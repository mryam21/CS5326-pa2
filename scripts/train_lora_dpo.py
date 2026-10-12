"""Task 5 ("Idiot Sandwich"): TRL DPOTrainer + PEFT/LoRA persona tuning of SmolLM2-360M-Instruct.

Unlike Tasks 3-4, the manual explicitly allows (and asks for) TRL's DPOTrainer and PEFT here.

LoRA choices (defaults below; justify them in the report, manual Section 11):
  * rank r=16, alpha=32 (scale alpha/r = 2): style is a low-dimensional change; r=16 gives
    ~8.7M trainable params (~2.4% of 362M), plenty for 200 pairs without memorizing them.
  * target modules = all attention + MLP projections: tone lives in word choice, so the MLP
    (where token-level features are stored) matters as much as attention.
  * lr 5e-5 with cosine decay + 10% warmup: LoRA adapters start at zero and need a much higher
    lr than full fine-tuning (1e-6 in Task 4); 5e-5 is a conservative LoRA-DPO value.
  * batch 4 x accumulation 2 = 8 pairs per step, 3 epochs: only ~200 training pairs, so a small
    batch gives ~75 optimizer steps; more epochs on 200 pairs would mostly memorize them.
  * beta 0.1, as in Task 4.

Reference model: with a PEFT model, DPOTrainer uses the SAME base weights with the adapter
switched off as the frozen reference, so no second copy of the model is loaded.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ALL_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--out_dir", default="outputs/lora_dpo_run")
    p.add_argument("--val_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=ALL_PROJECTIONS)
    # DPO / optimization
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()


def to_conversational(row):
    """TRL's conversational preference format: role/content message lists."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    val_idx, train_idx = order[: args.val_size], order[args.val_size:]
    # Save the exact split (manual 6.4: save the exact subsets used).
    (out_dir / "split.json").write_text(json.dumps({"train": train_idx, "val": val_idx}))

    train_ds = Dataset.from_list([to_conversational(rows[i]) for i in train_idx])
    val_ds = Dataset.from_list([to_conversational(rows[i]) for i in val_idx])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    # fp32 base weights + fp16 autocast (T4 has no bf16); LoRA weights stay fp32 for stable updates.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM")

    config = DPOConfig(
        output_dir=str(out_dir / "trainer"),
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        fp16=use_cuda,
        bf16=False,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )

    trainer = DPOTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tokenizer, peft_config=lora)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"train={len(train_ds)} val={len(val_ds)} | trainable params {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)", flush=True)

    before = trainer.evaluate()  # adapter starts at zero, so policy == reference here
    trainer.train()
    after = trainer.evaluate()

    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    # TRL logs the reward metrics into log_history rather than returning them from evaluate().
    keys = ["eval_loss", "eval_rewards/accuracies", "eval_rewards/margins",
            "eval_rewards/chosen", "eval_rewards/rejected"]
    last_eval = [e for e in trainer.state.log_history if "eval_rewards/accuracies" in e][-1]
    summary = {
        "config": vars(args),
        "trl_version": trl.__version__,
        "trainable_params": trainable,
        "total_params": total,
        # Adapter starts at zero => policy == reference, so rewards/accuracy/margin are exactly 0.
        "before_training": {"eval_loss": before.get("eval_loss")},
        "after_training": {k: last_eval.get(k, after.get(k)) for k in keys},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("before:", json.dumps(summary["before_training"]))
    print("after: ", json.dumps(summary["after_training"]))


if __name__ == "__main__":
    main()