"""
Persona DPO ("Idiot Sandwich") on SmolLM2-360M-Instruct with TRL's DPOTrainer and a LoRA adapter.

Defaults: r=16, alpha=32, dropout 0.05 on all attention and MLP projections (8.7M trainable
params, 2.3% of the model); lr 5e-5 with 10% warmup and cosine decay; batch 4 x 2 accumulation;
3 epochs over the 200 training pairs (75 steps); beta 0.1. With a PEFT model, DPOTrainer uses
the base weights with the adapter disabled as the reference, so no second model is loaded.
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
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=ALL_PROJECTIONS)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()


def to_conversational(row):
    """Convert a row to TRL's conversational format (lists of role/content messages)."""
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
    (out_dir / "split.json").write_text(json.dumps({"train": train_idx, "val": val_idx}))

    train_ds = Dataset.from_list([to_conversational(rows[i]) for i in train_idx])
    val_ds = Dataset.from_list([to_conversational(rows[i]) for i in val_idx])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    # fp32 weights with fp16 mixed precision (T4 has no bf16)
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

    before = trainer.evaluate()  # adapter is zero-initialized, so policy == reference here
    trainer.train()
    after = trainer.evaluate()

    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    # the reward metrics end up in log_history, not in the dict evaluate() returns
    keys = ["eval_loss", "eval_rewards/accuracies", "eval_rewards/margins",
            "eval_rewards/chosen", "eval_rewards/rejected"]
    last_eval = [e for e in trainer.state.log_history if "eval_rewards/accuracies" in e][-1]
    summary = {
        "config": vars(args),
        "trl_version": trl.__version__,
        "trainable_params": trainable,
        "total_params": total,
        "before_training": {"eval_loss": before.get("eval_loss")},
        "after_training": {k: last_eval.get(k, after.get(k)) for k in keys},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("before:", json.dumps(summary["before_training"]))
    print("after: ", json.dumps(summary["after_training"]))


if __name__ == "__main__":
    main()