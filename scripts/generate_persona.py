"""
Persona-prompt generations from SmolLM2-360M-Instruct, with or without the LoRA adapter.

Uses the model's own chat template; greedy decoding with repetition_penalty 1.1.

  python scripts/generate_persona.py --generator smollm2-360m-instruct --out outputs/generations/persona_base.json
  python scripts/generate_persona.py --adapter outputs/lora_dpo_run/adapter \
      --generator smollm2-360m-lora-dpo --out outputs/generations/persona_lora.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for the untuned model")
    p.add_argument("--eval_set", default="eval_sets/persona_style_prompts.json")
    p.add_argument("--out", required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model = model.to(device).eval()

    rows = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": r["instruction"]}],
                                                 tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], do_sample=False, max_new_tokens=args.max_new_tokens,
                                 repetition_penalty=args.repetition_penalty,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, text in zip(batch, texts):
            results.append({"id": r["id"], "category": r["category"], "instruction": r["instruction"],
                            "output": text.strip(), "generator": args.generator, "dataset": r["dataset"]})
        print(f"{len(results)}/{len(rows)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {k: v for k, v in vars(args).items()} | {"do_sample": False}
    Path(str(args.out) + ".config.json").write_text(json.dumps(config, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()