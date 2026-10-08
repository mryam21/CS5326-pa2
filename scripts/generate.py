"""Greedy generation on a fixed eval set. Reused for base, SFT, and DPO models.

Writes results after every batch and skips ids already in --out, so it can resume.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_format import STOP_STRING, format_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity check)")
    args = parser.parse_args()

    rows = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["id"] for r in results}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to generate")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda:0").eval()

    decoding = {
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "stop_strings": [STOP_STRING],
        "prompt_template": format_prompt("{instruction}"),
    }
    Path(str(out) + ".config.json").write_text(
        json.dumps({"model_path": args.model_path, "generator": args.generator, **decoding}, indent=2)
    )

    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        prompts = [format_prompt(r["instruction"]) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")

        with torch.no_grad():
            generated = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                stop_strings=[STOP_STRING],
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        texts = tokenizer.batch_decode(generated[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for row, text in zip(batch, texts):
            results.append(
                {
                    "id": row["id"],
                    "instruction": row["instruction"],
                    "output": text.split(STOP_STRING)[0].strip(),
                    "generator": args.generator,
                    "dataset": row["dataset"],
                }
            )

        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{len(results)}/{len(done) + len(todo)} saved")


if __name__ == "__main__":
    main()