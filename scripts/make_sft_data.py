"""Build the single-turn SFT train/validation sets from UltraChat (manual Section 6.2).

Each UltraChat conversation is cut down to its FIRST user turn and FIRST assistant reply,
so every example is one instruction -> one response.

Train comes from the `train_sft` split, validation from the separate `test_sft` split,
so no validation example can also appear in training.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from prompt_format import format_prompt


def first_turn_pair(messages):
    """Return (instruction, response) from the first user/assistant exchange, or None."""
    if len(messages) < 2:
        return None
    user, assistant = messages[0], messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    instruction = (user.get("content") or "").strip()
    response = (assistant.get("content") or "").strip()
    if not instruction or not response:
        return None
    return instruction, response


def extract_examples(rows, tokenizer, n, max_prompt_tokens, max_length, id_prefix):
    """Walk the (already shuffled) rows and keep the first n usable single-turn examples."""
    kept, stats = [], {"seen": 0, "bad_turns": 0, "prompt_too_long": 0, "will_be_truncated": 0}

    for row in rows:
        if len(kept) == n:
            break
        stats["seen"] += 1

        pair = first_turn_pair(row["messages"])
        if pair is None:
            stats["bad_turns"] += 1
            continue
        instruction, response = pair

        # Drop examples whose prompt alone eats most of the context window:
        # they would leave few or no response tokens to train on.
        prompt_len = len(tokenizer.encode(format_prompt(instruction), add_special_tokens=False))
        if prompt_len > max_prompt_tokens:
            stats["prompt_too_long"] += 1
            continue

        # Not dropped, just counted: these get cut at max_length during batching,
        # which also cuts off the EOS token (worth reporting).
        response_len = len(tokenizer.encode(response, add_special_tokens=False))
        if prompt_len + response_len + 1 > max_length:
            stats["will_be_truncated"] += 1

        kept.append({"id": f"{id_prefix}-{len(kept):05d}", "instruction": instruction, "response": response})

    return kept, stats


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n_train", type=int, default=30_000)
    parser.add_argument("--n_val", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=384)
    parser.add_argument("--out_dir", default="outputs/data/sft")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out_dir = Path(args.out_dir)
    summary = {"config": vars(args)}

    for split, n, name in [("train_sft", args.n_train, "train"), ("test_sft", args.n_val, "val")]:
        rows = load_dataset(args.dataset, split=split).shuffle(seed=args.seed)
        kept, stats = extract_examples(
            rows, tokenizer, n, args.max_prompt_tokens, args.max_length, id_prefix=f"sft-{name}"
        )
        if len(kept) < n:
            raise RuntimeError(f"Only found {len(kept)} usable {name} examples, wanted {n}")
        write_jsonl(kept, out_dir / f"{name}.jsonl")
        summary[name] = {"source_split": split, "kept": len(kept), **stats}
        print(f"{name}: kept {len(kept)} | {stats}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}/train.jsonl, val.jsonl, summary.json")


if __name__ == "__main__":
    main()