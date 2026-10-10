"""Build single-turn DPO preference data (manual Sections 6.3 and 10).

  --source hh  ->  Anthropic/hh-rlhf               (train / test splits)
  --source uf  ->  HuggingFaceH4/ultrafeedback_binarized   (train_prefs / test_prefs)

Every output row is {id, instruction, chosen, rejected}. Train and validation come from
separate source splits, so no validation pair can also appear in training.

Filters (all counted in summary.json):
  * multi-turn conversations are dropped (manual 6.3: single-turn reference setup)
  * empty responses and pairs where chosen == rejected are dropped
  * UltraFeedback ties (equal scores) are dropped: there is no preference to learn
  * prompts longer than 256 tokens after formatting are dropped (manual 10.4: max prompt 256)
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from prompt_format import format_prompt

SOURCES = {
    "hh": ("Anthropic/hh-rlhf", "train", "test"),
    "uf": ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs"),
}

HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def split_hh_transcript(text):
    """'\\n\\nHuman: q\\n\\nAssistant: a' -> (q, a), or None if not exactly one turn each."""
    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    before, answer = text.split(ASSISTANT)
    if not before.startswith(HUMAN):
        return None
    return before[len(HUMAN):].strip(), answer.strip()


def parse_hh(row):
    chosen, rejected = split_hh_transcript(row["chosen"]), split_hh_transcript(row["rejected"])
    if chosen is None or rejected is None:
        return None, "multi_turn"
    if chosen[0] != rejected[0]:
        return None, "prompt_mismatch"
    return (chosen[0], chosen[1], rejected[1]), None


def parse_uf(row):
    chosen, rejected = row["chosen"], row["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return None, "multi_turn"
    if [m["role"] for m in chosen] != ["user", "assistant"] or [m["role"] for m in rejected] != ["user", "assistant"]:
        return None, "multi_turn"
    if row.get("score_chosen") is not None and row["score_chosen"] == row["score_rejected"]:
        return None, "tied_scores"
    return (row["prompt"].strip(), chosen[1]["content"].strip(), rejected[1]["content"].strip()), None


PARSERS = {"hh": parse_hh, "uf": parse_uf}


def extract_pairs(rows, parse, tokenizer, n, max_prompt_tokens, max_length, id_prefix):
    kept = []
    stats = {"seen": 0, "multi_turn": 0, "prompt_mismatch": 0, "tied_scores": 0, "empty_or_identical": 0,
             "prompt_too_long": 0, "chosen_truncated": 0, "rejected_truncated": 0}

    for row in rows:
        if len(kept) == n:
            break
        stats["seen"] += 1

        triple, reason = parse(row)
        if triple is None:
            stats[reason] += 1
            continue
        instruction, chosen, rejected = triple
        if not instruction or not chosen or not rejected or chosen == rejected:
            stats["empty_or_identical"] += 1
            continue

        prompt_len = len(tokenizer.encode(format_prompt(instruction), add_special_tokens=False))
        if prompt_len > max_prompt_tokens:
            stats["prompt_too_long"] += 1
            continue

        # Counted, not dropped: these lose their tail (and EOS) at max_length during batching.
        for side, text in (("chosen", chosen), ("rejected", rejected)):
            if prompt_len + len(tokenizer.encode(text, add_special_tokens=False)) + 1 > max_length:
                stats[f"{side}_truncated"] += 1

        kept.append({"id": f"{id_prefix}-{len(kept):05d}", "instruction": instruction,
                     "chosen": chosen, "rejected": rejected})
    return kept, stats


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n_train", type=int, default=15_000)
    parser.add_argument("--n_val", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    dataset_name, train_split, val_split = SOURCES[args.source]
    out_dir = Path(args.out_dir or f"outputs/data/dpo_{args.source}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    summary = {"config": vars(args), "dataset": dataset_name}

    for split, n, name in [(train_split, args.n_train, "train"), (val_split, args.n_val, "val")]:
        rows = load_dataset(dataset_name, split=split).shuffle(seed=args.seed)
        kept, stats = extract_pairs(rows, PARSERS[args.source], tokenizer, n, args.max_prompt_tokens,
                                    args.max_length, id_prefix=f"{args.source}-{name}")
        if len(kept) < n:
            raise RuntimeError(f"Only found {len(kept)} usable {name} pairs, wanted {n}")
        write_jsonl(kept, out_dir / f"{name}.jsonl")
        summary[name] = {"source_split": split, "kept": len(kept), **stats}
        print(f"{args.source} {name}: kept {len(kept)} | {stats}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}/train.jsonl, val.jsonl, summary.json")


if __name__ == "__main__":
    main()