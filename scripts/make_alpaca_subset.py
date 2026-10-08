"""Build the fixed AlpacaEval evaluation subset (default: 300 prompts, seed 42)."""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_sets/alpaca_eval_300_seed42.json")
    args = parser.parse_args()

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    indices = random.Random(args.seed).sample(range(len(rows)), args.n)
    subset = [
        {
            "id": f"alpaca-eval-{args.seed}-{k:04d}",
            "source_index": i,
            "instruction": rows[i]["instruction"],
            "dataset": "alpaca_eval",
            "source_subset": rows[i].get("dataset"),
        }
        for k, i in enumerate(indices)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(subset)} of {len(rows)} prompts to {out} (seed={args.seed})")


if __name__ == "__main__":
    main()