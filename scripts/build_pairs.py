"""Build blinded pairwise-comparison files (manual Task 6).

For each prompt, the candidate and baseline outputs are placed as Response A / Response B
in a random order (fixed seed). The judge only ever sees `prompt`, `response_a`, `response_b`
(and, for the HH safety set, the eval set's own `expected_behavior` note).
Which model sits in which position is stored under `hidden`, and is only used AFTER judging.

  python scripts/build_pairs.py --candidate dpo_uf_alpaca.json --baseline sft_alpaca.json \
      --candidate_name dpo_uf --baseline_name sft --rubric general --out pairs/uf_dpo_vs_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, help="generation JSON of the model being tested (e.g. DPO)")
    p.add_argument("--baseline", required=True, help="generation JSON of the reference model (e.g. SFT)")
    p.add_argument("--candidate_name", required=True)
    p.add_argument("--baseline_name", required=True)
    p.add_argument("--rubric", choices=["general", "safety", "persona"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cand, base = load(args.candidate), load(args.baseline)
    if set(cand) != set(base):
        raise SystemExit(f"Prompt ids differ: {len(set(cand) ^ set(base))} unmatched")

    rng = random.Random(args.seed)
    rows, n_cand_first = [], 0
    for pid in sorted(cand):
        c, b = cand[pid], base[pid]
        if c["instruction"] != b["instruction"]:
            raise SystemExit(f"Instruction mismatch for {pid}")
        cand_first = rng.random() < 0.5
        n_cand_first += cand_first
        a, bb = (c, b) if cand_first else (b, c)
        row = {
            "pair_id": pid,
            "rubric": args.rubric,
            "prompt": c["instruction"],
            "response_a": a["output"],
            "response_b": bb["output"],
            "category": c.get("category"),
            "expected_behavior": c.get("expected_behavior"),
            # NEVER shown to the judge:
            "hidden": {
                "a_model": args.candidate_name if cand_first else args.baseline_name,
                "b_model": args.baseline_name if cand_first else args.candidate_name,
                "candidate": args.candidate_name,
                "baseline": args.baseline_name,
            },
        }
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} pairs to {out} | candidate shown as A in {n_cand_first}/{len(rows)}")


if __name__ == "__main__":
    main()