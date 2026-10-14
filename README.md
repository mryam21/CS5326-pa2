# CS5326 PA2 — Small-Model Alignment: SFT + DPO

**Author:** Mariyam Liaqat (LUMS) · **Report:** [`report/report.pdf`](report/report.pdf)

SFT and from-scratch DPO on `Qwen/Qwen2.5-0.5B` (HH-RLHF and UltraFeedback), plus a TRL `DPOTrainer` + LoRA
persona adapter on `HuggingFaceTB/SmolLM2-360M-Instruct`, evaluated with blinded pairwise judgments from
`google/gemini-3.6-flash` through `kaggle_benchmarks`.

## Results (judge: `google/gemini-3.6-flash`, n = 300 per set)

| Comparison | Eval set | Win / Loss / Tie | Win rate (excl. ties) |
|---|---|---|---|
| HH-DPO vs SFT | HH safety (synthetic, 300) | 158 / 98 / 44 | **61.7%** |
| UF-DPO vs SFT | AlpacaEval subset (300, seed 42) | 126 / 127 / 47 | 49.8% |
| SFT vs base (extra) | AlpacaEval subset | 124 / 149 / 27 | 45.4% |
| LoRA-DPO vs base, style | Persona prompts (24) | 18 / 1 / 5 | 94.7% |
| LoRA-DPO vs base, helpfulness | Persona prompts (24) | 10 / 13 / 1 | 43.5% |

## Where the code is

| What | File |
|---|---|
| SFT batching, response-only masking, log-probs, loss | `alignment/sft.py` |
| DPO batching, loss, rewards/metrics, reference handling | `alignment/dpo.py` |
| Optimizer, scheduler, logging, checkpointing | `alignment/train_utils.py` |
| SFT training loop (own loop, no HF `Trainer`) | `scripts/train_sft.py` |
| DPO training loop (own loop, no TRL) | `scripts/train_dpo.py` |
| LoRA persona DPO (TRL `DPOTrainer` + PEFT) | `scripts/train_lora_dpo.py` |
| Grader interface | `submission_adapter.py` |
| Prompt template | `prompt_format.py` |

Tests: `PYTHONPATH=. python -m pytest -q tests/` (public adapter smoke tests + my SFT/DPO unit tests; CPU only).

## Reproducing (Kaggle, GPU T4 x2, seed 42 everywhere)

Setup cell in every notebook (keeps Kaggle's PyTorch):

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

One notebook per stage; each stage's output is attached to the next via *Add Input → Your Work*.

| Stage (notebook) | Commands |
|---|---|
| Setup + baseline (`AdvGenAI PA2`) | `python scripts/make_alpaca_subset.py` · `python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --eval_set eval_sets/alpaca_eval_300_seed42.json --out outputs/generations/base_alpaca_eval.json --generator qwen2.5-0.5b-base --repetition_penalty 1.0` |
| SFT (`pa2-sft`) | `python scripts/make_sft_data.py` · `python scripts/train_sft.py` |
| HH DPO (`pa2-dpo-hh`) | `python scripts/make_dpo_data.py --source hh` · `python scripts/train_dpo.py --sft_checkpoint outputs/sft_run/best --train_file outputs/data/dpo_hh/train.jsonl --val_file outputs/data/dpo_hh/val.jsonl --run_dir outputs/dpo_hh_run` |
| UF DPO (`pa2-dpo-uf`) | same with `--source uf` and `dpo_uf` paths |
| LoRA persona (`pa2-lora`) | `python scripts/train_lora_dpo.py` (3 epochs; ablation `--epochs 1 --out_dir outputs/lora_dpo_1ep`) · `python scripts/generate_persona.py --adapter outputs/lora_dpo_run/adapter --out outputs/generations/persona_lora.json --generator lora` |
| Generation + pairs + plots (`pa2-generate`) | `scripts/generate.py` for each model/eval set (greedy, repetition penalty 1.1, 512 new tokens) · `scripts/build_pairs.py` · `scripts/plot_training.py` |
| Judging (`pa2-judge`, Kaggle **benchmark** notebook) | `python scripts/judge_pairwise_kaggle.py --pairs_path outputs/pairs/<name>.jsonl --output_path outputs/judgments/<name>.jsonl --judge_model google/gemini-3.6-flash --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments` |

## Outputs in this repository

```text
outputs/sft_run/, dpo_hh_run/, dpo_uf_run/   config.json, metrics.jsonl, summary.json
outputs/lora_dpo_run/, lora_dpo_1ep/          log_history.json, summary.json, split.json
outputs/data/*/summary.json                   data filtering counts
outputs/generations/                          all model outputs + *.config.json with exact decoding settings
outputs/pairs/                                blinded pairwise files (model identities only under "hidden")
outputs/judgments/                            per-pair verdicts, *_summary.json, full judge logs
outputs/figures/                              training plots used in the report
```

Model checkpoints are not committed (size); they are in the Kaggle notebook outputs listed above.
