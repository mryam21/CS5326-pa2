"""From-scratch DPO training (manual Task 4). Same script for the HH-RLHF and UltraFeedback runs.

Checklist from manual Section 10.1 / 10.3, and where each item lives:
  1. SFT model as trainable policy ........ `policy` (fp32 weights, cuda:0)
  2. frozen SFT copy as reference ......... `reference` (cuda:1, eval mode, requires_grad False)
  3. response-only chosen/rejected logps .. alignment.dpo.policy_and_reference_logps
  4. DPO loss ............................. alignment.dpo.dpo_loss_from_logps
  5. val DPO loss / reward acc / margin ... evaluate()
  6. best checkpoint by val reward acc .... save_checkpoint(... "best") when accuracy improves
  Recipe: beta 0.1, fixed lr 1e-6, 1 pair per micro-batch x 32 accumulation, clip 1.0, 1 epoch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.dpo import build_dpo_batch, dpo_loss_from_logps, dpo_metrics, policy_and_reference_logps
from alignment.train_utils import MetricsLogger, read_jsonl, save_checkpoint, set_seed
from prompt_format import format_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_checkpoint", required=True, help="SFT best checkpoint (policy init AND reference)")
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", required=True)
    p.add_argument("--run_dir", required=True)
    # Defaults = manual Section 10.3 / 10.4
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--reference_dtype", choices=["fp32", "fp16"], default="fp32",
                   help="fp32 makes policy and reference numerically identical at step 0")
    # Bookkeeping
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Pilot runs / memory
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_val_examples", type=int, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()


def make_loader(rows, tokenizer, max_length, batch_size, shuffle, seed):
    triples = [(format_prompt(r["instruction"]), r["chosen"], r["rejected"]) for r in rows]

    def collate(items):
        prompts, chosen, rejected = map(list, zip(*items))
        return build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length, pad_to_max_length=False)

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(triples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator)


def summarize(per_pair):
    """Average per-pair tensors collected over many batches."""
    cat = {k: torch.cat(v) for k, v in per_pair.items()}
    return {
        "loss": cat["loss"].mean().item(),
        "reward_accuracy": cat["correct"].mean().item(),
        "reward_margin": cat["margins"].mean().item(),
        "chosen_reward": cat["chosen_rewards"].mean().item(),
        "rejected_reward": cat["rejected_rewards"].mean().item(),
    }


def collect(store, logps, beta):
    """Per-pair loss + reward metrics for one batch, appended to `store`."""
    with torch.no_grad():
        per_pair_loss = torch.stack([
            dpo_loss_from_logps(*(t[i:i + 1] for t in logps), beta) for i in range(len(logps[0]))
        ])
    for key, value in dpo_metrics(*logps, beta).items():
        store.setdefault(key, []).append(value.float().cpu())
    store.setdefault("loss", []).append(per_pair_loss.float().cpu())


@torch.no_grad()
def evaluate(policy, reference, loader, beta, use_amp):
    policy.eval()
    store = {}
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logps = policy_and_reference_logps(policy, reference, batch)
        collect(store, logps, beta)
    policy.train()
    return summarize(store)


def main():
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    logger = MetricsLogger(run_dir / "metrics.jsonl")

    use_amp = torch.cuda.is_available()
    policy_device = torch.device("cuda:0" if use_amp else "cpu")
    reference_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else policy_device)
    reference_dtype = torch.float16 if (args.reference_dtype == "fp16" and use_amp) else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)

    policy = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, dtype=torch.float32).to(policy_device)
    policy.config.use_cache = False
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.train()

    # The reference is a separate, frozen copy of the SAME SFT checkpoint.
    reference = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, dtype=reference_dtype).to(reference_device)
    reference.config.use_cache = False
    reference.eval()
    reference.requires_grad_(False)

    train_rows = read_jsonl(args.train_file, args.max_train_examples)
    val_rows = read_jsonl(args.val_file, args.max_val_examples)
    train_loader = make_loader(train_rows, tokenizer, args.max_length, args.batch_size, True, args.seed)
    val_loader = make_loader(val_rows, tokenizer, args.max_length, args.eval_batch_size, False, args.seed)

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    params = [p for p in policy.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    else:
        optimizer = torch.optim.RMSprop(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # fixed LR: no scheduler (manual 10.4)

    print(
        f"train={len(train_rows)} val={len(val_rows)} | effective batch={args.batch_size * args.grad_accum} "
        f"| optimizer steps={total_steps} | beta={args.beta} lr={args.lr} | policy={policy_device} "
        f"reference={reference_device} ({args.reference_dtype})",
        flush=True,
    )

    val = evaluate(policy, reference, val_loader, args.beta, use_amp)
    logger.log(type="val", step=0, **val)
    print(f"step 0 | val {json.dumps({k: round(v, 4) for k, v in val.items()})}", flush=True)

    best_acc, step, skipped, window, start = -1.0, 0, 0, {}, time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            group_start = (i // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logps = policy_and_reference_logps(policy, reference, batch)
                loss = dpo_loss_from_logps(*logps, args.beta)
            scaler.scale(loss / group_size).backward()
            collect(window, [t.detach() for t in logps], args.beta)

            if i - group_start + 1 < group_size:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                train = summarize(window)
                window = {}
                logger.log(type="train", step=step, epoch=epoch, grad_norm=grad_norm, skipped_steps=skipped, **train)
                eta = (time.time() - start) / step * (total_steps - step) / 60
                print(f"step {step}/{total_steps} | loss {train['loss']:.4f} | acc {train['reward_accuracy']:.3f} "
                      f"| margin {train['reward_margin']:.4f} | grad_norm {grad_norm:.2f} | eta {eta:.0f} min", flush=True)

            if step % args.eval_every == 0 or step == total_steps:
                val = evaluate(policy, reference, val_loader, args.beta, use_amp)
                is_best = val["reward_accuracy"] > best_acc
                logger.log(type="val", step=step, is_best=is_best, **val)
                print(f"step {step} | val {json.dumps({k: round(v, 4) for k, v in val.items()})}"
                      f"{'  <- best acc, saving' if is_best else ''}", flush=True)
                if is_best:
                    best_acc = val["reward_accuracy"]
                    save_checkpoint(policy, tokenizer, run_dir / "best", {"step": step, **val})

    save_checkpoint(policy, tokenizer, run_dir / "final", {"step": step, **val})
    summary = {"total_steps": step, "best_val_reward_accuracy": best_acc, "final_val": val,
               "skipped_steps": skipped, "train_minutes": round((time.time() - start) / 60, 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()