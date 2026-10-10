"""Supervised fine-tuning of Qwen2.5-0.5B with our own training loop (manual Task 3).

Checklist from manual Section 9.1, and where each item lives:
  1. prompt formatting ............ prompt_format.format_prompt (same template as eval)
  2. response-only loss masking ... alignment.sft.build_lm_batch / sft_loss_from_batch
  3. padding + attention masks .... alignment.sft.build_lm_batch
  4. gradient accumulation ........ the micro-batch loop in train()
  5. fp16 autocast ................ torch.autocast(..., dtype=torch.float16)
  6. fp32 weights + grad scaling .. model loaded in fp32, torch.amp.GradScaler
  7. gradient clipping ............ clip_grad_norm_ after scaler.unscale_
  8. validation loss .............. evaluate()
  9. best + final checkpoints ..... save_checkpoint(...) to <run_dir>/best and /final
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

from alignment.sft import build_lm_batch, response_logprobs_from_batch
from alignment.train_utils import (
    MetricsLogger,
    build_adamw,
    read_jsonl,
    save_checkpoint,
    set_seed,
    warmup_cosine_scheduler,
)
from prompt_format import format_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train_file", default="outputs/data/sft/train.jsonl")
    p.add_argument("--val_file", default="outputs/data/sft/val.jsonl")
    p.add_argument("--run_dir", default="outputs/sft_run")
    # Defaults = manual Section 9.2
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    # Bookkeeping
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validations")
    p.add_argument("--log_every", type=int, default=10, help="optimizer steps between train-loss logs")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    # Pilot runs / memory
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_val_examples", type=int, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true", help="use if you hit CUDA OOM")
    return p.parse_args()


def make_loader(rows, tokenizer, max_length, batch_size, shuffle, seed):
    prompts = [format_prompt(r["instruction"]) for r in rows]
    responses = [r["response"] for r in rows]
    pairs = list(zip(prompts, responses))

    def collate(items):
        ps, rs = zip(*items)
        # Pad only to the longest row in this batch: same loss, much less wasted compute.
        return build_lm_batch(tokenizer, list(ps), list(rs), max_length, pad_to_max_length=False)

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator)


@torch.no_grad()
def evaluate(model, loader, use_amp):
    """Token-mean validation loss over the whole validation set (same reduction as training)."""
    model.eval()
    total_logp, total_tokens = 0.0, 0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            summed, counts = response_logprobs_from_batch(model, batch)
        total_logp += summed.sum().item()
        total_tokens += counts.sum().item()
    model.train()
    return -total_logp / max(1, total_tokens)


def main():
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    logger = MetricsLogger(run_dir / "metrics.jsonl")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # FP32 master weights (manual 9.1 item 6). Autocast runs the matmuls in fp16;
    # the optimizer updates the fp32 copy, so tiny updates are not rounded away.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.config.use_cache = False  # the KV cache is only useful for generation
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    train_rows = read_jsonl(args.train_file, args.max_train_examples)
    val_rows = read_jsonl(args.val_file, args.max_val_examples)
    train_loader = make_loader(train_rows, tokenizer, args.max_length, args.batch_size, True, args.seed)
    val_loader = make_loader(val_rows, tokenizer, args.max_length, args.eval_batch_size, False, args.seed)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = build_adamw(model, args.lr, args.weight_decay)
    scheduler, warmup_steps = warmup_cosine_scheduler(optimizer, total_steps, args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"train={len(train_rows)} val={len(val_rows)} | effective batch={args.batch_size * args.grad_accum} "
        f"| optimizer steps={total_steps} (warmup {warmup_steps}) | device={device} amp={use_amp}",
        flush=True,
    )

    # Validation before any training: the starting point of the val-loss curve.
    initial_val = evaluate(model, val_loader, use_amp)
    logger.log(type="val", step=0, val_loss=initial_val)
    print(f"step 0 | val_loss {initial_val:.4f}", flush=True)
    best_val = float("inf")

    step, window_loss, window_micro, skipped = 0, 0.0, 0, 0
    start = time.time()

    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            # Each optimizer step covers grad_accum micro-batches (fewer for the last one).
            group_start = (i // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                summed, counts = response_logprobs_from_batch(model, batch)
                loss = -summed.sum() / counts.sum().clamp_min(1)

            # Divide so the accumulated gradient is the AVERAGE over micro-batches, not their sum.
            scaler.scale(loss / group_size).backward()
            window_loss += loss.item()
            window_micro += 1

            if i - group_start + 1 < group_size:
                continue  # keep accumulating

            # Clip the TRUE gradient: undo the loss scaling first.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()

            scale_before = scaler.get_scale()
            scaler.step(optimizer)   # skipped automatically if the grads contain inf/NaN
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped += 1
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                elapsed = time.time() - start
                eta_min = elapsed / step * (total_steps - step) / 60
                record = logger.log(
                    type="train", step=step, epoch=epoch, loss=window_loss / window_micro,
                    lr=scheduler.get_last_lr()[0], grad_norm=grad_norm, skipped_steps=skipped,
                )
                print(
                    f"step {step}/{total_steps} | loss {record['loss']:.4f} | lr {record['lr']:.2e} "
                    f"| grad_norm {grad_norm:.2f} | eta {eta_min:.0f} min",
                    flush=True,
                )
                window_loss, window_micro = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                val_loss = evaluate(model, val_loader, use_amp)
                is_best = val_loss < best_val
                logger.log(type="val", step=step, val_loss=val_loss, is_best=is_best)
                print(f"step {step} | val_loss {val_loss:.4f}{'  <- best, saving' if is_best else ''}", flush=True)
                if is_best:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, run_dir / "best",
                                    {"step": step, "val_loss": val_loss})

    save_checkpoint(model, tokenizer, run_dir / "final", {"step": step, "val_loss": val_loss})
    summary = {"total_steps": step, "best_val_loss": best_val, "final_val_loss": val_loss,
               "skipped_steps": skipped, "train_minutes": round((time.time() - start) / 60, 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done:", summary, flush=True)


if __name__ == "__main__":
    main()