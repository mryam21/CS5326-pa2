"""Helpers shared by the SFT and DPO training loops."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_adamw(model, lr, weight_decay):
    """AdamW with weight decay on weight matrices only (no decay on biases / norm weights)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio):
    """Linear warmup, then cosine decay to 0."""
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor), warmup_steps


class MetricsLogger:
    """Appends one JSON line per event so a crashed run still leaves its logs."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, **record):
        record["elapsed_s"] = round(time.time() - self.start, 1)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record


def save_checkpoint(model, tokenizer, path, extra=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if extra is not None:
        (path / "checkpoint_info.json").write_text(json.dumps(extra, indent=2))