"""Training plots for the report (manual Section 14.4 / submission checklist item 3).

Reads the metrics.jsonl files written by train_sft.py and train_dpo.py and saves PNGs:

  sft_training.png         SFT train vs validation loss
  dpo_<name>_training.png  DPO loss, reward accuracy, reward margin, chosen/rejected rewards
  dpo_comparison.png       validation reward accuracy and margin, all DPO runs together

Usage:
  python scripts/plot_training.py --sft path/to/sft_run/metrics.jsonl \
      --dpo HH-RLHF=path/to/dpo_hh_run/metrics.jsonl --dpo UltraFeedback=path/to/dpo_uf_run/metrics.jsonl \
      --out_dir report/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (fixed order) + neutral inks.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False, "font.size": 9,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def read_metrics(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    train = [r for r in rows if r["type"] == "train"]
    val = [r for r in rows if r["type"] == "val"]
    return train, val


def series(rows, key):
    pts = [(r["step"], r[key]) for r in rows if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_line(ax, x, y, color, label, marker=False, alpha=1.0, end_label=True):
    kw = dict(marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5) if marker else {}
    ax.plot(x, y, color=color, label=label, alpha=alpha, **kw)
    if end_label and x:
        ax.annotate(f"{label} {y[-1]:.3f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2, annotation_clip=False)


def mark_best(ax, x, y, mode, text):
    """Ring the best validation point; annotate it unless it is the last point (the end label covers that)."""
    if not x:
        return
    i = max(range(len(y)), key=lambda k: y[k]) if mode == "max" else min(range(len(y)), key=lambda k: y[k])
    ax.scatter([x[i]], [y[i]], s=110, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=5)
    label = f"{text}: step {x[i]}" if i < len(x) - 1 else f"{text} = final step"
    ax.annotate(label, (x[i], y[i]), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=8, color=INK)


def finish(fig, path, title, subtitle):
    h = fig.get_size_inches()[1]
    fig.text(0.01, 1 - 0.25 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.52 / h, subtitle, ha="left", va="top", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 0.93, 1 - 0.8 / h))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def plot_sft(metrics_path, out_dir):
    train, val = read_metrics(metrics_path)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    plot_line(ax, *series(train, "loss"), BLUE, "train", alpha=0.85, end_label=False)
    vx, vy = series(val, "val_loss")
    plot_line(ax, vx, vy, ORANGE, "validation", marker=True)
    mark_best(ax, vx, vy, "min", "best")
    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token loss (nats / token)")
    ax.legend(loc="upper right")
    finish(fig, out_dir / "sft_training.png", "SFT on 30k UltraChat examples",
           "Token-mean cross-entropy on response tokens only. Validation on 1,000 held-out examples.")


def plot_dpo(name, metrics_path, out_dir):
    train, val = read_metrics(metrics_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    plot_line(ax, *series(train, "loss"), BLUE, "train", alpha=0.85, end_label=False)
    plot_line(ax, *series(val, "loss"), ORANGE, "val", marker=True)
    ax.axhline(0.6931, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.99, 0.6931, "log 2 (start) ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=INK_2)
    ax.set_title("DPO loss", loc="left")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    plot_line(ax, *series(train, "reward_accuracy"), BLUE, "train", alpha=0.85, end_label=False)
    vx, vy = series(val, "reward_accuracy")
    plot_line(ax, vx, vy, ORANGE, "val", marker=True)
    mark_best(ax, vx, vy, "max", "best checkpoint")
    ax.axhline(0.5, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.99, 0.5, "chance ", transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=8, color=INK_2)
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("Reward accuracy  (r_chosen > r_rejected)", loc="left")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    plot_line(ax, *series(train, "reward_margin"), BLUE, "train", alpha=0.85, end_label=False)
    plot_line(ax, *series(val, "reward_margin"), ORANGE, "val", marker=True)
    ax.set_title("Reward margin  (r_chosen − r_rejected)", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    plot_line(ax, *series(val, "chosen_reward"), AQUA, "chosen", marker=True)
    plot_line(ax, *series(val, "rejected_reward"), VIOLET, "rejected", marker=True)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_title("Validation rewards  β·(log π − log π_ref)", loc="left")
    ax.legend(loc="lower left")

    for ax in axes[1]:
        ax.set_xlabel("optimizer step (effective batch 32)")
    slug = name.lower().replace(" ", "_").replace("-", "_")
    finish(fig, out_dir / f"dpo_{slug}_training.png", f"DPO on {name} (from the SFT checkpoint)",
           "β = 0.1, fixed lr 1e-6. Train curves are 10-step averages; validation on 1,000 held-out pairs.")


def plot_comparison(runs, out_dir):
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (name, path), color in zip(runs, colors):
        _, val = read_metrics(path)
        plot_line(axes[0], *series(val, "reward_accuracy"), color, name, marker=True)
        plot_line(axes[1], *series(val, "reward_margin"), color, name, marker=True)
    axes[0].axhline(0.5, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].set_ylim(-0.03, 1.0)
    axes[0].set_title("Validation reward accuracy", loc="left")
    axes[1].set_title("Validation reward margin", loc="left")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.legend(loc="lower right")
    finish(fig, out_dir / "dpo_comparison.png", "DPO runs compared",
           "Same SFT starting point, β, learning rate and step count; only the preference data differs.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", help="SFT metrics.jsonl")
    p.add_argument("--dpo", action="append", default=[], help="NAME=path/to/metrics.jsonl (repeatable)")
    p.add_argument("--out_dir", default="report/figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.sft:
        plot_sft(args.sft, out_dir)
    runs = [tuple(item.split("=", 1)) for item in args.dpo]
    for name, path in runs:
        plot_dpo(name, path, out_dir)
    if len(runs) > 1:
        plot_comparison(runs, out_dir)


if __name__ == "__main__":
    main()