r"""Create reproducible MMA3001 report charts from existing YOLO11 runs.

Place this file in MMA3001_Project/src/plot_report_figures.py.
Run from the project root:
    .\.venv\Scripts\python.exe src\plot_report_figures.py

No training takes place. Results are read from Dataset V1 experiment folders.
Plots display per-epoch VALIDATION metrics, not final held-out test metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save image files; no desktop plot window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "results" / "experiments"
OUTPUT_DIR = ROOT / "results" / "report_figures"
MAP_COL = "metrics/mAP50-95(B)"

# These keys match the saved folders produced by run_architecture_experiments.py.
EXPERIMENTS = {
    "baseline": "Baseline",
    "narrower": "Narrower",
    "wider": "Wider",
    "deeper": "Deeper",
    "lr_low": "LR = 0.0003",
    "lr_high": "LR = 0.003",
    "combined": "Combined",
}


def read_runs():
    """Return run summaries and per-epoch histories from actual saved CSV files."""
    records = []
    histories = {}
    missing = []
    for key, label in EXPERIMENTS.items():
        folder = EXPERIMENTS_DIR / f"arch_{key}_scratch_30ep"
        csv_path = folder / "results.csv"
        metadata_path = folder / "mma3001_arch_metadata.json"
        if not csv_path.is_file() or not metadata_path.is_file():
            missing.append(label)
            continue

        history = pd.read_csv(csv_path)
        history.columns = history.columns.str.strip()
        if MAP_COL not in history.columns or "epoch" not in history.columns:
            raise ValueError(f"Required columns missing in {csv_path}")
        history[MAP_COL] = pd.to_numeric(history[MAP_COL], errors="raise")
        history["epoch"] = pd.to_numeric(history["epoch"], errors="raise")
        if history.empty or history[MAP_COL].isna().any():
            raise ValueError(f"Empty or invalid mAP values in {csv_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        peak_idx = history[MAP_COL].idxmax()
        records.append({
            "experiment": key,
            "label": label,
            "depth": float(metadata["depth"]),
            "width": float(metadata["width"]),
            "learning_rate": float(metadata["lr0"]),
            "parameters": int(metadata["parameters"]),
            "train_minutes": float(metadata["train_seconds"]) / 60,
            "peak_val_mAP50_95_pct": 100 * float(history.loc[peak_idx, MAP_COL]),
            "peak_epoch": int(history.loc[peak_idx, "epoch"]),
            "last_val_mAP50_95_pct": 100 * float(history.iloc[-1][MAP_COL]),
        })
        histories[key] = history

    if missing:
        print("Not found (skipped):", ", ".join(missing))
    if not records:
        raise FileNotFoundError(
            "No experiments found. Place this script inside MMA3001_Project/src "
            "and make sure results/experiments contains your saved runs."
        )
    return pd.DataFrame(records), histories


def save_figure(fig, basename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / basename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", path)


def comparison_chart(summary):
    """Peak versus last-epoch validation mAP for all seven experiments."""
    plot = summary.sort_values("peak_val_mAP50_95_pct", ascending=True)
    y = np.arange(len(plot))
    h = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.3))
    ax.barh(y + h / 2, plot["peak_val_mAP50_95_pct"], height=h,
            color="#315d91", label="Highest epoch mAP")
    ax.barh(y - h / 2, plot["last_val_mAP50_95_pct"], height=h,
            color="#94b5d4", label="Epoch 30 mAP")
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("Validation mAP@50–95 (%)")
    ax.set_title("YOLO11 architecture and learning-rate comparison")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    limit = float(max(plot["peak_val_mAP50_95_pct"].max(),
                      plot["last_val_mAP50_95_pct"].max())) + 3
    ax.set_xlim(0, limit)
    for i, (_, row) in enumerate(plot.iterrows()):
        for offset, col in ((h / 2, "peak_val_mAP50_95_pct"),
                            (-h / 2, "last_val_mAP50_95_pct")):
            val = row[col]
            ax.text(val + 0.15, i + offset, f"{val:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "figure_1_experiment_comparison.png")


def complexity_chart(summary):
    """Architecture variants: trade-off between capacity and peak val mAP."""
    # LR variants have the same parameter count as baseline and clutter this plot.
    variants = summary[summary["experiment"].isin(
        ["baseline", "narrower", "wider", "deeper", "combined"]
    )]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for _, row in variants.iterrows():
        x = row["parameters"] / 1e6
        y = row["peak_val_mAP50_95_pct"]
        color = "#256a5c" if row["experiment"] == "baseline" else "#315d91"
        ax.scatter([x], [y], s=95, c=color, zorder=3)
        ax.annotate(row["label"], (x, y), xytext=(7, 6),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Trainable parameters (millions; training model)")
    ax.set_ylabel("Peak validation mAP@50–95 (%)")
    ax.set_title("Detection quality versus model complexity")
    ax.grid(alpha=0.25)
    ax.margins(x=0.18, y=0.19)
    fig.tight_layout()
    save_figure(fig, "figure_2_model_complexity.png")


def learning_curves(histories):
    """Trace validation mAP over epochs for baseline and wider models."""
    if "baseline" not in histories or "wider" not in histories:
        print("Skipping learning curves: baseline or wider results.csv not found")
        return
    fig, ax = plt.subplots(figsize=(9, 5.1))
    for key, color in (("baseline", "#256a5c"), ("wider", "#315d91")):
        df = histories[key]
        ax.plot(df["epoch"], 100 * df[MAP_COL], marker="o", markersize=3,
                linewidth=1.8, label=EXPERIMENTS[key], color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP@50–95 (%)")
    ax.set_title("Validation learning curves: baseline versus wider YOLO11")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "figure_3_validation_learning_curves.png")


def main():
    summary, histories = read_runs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "report_experiment_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    comparison_chart(summary)
    complexity_chart(summary)
    learning_curves(histories)
    print("\nThese are VALIDATION metrics from training CSV histories, not test results.")
    print("Peak value means the maximum across epochs, NOT necessarily the best.pt score.")
    print("No models were trained or modified.")


if __name__ == "__main__":
    main()
