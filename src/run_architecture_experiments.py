r"""MMA3001: controlled YOLO11 network depth/width/learning-rate experiments.

Place at: MMA3001_Project/src/run_architecture_experiments.py
Run from repository root (Windows PowerShell):
    .\.venv\Scripts\python.exe src\run_architecture_experiments.py preview
    .\.venv\Scripts\python.exe src\run_architecture_experiments.py baseline
    .\.venv\Scripts\python.exe src\run_architecture_experiments.py wider
    .\.venv\Scripts\python.exe src\run_architecture_experiments.py deeper
    .\.venv\Scripts\python.exe src\run_architecture_experiments.py summary

Architecture experiments use custom YAML configurations derived from the YOLO11 template
installed with your version of Ultralytics. For a *controlled architecture comparison*,
default initialization is 'scratch' for ALL runs. These runs must not be compared as if
identical to your earlier PRETRAINED yolo11n.pt smoke test.

If --init transfer is selected, matching weights from yolo11n.pt are transferred.
When depth/width changes, different fractions of pretrained weights may match,
confounding pure architectural conclusions. Record this limitation in the report.

Each run gets a unique folder in results/experiments and never overwrites a prior run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import torch
import ultralytics
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "raw" / "dataset_v1" / "data.yaml"
RUNS_DIR = ROOT / "results" / "experiments"
MODEL_CONFIG_DIR = ROOT / "configs" / "generated"
ULTRA_MODEL_YAML = (
    Path(ultralytics.__file__).resolve().parent
    / "cfg" / "models" / "11" / "yolo11.yaml"
)

# Identical settings make effects of width, depth or learning rate interpretable.
TRAINING = {
    "epochs": 30,
    "imgsz": 640,
    "batch": 2,
    "workers": 2,
    "seed": 42,
    "deterministic": True,
    "optimizer": "AdamW",  # 'auto' may IGNORE lr0
    "plots": True,
    "verbose": True,
    "patience": 30,
}

# YOLO11n reference scale is depth=.50, width=.25, max_channels=1024.
# Width/depth adjust channels/repeated blocks; NOT input image pixel dimensions.
EXPERIMENTS = {
    "baseline": {"depth": 0.50, "width": 0.25, "lr0": 0.001},
    "narrower": {"depth": 0.50, "width": 0.1875, "lr0": 0.001}, 
    "wider": {"depth": 0.50, "width": 0.375, "lr0": 0.001},
    "deeper": {"depth": 1.00, "width": 0.25, "lr0": 0.001},
    "lr_low": {"depth": 0.50, "width": 0.25, "lr0": 0.0003},
    "lr_high": {"depth": 0.50, "width": 0.25, "lr0": 0.003},
    "combined": {"depth": 1.00, "width": 0.375, "lr0": 0.0003},
}

METRIC_COLS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}


def dataset_class_count() -> int:
    """Read number of class labels from the existing Roboflow data.yaml."""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Missing dataset file: {DATA_YAML}")
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    if "nc" in data:
        nc = int(data["nc"])
    elif "names" in data:
        nc = len(data["names"])
    else:
        raise ValueError(f"Neither 'nc' nor 'names' found in: {DATA_YAML}")
    if nc < 1:
        raise ValueError("Dataset must have at least one class")
    return nc


def make_custom_architecture(key: str) -> Path:
    """Generate YOLO11 YAML with individually controlled depth and width.

    The 'n' scale is used for ALL variants so differences in the framework's
    scale-dependent modules are not introduced unintentionally.
    """
    if not ULTRA_MODEL_YAML.is_file():
        raise FileNotFoundError(
            f"YOLO11 template not found in the installed Ultralytics: {ULTRA_MODEL_YAML}"
        )
    settings = EXPERIMENTS[key]
    config = yaml.safe_load(ULTRA_MODEL_YAML.read_text(encoding="utf-8"))
    config["nc"] = dataset_class_count()
    config["scales"] = {
        "n": [settings["depth"], settings["width"], 1024]
    }
    MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Start filename with yolo11n so Ultralytics recognises the n scale.
    output = MODEL_CONFIG_DIR / f"yolo11n_mma3001_{key}.yaml"
    if output.exists():
        old = yaml.safe_load(output.read_text(encoding="utf-8"))
        if old != config:
            raise FileExistsError(
                f"A DIFFERENT config already exists at {output}. "
                "Keep it for reproducibility; choose a new experiment name."
            )
    else:
        output.write_text(
            "# Generated from installed Ultralytics YOLO11 template.\n"
            + yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
    return output


def build_model(key: str, initialization: str):
    """Build custom detection architecture and optionally load matching pretrained weights."""
    config_path = make_custom_architecture(key)
    model = YOLO(str(config_path), task="detect")
    if initialization == "transfer":
        # Only compatible pretrained tensors will transfer for changed architectures.
        model.load("yolo11n.pt")
    return model, config_path


def preview() -> None:
    """Build models and inspect parameter counts WITHOUT training."""
    for key, settings in EXPERIMENTS.items():
        model, config_path = build_model(key, "scratch")
        n_params = sum(p.numel() for p in model.model.parameters())
        print(
            f"{key:12s} depth={settings['depth']:<4} "
            f"width={settings['width']:<7} "
            f"lr0={settings['lr0']:<7} "
            f"parameters={n_params:,} config={config_path.name}"
        )
    print("Preview only: NO model has been trained.")


def summary() -> None:
    """Read generated training results (no retraining) and save a comparison CSV."""
    rows = []
    for result_csv in sorted(RUNS_DIR.glob("*/results.csv")):
        with result_csv.open("r", newline="", encoding="utf-8-sig") as f:
            record_list = [
                {(k or "").strip(): v for k, v in row.items()}
                for row in csv.DictReader(f)
            ]
        if not record_list or any(
            metric not in record_list[-1] for metric in METRIC_COLS.values()
        ):
            continue
        last = record_list[-1]
        # A final-epoch result is explicit; best.pt may use a different fitness criterion.
        entry = {"run": result_csv.parent.name, "last_epoch": last.get("epoch", "")}
        entry.update({
            name: float(last[column]) for name, column in METRIC_COLS.items()
        })
        entry["peak_mAP50_95"] = max(
            float(row[METRIC_COLS["mAP50_95"]]) for row in record_list
        )
        metadata_path = result_csv.parent / "mma3001_arch_metadata.json"
        if metadata_path.is_file():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            entry.update({
                "depth": meta["depth"], "width": meta["width"],
                "lr0": meta["lr0"], "epochs_requested": meta["epochs_requested"],
                "initialization": meta["initialization"],
                "parameters": meta["parameters"],
                "train_seconds": round(meta["train_seconds"], 1),
            })
        rows.append(entry)
    if not rows:
        print(f"No run results.csv found under {RUNS_DIR}")
        return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUNS_DIR / "architecture_comparison.csv"
    columns = [
        "run", "last_epoch", "precision", "recall", "mAP50", "mAP50_95",
        "peak_mAP50_95", "depth", "width", "lr0", "epochs_requested",
        "initialization", "parameters", "train_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} validation result(s) to {output_path}")
    for entry in rows:
        print(f"{entry['run']}: mAP50-95={entry['mAP50_95']:.4f} ",
              f"recall={entry['recall']:.4f}")
    print("Compare ONLY runs with equivalent data, budget and initialization.")


def train_one(key: str, initialization: str, epochs: int) -> None:
    """Train one experiment in a new run folder and preserve full settings."""
    settings = EXPERIMENTS[key]
    name = f"arch_{key}_{initialization}_{epochs}ep"
    run_dir = RUNS_DIR / name
    if run_dir.exists():
        raise FileExistsError(
            f"Run already exists: {run_dir}\n"
            "To retain reproducibility, do not overwrite it."
        )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model, config_path = build_model(key, initialization)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training {name} on {device}")
    print(f"Depth={settings['depth']}, width={settings['width']}, "
          f"lr0={settings['lr0']}, initialization={initialization}")
    start = perf_counter()
    model.train(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        name=name,
        exist_ok=False,
        device=device,
        lr0=settings["lr0"],
        pretrained=(initialization == "transfer"),
        **{**TRAINING, "epochs": epochs},
    )
    elapsed = perf_counter() - start
    actual_params = sum(p.numel() for p in model.model.parameters())
    metadata = {
        "experiment": key,
        "depth": settings["depth"], "width": settings["width"],
        "lr0": settings["lr0"],
        "initialization": initialization,
        "initialization_source": (
            "yolo11n.pt (partial compatible weights only if dimensions change)"
            if initialization == "transfer" else "random (scratch)"
        ),
        "epochs_requested": epochs,
        "training": {**TRAINING, "epochs": epochs},
        "device": device,
        "architecture_yaml": str(config_path.relative_to(ROOT)),
        "dataset_yaml": str(DATA_YAML.relative_to(ROOT)),
        "parameters": actual_params,
        "train_seconds": elapsed,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "mma3001_arch_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Completed: {run_dir}")
    summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "preview", "summary"])
    parser.add_argument("--init", choices=["scratch", "transfer"], default="scratch")
    parser.add_argument("--epochs", type=int, default=TRAINING["epochs"])
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.experiment == "preview":
        preview()
    elif args.experiment == "summary":
        summary()
    else:
        train_one(args.experiment, args.init, args.epochs)
