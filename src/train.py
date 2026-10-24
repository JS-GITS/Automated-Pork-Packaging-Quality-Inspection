"""
Initial YOLO11n training smoke test for pork-packaging defect detection.

Trains a COCO-pretrained YOLO11n model for 10 epochs to verify
the dataset and training workflow.

This is NOT the final selected model. The seven controlled
architecture experiments are implemented in
run_architecture_experiments.py.

MMA3001 Project
"""

from pathlib import Path

import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

# Root directory:
# MMA3001_Project/
ROOT = Path(__file__).resolve().parents[1]

# Dataset configuration supplied by Roboflow
DATA_YAML = ROOT / "data" / "raw" / "dataset_v1" / "data.yaml"

# Where YOLO training results will be stored
RESULTS_DIR = ROOT / "results" / "experiments"


def train_model():
    """Train a YOLO11n object-detection model on Dataset V1."""

    # Make sure the dataset configuration exists
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Could not find dataset configuration:\n{DATA_YAML}"
        )

    # Use NVIDIA GPU if CUDA is available
    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("MMA3001 Pork Packaging Detection")
    print("=" * 60)
    print(f"Dataset: {DATA_YAML}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU unavailable - training on CPU")

    print("=" * 60)

    # Load pretrained YOLO11 Nano model
    model = YOLO("yolo11n.pt")

    # Train model
    results = model.train(
        data=str(DATA_YAML),

        # -------------------------
        # Training settings
        # -------------------------
        epochs=10,          # smoke test first
        imgsz=640,          # input image resolution
        batch=4,            # suitable starting point for 4 GB VRAM

        # -------------------------
        # Hardware
        # -------------------------
        device=device,
        workers=2,

        # -------------------------
        # Reproducibility
        # -------------------------
        seed=42,
        deterministic=True,

        # -------------------------
        # Output
        # -------------------------
        project=str(RESULTS_DIR),
        name="v1_yolo11n_smoke_test",
        exist_ok=True,

        # -------------------------
        # Other settings
        # -------------------------
        pretrained=True,
        plots=True,
        verbose=True,
    )

    print("\nTraining complete.")
    print(f"Results saved to: {results.save_dir}")

    return results


if __name__ == "__main__":
    train_model()