"""MMA3001 demonstration entry point for YOLO11 pork-packaging defect detection.

Run from the project root:
    python main.py check
    python main.py summary
    python main.py demo
    python main.py demo --image "path/to/validation_image.jpg" --show
    python main.py tests

This is a *demonstration* helper, not a model-training script. It does not
retrain or modify the saved checkpoints. Predictions are object detections,
not a validated package-level Pass/Fail classification.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "results" / "experiments"
CHECKPOINT = EXPERIMENTS / "arch_baseline_scratch_30ep" / "weights" / "best.pt"
SUMMARY_CSV = EXPERIMENTS / "architecture_comparison.csv"
VALIDATION_IMAGES = ROOT / "data" / "raw" / "dataset_v1" / "valid" / "images"
TESTS_DIR = ROOT / "tests"
DOCS_DIR = ROOT / "docs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check() -> None:
    """Print a readiness checklist without training or importing PyTorch."""
    paths = {
        "Selected model checkpoint": CHECKPOINT,
        "Experiment comparison CSV": SUMMARY_CSV,
        "Validation image folder": VALIDATION_IMAGES,
        "Test directory": TESTS_DIR,
        "Documentation directory": DOCS_DIR,
    }
    print("MMA3001 demo readiness check")
    print("-" * 55)
    for label, path in paths.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"[{status:7}] {label}: {path.relative_to(ROOT)}")

    test_files = sorted(TESTS_DIR.glob("test_*.py")) if TESTS_DIR.is_dir() else []
    print(f"Automated test modules discovered: {len(test_files)}")
    if not CHECKPOINT.is_file():
        print("\nThe demo requires the project-trained baseline best.pt checkpoint.")
        print("Do not substitute the COCO-pretrained yolo11n.pt file.")


def summary() -> None:
    """Show peak validation mAP from the seven saved experiment summaries."""
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(
            f"Missing comparison file: {SUMMARY_CSV}\n"
            "Generate it first using:\n"
            "  python src/run_architecture_experiments.py summary"
        )

    with SUMMARY_CSV.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Experiment comparison CSV is empty")

    required = {"run", "peak_mAP50_95", "depth", "width", "lr0", "parameters"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Experiment CSV lacks required columns: {sorted(missing)}")

    rows.sort(key=lambda row: float(row["peak_mAP50_95"]), reverse=True)
    print("\nYOLO11 architecture study — peak VALIDATION performance")
    print("-" * 104)
    print(f"{'Experiment':36} {'Depth':>7} {'Width':>7} {'LR':>9} {'Params':>12} {'mAP50-95':>10}")
    print("-" * 104)
    for row in rows:
        run = row["run"].removeprefix("arch_").removesuffix("_scratch_30ep")
        print(
            f"{run:36} {row['depth']:>7} {row['width']:>7} "
            f"{float(row['lr0']):9.4f} {int(float(row['parameters'])):12,} "
            f"{100 * float(row['peak_mAP50_95']):9.2f}%"
        )
    print("\nNote: these are highest observed validation values, NOT held-out test scores.")
    print("Final held-out test metrics and the confusion matrix are in the project report.")


def choose_image(user_image: Path | None) -> Path:
    """Resolve an explicit image, or use the first Dataset V1 validation image."""
    if user_image is not None:
        path = user_image.expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return path

    if not VALIDATION_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Validation image directory not found: {VALIDATION_IMAGES}\n"
            "Use: python main.py demo --image path/to/image.jpg"
        )
    images = sorted(
        p for p in VALIDATION_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found in {VALIDATION_IMAGES}")
    return images[0]


def demo(image_arg: Path | None, confidence: float, show: bool) -> None:
    """Predict on one image with the previously trained model, then save an overlay."""
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"The trained checkpoint is missing: {CHECKPOINT}\n"
            "Use your saved experiment weights/best.pt, not yolo11n.pt."
        )

    image_path = choose_image(image_arg)
    print(f"\nModel: {CHECKPOINT.relative_to(ROOT)}")
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {confidence:.2f}")

    # Lazy import: 'summary' and 'tests' work even if Ultralytics isn't installed.
    from ultralytics import YOLO

    model = YOLO(str(CHECKPOINT))
    results = model.predict(source=str(image_path), conf=confidence, imgsz=640, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no prediction result")
    result = results[0]

    count = len(result.boxes) if result.boxes is not None else 0
    print(f"\nDefect detections above threshold: {count}")
    if result.boxes is not None:
        boxes = sorted(
            zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        for class_id, score in boxes:
            label = result.names.get(int(class_id), str(int(class_id)))
            print(f"  {label}: {score:.1%} confidence")

    # The saved image makes the demo viewable from VS Code without any GUI requirement.
    destination = ROOT / "results" / "demo"
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = destination / f"prediction_{image_path.stem}_{timestamp}.jpg"
    result.save(filename=str(output_path))
    print(f"\nAnnotated result saved: {output_path.relative_to(ROOT)}")
    print("IMPORTANT: This is defect detection, not validated package Pass/Fail accuracy.")
    if count == 0:
        print("No box above threshold does NOT prove the package is defect-free.")
    if show:
        from PIL import Image
        Image.open(output_path).show()


def tests() -> int:
    """Run pytest and save both a readable report and a JUnit XML report."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if not TESTS_DIR.is_dir() or not list(TESTS_DIR.glob("test_*.py")):
        print("No test_*.py modules found in tests/. Add tests before demonstrating them.")
        return 2

    xml_path = DOCS_DIR / "test_report.xml"
    text_path = DOCS_DIR / "test_report.txt"
    command = [
        sys.executable, "-m", "pytest", "tests", "-v",
        f"--junitxml={xml_path}",
    ]
    print("Running automated tests (does NOT train YOLO models)...\n")
    completed = subprocess.run(
        command, cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )
    print(completed.stdout)
    text_path.write_text(
        f"Command: {' '.join(command)}\n"
        f"Exit code: {completed.returncode}\n\n"
        + completed.stdout,
        encoding="utf-8",
    )
    print(f"Saved test log: {text_path.relative_to(ROOT)}")
    if xml_path.exists():
        print(f"Saved JUnit report: {xml_path.relative_to(ROOT)}")
    if completed.returncode:
        print("Tests did not all pass; inspect the output before committing results.")
    return completed.returncode


def main() -> int:
    """Provide safe presenter-facing commands for the completed project."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=False)
    sub.add_parser("check", help="Check required project files")
    sub.add_parser("summary", help="Display saved seven-model validation comparison")
    predict_parser = sub.add_parser("demo", help="Run inference with the trained baseline checkpoint")
    predict_parser.add_argument("--image", type=Path, help="Image to inspect (default: first validation image)")
    predict_parser.add_argument("--conf", type=float, default=0.10, help="Detection confidence threshold (default: 0.10)")
    predict_parser.add_argument("--show", action="store_true", help="Open annotated output image after saving it")
    sub.add_parser("tests", help="Run pytest and generate docs/test_report.txt and .xml")
    args = parser.parse_args()

    try:
        if args.command is None or args.command == "check":
            check()
            if args.command is None:
                print("\nTry: python main.py summary | demo | tests")
        elif args.command == "summary":
            summary()
        elif args.command == "demo":
            if not 0 < args.conf <= 1:
                parser.error("--conf must be in (0, 1]")
            demo(args.image, args.conf, args.show)
        elif args.command == "tests":
            return tests()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
