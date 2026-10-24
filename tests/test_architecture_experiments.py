r"""Fast, data-free regression tests for the MMA3001 YOLO11 experiment runner.

Run from the project root:
    .\.venv\Scripts\python.exe -m pytest tests -v

These tests NEVER call model.train() and do not require your raw images or checkpoints.
They check configuration handling and experiment-results summarisation rather than
model predictive accuracy (which is assessed separately in the project report).
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "src" / "run_architecture_experiments.py"


@pytest.fixture(scope="module")
def runner():
    """Load the real project script without executing its command-line entry point."""
    assert SCRIPT_PATH.is_file(), f"Missing experiment script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("mma3001_experiment_runner", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_configurations_are_complete(runner):
    """All seven experiment configurations exist and have valid numeric settings."""
    assert set(runner.EXPERIMENTS) == {
        "baseline", "narrower", "wider", "deeper", "lr_low", "lr_high", "combined"
    }
    for settings in runner.EXPERIMENTS.values():
        assert settings["depth"] > 0
        assert settings["width"] > 0
        assert settings["lr0"] > 0


def test_controlled_experiment_design(runner):
    """Width/depth and learning-rate comparisons isolate intended variables."""
    base = runner.EXPERIMENTS["baseline"]
    for name in ("narrower", "wider"):
        assert runner.EXPERIMENTS[name]["depth"] == base["depth"]
        assert runner.EXPERIMENTS[name]["lr0"] == base["lr0"]
        assert runner.EXPERIMENTS[name]["width"] != base["width"]
    for name in ("lr_low", "lr_high"):
        assert runner.EXPERIMENTS[name]["depth"] == base["depth"]
        assert runner.EXPERIMENTS[name]["width"] == base["width"]
        assert runner.EXPERIMENTS[name]["lr0"] != base["lr0"]
    assert runner.TRAINING["optimizer"] == "AdamW"
    assert runner.TRAINING["epochs"] == 30
    assert runner.TRAINING["seed"] == 42


def test_dataset_class_count_reads_yaml(runner, monkeypatch, tmp_path):
    """Dataset class count is read from the configuration, not hard-coded."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("nc: 4\nnames: [a, b, c, d]\n", encoding="utf-8")
    monkeypatch.setattr(runner, "DATA_YAML", data_file)
    assert runner.dataset_class_count() == 4


def test_dataset_class_count_rejects_missing_file(runner, monkeypatch, tmp_path):
    """A missing dataset configuration produces a clear error."""
    monkeypatch.setattr(runner, "DATA_YAML", tmp_path / "not_here.yaml")
    with pytest.raises(FileNotFoundError, match="Missing dataset file"):
        runner.dataset_class_count()


def test_dataset_class_count_rejects_invalid_classes(runner, monkeypatch, tmp_path):
    """A dataset with no classes is rejected instead of silently accepted."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("nc: 0\n", encoding="utf-8")
    monkeypatch.setattr(runner, "DATA_YAML", data_file)
    with pytest.raises(ValueError, match="at least one class"):
        runner.dataset_class_count()


def test_generated_model_yaml_scales_width_and_depth(runner, monkeypatch, tmp_path):
    """The generated YAML contains the requested scales and four output classes."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("nc: 4\nnames: [a, b, c, d]\n", encoding="utf-8")
    model_template = tmp_path / "yolo11.yaml"
    model_template.write_text(
        "nc: 80\nscales:\n  n: [0.5, 0.25, 1024]\nbackbone: []\nhead: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "DATA_YAML", data_file)
    monkeypatch.setattr(runner, "ULTRA_MODEL_YAML", model_template)
    monkeypatch.setattr(runner, "MODEL_CONFIG_DIR", tmp_path / "generated")
    output = runner.make_custom_architecture("wider")
    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert generated["nc"] == 4
    assert generated["scales"]["n"] == [0.5, 0.375, 1024]
    assert output.name == "yolo11n_mma3001_wider.yaml"
    # Re-running with identical settings must preserve the same YAML.
    assert runner.make_custom_architecture("wider") == output
    # But a changed file must not be silently overwritten.
    output.write_text("nc: 99\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="DIFFERENT config"):
        runner.make_custom_architecture("wider")


def test_summary_saves_peak_and_last_epoch_separately(runner, monkeypatch, tmp_path):
    """Results summary distinguishes highest validation mAP from last-epoch mAP."""
    runs_dir = tmp_path / "experiments"
    run_dir = runs_dir / "sample_run"
    run_dir.mkdir(parents=True)
    result_csv = run_dir / "results.csv"
    columns = ["epoch", *runner.METRIC_COLS.values()]
    sample_rows = [
        {"epoch": 1, "metrics/precision(B)": 0.50, "metrics/recall(B)": 0.20,
         "metrics/mAP50(B)": 0.30, "metrics/mAP50-95(B)": 0.15},
        {"epoch": 2, "metrics/precision(B)": 0.55, "metrics/recall(B)": 0.22,
         "metrics/mAP50(B)": 0.35, "metrics/mAP50-95(B)": 0.21},
        {"epoch": 3, "metrics/precision(B)": 0.52, "metrics/recall(B)": 0.19,
         "metrics/mAP50(B)": 0.31, "metrics/mAP50-95(B)": 0.16},
    ]
    with result_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sample_rows)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    runner.summary()  # Processes synthetic CSV files only; NEVER trains a model.
    summary_file = runs_dir / "architecture_comparison.csv"
    assert summary_file.is_file()
    with summary_file.open("r", newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    assert len(records) == 1
    assert records[0]["run"] == "sample_run"
    assert float(records[0]["mAP50_95"]) == pytest.approx(0.16)
    assert float(records[0]["peak_mAP50_95"]) == pytest.approx(0.21)
