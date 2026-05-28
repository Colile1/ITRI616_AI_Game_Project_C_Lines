"""T-01 .. T-04: smoke tests for the training pipeline (run-folder aware)."""

import csv
import json
from pathlib import Path
import pytest
from src.training.train import train
import src.config as cfg


@pytest.fixture(autouse=True)
def _tmp_dirs(tmp_path, monkeypatch):
    """Redirect all model and result output to a temp directory."""
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cfg, "REGISTRY_PATH", tmp_path / "models" / "registry.json")
    import src.versioning.registry as reg
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path / "models")
    import src.training.snapshot as snap
    monkeypatch.setattr(snap, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    yield tmp_path


# ---------------------------------------------------------------------------
# T-01: train exits without exception
# ---------------------------------------------------------------------------
def test_T01_train_exits_cleanly(_tmp_dirs):
    train(board_size=8, mode="points_full", n_games=10, run_id="run_001")


# ---------------------------------------------------------------------------
# T-02: training log CSV is produced in run-specific folder
# ---------------------------------------------------------------------------
def test_T02_csv_produced(_tmp_dirs):
    train(board_size=8, mode="points_full", n_games=10, run_id="run_001")
    log = _tmp_dirs / "results" / "size_08" / "run_001" / "training_log.csv"
    assert log.exists(), f"Expected {log}"
    with open(log) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    # New columns present
    assert "win_rate_vs_heuristic" in rows[0]


# ---------------------------------------------------------------------------
# T-03: snapshot folder created under run directory
# ---------------------------------------------------------------------------
def test_T03_snapshot_created(_tmp_dirs):
    train(board_size=8, mode="points_full", n_games=10, run_id="run_001")
    run_dir = _tmp_dirs / "models" / "size_08" / "run_001"
    assert run_dir.exists(), "models/size_08/run_001/ missing"
    snap_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")]
    assert len(snap_dirs) >= 1


# ---------------------------------------------------------------------------
# T-04: per-run registry.json is populated
# ---------------------------------------------------------------------------
def test_T04_registry_populated(_tmp_dirs):
    train(board_size=8, mode="points_full", n_games=10, run_id="run_001")
    reg_path = _tmp_dirs / "models" / "size_08" / "run_001" / "registry.json"
    assert reg_path.exists(), f"Expected {reg_path}"
    data = json.loads(reg_path.read_text())
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["run_id"] == "run_001"
