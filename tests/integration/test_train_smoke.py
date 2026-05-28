"""T-01 .. T-04: smoke tests for the training pipeline."""

import csv
from pathlib import Path
import pytest
from src.training.train import train
from src.config import MODELS_DIR, LOGS_DIR


@pytest.fixture(autouse=True)
def _tmp_models(tmp_path, monkeypatch):
    """Redirect model and log output to a temp directory."""
    import src.config as cfg
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(cfg, "REGISTRY_PATH", tmp_path / "models" / "registry.json")
    monkeypatch.setattr(cfg, "TRAINING_LOG_PATH", tmp_path / "logs" / "training_log.csv")
    import src.versioning.registry as reg
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "models" / "registry.json")
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path / "models")
    import src.training.snapshot as snap
    monkeypatch.setattr(snap, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    yield tmp_path


# ---------------------------------------------------------------------------
# T-01: train exits without exception
# ---------------------------------------------------------------------------
def test_T01_train_exits_cleanly(_tmp_models):
    train(board_size=8, mode="points_full", n_games=10)


# ---------------------------------------------------------------------------
# T-02: training log CSV is produced
# ---------------------------------------------------------------------------
def test_T02_csv_produced(_tmp_models):
    train(board_size=8, mode="points_full", n_games=10)
    logs = list((_tmp_models / "logs").glob("training_log_size8.csv"))
    assert logs, "No training log CSV produced"
    with open(logs[0]) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1


# ---------------------------------------------------------------------------
# T-03: at least one snapshot folder is created
# ---------------------------------------------------------------------------
def test_T03_snapshot_created(_tmp_models):
    # SNAPSHOT_INTERVAL is 1000 by default, so final snapshot is always written
    train(board_size=8, mode="points_full", n_games=10)
    size_dir = _tmp_models / "models" / "size_08"
    assert size_dir.exists(), "size_08 directory missing"
    snap_dirs = [d for d in size_dir.iterdir() if d.is_dir()]
    assert len(snap_dirs) >= 1


# ---------------------------------------------------------------------------
# T-04: registry.json is populated
# ---------------------------------------------------------------------------
def test_T04_registry_populated(_tmp_models):
    import json
    train(board_size=8, mode="points_full", n_games=10)
    reg_path = _tmp_models / "models" / "registry.json"
    assert reg_path.exists()
    data = json.loads(reg_path.read_text())
    assert "size_08" in data
    assert len(data["size_08"]) >= 1
