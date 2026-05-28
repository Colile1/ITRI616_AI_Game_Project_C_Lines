"""V-01 .. V-09: versioning module tests."""

import json
import os
import pytest
from pathlib import Path


@pytest.fixture()
def registry_env(tmp_path, monkeypatch):
    """Redirect models dir and registry path to tmp_path."""
    import src.config as cfg
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "REGISTRY_PATH", tmp_path / "models" / "registry.json")
    import src.versioning.registry as reg
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "models" / "registry.json")
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir()
    return tmp_path


def _make_meta(board_size=8, games=100, vid="gen_001"):
    from src.versioning.metadata import SnapshotMetadata
    return SnapshotMetadata(
        version_id=vid,
        board_size=board_size,
        weights_path=f"models/size_{board_size:02d}/{vid}/weights.pt",
        created_at="2026-05-28T12:00:00Z",
        games_trained=games,
        gradient_steps=500,
        epsilon_at_freeze=0.3,
        parent_run_id="abc123",
        parent_version_id=None,
        win_rate_vs_random=0.62,
        win_rate_vs_heuristic=None,
        win_rate_vs_self=None,
        elo_rating=950.0,
        mean_episode_length=40.0,
        friendly_name="Apprentice",
        difficulty_band="novice",
    )


# ---------------------------------------------------------------------------
# V-01: save_metadata / load_metadata round-trip
# ---------------------------------------------------------------------------
def test_V01_metadata_roundtrip(tmp_path):
    from src.versioning.metadata import save_metadata, load_metadata
    meta = _make_meta()
    p = tmp_path / "metadata.json"
    save_metadata(meta, p)
    loaded = load_metadata(p)
    assert loaded.version_id == meta.version_id
    assert loaded.board_size == meta.board_size
    assert loaded.win_rate_vs_random == pytest.approx(meta.win_rate_vs_random)


# ---------------------------------------------------------------------------
# V-02: metadata JSON is valid JSON
# ---------------------------------------------------------------------------
def test_V02_metadata_valid_json(tmp_path):
    from src.versioning.metadata import save_metadata
    meta = _make_meta()
    p = tmp_path / "metadata.json"
    save_metadata(meta, p)
    data = json.loads(p.read_text())
    assert data["version_id"] == "gen_001"


# ---------------------------------------------------------------------------
# V-03: registry.register persists to disk
# ---------------------------------------------------------------------------
def test_V03_register_persists(registry_env):
    from src.versioning.registry import register
    meta = _make_meta()
    register(meta)
    reg_path = registry_env / "models" / "registry.json"
    assert reg_path.exists()
    data = json.loads(reg_path.read_text())
    assert "size_08" in data


# ---------------------------------------------------------------------------
# V-04: list_by_size returns registered snapshots sorted by games_trained
# ---------------------------------------------------------------------------
def test_V04_list_sorted(registry_env):
    from src.versioning.registry import register, list_by_size
    register(_make_meta(games=1000, vid="gen_001"))
    register(_make_meta(games=500, vid="gen_002"))
    metas = list_by_size(8)
    assert metas[0].games_trained <= metas[1].games_trained


# ---------------------------------------------------------------------------
# V-05: get() retrieves correct snapshot
# ---------------------------------------------------------------------------
def test_V05_get_by_id(registry_env):
    from src.versioning.registry import register, get
    register(_make_meta(vid="gen_001", games=100))
    register(_make_meta(vid="gen_002", games=200))
    m = get(8, "gen_002")
    assert m.version_id == "gen_002"
    assert m.games_trained == 200


# ---------------------------------------------------------------------------
# V-06: get() raises KeyError for missing version
# ---------------------------------------------------------------------------
def test_V06_get_missing_raises(registry_env):
    from src.versioning.registry import get
    with pytest.raises(KeyError):
        get(8, "gen_999")


# ---------------------------------------------------------------------------
# V-07: evict_oldest keeps only keep_n entries
# ---------------------------------------------------------------------------
def test_V07_evict_oldest(registry_env):
    from src.versioning.registry import register, evict_oldest, list_by_size
    for i in range(5):
        register(_make_meta(games=(i+1)*100, vid=f"gen_{i+1:03d}"))
    evicted = evict_oldest(8, keep_n=3)
    assert len(evicted) == 2
    remaining = list_by_size(8)
    assert len(remaining) == 3


# ---------------------------------------------------------------------------
# V-08: atomic write — registry not corrupted on re-read
# ---------------------------------------------------------------------------
def test_V08_atomic_write(registry_env):
    from src.versioning.registry import register, list_by_size
    for i in range(3):
        register(_make_meta(games=(i+1)*100, vid=f"gen_{i+1:03d}"))
    metas = list_by_size(8)
    assert len(metas) == 3


# ---------------------------------------------------------------------------
# V-09: next_version_id increments monotonically
# ---------------------------------------------------------------------------
def test_V09_version_id_monotonic(registry_env):
    from src.versioning.registry import register, next_version_id
    register(_make_meta(vid="gen_001", games=100))
    register(_make_meta(vid="gen_002", games=200))
    nxt = next_version_id(8)
    assert nxt == "gen_003"
