"""Snapshot registry — per-run registry.json files, atomic writes."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

from src.versioning.metadata import SnapshotMetadata, TrainingHistoryEntry
from src.config import MODELS_DIR


# ---------------------------------------------------------------------------
# Run-folder helpers
# ---------------------------------------------------------------------------

def _run_dir(board_size: int, run_id: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"size_{board_size:02d}" / run_id


def _registry_path(board_size: int, run_id: str, models_dir: Path = MODELS_DIR) -> Path:
    return _run_dir(board_size, run_id, models_dir) / "registry.json"


def next_run_id(board_size: int, models_dir: Path = MODELS_DIR) -> str:
    """Return the next auto-incremented run_NNN string for a given board size."""
    size_dir = models_dir / f"size_{board_size:02d}"
    if not size_dir.exists():
        return "run_001"
    existing = sorted(
        p.name for p in size_dir.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )
    if not existing:
        return "run_001"
    last = existing[-1]
    try:
        n = int(last.split("_")[1]) + 1
    except (IndexError, ValueError):
        n = len(existing) + 1
    return f"run_{n:03d}"


def next_version_id(board_size: int, run_id: str, models_dir: Path = MODELS_DIR) -> str:
    """Return next gen_NNN, derived from the registry.json (not filesystem scan)."""
    existing = list_by_size_run(board_size, run_id, models_dir)
    if not existing:
        return "gen_001"
    nums = []
    for m in existing:
        try:
            nums.append(int(m.version_id.split("_")[1]))
        except (IndexError, ValueError):
            pass
    n = (max(nums) + 1) if nums else 1
    return f"gen_{n:03d}"


# ---------------------------------------------------------------------------
# Registry read / write
# ---------------------------------------------------------------------------

def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _write(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    os.replace(tmp, path)


def _meta_to_entry(meta: SnapshotMetadata) -> dict:
    return {
        "version_id": meta.version_id,
        "run_id": meta.run_id,
        "weights_path": meta.weights_path,
        "friendly_name": meta.friendly_name,
        "difficulty_band": meta.difficulty_band,
        "games_trained": meta.games_trained,
        "win_rate_vs_random": meta.win_rate_vs_random,
        "win_rate_vs_heuristic": meta.win_rate_vs_heuristic,
        "elo_rating": meta.elo_rating,
        "created_at": meta.created_at,
        "model_version": meta.model_version,
    }


def _entry_to_meta(entry: dict, board_size: int) -> SnapshotMetadata:
    from src.versioning.metadata import SnapshotMetadata
    return SnapshotMetadata(
        version_id=entry["version_id"],
        run_id=entry.get("run_id", "run_001"),
        board_size=board_size,
        weights_path=entry["weights_path"],
        created_at=entry.get("created_at", ""),
        games_trained=entry.get("games_trained", 0),
        gradient_steps=entry.get("gradient_steps", 0),
        epsilon_at_freeze=entry.get("epsilon_at_freeze", 0.0),
        parent_run_id=entry.get("parent_run_id", ""),
        parent_version_id=entry.get("parent_version_id"),
        win_rate_vs_random=entry.get("win_rate_vs_random"),
        win_rate_vs_heuristic=entry.get("win_rate_vs_heuristic"),
        win_rate_vs_self=entry.get("win_rate_vs_self"),
        elo_rating=entry.get("elo_rating"),
        mean_episode_length=entry.get("mean_episode_length"),
        friendly_name=entry.get("friendly_name", "Apprentice"),
        difficulty_band=entry.get("difficulty_band", "novice"),
        notes=entry.get("notes", ""),
        model_version=entry.get("model_version", 1),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(meta: SnapshotMetadata, models_dir: Path = MODELS_DIR) -> None:
    path = _registry_path(meta.board_size, meta.run_id, models_dir)
    entries = _read(path)
    entries.append(_meta_to_entry(meta))
    _write(path, entries)


def list_by_size_run(
    board_size: int, run_id: str, models_dir: Path = MODELS_DIR
) -> list[SnapshotMetadata]:
    path = _registry_path(board_size, run_id, models_dir)
    return sorted(
        [_entry_to_meta(e, board_size) for e in _read(path)],
        key=lambda m: m.games_trained,
    )


def list_all_runs(board_size: int, models_dir: Path = MODELS_DIR) -> dict[str, list[SnapshotMetadata]]:
    """Return {run_id: [SnapshotMetadata, ...]} for all runs of a given board size."""
    size_dir = models_dir / f"size_{board_size:02d}"
    result: dict[str, list[SnapshotMetadata]] = {}
    if not size_dir.exists():
        return result
    for run_dir in sorted(size_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            result[run_dir.name] = list_by_size_run(board_size, run_dir.name, models_dir)
    return result


def list_by_size(board_size: int, models_dir: Path = MODELS_DIR) -> list[SnapshotMetadata]:
    """Return all snapshots across all runs for a board size (for UI level-select)."""
    all_runs = list_all_runs(board_size, models_dir)
    merged: list[SnapshotMetadata] = []
    for snaps in all_runs.values():
        merged.extend(snaps)
    return sorted(merged, key=lambda m: (m.run_id, m.games_trained))


def get(
    board_size: int, version_id: str,
    run_id: Optional[str] = None,
    models_dir: Path = MODELS_DIR,
) -> SnapshotMetadata:
    if run_id:
        for m in list_by_size_run(board_size, run_id, models_dir):
            if m.version_id == version_id:
                return m
    else:
        for m in list_by_size(board_size, models_dir):
            if m.version_id == version_id:
                return m
    raise KeyError(f"Snapshot {version_id} not found for size={board_size} run={run_id}")


def evict_oldest(
    board_size: int, run_id: str, keep_n: int, models_dir: Path = MODELS_DIR
) -> list[str]:
    path = _registry_path(board_size, run_id, models_dir)
    entries = _read(path)
    if len(entries) <= keep_n:
        return []
    sorted_entries = sorted(entries, key=lambda e: e.get("games_trained", 0))
    evicted_ids = [e["version_id"] for e in sorted_entries[:-keep_n]]
    _write(path, sorted_entries[-keep_n:])
    return evicted_ids
