"""Snapshot registry — atomic JSON CRUD over models/registry.json."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

from src.versioning.metadata import SnapshotMetadata, TrainingHistoryEntry
from src.config import REGISTRY_PATH, MODELS_DIR


def _read_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text())


def _write_registry(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, REGISTRY_PATH)  # atomic on POSIX and Windows


def _meta_to_entry(meta: SnapshotMetadata) -> dict:
    return {
        "version_id": meta.version_id,
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
    return SnapshotMetadata(
        version_id=entry["version_id"],
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


def _size_key(board_size: int) -> str:
    return f"size_{board_size:02d}"


def register(meta: SnapshotMetadata) -> None:
    data = _read_registry()
    key = _size_key(meta.board_size)
    entries = data.get(key, [])
    entries.append(_meta_to_entry(meta))
    data[key] = entries
    _write_registry(data)


def list_by_size(board_size: int) -> list[SnapshotMetadata]:
    data = _read_registry()
    key = _size_key(board_size)
    entries = data.get(key, [])
    metas = [_entry_to_meta(e, board_size) for e in entries]
    return sorted(metas, key=lambda m: m.games_trained)


def get(board_size: int, version_id: str) -> SnapshotMetadata:
    for meta in list_by_size(board_size):
        if meta.version_id == version_id:
            return meta
    raise KeyError(f"Snapshot {version_id} not found for board_size={board_size}")


def evict_oldest(board_size: int, keep_n: int) -> list[str]:
    data = _read_registry()
    key = _size_key(board_size)
    entries = data.get(key, [])
    if len(entries) <= keep_n:
        return []
    sorted_entries = sorted(entries, key=lambda e: e.get("games_trained", 0))
    evicted_ids = [e["version_id"] for e in sorted_entries[:-keep_n]]
    data[key] = sorted_entries[-keep_n:]
    _write_registry(data)
    return evicted_ids


def next_version_id(board_size: int) -> str:
    existing = list_by_size(board_size)
    nums = []
    for m in existing:
        try:
            nums.append(int(m.version_id.split("_")[1]))
        except (IndexError, ValueError):
            pass
    n = (max(nums) + 1) if nums else 1
    return f"gen_{n:03d}"
