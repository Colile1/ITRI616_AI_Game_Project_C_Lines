"""SnapshotMetadata dataclass and JSON persistence."""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TrainingHistoryEntry:
    games_trained: int
    win_rate_vs_random: Optional[float]
    elo_rating: Optional[float]


@dataclass
class SnapshotMetadata:
    # Identity
    version_id: str
    board_size: int
    weights_path: str

    # Provenance
    created_at: str
    games_trained: int
    gradient_steps: int
    epsilon_at_freeze: float
    parent_run_id: str
    parent_version_id: Optional[str]

    # Performance
    win_rate_vs_random: Optional[float]
    win_rate_vs_heuristic: Optional[float]
    win_rate_vs_self: Optional[float]
    elo_rating: Optional[float]
    mean_episode_length: Optional[float]

    # History
    training_history: list[TrainingHistoryEntry] = field(default_factory=list)

    # Presentation
    friendly_name: str = "Apprentice"
    difficulty_band: str = "novice"
    notes: str = ""

    # Compatibility
    model_version: int = 1


def save_metadata(meta: SnapshotMetadata, path: Path) -> None:
    d = asdict(meta)
    path = Path(path)
    path.write_text(json.dumps(d, indent=2))


def load_metadata(path: Path) -> SnapshotMetadata:
    d = json.loads(Path(path).read_text())
    history_raw = d.pop("training_history", [])
    history = [TrainingHistoryEntry(**e) for e in history_raw]
    return SnapshotMetadata(**d, training_history=history)
