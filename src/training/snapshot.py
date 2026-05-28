"""Freeze, load, and clone DQN agents as versioned snapshots."""

from __future__ import annotations
import copy
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from src.agents.dqn_agent import DQNAgent
from src.config import MODELS_DIR, FRIENDLY_NAMES, TRAINING_GAMES, MODEL_VERSION
from src.versioning.metadata import SnapshotMetadata, TrainingHistoryEntry, save_metadata
from src.versioning.registry import register, next_version_id


def _difficulty_band(games_trained: int, total_games: int) -> str:
    frac = games_trained / max(total_games, 1)
    if frac >= 1.0:
        return "master"
    elif frac >= 0.75:
        return "hard"
    elif frac >= 0.50:
        return "medium"
    elif frac >= 0.25:
        return "easy"
    return "novice"


def _friendly_name(band: str, existing_count: int) -> str:
    names = FRIENDLY_NAMES.get(band, ["Agent"])
    return names[existing_count % len(names)]


def freeze(
    agent: DQNAgent,
    game_idx: int,
    eval_stats: dict,
    board_size: int,
    parent_run_id: Optional[str] = None,
    parent_version_id: Optional[str] = None,
    gradient_steps: int = 0,
    history: Optional[list[TrainingHistoryEntry]] = None,
) -> SnapshotMetadata:
    """Serialise *agent* to disk and register the snapshot. Returns metadata."""
    version_id = next_version_id(board_size)
    size_dir = MODELS_DIR / f"size_{board_size:02d}" / version_id
    size_dir.mkdir(parents=True, exist_ok=True)

    weights_path = size_dir / "weights.pt"
    torch.save(agent.state_dict(), weights_path)

    band = _difficulty_band(game_idx, TRAINING_GAMES)
    existing_count = len([
        p for p in (MODELS_DIR / f"size_{board_size:02d}").iterdir()
        if p.is_dir()
    ]) - 1  # -1 because current dir was just created

    new_entry = TrainingHistoryEntry(
        games_trained=game_idx,
        win_rate_vs_random=eval_stats.get("win_rate_vs_random"),
        elo_rating=eval_stats.get("elo_rating"),
    )
    full_history = list(history or []) + [new_entry]

    meta = SnapshotMetadata(
        version_id=version_id,
        board_size=board_size,
        weights_path=str(weights_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        games_trained=game_idx,
        gradient_steps=gradient_steps,
        epsilon_at_freeze=agent.epsilon,
        parent_run_id=parent_run_id or str(uuid.uuid4()),
        parent_version_id=parent_version_id,
        win_rate_vs_random=eval_stats.get("win_rate_vs_random"),
        win_rate_vs_heuristic=eval_stats.get("win_rate_vs_heuristic"),
        win_rate_vs_self=eval_stats.get("win_rate_vs_self"),
        elo_rating=eval_stats.get("elo_rating"),
        mean_episode_length=eval_stats.get("mean_episode_length"),
        training_history=full_history,
        friendly_name=_friendly_name(band, max(existing_count, 0)),
        difficulty_band=band,
        model_version=MODEL_VERSION,
    )

    save_metadata(meta, size_dir / "metadata.json")
    register(meta)
    return meta


def load_snapshot(board_size: int, version_id: str) -> DQNAgent:
    """Load a frozen snapshot as a greedy eval-only agent."""
    from src.versioning.registry import get
    meta = get(board_size, version_id)
    agent = DQNAgent(board_size, eval_only=True)
    sd = torch.load(meta.weights_path, map_location="cpu", weights_only=True)
    agent.load_state_dict(sd)
    agent.set_epsilon(0.0)
    return agent


def clone_agent(agent: DQNAgent) -> DQNAgent:
    """Deep-copy a training agent for use as self-play opponent."""
    new_agent = DQNAgent(agent.board_size, eval_only=True)
    new_agent.load_state_dict(copy.deepcopy(agent.state_dict()))
    new_agent.set_epsilon(0.0)
    return new_agent
