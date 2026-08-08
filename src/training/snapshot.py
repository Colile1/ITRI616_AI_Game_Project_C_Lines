"""Freeze, load, and clone DQN agents as versioned snapshots (run-aware)."""

from __future__ import annotations
import copy
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from src.agents.dqn_agent import DQNAgent
from src.config import (
    MODELS_DIR, FRIENDLY_NAMES, MODEL_VERSION,
    STATE_CHANNELS_V2, NETWORK_ARCH,
)
from src.versioning.metadata import SnapshotMetadata, TrainingHistoryEntry, save_metadata
from src.versioning.registry import register, next_version_id


def _difficulty_band(eval_stats: dict) -> str:
    """Assign band from measured win rates, not game-count fraction.

    Primary signal: win rate vs heuristic (harder baseline = more meaningful).
    Fallback: win rate vs random (always present).
    """
    wr_h = eval_stats.get("win_rate_vs_heuristic")
    wr_r = eval_stats.get("win_rate_vs_random") or 0.0
    wr = wr_h if (wr_h is not None) else wr_r
    if wr >= 0.90:   return "master"
    elif wr >= 0.75: return "hard"
    elif wr >= 0.50: return "medium"
    elif wr >= 0.25: return "easy"
    return "novice"


def _friendly_name(band: str, existing_count: int) -> str:
    names = FRIENDLY_NAMES.get(band, ["Agent"])
    return names[existing_count % len(names)]


def freeze(
    agent: DQNAgent,
    game_idx: int,
    eval_stats: dict,
    board_size: int,
    run_id: str,
    parent_run_id: Optional[str] = None,
    parent_version_id: Optional[str] = None,
    gradient_steps: int = 0,
    history: Optional[list[TrainingHistoryEntry]] = None,
    models_dir: Path = MODELS_DIR,
    human_games_seen: int = 0,
) -> SnapshotMetadata:
    """Serialise *agent* to disk under models/size_NN/run_NNN/gen_NNN/ and register."""
    version_id = next_version_id(board_size, run_id, models_dir)
    snap_dir = models_dir / f"size_{board_size:02d}" / run_id / version_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    weights_path = snap_dir / "weights.pt"
    torch.save(agent.state_dict(), weights_path)

    band = _difficulty_band(eval_stats)
    from src.versioning.registry import list_by_size_run
    run_snaps = list_by_size_run(board_size, run_id, models_dir)
    existing_count = max(0, len(run_snaps))

    new_entry = TrainingHistoryEntry(
        games_trained=game_idx,
        win_rate_vs_random=eval_stats.get("win_rate_vs_random"),
        elo_rating=eval_stats.get("elo_rating"),
    )
    full_history = list(history or []) + [new_entry]

    meta = SnapshotMetadata(
        version_id=version_id,
        run_id=run_id,
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
        friendly_name=_friendly_name(band, existing_count),
        difficulty_band=band,
        model_version=MODEL_VERSION,
        state_channels=getattr(agent, "in_channels", STATE_CHANNELS_V2),
        network_arch=getattr(agent, "network_arch", NETWORK_ARCH),
        human_games_seen=human_games_seen,
    )

    save_metadata(meta, snap_dir / "metadata.json")
    register(meta, models_dir)
    return meta


def load_snapshot(
    board_size: int, version_id: str,
    run_id: Optional[str] = None,
    models_dir: Path = MODELS_DIR,
) -> DQNAgent:
    """Load a frozen snapshot as a greedy eval-only agent.

    The registry entry is only a summary — it carries no architecture fields, so
    trusting it would build a 6-channel plain network for every snapshot and then
    fail to load 10-channel ResNet weights.  The per-snapshot metadata.json is
    authoritative and is preferred whenever it is present.
    """
    from src.versioning.metadata import load_metadata
    from src.versioning.registry import get

    meta = get(board_size, version_id, run_id=run_id, models_dir=models_dir)
    snap_dir = models_dir / f"size_{board_size:02d}" / meta.run_id / meta.version_id

    sidecar = snap_dir / "metadata.json"
    if sidecar.exists():
        try:
            meta = load_metadata(sidecar)
        except (OSError, ValueError, TypeError):
            pass   # summary metadata is better than nothing

    in_channels = getattr(meta, "state_channels", 6)
    network_arch = getattr(meta, "network_arch", "plain_v1")

    # Recorded paths are absolute, so they break if the project is moved; fall
    # back to the snapshot directory we just derived.
    weights_path = Path(meta.weights_path)
    if not weights_path.exists():
        weights_path = snap_dir / "weights.pt"

    agent = DQNAgent(board_size, eval_only=True, in_channels=in_channels, network_arch=network_arch)
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    agent.load_state_dict(sd)
    agent.set_epsilon(0.0)
    return agent


def clone_agent(agent: DQNAgent) -> DQNAgent:
    """Deep-copy a training agent for use as self-play opponent."""
    in_ch = getattr(agent, "in_channels", STATE_CHANNELS_V2)
    arch  = getattr(agent, "network_arch", NETWORK_ARCH)
    new_agent = DQNAgent(agent.board_size, eval_only=True, in_channels=in_ch, network_arch=arch)
    new_agent.load_state_dict(copy.deepcopy(agent.state_dict()))
    new_agent.set_epsilon(0.0)
    return new_agent
