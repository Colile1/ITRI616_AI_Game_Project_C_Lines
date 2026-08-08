"""Player statistics — a small JSON file accumulating results across sessions.

Only games against the AI count towards win/loss records; hot-seat games are
counted separately because there is no "you" in them.  Every failure path here
is silent by design: statistics must never be able to break a game.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.config import PLAYER_STATS_PATH, PLAYER_1, DIFFICULTY_BANDS

STATS_SCHEMA = 1

RESULT_WIN = "win"
RESULT_LOSS = "loss"
RESULT_DRAW = "draw"

BAND_ORDER = list(DIFFICULTY_BANDS.keys())   # novice → master


def _empty_tally() -> dict[str, int]:
    return {RESULT_WIN: 0, RESULT_LOSS: 0, RESULT_DRAW: 0}


@dataclass
class PlayerStats:
    games_vs_ai: int = 0
    games_hotseat: int = 0

    wins: int = 0
    losses: int = 0
    draws: int = 0

    current_streak: int = 0        # consecutive wins; reset by a loss or draw
    best_streak: int = 0

    moves_played: int = 0
    fastest_win_moves: Optional[int] = None
    total_play_sec: float = 0.0

    by_band: dict[str, dict[str, int]] = field(default_factory=dict)
    by_mode: dict[str, dict[str, int]] = field(default_factory=dict)

    schema: int = STATS_SCHEMA

    # ------------------------------------------------------------------
    @property
    def decided_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> Optional[float]:
        """Win rate against the AI, or None with no games played yet."""
        total = self.decided_games
        return (self.wins / total) if total else None

    def band_tally(self, band: str) -> dict[str, int]:
        return self.by_band.get(band, _empty_tally())

    def mode_tally(self, mode: str) -> dict[str, int]:
        return self.by_mode.get(mode, _empty_tally())

    def best_band_beaten(self) -> Optional[str]:
        """The hardest difficulty band with at least one win."""
        for band in reversed(BAND_ORDER):
            if self.by_band.get(band, {}).get(RESULT_WIN, 0) > 0:
                return band
        return None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def result_for_human(winner: Optional[int], human_player: int = PLAYER_1) -> str:
    if winner is None:
        return RESULT_DRAW
    return RESULT_WIN if winner == human_player else RESULT_LOSS


def record_game(
    stats: PlayerStats,
    *,
    vs_ai: bool,
    winner: Optional[int],
    mode: str,
    move_count: int,
    duration_sec: float = 0.0,
    band: Optional[str] = None,
    human_player: int = PLAYER_1,
) -> PlayerStats:
    """Fold one finished game into *stats* (mutated in place and returned)."""
    stats.moves_played += max(0, int(move_count))
    stats.total_play_sec += max(0.0, float(duration_sec))

    if not vs_ai:
        stats.games_hotseat += 1
        return stats

    stats.games_vs_ai += 1
    outcome = result_for_human(winner, human_player)

    if outcome == RESULT_WIN:
        stats.wins += 1
        stats.current_streak += 1
        stats.best_streak = max(stats.best_streak, stats.current_streak)
        if stats.fastest_win_moves is None or move_count < stats.fastest_win_moves:
            stats.fastest_win_moves = int(move_count)
    elif outcome == RESULT_LOSS:
        stats.losses += 1
        stats.current_streak = 0
    else:
        stats.draws += 1
        stats.current_streak = 0

    if band:
        tally = stats.by_band.setdefault(band, _empty_tally())
        tally[outcome] = tally.get(outcome, 0) + 1

    mode_tally = stats.by_mode.setdefault(mode, _empty_tally())
    mode_tally[outcome] = mode_tally.get(outcome, 0) + 1

    return stats


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _sanitise(d: dict) -> dict:
    known = set(PlayerStats.__dataclass_fields__)
    return {k: v for k, v in d.items() if k in known}


def load_stats(path: Path = PLAYER_STATS_PATH) -> PlayerStats:
    """Read stats from disk; a fresh PlayerStats if absent or unreadable."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return PlayerStats()
    if not isinstance(data, dict):
        return PlayerStats()
    try:
        return PlayerStats(**_sanitise(data))
    except TypeError:
        return PlayerStats()


def save_stats(stats: PlayerStats, path: Path = PLAYER_STATS_PATH) -> bool:
    """Atomically write stats.  False if the write failed (never raises)."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(stats), indent=2))
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def reset_stats(path: Path = PLAYER_STATS_PATH) -> PlayerStats:
    fresh = PlayerStats()
    save_stats(fresh, path)
    return fresh
