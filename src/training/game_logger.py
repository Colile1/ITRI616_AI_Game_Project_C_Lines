"""Per-game result logger — writes one CSV row for every training game.

Columns
-------
game_idx        : int   — training game counter (0-based)
p1_label        : str   — who played as Player 1
p2_label        : str   — who played as Player 2
winner          : 1|2|draw — winner player number, or "draw"
p1_reward_sum   : float — sum of all rewards received by P1 during the episode
p2_reward_sum   : float — sum of all rewards received by P2 during the episode
episode_length  : int   — total moves played
timestamp       : str   — ISO-8601 wall-clock time of the game

Label conventions
-----------------
"dqn"           — the training DQN agent (current policy)
"human"         — human player (UI or terminal)
"random"        — RandomAgent
"heuristic"     — HeuristicAgent
"self"          — frozen clone of the current DQN (self-play)
"pool:gen_NNN"  — frozen snapshot from the self-play pool
"alphabeta_d4"  — AlphaBeta search agent at depth 4 (or other depth)
"""

from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path


class GameLogger:
    """Append-safe CSV logger — one row per game, opened for the life of a run."""

    _HEADER = [
        "game_idx",
        "p1_label", "p2_label",
        "winner",
        "p1_reward_sum", "p2_reward_sum",
        "episode_length",
        "timestamp",
    ]

    def __init__(self, log_path: Path):
        self._path = Path(log_path)
        write_header = not self._path.exists()
        self._file   = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(self._HEADER)
            self._file.flush()

    def log(
        self,
        game_idx: int,
        p1_label: str,
        p2_label: str,
        winner: "int | None",
        p1_transitions: list,
        p2_transitions: list,
    ) -> None:
        """Write one row.  transitions are lists of Transition objects from play_episode."""
        p1_rew = sum(t.reward for t in p1_transitions)
        p2_rew = sum(t.reward for t in p2_transitions)
        ep_len = len(p1_transitions) + len(p2_transitions)
        winner_str = str(winner) if winner is not None else "draw"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        self._writer.writerow([
            game_idx,
            p1_label, p2_label,
            winner_str,
            f"{p1_rew:.4f}", f"{p2_rew:.4f}",
            ep_len,
            ts,
        ])
        # Flush every 100 games to keep the file readable during a live run
        if game_idx % 100 == 0:
            self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()
