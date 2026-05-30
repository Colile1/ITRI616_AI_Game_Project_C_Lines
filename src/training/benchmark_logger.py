"""Fixed-benchmark logging — plays N games against an unchanging reference
agent after each training checkpoint and records aggregated results.

With games_per_check=1 the old single-game (0/1 binary) behaviour is preserved.
With games_per_check=32+ the result column becomes a true win-rate estimate.
"""

from __future__ import annotations
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.agents.base_agent import BaseAgent
from src.game.env import GameEnv
from src.training.self_play import play_episode


@dataclass
class BenchmarkResult:
    training_game: int
    phase_name: str
    benchmark_name: str
    win_rate: float          # wins / games_played  (0.0–1.0)
    score_diff: float        # mean (learner_reward_sum − bm_reward_sum)
    mean_episode_length: float
    games_played: int
    wall_clock_sec: float


class BenchmarkLogger:
    """Logs benchmark results to CSV — one aggregated row per checkpoint.

    The benchmark agent is constructed once and never updated, so changes in
    win_rate are a real signal of learner improvement, not opponent drift.
    """

    _HEADER = [
        "training_game", "phase_name", "benchmark_name",
        "win_rate", "mean_score_diff",
        "mean_episode_length", "games_played", "wall_clock_sec",
    ]

    def __init__(
        self,
        log_path: Path,
        benchmark_agent: BaseAgent,
        env: GameEnv,
        benchmark_name: str = "alphabeta_d4",
        games_per_check: int = 1,
    ):
        self._agent = benchmark_agent
        self._env = env
        self._name = benchmark_name
        self._games_per_check = games_per_check
        self._log_path = Path(log_path)

        write_header = not self._log_path.exists()
        self._file = self._log_path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(self._HEADER)
            self._file.flush()

    def close(self) -> None:
        self._file.close()

    def run(
        self,
        learner: BaseAgent,
        training_game: int,
        phase_name: str = "training",
        rng_seed: Optional[int] = None,
    ) -> BenchmarkResult:
        """Play games_per_check games, aggregate, write one CSV row."""
        base_seed = rng_seed if rng_seed is not None else training_game

        wins = 0
        score_diffs: list[float] = []
        ep_lengths: list[int] = []
        t0 = time.monotonic()

        for g in range(self._games_per_check):
            np.random.seed(base_seed * 1000 + g)

            if (training_game + g) % 2 == 0:
                t1, t2, winner = play_episode(learner, self._agent, self._env)
                learner_player = 1
            else:
                t1, t2, winner = play_episode(self._agent, learner, self._env)
                learner_player = 2

            if winner == learner_player:
                wins += 1

            learner_t = t1 if learner_player == 1 else t2
            bm_t      = t2 if learner_player == 1 else t1
            score_diffs.append(
                sum(t.reward for t in learner_t) - sum(t.reward for t in bm_t)
            )
            ep_lengths.append(len(t1) + len(t2))

        elapsed = time.monotonic() - t0
        result = BenchmarkResult(
            training_game=training_game,
            phase_name=phase_name,
            benchmark_name=self._name,
            win_rate=wins / self._games_per_check,
            score_diff=float(np.mean(score_diffs)),
            mean_episode_length=float(np.mean(ep_lengths)),
            games_played=self._games_per_check,
            wall_clock_sec=elapsed,
        )

        self._writer.writerow([
            result.training_game, result.phase_name, result.benchmark_name,
            f"{result.win_rate:.4f}", f"{result.score_diff:.4f}",
            f"{result.mean_episode_length:.1f}", result.games_played,
            f"{result.wall_clock_sec:.3f}",
        ])
        self._file.flush()
        return result
