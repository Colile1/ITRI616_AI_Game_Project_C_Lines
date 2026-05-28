"""Fixed-benchmark logging — plays one (or more) games against an unchanging
reference agent after each training game and records the result.

The benchmark agent is constructed once at the start of training and never
updated, so changes in the learner's win rate against it are a real signal
of improvement, not opponent drift.
"""

from __future__ import annotations
import csv
import time
from dataclasses import dataclass, field
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
    result: float           # 1=learner won, 0.5=draw, 0=lost
    score_diff: float       # learner_score − benchmark_score (Mode 2)
    episode_length: int
    seed: int
    wall_clock_sec: float


class BenchmarkLogger:
    """Logs benchmark game results to a CSV file (one row per training game).

    The logger is append-safe: it opens the CSV in append mode so a resumed
    session continues from where it left off.
    """

    _HEADER = [
        "training_game", "phase_name", "benchmark_name",
        "benchmark_result", "benchmark_score_diff",
        "benchmark_episode_length", "benchmark_seed", "wall_clock_sec",
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

        # Open CSV — write header only if the file does not yet exist
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
    ) -> list[BenchmarkResult]:
        """Play self._games_per_check benchmark games and log each result."""
        results = []
        seed = rng_seed if rng_seed is not None else training_game

        for g in range(self._games_per_check):
            game_seed = seed * 1000 + g
            np.random.seed(game_seed)

            t0 = time.monotonic()
            if training_game % 2 == 0:
                t1, t2, winner = play_episode(learner, self._agent, self._env)
                learner_player = 1
            else:
                t1, t2, winner = play_episode(self._agent, learner, self._env)
                learner_player = 2

            elapsed = time.monotonic() - t0

            if winner == learner_player:
                result = 1.0
            elif winner is None:
                result = 0.5
            else:
                result = 0.0

            ep_len = len(t1) + len(t2)

            # Score difference: sum of rewards for learner minus benchmark
            # (simplified: terminal reward is ±1, mid-game rewards small)
            all_learner = t1 if learner_player == 1 else t2
            all_bm      = t2 if learner_player == 1 else t1
            score_diff = (
                sum(t.reward for t in all_learner) -
                sum(t.reward for t in all_bm)
            )

            br = BenchmarkResult(
                training_game=training_game,
                phase_name=phase_name,
                benchmark_name=self._name,
                result=result,
                score_diff=float(score_diff),
                episode_length=ep_len,
                seed=game_seed,
                wall_clock_sec=elapsed,
            )
            results.append(br)

            self._writer.writerow([
                br.training_game, br.phase_name, br.benchmark_name,
                f"{br.result:.1f}", f"{br.score_diff:.4f}",
                br.episode_length, br.seed, f"{br.wall_clock_sec:.3f}",
            ])
            self._file.flush()

        return results
