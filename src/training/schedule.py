"""Training schedule abstractions: OpponentSource, TrainingPhase, TrainingSchedule.

A TrainingSchedule composes training phases (who the learner plays against, how
many games, what weight those transitions get in the replay buffer) together
with a fixed benchmark opponent that measures improvement.

Example inline schedule string:
    "self:50,human:10,self:100,pool:200,human:5"

Each token is <source>:<n_games>. Supported sources:
    random, heuristic, self, pool, alphabeta, alphabeta_dN, human, demo

JSON schedule files follow the structure in 10_human_in_loop_training_plan.md.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.config import DEFAULT_SOURCE_WEIGHTS, BENCHMARK_DEPTH_BY_SIZE


# ---------------------------------------------------------------------------
# OpponentSource protocol (plain class, not runtime Protocol for compat)
# ---------------------------------------------------------------------------

class OpponentSource:
    """Base class representing a source of opponent agents."""
    name: str = "base"
    _default_weight: float = 1.0

    def next_agent(self):
        """Return a BaseAgent instance for the next game."""
        raise NotImplementedError

    def is_blocking(self) -> bool:
        """True if this source requires user interaction (e.g. HumanSource)."""
        return False

    def default_weight(self) -> float:
        return DEFAULT_SOURCE_WEIGHTS.get(self.name, 1.0)


class RandomSource(OpponentSource):
    name = "random"

    def next_agent(self):
        from src.agents.random_agent import RandomAgent
        return RandomAgent()


class HeuristicSource(OpponentSource):
    name = "heuristic"

    def next_agent(self):
        from src.agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent()


class AlphaBetaSource(OpponentSource):
    def __init__(self, depth: Optional[int] = None):
        self._depth = depth
        self.name = f"alphabeta" if depth is None else f"alphabeta_d{depth}"

    def next_agent(self):
        from src.agents.alphabeta_agent import AlphaBetaAgent
        return AlphaBetaAgent(depth=self._depth)


class SelfSource(OpponentSource):
    """Returns a frozen clone of the current learner."""
    name = "self"

    def __init__(self, agent_factory):
        self._factory = agent_factory

    def next_agent(self):
        from src.training.snapshot import clone_agent
        return clone_agent(self._factory())


class PoolSource(OpponentSource):
    """Samples from the snapshot pool."""
    name = "pool"

    def __init__(self, pool: list):
        self._pool = pool

    def next_agent(self):
        from src.training.self_play import play_episode
        import numpy as np
        if not self._pool:
            from src.agents.random_agent import RandomAgent
            return RandomAgent()
        return self._pool[np.random.randint(len(self._pool))]


class BenchmarkSource(OpponentSource):
    """Fixed, frozen benchmark agent — never updated."""
    name = "benchmark"

    def __init__(self, agent):
        self._agent = agent

    def next_agent(self):
        return self._agent

    def is_frozen(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# TrainingPhase
# ---------------------------------------------------------------------------

@dataclass
class TrainingPhase:
    name: str
    opponent_name: str       # key for deserialization; actual source assigned at runtime
    n_games: int
    train: bool = True
    weight: float = 1.0


# ---------------------------------------------------------------------------
# TrainingSchedule
# ---------------------------------------------------------------------------

@dataclass
class TrainingSchedule:
    phases: list[TrainingPhase]
    benchmark_name: str = "alphabeta_d4"
    benchmark_every_n_games: int = 1
    benchmark_games_per_check: int = 1
    snapshot_every_n_games: int = 1000
    eval_every_n_games: int = 500

    def to_json(self) -> str:
        d = {
            "phases": [asdict(p) for p in self.phases],
            "benchmark": self.benchmark_name,
            "benchmark_every_n_games": self.benchmark_every_n_games,
            "benchmark_games_per_check": self.benchmark_games_per_check,
            "snapshot_every_n_games": self.snapshot_every_n_games,
            "eval_every_n_games": self.eval_every_n_games,
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "TrainingSchedule":
        d = json.loads(text)
        phases = [TrainingPhase(**p) for p in d.get("phases", [])]
        return cls(
            phases=phases,
            benchmark_name=d.get("benchmark", "alphabeta_d4"),
            benchmark_every_n_games=d.get("benchmark_every_n_games", 1),
            benchmark_games_per_check=d.get("benchmark_games_per_check", 1),
            snapshot_every_n_games=d.get("snapshot_every_n_games", 1000),
            eval_every_n_games=d.get("eval_every_n_games", 500),
        )

    @classmethod
    def from_string(cls, schedule_str: str, board_size: int = 8) -> "TrainingSchedule":
        """Parse inline schedule string like "self:50,human:10,pool:100".

        Weight defaults:
            human → 5.0, demo → 10.0, alphabeta* → 2.0, all others → 1.0
        """
        if not schedule_str.strip():
            raise ValueError("Empty schedule string")
        phases = []
        for token in schedule_str.split(","):
            token = token.strip()
            if not token:
                continue
            parts = token.split(":")
            if len(parts) != 2:
                raise ValueError(f"Bad token '{token}' — expected <source>:<n_games>")
            src, n_str = parts
            n = int(n_str)
            weight = DEFAULT_SOURCE_WEIGHTS.get(src.split("_")[0], 1.0)
            phases.append(TrainingPhase(
                name=f"{src}_{len(phases)}",
                opponent_name=src,
                n_games=n,
                weight=weight,
            ))
        if not phases:
            raise ValueError("Schedule produced no phases")
        return cls(phases=phases)

    def total_games(self) -> int:
        return sum(p.n_games for p in self.phases)


# ---------------------------------------------------------------------------
# Factory: build an OpponentSource from a name string
# ---------------------------------------------------------------------------

def make_source(name: str, pool: Optional[list] = None, agent_factory=None) -> OpponentSource:
    """Return the right OpponentSource for the given name."""
    if name == "random":
        return RandomSource()
    if name in ("heuristic", "heur"):
        return HeuristicSource()
    if name.startswith("alphabeta"):
        # Support alphabeta_d4 → depth=4
        depth = None
        if "_d" in name:
            try:
                depth = int(name.split("_d")[1])
            except ValueError:
                pass
        return AlphaBetaSource(depth=depth)
    if name == "self":
        if agent_factory is None:
            raise ValueError("SelfSource requires agent_factory")
        return SelfSource(agent_factory)
    if name in ("pool", "snapshot_pool"):
        return PoolSource(pool or [])
    raise ValueError(f"Unknown opponent source: '{name}'")
