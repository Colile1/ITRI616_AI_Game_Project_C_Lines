"""Random agent — selects uniformly from legal moves."""

from __future__ import annotations
import numpy as np

from src.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng or np.random.default_rng()

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        legal_indices = np.where(legal_mask)[0]
        return int(self._rng.choice(legal_indices))
