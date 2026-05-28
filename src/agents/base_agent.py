"""Abstract base class for all C_lines agents."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """All agents implement select_action; nothing else is required."""

    @abstractmethod
    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """Return a flat action index (must be legal).

        obs: (6, n, n) float32 state tensor
        legal_mask: (n*n,) bool array — True where placement is legal
        """
