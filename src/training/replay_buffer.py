"""Replay buffer for DQN off-policy training."""

from __future__ import annotations
from collections import deque
import numpy as np


class ReplayBuffer:
    """Circular buffer storing (state, action, reward, next_state, done, legal_mask_next)."""

    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        legal_mask_next: np.ndarray,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done, legal_mask_next))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.choice(len(self._buf), batch_size, replace=False)
        batch = [self._buf[i] for i in indices]
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "legal_masks_next": np.array(masks, dtype=bool),
        }

    def __len__(self) -> int:
        return len(self._buf)
