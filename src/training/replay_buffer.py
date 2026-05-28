"""Replay buffer with importance-weighted sampling and optional symmetry augmentation."""

from __future__ import annotations
from collections import deque
import numpy as np

from src.config import MAX_REPLAY_WEIGHT, USE_SYMMETRY_AUGMENTATION
from src.training.symmetry import augment_transition


_TRANSITION = tuple  # (state, action, reward, next_state, done, legal_mask_next)


class ReplayBuffer:
    """Circular buffer storing (state, action, reward, next_state, done, legal_mask_next).

    Weights support importance-weighted sampling: high-weight transitions
    (e.g. from human games) are sampled proportionally more often.
    Symmetry augmentation applies a random dihedral transform at sample time.
    """

    def __init__(
        self,
        capacity: int,
        use_augmentation: bool = USE_SYMMETRY_AUGMENTATION,
    ):
        self._buf: deque[_TRANSITION] = deque(maxlen=capacity)
        self._weights: deque[float] = deque(maxlen=capacity)
        self._use_augmentation = use_augmentation

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        legal_mask_next: np.ndarray,
        weight: float = 1.0,
    ) -> None:
        weight = min(float(weight), MAX_REPLAY_WEIGHT)
        self._buf.append((state, action, reward, next_state, done, legal_mask_next))
        self._weights.append(weight)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        w = np.array(self._weights, dtype=np.float64)
        probs = w / w.sum()
        indices = np.random.choice(len(self._buf), batch_size, replace=True, p=probs)

        batch = []
        for i in indices:
            t = self._buf[i]
            if self._use_augmentation:
                k = np.random.randint(8)
                if k != 0:
                    t = augment_transition(*t, k=k)
            batch.append(t)

        states, actions, rewards, next_states, dones, masks = zip(*batch)
        return {
            "states":           np.array(states,      dtype=np.float32),
            "actions":          np.array(actions,     dtype=np.int64),
            "rewards":          np.array(rewards,     dtype=np.float32),
            "next_states":      np.array(next_states, dtype=np.float32),
            "dones":            np.array(dones,       dtype=np.float32),
            "legal_masks_next": np.array(masks,       dtype=bool),
        }

    def __len__(self) -> int:
        return len(self._buf)
