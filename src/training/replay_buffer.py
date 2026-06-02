"""Replay buffer with importance-weighted sampling and optional symmetry augmentation."""

from __future__ import annotations
from collections import deque
import numpy as np

from src.config import MAX_REPLAY_WEIGHT, USE_SYMMETRY_AUGMENTATION, GAMMA
from src.training.symmetry import augment_transition


# (state, action, reward, next_state, done, legal_mask_next, gamma_n)
_TRANSITION = tuple


class ReplayBuffer:
    """Circular buffer storing (state, action, reward, next_state, done, legal_mask_next, gamma_n).

    Weights support importance-weighted sampling: high-weight transitions
    (e.g. from human games) are sampled proportionally more often.
    Symmetry augmentation applies a random dihedral transform at sample time.
    gamma_n stores the per-transition bootstrap discount (GAMMA^n for n-step).
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
        gamma_n: float = GAMMA,
    ) -> None:
        weight = min(float(weight), MAX_REPLAY_WEIGHT)
        self._buf.append((state, action, reward, next_state, done, legal_mask_next, gamma_n))
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
                    # augment_transition expects 6-tuple; strip gamma_n, augment, re-add
                    t6 = augment_transition(*t[:6], k=k)
                    t = (*t6, t[6])
            batch.append(t)

        states, actions, rewards, next_states, dones, masks, gammas = zip(*batch)
        return {
            "states":           np.array(states,      dtype=np.float32),
            "actions":          np.array(actions,     dtype=np.int64),
            "rewards":          np.array(rewards,     dtype=np.float32),
            "next_states":      np.array(next_states, dtype=np.float32),
            "dones":            np.array(dones,       dtype=np.float32),
            "legal_masks_next": np.array(masks,       dtype=bool),
            "gammas":           np.array(gammas,      dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self._buf)
