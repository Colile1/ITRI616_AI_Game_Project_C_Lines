"""D4 dihedral symmetry augmentation for square boards.

k=0  identity
k=1  rotate 90°  (CCW)
k=2  rotate 180°
k=3  rotate 270° (CCW)
k=4  mirror horizontal (flip left-right)
k=5  mirror + rotate 90°
k=6  mirror + rotate 180°
k=7  mirror + rotate 270°
"""

from __future__ import annotations
import numpy as np


def _rot90_action(r: int, c: int, n: int) -> tuple[int, int]:
    """CCW 90° rotation of (r, c) on an n×n board."""
    return c, n - 1 - r


def apply_symmetry_grid(grid: np.ndarray, k: int) -> np.ndarray:
    """Apply dihedral symmetry k to a 2-D (n, n) array."""
    if k >= 4:
        grid = np.fliplr(grid)
        k -= 4
    return np.rot90(grid, k=k)


def transform_obs(obs: np.ndarray, k: int) -> np.ndarray:
    """Apply symmetry k to a (C, n, n) observation — vectorised over all channels at once."""
    if k >= 4:
        obs = obs[:, :, ::-1]   # mirror left-right across last axis (all channels)
        k -= 4
    if k == 0:
        return np.ascontiguousarray(obs)
    return np.ascontiguousarray(np.rot90(obs, k=k, axes=(1, 2)))


def transform_action_index(idx: int, n: int, k: int) -> int:
    """Map a flat action index through symmetry k on an n×n board."""
    r, c = divmod(idx, n)
    # Apply mirror first if k >= 4
    if k >= 4:
        c = n - 1 - c
        k -= 4
    # Apply rotations
    for _ in range(k):
        r, c = _rot90_action(r, c, n)
    return r * n + c


def transform_mask(mask: np.ndarray, n: int, k: int) -> np.ndarray:
    """Apply symmetry k permutation to a flat (n*n,) boolean legal mask."""
    grid = mask.reshape(n, n)
    transformed = apply_symmetry_grid(grid.astype(np.float32), k)
    return transformed.flatten().astype(bool)


def augment_transition(
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    legal_mask_next: np.ndarray,
    k: int,
) -> tuple:
    """Return a symmetry-k copy of a (state, action, reward, next_state, done, mask) tuple."""
    n = state.shape[-1]
    return (
        transform_obs(state, k),
        transform_action_index(action, n, k),
        reward,
        transform_obs(next_state, k),
        done,
        transform_mask(legal_mask_next, n, k),
    )
