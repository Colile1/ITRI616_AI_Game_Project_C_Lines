"""State encoding and action space utilities for C_lines. Pure functions."""

from __future__ import annotations
import numpy as np

from src.engine.board import Board
from src.config import EMPTY, PLAYER_1, PLAYER_2, MIN_SCORING_LENGTH, STATE_CHANNELS_V1

_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def action_to_index(row: int, col: int, n: int) -> int:
    return row * n + col


def index_to_action(idx: int, n: int) -> tuple[int, int]:
    return divmod(idx, n)


def build_legal_mask(board: Board, phase: str = "placement") -> np.ndarray:
    n = board.size
    mask = np.zeros(n * n, dtype=bool)
    if phase == "placement":
        for r in range(n):
            for c in range(n):
                if board.grid[r, c] == EMPTY:
                    mask[action_to_index(r, c, n)] = True
    elif phase == "removal":
        target = board.current_player
        for r in range(n):
            for c in range(n):
                if board.grid[r, c] == target:
                    mask[action_to_index(r, c, n)] = True
    return mask


# ---------------------------------------------------------------------------
# Threat helpers
# ---------------------------------------------------------------------------

def _compute_open3_threats(grid: np.ndarray, player: int, n: int) -> np.ndarray:
    """Binary (n,n) plane — 1 at cells that are part of any open-3 run."""
    threat = np.zeros((n, n), dtype=np.float32)
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                # Only start from run heads
                pr, pc = r - dr, c - dc
                if 0 <= pr < n and 0 <= pc < n and grid[pr, pc] == player:
                    continue
                length = 0
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    length += 1
                    nr += dr
                    nc += dc
                if length == MIN_SCORING_LENGTH:
                    mr, mc = r, c
                    for _ in range(length):
                        threat[mr, mc] = 1.0
                        mr += dr
                        mc += dc
    return threat


def _compute_open4_threats(grid: np.ndarray, player: int, n: int) -> np.ndarray:
    """Binary (n,n) plane — 1 at cells that are part of any open-4 run."""
    threat = np.zeros((n, n), dtype=np.float32)
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < n and 0 <= pc < n and grid[pr, pc] == player:
                    continue
                length = 0
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    length += 1
                    nr += dr
                    nc += dc
                if length == 4:
                    # Check at least one end is open
                    before_r, before_c = r - dr, c - dc
                    after_r, after_c = nr, nc
                    before_open = (
                        0 <= before_r < n and 0 <= before_c < n
                        and grid[before_r, before_c] == EMPTY
                    )
                    after_open = (
                        0 <= after_r < n and 0 <= after_c < n
                        and grid[after_r, after_c] == EMPTY
                    )
                    if before_open or after_open:
                        mr, mc = r, c
                        for _ in range(length):
                            threat[mr, mc] = 1.0
                            mr += dr
                            mc += dc
    return threat


def _compute_immediate_wins(grid: np.ndarray, player: int, n: int) -> np.ndarray:
    """Binary (n,n) plane — 1 at empty cells where placing wins immediately."""
    plane = np.zeros((n, n), dtype=np.float32)
    for r in range(n):
        for c in range(n):
            if grid[r, c] != EMPTY:
                continue
            grid[r, c] = player
            if _player_has_run_4(grid, player, n):
                plane[r, c] = 1.0
            grid[r, c] = EMPTY
    return plane


def _compute_immediate_losses(grid: np.ndarray, player: int, n: int) -> np.ndarray:
    """Binary (n,n) plane — 1 at empty cells where the opponent would win if they played there."""
    opp = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    return _compute_immediate_wins(grid, opp, n)


def _player_has_run_4(grid: np.ndarray, player: int, n: int) -> bool:
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                cnt = 0
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    cnt += 1
                    if cnt >= 4:
                        return True
                    nr += dr
                    nc += dc
    return False


# ---------------------------------------------------------------------------
# State tensor constructors
# ---------------------------------------------------------------------------

def state_to_tensor(board: Board, n_channels: int | None = None) -> np.ndarray:
    """Encode board as (C, n, n) float32 tensor.

    n_channels=6  → legacy v1 encoding (6 channels)
    n_channels=10 → v2 encoding (10 channels, default)

    Ch 0 — current player's pieces
    Ch 1 — opponent's pieces
    Ch 2 — empty cells
    Ch 3 — current player's open-3 threats
    Ch 4 — opponent's open-3 threats
    Ch 5 — turn progress (turn / n²), constant plane
    Ch 6 — current player's open-4 threats         [v2 only]
    Ch 7 — opponent's open-4 threats               [v2 only]
    Ch 8 — cells where current player wins now      [v2 only]
    Ch 9 — cells where opponent would win now       [v2 only]
    """
    from src.config import STATE_CHANNELS_V2
    if n_channels is None:
        n_channels = STATE_CHANNELS_V2

    n = board.size
    grid = board.grid
    cur = board.current_player
    opp = PLAYER_2 if cur == PLAYER_1 else PLAYER_1

    ch0 = (grid == cur).astype(np.float32)
    ch1 = (grid == opp).astype(np.float32)
    ch2 = (grid == EMPTY).astype(np.float32)
    ch3 = _compute_open3_threats(grid, cur, n)
    ch4 = _compute_open3_threats(grid, opp, n)
    progress = min(board.turn / (n * n), 1.0)
    ch5 = np.full((n, n), progress, dtype=np.float32)

    channels = [ch0, ch1, ch2, ch3, ch4, ch5]

    if n_channels >= 10:
        ch6 = _compute_open4_threats(grid, cur, n)
        ch7 = _compute_open4_threats(grid, opp, n)
        ch8 = _compute_immediate_wins(grid, cur, n)
        ch9 = _compute_immediate_losses(grid, cur, n)
        channels.extend([ch6, ch7, ch8, ch9])

    return np.stack(channels, axis=0)
