"""Alpha-beta agent with iterative deepening and a hand-engineered evaluation function.

This is a fixed, non-learning benchmark agent. It does not update between games
and is intentionally deterministic given a fixed seed so benchmark results are
reproducible.

Evaluation function (from each player's perspective):
  eval = W_open3 * (own_open3 - opp_open3)
       + W_open4 * (own_open4 - opp_open4)
       + W_line  * (own_line_score - opp_line_score)
       + W_mob   * (own_mobility - opp_mobility)

Move ordering: immediate-win / immediate-block first, then centre cells, then
cells adjacent to existing pieces, then the rest.
"""

from __future__ import annotations
import math
from typing import Optional

import numpy as np

from src.agents.base_agent import BaseAgent
from src.engine.board import Board, clone_board
from src.engine.rules import apply_placement, legal_placements, check_terminal_mode1
from src.engine.scoring import score_board
from src.game.encoding import index_to_action, action_to_index
from src.config import (
    PLAYER_1, PLAYER_2, EMPTY,
    BENCHMARK_DEPTH_BY_SIZE, BENCHMARK_EVAL_WEIGHTS,
)

_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_INF = float("inf")


def _other(p: int) -> int:
    return PLAYER_2 if p == PLAYER_1 else PLAYER_1


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------

def _count_open_k(grid: np.ndarray, player: int, n: int, k: int) -> int:
    """Count runs of exactly length k that have at least one open extension."""
    count = 0
    opp = _other(player)
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
                if length != k:
                    continue
                # Check if at least one end is open
                before_r, before_c = r - dr, c - dc
                after_r, after_c = nr, nc  # one past the end
                before_open = (
                    0 <= before_r < n and 0 <= before_c < n
                    and grid[before_r, before_c] == EMPTY
                )
                after_open = (
                    0 <= after_r < n and 0 <= after_c < n
                    and grid[after_r, after_c] == EMPTY
                )
                if before_open or after_open:
                    count += 1
    return count


def _has_immediate_win(grid: np.ndarray, player: int, n: int) -> list[tuple[int, int]]:
    """Return list of cells that immediately win (create run >= 4) for player."""
    wins = []
    for r in range(n):
        for c in range(n):
            if grid[r, c] != EMPTY:
                continue
            grid[r, c] = player
            if _player_has_run(grid, player, n, 4):
                wins.append((r, c))
            grid[r, c] = EMPTY
    return wins


def _player_has_run(grid: np.ndarray, player: int, n: int, length: int) -> bool:
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                cnt = 0
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    cnt += 1
                    if cnt >= length:
                        return True
                    nr += dr
                    nc += dc
    return False


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def _evaluate(grid: np.ndarray, player: int, n: int, weights: dict[str, float]) -> float:
    opp = _other(player)
    own_o3 = _count_open_k(grid, player, n, 3)
    opp_o3 = _count_open_k(grid, opp, n, 3)
    own_o4 = _count_open_k(grid, player, n, 4)
    opp_o4 = _count_open_k(grid, opp, n, 4)
    own_ls  = score_board(grid, player, n)
    opp_ls  = score_board(grid, opp, n)
    # Mobility = number of empty neighbours of player's pieces (proxy)
    own_mob = _mobility(grid, player, n)
    opp_mob = _mobility(grid, opp, n)
    return (
        weights["open_3"]    * (own_o3 - opp_o3)
        + weights["open_4"]  * (own_o4 - opp_o4)
        + weights["line_score"] * (own_ls - opp_ls)
        + weights["mobility"] * (own_mob - opp_mob)
    )


def _mobility(grid: np.ndarray, player: int, n: int) -> int:
    """Count empty cells adjacent (8-connected) to player's pieces."""
    empty_adj: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            if grid[r, c] != player:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == EMPTY:
                        empty_adj.add((nr, nc))
    return len(empty_adj)


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

def _order_moves(
    board: Board, moves: list[tuple[int, int]], player: int
) -> list[tuple[int, int]]:
    """Order: win first, then block, then centre-near, then adjacent-to-pieces."""
    n = board.size
    grid = board.grid
    opp = _other(player)
    centre = n / 2.0

    wins  = _has_immediate_win(grid, player, n)
    wins_set = set(wins)
    blocks = _has_immediate_win(grid, opp, n)
    blocks_set = set(blocks)

    def priority(rc: tuple[int, int]) -> float:
        r, c = rc
        if rc in wins_set:
            return -3.0
        if rc in blocks_set:
            return -2.0
        dist = abs(r - centre) + abs(c - centre)
        adj = _has_adj_piece(grid, r, c, n)
        return dist - (0.5 if adj else 0.0)

    return sorted(moves, key=priority)


def _has_adj_piece(grid: np.ndarray, r: int, c: int, n: int) -> bool:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] != EMPTY:
                return True
    return False


# ---------------------------------------------------------------------------
# Alpha-beta search
# ---------------------------------------------------------------------------

def _alphabeta(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    maximising: bool,
    root_player: int,
    weights: dict[str, float],
) -> float:
    n = board.size
    done, winner = check_terminal_mode1(board)
    if done:
        if winner == root_player:
            return 10000.0
        elif winner is None:
            return 0.0
        else:
            return -10000.0

    if depth == 0:
        return _evaluate(board.grid, root_player, n, weights)

    moves = legal_placements(board)
    current_player = board.current_player
    ordered = _order_moves(board, moves, current_player)

    if maximising:
        value = -_INF
        for r, c in ordered:
            child = apply_placement(board, r, c)
            score = _alphabeta(child, depth - 1, alpha, beta, False, root_player, weights)
            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = _INF
        for r, c in ordered:
            child = apply_placement(board, r, c)
            score = _alphabeta(child, depth - 1, alpha, beta, True, root_player, weights)
            value = min(value, score)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AlphaBetaAgent(BaseAgent):
    """Fixed-policy iterative-deepening alpha-beta agent.

    Call set_board() before select_action(), same pattern as HeuristicAgent.
    """

    def __init__(
        self,
        board_size: Optional[int] = None,
        depth: Optional[int] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self._board_size = board_size
        self._depth = depth  # None → read from BENCHMARK_DEPTH_BY_SIZE
        self._weights = weights or dict(BENCHMARK_EVAL_WEIGHTS)
        self._board: Optional[Board] = None

    def set_board(self, board: Board) -> None:
        self._board = board

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        if self._board is None:
            raise RuntimeError("AlphaBetaAgent.set_board() must be called first.")
        board = self._board
        n = board.size
        depth = self._depth if self._depth is not None else BENCHMARK_DEPTH_BY_SIZE.get(n, 3)
        player = board.current_player

        moves = legal_placements(board)
        if not moves:
            legal_indices = np.where(legal_mask)[0]
            return int(np.random.choice(legal_indices))

        ordered = _order_moves(board, moves, player)
        best_move = ordered[0]
        best_score = -_INF

        for r, c in ordered:
            child = apply_placement(board, r, c)
            score = _alphabeta(
                child, depth - 1, -_INF, _INF,
                False,   # we just moved as maximiser; child is opponent's turn
                player, self._weights,
            )
            if score > best_score:
                best_score = score
                best_move = (r, c)

        return action_to_index(best_move[0], best_move[1], n)
