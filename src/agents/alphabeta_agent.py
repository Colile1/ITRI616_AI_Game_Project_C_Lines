"""Alpha-beta agent with iterative deepening and a hand-engineered evaluation function.

This is a fixed, non-learning benchmark agent. It does not update between games
and is intentionally deterministic given a fixed seed so benchmark results are
reproducible.

Speed design: depth is kept at 2 by default (configurable). Each move is also
capped at MAX_THINK_SEC seconds — if the time limit is hit the best move found
so far is returned. This keeps benchmark games under a few seconds in Python.

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
import time
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

# Hard wall-clock cap per move — search stops early if exceeded
MAX_THINK_SEC = 2.0


def _other(p: int) -> int:
    return PLAYER_2 if p == PLAYER_1 else PLAYER_1


# ---------------------------------------------------------------------------
# Pattern helpers — use numpy for speed
# ---------------------------------------------------------------------------

def _count_open_k(grid: np.ndarray, player: int, n: int, k: int) -> int:
    """Count runs of exactly length k with at least one open end."""
    count = 0
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
                if length != k:
                    continue
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
                    count += 1
    return count


def _has_immediate_win(grid: np.ndarray, player: int, n: int) -> list[tuple[int, int]]:
    wins = []
    for r in range(n):
        for c in range(n):
            if grid[r, c] != EMPTY:
                continue
            grid[r, c] = player
            if _player_has_run_4(grid, player, n):
                wins.append((r, c))
            grid[r, c] = EMPTY
    return wins


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
# Fast evaluation — only line scores + open-4 threats (skip slow open-3/mobility)
# ---------------------------------------------------------------------------

def _evaluate_fast(grid: np.ndarray, player: int, n: int, weights: dict) -> float:
    """Lightweight eval: line scores + open-4 only. Much faster than full eval."""
    opp = _other(player)
    own_ls = score_board(grid, player, n)
    opp_ls = score_board(grid, opp, n)
    own_o4 = _count_open_k(grid, player, n, 4)
    opp_o4 = _count_open_k(grid, opp, n, 4)
    return (
        weights["line_score"] * (own_ls - opp_ls)
        + weights["open_4"]   * (own_o4 - opp_o4)
    )


# ---------------------------------------------------------------------------
# Move ordering — cheap version (win/block first, then centre proximity)
# ---------------------------------------------------------------------------

def _order_moves(
    board: Board, moves: list[tuple[int, int]], player: int
) -> list[tuple[int, int]]:
    n = board.size
    grid = board.grid
    opp = _other(player)
    centre = n / 2.0

    wins_set  = set(_has_immediate_win(grid, player, n))
    blocks_set = set(_has_immediate_win(grid, opp, n))

    def priority(rc: tuple[int, int]) -> float:
        r, c = rc
        if rc in wins_set:
            return -3.0
        if rc in blocks_set:
            return -2.0
        return abs(r - centre) + abs(c - centre)

    return sorted(moves, key=priority)


# ---------------------------------------------------------------------------
# Alpha-beta search with time limit
# ---------------------------------------------------------------------------

def _alphabeta(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    maximising: bool,
    root_player: int,
    weights: dict,
    deadline: float,
) -> float:
    if time.monotonic() > deadline:
        raise TimeoutError

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
        return _evaluate_fast(board.grid, root_player, n, weights)

    moves = legal_placements(board)
    current_player = board.current_player
    ordered = _order_moves(board, moves, current_player)

    if maximising:
        value = -_INF
        for r, c in ordered:
            child = apply_placement(board, r, c)
            score = _alphabeta(child, depth - 1, alpha, beta, False,
                               root_player, weights, deadline)
            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = _INF
        for r, c in ordered:
            child = apply_placement(board, r, c)
            score = _alphabeta(child, depth - 1, alpha, beta, True,
                               root_player, weights, deadline)
            value = min(value, score)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AlphaBetaAgent(BaseAgent):
    """Fixed-policy alpha-beta agent.

    Default depth=2 keeps each move under ~1 second in Python.
    A hard time cap (MAX_THINK_SEC) cuts off any move that runs long.

    Call set_board() before select_action(), same pattern as HeuristicAgent.
    """

    def __init__(
        self,
        board_size: Optional[int] = None,
        depth: Optional[int] = None,
        weights: Optional[dict] = None,
        max_think_sec: float = MAX_THINK_SEC,
    ):
        self._board_size = board_size
        # Default depth: 2 (fast) instead of 4 (very slow in Python)
        self._depth = depth if depth is not None else 2
        self._weights = weights or dict(BENCHMARK_EVAL_WEIGHTS)
        self._max_think_sec = max_think_sec
        self._board: Optional[Board] = None

    def set_board(self, board: Board) -> None:
        self._board = board

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        if self._board is None:
            raise RuntimeError("AlphaBetaAgent.set_board() must be called first.")
        board = self._board
        n = board.size
        player = board.current_player

        moves = legal_placements(board)
        if not moves:
            legal_indices = np.where(legal_mask)[0]
            return int(np.random.choice(legal_indices))

        ordered = _order_moves(board, moves, player)

        # Immediate win — no search needed
        wins = _has_immediate_win(board.grid, player, n)
        if wins:
            return action_to_index(wins[0][0], wins[0][1], n)

        best_move = ordered[0]
        best_score = -_INF
        deadline = time.monotonic() + self._max_think_sec

        for r, c in ordered:
            try:
                child = apply_placement(board, r, c)
                score = _alphabeta(
                    child, self._depth - 1, -_INF, _INF,
                    False, player, self._weights, deadline,
                )
            except TimeoutError:
                break   # return best found so far
            if score > best_score:
                best_score = score
                best_move = (r, c)

        return action_to_index(best_move[0], best_move[1], n)
