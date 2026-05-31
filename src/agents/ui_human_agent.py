"""Human agent that communicates with the PyGame training board via thread-safe queues.

The training loop runs in a background thread.  When it is the human's turn,
select_action() puts the current board state into board_queue for the UI to
display and then blocks on move_queue until the user clicks a cell.

Protocol
--------
board_queue (training → UI):
    dict  {grid, legal_mask, current_player, board_size}

move_queue  (UI → training):
    int   action_idx  (−1 = sentinel: switch to auto, use any legal move)

stats_queue (training → UI, optional):
    dict  {game, epsilon, wr_random, wr_heuristic, best_wr, mode}
"""

from __future__ import annotations
from queue import Queue, Empty

import numpy as np

from src.agents.base_agent import BaseAgent
from src.engine.board import Board


class UIHumanAgent(BaseAgent):

    def __init__(
        self,
        board_queue: Queue,
        move_queue: Queue,
        stats_queue: "Queue | None" = None,
        switch_signal_path=None,
    ):
        self._board_q = board_queue
        self._move_q  = move_queue
        self._stats_q = stats_queue
        self._switch_signal_path = switch_signal_path
        self._board: Board | None = None
        self._tally: dict[str, int] = {"WIN": 0, "LOSS": 0, "DRAW": 0}

    # self_play.py calls set_board(env.board) before select_action
    def set_board(self, board: Board) -> None:
        self._board = board

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        if self._board is None:
            return int(np.where(legal_mask)[0][0])

        state = {
            "grid":           self._board.grid.copy(),
            "legal_mask":     legal_mask.copy(),
            "current_player": self._board.current_player,
            "board_size":     self._board.size,
        }
        # Replace any stale state so the UI always shows the latest position
        try:
            self._board_q.get_nowait()
        except Empty:
            pass
        self._board_q.put(state)

        # Block until UI delivers a move (or timeout after 10 minutes)
        try:
            action = self._move_q.get(timeout=600)
        except Empty:
            return int(np.where(legal_mask)[0][0])

        if action < 0:  # sentinel from "Switch to Auto" button
            if self._switch_signal_path:
                self._switch_signal_path.write_text("auto")
            return int(np.where(legal_mask)[0][0])

        return int(action)

    def notify_game_end(self, info: dict) -> None:
        """Send game-over result to the UI immediately after play_episode returns.

        info keys: grid, human_player, winner_player, human_result (WIN/LOSS/DRAW),
                   human_reward, agent_reward, game_idx
        """
        result = info.get("human_result", "DRAW")
        self._tally[result] = self._tally.get(result, 0) + 1

        msg = {**info, "game_over": True, "tally": dict(self._tally)}
        try:
            self._board_q.get_nowait()
        except Empty:
            pass
        try:
            self._board_q.put_nowait(msg)
        except Exception:
            pass

    def push_display(self, grid, winner, p1_label: str, p2_label: str, game_idx: int) -> None:
        """Send a display-only board update during auto-play (no move expected from UI)."""
        msg = {
            "display_only": True,
            "grid":      grid,
            "winner":    winner,
            "p1_label":  p1_label,
            "p2_label":  p2_label,
            "game_idx":  game_idx,
        }
        try:
            self._board_q.get_nowait()
        except Empty:
            pass
        try:
            self._board_q.put_nowait(msg)
        except Exception:
            pass

    def push_stats(self, stats: dict) -> None:
        """Called by train.py after each eval to update the sidebar."""
        if self._stats_q is not None:
            try:
                self._stats_q.get_nowait()   # drop stale stats
            except Empty:
                pass
            try:
                self._stats_q.put_nowait(stats)
            except Exception:
                pass
