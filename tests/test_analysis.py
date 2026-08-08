"""A-01 .. A-16 — position analysis used by the player-facing app."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import MODE_FIRST_TO_FOUR, MODE_POINTS_FULL, PLAYER_1, PLAYER_2
from src.engine.board import setup_board
from src.game import analysis


def _board(n: int = 8, cells: dict | None = None, current: int = PLAYER_1):
    b = setup_board(n)
    for (r, c), p in (cells or {}).items():
        b.grid[r, c] = p
    b.current_player = current
    return b


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def test_a01_algebraic_matches_board_labels():
    assert analysis.algebraic(0, 0) == "A1"
    assert analysis.algebraic(3, 3) == "D4"
    assert analysis.algebraic(7, 2) == "C8"


def test_a02_other_player():
    assert analysis.other_player(PLAYER_1) == PLAYER_2
    assert analysis.other_player(PLAYER_2) == PLAYER_1


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def test_a03_maximal_runs_counts_each_segment_once_per_direction():
    b = _board(cells={(2, c): PLAYER_1 for c in range(2, 6)})
    runs = analysis.maximal_runs(b.grid, PLAYER_1, b.size)
    horizontal = [r for r in runs if len(r) == 4]
    assert len(horizontal) == 1
    # The four pieces each also form a length-1 run in the other 3 directions.
    assert sum(1 for r in runs if len(r) == 1) == 12


def test_a04_longest_run_picks_the_longest():
    b = _board(cells={
        **{(0, c): PLAYER_1 for c in range(3)},
        **{(4, c): PLAYER_1 for c in range(5)},
    })
    assert len(analysis.longest_run(b, PLAYER_1)) == 5


def test_a05_longest_run_empty_for_absent_player():
    assert analysis.longest_run(_board(), PLAYER_2) == []


def test_a06_find_winning_line_requires_min_length():
    b = _board(cells={(1, c): PLAYER_1 for c in range(3)})
    assert analysis.find_winning_line(b, PLAYER_1, 4) is None
    b.grid[1, 3] = PLAYER_1
    line = analysis.find_winning_line(b, PLAYER_1, 4)
    assert line is not None and len(line) == 4


def test_a07_deciding_line_ftf_is_the_four():
    b = _board(cells={(r, r): PLAYER_2 for r in range(4)})
    line = analysis.deciding_line(b, MODE_FIRST_TO_FOUR, PLAYER_2)
    assert sorted(line) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_a08_deciding_line_points_uses_longest_run():
    b = _board(cells={(5, c): PLAYER_1 for c in range(3)})
    assert len(analysis.deciding_line(b, MODE_POINTS_FULL, PLAYER_1)) == 3


def test_a09_deciding_line_empty_on_draw():
    assert analysis.deciding_line(_board(), MODE_POINTS_FULL, None) == []


# ---------------------------------------------------------------------------
# Threats
# ---------------------------------------------------------------------------

def test_a10_threat_cells_finds_open_three():
    b = _board(cells={(3, c): PLAYER_1 for c in range(2, 5)})
    open3, open4 = analysis.threat_cells(b, PLAYER_1)
    assert open3 == {(3, 2), (3, 3), (3, 4)}
    assert open4 == set()


def test_a11_threat_cells_ignores_blocked_run():
    cells = {(0, c): PLAYER_1 for c in range(3)}
    cells[(0, 3)] = PLAYER_2          # far end blocked; near end is the board edge
    b = _board(cells=cells)
    open3, _ = analysis.threat_cells(b, PLAYER_1)
    assert open3 == set()


def test_a12_critical_cells_reports_win_and_loss_squares():
    b = _board(cells={
        **{(2, c): PLAYER_1 for c in range(1, 4)},
        **{(6, c): PLAYER_2 for c in range(1, 4)},
    })
    win_now, lose_now = analysis.critical_cells(b, PLAYER_1)
    assert (2, 0) in win_now and (2, 4) in win_now
    assert (6, 0) in lose_now and (6, 4) in lose_now


def test_a13_critical_cells_leaves_the_grid_untouched():
    b = _board(cells={(2, c): PLAYER_1 for c in range(1, 4)})
    before = b.grid.copy()
    analysis.critical_cells(b, PLAYER_1)
    assert np.array_equal(b.grid, before)


def test_a14_build_threat_overlay_is_perspective_aware():
    b = _board(cells={
        **{(2, c): PLAYER_1 for c in range(1, 4)},
        **{(6, c): PLAYER_2 for c in range(1, 4)},
    })
    p1_view = analysis.build_threat_overlay(b, PLAYER_1)
    p2_view = analysis.build_threat_overlay(b, PLAYER_2)
    assert p1_view.own3 == p2_view.opp3
    assert p1_view.opp3 == p2_view.own3
    assert not p1_view.is_empty()


# ---------------------------------------------------------------------------
# Hints and evaluation
# ---------------------------------------------------------------------------

def test_a15_suggest_move_returns_a_legal_cell():
    b = _board(cells={(2, c): PLAYER_1 for c in range(1, 4)})
    cell = analysis.suggest_move(b, MODE_FIRST_TO_FOUR)
    assert cell is not None
    r, c = cell
    assert b.grid[r, c] == 0


def test_a16_suggest_move_none_on_full_board():
    b = setup_board(4)
    b.grid[:, :] = PLAYER_1
    assert analysis.suggest_move(b, MODE_POINTS_FULL) is None


def test_a16b_suggest_move_reports_which_engine_answered():
    """A snapshot that fails must not have its fallback credited to the network."""
    class GoodAgent:
        in_channels = 10

        def q_values(self, obs):
            q = np.zeros(64, dtype=np.float32)
            q[27] = 10.0                      # (3, 3) on an 8x8 board
            return q

    class BrokenAgent:
        in_channels = 10

        def q_values(self, obs):
            raise RuntimeError("weights did not load")

    b = _board()
    assert analysis.suggest_move_detailed(b, MODE_POINTS_FULL, GoodAgent()) == ((3, 3), "network")

    cell, source = analysis.suggest_move_detailed(b, MODE_POINTS_FULL, BrokenAgent())
    assert source == "search" and cell is not None

    assert analysis.suggest_move_detailed(b, MODE_POINTS_FULL)[1] == "search"

    full = setup_board(4)
    full.grid[:, :] = PLAYER_1
    assert analysis.suggest_move_detailed(full, MODE_POINTS_FULL) == (None, "none")


def test_a16c_suggest_move_ignores_an_agent_with_the_wrong_action_space():
    """A 12x12 snapshot loaded against an 8x8 board must not silently mis-index."""
    class WrongSizeAgent:
        in_channels = 10

        def q_values(self, obs):
            return np.zeros(144, dtype=np.float32)

    b = _board()
    cell, source = analysis.suggest_move_detailed(b, MODE_POINTS_FULL, WrongSizeAgent())
    assert source == "search"
    assert cell is not None and b.grid[cell] == 0


def test_a17_evaluate_position_is_signed_from_p1_perspective():
    b = _board(cells={(3, c): PLAYER_1 for c in range(3)})
    assert analysis.evaluate_position(b, MODE_POINTS_FULL) > 0

    b2 = _board(cells={(3, c): PLAYER_2 for c in range(3)})
    assert analysis.evaluate_position(b2, MODE_POINTS_FULL) < 0


def test_a18_evaluate_position_is_zero_on_an_empty_board():
    assert analysis.evaluate_position(_board(), MODE_POINTS_FULL) == pytest.approx(0.0)
    assert analysis.evaluate_position(_board(), MODE_FIRST_TO_FOUR) == pytest.approx(0.0)


def test_a19_evaluate_position_stays_in_range():
    b = _board(cells={(r, c): PLAYER_1 for r in range(6) for c in range(6)})
    value = analysis.evaluate_position(b, MODE_POINTS_FULL)
    assert -1.0 <= value <= 1.0


def test_a20_evaluate_position_uses_the_agent_when_given():
    class StubAgent:
        in_channels = 10
        calls = 0

        def q_values(self, obs):
            StubAgent.calls += 1
            return np.full(64, 5.0, dtype=np.float32)

    b = _board(current=PLAYER_1)
    value = analysis.evaluate_position(b, MODE_POINTS_FULL, agent=StubAgent())
    assert StubAgent.calls == 1
    assert value > 0.9                      # tanh(5.0) saturates towards +1

    b.current_player = PLAYER_2
    assert analysis.evaluate_position(b, MODE_POINTS_FULL, agent=StubAgent()) < -0.9


def test_a21_evaluate_position_falls_back_when_the_agent_raises():
    class BrokenAgent:
        in_channels = 10

        def q_values(self, obs):
            raise RuntimeError("no weights")

    b = _board(cells={(3, c): PLAYER_1 for c in range(3)})
    assert analysis.evaluate_position(b, MODE_POINTS_FULL, agent=BrokenAgent()) > 0


def test_a22_line_breakdown_groups_by_length():
    b = _board(cells={
        **{(0, c): PLAYER_1 for c in range(4)},
        **{(2, c): PLAYER_1 for c in range(3)},
        **{(4, c): PLAYER_1 for c in range(3)},
    })
    assert analysis.line_breakdown(b, PLAYER_1) == {4: 1, 3: 2}
