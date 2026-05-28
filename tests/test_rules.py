"""R-01 .. R-14: rules module tests."""

import numpy as np
import pytest
from src.engine.board import setup_board
from src.engine.rules import (
    is_legal_placement,
    apply_placement,
    apply_removal,
    check_terminal_mode1,
    check_terminal_mode2,
    run_tiebreak,
    legal_placements,
    legal_removals,
)
from src.config import PLAYER_1, PLAYER_2


# ---------------------------------------------------------------------------
# R-01: fresh board has n*n legal placements
# ---------------------------------------------------------------------------
def test_R01_fresh_board_legal_placements():
    b = setup_board(8)
    assert len(legal_placements(b)) == 64


# ---------------------------------------------------------------------------
# R-02: placing a piece reduces legal placements by 1
# ---------------------------------------------------------------------------
def test_R02_placement_reduces_legal():
    b = setup_board(8)
    b2 = apply_placement(b, 0, 0)
    assert len(legal_placements(b2)) == 63


# ---------------------------------------------------------------------------
# R-03: occupied cell is not a legal placement
# ---------------------------------------------------------------------------
def test_R03_occupied_not_legal():
    b = setup_board(8)
    b2 = apply_placement(b, 3, 3)
    assert not is_legal_placement(b2, 3, 3)


# ---------------------------------------------------------------------------
# R-04: out-of-bounds placement is illegal
# ---------------------------------------------------------------------------
def test_R04_oob_illegal():
    b = setup_board(8)
    assert not is_legal_placement(b, -1, 0)
    assert not is_legal_placement(b, 0, 8)
    assert not is_legal_placement(b, 8, 8)


# ---------------------------------------------------------------------------
# R-05: apply_placement toggles current_player
# ---------------------------------------------------------------------------
def test_R05_player_toggles():
    b = setup_board(8)
    assert b.current_player == PLAYER_1
    b2 = apply_placement(b, 0, 0)
    assert b2.current_player == PLAYER_2
    b3 = apply_placement(b2, 0, 1)
    assert b3.current_player == PLAYER_1


# ---------------------------------------------------------------------------
# R-06: apply_placement is immutable (original board unchanged)
# ---------------------------------------------------------------------------
def test_R06_immutable_placement():
    b = setup_board(8)
    _ = apply_placement(b, 4, 4)
    assert b.grid[4, 4] == 0


# ---------------------------------------------------------------------------
# R-07: apply_removal clears the cell
# ---------------------------------------------------------------------------
def test_R07_removal_clears_cell():
    b = setup_board(8)
    b2 = apply_placement(b, 2, 2)
    b3 = apply_removal(b2, 2, 2)
    assert b3.grid[2, 2] == 0


# ---------------------------------------------------------------------------
# R-08: apply_removal is immutable
# ---------------------------------------------------------------------------
def test_R08_immutable_removal():
    b = setup_board(8)
    b2 = apply_placement(b, 2, 2)
    _ = apply_removal(b2, 2, 2)
    assert b2.grid[2, 2] == PLAYER_1


# ---------------------------------------------------------------------------
# R-09: Mode-1 win detected after 4-in-a-row
# ---------------------------------------------------------------------------
def test_R09_mode1_win():
    b = setup_board(8)
    # P1 gets 4 in a row horizontally
    for c in range(4):
        b.grid[0, c] = PLAYER_1
    b.current_player = PLAYER_2  # simulating P1 just moved
    done, winner = check_terminal_mode1(b)
    assert done is True
    assert winner == PLAYER_1


# ---------------------------------------------------------------------------
# R-10: Mode-1 no terminal on fresh board
# ---------------------------------------------------------------------------
def test_R10_mode1_no_terminal_fresh():
    b = setup_board(8)
    done, winner = check_terminal_mode1(b)
    assert done is False
    assert winner is None


# ---------------------------------------------------------------------------
# R-11: Mode-1 draw when board full with no winner
# ---------------------------------------------------------------------------
def test_R11_mode1_draw_full_board():
    b = setup_board(4)  # small board for speed
    # Pattern with no 4-in-a-row in any direction (rows alternate in pairs
    # so diagonals are mixed: [0,0],[1,1],[2,2],[3,3] = 1,2,1,2 — no run of 4)
    pattern = [
        [1, 2, 1, 2],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [2, 1, 2, 1],
    ]
    for r in range(4):
        for c in range(4):
            b.grid[r, c] = pattern[r][c]
    b.current_player = PLAYER_2
    done, winner = check_terminal_mode1(b)
    assert done is True
    assert winner is None


# ---------------------------------------------------------------------------
# R-12: Mode-2 not terminal on partial board
# ---------------------------------------------------------------------------
def test_R12_mode2_partial_not_terminal():
    b = setup_board(8)
    b.grid[0, 0] = 1
    done, _ = check_terminal_mode2(b)
    assert done is False


# ---------------------------------------------------------------------------
# R-13: Mode-2 terminal when board full, higher score wins
# ---------------------------------------------------------------------------
def test_R13_mode2_full_board_winner():
    b = setup_board(4)
    # Fill entirely P1 (simple — P1 wins easily)
    for r in range(4):
        for c in range(4):
            b.grid[r, c] = PLAYER_1
    b.current_player = PLAYER_2
    done, winner = check_terminal_mode2(b)
    assert done is True
    assert winner == PLAYER_1


# ---------------------------------------------------------------------------
# R-14: legal_removals returns only pieces belonging to given player
# ---------------------------------------------------------------------------
def test_R14_legal_removals_correct_player():
    b = setup_board(8)
    b.grid[0, 0] = PLAYER_1
    b.grid[1, 1] = PLAYER_2
    b.grid[2, 2] = PLAYER_1
    p1_removals = legal_removals(b, PLAYER_1)
    p2_removals = legal_removals(b, PLAYER_2)
    assert (0, 0) in p1_removals
    assert (2, 2) in p1_removals
    assert (1, 1) not in p1_removals
    assert (1, 1) in p2_removals
    assert len(p2_removals) == 1
