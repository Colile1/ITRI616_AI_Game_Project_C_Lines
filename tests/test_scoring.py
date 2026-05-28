"""S-01 .. S-13: scoring module tests."""

import numpy as np
import pytest
from src.engine.board import setup_board, clone_board
from src.engine.scoring import score_for_length, score_board, compute_scores


# ---------------------------------------------------------------------------
# S-01: score_for_length table values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length,expected", [
    (3, 0.25), (4, 1.0), (5, 2.0), (6, 3.0), (7, 4.0), (8, 5.0),
])
def test_S01_score_table(length, expected):
    assert score_for_length(length) == expected


# ---------------------------------------------------------------------------
# S-02: lengths below MIN_SCORING_LENGTH score 0
# ---------------------------------------------------------------------------
def test_S02_below_min_scores_zero():
    assert score_for_length(1) == 0.0
    assert score_for_length(2) == 0.0


# ---------------------------------------------------------------------------
# S-03: lengths above MAX_SCORING_LENGTH capped at 8
# ---------------------------------------------------------------------------
def test_S03_cap_at_max():
    assert score_for_length(9) == score_for_length(8)
    assert score_for_length(12) == score_for_length(8)


# ---------------------------------------------------------------------------
# S-04: empty board scores 0
# ---------------------------------------------------------------------------
def test_S04_empty_board():
    b = setup_board(8)
    p1, p2 = compute_scores(b)
    assert p1 == 0.0 and p2 == 0.0


# ---------------------------------------------------------------------------
# S-05: horizontal run of 4 scores 1.0
# ---------------------------------------------------------------------------
def test_S05_horizontal_4():
    b = setup_board(8)
    for c in range(4):
        b.grid[0, c] = 1
    p1, _ = compute_scores(b)
    assert p1 == 1.0


# ---------------------------------------------------------------------------
# S-06: vertical run of 3 scores 0.25
# ---------------------------------------------------------------------------
def test_S06_vertical_3():
    b = setup_board(8)
    for r in range(3):
        b.grid[r, 0] = 2
    _, p2 = compute_scores(b)
    assert p2 == 0.25


# ---------------------------------------------------------------------------
# S-07: diagonal run of 5 scores 2.0
# ---------------------------------------------------------------------------
def test_S07_diagonal_5():
    b = setup_board(8)
    for i in range(5):
        b.grid[i, i] = 1
    p1, _ = compute_scores(b)
    assert p1 == 2.0


# ---------------------------------------------------------------------------
# S-08: anti-diagonal run of 6 scores 3.0
# ---------------------------------------------------------------------------
def test_S08_antidiagonal_6():
    b = setup_board(8)
    for i in range(6):
        b.grid[i, 7 - i] = 1
    p1, _ = compute_scores(b)
    assert p1 == 3.0


# ---------------------------------------------------------------------------
# S-09: run of 10 capped to score of 8 (5.0)
# ---------------------------------------------------------------------------
def test_S09_long_run_capped():
    b = setup_board(12)
    for c in range(10):
        b.grid[0, c] = 1
    p1, _ = compute_scores(b)
    assert p1 == 5.0


# ---------------------------------------------------------------------------
# S-10: two separate runs on same row scored independently
# ---------------------------------------------------------------------------
def test_S10_two_separate_runs():
    b = setup_board(12)
    # Run of 4 at cols 0-3, run of 3 at cols 6-8 (gap at col 4,5)
    for c in range(4):
        b.grid[0, c] = 1
    for c in range(6, 9):
        b.grid[0, c] = 1
    p1, _ = compute_scores(b)
    assert p1 == pytest.approx(1.0 + 0.25)


# ---------------------------------------------------------------------------
# S-11: opponent pieces act as interruptions
# ---------------------------------------------------------------------------
def test_S11_interrupted_by_opponent():
    b = setup_board(8)
    # P1 at cols 0,1, P2 at col 2, P1 at cols 3,4,5,6 — interrupted
    for c in [0, 1]:
        b.grid[0, c] = 1
    b.grid[0, 2] = 2
    for c in [3, 4, 5, 6]:
        b.grid[0, c] = 1
    p1, _ = compute_scores(b)
    # P1 has run-of-2 (no score) + run-of-4 (1.0)
    assert p1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# S-12: same piece can score in multiple directions
# ---------------------------------------------------------------------------
def test_S12_multi_direction_fulcrum():
    b = setup_board(8)
    # Horizontal run of 4 across row 3, cols 2-5
    for c in range(2, 6):
        b.grid[3, c] = 1
    # Vertical run of 4 down col 4, rows 0-3 (cell (3,4) already set)
    for r in range(3):
        b.grid[r, 4] = 1
    p1, _ = compute_scores(b)
    # Both lines score independently: 1.0 + 1.0 = 2.0
    assert p1 == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# S-13: compute_scores returns both players correctly
# ---------------------------------------------------------------------------
def test_S13_both_players():
    b = setup_board(8)
    for c in range(3):
        b.grid[0, c] = 1   # P1 run-of-3 → 0.25
    for c in range(4):
        b.grid[7, c] = 2   # P2 run-of-4 → 1.0
    p1, p2 = compute_scores(b)
    assert p1 == pytest.approx(0.25)
    assert p2 == pytest.approx(1.0)
