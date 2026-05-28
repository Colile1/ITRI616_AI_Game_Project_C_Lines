"""L-01 .. L-05: Elo rating tests."""

import pytest
from src.evaluation.elo import expected_score, update_elo


# ---------------------------------------------------------------------------
# L-01: equal ratings → expected score = 0.5
# ---------------------------------------------------------------------------
def test_L01_equal_ratings():
    assert expected_score(1000, 1000) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# L-02: higher rating → expected score > 0.5
# ---------------------------------------------------------------------------
def test_L02_higher_rating_higher_expected():
    assert expected_score(1200, 1000) > 0.5
    assert expected_score(800, 1000) < 0.5


# ---------------------------------------------------------------------------
# L-03: update_elo — winner gains, loser loses
# ---------------------------------------------------------------------------
def test_L03_winner_gains():
    ra, rb = update_elo(1000, 1000, score_a=1.0)
    assert ra > 1000
    assert rb < 1000


# ---------------------------------------------------------------------------
# L-04: draw keeps ratings close to original for equal players
# ---------------------------------------------------------------------------
def test_L04_draw_equal_players():
    ra, rb = update_elo(1000, 1000, score_a=0.5)
    assert ra == pytest.approx(1000.0)
    assert rb == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# L-05: rating change sums to zero (zero-sum property)
# ---------------------------------------------------------------------------
def test_L05_zero_sum():
    r1, r2 = 1200, 800
    for score in [0.0, 0.5, 1.0]:
        new1, new2 = update_elo(r1, r2, score)
        assert (new1 - r1) + (new2 - r2) == pytest.approx(0.0)
