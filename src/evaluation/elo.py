"""Elo rating arithmetic."""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given ratings A and B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Return updated (rating_a, rating_b).

    score_a: 1.0 = A wins, 0.5 = draw, 0.0 = B wins.
    """
    ea = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - ea))
    return new_a, new_b
