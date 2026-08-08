"""Small time-based animation helpers.

Deliberately frame-rate independent: every animation is driven by wall-clock
milliseconds from `pygame.time.get_ticks()`, so a slow frame slows nothing down
and a fast machine does not fast-forward.  When "reduce motion" is on the app
simply does not create Tweens, and every call site treats a missing tween as
"already finished".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_out_back(t: float) -> float:
    """Overshoots slightly then settles — used for the piece drop."""
    t = max(0.0, min(1.0, t))
    c1, c3 = 1.70158, 2.70158
    return 1.0 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def pulse(period_ms: int = 1100, now_ms: int | None = None) -> float:
    """A 0..1 triangle-ish wave for breathing highlights."""
    now = pygame.time.get_ticks() if now_ms is None else now_ms
    if period_ms <= 0:
        return 1.0
    return 0.5 - 0.5 * math.cos(2 * math.pi * (now % period_ms) / period_ms)


# ---------------------------------------------------------------------------
# Tween
# ---------------------------------------------------------------------------

@dataclass
class Tween:
    """A one-shot 0 → 1 ramp over *duration_ms*, started at construction."""

    duration_ms: int
    start_ms: int = -1

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            self.start_ms = pygame.time.get_ticks()

    def raw(self, now_ms: int | None = None) -> float:
        """Linear progress in [0, 1]."""
        if self.duration_ms <= 0:
            return 1.0
        now = pygame.time.get_ticks() if now_ms is None else now_ms
        return max(0.0, min(1.0, (now - self.start_ms) / self.duration_ms))

    def value(self, now_ms: int | None = None) -> float:
        """Eased progress in [0, 1]."""
        return ease_out_cubic(self.raw(now_ms))

    def done(self, now_ms: int | None = None) -> bool:
        return self.raw(now_ms) >= 1.0

    def restart(self) -> None:
        self.start_ms = pygame.time.get_ticks()


@dataclass
class DropAnim:
    """The 'piece lands on the board' animation for one cell."""

    row: int
    col: int
    player: int
    tween: Tween

    @classmethod
    def start(cls, row: int, col: int, player: int, duration_ms: int) -> "DropAnim":
        return cls(row=row, col=col, player=player, tween=Tween(duration_ms))

    def scale(self) -> float:
        """Radius multiplier — drops in from 1.6× with a soft overshoot."""
        t = self.tween.raw()
        return 1.6 - 0.6 * ease_out_back(t)

    def alpha(self) -> int:
        return int(70 + 185 * ease_out_cubic(self.tween.raw()))

    def done(self) -> bool:
        return self.tween.done()
