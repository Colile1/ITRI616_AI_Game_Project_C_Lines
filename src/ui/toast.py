"""Transient status messages ("Undo", "Hint: D4", "Game saved").

Toasts stack upward from the bottom of the area they are drawn in and fade out
on their own.  They carry no interaction, so they can be drawn over anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from src.config import TOAST_MS, TOAST_FADE_MS
from src.ui.theme import (
    BG_TOP, TEXT_BRIGHT,
    P1_ACCENT, HL_THREAT_OPP, HL_LAST_MOVE, HL_HINT,
    font,
)

KIND_COLORS: dict[str, tuple[int, int, int]] = {
    "info":  P1_ACCENT,
    "good":  (110, 240, 170),
    "warn":  HL_LAST_MOVE,
    "error": HL_THREAT_OPP,
    "hint":  HL_HINT,
}

MAX_VISIBLE = 4
TOAST_H = 30
TOAST_GAP = 6
PAD_X = 14


@dataclass
class _Toast:
    text: str
    kind: str
    born_ms: int

    def age(self, now: int) -> int:
        return now - self.born_ms

    def alpha(self, now: int) -> int:
        """255 while fresh, ramping to 0 across the fade window."""
        age = self.age(now)
        if age <= TOAST_MS:
            return 255
        fade = age - TOAST_MS
        if fade >= TOAST_FADE_MS:
            return 0
        return int(255 * (1.0 - fade / TOAST_FADE_MS))

    def expired(self, now: int) -> bool:
        return self.age(now) >= TOAST_MS + TOAST_FADE_MS


class Toaster:
    """A short queue of fading messages."""

    def __init__(self) -> None:
        self._items: list[_Toast] = []

    # ------------------------------------------------------------------
    def push(self, text: str, kind: str = "info") -> None:
        """Show a message.  Re-pushing the same text just restarts its timer."""
        now = pygame.time.get_ticks()
        for item in self._items:
            if item.text == text and item.kind == kind:
                item.born_ms = now
                return
        self._items.append(_Toast(text=text, kind=kind, born_ms=now))
        if len(self._items) > MAX_VISIBLE:
            del self._items[: len(self._items) - MAX_VISIBLE]

    def clear(self) -> None:
        self._items.clear()

    def update(self) -> None:
        now = pygame.time.get_ticks()
        self._items = [t for t in self._items if not t.expired(now)]

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, area: pygame.Rect | None = None) -> None:
        """Draw the stack anchored to the bottom-centre of *area* (or the surface)."""
        self.update()
        if not self._items:
            return

        rect = area or surface.get_rect()
        now = pygame.time.get_ticks()
        f = font("small")

        for i, item in enumerate(reversed(self._items)):
            alpha = item.alpha(now)
            if alpha <= 0:
                continue
            label = f.render(item.text, True, TEXT_BRIGHT)
            box_w = label.get_width() + PAD_X * 2
            box_x = rect.x + (rect.width - box_w) // 2
            box_y = rect.bottom - 18 - TOAST_H - i * (TOAST_H + TOAST_GAP)

            col = KIND_COLORS.get(item.kind, P1_ACCENT)
            box = pygame.Surface((box_w, TOAST_H), pygame.SRCALPHA)
            box.fill((*BG_TOP, int(alpha * 0.86)))
            pygame.draw.rect(box, (*col, int(alpha * 0.75)), box.get_rect(), 1, border_radius=6)
            pygame.draw.rect(box, (*col, int(alpha * 0.9)), pygame.Rect(0, 0, 3, TOAST_H))
            label.set_alpha(alpha)
            box.blit(label, (PAD_X, (TOAST_H - label.get_height()) // 2))
            surface.blit(box, (box_x, box_y))
