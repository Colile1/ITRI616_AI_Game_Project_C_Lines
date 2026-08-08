"""Floating panels drawn over whatever screen is active: the game-over card,
a confirm dialog and the help sheet.

Each one owns its geometry, its hover state and its own event handling, and
returns a plain string action so the app never has to know their internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pygame

from src.config import MODE_FIRST_TO_FOUR, SCORE_FOR_LENGTH, PLAYER_1
from src.ui.theme import (
    TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P2_ACCENT, DRAW_GLOW, GLASS_EDGE,
    font, draw_button, draw_dim, truncate,
)

PANEL_FILL = (22, 38, 70, 232)


def _panel(surface: pygame.Surface, rect: pygame.Rect,
           accent: tuple[int, int, int], header_h: int = 0) -> None:
    """Rounded translucent panel with an optional coloured header band."""
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    panel.fill(PANEL_FILL)
    pygame.draw.rect(panel, (200, 220, 255, 70), panel.get_rect(), 2, border_radius=12)
    surface.blit(panel, rect.topleft)

    if header_h:
        hdr = pygame.Surface((rect.width, header_h), pygame.SRCALPHA)
        hdr.fill((*accent, 55))
        pygame.draw.rect(hdr, (*accent, 30), hdr.get_rect(), 0, border_radius=12)
        pygame.draw.rect(hdr, (*accent, 55), pygame.Rect(0, header_h - 12, rect.width, 12))
        surface.blit(hdr, rect.topleft)


def _centered(w: int, h: int, box_w: int, box_h: int) -> pygame.Rect:
    return pygame.Rect((w - box_w) // 2, (h - box_h) // 2, box_w, box_h)


# ---------------------------------------------------------------------------
# Confirm dialog
# ---------------------------------------------------------------------------

class ConfirmDialog:
    """A two-button yes/no panel.  Enter confirms, Escape cancels."""

    BOX_W, BOX_H = 420, 190
    BTN_W, BTN_H = 150, 40

    def __init__(
        self,
        title: str,
        message: str = "",
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        accent: tuple[int, int, int] = P2_ACCENT,
    ):
        self.title = title
        self.message = message
        self.confirm_label = confirm_label
        self.cancel_label = cancel_label
        self.accent = accent
        self._hovered: str | None = None

    def rects(self, w: int, h: int) -> dict[str, pygame.Rect]:
        box = _centered(w, h, self.BOX_W, self.BOX_H)
        gap = 16
        total = 2 * self.BTN_W + gap
        x = box.x + (box.width - total) // 2
        y = box.bottom - 22 - self.BTN_H
        return {
            "box": box,
            "cancel": pygame.Rect(x, y, self.BTN_W, self.BTN_H),
            "confirm": pygame.Rect(x + self.BTN_W + gap, y, self.BTN_W, self.BTN_H),
        }

    def draw(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        r = self.rects(w, h)
        box = r["box"]

        draw_dim(surface, 150)
        _panel(surface, box, self.accent, header_h=48)

        title = font("h2").render(self.title, True, TEXT_BRIGHT)
        surface.blit(title, (box.centerx - title.get_width() // 2, box.y + 12))

        if self.message:
            msg = font("small").render(
                truncate("small", self.message, box.width - 40), True, TEXT_MUTED
            )
            surface.blit(msg, (box.centerx - msg.get_width() // 2, box.y + 66))

        draw_button(surface, r["cancel"], self.cancel_label,
                    hovered=self._hovered == "cancel", base_surf=surface)
        draw_button(surface, r["confirm"], self.confirm_label,
                    hovered=self._hovered == "confirm", base_surf=surface,
                    accent=self.accent)

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> Optional[str]:
        w, h = surface.get_size()
        r = self.rects(w, h)

        if event.type == pygame.MOUSEMOTION:
            self._hovered = None
            for key in ("cancel", "confirm"):
                if r[key].collidepoint(event.pos):
                    self._hovered = key

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key in ("cancel", "confirm"):
                if r[key].collidepoint(event.pos):
                    return key
            if not r["box"].collidepoint(event.pos):
                return "cancel"   # click-away dismisses

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_y):
                return "confirm"
            if event.key in (pygame.K_ESCAPE, pygame.K_n):
                return "cancel"
        return None


# ---------------------------------------------------------------------------
# Game-over card
# ---------------------------------------------------------------------------

@dataclass
class GameOverData:
    winner: Optional[int]
    mode: str
    p1_label: str = "Player 1"
    p2_label: str = "Player 2"
    p1_score: float = 0.0
    p2_score: float = 0.0
    p1_lines: dict[int, int] = field(default_factory=dict)
    p2_lines: dict[int, int] = field(default_factory=dict)
    move_count: int = 0
    duration_sec: float = 0.0
    resigned_by: Optional[int] = None
    saved: bool = False
    can_save: bool = True


class GameOverCard:
    """Result panel with Replay / Save / New / Menu."""

    BOX_W, BOX_H = 560, 340
    HDR_H = 58
    BTN_H = 40

    ACTIONS = [
        ("replay", "Replay  (R)"),
        ("save",   "Save  (S)"),
        ("new",    "New  (N)"),
        ("menu",   "Menu  (M)"),
    ]

    def __init__(self, data: GameOverData):
        self.data = data
        self._hovered = -1

    # ------------------------------------------------------------------
    def rects(self, w: int, h: int) -> tuple[pygame.Rect, list[tuple[str, pygame.Rect]]]:
        box = _centered(w, h, self.BOX_W, self.BOX_H)
        gap = 10
        count = len(self.ACTIONS)
        btn_w = (box.width - 2 * 28 - gap * (count - 1)) // count
        y = box.bottom - 24 - self.BTN_H
        buttons = [
            (key, pygame.Rect(box.x + 28 + i * (btn_w + gap), y, btn_w, self.BTN_H))
            for i, (key, _) in enumerate(self.ACTIONS)
        ]
        return box, buttons

    # ------------------------------------------------------------------
    def _headline(self) -> tuple[str, tuple[int, int, int]]:
        d = self.data
        if d.winner is None:
            return "Draw", DRAW_GLOW
        label = d.p1_label if d.winner == PLAYER_1 else d.p2_label
        col = P1_ACCENT if d.winner == PLAYER_1 else P2_ACCENT
        verb = "win" if label == "You" else "wins"       # "You win", "Master wins"
        if d.resigned_by is not None:
            return f"{label} {verb} by resignation", col
        return f"{label} {verb}", col

    @staticmethod
    def _lines_text(lines: dict[int, int]) -> str:
        if not lines:
            return "no scoring lines"
        return "  ".join(
            f"{count}×{length} ({SCORE_FOR_LENGTH.get(min(length, 8), 0):g}p)"
            for length, count in lines.items()
        )

    # ------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, fade: float = 1.0) -> None:
        w, h = surface.get_size()
        box, buttons = self.rects(w, h)
        d = self.data

        draw_dim(surface, int(170 * max(0.0, min(1.0, fade))))
        headline, accent = self._headline()
        _panel(surface, box, accent, header_h=self.HDR_H)

        title = font("h2").render(truncate("h2", headline, box.width - 40), True, TEXT_BRIGHT)
        surface.blit(
            title,
            (box.centerx - title.get_width() // 2,
             box.y + (self.HDR_H - title.get_height()) // 2),
        )

        y = box.y + self.HDR_H + 18
        for label, score, lines, col in (
            (d.p1_label, d.p1_score, d.p1_lines, P1_ACCENT),
            (d.p2_label, d.p2_score, d.p2_lines, P2_ACCENT),
        ):
            name = font("ui").render(truncate("ui", label, 220), True, col)
            surface.blit(name, (box.x + 32, y))

            score_s = font("score").render(f"{score:.2f}", True, col)
            surface.blit(score_s, (box.right - 110 - score_s.get_width(), y))
            surface.blit(font("small").render("pts", True, TEXT_DIM), (box.right - 100, y + 5))

            breakdown = font("hint").render(
                truncate("hint", self._lines_text(lines), box.width - 64), True, TEXT_DIM
            )
            surface.blit(breakdown, (box.x + 32, y + 24))
            y += 56

        pygame.draw.line(
            surface, (*GLASS_EDGE, 40), (box.x + 28, y), (box.right - 28, y),
        )
        y += 12

        margin = abs(d.p1_score - d.p2_score)
        facts = [
            f"{d.move_count} moves",
            _format_duration(d.duration_sec),
        ]
        if d.mode != MODE_FIRST_TO_FOUR:
            facts.append(f"margin {margin:.2f} pts")
        facts_s = font("small").render("   ·   ".join(facts), True, TEXT_MUTED)
        surface.blit(facts_s, (box.centerx - facts_s.get_width() // 2, y))

        if d.saved:
            note = font("hint").render("game saved to results/games", True, TEXT_DIM)
            surface.blit(note, (box.centerx - note.get_width() // 2, y + 22))

        mx, my = pygame.mouse.get_pos()
        for i, (key, rect) in enumerate(buttons):
            label = dict(self.ACTIONS)[key]
            enabled = True
            if key == "save":
                enabled = d.can_save and not d.saved
                if d.saved:
                    label = "Saved"
            draw_button(
                surface, rect, label,
                hovered=(self._hovered == i or rect.collidepoint(mx, my)),
                base_surf=surface, enabled=enabled,
            )

    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> Optional[str]:
        w, h = surface.get_size()
        _, buttons = self.rects(w, h)

        if event.type == pygame.MOUSEMOTION:
            self._hovered = -1
            for i, (_, rect) in enumerate(buttons):
                if rect.collidepoint(event.pos):
                    self._hovered = i

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key, rect in buttons:
                if rect.collidepoint(event.pos):
                    if key == "save" and (self.data.saved or not self.data.can_save):
                        return None
                    return key

        if event.type == pygame.KEYDOWN:
            keymap = {
                pygame.K_r: "replay",
                pygame.K_s: "save",
                pygame.K_n: "new",
                pygame.K_m: "menu",
                pygame.K_ESCAPE: "menu",
            }
            action = keymap.get(event.key)
            if action == "save" and (self.data.saved or not self.data.can_save):
                return None
            return action
        return None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


# ---------------------------------------------------------------------------
# Help overlay
# ---------------------------------------------------------------------------

CONTROL_ROWS: list[tuple[str, str]] = [
    ("Click / Enter",  "Place a piece on the highlighted cell"),
    ("← ↑ → ↓",        "Move the keyboard cursor"),
    ("U  /  Ctrl+Z",   "Undo (rewinds the AI's reply too)"),
    ("Ctrl+Y",         "Redo"),
    ("H",              "Hint — the agent's own suggested move"),
    ("T",              "Toggle the threat overlay"),
    ("E",              "Toggle the evaluation meter"),
    ("Q",              "Resign (asks first)"),
    ("F1  /  ?",       "This help sheet"),
    ("ESC",            "Back / main menu"),
]


class HelpOverlay:
    """Controls and scoring reference.  Any key closes it."""

    BOX_W, BOX_H = 640, 430

    def draw(self, surface: pygame.Surface, mode: str = "") -> None:
        w, h = surface.get_size()
        box = _centered(w, h, min(self.BOX_W, w - 40), min(self.BOX_H, h - 40))

        draw_dim(surface, 190)
        _panel(surface, box, P1_ACCENT, header_h=52)

        title = font("h2").render("Controls & Scoring", True, TEXT_BRIGHT)
        surface.blit(title, (box.centerx - title.get_width() // 2, box.y + 13))

        y = box.y + 66
        for keys, desc in CONTROL_ROWS:
            if y > box.bottom - 150:
                break
            surface.blit(font("mono_sm").render(keys, True, P1_ACCENT), (box.x + 28, y))
            surface.blit(
                font("small").render(truncate("small", desc, box.width - 200), True, TEXT_MUTED),
                (box.x + 160, y - 1),
            )
            y += 21

        y += 10
        pygame.draw.line(surface, (*GLASS_EDGE, 40), (box.x + 24, y), (box.right - 24, y))
        y += 12

        surface.blit(font("h3").render("Line scoring", True, TEXT_BRIGHT), (box.x + 28, y))
        y += 22
        table = "   ".join(f"{length}→{pts:g}p" for length, pts in sorted(SCORE_FOR_LENGTH.items()))
        surface.blit(font("mono_sm").render(table, True, TEXT_MUTED), (box.x + 28, y))
        y += 22

        rule = (
            "Every maximal run is scored once per direction; runs shorter than 3 score nothing."
            if mode != MODE_FIRST_TO_FOUR else
            "First to complete a run of four or more wins immediately."
        )
        surface.blit(
            font("hint").render(truncate("hint", rule, box.width - 56), True, TEXT_DIM),
            (box.x + 28, y),
        )

        close = font("small").render("any key or click to close", True, TEXT_DIM)
        surface.blit(close, (box.centerx - close.get_width() // 2, box.bottom - 26))

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            return "close"
        return None
