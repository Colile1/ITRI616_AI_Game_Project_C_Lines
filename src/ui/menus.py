"""Menu screens: MainMenu, BoardSizePicker, ModePicker, SettingsScreen."""

from __future__ import annotations

import random
from typing import Optional

import pygame

from src.ui.theme import (
    BG_DEEP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P1_GLOW, P2_ACCENT, P2_GLOW,
    font, draw_button, draw_card, draw_toggle, draw_segmented, truncate,
)
from src.config import (
    UI_BOARD_SIZES as BOARD_SIZES,
    MODE_FIRST_TO_FOUR, MODE_POINTS_FULL,
    AI_SPEEDS,
)

# Fixed palette for decorative background pieces
_DECO_COLORS = [P1_ACCENT, P2_ACCENT, P1_GLOW, P2_GLOW, (140, 200, 180), (200, 160, 240)]


def _draw_deco(surface: pygame.Surface, deco: list) -> None:
    w, h = surface.get_size()
    for xf, yf, radius, color, alpha in deco:
        s = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        if radius < 14:
            pygame.draw.circle(s, (*color, alpha), (radius + 2, radius + 2), radius)
        else:
            pygame.draw.circle(s, (*color, alpha), (radius + 2, radius + 2), radius, 2)
        surface.blit(s, (int(w * xf) - radius - 2, int(h * yf) - radius - 2))


def _bake_deco(count: int = 20, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [
        (
            rng.uniform(0.04, 0.96),
            rng.uniform(0.04, 0.96),
            rng.randint(7, 26),
            _DECO_COLORS[rng.randint(0, len(_DECO_COLORS) - 1)],
            rng.randint(14, 42),
        )
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

class MainMenu:
    ITEMS: list[tuple[str, str]] = [
        ("play_vs_ai", "Play vs AI"),
        ("hot_seat",   "Hot-seat"),
        ("replays",    "Replays"),
        ("stats",      "Statistics"),
        ("settings",   "Settings"),
        ("quit",       "Quit"),
    ]
    BTN_W, BTN_H, BTN_GAP = 260, 48, 12

    def __init__(self):
        self._hovered: int = -1
        self._focus: int = 0          # keyboard focus
        self._deco = _bake_deco()
        self.footer: str = ""         # set by the app (e.g. "12 games · 42% win rate")

    # ------------------------------------------------------------------
    def _rects(self, surface: pygame.Surface) -> list[pygame.Rect]:
        w, h = surface.get_size()
        total_h = len(self.ITEMS) * (self.BTN_H + self.BTN_GAP) - self.BTN_GAP
        start_y = max(h // 5 + 96, (h - total_h) // 2 + 40)
        x = (w - self.BTN_W) // 2
        return [
            pygame.Rect(x, start_y + i * (self.BTN_H + self.BTN_GAP), self.BTN_W, self.BTN_H)
            for i in range(len(self.ITEMS))
        ]

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()
        _draw_deco(surface, self._deco)

        title = font("display").render("C_lines", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, h // 5))
        sub = font("ui").render("AI Board Game — ITRI616", True, TEXT_MUTED)
        surface.blit(sub, ((w - sub.get_width()) // 2, h // 5 + 48))

        for i, rect in enumerate(self._rects(surface)):
            label = self.ITEMS[i][1]
            draw_button(
                surface, rect, label,
                hovered=(self._hovered == i or self._focus == i),
                base_surf=surface,
            )

        if self.footer:
            fs = font("small").render(self.footer, True, TEXT_DIM)
            surface.blit(fs, ((w - fs.get_width()) // 2, h - 40))

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> Optional[str]:
        rects = self._rects(surface)

        if event.type == pygame.MOUSEMOTION:
            self._hovered = -1
            for i, rect in enumerate(rects):
                if rect.collidepoint(event.pos):
                    self._hovered = i
                    self._focus = i

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, rect in enumerate(rects):
                if rect.collidepoint(event.pos):
                    return self.ITEMS[i][0]

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DOWN, pygame.K_TAB):
                self._focus = (self._focus + 1) % len(self.ITEMS)
            elif event.key == pygame.K_UP:
                self._focus = (self._focus - 1) % len(self.ITEMS)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self.ITEMS[self._focus][0]
            # ESC is "back" everywhere else in the app; there is nowhere to go
            # back to from here, so it does nothing rather than quitting.
        return None


# ---------------------------------------------------------------------------
# Board size
# ---------------------------------------------------------------------------

class BoardSizePicker:
    CHIP_R = 36

    def __init__(self):
        # Must be a size the UI actually offers — otherwise Enter would start a
        # game on a board with no trained snapshots.
        self._selected: int = BOARD_SIZES[0]
        self._hovered: int = -1

    @property
    def selected(self) -> int:
        return self._selected

    def _centers(self, surface: pygame.Surface) -> list[tuple[int, int]]:
        w, h = surface.get_size()
        spacing = self.CHIP_R * 3
        total_w = len(BOARD_SIZES) * spacing
        start_x = (w - total_w) // 2 + self.CHIP_R
        cy = h // 2
        return [(start_x + i * spacing, cy) for i in range(len(BOARD_SIZES))]

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()

        title = font("display").render("Board Size", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, h // 4))
        hint = font("ui").render("Click a size to continue  ·  ESC = back", True, TEXT_MUTED)
        surface.blit(hint, ((w - hint.get_width()) // 2, h // 4 + 48))

        for i, (cx, cy) in enumerate(self._centers(surface)):
            sz = BOARD_SIZES[i]
            selected = sz == self._selected
            hovered = i == self._hovered
            color = P1_ACCENT if selected else (TEXT_MUTED if hovered else TEXT_DIM)
            pygame.draw.circle(surface, color, (cx, cy), self.CHIP_R, 0 if selected else 2)
            label = font("display" if selected else "ui").render(
                str(sz), True, BG_DEEP if selected else color
            )
            surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))
            cap = font("hint").render(f"{sz}×{sz}", True, TEXT_DIM)
            surface.blit(cap, (cx - cap.get_width() // 2, cy + self.CHIP_R + 10))

        if len(BOARD_SIZES) == 1:
            note = font("small").render(
                "Only 8×8 has trained agents — other sizes are disabled.", True, TEXT_DIM
            )
            surface.blit(note, ((w - note.get_width()) // 2, h // 2 + 110))

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> Optional[str]:
        centers = self._centers(surface)

        if event.type == pygame.MOUSEMOTION:
            self._hovered = -1
            for i, (cx, cy) in enumerate(centers):
                if (event.pos[0] - cx) ** 2 + (event.pos[1] - cy) ** 2 <= self.CHIP_R ** 2:
                    self._hovered = i

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, (cx, cy) in enumerate(centers):
                if (event.pos[0] - cx) ** 2 + (event.pos[1] - cy) ** 2 <= self.CHIP_R ** 2:
                    self._selected = BOARD_SIZES[i]
                    return "confirm"

        if event.type == pygame.KEYDOWN:
            idx = BOARD_SIZES.index(self._selected)
            if event.key == pygame.K_RIGHT:
                self._selected = BOARD_SIZES[(idx + 1) % len(BOARD_SIZES)]
            elif event.key == pygame.K_LEFT:
                self._selected = BOARD_SIZES[(idx - 1) % len(BOARD_SIZES)]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return "confirm"
            elif event.key == pygame.K_ESCAPE:
                return "back"
        return None


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------

class ModePicker:
    BTN_W, BTN_H = 340, 76
    GAP = 40

    MODES: list[tuple[str, str, str]] = [
        (MODE_FIRST_TO_FOUR, "First to Four", "First run of four wins immediately"),
        (MODE_POINTS_FULL, "Points Until Full", "Fill the board — most line points wins"),
    ]

    def __init__(self):
        self._selected: str = MODE_POINTS_FULL
        self._hovered: Optional[str] = None
        self._focus: int = 1

    @property
    def selected(self) -> str:
        return self._selected

    def _rects(self, surface: pygame.Surface) -> list[pygame.Rect]:
        w, h = surface.get_size()
        total_h = len(self.MODES) * (self.BTN_H + self.GAP) - self.GAP
        start_y = (h - total_h) // 2
        x = (w - self.BTN_W) // 2
        return [
            pygame.Rect(x, start_y + i * (self.BTN_H + self.GAP), self.BTN_W, self.BTN_H)
            for i in range(len(self.MODES))
        ]

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()

        title = font("display").render("Game Mode", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, h // 4 - 20))

        for i, rect in enumerate(self._rects(surface)):
            mode, label, desc = self.MODES[i]
            active = (mode == self._hovered) or (self._focus == i)
            draw_button(surface, rect, label, hovered=active, base_surf=surface)
            sub = font("small").render(desc, True, TEXT_DIM)
            surface.blit(sub, (rect.centerx - sub.get_width() // 2, rect.bottom + 8))

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> Optional[str]:
        rects = self._rects(surface)

        if event.type == pygame.MOUSEMOTION:
            self._hovered = None
            for i, rect in enumerate(rects):
                if rect.collidepoint(event.pos):
                    self._hovered = self.MODES[i][0]
                    self._focus = i

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, rect in enumerate(rects):
                if rect.collidepoint(event.pos):
                    self._selected = self.MODES[i][0]
                    return "confirm"

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DOWN, pygame.K_TAB):
                self._focus = (self._focus + 1) % len(self.MODES)
            elif event.key == pygame.K_UP:
                self._focus = (self._focus - 1) % len(self.MODES)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._selected = self.MODES[self._focus][0]
                return "confirm"
            elif event.key == pygame.K_ESCAPE:
                return "back"
        return None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsScreen:
    """Toggles plus the AI pace selector.

    `handle_event` returns "changed" whenever a value moved, so the app can
    persist settings immediately rather than only on exit.
    """

    # (kind, key, label, description)
    ROWS: list[tuple[str, str, str, str]] = [
        ("toggle", "show_legal",    "Show legal moves",   "Tint every cell you may play"),
        ("toggle", "show_threats",  "Threat overlay",     "Highlight open threes, fours and losing cells"),
        ("toggle", "show_coords",   "Board coordinates",  "Letters and numbers around the grid"),
        ("toggle", "show_eval",     "Evaluation meter",   "Live advantage bar in the sidebar"),
        ("toggle", "reduce_motion", "Reduce motion",      "Disable drops, pulses and fades"),
        ("toggle", "autosave",      "Auto-save games",    "Write finished games to results/games"),
        ("toggle", "sound",         "Sound",              "No audio ships with this build"),
        ("choice", "ai_speed",      "AI move pace",       "Minimum time the AI appears to think"),
    ]
    ROW_H = 50
    ROW_GAP = 4
    CARD_W = 560

    def __init__(self, settings: dict):
        self._settings = settings
        self._hovered = -1

    @property
    def settings(self) -> dict:
        return self._settings

    # ------------------------------------------------------------------
    def _card_x(self, surface: pygame.Surface) -> int:
        w, _ = surface.get_size()
        return (w - min(self.CARD_W, w - 60)) // 2

    def _card_w(self, surface: pygame.Surface) -> int:
        w, _ = surface.get_size()
        return min(self.CARD_W, w - 60)

    def _row_rect(self, idx: int, surface: pygame.Surface) -> pygame.Rect:
        return pygame.Rect(
            self._card_x(surface),
            126 + idx * (self.ROW_H + self.ROW_GAP),
            self._card_w(surface),
            self.ROW_H,
        )

    def _choice_rect(self, row: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(row.right - 236, row.y + 11, 224, 28)

    def _back_rect(self, surface: pygame.Surface) -> pygame.Rect:
        _, h = surface.get_size()
        return pygame.Rect(40, h - 50, 110, 34)

    # ------------------------------------------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()

        title = font("display").render("Settings", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, 34))
        hint = font("small").render(
            "Changes save immediately  ·  ESC = back", True, TEXT_DIM
        )
        surface.blit(hint, ((w - hint.get_width()) // 2, 84))

        mx, my = pygame.mouse.get_pos()
        for i, (kind, key, label, desc) in enumerate(self.ROWS):
            row = self._row_rect(i, surface)
            value = self._settings.get(key)
            on = bool(value) if kind == "toggle" else True

            draw_card(
                surface, row,
                stripe_color=P1_ACCENT if (kind == "toggle" and on) else None,
                fill_alpha=48 if (kind == "toggle" and on) else 20,
                border_alpha=90 if i == self._hovered else 40,
            )
            surface.blit(
                font("ui").render(label, True, TEXT_BRIGHT if on else TEXT_MUTED),
                (row.x + 16, row.y + 6),
            )
            surface.blit(
                font("hint").render(truncate("hint", desc, row.width - 260), True, TEXT_DIM),
                (row.x + 16, row.y + 28),
            )

            if kind == "toggle":
                draw_toggle(surface, row.right - 34, row.centery, bool(value))
            else:
                choice = self._choice_rect(row)
                idx = AI_SPEEDS.index(value) if value in AI_SPEEDS else 0
                hov = -1
                if choice.collidepoint(mx, my):
                    seg_w = choice.width // len(AI_SPEEDS)
                    hov = min(len(AI_SPEEDS) - 1, max(0, (mx - choice.x) // seg_w))
                draw_segmented(
                    surface, choice, [s.title() for s in AI_SPEEDS], idx, hovered_index=hov
                )

        back = self._back_rect(surface)
        draw_button(surface, back, "◄  Back", hovered=back.collidepoint(mx, my),
                    base_surf=surface)

    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> Optional[str]:
        if event.type == pygame.MOUSEMOTION:
            self._hovered = -1
            for i in range(len(self.ROWS)):
                if self._row_rect(i, surface).collidepoint(event.pos):
                    self._hovered = i

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._back_rect(surface).collidepoint(event.pos):
                return "back"
            for i, (kind, key, _, _) in enumerate(self.ROWS):
                row = self._row_rect(i, surface)
                if not row.collidepoint(event.pos):
                    continue
                if kind == "toggle":
                    self._settings[key] = not bool(self._settings.get(key, False))
                    return "changed"
                choice = self._choice_rect(row)
                if choice.collidepoint(event.pos):
                    seg_w = choice.width // len(AI_SPEEDS)
                    idx = min(len(AI_SPEEDS) - 1, max(0, (event.pos[0] - choice.x) // seg_w))
                    self._settings[key] = AI_SPEEDS[idx]
                    return "changed"
                return None

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "back"
        return None
