"""Saved-games browser — pick a recorded game and open it in the replay viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pygame

from src.config import GAMES_DIR, MODE_FIRST_TO_FOUR
from src.game.records import GameRecord, list_records, delete_record
from src.ui.theme import (
    BG_DEEP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P2_ACCENT, DRAW_GLOW, BAND_COLORS,
    font, draw_button, draw_card, truncate,
)

HEADER_H = 108
ROW_H = 62
ROW_GAP = 8
SIDE_PAD = 40
SCROLLBAR_W = 10
FOOTER_H = 56
DOUBLE_CLICK_MS = 400

RESULT_COLORS = {
    "Win":  (110, 240, 170),
    "Loss": (245, 100, 120),
    "Draw": DRAW_GLOW,
}


class ReplayBrowser:
    """A scrollable list of saved games with Open / Delete per row."""

    def __init__(self, directory: Path = GAMES_DIR):
        self._dir = Path(directory)
        self._rows: list[tuple[Path, GameRecord]] = []
        self._hovered = -1
        self._scroll_y = 0
        self._selected_path: Optional[Path] = None
        # Double-click tracking is per-row: a fast click on row A then row B is
        # two single clicks, not a double-click on B.
        self._last_click_idx = -1
        self._last_click_ms = 0
        self.refresh()

    def _is_double_click(self, idx: int) -> bool:
        now = pygame.time.get_ticks()
        same_row = idx == self._last_click_idx
        quick = (now - self._last_click_ms) <= DOUBLE_CLICK_MS
        self._last_click_idx = idx
        self._last_click_ms = now
        return same_row and quick

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._rows = list_records(self._dir)
        self._scroll_y = 0
        self._hovered = -1

    @property
    def selected_record(self) -> Optional[GameRecord]:
        for path, rec in self._rows:
            if path == self._selected_path:
                return rec
        return None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _viewport(self, surface: pygame.Surface) -> pygame.Rect:
        w, h = surface.get_size()
        return pygame.Rect(SIDE_PAD, HEADER_H, w - 2 * SIDE_PAD - SCROLLBAR_W,
                           max(0, h - HEADER_H - FOOTER_H))

    def _content_h(self) -> int:
        if not self._rows:
            return 0
        return len(self._rows) * (ROW_H + ROW_GAP) - ROW_GAP

    def _max_scroll(self, viewport_h: int) -> int:
        return max(0, self._content_h() - viewport_h)

    def _row_rect(self, idx: int, viewport: pygame.Rect) -> pygame.Rect:
        y = viewport.y + idx * (ROW_H + ROW_GAP) - self._scroll_y
        return pygame.Rect(viewport.x, y, viewport.width, ROW_H)

    @staticmethod
    def _row_buttons(row: pygame.Rect) -> dict[str, pygame.Rect]:
        return {
            "open":   pygame.Rect(row.right - 168, row.y + 15, 78, 32),
            "delete": pygame.Rect(row.right - 84, row.y + 15, 74, 32),
        }

    def _back_rect(self, surface: pygame.Surface) -> pygame.Rect:
        _, h = surface.get_size()
        return pygame.Rect(SIDE_PAD, h - 46, 110, 34)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()
        viewport = self._viewport(surface)

        title = font("display").render("Saved Games", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, 26))
        hint = font("ui").render(
            f"{len(self._rows)} saved  ·  double-click a row to open  ·  ESC = back",
            True, TEXT_MUTED,
        )
        surface.blit(hint, ((w - hint.get_width()) // 2, 70))

        if not self._rows:
            msg = font("ui").render(
                "No saved games yet — finish a game and press Save.", True, TEXT_DIM
            )
            surface.blit(msg, ((w - msg.get_width()) // 2, h // 2))
            self._draw_back(surface)
            return

        old_clip = surface.get_clip()
        surface.set_clip(viewport)
        for idx, (path, rec) in enumerate(self._rows):
            row = self._row_rect(idx, viewport)
            if row.bottom < viewport.y or row.top > viewport.bottom:
                continue
            self._draw_row(surface, row, rec, hovered=(idx == self._hovered),
                           selected=(path == self._selected_path))
        surface.set_clip(old_clip)

        self._draw_scrollbar(surface, viewport)
        self._draw_back(surface)

    def _draw_row(
        self, surface: pygame.Surface, row: pygame.Rect,
        rec: GameRecord, hovered: bool, selected: bool,
    ) -> None:
        result = rec.result_label
        accent = RESULT_COLORS.get(result, P1_ACCENT)
        draw_card(
            surface, row,
            stripe_color=accent if (hovered or selected) else None,
            fill_alpha=52 if (hovered or selected) else 22,
            border_alpha=110 if (hovered or selected) else 42,
        )

        # Result badge
        badge = pygame.Rect(row.x + 14, row.y + 18, 60, 26)
        badge_surf = pygame.Surface(badge.size, pygame.SRCALPHA)
        badge_surf.fill((*accent, 60))
        surface.blit(badge_surf, badge.topleft)
        pygame.draw.rect(surface, accent, badge, 1, border_radius=4)
        bl = font("small").render(result, True, TEXT_BRIGHT)
        surface.blit(bl, (badge.centerx - bl.get_width() // 2,
                          badge.centery - bl.get_height() // 2))

        # Primary line: opponent and mode
        mode_str = "First to Four" if rec.mode == MODE_FIRST_TO_FOUR else "Points Until Full"
        headline = f"{rec.opponent_label}  ·  {mode_str}  ·  {rec.board_size}×{rec.board_size}"
        surface.blit(
            font("ui").render(truncate("ui", headline, row.width - 260), True, TEXT_BRIGHT),
            (row.x + 88, row.y + 10),
        )

        # Secondary line: when, length, score
        when = rec.started_at[:16].replace("T", " ")
        detail = f"{when}   ·   {rec.move_count} moves   ·   {rec.p1_score:.2f} – {rec.p2_score:.2f}"
        surface.blit(
            font("hint").render(truncate("hint", detail, row.width - 260), True, TEXT_DIM),
            (row.x + 88, row.y + 34),
        )

        if rec.ai_band:
            band_col = BAND_COLORS.get(rec.ai_band, TEXT_MUTED)
            bs = font("hint").render(rec.ai_band.upper(), True, band_col)
            surface.blit(bs, (row.right - 176 - bs.get_width(), row.y + 22))

        mx, my = pygame.mouse.get_pos()
        buttons = self._row_buttons(row)
        draw_button(surface, buttons["open"], "Open",
                    hovered=buttons["open"].collidepoint(mx, my), base_surf=surface)
        draw_button(surface, buttons["delete"], "Delete",
                    hovered=buttons["delete"].collidepoint(mx, my), base_surf=surface,
                    accent=P2_ACCENT)

    def _draw_scrollbar(self, surface: pygame.Surface, viewport: pygame.Rect) -> None:
        max_sc = self._max_scroll(viewport.height)
        if max_sc <= 0:
            return
        bar_h = max(30, int(viewport.height * viewport.height / max(1, self._content_h())))
        bar_y = viewport.y + int((viewport.height - bar_h) * self._scroll_y / max_sc)
        bar_x = viewport.right + 4
        pygame.draw.rect(surface, (40, 48, 66),
                         pygame.Rect(bar_x, viewport.y, SCROLLBAR_W, viewport.height),
                         border_radius=4)
        pygame.draw.rect(surface, (92, 108, 138),
                         pygame.Rect(bar_x, bar_y, SCROLLBAR_W, bar_h), border_radius=4)

    def _draw_back(self, surface: pygame.Surface) -> None:
        rect = self._back_rect(surface)
        mx, my = pygame.mouse.get_pos()
        draw_button(surface, rect, "◄  Back",
                    hovered=rect.collidepoint(mx, my), base_surf=surface)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> Optional[str]:
        viewport = self._viewport(surface)
        max_sc = self._max_scroll(viewport.height)

        if event.type == pygame.MOUSEWHEEL:
            self._scroll_y = max(0, min(max_sc, self._scroll_y - event.y * 40))
            return None

        if event.type == pygame.MOUSEMOTION:
            self._hovered = -1
            for idx in range(len(self._rows)):
                rect = self._row_rect(idx, viewport)
                if rect.collidepoint(event.pos) and viewport.collidepoint(event.pos):
                    self._hovered = idx
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._back_rect(surface).collidepoint(event.pos):
                return "back"
            if not viewport.collidepoint(event.pos):
                return None

            for idx, (path, _) in enumerate(list(self._rows)):
                row = self._row_rect(idx, viewport)
                if not row.collidepoint(event.pos):
                    continue
                buttons = self._row_buttons(row)
                if buttons["delete"].collidepoint(event.pos):
                    delete_record(path)
                    self.refresh()
                    return None
                self._selected_path = path
                if buttons["open"].collidepoint(event.pos) or self._is_double_click(idx):
                    return "open"
                return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "back"
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self._selected_path:
                return "open"
            if event.key == pygame.K_F5:
                self.refresh()
        return None


