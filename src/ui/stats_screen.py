"""Player statistics screen — lifetime record against the trained snapshots."""

from __future__ import annotations

from typing import Optional

import pygame

from src.config import MODE_FIRST_TO_FOUR, MODE_POINTS_FULL
from src.game.stats import (
    PlayerStats, load_stats, reset_stats,
    BAND_ORDER, RESULT_WIN, RESULT_LOSS, RESULT_DRAW,
)
from src.ui.overlays import ConfirmDialog
from src.ui.theme import (
    BG_DEEP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P2_ACCENT, DRAW_GLOW, BAND_COLORS,
    font, draw_button, draw_card, draw_progress_bar, truncate,
)

TILE_W, TILE_H = 168, 84
TILE_GAP = 14
BAND_ROW_H = 34


class StatsScreen:
    """Headline tiles, a per-difficulty breakdown and a per-mode breakdown."""

    def __init__(self, stats: Optional[PlayerStats] = None):
        self._stats = stats if stats is not None else load_stats()
        self._confirm: Optional[ConfirmDialog] = None

    # ------------------------------------------------------------------
    @property
    def stats(self) -> PlayerStats:
        return self._stats

    def refresh(self) -> None:
        self._stats = load_stats()

    def set_stats(self, stats: PlayerStats) -> None:
        self._stats = stats

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _back_rect(self, surface: pygame.Surface) -> pygame.Rect:
        _, h = surface.get_size()
        return pygame.Rect(40, h - 50, 110, 34)

    def _reset_rect(self, surface: pygame.Surface) -> pygame.Rect:
        w, h = surface.get_size()
        return pygame.Rect(w - 40 - 130, h - 50, 130, 34)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()
        s = self._stats

        title = font("display").render("Your Record", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, 26))

        if s.games_vs_ai == 0 and s.games_hotseat == 0:
            msg = font("ui").render(
                "No games played yet — beat a snapshot and come back.", True, TEXT_DIM
            )
            surface.blit(msg, ((w - msg.get_width()) // 2, h // 2))
            self._draw_footer(surface)
            return

        y = self._draw_tiles(surface, w, 86)
        y = self._draw_bands(surface, w, y + 18)
        self._draw_modes(surface, w, y + 16)
        self._draw_footer(surface)

        if self._confirm is not None:
            self._confirm.draw(surface)

    # ------------------------------------------------------------------
    def _draw_tiles(self, surface: pygame.Surface, w: int, y: int) -> int:
        s = self._stats
        wr = s.win_rate
        best_band = s.best_band_beaten()

        tiles: list[tuple[str, str, tuple[int, int, int]]] = [
            ("Games vs AI", str(s.games_vs_ai), P1_ACCENT),
            ("Win rate", f"{wr:.0%}" if wr is not None else "—", P1_ACCENT),
            ("W / L / D", f"{s.wins}/{s.losses}/{s.draws}", TEXT_BRIGHT),
            ("Streak", f"{s.current_streak}  (best {s.best_streak})", P2_ACCENT),
            ("Fastest win",
             f"{s.fastest_win_moves} moves" if s.fastest_win_moves else "—", DRAW_GLOW),
            ("Hardest beaten",
             best_band.title() if best_band else "—",
             BAND_COLORS.get(best_band or "", TEXT_MUTED)),
        ]

        # Balance the grid rather than filling greedily: 6 tiles in a 5-wide
        # space should read as 3 + 3, not 5 + 1.
        fit = max(1, min(len(tiles), (w - 80 + TILE_GAP) // (TILE_W + TILE_GAP)))
        rows = (len(tiles) + fit - 1) // fit
        per_row = (len(tiles) + rows - 1) // rows

        for i, (label, value, col) in enumerate(tiles):
            row, col_i = divmod(i, per_row)
            count_in_row = min(per_row, len(tiles) - row * per_row)
            total_w = count_in_row * (TILE_W + TILE_GAP) - TILE_GAP
            x = (w - total_w) // 2 + col_i * (TILE_W + TILE_GAP)
            ry = y + row * (TILE_H + TILE_GAP)
            rect = pygame.Rect(x, ry, TILE_W, TILE_H)

            draw_card(surface, rect, stripe_color=col, fill_alpha=28, border_alpha=55)
            surface.blit(
                font("hint").render(label, True, TEXT_DIM), (rect.x + 14, rect.y + 12)
            )
            value_s = font("h2").render(truncate("h2", value, TILE_W - 26), True, col)
            surface.blit(value_s, (rect.x + 14, rect.y + 36))

        return y + rows * (TILE_H + TILE_GAP)

    def _draw_bands(self, surface: pygame.Surface, w: int, y: int) -> int:
        s = self._stats
        panel_w = min(620, w - 80)
        x = (w - panel_w) // 2

        surface.blit(font("h3").render("By difficulty", True, TEXT_BRIGHT), (x, y))
        y += 24

        any_row = False
        for band in BAND_ORDER:
            tally = s.by_band.get(band)
            if not tally:
                continue
            any_row = True
            wins = tally.get(RESULT_WIN, 0)
            losses = tally.get(RESULT_LOSS, 0)
            draws = tally.get(RESULT_DRAW, 0)
            played = wins + losses + draws
            col = BAND_COLORS.get(band, TEXT_MUTED)

            surface.blit(font("small").render(band.title(), True, col), (x, y + 4))
            bar = pygame.Rect(x + 92, y + 7, panel_w - 92 - 130, 12)
            draw_progress_bar(surface, bar, (wins / played) if played else 0.0, col)
            tally_s = font("mono_sm").render(
                f"{wins}W {losses}L {draws}D", True, TEXT_MUTED
            )
            surface.blit(tally_s, (bar.right + 12, y + 4))
            y += BAND_ROW_H

        if not any_row:
            surface.blit(
                font("small").render("no ranked games yet", True, TEXT_DIM), (x, y)
            )
            y += BAND_ROW_H
        return y

    def _draw_modes(self, surface: pygame.Surface, w: int, y: int) -> None:
        s = self._stats
        panel_w = min(620, w - 80)
        x = (w - panel_w) // 2

        surface.blit(font("h3").render("By mode", True, TEXT_BRIGHT), (x, y))
        y += 24

        labels = {
            MODE_FIRST_TO_FOUR: "First to Four",
            MODE_POINTS_FULL: "Points Until Full",
        }
        for mode, label in labels.items():
            tally = s.mode_tally(mode)
            played = sum(tally.values())
            text = (
                f"{label}:  {tally.get(RESULT_WIN, 0)}W  "
                f"{tally.get(RESULT_LOSS, 0)}L  {tally.get(RESULT_DRAW, 0)}D"
                if played else f"{label}:  —"
            )
            surface.blit(font("small").render(text, True, TEXT_MUTED), (x, y))
            y += 22

        extra = (
            f"{s.games_hotseat} hot-seat games   ·   "
            f"{s.moves_played} moves played   ·   "
            f"{s.total_play_sec / 60:.0f} min at the board"
        )
        surface.blit(font("hint").render(extra, True, TEXT_DIM), (x, y + 8))

    def _draw_footer(self, surface: pygame.Surface) -> None:
        mx, my = pygame.mouse.get_pos()
        back = self._back_rect(surface)
        reset = self._reset_rect(surface)
        draw_button(surface, back, "◄  Back", hovered=back.collidepoint(mx, my),
                    base_surf=surface)
        draw_button(surface, reset, "Reset stats", hovered=reset.collidepoint(mx, my),
                    base_surf=surface, accent=P2_ACCENT)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> Optional[str]:
        if self._confirm is not None:
            result = self._confirm.handle_event(event, surface)
            if result == "confirm":
                self._stats = reset_stats()
                self._confirm = None
                return "reset"
            if result == "cancel":
                self._confirm = None
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._back_rect(surface).collidepoint(event.pos):
                return "back"
            if self._reset_rect(surface).collidepoint(event.pos):
                self._confirm = ConfirmDialog(
                    "Reset all statistics?",
                    "Your win/loss record is deleted. Saved games are kept.",
                    confirm_label="Reset", cancel_label="Keep",
                )
                return None

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "back"
        return None
