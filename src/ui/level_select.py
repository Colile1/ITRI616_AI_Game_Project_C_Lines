"""Level selection screen — shows registered snapshots as difficulty cards."""

from __future__ import annotations
import pygame

from src.ui.theme import (
    BG_DEEP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P2_ACCENT, font, draw_button,
)
from src.versioning.registry import list_by_size
from src.versioning.metadata import SnapshotMetadata


_BAND_COLORS = {
    "novice": (100, 200, 140),
    "easy": (100, 180, 255),
    "medium": (200, 180, 80),
    "hard": (220, 100, 80),
    "master": (200, 80, 220),
}

CARD_W, CARD_H = 220, 130
CARD_GAP = 20
CARDS_PER_ROW = 4


class LevelSelectScreen:
    def __init__(self, board_size: int):
        self._board_size = board_size
        self._snapshots: list[SnapshotMetadata] = []
        self._hovered: int = -1
        self._selected: str | None = None
        self.refresh()

    def refresh(self) -> None:
        try:
            self._snapshots = list_by_size(self._board_size)
        except Exception:
            self._snapshots = []

    @property
    def selected_version_id(self) -> str | None:
        return self._selected

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()
        title = font("display").render("Select Difficulty", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, 30))
        hint = font("ui").render(
            f"Board size {self._board_size}×{self._board_size}  |  ESC = back",
            True, TEXT_MUTED,
        )
        surface.blit(hint, ((w - hint.get_width()) // 2, 76))

        if not self._snapshots:
            msg = font("ui").render(
                "No snapshots found. Run training first.", True, TEXT_DIM
            )
            surface.blit(msg, ((w - msg.get_width()) // 2, h // 2))
            return

        n_snaps = len(self._snapshots)
        rows = (n_snaps + CARDS_PER_ROW - 1) // CARDS_PER_ROW
        total_w = min(n_snaps, CARDS_PER_ROW) * (CARD_W + CARD_GAP) - CARD_GAP
        start_x = (w - total_w) // 2
        start_y = 120

        for idx, snap in enumerate(self._snapshots):
            row, col = divmod(idx, CARDS_PER_ROW)
            x = start_x + col * (CARD_W + CARD_GAP)
            y = start_y + row * (CARD_H + CARD_GAP)
            rect = pygame.Rect(x, y, CARD_W, CARD_H)

            # Card background
            band_col = _BAND_COLORS.get(snap.difficulty_band, TEXT_MUTED)
            hov = self._hovered == idx
            card_surf = pygame.Surface((CARD_W, CARD_H), pygame.SRCALPHA)
            card_surf.fill((*band_col, 35 if hov else 20))
            pygame.draw.rect(card_surf, (*band_col, 100 if hov else 60),
                             card_surf.get_rect(), 1, border_radius=6)
            surface.blit(card_surf, (x, y))

            # Name
            name_surf = font("ui").render(snap.friendly_name, True, TEXT_BRIGHT)
            surface.blit(name_surf, (x + 8, y + 8))

            # Band
            band_surf = font("small").render(snap.difficulty_band.upper(), True, band_col)
            surface.blit(band_surf, (x + 8, y + 28))

            # Stats
            wr = f"{snap.win_rate_vs_random:.0%}" if snap.win_rate_vs_random is not None else "?"
            games = f"{snap.games_trained:,} games"
            for li, line in enumerate([games, f"WR vs random: {wr}"]):
                lsurf = font("small").render(line, True, TEXT_MUTED)
                surface.blit(lsurf, (x + 8, y + 50 + li * 16))

            # Play button
            btn_rect = pygame.Rect(x + CARD_W - 60, y + CARD_H - 28, 52, 22)
            draw_button(surface, btn_rect, "Play", hovered=hov, base_surf=surface)

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> str | None:
        w, h = surface.get_size()
        n_snaps = len(self._snapshots)
        if not n_snaps:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "back"
            return None

        total_w = min(n_snaps, CARDS_PER_ROW) * (CARD_W + CARD_GAP) - CARD_GAP
        start_x = (w - total_w) // 2
        start_y = 120

        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._hovered = -1
            for idx in range(n_snaps):
                row, col = divmod(idx, CARDS_PER_ROW)
                x = start_x + col * (CARD_W + CARD_GAP)
                y = start_y + row * (CARD_H + CARD_GAP)
                if pygame.Rect(x, y, CARD_W, CARD_H).collidepoint(mx, my):
                    self._hovered = idx

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            for idx, snap in enumerate(self._snapshots):
                row, col = divmod(idx, CARDS_PER_ROW)
                x = start_x + col * (CARD_W + CARD_GAP)
                y = start_y + row * (CARD_H + CARD_GAP)
                btn_rect = pygame.Rect(x + CARD_W - 60, y + CARD_H - 28, 52, 22)
                if btn_rect.collidepoint(mx, my):
                    self._selected = snap.version_id
                    return "play"

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "back"
        return None
