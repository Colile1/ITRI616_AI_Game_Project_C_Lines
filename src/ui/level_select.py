"""Level selection screen — shows registered snapshots as difficulty cards."""

from __future__ import annotations
import time
import pygame

from src.ui.theme import (
    BG_DEEP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    BAND_COLORS as _BAND_COLORS, font, glyph, draw_button, draw_segmented, truncate,
)
from src.versioning.registry import list_by_size
from src.versioning.metadata import SnapshotMetadata

CARD_W, CARD_H   = 220, 130
CARD_GAP         = 20
CARDS_PER_ROW    = 4
HEADER_H         = 152   # title + hint + the band filter row
SCROLLBAR_W      = 10
DOUBLE_CLICK_SEC = 0.40  # max gap between two clicks to register as double-click

# Filter chips, weakest first.  "all" is always index 0.
BANDS = ["all", "novice", "easy", "medium", "hard", "master"]
FILTER_W, FILTER_H = 540, 28


def _strength_key(snap: SnapshotMetadata) -> tuple:
    """Order snapshots weakest → strongest.

    Band first, so the grid reads in the same order as the filter chips.  Within
    a band, measured win rate then Elo then training length.  Elo cannot lead:
    it is absent on some snapshots, which would sort them to the wrong end.
    """
    try:
        band_rank = BANDS.index(snap.difficulty_band)
    except ValueError:
        band_rank = 0
    return (
        band_rank,
        snap.win_rate_vs_random if snap.win_rate_vs_random is not None else -1.0,
        snap.elo_rating if snap.elo_rating is not None else -1.0,
        snap.games_trained,
        snap.run_id,
        snap.version_id,
    )


class LevelSelectScreen:
    def __init__(self, board_size: int):
        self._board_size  = board_size
        self._all: list[SnapshotMetadata] = []
        self._snapshots: list[SnapshotMetadata] = []   # after filtering
        self._band_idx    = 0
        self._hovered     = -1
        self._selected: str | None = None
        self._selected_idx = -1        # index into _snapshots (ids repeat across runs)
        self._scroll_y    = 0          # pixels scrolled down
        self._drag_scroll = False      # True while dragging the scrollbar thumb
        self._drag_offset = 0          # y-offset within thumb at drag start
        self._last_click_time  = 0.0
        self._last_click_idx   = -1
        self.refresh()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        try:
            self._all = sorted(list_by_size(self._board_size), key=_strength_key)
        except Exception:
            self._all = []
        self._apply_filter()

    def _apply_filter(self) -> None:
        band = BANDS[self._band_idx]
        self._snapshots = (
            list(self._all) if band == "all"
            else [s for s in self._all if s.difficulty_band == band]
        )
        self._scroll_y = 0
        self._hovered = -1
        self._selected_idx = -1
        self._selected = None

    def band_counts(self) -> dict[str, int]:
        counts = {"all": len(self._all)}
        for snap in self._all:
            counts[snap.difficulty_band] = counts.get(snap.difficulty_band, 0) + 1
        return counts

    @property
    def selected_version_id(self) -> str | None:
        return self._selected

    @property
    def selected_meta(self) -> SnapshotMetadata | None:
        """Full metadata for the chosen snapshot — the app labels the AI with it.

        Resolved by index, not by version_id: the same gen_NNN exists in every
        run, so an id alone is ambiguous once more than one run is registered.
        """
        if 0 <= self._selected_idx < len(self._snapshots):
            return self._snapshots[self._selected_idx]
        return None

    @property
    def snapshots(self) -> list[SnapshotMetadata]:
        return list(self._snapshots)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _grid_height(self) -> int:
        n = len(self._snapshots)
        rows = max(1, (n + CARDS_PER_ROW - 1) // CARDS_PER_ROW)
        return rows * (CARD_H + CARD_GAP) - CARD_GAP

    def _max_scroll(self, viewport_h: int) -> int:
        return max(0, self._grid_height() - viewport_h)

    def _filter_rect(self, w: int) -> pygame.Rect:
        width = min(FILTER_W, w - 80)
        return pygame.Rect((w - width) // 2, HEADER_H - FILTER_H - 12, width, FILTER_H)

    def _card_rect(self, idx: int, w: int) -> pygame.Rect:
        n = len(self._snapshots)
        total_w = min(n, CARDS_PER_ROW) * (CARD_W + CARD_GAP) - CARD_GAP
        start_x = (w - total_w) // 2
        row, col = divmod(idx, CARDS_PER_ROW)
        x = start_x + col * (CARD_W + CARD_GAP)
        y = HEADER_H + row * (CARD_H + CARD_GAP) - self._scroll_y
        return pygame.Rect(x, y, CARD_W, CARD_H)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()

        # Title
        title = font("display").render("Select Difficulty", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, 20))
        hint = font("small").render(
            f"Board {self._board_size}×{self._board_size}  ·  weakest first  ·  "
            f"double-click or Play to start  ·  ESC = back",
            True, TEXT_MUTED,
        )
        surface.blit(hint, ((w - hint.get_width()) // 2, 64))

        if not self._all:
            msg = font("ui").render(
                "No snapshots found. Run training first.", True, TEXT_DIM
            )
            surface.blit(msg, ((w - msg.get_width()) // 2, h // 2))
            return

        self._draw_filter(surface, w)

        if not self._snapshots:
            msg = font("ui").render(
                f"No {BANDS[self._band_idx]} agents trained yet.", True, TEXT_DIM
            )
            surface.blit(msg, ((w - msg.get_width()) // 2, h // 2))
            return

        # Clip card drawing to the area below the header
        clip_rect = pygame.Rect(0, HEADER_H, w - SCROLLBAR_W - 4, h - HEADER_H)
        old_clip  = surface.get_clip()
        surface.set_clip(clip_rect)

        for idx, snap in enumerate(self._snapshots):
            rect = self._card_rect(idx, w)
            if rect.bottom < HEADER_H or rect.top > h:
                continue  # off-screen — skip

            band_col = _BAND_COLORS.get(snap.difficulty_band, TEXT_MUTED)
            hov = self._hovered == idx
            card_surf = pygame.Surface((CARD_W, CARD_H), pygame.SRCALPHA)
            card_surf.fill((*band_col, 40 if hov else 20))
            pygame.draw.rect(card_surf, (*band_col, 120 if hov else 60),
                             card_surf.get_rect(), 1, border_radius=6)
            surface.blit(card_surf, rect.topleft)

            x, y = rect.x, rect.y
            surface.blit(font("ui").render(snap.friendly_name, True, TEXT_BRIGHT),  (x + 8, y + 8))
            surface.blit(font("small").render(snap.difficulty_band.upper(), True, band_col), (x + 8, y + 28))

            # Provenance: without it, snapshots from different runs are
            # indistinguishable — the same gen_NNN exists in every run.
            origin = font("hint").render(
                truncate("hint", f"{_short_run(snap.run_id)}/{snap.version_id}", 96),
                True, TEXT_DIM,
            )
            surface.blit(origin, (x + CARD_W - origin.get_width() - 8, y + 10))

            wr    = f"{snap.win_rate_vs_random:.0%}" if snap.win_rate_vs_random is not None else "?"
            games = f"{snap.games_trained:,} games"
            elo   = f"Elo {snap.elo_rating:.0f}" if getattr(snap, "elo_rating", None) else ""
            for li, line in enumerate([games, f"WR vs random: {wr}", elo]):
                if not line:
                    continue
                surface.blit(font("small").render(line, True, TEXT_MUTED), (x + 8, y + 50 + li * 16))

            btn_rect = pygame.Rect(x + CARD_W - 60, y + CARD_H - 28, 52, 22)
            draw_button(surface, btn_rect, "Play", hovered=hov, base_surf=surface)

        surface.set_clip(old_clip)

        # Scrollbar
        viewport_h = h - HEADER_H
        max_sc = self._max_scroll(viewport_h)
        if max_sc > 0:
            total_content = self._grid_height()
            bar_h = max(30, int(viewport_h * viewport_h / total_content))
            bar_y = HEADER_H + int((viewport_h - bar_h) * self._scroll_y / max_sc)
            bar_x = w - SCROLLBAR_W - 2
            # Track
            pygame.draw.rect(surface, (40, 40, 50),
                             pygame.Rect(bar_x, HEADER_H, SCROLLBAR_W, viewport_h), border_radius=4)
            # Thumb
            thumb_col = (120, 120, 140) if self._drag_scroll else (80, 80, 100)
            pygame.draw.rect(surface, thumb_col,
                             pygame.Rect(bar_x, bar_y, SCROLLBAR_W, bar_h), border_radius=4)

        # Scroll hint at bottom if there's more content below
        if self._max_scroll(h - HEADER_H) > 0 and self._scroll_y < self._max_scroll(h - HEADER_H):
            more = font("small").render(
                f"{glyph('▼', 'v')} scroll for more", True, TEXT_DIM
            )
            surface.blit(more, ((w - more.get_width()) // 2, h - 22))

    def _draw_filter(self, surface: pygame.Surface, w: int) -> None:
        counts = self.band_counts()
        labels = [
            f"{b.title()} ({counts.get(b, 0)})" if b != "all" else f"All ({counts['all']})"
            for b in BANDS
        ]
        rect = self._filter_rect(w)
        mx, my = pygame.mouse.get_pos()
        hovered = -1
        if rect.collidepoint(mx, my):
            seg_w = rect.width // len(BANDS)
            hovered = min(len(BANDS) - 1, max(0, (mx - rect.x) // seg_w))
        draw_segmented(surface, rect, labels, self._band_idx, hovered_index=hovered)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> str | None:
        w, h = surface.get_size()
        viewport_h = h - HEADER_H
        max_sc = self._max_scroll(viewport_h)

        # ---- Mouse wheel scroll ----
        if event.type == pygame.MOUSEWHEEL:
            self._scroll_y = max(0, min(max_sc, self._scroll_y - event.y * 30))
            return None

        # ---- Band filter ----
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self._all:
            rect = self._filter_rect(w)
            if rect.collidepoint(event.pos):
                seg_w = rect.width // len(BANDS)
                idx = min(len(BANDS) - 1, max(0, (event.pos[0] - rect.x) // seg_w))
                if idx != self._band_idx:
                    self._band_idx = int(idx)
                    self._apply_filter()
                return None

        # ---- Scrollbar drag ----
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and max_sc > 0:
            bar_x = w - SCROLLBAR_W - 2
            if event.pos[0] >= bar_x:
                total_content = self._grid_height()
                bar_h = max(30, int(viewport_h * viewport_h / total_content))
                bar_y = HEADER_H + int((viewport_h - bar_h) * self._scroll_y / max_sc)
                thumb = pygame.Rect(bar_x, bar_y, SCROLLBAR_W, bar_h)
                if thumb.collidepoint(event.pos):
                    self._drag_scroll = True
                    self._drag_offset = event.pos[1] - bar_y
                    return None

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag_scroll = False

        if event.type == pygame.MOUSEMOTION and self._drag_scroll and max_sc > 0:
            total_content = self._grid_height()
            bar_h = max(30, int(viewport_h * viewport_h / total_content))
            rel_y = event.pos[1] - HEADER_H - self._drag_offset
            ratio = rel_y / max(1, viewport_h - bar_h)
            self._scroll_y = max(0, min(max_sc, int(ratio * max_sc)))
            return None

        n_snaps = len(self._snapshots)

        # ---- Hover update ----
        if event.type == pygame.MOUSEMOTION:
            self._hovered = -1
            for idx in range(n_snaps):
                if self._card_rect(idx, w).collidepoint(event.pos):
                    self._hovered = idx
            return None

        # ---- Click (single & double) ----
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if my < HEADER_H:
                return None
            now = time.monotonic()
            for idx, snap in enumerate(self._snapshots):
                rect = self._card_rect(idx, w)
                if not rect.collidepoint(mx, my):
                    continue

                # Double-click on ANY part of card → play
                if (idx == self._last_click_idx
                        and (now - self._last_click_time) <= DOUBLE_CLICK_SEC):
                    self._selected = snap.version_id
                    self._selected_idx = idx
                    self._last_click_idx = -1
                    return "play"

                self._last_click_time = now
                self._last_click_idx  = idx

                # Single-click on Play button → play
                btn_rect = pygame.Rect(
                    rect.x + CARD_W - 60, rect.y + CARD_H - 28, 52, 22
                )
                if btn_rect.collidepoint(mx, my):
                    self._selected = snap.version_id
                    self._selected_idx = idx
                    return "play"
                return None  # card clicked but not Play — just update hover

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "back"
            # Left/right cycle the band filter — the whole grid stays keyboard-usable.
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                step = 1 if event.key == pygame.K_RIGHT else -1
                self._band_idx = (self._band_idx + step) % len(BANDS)
                self._apply_filter()
        return None


def _short_run(run_id: str) -> str:
    """"run_pts_005" → "pts_005"; keeps the card label readable."""
    return run_id[4:] if run_id.startswith("run_") else run_id
