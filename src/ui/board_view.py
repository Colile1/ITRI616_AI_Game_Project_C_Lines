"""Board rendering: responsive grid, pieces, highlights and overlays.

The view owns its own metrics (cell size, margin, screen origin).  Call `fit()`
once per frame with the rectangle the board is allowed to occupy and everything
— hit testing included — follows the new size, so resizing the window Just Works.
"""

from __future__ import annotations

import pygame

from src.engine.board import Board
from src.config import (
    PLAYER_1, PLAYER_2, EMPTY, CELL_PX, BOARD_MARGIN, MIN_CELL_PX, MAX_CELL_PX,
)
from src.game.analysis import ThreatOverlay
from src.ui.animations import DropAnim, pulse
from src.ui.theme import (
    BG_MID, TEXT_DIM, TEXT_MUTED,
    HL_LAST_MOVE, HL_LEGAL, HL_HINT, HL_CURSOR, HL_WIN_LINE,
    HL_THREAT_OWN, HL_THREAT_OPP,
    draw_piece, font,
)

Cell = tuple[int, int]


class BoardView:
    """Renders an N×N board into a rectangle of a surface."""

    GRID_COLOR = (60, 80, 120)
    EDGE_COLOR = (86, 116, 170)

    def __init__(self, cell_px: int = CELL_PX, margin: int = BOARD_MARGIN):
        self.cell_px = cell_px
        self.margin = margin
        self.origin: tuple[int, int] = (0, 0)
        # Reusable translucent layers, re-allocated only when the board resizes.
        self._layers: list[pygame.Surface] = []
        self._layer_side: int = -1

    def _layers_for(self, side: int, count: int = 3) -> list[pygame.Surface]:
        """Scratch alpha surfaces the size of the board, cleared and ready.

        Cached across frames: a resize reallocates, a redraw only clears.
        """
        if self._layer_side != side or len(self._layers) < count:
            self._layers = [
                pygame.Surface((side, side), pygame.SRCALPHA) for _ in range(count)
            ]
            self._layer_side = side
        for layer in self._layers[:count]:
            layer.fill((0, 0, 0, 0))
        return self._layers[:count]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def fit(self, area: pygame.Rect, n: int) -> pygame.Rect:
        """Size and centre an *n*×*n* board inside *area*; returns the board rect.

        The margin holds the coordinate labels and scales with the cell, so the
        board keeps its proportions from a 900 px window up to a 4K one.
        """
        n = max(1, n)
        # A board occupies n cells plus two half-cell margins → (n + 1) cells.
        raw = min(area.width, area.height) / (n + 1.0)
        self.cell_px = int(max(MIN_CELL_PX, min(MAX_CELL_PX, raw)))
        self.margin = int(max(20, min(48, self.cell_px * 0.5)))

        side = self.board_side(n)
        x = area.x + max(0, (area.width - side) // 2)
        y = area.y + max(0, (area.height - side) // 2)
        self.origin = (x, y)
        return pygame.Rect(x, y, side, side)

    def board_side(self, n: int) -> int:
        return n * self.cell_px + 2 * self.margin

    def board_pixel_size(self, n: int) -> tuple[int, int]:
        side = self.board_side(n)
        return side, side

    def board_rect(self, n: int) -> pygame.Rect:
        side = self.board_side(n)
        return pygame.Rect(self.origin[0], self.origin[1], side, side)

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        ox, oy = self.origin
        x = ox + self.margin + col * self.cell_px + self.cell_px // 2
        y = oy + self.margin + row * self.cell_px + self.cell_px // 2
        return x, y

    def cell_rect(self, row: int, col: int) -> pygame.Rect:
        cx, cy = self.cell_center(row, col)
        half = self.cell_px // 2
        return pygame.Rect(cx - half, cy - half, self.cell_px, self.cell_px)

    def pixel_to_cell(self, px: int, py: int, n: int) -> Cell | None:
        ox, oy = self.origin
        col = (px - ox - self.margin) // self.cell_px
        row = (py - oy - self.margin) // self.cell_px
        if 0 <= row < n and 0 <= col < n:
            return int(row), int(col)
        return None

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        board: Board,
        last_move: Cell | None = None,
        hover_cell: Cell | None = None,
        cursor_cell: Cell | None = None,
        show_legal: bool = True,
        legal_mask=None,
        show_coords: bool = True,
        allow_ghost: bool = True,
        threats: ThreatOverlay | None = None,
        hint_cell: Cell | None = None,
        win_line: list[Cell] | None = None,
        win_progress: float = 1.0,
        drop: DropAnim | None = None,
        reduce_motion: bool = False,
    ) -> None:
        n = board.size
        cp = self.cell_px
        ox, oy = self.origin
        side = self.board_side(n)

        # --- Backdrop ------------------------------------------------
        surface.fill(BG_MID, pygame.Rect(ox, oy, side, side))

        # Translucent washes go onto three cached layers — one blit each, rather
        # than one small blit per cell — and they alpha-blend with each other.
        checker, legal_layer, threat_layer = self._layers_for(side, 3)

        def cell_local(r: int, c: int) -> pygame.Rect:
            return pygame.Rect(
                self.margin + c * cp + 1, self.margin + r * cp + 1, cp - 2, cp - 2
            )

        for r in range(n):
            for c in range(n):
                if (r + c) % 2 == 0:
                    checker.fill((255, 255, 255, 7), cell_local(r, c))
        surface.blit(checker, (ox, oy))

        if show_legal and legal_mask is not None:
            for r in range(n):
                for c in range(n):
                    if legal_mask[r * n + c]:
                        legal_layer.fill((*HL_LEGAL, 22), cell_local(r, c))
            surface.blit(legal_layer, (ox, oy))

        if threats is not None:
            for cells, col, alpha in (
                (threats.opp3, HL_THREAT_OPP, 42),
                (threats.opp4, HL_THREAT_OPP, 78),
                (threats.own3, HL_THREAT_OWN, 42),
                (threats.own4, HL_THREAT_OWN, 78),
            ):
                for r, c in cells:
                    if 0 <= r < n and 0 <= c < n:
                        threat_layer.fill((*col, alpha), cell_local(r, c))
            surface.blit(threat_layer, (ox, oy))

        # --- Grid ----------------------------------------------------
        total = n * cp
        gx, gy = ox + self.margin, oy + self.margin
        for i in range(n + 1):
            x = gx + i * cp
            y = gy + i * cp
            pygame.draw.line(surface, self.GRID_COLOR, (x, gy), (x, gy + total))
            pygame.draw.line(surface, self.GRID_COLOR, (gx, y), (gx + total, y))
        pygame.draw.rect(surface, self.EDGE_COLOR, pygame.Rect(gx, gy, total, total), 1)

        # --- Coordinate labels ---------------------------------------
        if show_coords and self.margin >= 20:
            self._draw_coords(surface, n)

        # --- Critical-cell markers (part of the threat layer) --------
        if threats is not None:
            for r, c in threats.lose_now:
                self._ring(surface, r, c, HL_THREAT_OPP, 0.34, 2)
            for r, c in threats.win_now:
                self._ring(surface, r, c, HL_THREAT_OWN, 0.40, 2)

        # --- Last move -----------------------------------------------
        if last_move is not None:
            r, c = last_move
            cx, cy = self.cell_center(r, c)
            glow_a = 45
            if not reduce_motion:
                glow_a = int(30 + 40 * pulse(1400))
            glow = pygame.Surface((cp, cp), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*HL_LAST_MOVE, glow_a), (cp // 2, cp // 2), int(cp * 0.42))
            surface.blit(glow, (cx - cp // 2, cy - cp // 2))
            pygame.draw.circle(surface, HL_LAST_MOVE, (cx, cy), int(cp * 0.42), 2)

        # --- Hint -----------------------------------------------------
        if hint_cell is not None:
            r, c = hint_cell
            width = 3 if reduce_motion else 2 + int(2 * pulse(900))
            self._ring(surface, r, c, HL_HINT, 0.44, width)

        # --- Hover ghost / plain hover -------------------------------
        if hover_cell is not None:
            r, c = hover_cell
            if 0 <= r < n and 0 <= c < n and board.grid[r, c] == EMPTY:
                cx, cy = self.cell_center(r, c)
                if allow_ghost:
                    draw_piece(surface, (cx, cy), board.current_player, cp, alpha=70)
                else:
                    hov = pygame.Surface((cp - 2, cp - 2), pygame.SRCALPHA)
                    hov.fill((*HL_LEGAL, 55))
                    surface.blit(hov, (cx - cp // 2 + 1, cy - cp // 2 + 1))

        # --- Keyboard cursor -----------------------------------------
        if cursor_cell is not None:
            r, c = cursor_cell
            if 0 <= r < n and 0 <= c < n:
                rect = self.cell_rect(r, c).inflate(-4, -4)
                pygame.draw.rect(surface, HL_CURSOR, rect, 2, border_radius=4)

        # --- Pieces ---------------------------------------------------
        for r in range(n):
            for c in range(n):
                cell = int(board.grid[r, c])
                if cell not in (PLAYER_1, PLAYER_2):
                    continue
                if drop is not None and (r, c) == (drop.row, drop.col) and not drop.done():
                    continue  # drawn by the drop animation below
                cx, cy = self.cell_center(r, c)
                draw_piece(surface, (cx, cy), cell, cp)

        # --- Drop animation ------------------------------------------
        if drop is not None and 0 <= drop.row < n and 0 <= drop.col < n:
            cx, cy = self.cell_center(drop.row, drop.col)
            if drop.done():
                draw_piece(surface, (cx, cy), drop.player, cp)
            else:
                draw_piece(
                    surface, (cx, cy), drop.player, cp,
                    alpha=drop.alpha(), radius_scale=drop.scale(),
                )

        # --- Winning line ---------------------------------------------
        if win_line:
            self._draw_win_line(surface, win_line, win_progress)

    # ------------------------------------------------------------------
    # Pieces of the drawing above
    # ------------------------------------------------------------------

    def _draw_coords(self, surface: pygame.Surface, n: int) -> None:
        ox, oy = self.origin
        cp, mg = self.cell_px, self.margin
        coord_f = font("coord")
        for i in range(n):
            col_lbl = coord_f.render(chr(ord("A") + i), True, TEXT_DIM)
            surface.blit(
                col_lbl,
                (ox + mg + i * cp + cp // 2 - col_lbl.get_width() // 2,
                 oy + max(2, (mg - col_lbl.get_height()) // 2)),
            )
            row_lbl = coord_f.render(str(i + 1), True, TEXT_DIM)
            surface.blit(
                row_lbl,
                (ox + mg - row_lbl.get_width() - 5,
                 oy + mg + i * cp + cp // 2 - row_lbl.get_height() // 2),
            )

    def _ring(
        self, surface: pygame.Surface, row: int, col: int,
        color: tuple[int, int, int], radius_frac: float, width: int,
    ) -> None:
        cx, cy = self.cell_center(row, col)
        pygame.draw.circle(surface, color, (cx, cy), int(self.cell_px * radius_frac), width)

    def _draw_win_line(
        self, surface: pygame.Surface, cells: list[Cell], progress: float,
    ) -> None:
        """Sweep a bright connector along the deciding run."""
        if len(cells) < 2:
            return
        progress = max(0.0, min(1.0, progress))
        start = self.cell_center(*cells[0])
        end = self.cell_center(*cells[-1])
        tip = (
            int(start[0] + (end[0] - start[0]) * progress),
            int(start[1] + (end[1] - start[1]) * progress),
        )

        width = max(3, self.cell_px // 9)
        pygame.draw.line(surface, HL_WIN_LINE, start, tip, width)

        revealed = max(1, int(round(len(cells) * progress)))
        for r, c in cells[:revealed]:
            self._ring(surface, r, c, HL_WIN_LINE, 0.44, 3)

    # ------------------------------------------------------------------
    # Static-position rendering (replay viewer, thumbnails)
    # ------------------------------------------------------------------

    def draw_caption(
        self, surface: pygame.Surface, n: int, text: str,
        color: tuple[int, int, int] = TEXT_MUTED,
    ) -> None:
        """Centre a caption just under the board."""
        rect = self.board_rect(n)
        label = font("small").render(text, True, color)
        surface.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom + 6))
