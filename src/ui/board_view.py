"""Board rendering: grid lines, pieces, highlights."""

from __future__ import annotations
import pygame

from src.engine.board import Board
from src.config import PLAYER_1, PLAYER_2, EMPTY
from src.ui.theme import (
    BG_MID, TEXT_DIM, HL_LAST_MOVE, HL_LEGAL,
    draw_piece, font,
)


class BoardView:
    """Renders an N×N board within a surface."""

    GRID_COLOR = (60, 80, 120)

    def __init__(self, cell_px: int = 56, margin: int = 36):
        self.cell_px = cell_px
        self.margin = margin

    def board_pixel_size(self, n: int) -> tuple[int, int]:
        side = n * self.cell_px + 2 * self.margin
        return side, side

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        x = self.margin + col * self.cell_px + self.cell_px // 2
        y = self.margin + row * self.cell_px + self.cell_px // 2
        return x, y

    def pixel_to_cell(self, px: int, py: int, n: int) -> tuple[int, int] | None:
        col = (px - self.margin) // self.cell_px
        row = (py - self.margin) // self.cell_px
        if 0 <= row < n and 0 <= col < n:
            return row, col
        return None

    def draw(
        self,
        surface: pygame.Surface,
        board: Board,
        last_move: tuple[int, int] | None = None,
        hover_cell: tuple[int, int] | None = None,
        show_legal: bool = True,
        legal_mask: "np.ndarray | None" = None,  # noqa: F821
        show_coords: bool = True,
        allow_ghost: bool = True,
    ) -> None:
        n = board.size
        cp = self.cell_px
        mg = self.margin

        surface.fill(BG_MID)

        # Subtle checkerboard tint
        for r in range(n):
            for c in range(n):
                if (r + c) % 2 == 0:
                    cx, cy = self.cell_center(r, c)
                    s = pygame.Surface((cp - 1, cp - 1), pygame.SRCALPHA)
                    s.fill((255, 255, 255, 7))
                    surface.blit(s, (cx - cp // 2 + 1, cy - cp // 2 + 1))

        # Grid lines
        total = n * cp
        for i in range(n + 1):
            x = mg + i * cp
            y = mg + i * cp
            pygame.draw.line(surface, self.GRID_COLOR, (x, mg), (x, mg + total))
            pygame.draw.line(surface, self.GRID_COLOR, (mg, y), (mg + total, y))

        # Coordinate labels (column letters top, row numbers left)
        if show_coords and mg >= 28:
            coord_f = font("coord")
            for i in range(n):
                col_lbl = chr(ord("A") + i)
                cs = coord_f.render(col_lbl, True, TEXT_DIM)
                lx = mg + i * cp + cp // 2 - cs.get_width() // 2
                surface.blit(cs, (lx, max(2, (mg - cs.get_height()) // 2)))

                row_lbl = str(i + 1)
                rs = coord_f.render(row_lbl, True, TEXT_DIM)
                ry = mg + i * cp + cp // 2 - rs.get_height() // 2
                surface.blit(rs, (mg - rs.get_width() - 4, ry))

        # Legal move highlights
        if show_legal and legal_mask is not None:
            for r in range(n):
                for c in range(n):
                    if legal_mask[r * n + c]:
                        cx, cy = self.cell_center(r, c)
                        s = pygame.Surface((cp - 2, cp - 2), pygame.SRCALPHA)
                        s.fill((*HL_LEGAL, 22))
                        surface.blit(s, (cx - cp // 2 + 1, cy - cp // 2 + 1))

        # Hover: ghost piece (human turn) or plain blue highlight (AI turn)
        if hover_cell is not None:
            r, c = hover_cell
            if board.grid[r, c] == EMPTY:
                cx, cy = self.cell_center(r, c)
                if allow_ghost:
                    draw_piece(surface, (cx, cy), board.current_player, cp, alpha=70)
                else:
                    s = pygame.Surface((cp - 2, cp - 2), pygame.SRCALPHA)
                    s.fill((*HL_LEGAL, 55))
                    surface.blit(s, (cx - cp // 2 + 1, cy - cp // 2 + 1))

        # Last-move: amber glow fill + ring
        if last_move is not None:
            r, c = last_move
            cx, cy = self.cell_center(r, c)
            glow = pygame.Surface((cp, cp), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*HL_LAST_MOVE, 45), (cp // 2, cp // 2), int(cp * 0.42))
            surface.blit(glow, (cx - cp // 2, cy - cp // 2))
            pygame.draw.circle(surface, HL_LAST_MOVE, (cx, cy), int(cp * 0.42), 2)

        # Pieces
        for r in range(n):
            for c in range(n):
                cell = board.grid[r, c]
                if cell in (PLAYER_1, PLAYER_2):
                    cx, cy = self.cell_center(r, c)
                    draw_piece(surface, (cx, cy), int(cell), cp)
