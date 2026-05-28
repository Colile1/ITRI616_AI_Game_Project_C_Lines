"""Board rendering: grid lines, pieces, highlights."""

from __future__ import annotations
import pygame

from src.engine.board import Board
from src.config import PLAYER_1, PLAYER_2, EMPTY
from src.ui.theme import (
    BG_MID, TEXT_MUTED, HL_LAST_MOVE, HL_LEGAL, PLAYER_ACCENT,
    draw_piece,
)


class BoardView:
    """Renders an N×N board within a surface."""

    GRID_COLOR = (60, 80, 120)
    CELL_HOVER = (160, 220, 255, 40)

    def __init__(self, cell_px: int = 56, margin: int = 24):
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
    ) -> None:
        n = board.size
        cp = self.cell_px
        mg = self.margin

        # Background
        surface.fill(BG_MID)

        # Grid lines
        total = n * cp
        for i in range(n + 1):
            x = mg + i * cp
            y = mg + i * cp
            pygame.draw.line(surface, self.GRID_COLOR, (x, mg), (x, mg + total))
            pygame.draw.line(surface, self.GRID_COLOR, (mg, y), (mg + total, y))

        # Legal highlights
        if show_legal and legal_mask is not None:
            for r in range(n):
                for c in range(n):
                    idx = r * n + c
                    if legal_mask[idx]:
                        cx, cy = self.cell_center(r, c)
                        rect = pygame.Rect(cx - cp // 2 + 1, cy - cp // 2 + 1, cp - 2, cp - 2)
                        s = pygame.Surface((cp - 2, cp - 2), pygame.SRCALPHA)
                        s.fill((*HL_LEGAL, 25))
                        surface.blit(s, (rect.x, rect.y))

        # Hover highlight
        if hover_cell is not None:
            r, c = hover_cell
            if board.grid[r, c] == EMPTY:
                cx, cy = self.cell_center(r, c)
                s = pygame.Surface((cp - 2, cp - 2), pygame.SRCALPHA)
                s.fill((*HL_LEGAL, 60))
                surface.blit(s, (cx - cp // 2 + 1, cy - cp // 2 + 1))

        # Last-move ring
        if last_move is not None:
            r, c = last_move
            cx, cy = self.cell_center(r, c)
            pygame.draw.circle(surface, HL_LAST_MOVE, (cx, cy), int(cp * 0.42), 2)

        # Pieces
        for r in range(n):
            for c in range(n):
                cell = board.grid[r, c]
                if cell in (PLAYER_1, PLAYER_2):
                    cx, cy = self.cell_center(r, c)
                    draw_piece(surface, (cx, cy), int(cell), cp)
