"""Replay viewer — step through a saved move list with arrow keys."""

from __future__ import annotations
import pygame

from src.engine.board import setup_board, Board
from src.engine.rules import apply_placement
from src.ui.board_view import BoardView
from src.ui.theme import BG_DEEP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM, font
from src.game.encoding import action_to_index


class ReplayViewer:
    """Replay a list of (player, row, col) moves."""

    def __init__(
        self,
        board_size: int,
        moves: list[tuple[int, int, int]],  # (player, row, col)
        cell_px: int = 56,
        margin: int = 24,
    ):
        self._board_size = board_size
        self._moves = moves
        self._step = 0
        self._bv = BoardView(cell_px=cell_px, margin=margin)
        self._boards: list[Board] = self._build_states()

    def _build_states(self) -> list[Board]:
        states = [setup_board(self._board_size)]
        board = states[0]
        for _, r, c in self._moves:
            board = apply_placement(board, r, c)
            states.append(board)
        return states

    @property
    def current_board(self) -> Board:
        return self._boards[self._step]

    def draw(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        n = self._board_size
        bw, bh = self._bv.board_pixel_size(n)
        board_surf = pygame.Surface((bw, bh))
        last_move = None
        if self._step > 0:
            _, lr, lc = self._moves[self._step - 1]
            last_move = (lr, lc)
        self._bv.draw(board_surf, self.current_board, last_move=last_move)
        surface.blit(board_surf, ((w - bw) // 2, (h - bh) // 2 - 30))

        info = f"Move {self._step} / {len(self._moves)}  ←→ to navigate  ESC to exit"
        info_surf = font("ui").render(info, True, TEXT_MUTED)
        surface.blit(info_surf, ((w - info_surf.get_width()) // 2, h - 50))

        if self._step > 0:
            p, r, c = self._moves[self._step - 1]
            move_str = f"Player {p} placed at ({r}, {c})"
            ms = font("small").render(move_str, True, TEXT_DIM)
            surface.blit(ms, ((w - ms.get_width()) // 2, h - 28))

    def handle_event(self, event: pygame.event.Event) -> str | None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and self._step < len(self._moves):
                self._step += 1
            elif event.key == pygame.K_LEFT and self._step > 0:
                self._step -= 1
            elif event.key == pygame.K_HOME:
                self._step = 0
            elif event.key == pygame.K_END:
                self._step = len(self._moves)
            elif event.key == pygame.K_ESCAPE:
                return "back"
        return None
