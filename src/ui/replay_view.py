"""Replay viewer — step, scrub or auto-play through a recorded game."""

from __future__ import annotations

from typing import Optional

import pygame

from src.config import EMPTY, MODE_FIRST_TO_FOUR, MODE_POINTS_FULL, PLAYER_1, PLAYER_2
from src.engine.board import setup_board, Board
from src.engine.rules import apply_placement
from src.engine.scoring import compute_scores
from src.game.analysis import algebraic, deciding_line
from src.ui.board_view import BoardView
from src.ui.theme import (
    BG_DEEP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P2_ACCENT, GLASS_EDGE,
    font, glyph, draw_button, draw_segmented, truncate,
)

# (label, milliseconds between moves)
SPEEDS: list[tuple[str, int]] = [("0.5×", 1200), ("1×", 620), ("2×", 310), ("4×", 155)]
DEFAULT_SPEED = 1

CONTROL_H = 116
SCRUB_H = 10
BTN_W, BTN_H = 54, 32


class ReplayViewer:
    """Replay a list of (player, row, col) moves with full transport controls."""

    def __init__(
        self,
        board_size: int,
        moves: list[tuple[int, int, int]],
        cell_px: int = 56,
        margin: int = 24,
        mode: str = MODE_POINTS_FULL,
        title: str = "Replay",
        subtitle: str = "",
    ):
        self._board_size = board_size
        self._moves = list(moves)
        self._mode = mode
        self._title = title
        self._subtitle = subtitle

        self._step = 0
        self._bv = BoardView(cell_px=cell_px, margin=margin)
        self._boards: list[Board] = self._build_states()

        self._playing = False
        self._speed_idx = DEFAULT_SPEED
        self._last_advance_ms = 0
        self._hovered: str | None = None
        self._scrubbing = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _build_states(self) -> list[Board]:
        """Replay the move list into a board per step.

        Saved records are user-editable JSON, so the moves are validated here:
        a bad coordinate truncates the replay rather than raising out of the
        constructor and taking the window down with it.
        """
        n = self._board_size
        states = [setup_board(n)]
        board = states[0]
        playable: list[tuple[int, int, int]] = []

        for player, r, c in self._moves:
            if not (0 <= r < n and 0 <= c < n) or board.grid[r, c] != EMPTY:
                break   # corrupt or contradictory record — stop where it still made sense
            board = apply_placement(board, r, c)
            states.append(board)
            playable.append((player, r, c))

        self._moves = playable
        return states

    @property
    def current_board(self) -> Board:
        return self._boards[self._step]

    @property
    def total(self) -> int:
        return len(self._moves)

    def _at_end(self) -> bool:
        return self._step >= self.total

    def _seek(self, step: int) -> None:
        self._step = max(0, min(self.total, int(step)))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _layout(self, surface: pygame.Surface) -> dict[str, pygame.Rect]:
        w, h = surface.get_size()
        board_area = pygame.Rect(20, 56, w - 40, h - 56 - CONTROL_H)
        board_rect = self._bv.fit(board_area, self._board_size)

        bar_y = h - CONTROL_H + 14
        scrub = pygame.Rect(40, bar_y, w - 80, SCRUB_H)

        keys = ["first", "prev", "play", "next", "last"]
        gap = 8
        total_w = len(keys) * BTN_W + (len(keys) - 1) * gap
        x = (w - total_w) // 2
        y = bar_y + SCRUB_H + 14
        rects = {k: pygame.Rect(x + i * (BTN_W + gap), y, BTN_W, BTN_H)
                 for i, k in enumerate(keys)}

        rects["board"] = board_rect
        rects["scrub"] = scrub
        rects["speed"] = pygame.Rect(w - 40 - 180, y, 180, BTN_H)
        rects["back"] = pygame.Rect(40, y, 96, BTN_H)
        return rects

    # ------------------------------------------------------------------
    # Update (auto-play)
    # ------------------------------------------------------------------

    def update(self) -> None:
        if not self._playing:
            return
        if self._at_end():
            self._playing = False
            return
        now = pygame.time.get_ticks()
        interval = SPEEDS[self._speed_idx][1]
        if now - self._last_advance_ms >= interval:
            self._last_advance_ms = now
            self._seek(self._step + 1)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        self.update()
        surface.fill(BG_DEEP)
        w, h = surface.get_size()
        rects = self._layout(surface)
        board = self.current_board

        # --- Header ---------------------------------------------------
        title = font("h2").render(truncate("h2", self._title, w - 40), True, TEXT_BRIGHT)
        surface.blit(title, (20, 14))
        if self._subtitle:
            sub = font("small").render(truncate("small", self._subtitle, w - 300), True, TEXT_DIM)
            surface.blit(sub, (22, 40))

        # --- Board ----------------------------------------------------
        last_move = None
        if self._step > 0:
            _, lr, lc = self._moves[self._step - 1]
            last_move = (lr, lc)

        win_line = []
        if self._at_end() and self.total:
            win_line = deciding_line(board, self._mode, _infer_winner(board, self._mode))

        self._bv.draw(
            surface, board,
            last_move=last_move,
            show_legal=False,
            show_coords=True,
            allow_ghost=False,
            win_line=win_line,
            win_progress=1.0,
            reduce_motion=True,
        )

        # --- Live score -----------------------------------------------
        p1, p2 = compute_scores(board)
        self._draw_scoreline(surface, rects["board"], p1, p2)

        # --- Transport ------------------------------------------------
        self._draw_scrub(surface, rects["scrub"])
        self._draw_buttons(surface, rects)
        self._draw_move_caption(surface, w, rects["scrub"].y - 22)

    def _draw_scoreline(
        self, surface: pygame.Surface, board_rect: pygame.Rect, p1: float, p2: float
    ) -> None:
        y = max(30, board_rect.y - 24)
        left = font("mono").render(f"P1  {p1:.2f}", True, P1_ACCENT)
        right = font("mono").render(f"{p2:.2f}  P2", True, P2_ACCENT)
        surface.blit(left, (board_rect.x, y))
        surface.blit(right, (board_rect.right - right.get_width(), y))

    def _draw_move_caption(self, surface: pygame.Surface, w: int, y: int) -> None:
        if self._step > 0:
            p, r, c = self._moves[self._step - 1]
            col = P1_ACCENT if p == PLAYER_1 else P2_ACCENT
            text = f"move {self._step} / {self.total}   ·   P{p} → {algebraic(r, c)}"
        else:
            col = TEXT_MUTED
            text = f"start position   ·   {self.total} moves recorded"
        s = font("small").render(text, True, col)
        surface.blit(s, ((w - s.get_width()) // 2, y))

    def _draw_scrub(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        track = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        track.fill((60, 80, 120, 130))
        surface.blit(track, rect.topleft)

        frac = (self._step / self.total) if self.total else 0.0
        span = int(rect.width * frac)
        if span > 0:
            fill = pygame.Surface((span, rect.height), pygame.SRCALPHA)
            fill.fill((*P1_ACCENT, 190))
            surface.blit(fill, rect.topleft)

        knob_x = rect.x + span
        pygame.draw.circle(surface, TEXT_BRIGHT, (knob_x, rect.centery), rect.height // 2 + 3)
        pygame.draw.rect(surface, (*GLASS_EDGE, 50), rect, 1, border_radius=4)

    def _draw_buttons(self, surface: pygame.Surface, rects: dict[str, pygame.Rect]) -> None:
        mx, my = pygame.mouse.get_pos()
        back = glyph("◀", "◄")
        fwd = glyph("▶", "►")
        labels = {
            "first": f"|{back}{back}",
            "prev":  back,
            "play":  glyph("❚❚", "II") if self._playing else fwd,
            "next":  fwd,
            "last":  f"{fwd}{fwd}|",
        }
        enabled = {
            "first": self._step > 0,
            "prev":  self._step > 0,
            "play":  self.total > 0,
            "next":  not self._at_end(),
            "last":  not self._at_end(),
        }
        for key, label in labels.items():
            rect = rects[key]
            draw_button(
                surface, rect, label,
                hovered=rect.collidepoint(mx, my),
                base_surf=surface, enabled=enabled[key],
            )

        draw_segmented(
            surface, rects["speed"], [s[0] for s in SPEEDS], self._speed_idx,
            hovered_index=_segment_hit(rects["speed"], len(SPEEDS), (mx, my)),
        )
        draw_button(
            surface, rects["back"], "◄ Back",
            hovered=rects["back"].collidepoint(mx, my), base_surf=surface,
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface | None = None) -> Optional[str]:
        surface = surface or pygame.display.get_surface()
        if surface is None:
            return None
        rects = self._layout(surface)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            scrub = rects["scrub"].inflate(0, 16)
            if scrub.collidepoint(event.pos):
                self._scrubbing = True
                self._playing = False
                self._seek_from_pixel(event.pos[0], rects["scrub"])
                return None
            for key in ("first", "prev", "play", "next", "last"):
                if rects[key].collidepoint(event.pos):
                    self._do(key)
                    return None
            if rects["back"].collidepoint(event.pos):
                return "back"
            seg = _segment_hit(rects["speed"], len(SPEEDS), event.pos)
            if seg >= 0:
                self._speed_idx = seg
                return None

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._scrubbing = False

        if event.type == pygame.MOUSEMOTION and self._scrubbing:
            self._seek_from_pixel(event.pos[0], rects["scrub"])
            return None

        if event.type == pygame.MOUSEWHEEL:
            self._playing = False
            self._seek(self._step + event.y)
            return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self._do("next")
            elif event.key == pygame.K_LEFT:
                self._do("prev")
            elif event.key == pygame.K_HOME:
                self._do("first")
            elif event.key == pygame.K_END:
                self._do("last")
            elif event.key == pygame.K_SPACE:
                self._do("play")
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self._speed_idx = min(len(SPEEDS) - 1, self._speed_idx + 1)
            elif event.key == pygame.K_MINUS:
                self._speed_idx = max(0, self._speed_idx - 1)
            elif event.key == pygame.K_ESCAPE:
                return "back"
        return None

    def _do(self, action: str) -> None:
        if action == "first":
            self._playing = False
            self._seek(0)
        elif action == "prev":
            self._playing = False
            self._seek(self._step - 1)
        elif action == "next":
            self._playing = False
            self._seek(self._step + 1)
        elif action == "last":
            self._playing = False
            self._seek(self.total)
        elif action == "play":
            if self._at_end():
                self._seek(0)
            self._playing = not self._playing
            self._last_advance_ms = pygame.time.get_ticks()

    def _seek_from_pixel(self, px: int, scrub: pygame.Rect) -> None:
        if scrub.width <= 0 or self.total == 0:
            return
        frac = (px - scrub.x) / scrub.width
        self._seek(round(frac * self.total))


# ---------------------------------------------------------------------------

def _segment_hit(rect: pygame.Rect, count: int, pos: tuple[int, int]) -> int:
    if count <= 0 or not rect.collidepoint(pos):
        return -1
    seg_w = rect.width // count
    return min(count - 1, max(0, (pos[0] - rect.x) // seg_w))


def _infer_winner(board: Board, mode: str) -> Optional[int]:
    """Who the final position favours — used only to pick the line to highlight."""
    from src.engine.rules import check_terminal_mode1

    if mode == MODE_FIRST_TO_FOUR:
        done, winner = check_terminal_mode1(board)
        return winner if done else None
    p1, p2 = compute_scores(board)
    if p1 > p2:
        return PLAYER_1
    if p2 > p1:
        return PLAYER_2
    return None
