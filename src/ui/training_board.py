"""PyGame training board window — opens when --human flag is passed to train.py.

Runs in the main thread.  The training loop runs in a background thread and
communicates via queues (see UIHumanAgent for the protocol).

Layout
------
  [ board area  ] [ sidebar 280px ]

Sidebar shows:
  - Training Session header
  - Game mode
  - Live stats (game #, epsilon, WR vs random/heuristic, best WR)
  - YOUR TURN indicator (pulsing) when waiting for a move
  - AI PLAYING indicator when the AI is computing
  - [Switch to Auto]  button
  - [Quit Training]   button
"""

from __future__ import annotations
import threading
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import pygame

from src.config import PLAYER_1, PLAYER_2, EMPTY as CELL_EMPTY
from src.ui.theme import (
    BG_DEEP, BG_MID, BG_TOP,
    GLASS_TINT, GLASS_EDGE,
    P1_ACCENT, P1_GLOW, P2_ACCENT, P2_GLOW,
    TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    HL_LEGAL, HL_LAST_MOVE,
    PLAYER_ACCENT,
    init_fonts, font,
    draw_piece, draw_button, draw_card,
)
from src.ui.board_view import BoardView


SIDEBAR_W = 280
FPS       = 30


class TrainingBoard:
    """PyGame window for human-vs-AI training sessions."""

    def __init__(
        self,
        board_size: int,
        mode: str,
        board_queue: Queue,
        move_queue: Queue,
        stats_queue: "Queue | None" = None,
        switch_signal_path: "Path | None" = None,
        training_thread: "threading.Thread | None" = None,
    ):
        self._n       = board_size
        self._mode    = mode
        self._board_q = board_queue
        self._move_q  = move_queue
        self._stats_q = stats_queue
        self._switch_path   = switch_signal_path
        self._train_thread  = training_thread

        self._view = BoardView(cell_px=54, margin=32)
        board_w, board_h = self._view.board_pixel_size(board_size)
        self._win_w = board_w + SIDEBAR_W
        self._win_h = max(board_h, 560)
        self._board_w = board_w

        # State shown by the UI
        self._grid        = np.zeros((board_size, board_size), dtype=int)
        self._legal_mask  = np.zeros(board_size * board_size, dtype=bool)
        self._cur_player  = PLAYER_1
        self._last_move   = None
        self._waiting_for_move = False   # True when human turn
        self._auto_mode   = False        # True after "Switch to Auto"
        self._hover       = None
        self._pulse       = 0            # frame counter for pulsing indicator

        # Game-over result overlay
        self._game_over_info: dict | None = None   # set after each episode
        self._learning_frames = 0                  # counts down "AI learning" display

        # Running score tally for this training session
        self._tally: dict[str, int] = {"WIN": 0, "LOSS": 0, "DRAW": 0}

        # Stats sidebar data
        self._stats: dict = {}

        # Button rects (computed after pygame.init)
        self._btn_auto  = None
        self._btn_quit  = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        pygame.init()
        init_fonts()

        screen = pygame.display.set_mode((self._win_w, self._win_h))
        pygame.display.set_caption("C_lines — Training Session")
        clock = pygame.time.Clock()

        # Build a fake Board for BoardView (it only needs .grid, .size, .current_player)
        from src.engine.board import setup_board
        self._fake_board = setup_board(self._n)

        # Sidebar button rects
        bx = self._board_w + 16
        self._btn_auto = pygame.Rect(bx, self._win_h - 110, SIDEBAR_W - 32, 38)
        self._btn_quit = pygame.Rect(bx, self._win_h - 60,  SIDEBAR_W - 32, 38)

        running = True
        while running:
            self._pulse = (self._pulse + 1) % 60

            # -- Poll training thread state --
            if self._train_thread and not self._train_thread.is_alive():
                # Training finished — show overlay briefly then exit
                self._draw(screen)
                self._draw_finished_overlay(screen)
                pygame.display.flip()
                pygame.time.wait(3000)
                running = False
                break

            # -- Read from board_queue (non-blocking) --
            try:
                state = self._board_q.get_nowait()
                if state.get("game_over"):
                    # Episode just ended — show result overlay, freeze board
                    self._game_over_info   = state
                    self._grid             = state["grid"]
                    self._waiting_for_move = False
                    self._learning_frames  = FPS * 6   # show "AI learning" for ~6 s
                    self._fake_board.grid[:] = self._grid
                    if "tally" in state:
                        self._tally = state["tally"]
                else:
                    # New human turn starting — dismiss any previous result
                    self._game_over_info   = None
                    self._learning_frames  = 0
                    self._grid             = state["grid"]
                    self._legal_mask       = state["legal_mask"]
                    self._cur_player       = state["current_player"]
                    self._waiting_for_move = True
                    self._fake_board.grid[:] = self._grid
                    self._fake_board.current_player = self._cur_player
            except Empty:
                pass

            # Count down the "AI learning" timer
            if self._learning_frames > 0:
                self._learning_frames -= 1

            # -- Read stats --
            if self._stats_q:
                try:
                    self._stats = self._stats_q.get_nowait()
                except Empty:
                    pass

            # -- Events --
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if mx < self._board_w:
                        cell = self._view.pixel_to_cell(mx, my, self._n)
                        self._hover = cell
                    else:
                        self._hover = None

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    running = self._handle_click(mx, my)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # -- Draw --
            self._draw(screen)
            pygame.display.flip()
            clock.tick(FPS)

        # Signal training to stop if still running
        if self._switch_path:
            self._switch_path.write_text("quit")
        # Unblock any waiting select_action
        try:
            self._move_q.put_nowait(-1)
        except Exception:
            pass

        pygame.quit()

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _handle_click(self, mx: int, my: int) -> bool:
        """Returns False to quit, True to continue."""
        # Board click — only valid when waiting for a human move
        if mx < self._board_w and self._waiting_for_move and not self._auto_mode:
            cell = self._view.pixel_to_cell(mx, my, self._n)
            if cell is not None:
                r, c = cell
                idx = r * self._n + c
                if self._legal_mask[idx]:
                    self._last_move = cell
                    self._waiting_for_move = False
                    self._move_q.put(idx)
            return True

        # Sidebar buttons
        if self._btn_auto and self._btn_auto.collidepoint(mx, my):
            if self._auto_mode:
                # Resume human play
                self._auto_mode = False
                if self._switch_path:
                    self._switch_path.write_text("human")
            else:
                # Switch to auto
                self._auto_mode = True
                if self._switch_path:
                    self._switch_path.write_text("auto")
                # Unblock any waiting select_action with sentinel
                try:
                    self._move_q.put_nowait(-1)
                except Exception:
                    pass
                self._waiting_for_move = False
            return True

        if self._btn_quit and self._btn_quit.collidepoint(mx, my):
            return False

        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw(self, screen: pygame.Surface) -> None:
        screen.fill(BG_DEEP)

        # Board
        board_surf = pygame.Surface((self._board_w, self._win_h))
        self._fake_board.grid[:] = self._grid
        self._fake_board.current_player = self._cur_player

        self._view.draw(
            board_surf,
            self._fake_board,
            last_move=self._last_move,
            hover_cell=self._hover if (self._waiting_for_move and not self._auto_mode) else None,
            show_legal=self._waiting_for_move and not self._auto_mode,
            legal_mask=self._legal_mask,
        )
        screen.blit(board_surf, (0, 0))

        # Game-over overlay (drawn on top of the board)
        if self._game_over_info is not None:
            self._draw_game_over_overlay(screen)

        # Sidebar
        self._draw_sidebar(screen)

    def _draw_sidebar(self, screen: pygame.Surface) -> None:
        sx = self._board_w
        sw = SIDEBAR_W
        sh = self._win_h

        # Background panel
        panel = pygame.Surface((sw, sh), pygame.SRCALPHA)
        panel.fill((*BG_TOP, 200))
        pygame.draw.line(panel, (*GLASS_EDGE, 40), (0, 0), (0, sh))
        screen.blit(panel, (sx, 0))

        x = sx + 16
        y = 18

        def lbl(text, fkey="h2", color=TEXT_BRIGHT, offset=0):
            nonlocal y
            s = font(fkey).render(text, True, color)
            screen.blit(s, (x + offset, y))
            y += s.get_height() + 6

        def sep():
            nonlocal y
            pygame.draw.line(screen, (*GLASS_EDGE, 25), (sx + 8, y), (sx + sw - 8, y))
            y += 10

        # Title
        lbl("TRAINING SESSION", "h2", TEXT_BRIGHT)
        mode_str = "First to Four" if "first" in self._mode else "Points Full"
        lbl(mode_str, "small", TEXT_DIM)
        sep()

        # Turn indicator
        if self._auto_mode:
            lbl("AUTO MODE", "h2", TEXT_MUTED)
            lbl("AI is training itself", "small", TEXT_DIM)
        elif self._game_over_info is not None:
            result = self._game_over_info.get("human_result", "DRAW")
            if result == "WIN":
                lbl("YOU WON!", "h2", (80, 240, 160))
            elif result == "LOSS":
                lbl("AI WON", "h2", (255, 100, 100))
            else:
                lbl("DRAW", "h2", (200, 200, 220))
            if self._learning_frames > 0:
                dot_count = (self._pulse // 10) % 4
                lbl(f"AI learning{'.' * dot_count}", "small", TEXT_DIM)
            else:
                lbl("Next game starting...", "small", TEXT_DIM)
        elif self._waiting_for_move:
            surf = font("h2").render("▶ YOUR TURN", True, PLAYER_ACCENT.get(self._cur_player, P1_ACCENT))
            screen.blit(surf, (x, y))
            y += surf.get_height() + 4
            lbl("Click a cell to place", "small", TEXT_DIM)
        else:
            lbl("AI PLAYING...", "h2", TEXT_MUTED)
        sep()

        # Stats
        lbl("Live stats", "small", TEXT_DIM)
        game_n  = self._stats.get("game", "—")
        eps     = self._stats.get("epsilon", "—")
        wr_r    = self._stats.get("wr_random", None)
        wr_h    = self._stats.get("wr_heuristic", None)
        best_wr = self._stats.get("best_wr", None)

        lbl(f"Game:      {game_n}", "mono", TEXT_MUTED)
        lbl(f"Epsilon:   {eps if isinstance(eps, str) else f'{eps:.3f}'}", "mono", TEXT_MUTED)
        lbl(f"WR random: {wr_r if wr_r is None else f'{wr_r:.0%}'}", "mono", TEXT_MUTED)
        lbl(f"WR heur:   {wr_h if wr_h is None else f'{wr_h:.0%}'}", "mono", TEXT_MUTED)
        lbl(f"Best WR:   {best_wr if best_wr is None else f'{best_wr:.0%}'}", "mono", TEXT_BRIGHT)
        sep()

        # Score tally
        wins   = self._tally.get("WIN",  0)
        losses = self._tally.get("LOSS", 0)
        draws  = self._tally.get("DRAW", 0)
        total  = wins + losses + draws

        lbl("Session score", "small", TEXT_DIM)

        # Tally bar — coloured segments
        bar_x  = x
        bar_y  = y
        bar_w  = SIDEBAR_W - 32
        bar_h  = 14
        y     += bar_h + 4

        bar_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((40, 55, 80, 180))
        pygame.draw.rect(bar_surf, (*GLASS_EDGE, 30), bar_surf.get_rect(), 1, border_radius=4)
        if total > 0:
            w_px = int(bar_w * wins   / total)
            d_px = int(bar_w * draws  / total)
            l_px = bar_w - w_px - d_px
            if w_px > 0:
                pygame.draw.rect(bar_surf, (80, 220, 140, 200),  pygame.Rect(0,      0, w_px, bar_h), border_radius=4)
            if d_px > 0:
                pygame.draw.rect(bar_surf, (160, 165, 190, 180), pygame.Rect(w_px,   0, d_px, bar_h))
            if l_px > 0:
                pygame.draw.rect(bar_surf, (220, 90, 100, 200),  pygame.Rect(w_px + d_px, 0, l_px, bar_h))
        screen.blit(bar_surf, (bar_x, bar_y))

        # W / D / L counts
        w_pct = f"{wins/total:.0%}" if total > 0 else "—"
        tally_line = (
            f"W {wins} ({w_pct})   D {draws}   L {losses}   of {total}"
        )
        lbl(tally_line, "small", TEXT_MUTED)
        sep()

        # Hint
        if self._waiting_for_move and not self._auto_mode:
            lbl("Legal cells are highlighted", "hint", TEXT_DIM)
            lbl("Hover to preview your piece", "hint", TEXT_DIM)
            y += 6

        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        btn_label = "Resume Human Play" if self._auto_mode else "Switch to Auto"
        draw_button(screen, self._btn_auto, btn_label,
                    hovered=self._btn_auto.collidepoint(mouse_pos))
        draw_button(screen, self._btn_quit, "Quit Training",
                    hovered=self._btn_quit.collidepoint(mouse_pos))

    def _draw_game_over_overlay(self, screen: pygame.Surface) -> None:
        info    = self._game_over_info
        result  = info.get("human_result", "DRAW")
        h_rew   = info.get("human_reward", 0.0)
        a_rew   = info.get("agent_reward", 0.0)
        game_n  = info.get("game_idx", "?")

        # Semi-transparent dark veil over the board
        veil = pygame.Surface((self._board_w, self._win_h), pygame.SRCALPHA)
        veil.fill((8, 16, 36, 195))
        screen.blit(veil, (0, 0))

        cx = self._board_w // 2
        cy = self._win_h  // 2

        # Result headline
        if result == "WIN":
            headline  = "YOU WIN!"
            h_col     = (80, 240, 160)
            emoji_str = "+1.00"
        elif result == "LOSS":
            headline  = "AI WINS"
            h_col     = (255, 100, 100)
            emoji_str = "-1.00"
        else:
            headline  = "DRAW"
            h_col     = (200, 200, 220)
            emoji_str = " 0.00"

        hs = font("display").render(headline, True, h_col)
        screen.blit(hs, (cx - hs.get_width() // 2, cy - 80))

        # Reward row
        def _rew_text(val: float) -> str:
            return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"

        rew_line = font("h2").render(
            f"Your reward: {_rew_text(h_rew)}      AI reward: {_rew_text(a_rew)}",
            True, TEXT_MUTED,
        )
        screen.blit(rew_line, (cx - rew_line.get_width() // 2, cy - 24))

        # Learning / next-game status
        if self._learning_frames > 0:
            dot_count = (self._pulse // 10) % 4
            dots = "." * dot_count
            status = font("ui").render(f"AI is updating its strategy{dots}", True, TEXT_DIM)
        else:
            status = font("ui").render("Waiting for next game...", True, TEXT_DIM)
        screen.blit(status, (cx - status.get_width() // 2, cy + 22))

        # Game number
        gn = font("small").render(f"Game #{game_n}", True, TEXT_DIM)
        screen.blit(gn, (cx - gn.get_width() // 2, cy + 52))

    def _draw_finished_overlay(self, screen: pygame.Surface) -> None:
        overlay = pygame.Surface((self._win_w, self._win_h), pygame.SRCALPHA)
        overlay.fill((10, 20, 40, 180))
        screen.blit(overlay, (0, 0))
        msg = font("display").render("Training Complete!", True, TEXT_BRIGHT)
        screen.blit(msg, (self._win_w // 2 - msg.get_width() // 2,
                           self._win_h // 2 - msg.get_height() // 2))
