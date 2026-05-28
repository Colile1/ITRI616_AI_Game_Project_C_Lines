"""C_lines PyGame application — main loop and screen state machine."""

from __future__ import annotations
import pygame
import sys

from src.config import (
    DEFAULT_BOARD_SIZE, DEFAULT_MODE,
    MODE_FIRST_TO_FOUR, MODE_POINTS_FULL,
    CELL_PX, BOARD_MARGIN, SIDEBAR_W, PLAYER_1, PLAYER_2,
)
from src.engine.board import Board
from src.engine.rules import check_terminal_mode1, check_terminal_mode2, is_legal_placement
from src.engine.scoring import compute_scores
from src.game.env import GameEnv
from src.game.encoding import index_to_action, action_to_index, build_legal_mask, state_to_tensor
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.ui.theme import (
    BG_DEEP, BG_MID, BG_TOP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P2_ACCENT, WIN_GLOW, DRAW_GLOW,
    PLAYER_ACCENT, font, init_fonts, make_frost_surface, draw_button,
)
from src.ui.board_view import BoardView
from src.ui.menus import MainMenu, BoardSizePicker, ModePicker, SettingsScreen
from src.ui.level_select import LevelSelectScreen
from src.ui.replay_view import ReplayViewer


FPS = 60

SCREEN_MAIN_MENU   = "main_menu"
SCREEN_SIZE_PICK   = "size_pick"
SCREEN_MODE_PICK   = "mode_pick"
SCREEN_LEVEL_SELECT = "level_select"
SCREEN_INGAME      = "ingame"
SCREEN_GAME_OVER   = "game_over"
SCREEN_REPLAY      = "replay"
SCREEN_SETTINGS    = "settings"


def _window_size(n: int) -> tuple[int, int]:
    w = n * CELL_PX + 2 * BOARD_MARGIN + SIDEBAR_W
    h = n * CELL_PX + 2 * BOARD_MARGIN
    return max(w, 820), max(h, 540)


# ---------------------------------------------------------------------------
# Helper: draw a frosted modal popup box
# ---------------------------------------------------------------------------
def _draw_modal(
    surface: pygame.Surface,
    lines: list[tuple[str, tuple[int,int,int], str]],   # (text, colour, font_key)
    buttons: list[tuple[str, pygame.Rect]],
    hovered_btn: int,
    box_w: int = 480,
    box_h: int = 340,
) -> None:
    w, h = surface.get_size()
    # Dim backdrop
    dim = pygame.Surface((w, h), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 160))
    surface.blit(dim, (0, 0))

    bx = (w - box_w) // 2
    by = (h - box_h) // 2

    # Frosted panel
    panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    panel.fill((28, 48, 90, 200))
    pygame.draw.rect(panel, (200, 220, 255, 80), panel.get_rect(), 2, border_radius=12)
    surface.blit(panel, (bx, by))

    # Lines
    y_cursor = by + 30
    for text, colour, fkey in lines:
        s = font(fkey).render(text, True, colour)
        surface.blit(s, (bx + (box_w - s.get_width()) // 2, y_cursor))
        y_cursor += s.get_height() + 10

    # Buttons
    for i, (label, rect) in enumerate(buttons):
        draw_button(surface, rect, label, hovered=(i == hovered_btn), base_surf=surface)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("C_lines")
        self._n = DEFAULT_BOARD_SIZE
        self._mode = DEFAULT_MODE
        self._ai_version_id: str | None = None
        self._vs_ai = False
        self._settings: dict = {
            "show_legal": True,
            "show_threats": False,
            "reduce_motion": False,
            "sound": False,
        }

        w, h = _window_size(self._n)
        self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        init_fonts()

        self._clock = pygame.time.Clock()
        self._state = SCREEN_MAIN_MENU

        self._main_menu    = MainMenu()
        self._size_picker  = BoardSizePicker()
        self._mode_picker  = ModePicker()
        self._settings_scr = SettingsScreen(self._settings)
        self._level_screen: LevelSelectScreen | None = None
        self._board_view   = BoardView(CELL_PX, BOARD_MARGIN)

        # In-game state
        self._env: GameEnv | None = None
        self._ai_agent = None
        self._hover_cell: tuple[int, int] | None = None
        self._last_move: tuple[int, int] | None = None
        self._move_log: list[tuple[int, int, int]] = []
        self._winner: int | None = None
        self._game_done  = False
        self._obs = None

        # Game-over modal button hover index
        self._modal_hovered = -1

        self._replay_viewer: ReplayViewer | None = None
        self._prev_state = SCREEN_MAIN_MENU

    # ------------------------------------------------------------------
    def run(self) -> None:
        while True:
            self._clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                self._dispatch_event(event)
            self._screen.fill(BG_DEEP)
            self._draw()
            pygame.display.flip()

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------
    def _dispatch_event(self, event: pygame.event.Event) -> None:
        s = self._state
        if   s == SCREEN_MAIN_MENU:    self._ev_main_menu(event)
        elif s == SCREEN_SIZE_PICK:    self._ev_size_pick(event)
        elif s == SCREEN_MODE_PICK:    self._ev_mode_pick(event)
        elif s == SCREEN_LEVEL_SELECT: self._ev_level_select(event)
        elif s == SCREEN_INGAME:       self._ev_ingame(event)
        elif s == SCREEN_GAME_OVER:    self._ev_game_over(event)
        elif s == SCREEN_REPLAY:       self._ev_replay(event)
        elif s == SCREEN_SETTINGS:     self._ev_settings(event)

    # ------------------------------------------------------------------
    # Screen event handlers
    # ------------------------------------------------------------------
    def _ev_main_menu(self, event: pygame.event.Event) -> None:
        result = self._main_menu.handle_event(event, self._screen)
        if result == "play_vs_ai":
            self._vs_ai = True;  self._state = SCREEN_SIZE_PICK
        elif result == "hot-seat":
            self._vs_ai = False; self._state = SCREEN_SIZE_PICK
        elif result == "settings":
            self._prev_state = SCREEN_MAIN_MENU; self._state = SCREEN_SETTINGS
        elif result == "quit":
            pygame.quit(); sys.exit()

    def _back_button_clicked(self, event: pygame.event.Event) -> bool:
        """Return True if the visible Back button was clicked."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            w, h = self._screen.get_size()
            rect = pygame.Rect(20, h - 56, 110, 36)
            if rect.collidepoint(event.pos):
                return True
        return False

    def _ev_size_pick(self, event: pygame.event.Event) -> None:
        if self._back_button_clicked(event):
            self._state = SCREEN_MAIN_MENU; return
        result = self._size_picker.handle_event(event, self._screen)
        if result == "confirm": self._state = SCREEN_MODE_PICK
        elif result == "back":  self._state = SCREEN_MAIN_MENU

    def _ev_mode_pick(self, event: pygame.event.Event) -> None:
        if self._back_button_clicked(event):
            self._state = SCREEN_SIZE_PICK; return
        result = self._mode_picker.handle_event(event, self._screen)
        if result == "confirm":
            self._mode = self._mode_picker.selected
            if self._vs_ai:
                self._level_screen = LevelSelectScreen(self._size_picker.selected)
                self._state = SCREEN_LEVEL_SELECT
            else:
                self._n = self._size_picker.selected
                self._start_game()
        elif result == "back":
            self._state = SCREEN_SIZE_PICK

    def _ev_level_select(self, event: pygame.event.Event) -> None:
        assert self._level_screen is not None
        result = self._level_screen.handle_event(event, self._screen)
        if result == "play":
            self._ai_version_id = self._level_screen.selected_version_id
            self._n = self._size_picker.selected
            self._start_game()
        elif result == "back":
            self._state = SCREEN_MODE_PICK

    def _ev_ingame(self, event: pygame.event.Event) -> None:
        assert self._env is not None
        board = self._env.board

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._state = SCREEN_MAIN_MENU; return
            if event.key == pygame.K_q:    # resign
                self._game_done = True
                self._winner = PLAYER_2 if board.current_player == PLAYER_1 else PLAYER_1
                self._state = SCREEN_GAME_OVER; return

        if event.type == pygame.MOUSEMOTION:
            pos = event.pos
            self._hover_cell = self._board_view.pixel_to_cell(pos[0], pos[1], self._n)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self._game_done:
            # Check sidebar buttons first
            w, h = self._screen.get_size()
            bw = self._n * CELL_PX + 2 * BOARD_MARGIN
            sx = bw
            resign_rect, settings_rect, menu_rect = self._sidebar_button_rects(sx, h)
            mx, my = event.pos
            if resign_rect.collidepoint(mx, my):
                self._game_done = True
                self._winner = PLAYER_2 if board.current_player == PLAYER_1 else PLAYER_1
                self._state = SCREEN_GAME_OVER; return
            if settings_rect.collidepoint(mx, my):
                self._prev_state = SCREEN_INGAME; self._state = SCREEN_SETTINGS; return
            if menu_rect.collidepoint(mx, my):
                self._state = SCREEN_MAIN_MENU; return

            # Board click — only human turn
            if self._vs_ai and board.current_player == PLAYER_2:
                return
            cell = self._board_view.pixel_to_cell(event.pos[0], event.pos[1], self._n)
            if cell is not None:
                r, c = cell
                if is_legal_placement(board, r, c):
                    self._do_placement(r, c)
                    if self._game_done:
                        self._state = SCREEN_GAME_OVER

    def _ev_game_over(self, event: pygame.event.Event) -> None:
        w, h = self._screen.get_size()
        btns = self._modal_button_rects(w, h)

        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._modal_hovered = -1
            for i, (_, r) in enumerate(btns):
                if r.collidepoint(mx, my):
                    self._modal_hovered = i

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            for i, (action, r) in enumerate(btns):
                if r.collidepoint(mx, my):
                    self._handle_game_over_action(action); return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r: self._handle_game_over_action("replay")
            elif event.key == pygame.K_n: self._handle_game_over_action("new")
            elif event.key in (pygame.K_m, pygame.K_ESCAPE): self._handle_game_over_action("menu")

    def _handle_game_over_action(self, action: str) -> None:
        if action == "replay":
            self._replay_viewer = ReplayViewer(self._n, self._move_log, CELL_PX, BOARD_MARGIN)
            self._state = SCREEN_REPLAY
        elif action == "new":
            self._start_game()
        elif action == "menu":
            self._state = SCREEN_MAIN_MENU

    def _ev_replay(self, event: pygame.event.Event) -> None:
        assert self._replay_viewer is not None
        result = self._replay_viewer.handle_event(event)
        if result == "back":
            self._state = SCREEN_GAME_OVER

    def _ev_settings(self, event: pygame.event.Event) -> None:
        result = self._settings_scr.handle_event(event, self._screen)
        if result == "back":
            self._state = self._prev_state

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------
    def _start_game(self) -> None:
        w, h = _window_size(self._n)
        self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self._env = GameEnv(board_size=self._n, mode=self._mode)
        self._obs = self._env.reset()
        self._move_log = []
        self._last_move = None
        self._game_done = False
        self._winner = None
        self._modal_hovered = -1

        if self._vs_ai and self._ai_version_id:
            try:
                from src.training.snapshot import load_snapshot
                self._ai_agent = load_snapshot(self._n, self._ai_version_id)
            except Exception:
                self._ai_agent = RandomAgent()
        elif self._vs_ai:
            self._ai_agent = RandomAgent()
        else:
            self._ai_agent = None

        self._state = SCREEN_INGAME

    def _do_placement(self, r: int, c: int) -> None:
        assert self._env is not None
        player = self._env.board.current_player
        action = action_to_index(r, c, self._n)
        self._obs, _, done, info = self._env.step(action)
        self._last_move = (r, c)
        self._move_log.append((player, r, c))
        if done:
            self._game_done = True
            self._winner = info.get("winner")

    def _ai_move(self) -> None:
        assert self._env is not None and self._ai_agent is not None
        board = self._env.board
        mask = self._env.legal_mask()
        if not mask.any():
            return
        if isinstance(self._ai_agent, HeuristicAgent):
            self._ai_agent.set_board(board)
        action = self._ai_agent.select_action(self._obs, mask)
        r, c = index_to_action(action, self._n)
        self._do_placement(r, c)
        if self._game_done:
            self._state = SCREEN_GAME_OVER

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _sidebar_button_rects(self, sx: int, h: int) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        bw, bh = 130, 36
        resign   = pygame.Rect(sx + 20, h - 140, bw, bh)
        settings = pygame.Rect(sx + 20, h - 96,  bw, bh)
        menu     = pygame.Rect(sx + 20, h - 52,  bw, bh)
        return resign, settings, menu

    def _modal_button_rects(self, w: int, h: int) -> list[tuple[str, pygame.Rect]]:
        box_w, box_h = 480, 340
        bx = (w - box_w) // 2
        by = (h - box_h) // 2
        btn_w, btn_h = 130, 40
        gap = 16
        total = 3 * btn_w + 2 * gap
        start_x = bx + (box_w - total) // 2
        btn_y = by + box_h - 70
        return [
            ("replay", pygame.Rect(start_x,                  btn_y, btn_w, btn_h)),
            ("new",    pygame.Rect(start_x + btn_w + gap,    btn_y, btn_w, btn_h)),
            ("menu",   pygame.Rect(start_x + 2*(btn_w+gap),  btn_y, btn_w, btn_h)),
        ]

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    def _draw(self) -> None:
        s = self._state
        if   s == SCREEN_MAIN_MENU:    self._main_menu.draw(self._screen)
        elif s == SCREEN_SIZE_PICK:    self._draw_size_pick()
        elif s == SCREEN_MODE_PICK:    self._draw_mode_pick()
        elif s == SCREEN_LEVEL_SELECT and self._level_screen:
                                       self._level_screen.draw(self._screen)
        elif s == SCREEN_INGAME:       self._draw_ingame()
        elif s == SCREEN_GAME_OVER:    self._draw_ingame(); self._draw_game_over_modal()
        elif s == SCREEN_REPLAY and self._replay_viewer:
                                       self._screen.fill(BG_DEEP); self._replay_viewer.draw(self._screen)
        elif s == SCREEN_SETTINGS:     self._settings_scr.draw(self._screen)

    # ------------------------------------------------------------------
    def _draw_size_pick(self) -> None:
        self._size_picker.draw(self._screen)
        self._draw_back_button()

    def _draw_mode_pick(self) -> None:
        self._mode_picker.draw(self._screen)
        self._draw_back_button()

    def _draw_back_button(self) -> None:
        w, h = self._screen.get_size()
        rect = pygame.Rect(20, h - 56, 110, 36)
        mx, my = pygame.mouse.get_pos()
        hov = rect.collidepoint(mx, my)
        draw_button(self._screen, rect, "◄  Back", hovered=hov, base_surf=self._screen)

    # ------------------------------------------------------------------
    def _draw_ingame(self) -> None:
        assert self._env is not None
        board = self._env.board
        n = board.size
        w, h = self._screen.get_size()
        bw, bh = self._board_view.board_pixel_size(n)
        sx = bw

        # Board
        board_surf = pygame.Surface((bw, bh))
        mask = self._env.legal_mask() if self._settings.get("show_legal") else None
        self._board_view.draw(
            board_surf, board,
            last_move=self._last_move,
            hover_cell=self._hover_cell,
            show_legal=bool(self._settings.get("show_legal")),
            legal_mask=mask,
        )
        self._screen.blit(board_surf, (0, 0))

        # Sidebar background
        self._screen.fill(BG_MID, pygame.Rect(sx, 0, w - sx, h))
        pygame.draw.line(self._screen, (60, 80, 120), (sx, 0), (sx, h))

        # Title
        title = font("display").render("C_lines", True, TEXT_BRIGHT)
        self._screen.blit(title, (sx + 20, 18))

        # Mode / size info
        mode_str = "First to Four" if self._mode == MODE_FIRST_TO_FOUR else "Points Until Full"
        for li, line in enumerate([
            f"Mode: {mode_str}",
            f"Board: {n}×{n}",
            f"Turn: {board.turn}",
        ]):
            surf = font("ui").render(line, True, TEXT_MUTED)
            self._screen.blit(surf, (sx + 20, 72 + li * 24))

        # Score panel
        divider_y = 164
        pygame.draw.line(self._screen, (60, 80, 120), (sx + 12, divider_y), (w - 12, divider_y))
        p1s, p2s = compute_scores(board)
        for pi, (label, score, col) in enumerate([
            ("Player 1", p1s, P1_ACCENT),
            ("Player 2", p2s, P2_ACCENT),
        ]):
            py = 174 + pi * 46
            self._screen.blit(font("ui").render(label, True, col), (sx + 20, py))
            self._screen.blit(font("mono").render(f"{score:.2f} pts", True, col), (sx + 20, py + 18))

        # Current player
        pygame.draw.line(self._screen, (60, 80, 120), (sx + 12, 274), (w - 12, 274))
        cp = board.current_player
        cp_col = P1_ACCENT if cp == PLAYER_1 else P2_ACCENT
        actor = "AI" if (self._vs_ai and cp == PLAYER_2) else f"Player {cp}"
        self._screen.blit(
            font("ui").render(f"► {actor} to move", True, cp_col),
            (sx + 20, 284),
        )

        # Sidebar action buttons
        resign_r, settings_r, menu_r = self._sidebar_button_rects(sx, h)
        mx, my = pygame.mouse.get_pos()
        draw_button(self._screen, resign_r,   "Resign",   hovered=resign_r.collidepoint(mx, my),   base_surf=self._screen)
        draw_button(self._screen, settings_r, "Settings", hovered=settings_r.collidepoint(mx, my), base_surf=self._screen)
        draw_button(self._screen, menu_r,     "Main Menu",hovered=menu_r.collidepoint(mx, my),     base_surf=self._screen)

        # Key hints
        hints_y = h - 180
        pygame.draw.line(self._screen, (60, 80, 120), (sx + 12, hints_y - 6), (w - 12, hints_y - 6))
        for li, hint in enumerate(["Click board = place piece", "Q = Resign   ESC = Menu"]):
            hs = font("small").render(hint, True, TEXT_DIM)
            self._screen.blit(hs, (sx + 20, hints_y + li * 16))

        # AI move trigger
        if (
            not self._game_done
            and self._vs_ai
            and board.current_player == PLAYER_2
            and self._ai_agent is not None
        ):
            self._ai_move()

    # ------------------------------------------------------------------
    def _draw_game_over_modal(self) -> None:
        assert self._env is not None
        w, h = self._screen.get_size()
        p1s, p2s = compute_scores(self._env.board)

        if self._winner is None:
            result_text = "Draw!"
            col = DRAW_GLOW
        elif self._winner == PLAYER_1:
            result_text = "Player 1 Wins!"
            col = P1_ACCENT
        else:
            result_text = "Player 2 Wins!"
            col = P2_ACCENT

        btns = self._modal_button_rects(w, h)
        btn_labels = {"replay": "Replay (R)", "new": "New Game (N)", "menu": "Menu (M)"}
        named_btns = [(btn_labels[a], r) for a, r in btns]

        lines: list[tuple[str, tuple, str]] = [
            ("Game Over", TEXT_MUTED, "ui"),
            (result_text, col, "display"),
            ("", TEXT_DIM, "small"),
            (f"Player 1:  {p1s:.2f} pts", P1_ACCENT, "mono"),
            (f"Player 2:  {p2s:.2f} pts", P2_ACCENT, "mono"),
        ]
        _draw_modal(self._screen, lines, named_btns, self._modal_hovered)


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
