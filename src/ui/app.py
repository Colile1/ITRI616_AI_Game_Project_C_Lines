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
from src.engine.rules import check_terminal_mode1, check_terminal_mode2
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

# Screen states
SCREEN_MAIN_MENU = "main_menu"
SCREEN_SIZE_PICK = "size_pick"
SCREEN_MODE_PICK = "mode_pick"
SCREEN_LEVEL_SELECT = "level_select"
SCREEN_INGAME = "ingame"
SCREEN_GAME_OVER = "game_over"
SCREEN_REPLAY = "replay"
SCREEN_SETTINGS = "settings"


def _window_size(n: int) -> tuple[int, int]:
    w = n * CELL_PX + 2 * BOARD_MARGIN + SIDEBAR_W
    h = n * CELL_PX + 2 * BOARD_MARGIN
    return max(w, 800), max(h, 520)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("C_lines")
        self._n = DEFAULT_BOARD_SIZE
        self._mode = DEFAULT_MODE
        self._ai_version_id: str | None = None
        self._vs_ai = False
        self._settings = {
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

        # Screen objects
        self._main_menu = MainMenu()
        self._size_picker = BoardSizePicker()
        self._mode_picker = ModePicker()
        self._settings_screen = SettingsScreen(self._settings)
        self._level_screen: LevelSelectScreen | None = None
        self._board_view = BoardView(CELL_PX, BOARD_MARGIN)

        # In-game state
        self._env: GameEnv | None = None
        self._ai_agent = None
        self._hover_cell: tuple[int, int] | None = None
        self._last_move: tuple[int, int] | None = None
        self._move_log: list[tuple[int, int, int]] = []  # (player, row, col)
        self._winner: int | None = None
        self._game_done = False
        self._obs = None

        # Replay
        self._replay_viewer: ReplayViewer | None = None

        # Return-to state for settings/ESC
        self._prev_state = SCREEN_MAIN_MENU

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            dt = self._clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self._dispatch_event(event)

            self._screen.fill(BG_DEEP)
            self._draw()
            pygame.display.flip()

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _dispatch_event(self, event: pygame.event.Event) -> None:
        if self._state == SCREEN_MAIN_MENU:
            result = self._main_menu.handle_event(event, self._screen)
            if result == "play_vs_ai":
                self._vs_ai = True
                self._state = SCREEN_SIZE_PICK
            elif result == "hot-seat":
                self._vs_ai = False
                self._state = SCREEN_SIZE_PICK
            elif result == "settings":
                self._prev_state = SCREEN_MAIN_MENU
                self._state = SCREEN_SETTINGS
            elif result == "quit":
                pygame.quit()
                sys.exit()

        elif self._state == SCREEN_SIZE_PICK:
            result = self._size_picker.handle_event(event, self._screen)
            if result == "confirm":
                self._n = self._size_picker.selected
                self._state = SCREEN_MODE_PICK
            elif result == "back":
                self._state = SCREEN_MAIN_MENU

        elif self._state == SCREEN_MODE_PICK:
            result = self._mode_picker.handle_event(event, self._screen)
            if result == "confirm":
                self._mode = self._mode_picker.selected
                if self._vs_ai:
                    self._level_screen = LevelSelectScreen(self._n)
                    self._state = SCREEN_LEVEL_SELECT
                else:
                    self._start_game()
            elif result == "back":
                self._state = SCREEN_SIZE_PICK

        elif self._state == SCREEN_LEVEL_SELECT:
            assert self._level_screen is not None
            result = self._level_screen.handle_event(event, self._screen)
            if result == "play":
                self._ai_version_id = self._level_screen.selected_version_id
                self._start_game()
            elif result == "back":
                self._state = SCREEN_MODE_PICK

        elif self._state == SCREEN_INGAME:
            self._handle_ingame_event(event)

        elif self._state == SCREEN_GAME_OVER:
            self._handle_game_over_event(event)

        elif self._state == SCREEN_REPLAY:
            assert self._replay_viewer is not None
            result = self._replay_viewer.handle_event(event)
            if result == "back":
                self._state = SCREEN_GAME_OVER

        elif self._state == SCREEN_SETTINGS:
            result = self._settings_screen.handle_event(event, self._screen)
            if result == "back":
                self._state = self._prev_state

    # ------------------------------------------------------------------
    # In-game logic
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

    def _handle_ingame_event(self, event: pygame.event.Event) -> None:
        assert self._env is not None
        board = self._env.board

        # Keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._state = SCREEN_MAIN_MENU
                return
            if event.key == pygame.K_r and self._game_done:
                self._state = SCREEN_REPLAY
                self._replay_viewer = ReplayViewer(self._n, self._move_log, CELL_PX, BOARD_MARGIN)
                return

        # Mouse hover
        if event.type == pygame.MOUSEMOTION:
            pos = event.pos
            cell = self._board_view.pixel_to_cell(pos[0], pos[1], self._n)
            self._hover_cell = cell

        # Human placement click
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self._game_done:
            if self._vs_ai and board.current_player == PLAYER_2:
                return  # AI's turn
            pos = event.pos
            cell = self._board_view.pixel_to_cell(pos[0], pos[1], self._n)
            if cell is not None:
                r, c = cell
                from src.engine.rules import is_legal_placement
                if is_legal_placement(board, r, c):
                    self._do_placement(r, c)

        # AI turn (triggered each frame when it's AI's turn)
        if (
            not self._game_done
            and self._vs_ai
            and not self._game_done
            and self._env.board.current_player == PLAYER_2
            and event.type == pygame.USEREVENT  # not used — AI moves happen in draw loop
        ):
            pass

    def _do_placement(self, r: int, c: int) -> None:
        assert self._env is not None
        player = self._env.board.current_player
        action = action_to_index(r, c, self._n)
        self._obs, reward, done, info = self._env.step(action)
        self._last_move = (r, c)
        self._move_log.append((player, r, c))
        if done:
            self._game_done = True
            self._winner = info.get("winner")

    def _ai_move(self) -> None:
        """Let the AI pick and execute its move."""
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

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        if self._state == SCREEN_MAIN_MENU:
            self._main_menu.draw(self._screen)

        elif self._state == SCREEN_SIZE_PICK:
            self._size_picker.draw(self._screen)

        elif self._state == SCREEN_MODE_PICK:
            self._mode_picker.draw(self._screen)

        elif self._state == SCREEN_LEVEL_SELECT and self._level_screen:
            self._level_screen.draw(self._screen)

        elif self._state == SCREEN_INGAME:
            self._draw_ingame()

        elif self._state == SCREEN_GAME_OVER:
            self._draw_ingame()
            self._draw_game_over_overlay()

        elif self._state == SCREEN_REPLAY and self._replay_viewer:
            self._screen.fill(BG_DEEP)
            self._replay_viewer.draw(self._screen)

        elif self._state == SCREEN_SETTINGS:
            self._settings_screen.draw(self._screen)

    def _draw_ingame(self) -> None:
        assert self._env is not None
        board = self._env.board
        n = board.size
        w, h = self._screen.get_size()
        bw, bh = self._board_view.board_pixel_size(n)

        # Board area
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

        # Sidebar
        sx = bw
        self._screen.fill(BG_MID, pygame.Rect(sx, 0, w - sx, h))

        # Title
        title = font("display").render("C_lines", True, TEXT_BRIGHT)
        self._screen.blit(title, (sx + 20, 20))

        # Mode / size
        mode_str = "First to Four" if self._mode == MODE_FIRST_TO_FOUR else "Points Until Full"
        for li, line in enumerate([
            f"Mode: {mode_str}",
            f"Board: {n}×{n}",
            f"Turn: {board.turn}",
        ]):
            surf = font("ui").render(line, True, TEXT_MUTED)
            self._screen.blit(surf, (sx + 20, 80 + li * 24))

        # Scores
        p1s, p2s = compute_scores(board)
        self._screen.blit(font("ui").render("─" * 22, True, TEXT_DIM), (sx + 20, 168))
        for pi, (label, score, col) in enumerate([
            ("Player 1", p1s, P1_ACCENT),
            ("Player 2", p2s, P2_ACCENT),
        ]):
            self._screen.blit(font("ui").render(label, True, col), (sx + 20, 180 + pi * 40))
            self._screen.blit(
                font("mono").render(f"{score:.2f} pts", True, col),
                (sx + 20, 198 + pi * 40),
            )

        # Current player indicator
        cp = board.current_player
        cp_col = P1_ACCENT if cp == PLAYER_1 else P2_ACCENT
        cp_surf = font("ui").render(f"► Player {cp} to move", True, cp_col)
        self._screen.blit(cp_surf, (sx + 20, 275))

        # Controls hint
        hints = ["Click = place", "ESC = menu", "R = replay (after game)"]
        for li, hint in enumerate(hints):
            hs = font("small").render(hint, True, TEXT_DIM)
            self._screen.blit(hs, (sx + 20, h - 80 + li * 18))

        # AI move
        if (
            not self._game_done
            and self._vs_ai
            and board.current_player == PLAYER_2
            and self._ai_agent is not None
        ):
            self._ai_move()

    def _draw_game_over_overlay(self) -> None:
        w, h = self._screen.get_size()
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self._screen.blit(overlay, (0, 0))

        if self._winner is None:
            result_text = "Draw!"
            col = DRAW_GLOW
        else:
            result_text = f"Player {self._winner} wins!"
            col = P1_ACCENT if self._winner == PLAYER_1 else P2_ACCENT

        title = font("display").render(result_text, True, col)
        self._screen.blit(title, ((w - title.get_width()) // 2, h // 3))

        p1s, p2s = compute_scores(self._env.board)
        for li, line in enumerate([
            f"Player 1: {p1s:.2f} pts",
            f"Player 2: {p2s:.2f} pts",
        ]):
            ls = font("ui").render(line, True, TEXT_MUTED)
            self._screen.blit(ls, ((w - ls.get_width()) // 2, h // 3 + 60 + li * 28))

        options = [
            ("R — Replay", "replay"),
            ("N — New game", "new"),
            ("M — Main menu", "menu"),
        ]
        for li, (label, _) in enumerate(options):
            os = font("ui").render(label, True, TEXT_DIM)
            self._screen.blit(os, ((w - os.get_width()) // 2, h // 2 + 60 + li * 28))

    def _handle_game_over_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self._state = SCREEN_REPLAY
                self._replay_viewer = ReplayViewer(self._n, self._move_log, CELL_PX, BOARD_MARGIN)
            elif event.key == pygame.K_n:
                self._start_game()
            elif event.key in (pygame.K_m, pygame.K_ESCAPE):
                self._state = SCREEN_MAIN_MENU


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
