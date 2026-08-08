"""C_lines PyGame application — main loop and screen state machine.

The loop is a strict three-phase cycle: handle events, `_update()` (the only
place game state is allowed to change), then `_draw()`.  Keeping the AI's move
out of the draw phase is what lets the window stay responsive and the
"Thinking…" indicator actually appear.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Optional

import pygame

from src.config import (
    DEFAULT_MODE, UI_BOARD_SIZES,
    CELL_PX, BOARD_MARGIN, SIDEBAR_W, PLAYER_1, PLAYER_2,
    MIN_WINDOW_W, MIN_WINDOW_H,
    AI_THINK_DELAY_MS, DEFAULT_AI_SPEED,
    DROP_ANIM_MS, WIN_LINE_ANIM_MS, MODAL_FADE_MS,
    MAX_UNDO_PLIES, STATE_CHANNELS,
)
from src.engine.board import Board, clone_board
from src.engine.rules import is_legal_placement
from src.engine.scoring import compute_scores
from src.game.env import GameEnv
from src.game.encoding import index_to_action, action_to_index
from src.game.analysis import (
    algebraic, build_threat_overlay, deciding_line, evaluate_position,
    line_breakdown, suggest_move_detailed, ThreatOverlay,
)
from src.game.records import build_record, save_record
from src.game.stats import load_stats, save_stats, record_game
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.ui.animations import DropAnim, Tween
from src.ui.board_view import BoardView
from src.ui.level_select import LevelSelectScreen
from src.ui.menus import MainMenu, BoardSizePicker, ModePicker, SettingsScreen
from src.ui.overlays import ConfirmDialog, GameOverCard, GameOverData, HelpOverlay
from src.ui.replay_browser import ReplayBrowser
from src.ui.replay_view import ReplayViewer
from src.ui.settings_store import load_settings, save_settings
from src.ui.sidebar import (
    SidebarState, action_enabled, draw as draw_sidebar, layout as sidebar_layout,
)
from src.ui.stats_screen import StatsScreen
from src.ui.theme import BG_DEEP, init_fonts, draw_button
from src.ui.toast import Toaster

FPS = 60
HUMAN_PLAYER = PLAYER_1

SCREEN_MAIN_MENU      = "main_menu"
SCREEN_SIZE_PICK      = "size_pick"
SCREEN_MODE_PICK      = "mode_pick"
SCREEN_LEVEL_SELECT   = "level_select"
SCREEN_INGAME         = "ingame"
SCREEN_GAME_OVER      = "game_over"
SCREEN_REPLAY         = "replay"
SCREEN_REPLAY_BROWSER = "replay_browser"
SCREEN_STATS          = "stats"
SCREEN_SETTINGS       = "settings"


def _preferred_window_size(n: int) -> tuple[int, int]:
    w = n * CELL_PX + 2 * BOARD_MARGIN + SIDEBAR_W
    h = n * CELL_PX + 2 * BOARD_MARGIN
    return max(w, MIN_WINDOW_W), max(h, MIN_WINDOW_H)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("C_lines")

        self._settings: dict = load_settings()
        self._stats = load_stats()

        self._n = UI_BOARD_SIZES[0]
        self._mode = DEFAULT_MODE
        self._vs_ai = False
        self._ai_version_id: Optional[str] = None
        self._ai_meta = None

        w, h = _preferred_window_size(self._n)
        self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        init_fonts()

        self._clock = pygame.time.Clock()
        self._state = SCREEN_MAIN_MENU
        self._prev_state = SCREEN_MAIN_MENU

        # Screens
        self._main_menu     = MainMenu()
        self._size_picker   = BoardSizePicker()
        self._mode_picker   = ModePicker()
        self._settings_scr  = SettingsScreen(self._settings)
        self._stats_screen  = StatsScreen(self._stats)
        self._replay_browser: Optional[ReplayBrowser] = None
        self._level_screen: Optional[LevelSelectScreen] = None
        self._board_view    = BoardView(CELL_PX, BOARD_MARGIN)
        self._replay_viewer: Optional[ReplayViewer] = None

        # Floating layers
        self._toaster = Toaster()
        self._help = HelpOverlay()
        self._help_open = False
        self._confirm: Optional[ConfirmDialog] = None
        self._confirm_action: str = ""

        # In-game state
        self._env: Optional[GameEnv] = None
        self._ai_agent = None
        self._move_log: list[tuple[int, int, int]] = []
        self._board_history: list[Board] = []      # state *before* each move
        self._redo_stack: list[tuple[int, int, int]] = []
        self._hover_cell: Optional[tuple[int, int]] = None
        self._cursor_cell: Optional[tuple[int, int]] = None
        self._last_move: Optional[tuple[int, int]] = None
        self._hint_cell: Optional[tuple[int, int]] = None
        self._winner: Optional[int] = None
        self._resigned_by: Optional[int] = None
        self._game_done = False
        self._obs = None
        self._started_at = datetime.now(timezone.utc)

        # AI pacing
        self._ai_pending_since: Optional[int] = None
        self._ai_times: list[float] = []

        # Presentation caches / animation
        self._drop: Optional[DropAnim] = None
        self._win_line: list[tuple[int, int]] = []
        self._win_tween: Optional[Tween] = None
        self._modal_tween: Optional[Tween] = None
        self._analysis_key: Optional[tuple] = None
        self._cached_eval: Optional[float] = None
        self._cached_threats: Optional[ThreatOverlay] = None

        self._game_over_card: Optional[GameOverCard] = None
        self._record_saved = False
        self._stats_counted = False

        self._refresh_menu_footer()

    # ==================================================================
    # Main loop
    # ==================================================================

    def run(self) -> None:
        while True:
            self._clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._quit()
                if event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(
                        (max(640, event.w), max(480, event.h)), pygame.RESIZABLE
                    )
                    continue
                self._route_event(event)
            self._update()
            self._screen.fill(BG_DEEP)
            self._draw()
            pygame.display.flip()

    def _quit(self) -> None:
        save_settings(self._settings)
        save_stats(self._stats)
        pygame.quit()
        sys.exit()

    # ==================================================================
    # Event routing
    # ==================================================================

    def _route_event(self, event: pygame.event.Event) -> None:
        # Floating layers swallow everything while they are up.
        if self._help_open:
            if self._help.handle_event(event) == "close":
                self._help_open = False
            return

        if self._confirm is not None:
            result = self._confirm.handle_event(event, self._screen)
            if result == "confirm":
                action, self._confirm, self._confirm_action = self._confirm_action, None, ""
                self._apply_confirmed(action)
            elif result == "cancel":
                self._confirm, self._confirm_action = None, ""
            return

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_F1, pygame.K_QUESTION):
            self._help_open = True
            return

        handler = {
            SCREEN_MAIN_MENU:      self._ev_main_menu,
            SCREEN_SIZE_PICK:      self._ev_size_pick,
            SCREEN_MODE_PICK:      self._ev_mode_pick,
            SCREEN_LEVEL_SELECT:   self._ev_level_select,
            SCREEN_INGAME:         self._ev_ingame,
            SCREEN_GAME_OVER:      self._ev_game_over,
            SCREEN_REPLAY:         self._ev_replay,
            SCREEN_REPLAY_BROWSER: self._ev_replay_browser,
            SCREEN_STATS:          self._ev_stats,
            SCREEN_SETTINGS:       self._ev_settings,
        }.get(self._state)
        if handler:
            handler(event)

    # ------------------------------------------------------------------
    def _ev_main_menu(self, event: pygame.event.Event) -> None:
        result = self._main_menu.handle_event(event, self._screen)
        if result == "play_vs_ai":
            self._vs_ai = True
            self._state = SCREEN_SIZE_PICK
        elif result == "hot_seat":
            self._vs_ai = False
            self._state = SCREEN_SIZE_PICK
        elif result == "replays":
            self._replay_browser = ReplayBrowser()
            self._state = SCREEN_REPLAY_BROWSER
        elif result == "stats":
            self._stats_screen.set_stats(self._stats)
            self._state = SCREEN_STATS
        elif result == "settings":
            self._prev_state = SCREEN_MAIN_MENU
            self._state = SCREEN_SETTINGS
        elif result == "quit":
            self._quit()

    def _ev_size_pick(self, event: pygame.event.Event) -> None:
        if self._back_button_clicked(event):
            self._state = SCREEN_MAIN_MENU
            return
        result = self._size_picker.handle_event(event, self._screen)
        if result == "confirm":
            self._state = SCREEN_MODE_PICK
        elif result == "back":
            self._state = SCREEN_MAIN_MENU

    def _ev_mode_pick(self, event: pygame.event.Event) -> None:
        if self._back_button_clicked(event):
            self._state = SCREEN_SIZE_PICK
            return
        result = self._mode_picker.handle_event(event, self._screen)
        if result == "confirm":
            self._mode = self._mode_picker.selected
            self._n = self._size_picker.selected
            if self._vs_ai:
                self._level_screen = LevelSelectScreen(self._n)
                self._state = SCREEN_LEVEL_SELECT
            else:
                self._ai_version_id, self._ai_meta = None, None
                self._start_game()
        elif result == "back":
            self._state = SCREEN_SIZE_PICK

    def _ev_level_select(self, event: pygame.event.Event) -> None:
        assert self._level_screen is not None
        result = self._level_screen.handle_event(event, self._screen)
        if result == "play":
            self._ai_version_id = self._level_screen.selected_version_id
            self._ai_meta = self._level_screen.selected_meta
            self._n = self._size_picker.selected
            self._start_game()
        elif result == "back":
            self._state = SCREEN_MODE_PICK

    # ------------------------------------------------------------------
    def _ev_ingame(self, event: pygame.event.Event) -> None:
        assert self._env is not None
        board = self._env.board

        if event.type == pygame.KEYDOWN:
            if self._handle_ingame_key(event):
                return

        if event.type == pygame.MOUSEMOTION:
            self._hover_cell = self._board_view.pixel_to_cell(*event.pos, self._n)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            lay = sidebar_layout(self._sidebar_area())
            for key, rect in lay.buttons.items():
                if not rect.collidepoint(event.pos):
                    continue
                # Only act if the button was drawn live — a greyed-out control
                # must not fire.
                if action_enabled(key, self._sidebar_state()):
                    self._sidebar_action(key)
                return
            if self._game_done or self._is_ai_turn():
                return
            cell = self._board_view.pixel_to_cell(*event.pos, self._n)
            if cell is not None and is_legal_placement(board, *cell):
                self._cursor_cell = cell
                self._do_placement(*cell)

    def _handle_ingame_key(self, event: pygame.event.Event) -> bool:
        """Returns True when the key was consumed."""
        mods = pygame.key.get_mods()
        ctrl = bool(mods & pygame.KMOD_CTRL)
        key = event.key

        if key == pygame.K_ESCAPE:
            self._state = SCREEN_MAIN_MENU
            return True
        if key == pygame.K_q:
            self._ask_resign()
            return True
        if key == pygame.K_u or (ctrl and key == pygame.K_z):
            self._undo()
            return True
        if ctrl and key == pygame.K_y:
            self._redo()
            return True
        if key == pygame.K_h:
            self._show_hint()
            return True
        if key == pygame.K_t:
            self._toggle_setting("show_threats", "Threat overlay")
            return True
        if key == pygame.K_e:
            self._toggle_setting("show_eval", "Evaluation meter")
            return True

        # Keyboard board navigation
        deltas = {
            pygame.K_LEFT: (0, -1), pygame.K_RIGHT: (0, 1),
            pygame.K_UP: (-1, 0), pygame.K_DOWN: (1, 0),
        }
        if key in deltas:
            self._move_cursor(*deltas[key])
            return True
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._place_at_cursor()
            return True
        return False

    def _ev_game_over(self, event: pygame.event.Event) -> None:
        if self._game_over_card is None:
            return
        # Undo is still reachable here — taking back the losing move is the most
        # common thing anyone wants from this screen.
        if event.type == pygame.KEYDOWN:
            ctrl = bool(pygame.key.get_mods() & pygame.KMOD_CTRL)
            if event.key == pygame.K_u or (ctrl and event.key == pygame.K_z):
                self._undo()
                return

        action = self._game_over_card.handle_event(event, self._screen)
        if action == "replay":
            self._open_replay(
                self._move_log, self._n, self._mode, "Replay — last game",
                self._opponent_label(),
            )
        elif action == "save":
            self._save_current_game(manual=True)
        elif action == "new":
            self._start_game()
        elif action == "menu":
            self._state = SCREEN_MAIN_MENU

    def _ev_replay(self, event: pygame.event.Event) -> None:
        assert self._replay_viewer is not None
        if self._replay_viewer.handle_event(event, self._screen) == "back":
            self._state = self._prev_state

    def _ev_replay_browser(self, event: pygame.event.Event) -> None:
        assert self._replay_browser is not None
        result = self._replay_browser.handle_event(event, self._screen)
        if result == "back":
            self._state = SCREEN_MAIN_MENU
        elif result == "open":
            rec = self._replay_browser.selected_record
            if rec is None or not rec.moves:
                self._toaster.push("That game has no moves to replay", "warn")
                return
            self._prev_state = SCREEN_REPLAY_BROWSER
            self._open_replay(
                rec.move_tuples, rec.board_size, rec.mode,
                f"Replay — {rec.opponent_label}", rec.summary_line(),
                keep_prev=True,
            )

    def _ev_stats(self, event: pygame.event.Event) -> None:
        result = self._stats_screen.handle_event(event, self._screen)
        if result == "back":
            self._state = SCREEN_MAIN_MENU
        elif result == "reset":
            self._stats = self._stats_screen.stats
            self._refresh_menu_footer()
            self._toaster.push("Statistics reset", "warn")

    def _ev_settings(self, event: pygame.event.Event) -> None:
        result = self._settings_scr.handle_event(event, self._screen)
        if result == "changed":
            save_settings(self._settings)
            self._invalidate_analysis()
        elif result == "back":
            save_settings(self._settings)
            self._state = self._prev_state

    # ------------------------------------------------------------------
    def _back_button_clicked(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self._back_rect().collidepoint(event.pos)
        return False

    def _back_rect(self) -> pygame.Rect:
        _, h = self._screen.get_size()
        return pygame.Rect(20, h - 56, 110, 36)

    # ==================================================================
    # Update phase — the only place game state changes on its own
    # ==================================================================

    def _update(self) -> None:
        self._toaster.update()

        if self._drop is not None and self._drop.done():
            self._drop = None

        if self._state != SCREEN_INGAME or self._game_done or self._env is None:
            return

        if not self._is_ai_turn() or self._ai_agent is None:
            self._ai_pending_since = None
            return

        now = pygame.time.get_ticks()
        if self._ai_pending_since is None:
            # Start the visible "thinking" window — the move is played after it.
            self._ai_pending_since = now
            return

        delay = AI_THINK_DELAY_MS.get(
            self._settings.get("ai_speed", DEFAULT_AI_SPEED), 0
        )
        if now - self._ai_pending_since >= delay:
            self._ai_pending_since = None
            self._ai_move()

    # ==================================================================
    # Game lifecycle
    # ==================================================================

    def _start_game(self) -> None:
        self._ensure_window_for_board()

        state_channels = STATE_CHANNELS
        self._ai_agent = None
        if self._vs_ai:
            self._ai_agent, state_channels = self._load_ai()

        self._env = GameEnv(
            board_size=self._n, mode=self._mode, state_channels=state_channels
        )
        self._obs = self._env.reset()

        self._move_log = []
        self._board_history = []
        self._redo_stack = []
        self._last_move = None
        self._hint_cell = None
        self._hover_cell = None
        self._cursor_cell = (self._n // 2, self._n // 2)
        self._winner = None
        self._resigned_by = None
        self._game_done = False
        self._started_at = datetime.now(timezone.utc)
        self._ai_times = []
        self._ai_pending_since = None
        self._drop = None
        self._win_line = []
        self._win_tween = None
        self._modal_tween = None
        self._game_over_card = None
        self._record_saved = False
        self._stats_counted = False
        self._invalidate_analysis()

        # Establish board metrics now so a click landing before the first draw
        # still hit-tests against the right geometry.
        self._board_view.fit(self._board_area(), self._n)
        self._state = SCREEN_INGAME

    def _load_ai(self) -> tuple[object, int]:
        """Load the chosen snapshot; fall back to a random agent if anything fails."""
        if not self._ai_version_id:
            return RandomAgent(), STATE_CHANNELS
        try:
            from src.training.snapshot import load_snapshot

            run_id = getattr(self._ai_meta, "run_id", None)
            agent = load_snapshot(self._n, self._ai_version_id, run_id=run_id)
            return agent, agent.in_channels
        except Exception:
            self._toaster.push("Could not load that agent — using a random opponent", "error")
            self._ai_meta = None
            return RandomAgent(), STATE_CHANNELS

    def _ensure_window_for_board(self) -> None:
        want_w, want_h = _preferred_window_size(self._n)
        have_w, have_h = self._screen.get_size()
        if have_w < want_w or have_h < want_h:
            self._screen = pygame.display.set_mode(
                (max(have_w, want_w), max(have_h, want_h)), pygame.RESIZABLE
            )

    # ------------------------------------------------------------------
    def _do_placement(self, r: int, c: int) -> None:
        assert self._env is not None
        board = self._env.board
        player = board.current_player

        self._board_history.append(clone_board(board))
        if len(self._board_history) > MAX_UNDO_PLIES:
            self._board_history.pop(0)
        self._redo_stack.clear()

        self._obs, _, done, info = self._env.step(action_to_index(r, c, self._n))
        self._last_move = (r, c)
        self._move_log.append((player, r, c))
        self._hint_cell = None
        self._invalidate_analysis()

        if not self._settings.get("reduce_motion"):
            self._drop = DropAnim.start(r, c, player, DROP_ANIM_MS)

        if done:
            self._finish_game(info.get("winner"))

    def _ai_move(self) -> None:
        assert self._env is not None and self._ai_agent is not None
        board = self._env.board
        mask = self._env.legal_mask()
        if not mask.any():
            return

        if isinstance(self._ai_agent, HeuristicAgent):
            self._ai_agent.set_board(board)

        t0 = time.perf_counter()
        try:
            action = int(self._ai_agent.select_action(self._obs, mask))
        except Exception:
            self._toaster.push("AI failed to move — playing at random", "error")
            self._ai_agent = RandomAgent()
            action = int(self._ai_agent.select_action(self._obs, mask))
        self._ai_times.append(time.perf_counter() - t0)

        self._do_placement(*index_to_action(action, self._n))

    def _place_at_cursor(self) -> None:
        if self._env is None or self._game_done or self._is_ai_turn():
            return
        if self._cursor_cell is None:
            self._cursor_cell = (self._n // 2, self._n // 2)
            return
        r, c = self._cursor_cell
        if is_legal_placement(self._env.board, r, c):
            self._do_placement(r, c)
        else:
            self._toaster.push(f"{algebraic(r, c)} is already taken", "warn")

    def _move_cursor(self, dr: int, dc: int) -> None:
        r, c = self._cursor_cell or (self._n // 2, self._n // 2)
        self._cursor_cell = (
            max(0, min(self._n - 1, r + dr)),
            max(0, min(self._n - 1, c + dc)),
        )

    # ------------------------------------------------------------------
    def _finish_game(self, winner: Optional[int], resigned_by: Optional[int] = None) -> None:
        assert self._env is not None
        self._game_done = True
        self._winner = winner
        self._resigned_by = resigned_by
        self._ai_pending_since = None

        board = self._env.board
        p1_score, p2_score = compute_scores(board)

        self._win_line = deciding_line(board, self._mode, winner)
        if self._win_line and not self._settings.get("reduce_motion"):
            self._win_tween = Tween(WIN_LINE_ANIM_MS)
        self._modal_tween = (
            None if self._settings.get("reduce_motion") else Tween(MODAL_FADE_MS)
        )

        duration = max(0.0, (datetime.now(timezone.utc) - self._started_at).total_seconds())

        # A game counts once. Undo deliberately does not clear this: reaching the
        # end again after taking a move back must not inflate the record, and the
        # first outcome is the honest one.
        if not self._stats_counted:
            self._stats_counted = True
            record_game(
                self._stats,
                vs_ai=self._vs_ai,
                winner=winner,
                mode=self._mode,
                move_count=len(self._move_log),
                duration_sec=duration,
                band=getattr(self._ai_meta, "difficulty_band", None) if self._vs_ai else None,
                human_player=HUMAN_PLAYER,
            )
            save_stats(self._stats)
            self._refresh_menu_footer()

        self._game_over_card = GameOverCard(GameOverData(
            winner=winner,
            mode=self._mode,
            p1_label="You" if self._vs_ai else "Player 1",
            p2_label=self._opponent_label(),
            p1_score=p1_score,
            p2_score=p2_score,
            p1_lines=line_breakdown(board, PLAYER_1),
            p2_lines=line_breakdown(board, PLAYER_2),
            move_count=len(self._move_log),
            duration_sec=duration,
            resigned_by=resigned_by,
            # A game is written once; if it was already saved before an undo,
            # the button must read "Saved" rather than look live and no-op.
            saved=self._record_saved,
            can_save=bool(self._move_log),
        ))

        if self._settings.get("autosave") and self._move_log:
            self._save_current_game(manual=False)

        self._state = SCREEN_GAME_OVER

    def _save_current_game(self, manual: bool) -> None:
        if self._env is None or self._record_saved or not self._move_log:
            return
        p1_score, p2_score = compute_scores(self._env.board)
        record = build_record(
            board_size=self._n,
            mode=self._mode,
            moves=self._move_log,
            winner=self._winner,
            p1_score=p1_score,
            p2_score=p2_score,
            vs_ai=self._vs_ai,
            started_at=self._started_at,
            ai_version_id=self._ai_version_id,
            ai_run_id=getattr(self._ai_meta, "run_id", None),
            ai_name=self._opponent_label() if self._vs_ai else None,
            ai_band=getattr(self._ai_meta, "difficulty_band", None),
            resigned_by=self._resigned_by,
        )
        path = save_record(record)
        if path is None:
            self._toaster.push("Could not write the save file", "error")
            return
        self._record_saved = True
        if self._game_over_card is not None:
            self._game_over_card.data.saved = True
        self._toaster.push("Game saved" if manual else "Game auto-saved", "good")

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def _can_undo(self) -> bool:
        return bool(self._move_log) and self._state == SCREEN_INGAME

    def _undo(self) -> None:
        if self._env is None or not self._move_log:
            self._toaster.push("Nothing to undo", "warn")
            return
        if self._state not in (SCREEN_INGAME, SCREEN_GAME_OVER):
            return

        popped = 0
        while self._move_log:
            self._pop_ply()
            popped += 1
            # Versus the AI, rewind past its reply so it is your turn again.
            if not self._vs_ai or self._env.board.current_player == HUMAN_PLAYER:
                break

        self._game_done = False
        self._winner = None
        self._resigned_by = None
        self._win_line = []
        self._win_tween = None
        self._game_over_card = None
        self._ai_pending_since = None
        self._drop = None
        self._hint_cell = None
        self._state = SCREEN_INGAME
        self._invalidate_analysis()
        self._toaster.push(f"Undid {popped} move{'s' if popped != 1 else ''}", "info")

    def _pop_ply(self) -> None:
        assert self._env is not None
        move = self._move_log.pop()
        self._redo_stack.append(move)
        board = self._board_history.pop()
        self._obs = self._env.set_state(board)
        self._last_move = (
            (self._move_log[-1][1], self._move_log[-1][2]) if self._move_log else None
        )

    def _redo(self) -> None:
        if self._env is None or not self._redo_stack:
            self._toaster.push("Nothing to redo", "warn")
            return

        replayed = 0
        while self._redo_stack:
            _, r, c = self._redo_stack[-1]
            if not is_legal_placement(self._env.board, r, c):
                self._redo_stack.clear()
                break
            self._redo_stack.pop()
            self._apply_redo_move(r, c)
            replayed += 1
            if self._game_done:
                break
            if self._vs_ai and self._env.board.current_player == HUMAN_PLAYER:
                break
            if not self._vs_ai:
                break

        if replayed:
            self._toaster.push(f"Redid {replayed} move{'s' if replayed != 1 else ''}", "info")

    def _apply_redo_move(self, r: int, c: int) -> None:
        """Replay one move without clearing the rest of the redo stack."""
        assert self._env is not None
        board = self._env.board
        player = board.current_player
        self._board_history.append(clone_board(board))
        self._obs, _, done, info = self._env.step(action_to_index(r, c, self._n))
        self._move_log.append((player, r, c))
        self._last_move = (r, c)
        self._invalidate_analysis()
        if done:
            self._finish_game(info.get("winner"))

    # ------------------------------------------------------------------
    # Hint, resign, settings toggles
    # ------------------------------------------------------------------

    def _show_hint(self) -> None:
        if self._env is None or self._game_done or self._is_ai_turn():
            return
        board = self._env.board
        # Observations are always encoded "side to move first", so the trained
        # network can be asked what *you* should play, not just what it plays.
        cell, source = suggest_move_detailed(
            board, self._mode, agent=self._ai_agent if self._has_network() else None
        )
        if cell is None:
            self._toaster.push("No legal moves left", "warn")
            return
        self._hint_cell = cell
        label = {"network": "agent", "search": "search"}.get(source, "guess")
        self._toaster.push(f"Hint ({label}): {algebraic(*cell)}", "hint")

    def _ask_resign(self) -> None:
        if self._game_done or self._env is None:
            return
        self._confirm_action = "resign"
        self._confirm = ConfirmDialog(
            "Resign this game?",
            "Your opponent is awarded the win.",
            confirm_label="Resign", cancel_label="Keep playing",
        )

    def _apply_confirmed(self, action: str) -> None:
        if action == "resign" and self._env is not None and not self._game_done:
            loser = self._env.board.current_player
            winner = PLAYER_2 if loser == PLAYER_1 else PLAYER_1
            self._finish_game(winner, resigned_by=loser)

    def _sidebar_action(self, key: str) -> None:
        if key == "undo":
            self._undo()
        elif key == "hint":
            self._show_hint()
        elif key == "resign":
            self._ask_resign()
        elif key == "settings":
            self._prev_state = SCREEN_INGAME
            self._state = SCREEN_SETTINGS
        elif key == "menu":
            self._state = SCREEN_MAIN_MENU

    def _toggle_setting(self, key: str, label: str) -> None:
        value = not bool(self._settings.get(key, False))
        self._settings[key] = value
        save_settings(self._settings)
        self._invalidate_analysis()
        self._toaster.push(f"{label} {'on' if value else 'off'}", "info")

    # ------------------------------------------------------------------
    # Replay helpers
    # ------------------------------------------------------------------

    def _open_replay(
        self, moves: list[tuple[int, int, int]], size: int, mode: str,
        title: str, subtitle: str = "", keep_prev: bool = False,
    ) -> None:
        if not keep_prev:
            self._prev_state = SCREEN_GAME_OVER
        self._replay_viewer = ReplayViewer(
            size, moves, CELL_PX, BOARD_MARGIN,
            mode=mode, title=title, subtitle=subtitle,
        )
        self._state = SCREEN_REPLAY

    # ==================================================================
    # Derived state
    # ==================================================================

    def _is_ai_turn(self) -> bool:
        return (
            self._vs_ai
            and self._env is not None
            and self._env.board.current_player == PLAYER_2
        )

    def _has_network(self) -> bool:
        """True only when a real DQN snapshot is loaded.

        The RandomAgent fallback has no `q_values`, so hints and the evaluation
        meter must not be attributed to "the network" when it is in play.
        """
        return self._vs_ai and callable(getattr(self._ai_agent, "q_values", None))

    def _sidebar_state(self, evaluation: Optional[float] = None) -> SidebarState:
        """Assemble the sidebar's view of the game.

        Shared by drawing and hit testing so a button's enabled state is decided
        exactly once.
        """
        assert self._env is not None
        board = self._env.board
        p1_score, p2_score = compute_scores(board)
        is_ai_turn = self._is_ai_turn()

        return SidebarState(
            mode=self._mode,
            board_size=board.size,
            turn=board.turn,
            current_player=board.current_player,
            p1_score=p1_score,
            p2_score=p2_score,
            p1_label="You" if self._vs_ai else "Player 1",
            p2_label=self._opponent_label(),
            vs_ai=self._vs_ai,
            ai_thinking=is_ai_turn and not self._game_done,
            game_over=self._game_done,
            evaluation=evaluation,
            eval_source="network" if self._has_network() else "heuristic",
            show_eval=bool(self._settings.get("show_eval")),
            moves=self._move_log,
            ai_last_sec=self._ai_times[-1] if self._ai_times else None,
            ai_avg_sec=(
                sum(self._ai_times) / len(self._ai_times) if self._ai_times else None
            ),
            can_undo=self._can_undo(),
            can_redo=bool(self._redo_stack),
            hint_available=not is_ai_turn,
        )

    def _opponent_label(self) -> str:
        if not self._vs_ai:
            return "Player 2"
        name = getattr(self._ai_meta, "friendly_name", None)
        band = getattr(self._ai_meta, "difficulty_band", None)
        if name and band:
            return f"{name} ({band})"
        return name or "AI"

    def _invalidate_analysis(self) -> None:
        self._analysis_key = None
        self._cached_eval = None
        self._cached_threats = None

    def _analysis(self) -> tuple[Optional[float], Optional[ThreatOverlay]]:
        """Evaluation and threat overlay for the current position.

        Both are O(n^4)-ish, so they are computed once per ply and reused for
        every frame until the position changes.
        """
        if self._env is None:
            return None, None

        board = self._env.board
        want_eval = bool(self._settings.get("show_eval"))
        want_threats = bool(self._settings.get("show_threats"))
        key = (len(self._move_log), board.current_player, want_eval, want_threats)
        if key == self._analysis_key:
            return self._cached_eval, self._cached_threats

        self._analysis_key = key
        self._cached_eval = (
            evaluate_position(
                board, self._mode, agent=self._ai_agent if self._has_network() else None
            )
            if want_eval else None
        )
        self._cached_threats = (
            build_threat_overlay(board, board.current_player) if want_threats else None
        )
        return self._cached_eval, self._cached_threats

    def _refresh_menu_footer(self) -> None:
        s = self._stats
        if s.games_vs_ai == 0:
            self._main_menu.footer = "no games played yet — F1 for controls"
            return
        wr = s.win_rate
        parts = [f"{s.games_vs_ai} games vs AI"]
        if wr is not None:
            parts.append(f"{wr:.0%} win rate")
        if s.current_streak:
            parts.append(f"{s.current_streak}-game streak")
        self._main_menu.footer = "   ·   ".join(parts)

    # ==================================================================
    # Layout
    # ==================================================================

    def _sidebar_area(self) -> pygame.Rect:
        w, h = self._screen.get_size()
        sw = max(280, min(SIDEBAR_W, w // 3))
        return pygame.Rect(w - sw, 0, sw, h)

    def _board_area(self) -> pygame.Rect:
        w, h = self._screen.get_size()
        sidebar = self._sidebar_area()
        return pygame.Rect(12, 12, max(80, sidebar.x - 24), max(80, h - 24))

    # ==================================================================
    # Draw
    # ==================================================================

    def _draw(self) -> None:
        s = self._state
        if s == SCREEN_MAIN_MENU:
            self._main_menu.draw(self._screen)
        elif s == SCREEN_SIZE_PICK:
            self._size_picker.draw(self._screen)
            self._draw_back_button()
        elif s == SCREEN_MODE_PICK:
            self._mode_picker.draw(self._screen)
            self._draw_back_button()
        elif s == SCREEN_LEVEL_SELECT and self._level_screen:
            self._level_screen.draw(self._screen)
        elif s == SCREEN_INGAME:
            self._draw_ingame()
        elif s == SCREEN_GAME_OVER:
            self._draw_ingame()
            if self._game_over_card:
                fade = self._modal_tween.value() if self._modal_tween else 1.0
                self._game_over_card.draw(self._screen, fade)
        elif s == SCREEN_REPLAY and self._replay_viewer:
            self._replay_viewer.draw(self._screen)
        elif s == SCREEN_REPLAY_BROWSER and self._replay_browser:
            self._replay_browser.draw(self._screen)
        elif s == SCREEN_STATS:
            self._stats_screen.draw(self._screen)
        elif s == SCREEN_SETTINGS:
            self._settings_scr.draw(self._screen)

        self._toaster.draw(self._screen, self._toast_area())

        if self._confirm is not None:
            self._confirm.draw(self._screen)
        if self._help_open:
            self._help.draw(self._screen, self._mode)

    def _toast_area(self) -> pygame.Rect:
        if self._state in (SCREEN_INGAME, SCREEN_GAME_OVER):
            return self._board_area()
        return self._screen.get_rect()

    def _draw_back_button(self) -> None:
        rect = self._back_rect()
        draw_button(
            self._screen, rect, "◄  Back",
            hovered=rect.collidepoint(*pygame.mouse.get_pos()),
            base_surf=self._screen,
        )

    # ------------------------------------------------------------------
    def _draw_ingame(self) -> None:
        assert self._env is not None
        board = self._env.board
        n = board.size

        self._board_view.fit(self._board_area(), n)
        evaluation, threats = self._analysis()

        is_ai_turn = self._is_ai_turn()
        legal_mask = self._env.legal_mask() if self._settings.get("show_legal") else None
        win_progress = self._win_tween.value() if self._win_tween else 1.0

        self._board_view.draw(
            self._screen, board,
            last_move=self._last_move,
            hover_cell=None if (is_ai_turn or self._game_done) else self._hover_cell,
            cursor_cell=None if self._game_done else self._cursor_cell,
            show_legal=bool(self._settings.get("show_legal")),
            legal_mask=legal_mask,
            show_coords=bool(self._settings.get("show_coords")),
            allow_ghost=not is_ai_turn,
            threats=threats,
            hint_cell=self._hint_cell,
            win_line=self._win_line,
            win_progress=win_progress,
            drop=self._drop,
            reduce_motion=bool(self._settings.get("reduce_motion")),
        )

        draw_sidebar(self._screen, self._sidebar_area(), self._sidebar_state(evaluation))


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
