"""U-01 .. U-25 — headless UI smoke tests.

Runs against SDL's dummy video driver, so every screen is really rendered and
every event path really executed; nothing is mocked except the four functions
that would otherwise write into the user's results/ folder.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from src.config import (
    MODE_FIRST_TO_FOUR, MODE_POINTS_FULL, PLAYER_1, PLAYER_2,
    UI_BOARD_SIZES, DEFAULT_UI_SETTINGS,
)
from src.engine.rules import is_legal_placement
from src.game.stats import PlayerStats
from src.ui import app as app_mod
from src.ui.replay_browser import ReplayBrowser
from src.ui.replay_view import ReplayViewer
from src.ui.stats_screen import StatsScreen

SIZE = UI_BOARD_SIZES[0]


class StubAgent:
    """Deterministic opponent: always the first legal cell, scanning order."""

    in_channels = 10

    def select_action(self, obs, legal_mask):
        return int(np.flatnonzero(legal_mask)[0])

    def q_values(self, obs):
        return np.zeros(SIZE * SIZE, dtype=np.float32)


@pytest.fixture
def app(tmp_path, monkeypatch):
    """An App whose persistence is redirected away from the real results/ dir."""
    saved: dict = {"settings": dict(DEFAULT_UI_SETTINGS), "stats": None, "records": []}

    monkeypatch.setattr(app_mod, "load_settings", lambda *a, **k: dict(DEFAULT_UI_SETTINGS))
    monkeypatch.setattr(app_mod, "save_settings",
                        lambda s, *a, **k: saved.__setitem__("settings", dict(s)) or True)
    monkeypatch.setattr(app_mod, "load_stats", lambda *a, **k: PlayerStats())
    monkeypatch.setattr(app_mod, "save_stats",
                        lambda s, *a, **k: saved.__setitem__("stats", s) or True)
    monkeypatch.setattr(app_mod, "save_record",
                        lambda r, *a, **k: saved["records"].append(r) or (tmp_path / "g.json"))
    monkeypatch.setattr(app_mod, "ReplayBrowser", lambda *a, **k: ReplayBrowser(tmp_path))

    instance = app_mod.App()
    instance.test_saved = saved       # let tests inspect what would have been written
    yield instance
    pygame.display.quit()


def _click(pos, button: int = 1) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": button})


def _key(key: int, mod: int = 0) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": mod, "unicode": ""})


def _last_toast(app) -> str:
    return app._toaster._items[-1].text if app._toaster._items else ""


def _start(app, vs_ai: bool = False, mode: str = MODE_POINTS_FULL):
    app._vs_ai = vs_ai
    app._mode = mode
    app._n = SIZE
    app._start_game()
    if vs_ai:
        app._ai_agent = StubAgent()
    return app


# ---------------------------------------------------------------------------
# Every screen renders
# ---------------------------------------------------------------------------

def test_u01_all_menu_screens_render(app):
    for state in (
        app_mod.SCREEN_MAIN_MENU,
        app_mod.SCREEN_SIZE_PICK,
        app_mod.SCREEN_MODE_PICK,
        app_mod.SCREEN_SETTINGS,
        app_mod.SCREEN_STATS,
    ):
        app._state = state
        app._draw()


def test_u02_ingame_and_game_over_render(app):
    _start(app)
    app._draw()
    app._finish_game(PLAYER_1)
    assert app._state == app_mod.SCREEN_GAME_OVER
    app._draw()


def test_u03_replay_and_browser_render(app, tmp_path):
    _start(app)
    app._do_placement(3, 3)
    app._do_placement(4, 4)
    app._open_replay(app._move_log, SIZE, app._mode, "Replay")
    assert app._state == app_mod.SCREEN_REPLAY
    app._draw()

    # Empty browser (no saved games yet) …
    app._replay_browser = ReplayBrowser(tmp_path)
    app._state = app_mod.SCREEN_REPLAY_BROWSER
    app._draw()

    # … and a populated one.
    from src.game.records import build_record, save_record
    from datetime import datetime, timezone

    save_record(
        build_record(
            board_size=SIZE, mode=MODE_POINTS_FULL, moves=app._move_log,
            winner=PLAYER_1, p1_score=0.0, p2_score=0.0, vs_ai=True,
            started_at=datetime.now(timezone.utc), ai_name="Master (master)",
            ai_band="master",
        ),
        tmp_path,
    )
    app._replay_browser.refresh()
    assert len(app._replay_browser._rows) == 1
    app._draw()


def test_u04_level_select_renders_with_or_without_snapshots(app):
    from src.ui.level_select import LevelSelectScreen

    screen = LevelSelectScreen(SIZE)
    app._level_screen = screen
    app._state = app_mod.SCREEN_LEVEL_SELECT
    app._draw()
    assert screen.selected_meta is None or screen.selected_meta.board_size == SIZE


def test_u04b_level_select_sorts_weakest_first_and_filters_by_band(app):
    from src.ui.level_select import BANDS, LevelSelectScreen, _strength_key, _short_run

    screen = LevelSelectScreen(SIZE)
    if not screen.snapshots:
        pytest.skip("no trained snapshots on this machine")

    keys = [_strength_key(s) for s in screen.snapshots]
    assert keys == sorted(keys), "cards must read weakest -> strongest"

    counts = screen.band_counts()
    assert counts["all"] == len(screen.snapshots)
    assert sum(v for k, v in counts.items() if k != "all") == counts["all"]

    for idx, band in enumerate(BANDS):
        screen._band_idx = idx
        screen._apply_filter()
        assert len(screen.snapshots) == counts.get(band, 0)
        if band != "all":
            assert all(s.difficulty_band == band for s in screen.snapshots)
        app._level_screen = screen
        app._state = app_mod.SCREEN_LEVEL_SELECT
        app._draw()

    assert _short_run("run_pts_005") == "pts_005"
    assert _short_run("custom") == "custom"


def test_u04c_changing_the_filter_clears_a_stale_selection(app):
    from src.ui.level_select import LevelSelectScreen

    screen = LevelSelectScreen(SIZE)
    if not screen.snapshots:
        pytest.skip("no trained snapshots on this machine")

    screen._selected = screen.snapshots[0].version_id
    screen._selected_idx = 0
    screen._band_idx = 1
    screen._apply_filter()
    assert screen.selected_version_id is None
    assert screen.selected_meta is None


def test_u05_overlays_render_over_any_screen(app):
    _start(app)
    app._help_open = True
    app._draw()
    app._help_open = False

    app._ask_resign()
    assert app._confirm is not None
    app._draw()


def test_u06_toasts_render(app):
    _start(app)
    app._toaster.push("hello", "info")
    app._toaster.push("careful", "warn")
    assert len(app._toaster) == 2
    app._draw()


# ---------------------------------------------------------------------------
# Responsive layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(900, 640), (1280, 900), (1920, 1080), (700, 520)])
def test_u07_board_rescales_and_hit_testing_follows(app, size):
    _start(app)
    app._screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    app._draw()

    view = app._board_view
    board_rect = view.board_rect(SIZE)
    assert app._board_area().contains(board_rect) or board_rect.width <= app._board_area().width

    # Every cell centre must map back to its own cell at every window size.
    for cell in [(0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE // 2, SIZE // 2)]:
        assert view.pixel_to_cell(*view.cell_center(*cell), SIZE) == cell


def test_u08_clicks_outside_the_grid_are_rejected(app):
    _start(app)
    app._draw()
    view = app._board_view
    rect = view.board_rect(SIZE)
    assert view.pixel_to_cell(rect.x + 2, rect.y + 2, SIZE) is None      # margin
    assert view.pixel_to_cell(rect.right + 40, rect.centery, SIZE) is None


def test_u09_board_never_shrinks_below_the_minimum(app):
    from src.config import MIN_CELL_PX, MAX_CELL_PX

    _start(app)
    for size in [(640, 480), (900, 640), (2560, 1440)]:
        app._screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        app._draw()
        assert MIN_CELL_PX <= app._board_view.cell_px <= MAX_CELL_PX


# ---------------------------------------------------------------------------
# Playing
# ---------------------------------------------------------------------------

def test_u10_placement_records_history_and_last_move(app):
    _start(app)
    app._do_placement(2, 3)
    assert app._move_log == [(PLAYER_1, 2, 3)]
    assert app._last_move == (2, 3)
    assert len(app._board_history) == 1
    assert app._env.board.current_player == PLAYER_2


def test_u11_clicking_a_cell_places_a_piece(app):
    _start(app)
    app._draw()
    pos = app._board_view.cell_center(4, 4)
    app._route_event(_click(pos))
    assert app._move_log == [(PLAYER_1, 4, 4)]


def test_u12_clicking_an_occupied_cell_does_nothing(app):
    _start(app)
    app._draw()
    pos = app._board_view.cell_center(4, 4)
    app._route_event(_click(pos))
    app._route_event(_click(pos))
    assert len(app._move_log) == 1


def test_u13_keyboard_cursor_moves_and_places(app):
    _start(app)
    start = app._cursor_cell
    app._route_event(_key(pygame.K_RIGHT))
    app._route_event(_key(pygame.K_DOWN))
    assert app._cursor_cell == (start[0] + 1, start[1] + 1)

    app._route_event(_key(pygame.K_RETURN))
    assert app._move_log == [(PLAYER_1, *app._cursor_cell)]


def test_u14_cursor_is_clamped_to_the_board(app):
    _start(app)
    for _ in range(SIZE + 5):
        app._route_event(_key(pygame.K_UP))
        app._route_event(_key(pygame.K_LEFT))
    assert app._cursor_cell == (0, 0)


# ---------------------------------------------------------------------------
# AI pacing
# ---------------------------------------------------------------------------

def test_u15_ai_moves_from_update_not_from_draw(app):
    _start(app, vs_ai=True)
    app._settings["ai_speed"] = "instant"
    app._do_placement(0, 0)
    assert app._is_ai_turn()

    app._draw()                       # drawing alone must never advance the game
    assert len(app._move_log) == 1

    app._update()                     # first update opens the thinking window
    app._update()                     # second one plays the move
    assert len(app._move_log) == 2
    assert app._move_log[1][0] == PLAYER_2


def test_u16_ai_delay_is_honoured(app, monkeypatch):
    _start(app, vs_ai=True)
    app._settings["ai_speed"] = "slow"
    app._do_placement(0, 0)

    now = {"t": 10_000}
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: now["t"])

    app._update()
    app._update()
    assert len(app._move_log) == 1    # still inside the think window

    now["t"] += 5_000
    app._update()
    assert len(app._move_log) == 2


def test_u17_ai_move_time_is_recorded(app):
    _start(app, vs_ai=True)
    app._settings["ai_speed"] = "instant"
    app._do_placement(0, 0)
    app._update()
    app._update()
    assert len(app._ai_times) == 1 and app._ai_times[0] >= 0.0


# ---------------------------------------------------------------------------
# Undo / redo
# ---------------------------------------------------------------------------

def test_u18_undo_one_ply_in_hot_seat(app):
    _start(app)
    app._do_placement(1, 1)
    app._do_placement(2, 2)
    app._undo()
    assert app._move_log == [(PLAYER_1, 1, 1)]
    assert app._env.board.grid[2, 2] == 0
    assert app._env.board.current_player == PLAYER_2
    assert app._last_move == (1, 1)


def test_u19_undo_vs_ai_rewinds_to_your_turn(app):
    _start(app, vs_ai=True)
    app._settings["ai_speed"] = "instant"
    app._do_placement(0, 0)
    app._update(); app._update()
    assert len(app._move_log) == 2

    app._undo()
    assert app._move_log == []
    assert app._env.board.current_player == PLAYER_1


def test_u20_redo_restores_the_undone_moves(app):
    _start(app)
    app._do_placement(1, 1)
    app._do_placement(2, 2)
    app._undo()
    app._redo()
    assert app._move_log == [(PLAYER_1, 1, 1), (PLAYER_2, 2, 2)]
    assert app._env.board.grid[2, 2] == PLAYER_2


def test_u21_a_new_move_clears_the_redo_stack(app):
    _start(app)
    app._do_placement(1, 1)
    app._undo()
    assert app._redo_stack
    app._do_placement(5, 5)
    assert app._redo_stack == []


def test_u22_undo_with_no_moves_is_a_no_op(app):
    _start(app)
    app._undo()
    assert app._move_log == []
    assert app._env.board.turn == 0


def test_u23_undo_reopens_a_finished_game(app):
    _start(app, mode=MODE_FIRST_TO_FOUR)
    for col in range(3):
        app._do_placement(0, col)          # P1
        app._do_placement(7, col)          # P2
    app._do_placement(0, 3)                # P1 completes four
    assert app._game_done and app._winner == PLAYER_1

    app._undo()
    assert not app._game_done
    assert app._winner is None
    assert app._state == app_mod.SCREEN_INGAME
    assert app._win_line == []


# ---------------------------------------------------------------------------
# Hints, resign, settings
# ---------------------------------------------------------------------------

def test_u24_hint_marks_a_legal_cell(app):
    _start(app, vs_ai=True)
    app._show_hint()
    assert app._hint_cell is not None
    assert is_legal_placement(app._env.board, *app._hint_cell)
    assert len(app._toaster) == 1

    app._do_placement(*app._hint_cell)
    assert app._hint_cell is None          # a move clears the hint


def test_u25_resign_needs_confirmation(app):
    _start(app)
    app._route_event(_key(pygame.K_q))
    assert app._confirm is not None and not app._game_done

    app._route_event(_key(pygame.K_ESCAPE))          # cancel
    assert app._confirm is None and not app._game_done

    app._route_event(_key(pygame.K_q))
    app._route_event(_key(pygame.K_RETURN))          # confirm
    assert app._game_done
    assert app._winner == PLAYER_2 and app._resigned_by == PLAYER_1


def test_u26_setting_toggles_persist_and_take_effect(app):
    _start(app)
    assert app._settings["show_threats"] is False
    app._route_event(_key(pygame.K_t))
    assert app._settings["show_threats"] is True
    assert app.test_saved["settings"]["show_threats"] is True

    app._do_placement(3, 3)
    _, threats = app._analysis()
    assert threats is not None
    app._draw()


def test_u27_analysis_is_cached_per_ply(app, monkeypatch):
    _start(app)
    app._settings["show_eval"] = True
    calls = {"n": 0}
    real = app_mod.evaluate_position

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(app_mod, "evaluate_position", counted)

    app._analysis(); app._analysis(); app._analysis()
    assert calls["n"] == 1
    app._do_placement(2, 2)
    app._analysis()
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Finishing a game
# ---------------------------------------------------------------------------

def test_u28_finishing_updates_stats_and_autosaves(app):
    _start(app, vs_ai=True, mode=MODE_FIRST_TO_FOUR)
    for col in range(3):
        app._do_placement(0, col)
        app._do_placement(7, col)
    app._do_placement(0, 3)

    assert app._game_done and app._winner == PLAYER_1
    assert app._stats.wins == 1 and app._stats.games_vs_ai == 1
    assert app._record_saved is True
    assert len(app.test_saved["records"]) == 1

    record = app.test_saved["records"][0]
    assert record.move_tuples == app._move_log
    assert record.winner == PLAYER_1


def test_u29_win_line_is_found_on_a_win(app):
    _start(app, mode=MODE_FIRST_TO_FOUR)
    for col in range(3):
        app._do_placement(0, col)
        app._do_placement(7, col)
    app._do_placement(0, 3)
    assert sorted(app._win_line) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    app._draw()


def test_u30_manual_save_is_not_duplicated(app):
    _start(app)
    app._do_placement(1, 1)
    app._finish_game(PLAYER_1)
    assert len(app.test_saved["records"]) == 1
    app._save_current_game(manual=True)
    assert len(app.test_saved["records"]) == 1


def test_u30b_undo_then_redo_does_not_double_count(app):
    """Reaching the end twice must not inflate stats or write a second file."""
    _start(app, vs_ai=True, mode=MODE_FIRST_TO_FOUR)
    for col in range(3):
        app._do_placement(0, col)
        app._do_placement(7, col)
    app._do_placement(0, 3)
    assert app._stats.wins == 1 and len(app.test_saved["records"]) == 1

    app._undo()
    assert not app._game_done
    app._redo()
    assert app._game_done and app._winner == PLAYER_1

    assert app._stats.wins == 1
    assert app._stats.games_vs_ai == 1
    assert len(app.test_saved["records"]) == 1


def test_u30c_save_button_reads_saved_after_undo_and_refinish(app):
    """The card must not advertise a Save that would silently do nothing."""
    _start(app, vs_ai=True, mode=MODE_FIRST_TO_FOUR)
    for col in range(3):
        app._do_placement(0, col)
        app._do_placement(7, col)
    app._do_placement(0, 3)
    assert app._game_over_card.data.saved is True

    app._undo()
    app._redo()
    assert app._game_over_card.data.saved is True
    assert app._game_over_card.handle_event(_key(pygame.K_s), app._screen) is None


def test_u30d_disabled_sidebar_buttons_do_not_fire(app):
    """A greyed-out control must be inert, not merely dim."""
    from src.ui.sidebar import action_enabled, layout as sidebar_layout

    _start(app, vs_ai=True)
    app._settings["ai_speed"] = "instant"
    app._draw()

    state = app._sidebar_state()
    assert action_enabled("undo", state) is False       # nothing to undo yet

    buttons = sidebar_layout(app._sidebar_area()).buttons
    app._route_event(_click(buttons["undo"].center))
    assert app._move_log == []
    assert len(app._toaster) == 0                       # not even a "nothing to undo"

    app._do_placement(0, 0)
    app._update(); app._update()
    state = app._sidebar_state()
    assert action_enabled("undo", state) is True
    app._route_event(_click(buttons["undo"].center))
    assert app._move_log == []                          # rewound both plies


def test_u30e_sidebar_is_inert_while_the_result_card_is_up(app):
    from src.ui.sidebar import action_enabled, layout as sidebar_layout

    _start(app)
    app._do_placement(1, 1)
    app._finish_game(PLAYER_1)

    state = app._sidebar_state()
    for key in ("undo", "hint", "resign", "settings", "menu"):
        assert action_enabled(key, state) is False, key

    buttons = sidebar_layout(app._sidebar_area()).buttons
    app._route_event(_click(buttons["menu"].center))
    assert app._state == app_mod.SCREEN_GAME_OVER       # the card still owns input


def test_u30f_eval_and_hint_are_labelled_by_their_real_source(app):
    """A RandomAgent fallback must never be described as "the network"."""
    _start(app, vs_ai=True)
    app._ai_agent = app_mod.RandomAgent()               # no q_values on this one
    assert app._has_network() is False
    assert app._sidebar_state().eval_source == "heuristic"

    app._show_hint()
    assert "(search)" in _last_toast(app)

    app._ai_agent = StubAgent()
    assert app._has_network() is True
    assert app._sidebar_state().eval_source == "network"

    app._toaster.clear()
    app._hint_cell = None
    app._show_hint()
    assert "(agent)" in _last_toast(app)


def test_u31_hot_seat_result_does_not_count_as_a_win(app):
    _start(app, vs_ai=False)
    app._do_placement(1, 1)
    app._finish_game(PLAYER_1)
    assert app._stats.games_hotseat == 1
    assert app._stats.wins == 0


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def test_u32_main_menu_routes_to_every_destination(app):
    routes = {
        "play_vs_ai": app_mod.SCREEN_SIZE_PICK,
        "hot_seat":   app_mod.SCREEN_SIZE_PICK,
        "replays":    app_mod.SCREEN_REPLAY_BROWSER,
        "stats":      app_mod.SCREEN_STATS,
        "settings":   app_mod.SCREEN_SETTINGS,
    }
    for i, (action, expected) in enumerate(routes.items()):
        assert app_mod.MainMenu.ITEMS[i][0] == action
        app._state = app_mod.SCREEN_MAIN_MENU
        app._main_menu._focus = i
        app._route_event(_key(pygame.K_RETURN))
        assert app._state == expected, action
        app._draw()


def test_u33_board_size_picker_defaults_to_an_offered_size(app):
    assert app._size_picker.selected in UI_BOARD_SIZES

    app._state = app_mod.SCREEN_SIZE_PICK
    app._route_event(_key(pygame.K_RETURN))
    assert app._state == app_mod.SCREEN_MODE_PICK
    assert app._size_picker.selected in UI_BOARD_SIZES


def test_u33b_escape_on_the_main_menu_does_not_quit(app):
    """ESC means "back" everywhere; from the main menu there is nowhere to go."""
    app._state = app_mod.SCREEN_MAIN_MENU
    app._route_event(_key(pygame.K_ESCAPE))
    assert app._state == app_mod.SCREEN_MAIN_MENU

    app._main_menu._focus = len(app_mod.MainMenu.ITEMS) - 1
    assert app_mod.MainMenu.ITEMS[app._main_menu._focus][0] == "quit"


def test_u34_help_overlay_swallows_input(app):
    _start(app)
    app._route_event(_key(pygame.K_F1))
    assert app._help_open

    app._route_event(_key(pygame.K_RIGHT))            # consumed by the overlay
    assert not app._help_open
    assert app._move_log == []


def test_u35_replay_viewer_transport(app):
    moves = [(PLAYER_1, 0, 0), (PLAYER_2, 1, 1), (PLAYER_1, 0, 1), (PLAYER_2, 2, 2)]
    viewer = ReplayViewer(SIZE, moves, mode=MODE_POINTS_FULL, title="t")
    surface = pygame.display.set_mode((900, 640), pygame.RESIZABLE)

    assert viewer.total == 4
    viewer.handle_event(_key(pygame.K_RIGHT), surface)
    viewer.handle_event(_key(pygame.K_RIGHT), surface)
    assert viewer.current_board.grid[1, 1] == PLAYER_2

    viewer.handle_event(_key(pygame.K_END), surface)
    assert viewer._step == 4
    viewer.handle_event(_key(pygame.K_HOME), surface)
    assert viewer._step == 0
    assert viewer.current_board.grid.sum() == 0

    viewer.handle_event(_key(pygame.K_SPACE), surface)
    assert viewer._playing is True
    viewer.draw(surface)

    assert viewer.handle_event(_key(pygame.K_ESCAPE), surface) == "back"


def test_u36_replay_viewer_survives_an_empty_game(app):
    viewer = ReplayViewer(SIZE, [], mode=MODE_POINTS_FULL)
    surface = pygame.display.set_mode((900, 640), pygame.RESIZABLE)
    viewer.draw(surface)
    viewer.handle_event(_key(pygame.K_RIGHT), surface)
    assert viewer._step == 0


@pytest.mark.parametrize("bad_move", [
    (PLAYER_1, 99, 0),        # row out of range
    (PLAYER_1, 0, -3),        # negative column
    (PLAYER_1, 0, 0),         # duplicate of an earlier move
])
def test_u36b_replay_viewer_truncates_a_corrupt_record(app, bad_move):
    """Saved games are user-editable JSON — a bad one must not crash the window."""
    moves = [(PLAYER_1, 0, 0), (PLAYER_2, 1, 1), bad_move, (PLAYER_2, 2, 2)]
    viewer = ReplayViewer(SIZE, moves, mode=MODE_POINTS_FULL)
    surface = pygame.display.set_mode((900, 640), pygame.RESIZABLE)

    assert viewer.total == 2                    # stops at the last sane move
    viewer.handle_event(_key(pygame.K_END), surface)
    viewer.draw(surface)


def test_u36c_replay_browser_double_click_is_per_row(app, tmp_path):
    """Fast clicks on two different rows are two single clicks, not a double."""
    from datetime import datetime, timezone
    from src.game.records import build_record, save_record

    for i in range(2):
        save_record(
            build_record(
                board_size=SIZE, mode=MODE_POINTS_FULL,
                moves=[(PLAYER_1, i, i)], winner=PLAYER_1,
                p1_score=0.0, p2_score=0.0, vs_ai=False,
                started_at=datetime.now(timezone.utc),
            ),
            tmp_path,
        )

    browser = ReplayBrowser(tmp_path)
    surface = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    browser.draw(surface)
    viewport = browser._viewport(surface)

    row0 = browser._row_rect(0, viewport)
    row1 = browser._row_rect(1, viewport)
    body0 = (row0.x + 100, row0.centery)        # away from the Open/Delete buttons
    body1 = (row1.x + 100, row1.centery)

    assert browser.handle_event(_click(body0), surface) is None
    assert browser.handle_event(_click(body1), surface) is None   # different row
    assert browser.handle_event(_click(body1), surface) == "open"  # same row, quick


def test_u37_stats_screen_renders_empty_and_populated():
    pygame.display.set_mode((900, 640), pygame.RESIZABLE)
    surface = pygame.display.get_surface()

    screen = StatsScreen(PlayerStats())
    screen.draw(surface)

    from src.game.stats import record_game

    filled = PlayerStats()
    record_game(filled, vs_ai=True, winner=PLAYER_1, mode=MODE_POINTS_FULL,
                move_count=22, band="master", duration_sec=120)
    record_game(filled, vs_ai=True, winner=PLAYER_2, mode=MODE_FIRST_TO_FOUR,
                move_count=9, band="easy")
    screen.set_stats(filled)
    screen.draw(surface)
