"""P-01 .. P-12 — player statistics accumulation and persistence."""

from __future__ import annotations

import json

from src.config import MODE_FIRST_TO_FOUR, MODE_POINTS_FULL, PLAYER_1, PLAYER_2
from src.game import stats as st


def _win(s: st.PlayerStats, **kw) -> st.PlayerStats:
    args = dict(vs_ai=True, winner=PLAYER_1, mode=MODE_POINTS_FULL, move_count=20)
    args.update(kw)
    return st.record_game(s, **args)


# ---------------------------------------------------------------------------

def test_p01_fresh_stats_are_empty():
    s = st.PlayerStats()
    assert s.games_vs_ai == 0
    assert s.win_rate is None
    assert s.best_band_beaten() is None


def test_p02_result_for_human():
    assert st.result_for_human(PLAYER_1) == st.RESULT_WIN
    assert st.result_for_human(PLAYER_2) == st.RESULT_LOSS
    assert st.result_for_human(None) == st.RESULT_DRAW


def test_p03_win_updates_totals_and_streak():
    s = _win(_win(st.PlayerStats()))
    assert (s.games_vs_ai, s.wins, s.losses, s.draws) == (2, 2, 0, 0)
    assert s.current_streak == 2 and s.best_streak == 2
    assert s.win_rate == 1.0


def test_p04_loss_breaks_the_streak_but_keeps_the_best():
    s = _win(_win(st.PlayerStats()))
    st.record_game(s, vs_ai=True, winner=PLAYER_2, mode=MODE_POINTS_FULL, move_count=10)
    assert s.current_streak == 0
    assert s.best_streak == 2
    assert s.losses == 1


def test_p05_draw_also_breaks_the_streak():
    s = _win(st.PlayerStats())
    st.record_game(s, vs_ai=True, winner=None, mode=MODE_POINTS_FULL, move_count=64)
    assert s.draws == 1 and s.current_streak == 0


def test_p06_fastest_win_only_tracks_wins():
    s = st.PlayerStats()
    _win(s, move_count=30)
    _win(s, move_count=12)
    _win(s, move_count=40)
    st.record_game(s, vs_ai=True, winner=PLAYER_2, mode=MODE_POINTS_FULL, move_count=4)
    assert s.fastest_win_moves == 12


def test_p07_hotseat_games_are_counted_separately():
    s = st.PlayerStats()
    st.record_game(s, vs_ai=False, winner=PLAYER_2, mode=MODE_POINTS_FULL, move_count=15)
    assert s.games_hotseat == 1
    assert s.games_vs_ai == 0
    assert (s.wins, s.losses, s.draws) == (0, 0, 0)
    assert s.moves_played == 15          # still counts towards time at the board


def test_p08_band_and_mode_tallies():
    s = st.PlayerStats()
    _win(s, band="master")
    st.record_game(s, vs_ai=True, winner=PLAYER_2, mode=MODE_FIRST_TO_FOUR,
                   move_count=9, band="easy")

    assert s.band_tally("master")[st.RESULT_WIN] == 1
    assert s.band_tally("easy")[st.RESULT_LOSS] == 1
    assert s.band_tally("novice") == {"win": 0, "loss": 0, "draw": 0}
    assert s.mode_tally(MODE_POINTS_FULL)[st.RESULT_WIN] == 1
    assert s.mode_tally(MODE_FIRST_TO_FOUR)[st.RESULT_LOSS] == 1


def test_p09_best_band_beaten_prefers_the_hardest():
    s = st.PlayerStats()
    _win(s, band="easy")
    _win(s, band="hard")
    st.record_game(s, vs_ai=True, winner=PLAYER_2, mode=MODE_POINTS_FULL,
                   move_count=8, band="master")
    assert s.best_band_beaten() == "hard"


def test_p10_persistence_round_trip(tmp_path):
    path = tmp_path / "player_stats.json"
    s = _win(st.PlayerStats(), band="medium")
    assert st.save_stats(s, path) is True

    loaded = st.load_stats(path)
    assert loaded.wins == 1
    assert loaded.band_tally("medium")[st.RESULT_WIN] == 1


def test_p11_load_tolerates_missing_and_corrupt_files(tmp_path):
    assert st.load_stats(tmp_path / "absent.json").games_vs_ai == 0

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert st.load_stats(bad).games_vs_ai == 0

    listy = tmp_path / "list.json"
    listy.write_text("[]")
    assert st.load_stats(listy).games_vs_ai == 0


def test_p12_unknown_fields_are_dropped(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"wins": 3, "mystery": True}))
    assert st.load_stats(path).wins == 3


def test_p13_reset_clears_the_file(tmp_path):
    path = tmp_path / "stats.json"
    st.save_stats(_win(st.PlayerStats()), path)
    fresh = st.reset_stats(path)
    assert fresh.wins == 0
    assert st.load_stats(path).wins == 0
