"""G-01 .. G-12 — saved-game records and UI settings persistence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.config import MODE_FIRST_TO_FOUR, MODE_POINTS_FULL, DEFAULT_UI_SETTINGS
from src.game import records
from src.ui.settings_store import load_settings, save_settings

MOVES = [(1, 0, 0), (2, 1, 1), (1, 0, 1), (2, 2, 2)]


def _record(**overrides) -> records.GameRecord:
    base = dict(
        board_size=8,
        mode=MODE_POINTS_FULL,
        moves=[[p, r, c] for p, r, c in MOVES],
        winner=1,
        p1_score=1.0,
        p2_score=0.25,
        vs_ai=True,
        ai_name="Master (master)",
        ai_band="master",
        started_at="2026-08-07T10:00:00+00:00",
    )
    base.update(overrides)
    return records.GameRecord(**base)


# ---------------------------------------------------------------------------

def test_g01_round_trip(tmp_path):
    path = records.save_record(_record(), tmp_path)
    assert path is not None and path.exists()

    loaded = records.load_record(path)
    assert loaded is not None
    assert loaded.move_tuples == MOVES
    assert loaded.board_size == 8
    assert loaded.ai_band == "master"


def test_g02_list_records_is_newest_first(tmp_path):
    for i in range(3):
        records.save_record(_record(p1_score=float(i)), tmp_path)
    rows = records.list_records(tmp_path)
    assert len(rows) == 3
    assert [r.p1_score for _, r in rows] == [2.0, 1.0, 0.0]


def test_g03_list_records_on_missing_dir(tmp_path):
    assert records.list_records(tmp_path / "nope") == []


def test_g04_corrupt_files_are_skipped_not_fatal(tmp_path):
    records.save_record(_record(), tmp_path)
    (tmp_path / "game_broken.json").write_text("{not json")
    (tmp_path / "game_list.json").write_text("[1, 2, 3]")

    rows = records.list_records(tmp_path)
    assert len(rows) == 1


def test_g05_unknown_fields_are_dropped(tmp_path):
    path = tmp_path / "game_future.json"
    path.write_text(json.dumps({
        "board_size": 8, "mode": MODE_POINTS_FULL, "moves": [], "quantum_flux": 7,
    }))
    rec = records.load_record(path)
    assert rec is not None and rec.board_size == 8


def test_g06_prune_keeps_the_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "MAX_SAVED_GAMES", 3)
    for i in range(6):
        records.save_record(_record(p1_score=float(i)), tmp_path)
    rows = records.list_records(tmp_path)
    assert len(rows) == 3
    assert [r.p1_score for _, r in rows] == [5.0, 4.0, 3.0]


def test_g07_delete_record(tmp_path):
    path = records.save_record(_record(), tmp_path)
    assert records.delete_record(path) is True
    assert records.list_records(tmp_path) == []
    assert records.delete_record(path) is False


def test_g08_labels_and_notation():
    rec = _record()
    assert rec.move_count == 4
    assert rec.opponent_label == "Master (master)"
    assert rec.result_label == "Win"
    assert rec.notation() == "A1 B2 B1 C3"
    assert rec.notation(limit=2) == "A1 B2 …"

    assert _record(winner=2).result_label == "Loss"
    assert _record(winner=None).result_label == "Draw"
    assert _record(vs_ai=False, winner=2).result_label == "P2"
    assert _record(vs_ai=False).opponent_label == "Hot-seat"


def test_g09_build_record_computes_duration():
    started = datetime.now(timezone.utc) - timedelta(seconds=90)
    rec = records.build_record(
        board_size=8, mode=MODE_FIRST_TO_FOUR, moves=MOVES, winner=1,
        p1_score=1.0, p2_score=0.0, vs_ai=False, started_at=started,
    )
    assert 89 <= rec.duration_sec <= 95
    assert rec.moves == [[1, 0, 0], [2, 1, 1], [1, 0, 1], [2, 2, 2]]
    assert rec.finished_at


def test_g10_build_record_accepts_naive_start_time():
    rec = records.build_record(
        board_size=8, mode=MODE_POINTS_FULL, moves=[], winner=None,
        p1_score=0.0, p2_score=0.0, vs_ai=False,
        started_at=datetime.now(),          # no tzinfo
    )
    assert rec.duration_sec >= 0.0


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_g11_settings_round_trip(tmp_path):
    path = tmp_path / "ui_settings.json"
    settings = dict(DEFAULT_UI_SETTINGS)
    settings["show_threats"] = True
    settings["ai_speed"] = "slow"

    assert save_settings(settings, path) is True
    assert load_settings(path)["show_threats"] is True
    assert load_settings(path)["ai_speed"] == "slow"


def test_g12_settings_reject_bad_values_and_fill_gaps(tmp_path):
    path = tmp_path / "ui_settings.json"
    path.write_text(json.dumps({
        "show_legal": "yes",       # wrong type — ignored
        "ai_speed": "warp",        # not a known speed — ignored
        "unknown_key": 1,          # dropped
        "show_threats": True,      # kept
    }))
    loaded = load_settings(path)
    assert loaded["show_legal"] is DEFAULT_UI_SETTINGS["show_legal"]
    assert loaded["ai_speed"] == DEFAULT_UI_SETTINGS["ai_speed"]
    assert loaded["show_threats"] is True
    assert "unknown_key" not in loaded
    assert set(loaded) == set(DEFAULT_UI_SETTINGS)


def test_g13_settings_missing_file_gives_defaults(tmp_path):
    assert load_settings(tmp_path / "absent.json") == DEFAULT_UI_SETTINGS
