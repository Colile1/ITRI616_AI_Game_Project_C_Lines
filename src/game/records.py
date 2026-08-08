"""Saved-game records — a finished game serialised to JSON so it can be
re-opened in the replay viewer later.

The format is deliberately flat and human-readable: a marker can open one of
these files and read the whole game off it.  Every read path is tolerant of
missing or malformed files; a corrupt record is skipped, never fatal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import GAMES_DIR, MAX_SAVED_GAMES, MODE_POINTS_FULL
from src.game.analysis import algebraic

RECORD_SCHEMA = 1


@dataclass
class GameRecord:
    """One completed (or abandoned) game."""

    board_size: int
    mode: str = MODE_POINTS_FULL
    moves: list[list[int]] = field(default_factory=list)   # [[player, row, col], ...]

    vs_ai: bool = False
    ai_version_id: Optional[str] = None
    ai_run_id: Optional[str] = None
    ai_name: Optional[str] = None
    ai_band: Optional[str] = None

    winner: Optional[int] = None          # 1, 2, or None for a draw
    p1_score: float = 0.0
    p2_score: float = 0.0
    resigned_by: Optional[int] = None

    started_at: str = ""
    finished_at: str = ""
    duration_sec: float = 0.0

    schema: int = RECORD_SCHEMA

    # ------------------------------------------------------------------
    @property
    def move_count(self) -> int:
        return len(self.moves)

    @property
    def move_tuples(self) -> list[tuple[int, int, int]]:
        """Moves as (player, row, col) tuples — what ReplayViewer expects."""
        return [(int(p), int(r), int(c)) for p, r, c in self.moves]

    @property
    def opponent_label(self) -> str:
        if not self.vs_ai:
            return "Hot-seat"
        return self.ai_name or self.ai_version_id or "AI"

    @property
    def result_label(self) -> str:
        if self.winner is None:
            return "Draw"
        if self.vs_ai:
            return "Win" if self.winner == 1 else "Loss"
        return f"P{self.winner}"

    def summary_line(self) -> str:
        when = self.started_at[:16].replace("T", " ")
        return f"{when}  ·  {self.result_label}  ·  {self.move_count} moves"

    def notation(self, limit: Optional[int] = None) -> str:
        """Moves in the same algebraic form the board labels use."""
        cells = [algebraic(int(r), int(c)) for _, r, c in self.moves]
        if limit is not None and len(cells) > limit:
            cells = cells[:limit] + ["…"]
        return " ".join(cells)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _sanitise(d: dict) -> dict:
    """Keep only known fields — lets old records load after a schema change."""
    known = set(GameRecord.__dataclass_fields__)
    return {k: v for k, v in d.items() if k in known}


def record_path(directory: Path = GAMES_DIR, when: Optional[datetime] = None) -> Path:
    when = when or datetime.now()
    stamp = when.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return Path(directory) / f"game_{stamp}.json"


def save_record(record: GameRecord, directory: Path = GAMES_DIR) -> Optional[Path]:
    """Write *record* to *directory*, pruning old files.  None if the write failed."""
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = record_path(directory)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(record), indent=2))
        os.replace(tmp, path)
    except OSError:
        return None

    _prune(directory)
    return path


def _prune(directory: Path, keep: int | None = None) -> None:
    # Read the module global at call time, not at def time, so the cap stays
    # overridable (tests do exactly this).
    keep = MAX_SAVED_GAMES if keep is None else keep
    try:
        files = sorted(Path(directory).glob("game_*.json"))
    except OSError:
        return
    if len(files) <= keep:
        return
    for stale in files[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass


def load_record(path: Path) -> Optional[GameRecord]:
    """Read one record.  None if the file is missing or unparseable."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return GameRecord(**_sanitise(data))
    except TypeError:
        return None


def list_records(directory: Path = GAMES_DIR) -> list[tuple[Path, GameRecord]]:
    """All readable records, newest first.  Empty list if the folder is absent."""
    directory = Path(directory)
    if not directory.exists():
        return []
    out: list[tuple[Path, GameRecord]] = []
    try:
        paths = sorted(directory.glob("game_*.json"), reverse=True)
    except OSError:
        return []
    for path in paths:
        rec = load_record(path)
        if rec is not None:
            out.append((path, rec))
    return out


def delete_record(path: Path) -> bool:
    try:
        Path(path).unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Construction helper
# ---------------------------------------------------------------------------

def build_record(
    board_size: int,
    mode: str,
    moves: list[tuple[int, int, int]],
    winner: Optional[int],
    p1_score: float,
    p2_score: float,
    vs_ai: bool,
    started_at: datetime,
    ai_version_id: Optional[str] = None,
    ai_run_id: Optional[str] = None,
    ai_name: Optional[str] = None,
    ai_band: Optional[str] = None,
    resigned_by: Optional[int] = None,
) -> GameRecord:
    finished = datetime.now(timezone.utc)
    started = started_at if started_at.tzinfo else started_at.replace(tzinfo=timezone.utc)
    return GameRecord(
        board_size=board_size,
        mode=mode,
        moves=[[int(p), int(r), int(c)] for p, r, c in moves],
        vs_ai=vs_ai,
        ai_version_id=ai_version_id,
        ai_run_id=ai_run_id,
        ai_name=ai_name,
        ai_band=ai_band,
        winner=winner,
        p1_score=float(p1_score),
        p2_score=float(p2_score),
        resigned_by=resigned_by,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        duration_sec=max(0.0, (finished - started).total_seconds()),
    )
