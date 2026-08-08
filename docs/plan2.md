# Plan 2 — Application UI & Feature Upgrade

**Author:** Colile Sibanda
**Date:** 2026-08-07
**Scope:** the *playable application* (`src/ui/`, plus supporting pure logic in `src/game/`).
**Out of scope:** training, network architecture, reward shaping, evaluation pipeline — all of that is frozen (see `MEMORY.md` / `docs/plan.md`).

---

## 1. Why

The engine, training loop and snapshot pipeline are finished and validated. What the marker actually *interacts with* is `python -m src.ui.app`, and that layer is the thinnest part of the project:

* Four settings toggles exist in the menu; **none of them are wired to anything**.
* The AI's move is computed **inside the draw function**, so the window freezes while it thinks and the "Thinking…" label it renders is never visible.
* The default board size in the picker (`10`) is **not in `UI_BOARD_SIZES` (`[8]`)**, so pressing Enter starts a game against a size for which no snapshot exists.
* Nothing that happens in a game is kept: no undo, no saved games, no record of whether you have ever beaten the "Master" snapshot.
* The window is `RESIZABLE`, but the board is drawn at a fixed 56 px cell and always pinned to the top-left.

None of that is hard to fix, and fixing it makes the AI work far more legible — a win-probability read-out and a hint button turn the trained snapshot from "an opponent" into "something you can inspect".

---

## 2. Design rules for this plan

1. **No engine changes.** `engine/`, `training/`, `agents/` stay byte-identical except one additive method on `GameEnv` (`set_state`, needed for undo).
2. **Game logic that is not drawing goes in `src/game/`,** so it is unit-testable without a display. New: `analysis.py`, `records.py`, `stats.py`.
3. **`app.py` orchestrates; it does not draw.** Panels move into their own modules.
4. **Everything degrades.** No snapshots, no saved games, unwritable `results/` — the app must still run.
5. **Update and draw are separate.** Nothing that mutates game state may live inside a `draw()`.

---

## 3. Work items

### A — Correctness fixes (foundation)

| ID | Item |
|----|------|
| A1 | Board-size picker defaults to `UI_BOARD_SIZES[0]`; Enter can never confirm an unlisted size. |
| A2 | Move AI move-selection out of `_draw_ingame()` into an `_update()` phase, with a configurable think-delay so "Thinking…" is actually shown and the window stays responsive. |
| A3 | Result text says "AI Wins!" when you are playing the AI, not "Player 2 Wins!". |
| A4 | Wire the dead settings (`show_legal`, `show_threats`, `reduce_motion`, `sound`) to real behaviour, and persist settings to disk. |
| A5 | Resign no longer fires instantly off a stray `Q` — it goes through a confirm dialog. |

### B — UI improvements

| ID | Item |
|----|------|
| B1 | **Responsive board.** Cell size is derived from the window each frame and clamped to `[MIN_CELL_PX, MAX_CELL_PX]`; the board is centred in the space left of the sidebar. Resizing works properly. |
| B2 | **Motion.** Drop-in animation for a placed piece, pulse on the last move, animated win-line reveal — all disabled by `reduce_motion`. |
| B3 | **Winning-line highlight.** On game over the deciding run is drawn as a bright connected line (FTF: the 4+ run; PTS: the winner's longest run). |
| B4 | **Threat overlay** (`show_threats`): own open-3/open-4 cells tinted green, opponent's red, and empty cells that win-now / lose-now ringed. Uses the `HL_THREAT_*` tokens already in `theme.py`. |
| B5 | **Move-history panel** in the sidebar — algebraic coordinates (`D4`), paired per turn, auto-scrolled to the latest. |
| B6 | **Toasts** — transient bottom-of-board messages ("Undo", "Hint: D4", "Game saved"). |
| B7 | **Keyboard play.** Arrow keys move a board cursor, Enter/Space places. The whole game is playable without a mouse. |
| B8 | **Richer game-over card** — winner, move count, wall-clock duration, per-player score and the line breakdown, plus Save/Replay/New/Menu. |
| B9 | **Real replay controls** — play/pause, step, first/last, a draggable scrub bar and a speed selector, all mouse-clickable. |
| B10 | **Help overlay** (`F1` / `?`) — controls plus the scoring table, available from any screen. |

### C — New features

| ID | Item |
|----|------|
| C1 | **Undo / redo.** `U` / `Ctrl+Z` and `Ctrl+Y`. Versus AI, undo rewinds to your turn (normally two plies); redo replays. |
| C2 | **Hint.** `H` asks for a suggested move — from the loaded snapshot's own Q-values when playing the AI, otherwise from the alpha-beta benchmark agent — and highlights it. |
| C3 | **Position evaluation bar.** A live advantage meter in the sidebar. With a snapshot loaded it shows *the network's own value estimate*; otherwise a heuristic score/threat differential. Recomputed once per ply, not per frame. |
| C4 | **Save / load games.** Finished games are written to `results/games/*.json`; a **Replays** browser on the main menu lists them and opens any of them in the replay viewer. |
| C5 | **Player statistics.** `results/player_stats.json` accumulates W/L/D overall, per difficulty band and per mode, plus current/best win streak and fastest win. A **Stats** screen renders it. |
| C6 | **AI move-time read-out** in the sidebar (last / average seconds per move) — cheap, and it makes the snapshot's cost visible. |
| C7 | **AI speed setting** — instant / fast / normal / slow think-delay, so the AI's move is watchable during a demo. |

---

## 4. New / changed files

```
src/game/analysis.py       NEW  win lines, threat cells, hint search, position eval
src/game/records.py        NEW  GameRecord dataclass + JSON save/list/load
src/game/stats.py          NEW  PlayerStats accumulation + JSON persistence
src/game/env.py            +    set_state() for undo

src/ui/animations.py       NEW  Tween / Timeline helpers
src/ui/toast.py            NEW  transient message queue
src/ui/sidebar.py          NEW  in-game HUD (cards, eval bar, history, buttons)
src/ui/overlays.py         NEW  help overlay, confirm dialog, game-over card
src/ui/replay_browser.py   NEW  saved-game list screen
src/ui/stats_screen.py     NEW  statistics screen
src/ui/settings_store.py   NEW  load/save UI settings JSON

src/ui/theme.py            ~    extra tokens, panel/bar/segmented-control helpers
src/ui/board_view.py       ~    responsive metrics, overlays, animation, cursor
src/ui/replay_view.py      ~    full transport controls
src/ui/menus.py            ~    new menu entries, size-picker fix, more settings
src/ui/app.py              ~    update/draw split, undo/redo, hint, persistence
src/config.py              +    UI + feature constants

tests/test_analysis.py     NEW
tests/test_records.py      NEW
tests/test_stats.py        NEW
tests/test_ui_smoke.py     NEW  headless (SDL dummy) screen-render smoke test
```

---

## 5. Screen map (after)

```
main_menu ─┬─ Play vs AI ──► size_pick ─► mode_pick ─► level_select ─► ingame
           ├─ Hot-seat ────► size_pick ─► mode_pick ──────────────────► ingame
           ├─ Replays ─────► replay_browser ─► replay
           ├─ Stats ───────► stats
           ├─ Settings ────► settings
           └─ Quit

ingame ─► game_over ─┬─ Replay ─► replay
                     ├─ Save
                     ├─ New Game
                     └─ Menu

(help overlay and confirm dialog float above whatever screen is active)
```

---

## 6. Acceptance criteria

* `pytest` green, including the new analysis/records/stats tests.
* A headless smoke test renders every screen with `SDL_VIDEODRIVER=dummy` without raising.
* Manual: resize the window mid-game — board rescales and stays centred, clicks still land on the right cell.
* Manual: every settings toggle visibly changes behaviour.
* Manual: undo/redo, hint, save, replay-from-browser and the stats screen all work with real snapshots from `models/size_08/`.

---

## 7. Outcome

**Status: implemented.** All of A1–A5, B1–B10 and C1–C7 are in. 177 unit tests pass, including 52 new headless UI tests that render every screen and drive every event path under `SDL_VIDEODRIVER=dummy`. The integration suite passes too but is slow for reasons unrelated to this work: one `alphabeta_d4` benchmark check plays 128 search-heavy games, so `test_T04_registry_populated` alone takes 28 minutes.

### Two things found during the work that were not in the plan

**1. Every trained snapshot was silently loading as a random opponent.**
`load_snapshot()` read the architecture from the *registry* entry, but `registry._meta_to_entry()` never wrote `state_channels` or `network_arch`. So every snapshot was rebuilt as a 6-channel `plain_v1` network, the 10-channel `resnet_v1` weights failed to load, and the exception handler in the UI fell back to `RandomAgent`. All 84 registered 8×8 snapshots were affected, and the old UI swallowed the failure without a word — you were playing noise while the card said "Master".

Fixed in three parts:
* `snapshot.load_snapshot()` now prefers the per-snapshot `metadata.json`, which has always carried the correct fields — so all existing snapshots work with no migration.
* `registry._meta_to_entry()` / `_entry_to_meta()` now round-trip both fields, so new registrations are complete.
* Loading also falls back to the derived snapshot directory when the recorded absolute `weights_path` does not resolve, which makes the models tree portable.

Covered by `tests/test_snapshot_load.py` (S-01 … S-05), including a test that strips the fields back out of the registry to reproduce the original on-disk shape.

**2. Level select was unusable at 84 snapshots.** Cards were ordered by run id, carried no provenance, and the same `gen_NNN` appears in every run — so a dozen cards read identically. It now sorts weakest-first (band, then win rate, then Elo, then games), labels each card with its source run, and offers a difficulty-band filter with counts.

### Also fixed while testing
* `records._prune()` bound `MAX_SAVED_GAMES` as a default argument, so the cap could never be overridden.
* Undo-then-redo past the end double-counted a game in statistics and saved a second copy. A game is now recorded once per `_start_game()`; undo deliberately does not clear that, so taking a move back cannot inflate your record.
* Decorative glyphs (`↶ ◎ ✕ ⚙ ▶ ❚❚`) do not exist in Arial and rendered as tofu boxes. `theme.glyph()` probes by rendering and comparing against an undefined codepoint — `Font.metrics()` is no help, it reports metrics for the box itself — and substitutes a glyph the font actually has.

### Found by the code review and fixed

* Sidebar buttons were hit-tested without consulting the enabled state they were drawn with, so a greyed-out Undo still fired, and Settings/Main Menu stayed live underneath the result card. `sidebar.action_enabled()` is now the single source of truth for both drawing and dispatch, fed by one shared `App._sidebar_state()`.
* The Save button on a re-finished game looked live but silently no-opped; the card now carries the real `saved` flag.
* The evaluation meter captioned itself "network" and hints claimed "(agent)" even when the RandomAgent fallback was in play or the snapshot's `q_values` had failed. `suggest_move_detailed()` now returns its provenance and `App._has_network()` gates the caption.
* `ReplayViewer` replayed saved JSON without bounds-checking, so a hand-edited record raised out of the constructor. It now truncates at the last coherent move.
* The replay browser's double-click used one module-global timestamp with no row identity, so two quick clicks on different rows opened the second game.
* ESC quit the app outright from the main menu, against the ESC-means-back convention everywhere else.
* `critical_cells()` probes by writing into the caller's grid; the restore is now in a `try/finally`.

Each of these has regression cover in `tests/test_ui_smoke.py` or `tests/test_analysis.py`.

---

## 8. Deliberately not done

* Sound assets — the toggle exists and is honoured by the code path, but no audio files ship with the project.
* Online play, accounts, animation of the tie-break removal sequence.
* Board sizes other than 8 in the UI: `UI_BOARD_SIZES` stays `[8]` because that is the only size with trained snapshots.
