# C_lines — Agent Versioning and Snapshot Specification

**Author:** *----* *S----*
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Schema locked; implementation pending Phase 8

---

## 1. Purpose

C_lines requires the agent to be frozen at points in training, stored immutably, exposed to the user as selectable difficulty levels in single-player mode, and never to train further once registered. This document specifies the on-disk layout, metadata schema, registry format, and runtime contract that the versioning system honours.

## 2. On-disk layout

```
models/
├── registry.json                # global index across all board sizes
├── size_08/
│   ├── gen_001/
│   │   ├── weights.pt           # torch.save(state_dict)
│   │   ├── metadata.json        # SnapshotMetadata serialised
│   │   └── training_history.json  # per-snapshot training-log slice
│   ├── gen_002/
│   │   └── ...
│   └── ...
├── size_09/
│   └── ...
├── size_10/
│   └── ...
├── size_11/
│   └── ...
└── size_12/
    └── ...
```

The board size is part of the directory tree, not the snapshot ID, so two snapshots can both be `gen_007` on different board sizes without colliding. Within a board size, generation IDs are zero-padded to three digits and strictly monotonic.

## 3. SnapshotMetadata schema

The metadata is a dataclass in `src/versioning/metadata.py`. The on-disk JSON is the same shape.

```python
@dataclass
class SnapshotMetadata:
    # Identity
    version_id        : str       # e.g. "gen_007"
    board_size        : int       # 8 .. 12
    weights_path      : str       # relative path from project root

    # Training provenance
    created_at        : str       # ISO-8601 UTC, e.g. "2026-05-28T14:32:01Z"
    games_trained     : int       # cumulative games at freeze time
    gradient_steps    : int       # cumulative optimiser steps at freeze time
    epsilon_at_freeze : float     # exploration rate when frozen
    parent_run_id     : str       # uuid4 of the training run this came from
    parent_version_id : str | None  # previous snapshot in same run, or None

    # Performance at freeze time
    win_rate_vs_random    : float | None    # over EVAL_GAMES games
    win_rate_vs_heuristic : float | None
    win_rate_vs_self      : float | None    # vs the previous snapshot
    elo_rating            : float | None    # within the snapshot pool
    mean_episode_length   : float | None

    # Training history summary (compact)
    training_history      : list[TrainingHistoryEntry]

    # Presentation
    friendly_name     : str       # e.g. "Apprentice" / "Tactician" / "Master"
    difficulty_band   : str       # one of: novice | easy | medium | hard | master
    notes             : str       # free-text annotation
```

`TrainingHistoryEntry` is a compact per-checkpoint record:

```python
@dataclass
class TrainingHistoryEntry:
    games_trained         : int
    win_rate_vs_random    : float
    mean_reward           : float
    mean_loss             : float
```

The list contains all snapshots' history entries up to and including this one, so a single snapshot's metadata is self-contained — you can plot its full learning curve from `metadata.json` alone without re-reading the global log.

## 4. Registry format

`models/registry.json` is the global index. It is a single JSON document with one top-level key per board size:

```json
{
  "size_08": [
    { "version_id": "gen_001", "weights_path": "models/size_08/gen_001/weights.pt",
      "friendly_name": "Apprentice", "difficulty_band": "novice",
      "games_trained": 1000, "win_rate_vs_random": 0.62,
      "elo_rating": 950, "created_at": "2026-05-28T12:00:00Z" },
    { "version_id": "gen_002", ... }
  ],
  "size_09": [ ... ],
  "size_10": [ ... ],
  "size_11": [ ... ],
  "size_12": [ ... ]
}
```

Each entry stores the *display-relevant* subset of `SnapshotMetadata` so the level-select screen does not need to open every per-snapshot `metadata.json`. The full metadata is always reachable by loading `models/<size_NN>/<version_id>/metadata.json`.

Concurrency note: the registry is rewritten atomically — write to `registry.json.tmp`, then rename to `registry.json` — so partial writes never leave the index corrupt.

## 5. Runtime contract — `versioning/registry.py`

```python
def register(meta: SnapshotMetadata) -> None:
    """Append meta to registry.json under the right board-size bucket. Atomic."""

def list_by_size(board_size: int) -> list[SnapshotMetadata]:
    """Return all snapshots for board_size, sorted by games_trained ascending."""

def get(board_size: int, version_id: str) -> SnapshotMetadata:
    """Load full SnapshotMetadata for the given snapshot. Raises if missing."""

def evict_oldest(board_size: int, keep_n: int) -> list[str]:
    """Remove all but the newest keep_n snapshots for the size. Returns evicted ids."""
```

The level-select screen uses `list_by_size`. The training loop uses `register` and (optionally) `evict_oldest`. `get` is used by both to load weights and metadata together.

## 6. Loading a snapshot for play

```python
def load_snapshot(board_size: int, version_id: str) -> DQNAgent:
    """Return a fully-loaded, frozen DQNAgent."""
    meta = registry.get(board_size, version_id)
    agent = DQNAgent(board_size=board_size, eval_only=True)
    agent.load_state_dict(torch.load(meta.weights_path, map_location="cpu"))
    agent.eval()
    agent.set_epsilon(0.0)
    return agent
```

`eval_only=True` instantiates the agent without an optimiser, replay buffer, or target network — none are needed at inference time. `set_epsilon(0.0)` ensures the loaded snapshot plays greedily; it does **not** explore.

## 7. Immutability rule

Once a snapshot is registered, its files are never re-written. The training loop only ever creates new gen_NNN folders; existing folders are read-only by convention. The implementation enforces this by:

* Constructing the gen_NNN directory only after picking the next sequential integer (`max(existing) + 1`), so collisions are impossible.
* Refusing to overwrite an existing `weights.pt` or `metadata.json` (`open(..., "x")` instead of `"w"`).
* Treating `models/` as untouched by any code path outside `versioning/`.

If a previously-trained snapshot becomes a problem (corrupt, mislabelled), it can be deleted manually, but the system never silently mutates a registered snapshot.

## 8. Pool sampling during training

The training loop maintains a `snapshot_pool` — a Python list of recently-frozen metadata. Pool sampling rules:

* `MAX_POOL_SIZE` caps the pool at 20 snapshots (configurable).
* On overflow, evict the **oldest** (FIFO) — this keeps the pool biased toward stronger recent opponents while retaining a long tail of generational diversity.
* Each self-play episode draws one opponent uniformly from the pool with probability `SELF_PLAY_MIX_PROB` (default 0.5); otherwise plays against the current agent.

The pool used for *training* is distinct from the registry's *presentation* list. The registry retains all snapshots forever (subject to disk space), so the player always sees the full learning timeline in the level-select screen.

## 9. Difficulty banding for the player UI

The UI presents snapshots as named difficulty bands rather than raw generation IDs. The mapping is heuristic, set at registration time:

| Band     | Trigger                                              | Friendly name examples |
|----------|------------------------------------------------------|------------------------|
| novice   | first 25% of TRAINING_GAMES                          | Apprentice, Beginner   |
| easy     | 25% to 50%                                           | Improver, Cadet        |
| medium   | 50% to 75%                                           | Tactician, Adept       |
| hard     | 75% to 100% (but not the most recent)                | Strategist, Veteran    |
| master   | the single most-recent snapshot at end of training   | Master, Champion       |

The friendly_name is chosen from a small pool per band to make the level-select feel less mechanical. The banding can be regenerated by re-scanning the registry; it is not load-bearing.

## 10. What "training history" means in the metadata

`training_history` inside each snapshot is the per-checkpoint timeline up to and including that snapshot. This means:

* `gen_001.metadata.json` has a `training_history` list of length 1.
* `gen_002.metadata.json` has a list of length 2 — the snapshots that preceded it plus itself.
* `gen_010.metadata.json` has a list of length 10.

This redundancy is intentional. Any single snapshot can render its own learning curve without depending on anything else. The cost is small — each entry is ~80 bytes — and the benefit is that snapshots remain meaningful even if `registry.json` is lost or the training run is partially replayed.

## 11. Storage budget

Per snapshot, on disk:

```
weights.pt          ~ 10 MB    (2.4M float32 params for 12x12)
metadata.json       ~ 5 KB
training_history    ~ N × 80 bytes
Total per snapshot  ~ 10 MB
```

At `SNAPSHOT_INTERVAL = 1000` and `TRAINING_GAMES = 10000` per size, that is 10 snapshots × 5 sizes = 50 snapshots × 10 MB = **500 MB**. Comfortable for a student laptop; the user can also drop `MAX_POOL_SIZE` and `SNAPSHOT_INTERVAL` to halve this if needed.

## 12. Migration / version compatibility

If the network architecture changes (e.g. an extra input channel is added), older snapshots become unloadable. The recipe:

* Bump `MODEL_VERSION` constant in `config.py` from `1` to `2`.
* Stamp `model_version` into every `SnapshotMetadata` going forward.
* On load, `versioning.get` raises `IncompatibleSnapshotError` if `meta.model_version != MODEL_VERSION` — the UI surfaces this clearly ("This level was trained with an older agent and cannot be loaded.").
* Optionally provide a `--migrate-from N` flag on the training script that uses an old snapshot as a warm start for a new run, but does not re-register the old snapshot under the new architecture.

`model_version` lives in metadata, not on the registry side, so partial migrations leave the older snapshots cleanly inaccessible without rewriting any older files.

## 13. Quick API summary

```python
from src.versioning import registry, metadata, snapshot

# during training
snapshot.freeze(dqn_agent, games_trained, eval_stats)   # writes files + registers

# during UI launch
levels = registry.list_by_size(board_size=10)           # for level-select screen

# during gameplay
opponent = snapshot.load_snapshot(board_size=10, version_id="gen_007")
action_idx = opponent.select_action(obs, legal_mask)
```

---

*End of versioning specification.*
