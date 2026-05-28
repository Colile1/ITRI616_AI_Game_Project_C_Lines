# engine/

Pure game logic — no I/O, no side effects, no randomness.

| Module | Responsibility |
|--------|---------------|
| `board.py` | `Board` dataclass, `setup_board`, `clone_board` |
| `scoring.py` | Line counting, `score_board`, `compute_scores` |
| `rules.py` | Placement/removal, terminal detection, tie-break |
