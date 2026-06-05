# src/engine/

Pure game logic. No ML dependencies. All functions are stateless — they take a `Board` and return results without modifying it.

| File | Contents |
|------|----------|
| `board.py` | `Board` dataclass; `setup_board()` |
| `rules.py` | `apply_placement()`, `is_legal_placement()`, `check_terminal_mode1/2()` |
| `scoring.py` | `compute_scores()`, `compute_threats()`, `compute_double_threats()` |

### Scoring functions
- `compute_scores(board)` -> `(p1_score, p2_score)` for points-full mode
- `compute_threats(board)` -> `(p1_open3, p2_open3)` — open-3 threat counts used in FTF reward shaping
- `compute_double_threats(board)` -> `(p1_dbl, p2_dbl)` — forced-win threats (3-in-a-row with both ends free)

These are called every game step; they use Python loops over the board grid so they are the main simulation bottleneck. Safe to call speculatively (e.g. in alpha-beta search) as they have no side effects.
