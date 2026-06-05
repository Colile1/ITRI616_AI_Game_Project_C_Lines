# Game Design Skeleton — AI Board Game (ITRI616 Template)

**Author:** *----* *S----*  
**Purpose:** Reusable prompt attachment for building new AI board games in the Masimo stack.  
**How to use:** Attach this file to a new conversation, then describe only what is *different*
about the new game (rules, board, actions). Everything else — architecture, conventions,
AI design, build order — is already specified here. The LLM should follow this skeleton
exactly unless you explicitly override a section.

---

## 1. Project Identity (fill in for each new game)

```
Game name        : _______________
Short description: _______________   (1 sentence)
Board type       : _______________   (grid / graph / linear / custom)
Grid size        : _______________   (e.g. 8×8, or variable)
Number of players: _______________   (usually 2)
Turn structure   : _______________   (alternating / simultaneous)
Information      : _______________   (perfect / imperfect)
Originality note : _______________   (invented / variant of ___ / traditional ___)
```

---

## 2. Game Rules (fill in for each new game)

Describe each section clearly. The engine will be built directly from this spec.

### 2.1 Board Setup
- What the board looks like at game start
- Which cells/nodes are owned by whom, how many pieces/seeds/tokens

### 2.2 On Your Turn — Legal Actions
List every distinct action type a player may take. For each:
```
Action name : _______________
Precondition: _______________   (what must be true to use it)
Effect      : _______________   (what changes on the board)
```

### 2.3 Interaction / Capture Rules
- What happens when pieces meet (erode, capture, convert, bounce, etc.)

### 2.4 Terminal Condition
- When does the game end? (no legal moves / turn cap / piece count / score target)
- What determines the winner? (score formula, piece count, territory, etc.)
- Tie-breaking rule (e.g. first player wins ties — prefer **no draws**)

### 2.5 Tunable Parameters
List every value that should live in `config.py`:
```
BOARD_SIZE      = ___
TURN_CAP        = ___
MIN_X           = ___   # describe what X is
CONVERT_ON_Y    = ___   # True/False mechanic flags
```

---

## 3. Standard Project Structure

Every game uses this exact directory layout. Do not deviate.

```
project_root/
├── src/
│   ├── __init__.py
│   ├── config.py               ← ALL tunable parameters. Single source of truth.
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── board.py            ← Board dataclass, setup_board(), clone_board()
│   │   └── rules.py            ← PURE functions only: legal actions, apply actions,
│   │                              is_terminal(), get_winner()
│   ├── game/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── encoding.py         ← state_to_tensor(), action_to_index(), index_to_action(),
│   │   │                          build_legal_mask()
│   │   └── env.py              ← GameEnv: reset(), step(), legal_mask()
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── base_agent.py       ← BaseAgent ABC: select_action(obs, legal_mask) → int
│   │   ├── random_agent.py     ← uniform random over legal actions (baseline floor)
│   │   ├── heuristic_agent.py  ← hand-coded greedy policy (second baseline)
│   │   └── dqn_agent.py        ← epsilon-greedy DQN wrapper
│   ├── training/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── network.py          ← DQNNetwork (CNN or MLP depending on state shape)
│   │   ├── replay_buffer.py    ← ReplayBuffer: push(), sample(), __len__()
│   │   ├── self_play.py        ← run_episode(), self_play_loop()
│   │   ├── train.py            ← get_training_config(), train(), main()
│   │   └── snapshot.py         ← freeze(), load_snapshot(), clone_agent()
│   ├── versioning/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── metadata.py         ← SnapshotMetadata dataclass, save(), load()
│   │   └── registry.py         ← models/registry.json: register, list, get
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── evaluator.py        ← evaluate(agent, opponent, n_games) → dict
│   │   ├── elo.py              ← update_elo(), expected_score()
│   │   └── plots.py            ← generate_all_plots() → 5 PNGs in results/figures/
│   └── ui/
│       ├── __init__.py
│       ├── README.md
│       ├── app.py              ← Pygame main loop, entry point
│       ├── board_view.py       ← render_board(), render_sidebar()
│       ├── menus.py            ← main_menu() → (mode, board_size)
│       └── level_select.py     ← level_select_screen() → SnapshotMetadata
├── tests/
│   ├── __init__.py
│   ├── test_rules.py
│   ├── test_env.py
│   ├── test_agent.py
│   └── test_snapshot.py
├── models/
│   └── registry.json           ← [] initially
├── results/
│   ├── figures/                ← 6 PNG plots generated after training
│   └── logs/                   ← training_log.csv
├── docs/
│   ├── plan.md
│   ├── todo.md
│   ├── log.md
│   ├── report.md               ← fill with actual results
│   ├── starter_guide.md
│   └── tep_definitions.md
├── requirements.txt
└── README.md
```

---

## 4. Python Coding Rules (non-negotiable)

These apply to every file in the project.

### Structure
- Every package has `__init__.py` + `README.md`
- `config.py` is the only place constants live — never hardcode values elsewhere
- `main()` in entry-point files orchestrates only — no logic
- No code executed at import time

### File limits
- Soft limit: **120 lines per file**. Split by responsibility if exceeded.

### Function types (keep strictly separated)
| Type | Naming | Rule |
|------|--------|------|
| Pure function | `calculate_x`, `apply_x`, `get_winner` | No I/O, deterministic, no side effects |
| Input wrapper | `get_calculate_x` | Collects input, calls pure function |
| Output handler | `display_x`, `render_x` | Handles printing/drawing only |

### Rules for `engine/rules.py` specifically
- **Zero I/O** — no `print`, no `input`, no file access
- Every function returns a new board (pure); never mutates the input board
- Use `clone_board()` at the top of every `apply_*` function
- `_land_seed(board, r, c, player)` is the single shared helper for placing one piece

### Naming
- `snake_case` everywhere
- Private helpers: `_underscore_prefix`
- Dataclasses: `PascalCase`

### Comments
- File-level docstring required (one line describing purpose)
- Every public function: one-line docstring
- No multi-line comment blocks; no "what the code does" comments

---

## 5. AI Design Pattern (DQN Self-Play)

### Algorithm
Deep Q-Network (DQN) with:
- Experience replay buffer (i.i.d. sampling)
- Target network (periodically synced copy)
- **Action masking** — illegal actions set to −∞ before argmax
- Epsilon-greedy exploration (linearly decaying)

### State Encoding
Encode board as a multi-channel float32 tensor `(C, H, W)` or `(C, N)` for non-grid games.

**Standard channels (adapt to your game):**
```
ch 0  : current player's piece counts / normaliser
ch 1  : opponent's piece counts / normaliser
ch 2  : current player's locked/permanent territory (binary)
ch 3  : opponent's locked/permanent territory (binary)
ch 4  : empty / neutral cells (binary)
ch 5+ : add more as needed for game-specific features
```
Always normalise continuous values to [0, 1].

### Action Space
```
Total actions  = (number of action types) × (number of source positions) × (number of parameters)
```
- Enumerate every possible (type, position, param) triple into a flat integer index
- `action_to_index(action_tuple) → int`
- `index_to_action(int) → action_tuple`
- Build the legal mask in `build_legal_mask(board, player) → np.ndarray(bool, shape=(ACTION_SPACE_SIZE,))`

### Reward Function
```
Per step  : Δ(my_score − opp_score) × STEP_REWARD_SCALE     (use 0.05 default)
Terminal  : +1.0 win  |  −1.0 loss  |  0.0 draw (avoid draws by design)
Optional  : small +0.01 per capture (off by default, add to config)
```

### Network Architecture
**For grid boards (H×W):**
```
Conv2d(C_in→32, 3×3, pad=1) → ReLU
Conv2d(32→64,  3×3, pad=1) → ReLU
Conv2d(64→64,  3×3, pad=1) → ReLU
Flatten → Linear(64·H·W, 256) → ReLU → Linear(256, ACTION_SPACE_SIZE)
```
**For non-grid boards (N nodes):**
```
Linear(N·C, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, ACTION_SPACE_SIZE)
```

### Self-Play Curriculum
1. **Warm-up** (`WARMUP_GAMES` games): DQN vs `RandomAgent`
2. **Self-play** (remaining games): DQN vs current copy AND past frozen snapshots (version pool)

### Standard Hyperparameters (put all in `config.py`)
```python
LR               = 1e-3
GAMMA            = 0.99
REPLAY_CAPACITY  = 100_000
BATCH_SIZE       = 64
TARGET_SYNC_STEPS= 1_000
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY_GAMES  = 5_000
SNAPSHOT_INTERVAL= 1_000
TRAINING_GAMES   = 10_000
WARMUP_GAMES     = 1_000
STEP_REWARD_SCALE= 0.05
WIN_REWARD       = 1.0
LOSS_REWARD      = -1.0
```

---

## 6. TEP Formulation (Mitchell Framework)

Fill this in for every new game. Required for the ITRI616 report.

```
Task (T):
  State space S   : describe all possible board configurations
  Action space A  : enumerate legal action types
  Transition δ    : describe how board changes after each action
  Task category   : sequential decision-making (MDP)
  Cardinality est.: |S| ≈ ___ (rough upper bound, show working)
  |A|             : exact number of network outputs

Experience (E):
  Type            : self-play simulated episodes
  Opponents faced : random (warm-up) → current + snapshot pool (self-play)
  Storage         : replay buffer — (s_t, a_t, r_t, s_{t+1}, done) transitions
  Feedback type   : indirect / delayed (win/loss at episode end)

Performance (P):
  Primary metric  : win rate W(k) vs random agent at checkpoint k
  Secondary       : Elo rating, average reward per episode, episode length
  Hypothesis      : P improves monotonically (in trend) with E
```

---

## 7. Environment Contract (`game/env.py`)

The `GameEnv` class must satisfy this interface exactly:

```python
class GameEnv:
    def reset(self) -> np.ndarray:
        """Return initial obs for player 1. Apply any opening mechanics."""

    def legal_mask(self) -> np.ndarray:
        """Return bool mask shape (ACTION_SPACE_SIZE,) for current player."""

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        """Apply action; return (obs, reward, done, info).
        Info must contain 'winner' key (int or None) when done=True.
        After switching player: apply any turn-start mechanics (explosions etc.)
        then check if new player has legal moves; if not, set done=True.
        """
```

---

## 8. Standard Build Order

Follow these phases in sequence. Do not skip ahead.

| Phase | What to build | Acceptance test |
|-------|--------------|-----------------|
| 1 | `config.py` + `engine/board.py` | `setup_board()` returns correct seed counts |
| 2 | `engine/rules.py` | All `test_rules.py` tests pass |
| 3 | `game/encoding.py` + `game/env.py` | All `test_env.py` tests pass |
| 4 | `agents/` (base, random, heuristic) | All `test_agent.py` tests pass |
| 5 | `training/network.py` + `replay_buffer.py` | Forward pass returns correct shape |
| 6 | `agents/dqn_agent.py` | Legal action always selected |
| 7 | `training/self_play.py` + `train.py` + `snapshot.py` | `train --games 10` exits cleanly |
| 8 | `versioning/` | snapshot round-trip: `test_snapshot.py` passes |
| 9 | `evaluation/` | evaluate() returns dict with win_rate key |
| 10 | `ui/` | Window opens; hotseat game playable |
| 11 | Full training run | `training_log.csv` populated; plots generated |

---

## 9. Required Test Coverage

Every game must ship with these four test files. Minimum test cases per file:

### `test_rules.py`
- Setup: total piece count correct
- Setup: player ownership correct
- `legal_*_actions` returns only valid positions
- `apply_*` empties source correctly
- `apply_*` distributes pieces correctly
- Capture / interaction mechanic works
- `is_terminal` fires at turn cap
- `get_winner` returns correct player with clear lead

### `test_env.py`
- `reset()` returns correct tensor shape and dtype
- `legal_mask()` shape is `(ACTION_SPACE_SIZE,)` and has at least one True at start
- `step()` on legal action does not raise
- Turn advances after step
- Player alternates after step
- Game terminates within turn cap

### `test_agent.py`
- `RandomAgent` always selects a legal action
- `RandomAgent` varies choices (not always same action)
- `DQNAgent` (epsilon=0) selects legal action
- `DQNAgent` (epsilon=1) selects legal action
- `set_epsilon()` updates epsilon

### `test_snapshot.py`
- `freeze()` creates a `.pt` file
- `load_snapshot()` produces legal action on fresh obs
- Q-values from loaded snapshot match original (atol 1e-5)
- `clone_agent()` Q-values match original

---

## 10. Required Output Figures

After training, `plots.py` must generate all five PNGs in `results/figures/`:

| File | X axis | Y axis | Key line |
|------|--------|--------|----------|
| `win_rate.png` | Training game | Win rate (rolling 200) | 50% random baseline |
| `reward_curve.png` | Training game | Cumulative reward | Rolling mean |
| `episode_length.png` | Training game | Transitions per episode | Rolling mean |
| `epsilon_decay.png` | Training game | Epsilon value | — |
| `loss_curve.png` | Training game | MSE TD loss | Rolling mean |

All figures must have labelled axes and a title. Save at `dpi=120`.

---

## 11. UI Pattern (Pygame)

### Window layout
```
┌─────────────────────────────────┬───────────────────────┐
│                                 │  GAME TITLE           │
│   Board grid                    │  Turn N | Player P    │
│   (render_board)                │  Mode: SOW / CLAIM    │
│                                 │  ──────────────────── │
│   Yellow border = sow legal     │  Player 1             │
│   Green border  = claim legal   │    Score: XXX         │
│   Orange fill   = about to      │  Player 2             │
│                   explode       │    Score: XXX         │
│                                 │  ──────────────────── │
│                                 │  Controls             │
│                                 │  Click – act          │
│                                 │  TAB   – toggle mode  │
│                                 │  ESC   – menu         │
└─────────────────────────────────┴───────────────────────┘
```

### Colour palette (dark theme)
```python
BG        = (14, 14, 24)
PANEL     = (22, 22, 36)
EMPTY_PIT = (38, 40, 56)
GRID_LINE = (55, 57, 75)
GOLD      = (255, 195, 40)     # locked territory, title
WARN      = (255, 130, 0)      # about to explode / critical
SOW_HL    = (255, 230, 30)     # yellow border — sow highlight
CLAIM_HL  = (40, 220, 100)     # green border — claim highlight
P1_ACCENT = (70, 140, 255)     # player 1 label / pieces
P2_ACCENT = (255, 85, 45)      # player 2 label / pieces
TEXT      = (215, 218, 235)
DIM       = (90, 93, 115)      # hints, secondary labels
```

### Piece density heat-map
- Interpolate from light colour (few pieces) to dark colour (many pieces)
- Show the piece count as a centred number label
- Flash `WARN` colour when a piece count reaches an explosion/critical threshold

### Menu must support
- Board size selection (6×6 / 8×8 / default) — cycles on ENTER
- Returns `(mode_string, board_size)` tuple
- Window resizes dynamically when board size changes

### Action input (hotseat)
- **TAB** toggles between SOW mode and CLAIM mode
- **Left-click** on a highlighted pit executes the action
- In CLAIM mode: auto-select the longest legal run from the clicked anchor
- In vs-AI mode: AI calls `agent.select_action(obs, mask)` on its turns

---

## 12. Snapshot / Versioning

Every frozen agent snapshot stored under `models/<version_id>/`:
```
models/
  registry.json          ← master index (list of SnapshotMetadata dicts)
  gen_001/
    weights.pt           ← torch state dict
    metadata.json        ← SnapshotMetadata fields
  gen_002/ ...
```

`SnapshotMetadata` required fields:
```python
version_id        : str       # e.g. "gen_007"
games_trained     : int
epsilon_at_freeze : float
weights_path      : str
created_at        : str       # ISO timestamp
win_rate_vs_random    : float | None
win_rate_vs_heuristic : float | None
elo_rating            : float | None
friendly_name     : str
```

---

## 13. Documentation Templates

### `docs/log.md` — entry format
```markdown
## YYYY-MM-DD

**Session N — short title**

**Actions taken:**
- bullet per significant action

**Why:**
One-paragraph motivation.

**Issues encountered:**
- bullet per problem and how it was resolved

**Impact:**
What is now true that wasn't before.

**Next:**
- bullet per next step
```

### `docs/todo.md` — sections
```markdown
## Done
- [x] item

## In Progress
- [ ] item

## Not Started
- [ ] item
```

### `docs/plan.md` — required sections
1. Overview (1 paragraph)
2. Module map (directory tree with one-line descriptions)
3. Build phases table
4. State encoding table
5. Action space description
6. Reward function
7. Hyperparameter defaults table
8. Required output figures list

---

## 14. Submission Readiness Checklist

Before submitting any project built on this skeleton:

**Code**
- [ ] `python -m pytest tests/ -v` — all pass from a fresh terminal
- [ ] No `__pycache__` committed
- [ ] All public functions have one-line docstrings
- [ ] No file exceeds 120 lines meaningfully
- [ ] `requirements.txt` lists all imports; `pip install -r requirements.txt` works

**Training**
- [ ] `training_log.csv` is one coherent run (not concatenated re-runs)
- [ ] `results/figures/` contains exactly the 5 required PNGs
- [ ] All figures have axis labels and titles
- [ ] Numbers in `report.md` match `training_log.csv`

**Documentation**
- [ ] `docs/log.md` has an entry for every work session
- [ ] `docs/todo.md` has no "In Progress" items on submission day
- [ ] `docs/report.md` filled with actual results (not stubs)
- [ ] `docs/tep_definitions.md` has formal T, E, P definitions
- [ ] `docs/starter_guide.md` gives a fresh-clone reproduction recipe
- [ ] `README.md` has one-line install + one-line run

**Report**
- [ ] TEP definitions are explicit, formal, and lead the report
- [ ] Algorithm choice (DQN) is justified vs alternatives
- [ ] At least 3 concrete limitations with rationale
- [ ] Critical analysis section answers: "Did performance improve with experience?"
- [ ] Critical analysis section answers: "Was the problem well-posed?"

---

## 15. How to Describe a New Game to the LLM

When starting a new project, attach this file and write a message structured like this:

```
I want to build a new AI board game using the architecture in this skeleton.

GAME NAME: [name]

BOARD: [describe shape, size, starting configuration]

PLAYERS: [how many, who goes first]

ACTIONS:
  [Action 1 name]: [precondition] → [effect]
  [Action 2 name]: [precondition] → [effect]
  ...

INTERACTION: [what happens when pieces meet / compete]

TERMINAL: [when does the game end]

WINNER: [scoring formula — must produce no draws, or specify tie-break]

SPECIAL MECHANICS:
  [any mechanic not covered above, e.g. explosions, chain reactions, power-ups]

TUNABLE PARAMETERS:
  [list any thresholds or toggle flags that belong in config.py]

DIFFERENCES FROM DEFAULT SKELETON:
  [anything you want changed from the standard architecture above]
```

The LLM will then build the complete project using this skeleton as the spec.
Everything not mentioned defaults to the skeleton's standard implementation.
```

---

*This document was extracted from the Masimo project (ITRI616, 2026).  
Reference implementation: `ITRI616_AI_Game_Project_Masimo/`*
