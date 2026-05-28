"""Single source of truth for all C_lines constants and hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (base dirs — run-specific paths are built in train.py)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
REGISTRY_PATH = MODELS_DIR / "registry.json"        # legacy flat path (kept for compat)
TRAINING_LOG_PATH = RESULTS_DIR / "training_log.csv" # legacy (train.py overrides per-run)

# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------
BOARD_SIZES = [8, 9, 10, 11, 12]
DEFAULT_BOARD_SIZE = 10

PLAYER_1 = 1
PLAYER_2 = 2
EMPTY = 0

# ---------------------------------------------------------------------------
# Game modes
# ---------------------------------------------------------------------------
MODE_FIRST_TO_FOUR = "first_to_four"
MODE_POINTS_FULL = "points_full"
DEFAULT_MODE = MODE_POINTS_FULL

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
SCORE_FOR_LENGTH: dict[int, float] = {
    3: 0.25,
    4: 1.0,
    5: 2.0,
    6: 3.0,
    7: 4.0,
    8: 5.0,
}
MIN_SCORING_LENGTH = 3
MAX_SCORING_LENGTH = 8

# ---------------------------------------------------------------------------
# Tie-break
# ---------------------------------------------------------------------------
TIEBREAK_MAX_ROUNDS = 3

# ---------------------------------------------------------------------------
# Optional rule toggles
# ---------------------------------------------------------------------------
USE_SWAP_RULE = False
USE_OPENING_RESTRICT = False

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
STEP_REWARD_SCALE = 0.05
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = 0.0

# ---------------------------------------------------------------------------
# DQN training hyperparameters — v2 (improved)
# ---------------------------------------------------------------------------
LR = 1e-3                   # initial LR; decays at 40% and 75% of training
GAMMA = 0.99
REPLAY_CAPACITY = 100_000
BATCH_SIZE = 128            # ↑ from 64 — more stable gradients (P9)
TARGET_SYNC_STEPS = 500     # ↓ from 1000 — sync target net more often
GRADIENT_CLIP = 10.0
GRADIENT_STEPS_PER_GAME = 4 # ↑ from 1 — exploit collected experience fully (P5)

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_GAMES = 5_000

TRAINING_GAMES = 10_000
WARMUP_GAMES = 2_000        # ↑ from 1000 — longer mixed-opponent warmup (P3)
WARMUP_HEURISTIC_PROB = 0.3 # fraction of warmup games using HeuristicAgent (P3)
SELF_PLAY_MIX_PROB = 0.5
RECENT_POOL_BIAS = 0.7      # prob of sampling from top-5 recent snapshots (P4)
RECENT_POOL_TOP_N = 5       # how many "recent" snapshots count as strong (P4)

SNAPSHOT_INTERVAL = 500     # ↓ from 1000 — finer-grained pool diversity (P8)
MAX_POOL_SIZE = 20

EVAL_INTERVAL = 100         # ↓ from 500 — 10× more curve resolution (P1)
EVAL_GAMES = 100            # ↑ from 50 — halves WR estimate noise (P7)

# LR schedule milestones (fraction of TRAINING_GAMES)
LR_DECAY_MILESTONES = [0.40, 0.75]   # halve LR at these fractions (P6)
LR_DECAY_FACTOR = 0.5

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
MODEL_VERSION = 2   # bumped from 1 — architecture unchanged but training protocol changed

DIFFICULTY_BANDS = {
    "novice": (0.00, 0.25),
    "easy": (0.25, 0.50),
    "medium": (0.50, 0.75),
    "hard": (0.75, 1.00),
    "master": (1.00, 1.00),
}

FRIENDLY_NAMES = {
    "novice": ["Apprentice", "Beginner"],
    "easy": ["Improver", "Cadet"],
    "medium": ["Tactician", "Adept"],
    "hard": ["Strategist", "Veteran"],
    "master": ["Master", "Champion"],
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CELL_PX = 56
BOARD_MARGIN = 24
SIDEBAR_W = 320
ANIMATION_DURATION_MS = 120
