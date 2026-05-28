"""Single source of truth for all C_lines constants and hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
REGISTRY_PATH = MODELS_DIR / "registry.json"
TRAINING_LOG_PATH = LOGS_DIR / "training_log.csv"

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
# DQN training hyperparameters
# ---------------------------------------------------------------------------
LR = 1e-3
GAMMA = 0.99
REPLAY_CAPACITY = 100_000
BATCH_SIZE = 64
TARGET_SYNC_STEPS = 1_000
GRADIENT_CLIP = 10.0

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_GAMES = 5_000

TRAINING_GAMES = 10_000
WARMUP_GAMES = 1_000
SELF_PLAY_MIX_PROB = 0.5

SNAPSHOT_INTERVAL = 1_000
MAX_POOL_SIZE = 20

EVAL_INTERVAL = 500
EVAL_GAMES = 200

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
MODEL_VERSION = 1

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
