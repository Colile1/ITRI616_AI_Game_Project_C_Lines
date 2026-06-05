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
BOARD_SIZES = [8, 9, 10, 11, 12]          # all supported sizes
UI_BOARD_SIZES = [8]                       # sizes shown in the UI (add back once agents trained)
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
EPS_DECAY_GAMES = 7_000     # extended from 5000 — more exploration into late training

TRAINING_GAMES = 10_000
WARMUP_GAMES = 2_000        # ↑ from 1000 — longer mixed-opponent warmup (P3)
WARMUP_HEURISTIC_PROB = 0.3 # fraction of warmup games using HeuristicAgent (P3)
SELF_PLAY_MIX_PROB = 0.5
RECENT_POOL_BIAS = 0.5      # ↓ from 0.7 — reduce recency bias for more diversity (F4)
RECENT_POOL_TOP_N = 5       # how many "recent" snapshots count as strong (P4)

SNAPSHOT_INTERVAL = 1000     # ↓ from 1000 — finer-grained pool diversity (P8)
MAX_POOL_SIZE = 30           # ↑ from 20 — more pool diversity (F4)

EVAL_INTERVAL = 100         # ↓ from 500 — 10× more curve resolution (P1)
EVAL_GAMES = 100            # legacy — kept for backward compat; new code uses the two below
EVAL_GAMES_VS_RANDOM    = 200   # doubled from 100 — tighter WR estimate
EVAL_GAMES_VS_HEURISTIC = 100   # doubled from 50 — tighter WR estimate

# LR schedule — plateau-based (replaces fixed milestones)
LR_DECAY_MILESTONES = [0.40, 0.75]   # kept for legacy schedule path only
LR_DECAY_FACTOR = 0.5
PLATEAU_PATIENCE = 10       # eval intervals with no WR improvement before LR halves

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
# State encoding
# ---------------------------------------------------------------------------
STATE_CHANNELS_V1 = 6    # original 6-channel encoding
STATE_CHANNELS_V2 = 10   # 10-channel encoding (adds open-4, closed-4, win/loss cells)
STATE_CHANNELS = STATE_CHANNELS_V2  # default for new training

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
NUM_RES_BLOCKS = 4
RES_CHANNELS = 64
NETWORK_ARCH = "resnet_v1"   # "resnet_v1" or "plain_v1"

# ---------------------------------------------------------------------------
# Symmetry augmentation (D4 dihedral group on square board)
# ---------------------------------------------------------------------------
USE_SYMMETRY_AUGMENTATION = True   # apply 8-fold augmentation at sample time

# ---------------------------------------------------------------------------
# MCTS (used by MCTSAgent at inference)
# ---------------------------------------------------------------------------
MCTS_SIMS_BY_CONTEXT: dict[str, int] = {
    "self_play":       200,
    "snapshot_play":   200,
    "alphabeta":       200,
    "human_play":       80,   # keep move latency tolerable for humans
    "benchmark_eval":  400,
}
MCTS_C_PUCT = 1.4
MCTS_LEAF_EVAL = "q_value"   # or "random_rollout"
MCTS_MAX_THINK_SEC = 1.5     # hard wall-clock cap per move

# ---------------------------------------------------------------------------
# Alpha-beta benchmark agent
# ---------------------------------------------------------------------------
BENCHMARK_DEPTH_BY_SIZE: dict[int, int] = {8: 2, 9: 2, 10: 2, 11: 2, 12: 2}
BENCHMARK_EVAL_WEIGHTS: dict[str, float] = {
    "open_3": 1.0,
    "open_4": 5.0,
    "line_score": 1.0,
    "mobility": 0.1,
}

# ---------------------------------------------------------------------------
# Weighted replay buffer
# ---------------------------------------------------------------------------
MAX_REPLAY_WEIGHT = 20.0     # cap to prevent any transition from dominating
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "self":       1.0,
    "pool":       1.0,
    "random":     0.5,
    "heuristic":  1.0,
    "alphabeta":  2.0,
    "human":      5.0,
    "demo":      10.0,
}

# ---------------------------------------------------------------------------
# Benchmark logging
# ---------------------------------------------------------------------------
BENCHMARK_EVERY_N_GAMES_SMALL = 100  # board sizes <= 10
BENCHMARK_EVERY_N_GAMES_LARGE = 200  # board sizes >= 11
BENCHMARK_GAMES_PER_CHECK = 64       # ↑ from 32 — halves sampling noise (R2)

# Self-play pool quality gate
MIN_POOL_WR = 0.50          # only add snapshot to pool if WR vs random >= this

# First-to-four mode reward shaping
FTF_THREAT_SCALE      = 0.10   # reward scale for open-3 threat delta in ftf mode
FTF_THREAT_SCALE_4    = 0.30   # reward scale for double-open-3 ("forced-win") threats (F2)
FTF_SURVIVAL_SCALE    = 0.0    # intentionally zero: survival bonus rewards stalling (wrong objective)
FTF_EARLY_LOSS_TURNS  = 8      # total moves at or below which a loss is "early" (adaptive in train.py, F6)
FTF_EARLY_LOSS_EXTRA  = 0.5    # extra penalty on top of LOSS_REWARD for early collapse

# N-step returns
N_STEP_RETURNS = 3             # n-step lookahead for TD target (1 = classic 1-step, F5/R3)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CELL_PX = 56
BOARD_MARGIN = 36
SIDEBAR_W = 320
ANIMATION_DURATION_MS = 120
