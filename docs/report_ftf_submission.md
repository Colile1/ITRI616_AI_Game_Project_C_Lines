# C_lines: A Self-Play Reinforcement Learning Agent for a Southern African Line-Formation Game
## ITRI 616 Mini-Project — Technical Report

**Student:** Colile Sibanda | **Module:** ITRI 616 — Artificial Intelligence 1
**Game:** C_lines (original) | **Algorithm:** DQN with self-play
**Code repository:** [GitHub link]

---

## 1. Introduction and Motivation

This project digitises and extends a class of Southern African line-formation games — traditional stone and seed games where players compete to place pieces on a grid and form lines. The game, **C_lines**, was built specifically for this project as both a cultural preservation exercise (making a traditional gaming format freely available digitally) and as a learning environment for an AI agent.

C_lines is played on an 8×8 grid. Players alternate placing one piece per turn anywhere on the board. The goal is simple: **be the first to complete a line of four or more pieces in any direction** — horizontal, vertical, or diagonal. The first player to achieve this wins immediately. If the board fills without either player completing a line of four, the player with the highest cumulative line score wins.

This "First-to-Four" mode was chosen because it provides an unambiguous, immediate objective — winning means completing a specific pattern as fast as possible — which creates a clean test of whether the agent can learn goal-directed tactical play. The agent has no access to human expertise or pre-programmed strategy; every skill it acquires comes from playing against itself.

---

## 2. TEP Framework — Formal Learning Problem Definition

### 2.1 Task (T)

**Task category:** Sequential decision-making — a finite-horizon Markov Decision Process (MDP). Not classification, not prediction; the agent learns a *policy*, a function from board states to actions, that maximises expected return.

**Formal MDP:**

| Component | Specification |
|---|---|
| **State S** | All reachable 8×8 board configurations, encoded as a 10-channel float32 tensor. Channels: own pieces, opponent pieces, empty cells, own/opponent open-3 threats, own/opponent open-4 threats, immediate-win cells, immediate-lose cells, turn-progress fraction. |
| **Action A** | 64 cell indices (`i = row × 8 + col`). Occupied cells set to −10⁹ before action selection — the agent never plays an illegal move. |
| **Transition δ** | Deterministic: place piece at chosen cell, advance turn, check for 4-in-a-row. Implemented as a pure function with no side effects. |
| **Reward R** | Terminal: `+1` win, `−1` loss (or `−1.5` for games ending in ≤ 8 total moves — heavier early-collapse penalty). Non-terminal: `(open-3 threat delta) × 0.10` — a shaping term aligned with the mode objective. No survival bonus: rewarding prolonged games is counter to the First-to-Four goal. |
| **Discount γ** | 0.99 |
| **Horizon T_max** | 64 moves (board full) — but in practice the agent wins in 8–10 moves by late training. |

**Relevance of the task design:** The reward structure is deliberately mode-specific. In First-to-Four, the only thing that matters is *building a 4-line before your opponent*. Shaping that rewards general line-building (any length) would teach the wrong objective — a mistake that earlier runs confirmed empirically. The open-3 threat shaping specifically rewards advancing toward the 4-line goal.

### 2.2 Experience (E)

The agent learns from **self-play simulated episodes** only. No human expert data exists for C_lines.

**Training curriculum:**

| Phase | Games | Opponent | Purpose |
|---|---|---|---|
| Warm-up | 0–2,000 | 30% Heuristic, 70% Random | Bootstrap replay buffer with diverse positions |
| Self-play | 2,000–10,000 | 45% pool snapshots, 45% self-clone, 10% anchors | Main learning from improving opponents |

A **snapshot pool** (max 20 frozen checkpoints, capped by win-rate quality gate) provides a self-organising difficulty ladder. A 10% permanent anchor rate (Random / Heuristic games) prevents curriculum collapse.

Each game produces transitions `(s_t, a_t, r_t, s_{t+1}, done, legal_mask_{t+1})` stored in a 100,000-capacity replay buffer. Four gradient steps per game, 128-sample mini-batches, D4 symmetry augmentation (8-fold) at sample time.

**Special mechanism for FTF credit assignment:** In First-to-Four mode the *losing* player never makes the final move, so they never receive a terminal reward from the environment. To close this credit-assignment gap, the training loop manually pushes a synthetic terminal transition to the losing player's replay buffer after each episode, carrying the graduated loss reward (`−1.0` for normal loss, `−1.5` for early collapse). Without this, the loss signal is completely invisible to the Q-function.

### 2.3 Performance (P)

Primary metrics tracked every 100 games:

| Metric | Definition | Relevance |
|---|---|---|
| Win rate vs Random (W_r) | Fraction of 200 games won against RandomAgent | Baseline — any learning should exceed 50% |
| Win rate vs Heuristic (W_h) | Fraction of 100 games won against HeuristicAgent | Non-learning tactical ceiling with explicit 4-threat blocking |
| Mean episode length | Average total moves per training game | Key FTF signal — shorter = agent wins faster = mode-correct learning |
| Elo rating | Updated against Random (anchor 800) and Heuristic (anchor 900) every 100 games | Skill estimate (note: less reliable in FTF mode — see §5) |
| Benchmark win rate | 32 games vs Alpha-Beta depth-4 every 100 games | Strength against principled tree search |

**Formal hypotheses:**
- **H1:** W_r trends upward, ends ≥ 0.75
- **H2:** Mean episode length drops measurably (agent learns to win faster)
- **H3:** Agent wins benchmark games against alpha-beta depth-4

---

## 3. Learning Algorithm — Design and Implementation

### 3.1 Algorithm: Deep Q-Network (DQN)

DQN was selected because: (1) C_lines is a discrete-action, perfect-information MDP — the textbook DQN domain; (2) it provides a clear win-rate improvement curve for the brief; (3) the target-network state dict doubles as the snapshot artefact for the difficulty ladder.

### 3.2 Network Architecture (ResNet-v1)

```
Input:  (B, 10, 8, 8)   — 10-channel board encoding

Conv stem:  Conv2d(10→64, 3×3, pad=1) → BatchNorm → ReLU
4× Residual block: [Conv→BN→ReLU→Conv→BN + skip → ReLU]
Q-head:     Conv2d(64→1, 1×1) → Flatten → Linear(64→64)

Output: (B, 64)  — Q-value per cell
```

### 3.3 Critical Implementation Details

**Negamax TD target (decisive change):**
Standard DQN: `target = r + γ · max_a Q(s', a)`
In a two-player game, `s'` is encoded from the *opponent's* perspective. Adding the opponent's Q-value tells the agent "great for opponent = great for me." The correct zero-sum formula negates it:
```
target = r − γ · Q_target(s', argmax_a Q_online(s', a))
```
This change — one minus sign — was the difference between three completely failed runs and a run that learned. Every gradient step in the earlier runs was pointing in the wrong direction.

**Double DQN:** Online network selects the best action; target network evaluates it. Eliminates overestimation bias and prevents late-training collapse.

**Huber loss:** Smooth-L1 replaces MSE — robust to large TD error spikes during the early re-learning phase.

**Plateau-based LR decay:** LR halves when the primary metric plateaus for 10 consecutive evaluation intervals. Four decays triggered: 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4.

---

## 4. Experimental Results

**Training run:** `run_ftf_003` | 10,000 games | ~51 hours on CPU

### 4.1 The Decisive Signal — Episode Length

Mean episode length is the clearest evidence of mode-correct learning in First-to-Four:

| Game | Episode Length | Interpretation |
|------|------|---|
| 0 | 33.3 moves | Near-random play |
| 2,000 | 27.5 moves | Agent starting to race for 4-lines |
| 4,000 | 18.1 moves | Major improvement — wins in ~9 own turns |
| 6,000 | 13.1 moves | Strong tactical play |
| 7,000 | 9.9 moves | Wins in under 5 own turns |
| 9,000 | 8.9 moves | Near-minimum: 4 pieces placed in 4 turns |

By game 7,000 the agent completes a 4-in-a-row in approximately 4–5 of its own turns. This is the mode-correct objective fully learned.

### 4.2 Snapshot Performance

| Gen | Games | WR vs Random | WR vs Heuristic | Ep. Length |
|-----|-------|:---:|:---:|:---:|
| gen_001 | 1,000 | 48.5% | 0% | 31.8 |
| gen_003 | 3,000 | 88% | 1% | 23.9 |
| gen_005 | 5,000 | 96.5% | 5% | 17.1 |
| gen_007 | 7,000 | **100%** | 6% | 9.9 |
| gen_008 | 8,000 | **100%** | **17%** | 9.1 |
| gen_010 | 10,000 | 99.5% | **13%** | 10.2 |

### 4.3 Win Rate Progression

**vs Random:** 48.5% → 77.5% (game 2,000) → 95% (game 4,000) → **100%** (game 7,000, stable thereafter).

**vs Heuristic:** 0% for first 4,100 games → first non-trivial values at game 4,200 (4%) → 10% by game 5,200 → **31% peak at game 9,800**. Slow but real progress against an opponent whose explicit purpose is to block 4-threats.

### 4.4 Benchmark vs Alpha-Beta Depth-4

- Games 0–4,400: **0%** (agent loses all 44 consecutive checks)
- Game 4,500: first win (3.1%)
- Game 6,300: **43.75%** — first major result
- Game 7,700: **78.1%** — strong performance
- Game 8,800: **87.5%** — best single check
- Late run (9,000–9,900): 40–47% typical

The benchmark is volatile: the agent has learned a strong tactical fork pattern that beats alpha-beta from favourable positions but has not generalised to all starting configurations. This is the primary remaining limitation.

### 4.5 Hypothesis Results

| Hypothesis | Threshold | Result | Met? |
|---|---|---|:---:|
| H1 WR_random ≥ 0.75 | 0.75 | **1.00** | ✅ |
| H2 Episode length decreases | Measurable drop | **33 → 8.9 moves (−73%)** | ✅ |
| H3 Beats alpha-beta d4 | >0% | **Peak 87.5%** | ✅ (volatile) |

---

## 5. Critical Analysis and Reflection

### Was the Problem Well-Posed?

By Mitchell's three criteria: the task is fully specified (pure functions, no hidden state), experience is reproducible (fixed seed → identical curve), and performance measures are operationally defined. **Yes, well-posed.**

### What Worked and Why

1. **Negamax sign fix** — the single decisive change. All prior runs used `+γ`. Fixing it to `−γ` immediately produced a learning curve that had never appeared in three prior runs combined.
2. **Threat-delta reward shaping** — specifically aligned with the FTF objective. No survival bonus, no general line scoring: only open-3 threat building. Episode length dropping to 8–10 confirms this teaches the right behaviour.
3. **Early-loss synthetic transition** — closing the credit-assignment gap for the loser in FTF mode. Without it, the loss signal is completely invisible.

### What Didn't Work as Expected

**Elo as a FTF metric.** The Elo computation anchors against Random (800) and Heuristic (900). In FTF mode the agent wins nearly all random games (large upward Elo pressure) but loses most heuristic games (large downward pressure). These nearly cancel, leaving Elo flat at ~790 for the entire run — useless as a skill signal. Worse, the plateau-based LR scheduler uses Elo as its signal, so it detected "no improvement" by game 1,000 and decayed LR to its minimum by game 4,200. The agent spent 58% of training at LR=1.25e-4 — possibly limiting heuristic learning.

**The right metric for FTF is episode length**, not Elo. A decreasing episode length means the agent is winning faster, which is the only thing that matters in First-to-Four.

### Honest Limitations

1. **WR vs heuristic peaks at 31%.** The HeuristicAgent's blocking logic is a direct counter to the FTF racing strategy. The agent learned the first half of the strategy (build fast) but not the second half (build around blocks). This would require open-4 threat shaping — specifically rewarding the creation of forced-win patterns (two simultaneous 4-threats) that even the blocking heuristic cannot stop.

2. **Bimodal benchmark policy.** The agent wins 78–87% from some opening positions and 0% from others. It has one strong fork pattern, not a general strategy. Diversifying the self-play pool and the benchmark seeds would expose it to more opening configurations.

3. **Single seed, single run.** Statistical confidence requires multiple seeds. One run with these numbers is promising; three runs with error bars would be definitive.

### Reflection on the Project

The most important lesson is that algorithm choice alone is insufficient — the theoretical correctness of the learning rule matters as much as the implementation. The same architecture, same curriculum, and same hyperparameters produced radically different results depending solely on whether the TD target sign was correct. This would not have been discovered without the systematic post-run analysis that compared the learning curves across runs.

The FTF mode also revealed that metrics must be chosen to match the game mode. Elo, which worked perfectly for points-full training, failed completely for FTF because its two anchor opponents gave contradictory signals. Episode length is the natural FTF metric, and designing the evaluation system around it would have produced better training decisions (earlier identification of the heuristic gap, more appropriate LR schedule).

---

## 6. Code Documentation

### 6.1 Installation

**Requirements:** Python 3.11+, PyTorch 2.2+, PyGame 2.5+, NumPy 1.26+, Matplotlib 3.8+

```bash
git clone [repository_url]
cd ITRI616_AI_Game_Project_Flat_4_in_Row
pip install -r requirements.txt
```

### 6.2 Running the Game (Playable UI)

```bash
python -m src.ui.app
```

Navigate: Main Menu → Board Size (8×8) → Game Mode (First to Four) → Level Select → Play.

The level-select screen shows trained snapshots as difficulty cards. Snapshots from `run_ftf_003` are labeled Apprentice/Beginner (all in the novice band) — all are playable opponents.

### 6.3 Training a New Agent

```bash
# Standard auto-training with benchmark
python -m src.training.train --games 10000 --size 8 --mode first_to_four --benchmark alphabeta_d4

# With human-in-the-loop (opens PyGame training board)
python -m src.training.train --games 10000 --size 8 --mode first_to_four --human

# Resume a previous run
python -m src.training.train --games 10000 --size 8 --mode first_to_four --run-id run_ftf_003 --resume
```

Outputs (auto-named `run_ftf_NNN`):
- `results/size_08/run_ftf_NNN/training_log.csv` — eval metrics every 100 games
- `results/size_08/run_ftf_NNN/benchmark_log.csv` — benchmark every 100 games
- `results/size_08/run_ftf_NNN/game_log.csv` — every individual game result
- `results/size_08/run_ftf_NNN/best/weights.pt` — best model (by WR_heuristic)
- `models/size_08/run_ftf_NNN/gen_NNN/` — snapshots every 1,000 games

### 6.4 Analysing Results

```python
import pandas as pd

# Load training log
log = pd.read_csv("results/size_08/run_ftf_003/training_log.csv")
print(log[["game", "win_rate_vs_random", "win_rate_vs_heuristic", "mean_ep_len"]].tail(10))

# Load game-level log (every individual game)
games = pd.read_csv("results/size_08/run_ftf_003/game_log.csv")
human_games = games[games["p1_label"] == "human"]  # filter human games

# Load benchmark
bench = pd.read_csv("results/size_08/run_ftf_003/benchmark_log.csv")
print(bench[["training_game", "win_rate", "mean_episode_length"]].tail(20))
```

Generate plots:
```python
from src.evaluation.plots import generate_all_plots, generate_benchmark_plot
figs = generate_all_plots("results/size_08/run_ftf_003/training_log.csv", 8,
                           "results/size_08/run_ftf_003/figures")
```

### 6.5 Key Source Files

| File | Purpose |
|---|---|
| `src/config.py` | All hyperparameters — single source of truth |
| `src/game/env.py` | RL environment (reward function, state encoding) |
| `src/engine/rules.py` | Game rules (pure functions, no side effects) |
| `src/engine/scoring.py` | Line scoring + threat counting |
| `src/agents/dqn_agent.py` | DQN: negamax target, Double DQN, Huber loss |
| `src/training/train.py` | Main training loop (curriculum, Elo, plateau LR, logging) |
| `src/training/self_play.py` | Episode execution |
| `src/training/replay_buffer.py` | Weighted replay buffer with D4 augmentation |
| `src/versioning/registry.py` | Snapshot registry (per-run JSON files) |
| `src/ui/app.py` | PyGame UI — full game + level select + replay |

---

## 7. References

- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill, Ch. 1. — TEP framework.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Van Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement learning with Double Q-learning. *AAAI*, 30(1).
- Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Silver, D. et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529, 484–489.
