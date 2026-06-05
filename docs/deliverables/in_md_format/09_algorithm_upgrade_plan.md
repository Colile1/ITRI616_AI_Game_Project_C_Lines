# Algorithm Upgrade — Implementation Plan

**Author:** *----* *S----*
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Plan locked, implementation pending
**Depends on:** `08_similar_games_research.md`

---

## 1. Goal

Take C_lines's existing DQN agent (8×8 reaches 86% vs random, plateaued at weak human-perceived play) and convert it into an agent that visibly plays well against a thinking opponent. The benchmark for success is set in `10_human_in_loop_training_plan.md`: the upgraded agent should beat the new alpha-beta classical benchmark at least 40% of the time on 8×8, where the current DQN scores ≤ 10%.

The plan is staged in five phases, ordered by impact-per-hour and by minimum-blast-radius. Each phase can ship independently and improves play measurably; later phases compound.

## 2. Constraints inherited from the existing build

* Keep the existing `BaseAgent` / `DQNAgent` / `SnapshotMetadata` / `registry` contracts intact. All upgrades extend, none break.
* Keep the existing `engine/`, `game/encoding.py`, `game/env.py` modules unchanged for the encoding upgrade; new channels are appended at the end so prior snapshots remain readable when re-loaded under a compatibility flag.
* `config.py` remains the only home for constants; new flags live there.
* The frosted-glass UI does not need to change for any of these upgrades; the UI calls `agent.select_action(obs, mask)` regardless of what is inside the agent.

## 3. Phase A — Symmetry augmentation in the replay buffer

**Goal.** Multiply effective training data per episode by 8 (the dihedral group D₄ of a square board) without playing more games.

### 3.1 Why it works

A square board has 8 symmetries: 4 rotations × 2 mirror options. Each transition `(s, a, r, s', done, legal_mask)` has 7 symmetry-equivalent twins. Sampling from these increases the diversity of any minibatch and frees the network from re-learning that "the top-left corner is the same as the bottom-right corner". Documented to cut time-to-strength by 30–60% on Gomoku and Go.

### 3.2 What to build

* `src/training/symmetry.py` — new file. Pure functions:
  * `apply_symmetry(grid: np.ndarray, k: int) -> np.ndarray` — k ∈ 0..7, rotates and optionally reflects.
  * `transform_action_index(idx: int, n: int, k: int) -> int` — maps a flat action index through symmetry k for a board of side n.
  * `transform_mask(mask: np.ndarray, n: int, k: int) -> np.ndarray` — applies the same permutation to a flat legal mask.
  * `transform_obs(obs: np.ndarray, k: int) -> np.ndarray` — applies symmetry to every channel.
* Modify `src/training/replay_buffer.py`:
  * Either (a) **push-time** — duplicate each pushed transition 8x with symmetry indices, OR
  * (b) **sample-time** — sample a random k for every transition in a minibatch.
  * Recommend **(b)** — same statistical benefit, 8x less memory, no buffer-capacity inflation.

### 3.3 Acceptance test

* New `tests/test_symmetry.py`:
  * For every k, applying symmetry k and its inverse returns the original board.
  * `transform_action_index` is a bijection on `{0..N²-1}` for every k.
  * Symmetric self-consistency: `score_board(transform(board, k), player)` equals `score_board(board, player)` for every k.
* Smoke training: `train --games 200 --size 8 --augment` reaches a higher win-rate against random in 200 games than the same run with `--no-augment` (expected to be visible by game 100).

### 3.4 Cost

1–2 hours of code. No retraining wall-clock cost (this *reduces* it). No risk to existing snapshots.

## 4. Phase B — Richer state-encoding channels (open-4, closed-4, immediate-win, immediate-loss)

**Goal.** Hand the network the four most strategically important pattern features so it spends capacity on strategy rather than pattern detection.

### 4.1 New channels

Extend the state tensor from 6 to 10 channels (`STATE_CHANNELS = 10`):

| ch | Content (new in bold)                                            |
|----|------------------------------------------------------------------|
| 0  | Current player's pieces (binary)                                 |
| 1  | Opponent's pieces (binary)                                       |
| 2  | Empty cells (binary)                                             |
| 3  | Current player's open-3 threats (binary)                         |
| 4  | Opponent's open-3 threats (binary)                               |
| 5  | Turn number / N² (constant plane)                                |
| 6  | **Current player's open-4 threats (binary)**                     |
| 7  | **Opponent's open-4 threats (binary)**                           |
| 8  | **Cells where placing now would immediately win (binary)**       |
| 9  | **Cells where the opponent would immediately win next turn (binary)** |

Channels 6–9 are computed by running the existing line-counter at length 4 on the board with the candidate cell hypothetically filled. The scanning logic exists; only the wrapper is new.

### 4.2 What to build

* `src/game/encoding.py`:
  * Add `_compute_open4_threats(grid, player, n)`, `_compute_immediate_wins(grid, player, n)`, `_compute_immediate_losses(grid, player, n)`.
  * Extend `state_to_tensor` to produce the new channels.
  * Add `STATE_CHANNELS_V2 = 10` to `config.py`; keep `STATE_CHANNELS_V1 = 6` for back-compat.
* `src/training/network.py`:
  * `DQNNetwork(board_size, in_channels=STATE_CHANNELS_V2)` — parameterised input channels.
* `src/versioning/metadata.py`:
  * Add `state_channels: int` to `SnapshotMetadata`. `load_snapshot` reads it and instantiates the network with the correct `in_channels`.

### 4.3 Acceptance test

* `tests/test_encoding.py`:
  * Open-4 channel is set on exactly the cells that would form a length-4 open run if filled.
  * Immediate-win channel is 1 on at least one cell when an open-3 exists with both extensions empty.
  * Old 6-channel snapshots still load via the compatibility flag.
* No regression in the existing `test_env.py` tests after the channel count bump.

### 4.4 Cost

Half a day of code. Retraining the agent is required to benefit from the new channels, but a small smoke run (200 games on 8×8) will already show whether the new channels are wired correctly.

## 5. Phase C — Residual-block network

**Goal.** Replace the plain 3-conv stack with 3–5 residual blocks for better feature reuse and stabler training of a deeper network.

### 5.1 Architecture

```
Input  : (B, 10, N, N)
ConvBNReLU(10 -> 64, 3×3)

# 3-5 residual blocks, each:
#   x  -> Conv(64->64,3x3) -> BN -> ReLU -> Conv(64->64,3x3) -> BN -> +x -> ReLU
[ResBlock(64)] × NUM_RES_BLOCKS

# Q-head
Conv(64 -> 32, 1×1) -> BN -> ReLU
Flatten -> Linear(32·N·N -> 256) -> ReLU -> Linear(256 -> N·N)

Output : (B, N*N)
```

For N=12, parameter count rises to ~3.5M with 4 res blocks — still CPU-feasible.

### 5.2 What to build

* `src/training/network.py`:
  * Add `class ResidualBlock(nn.Module)`.
  * Refactor `DQNNetwork` to accept `num_res_blocks: int = 4`. Old plain-CNN constructor kept under `DQNNetworkPlain` for back-compat with older snapshots.
* `config.py`:
  * `NUM_RES_BLOCKS = 4`, `RES_CHANNELS = 64`.
  * `NETWORK_ARCH = "resnet_v1"` (string tag persisted in metadata).
* `src/versioning/metadata.py`:
  * Add `network_arch: str` to `SnapshotMetadata`. Loader picks the right class.

### 5.3 Acceptance test

* Forward-pass shape unchanged: `(B, 10, N, N) -> (B, N*N)`.
* New res-block model trains stably on 200-game smoke run (no loss explosion, no NaN).
* Old snapshots still loadable under `DQNNetworkPlain` (`network_arch` switches the loader).

### 5.4 Cost

1 day of code + a full retrain wall-clock per board size. Retraining is the long pole.

## 6. Phase D — MCTS at inference (the big strength jump)

**Goal.** Add UCT/PUCT search on top of the trained DQN. No retraining needed; uses the existing Q-network as a search prior.

### 6.1 Algorithm

PUCT (Polynomial Upper Confidence applied to Trees) per AlphaZero, but with priors derived from the existing Q-network. For each move, run `MCTS_SIMULATIONS` simulations (default 200):

```
For each simulation:
    1. SELECTION: from the root, descend by argmax_a [Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))]
       until reaching a leaf or terminal.
    2. EXPANSION: at the leaf, expand all legal children. Initialise their priors P(s,a)
       from a softmax over the trained DQN's Q-values, masked to legal moves.
    3. EVALUATION: leaf value = DQN's max Q-value at this state, scaled into [-1, 1]
       via tanh or clipped linear. (Or: a quick random rollout to terminal, if time
       permits — slower but more accurate.)
    4. BACKUP: walk the path back to root, updating N(s,a) and W(s,a); Q(s,a) = W/N.

Final move = argmax over root children of N(s,a) (visit count — more robust than Q).
```

### 6.2 What to build

* `src/agents/mcts_agent.py` — new file. Wraps an underlying `BaseAgent` that provides priors and leaf evaluations; standalone PUCT loop.
  * `class MCTSAgent(BaseAgent)`:
    * `__init__(prior_agent, n_simulations, c_puct, leaf_eval="q_value" | "random_rollout")`
    * `select_action(obs, legal_mask)` runs PUCT and returns the visit-argmax.
  * `class MCTSNode` — children dict, N, W, Q, P, terminal flag.
* `src/training/network.py`:
  * Add `q_to_policy_prior(q_values, mask, temperature=1.0)` helper — softmax over masked Q-values.
* `config.py`:
  * `MCTS_SIMULATIONS = 200` (default; 100 for human-vs-agent play to keep moves <1 s, 800 for benchmark games).
  * `MCTS_C_PUCT = 1.4`.
  * `MCTS_LEAF_EVAL = "q_value"`.

### 6.3 Acceptance test

* `tests/test_mcts.py`:
  * `MCTSAgent(prior=RandomAgent(), n_sims=50)` always returns a legal action.
  * With 200 sims and the trained DQN as prior, MCTSAgent beats the bare DQNAgent ≥ 70% in 100 head-to-head 8×8 games.
  * Move latency at 200 sims is < 0.5 s on the 12×12 board on CPU.
* `tests/integration/test_mcts_eval.py`:
  * `evaluate(MCTSAgent(dqn, 200), RandomAgent(), 100)` reaches ≥ 95% win rate (vs 86% for bare DQN).

### 6.4 Cost

1–2 days of code, no retraining. Highest impact-per-hour of any phase.

## 7. Phase E — Classical alpha-beta benchmark agent (fixed reference)

**Goal.** Provide a fixed, non-learning, thinking opponent against which improvement can be measured rigorously. Required by `10_human_in_loop_training_plan.md` as the benchmark logged after every training game.

### 7.1 What to build

* `src/agents/alphabeta_agent.py`:
  * Iterative-deepening alpha-beta on a bitboard representation, with a hand-engineered evaluation function:
    * `+W_3 * (open_3 - opp_open_3)`
    * `+W_4 * (open_4 - opp_open_4)`
    * `+W_LINE * (sum_of_line_scores_me - sum_of_line_scores_opp)`
    * Mobility term (number of legal moves) for tie-break.
  * Depth budget: 4 ply on 8×8, 3 ply on 12×12, by default. Configurable.
  * Move ordering: try centre cells first, then cells adjacent to existing pieces.
  * Transposition table optional (Phase E.1, deferred).
* `src/engine/bitboard.py` (optional but recommended):
  * `pack_grid(grid) -> tuple[int, int]` — two big integers, one per player.
  * `lines_touching(bitboard, r, c, n)` — fast pattern scan.
* `config.py`:
  * `BENCHMARK_DEPTH_BY_SIZE = {8: 4, 9: 4, 10: 3, 11: 3, 12: 3}`.
  * `BENCHMARK_EVAL_WEIGHTS = {"open_3": 1.0, "open_4": 5.0, "line_score": 1.0, "mobility": 0.1}`.

### 7.2 Acceptance test

* `tests/test_alphabeta.py`:
  * Always returns a legal action.
  * On a forced-mate-in-1 position, picks the mating move at depth 1.
  * On a fork position, finds it at depth 4.
  * Beats `RandomAgent` ≥ 95% over 50 games on 8×8.
  * Beats `HeuristicAgent` ≥ 70% over 50 games on 8×8.

### 7.3 Cost

Half a day to a day. Reusable across both the upgrade plan and the benchmark plan; doubles as a Mode-1 quick-play opponent for the UI's "vs strong baseline" option.

## 8. Sequencing and dependencies

```
A. Symmetry augmentation    -- independent; ship first
B. Richer state channels    -- independent; can ship in parallel with A
C. Residual-block network   -- depends on B (more channels)
                                or can ship without B if architecture-only
D. MCTS at inference        -- works with any trained DQN; ship after C is retrained
                                (or earlier with the current snapshot)
E. Alpha-beta benchmark     -- independent of all of the above; required by Plan 10
```

A pragmatic schedule: ship A immediately (no retrain), ship E in parallel (no retrain). Re-run training with A + B + C, which gives a stronger base DQN. Then add D on top of the new DQN.

## 9. Updated success criteria (vs the originals in `01_project_brief.md`)

| Criterion | Old target | New target |
|-----------|-----------|-----------|
| Final WR vs RandomAgent (8×8) | ≥ 75% | ≥ 95% (with MCTS-at-inference) |
| Final WR vs HeuristicAgent (8×8) | ≥ 55% | ≥ 85% (with MCTS-at-inference) |
| Final WR vs AlphaBetaAgent (8×8) | n/a | ≥ 40% (with MCTS-at-inference) |
| Training-curve trend | upward | upward AND benchmark-vs-fixed shows monotone climb |

These are stretch goals; mid-phase milestones are below.

## 10. Mid-phase milestones (what to expect at each stage)

| Stage shipped | Expected WR vs random | Expected WR vs heuristic | Expected WR vs alpha-beta |
|---------------|----------------------|--------------------------|---------------------------|
| Today (baseline) | 86% | 30-50% | ≤ 10% |
| + A (symmetry) | 90% | 40-55% | 10-15% |
| + B (channels) | 92% | 50-60% | 15-20% |
| + C (resnet) | 94% | 60-70% | 20-30% |
| + D (MCTS) | 98%+ | 80-90% | 40-50% |

The MCTS step delivers the biggest visible jump and the milestone the user is most likely to feel during play.

## 11. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| MCTS too slow for human play on 12×12 | Medium | Medium | Configurable `MCTS_SIMULATIONS` per opponent context; default 100 for vs-human, 200 for vs-self, 800 for benchmark games |
| Residual network overfits to the smaller boards | Low | Medium | Add weight decay (1e-4) and dropout (0.1) in the head; standard hygiene |
| New channels break old snapshots | Medium | Low | `state_channels` field in metadata + back-compat loader; clearly surface "this snapshot was trained with an older encoder" in the UI |
| Alpha-beta benchmark itself is too weak on 12×12 (depth 3 is shallow) | Medium | Low | Document the depth; benchmark is *fixed* so a weaker benchmark still detects relative improvement |
| Retraining wall-clock per size triples with res-blocks + MCTS targets | High | Medium | Re-train 8×8 first (vertical slice); only scale up after 8×8 demonstrates the expected gains |

## 12. What this plan does **not** include

Explicitly out of scope for this upgrade (deferred to a future project or to `11_unified_upgrade_plan.md`):

* Full AlphaZero (policy + value heads with MCTS-policy-as-target during training). The MCTS-at-inference in Phase D is the AlphaZero-light stepping stone.
* MuZero (learned model).
* Distributed self-play / population-based hyperparameter search.
* Bitboard rewrite of the entire engine. The bitboard in Phase E is local to the alpha-beta agent.
* GPU-specific tuning.

---

*End of algorithm upgrade plan.*
