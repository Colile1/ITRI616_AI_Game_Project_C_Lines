# Formal TEP Definitions — C_lines AI Agent

**Framework:** Mitchell, T. M. (1997). *Machine Learning*, Chapter 1 — "A Well-Posed Learning Problem".
**Game:** C_lines (see `deliverables/in_md_format/03_game_description.md`)
**Author:** *----* *S----* — ITRI 616, 2026

---

## Task (T)

### Informal statement

Given a partially-filled C_lines board and the indication of whose turn it is, choose a placement (or, during the Mode-2 tie-break, a removal) that maximises the expected end-of-game return.

### Formal statement

The task is a **sequential decision-making problem** structured as a finite-horizon Markov Decision Process (MDP):

```
MDP = (S, A, P, R, γ, T_max)
```

* **State space S** — all reachable configurations of an N×N board where each cell is one of {empty, P1, P2}, augmented by (i) whose turn it is, (ii) whether the game is in placement phase or tie-break removal phase, and (iii) for Mode 2, the per-player cumulative score and the tie-break round counter.

  Cardinality upper bound: |S| ≤ 3^(N²) × 2 × 2 × 4 (the last two factors account for active-player, phase, tie-break round). For N=12 the dominant term is 3^144 ≈ 1.66 × 10^68 — far too large to enumerate, so function approximation (a neural network) is required.

* **Action space A** — for each board size N, |A| = N². The same flat integer index `i = r·N + c` is used both for placement (in regular play) and for removal (during the tie-break), with a phase-conditional legal mask.

* **Transition function δ : S × A → S** — deterministic. Given state s and a legal action a, the next state s' is computed by either `apply_placement(board, action, player)` or `apply_removal(board, action, opponent)`, followed by turn advancement, scoring update, and terminal/phase checks. Illegal actions are filtered before they reach the transition function, so δ is never invoked on an undefined input.

* **Reward function R : S × A × S → ℝ** — mode-conditional and defined in `04_implementation_plan.md` §7. For Mode 1, the per-step reward is `+STEP_REWARD_SCALE` if the placement completes a line of length ≥ 4, else 0. For Mode 2, the per-step reward is `(Δown_score − Δopp_score) × STEP_REWARD_SCALE` — a delta-score shaping that converts the sparse end-of-game reward into a dense per-step signal aligned with the eventual win/loss outcome. The terminal reward is `+WIN_REWARD = +1` on a win, `LOSS_REWARD = −1` on a loss, and 0 on a draw.

* **Discount γ** — set to 0.99 (configurable). Games are short (≤ N² steps), so γ near 1 is appropriate: future reward should be valued similarly to immediate reward.

* **Horizon T_max** — bounded above by N² placements (board fills) plus up to 6 removals (3 tie-break rounds × 2 players). For N = 12 the worst-case horizon is 144 + 6 = 150 steps.

### Task category

**Sequential decision-making under perfect information and deterministic dynamics.** Not classification (no labelled examples of "the correct move"). Not prediction (no continuous variable to forecast). Not control in the engineering sense (no continuous state). The agent's primary objective is to learn a policy π : S → A that maximises expected return.

### Originality of the task

C_lines is a game we built for this project. There is no published expert-play corpus, no opening book, no solver. The agent therefore cannot learn from human data — experience must come from interaction with itself or with synthetic opponents, which is exactly what a self-play RL setup provides.

---

## Experience (E)

### Type

**Self-play simulated episodes**, played by the agent against (i) a `RandomAgent` baseline during the warm-up phase and (ii) a mix of the current agent and a pool of frozen past snapshots during the main phase.

### Structure of the experience stream

Each game produces a trajectory `((s_0, a_0, r_0), (s_1, a_1, r_1), …, (s_T, a_T, r_T))`. Each trajectory is decomposed into single-step transitions `(s_t, a_t, r_t, s_{t+1}, done, legal_mask_{t+1})` and pushed into a fixed-capacity replay buffer (FIFO eviction once `REPLAY_CAPACITY` is reached).

### Opponents faced during training

1. **Warm-up phase** (games 0 to `WARMUP_GAMES`, default 1000) — `RandomAgent`. Bootstraps the network with non-degenerate experience before introducing the moving-target problem of pure self-play.
2. **Self-play phase** (games `WARMUP_GAMES` to `TRAINING_GAMES`) — with probability `SELF_PLAY_MIX_PROB = 0.5`, the opponent is a uniformly-sampled snapshot from the snapshot pool (capped at `MAX_POOL_SIZE = 20`). Otherwise the opponent is a frozen copy of the current agent.

### Feedback type

**Indirect, delayed, and shaped.** The eventual win/loss is the canonical signal; per-step delta-score shaping (Mode 2) makes the signal dense enough that the agent can credit-assign within episodes that may exceed 100 steps. The shaping is potential-based in the sense that it sums to the terminal score difference, so the optimal policy is unchanged.

### Distribution shift over training

The opponent distribution evolves with the agent's own strength: as snapshots are added, the average difficulty of pool opponents grows. This is by design — it provides a self-curriculum without requiring a hand-crafted opponent ladder.

### Quantity

Default total experience for one board-size agent family:
* `TRAINING_GAMES = 10_000` games
* Average game length ≈ N² / 2 steps (Mode 2), bounded above by N²
* Total transitions ≈ 10_000 × N²/2 ≈ 720_000 for N = 12

That is a substantial replay buffer turnover (`REPLAY_CAPACITY = 100_000` rotates ~7 times over the run), giving each transition multiple opportunities to influence the network through minibatch sampling.

### Storage

Transitions live in an in-memory ReplayBuffer. The model and its metadata are persisted to disk only at snapshot time (`SNAPSHOT_INTERVAL = 1000` games). Training logs are appended live to `results/logs/training_log.csv`.

---

## Performance (P)

### Primary metric

**Win rate W(k) against `RandomAgent`**, evaluated every `EVAL_INTERVAL = 500` games, over `EVAL_GAMES = 200` games with a fixed seed list (so successive evaluations are directly comparable). The agent plays half the evaluation games as Player 1 and half as Player 2, so first-mover advantage cannot inflate the score.

```
W(k) = (#wins as P1 + #wins as P2) / EVAL_GAMES
```

This is the canonical "performance improves with experience" curve and the headline figure in `results/figures/win_rate.png`.

### Secondary metrics

| Metric                    | Source                                             | Why it matters                               |
|---------------------------|----------------------------------------------------|----------------------------------------------|
| Win rate vs heuristic     | Same evaluator with `HeuristicAgent` opponent      | Random is the floor; heuristic is the ceiling that a non-learning agent can reach. The DQN should clear it. |
| Elo rating                | `evaluation/elo.py` updated from snapshot-vs-snapshot tournaments | Tracks relative strength across the snapshot pool without anchoring to a baseline. |
| Mean reward per episode   | Per-episode cumulative-reward log                  | Sanity check that the reward shaping aligns with winning. |
| Mean episode length       | Per-episode step count                             | Decreasing length over training suggests faster decisive play. |
| TD loss                   | Per-gradient-step minibatch loss                   | Stability check — a runaway-or-collapsing loss curve signals divergence. |

### Hypotheses (formal)

* **H1 (improvement-with-experience)** — `W(k)` is monotonically non-decreasing in trend (allowing local noise) over the course of training and ends materially above the 50% random-baseline.
* **H2 (heuristic-beating)** — by the end of training, `win_rate_vs_heuristic ≥ 0.55` on the trained board size, i.e. the agent strictly outperforms a hand-coded 1-ply tactical opponent.
* **H3 (intra-pool dominance)** — Elo of the latest snapshot is at least 200 points above the Elo of the snapshot at game `WARMUP_GAMES`.

The TEP design is "well-posed" if H1 holds, the metric is single-valued and reproducible across seeds, and the agent cannot trivially game the metric (e.g. by stalling). All three are checked in `07_test_plan.md` Section 4.

### Performance reporting in the final report

`docs/report.md` reports:
* The final `W(k)` figure and the curve.
* `win_rate_vs_heuristic` at training end.
* Elo trajectory across the snapshot pool.
* The five required PNGs.
* A critical-analysis section answering "Did performance improve with experience?" (yes/no with supporting numbers) and "Was the problem well-posed?" (review of T, E, P against Mitchell's criteria).

---

## Why this TEP framing is appropriate for C_lines

The MDP cast is exact — C_lines is fully observable, deterministic, and turn-based, which is the textbook setting for sequential decision-making with a value- or Q-function. Self-play is the only viable experience source given that no expert data exists. The win-rate-against-random metric is operationally simple, comparable across snapshots, and directly answers the course's "did performance improve" question. Delta-score reward shaping is necessary in Mode 2 because the terminal-only reward would otherwise be far too sparse for episodes of 144 steps; shaping is potential-based so it does not bias the optimal policy.

---

## Reference

Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill, Chapter 1: "Designing a Learning System".
