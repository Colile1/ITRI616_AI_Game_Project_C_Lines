# ML Training Approach — Improvement Plan & Implementation Guide

**Project:** C_lines (ITRI 616) · Author: *----* *S----*
**Plan written:** 2026-05-31
**Companion to:** `12_ML_Training_Analysis_Report.md`
**Status:** Proposal. This document does not modify any source file. Every change below is described with the file to touch and a code sketch, so it can be applied deliberately.

---

## 0. How to read this plan

The work is sequenced in four phases, ordered by *return on effort*. Phase 0 is small, surgical, and is expected to produce most of the gain because it repairs the learning rule itself. Phases 1–2 make the run stable and the evidence trustworthy. Phase 3 is the high-ceiling, higher-effort upgrade that turns the existing residual-net + MCTS into a proper AlphaZero-style learner.

Each item lists: the **problem** (cross-referenced to the analysis report §), the **change**, the **file(s)**, a **code sketch**, and the **metric that confirms it worked**. Do them one at a time and re-run a short smoke test between each so the effect of every change is isolated.

Golden rule for this project: **change one thing, run a 1,500–2,000 game smoke run, look at win-rate-vs-random.** A correct agent should pass ~90% vs random quickly. That single curve is the fastest oracle you have.

---

## Phase 0 — Correctness of the learning rule (do these first)

### P0.1 — Fix the TD target sign (negamax) — *critical* (report §5.1)

**Problem.** `next_state` is encoded from the opponent's perspective, so `max_a Q(s′,a)` is the opponent's value. The target currently **adds** it; it must **subtract** it.

**File.** `src/agents/dqn_agent.py`, `update()`.

**Change.** Negate the bootstrap term:

```python
with torch.no_grad():
    q_next = self._target(next_states)
    q_next[~legal_next] = _NEG_INF
    max_q_next = q_next.max(dim=1).values
    # Two-player zero-sum: s' is the OPPONENT's turn → negate the bootstrap.
    targets = rewards - GAMMA * max_q_next * (1.0 - dones)
```

**Confirm.** Re-run a 2,000-game smoke run on size 8, points-full. Win rate vs random should rise past ~0.85 and *keep climbing* instead of oscillating around 0.5. This is the make-or-break change.

> Note: P0.1 and P0.2 should be applied together — Double DQN's action selection must use the same negamax convention (see P0.2 sketch).

### P0.2 — Double DQN action selection (report §5.2)

**Problem.** `max` over the target net for both selection and evaluation overestimates values.

**File.** `src/agents/dqn_agent.py`, `update()`.

**Change.** Select the argmax with the **online** net, evaluate it with the **target** net, then negate (per P0.1):

```python
with torch.no_grad():
    # action chosen by ONLINE net (masked)
    q_next_online = self._online(next_states)
    q_next_online[~legal_next] = _NEG_INF
    next_actions = q_next_online.argmax(dim=1, keepdim=True)
    # value READ from TARGET net at that action
    q_next_target = self._target(next_states)
    next_q = q_next_target.gather(1, next_actions).squeeze(1)
    targets = rewards - GAMMA * next_q * (1.0 - dones)
```

**Confirm.** Loss curve is smoother; late-training collapse (as in `run_pts_001`) disappears.

### P0.3 — Huber (smooth-L1) loss (report §5.3)

**File.** `src/agents/dqn_agent.py`, `update()`.

**Change.**

```python
loss = nn.functional.smooth_l1_loss(q_pred_actions, targets)
```

**Confirm.** Lower loss variance, no change to the win-rate trend other than smoother.

### P0.4 — Align first-to-four reward with the objective (report §5.4)

**Problem.** First-to-four shaping rewards line-building and *survival*; the objective is to complete a 4-line first. The survival bonus actively rewards stalling.

**File.** `src/game/env.py`, `_compute_reward`, plus the `FTF_*` constants in `src/config.py`.

**Change (recommended: the project's own "Option C", completion-focused).** In first-to-four mode, drop the per-step survival bonus and keep only a small reward for *creating an own open-4 threat* and a matching penalty for *allowing an opponent open-4*, with the terminal win/loss dominating:

```python
# first_to_four, non-terminal:
own_open4_delta, opp_open4_delta = ...   # from compute_threats / open-4 count
return (own_open4_delta - opp_open4_delta) * FTF_THREAT_SCALE   # no survival term
```

Set `FTF_SURVIVAL_SCALE = 0.0` (keep the constant for compatibility). Keep the graduated early-loss penalty.

**Confirm.** In a 2,000-game ftf smoke run, win rate vs heuristic should leave 0% within the first ~1,500 games. If it is still 0% at 3,000, the shaping is still wrong.

**Phase-0 acceptance gate:** a fresh size-8 points-full run reaches ≥ 0.9 vs random and shows a clearly rising vs-heuristic curve. Do not proceed until this holds — everything downstream assumes a correct update rule.

---

## Phase 1 — Trustworthy evaluation & stable self-play

### P1.1 — Wire up Elo as the primary skill curve (report §5.8)

**Problem.** `elo.py` exists but `elo_rating` is always null; win-rate vs fixed opponents saturates/floors and is noisy.

**Files.** `src/evaluation/elo.py` (already present), `src/training/train.py` (evaluation block + snapshot metadata).

**Change.** At each eval, run a small round-robin of the current agent vs a fixed anchor set (random, heuristic, alpha-beta-d2, and 2–3 pool snapshots), update Elo, and write `elo_rating` into the snapshot metadata and the training log. Plot Elo over games as the headline "improves with experience" figure.

**Confirm.** A smooth, monotone-ish Elo curve — the single most rubric-relevant graph for the 30% evaluation criterion.

### P1.2 — Report win rate with confidence intervals (report §5.8)

**File.** `src/evaluation/evaluator.py`.

**Change.** Return a Wilson 95% CI alongside each win rate (`p ± 1.96·√(p(1−p)/n)` as a first approximation, Wilson preferred). Use the lower CI bound — not the point estimate — to drive the plateau LR scheduler and best-model selection so neither is fooled by noise.

**Confirm.** "Best model" stops flipping between snapshots that are statistically tied.

### P1.3 — Replace the uninformative episode-length metric (report §4)

**File.** `src/training/train.py` eval; `src/training/benchmark_logger.py`.

**Change.** In points-full mode the board always fills, so log **final score margin** instead. In first-to-four mode log **moves-to-win** (a falling trend = genuine improvement). Keep `mean_ep_len` only where it varies.

### P1.4 — Anchor the self-play pool and gate by rating, not raw WR (report §5.6)

**File.** `src/training/train.py` (`_sample_pool_opponent`, pool-admission block).

**Change.** Always keep `RandomAgent`, `HeuristicAgent`, and `AlphaBetaAgent(depth=2)` as permanent low-probability opponents so the curriculum never collapses into "weak clone vs weak clone." Admit a snapshot to the pool on **relative Elo gain** over the current pool median rather than the noisy "WR-vs-random ≥ 0.5" test.

**Confirm.** Win-rate variance between consecutive evals drops; no more stretches where the agent only ever faces near-random clones.

---

## Phase 2 — Sample efficiency

### P2.1 — Prioritized Experience Replay by TD error (report §5.5)

**Problem.** Sampling is by source weight only; the agent never focuses on the transitions it currently gets most wrong, and there is no IS correction.

**Files.** `src/training/replay_buffer.py`, `src/agents/dqn_agent.py`.

**Change.** Store a priority `pᵢ = (|TD error| + ε)^α` per transition (combine multiplicatively with the existing source weight so demonstrations stay emphasised), sample proportional to priority, and apply the importance-sampling correction `wᵢ = (1/(N·P(i)))^β` to the per-sample loss, with β annealed 0.4 → 1.0. Update priorities with the fresh TD error after each batch. A sum-tree is the efficient structure, but a simple proportional implementation is fine at 100k capacity.

**Confirm.** Faster rise of the vs-random / Elo curve per game (better sample efficiency), especially early.

### P2.2 — n-step returns (report §5.7)

**File.** `src/training/self_play.py` (accumulate n-step transitions) and `dqn_agent.update` (use `γ^n`).

**Change.** Use n = 3 returns so end-of-game outcomes propagate three plies per update instead of one. With negamax targets this also makes the first-to-four "synthetic loss transition" hack (`train.py`) unnecessary — the terminal signal reaches the loser's earlier states naturally.

**Confirm.** Sparse-reward first-to-four learns faster; you can delete the synthetic-transition block and verify win rate is unaffected or better.

### P2.3 — Throughput & reproducibility (report §5.10)

**Files.** `src/training/train.py`, `src/config.py`.

**Change.** Add a `--seed` flag and seed `torch`/`numpy`/`random`; report results as mean ± std over ≥ 3 seeds so real improvement is separable from run-to-run noise. If feasible, vectorise self-play (batch several environments per gradient step) to lift the ~1,400 games/hour ceiling and allow longer runs.

**Confirm.** Two runs with the same seed match; the multi-seed band is narrow enough to claim improvement with confidence.

---

## Phase 3 — High-ceiling upgrade (optional, biggest payoff)

### P3.1 — AlphaZero-style policy + value head with MCTS targets (report §5.9)

**Why.** The project already has the two expensive pieces: a residual network and an MCTS agent. AlphaZero-class methods are the strongest known approach for small two-player perfect-information board games, and they fit this codebase with minimal new infrastructure.

**Files.** `src/training/network.py` (add a policy head: a 1×1 conv → softmax over N² cells, alongside the existing value output collapsed to a single scalar via a value head); `src/agents/mcts_agent.py` (already produces visit counts); a new training target.

**Change (outline).**
1. Convert the value output to a single scalar `v ∈ [−1, 1]` (tanh) and add a policy head `p` over cells.
2. During self-play, run MCTS at each move and record the **visit-count distribution π** and the eventual **game outcome z**.
3. Train with the AlphaZero loss: `(z − v)² − πᵀ log p + c‖θ‖²` (value regression + policy cross-entropy + L2). No replay-sign subtlety — outcomes are stored from each state's own perspective.
4. The snapshot/pool/Elo/benchmark machinery is reused unchanged.

**Confirm.** Should beat alpha-beta depth-4 (which the DQN never did) and produce the cleanest improves-with-experience curve. This is the strongest single thing you could show for the evaluation and critical-analysis marks — but only attempt it once Phase 0 has proven the rest of the pipeline is sound.

---

## Verification plan (applies to every phase)

1. **Unit test the target maths.** Add `tests/test_dqn_target.py`: construct a tiny 2-ply position with a known winning reply, compute the target by hand, and assert `update()`'s target matches (correct sign, correct Double-DQN action, correct discount). This pins P0.1–P0.2 permanently.
2. **Smoke run between every change.** 1,500–2,000 games, size 8, fixed seed; assert win-rate-vs-random crosses a threshold (e.g. ≥ 0.85) by the end. Wire this into `tests/integration/test_train_smoke.py` as a slow-marked test.
3. **Ablation table for the report.** Run {baseline, +P0.1, +P0.2, +P0.3, +P0.4} for a fixed budget and tabulate final Elo / WR-vs-random / WR-vs-heuristic. This *is* your experimental-evaluation deliverable and directly evidences "performance improves with experience."
4. **Multi-seed final run.** Report the chosen configuration over ≥ 3 seeds with confidence bands.

---

## Suggested order & effort

| Step | Effort | Expected impact |
|---|---|---|
| P0.1 negamax sign | ~1 line | **Decisive** — fixes the core bug |
| P0.2 Double DQN | ~6 lines | High — stability, no late collapse |
| P0.3 Huber loss | ~1 line | Medium — smoother training |
| P0.4 ftf reward | ~15 lines | High (ftf mode) — correct objective |
| P1.1 Elo curve | small | High for marks — the headline graph |
| P1.2 CIs on WR | small | Medium — trustworthy decisions |
| P1.3 better metric | small | Medium — readable curves |
| P1.4 anchored pool | medium | Medium — stable curriculum |
| P2.1 PER | medium | Medium — sample efficiency |
| P2.2 n-step | medium | Medium — sparse-reward credit |
| P2.3 seeds/throughput | small–large | Medium — credible evidence |
| P3.1 AlphaZero head | large | **Highest ceiling** — beats alpha-beta |

Start at the top. The first four items are short and are expected to turn the flat, noisy curves into the monotone improvement the assignment is asking you to demonstrate.

---

## Sources informing the recommendations

- Two-player TD target / negamax sign correction: discussion of self-play DQN for board games — chiamp, *dqn-boardgames* (GitHub); arXiv 1905.07102 (*Mastering Sungka from Random Play*).
- Double DQN, Dueling, Prioritized Experience Replay reducing overestimation and improving sample efficiency: Simonini, *Improvements in Deep Q-Learning* (freeCodeCamp); MDPI *Electronics* 13(12):2423.
