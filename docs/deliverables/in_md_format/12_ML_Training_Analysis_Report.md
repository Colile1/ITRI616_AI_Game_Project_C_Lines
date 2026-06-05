# ML Training Approach — Current-State Analysis Report

**Project:** C_lines — open-placement four-in-a-row variant with a self-play DQN agent
**Author of project:** *----* *S----* · NWU · ITRI 616
**Report written:** 2026-05-31
**Scope:** This report reviews the project *as it currently stands*, with the focus on the machine-learning model and the training approach. It is a standalone document and does not modify any existing file. A companion document, `13_ML_Training_Improvement_Plan.md`, proposes and sequences the fixes.

---

## 1. Executive summary

C_lines is an unusually well-engineered student RL project. The infrastructure around the model — versioning, snapshot registry, run management, logging, symmetry augmentation, a residual network, an MCTS inference agent, an alpha-beta benchmark, and a human-in-the-loop training mode — is of a standard well above what the brief requires. The TEP framing is sound and the problem is correctly identified as a sequential decision / control problem solved with value-based reinforcement learning.

The learning *results*, however, do not match the quality of the scaffolding. Across three representative runs the agent never reliably beats a random opponent, never beats the alpha-beta benchmark, and shows large, non-converging variance in win rate. The single most likely root cause is a **sign error in the temporal-difference (TD) target** for this two-player game: the bootstrap term is added when it should be subtracted. That one defect is sufficient to explain most of the observed behaviour, and it is compounded by vanilla-DQN value overestimation, reward shaping that targets the wrong objective in one game mode, and evaluation metrics too noisy to steer the run. None of these are infrastructure problems — they are in the core learning rule — which is good news, because the expensive parts are already built.

---

## 2. How the learning problem is currently posed (TEP)

**Task.** Choose a placement (a cell on an N×N grid, N ∈ {8…12}) that maximises the chance of winning, under two modes: *first-to-four* (first to complete a line of length ≥ 4 wins) and *points-until-full* (board fills, highest line-score wins). Problem category: sequential **decision-making / control**, not classification or prediction.

**Experience.** Self-play episodes, with a warm-up phase against `RandomAgent` and `HeuristicAgent`, then a mix of self-clones and a pool of past snapshots. Experience is stored in a replay buffer and reused for several gradient steps per game. Human games and demonstrations can be injected with higher sampling weight.

**Performance.** Win rate vs `RandomAgent` and vs `HeuristicAgent` (every 100 games), an alpha-beta benchmark check, mean episode length, training loss, and a best-model checkpoint keyed on win-rate-vs-heuristic.

The framing is correct and matches Mitchell's well-posed-learning-problem template. The weaknesses below are in the *realisation*, not the framing.

---

## 3. Model and pipeline as built

**Network (`src/training/network.py`).** A residual Q-network: a conv stem, four residual blocks at 64 channels with BatchNorm, and a Q-head producing one value per cell (N² outputs). This is an AlphaZero-shaped *value* network — but with no policy head. A legacy plain CNN is retained for back-compat.

**State encoding (`src/game/encoding.py`).** Ten channels from the current player's perspective: own/opponent/empty, own/opponent open-3 threats, turn-progress, own/opponent open-4 threats, and immediate-win / immediate-loss cells. This is a rich, well-chosen representation that hands the network most of the tactical features explicitly.

**Agent (`src/agents/dqn_agent.py`).** Epsilon-greedy action selection with legal-move masking, an online + target network, Adam, gradient clipping, MSE loss.

**Training loop (`src/training/train.py`).** Linear epsilon decay over 7,000 games; warm-up vs random/heuristic; self-play mix with a recency-biased snapshot pool; weighted replay; D4 symmetry augmentation; plateau-based LR halving; per-100-game evaluation; best-model checkpointing; snapshotting every 1,000 games with a "WR-vs-random ≥ 0.5" pool-admission gate; optional alpha-beta benchmark; a schedule-based variant and a PyGame human-in-the-loop variant.

**Replay buffer (`src/training/replay_buffer.py`).** A circular buffer with *source-based* importance weights (human/demo/alpha-beta transitions sampled more often) and random D4 augmentation at sample time.

The engineering here is genuinely strong and modular; the issues are concentrated in a handful of functions.

---

## 4. What the results actually show

Numbers are taken directly from the committed training logs.

**`size_09/run_001` (points-full, 10,000 games).** Win rate vs random oscillates between ~0.47 and ~0.83 the whole way and the *final* snapshot (`gen_020`) records **0.47 vs random** — at or below chance — with 0.54 vs heuristic. There is no monotone climb; the curve is noise around a low ceiling.

**`size_08/run_pts_001` (points-full, 10,000 games).** The *final* model is the *worst* of the run (0.06 vs random, 0.10 vs heuristic at game 9,999), classic catastrophic forgetting / divergence late in training.

**`size_08/run_ftf_002` (first-to-four, 10,000 games).** Win rate vs heuristic is **0% for nine of ten snapshots** and 6% at the end. The agent **lost all 101 benchmark games** against alpha-beta depth-4, by an almost-identical margin every time.

**Mean episode length** is reported as a flat constant (81 for 9×9, 64 for 8×8) in points-full mode — i.e. the board simply fills every game — so it carries no learning signal in that mode.

**Elo** columns exist in the metadata and an `elo.py` module is present, but `elo_rating` is `null` in every snapshot; the rating system is never wired into a run.

A correctly-learning agent on a tactical game with this much feature engineering should approach ~100% vs random within a couple of thousand games. The observed behaviour — near-chance, high-variance, late-training collapse, total failure vs a 4-ply search — is the fingerprint of an incorrect learning target rather than of "not enough training."

---

## 5. Root-cause analysis of the training approach

### 5.1 The TD target has the wrong sign for a two-player game (critical)

In `DQNAgent.update` the target is

```python
targets = rewards + GAMMA * max_q_next * (1.0 - dones)
```

But look at where `next_state` comes from. In `play_episode` the stored `next_state` is the observation returned by `env.step` *after the acting player's move*, and `env._obs()` always encodes the board **from the perspective of whoever is to move next** — i.e. the **opponent**. So `max_q_next = max_a Q(s', a)` estimates *the opponent's* best continuation value. Adding it means the agent treats "a position that is great for my opponent" as "great for me." In a zero-sum, alternating-turn game the bootstrap must be **negated**:

```
target = r − γ · max_a Q(s′, a)        (negamax / two-player TD)
```

This is the standard correction for self-play value learning in two-player zero-sum games: one player's value is the negative of the other's, so the TD error becomes `(r − γ max Q(s′) − Q(s,a))²`. With the sign as currently written, every bootstrap pushes the value function in the wrong direction; the only correct signal the agent receives is the terminal reward, and even that is then propagated backwards with the wrong sign. This single defect is consistent with *all* of the symptoms in §4.

### 5.2 Vanilla DQN overestimation, no Double-DQN decoupling

Even with the sign fixed, the target uses `max` over the *target* network for both action selection and evaluation. This is the well-documented overestimation bias of vanilla DQN. The standard remedy — Double DQN, where the *online* net picks the argmax action and the *target* net evaluates it — reduces this bias and improves stability. Combined with the sign error, the current setup is close to a worst case for stable value learning, which fits the late-run divergence seen in `run_pts_001`.

### 5.3 MSE loss is fragile to outlier targets

The loss is plain `mse_loss`. With bootstrapped targets that can spike (especially before the sign is fixed), squared error lets a few large TD errors dominate the gradient. Huber / smooth-L1 loss is the usual choice for DQN for exactly this reason.

### 5.4 Reward shaping optimises the wrong objective in first-to-four mode

`env._compute_reward` applies delta-*score* shaping in points-full mode (correct: the objective *is* cumulative score) but in first-to-four it mixes a threat-delta term with a small per-step survival bonus. The project's own `run_ftf_002` post-mortem already identifies that the agent spent thousands of games optimising line-building rather than racing to complete a 4-line. The survival bonus in particular rewards *prolonging* games, which is directly opposed to the first-to-four objective.

### 5.5 Replay "weights" are source-based, not learning-based

The buffer over-samples by *origin* (human, demo, alpha-beta) but never by *how much a transition would teach the network* (its TD error). It is not Prioritized Experience Replay. Moreover, the non-uniform sampling is applied with **no importance-sampling correction** in the loss, so it biases the value estimate toward over-weighted sources. For the source-weighting use-case that bias may be acceptable, but there is no mechanism to focus learning on the transitions the agent is currently getting wrong.

### 5.6 The self-play opponent pool can be a pool of weak players

A snapshot is admitted to the opponent pool only if its win rate vs random ≥ 0.50. Given that the agent barely clears that bar, the pool is frequently populated with near-random clones, so "self-play" provides little curricular pressure. There is no permanent anchor opponent (random / heuristic / alpha-beta always present) and admission is gated on a noisy 200-game estimate rather than a relative rating.

### 5.7 The first-to-four "synthetic loss transition" is a patch over a credit-assignment gap

Because the loser never makes the final move in first-to-four, the code manually pushes a synthetic terminal transition carrying the loss reward. This works, but it is a symptom of sparse terminal credit. n-step returns (or the value-correct negamax target) would propagate end-of-game outcomes far more naturally than a hand-inserted transition.

### 5.8 Evaluation is too noisy to steer training, and Elo is unused

Win rate is estimated on 100–200 games with no confidence interval, against opponents (random, deterministic heuristic) that saturate or floor quickly. The plateau-based LR scheduler and the best-model checkpoint both key off this noisy number, so LR decisions and "best" selection are partly driven by sampling noise. The Elo machinery that would give a smooth, comparable skill curve across snapshots is built but never connected.

### 5.9 The architecture is AlphaZero-shaped but trained as plain DQN

The project already has a residual net *and* an MCTS agent at inference. It is one design decision away from an AlphaZero-style policy+value learner — which is exactly the family that excels on small two-player board games — yet training is value-only DQN and the MCTS search is used only to *play*, never to *generate improved training targets*. This is a large unused ceiling, noted here as an opportunity rather than a defect.

### 5.10 Minor engineering notes

- BatchNorm with `.eval()/.train()` toggled on every single greedy action call adds overhead and subtle train/eval drift; acceptable but worth revisiting.
- All training is single-environment, CPU-only (~1,400 games/hour). 10,000 games is modest for board-game RL; throughput limits how much the above fixes can be validated.
- No seed control or multi-seed reporting, so run-to-run variance (which is clearly large) cannot be separated from real improvement.

---

## 6. Assessment against the grading rubric

| Rubric area | Current state |
|---|---|
| **TEP definitions (15%)** | Strong. Task/Experience/Performance are explicit and correctly categorised. |
| **Algorithm implementation (30%)** | Ambitious and modular, but the core learning rule contains a sign error that undermines convergence. High effort, partially-correct realisation. |
| **Experimental evaluation (30%)** | Plenty of logging and plots, but the headline metrics are too noisy and the curves do not demonstrate the required "performance improves with experience" — in two of three runs the final model is among the weakest. |
| **Critical analysis (25%)** | Already good — the per-run post-mortems are honest and specific. This report and the companion plan extend that. |
| **Code quality (10%)** | Excellent: clear modules, docstrings, config-as-single-source-of-truth, tests. |

The decisive gap is between the quality of the engineering and the absence of a clean "improves with experience" curve. Fixing §5.1–§5.4 is what converts the existing work into evidence that satisfies the 30% evaluation criterion.

---

## 7. Conclusion

The project is built well; it is *learning* wrongly. The priority is not more compute or more features — it is correctness of the value-update rule (negamax sign, Double DQN, Huber loss), a reward signal aligned to each mode's objective, and an evaluation that is precise enough to trust. With those in place the existing snapshot/pool/benchmark/Elo infrastructure should, for the first time, produce the monotone skill curve the assignment asks for. Concrete, sequenced, code-level steps are in `13_ML_Training_Improvement_Plan.md`.
