# Unified Upgrade Plan — Combining Algorithm Upgrade with Human-in-the-Loop Training

**Author:** Colile Sibanda
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Plan locked; this is the binding execution plan for the upgrade work
**Supersedes:** the phasing in `09_algorithm_upgrade_plan.md` and `10_human_in_loop_training_plan.md` where they disagree

---

## 1. Purpose

`09_algorithm_upgrade_plan.md` strengthens the agent. `10_human_in_loop_training_plan.md` gives the user control over how the agent is trained and produces objective evidence of improvement. The two plans are largely orthogonal but they collide in five specific places: MCTS cost during human play, sample scarcity, benchmark cost, snapshot-schema drift, and state-encoder drift breaking older snapshots.

This document is the single execution plan: it sequences both plans together, calls out each conflict explicitly, specifies the resolution, and ends with a phase-by-phase build order that subsumes both 09 and 10.

## 2. Where the two plans agree (combine for free)

Most of the work composes cleanly:

| Item | 09 (algo upgrade) | 10 (human-loop) | Composition |
|------|-------------------|-----------------|-------------|
| Use `BaseAgent` interface | unchanged | uses it for benchmark and human sources | works as-is |
| Snapshot/registry system | unchanged | benchmark needs persistence too | reuse for benchmark "snapshots" |
| Reward function | unchanged | unchanged | same in all training phases |
| Symmetry augmentation (09-A) | required | independent | applies to all sources, including human transitions |
| Richer state channels (09-B) | required | independent | works for human transitions and benchmark games equally |
| Residual network (09-C) | required | independent | works for human transitions equally |
| Alpha-beta agent (09-E) | required | required as benchmark | one implementation, two consumers |

Alpha-beta in particular is a shared dependency — building it once satisfies both plans. This is the single biggest co-build win.

## 3. Where the two plans collide

Five real conflicts, each with a concrete remedy below.

### Conflict 3.1 — MCTS at inference is slow during human play

**The problem.** `09-D` adds MCTS-at-inference to the agent. Default `MCTS_SIMULATIONS = 200` is fine for self-play (run in batch) but is noticeable during interactive human play, especially on 12×12 with the deeper residual network. The human waits while the agent thinks. Worse, this is exactly when responsiveness matters most for user experience.

**Remedy.** Make MCTS simulations a per-context configuration, not a global constant.

```python
# config.py
MCTS_SIMS_BY_CONTEXT = {
    "self_play":    200,
    "snapshot_play":200,
    "alphabeta":    200,   # learner uses MCTS during benchmark games too
    "human_play":   80,    # ~0.3 s on 12x12 CPU, comfortable for humans
    "benchmark_eval":400,  # benchmark games for the eval CSV use stronger search
}
```

`MCTSAgent.select_action(obs, mask, context: str = "self_play")` reads its budget from this map. The opponent-context is set by the training loop on a per-game basis. Human players never wait more than a noticeable but tolerable interval.

A secondary safeguard: a hard wall-clock cap (`MCTS_MAX_THINK_SEC = 1.5`) that stops simulations early on slow machines.

### Conflict 3.2 — Human games are too few to influence a stronger network

**The problem.** A residual network with more channels (~3.5M parameters at N=12) has higher data demand. Self-play can supply this. Human games cannot — even 10 games per evening is 10 trajectories of ~100 transitions, a thousand samples against a buffer that turns over 100 000.

**Remedy.** Three-part:

1. **Weighted sampling in the buffer** (already in plan 10 §7): default weight 5.0 for human transitions, capped at 20.0.
2. **Optional demonstration buffer** (plan 10 §7.1) — second buffer at fixed 25% mixing ratio. Recommended on by default once 50+ human games are accumulated.
3. **Symmetry augmentation amplifies human data, too.** Plan 09-A applies at sample time, so each human transition turns into 8 effective samples through the dihedral group. A 10-game human session contributes the same data as an 80-game session would without augmentation.

Together these make 10 human games per evening a meaningful contribution rather than a drop in the bucket.

### Conflict 3.3 — Benchmark game cost is high if benchmark uses MCTS

**The problem.** Plan 10 calls for a benchmark game after every training game (`benchmark_every_n_games=1`). Plan 09 adds MCTS to the learner. If the benchmark uses MCTS too, every training game now triggers two MCTS-driven games — wall-clock cost doubles or worse.

**Remedy.** The benchmark agent is **alpha-beta only — no MCTS, no DQN, no learning**. It is a fixed, deterministic, cheap reference. The learner uses MCTS during benchmark games (that is fine — the learner is what we're measuring), but the benchmark itself does not.

In numbers: alpha-beta depth-4 on 8×8 plays a full game in ~0.5 s; the learner with MCTS-80 plays its half of the game in ~0.5 s too; total benchmark game cost is ~1 s. Even with `benchmark_every_n_games=1` this adds a fixed overhead per training game that is roughly the same order as the training game itself. On 12×12 the benchmark cost grows; default `benchmark_every_n_games` is raised to 5 in `config.py` for sizes ≥ 11.

### Conflict 3.4 — Snapshot schema drift breaks older snapshots

**The problem.** Plan 09 adds new fields to `SnapshotMetadata`: `state_channels`, `network_arch`. Plan 10 adds new fields too: `parent_schedule`, `human_games_seen`, `benchmark_results_summary`. Both planning passes the schema concurrently risks one party's loader rejecting the other's snapshots.

**Remedy.** Single coordinated schema bump to `MODEL_VERSION = 2` (from 1). The new schema is the union of both plans' additions, plus a back-compat loader that fills missing fields with defaults when reading older v1 snapshots:

```python
@dataclass
class SnapshotMetadata:
    # v1 fields (unchanged)
    version_id, board_size, weights_path, created_at, games_trained,
    gradient_steps, epsilon_at_freeze, parent_run_id, parent_version_id,
    win_rate_vs_random, win_rate_vs_heuristic, win_rate_vs_self, elo_rating,
    mean_episode_length, training_history, friendly_name, difficulty_band, notes

    # v2 additions (this plan)
    model_version: int = 2
    state_channels: int = 10           # plan 09
    network_arch: str = "resnet_v1"    # plan 09
    parent_schedule_path: str | None = None   # plan 10
    human_games_seen: int = 0          # plan 10
    benchmark_summary: dict | None = None     # plan 10 — {wins, losses, draws}
```

All fields have defaults so deserialising older v1 JSON works without raising. The training loop sets the new fields from runtime state.

### Conflict 3.5 — State encoder drift breaks older snapshots more deeply

**The problem.** Even with a schema-tolerant loader, plan 09-B changes the network input shape (6 → 10 channels). An older 6-channel snapshot's weights cannot be loaded into a 10-channel network at all; the input convolution's weight shape changes.

**Remedy.** Two-part:

1. **`network_arch` and `state_channels` are persisted** (resolved above). The loader inspects them and instantiates the correct legacy network (`DQNNetworkPlain`, 6 channels) for old snapshots. They keep working in the UI as historical levels.
2. **A migration script** `python -m src.versioning.migrate --from 1 --to 2` transfers an old snapshot's parameters into the new architecture as a warm start: the first input-conv layer's weights are zero-padded along the channel axis (so the new threat channels start with neutral weights), the rest of the network is copied verbatim. The migrated snapshot is registered as a new gen_NNN with `notes = "migrated from gen_MMM (v1)"`. The old snapshot stays in place — never overwritten.

Result: every old snapshot remains playable, and migration is an option, not a requirement.

## 4. What "plans 1 and 2 are not compatible" really means

The honest enumeration:

* They **are** compatible at the architectural level — they touch different files mostly, and the shared files (config, metadata, registry) take additions cleanly.
* They **are** compatible at the schedule level — once MCTS budgets are context-aware (§3.1), human play remains pleasant.
* They **are** compatible at the data level — once weighted sampling + demonstration buffer + symmetry augmentation are in place (§3.2), human-game scarcity stops mattering.
* They **are** compatible at the cost level — once the benchmark is locked to a cheap alpha-beta agent (§3.3), per-game overhead is tolerable.
* They **are** compatible at the persistence level — once `MODEL_VERSION = 2` is bumped with a union schema (§3.4) and a migration script (§3.5) is provided, old snapshots stay playable.

There are no fundamental incompatibilities. The remedies above turn what looked like five clashes into five small specifications.

## 5. Unified phase order

This is the **binding** sequence. It supersedes the standalone orderings in plans 09 and 10.

| # | Phase | Comes from | Build time | Retrain needed? |
|---|-------|-----------|-----------|-----------------|
| **U1** | Symmetry augmentation (sample-time) | 09-A | 1-2 h | No — applies to next run |
| **U2** | Alpha-beta agent + bitboard helper | 09-E | 1 day | No |
| **U3** | Benchmark logger + new CSV + plot | 10 §5 | half day | No (logs from next run) |
| **U4** | OpponentSource abstraction + parsed schedule string | 10 §3-4 | 1 day | No |
| **U5** | Weighted replay buffer + demonstration buffer skeleton | 10 §7 | half day | No |
| **U6** | Schema bump to MODEL_VERSION = 2 + back-compat loader + migration script | §3.4-3.5 | half day | No |
| **U7** | Richer state channels (10 channels) | 09-B | half day | **Yes** |
| **U8** | Residual-block network | 09-C | 1 day | **Yes** |
| **U9** | MCTS agent with context-aware sim budget | 09-D + §3.1 | 1-2 days | No (works on top of any DQN) |
| **U10** | Human-game capture in the UI (training mode banner, suppressed victory screen) | 10 §6 | half day | No |
| **U11** | Full retraining run on 8×8 with the new architecture (U1+U7+U8+U9 active) | 09 §10 | 4-8 h wall-clock | — |
| **U12** | End-to-end mixed-schedule session (the user's first interactive evening) | 10 §9.3 | 2-3 h | — |
| **U13** | Submission documentation update — plot the benchmark curve, populate the report | 10 §5.4 | half day | — |

Total engineering time: ~7-9 days of focused work to U10. The retraining wall-clock dominates after that.

## 6. Recommended first session (vertical slice end-to-end)

The smallest credible end-to-end demonstration of the unified plan, in one evening:

```bash
# 1. Build phases U1 → U6 (no model changes yet)
# 2. Run a quick schedule against the existing 8x8 DQN
python -m src.training.train --size 8 \
    --schedule "self:20,human:3,self:20" \
    --benchmark alphabeta_d4 \
    --benchmark-every 1 \
    --no-augment-yet false

# Expected:
#   - training_log.csv has 43 rows
#   - benchmark_log.csv has 43 rows
#   - Looking at benchmark_log.csv: does the rolling-mean win-rate vs alphabeta
#     trend upward (even noisily) across the 43 games?
# Stops here if the answer is "no" — the existing DQN may not yet be benefiting
# from the new symmetry/weighting. Move to U7-U9 in that case.
```

This catches integration issues with U1-U6 before the larger retraining commitment in U7-U11.

## 7. What stays unchanged from the original plans

Carrying forward verbatim:

* `01_project_brief.md` — scope, stakeholders, deliverable list (with the four new files added).
* `02_ml_methods_research.md` — DQN remains the chosen family; MCTS is an inference-time augmentation that does not change the algorithm choice for the report.
* `03_game_description.md` — game rules untouched.
* `05_ui_ux_spec.md` — frosted-glass theme untouched. The only addition is the small "Training mode" banner in U10.
* `06_versioning_spec.md` — schema is bumped per §3.4 but the file's overall structure stays.
* `07_test_plan.md` — extended with the new test files listed in plan 10 §10 and plan 09 §3.3/4.3/5.3/6.3/7.2.
* TEP definitions — unchanged. Experience now legitimately includes human games (Mitchell's framework explicitly allows this — "labelled training examples" / "supervised feedback from a teacher" / "direct sources").

## 8. Updates to TEP

Section 2 of `docs/tep_definitions.md` should be updated to reflect that experience now includes:

* Self-play episodes (as before)
* Warm-up vs `RandomAgent` (as before)
* Snapshot-pool opponents (as before)
* **Human games (new)** — the user as a teacher. Transitions weighted higher in the buffer.
* **Alpha-beta opponent games (new, optional)** — a cheap stronger-than-heuristic opponent.

The benchmark game stream is **measurement**, not experience — the agent does not train on benchmark transitions. This is important enough to call out explicitly in the TEP doc so the assessor sees the distinction.

## 9. Updates to success criteria

Both the original brief criteria and the upgrade plan's new targets remain. Adding:

* **Benchmark improvement criterion (new):** the rolling-100 win rate against the fixed alpha-beta benchmark in `benchmark_log.csv` must be higher in the final 1000 training games than in the first 1000. This is now the headline "did performance improve" criterion in the report — it is rigorous because the opponent never changed.

* **Schedule-coverage criterion (new):** the submitted run includes at least one training session with a mixed schedule containing all of `self`, `human`, `snapshot_pool`, and `alphabeta` phases. This demonstrates the full human-in-loop capability.

## 10. Updates to the report template

`docs/report.md` Section 4 (Experimental Results) needs three additions:

* A new sub-section "**Benchmark curve**" with `benchmark_curve.png` and a discussion paragraph.
* A new sub-section "**Mixed-schedule session**" documenting one end-to-end interactive session — what schedule was run, how many human games were played, and what the benchmark curve looked like during that session.
* A new sub-section "**MCTS-at-inference ablation**" — table of WR vs each baseline for the agent with and without MCTS (the same trained network, two evaluations). This is a textbook ablation result that strengthens the report.

## 11. Risk register (combined)

Picks up from `01_project_brief.md` §10 and `09 §11` and `10 §11`; new combined-plan risks:

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Engineering scope (~9 days) exceeds available time | Medium | High | Ship U1–U6 as v1.5 (no retrain needed); leave U7–U9 as v2.0 if time permits. The benchmark curve alone (U3) makes the report dramatically stronger. |
| Migration script silently corrupts a snapshot | Low | High | Migration script writes to a new gen_NNN, never overwrites; tested in `tests/test_migration.py` |
| Human-game weighting causes the agent to over-fit to the user's style | Medium | Medium | Cap human weight at 5.0 by default; demonstration buffer mix ratio capped at 25% |
| New plots are unreadable due to noise | Low | Low | Rolling-100 smoothing on all plots; raw points shown as scatter behind the smoothed line |
| User stops mid-session and the schedule state is corrupt | Low | Medium | Atomic writes for `schedule.json` and `learner_latest.pt`; documented in `docs/starter_guide.md` |

## 12. What this plan does **not** introduce

Explicitly out of scope, even after combining:

* Full AlphaZero (separate policy/value heads with MCTS-policy-as-target during training). MCTS in U9 uses the DQN as prior; AlphaZero is a follow-on project.
* MuZero, distributed self-play, GPU-optimised training.
* Curriculum auto-design or RLHF-style preference learning. The user composes the schedule.
* A second human player in the loop simultaneously.
* The mobile or web ports.

## 13. Definition of done for the unified upgrade

* All tests from plans 09 + 10 pass: `pytest tests/ -v` is green.
* A retraining run with U1 + U7 + U8 active produces a `benchmark_log.csv` that demonstrably trends upward (rolling-100 win rate vs alpha-beta climbs by ≥ 20 percentage points from first 1000 to last 1000 games on 8×8).
* The MCTS-at-inference (U9) agent, evaluated against alpha-beta over 200 fixed-seed games, wins ≥ 40% on 8×8.
* At least one end-to-end mixed-schedule session has been run successfully and is referenced in the report.
* `docs/report.md` includes the three new sub-sections from §10.
* `docs/tep_definitions.md` Experience section is updated per §8.
* Old snapshots remain loadable via the back-compat loader and playable in the UI.

---

*End of unified upgrade plan. This is the binding execution document for the upgrade work — when it contradicts plans 09 or 10, this document wins.*
