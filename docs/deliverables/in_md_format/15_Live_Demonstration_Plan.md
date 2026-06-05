# Live Demonstration Plan — C_lines (8×8, points-until-full)

**Author:** Colile Sibanda · ITRI 616
**Date:** 2026-06-01
**Run demonstrated:** `run_pts_002` (10,000 self-play games, 8×8, points-until-full)
**Target duration:** 6–8 minutes of demo within the presentation slot

This plan tells you exactly what to run, in what order, what to say while it runs, what the marker should see, and what to do if something fails. Every command has been chosen so the visible output is a clear number or a moving game. **Rehearse the whole sequence at least twice on the actual machine** before the session.

---

## 0. Pre-flight checklist (do this 15 minutes before)

Run these once, before anyone is watching, so the live portion never waits on setup.

```bash
cd "<project folder>"
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on mac/linux)
python -m pytest tests/ -q        # confirm the code base is green
python -m scripts.demo_eval --gen 10 --games 20   # warm the model, confirm it loads
```

Confirm before you start:

- [ ] The virtual environment activates and `pytest` passes.
- [ ] `python -m src.ui.app` opens the game window.
- [ ] The figures exist: `results/size_08/run_pts_002/figures/combined_progress.png`, `elo_curve.png`, `benchmark_vs_alphabeta.png`.
- [ ] Have the three figures **open in an image viewer already** as a fallback.
- [ ] Screen resolution / projector tested; font size in the terminal bumped up so output is readable from the back.
- [ ] Close email, chat, notifications.

---

## 1. Demonstration arc (the story the marker should leave with)

> "I posed C_lines as a reinforcement-learning control problem. I trained one agent by self-play for 10,000 games. It went from random-level to beating a 4-ply search 100% of the time, and I can show you the learned agent playing right now."

Three beats, mapped to the rubric:

1. **It learned** — show the improvement curves (Experimental evaluation, 30%).
2. **It's genuinely strong** — produce a live win-rate number against a real opponent (Evaluation + Algorithm, 30%+30%).
3. **It plays like a strategist** — let the marker watch (or play) the trained agent (Critical analysis / "is it well-posed", 25%).

---

## 2. Step-by-step demo

### Step 1 — Show the headline result (1.5 min)

Open `combined_progress.png` and `elo_curve.png` full-screen.

**Say:** "This is one training run. Blue and green are win rates versus a random and a heuristic opponent; the purple dashed line is the agent's Elo rating. All three climb together from random-level to ceiling. The Elo rises monotonically from 796 to 1190 — that single curve is the 'performance improves with experience' claim the assignment asks for."

**Marker sees:** a clean, monotone learning curve. This is your strongest evaluation evidence and it can't fail live (it's a static image).

### Step 2 — The decisive benchmark (1 min)

Open `benchmark_vs_alphabeta.png`.

**Say:** "The hardest test is the opponent the agent never trained against — a 4-ply alpha-beta search. Early on it wins about 19% of games and loses on score. By the end it wins ~100% and the average score margin flips from −0.83 to +1.17. It learned to out-play a principled lookahead search purely from self-play."

**Marker sees:** the reversal from losing to winning against a non-trivial adversary.

### Step 3 — Prove the number live (2 min)

In the terminal:

```bash
python -m scripts.demo_eval --gen 10 --games 50
```

Expected output (numbers will vary slightly by run):

```
Snapshot: gen_010  (run run_pts_002, 8x8, points-full)
Win rate vs Random:    100%   (50 games)
Win rate vs Heuristic: 100%   (50 games)
```

**Say:** "That's the final snapshot playing 50 fresh games live, not numbers from a log. It's effectively perfect against both fixed opponents."

*Optional, if time and patience allow (alpha-beta is slower — ~1–2 min for 16 games):*

```bash
python -m scripts.demo_eval --gen 10 --games 16 --alphabeta
```

**Marker sees:** a result generated in front of them — this defeats any suspicion that the figures were cherry-picked.

### Step 4 — Watch it play (2 min)

```bash
python -m src.ui.app
```

Navigate: choose the 8×8 board → points-until-full mode → the **Level Select** screen, which lists the ten trained snapshots as difficulty cards (Improver → Champion, read live from the model registry).

Do **one** of these (pick in advance):

- **(A) Easy vs Hard, agent vs agent / you watch.** Pick a low snapshot (e.g. `gen_001`, "Improver") and mention it, then pick the top snapshot (`gen_010`, "Champion") and let it play. Point out that the strong snapshot builds long lines and blocks yours.
- **(B) You play the Champion.** Play a few moves against `gen_010`. Narrate one move where it blocks your line or extends its own — concrete evidence of learned strategy.

**Say:** "Each difficulty level is literally a checkpoint from the same training run — the registry assigns the difficulty band from the snapshot's measured Elo and win rate, so 'harder' means 'later in learning'."

**Marker sees:** the abstraction made tangible — the curve from Step 1 is the same thing they're now playing against.

### Step 5 (optional, only if you have ≥3 spare minutes) — Show it learning live

```bash
python -m src.training.train --games 300 --size 8 --run-id demo_live
```

**Say:** "A few hundred games is too short to reach skill, but you can see the pipeline running: epsilon decaying, loss updating, snapshots being written, evaluation every 100 games." Then **Ctrl-C** to stop cleanly.

**Marker sees:** the training machinery is real and reproducible, not a one-off artefact. *Skip this if you are tight on time — Steps 1–4 are the core.*

---

## 3. Timing budget

| Step | Content | Time |
|---|---|---|
| 1 | Improvement curves (Elo + win rate) | 1.5 min |
| 2 | Benchmark reversal vs alpha-beta | 1.0 min |
| 3 | Live evaluation numbers | 2.0 min |
| 4 | Watch / play the trained agent | 2.0 min |
| 5 | (Optional) live training pipeline | +2–3 min |
| — | **Core total** | **~6.5 min** |

If you have only 4 minutes: do Steps 1, 3, 4. The static curve + a live number + watching it play is enough to carry every rubric line.

---

## 4. Fallback plan (assume something will break)

| If this fails… | Do this instead |
|---|---|
| `demo_eval` errors / model won't load | Switch to the pre-opened figures; quote the numbers from `training_log.csv` (final row: 100% vs random, 100% vs heuristic, Elo 1190). |
| PyGame window won't open (no display / driver) | Show `benchmark_vs_alphabeta.png` and `combined_progress.png`; describe the difficulty levels from `models/size_08/run_pts_002/registry.json`. |
| Terminal output too slow / alpha-beta hangs | Ctrl-C, drop the `--alphabeta` flag, run the fast `--games 50` version. |
| Projector cuts the terminal off | Everything you need is also in the three figures — narrate from those. |
| Total tech failure | The report (`14_Final_Project_Report_run_pts_002.md`) contains every figure and table; present from it directly. |

**Golden rule:** the static figures are your safety net. They contain the entire evaluation story and cannot crash. Never let a failed command cost you more than 15 seconds — fall back and keep talking.

---

## 5. One-paragraph script to memorise (in case you go fully manual)

> "C_lines is an original 8×8 line-scoring game I posed as a reinforcement-learning control problem and solved with a self-play Deep Q-Network. Over one 10,000-game run the agent's Elo rose monotonically from 796 to 1190; its win rate against random and heuristic opponents reached 100%; and against a 4-ply alpha-beta search it never trained on, it went from winning 19% of games to 100%, flipping the score margin from −0.83 to +1.17. Each of the ten difficulty levels you can play against is a checkpoint from that same run, so the difficulty you face is literally the agent's stage of learning."

---

## 6. Commands quick-reference card (print this)

```bash
# activate
.venv\Scripts\activate

# 1. confirm green
python -m pytest tests/ -q

# 2. live win-rate vs Random + Heuristic (fast)
python -m scripts.demo_eval --gen 10 --games 50

# 3. live win-rate including Alpha-Beta d4 (slower)
python -m scripts.demo_eval --gen 10 --games 16 --alphabeta

# 4. open the game (Level Select shows the 10 snapshots)
python -m src.ui.app

# 5. (optional) show training pipeline live, then Ctrl-C
python -m src.training.train --games 300 --size 8 --run-id demo_live
```
