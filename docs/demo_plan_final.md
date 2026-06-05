# Live Demo Plan — Final Version
## ITRI 616 Presentation

---

## Pre-Demo Setup (do before entering the room)

```
1. Terminal open at project root
2. VS Code open with:
   - results/size_08/run_pts_002/training_log.csv
   - results/size_08/run_pts_002/benchmark_log.csv
3. Test: python -m src.ui.app  (ensure it launches and gen_010 loads)
4. Close all other applications
5. Font size in VS Code: 16pt minimum (visible to audience)
```

---

## Demo Script (fits inside the presentation)

### Block 1 — Training Evidence (2 min)

**Open training_log.csv. Scroll slowly top to bottom while talking.**

Rows to highlight:
| Row | What to say |
|---|---|
| Row 2 (game 0) | "Starts here. 52.5% vs random, Elo 796. Barely above a coin flip." |
| Row 32 (game 3100) | "Elo crosses 900 — stronger than the heuristic agent." |
| Row 72 (game 7000) | "100% vs random, 100% vs heuristic, Elo 1,092." |
| Last row (game 9999) | "Final. 100%/100%, Elo 1,190. Every row in the Elo column is higher than the last." |

**Then switch to benchmark_log.csv.**

| Row | What to say |
|---|---|
| Row 2 (game 0) | "18.8% against alpha-beta tree search." |
| Row 24 (game 2300) | "50% — first time it matches tree search." |
| Row 59 (game 5700) | "100% — wins every game." |
| Rows 67–71 | "Four consecutive perfect checks." |

---

### Block 2 — Live Game (5 min)

```bash
python -m src.ui.app
```

**Step-by-step:**

1. Main menu → click **8×8**
2. Click **Points Until Full**
3. Level select appears
   - "Each card is a real training checkpoint. Labels from measured win rates."

4. Click **gen_001 (Easy / Improver)**
   - Play 4 moves each. Let it play weakly.
   - "1,000 games. Legal but not strategic. Doesn't block, doesn't extend."

5. Back → click **gen_010 (Master / Champion)**
   - Play 6–8 moves. Narrate each AI move.
   - Point at: opponent blocking your line → "It learned blocking is valuable."
   - Point at: opponent extending diagonally → "Fork threat. Learned from self-play."

**If AI beats you:** "Good — that is the intended result."
**If you are winning:** "I know this particular opening. It has a known weakness I discuss in limitations."

---

## Fallback Plan (if UI doesn't launch)

1. Open training_log.csv — show Elo column from top to bottom. "This is the evidence."
2. Open benchmark_log.csv — show win_rate column from 0.19 to 1.00.
3. Say: "The game is demonstrated in the code repository at the link in my report."

---

## What each demo moment proves for each rubric

| Rubric criterion | Demo moment |
|---|---|
| **Problem definition / TEP** | "The agent generates its own experience — there is no human data" (say during training_log walkthrough) |
| **Technical understanding** | Show the Elo column: "This is updating every 100 games against two fixed reference agents using the Elo formula" |
| **Results and analysis** | training_log rows 0→9999. "Monotone increase in Elo. 796 to 1,190." |
| **Communication** | Narrate every AI move during live game. Keep sentences short. |
| **Critical thinking (Q&A)** | When asked about the 0% at game 8300: "That is the limitation I disclosed." |

---

## Numbers to memorise

| Metric | Start | End |
|---|---|---|
| Win rate vs random | 52.5% | 100% |
| Win rate vs heuristic | 46% | 100% |
| Elo | 796 | 1,190 |
| Benchmark vs alpha-beta | 18.8% | mostly 90–100% |
| First beats alpha-beta 50/50 | game 2,300 | — |
| First 100% benchmark | game 5,700 | — |
| Total games trained | 10,000 | — |
| Training duration | ~29 hours | — |
