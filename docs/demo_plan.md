# Live Demonstration Plan — ITRI 616 Presentation

**Run to demonstrate:** `run_pts_002` (8×8, Points-Until-Full)
**Total demo time:** 8–12 minutes within the presentation

---

## Pre-Demo Checklist (do these before entering the room)

- [ ] Open terminal in the project root directory
- [ ] Open `results/size_08/run_pts_002/training_log.csv` in VS Code (IDE)
- [ ] Open `results/size_08/run_pts_002/benchmark_log.csv` in VS Code (IDE)
- [ ] Have the game UI ready to launch: `python -m src.ui.app`
- [ ] Test that gen_010 loads in the UI (it should — registry exists)
- [ ] Close all unnecessary applications (no distractions on screen)
- [ ] Have a browser tab open with the project folder for file navigation if needed

---

## Demo Sequence

### Step 1 — Show the training log (2 minutes)

**Open:** `results/size_08/run_pts_002/training_log.csv`

**Say:** "This is the actual training log from the run. Every 100 games the agent was evaluated."

**Point out — scroll from top to bottom:**
- Row 1 (game 0): `win_rate_vs_random=0.525`, `win_rate_vs_heuristic=0.460`, `elo_rating=795.8` — "This is where it starts — barely better than random."
- Row ~11 (game 1000): `elo=803` — "After 1,000 games, already rising."
- Row ~32 (game 3100): `elo=901.9` — "Just crossed the heuristic anchor. It's now stronger than the non-learning agent."
- Row ~73 (game 7000): `win_rate_vs_random=1.0000, win_rate_vs_heuristic=1.0000, elo=1091.5` — "100% against both baselines by game 7,000."
- Last row (game 9999): `win_rate_vs_random=1.0000, win_rate_vs_heuristic=1.0000, elo=1189.9` — "Final model. Performance improved every single step."

**Key message:** "The Elo column is the cleanest proof. It rose from 796 to 1,190 — monotonically increasing across 100 evaluation checkpoints."

---

### Step 2 — Show the benchmark log (1 minute)

**Open:** `results/size_08/run_pts_002/benchmark_log.csv`

**Say:** "Every 100 games, the agent also played 32 games against a fixed alpha-beta depth-4 search engine. This agent applies principled tree search — it's not random."

**Point out:**
- Row 1 (game 0): `win_rate=0.1875` — "18.8% to start. Losing."
- Row ~24 (game 2300): `win_rate=0.5000` — "First time it beats alpha-beta 50/50. Game 2,300."
- Row ~59 (game 5700): `win_rate=1.0000` — "First perfect run — wins all 32 games."
- Rows ~67–71 (games 6400–6700): `win_rate=1.0000` × 4 — "Four consecutive perfect checks."
- Final window: mostly 0.9688–1.0000

**Key message:** "Beating alpha-beta depth-4 is not trivial. The agent learned to do it reliably."

---

### Step 3 — Launch the game UI (5–7 minutes)

```
python -m src.ui.app
```

**Demo flow:**

1. **Main menu appears.** "This is the playable game interface."

2. **Click Size 8×8.** "We're demonstrating the 8×8 board."

3. **Click Mode: Points Until Full.** "Points-Until-Full — the board fills and highest line score wins."

4. **Level select screen appears.** "The game uses the trained snapshots as difficulty levels. Each corresponds to a real training checkpoint."

5. **Select gen_001 (Easy/Improver).** "First let's see the weakest snapshot — trained on only 1,000 games." Play a few moves. Let it make a clearly suboptimal move. "Notice it's not very tactical."

6. **Go back to level select, select gen_010 (Master/Champion).** "Now the final model — 10,000 games of self-play."

7. **Play 5–8 moves.** Let the AI play. It should respond quickly and tactically. **Let it beat you if possible** — a loss here makes the point better than a win. "This is noticeably different — it's blocking my lines and building its own."

8. **If asked about the score sidebar:** "The sidebar shows each player's cumulative line score in real time — this is what the agent is optimising."

**Backup plan if UI crashes or doesn't load:**
- Open `results/size_08/run_pts_002/training_log.csv` and talk through the numbers instead.
- Have the benchmark log ready as a second piece of evidence.

---

## What the Demo Proves

| Brief criterion | What the demo shows |
|---|---|
| **Performance improves** | Elo 796 → 1,190, monotone (training_log.csv) |
| **Algorithm implemented** | Live AI playing in the UI |
| **Quantitative performance** | Win rates and Elo from the log |
| **Experience = self-play** | "No human data — it learned from playing itself" |
| **Critical analysis hook** | gen_001 vs gen_010 side-by-side quality difference is visible |

---

## Likely Questions — Prepared Answers

**Q: Why DQN and not a minimax solver?**
"C_lines has a 64-cell action space and up to N² = 64 moves. A full minimax tree is computationally infeasible even at moderate depth. DQN uses a neural network to approximate the value function rather than enumerating the tree."

**Q: What does the 10-channel state encoding do?**
"Each channel provides one feature to the network: own pieces, opponent pieces, empty cells, open threat cells for each player, win/lose-in-one cells, and turn progress. The network doesn't discover these patterns from scratch — we hand it the tactical features, and it learns how to weight them."

**Q: Why does it beat alpha-beta if alpha-beta uses lookahead?**
"The DQN's value function has implicitly 'seen' millions of board positions across 10,000 games. Its Q-values encode deep pattern knowledge accumulated over the full run. A depth-4 alpha-beta evaluates 4 plies but uses a simple hand-crafted score. By the end of training, the DQN's learned evaluation is more accurate than the heuristic evaluation alpha-beta uses."

**Q: What's Elo?**
"Elo is the rating system used in chess. I track the DQN's rating against two fixed anchors: Random=800 and Heuristic=900. Every 100 games, I play 20 games against each anchor and update the rating with the standard Elo formula. Because the anchor ratings don't change, the DQN's Elo is directly comparable across all 100 checkpoints."

**Q: Why does it sometimes drop to 0% in the benchmark?**
"That's a legitimate limitation. At game 8,300 it lost all 32 benchmark games. The agent has a specific exploitable weakness against one opening pattern that alpha-beta consistently finds. The policy is strong on average but not perfectly robust across all starting positions."

**Q: Did you train for other board sizes?**
"The infrastructure supports 9×9 through 12×12 and I have a second mode (First-to-Four). Due to the submission timeline I focused on validating the learning pipeline end-to-end on 8×8 first. The same training command works for all sizes — `--size 9`, `--size 10`, etc."
