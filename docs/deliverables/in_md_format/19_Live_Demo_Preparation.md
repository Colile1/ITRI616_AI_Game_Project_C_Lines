# In-Person Live Demo — Preparation (Presenting Tomorrow, 2026-06-03)

**Author:** *----* *S----* · ITRI 616
**Configuration:** C_lines 8×8, points-until-full · run `run_pts_002`
**Use with:** the slide deck (`C_lines_presentation.pptx`) and the video script (`20_Video_Presentation_Script.md`)

This is your readiness checklist and run-of-show for presenting **in person** tomorrow. It is written to score on all five points of the oral rubric: *Problem + TEP*, *Technical understanding*, *Results + analysis*, *Communication + organisation*, and *Response to questions + critical thinking*.

---

## A. The night before

- [ ] **Pull everything onto the presentation laptop** and confirm it runs offline (no reliance on Wi-Fi).
- [ ] Create/activate the virtual environment and install dependencies (`pip install -r requirements.txt`).
- [ ] Run `python -m pytest tests/ -q` → must be green.
- [ ] Run `python -m scripts.demo_eval --gen 10 --games 50` → confirm ~100% / ~100%.
- [ ] Run `python -m src.ui.app` → confirm the window opens and Level Select shows the snapshots.
- [ ] Open the three figures in an image viewer and **leave them open** as a fallback: `combined_progress.png`, `elo_curve.png`, `benchmark_vs_alphabeta.png` (in `results/size_08/run_pts_002/figures/`).
- [ ] Open the slide deck and click through once end-to-end.
- [ ] Charge the laptop; bring the charger and an HDMI/USB-C adapter.
- [ ] Put the **quick-reference commands** (section F) on a phone or index card.

## B. 15 minutes before you present

- [ ] Plug in, mirror to the projector, set terminal font large (so the back row can read output).
- [ ] Activate the venv in a terminal and pre-run `demo_eval --games 20` once to warm caches (so the live run is fast).
- [ ] Close email, chat, and notifications. Put the laptop on Do-Not-Disturb.
- [ ] Have the slide deck open on the title slide and one terminal ready.

## C. Run of show (target ~10 min talk + demo)

The order is built so each block earns one rubric criterion. Keep moving; if a demo misbehaves, fall back to the figure within 15 seconds.

| # | What you do | Rubric criterion it scores |
|---|---|---|
| 1 | **Title + one-line promise.** "Original game, self-play agent, evidence it improves with experience." | Communication |
| 2 | **Explain the game** on the board slide — open placement, line-scoring, board fills. | Problem definition |
| 3 | **TEP slide** — say Task, Experience, Performance out loud, with the problem category ("control"). | Problem + TEP |
| 4 | **Algorithm slide** — DQN, residual net, and the three choices that mattered (negamax sign, Double DQN, symmetry). | Technical understanding |
| 5 | **Results slide 1** — the combined Elo + win-rate curve; "monotone 796 → 1190". | Results + analysis |
| 6 | **Results slide 2** — benchmark reversal "19% → 100% vs a 4-ply search". | Results + analysis |
| 7 | **DEMO — live numbers:** `python -m scripts.demo_eval --gen 10 --games 50`. Read the result aloud. | Results + Technical |
| 8 | **DEMO — watch it play:** `python -m src.ui.app`, Level Select → Champion (gen_010); play 3–4 moves, narrate one block/extend. | Technical + Communication |
| 9 | **Critical analysis slide** — well-posed? improved? assumptions, and *volunteer the limitations*. | Critical thinking |
| 10 | **Reflection + close** — the negamax lesson, Elo-as-metric lesson; one-line summary. | Critical thinking + Communication |
| 11 | **Q&A** — see section E. | Response to questions |

## D. What to say at each demo (so the live part never goes quiet)

- **While `demo_eval` runs:** "This is the final snapshot playing 50 brand-new games against each fixed opponent, right now — not numbers from a log." Then read: "100% and 100%."
- **In Level Select:** "Each difficulty level is a checkpoint from the same run — the registry sets the difficulty from each snapshot's measured Elo, so 'harder' literally means 'later in learning'."
- **While playing the Champion:** narrate exactly one move — "It just blocked my line instead of extending its own; that offence-versus-defence judgement is learned, not coded."

## E. Q&A preparation (rubric criterion 5)

Have these answers ready and short:

**Why DQN and not policy gradient / AlphaZero?** "DQN fits a discrete action space with delayed reward and was enough to beat a 4-ply search. An AlphaZero-style policy+value net with MCTS is the natural next step — the network is already residual, so it's a small extension."

**How do you know it isn't memorising?** "It wins against alpha-beta, which it never trains against, and open placement gives a 64-branch tree — far too large to memorise; it has to generalise."

**Why is the win rate noisy between checkpoints?** "Each point is only 100–200 games, so there's sampling noise; that's exactly why the headline metric is Elo, which aggregates and comes out smooth."

**Single most important thing that made it work?** "The two-player value target — the next state is the opponent's, so the bootstrap is subtracted, not added. Wrong sign, it learns the opposite; right sign, it converges."

**Isn't the reward shaping doing the work?** "The shaping is the change in score margin — the same quantity as the final objective, just measured per move. It speeds learning without changing *what* is optimised."

**Hardware / time?** "CPU only, ~450–620 games/hour, several hours for a full run. No GPU."

**What would you improve?** "Multiple seeds for confidence bands, a deeper benchmark, and an AlphaZero-style extension."

## F. Quick-reference commands (card)

```bash
.venv\Scripts\activate
python -m pytest tests/ -q                              # green check
python -m scripts.demo_eval --gen 10 --games 50         # live win rates (fast)
python -m scripts.demo_eval --gen 10 --games 16 --alphabeta   # incl. alpha-beta (slower)
python -m src.ui.app                                    # play / watch (Level Select)
```

## G. Fallback plan (assume one thing will break)

| If this fails… | Do this |
|---|---|
| `demo_eval` errors / model won't load | Switch to the open figures; quote `training_log.csv` final row (100% / 100%, Elo 1190). |
| PyGame window won't open | Show `combined_progress.png` and `benchmark_vs_alphabeta.png`; describe the levels from `registry.json`. |
| Alpha-beta run hangs | Ctrl-C, drop `--alphabeta`, run the fast `--games 50`. |
| Total tech failure | Present straight from the slide deck + report — every figure and number is in them. |

**Golden rule:** the static figures are your safety net; they hold the entire results story and cannot crash. Never let a failed command cost you more than 15 seconds.

## H. Two numbers you must be able to say without notes

1. **"Elo rose monotonically from 796 to 1190."**
2. **"From 19% to 100% against a 4-ply search it never trained on."**

If you remember nothing else, those two sentences carry the results criterion.

---

### Note on tomorrow vs Friday
If you are ready, present **in person tomorrow** using this plan. If you are not ready, the lecturer allows a **recorded video** submitted Friday — for that, use the same slide deck with `20_Video_Presentation_Script.md` as the narration (no live terminal needed; screen-record the figures and a short clip of the GUI instead).
