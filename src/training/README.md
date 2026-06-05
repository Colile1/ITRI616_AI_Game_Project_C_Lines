# src/training/

Everything needed to train a DQN agent via self-play.

| File | Purpose |
|------|---------|
| `train.py` | Main CLI training loop — opponent selection, logging, eval, snapshots, benchmark |
| `network.py` | `DQNNetwork` (ResNet 4-block 64-ch), `build_network()` factory |
| `replay_buffer.py` | Weighted circular buffer with D4 symmetry augmentation and `gamma_n` support |
| `self_play.py` | `play_episode()` with n=3 step returns; `linear_epsilon()` |
| `symmetry.py` | D4 dihedral transforms — vectorised `transform_obs()`, `augment_transition()` |
| `snapshot.py` | `freeze()` — save agent to disk; `load_snapshot()`, `clone_agent()` |
| `benchmark_logger.py` | Logs benchmark game results to CSV |
| `game_logger.py` | Logs every individual game result to CSV |
| `schedule.py` | `TrainingSchedule` — multi-phase opponent curriculum |

### Key CLI flags (train.py)
```powershell
python -m src.training.train --size 8 --mode first_to_four --games 10000 --benchmark alphabeta_d4
python -m src.training.train --size 8 --mode first_to_four --games 20000 --run-id run_ftf_004 --resume --workers 4 --prev-best results/size_08/run_ftf_003/best/weights.pt
```

| Flag | Effect |
|------|--------|
| `--workers N` | Parallel episode collection via multiprocessing Pool (use 4 for single run) |
| `--prev-best PATH` | Frozen model used as static training opponent instead of RandomAgent |
| `--resume` | Load latest snapshot and continue; auto-deletes stale mode.txt "quit" |
| `--load-weights PATH` | Seed agent weights from a file (use with --start-game N) |

### Parallel workers
Workers weight sync happens every 50 games (not every game — the 500KB pickle overhead negated all speedup when syncing every step). Benchmark: 4 workers gives 1.4x speedup over sequential.

### n-step returns (N_STEP_RETURNS=3)
Full episode trajectory is collected, then n=3 step returns are computed with negamax alternating sign:
`G = r_t - gamma*r_{t+1} + gamma^2*r_{t+2}`, bootstrap discount `gamma_n = gamma^3 = 0.9703` stored per-transition.
