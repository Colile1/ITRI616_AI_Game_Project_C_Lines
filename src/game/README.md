# src/game/

Gym-style game environment and state encoding.

| File | Contents |
|------|----------|
| `env.py` | `GameEnv` — `reset()`, `step(action)`, `legal_mask()` |
| `encoding.py` | `state_to_tensor()` — Board -> 10-channel (C, N, N) float32 array |

### GameEnv usage
```python
env = GameEnv(board_size=8, mode="first_to_four")
obs  = env.reset()                        # (10, 8, 8) float32
mask = env.legal_mask()                   # (64,) bool
obs, reward, done, info = env.step(42)    # info["winner"] = 1 | 2 | None
```
Reward is always from the perspective of the player who just acted.

### 10-channel encoding
| Ch | Content |
|----|---------|
| 0 | Current player pieces |
| 1 | Opponent pieces |
| 2 | Current player open-3 threats |
| 3 | Opponent open-3 threats |
| 4 | Current player forced-win threats (double-open-3) |
| 5 | Opponent forced-win threats |
| 6 | Turn progress (0->1) |
| 7 | Immediate-win cells for current player |
| 8 | Immediate-loss cells |
| 9 | Empty cells |

### FTF reward shaping
Non-terminal: `open3_delta * 0.10 + double_open3_delta * 0.30`
Terminal: win=+1.0, loss=-1.0 (or -1.5 if early loss within adaptive threshold), draw=0.0
