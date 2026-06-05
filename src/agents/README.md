# src/agents/

All agent implementations. Every agent inherits `BaseAgent` and implements `select_action(obs, legal_mask) -> int`.

| File | Class | Description |
|------|-------|-------------|
| `base_agent.py` | `BaseAgent` | Abstract interface |
| `dqn_agent.py` | `DQNAgent` | Double DQN + ResNet; trains via `update(batch)` |
| `alphabeta_agent.py` | `AlphaBetaAgent` | Iterative-deepening alpha-beta search |
| `mcts_agent.py` | `MCTSAgent` | PUCT MCTS wrapping any DQNAgent |
| `heuristic_agent.py` | `HeuristicAgent` | Rule-based threat-counting agent |
| `random_agent.py` | `RandomAgent` | Uniform random legal move |
| `terminal_human_agent.py` | `TerminalHumanAgent` | Human input via terminal |
| `ui_human_agent.py` | `UIHumanAgent` | Human input via PyGame (thread-safe queues) |

### DQNAgent quick reference
```python
agent = DQNAgent(board_size=8, in_channels=10, network_arch="resnet_v1")
action = agent.select_action(obs, legal_mask)  # epsilon-greedy
loss   = agent.update(batch)                    # one gradient step
agent.sync_target()                             # copy online -> target
agent.set_epsilon(0.05)
sd = agent.state_dict()                         # save
agent.load_state_dict(sd)                       # load
```

Device selection: `_best_device()` returns CUDA if available, otherwise CPU.
DirectML (Intel UHD) is excluded — Adam's `lerp` op has no DML kernel, causing 100x slowdown via CPU fallback.
