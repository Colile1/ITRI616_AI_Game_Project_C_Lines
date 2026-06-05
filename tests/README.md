# tests/

Unit and integration test suite. 72 tests, all passing.

```powershell
python -m pytest tests/ -v             # all tests
python -m pytest tests/test_agent.py  # one file
```

| File | Tags | What it covers |
|------|------|----------------|
| `test_agent.py` | A01-A08 | DQNAgent: legal actions, update step, target sync, loss returned |
| `test_env.py` | E01-E11 | GameEnv: obs shape (10ch), channel values, step, illegal move penalty |
| `test_rules.py` | R01-R14 | Board placement, removal, immutability, terminal conditions |
| `test_scoring.py` | S01-S13 | Score computation, all line directions, capping, multi-player |
| `test_elo.py` | — | Elo update formula and expected score |
| `test_snapshot.py` | V01-V09 | Metadata roundtrip, registry persistence, monotone version IDs |
| `integration/` | — | End-to-end evaluator and training smoke tests |

Tests use 10-channel encoding throughout (STATE_CHANNELS_V2=10). Do not revert to 6 channels.
