# tests/integration/

End-to-end tests exercising the full training and evaluation pipeline together.

| File | What it tests |
|------|--------------|
| `test_evaluator.py` | `evaluate()` runs N games between two agents and returns correct statistics |
| `test_train_smoke.py` | Full training loop: 50 games, gradient steps, snapshot saved, log file written |

These take 30-120 seconds each. Run before any major commit to confirm the full pipeline works.

```powershell
python -m pytest tests/integration/ -v
```
