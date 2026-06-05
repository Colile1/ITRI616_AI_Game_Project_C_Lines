# src/versioning/

Snapshot lifecycle management — saving, loading, and registering trained model checkpoints.

| File | Purpose |
|------|---------|
| `metadata.py` | `SnapshotMetadata` dataclass (v2 schema); `save_metadata()`, `load_metadata()` |
| `registry.py` | Per-run JSON index; `register()`, `get()`, `list_by_size_run()`, `next_version_id()` |
| `migrate.py` | Zero-pads v1 (6-channel) weights to v2 (10-channel) format |

### Layout
```
models/size_08/run_ftf_004/
    registry.json           # flat index of all snapshots in this run
    gen_001/
        weights.pt          # PyTorch state_dict
        metadata.json       # full snapshot record
    gen_002/ ...
```

### Key metadata fields
`version_id`, `run_id`, `games_trained`, `win_rate_vs_random`, `win_rate_vs_heuristic`, `elo_rating`, `difficulty_band` (novice/easy/medium/hard/master), `friendly_name`, `state_channels`, `network_arch`

### Loading a snapshot as an eval agent
```python
from src.training.snapshot import load_snapshot
agent = load_snapshot(board_size=8, version_id="gen_009", run_id="run_ftf_004")
# Returns eval-only DQNAgent with epsilon=0
```
