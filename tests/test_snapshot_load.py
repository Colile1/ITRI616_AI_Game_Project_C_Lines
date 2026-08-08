"""S-01 .. S-05 — snapshot round-trip, focused on architecture fidelity.

Regression cover for the bug where the registry entry carried no architecture
fields, so every 10-channel ResNet snapshot was rebuilt as a 6-channel plain
network and silently failed to load — which the UI then swallowed as "play a
random opponent instead".
"""

from __future__ import annotations

import json

import pytest

from src.agents.dqn_agent import DQNAgent
from src.config import STATE_CHANNELS_V2
from src.training.snapshot import freeze, load_snapshot
from src.versioning.registry import list_by_size, _registry_path

BOARD = 8
RUN = "run_test_001"


@pytest.fixture
def frozen(tmp_path):
    """A real ResNet snapshot frozen into a temporary models dir."""
    agent = DQNAgent(BOARD, in_channels=STATE_CHANNELS_V2, network_arch="resnet_v1")
    meta = freeze(
        agent, game_idx=100, eval_stats={"win_rate_vs_random": 0.9},
        board_size=BOARD, run_id=RUN, models_dir=tmp_path,
    )
    return meta, tmp_path, agent


def test_s01_freeze_writes_architecture_to_both_files(frozen):
    meta, models_dir, _ = frozen

    sidecar = json.loads(
        (models_dir / f"size_{BOARD:02d}" / RUN / meta.version_id / "metadata.json").read_text()
    )
    assert sidecar["state_channels"] == STATE_CHANNELS_V2
    assert sidecar["network_arch"] == "resnet_v1"

    entry = json.loads(_registry_path(BOARD, RUN, models_dir).read_text())[0]
    assert entry["state_channels"] == STATE_CHANNELS_V2
    assert entry["network_arch"] == "resnet_v1"


def test_s02_load_snapshot_rebuilds_the_same_architecture(frozen):
    meta, models_dir, original = frozen
    loaded = load_snapshot(BOARD, meta.version_id, run_id=RUN, models_dir=models_dir)

    assert loaded.in_channels == STATE_CHANNELS_V2
    assert loaded.network_arch == "resnet_v1"
    assert loaded.epsilon == 0.0

    for (ka, va), (kb, vb) in zip(
        original.state_dict().items(), loaded.state_dict().items()
    ):
        assert ka == kb
        assert va.shape == vb.shape


def test_s03_legacy_registry_without_arch_fields_still_loads(frozen):
    """The 84 snapshots already on disk have exactly this shape."""
    meta, models_dir, _ = frozen

    path = _registry_path(BOARD, RUN, models_dir)
    entries = json.loads(path.read_text())
    for entry in entries:
        entry.pop("state_channels", None)
        entry.pop("network_arch", None)
    path.write_text(json.dumps(entries, indent=2))

    # The registry now claims nothing; metadata.json must carry the day.
    assert list_by_size(BOARD, models_dir)[0].state_channels == 6

    loaded = load_snapshot(BOARD, meta.version_id, run_id=RUN, models_dir=models_dir)
    assert loaded.in_channels == STATE_CHANNELS_V2
    assert loaded.network_arch == "resnet_v1"


def test_s04_absolute_weights_path_survives_a_moved_project(frozen, tmp_path):
    """Recorded paths are absolute; loading must not depend on them."""
    meta, models_dir, _ = frozen
    snap_dir = models_dir / f"size_{BOARD:02d}" / RUN / meta.version_id

    sidecar = snap_dir / "metadata.json"
    data = json.loads(sidecar.read_text())
    data["weights_path"] = str(tmp_path / "somewhere" / "else" / "weights.pt")
    sidecar.write_text(json.dumps(data))

    loaded = load_snapshot(BOARD, meta.version_id, run_id=RUN, models_dir=models_dir)
    assert loaded.in_channels == STATE_CHANNELS_V2


def test_s05_plain_v1_snapshots_still_round_trip(tmp_path):
    agent = DQNAgent(BOARD, in_channels=6, network_arch="plain_v1")
    meta = freeze(
        agent, game_idx=10, eval_stats={"win_rate_vs_random": 0.2},
        board_size=BOARD, run_id=RUN, models_dir=tmp_path,
    )
    loaded = load_snapshot(BOARD, meta.version_id, run_id=RUN, models_dir=tmp_path)
    assert loaded.in_channels == 6
    assert loaded.network_arch == "plain_v1"
