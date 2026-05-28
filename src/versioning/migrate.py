"""Snapshot migration utility.

Migrates a v1 (6-channel plain CNN) snapshot to the v2 (10-channel resnet_v1) format.
The old snapshot is never overwritten; a new gen_NNN entry is created with a migration note.

Usage:
    python -m src.versioning.migrate --version-id gen_003 --size 8
"""

from __future__ import annotations
import argparse
import copy
from pathlib import Path

import torch

import src.config as _cfg
from src.versioning.metadata import load_metadata, save_metadata, SnapshotMetadata
from src.versioning.registry import list_by_size_run, register, next_version_id


def migrate_snapshot(
    version_id: str,
    board_size: int,
    run_id: str,
    models_dir: Path | None = None,
) -> SnapshotMetadata:
    """Zero-pad a v1 (6-ch) snapshot's first conv weights into a 10-ch network.

    Copies all other weights verbatim. The 4 new input channels start with
    neutral (zero) weights so the network behaves identically to the original
    on positions where the new channels are all zeros.
    """
    models_dir = models_dir or _cfg.MODELS_DIR

    # Locate the old snapshot
    size_dir = models_dir / f"size_{board_size:02d}" / run_id
    snap_dir = size_dir / version_id
    meta_path = snap_dir / "metadata.json"
    old_meta = load_metadata(meta_path)

    if old_meta.state_channels == _cfg.STATE_CHANNELS_V2:
        print(f"[migrate] {version_id} already has {_cfg.STATE_CHANNELS_V2} channels — skipping.")
        return old_meta

    weights_path = models_dir / old_meta.weights_path
    old_sd = torch.load(weights_path, map_location="cpu")

    # The first conv layer is "conv.0.weight" shape (32, 6, 3, 3) in the plain network
    first_key = next(k for k in old_sd if "conv" in k and "weight" in k)
    old_w = old_sd[first_key]       # (out, 6, kH, kW)
    out_ch, in_ch, kH, kW = old_w.shape
    new_in_ch = _cfg.STATE_CHANNELS_V2

    new_w = torch.zeros(out_ch, new_in_ch, kH, kW)
    new_w[:, :in_ch, :, :] = old_w   # copy original 6 channels; new 4 start at 0
    new_sd = copy.deepcopy(old_sd)
    new_sd[first_key] = new_w

    # Assign a new version id
    new_ver = next_version_id(board_size, run_id, models_dir)
    new_snap_dir = size_dir / new_ver
    new_snap_dir.mkdir(parents=True, exist_ok=True)

    new_weights_rel = str(
        (new_snap_dir / "weights.pt").relative_to(models_dir)
    ).replace("\\", "/")
    torch.save(new_sd, new_snap_dir / "weights.pt")

    new_meta = copy.deepcopy(old_meta)
    new_meta.version_id = new_ver
    new_meta.weights_path = new_weights_rel
    new_meta.state_channels = new_in_ch
    new_meta.network_arch = "plain_v1"   # still plain CNN, just padded channels
    new_meta.notes = f"migrated from {version_id} (v1 → v2 channel padding)"
    new_meta.model_version = 2

    save_metadata(new_meta, new_snap_dir / "metadata.json")
    register(new_meta, board_size, run_id, models_dir)

    print(f"[migrate] {version_id} → {new_ver} written to {new_snap_dir}")
    return new_meta


def main() -> None:
    p = argparse.ArgumentParser(description="Migrate a C_lines snapshot from v1 to v2 channels")
    p.add_argument("--version-id", required=True, help="e.g. gen_003")
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--run-id", default="run_001")
    args = p.parse_args()
    migrate_snapshot(args.version_id, args.size, args.run_id)


if __name__ == "__main__":
    main()
