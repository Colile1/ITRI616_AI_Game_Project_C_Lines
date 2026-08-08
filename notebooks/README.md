# notebooks/

Colab notebooks for running training on rented GPU hardware instead of the
local laptop.

| File | Purpose |
|------|---------|
| `colab_train.ipynb` | Trains `run_ftf_005` (first-to-four) on a Colab GPU, seeded from `run_ftf_004/gen_009`. |

## Before the first session

`colab_train.ipynb` clones this repo from GitHub, so any training-code change it
depends on must already be pushed to `origin/submission`. The notebook checks
this in its setup cell and stops with an explicit message if the clone is stale,
rather than failing later inside a detached background process.

## How a session works

1. Cells 1–6 attach the GPU, mount Drive, clone the repo and restore any
   previous progress for the run.
2. Cell 7 builds the training command. It detects an existing `registry.json`
   for the run and switches between a seeded fresh start and `--resume`,
   carrying the Elo forward either way.
3. Cell 8 launches the trainer detached, plus a background loop that mirrors
   `models/size_08/<run>/` and `results/size_08/<run>/` to Drive every two
   minutes.
4. Cell 9 reports progress and can be re-run any time, including from a fresh
   session after a disconnect.

Training writes to normal folders inside the clone and is mirrored to Drive
afterwards, rather than writing directly into the Drive mount. Symlinking the
run folders into Drive would make git record them as symlinks, which breaks the
push-back path in cell 10. The mirror is copy-only and skips empty sources, so
it can never delete a Drive backup.

## Why several sessions are needed

Free Colab caps a session at roughly 12 hours and disconnects after about 90
minutes idle. A snapshot is written every 1000 games, so at worst a dropped
session costs the games since the last snapshot plus up to two minutes of logs.
Re-running cells 1–8 in a new session resumes from the last snapshot.

## Throughput note

Wall-clock time in this project is dominated by measurement rather than by
learning. Every 100 training games the loop also plays 340 evaluation games
(200 vs random, 100 vs heuristic, 40 for Elo), and by default a further 128
games against a depth-4 alpha-beta search.

The evaluation games feed the metrics the report is built on and are left alone.
The alpha-beta benchmark is purely diagnostic, so the notebook passes
`--benchmark-every 500` instead of the default 100. Set `BENCHMARK_EVERY = 100`
in the config cell if you want the finer benchmark curve and can spare the time.

## Choosing a device

The network is small — a 4-block ResNet, 64 channels, on an 8×8 board. Training
mixes many single-state forward passes (simulation, evaluation, benchmarking)
with a few batched gradient steps. GPUs win clearly on the latter and can lose
on the former, where kernel launch latency dominates. Whether CUDA is faster
here is therefore an empirical question: cell 6 measures both and picks the
winner. `FLAT4_DEVICE=cpu|cuda` overrides the choice anywhere in the project.
