"""One-shot cleanup — removes all files not needed for grading."""
import shutil, pathlib, os

ROOT = pathlib.Path(__file__).parent

def rm(rel):
    p = ROOT / rel
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print("  removed:", rel)
    else:
        print("  skip (not found):", rel)

print("=== Removing old/test training runs ===")
# Results — keep only run_ftf_003, run_ftf_004, run_pts_002, run_pts_003
for r in ["run_001", "run_002", "run_ftf_001", "run_ftf_002",
          "run_pts_001", "run_showcase", "run_ftf_005",
          "test_no_bm", "test_upgrade_smoke"]:
    rm(f"results/size_08/{r}")

# Legacy empty result dirs
for r in ["checkpoints", "figures", "logs"]:
    rm(f"results/{r}")

# Size 09 results (no trained agent)
rm("results/size_09")

print()
print("=== Removing old/test model snapshots ===")
# Loose gen_* folders at models/size_08 root (not inside a run)
for i in range(1, 12):
    rm(f"models/size_08/gen_{i:03d}")

# Model runs — keep run_ftf_003, run_ftf_004, run_pts_002, run_pts_003
for r in ["run_001", "run_ftf_001", "run_ftf_002",
          "run_pts_001", "test_no_bm", "test_upgrade_smoke"]:
    rm(f"models/size_08/{r}")

# Size 09 models
rm("models/size_09")

print()
print("=== Removing dev/test files ===")
rm("benchmark_workers.py")
rm(".pytest_cache")

print()
print("=== Removing __pycache__ directories ===")
for p in ROOT.rglob("__pycache__"):
    if ".venv" not in str(p) and ".git" not in str(p):
        shutil.rmtree(p)
        print("  removed:", p.relative_to(ROOT))

print()
print("=== Cleaning stale mode.txt files ===")
for p in ROOT.rglob("mode.txt"):
    if ".git" not in str(p):
        p.unlink()
        print("  removed:", p.relative_to(ROOT))

print()
print("Done. Remaining results/models:")
for p in sorted((ROOT / "results/size_08").iterdir()):
    print(" ", p.name)
print()
for p in sorted((ROOT / "models/size_08").iterdir()):
    print(" ", p.name)
