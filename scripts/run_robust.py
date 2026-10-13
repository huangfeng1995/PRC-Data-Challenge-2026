"""
Robust runner (learned from 2025 2nd place team_jubilant_vase)

Usage:
    python -m scripts.run_robust --train \
        --feature-set all --models xgb,lgb,cat --n-seeds 5

Why this is needed:
- Blending 3 models x 5 seeds = 15 independent models, takes 1-2 hours
- Long runs can crash on OOM, a full disk, an extreme random seed, etc.
- Auto restart + a blacklist saves you time

Supported commands:
    --train: run train_ensemble
    --predict: run predict_ensemble
    --tune: run tune
    --eda: run eda

    Restart limit is 5; the blacklist automatically records the crashed flight / error info.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Robustness configuration
MAX_RESTARTS = 5
RESTART_COOLDOWN_SEC = 30  # wait 30 seconds before restarting (let the OS release resources)
BLACKLIST_FILE = Path("models/.blacklist.txt")


def run_with_restart(cmd: list[str], max_restarts: int = MAX_RESTARTS) -> int:
    """Run a command and automatically restart it on failure (up to max_restarts times).

    Returns the exit code of the last run (0 = success).
    """
    attempt = 0
    while attempt <= max_restarts:
        attempt += 1
        print(f"\n{'=' * 70}")
        print(f"🚀 start (attempt {attempt}/{max_restarts + 1})")
        print(f"  command: {' '.join(cmd)}")
        print(f"{'=' * 70}\n")

        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                print(f"\n✅ finished successfully!exit code = 0")
                return 0
            else:
                print(f"\n⚠️ failed,exit code = {result.returncode}")
        except KeyboardInterrupt:
            print(f"\n⚠️ interrupted by the user")
            return 130
        except Exception as e:
            print(f"\n❌ exception: {e}")

        # Not successful, prepare to restart
        if attempt > max_restarts:
            print(f"\n❌ reached the max retry count {max_restarts},giving up")
            return 1

        # Blacklist logging
        BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BLACKLIST_FILE, "a") as f:
            from datetime import datetime
            f.write(f"{datetime.now().isoformat()}\tattempt={attempt}\tcmd={' '.join(cmd)}\n")

        print(f"\n⏳ waiting {RESTART_COOLDOWN_SEC} seconds before restart...")
        time.sleep(RESTART_COOLDOWN_SEC)

    return 1


def main():
    parser = argparse.ArgumentParser(description="Robust runner")
    parser.add_argument("--train", action="store_true", help="run train_ensemble")
    parser.add_argument("--predict", action="store_true", help="run predict_ensemble")
    parser.add_argument("--tune", action="store_true", help="run tune")
    parser.add_argument("--eda", action="store_true", help="run eda")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--feature-set", default="all", choices=["basic", "domain", "aggregate", "all"])
    parser.add_argument("--models", default="xgb,lgb,cat")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS)
    args = parser.parse_args()

    # Build the subcommand
    if args.train:
        cmd = [
            sys.executable, "-m", "src.train_ensemble",
            "--data-dir", args.data_dir,
            "--feature-set", args.feature_set,
            "--models", args.models,
            "--n-seeds", str(args.n_seeds),
        ]
    elif args.predict:
        cmd = [
            sys.executable, "-m", "src.predict_ensemble",
            "--data-dir", args.data_dir,
            "--output", "submissions/robust_v1.parquet",
        ]
    elif args.tune:
        cmd = [
            sys.executable, "-m", "src.tune",
            "--data-dir", args.data_dir,
            "--feature-set", args.feature_set,
            "--n-trials", "50",
        ]
    elif args.eda:
        cmd = [
            sys.executable, "-m", "scripts.eda",
            "--data-dir", args.data_dir,
        ]
    else:
        parser.error("specify one of --train / --predict / --tune / --eda")

    exit_code = run_with_restart(cmd, max_restarts=args.max_restarts)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()