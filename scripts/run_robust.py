"""
鲁棒运行器(从 2025 第 2 名 team_jubilant_vase 学到)

用法:
    python -m scripts.run_robust --train \
        --feature-set all --models xgb,lgb,cat --n-seeds 5

为什么需要这个:
- 三模型 5-seed 融合 = 15 个独立模型,跑 1-2 小时
- 长跑可能因为 OOM、磁盘满、random seed 极端等崩溃
- 自动重启 + 黑名单可以节省你的时间

支持的命令:
    --train: 跑 train_ensemble
    --predict: 跑 predict_ensemble
    --tune: 跑 tune
    --eda: 跑 eda

    重启上限 5 次,黑名单会自动记录崩溃的航班/异常信息。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# 鲁棒性配置
MAX_RESTARTS = 5
RESTART_COOLDOWN_SEC = 30  # 重启前等待 30 秒(让 OS 资源释放)
BLACKLIST_FILE = Path("models/.blacklist.txt")


def run_with_restart(cmd: list[str], max_restarts: int = MAX_RESTARTS) -> int:
    """运行命令,如果失败自动重启(最多 max_restarts 次)。

    返回最后一次运行的 exit code(0 = 成功)。
    """
    attempt = 0
    while attempt <= max_restarts:
        attempt += 1
        print(f"\n{'=' * 70}")
        print(f"🚀 启动 (尝试 {attempt}/{max_restarts + 1})")
        print(f"  命令: {' '.join(cmd)}")
        print(f"{'=' * 70}\n")

        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                print(f"\n✅ 成功完成!exit code = 0")
                return 0
            else:
                print(f"\n⚠️ 失败,exit code = {result.returncode}")
        except KeyboardInterrupt:
            print(f"\n⚠️ 用户中断")
            return 130
        except Exception as e:
            print(f"\n❌ 异常: {e}")

        # 没成功,准备重启
        if attempt > max_restarts:
            print(f"\n❌ 已达最大重试次数 {max_restarts},放弃")
            return 1

        # 黑名单记录
        BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BLACKLIST_FILE, "a") as f:
            from datetime import datetime
            f.write(f"{datetime.now().isoformat()}\tattempt={attempt}\tcmd={' '.join(cmd)}\n")

        print(f"\n⏳ 等待 {RESTART_COOLDOWN_SEC} 秒后重启...")
        time.sleep(RESTART_COOLDOWN_SEC)

    return 1


def main():
    parser = argparse.ArgumentParser(description="鲁棒运行器")
    parser.add_argument("--train", action="store_true", help="跑 train_ensemble")
    parser.add_argument("--predict", action="store_true", help="跑 predict_ensemble")
    parser.add_argument("--tune", action="store_true", help="跑 tune")
    parser.add_argument("--eda", action="store_true", help="跑 eda")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--feature-set", default="all", choices=["basic", "domain", "aggregate", "all"])
    parser.add_argument("--models", default="xgb,lgb,cat")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS)
    args = parser.parse_args()

    # 构造子命令
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
        parser.error("请指定 --train / --predict / --tune / --eda 之一")

    exit_code = run_with_restart(cmd, max_restarts=args.max_restarts)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()