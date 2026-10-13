"""
下载 final 阶段四个月数据 (1+2+6+7 月) from OpenSky MinIO
==========================================================
- 只读下载，不覆盖现有 ranking.parquet
- 不提交/不向 prc-2026-jubilant-puzzle 上传任何对象
- 凭据失效时如实报告，不伪造数据

凭据来源: .env (OSN_ACCESS_KEY / OSN_SECRET_KEY)
前置:
  pip install boto3 python-dotenv  (boto3 + python-dotenv)
  mc 不可用时用 boto3 fallback

用法:
  python3 scripts/download_final_ranking.py
"""
from __future__ import annotations
import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
DATA_RAW = ROOT / "data" / "raw"

# 目标桶与文件 (实际 final_* 在 prc-2026-datasets 桶, competition-data 桶无权)
BUCKET = "prc-2026-datasets"
OBJ_FINAL_RANKING = "final_ranking.parquet"
OBJ_FINAL_SUBMITTING = "final_submitting.parquet"
LOCAL_FINAL_RANKING = DATA_RAW / OBJ_FINAL_RANKING
LOCAL_FINAL_SUBMITTING = DATA_RAW / OBJ_FINAL_SUBMITTING

# 凭据加载
def load_creds():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None, None
    creds = {}
    for ln in env_file.read_text().splitlines():
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            creds[k.strip()] = v.strip().strip('"').strip("'")
    return creds.get("OSN_ACCESS_KEY"), creds.get("OSN_SECRET_KEY")


# ==== mc CLI ====
def configure_mc_alias(alias="dc26"):
    """用 .env 凭据配置 mc alias, 返回 (alias, mc_path)"""
    ak, sk = load_creds()
    if not ak or not sk:
        raise RuntimeError("OSN_ACCESS_KEY / OSN_SECRET_KEY 缺失于 .env")
    mc_paths = [
        Path("/Users/openclaw/.trae-cn/work/6aaca8f8b524b5e1e3ae7a73/bin/mc"),
        Path.home() / ".local" / "bin" / "mc",
    ]
    mc = None
    for p in mc_paths:
        if p.exists() and p.is_file():
            mc = p
            break
    if mc is None:
        for cmd in ["which mc", "command -v mc"]:
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip():
                    mc = Path(r.stdout.strip())
                    break
            except Exception:
                pass
    if mc is None:
        raise RuntimeError("mc CLI 未找到. 请 brew install minio/stable/mc 或下载 GitHub release")
    # alias set
    r = subprocess.run(
        [str(mc), "alias", "set", alias,
         "https://s3.opensky-network.org", ak, sk],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"mc alias set 失败: {r.stderr.strip()}")
    return alias, mc


def mc_list(alias, mc_path):
    """列 competition-data/"""
    r = subprocess.run(
        [str(mc_path), "ls", f"{alias}/{BUCKET}/"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"mc ls 失败: {r.stderr.strip()}")
    return r.stdout.strip()


def mc_stat(alias, key, mc_path):
    """获取对象大小和修改时间"""
    r = subprocess.run(
        [str(mc_path), "stat", f"{alias}/{BUCKET}/{key}"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def mc_download(alias, key, dest, mc_path):
    """下载对象到本地路径"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 临时占位文件先标记, 避免覆盖已有 ranking.parquet
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    r = subprocess.run(
        [str(mc_path), "cp", f"{alias}/{BUCKET}/{key}", str(tmp)],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"mc cp 失败: {r.stderr.strip()}")
    tmp.rename(dest)
    return dest.stat().st_size


# ==== boto3 fallback ====
def boto3_list(ak, sk):
    import boto3
    from botocore.exceptions import ClientError
    s3 = boto3.client(
        "s3", endpoint_url="https://s3.opensky-network.org",
        aws_access_key_id=ak, aws_secret_access_key=sk,
    )
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, MaxKeys=50)
    except ClientError as e:
        raise RuntimeError(f"boto3 list_objects_v2: {e}")
    return resp.get("Contents", [])


def boto3_download(ak, sk, key, dest):
    import boto3
    from botocore.exceptions import ClientError
    s3 = boto3.client(
        "s3", endpoint_url="https://s3.opensky-network.org",
        aws_access_key_id=ak, aws_secret_access_key=sk,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        s3.download_file(BUCKET, key, str(tmp))
    except ClientError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"boto3 download_file {key}: {e}")
    tmp.rename(dest)
    return dest.stat().st_size


# ==== 自检 + 核对 ====
def audit(path):
    """读 parquet 给出汇总统计 + 必要 assert"""
    import pandas as pd
    df = pd.read_parquet(path)
    n_total = len(df)
    cols = list(df.columns)
    print(f"\n  === {path.name} ===")
    print(f"  rows: {n_total:,}")
    print(f"  cols: {cols[:10]}{' ...' if len(cols)>10 else ''}")

    if "PHASE_mvt" in df.columns:
        dep = df[df["PHASE_mvt"] == "DEP"].copy()
        if "MVT_TIME_UTC_mvt" in dep.columns:
            dep["mo"] = pd.to_datetime(dep["MVT_TIME_UTC_mvt"],
                                       utc=True, errors="coerce").dt.month
            by_month = dep["mo"].value_counts().sort_index()
            print(f"  DEP rows: {len(dep):,}")
            print(f"  DEP by month: {by_month.to_dict()}")
            for mn in (1, 2, 6, 7):
                if mn not in by_month.index:
                    print(f"    ⚠ 月 {mn} 无 DEP 行")
            # MVT_ID 范围
            for mn in sorted(by_month.index):
                m_ids = dep.loc[dep["mo"] == mn, "MVT_ID_mvt"].astype(float).values
                if len(m_ids):
                    print(f"    月 {mn}: MVT_ID [{m_ids.min():.0f}, {m_ids.max():.0f}]")
        else:
            print(f"  DEP rows (无 MVT_TIME_UTC_mvt): {len(dep):,}")
        # 确认 BLOCK_TIME_UTC_mvt / TAXITIME_SEC_mvt 在 DEP 行为空
        if "BLOCK_TIME_UTC_mvt" in df.columns:
            n_blk_dep = int(dep["BLOCK_TIME_UTC_mvt"].notna().sum())
            print(f"  DEP 行中 BLOCK 非空: {n_blk_dep}  (期望 0)")
            if n_blk_dep > 0:
                print(f"    ⚠ DEP 行的 BLOCK_TIME_UTC_mvt 应空, 实际 {n_blk_dep} 行非空")
        if "TAXITIME_SEC_mvt" in df.columns:
            n_tx_dep = int(dep["TAXITIME_SEC_mvt"].notna().sum())
            print(f"  DEP 行中 TAXITIME 非空: {n_tx_dep}  (期望 0)")
            if n_tx_dep > 0:
                print(f"    ⚠ DEP 行的 TAXITIME_SEC_mvt 应空, 实际 {n_tx_dep} 行非空")
        # ARR
        if "PHASE_mvt" in df.columns:
            arr = df[df["PHASE_mvt"] == "ARR"]
            print(f"  ARR rows: {len(arr):,}")
    else:
        print(f"  ⚠ 无 PHASE_mvt 列")
    return df


# ==== 主流程 ====
def main():
    t0 = time.time()
    print("=" * 70)
    print("[下载 final 阶段四个月数据] OSN MinIO → data/raw/")
    print("=" * 70)
    print(f"  时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")

    # 0. 加载凭据
    ak, sk = load_creds()
    if not ak or not sk:
        print("\n  ❌ 凭据缺失: .env 没有 OSN_ACCESS_KEY / OSN_SECRET_KEY")
        print("  请用户提供新凭据后重跑")
        sys.exit(1)
    print(f"  ✓ 凭据加载: ak={ak[:8]}... sk={sk[:8]}...")

    # 1. 配置 mc alias (可选)
    mc_alias = None
    MC = None
    try:
        mc_alias, MC = configure_mc_alias()
        print(f"  ✓ mc alias '{mc_alias}' 已配置 ({MC})")
    except RuntimeError as e:
        print(f"  ⚠ mc 不可用 ({e}), fallback 到 boto3")

    # 2. 列出 competition-data 桶, 确认版本与时间
    print(f"\n[2] 列 {BUCKET}/ 桶内容")
    if mc_alias:
        try:
            listing = mc_list(mc_alias, MC)
            print("  --- listing ---")
            print(listing)
            for key in (OBJ_FINAL_RANKING, OBJ_FINAL_SUBMITTING):
                try:
                    stat = mc_stat(mc_alias, key, MC)
                    print(f"  --- stat {key} ---")
                    print(stat)
                except RuntimeError as e:
                    print(f"  ⚠ {key}: {e}")
        except RuntimeError as e:
            print(f"  ❌ mc ls 失败: {e}")
            print("  fallback 到 boto3 ...")
            mc_alias = None

    if not mc_alias:
        try:
            contents = boto3_list(ak, sk)
            print("  --- boto3 listing ---")
            for o in contents:
                size_mb = o["Size"] / 1024 / 1024
                print(f"  {o['Key']}  size={o['Size']:,} ({size_mb:.2f} MB)  "
                      f"modified={o['LastModified'].isoformat()}")
        except RuntimeError as e:
            print(f"\n  ❌ boto3 列出桶失败: {e}")
            print("  (凭据可能失效 — InvalidAccessKeyId)")
            print("  请用户提供新凭据后重跑")
            sys.exit(2)

    # 3. 下载两个文件
    print(f"\n[3] 下载 → data/raw/")
    downloaded = {}
    for obj_key, local_path in [
        (OBJ_FINAL_RANKING, LOCAL_FINAL_RANKING),
        (OBJ_FINAL_SUBMITTING, LOCAL_FINAL_SUBMITTING),
    ]:
        if local_path.exists():
            print(f"  ⚠ {local_path} 已存在 ({local_path.stat().st_size:,} bytes), 跳过")
            downloaded[obj_key] = {"exists": True, "size": local_path.stat().st_size,
                                   "path": str(local_path)}
            continue
        try:
            # mc 在本环境下 85MB 文件传输 >10min, boto3 ~5min 内完成
            # 默认走 boto3; mc 保留为 fallback 路径 (--prefer-mc 可启用)
            if False and mc_alias:
                size = mc_download(mc_alias, obj_key, local_path, MC)
            else:
                size = boto3_download(ak, sk, obj_key, local_path)
            print(f"  ✓ {obj_key} → {local_path.name}  ({size:,} bytes)")
            downloaded[obj_key] = {"exists": False, "size": size, "path": str(local_path)}
        except RuntimeError as e:
            print(f"  ❌ 下载 {obj_key} 失败: {e}")
            downloaded[obj_key] = {"exists": False, "size": None, "error": str(e)}

    # 4. 核对
    print(f"\n[4] 核对")
    summary = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "downloaded": downloaded,
        "delta_vs_2nd_hour_check": "未执行 (用户未指定具体时间)",
    }
    if LOCAL_FINAL_RANKING.exists():
        df_rank = audit(LOCAL_FINAL_RANKING)
        summary["final_ranking"] = {
            "n_total": int(len(df_rank)),
            "cols": list(df_rank.columns),
        }
    if LOCAL_FINAL_SUBMITTING.exists():
        df_sub = audit(LOCAL_FINAL_SUBMITTING)
        summary["final_submitting"] = {
            "n_total": int(len(df_sub)),
            "cols": list(df_sub.columns),
        }

    # final_submitting 与 final_ranking DEP 一致性
    if LOCAL_FINAL_SUBMITTING.exists() and LOCAL_FINAL_RANKING.exists():
        import pandas as pd
        rank_ids = set(df_rank[df_rank["PHASE_mvt"] == "DEP"]["MVT_ID_mvt"].astype(float).values)
        sub_ids = set(df_sub["MVT_ID_mvt"].astype(float).values)
        n_diff = len(rank_ids ^ sub_ids)
        print(f"\n  submitting MVT_ID vs ranking DEP MVT_ID: "
              f"差集 {n_diff} 个 ID (期望 0)")
        if n_diff > 0:
            sample = list(rank_ids ^ sub_ids)[:5]
            print(f"    示例: {sample}")
        summary["DEP_MVTID_match"] = {
            "rank_n": len(rank_ids), "sub_n": len(sub_ids),
            "diff_n": n_diff, "sample_diff": list(rank_ids ^ sub_ids)[:5],
        }

    # 落日志
    log_fn = DATA_RAW / "final_download_log.json"
    log_fn.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\n  → 日志: {log_fn}")
    print(f"  总耗时: {(time.time()-t0):.0f}s")
    print(f"\n  === 任务完成 ===")


if __name__ == "__main__":
    main()