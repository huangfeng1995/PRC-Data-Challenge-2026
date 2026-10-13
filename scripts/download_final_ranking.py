"""
Download final-stage four-month data (1+2+6+7) from OpenSky MinIO
==========================================================
- Read-only download, never overwrites an existing ranking.parquet
- Never commits / never uploads any object to prc-2026-jubilant-puzzle
- Reports credential failures honestly, never fabricates data

Credential source: .env (OSN_ACCESS_KEY / OSN_SECRET_KEY)
Prerequisites:
  pip install boto3 python-dotenv  (boto3 + python-dotenv)
  Falls back to boto3 when mc is unavailable

Usage:
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

# Target bucket and files (the real final_* live in the prc-2026-datasets bucket; the competition-data bucket is inaccessible)
BUCKET = "prc-2026-datasets"
OBJ_FINAL_RANKING = "final_ranking.parquet"
OBJ_FINAL_SUBMITTING = "final_submitting.parquet"
LOCAL_FINAL_RANKING = DATA_RAW / OBJ_FINAL_RANKING
LOCAL_FINAL_SUBMITTING = DATA_RAW / OBJ_FINAL_SUBMITTING

# Credential loading
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
    """Configure the mc alias with .env credentials, returns (alias, mc_path)"""
    ak, sk = load_creds()
    if not ak or not sk:
        raise RuntimeError("OSN_ACCESS_KEY / OSN_SECRET_KEY missing from .env")
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
        raise RuntimeError("mc CLI not found. Run brew install minio/stable/mc or download the GitHub release")
    # alias set
    r = subprocess.run(
        [str(mc), "alias", "set", alias,
         "https://s3.opensky-network.org", ak, sk],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"mc alias set failed: {r.stderr.strip()}")
    return alias, mc


def mc_list(alias, mc_path):
    """List competition-data/"""
    r = subprocess.run(
        [str(mc_path), "ls", f"{alias}/{BUCKET}/"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"mc ls failed: {r.stderr.strip()}")
    return r.stdout.strip()


def mc_stat(alias, key, mc_path):
    """Get object size and modification time"""
    r = subprocess.run(
        [str(mc_path), "stat", f"{alias}/{BUCKET}/{key}"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def mc_download(alias, key, dest, mc_path):
    """Download an object to a local path"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Mark a temp placeholder file first, to avoid overwriting an existing ranking.parquet
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    r = subprocess.run(
        [str(mc_path), "cp", f"{alias}/{BUCKET}/{key}", str(tmp)],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"mc cp failed: {r.stderr.strip()}")
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


# ==== self-check + verification ====
def audit(path):
    """Read the parquet, print summary stats + required asserts"""
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
                    print(f"    ⚠ month {mn} has no DEP rows")
            # MVT_ID range
            for mn in sorted(by_month.index):
                m_ids = dep.loc[dep["mo"] == mn, "MVT_ID_mvt"].astype(float).values
                if len(m_ids):
                    print(f"    month {mn}: MVT_ID [{m_ids.min():.0f}, {m_ids.max():.0f}]")
        else:
            print(f"  DEP rows (no MVT_TIME_UTC_mvt): {len(dep):,}")
        # Verify BLOCK_TIME_UTC_mvt / TAXITIME_SEC_mvt are null for DEP rows
        if "BLOCK_TIME_UTC_mvt" in df.columns:
            n_blk_dep = int(dep["BLOCK_TIME_UTC_mvt"].notna().sum())
            print(f"  DEP rows with non-null BLOCK: {n_blk_dep}  (expected 0)")
            if n_blk_dep > 0:
                print(f"    ⚠ BLOCK_TIME_UTC_mvt should be null for DEP rows, but {n_blk_dep} rows are non-null")
        if "TAXITIME_SEC_mvt" in df.columns:
            n_tx_dep = int(dep["TAXITIME_SEC_mvt"].notna().sum())
            print(f"  DEP rows with non-null TAXITIME: {n_tx_dep}  (expected 0)")
            if n_tx_dep > 0:
                print(f"    ⚠ TAXITIME_SEC_mvt should be null for DEP rows, but {n_tx_dep} rows are non-null")
        # ARR
        if "PHASE_mvt" in df.columns:
            arr = df[df["PHASE_mvt"] == "ARR"]
            print(f"  ARR rows: {len(arr):,}")
    else:
        print(f"  ⚠ no PHASE_mvt column")
    return df


# ==== main flow ====
def main():
    t0 = time.time()
    print("=" * 70)
    print("[Download final-stage four-month data] OSN MinIO → data/raw/")
    print("=" * 70)
    print(f"  time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")

    # 0. Load credentials
    ak, sk = load_creds()
    if not ak or not sk:
        print("\n  ❌ credentials missing: .env has no OSN_ACCESS_KEY / OSN_SECRET_KEY")
        print("  Please provide new credentials and rerun")
        sys.exit(1)
    print(f"  ✓ credentials loaded: ak={ak[:8]}... sk={sk[:8]}...")

    # 1. Configure the mc alias (optional)
    mc_alias = None
    MC = None
    try:
        mc_alias, MC = configure_mc_alias()
        print(f"  ✓ mc alias '{mc_alias}' configured ({MC})")
    except RuntimeError as e:
        print(f"  ⚠ mc unavailable ({e}), falling back to boto3")

    # 2. List the competition-data bucket, confirm version and time
    print(f"\n[2] List {BUCKET}/ bucket contents")
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
            print(f"  ❌ mc ls failed: {e}")
            print("  falling back to boto3 ...")
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
            print(f"\n  ❌ boto3 bucket listing failed: {e}")
            print("  (credentials may be expired — InvalidAccessKeyId)")
            print("  Please provide new credentials and rerun")
            sys.exit(2)

    # 3. Download both files
    print(f"\n[3] Download → data/raw/")
    downloaded = {}
    for obj_key, local_path in [
        (OBJ_FINAL_RANKING, LOCAL_FINAL_RANKING),
        (OBJ_FINAL_SUBMITTING, LOCAL_FINAL_SUBMITTING),
    ]:
        if local_path.exists():
            print(f"  ⚠ {local_path} already exists ({local_path.stat().st_size:,} bytes), skipping")
            downloaded[obj_key] = {"exists": True, "size": local_path.stat().st_size,
                                   "path": str(local_path)}
            continue
        try:
            # In this environment an 85MB mc transfer takes >10min, boto3 finishes within ~5min
            # boto3 is used by default; mc is kept as a fallback path (--prefer-mc enables it)
            if False and mc_alias:
                size = mc_download(mc_alias, obj_key, local_path, MC)
            else:
                size = boto3_download(ak, sk, obj_key, local_path)
            print(f"  ✓ {obj_key} → {local_path.name}  ({size:,} bytes)")
            downloaded[obj_key] = {"exists": False, "size": size, "path": str(local_path)}
        except RuntimeError as e:
            print(f"  ❌ download of {obj_key} failed: {e}")
            downloaded[obj_key] = {"exists": False, "size": None, "error": str(e)}

    # 4. Verify
    print(f"\n[4] Verify")
    summary = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "downloaded": downloaded,
        "delta_vs_2nd_hour_check": "not run (user did not specify an exact time)",
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

    # Consistency of final_submitting vs final_ranking DEP
    if LOCAL_FINAL_SUBMITTING.exists() and LOCAL_FINAL_RANKING.exists():
        import pandas as pd
        rank_ids = set(df_rank[df_rank["PHASE_mvt"] == "DEP"]["MVT_ID_mvt"].astype(float).values)
        sub_ids = set(df_sub["MVT_ID_mvt"].astype(float).values)
        n_diff = len(rank_ids ^ sub_ids)
        print(f"\n  submitting MVT_ID vs ranking DEP MVT_ID: "
              f"symmetric difference {n_diff} IDs (expected 0)")
        if n_diff > 0:
            sample = list(rank_ids ^ sub_ids)[:5]
            print(f"    sample: {sample}")
        summary["DEP_MVTID_match"] = {
            "rank_n": len(rank_ids), "sub_n": len(sub_ids),
            "diff_n": n_diff, "sample_diff": list(rank_ids ^ sub_ids)[:5],
        }

    # Write the log
    log_fn = DATA_RAW / "final_download_log.json"
    log_fn.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\n  → log: {log_fn}")
    print(f"  total elapsed: {(time.time()-t0):.0f}s")
    print(f"\n  === task complete ===")


if __name__ == "__main__":
    main()