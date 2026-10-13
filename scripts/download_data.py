"""
Download the PRC Data Challenge 2026 data (using Python + the minio package)

Credential sources (by priority, later ones override earlier ones):
    1. Command-line arguments --access-key / --secret-key / --bucket
    2. Environment variables OSN_ACCESS_KEY / OSN_SECRET_KEY / OSN_BUCKET
    3. The .env file in the project root (key=value format, never committed)

Usage:
    # Recommended: put the .env file in the project root, then just run
    python3 scripts/download_data.py --bucket prc-2026-datasets --dest data/raw

    # Or enter values interactively
    python3 scripts/download_data.py --interactive

    # Or via CLI/env
    OSN_ACCESS_KEY=xxx OSN_SECRET_KEY=yyy python3 scripts/download_data.py \\
        --bucket prc-2026-datasets
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error


# OSN competition data endpoint (https enforced)
ENDPOINT = os.environ.get("OSN_ENDPOINT", "s3.opensky-network.org").replace("https://", "").replace("http://", "")


def _load_env_file(env_path: Path) -> dict:
    """Simple .env parser, no python-dotenv dependency.
    Reads only key=value / key="value" lines, ignores # comments and blank lines."""
    out = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Strip quotes
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k] = v
    return out


def download_all(
    access_key: str,
    secret_key: str,
    bucket: str,
    dest: str = "data/raw",
    quiet: bool = False,
) -> int:
    """Download every file in the bucket.

    Returns the number of downloaded files.
    """
    client = Minio(
        ENDPOINT,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
    )

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # List every object in the bucket
    objects = list(client.list_objects(bucket, recursive=True))
    if not objects:
        print(f"⚠️ bucket '{bucket}' is empty, confirm the bucket name is correct")
        return 0

    print(f"Found {len(objects)} files, starting download to {dest_path}/")
    print("=" * 60)

    downloaded = 0
    total_bytes = 0
    for obj in objects:
        target = dest_path / obj.object_name
        # Skip files that already exist
        if target.exists() and target.stat().st_size == obj.size:
            if not quiet:
                print(f"  skip (already exists): {obj.object_name}")
            continue

        if not quiet:
            print(f"  download: {obj.object_name} ({obj.size / 1e6:.1f} MB)...", end="", flush=True)
        try:
            client.fget_object(bucket, obj.object_name, str(target))
            downloaded += 1
            total_bytes += obj.size
            if not quiet:
                print(" ✓")
        except S3Error as e:
            print(f" ✗\n    error: {e}")

    print("=" * 60)
    print(f"✅ done! downloaded {downloaded} files this run, {total_bytes / 1e6:.1f} MB in total")
    print(f"   data directory: {dest_path}/")
    print(f"   file count: {len(list(dest_path.glob('*')))}")

    return downloaded


def list_bucket(access_key: str, secret_key: str, bucket: str) -> None:
    """List the bucket contents (view only, no download)."""
    client = Minio(
        ENDPOINT,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
    )
    print(f"\n=== {bucket} contents ===")
    objects = list(client.list_objects(bucket, recursive=True))
    if not objects:
        print("(empty)")
        return
    for obj in objects:
        print(f"  {obj.object_name}  ({obj.size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download the PRC 2026 data")
    parser.add_argument("--access-key", help="OSN MinIO Access Key")
    parser.add_argument("--secret-key", help="OSN MinIO Secret Key")
    parser.add_argument("--bucket", default="prc-2026-datasets",
                        help="Bucket holding the data (note: the data is in a public bucket, not a personal bucket)")
    parser.add_argument("--dest", default="data/raw", help="Target directory")
    parser.add_argument("--list-only", action="store_true", help="Only list the bucket contents, do not download")
    parser.add_argument("--interactive", "-i", action="store_true", help="Enter keys interactively")
    args = parser.parse_args()

    # Interactive input
    if args.interactive:
        args.access_key = getpass.getpass("Access Key: ")
        args.secret_key = getpass.getpass("Secret Key: ")
        args.bucket = input("Bucket name: ").strip()

    # Also read from environment variables (avoids leaking secrets into shell history)
    if not args.access_key:
        args.access_key = os.environ.get("OSN_ACCESS_KEY")
    if not args.secret_key:
        args.secret_key = os.environ.get("OSN_SECRET_KEY")
    if not args.bucket:
        args.bucket = os.environ.get("OSN_BUCKET")

    # Finally read from the .env file in the project root (lowest priority, only fills missing fields)
    if not all([args.access_key, args.secret_key, args.bucket]):
        env_file = Path(__file__).resolve().parents[1] / ".env"
        env = _load_env_file(env_file)
        if not args.access_key:
            args.access_key = env.get("OSN_ACCESS_KEY")
        if not args.secret_key:
            args.secret_key = env.get("OSN_SECRET_KEY")
        if not args.bucket:
            args.bucket = env.get("OSN_BUCKET", args.bucket)

    if not all([args.access_key, args.secret_key, args.bucket]):
        print("❌ missing required arguments, use --interactive or --access-key/--secret-key/--bucket")
        return 1

    if args.list_only:
        list_bucket(args.access_key, args.secret_key, args.bucket)
    else:
        download_all(args.access_key, args.secret_key, args.bucket, args.dest)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())