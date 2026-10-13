"""
Upload a submission parquet to the OSN bucket (prc-2026-jubilant-puzzle)

Usage:
    python -m scripts.upload_submission \
        --access-key "ABC..." \
        --secret-key "xyz..." \
        --local submissions/jubilant-puzzle_v2.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from minio import Minio


ENDPOINT = "s3.opensky-network.org"
SUBMISSION_BUCKET = "prc-2026-jubilant-puzzle"


def upload_file(
    access_key: str,
    secret_key: str,
    local_path: str,
    bucket: str = SUBMISSION_BUCKET,
    object_name: str | None = None,
) -> str:
    """Upload a local file to the bucket. Returns the object name."""
    client = Minio(
        ENDPOINT,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        region="eu-west-1",
    )

    local = Path(local_path)
    if not local.exists():
        raise FileNotFoundError(local)

    object_name = object_name or local.name
    size = local.stat().st_size
    print(f"uploading {local} ({size/1024:.1f} KB) -> {bucket}/{object_name}")

    # content_type: parquet
    import mimetypes
    ctype, _ = mimetypes.guess_type(str(local))
    ctype = ctype or "application/octet-stream"

    client.fput_object(
        bucket_name=bucket,
        object_name=object_name,
        file_path=str(local),
        content_type=ctype,
    )

    print(f"✅ upload succeeded: s3://{bucket}/{object_name}")
    return object_name


def main():
    parser = argparse.ArgumentParser(description="Upload the submission to OSN")
    parser.add_argument("--access-key", default=os.environ.get("OSN_ACCESS_KEY"))
    parser.add_argument("--secret-key", default=os.environ.get("OSN_SECRET_KEY"))
    parser.add_argument("--local", required=True, help="Local parquet path")
    parser.add_argument("--bucket", default=SUBMISSION_BUCKET)
    parser.add_argument("--object-name", default=None, help="Object name (defaults to the file name)")
    parser.add_argument("--env-file", default=".env", help=".env file path, credentials are read automatically")
    args = parser.parse_args()

    # If the environment variables are not set, read them from .env
    if not args.access_key or not args.secret_key:
        env_path = Path(args.env_file)
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        args.access_key = args.access_key or os.environ.get("OSN_ACCESS_KEY")
        args.secret_key = args.secret_key or os.environ.get("OSN_SECRET_KEY")

    if not args.access_key or not args.secret_key:
        parser.error("need --access-key and --secret-key, or a .env file, or the OSN_ACCESS_KEY/OSN_SECRET_KEY environment variables")

    upload_file(
        access_key=args.access_key,
        secret_key=args.secret_key,
        local_path=args.local,
        bucket=args.bucket,
        object_name=args.object_name,
    )


if __name__ == "__main__":
    main()