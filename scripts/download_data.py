"""
下载 PRC Data Challenge 2026 数据(用 Python + minio 包)

凭据来源(优先级, 后者优先覆盖):
    1. 命令行参数 --access-key / --secret-key / --bucket
    2. 环境变量 OSN_ACCESS_KEY / OSN_SECRET_KEY / OSN_BUCKET
    3. 项目根目录 .env 文件 (key=value 格式, 不会 commit)

用法:
    # 推荐: 把 .env 文件放到项目根目录, 然后直接跑
    python3 scripts/download_data.py --bucket prc-2026-datasets --dest data/raw

    # 或交互式输入
    python3 scripts/download_data.py --interactive

    # 或命令行/env
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


# OSN 比赛数据 endpoint(https 强制)
ENDPOINT = os.environ.get("OSN_ENDPOINT", "s3.opensky-network.org").replace("https://", "").replace("http://", "")


def _load_env_file(env_path: Path) -> dict:
    """简单解析 .env 文件, 不依赖 python-dotenv.
    只读 key=value / key="value" 行, 忽略 # 注释和空行."""
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
        # 去引号
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
    """全量下载 bucket 里的所有文件。

    返回下载的文件数。
    """
    client = Minio(
        ENDPOINT,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
    )

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # 列出 bucket 内所有对象
    objects = list(client.list_objects(bucket, recursive=True))
    if not objects:
        print(f"⚠️ bucket '{bucket}' 为空,确认 bucket 名是否正确")
        return 0

    print(f"找到 {len(objects)} 个文件,开始下载到 {dest_path}/")
    print("=" * 60)

    downloaded = 0
    total_bytes = 0
    for obj in objects:
        target = dest_path / obj.object_name
        # 跳过已存在的
        if target.exists() and target.stat().st_size == obj.size:
            if not quiet:
                print(f"  跳过(已存在): {obj.object_name}")
            continue

        if not quiet:
            print(f"  下载: {obj.object_name} ({obj.size / 1e6:.1f} MB)...", end="", flush=True)
        try:
            client.fget_object(bucket, obj.object_name, str(target))
            downloaded += 1
            total_bytes += obj.size
            if not quiet:
                print(" ✓")
        except S3Error as e:
            print(f" ✗\n    错误: {e}")

    print("=" * 60)
    print(f"✅ 完成!本次下载 {downloaded} 个文件,共 {total_bytes / 1e6:.1f} MB")
    print(f"   数据目录: {dest_path}/")
    print(f"   文件数: {len(list(dest_path.glob('*')))}")

    return downloaded


def list_bucket(access_key: str, secret_key: str, bucket: str) -> None:
    """列出 bucket 内容(只看不下载)。"""
    client = Minio(
        ENDPOINT,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
    )
    print(f"\n=== {bucket} 内容 ===")
    objects = list(client.list_objects(bucket, recursive=True))
    if not objects:
        print("(空)")
        return
    for obj in objects:
        print(f"  {obj.object_name}  ({obj.size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="下载 PRC 2026 数据")
    parser.add_argument("--access-key", help="OSN MinIO Access Key")
    parser.add_argument("--secret-key", help="OSN MinIO Secret Key")
    parser.add_argument("--bucket", default="prc-2026-datasets",
                        help="数据所在 bucket(注意:数据在公共 bucket,不是个人 bucket)")
    parser.add_argument("--dest", default="data/raw", help="目标目录")
    parser.add_argument("--list-only", action="store_true", help="只列出 bucket 内容,不下载")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式输入密钥")
    args = parser.parse_args()

    # 交互式输入
    if args.interactive:
        args.access_key = getpass.getpass("Access Key: ")
        args.secret_key = getpass.getpass("Secret Key: ")
        args.bucket = input("Bucket 名: ").strip()

    # 也支持从环境变量读(防止命令行历史泄露)
    if not args.access_key:
        args.access_key = os.environ.get("OSN_ACCESS_KEY")
    if not args.secret_key:
        args.secret_key = os.environ.get("OSN_SECRET_KEY")
    if not args.bucket:
        args.bucket = os.environ.get("OSN_BUCKET")

    # 再从项目根目录 .env 文件读(最低优先级, 仅补缺失字段)
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
        print("❌ 缺少必要参数,使用 --interactive 或 --access-key/--secret-key/--bucket")
        return 1

    if args.list_only:
        list_bucket(args.access_key, args.secret_key, args.bucket)
    else:
        download_all(args.access_key, args.secret_key, args.bucket, args.dest)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())