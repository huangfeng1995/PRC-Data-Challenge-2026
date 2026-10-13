# MinIO Data Download Tutorial (Command Line + GUI)

**Use case**: after you get the OSN access key, download the competition data to your local machine
**Estimated time**: 10 minutes for the first run, 5 minutes afterwards
**Author**: jubilant-puzzle team

---

## 0. Prerequisites

### 0.1 The 4 things you need

| Item | Source | Notes |
|---|---|---|
| OSN account | https://opensky-network.org/ | You already have one |
| **Access Key** | OSN Dashboard → MinIO Credentials | It can only be generated after your application is approved |
| **Secret Key** | Same as above, generated together with the Access Key | **Store it safely, never leak it** |
| Your **bucket name** | Visible on the OSN Dashboard, usually named `dc2026-<team-name>` or similar | Assigned after you submit the form |

### 0.2 How to generate the Access Key

Log in to the OSN Dashboard → find "MinIO credentials" or a similar entry → generate a new access/secret pair → **copy and save it** (it is only displayed once!)

---

## Option A: command line (the `mc` client) ⭐⭐⭐ Recommended

**Pros**: fast, automatable, and it can be written into a reproducible script in the README
**Cons**: you have to type a few lines of commands

### Step 1: install the MinIO client `mc`

#### macOS (brew)
```bash
brew install minio/stable/mc
```
> If brew does not work, try the following:

#### macOS (download the binary directly, no brew needed)
```bash
curl -O https://dl.min.io/client/mc/release/darwin-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/
```

#### Linux
```bash
curl -O https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/
```

#### Windows
Go to https://dl.min.io/client/mc/release/windows-amd64/mc.exe to download it, and add `mc.exe` to PATH

### Step 2: verify the installation

```bash
mc --version
```
It should output something like `mc version RELEASE.2024-...`

### Step 3: configure the OSN endpoint

Replace `<YOUR_ACCESS_KEY>` and `<YOUR_SECRET_KEY>` in the command below with the values you received:

```bash
mc alias set osn https://datacomp.opensky-network.org <YOUR_ACCESS_KEY> <YOUR_SECRET_KEY>
```

On success you will see:
```
Added `osn` successfully.
```

### Step 4: see what is in the bucket

```bash
mc ls osn/
```
It should list the bucket assigned to your team, for example:
```
[2026-09-15 14:23]     0B prc-2026-jubilant-puzzle/
```
Write down this bucket name (it is used below).

### Step 5: see exactly what is inside the bucket

```bash
mc ls osn/<your-bucket-name>/
```
You will see something like:
```
[2026-09-15 14:23]  22MB training_2025-01-01_2025-02-01.parquet
[2026-09-15 14:23]  19MB training_2025-02-01_2025-03-01.parquet
...
[2026-09-15 14:23]  27MB ranking.parquet
[2026-09-15 14:23] 1.1MB submitting.parquet
```

### Step 6: download the data into the project directory

Create the data directory:
```bash
mkdir -p ~/Documents/PRC_data_challenge/data/raw
cd ~/Documents/PRC_data_challenge
```

**Full download** (12 training files + ranking + submitting):
```bash
mc cp --recursive osn/<your-bucket-name>/training_2025-* data/raw/
mc cp osn/<your-bucket-name>/ranking.parquet data/raw/
mc cp osn/<your-bucket-name>/submitting.parquet data/raw/
```

Download **only the latest day** (quick validation):
```bash
mc cp osn/<your-bucket-name>/training_2025-12-01_2026-01-01.parquet data/raw/
```

### Step 7: verify the download succeeded

```bash
ls -lh data/raw/
```
You should see:
```
total 290M
-rw-r--r--  22M  training_2025-01-01_2025-02-01.parquet
-rw-r--r--  19M  training_2025-02-01_2025-03-01.parquet
...
-rw-r--r-- 1.1M  submitting.parquet
```

### Step 8 (optional): run EDA for the first time to confirm the data is complete

```bash
cd ~/Documents/PRC_data_challenge
python -m scripts.eda --data-dir data/raw --output-dir notebooks/eda
```

After 5–10 minutes, look at the 8 generated images in `notebooks/eda/` and confirm:
- taxi_out_distribution.png ← you should see the distribution
- by_airport.png ← you should see the boxplot of the 10 airports

---

## Option B: GUI (browser-based MinIO Console)

**Pros**: visual, suitable for students who do not want to type commands
**Cons**: only one file can be downloaded at a time, and large batches require many repeated clicks

### Step 1: open the MinIO Console

Open it in the browser:
```
https://datacomp.opensky-network.org
```
> **Note**: this is the same endpoint, and the Console also goes through it

If the browser asks you to log in, enter your **Access Key** as the user name and your **Secret Key** as the password.

### Step 2: navigate to your bucket

Left menu → Object Browser → select your bucket (something like `prc-2026-jubilant-puzzle`)

### Step 3: download the files one by one

Each file has a ⋮ menu on its right → click **Download**

> ⚠️ **Slow**: 12 training files + 2 others = **14 downloads**, and you have to click each time
>
> Recommendation: use the command line from Option A, it is **done in a few seconds**

---

## Option C: use Python (the `minio` package) ⭐⭐

**Pros**: can be written into a Python script and automated
**Cons**: you need to run `pip install minio` first

### Step 1: install the package
```bash
pip install minio
```

### Step 2: download script

Save it as `scripts/download_data.py`:

```python
"""
Download the PRC Data Challenge 2026 data.
Usage: python -m scripts.download_data --access-key <KEY> --secret-key <SECRET> --bucket <BUCKET>
"""
import argparse
from pathlib import Path
from minio import Minio
from minio.error import S3Error


def download_all(access_key: str, secret_key: str, bucket: str, dest: str = "data/raw"):
    client = Minio(
        "datacomp.opensky-network.org",
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
    )

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    objects = list(client.list_objects(bucket, recursive=True))
    print(f"Found {len(objects)} files, starting download...")

    for obj in objects:
        target = dest_path / obj.object_name
        if target.exists():
            print(f"  Skipping (already exists): {obj.object_name}")
            continue
        print(f"  Downloading: {obj.object_name} ({obj.size / 1e6:.1f} MB)")
        client.fget_object(bucket, obj.object_name, str(target))

    print(f"\n✅ Done, files are in {dest_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dest", default="data/raw")
    args = parser.parse_args()
    download_all(args.access_key, args.secret_key, args.bucket, args.dest)
```

### Step 3: run it

```bash
python -m scripts.download_data \
    --access-key "YOUR_ACCESS_KEY" \
    --secret-key "YOUR_SECRET_KEY" \
    --bucket "your_bucket_name"
```

---

## FAQ

### Q1: `mc: command not found`
**Answer**: either brew is not installed, or PATH is not configured. Use the "download the binary directly" option.

### Q2: `mc cp` reports `AccessDenied`
**Answer**:
- Check whether the access key / secret key were copied incorrectly (especially the trailing `=` `+` `/`)
- Check whether the bucket name is misspelled (re-confirm with `mc ls osn/`)
- Make sure you **have already received the OSN approval email**

### Q3: the download was interrupted halfway
**Answer**: `mc cp` supports resuming interrupted transfers, so just run the same command again; it will automatically skip the files already downloaded.

### Q4: the network is too slow
**Answer**:
- Add `--quiet` to reduce output overhead
- Download in batches (January first, then February, ...)
- Users in mainland China should consider using a proxy

### Q5: I do not know the bucket name
**Answer**:
```bash
mc ls osn/
```
It will list all the buckets you can access.

### Q6: `mc` reports "cannot be opened because the developer cannot be verified" on macOS
**Answer**:
```bash
xattr -d com.apple.quarantine $(which mc)
# Or directly open "System Settings → Privacy & Security → Open Anyway"
```

### Q7: the files are downloaded but EDA reports an error
**Answer**:
- Check whether the files under `data/raw/` are complete (`ls data/raw/ | wc -l` should be 14)
- Check whether the file sizes are correct (`ls -lh data/raw/`, each one should be in the MB range)
- If any file is unusually small (a few KB), run `mc cp` again for that one

---

## What you should see after it works

```bash
$ ls -lh ~/Documents/PRC_data_challenge/data/raw/
# 12 training_2025-XX-XX_2025-YY-YY.parquet files, each 19-25 MB
# ranking.parquet, about 27 MB
# submitting.parquet, about 1.1 MB
# about 280 MB in total

$ du -sh ~/Documents/PRC_data_challenge/data/raw/
280M    /Users/xxx/Documents/PRC_data_challenge/data/raw
```

Once the data is complete, **tell me immediately** and I will run the first baseline for you.