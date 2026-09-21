# MinIO 数据下载教程(命令行 + GUI 两种)

**适用场景**:拿到 OSN access key 之后,把比赛数据下载到本地
**预计时间**:首次 10 分钟,后续 5 分钟
**作者**:jubilant-puzzle team

---

## 0. 前置准备

### 0.1 你需要的 4 样东西

| 项目 | 来源 | 备注 |
|---|---|---|
| OSN 账号 | https://opensky-network.org/ | 你已经有了 |
| **Access Key** | OSN Dashboard → MinIO 凭证 | 审核通过后才能生成 |
| **Secret Key** | 同上,跟 Access Key 一起生成 | **妥善保管,不能泄露** |
| 你的 **bucket 名** | OSN Dashboard 上能看到,通常叫 `dc2026-<team-name>` 或类似 | 提交表单后会分配 |

### 0.2 怎么生成 Access Key

登录 OSN Dashboard → 找 "MinIO credentials" 或类似入口 → 生成新的 access/secret pair → **复制保存**(只显示一次!)

---

## 方案 A:命令行(`mc` 客户端)⭐⭐⭐ 推荐

**优点**:速度快、可自动化、可以写进 README 复现脚本里
**缺点**:要敲几行命令

### 步骤 1:安装 MinIO 客户端 `mc`

#### macOS(brew)
```bash
brew install minio/stable/mc
```
> 如果 brew 装不上,试试下面这个:

#### macOS(直接下载二进制,无需 brew)
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
去 https://dl.min.io/client/mc/release/windows-amd64/mc.exe 下载,把 `mc.exe` 加到 PATH

### 步骤 2:验证安装

```bash
mc --version
```
应该输出类似 `mc version RELEASE.2024-...`

### 步骤 3:配置 OSN endpoint

把下面命令里的 `<YOUR_ACCESS_KEY>` 和 `<YOUR_SECRET_KEY>` 替换成你拿到的值:

```bash
mc alias set osn https://datacomp.opensky-network.org <YOUR_ACCESS_KEY> <YOUR_SECRET_KEY>
```

成功会看到:
```
Added `osn` successfully.
```

### 步骤 4:看看 bucket 里有什么

```bash
mc ls osn/
```
应该列出你团队对应的 bucket,例如:
```
[2026-09-15 14:23]     0B prc-2026-jubilant-puzzle/
```
记下这个 bucket 名(下面要用)。

### 步骤 5:看看 bucket 里具体有什么

```bash
mc ls osn/<your-bucket-name>/
```
你会看到类似:
```
[2026-09-15 14:23]  22MB training_2025-01-01_2025-02-01.parquet
[2026-09-15 14:23]  19MB training_2025-02-01_2025-03-01.parquet
...
[2026-09-15 14:23]  27MB ranking.parquet
[2026-09-15 14:23] 1.1MB submitting.parquet
```

### 步骤 6:下载数据到项目目录

创建数据目录:
```bash
mkdir -p ~/Documents/PRC_data_challenge/data/raw
cd ~/Documents/PRC_data_challenge
```

**全量下载**(12 个训练 + ranking + submitting):
```bash
mc cp --recursive osn/<your-bucket-name>/training_2025-* data/raw/
mc cp osn/<your-bucket-name>/ranking.parquet data/raw/
mc cp osn/<your-bucket-name>/submitting.parquet data/raw/
```

只下载**最新一天**(快速验证):
```bash
mc cp osn/<your-bucket-name>/training_2025-12-01_2026-01-01.parquet data/raw/
```

### 步骤 7:验证下载成功

```bash
ls -lh data/raw/
```
应该看到:
```
total 290M
-rw-r--r--  22M  training_2025-01-01_2025-02-01.parquet
-rw-r--r--  19M  training_2025-02-01_2025-03-01.parquet
...
-rw-r--r-- 1.1M  submitting.parquet
```

### 步骤 8(可选):第一次跑 EDA,确认数据完整

```bash
cd ~/Documents/PRC_data_challenge
python -m scripts.eda --data-dir data/raw --output-dir notebooks/eda
```

5–10 分钟后,看 `notebooks/eda/` 里生成的 8 张图,确认:
- taxi_out_distribution.png ← 应该看到分布
- by_airport.png ← 应该看到 10 个机场的 boxplot

---

## 方案 B:GUI(浏览器-based MinIO Console)

**优点**:可视化,适合不想敲命令的同学
**缺点**:一次只能下一个文件,大批量要重复点很多次

### 步骤 1:打开 MinIO Console

浏览器打开:
```
https://datacomp.opensky-network.org
```
> **注意**:这是同一个 endpoint,Console 也走这里

如果浏览器提示需要登录,输入你的 **Access Key** 作为用户名,**Secret Key** 作为密码。

### 步骤 2:导航到你的 bucket

左侧菜单 → Object Browser → 选择你的 bucket(类似 `prc-2026-jubilant-puzzle`)

### 步骤 3:逐个下载文件

每个文件右侧有一个 ⋮ 菜单 → 点 **Download**

> ⚠️ **慢**:12 个训练文件 + 2 个其他 = **14 次下载**,每次要点一次
>
> 建议:用方案 A 的命令行,**几秒搞定**

---

## 方案 C:用 Python(`minio` 包)⭐⭐

**优点**:可以写进 Python 脚本,自动化
**缺点**:要先 `pip install minio`

### 步骤 1:装包
```bash
pip install minio
```

### 步骤 2:下载脚本

保存为 `scripts/download_data.py`:

```python
"""
下载 PRC Data Challenge 2026 数据
用法:python -m scripts.download_data --access-key <KEY> --secret-key <SECRET> --bucket <BUCKET>
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
    print(f"找到 {len(objects)} 个文件,开始下载...")

    for obj in objects:
        target = dest_path / obj.object_name
        if target.exists():
            print(f"  跳过(已存在): {obj.object_name}")
            continue
        print(f"  下载: {obj.object_name} ({obj.size / 1e6:.1f} MB)")
        client.fget_object(bucket, obj.object_name, str(target))

    print(f"\n✅ 完成,文件在 {dest_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dest", default="data/raw")
    args = parser.parse_args()
    download_all(args.access_key, args.secret_key, args.bucket, args.dest)
```

### 步骤 3:运行

```bash
python -m scripts.download_data \
    --access-key "你的_ACCESS_KEY" \
    --secret-key "你的_SECRET_KEY" \
    --bucket "你的_bucket 名"
```

---

## 常见问题

### Q1: `mc: command not found`
**答**:要么 brew 没装,要么 PATH 没配。用"直接下载二进制"那条方案。

### Q2: `mc cp` 报 `AccessDenied`
**答**:
- 检查 access key / secret key 有没有复制错(尤其是末尾的 `=` `+` `/`)
- 检查 bucket 名有没有拼错(`mc ls osn/` 重新确认)
- 确认你**已经收到 OSN 的审核通过邮件**

### Q3: 下载到一半中断了
**答**:`mc cp` 支持断点续传,重新跑同一条命令就行,会自动跳过已下载的。

### Q4: 网络太慢
**答**:
- 加 `--quiet` 减少输出开销
- 分批下载(先下 1 月份,再下 2 月...)
- 国内用户考虑挂代理

### Q5: 不知道 bucket 名
**答**:
```bash
mc ls osn/
```
会列出你能访问的所有 bucket。

### Q6: macOS 上 `mc` 报 "cannot be opened because the developer cannot be verified"
**答**:
```bash
xattr -d com.apple.quarantine $(which mc)
# 或者直接打开"系统设置 → 隐私与安全性 → 仍要打开"
```

### Q7: 文件下完了但 EDA 报错
**答**:
- 检查 `data/raw/` 下文件是不是齐全(`ls data/raw/ | wc -l` 应该是 14)
- 检查文件大小是不是对的(`ls -lh data/raw/`,每个都应该是 MB 级)
- 如果哪个文件特别小(几 KB),重新 `mc cp` 那一个

---

## 成功后你应该看到的状态

```bash
$ ls -lh ~/Documents/PRC_data_challenge/data/raw/
# 12 个 training_2025-XX-XX_2025-YY-YY.parquet,每个 19-25 MB
# ranking.parquet,约 27 MB
# submitting.parquet,约 1.1 MB
# 总共约 280 MB

$ du -sh ~/Documents/PRC_data_challenge/data/raw/
280M    /Users/xxx/Documents/PRC_data_challenge/data/raw
```

数据齐了之后,**第一时间告诉我**,我帮你跑第一个 baseline。