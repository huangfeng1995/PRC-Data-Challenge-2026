"""
阶段 0:固化特征列清单(basic30 / cnt42),写 JSON,供 train / holdout / ranking 三端共享。
=========================================================================

依据阶段 0 红线:
- basic30 = BASELINE_FEATURE_COLS(30 列)
- cnt42   = basic30 + cnt_dep_* (12 列,4 机场 × 3 窗口,但只保留 cnt,不要 mean/p90)

输出:
    models/feature_lists/feat_basic30.json
    models/feature_lists/feat_cnt42.json
"""
import sys, json
from pathlib import Path

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

from src.features import BASELINE_FEATURE_COLS
from src.data_loader import load_training
from src.features import build_features
from src.features_aggregate import build_features_v2

OUT = ROOT / "models" / "feature_lists"
OUT.mkdir(exist_ok=True, parents=True)

print("加载训练数据(只为获取列名)...")
df = load_training("data/raw")
df = build_features_v2(df, include_aggregate=True, include_lead=True)

basic_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns]
print(f"basic 列数: {len(basic_cols)}")

# cnt_dep_* 列(去掉 mean/p90)
cnt_cols = [c for c in df.columns if c.startswith("cnt_dep_") and c.endswith("m")]
print(f"cnt_dep_* 列数: {len(cnt_cols)}: {cnt_cols}")

basic30 = basic_cols
cnt42 = basic_cols + cnt_cols

(OUT / "feat_basic30.json").write_text(json.dumps(basic30, indent=2))
(OUT / "feat_cnt42.json").write_text(json.dumps(cnt42, indent=2))

print(f"\nfeat_basic30.json: {len(basic30)} 列")
print(f"feat_cnt42.json: {len(cnt42)} 列")
print(f"保存到 {OUT}")
