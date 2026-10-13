"""
逐特征组 ablation: 在 holdout 上, 把每组特征替换为全局均值 (变成 G0 弱基线)
看哪一组的"信息量"最大 → 这组定义最容易出错
"""
import sys
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

# 复用 holdout_r7_rebuild 的所有设置
spec = importlib.util.spec_from_file_location(
    "ho", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
import builtins
_orig_open = builtins.open  # 不需要, 直接 exec
ho_code = (ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py").read_text()

# 在脚本上下文里 exec 到 "df_ho_dep_full = df_ho_full[dep_mask_full].copy()" 之前
# 我们重写为把过程跑完并保存中间变量
import sys
class _Catcher:
    def __init__(self): self.vars = {}
exec_globals = {"__name__": "__ablation__"}
# 改成: 直接运行一次, 保存特征 DataFrame 到 pickle
import pickle
ho_code_with_save = ho_code.replace(
    "# ============================================================\n# 5. 用留存模型推理\n",
    """# 缓存中间结果到 pickle
import pickle
with open('/tmp/holdout_r7_features.pkl', 'wb') as f:
    pickle.dump({
        'df_ho_dep_full': df_ho_dep_full,
        'FEATS69': FEATS69,
        'bst_str': 'models/xgb_v10_R7slim_12m_seed0_final.json',
        'lut': lut,
    }, f)
print(f'[CACHE] 已保存特征 DataFrame ({df_ho_dep_full.shape}) 到 /tmp/holdout_r7_features.pkl')

# ============================================================
# 5. 用留存模型推理
"""
)
(ROOT / "notebooks/diag/d0/_holdout_r7_with_cache.py").write_text(ho_code_with_save)
print("[*] 已生成缓存脚本, 开始 ablation 分析")