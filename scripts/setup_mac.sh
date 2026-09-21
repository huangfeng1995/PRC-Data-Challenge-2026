#!/bin/bash
# 一键环境准备脚本 —— 回家后第一件事
# 用法:bash scripts/setup_mac.sh
set -e

echo "==============================================="
echo "  PRC 2026 Mac 环境一键准备"
echo "==============================================="
echo ""

# 1. 检查 brew
if ! command -v brew &> /dev/null; then
    echo "❌ 没装 Homebrew,先装 brew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi
echo "✅ Homebrew OK"

# 2. 装 libomp(关键!)
if ! brew list libomp &> /dev/null; then
    echo "📦 装 libomp..."
    brew install libomp
else
    echo "✅ libomp 已装"
fi

# 3. 装 Python 依赖
echo ""
echo "📦 装 Python 依赖..."
pip3 install --upgrade pip --break-system-packages 2>&1 | tail -1
pip3 install --break-system-packages \
    pandas numpy pyarrow \
    scikit-learn matplotlib seaborn \
    xgboost lightgbm catboost \
    mlflow optuna minio tqdm 2>&1 | tail -3

# 4. 验证
echo ""
echo "==============================================="
echo "  验证"
echo "==============================================="
python3 -c "
import sys
print(f'Python {sys.version.split()[0]}')

try:
    import xgboost as xgb
    print(f'✅ XGBoost {xgb.__version__}')
except Exception as e:
    print(f'❌ XGBoost: {e}')

try:
    import lightgbm as lgb
    print(f'✅ LightGBM {lgb.__version__}')
except Exception as e:
    print(f'❌ LightGBM: {e}')

try:
    import catboost
    print(f'✅ CatBoost {catboost.__version__}')
except Exception as e:
    print(f'❌ CatBoost: {e}')

try:
    import mlflow
    print(f'✅ MLflow {mlflow.__version__}')
except Exception as e:
    print(f'❌ MLflow: {e}')

try:
    import optuna
    print(f'✅ Optuna {optuna.__version__}')
except Exception as e:
    print(f'❌ Optuna: {e}')

try:
    from minio import Minio
    print(f'✅ Minio OK')
except Exception as e:
    print(f'❌ Minio: {e}')
"

# 5. 验证数据
echo ""
echo "==============================================="
echo "  数据验证"
echo "==============================================="
if [ -d "data/raw" ]; then
    n_files=$(ls data/raw/*.parquet 2>/dev/null | wc -l)
    if [ "$n_files" -ge 14 ]; then
        echo "✅ 数据完整: $n_files 个 parquet 文件"
    else
        echo "⚠️  数据不完整: 只有 $n_files 个文件,需要重新下载"
    fi
else
    echo "❌ data/raw/ 不存在,需要先下载数据"
fi

echo ""
echo "==============================================="
echo "  下一步"
echo "==============================================="
echo "  1. cd ~/Documents/PRC_data_challenge"
echo "  2. python -m src.train_xgb --data-dir data/raw --feature-set basic"
echo "  3. python -m src.train_xgb --data-dir data/raw --feature-set all"
echo "  4. python -m src.train_ensemble --data-dir data/raw --feature-set all --models xgb,lgb,cat --n-seeds 5"
echo "  5. python -m src.predict_ensemble --output submissions/jubilant-puzzle_v1.parquet"
echo ""
echo "  上传:"
echo "  mc cp submissions/jubilant-puzzle_v1.parquet osn/prc-2026-jubilant-puzzle/"
echo ""
echo "🎉 完成!祝比赛顺利!"