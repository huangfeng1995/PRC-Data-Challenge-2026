import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path('/Users/openclaw/Documents/PRC_data_challenge'); D0 = ROOT / 'notebooks/diag/d0'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
from v15_clean_retrain import FEATS71, LOW_CARD_CATS
LP = {"objective":"regression","metric":"rmse","learning_rate":0.0147555939449363,
 "num_leaves":127,"max_depth":11,"min_data_in_leaf":5,"feature_fraction":0.6720708449966734,
 "bagging_fraction":0.9813515130184908,"bagging_freq":5,"reg_alpha":2.4624169162161342,
 "reg_lambda":1.9936339729896335,"verbose":-1,"num_threads":-1}
dep = pd.read_pickle(D0 / '_corr_feat_cache.pkl')
mo = dep['month'].to_numpy(); y = dep['TAXITIME_SEC_mvt'].to_numpy(float)
X = dep[FEATS71]
fit = np.where(np.isin(mo, [2,3,4,5,6,8,9,10]))[0]
ev  = np.where(np.isin(mo, [11,12]))[0]
ho  = np.where(np.isin(mo, [1,7]))[0]
dtr = lgb.Dataset(X.iloc[fit], label=y[fit], categorical_feature=LOW_CARD_CATS)
dev = lgb.Dataset(X.iloc[ev],  label=y[ev],  categorical_feature=LOW_CARD_CATS, reference=dtr)
b = lgb.train(dict(LP, seed=42, bagging_seed=42, feature_fraction_seed=42, data_random_seed=42),
              dtr, num_boost_round=2000, valid_sets=[dev],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
bi = b.best_iteration or 2000
def rep(tag, idx):
    p = np.clip(b.predict(X.iloc[idx], num_iteration=bi), 30, 10800); t = y[idx]
    c = (t > 0) & (t <= 10800)
    print(f'  {tag:32s} n={len(idx):>7,}  full={np.sqrt(np.mean((p-t)**2)):7.2f}  '
          f'core={np.sqrt(np.mean((p[c]-t[c])**2)):7.2f}', flush=True)
print(f'best_iter={bi}   (同一个模型, 两个测试集)', flush=True)
rep('M11/12 (折外, 可比 OOF)', ev)
rep('holdout 1+7', ho)
for m, nm in [(1, 'Jan'), (7, 'Jul')]:
    rep(nm, np.where(mo == m)[0])
print()
print('  对照: OOF缓存 M11=259.65 M12=269.00 | P0 九模型 holdout core=304.94', flush=True)
