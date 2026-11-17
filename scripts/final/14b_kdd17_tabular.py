"""Our tabular models on the KDD17 benchmark with the official Adv-ALSTM preprocessing (11 features x last 5 days = 55 inputs),
official split (train 2007-01-03.., val 2015-01-02.., test 2016-01-04..) and labels (+1 / -1; 0 excluded). 5 seeds. Output: results/final/kdd17_tabular.json"""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from common import RES, ROOT, LGB_PARAMS, best_threshold, dump
L = 5; files = sorted(glob.glob(str(ROOT / "data/raw/kdd17/ourpped/*.csv"))); dates = np.genfromtxt(ROOT / "data/raw/kdd17/trading_dates.csv", dtype=str)
rows = []
for f in files:
    a = np.genfromtxt(f, delimiter=","); miss = (a[:, :11] < -123320).any(1)
    for t in range(L, len(a)):
        if a[t, -2] in (1, -1) and not miss[t - L:t].any(): rows.append((dates[t], a[t - L:t, :11].ravel(), 1.0 if a[t, -2] > 0 else 0.0))
D = np.array([r[0] for r in rows]); X = np.stack([r[1] for r in rows]); y = np.array([r[2] for r in rows])
tr, va, te = (D < "2015-01-02"), (D >= "2015-01-02") & (D < "2016-01-04"), D >= "2016-01-04"
print("instances", tr.sum(), va.sum(), te.sum(), "test up-rate", round(y[te].mean(), 4))
out = {"n": dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()), test_up_rate=float(y[te].mean()))}
def ev(p, thr): pr = (p >= thr).astype(int); return dict(acc=accuracy_score(y[te], pr), mcc=matthews_corrcoef(y[te], pr), auroc=roc_auc_score(y[te], p))
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y[tr]); thr = best_threshold(y[va], m.predict_proba(X[va])[:, 1]); m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr | va], y[tr | va]); out["LR"] = ev(m.predict_proba(X[te])[:, 1], thr)
for kind in ["LightGBM", "XGBoost"]:
    rs = []
    for seed in [42, 1, 2, 3, 4]:
        if kind == "LightGBM":
            m = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **LGB_PARAMS).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)]); best = int(m.best_iteration_ or 1); pv = m.predict_proba(X[va])[:, 1]
            f = lgb.LGBMClassifier(n_estimators=best, random_state=seed, **LGB_PARAMS).fit(X[tr | va], y[tr | va])
        else:
            P = dict(learning_rate=0.03, max_depth=5, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", n_jobs=8, random_state=seed)
            m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, eval_metric="auc", **P).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False); best = int(m.best_iteration + 1); pv = m.predict_proba(X[va])[:, 1]; f = xgb.XGBClassifier(n_estimators=best, **P).fit(X[tr | va], y[tr | va])
        r = ev(f.predict_proba(X[te])[:, 1], best_threshold(y[va], pv)); r["thr0.5"] = ev(f.predict_proba(X[te])[:, 1], 0.5); r["best_iter"] = best; rs.append(r)
    out[kind] = dict(mean={k: float(np.mean([r[k] for r in rs])) for k in ["acc", "mcc", "auroc"]}, std={k: float(np.std([r[k] for r in rs])) for k in ["acc", "mcc", "auroc"]}, mean_thr05={k: float(np.mean([r["thr0.5"][k] for r in rs])) for k in ["acc", "mcc"]}, runs=rs); print(kind, out[kind]["mean"], flush=True)
out["published_causalstock_table1"] = {"LSTM": [51.18, 0.0187], "ALSTM": [51.66, 0.0316], "Adv-ALSTM": [51.69, 0.0333], "DTML": [53.53, 0.0733], "CausalStock": [56.09, 0.1235]}
dump(out, "kdd17_tabular.json")
