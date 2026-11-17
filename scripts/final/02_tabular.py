"""Tabular models under the common walk-forward protocol.
Arms: all 31 non-empty source subsets (for exact source Shapley) x {LR, LightGBM}; key arms x 5 seeds for LightGBM and XGBoost; rule baselines.
Output: results/final/oos_tabular.parquet (one column per model|arm|seed), results/final/tabular_fit_info.json
"""
import sys, json, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from common import *

df = load(); LABEL = "y_up"
SUBSETS = {}
for r in range(1, 6):
    for combo in itertools.combinations(list(SRC), r):
        SUBSETS["+".join(combo)] = sum((SRC[k] for k in combo), [])
KEY = {"Full": ALL, "S4-only": SRC["S4"], "TW-only": TW}
FOLDS = list(folds(df))

def run_lgb(feats, seed, label=LABEL, params=None, data=None):
    d = data if data is not None else df
    prob = pd.Series(np.nan, index=d.index); info = {}
    P = dict(LGB_PARAMS, **(params or {}))
    for yr, tr, itr, iva, te in (FOLDS if data is None else list(folds(d))):
        itr, iva, tr, te = [i[d.loc[i, label].notna()] for i in (itr, iva, tr, te)]
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **P)
        m.fit(d.loc[itr, feats], d.loc[itr, label], eval_set=[(d.loc[iva, feats], d.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        best = int(m.best_iteration_ or 1); thr = best_threshold(d.loc[iva, label].values, m.predict_proba(d.loc[iva, feats])[:, 1])
        f = lgb.LGBMClassifier(n_estimators=best, random_state=seed, **P).fit(d.loc[tr, feats], d.loc[tr, label])
        prob.loc[te] = f.predict_proba(d.loc[te, feats])[:, 1]; info[str(yr)] = dict(best_iter=best, thr=thr)
    return prob, info

def run_xgb(feats, seed, label=LABEL):
    prob = pd.Series(np.nan, index=df.index); info = {}
    P = dict(learning_rate=0.03, max_depth=5, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", n_jobs=4, random_state=seed, eval_metric="auc")
    for yr, tr, itr, iva, te in FOLDS:
        m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, **P)
        m.fit(df.loc[itr, feats], df.loc[itr, label], eval_set=[(df.loc[iva, feats], df.loc[iva, label])], verbose=False)
        best = int(m.best_iteration + 1); thr = best_threshold(df.loc[iva, label].values, m.predict_proba(df.loc[iva, feats])[:, 1])
        f = xgb.XGBClassifier(n_estimators=best, **{k: v for k, v in P.items() if k != "eval_metric"}).fit(df.loc[tr, feats], df.loc[tr, label])
        prob.loc[te] = f.predict_proba(df.loc[te, feats])[:, 1]; info[str(yr)] = dict(best_iter=best, thr=thr)
    return prob, info

def run_lr(feats, label=LABEL, C=0.1):
    prob = pd.Series(np.nan, index=df.index); info = {}
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    for yr, tr, itr, iva, te in FOLDS:
        thr = best_threshold(df.loc[iva, label].values, mk().fit(df.loc[itr, feats], df.loc[itr, label]).predict_proba(df.loc[iva, feats])[:, 1])
        prob.loc[te] = mk().fit(df.loc[tr, feats], df.loc[tr, label]).predict_proba(df.loc[te, feats])[:, 1]; info[str(yr)] = dict(thr=thr)
    return prob, info

def job(kind, arm, feats, seed):
    t = time.time()
    p, info = run_lgb(feats, seed) if kind == "LightGBM" else run_xgb(feats, seed) if kind == "XGBoost" else run_lr(feats)
    print(f"{kind:8s} {arm:16s} seed {seed} auroc={auc_stat(df.loc[p.notna(), LABEL].values, p.dropna().values):.4f} ({time.time()-t:.0f}s)", flush=True)
    return f"{kind}|{arm}|{seed}", p, info

if __name__ == "__main__":
    jobs = [("LightGBM", a, f, 42) for a, f in SUBSETS.items()] + [("LR", a, f, 0) for a, f in SUBSETS.items()]
    jobs += [("LightGBM", a, f, s) for a, f in KEY.items() for s in [1, 2, 3, 4]]
    jobs += [("XGBoost", a, f, s) for a, f in KEY.items() for s in [42, 1, 2, 3, 4]] + [("XGBoost", f"{k}", v, 42) for k, v in SRC.items() if k != "S4"]
    out = Parallel(n_jobs=6, prefer="threads")(delayed(job)(*j) for j in jobs)
    te_mask = df.year.isin(TEST_YEARS)
    res = df.loc[te_mask, ["date", "date_next", "stock_id", "sector_en", "year", "y_up", "y_tr", "y_intra", "y_gap", "y_rel", "ret_next", "ret_next_tr", "ret_intra", "ret_gap", "open_next", "close", "close_next", "taiex_ret_next", "us_sox_ret", "ret1", "vol20"]].copy()
    info = {}
    for name, p, inf in out: res[name] = p.loc[te_mask].values; info[name] = inf
    # rule baselines (scores) -- majority = constant, previous-day sign, SOX sign, SOX continuous, per-stock / per-sector train base rate
    res["Rule|majority|0"] = 0.0; res["Rule|prev-day sign|0"] = (res.ret1 > 0).astype(float); res["Rule|SOX sign|0"] = (res.us_sox_ret > 0).astype(float); res["Rule|SOX return score|0"] = res.us_sox_ret
    for yr, tr, *_ in FOLDS:
        m = res.year == yr
        res.loc[m, "Rule|stock base-rate|0"] = res.loc[m, "stock_id"].map(df.loc[tr].groupby("stock_id").y_up.mean())
        res.loc[m, "Rule|sector base-rate|0"] = res.loc[m, "sector_en"].map(df.loc[tr].groupby("sector_en").y_up.mean())
    res.to_parquet(RES / "oos_tabular.parquet"); dump(info, "tabular_fit_info.json"); print("saved oos", res.shape)
