"""Place our tabular pipeline on the public StockNet benchmark (Xu & Cohen, ACL 2018; 88 U.S. stocks) used by ALSTM / Adv-ALSTM /
PULSE-KAN (price-only task) so that the comparison is on the SAME dataset, split and label rule.
Protocol (StockNet): train 2014-01-01..2015-07-31, val 2015-08-01..2015-09-30, test 2015-10-01..2015-12-31;
label from the adjusted-close movement ratio: >= +0.55 % up, <= -0.50 % down, in-between removed; 5-day look-back.
Our models: price-only features (returns, MA ratios, volatility, range, volume ratio; same spirit as S1) -> LR / LightGBM / XGBoost,
early stopping + threshold on the validation split; 5 seeds. Metrics: ACC, MCC (as in the literature) + AUROC.
Output: results/final/sota_stocknet.json
"""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, matthews_corrcoef
from common import *
RAW = ROOT / "data/raw/stocknet"; frames = []
for f in sorted(glob.glob(str(RAW / "*.csv"))):
    d = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date"); d["stock"] = Path(f).stem; frames.append(d)
px = pd.concat(frames, ignore_index=True).rename(columns=str.lower).rename(columns={"adj close": "adj"})
g = px.groupby("stock")
px["mov"] = g["adj"].shift(-1) / px["adj"] - 1
for k in (1, 2, 3, 5, 10, 20): px[f"ret{k}"] = g["adj"].pct_change(k)
for w in (5, 20, 60): px[f"ma{w}_ratio"] = px["adj"] / g["adj"].transform(lambda s: s.rolling(w).mean()) - 1
px["vol20"] = g["ret1"].transform(lambda s: s.rolling(20).std()); px["volume_ratio20"] = px["volume"] / g["volume"].transform(lambda s: s.rolling(20).mean())
px["hl_range"] = (px["high"] - px["low"]) / px["close"]; px["gap_open"] = px["open"] / g["close"].shift(1) - 1
delta = g["adj"].diff(); up = delta.clip(lower=0); dn = -delta.clip(upper=0)
px["rsi14"] = 100 - 100 / (1 + up.groupby(px["stock"]).transform(lambda s: s.rolling(14).mean()) / dn.groupby(px["stock"]).transform(lambda s: s.rolling(14).mean()))
# lagged raw movements (the information the sequence baselines see: 5-day look-back of normalised prices)
for k in range(1, 5): px[f"ret1_lag{k}"] = g["ret1"].shift(k)
FEATS = ["ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ma5_ratio", "ma20_ratio", "ma60_ratio", "vol20", "volume_ratio20", "hl_range", "gap_open", "rsi14"] + [f"ret1_lag{k}" for k in range(1, 5)]
px["y"] = np.where(px.mov >= 0.0055, 1, np.where(px.mov <= -0.005, 0, np.nan))
d = px[(px.date >= "2014-01-01") & (px.date < "2016-01-01")].dropna(subset=["y", "ma60_ratio"]).copy()
tr = d[d.date < "2015-08-01"]; va = d[(d.date >= "2015-08-01") & (d.date < "2015-10-01")]; te = d[d.date >= "2015-10-01"]
print("rows train/val/test", len(tr), len(va), len(te), "| up-rate test", round(te.y.mean(), 4))
out = {"n": dict(train=len(tr), val=len(va), test=len(te), test_up_rate=float(te.y.mean()), n_stocks=int(d.stock.nunique()))}
def evalp(p, thr): pr = (p >= thr).astype(int); return dict(acc=accuracy_score(te.y, pr), mcc=matthews_corrcoef(te.y, pr), auroc=roc_auc_score(te.y, p), thr=float(thr))
# LR
mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
m = mk().fit(tr[FEATS], tr.y); thr = best_threshold(va.y.values, m.predict_proba(va[FEATS])[:, 1]); m = mk().fit(pd.concat([tr, va])[FEATS], pd.concat([tr, va]).y)
out["LR"] = evalp(m.predict_proba(te[FEATS])[:, 1], thr); out["LR_thr0.5"] = evalp(m.predict_proba(te[FEATS])[:, 1], 0.5)
# LightGBM / XGBoost, 5 seeds
for kind in ["LightGBM", "XGBoost"]:
    rows = []
    for seed in [42, 1, 2, 3, 4]:
        if kind == "LightGBM":
            m = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **dict(LGB_PARAMS, min_child_samples=100)).fit(tr[FEATS], tr.y, eval_set=[(va[FEATS], va.y)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
            best = int(m.best_iteration_ or 1); pv = m.predict_proba(va[FEATS])[:, 1]
            f = lgb.LGBMClassifier(n_estimators=best, random_state=seed, **dict(LGB_PARAMS, min_child_samples=100)).fit(pd.concat([tr, va])[FEATS], pd.concat([tr, va]).y)
        else:
            P = dict(learning_rate=0.03, max_depth=5, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", n_jobs=8, random_state=seed)
            m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, eval_metric="auc", **P).fit(tr[FEATS], tr.y, eval_set=[(va[FEATS], va.y)], verbose=False)
            best = int(m.best_iteration + 1); pv = m.predict_proba(va[FEATS])[:, 1]; f = xgb.XGBClassifier(n_estimators=best, **P).fit(pd.concat([tr, va])[FEATS], pd.concat([tr, va]).y)
        thr = best_threshold(va.y.values, pv); r = evalp(f.predict_proba(te[FEATS])[:, 1], thr); r["best_iter"] = best; r["thr0.5"] = evalp(f.predict_proba(te[FEATS])[:, 1], 0.5); rows.append(r)
    out[kind] = dict(mean={k: float(np.mean([r[k] for r in rows])) for k in ["acc", "mcc", "auroc"]}, std={k: float(np.std([r[k] for r in rows])) for k in ["acc", "mcc", "auroc"]},
                     mean_thr05={k: float(np.mean([r["thr0.5"][k] for r in rows])) for k in ["acc", "mcc"]}, runs=rows)
    print(kind, out[kind]["mean"], out[kind]["std"], flush=True)
# published numbers on the same benchmark (price-only StockNet task) -- verified from PULSE-KAN Table 1 (exports/PULSE-KAN_Mathematics_2026.pdf)
out["published_pulsekan_table1_stocknet"] = {"LSTM": [0.5148, 0.0273], "GRU": [0.5187, 0.0351], "Transformer": [0.5230, 0.0442], "ALSTM": [0.5610, 0.1234], "Adv-ALSTM": [0.5624, 0.1263], "PULSE-KAN (clean)": [0.5717, 0.1437], "PULSE-KAN (adv)": [0.5756, 0.1509]}
out["majority_acc_test"] = float(max(te.y.mean(), 1 - te.y.mean()))
dump(out, "sota_stocknet.json")
