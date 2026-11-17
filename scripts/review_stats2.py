"""Day-clustered CIs for remaining ablations + cross-sectional AUROC CIs + day-level paired accuracy tests."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data/processed/dataset.parquet"); te = pd.read_parquet(ROOT / "data/processed/review_oos_preds.parquet")
SRC = {"S1": ["ret1","ret2","ret3","ret5","ret10","ret20","ma5_ratio","ma20_ratio","ma60_ratio","vol20","volume_ratio20","hl_range","gap_open","rsi14","spread"],
 "S2": ["foreign_net_ratio","foreign_net_ratio5","trust_net_ratio","trust_net_ratio5","dealer_net_ratio","dealer_net_ratio5","inst_net_ratio","margin_chg","margin_chg5","short_chg"],
 "S3": ["taiex_ret1","taiex_ret5","taiex_ma20_ratio","taiex_vol_ratio20","taiex_vol20","rel_ret1"],
 "S4": ["us_sp500_ret","us_nasdaq_ret","us_sox_ret","us_tsm_adr_ret","us_dxy_ret","us_nikkei_ret","us_vix_level","us_vix_chg","us_usdtwd_chg","us_10y_chg","us_stale"],
 "S5": ["weekday","month","is_month_end"]}
ALL = sum(SRC.values(), []); TEST = [2023, 2024, 2025, 2026]
def lg(): return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=42, n_jobs=4)
for name, f in {"All-S4": [c for c in ALL if c not in SRC["S4"]], "All-S5": [c for c in ALL if c not in SRC["S5"]], "S4+S1": SRC["S4"] + SRC["S1"], "S4+S2": SRC["S4"] + SRC["S2"], "S4+S3": SRC["S4"] + SRC["S3"]}.items():
    col = f"LightGBM|{name}"
    if col in te: continue
    p = pd.Series(np.nan, index=te.index)
    for yr in TEST:
        tr = df[df.year < yr]; m = te.year == yr
        p[m] = lg().fit(tr[f], tr.y_up).predict_proba(te.loc[m, f])[:, 1]
    te[col] = p; print("done", name, flush=True)
te.to_parquet(ROOT / "data/processed/review_oos_preds.parquet")
y = te.y_up.values; grp = te.groupby("date").indices; dates = np.array(list(grp)); rng = np.random.default_rng(11)
def clu(a, b, B=400):
    pa, pb = te[a].values, te[b].values; d = roc_auc_score(y, pb) - roc_auc_score(y, pa); v = []
    for _ in range(B):
        j = np.concatenate([grp[x] for x in rng.choice(dates, len(dates))]); v.append(roc_auc_score(y[j], pb[j]) - roc_auc_score(y[j], pa[j]))
    return [round(float(d), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
K = "LightGBM|"
res = {f"drop {k}": clu(K + f"All-{k}", K + "All") for k in ["S1", "S2", "S3", "S4", "S5"]}
res["All vs S4"] = clu(K + "S4", K + "All"); res["S1-S3 joint, add on S4"] = clu(K + "S4", K + "All-S5")
for k in ["S1", "S2", "S3"]: res[f"S4+{k} vs S4"] = clu(K + "S4", K + f"S4+{k}")
res["LR S1-S3 -> S1-S4"] = clu("LR|S123", "LR|S1234")
# cross-sectional (within-day) AUROC, CI by resampling days
def daily_auc(col):
    out = {}
    for d, ii in grp.items():
        yy = y[ii]
        if 0 < yy.sum() < len(yy): out[d] = roc_auc_score(yy, te[col].values[ii])
    return pd.Series(out)
cs = {c: daily_auc(c) for c in [K + "All", K + "S4", K + "S123", "LR|S123", "LR|S1234"]}
def ci_mean(s, B=2000):
    v = s.values; b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
csr = {c: ci_mean(s) for c, s in cs.items()}
dd = (cs[K + "All"] - cs[K + "S123"]).dropna(); csr["All minus S1-S3 (paired days)"] = ci_mean(dd)
# day-level paired accuracy tests
def daytest(a, b):
    ca = ((te[a] > .5) == te.y_up).groupby(te.date).mean(); cb = ((te[b] > .5) == te.y_up).groupby(te.date).mean(); d = cb - ca
    return {"acc_diff": round(float(d.mean()), 4), "t": round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), 2), "days": int(len(d))}
tests = {"LR S1-S3 -> S1-S4": daytest("LR|S123", "LR|S1234"), "LGBM S1-S3 -> All": daytest(K + "S123", K + "All"), "LGBM S4 -> All": daytest(K + "S4", K + "All")}
out = {"auroc_diff_dayclustered_ci95": res, "cross_sectional_auroc_mean_ci95": csr, "day_level_paired_accuracy": tests, "n_test_days": int(len(dates))}
json.dump(out, open(ROOT / "figures/review_stats2.json", "w"), indent=1); print(json.dumps(out, indent=1))
