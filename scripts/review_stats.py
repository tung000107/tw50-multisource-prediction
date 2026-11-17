"""Review check: date-clustered bootstrap CIs and within-day cross-sectional AUROC for the key source-ablation claims."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(ROOT / "data/processed/dataset.parquet")
SRC = {"S1": ["ret1","ret2","ret3","ret5","ret10","ret20","ma5_ratio","ma20_ratio","ma60_ratio","vol20","volume_ratio20","hl_range","gap_open","rsi14","spread"],
 "S2": ["foreign_net_ratio","foreign_net_ratio5","trust_net_ratio","trust_net_ratio5","dealer_net_ratio","dealer_net_ratio5","inst_net_ratio","margin_chg","margin_chg5","short_chg"],
 "S3": ["taiex_ret1","taiex_ret5","taiex_ma20_ratio","taiex_vol_ratio20","taiex_vol20","rel_ret1"],
 "S4": ["us_sp500_ret","us_nasdaq_ret","us_sox_ret","us_tsm_adr_ret","us_dxy_ret","us_nikkei_ret","us_vix_level","us_vix_chg","us_usdtwd_chg","us_10y_chg","us_stale"],
 "S5": ["weekday","month","is_month_end"]}
ALL = sum(SRC.values(), [])
CFG = {("LightGBM", "All"): ALL, ("LightGBM", "S4"): SRC["S4"], ("LightGBM", "All-S1"): [c for c in ALL if c not in SRC["S1"]],
       ("LightGBM", "All-S2"): [c for c in ALL if c not in SRC["S2"]], ("LightGBM", "All-S3"): [c for c in ALL if c not in SRC["S3"]],
       ("LightGBM", "S123"): SRC["S1"] + SRC["S2"] + SRC["S3"], ("LR", "S123"): SRC["S1"] + SRC["S2"] + SRC["S3"], ("LR", "S1234"): SRC["S1"] + SRC["S2"] + SRC["S3"] + SRC["S4"]}
TEST = [2023, 2024, 2025, 2026]; te = df[df.year.isin(TEST)].copy()
def model(k):
    if k == "LR": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=42, n_jobs=4)
for (k, n), f in CFG.items():
    p = pd.Series(np.nan, index=te.index)
    for yr in TEST:
        tr = df[df.year < yr]; m = te.year == yr
        p[m] = model(k).fit(tr[f], tr.y_up).predict_proba(te.loc[m, f])[:, 1]
    te[f"{k}|{n}"] = p; print("done", k, n, flush=True)
te.to_parquet(ROOT / "data/processed/review_oos_preds.parquet")
dates = te.date.unique(); rng = np.random.default_rng(7); y = te.y_up.values
grp = te.groupby("date").indices
def cs_auc(col, idx=None):  # mean within-day AUROC over days with both classes
    vals = []
    for d, ii in grp.items():
        yy = y[ii]
        if 0 < yy.sum() < len(yy): vals.append(roc_auc_score(yy, te[col].values[ii]))
    return float(np.mean(vals))
def boot(a, b, B=300):
    pa, pb = te[a].values, te[b].values; base = roc_auc_score(y, pb) - roc_auc_score(y, pa)
    iid, clu = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y)); iid.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
        ds = rng.choice(dates, len(dates)); j = np.concatenate([grp[d] for d in ds]); clu.append(roc_auc_score(y[j], pb[j]) - roc_auc_score(y[j], pa[j]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"diff": round(float(base), 4), "iid_ci": q(iid), "day_cluster_ci": q(clu)}
out = {"pooled_auroc": {c: round(float(roc_auc_score(y, te[c])), 4) for c in te.columns if "|" in c},
       "cross_sectional_auroc": {c: round(cs_auc(c), 4) for c in te.columns if "|" in c},
       "tests": {"All vs S4 (LGBM)": boot("LightGBM|S4", "LightGBM|All"), "drop S1": boot("LightGBM|All-S1", "LightGBM|All"),
                 "drop S2": boot("LightGBM|All-S2", "LightGBM|All"), "drop S3": boot("LightGBM|All-S3", "LightGBM|All"),
                 "S123->S1234 (LR)": boot("LR|S123", "LR|S1234")}}
# McNemar effective: day-level paired accuracy difference with day-cluster
a = ((te["LR|S123"] > .5) == te.y_up).groupby(te.date).mean(); b = ((te["LR|S1234"] > .5) == te.y_up).groupby(te.date).mean()
dd = (b - a); out["LR_S1234_vs_S123_daylevel_acc"] = {"mean_diff": round(float(dd.mean()), 4), "t_stat_days": round(float(dd.mean() / (dd.std() / np.sqrt(len(dd)))), 2), "n_days": int(len(dd))}
json.dump(out, open(ROOT / "figures/review_stats.json", "w"), indent=1); print(json.dumps(out, indent=1))
