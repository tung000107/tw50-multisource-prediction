"""Leakage / protocol audits with LightGBM on the Full arm.
  1. random stratified 5-fold (shuffled rows, same features) vs walk-forward  -> leakage ratio
  2. block label permutation null: shuffle y_up in 20-trading-day blocks inside the training window -> test AUROC should be ~0.5
  3. future-shift test: replace the overnight U.S. block by the session dated t+1 (deliberate leak) -> AUROC should jump
  4. seed sensitivity (already in 02) and 'us_stale excluded' robustness
Output: results/final/audits.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])   # reuse run_lgb / df / FOLDS
out = {}
te_mask = df.year.isin(TEST_YEARS); y_te = df.loc[te_mask, "y_up"].values; bs = DayBootstrap(df.loc[te_mask, "date"].values, B=500)
# walk-forward reference
p_wf, _ = run_lgb(ALL, 42); p_wf = p_wf.loc[te_mask].values
out["walk_forward_full"] = dict(auroc=auc_stat(y_te, p_wf), ci=bs.ci(auc_stat, y_te, p_wf))
# 1. random stratified 5-fold on the same rows (2019-2026), fixed 400 trees (no early stopping possible without leaking) and early-stopped variant on a random inner split
skf = StratifiedKFold(5, shuffle=True, random_state=42); p_rand = pd.Series(np.nan, index=df.index)
for tr, te in skf.split(df, df.y_up):
    tr, te = df.index[tr], df.index[te]
    m = lgb.LGBMClassifier(n_estimators=400, random_state=42, **LGB_PARAMS).fit(df.loc[tr, ALL], df.loc[tr, "y_up"]); p_rand.loc[te] = m.predict_proba(df.loc[te, ALL])[:, 1]
out["random_5fold_full"] = dict(auroc_all_years=auc_stat(df.y_up.values, p_rand.values), auroc_test_years_only=auc_stat(y_te, p_rand.loc[te_mask].values))
out["leakage_ratio"] = out["random_5fold_full"]["auroc_test_years_only"] / out["walk_forward_full"]["auroc"]
print("random 5-fold", out["random_5fold_full"], "wf", out["walk_forward_full"]["auroc"], flush=True)
# same comparison for a fixed-400-tree walk-forward (apples to apples with the random split's fixed trees)
p_wf400 = pd.Series(np.nan, index=df.index)
for yr, tr, itr, iva, te in FOLDS:
    p_wf400.loc[te] = lgb.LGBMClassifier(n_estimators=400, random_state=42, **LGB_PARAMS).fit(df.loc[tr, ALL], df.loc[tr, "y_up"]).predict_proba(df.loc[te, ALL])[:, 1]
out["walk_forward_full_fixed400"] = auc_stat(y_te, p_wf400.loc[te_mask].values)
# does the Progress Update's fusion gain come from the fixed tree count? Full - S4-only with 400 fixed trees vs with early stopping (same seed, same folds)
p_s4400 = pd.Series(np.nan, index=df.index)
for yr, tr, itr, iva, te in FOLDS:
    p_s4400.loc[te] = lgb.LGBMClassifier(n_estimators=400, random_state=42, **LGB_PARAMS).fit(df.loc[tr, SRC["S4"]], df.loc[tr, "y_up"]).predict_proba(df.loc[te, SRC["S4"]])[:, 1]
p_s4es, _ = run_lgb(SRC["S4"], 42); p_s4es = p_s4es.loc[te_mask].values; bs1k = DayBootstrap(df.loc[te_mask, "date"].values, B=1000)
out["fusion_gain_fixed400_vs_early_stopping"] = dict(
    fixed400=dict(full=auc_stat(y_te, p_wf400.loc[te_mask].values), s4=auc_stat(y_te, p_s4400.loc[te_mask].values), diff_ci=bs1k.ci_diff(auc_stat, y_te, p_s4400.loc[te_mask].values, p_wf400.loc[te_mask].values)),
    early_stopping=dict(full=auc_stat(y_te, p_wf), s4=auc_stat(y_te, p_s4es), diff_ci=bs1k.ci_diff(auc_stat, y_te, p_s4es, p_wf)))
print(out["fusion_gain_fixed400_vs_early_stopping"], flush=True)
# 2. block permutation null (3 permutations)
nulls = []
for s in range(3):
    rng = np.random.default_rng(100 + s); d2 = df.copy()
    for yr, tr, itr, iva, te in FOLDS:
        pass
    # permute labels within training rows in 20-day blocks: shuffle block order of the daily label vectors
    dates = np.sort(d2.date.unique()); blocks = [dates[i:i + 20] for i in range(0, len(dates), 20)]; order = rng.permutation(len(blocks))
    mapping = {}
    for b_from, b_to in zip(blocks, [blocks[i] for i in order]):
        for a, b in zip(b_from, b_to): mapping[a] = b
    lab = d2.set_index(["date", "stock_id"])["y_up"]
    newd = d2.date.map(mapping); key = pd.MultiIndex.from_arrays([newd, d2.stock_id]); d2["y_perm"] = lab.reindex(key).values
    d2["y_perm"] = d2["y_perm"].fillna(d2.y_up)
    d2.loc[te_mask, "y_perm"] = d2.loc[te_mask, "y_up"]   # test labels untouched; only training labels permuted
    p, _ = run_lgb(ALL, 42, label="y_perm", data=d2); nulls.append(auc_stat(y_te, p.loc[te_mask].values)); print("null", nulls[-1], flush=True)
out["block_permutation_null_auroc"] = nulls
# 3. future-shift: use the U.S. session dated t+1 (available only after the TW close of t+1) -> deliberate leak
us = df[["date"] + SRC["S4"]].drop_duplicates("date").sort_values("date"); nxt = us.copy(); nxt["date"] = nxt["date"].shift(1)   # row dated t gets features of the next TW trading day's US session
d3 = df.drop(columns=SRC["S4"]).merge(nxt.dropna(subset=["date"]), on="date", how="left")
d3.index = df.index
p, _ = run_lgb(ALL, 42, data=d3); out["future_shift_full"] = auc_stat(y_te, p.loc[te_mask].values); print("future shift", out["future_shift_full"], flush=True)
# 4. exclude us_stale rows (US holiday / stale) at test time
m = te_mask & (df.us_stale == 0); out["exclude_us_stale"] = dict(n=int(m.sum()), auroc=auc_stat(df.loc[m, "y_up"].values, pd.Series(p_wf, index=df.index[te_mask]).loc[m].values))
dump(out, "audits.json")
