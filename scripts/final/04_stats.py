"""Main result tables with inference. Reads oos_tabular / oos_sequence (and oos_labels if present).
For every arm: pooled + per-year metrics (AUROC, PR-AUC, ACC, BA, Macro-F1, MCC, Brier) with 5-day-block trading-day bootstrap 95 % CIs;
seed mean +- std where several seeds exist; paired differences (bootstrap CI, DeLong p, day-level paired t) for the pre-registered comparisons;
Holm correction across the comparison family. Output: results/final/main_results.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp
from common import *
oos = pd.read_parquet(RES / "oos_tabular.parquet"); seq = pd.read_parquet(RES / "oos_sequence.parquet")
for c in seq.columns:
    if "|" in c: oos[c] = seq[c].values
info = {**json.load(open(RES / "tabular_fit_info.json")), **json.load(open(RES / "sequence_info.json"))}
y = oos.y_up.values; bs = DayBootstrap(oos.date.values, B=1000); out = {"arms": {}, "comparisons": {}}
cols = [c for c in oos.columns if "|" in c]
def thr_of(c):
    # decision threshold from inner validation per year (rules: 0.5)
    t = np.full(len(oos), 0.5)
    for yr in TEST_YEARS:
        if c in info and str(yr) in info[c] and "thr" in info[c][str(yr)]: t[oos.year.values == yr] = info[c][str(yr)]["thr"]
    return t
def seed_group(c): k, a, s = c.split("|"); return f"{k}|{a}"
groups = {}
for c in cols: groups.setdefault(seed_group(c), []).append(c)
for gname, cs in groups.items():
    rows = []
    for c in cs:
        p = oos[c].values; m = ~np.isnan(p); t = thr_of(c)
        r = dict(pooled=metrics(y[m], p[m], 0.5) if gname.startswith("Rule") else {**metrics(y[m], p[m], 0.5), **{f"{k}_tuned": v for k, v in metrics(y[m], np.where(p[m] >= t[m], 1.0, 0.0), 0.5).items() if k in ("acc", "bal_acc", "macro_f1", "mcc")}})
        r["by_year"] = {str(yr): metrics(y[m & (oos.year.values == yr)], p[m & (oos.year.values == yr)], 0.5) for yr in TEST_YEARS}
        rows.append(r)
    pooled_auc = [r["pooled"]["auroc"] for r in rows]; c0 = cs[0]; p0 = oos[c0].values; m0 = ~np.isnan(p0)
    pm = oos[cs].mean(axis=1).values   # seed-averaged probability
    out["arms"][gname] = dict(n_seeds=len(cs), columns=cs, auroc_seed_mean=float(np.mean(pooled_auc)), auroc_seed_std=float(np.std(pooled_auc)),
                              first_seed=rows[0], seed_mean_prob=dict(pooled=metrics(y[m0], pm[m0]), ci_auroc=bs.ci(auc_stat, y, pm) if m0.all() else None,
                                                                   by_year={str(yr): metrics(y[(oos.year.values == yr)], pm[(oos.year.values == yr)]) for yr in TEST_YEARS}))
    print(f"{gname:34s} AUROC {np.mean(pooled_auc):.4f} ± {np.std(pooled_auc):.4f} (n={len(cs)})", flush=True)
# ---- pre-registered comparisons on seed-mean probabilities ----
def P(g): return oos[groups[g]].mean(axis=1).values
def compare(a, b):
    pa, pb = P(a), P(b); m = ~(np.isnan(pa) | np.isnan(pb))
    d = auc_stat(y[m], pb[m]) - auc_stat(y[m], pa[m]); ci = bs.ci_diff(auc_stat, y, np.where(m, pa, 0.5), np.where(m, pb, 0.5))
    _, _, z, pdl = delong(y[m], pb[m], pa[m])
    da = ((pb >= 0.5) == y).astype(float) - ((pa >= 0.5) == y).astype(float); dd = pd.Series(da[m]).groupby(oos.date.values[m]).mean()
    return dict(diff=float(d), ci=ci[:2], boot_p_le0=ci[2], delong_z=z, delong_p=pdl, day_acc_diff=float(dd.mean()), day_t=float(ttest_1samp(dd, 0).statistic), n_days=int(len(dd)),
                by_year={str(yr): float(auc_stat(y[m & (oos.year.values == yr)], pb[m & (oos.year.values == yr)]) - auc_stat(y[m & (oos.year.values == yr)], pa[m & (oos.year.values == yr)])) for yr in TEST_YEARS})
fam = {"RQ1 Full vs S4-only (LightGBM)": ("LightGBM|S4-only", "LightGBM|Full"), "RQ1 Full vs TW-only (LightGBM)": ("LightGBM|TW-only", "LightGBM|Full"),
       "RQ1 Full vs S4-only (LR)": ("LR|S4", "LR|S1+S2+S3+S4+S5"), "RQ1 Full vs TW-only (LR)": ("LR|S1+S2+S3", "LR|S1+S2+S3+S4+S5"),
       "RQ1 Full vs S4-only (XGBoost)": ("XGBoost|S4-only", "XGBoost|Full"), "RQ1 Full vs S4-only (LSTM)": ("LSTM-W5|S4-only", "LSTM-W5|Full"), "RQ1 Full vs S4-only (Transformer)": ("Transformer-W5|S4-only", "Transformer-W5|Full"),
       "Full vs SOX-return rule": ("Rule|SOX return score", "LightGBM|Full"), "S4-only LightGBM vs SOX-return rule": ("Rule|SOX return score", "LightGBM|S4-only"),
       "RQ3 LightGBM Full vs LR Full": ("LR|S1+S2+S3+S4+S5", "LightGBM|Full"), "RQ3 LSTM Full vs LightGBM Full": ("LightGBM|Full", "LSTM-W5|Full"), "RQ3 Transformer Full vs LightGBM Full": ("LightGBM|Full", "Transformer-W5|Full"), "RQ3 XGBoost Full vs LightGBM Full": ("LightGBM|Full", "XGBoost|Full"),
       "RQ2 drop S1": ("LightGBM|S2+S3+S4+S5", "LightGBM|Full"), "RQ2 drop S2": ("LightGBM|S1+S3+S4+S5", "LightGBM|Full"), "RQ2 drop S3": ("LightGBM|S1+S2+S4+S5", "LightGBM|Full"), "RQ2 drop S4": ("LightGBM|S1+S2+S3+S5", "LightGBM|Full"), "RQ2 drop S5": ("LightGBM|S1+S2+S3+S4", "LightGBM|Full"),
       "RQ2 S4+S1 vs S4": ("LightGBM|S4", "LightGBM|S1+S4"), "RQ2 S4+S2 vs S4": ("LightGBM|S4", "LightGBM|S2+S4"), "RQ2 S4+S3 vs S4": ("LightGBM|S4", "LightGBM|S3+S4"), "RQ2 S4+S1+S2+S3 vs S4": ("LightGBM|S4", "LightGBM|S1+S2+S3+S4")}
for k, (a, b) in fam.items():
    if a in groups and b in groups: out["comparisons"][k] = compare(a, b); print(k, round(out["comparisons"][k]["diff"], 4), out["comparisons"][k]["ci"], "DeLong p", round(out["comparisons"][k]["delong_p"], 4), flush=True)
# Holm correction on DeLong p-values within the RQ1 family and the RQ2 family
for famname, keys in {"RQ1": [k for k in out["comparisons"] if k.startswith("RQ1")], "RQ2": [k for k in out["comparisons"] if k.startswith("RQ2")], "RQ3": [k for k in out["comparisons"] if k.startswith("RQ3")]}.items():
    ps = sorted([(out["comparisons"][k]["delong_p"], k) for k in keys]); mcount = len(ps); adj = {}
    running = 0
    for i, (p, k) in enumerate(ps): running = max(running, p * (mcount - i)); adj[k] = min(1.0, running)
    for k in keys: out["comparisons"][k]["holm_p"] = adj[k]
# LSTM window test
out["lstm_window"] = {c: dict(auroc=auc_stat(y, oos[c].values), seconds=info[c].get("seconds"), epochs={yr: info[c][yr]["epochs"] for yr in info[c] if yr.isdigit()}) for c in cols if c.startswith("LSTM-W") and c.endswith("|Full|0")}
out["fit_info_summary"] = {g: {yr: {k: v for k, v in info[groups[g][0]][yr].items()} for yr in map(str, TEST_YEARS) if yr in info.get(groups[g][0], {})} for g in groups if groups[g][0] in info}
dump(out, "main_results.json")
