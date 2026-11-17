"""RQ2 / RQ3: (a) exact Shapley value of each source over the 31 subset arms (LightGBM and LR; value = pooled test AUROC, v(empty)=0.5),
with trading-day bootstrap CIs; (b) pairwise source interactions I(i,j); (c) grouped SHAP by source inside the Full LightGBM
(TreeSHAP, per test year, mean |sum of SHAP over the source's features|); (d) per-sector and per-stock Full-minus-S4 gains with CIs;
(e) cross-sectional (within-day) AUROC and daily rank IC. Output: results/final/explain.json
"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb, shap
from scipy.stats import spearmanr, ttest_1samp
from math import factorial
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
oos = pd.read_parquet(RES / "oos_tabular.parquet").reset_index(drop=True); y = oos.y_up.values; bs = DayBootstrap(oos.date.values, B=500)
out = {}
S = list(SRC); n = len(S)
def subset_name(c): return "+".join(c)
def shapley(valfn):
    """valfn(frozenset) -> value; returns dict source -> shapley value"""
    phi = {}
    for i in S:
        tot = 0.0
        for r in range(n):
            for c in itertools.combinations([s for s in S if s != i], r):
                w = factorial(len(c)) * factorial(n - len(c) - 1) / factorial(n)
                tot += w * (valfn(frozenset(c) | {i}) - valfn(frozenset(c)))
        phi[i] = tot
    return phi
for kind, seed in [("LightGBM", 42), ("LR", 0)]:
    cols = {frozenset(c): f"{kind}|{subset_name(c)}|{seed}" for r in range(1, 6) for c in itertools.combinations(S, r)}
    def val(fs, idx=None):
        if not fs: return 0.5
        p = oos[cols[fs]].values; return auc_stat(y if idx is None else y[idx], p if idx is None else p[idx])
    phi = shapley(val); full = val(frozenset(S))
    # bootstrap CIs for the Shapley values (same day-block resamples); cache the 31 subset AUROCs per resample
    boots = {s: [] for s in S}
    for j in bs.samples:
        cache = {fs: val(fs, j) for fs in cols}
        ph = shapley(lambda fs: 0.5 if not fs else cache[fs])
        for s in S: boots[s].append(ph[s])
    inter = {}
    for i, k in itertools.combinations(S, 2):
        # average pairwise interaction over all contexts (Shapley interaction index, exact)
        tot = 0.0
        for r in range(n - 1):
            for c in itertools.combinations([s for s in S if s not in (i, k)], r):
                w = factorial(len(c)) * factorial(n - len(c) - 2) / factorial(n - 1); c = frozenset(c)
                tot += w * (val(c | {i, k}) - val(c | {i}) - val(c | {k}) + val(c))
        inter[f"{i}x{k}"] = tot
    out[f"shapley_{kind}"] = dict(full_auroc=full, phi={s: dict(value=phi[s], ci=[float(np.percentile(boots[s], 2.5)), float(np.percentile(boots[s], 97.5))]) for s in S},
                                  sum_phi=sum(phi.values()), interaction=inter,
                                  single={s: val(frozenset([s])) for s in S}, leave_one_out={s: full - val(frozenset(S) - {s}) for s in S})
    print(kind, {s: round(phi[s], 4) for s in S}, "full", round(full, 4), flush=True)
# ---- (c) grouped SHAP inside the Full LightGBM, per test year ----
gs = {}
for yr, tr, itr, iva, te in FOLDS:
    m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(df.loc[itr, ALL], df.loc[itr, "y_up"], eval_set=[(df.loc[iva, ALL], df.loc[iva, "y_up"])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    f = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(df.loc[tr, ALL], df.loc[tr, "y_up"])
    sv = shap.TreeExplainer(f).shap_values(df.loc[te, ALL]); sv = sv[1] if isinstance(sv, list) else sv
    svd = pd.DataFrame(sv, columns=ALL, index=te)
    grp = {s: float(svd[SRC[s]].sum(axis=1).abs().mean()) for s in S}; tot = sum(grp.values())
    gs[str(yr)] = dict(mean_abs=grp, share={s: grp[s] / tot for s in S}, top_features=svd.abs().mean().sort_values(ascending=False).head(10).round(4).to_dict(),
                       by_sector={sec: {s: float(svd.loc[g.index, SRC[s]].sum(axis=1).abs().mean()) for s in S} for sec, g in df.loc[te].groupby("sector_en")})
    print("SHAP", yr, {s: round(v, 3) for s, v in gs[str(yr)]["share"].items()}, flush=True)
out["grouped_shap"] = gs
# ---- (d) sector / stock gains (LightGBM seed-mean of 5 seeds for Full and S4-only) ----
pf = oos[[c for c in oos if c.startswith("LightGBM|Full|")]].mean(axis=1).values; ps = oos[[c for c in oos if c.startswith("LightGBM|S4-only|")]].mean(axis=1).values; pt = oos[[c for c in oos if c.startswith("LightGBM|TW-only|")]].mean(axis=1).values
sec = {}
for s, g in oos.groupby("sector_en"):
    ix = g.index.values; b = DayBootstrap(g.date.values, B=1000)
    sec[s] = dict(n=int(len(g)), auroc_S4=auc_stat(y[ix], ps[ix]), auroc_TW=auc_stat(y[ix], pt[ix]), auroc_Full=auc_stat(y[ix], pf[ix]), full_minus_S4=b.ci_diff(auc_stat, y[ix], ps[ix], pf[ix]), full_minus_TW=b.ci_diff(auc_stat, y[ix], pt[ix], pf[ix]))
out["by_sector"] = sec
stk = []
for s, g in oos.groupby("stock_id"):
    ix = g.index.values; b = DayBootstrap(g.date.values, B=400, seed=int(s))
    d = b.ci_diff(auc_stat, y[ix], ps[ix], pf[ix]); stk.append(dict(stock=s, sector=g.sector_en.iloc[0], auroc_S4=auc_stat(y[ix], ps[ix]), auroc_Full=auc_stat(y[ix], pf[ix]), diff=auc_stat(y[ix], pf[ix]) - auc_stat(y[ix], ps[ix]), ci=d[:2]))
out["by_stock"] = stk; out["by_stock_summary"] = dict(n_better=int(sum(x["diff"] > 0 for x in stk)), n_sig_up=int(sum(x["ci"][0] > 0 for x in stk)), n_sig_down=int(sum(x["ci"][1] < 0 for x in stk)))
# per-year
out["by_year"] = {str(yr): dict(S4=auc_stat(y[ix], ps[ix]), TW=auc_stat(y[ix], pt[ix]), Full=auc_stat(y[ix], pf[ix])) for yr, ix in oos.groupby("year").indices.items()}
# ---- (e) within-day AUROC and rank IC ----
def daily(fn, p):
    v = {}
    for d, ix in oos.groupby("date").indices.items():
        yy = y[ix]
        if 0 < yy.sum() < len(yy): v[d] = fn(yy, p[ix], oos.ret_next.values[ix])
    return pd.Series(v)
cs = {k: daily(lambda a, b, r: roc_auc_score(a, b), p) for k, p in [("Full", pf), ("S4-only", ps), ("TW-only", pt), ("stock base-rate", oos["Rule|stock base-rate|0"].values)]}
ic = {k: daily(lambda a, b, r: spearmanr(b, r).correlation, p) for k, p in [("Full", pf), ("TW-only", pt)]}
out["within_day"] = {k: dict(mean=float(v.mean()), t=float(ttest_1samp(v.dropna(), 0.5 if k in cs else 0).statistic), share_gt=float((v > (0.5 if k in cs else 0)).mean()), n_days=int(v.notna().sum())) for k, v in {**cs, **{f"IC {k}": v for k, v in ic.items()}}.items()}
dd = (cs["Full"] - cs["S4-only"]).dropna(); out["within_day"]["Full_minus_S4_days"] = dict(mean=float(dd.mean()), t=float(ttest_1samp(dd, 0).statistic))
dd = (cs["Full"] - cs["stock base-rate"]).dropna(); out["within_day"]["Full_minus_baserate_days"] = dict(mean=float(dd.mean()), t=float(ttest_1samp(dd, 0).statistic))
pd.DataFrame({"cs_full": cs["Full"], "cs_s4": cs["S4-only"], "cs_tw": cs["TW-only"], "ic_full": ic["Full"], "ic_tw": ic["TW-only"]}).to_parquet(RES / "daily_crosssection.parquet")
print(out["within_day"]); dump(out, "explain.json")
