"""Items promised in the Progress Update (p.11 challenges 4-5, p.11 challenge 1) that the first final pipeline skipped — see final_present/04_覆核_缺口與待辦.md A1-A3.
 A3 reduced S4 = {SOX, VIX level, VIX change, TSM ADR}: alone, and replacing the 11-feature S4 inside Full (collinearity check)
 A2 heterogeneity: pooled LightGBM (Full) vs Full + sector indicators vs one LightGBM per sector vs one logistic regression per stock
 A1 McNemar (exact binomial on discordant pairs) Full vs S4-only at each fold's inner-validation threshold. Rows are cross-sectionally dependent
    (50 stocks share each day), so this p-value is optimistic; the trading-day block bootstrap remains the primary test.
Same protocol as 02_tabular.py (seed 42). Output: results/final/gap_fills.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import binomtest
from joblib import Parallel, delayed
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
S4R = ["us_sox_ret", "us_vix_level", "us_vix_chg", "us_tsm_adr_ret"]; FULL_R = [c for c in ALL if c not in SRC["S4"]] + S4R
SECD = pd.get_dummies(df.sector_en, prefix="sec").astype(float); df = pd.concat([df, SECD], axis=1); SEC = list(SECD.columns)
arms = {"Full": ALL, "S4-only": SRC["S4"], "S4-reduced-only": S4R, "Full with reduced S4": FULL_R, "Full + sector indicators": ALL + SEC}
res = dict(Parallel(n_jobs=5, prefer="threads")(delayed(lambda a: (a, run_lgb(arms[a], 42)))(a) for a in arms)); prob = {a: r[0] for a, r in res.items()}; info = {a: r[1] for a, r in res.items()}
# per-sector LightGBM
def per_sector(sec):
    sub = df[df.sector_en == sec].copy(); p, _ = run_lgb(ALL, 42, data=sub); return p
ps = Parallel(n_jobs=6, prefer="threads")(delayed(per_sector)(s) for s in sorted(df.sector_en.unique())); prob["Per-sector LightGBM"] = pd.concat(ps).reindex(df.index)
# per-stock logistic regression
def per_stock(sid):
    sub = df[df.stock_id == sid]; p = pd.Series(np.nan, index=sub.index)
    for yr, tr, itr, iva, te in folds(sub):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(sub.loc[tr, ALL], sub.loc[tr, "y_up"]); p.loc[te] = m.predict_proba(sub.loc[te, ALL])[:, 1]
    return p
pst = Parallel(n_jobs=8, prefer="threads")(delayed(per_stock)(s) for s in sorted(df.stock_id.unique())); prob["Per-stock LR"] = pd.concat(pst).reindex(df.index)
te = df.year.isin(TEST_YEARS); y = df.loc[te, "y_up"].values; bs = DayBootstrap(df.loc[te, "date"].values, B=1000)
out = {"arms": {a: dict(auroc=auc_stat(y, p.loc[te].values), ci=bs.ci(auc_stat, y, p.loc[te].values), by_year={str(yr): auc_stat(df.loc[te & (df.year == yr), "y_up"].values, p.loc[te & (df.year == yr)].values) for yr in TEST_YEARS}) for a, p in prob.items()}}
out["diffs"] = {f"{a} minus Full": bs.ci_diff(auc_stat, y, prob["Full"].loc[te].values, prob[a].loc[te].values) for a in prob if a != "Full"}
out["diffs"]["S4-reduced-only minus S4-only"] = bs.ci_diff(auc_stat, y, prob["S4-only"].loc[te].values, prob["S4-reduced-only"].loc[te].values)
# McNemar Full vs S4-only at the inner-validation threshold
def preds(a):
    p = prob[a].copy()
    for yr in TEST_YEARS: p[df.year == yr] = (p[df.year == yr] >= info[a][str(yr)]["thr"]).astype(float)
    return p.loc[te].values
pf, p4 = preds("Full"), preds("S4-only"); cf, c4 = pf == y, p4 == y; b = int((cf & ~c4).sum()); c = int((~cf & c4).sum())
out["mcnemar_Full_vs_S4"] = dict(full_right_s4_wrong=b, s4_right_full_wrong=c, n=int(len(y)), acc_full=float(cf.mean()), acc_s4=float(c4.mean()), p_exact=float(binomtest(b, b + c, 0.5).pvalue), note="rows are cross-sectionally dependent (50 stocks per day): optimistic; block bootstrap is the primary test")
dump(out, "gap_fills.json"); print({a: round(v["auroc"], 4) for a, v in out["arms"].items()}); print({k: [round(x, 4) for x in v] for k, v in out["diffs"].items()}); print(out["mcnemar_Full_vs_S4"]); print("done")
