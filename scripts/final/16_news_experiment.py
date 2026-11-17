"""Does a leakage-safe news source (S6) add to the five sources? Same protocol as 02/06: LightGBM, walk-forward, trading-day block
bootstrap. Arms: S6-only, Full+S6, S4+S6, TW+S6 vs Full / S4 / TW on y_up, y_gap, y_intra, y_rel. Output: results/final/news.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); df = df.merge(nf, on=["date", "stock_id"], how="left")
# fold-specific text scores -> one column used per test year (training rows use the out-of-fold score of that fold)
for yr in TEST_YEARS: pass
S6_BASE = ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news"]
def with_fold_scores(d, yr):
    d = d.copy(); d["text_score"] = d[f"text_score_{yr}"]; d["market_text_score"] = d[f"market_text_score_{yr}"]; return d
S6 = S6_BASE + ["text_score", "market_text_score"]
ARMS6 = {"S6-only": S6, "Full+S6": ALL + S6, "S4+S6": SRC["S4"] + S6, "TW+S6": TW + S6, "Full": ALL, "S4-only": SRC["S4"], "TW-only": TW}
def run(feats, label):
    prob = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, te in FOLDS:
        d = with_fold_scores(df, yr); itr, iva, tr, te = [i[d.loc[i, label].notna()] for i in (itr, iva, tr, te)]
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(d.loc[itr, feats], d.loc[itr, label], eval_set=[(d.loc[iva, feats], d.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(d.loc[tr, feats], d.loc[tr, label]); prob.loc[te] = f.predict_proba(d.loc[te, feats])[:, 1]
    return prob
out = {"coverage": dict(share_stock_days_with_news=float((df.n_news > 0).mean()), mean_news_per_stock_day=float(df.n_news.mean()), by_year={str(y): float((g.n_news > 0).mean()) for y, g in df.groupby("year")})}
te_mask = df.year.isin(TEST_YEARS)
for label in ["y_up", "y_gap", "y_intra", "y_rel"]:
    res = Parallel(n_jobs=7, prefer="threads")(delayed(lambda a: (a, run(ARMS6[a], label)))(a) for a in ARMS6)
    P = dict(res); m = te_mask & df[label].notna(); y = df.loc[m, label].values; bs = DayBootstrap(df.loc[m, "date"].values, B=1000)
    r = {a: dict(auroc=auc_stat(y, P[a].loc[m].values), ci=bs.ci(auc_stat, y, P[a].loc[m].values)) for a in ARMS6}
    for a, b in [("Full", "Full+S6"), ("S4-only", "S4+S6"), ("TW-only", "TW+S6")]: r[f"{b} minus {a}"] = bs.ci_diff(auc_stat, y, P[a].loc[m].values, P[b].loc[m].values)
    r["by_year"] = {str(yr): {a: auc_stat(y[(df.loc[m, "year"] == yr).values], P[a].loc[m].values[(df.loc[m, "year"] == yr).values]) for a in ARMS6} for yr in TEST_YEARS}
    out[label] = r; print(label, {a: round(r[a]["auroc"], 4) for a in ARMS6}, {k: v for k, v in r.items() if "minus" in k}, flush=True)
dump(out, "news.json")
