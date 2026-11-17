"""Extended source study: which external dataset helps? Sources S1..S5 (original) + S6 news + S7 cross-stock + S8 derivatives/positioning
+ S9 stock-level flows. LightGBM, same protocol. (a) add-on to Full (S1-S5) and to S4-only for each new source, on y_up / y_gap / y_intra / y_rel
with trading-day block-bootstrap CIs; (b) exact Shapley over all 9 sources on y_up (511 subsets x 4 folds). Output: results/final/source_study.json"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from math import factorial
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); df = df.merge(ex, on=["date", "stock_id"], how="left")
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); df = df.merge(nf, on=["date", "stock_id"], how="left")
NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"],
       "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"],
       "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
NAME = {**SRC_NAME, "S6": "news headlines (cnyes)", "S7": "cross-stock / sector / leader", "S8": "TX futures night session, positioning", "S9": "day-trading, lending, revenue"}
ALLSRC = {**SRC, **NEW}
def run(feats, label):
    prob = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, te in FOLDS:
        d = df.copy(); d["text_score"] = d[f"text_score_{yr}"]; d["market_text_score"] = d[f"market_text_score_{yr}"]
        itr, iva, tr, te = [i[d.loc[i, label].notna()] for i in (itr, iva, tr, te)]
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(d.loc[itr, feats], d.loc[itr, label], eval_set=[(d.loc[iva, feats], d.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(d.loc[tr, feats], d.loc[tr, label]); prob.loc[te] = f.predict_proba(d.loc[te, feats])[:, 1]
    return prob
te_mask = df.year.isin(TEST_YEARS); out = {"sources": NAME, "n_features": {k: len(v) for k, v in ALLSRC.items()}}
# ---- (a) add-on tests ----
arms = {"Full": ALL, "S4-only": SRC["S4"], "All9": ALL + sum(NEW.values(), [])}
for k, v in NEW.items(): arms[f"{k}-only"] = v; arms[f"Full+{k}"] = ALL + v; arms[f"S4+{k}"] = SRC["S4"] + v
for label in ["y_up", "y_gap", "y_intra", "y_rel"]:
    res = dict(Parallel(n_jobs=8, prefer="threads")(delayed(lambda a: (a, run(arms[a], label)))(a) for a in arms))
    m = te_mask & df[label].notna(); y = df.loc[m, label].values; bs = DayBootstrap(df.loc[m, "date"].values, B=1000)
    r = {a: dict(auroc=auc_stat(y, res[a].loc[m].values), ci=bs.ci(auc_stat, y, res[a].loc[m].values)) for a in arms}
    for k in NEW:
        r[f"Full+{k} minus Full"] = bs.ci_diff(auc_stat, y, res["Full"].loc[m].values, res[f"Full+{k}"].loc[m].values)
        r[f"S4+{k} minus S4"] = bs.ci_diff(auc_stat, y, res["S4-only"].loc[m].values, res[f"S4+{k}"].loc[m].values)
    r["All9 minus Full"] = bs.ci_diff(auc_stat, y, res["Full"].loc[m].values, res["All9"].loc[m].values)
    out[label] = r; print(label, {a: round(r[a]["auroc"], 4) for a in arms}, {k: [round(x, 4) for x in v] for k, v in r.items() if "minus" in k}, flush=True)
dump(out, "source_study.json")
# ---- (b) exact Shapley over 9 sources on y_up ----
S = list(ALLSRC); subsets = {}
for r_ in range(1, len(S) + 1):
    for c in itertools.combinations(S, r_): subsets[frozenset(c)] = sum((ALLSRC[k] for k in c), [])
print("subsets", len(subsets), flush=True)
y = df.loc[te_mask, "y_up"].values
vals = dict(Parallel(n_jobs=8, prefer="threads")(delayed(lambda fs: (fs, auc_stat(y, run(subsets[fs], "y_up").loc[te_mask].values)))(fs) for fs in subsets))
def val(fs): return 0.5 if not fs else vals[fs]
n = len(S); phi = {}
for i in S:
    tot = 0.0
    for r_ in range(n):
        for c in itertools.combinations([s for s in S if s != i], r_):
            w = factorial(len(c)) * factorial(n - len(c) - 1) / factorial(n); tot += w * (val(frozenset(c) | {i}) - val(frozenset(c)))
    phi[i] = tot
out["shapley9_y_up"] = dict(phi=phi, full=val(frozenset(S)), single={s: val(frozenset([s])) for s in S}, leave_one_out={s: val(frozenset(S)) - val(frozenset(S) - {s}) for s in S}, subset_values={"+".join(sorted(k)): v for k, v in vals.items()})
print("Shapley9", {k: round(v, 4) for k, v in phi.items()}); dump(out, "source_study.json")
