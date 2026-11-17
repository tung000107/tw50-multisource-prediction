"""Graph-derived features for LightGBM (S10): does an explicit Taiwan-50 relationship graph add information beyond the five sources,
beyond the nine sources of 19_source_study.py, and does propagating news over the graph help?
S10g (graph-price):  mean ret1 / ret5 of prior-graph neighbours (sector + business ties) and of business-tie neighbours only;
                     fold-specific Lasso-graph prediction (sum of parent coefficients x standardised market-adjusted returns, training-window fit);
                     number of Lasso parents; mean market-adjusted return of significant lead-lag leaders (fold-specific)
S10n (graph-news):   mean fold-trained text score and mean log news count of prior-graph neighbours; same for business-tie neighbours
Arms: Full, Full+S10g, Full+S10n, Full+S10, Full+S6, Full+S6+S10n, All9, All9+S10, S4+S10g, S10-only; labels y_up, y_gap, y_rel, y_intra.
Output: results/final/graph_features.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); df = df.merge(ex, on=["date", "stock_id"], how="left")
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); df = df.merge(nf, on=["date", "stock_id"], how="left")
graphs = json.load(open(ROOT / "data/processed/tw_graphs.json")); S = graphs["stocks"]
nb_all = {s: set() for s in S}; nb_tie = {s: set() for s in S}
for e in graphs["prior"]:
    nb_all[e["a"]].add(e["b"]); nb_all[e["b"]].add(e["a"])
    if not e["reason"].startswith("same sector") or ";" in e["reason"]: nb_tie[e["a"]].add(e["b"]); nb_tie[e["b"]].add(e["a"])
wide = {c: df.pivot(index="date", columns="stock_id", values=c).reindex(columns=S) for c in ["ret1", "ret5", "log_news"] + [f"text_score_{y}" for y in TEST_YEARS]}
taiex = df.groupby("date").taiex_ret1.first(); radj = wide["ret1"].sub(taiex.reindex(wide["ret1"].index), axis=0)
def nb_mean(W, nb):
    return pd.DataFrame({s: W[list(nb[s])].mean(1) if nb[s] else pd.Series(np.nan, index=W.index) for s in S})
feat = {}
for nm, nb in [("prior", nb_all), ("tie", nb_tie)]:
    feat[f"{nm}_nb_ret1"] = nb_mean(wide["ret1"], nb); feat[f"{nm}_nb_ret5"] = nb_mean(wide["ret5"], nb); feat[f"{nm}_nb_lognews"] = nb_mean(wide["log_news"], nb)
    for y in TEST_YEARS: feat[f"{nm}_nb_text_{y}"] = nb_mean(wide[f"text_score_{y}"], nb)
for y in TEST_YEARS:
    t0 = pd.Timestamp(f"{y}-01-01"); R = radj[radj.index < t0 - pd.Timedelta(days=1)]; mu, sd = R.mean(), R.std(); tm, ts = taiex[taiex.index < t0 - pd.Timedelta(days=1)].mean(), taiex[taiex.index < t0 - pd.Timedelta(days=1)].std()
    Z = (radj - mu) / sd; zt = (taiex.reindex(radj.index) - tm) / ts; pred = pd.DataFrame(0.0, index=radj.index, columns=S); npar = pd.Series(0, index=S, dtype=float)
    for child, lst in graphs["lasso"][str(y)].items():
        for par, coef in lst: pred[child] += coef * (zt if par == "TAIEX" else Z[par]); npar[child] += 1
    feat[f"lasso_pred_{y}"] = pred; feat[f"n_parents_{y}"] = pd.DataFrame(np.tile(npar.values, (len(radj), 1)), index=radj.index, columns=S)
    lead = {s: set() for s in S}
    for a, b, c in graphs["leadlag_sig"][str(y)]: lead[b].add(a)
    feat[f"leader_radj_{y}"] = nb_mean(radj, lead).fillna(0)
long = pd.concat({k: v.stack(future_stack=True) for k, v in feat.items()}, axis=1); long.index.names = ["date", "stock_id"]; long = long.reset_index()
df = df.merge(long, on=["date", "stock_id"], how="left")
S10g = ["prior_nb_ret1", "prior_nb_ret5", "tie_nb_ret1", "tie_nb_ret5", "lasso_pred", "n_parents", "leader_radj"]; S10n = ["prior_nb_lognews", "tie_nb_lognews", "prior_nb_text", "tie_nb_text"]
FOLDSPEC = ["text_score", "market_text_score", "lasso_pred", "n_parents", "leader_radj", "prior_nb_text", "tie_nb_text"]
NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"],
       "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"],
       "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
ALL9 = ALL + sum(NEW.values(), [])
def run(feats, label):
    prob = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, te in FOLDS:
        d = df.copy()
        for c in FOLDSPEC: d[c] = d[f"{c}_{yr}"]
        itr, iva, tr, te = [i[d.loc[i, label].notna()] for i in (itr, iva, tr, te)]
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(d.loc[itr, feats], d.loc[itr, label], eval_set=[(d.loc[iva, feats], d.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(d.loc[tr, feats], d.loc[tr, label]); prob.loc[te] = f.predict_proba(d.loc[te, feats])[:, 1]
    return prob
arms = {"Full": ALL, "Full+S10g": ALL + S10g, "Full+S10n": ALL + S10n, "Full+S10": ALL + S10g + S10n, "Full+S6": ALL + NEW["S6"], "Full+S6+S10n": ALL + NEW["S6"] + S10n,
        "All9": ALL9, "All9+S10": ALL9 + S10g + S10n, "S4+S10g": SRC["S4"] + S10g, "S10-only": S10g + S10n, "S7-only": NEW["S7"], "S10g-only": S10g}
PAIRS = [("Full+S10g", "Full"), ("Full+S10n", "Full"), ("Full+S10", "Full"), ("Full+S6+S10n", "Full+S6"), ("Full+S6+S10n", "Full"), ("All9+S10", "All9"), ("S4+S10g", "S4-only"), ("S10g-only", "S7-only")]
arms["S4-only"] = SRC["S4"]
te_mask = df.year.isin(TEST_YEARS); out = {"features": {"S10g": S10g, "S10n": S10n}, "n_rows_with_tie_neighbours": int(df.tie_nb_ret1.notna().sum())}
for label in ["y_up", "y_gap", "y_rel", "y_intra"]:
    res = dict(Parallel(n_jobs=3, prefer="threads")(delayed(lambda a: (a, run(arms[a], label)))(a) for a in arms))
    m = te_mask & df[label].notna(); y = df.loc[m, label].values; bs = DayBootstrap(df.loc[m, "date"].values, B=1000)
    r = {a: dict(auroc=auc_stat(y, res[a].loc[m].values), ci=bs.ci(auc_stat, y, res[a].loc[m].values)) for a in arms}
    for a, b in PAIRS: r[f"{a} minus {b}"] = bs.ci_diff(auc_stat, y, res[b].loc[m].values, res[a].loc[m].values)
    out[label] = r; print(label, {a: round(r[a]["auroc"], 4) for a in arms}, {k: [round(x, 4) for x in v] for k, v in r.items() if "minus" in k}, flush=True)
    dump(out, "graph_features.json")
print("done")
