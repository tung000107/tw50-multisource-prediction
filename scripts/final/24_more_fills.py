"""Further fills for final_present/04_覆核_缺口與待辦.md (A4-A8, D6-D8), all under the common protocol:
 (1) Diebold-Mariano on daily mean log-loss, Full vs S4-only and Full vs TW-only (LightGBM seed-mean probabilities; Newey-West lag 5)
 (2) tertiary 3-class ±0.5 % (down / flat / up): LightGBM multiclass, Full vs S4-only vs TW-only, macro-F1 / accuracy / majority
 (3) LightGBM hyper-parameter grid chosen on the inner validation of each fold (27 configs) vs the fixed default
 (4) ensemble: mean of LightGBM Full and LSTM-W5 Full seed-mean probabilities
 (5) regimes: AUROC of Full / S4-only / TW-only by VIX tercile, by TAIEX above/below its 20-day mean, by |overnight SOX| tercile
 (6) magnitude-gated intraday backtest: trade only stocks whose predicted P(|r| > k) (TW-only magnitude model) is in the day's top 10, direction by the
     Open→Close model; same costs as 10_backtest.py
 (7) U.S.-holiday subsample: rows whose overnight U.S. features are stale (us_stale = 1): Full, Full + news (S6), S4-only, TW-only
Output: results/final/more_fills.json"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import norm, skew, kurtosis
from sklearn.metrics import f1_score, accuracy_score
from joblib import Parallel, delayed
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
oos = pd.read_parquet(RES / "oos_tabular.parquet"); lab = pd.read_parquet(RES / "oos_labels.parquet"); seq = pd.read_parquet(RES / "oos_sequence.parquet")
pf = {a: oos[[c for c in oos if c.startswith(f"LightGBM|{a}|")]].mean(axis=1).values for a in ["Full", "S4-only", "TW-only"]}; y = oos.y_up.values; dates = oos.date.values; bs = DayBootstrap(dates, B=1000); out = {}
# (1) DM
def dm(pa, pb, lags=5):
    la = -(y * np.log(np.clip(pa, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - pa, 1e-6, 1 - 1e-6))); lb = -(y * np.log(np.clip(pb, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - pb, 1e-6, 1 - 1e-6)))
    d = pd.Series(la - lb).groupby(oos.date.values).mean().values; T = len(d); g = lambda k: float(np.sum((d[k:] - d.mean()) * (d[:T - k] - d.mean())) / T)
    var = g(0) + 2 * sum((1 - k / (lags + 1)) * g(k) for k in range(1, lags + 1)); stat = d.mean() / np.sqrt(var / T); return dict(mean_daily_logloss_diff=float(d.mean()), dm_stat=float(stat), p_two_sided=float(2 * norm.sf(abs(stat))), n_days=int(T), note="negative = first model has lower loss")
out["diebold_mariano_logloss"] = {"Full vs S4-only": dm(pf["Full"], pf["S4-only"]), "Full vs TW-only": dm(pf["Full"], pf["TW-only"])}; print(out["diebold_mariano_logloss"], flush=True)
# (2) tertiary
df["y3"] = np.select([df.ret_next < -0.005, df.ret_next > 0.005], [0, 2], 1)
def run3(feats):
    prob = pd.DataFrame(np.nan, index=df.index, columns=[0, 1, 2])
    for yr, tr, itr, iva, te in FOLDS:
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, random_state=42, **LGB_PARAMS).fit(df.loc[itr, feats], df.loc[itr, "y3"], eval_set=[(df.loc[iva, feats], df.loc[iva, "y3"])], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(objective="multiclass", n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(df.loc[tr, feats], df.loc[tr, "y3"]); prob.loc[te] = f.predict_proba(df.loc[te, feats])
    return prob
te_mask = df.year.isin(TEST_YEARS); y3 = df.loc[te_mask, "y3"].values; r3 = dict(Parallel(n_jobs=3, prefer="threads")(delayed(lambda a: (a, run3(KEY[a])))(a) for a in KEY))
out["tertiary_3class"] = {"class_shares_test": {str(k): float((y3 == k).mean()) for k in range(3)}, "majority_acc": float(max((y3 == k).mean() for k in range(3))), "majority_macro_f1": float(f1_score(y3, np.full_like(y3, int(np.bincount(y3).argmax())), average="macro"))}
for a in KEY:
    p = r3[a].loc[te_mask].values.argmax(1); out["tertiary_3class"][a] = dict(macro_f1=float(f1_score(y3, p, average="macro")), acc=float(accuracy_score(y3, p)), macro_f1_ci=bs.ci(lambda yy, pp: f1_score(yy, pp, average="macro"), y3, p))
out["tertiary_3class"]["Full minus S4 macro_f1"] = bs.ci_diff(lambda yy, pp: f1_score(yy, pp, average="macro"), y3, r3["S4-only"].loc[te_mask].values.argmax(1), r3["Full"].loc[te_mask].values.argmax(1)); print(out["tertiary_3class"], flush=True)
# (3) tuning grid
GRID = [dict(num_leaves=a, learning_rate=b, min_child_samples=c) for a in (15, 31, 63) for b in (0.01, 0.03, 0.1) for c in (50, 200, 500)]
def inner_auc(cfg, yr, itr, iva):
    P = dict(LGB_PARAMS, **cfg); m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **P).fit(df.loc[itr, ALL], df.loc[itr, "y_up"], eval_set=[(df.loc[iva, ALL], df.loc[iva, "y_up"])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)]); return float(m.best_score_["valid_0"]["auc"]), int(m.best_iteration_ or 1)
prob_t = pd.Series(np.nan, index=df.index); chosen = {}
for yr, tr, itr, iva, te in FOLDS:
    res = Parallel(n_jobs=6, prefer="threads")(delayed(inner_auc)(c, yr, itr, iva) for c in GRID); k = int(np.argmax([r[0] for r in res])); cfg, (va, best) = GRID[k], res[k]
    f = lgb.LGBMClassifier(n_estimators=best, random_state=42, **dict(LGB_PARAMS, **cfg)).fit(df.loc[tr, ALL], df.loc[tr, "y_up"]); prob_t.loc[te] = f.predict_proba(df.loc[te, ALL])[:, 1]; chosen[str(yr)] = dict(cfg, inner_val_auc=va, n_estimators=best, default_inner_val_auc=res[GRID.index(dict(num_leaves=31, learning_rate=0.03, min_child_samples=200))][0]); print("tuned", yr, chosen[str(yr)], flush=True)
pt = prob_t.loc[te_mask].values; out["tuning"] = dict(chosen_by_fold=chosen, tuned_auroc=auc_stat(y, pt), default_auroc=auc_stat(y, pf["Full"]), default_auroc_seed42=auc_stat(y, oos["LightGBM|S1+S2+S3+S4+S5|42"].values), tuned_minus_default_seed42=bs.ci_diff(auc_stat, y, oos["LightGBM|S1+S2+S3+S4+S5|42"].values, pt), note="27-config grid chosen on the inner validation only; compared with the fixed default (seed 42)")
# (4) ensemble
pl = seq[[c for c in seq if c.startswith("LSTM-W5|Full|")]].mean(axis=1).values; pe = (pf["Full"] + pl) / 2; pe_rank = (pd.Series(pf["Full"]).rank(pct=True).values + pd.Series(pl).rank(pct=True).values) / 2
out["ensemble"] = dict(lightgbm=auc_stat(y, pf["Full"]), lstm=auc_stat(y, pl), mean_prob=auc_stat(y, pe), mean_rank=auc_stat(y, pe_rank), mean_prob_minus_lightgbm=bs.ci_diff(auc_stat, y, pf["Full"], pe), corr_lightgbm_lstm=float(np.corrcoef(pf["Full"], pl)[0, 1]))
# (5) regimes
m = oos[["date", "stock_id"]].merge(df[["date", "stock_id", "us_vix_level", "taiex_ma20_ratio", "us_sox_ret"]], on=["date", "stock_id"], how="left")
def by(groups, name):
    groups = pd.Series(np.asarray(groups).astype(str)); r = {}
    for g in sorted(groups.unique()):
        ix = (groups == g).values; r[str(g)] = dict(n=int(ix.sum()), n_days=int(pd.Series(dates[ix]).nunique()), **{a: auc_stat(y[ix], pf[a][ix]) for a in pf}, up_rate=float(y[ix].mean()))
    out.setdefault("regimes", {})[name] = r
# taiex_ma20_ratio is Close / MA20 - 1, so "above the average" is > 0 (a "> 1" here left one group empty; fixed 2026-11-14)
by(pd.qcut(m.us_vix_level, 3, labels=["low VIX", "mid VIX", "high VIX"]), "VIX tercile"); by(np.where(m.taiex_ma20_ratio > 0, "TAIEX above MA20", "TAIEX below MA20"), "TAIEX trend"); by(pd.qcut(m.us_sox_ret.abs(), 3, labels=["small |SOX|", "mid |SOX|", "large |SOX|"]), "|overnight SOX| tercile")
# (6) magnitude-gated intraday backtest
COST_RT = 0.001425 * 2 + 0.0015
def stats_(r):
    r = pd.Series(r).dropna(); n = len(r); mu, sd = r.mean(), r.std(ddof=1); sr = mu / sd * np.sqrt(252) if sd > 0 else np.nan; eq = (1 + r).cumprod()
    return dict(n_days=int(n), mean_daily=float(mu), ann_return=float((1 + mu) ** 252 - 1), sharpe=float(sr), max_drawdown=float((eq / eq.cummax() - 1).min()), hit_rate=float((r > 0).mean()), break_even_cost=float(mu + COST_RT))
oos["p_mag"] = lab["ymag_raw|TW-only"].values; oos["p_oc"] = lab["y_intra|TW-only"].values; oos["p_oc_full"] = lab["y_intra|Full"].values; tie = np.random.default_rng(1).random(len(oos)) * 1e-9; oos["p_oc"] += tie; oos["p_mag"] += tie
rows = []
for d, g in oos.groupby("date"):
    g = g.dropna(subset=["ret_intra"]); gated = g.sort_values("p_mag", ascending=False).head(10); top = gated.sort_values("p_oc", ascending=False).head(5); bot = gated.sort_values("p_oc").head(5); un = g.sort_values("p_oc", ascending=False).head(5); hi = g.sort_values("p_mag", ascending=False).head(5)
    rows.append(dict(date=d, gated_long_net=top.ret_intra.mean() - COST_RT, gated_long_gross=top.ret_intra.mean(), gated_ls_gross=top.ret_intra.mean() - bot.ret_intra.mean(), ungated_long_gross=un.ret_intra.mean(), abs_move_gated=gated.ret_intra.abs().mean(), abs_move_all=g.ret_intra.abs().mean(), mag_top5_abs=hi.ret_intra.abs().mean()))
r = pd.DataFrame(rows).set_index("date"); out["magnitude_gated_backtest"] = {k: stats_(r[k]) for k in ["gated_long_net", "gated_long_gross", "gated_ls_gross", "ungated_long_gross"]}
out["magnitude_gated_backtest"]["mean_abs_intraday_move"] = dict(gated_top10=float(r.abs_move_gated.mean()), all_50=float(r.abs_move_all.mean()), magnitude_top5=float(r.mag_top5_abs.mean()), ci_gated_minus_all=DayBootstrap(r.index.values, B=1000).ci(lambda a, b: float((a - b).mean()), r.abs_move_gated.values, r.abs_move_all.values))
# (7) U.S.-holiday subsample with news
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); d6 = df.merge(nf, on=["date", "stock_id"], how="left"); S6 = ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"]
def run6(feats):
    prob = pd.Series(np.nan, index=d6.index)
    for yr, tr, itr, iva, te in FOLDS:
        d = d6.copy(); d["text_score"] = d[f"text_score_{yr}"]; d["market_text_score"] = d[f"market_text_score_{yr}"]
        mm = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(d.loc[itr, feats], d.loc[itr, "y_up"], eval_set=[(d.loc[iva, feats], d.loc[iva, "y_up"])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(n_estimators=int(mm.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(d.loc[tr, feats], d.loc[tr, "y_up"]); prob.loc[te] = f.predict_proba(d.loc[te, feats])[:, 1]
    return prob
p6 = dict(Parallel(n_jobs=3, prefer="threads")(delayed(lambda a, f: (a, run6(f)))(a, f) for a, f in [("Full", ALL), ("Full+S6", ALL + S6), ("TW+S6", TW + S6)]))
stale = d6.loc[te_mask, "us_stale"].values.astype(bool); yy = d6.loc[te_mask, "y_up"].values; dd = d6.loc[te_mask, "date"].values
r7 = {}
for nm, ix in [("stale (U.S. closed the night before)", stale), ("normal", ~stale)]:
    b = DayBootstrap(dd[ix], B=1000); r7[nm] = dict(n=int(ix.sum()), n_days=int(pd.Series(dd[ix]).nunique()), **{a: dict(auroc=auc_stat(yy[ix], p6[a].loc[te_mask].values[ix]), ci=b.ci(auc_stat, yy[ix], p6[a].loc[te_mask].values[ix])) for a in p6}, **{a: auc_stat(yy[ix], pf[a][ix]) for a in ["S4-only", "TW-only"]},
                   **{"Full+S6 minus Full": b.ci_diff(auc_stat, yy[ix], p6["Full"].loc[te_mask].values[ix], p6["Full+S6"].loc[te_mask].values[ix])})
out["us_holiday_subsample"] = r7
dump(out, "more_fills.json"); print({k: out[k] for k in ("tuning", "ensemble", "magnitude_gated_backtest", "us_holiday_subsample")}); print("done")
