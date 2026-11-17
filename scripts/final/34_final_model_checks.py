"""Checks on the final model (mean of LightGBM and LSTM on nine sources, AUROC 0.610), from the out-of-sample predictions of 33_sota_headtohead.py.
 A  where does the gain come from: sources x ensemble 2x2, paired trading-day block-bootstrap differences, per-year sign
 B  which day vs which stock: AUROC of the day-mean score, within-day AUROC, daily rank IC, top-5 minus bottom-5 return, day-level paired t
    + intraday top-k backtest of the final score (gross / net, drawdown, turnover, cost-sensitivity curve, slippage scenario)
 C  what does the TX night session (S8) add to the overnight U.S. source (S4)? targeted ablation on y_up / y_gap / y_intra (LightGBM, seed 42)
 D  is the final model's confidence usable: reliability bins and accuracy by confidence quantile (descriptive, no threshold is selected on the test set)
 E  robustness of S4 to the two features whose daily-bar close time was not verified field by field (USD/TWD, DXY)
Output: results/final/final_model_checks.json, figures/final/f21_final_model_checks.png"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr, ttest_1samp, norm, skew, kurtosis
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from common import *
KEY = ["date", "stock_id"]; out = {}
P = pd.read_parquet(RES / "oos_sota_headtohead.parquet"); P["stock_id"] = P.stock_id.astype(str); P["date"] = pd.to_datetime(P.date)
tab = pd.read_parquet(RES / "oos_tabular.parquet"); tab["stock_id"] = tab.stock_id.astype(str); tab["date"] = pd.to_datetime(tab.date)
P = P.merge(tab[KEY + ["date_next", "ret_next", "ret_intra", "Rule|stock base-rate|0"]], on=KEY, how="left").rename(columns={"Rule|stock base-rate|0": "stock base-rate"})
LF, LA, EF, EA = "LightGBM Full (S1-S5)", "LightGBM All9 (S1-S9)", "Ensemble LightGBM+LSTM (S1-S5)", "Ensemble LightGBM+LSTM All9 (ours, final)"
y = P.y_up.values; bs = DayBootstrap(P.date.values, B=1000); yrs = P.year.values
def pair(a, b):   # b minus a
    ci = bs.ci_diff(auc_stat, y, P[a].values, P[b].values); by = {int(t): float(auc_stat(y[yrs == t], P.loc[yrs == t, b].values) - auc_stat(y[yrs == t], P.loc[yrs == t, a].values)) for t in TEST_YEARS}
    return dict(diff=float(auc_stat(y, P[b].values) - auc_stat(y, P[a].values)), ci=ci[:2], p_boot=ci[2], by_year=by, same_sign_years=int(sum(np.sign(v) == np.sign(sum(by.values())) for v in by.values())))
# ---------------------------------------------------------------- A
out["A_decomposition"] = dict(auroc={k: float(auc_stat(y, P[k].values)) for k in (LF, LA, EF, EA)},
    more_sources_single_model=pair(LF, LA), more_sources_ensemble=pair(EF, EA), ensemble_on_five_sources=pair(LF, EF), ensemble_on_nine_sources=pair(LA, EA), total=pair(LF, EA))
print("A", {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"]) for k, v in out["A_decomposition"].items() if k != "auroc"}, flush=True)
# ---------------------------------------------------------------- B
MODELS = [LF, LA, EF, EA, "stock base-rate", "SOX-return rule"]; g = P.groupby("date")
def wd_auc(d, c):
    return roc_auc_score(d.y_up, d[c]) if d.y_up.nunique() == 2 and d[c].nunique() > 1 else (0.5 if d.y_up.nunique() == 2 else np.nan)
W = pd.DataFrame({c: g.apply(lambda d: wd_auc(d, c)) for c in MODELS}); IC = pd.DataFrame({c: g.apply(lambda d: spearmanr(d[c], d.ret_next).correlation if d[c].nunique() > 1 else 0.0) for c in MODELS})
SP = pd.DataFrame({c: g.apply(lambda d: d.nlargest(5, c).ret_next.mean() - d.nsmallest(5, c).ret_next.mean()) for c in MODELS if c != "SOX-return rule"})
daymean = {c: P.groupby("date")[c].transform("mean").values for c in MODELS}
B = {}
for c in MODELS:
    w = W[c].dropna(); B[c] = dict(pooled_auroc=float(auc_stat(y, P[c].values)), which_day_auroc=float(auc_stat(y, daymean[c])), within_day_auroc=float(w.mean()), within_day_t=float(ttest_1samp(w, 0.5).statistic) if w.std() > 0 else None,
                                   rank_ic=float(IC[c].mean()), rank_ic_t=float(ttest_1samp(IC[c].dropna(), 0).statistic) if IC[c].std() > 0 else None, share_days_ic_pos=float((IC[c] > 0).mean()),
                                   within_day_by_year={int(t): float(W.loc[W.index.year == t, c].mean()) for t in TEST_YEARS})
    if c in SP: bd = DayBootstrap(SP.index.values, B=1000); B[c]["top5_minus_bottom5_next_ret"] = dict(mean=float(SP[c].mean()), ci=bd.ci(lambda x: x.mean(), SP[c].values)[:2])
def dpair(a, b): d = (W[b] - W[a]).dropna(); return dict(mean=float(d.mean()), t=float(ttest_1samp(d, 0).statistic))
B["paired_within_day"] = {"final minus stock base-rate": dpair("stock base-rate", EA), "final minus LightGBM Full": dpair(LF, EA), "LightGBM All9 minus LightGBM Full": dpair(LF, LA), "LightGBM Full minus stock base-rate": dpair("stock base-rate", LF)}
out["B_which_day_vs_which_stock"] = B
print("B", {c: (round(B[c]["pooled_auroc"], 3), round(B[c]["which_day_auroc"], 3), round(B[c]["within_day_auroc"], 3), round(B[c]["rank_ic"], 3)) for c in MODELS}, B["paired_within_day"], flush=True)
# backtest of the final score: top-k at the t+1 open, sold at the t+1 close (same rule and costs as 10_backtest.py)
COST = 0.001425 * 2 + 0.0015
def stats(r):
    r = pd.Series(r).dropna(); mu, sd = r.mean(), r.std(ddof=1); eq = (1 + r).cumprod()
    return dict(n_days=int(len(r)), mean_daily=float(mu), ann_return=float((1 + mu) ** 252 - 1), sharpe=float(mu / sd * np.sqrt(252)) if sd > 0 else None, max_drawdown=float((eq / eq.cummax() - 1).min()), hit_rate=float((r > 0).mean()))
BT = {}; tie = np.random.default_rng(1).random(len(P)) * 1e-9
for c in [LF, EA, "stock base-rate", "SOX-return rule"]:
    for k in (5, 10):
        sel = {}; rows = []
        for d, dd in P.assign(_s=P[c].values + tie).dropna(subset=["ret_intra"]).groupby("date"):
            top = dd.nlargest(k, "_s"); sel[d] = set(top.stock_id); rows.append((d, top.ret_intra.mean()))
        r = pd.Series(dict(rows)); ds = sorted(sel); turn = float(np.mean([1 - len(sel[a] & sel[b]) / k for a, b in zip(ds[:-1], ds[1:])]))
        BT[f"{c}|top{k}"] = dict(gross=stats(r), net=stats(r - COST), net_with_slippage_5bp_per_side=stats(r - COST - 0.001), break_even_round_trip_cost=float(r.mean()), daily_name_turnover=turn, round_trips_per_year=252 * k,
                                 net_ann_return_by_cost={f"{100 * cst:.2f}%": float((1 + (r - cst).mean()) ** 252 - 1) for cst in (0.0, 0.0005, 0.001, 0.002, 0.00264, 0.00435, 0.00585)})
out["B_backtest_final_score_intraday"] = dict(cost_round_trip=COST, note="every position is opened and closed within the day, so trading turnover is 100 % per day by construction; 'daily_name_turnover' is the share of the top-k names replaced from one day to the next", strategies=BT)
print("BT", {k: (round(v["gross"]["sharpe"], 2), round(v["net"]["ann_return"], 2), round(100 * v["break_even_round_trip_cost"], 3)) for k, v in BT.items()}, flush=True)
# ---------------------------------------------------------------- D
p = P[EA].values; conf = np.abs(p - 0.5); q = pd.qcut(conf, 5, labels=False); rb = pd.qcut(p, 10, labels=False, duplicates="drop")
out["D_confidence"] = dict(brier=float(brier_score_loss(y, p)), brier_base_rate=float(brier_score_loss(y, np.full(len(y), y.mean()))), score_range=[float(np.quantile(p, 0.01)), float(np.quantile(p, 0.99))],
    reliability_deciles=[dict(mean_p=float(p[rb == i].mean()), up_rate=float(y[rb == i].mean()), n=int((rb == i).sum())) for i in range(int(rb.max()) + 1)],
    accuracy_by_confidence_quintile=[dict(quintile=int(i) + 1, mean_abs_p_minus_half=float(conf[q == i].mean()), accuracy=float(accuracy_score(y[q == i], p[q == i] >= 0.5)), auroc=float(auc_stat(y[q == i], p[q == i])), n=int((q == i).sum()),
                                          accuracy_by_year={int(t): float(accuracy_score(y[(q == i) & (yrs == t)], p[(q == i) & (yrs == t)] >= 0.5)) for t in TEST_YEARS}) for i in range(5)],
    coverage_curve=[dict(coverage=float(cv), accuracy=float(accuracy_score(y[conf >= np.quantile(conf, 1 - cv)], p[conf >= np.quantile(conf, 1 - cv)] >= 0.5))) for cv in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1)],
    note="descriptive: quantiles of the pooled test-set confidence; no abstention threshold was selected on the test set")
print("D", [(d["quintile"], round(d["accuracy"], 3)) for d in out["D_confidence"]["accuracy_by_confidence_quintile"]], flush=True)
dump(out, "final_model_checks.json")
# ---------------------------------------------------------------- C and E (LightGBM fits, seed 42)
NEW = {"S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"]}
df = load(); ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); ex["date"] = pd.to_datetime(ex["date"]); m = df[KEY].merge(ex, on=KEY, how="left"); m.index = df.index
for c in NEW["S7"] + NEW["S8"]: df[c] = m[c].values
FOLDS = list(folds(df)); te = df.year.isin(TEST_YEARS); TW = SRC["S1"] + SRC["S2"] + SRC["S3"]
def run(feats, label):
    prob = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, tst in FOLDS:
        itr, iva, tr = [i[df.loc[i, label].notna()] for i in (itr, iva, tr)]; Pm = dict(LGB_PARAMS, n_jobs=8)
        mm = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **Pm).fit(df.loc[itr, feats], df.loc[itr, label], eval_set=[(df.loc[iva, feats], df.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        prob.loc[tst] = lgb.LGBMClassifier(n_estimators=int(mm.best_iteration_ or 1), random_state=42, **Pm).fit(df.loc[tr, feats], df.loc[tr, label]).predict_proba(df.loc[tst, feats])[:, 1]
    return prob
ARMS = {"TW": TW, "TW+S4": TW + SRC["S4"], "TW+S8": TW + NEW["S8"], "TW+S4+S8": TW + SRC["S4"] + NEW["S8"], "TW+S4+S7": TW + SRC["S4"] + NEW["S7"], "TW+S4+S7+S8": TW + SRC["S4"] + NEW["S7"] + NEW["S8"]}
C = {}
for label in ["y_up", "y_gap", "y_intra"]:
    mk = te & df[label].notna(); yy = df.loc[mk, label].values; b2 = DayBootstrap(df.loc[mk, "date"].values, B=1000); pr = {a: run(f, label).loc[mk].values for a, f in ARMS.items()}
    d2 = lambda a, b: (lambda ci: dict(diff=float(auc_stat(yy, pr[b]) - auc_stat(yy, pr[a])), ci=ci[:2]))(b2.ci_diff(auc_stat, yy, pr[a], pr[b]))
    C[label] = dict(auroc={a: float(auc_stat(yy, pr[a])) for a in ARMS}, S8_on_top_of_S4=d2("TW+S4", "TW+S4+S8"), S4_on_top_of_S8=d2("TW+S8", "TW+S4+S8"), S8_alone_vs_TW=d2("TW", "TW+S8"), S4_alone_vs_TW=d2("TW", "TW+S4"),
                    S7_on_top_of_S4=d2("TW+S4", "TW+S4+S7"), S7_on_top_of_S4_S8=d2("TW+S4+S8", "TW+S4+S7+S8"))
    print("C", label, {k: round(v, 4) for k, v in C[label]["auroc"].items()}, {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]]) for k, v in C[label].items() if k != "auroc"}, flush=True)
out["C_S4_vs_S8"] = C; dump(out, "final_model_checks.json")
mk = te; yy = df.loc[mk, "y_up"].values; b2 = DayBootstrap(df.loc[mk, "date"].values, B=1000); drop = ["us_usdtwd_chg", "us_dxy_ret"]
pf = run(ALL, "y_up").loc[mk].values; pd_ = run([c for c in ALL if c not in drop], "y_up").loc[mk].values; ci = b2.ci_diff(auc_stat, yy, pf, pd_)
out["E_drop_fx_features"] = dict(dropped=drop, full=float(auc_stat(yy, pf)), without=float(auc_stat(yy, pd_)), diff=float(auc_stat(yy, pd_) - auc_stat(yy, pf)), ci=ci[:2])
print("E", out["E_drop_fx_features"], flush=True); dump(out, "final_model_checks.json")
# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4)); A = out["A_decomposition"]
a = ax[0]; lab = ["more sources\n(LightGBM)", "more sources\n(ensemble)", "ensemble\n(5 sources)", "ensemble\n(9 sources)", "total"]; ks = ["more_sources_single_model", "more_sources_ensemble", "ensemble_on_five_sources", "ensemble_on_nine_sources", "total"]
v = [A[k]["diff"] for k in ks]; lo = [A[k]["diff"] - A[k]["ci"][0] for k in ks]; hi = [A[k]["ci"][1] - A[k]["diff"] for k in ks]
a.bar(range(5), v, yerr=[lo, hi], color=["#4a3aa7", "#4a3aa7", "#eb6834", "#eb6834", "#8a8a8a"], capsize=3, width=0.62); a.axhline(0, color="#999", lw=0.8); a.set_xticks(range(5)); a.set_xticklabels(lab, fontsize=8)
for i, x in enumerate(v): a.text(i, A[ks[i]]["ci"][1] + 0.0008, f"{x:+.3f}", ha="center", fontsize=8.5)
a.set_ylabel("ΔAUROC (95 % trading-day block CI)"); a.set_title("(a) Where the gain of the final model comes from", fontsize=10, fontweight="bold")
a = ax[1]; ms = [LF, LA, EA, "stock base-rate", "SOX-return rule"]; nm = ["LightGBM\n5 sources", "LightGBM\n9 sources", "final\n(ensemble, 9)", "per-stock\nbase rate", "SOX-return\nrule"]; x = np.arange(len(ms))
a.bar(x - 0.2, [B[c]["which_day_auroc"] for c in ms], 0.38, color="#eda100", label="which day (AUROC of the day-mean score)"); a.bar(x + 0.2, [B[c]["within_day_auroc"] for c in ms], 0.38, color="#4a3aa7", label="which stock (mean within-day AUROC)")
for i, c in enumerate(ms): a.text(i - 0.2, B[c]["which_day_auroc"] + 0.002, f"{B[c]['which_day_auroc']:.3f}", ha="center", fontsize=8); a.text(i + 0.2, B[c]["within_day_auroc"] + 0.002, f"{B[c]['within_day_auroc']:.3f}", ha="center", fontsize=8)
a.axhline(0.5, color="#999", lw=0.8); a.set_ylim(0.47, 0.66); a.set_xticks(x); a.set_xticklabels(nm, fontsize=8); a.legend(fontsize=8, frameon=False, loc="upper right"); a.set_ylabel("AUROC"); a.set_title("(b) Which day vs which stock", fontsize=10, fontweight="bold")
a = ax[2]; Cc = out["C_S4_vs_S8"]; ar = ["TW", "TW+S8", "TW+S4", "TW+S4+S8", "TW+S4+S7+S8"]; x = np.arange(len(ar)); cols = {"y_up": "#4a3aa7", "y_gap": "#eb6834", "y_intra": "#1baf7a"}; nm2 = {"y_up": "Close→Close", "y_gap": "Close→Open (gap)", "y_intra": "Open→Close"}
for j, l in enumerate(["y_up", "y_gap", "y_intra"]):
    a.bar(x + (j - 1) * 0.27, [Cc[l]["auroc"][k] for k in ar], 0.26, color=cols[l], label=nm2[l])
    for i, k in enumerate(ar): a.text(i + (j - 1) * 0.27, Cc[l]["auroc"][k] + 0.004, f"{Cc[l]['auroc'][k]:.3f}", ha="center", fontsize=6.5, rotation=90)
a.axhline(0.5, color="#999", lw=0.8); a.set_ylim(0.47, 0.90); a.set_xticks(x); a.set_xticklabels(["TW only\n(S1–S3)", "TW + S8\n(TX night)", "TW + S4\n(U.S.)", "TW + S4\n+ S8", "TW + S4\n+ S7 + S8"], fontsize=8); a.legend(fontsize=8, frameon=False, loc="upper center", ncol=3); a.set_ylabel("AUROC (LightGBM, seed 42)")
a.set_title("(c) What the TX night session adds to the U.S. source", fontsize=10, fontweight="bold")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "figures/final/f21_final_model_checks.png", dpi=200); print("figure saved; done")
