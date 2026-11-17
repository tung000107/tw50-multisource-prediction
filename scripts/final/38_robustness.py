"""Robustness of the headline conclusions: is each one supported by more than a single number?
 main   (existing out-of-sample predictions + a few LightGBM refits)
   A  rules: TX night-return rule vs SOX-return rule, by year and on the fresh hold-out
   B  final nine-source model vs the night rule, vs LightGBM five sources, vs LightGBM nine sources: pooled CI + per-year sign
   C  S4 vs S8 targeted ablation repeated over 4 seeds (the first version used seed 42 only): per-seed differences, seed-mean CI, per-year sign
   D  seed spread of our own nine-source models (LightGBM, LSTM) -> how large is "one seed" on this task
   E  Taiwan-only sources near chance: LightGBM and LR, pooled and by year
   F  within-day AUROC of the final model vs the per-stock base rate: block-bootstrap CI over days (the first version reported a t statistic only)
 alstm  official ALSTM on nine sources re-run 3x with a free random initialisation (38_run_alstm_reruns.sh): per-run AUROC,
        final model vs every run, seed-averaged vs seed-averaged, single seed vs single run
Output: results/final/robustness.json"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import *
STAGE = sys.argv[1] if len(sys.argv) > 1 else "main"; KEY = ["date", "stock_id"]; OUTF = RES / "robustness.json"
out = json.load(open(OUTF)) if OUTF.exists() else {}
P = pd.read_parquet(RES / "oos_sota_headtohead.parquet"); P["stock_id"] = P.stock_id.astype(str); P["date"] = pd.to_datetime(P.date)
FIN, LA9, LF5 = "Ensemble LightGBM+LSTM All9 (ours, final)", "LightGBM All9 (S1-S9)", "LightGBM Full (S1-S5)"
def paired(d, a, b, ycol="y_up"):   # b minus a on the rows of d, trading-day block bootstrap + per-year sign
    d = d.dropna(subset=[a, b]); y = d[ycol].values; bs = DayBootstrap(d.date.values, B=1000); ci = bs.ci_diff(auc_stat, y, d[a].values, d[b].values); yr = d.date.dt.year.values
    by = {int(t): float(auc_stat(y[yr == t], d.loc[yr == t, b].values) - auc_stat(y[yr == t], d.loc[yr == t, a].values)) for t in sorted(set(yr))}; df_ = float(auc_stat(y, d[b].values) - auc_stat(y, d[a].values))
    return dict(n=int(len(d)), a=float(auc_stat(y, d[a].values)), b=float(auc_stat(y, d[b].values)), diff=df_, ci=ci[:2], p_boot=ci[2], by_year=by, same_sign_years=int(sum(np.sign(v) == np.sign(df_) for v in by.values())), n_years=len(by))

if STAGE == "main":
    df = load(); ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); ex["date"] = pd.to_datetime(ex["date"]); m = df[KEY].merge(ex, on=KEY, how="left"); m.index = df.index
    S8 = ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"]
    assert all(c in m.columns for c in S8)
    for c in S8: df[c] = m[c].values
    df["stock_id"] = df.stock_id.astype(str); P = P.merge(df[KEY + ["tx_night_ret"]], on=KEY, how="left"); P["night rule"] = P.tx_night_ret.fillna(0); P["SOX rule"] = P["SOX-return rule"]
    # ---- A rules
    A = dict(walk_forward=paired(P, "SOX rule", "night rule"))
    ho = pd.read_parquet(RES / "oos_fresh_holdout.parquet"); ho["stock_id"] = ho.stock_id.astype(str); ho["date"] = pd.to_datetime(ho.date)
    hx = pd.read_parquet(ROOT / "data/holdout/extra_features.parquet"); hx["date"] = pd.to_datetime(hx.date); hx["stock_id"] = hx.stock_id.astype(str)
    ho = ho.merge(hx[KEY + ["tx_night_ret"]], on=KEY, how="left"); ho["night rule"] = ho.tx_night_ret.fillna(0); ho["SOX rule"] = ho["SOX-return rule"]
    A["fresh_holdout"] = paired(ho, "SOX rule", "night rule"); A["fresh_holdout"]["n_days"] = int(ho.date.nunique()); A["fresh_holdout"]["night_rule_ci"] = DayBootstrap(ho.date.values, B=1000).ci(auc_stat, ho.y_up.values, ho["night rule"].values)[:2]
    if "Full+S7+S8" in ho: A["fresh_holdout_model_vs_night_rule"] = paired(ho, "night rule", "Full+S7+S8")
    out["A_night_rule_vs_sox_rule"] = A; print("A", {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"], v["n_years"]) for k, v in A.items()}, flush=True)
    # ---- B final model
    out["B_final_model"] = dict(vs_night_rule=paired(P, "night rule", FIN), vs_lightgbm_five=paired(P, LF5, FIN), vs_lightgbm_nine=paired(P, LA9, FIN), lightgbm_nine_vs_night_rule=paired(P, "night rule", LA9), lightgbm_nine_vs_five=paired(P, LF5, LA9))
    print("B", {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"]) for k, v in out["B_final_model"].items()}, flush=True)
    # ---- C S4 vs S8 over seeds
    FOLDS = list(folds(df)); te = df.year.isin(TEST_YEARS); TW = SRC["S1"] + SRC["S2"] + SRC["S3"]; ARMS = {"TW+S4": TW + SRC["S4"], "TW+S8": TW + S8, "TW+S4+S8": TW + SRC["S4"] + S8}; SEEDS = [42, 1, 2, 3]
    def run(feats, label, seed):
        prob = pd.Series(np.nan, index=df.index)
        for yr, tr, itr, iva, tst in FOLDS:
            itr, iva, tr = [i[df.loc[i, label].notna()] for i in (itr, iva, tr)]; Pm = dict(LGB_PARAMS, n_jobs=8)
            mm = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **Pm).fit(df.loc[itr, feats], df.loc[itr, label], eval_set=[(df.loc[iva, feats], df.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
            prob.loc[tst] = lgb.LGBMClassifier(n_estimators=int(mm.best_iteration_ or 1), random_state=seed, **Pm).fit(df.loc[tr, feats], df.loc[tr, label]).predict_proba(df.loc[tst, feats])[:, 1]
        return prob
    C = {}
    for label in ["y_up", "y_gap"]:
        mk = te & df[label].notna(); base = df.loc[mk, KEY + [label]].copy(); pr = {(a, s): run(f, label, s).loc[mk].values for a, f in ARMS.items() for s in SEEDS}
        per_seed = {str(s): {a: float(auc_stat(base[label].values, pr[(a, s)])) for a in ARMS} for s in SEEDS}
        for a in ARMS: base[a] = np.mean([pr[(a, s)] for s in SEEDS], axis=0)
        d_s4 = [per_seed[str(s)]["TW+S4+S8"] - per_seed[str(s)]["TW+S8"] for s in SEEDS]; d_s8 = [per_seed[str(s)]["TW+S4+S8"] - per_seed[str(s)]["TW+S4"] for s in SEEDS]; d_alone = [per_seed[str(s)]["TW+S8"] - per_seed[str(s)]["TW+S4"] for s in SEEDS]
        C[label] = dict(seeds=SEEDS, per_seed_auroc=per_seed, S4_on_top_of_S8_per_seed=d_s4, S8_on_top_of_S4_per_seed=d_s8, S8_arm_minus_S4_arm_per_seed=d_alone,
                        seed_mean=dict(S4_on_top_of_S8=paired(base, "TW+S8", "TW+S4+S8", label), S8_on_top_of_S4=paired(base, "TW+S4", "TW+S4+S8", label), S8_arm_minus_S4_arm=paired(base, "TW+S4", "TW+S8", label)))
        print("C", label, {k: [round(x, 4) for x in v] for k, v in C[label].items() if k.endswith("per_seed")}, {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"]) for k, v in C[label]["seed_mean"].items()}, flush=True)
    out["C_S4_vs_S8_over_seeds"] = C
    # ---- D seed spread of our nine-source models
    L = pd.read_parquet(RES / "oos_sota_lgb.parquet"); S = pd.read_parquet(RES / "oos_sota_seq.parquet"); D = {}
    for nm, d, pre in [("LightGBM All9", L, "LightGBM|All9|"), ("LightGBM Full+S7+S8", L, "LightGBM|Full+S7+S8|"), ("LSTM All9", S, "LSTM-W5|All9|"), ("Transformer All9", S, "Transformer-W5|All9|")]:
        v = [float(auc_stat(d.y_up.values, d[c].values)) for c in d.columns if c.startswith(pre)]; D[nm] = dict(per_seed=v, mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), min=float(min(v)), max=float(max(v)), seed_mean_probability=float(auc_stat(d.y_up.values, d[[c for c in d.columns if c.startswith(pre)]].mean(axis=1).values)))
    out["D_seed_spread_ours"] = D; print("D", {k: (round(v["mean"], 4), round(v["std"], 4), round(v["seed_mean_probability"], 4)) for k, v in D.items()}, flush=True)
    # ---- E Taiwan-only near chance: two model families
    tab = pd.read_parquet(RES / "oos_tabular.parquet"); tab["date"] = pd.to_datetime(tab.date); bs = DayBootstrap(tab.date.values, B=1000); E = {}
    for nm, c in [("LightGBM TW-only (seed 42)", "LightGBM|S1+S2+S3|42"), ("LR TW-only", "LR|S1+S2+S3|0"), ("LightGBM S4-only (seed 42)", "LightGBM|S4|42"), ("LR S4-only", "LR|S4|0"), ("stock base-rate", "Rule|stock base-rate|0")]:
        if c in tab: E[nm] = dict(auroc=float(auc_stat(tab.y_up.values, tab[c].values)), ci=bs.ci(auc_stat, tab.y_up.values, tab[c].values)[:2], by_year={int(t): float(auc_stat(tab.y_up.values[tab.year == t], tab.loc[tab.year == t, c].values)) for t in TEST_YEARS})
    out["E_taiwan_only"] = E; print("E", {k: (round(v["auroc"], 4), [round(x, 4) for x in v["ci"]]) for k, v in E.items()}, flush=True)
    # ---- F within-day AUROC: block bootstrap over days
    P2 = P.merge(tab[KEY + ["Rule|stock base-rate|0"]].assign(stock_id=tab.stock_id.astype(str)), on=KEY, how="left"); g = P2.groupby("date")
    wd = lambda c: g.apply(lambda d: roc_auc_score(d.y_up, d[c]) if d.y_up.nunique() == 2 and d[c].nunique() > 1 else np.nan)
    W = pd.DataFrame({c: wd(c) for c in [FIN, LA9, LF5, "Rule|stock base-rate|0"]}).dropna(); rng = np.random.default_rng(2026); n = len(W); blk = 5
    def boot(v):
        r = []
        for _ in range(2000):
            st = rng.integers(0, n - blk + 1, size=int(np.ceil(n / blk))); idx = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n]; r.append(v[idx].mean())
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    F = dict(n_days=int(n), level={c: dict(mean=float(W[c].mean()), ci=boot(W[c].values)) for c in W}, by_year_final_minus_base={int(t): float((W[FIN] - W["Rule|stock base-rate|0"])[W.index.year == t].mean()) for t in TEST_YEARS})
    for nm, a, b in [("final_minus_base_rate", "Rule|stock base-rate|0", FIN), ("lightgbm_nine_minus_base_rate", "Rule|stock base-rate|0", LA9), ("lightgbm_five_minus_base_rate", "Rule|stock base-rate|0", LF5), ("final_minus_lightgbm_five", LF5, FIN)]:
        v = (W[b] - W[a]).values; F[nm] = dict(mean=float(v.mean()), ci=boot(v))
    out["F_within_day"] = F; print("F", {k: (round(v["mean"], 4), [round(x, 4) for x in v["ci"]]) for k, v in F.items() if isinstance(v, dict) and "ci" in v}, F["by_year_final_minus_base"], flush=True)
    dump(out, "robustness.json")

elif STAGE == "alstm":
    import importlib.util, glob
    base = ROOT / "data/raw/tw50_advalstm74"; meta = json.load(open(base / "meta.json")); seq = 15
    def instances(path, tes_date):
        fn = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]; E = [np.genfromtxt(os.path.join(path, f), delimiter=",") for f in fn]
        dates = np.genfromtxt(os.path.join(path, "..", "trading_dates.csv"), dtype=str, delimiter=","); ti = list(dates).index(tes_date); o = []
        for di in range(ti, len(dates)):
            if di < seq: continue
            for k in range(len(fn)):
                if abs(E[k][di][-2]) > 1e-8 and E[k][di - seq:di, :].min() > -123320: o.append((pd.Timestamp(str(dates[di - 1])), fn[k].replace(".csv", ""), (E[k][di][-2] + 1) / 2))
        return o
    INS = {str(y): instances(str(base / str(y) / "ourpped"), meta[str(y)]["tes"]) for y in TEST_YEARS}; runs = {}
    for tag in ["33_adv74", "33_adv74r1", "33_adv74r2", "33_adv74r3"]:
        parts = []
        for y in TEST_YEARS:
            pf = ROOT / "external/Adv-ALSTM/saved_model" / f"{tag}_{y}_ALSTM" / "model_tes_prediction.csv"
            if not pf.exists(): parts = None; print("missing", pf); break
            sc = np.loadtxt(pf); ins = INS[str(y)]; assert len(sc) == len(ins), (tag, y, len(sc), len(ins)); parts.append(pd.DataFrame({"date": [i[0] for i in ins], "stock_id": [i[1] for i in ins], "y_chk": [i[2] for i in ins], tag: sc}))
        if parts: runs[tag] = pd.concat(parts)
    d = P[KEY + ["y_up", FIN, LA9]].copy()
    for tag, r in runs.items(): d = d.merge(r[KEY + [tag]], on=KEY, how="left")
    tags = list(runs); d = d.dropna(subset=tags + [FIN]); d["ALSTM mean of runs"] = d[tags].mean(axis=1)
    L = pd.read_parquet(RES / "oos_sota_lgb.parquet"); L["stock_id"] = L.stock_id.astype(str); L["date"] = pd.to_datetime(L.date); d = d.merge(L[KEY + ["LightGBM|All9|1"]], on=KEY, how="left")
    per = {t: float(auc_stat(d.y_up.values, d[t].values)) for t in tags}; v = list(per.values())
    R = dict(n_rows=int(len(d)), runs={"33_adv74": "fixed TF seed 123456 (-f 1), the run reported so far", "33_adv74r1..r3": "free random initialisation (-f 0)"}, per_run_auroc=per, run_mean=float(np.mean(v)), run_std=float(np.std(v, ddof=1)) if len(v) > 1 else None, run_min=float(min(v)), run_max=float(max(v)),
             final_vs_each_run={t: paired(d, t, FIN) for t in tags}, final_vs_mean_of_runs=paired(d, "ALSTM mean of runs", FIN), lightgbm_nine_seed_mean_vs_mean_of_runs=paired(d, "ALSTM mean of runs", LA9),
             single_seed_lightgbm_nine_vs_each_run={t: paired(d, t, "LightGBM|All9|1") for t in tags})
    _y = d.y_up.values; _m = d["ALSTM mean of runs"].values; _b = DayBootstrap(d.date.values, B=1000); _thr = float(np.median(_m)); R["mean_of_runs_scores"] = dict(how="the four runs' test scores are averaged row by row, then one AUROC is computed", auroc=float(auc_stat(_y, _m)), ci=_b.ci(auc_stat, _y, _m)[:2], mean_of_the_four_aurocs=float(np.mean(v)))
    out["G_official_alstm_reruns"] = R
    print("G per run", {k: round(x, 4) for k, x in per.items()}, "std", R["run_std"]); print("final vs each", {t: (round(x["diff"], 4), [round(z, 4) for z in x["ci"]], x["same_sign_years"]) for t, x in R["final_vs_each_run"].items()})
    print("final vs mean of runs", round(R["final_vs_mean_of_runs"]["b"], 4), round(R["final_vs_mean_of_runs"]["a"], 4), round(R["final_vs_mean_of_runs"]["diff"], 4), R["final_vs_mean_of_runs"]["ci"], R["final_vs_mean_of_runs"]["same_sign_years"])
    print("single-seed LightGBM nine vs each run", {t: (round(x["diff"], 4), [round(z, 4) for z in x["ci"]]) for t, x in R["single_seed_lightgbm_nine_vs_each_run"].items()})
    dump(out, "robustness.json")

elif STAGE == "bt":
    # H  is the gross intraday top-5 result more than one lucky draw? 2,000 random top-5 portfolios (the first version showed one), and the
    #    selection effect itself: top-5 minus the equal-weight 50-stock basket on the same day, block bootstrap over days
    tab = pd.read_parquet(RES / "oos_tabular.parquet"); lab = pd.read_parquet(RES / "oos_labels.parquet"); tab["stock_id"] = tab.stock_id.astype(str); tab["date"] = pd.to_datetime(tab.date)
    tab["TW-only (Open->Close model)"] = lab["y_intra|TW-only"].values; tab["Full (Open->Close model)"] = lab["y_intra|Full"].values
    tab = tab.merge(P[KEY + [FIN]].rename(columns={FIN: "final nine-source model (Close->Close score)"}), on=KEY, how="left"); tab = tab.dropna(subset=["ret_intra"])
    R = tab.pivot(index="date", columns="stock_id", values="ret_intra"); ok = R.notna().values; Rv = np.nan_to_num(R.values); nd, ns = Rv.shape; rng = np.random.default_rng(2026); sh = lambda r: float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    rs = []
    for _ in range(2000):
        u = rng.random((nd, ns)); u[~ok] = -1; idx = np.argsort(-u, axis=1)[:, :5]; rs.append(sh(np.take_along_axis(Rv, idx, axis=1).mean(axis=1)))
    rs = np.array(rs); basket = np.array([Rv[i, ok[i]].mean() for i in range(nd)]); H = dict(n_days=int(nd), random_top5_sharpe=dict(n_draws=2000, mean=float(rs.mean()), p2_5=float(np.percentile(rs, 2.5)), p97_5=float(np.percentile(rs, 97.5)), max=float(rs.max())), equal_weight_basket_sharpe=sh(basket), strategies={})
    def boot(v, blk=5):
        n = len(v); r = []
        for _ in range(2000):
            st = rng.integers(0, n - blk + 1, size=int(np.ceil(n / blk))); ix = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n]; r.append(v[ix].mean())
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    tie = np.random.default_rng(1).random(len(tab)) * 1e-9
    for c in ["TW-only (Open->Close model)", "Full (Open->Close model)", "final nine-source model (Close->Close score)"]:
        S = tab.assign(_s=tab[c].values + tie).pivot(index="date", columns="stock_id", values="_s").reindex(index=R.index, columns=R.columns).values; S = np.where(ok & ~np.isnan(S), S, -np.inf); idx = np.argsort(-S, axis=1)[:, :5]; r = np.take_along_axis(Rv, idx, axis=1).mean(axis=1); ex_ = r - basket
        yr = R.index.year.values; H["strategies"][c] = dict(gross_sharpe=sh(r), share_of_random_draws_below=float((rs < sh(r)).mean()), mean_daily_excess_over_basket=float(ex_.mean()), excess_ci=boot(ex_), excess_by_year={int(t): float(ex_[yr == t].mean()) for t in TEST_YEARS}, positive_years=int(sum(ex_[yr == t].mean() > 0 for t in TEST_YEARS)))
    out["H_backtest_gross_vs_random"] = H; print("H random", H["random_top5_sharpe"], "basket", round(H["equal_weight_basket_sharpe"], 2)); print({k: (round(v["gross_sharpe"], 2), round(v["share_of_random_draws_below"], 3), round(v["mean_daily_excess_over_basket"] * 1e4, 2), [round(x * 1e4, 2) for x in v["excess_ci"]], v["positive_years"]) for k, v in H["strategies"].items()})
    dump(out, "robustness.json")

elif STAGE in ("more", "more2"):   # "more2" reuses the stored I and J blocks and adds N-Q
    # Checks requested by the second review (Codex, 2026-11-14): results that had been reported from seed 42 only, or with a mismatched reference
    import itertools
    from math import factorial
    df = load(); ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); ex["date"] = pd.to_datetime(ex["date"]); nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); nf["date"] = pd.to_datetime(nf["date"]); df = df.merge(ex, on=KEY, how="left").merge(nf, on=KEY, how="left")
    NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"], "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
           "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"], "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
    FOLDS = list(folds(df)); te = df.year.isin(TEST_YEARS); TW = SRC["S1"] + SRC["S2"] + SRC["S3"]; SEEDS = [42, 1, 2, 3]
    def run(feats, label, seed):
        prob = pd.Series(np.nan, index=df.index); Pm = dict(LGB_PARAMS, n_jobs=6)
        for yr, tr, itr, iva, tst in FOLDS:
            itr, iva, tr = [i[df.loc[i, label].notna()] for i in (itr, iva, tr)]; X = df[[f for f in feats if f not in ("text_score", "market_text_score")]]
            if "text_score" in feats: X = X.assign(text_score=df[f"text_score_{yr}"].values, market_text_score=df[f"market_text_score_{yr}"].values)   # fold-trained text scores
            mm = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **Pm).fit(X.loc[itr], df.loc[itr, label], eval_set=[(X.loc[iva], df.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
            prob.loc[tst] = lgb.LGBMClassifier(n_estimators=int(mm.best_iteration_ or 1), random_state=seed, **Pm).fit(X.loc[tr], df.loc[tr, label]).predict_proba(X.loc[tst])[:, 1]
        return prob
    from joblib import Parallel, delayed
    def block(label, arms, pairs):
        mk = te & df[label].notna(); base = df.loc[mk, KEY + [label]].copy(); jobs = [(a, s) for a in arms for s in SEEDS]; pr = dict(zip(jobs, Parallel(n_jobs=4, prefer="threads")(delayed(lambda a, s: run(arms[a], label, s).loc[mk].values)(a, s) for a, s in jobs)))
        per = {str(s): {a: float(auc_stat(base[label].values, pr[(a, s)])) for a in arms} for s in SEEDS}
        for a in arms: base[a] = np.mean([pr[(a, s)] for s in SEEDS], axis=0)
        bs_ = DayBootstrap(base.date.values, B=1000); r = dict(seeds=SEEDS, per_seed_auroc=per, seed_mean_auroc={a: dict(auroc=float(auc_stat(base[label].values, base[a].values)), ci=bs_.ci(auc_stat, base[label].values, base[a].values)[:2]) for a in arms}, paired={})
        for nm, (a, b) in pairs.items(): r["paired"][nm] = dict(paired(base, a, b, label), per_seed=[per[str(s)][b] - per[str(s)][a] for s in SEEDS])
        return r
    # I  label variants (R4) over four seeds
    I = out["I_label_variants_over_seeds"] if STAGE == "more2" else {lab: block(lab, {"Full": ALL, "S4-only": SRC["S4"], "TW-only": TW}, {"Full minus S4-only": ("S4-only", "Full"), "Full minus TW-only": ("TW-only", "Full")}) for lab in ["y_gap", "y_intra", "y_rel"]}
    out["I_label_variants_over_seeds"] = I
    for lab in I: print("I", lab, {a: (round(v["auroc"], 4), [round(x, 4) for x in v["ci"]]) for a, v in I[lab]["seed_mean_auroc"].items()}, {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"], [round(x, 4) for x in v["per_seed"]]) for k, v in I[lab]["paired"].items()}, flush=True)
    # J  add-on sources on top of the five (R12) over four seeds
    J_ = out["J_addon_sources_over_seeds"] if STAGE == "more2" else block("y_up", {"Full": ALL, **{f"Full+{k}": ALL + NEW[k] for k in NEW}}, {f"Full+{k} minus Full": ("Full", f"Full+{k}") for k in NEW}); out["J_addon_sources_over_seeds"] = J_
    print("J", {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"], [round(x, 4) for x in v["per_seed"]]) for k, v in J_["paired"].items()}, flush=True)
    # K  matched leave-one-out (seed 42 Full vs seed 42 drop-one; R3 compared the 4-seed Full with single-seed subsets)
    tab = pd.read_parquet(RES / "oos_tabular.parquet"); tab["date"] = pd.to_datetime(tab.date); S = list(SRC); nm = lambda c: "+".join(c); fullc = f"LightGBM|{nm(S)}|42"; Kd = {}
    for s_ in S: Kd[f"drop {s_}"] = paired(tab, f"LightGBM|{nm([x for x in S if x != s_])}|42", fullc)
    out["K_matched_leave_one_out_seed42"] = Kd; print("K", {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"]) for k, v in Kd.items()}, flush=True)
    # L  bootstrap intervals for the pairwise Shapley interaction indices (same formula as 09_explain.py)
    y = tab.y_up.values; bsL = DayBootstrap(tab.date.values, B=500); n = len(S); Ld = {}
    for kind, seed in [("LightGBM", 42), ("LR", 0)]:
        cols = {frozenset(c): f"{kind}|{nm(c)}|{seed}" for r in range(1, 6) for c in itertools.combinations(S, r)}
        def inter(v):
            o = {}
            for i, k in itertools.combinations(S, 2):
                tot = 0.0
                for r in range(n - 1):
                    for c in itertools.combinations([s for s in S if s not in (i, k)], r):
                        w = factorial(len(c)) * factorial(n - len(c) - 2) / factorial(n - 1); c = frozenset(c); tot += w * (v(c | {i, k}) - v(c | {i}) - v(c | {k}) + v(c))
                o[f"{i}x{k}"] = tot
            return o
        point = inter(lambda fs: 0.5 if not fs else auc_stat(y, tab[cols[fs]].values)); bo = {k: [] for k in point}
        for j in bsL.samples:
            cache = {fs: auc_stat(y[j], tab[c].values[j]) for fs, c in cols.items()}; b_ = inter(lambda fs: 0.5 if not fs else cache[fs])
            for k in b_: bo[k].append(b_[k])
        Ld[kind] = {k: dict(value=float(point[k]), ci=[float(np.percentile(bo[k], 2.5)), float(np.percentile(bo[k], 97.5))]) for k in point}
    out["L_interaction_index_ci"] = Ld; print("L", {kd: {k: (round(v["value"], 4), [round(x, 4) for x in v["ci"]]) for k, v in d.items() if abs(v["value"]) > 0.004} for kd, d in Ld.items()}, flush=True)
    if STAGE == "more2":
        # N  the remaining label variants of R4 over four seeds (dividend-adjusted, flat days excluded) and the primary label with the same seeds
        df["y_up_noflat"] = df.y_up.where(df.ret_next != 0)
        N = {lab: block(lab, {"Full": ALL, "S4-only": SRC["S4"], "TW-only": TW}, {"Full minus S4-only": ("S4-only", "Full"), "Full minus TW-only": ("TW-only", "Full")}) for lab in ["y_up", "y_tr", "y_up_noflat"]}
        out["N_more_label_variants_over_seeds"] = N
        for lab in N: print("N", lab, {a: round(v["auroc"], 4) for a, v in N[lab]["seed_mean_auroc"].items()}, {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"]) for k, v in N[lab]["paired"].items()}, flush=True)
        # O  add-on sources on the other three targets over four seeds (R12 showed seed 42)
        O = {lab: block(lab, {"Full": ALL, **{f"Full+{k}": ALL + NEW[k] for k in NEW}, "All9": ALL + sum(NEW.values(), [])}, {**{f"Full+{k} minus Full": ("Full", f"Full+{k}") for k in NEW}, "All9 minus Full": ("Full", "All9")}) for lab in ["y_gap", "y_intra", "y_rel"]}
        out["O_addon_sources_other_targets_over_seeds"] = O
        for lab in O: print("O", lab, {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"]) for k, v in O[lab]["paired"].items()}, flush=True)
        # P  within-day ranking: nine-source LightGBM vs the final ensemble (block bootstrap over days)
        from sklearn.metrics import roc_auc_score as _ras
        g = P.groupby("date"); wdf = lambda c: g.apply(lambda d: _ras(d.y_up, d[c]) if d.y_up.nunique() == 2 and d[c].nunique() > 1 else np.nan); W = pd.DataFrame({c: wdf(c) for c in [FIN, LA9]}).dropna(); rng = np.random.default_rng(2026); n_ = len(W); v_ = (W[LA9] - W[FIN]).values; r_ = []
        for _ in range(2000):
            st = rng.integers(0, n_ - 5 + 1, size=int(np.ceil(n_ / 5))); ix = (st[:, None] + np.arange(5)[None, :]).ravel()[:n_]; r_.append(v_[ix].mean())
        out["P_within_day_lightgbm_nine_minus_final"] = dict(lightgbm_nine=float(W[LA9].mean()), final=float(W[FIN].mean()), mean=float(v_.mean()), ci=[float(np.percentile(r_, 2.5)), float(np.percentile(r_, 97.5))], by_year={int(t): float(v_[W.index.year == t].mean()) for t in TEST_YEARS})
        print("P", out["P_within_day_lightgbm_nine_minus_final"], flush=True)
    # M  XGBoost Taiwan-only and the day-mean ("which day") comparison with the night rule are read from existing files by the deck
    dump(out, "robustness.json")
