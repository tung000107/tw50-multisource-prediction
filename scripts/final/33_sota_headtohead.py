"""Head-to-head on the Taiwan 50: SOTA architectures vs our multi-source models under ONE protocol (same rows, walk-forward 2023-2026,
label y_up). Two questions: (1) do the published architectures (CausalStock, Adv-ALSTM / ALSTM / LSTM) beat our tabular / sequence
models on the same data? (2) when the SOTA architectures are given OUR inputs (45 features S1-S5; all 9 sources), do they catch up?
If (2) is yes, the gain is information, not architecture.
Stages (argv[1]):
  causal      CausalStock v2 (our re-implementation, 14_causalstock_v2.py recipe) with 45 / All9 features, 3 seeds x 4 folds  -> oos_sota_causal.parquet
  seq         our LSTM / Transformer (03_sequence.py) on All9, 3 seeds                                                            -> oos_sota_seq.parquet
  lgb         LightGBM on All9 and Full+S7+S8, 4 seeds (02_tabular policy)                                                       -> oos_sota_lgb.parquet
  adv_gen     write Adv-ALSTM input files with our 45 features (data/raw/tw50_advalstm45), same format as 28_advalstm_tw50.py
  adv_collect parse the official-code logs + saved test scores (both 11-feature and 45-feature runs)                             -> oos_sota_adv.parquet
  assemble    metrics with trading-day block-bootstrap CIs, paired differences, figure f20                                       -> sota_headtohead.json
"""
import sys, os, json, glob, re, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import *
STAGE = sys.argv[1]
NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"],
       "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"],
       "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
ALL9 = ALL + sum(NEW.values(), []); FS = {"S1-S5": ALL, "All9": ALL9, "Full+S7+S8": ALL + NEW["S7"] + NEW["S8"]}
ADV11 = ROOT / "data/raw/tw50_advalstm"; ADV45 = ROOT / "data/raw/tw50_advalstm45"; ADV74 = ROOT / "data/raw/tw50_advalstm74"; KEY = ["date", "stock_id"]

def with_extra(df):
    ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet")
    for e in (ex, nf): e["date"] = pd.to_datetime(e["date"])
    m = df[KEY].merge(ex, on=KEY, how="left").merge(nf, on=KEY, how="left"); m.index = df.index
    for c in m.columns:
        if c not in KEY: df[c] = m[c].values
    return df
def fold_text(d, yr):
    d["text_score"] = d[f"text_score_{yr}"]; d["market_text_score"] = d[f"market_text_score_{yr}"]; return d

# ---------------------------------------------------------------- causal / causal_tuned
if STAGE in ("causal", "causal_tuned"):
    import torch
    src = open(Path(__file__).parent / "14_causalstock_v2.py").read().split("out = json.load(open(RES")[0]
    sys.argv = ["x", "none"]; g = {"__name__": "cs", "__file__": str(Path(__file__).parent / "14_causalstock_v2.py")}; exec(src, g); run_restarts, windows, zscore, T, L = g["run_restarts"], g["windows"], g["zscore"], g["T"], g["L"]
    TUNED = STAGE == "causal_tuned"; KW = {}; SUF = ""
    if TUNED:   # same architecture, a fair budget: wider price encoder (d_p 16 instead of 4), lr 1e-3 instead of 1e-5/1e-4, patience 20 (selection still on validation accuracy, as in the recipe)
        Orig = g["CausalStockV2"]; g["CausalStockV2"] = lambda D, F, n=False, t=True, l=True: Orig(D, F, n, t, l, d_p=16); KW = dict(lr=1e-3, patience=20); SUF = ", tuned"
    df = with_extra(load()); out = {}; oos = []
    for fs in ["S1-S5", "All9"]:
        feats = FS[fs]
        for yr in TEST_YEARS:
            d = fold_text(df.copy(), yr); t0 = pd.Timestamp(f"{yr}-01-01"); v0 = t0 - pd.DateOffset(months=12)
            piv = {c: d.pivot(index="date", columns="stock_id", values=c) for c in feats}; days = piv[feats[0]].index; cols = list(piv[feats[0]].columns)
            F3 = np.stack([piv[c].values for c in feats], -1); miss = np.isnan(F3).all(-1); F3 = np.nan_to_num(F3)   # a stock-day with no row at all -> masked
            Ypv = d.pivot(index="date", columns="stock_id", values="y_up").shift(1); Y2 = np.nan_to_num(Ypv.values); M2 = ~np.isnan(Ypv.values)   # label shifted one row: the window (t-L+1..t) ends on the feature day t, as our own models see it
            validwin = np.stack([~miss[t - L:t].any(0) for t in range(L, len(days))], 0); trm = (days < v0)
            Fz = np.clip(zscore(F3, trm), -10, 10); X, Y, Mk, _ = windows(Fz, Y2, M2); Mk = Mk * validwin; dd = days[L - 1:-1]
            itr, iva, ite = (dd < v0), (dd >= v0) & (dd < t0), (dd >= t0) & (dd < pd.Timestamp(f"{yr + 1}-01-01")); D = X.shape[1]
            pairs = [(day, s) for day, mrow in zip(dd[ite], Mk[ite]) for s, mk in zip(cols, mrow) if mk > 0]; P = []
            for seed in range(3):
                t = time.time(); r, pp, yy = run_restarts(T(X[itr]), T(Y[itr]), T(Mk[itr]), None, T(X[iva]), T(Y[iva]), T(Mk[iva]), None, T(X[ite]), T(Y[ite]), T(Mk[ite]), None, D, len(feats), seed, **KW)
                r["seconds"] = round(time.time() - t); out.setdefault(fs, {}).setdefault(str(yr), []).append(r); P.append(pp); print("causal", fs, yr, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
            assert len(pairs) == len(yy); oos.append(pd.DataFrame({"date": [p[0] for p in pairs], "stock_id": [p[1] for p in pairs], "y_up": yy, f"CausalStock v2 + {fs}{SUF}": np.mean(P, 0)}))
            dump(out, "sota_headtohead_causal_runs.json" if not TUNED else "sota_headtohead_causal_tuned_runs.json")
    byfs = {}
    for r in oos: byfs.setdefault(r.columns[-1], []).append(r)
    o = None
    for r in [pd.concat(v) for v in byfs.values()]: o = r if o is None else o.merge(r, on=KEY + ["y_up"], how="outer")
    o.to_parquet(RES / ("oos_sota_causal_tuned.parquet" if TUNED else "oos_sota_causal.parquet")); print("saved", o.shape)
# ---------------------------------------------------------------- seq
elif STAGE == "seq":
    src = open(Path(__file__).parent / "03_sequence.py").read().split("if __name__")[0]; g = {"__name__": "sq", "__file__": str(Path(__file__).parent / "03_sequence.py")}; exec(src, g)
    df = g["df"]; with_extra(df); ALLF = g["FOLDS"]; te = df.year.isin(TEST_YEARS); res = df.loc[te, KEY + ["year", "y_up"]].copy(); info = {}
    for kind in ["LSTM", "Transformer"]:
        for seed in [0, 1, 2]:
            name = f"{kind}-W5|All9|{seed}"; prob = pd.Series(np.nan, index=df.index); t = time.time()
            for fold in ALLF:
                yr = fold[0]; fold_text(df, yr); g["FOLDS"] = [fold]; p, inf = g["fit_predict"](kind, ALL9, 5, seed); prob[p.notna()] = p[p.notna()]; info.setdefault(name, {})[str(yr)] = inf[str(yr)]
            res[name] = prob.loc[te].values; y = res.loc[res[name].notna(), "y_up"].values; print(f"{name} auroc={auc_stat(y, res.loc[res[name].notna(), name].values):.4f} ({time.time()-t:.0f}s)", flush=True)
            res.to_parquet(RES / "oos_sota_seq.parquet"); dump(info, "sota_headtohead_seq_info.json")
# ---------------------------------------------------------------- lgb
elif STAGE == "lgb":
    import lightgbm as lgb
    df = with_extra(load()); FOLDS = list(folds(df)); te = df.year.isin(TEST_YEARS); res = df.loc[te, KEY + ["year", "y_up"]].copy()
    for fs in ["All9", "Full+S7+S8"]:
        for seed in [1, 2, 3, 4]:
            prob = pd.Series(np.nan, index=df.index)
            for yr, tr, itr, iva, tst in FOLDS:
                d = fold_text(df, yr); P = dict(LGB_PARAMS, n_jobs=6)
                m = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **P).fit(d.loc[itr, FS[fs]], d.loc[itr, "y_up"], eval_set=[(d.loc[iva, FS[fs]], d.loc[iva, "y_up"])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
                f = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=seed, **P).fit(d.loc[tr, FS[fs]], d.loc[tr, "y_up"]); prob.loc[tst] = f.predict_proba(d.loc[tst, FS[fs]])[:, 1]
            name = f"LightGBM|{fs}|{seed}"; res[name] = prob.loc[te].values; print(name, round(auc_stat(res.y_up.values, res[name].values), 4), flush=True)
    res.to_parquet(RES / "oos_sota_lgb.parquet")
# ---------------------------------------------------------------- adv_gen
elif STAGE == "adv_gen":
    NINE = len(sys.argv) > 2 and sys.argv[2] == "all9"; OUTD = ADV74 if NINE else ADV45; feats = ALL9 if NINE else ALL   # "adv_gen all9": the same files with all nine sources (74 features)
    df0 = with_extra(load()) if NINE else load(); days = np.sort(df0.date.unique()); meta = json.load(open(ADV11 / "meta.json"))
    for yr in TEST_YEARS:
        df = fold_text(df0.copy(), yr) if NINE else df0
        t0 = pd.Timestamp(f"{yr}-01-01"); v0 = pd.Timestamp(f"{yr - 1}-01-01"); t1 = pd.Timestamp(f"{yr + 1}-01-01"); dd = days[days < t1]; dd = np.concatenate([dd, days[days >= t1][:1]]); trm = df.date < v0
        mu = df.loc[trm, feats].mean(); sd = df.loc[trm, feats].std().replace(0, 1); d = OUTD / str(yr) / "ourpped"; d.mkdir(parents=True, exist_ok=True)
        for sid, gg in df.groupby("stock_id"):
            gg = gg.set_index("date").reindex(dd); X = ((gg[feats] - mu) / sd).clip(-10, 10); X = X.where(X.notna() | gg.close.isna().values[:, None], 0.0); ys = gg.y_up.shift(1); lab = np.where(ys == 1, 1.0, np.where(ys == 0, -1.0, 0.0)); arr = np.column_stack([X.values, lab, gg.close.values]); bad = np.isnan(arr).any(1); arr[bad] = -123321.0; arr[:, -2] = lab   # label one row later -> window ends on the feature day (see 28_advalstm_tw50.py); a missing feature on an existing row -> training mean (0 after z-scoring), only absent rows are padded
            np.savetxt(d / f"{sid}.csv", arr, delimiter=",", fmt="%.6f")
        pd.Series([pd.Timestamp(x).strftime("%Y-%m-%d") for x in dd]).to_csv(OUTD / str(yr) / "trading_dates.csv", index=False, header=False)
        tes = df[(df.date >= t0) & (df.date < t1)]; meta[str(yr)] = dict(tra=pd.Timestamp(dd[0]).strftime("%Y-%m-%d"), val=pd.Timestamp(dd[dd >= v0][1]).strftime("%Y-%m-%d"), tes=pd.Timestamp(dd[dd >= t0][1]).strftime("%Y-%m-%d"), majority_acc=float(max(tes.y_up.mean(), 1 - tes.y_up.mean())), n_test=int(len(tes)))
        print(yr, meta[str(yr)], len(feats), "features", flush=True)
    json.dump(meta, open(OUTD / "meta.json", "w"))
# ---------------------------------------------------------------- adv_collect
elif STAGE == "adv_collect":
    def instances(path, tes_date, seq):
        """replicate load.py's instance order (os.listdir order, date-major) for the test block."""
        fn = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]; E = [np.genfromtxt(os.path.join(path, f), delimiter=",") for f in fn]
        dates = np.genfromtxt(os.path.join(path, "..", "trading_dates.csv"), dtype=str, delimiter=","); ti = list(dates).index(tes_date); out = []
        for di in range(ti, len(dates)):
            if di < seq: continue
            for k in range(len(fn)):
                if abs(E[k][di][-2]) > 1e-8 and E[k][di - seq:di, :].min() > -123320: out.append((pd.Timestamp(str(dates[di - 1])), fn[k].replace(".csv", ""), (E[k][di][-2] + 1) / 2))
        return out
    XR = json.load(open(RES / "external_runs.json")); rows = []; summ = {}
    for tag, base in [("11 price features", ADV11), ("our 45 features", ADV45), ("all nine sources, 74 features", ADV74)]:
        meta = json.load(open(base / "meta.json")); pref = {ADV11: "33_adv11", ADV45: "33_adv45", ADV74: "33_adv74"}[base]
        if not (base / "meta.json").exists(): continue
        for lg in sorted(glob.glob(str(RES / f"log_{pref}_*.txt"))):
            m = re.search(rf"log_{pref}_(\d{{4}})_(\w+)\.txt", lg); yr, mdl = m.group(1), m.group(2); txt = open(lg).read()
            b = re.search(r"Best Test performance: \{'acc': ([\d.]+), 'mcc': ([-\d.e]+)\}", txt); v = re.search(r"Best Valid performance: \{'acc': ([\d.]+)", txt)
            if not b: print("no result", lg); continue
            name = f"{mdl} official ({tag})"; summ.setdefault(name, {})[yr] = dict(test_acc=float(b.group(1)), test_mcc=float(b.group(2)), val_acc=float(v.group(1)) if v else None, majority_acc=meta[yr]["majority_acc"], n_test=meta[yr]["n_test"])
            pf = ROOT / "external/Adv-ALSTM/saved_model" / f"{pref}_{yr}_{mdl}" / "model_tes_prediction.csv"
            if pf.exists():
                sc = np.loadtxt(pf); ins = instances(str(base / yr / "ourpped"), meta[yr]["tes"], 5 if mdl == "LSTM" else 15); assert len(sc) == len(ins), (lg, len(sc), len(ins))
                rows.append(pd.DataFrame({"date": [i[0] for i in ins], "stock_id": [i[1] for i in ins], "y_up": [i[2] for i in ins], name: sc}))
                summ[name][yr]["auroc_from_scores"] = float(auc_stat(np.array([i[2] for i in ins]), sc))
    for n in summ: summ[n]["mean"] = {k: float(np.mean([summ[n][str(y)][k] for y in TEST_YEARS if str(y) in summ[n] and summ[n][str(y)].get(k) is not None] or [np.nan])) for k in ("test_acc", "test_mcc", "majority_acc", "auroc_from_scores")}
    byn = {}
    for r in rows: byn.setdefault(r.columns[-1], []).append(r)
    o = None
    for r in [pd.concat(v) for v in byn.values()]: o = r if o is None else o.merge(r, on=KEY + ["y_up"], how="outer")
    o.to_parquet(RES / "oos_sota_adv.parquet")
    XR["adv_alstm_official_tw50_headtohead"] = dict(summ, note="official Adv-ALSTM code (Feng et al. 2019), KDD17 hyper-parameters, walk-forward by year (validation = previous year), label y_up; models saved and re-scored with the code's test action to obtain raw scores for AUROC; 11 'ourpped' price features vs our 45 S1-S5 features")
    json.dump(XR, open(RES / "external_runs.json", "w"), indent=1, ensure_ascii=False); print(json.dumps({n: summ[n]["mean"] for n in summ}, indent=1)); print("saved", o.shape)
# ---------------------------------------------------------------- assemble
elif STAGE == "assemble":
    tab = pd.read_parquet(RES / "oos_tabular.parquet"); seq = pd.read_parquet(RES / "oos_sequence.parquet"); cs = pd.read_parquet(RES / "oos_causalstock_v2_tw50.parquet")
    cs["stock_id"] = cs.stock_id.astype(str); cs["date"] = pd.to_datetime(cs.date)
    base = tab[KEY + ["year", "y_up"]].copy(); base["stock_id"] = base.stock_id.astype(str)
    def mean_cols(d, prefix): return d[[c for c in d.columns if c.startswith(prefix)]].mean(axis=1).values
    P = pd.DataFrame(index=base.index); P["SOX-return rule"] = tab["Rule|SOX return score|0"].values; P["LR Full (S1-S5)"] = tab["LR|S1+S2+S3+S4+S5|0"].values
    P["LightGBM Full (S1-S5)"] = mean_cols(tab, "LightGBM|Full|"); P["XGBoost Full (S1-S5)"] = mean_cols(tab, "XGBoost|Full|")
    sq = base[KEY].merge(seq.assign(stock_id=seq.stock_id.astype(str)), on=KEY, how="left"); P["LSTM Full (S1-S5)"] = mean_cols(sq, "LSTM-W5|Full|"); P["Transformer Full (S1-S5)"] = mean_cols(sq, "Transformer-W5|Full|")
    P["Ensemble LightGBM+LSTM (S1-S5)"] = (P["LightGBM Full (S1-S5)"] + P["LSTM Full (S1-S5)"]) / 2
    for f_, pre, nm in [("oos_sota_lgb.parquet", "LightGBM|All9|", "LightGBM All9 (S1-S9)"), ("oos_sota_lgb.parquet", "LightGBM|Full+S7+S8|", "LightGBM Full+S7+S8"), ("oos_sota_seq.parquet", "LSTM-W5|All9|", "LSTM All9 (S1-S9)"), ("oos_sota_seq.parquet", "Transformer-W5|All9|", "Transformer All9 (S1-S9)")]:
        if (RES / f_).exists():
            d = pd.read_parquet(RES / f_); d["stock_id"] = d.stock_id.astype(str); d = base[KEY].merge(d, on=KEY, how="left")
            if any(c.startswith(pre) for c in d.columns): P[nm] = mean_cols(d, pre)
    if "LightGBM All9 (S1-S9)" in P and "LSTM All9 (S1-S9)" in P: P["Ensemble LightGBM+LSTM All9 (ours, final)"] = (P["LightGBM All9 (S1-S9)"] + P["LSTM All9 (S1-S9)"]) / 2
    d = base[KEY].merge(cs, on=KEY, how="left"); P["CausalStock v2 (price-only, 11)"] = d["CausalStock v2 (price-only, 11)"].values
    for f_ in ["oos_sota_causal.parquet", "oos_sota_causal_tuned.parquet", "oos_sota_adv.parquet"]:
        if (RES / f_).exists():
            d = pd.read_parquet(RES / f_); d["stock_id"] = d.stock_id.astype(str); d["date"] = pd.to_datetime(d.date); d = base[KEY].merge(d, on=KEY, how="left")
            for c in d.columns:
                if c not in KEY + ["y_up"]: P[c] = d[c].values
    SOTA = [c for c in P.columns if c.startswith("CausalStock") or "official" in c]; OURS = [c for c in P.columns if c not in SOTA]
    y = base.y_up.values; dates = base.date.values; bs = DayBootstrap(dates, B=1000); out = {"rows": {}, "n_rows": int(len(base)), "sota_models": SOTA, "our_models": OURS}
    def acc_mcc(yy, pp, thr):
        from sklearn.metrics import matthews_corrcoef; pr = (pp >= thr).astype(int); return float((pr == yy).mean()), float(matthews_corrcoef(yy, pr))
    for c in P.columns:
        p = P[c].values; ok = ~np.isnan(p); thr = 0.0 if "official" in c else 0.5; a, m = acc_mcc(y[ok], p[ok], thr)
        out["rows"][c] = dict(auroc=float(auc_stat(y[ok], p[ok])), ci=bs.ci(auc_stat, y, np.where(ok, p, np.nan)) if ok.all() else DayBootstrap(dates[ok], B=1000).ci(auc_stat, y[ok], p[ok]), acc=a, mcc=m, n=int(ok.sum()), group="SOTA architecture" if c in SOTA else "ours",
                               by_year={str(yr): float(auc_stat(y[ok & (base.year.values == yr)], p[ok & (base.year.values == yr)])) for yr in TEST_YEARS})
        print(f"{c:48s} auroc {out['rows'][c]['auroc']:.4f} acc {a:.4f} mcc {m:.4f} n {ok.sum()}", flush=True)
    FINAL = "Ensemble LightGBM+LSTM All9 (ours, final)" if "Ensemble LightGBM+LSTM All9 (ours, final)" in P else "LightGBM Full (S1-S5)"; out["final_model"] = FINAL; out["paired"] = {}
    for ref in ["LightGBM Full (S1-S5)", FINAL]:
        for c in SOTA + ["LightGBM Full (S1-S5)", "SOX-return rule"]:
            if c == ref: continue
            ok = ~np.isnan(P[c].values) & ~np.isnan(P[ref].values); b2 = bs if ok.all() else DayBootstrap(dates[ok], B=1000)
            ci = b2.ci_diff(auc_stat, y[ok], P[c].values[ok], P[ref].values[ok]); same = sum(np.sign(out["rows"][ref]["by_year"][str(yr)] - out["rows"][c]["by_year"][str(yr)]) == np.sign(ci[0] + ci[1]) for yr in TEST_YEARS)
            out["paired"][f"{ref} minus {c}"] = dict(diff=float(out["rows"][ref]["auroc"] - out["rows"][c]["auroc"]), ci=ci[:2], p_boot=ci[2], same_sign_years=int(same), n=int(ok.sum()))
            print(f"{ref} - {c}: {out['paired'][f'{ref} minus {c}']}", flush=True)
    # architecture vs information: same architecture, 11 -> 45 -> All9 inputs
    out["information_vs_architecture"] = {k: out["rows"][k]["auroc"] for k in P.columns if k.startswith("CausalStock") or "official" in k}
    dump(out, "sota_headtohead.json"); P.assign(**{k: base[k].values for k in KEY + ["year", "y_up"]}).to_parquet(RES / "oos_sota_headtohead.parquet")
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = sorted(out["rows"].items(), key=lambda kv: kv[1]["auroc"]); names = [k for k, _ in rows]; v = [r["auroc"] for _, r in rows]; lo = [r["auroc"] - r["ci"][0] for _, r in rows]; hi = [r["ci"][1] - r["auroc"] for _, r in rows]
    col = ["#eb6834" if r["group"] != "ours" else ("#4a3aa7" if "final" in k or "Ensemble" in k else "#9b8fd6") for k, r in rows]
    fig, ax = plt.subplots(figsize=(13.56, 4.5)); ax.barh(names, v, xerr=[lo, hi], color=col, error_kw=dict(lw=0.8, ecolor="#555"), height=0.62); ax.tick_params(axis="y", labelsize=7.2)
    for i, (k, r) in enumerate(rows): ax.text(r["ci"][1] + 0.003, i, f"{r['auroc']:.3f}   acc {r['acc']:.3f}  MCC {r['mcc']:.3f}", va="center", fontsize=6.8)
    ax.axvline(0.5, color="#999", lw=0.8); ax.set_xlim(0.47, 0.68); ax.set_xlabel("AUROC, pooled 2023-2026 (95 % trading-day block-bootstrap CI)"); ax.grid(axis="x", color="#e4e3dc"); ax.grid(False, axis="y")
    from matplotlib.patches import Patch; ax.legend(handles=[Patch(color="#eb6834", label="published SOTA architecture (price-only inputs, or given our inputs)"), Patch(color="#9b8fd6", label="our models"), Patch(color="#4a3aa7", label="our final model")], loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Same data, same walk-forward split, same label: published architectures vs our multi-source models (Taiwan 50)", fontsize=10.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(ROOT / "figures/final/f20_sota_headtohead.png", dpi=200, bbox_inches="tight"); print("figure saved")
print("done", STAGE)
