"""Fresh hold-out: data that neither the models nor the authors had seen when the pipeline was fixed (feature days after 2026-09-23).
The study's test period (2023 -> 2026-09-23) had been inspected by the authors during the proposal / progress stages (Limitation 1).
This script freezes the pipeline (features, hyper-parameters, fold policy of common.py) and applies ONE more walk-forward step.

PLAN, written before any new outcome was looked at (2026-11-13):
  data      FinMind mirror + Yahoo Finance, fetched for 2026-09-01 -> 2026-11-13 and appended to the frozen raw files (old rows are kept as they are)
  check     the rebuilt table must reproduce the frozen table on every feature day <= 2026-09-23
  training  all rows whose label is realised before the first new feature day; inner validation = last 12 months (early stopping, threshold); refit; 1-day purge
  models    LightGBM Full / S4-only / TW-only (seeds 1-4, seed-mean probability), LR Full, SOX-return rule, per-stock base rate; extension: LightGBM Full+S7+S8 if the S7/S8 inputs can be rebuilt
  primary   AUROC of LightGBM Full on y_up; paired Full - S4-only, Full - TW-only, Full - SOX rule (5-day-block trading-day bootstrap, B = 1000)
  secondary the same arms on y_gap / y_intra / y_rel; accuracy and MCC at the inner-validation threshold
  reading   'consistent' = the hold-out CI contains the 2023-2026 walk-forward value. With ~35 trading days the CI is wide; nothing is re-tuned afterwards.
Stages: fetch | build | score.  Output: data/holdout/, results/final/fresh_holdout.json"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from common import *
STAGE = sys.argv[1]; HO = ROOT / "data/holdout"; NEWRAW = HO / "new"; RAWH = HO / "raw"; PROC = HO / "processed"
F_START, F_END = "2026-09-01", "2026-11-13"; CUT = pd.Timestamp("2026-09-24")   # frozen raw files end on 2026-09-24; frozen model table ends on feature day 2026-09-23
DS = ["TaiwanStockPrice", "TaiwanStockInstitutionalInvestorsBuySell", "TaiwanStockMarginPurchaseShortSale"]
if STAGE == "fetch":
    import requests, yfinance as yf
    from universe import TAIWAN50
    NEWRAW.mkdir(parents=True, exist_ok=True); API = "https://api.finmindtrade.com/api/v4/data"
    def get(dataset, data_id):
        for i in range(4):
            r = requests.get(API, params=dict(dataset=dataset, data_id=data_id, start_date=F_START, end_date=F_END), timeout=60)
            if r.status_code == 200 and r.json().get("status") == 200: return pd.DataFrame(r.json()["data"])
            print("retry", dataset, data_id, r.status_code, r.text[:100], flush=True); time.sleep(20 * (i + 1))
        raise RuntimeError(f"failed {dataset} {data_id}")
    for ds in DS:
        if (NEWRAW / f"{ds}.parquet").exists(): continue
        fr = []
        for sid in TAIWAN50: fr.append(get(ds, sid)); time.sleep(0.5)
        pd.concat(fr, ignore_index=True).to_parquet(NEWRAW / f"{ds}.parquet"); print(ds, sum(len(x) for x in fr), flush=True)
    if not (NEWRAW / "TAIEX.parquet").exists(): get("TaiwanStockPrice", "TAIEX").to_parquet(NEWRAW / "TAIEX.parquet")
    T = {"^GSPC": "SP500", "^IXIC": "NASDAQ", "^SOX": "SOX", "^VIX": "VIX", "TSM": "TSM_ADR", "TWD=X": "USDTWD", "DX-Y.NYB": "DXY", "^TNX": "US10Y", "^N225": "NIKKEI"}
    raw = yf.download(list(T), start="2026-08-01", end="2026-11-14", auto_adjust=False, progress=False, group_by="ticker"); close = pd.concat({T[t]: raw[t]["Close"] for t in T}, axis=1); close.index.name = "date"
    close.to_parquet(NEWRAW / "us_close.parquet"); print(close.tail(3)); json.dump(dict(fetched_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"), start=F_START, end=F_END), open(NEWRAW / "fetch_info.json", "w")); print("done fetch")
elif STAGE == "build":
    (RAWH / "finmind").mkdir(parents=True, exist_ok=True); (RAWH / "us").mkdir(parents=True, exist_ok=True); PROC.mkdir(parents=True, exist_ok=True); rep = {}
    for ds in DS + ["TAIEX"]:
        old = pd.read_parquet(ROOT / "data/raw/finmind" / f"{ds}.parquet"); new = pd.read_parquet(NEWRAW / f"{ds}.parquet"); last = pd.to_datetime(old["date"]).max()
        ov = new[pd.to_datetime(new["date"]) <= last]; add = new[pd.to_datetime(new["date"]) > last]
        if ds in ("TaiwanStockPrice", "TAIEX"):   # do the re-fetched overlapping rows agree with the frozen ones?
            k = ["stock_id", "date"] if "stock_id" in old else ["date"]; mm = old.merge(ov, on=k, suffixes=("", "_new")); rep[ds] = dict(overlap_rows=int(len(mm)), close_identical=float((mm.close == mm.close_new).mean()))
        pd.concat([old, add[old.columns.intersection(add.columns)]], ignore_index=True).to_parquet(RAWH / "finmind" / f"{ds}.parquet"); rep.setdefault(ds, {})["rows_added"] = int(len(add))
    old = pd.read_parquet(ROOT / "data/raw/us/us_close.parquet"); new = pd.read_parquet(NEWRAW / "us_close.parquet"); old.index = pd.to_datetime(old.index); new.index = pd.to_datetime(new.index)
    add = new[new.index > old.index.max()][old.columns]; pd.concat([old, add]).to_parquet(RAWH / "us" / "us_close.parquet"); rep["us"] = dict(rows_added=int(len(add)), last=str(add.index.max().date()))
    src = open(ROOT / "scripts/build_dataset.py").read(); a = 'RAW = ROOT / "data" / "raw"; OUT = ROOT / "data" / "processed"; OUT.mkdir(exist_ok=True)'; assert a in src
    exec(src.replace(a, 'RAW = ROOT / "data" / "holdout" / "raw"; OUT = ROOT / "data" / "holdout" / "processed"'), {"__name__": "bd", "__file__": str(ROOT / "scripts/build_dataset.py")})
    new = pd.read_parquet(PROC / "dataset.parquet"); new["date"] = pd.to_datetime(new.date); frozen = load(); K = ["date", "stock_id"]
    m = frozen[K + ALL + ["y_up"]].merge(new[K + ALL + ["y_up"]], on=K, how="left", suffixes=("", "_h")); rep["reproduces_frozen_table"] = dict(frozen_rows=int(len(frozen)), matched=int(m["y_up_h"].notna().sum()),
        max_abs_feature_diff=float(max(np.nanmax(np.abs(m[c].astype(float) - m[c + "_h"].astype(float))) for c in ALL)), labels_identical=float((m.y_up == m.y_up_h).mean()))
    nr = new[new.date > frozen.date.max()]; rep["new_rows"] = dict(n=int(len(nr)), days=int(nr.date.nunique()), first=str(nr.date.min().date()), last=str(nr.date.max().date()), up_rate=float(nr.y_up.mean()))
    json.dump(rep, open(HO / "build_report.json", "w"), indent=1); print(json.dumps(rep, indent=1)); print("done build")
elif STAGE == "extra":   # S7 (from our own prices) and S8 (TX futures, positioning, total margin) for the hold-out table: the SAME code as 18_extra_features.py, pointed at the hold-out files
    import requests
    API = "https://api.finmindtrade.com/api/v4/data"; XR_ = HO / "raw/finmind_extra"; XR_.mkdir(parents=True, exist_ok=True)
    for ds, did in [("TaiwanFuturesDaily", "TX"), ("TaiwanFuturesInstitutionalInvestors", "TX"), ("TaiwanStockTotalMarginPurchaseShortSale", None)]:
        prm = dict(dataset=ds, start_date=F_START, end_date=F_END); prm.update(data_id=did) if did else None
        r = requests.get(API, params=prm, timeout=90); assert r.status_code == 200 and r.json().get("status") == 200, r.text[:200]; new = pd.DataFrame(r.json()["data"]); old = pd.read_parquet(ROOT / "data/raw/finmind_extra" / f"{ds}.parquet")
        add = new[pd.to_datetime(new["date"]) > pd.to_datetime(old["date"]).max()]; pd.concat([old, add[old.columns.intersection(add.columns)]], ignore_index=True).to_parquet(XR_ / f"{ds}.parquet"); print(ds, "rows added", len(add), flush=True); time.sleep(1)
    h = pd.read_parquet(PROC / "dataset.parquet"); h["date"] = pd.to_datetime(h.date); sec = load().drop_duplicates("stock_id").set_index("stock_id").sector_en; h["sector_en"] = h.stock_id.map(sec); h = h.sort_values(["date", "stock_id"]).reset_index(drop=True)
    src = open(ROOT / "scripts/final/18_extra_features.py").read(); i0 = src.index("# ---- S7 cross-stock ----"); i1 = src.index("# ---- S9 stock-level flows ----"); body = src[i0:i1]
    a_ = 'ROOT / "data/raw/finmind/TAIEX.parquet"'; assert a_ in body; body = body.replace(a_, 'HO / "raw/finmind/TAIEX.parquet"')
    g_ = dict(np=np, pd=pd, ROOT=ROOT, HO=HO, R=XR_, df=h[["date", "stock_id", "sector_en", "ret1", "ret5", "volume", "close"]].copy()); exec(body, g_); o = g_["df"][["date", "stock_id"] + g_["S7"] + g_["S8"]]
    fz = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); fz["date"] = pd.to_datetime(fz.date); m = fz.merge(o, on=["date", "stock_id"], suffixes=("", "_h"))
    chk = float(max(np.nanmax(np.abs(m[c].astype(float) - m[c + "_h"].astype(float))) for c in g_["S7"] + g_["S8"])); print("frozen rows matched", len(m), "of", len(fz), "| max abs diff of S7/S8 vs the frozen extra features", chk)
    o.to_parquet(HO / "extra_features.parquet"); rep = json.load(open(HO / "build_report.json")); rep["extra_S7_S8"] = dict(matched=int(len(m)), max_abs_diff_vs_frozen=chk, missing_share_new_rows={c: float(o.loc[o.date > fz.date.max(), c].isna().mean()) for c in g_["S8"][:3]}); json.dump(rep, open(HO / "build_report.json", "w"), indent=1); print("done extra")
elif STAGE == "score":
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    df = pd.read_parquet(PROC / "dataset.parquet"); df["date"] = pd.to_datetime(df.date); df["date_next"] = pd.to_datetime(df.date_next); full = pd.read_parquet(PROC / "dataset_full.parquet").sort_values(["stock_id", "date"]); full["date"] = pd.to_datetime(full.date)
    full["open_next"] = full.groupby("stock_id")["open"].shift(-1); df = df.merge(full[["stock_id", "date", "open_next"]], on=["stock_id", "date"], how="left")
    tx = pd.read_parquet(RAWH / "finmind/TAIEX.parquet"); tx["date"] = pd.to_datetime(tx.date); tx = tx.sort_values("date"); tx["taiex_ret_next"] = tx.close.shift(-1) / tx.close - 1; df = df.merge(tx[["date", "taiex_ret_next"]], on="date", how="left")
    df["y_gap"] = (df.open_next > df.close).astype(float); df["y_intra"] = (df.close_next > df.open_next).astype(float); df["y_rel"] = (df.ret_next > df.taiex_ret_next).astype(float)   # same definitions as 01_prepare.py
    df = df.sort_values(["date", "stock_id"]).reset_index(drop=True); frozen_last = load().date.max(); te = df.index[df.date > frozen_last]; t0 = df.loc[te, "date"].min(); v0 = t0 - pd.DateOffset(months=12)
    tr = df.index[(df.date < t0) & (df.date_next < t0)]; iva = df.index[(df.date >= v0) & (df.date < t0) & (df.date_next < t0)]; itr = df.index[(df.date < v0) & (df.date_next < v0)]   # common.folds policy, one more step
    FEATS = {"Full": ALL, "S4-only": SRC["S4"], "TW-only": SRC["S1"] + SRC["S2"] + SRC["S3"]}; EXT = None
    if (HO / "extra_features.parquet").exists():
        ex = pd.read_parquet(HO / "extra_features.parquet"); ex["date"] = pd.to_datetime(ex.date); S78 = [c for c in ex.columns if c not in ("date", "stock_id")]; m = df[["date", "stock_id"]].merge(ex, on=["date", "stock_id"], how="left")
        for c in S78: df[c] = m[c].values
        FEATS["Full+S7+S8"] = ALL + S78; EXT = S78
    def lgbm(feats, label, seed):
        P = dict(LGB_PARAMS, n_jobs=8); m = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **P).fit(df.loc[itr, feats], df.loc[itr, label], eval_set=[(df.loc[iva, feats], df.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        n = int(m.best_iteration_ or 1); pv = m.predict_proba(df.loc[iva, feats])[:, 1]
        return lgb.LGBMClassifier(n_estimators=n, random_state=seed, **P).fit(df.loc[tr, feats], df.loc[tr, label]).predict_proba(df.loc[te, feats])[:, 1], pv, n
    out = dict(plan=__doc__, build=json.load(open(HO / "build_report.json")), fetch=json.load(open(NEWRAW / "fetch_info.json")), train_rows=int(len(tr)), first_new_feature_day=str(t0.date()), last_new_feature_day=str(df.loc[te, "date"].max().date()),
               n_rows=int(len(te)), n_days=int(df.loc[te, "date"].nunique()), extension_features=EXT, labels={})
    WF = json.load(open(RES / "main_results.json")); LBJ = json.load(open(RES / "labels.json"))
    for label in ["y_up", "y_gap", "y_intra", "y_rel"]:
        y = df.loc[te, label].values; bs = DayBootstrap(df.loc[te, "date"].values, B=1000); pr = {}; thr = {}; R = dict(up_rate=float(y.mean()), arms={})
        for arm, feats in FEATS.items():
            runs = [lgbm(feats, label, s) for s in (1, 2, 3, 4)]; pr[arm] = np.mean([r[0] for r in runs], 0); thr[arm] = best_threshold(df.loc[iva, label].values, np.mean([r[1] for r in runs], 0)); R["arms"][f"LightGBM|{arm}"] = dict(trees=[r[2] for r in runs])
        if label == "y_up":
            mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)); pr["LR Full"] = mk().fit(df.loc[tr, ALL], df.loc[tr, label]).predict_proba(df.loc[te, ALL])[:, 1]
            thr["LR Full"] = best_threshold(df.loc[iva, label].values, mk().fit(df.loc[itr, ALL], df.loc[itr, label]).predict_proba(df.loc[iva, ALL])[:, 1])
        pr["SOX-return rule"] = df.loc[te, "us_sox_ret"].fillna(0).values; pr["stock base-rate"] = df.loc[te, "stock_id"].map(df.loc[tr].groupby("stock_id")[label].mean()).values
        for k, p in pr.items():
            d = R["arms"].setdefault(k if k in ("LR Full", "SOX-return rule", "stock base-rate") else f"LightGBM|{k}", {}); d.update(auroc=float(auc_stat(y, p)), ci=bs.ci(auc_stat, y, p)[:2])
            if k in thr: d.update(thr=float(thr[k]), acc=float(accuracy_score(y, p >= thr[k])), mcc=float(matthews_corrcoef(y, p >= thr[k])))
        R["majority_acc"] = float(max(y.mean(), 1 - y.mean()))
        dd = lambda a, b: (lambda ci: dict(diff=float(auc_stat(y, pr[b]) - auc_stat(y, pr[a])), ci=ci[:2], p_boot=ci[2]))(bs.ci_diff(auc_stat, y, pr[a], pr[b]))
        R["paired"] = {"Full minus S4-only": dd("S4-only", "Full"), "Full minus TW-only": dd("TW-only", "Full"), "Full minus SOX-return rule": dd("SOX-return rule", "Full")}
        if "Full+S7+S8" in pr: R["paired"]["Full+S7+S8 minus Full"] = dd("Full", "Full+S7+S8"); R["paired"]["Full+S7+S8 minus S4-only"] = dd("S4-only", "Full+S7+S8")
        wf = LBJ["binary_variants"][label] if label != "y_up" else None
        R["walk_forward_2023_2026"] = ({a: float(wf[a]["auroc"]) for a in ("Full", "S4-only", "TW-only")} if wf else {a: float(WF["arms"][f"LightGBM|{a}"]["seed_mean_prob"]["pooled"]["auroc"]) for a in ("Full", "S4-only", "TW-only")})
        R["consistent_with_walk_forward"] = {a: bool(R["arms"][f"LightGBM|{a}"]["ci"][0] <= R["walk_forward_2023_2026"][a] <= R["arms"][f"LightGBM|{a}"]["ci"][1]) for a in ("Full", "S4-only", "TW-only")}
        out["labels"][label] = R; print(label, {k: (round(v["auroc"], 3), [round(x, 3) for x in v["ci"]]) for k, v in R["arms"].items()}, {k: (round(v["diff"], 3), [round(x, 3) for x in v["ci"]]) for k, v in R["paired"].items()}, flush=True)
        if label == "y_up": pd.DataFrame(dict(date=df.loc[te, "date"].values, stock_id=df.loc[te, "stock_id"].values, y_up=y, **{k: v for k, v in pr.items()})).to_parquet(RES / "oos_fresh_holdout.parquet")
    dump(out, "fresh_holdout.json"); print("done score")
