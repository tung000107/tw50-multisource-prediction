"""Run the OFFICIAL Adv-ALSTM code (Feng et al., IJCAI 2019; external/Adv-ALSTM, security-audited) on the Taiwan 50 under our walk-forward protocol.
'gen': write data/raw/tw50_advalstm/<year>/ourpped/<stock>.csv (11 features as in the KDD17 'ourpped' files: c_open, c_high, c_low, n_close, n_adj,
5/10/15/20/25/30-day MA ratios, z-scored on the training window; column 12 = label +1 up / -1 non-up (our y_up, written one row later so that the loader's window ends on the feature day); column 13 = close; padding -123321)
and trading_dates.csv truncated at the end of the test year. Train = before (year-1)-01-01, validation = year-1, test = year (their loader: tra/val/tes dates).
'collect': parse the logs into results/final/external_runs.json['adv_alstm_official_tw50']."""
import sys, json, re, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import ROOT, RES, TEST_YEARS
OUT = ROOT / "data/raw/tw50_advalstm"
if sys.argv[1] == "gen":
    df = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet").sort_values(["stock_id", "date"]); df["date"] = pd.to_datetime(df["date"]); g = df.groupby("stock_id")
    f = pd.DataFrame({"c_open": df.open / df.close - 1, "c_high": df.high / df.close - 1, "c_low": df.low / df.close - 1, "n_close": df.close / g.close.shift(1) - 1, "n_adj": df.close / g.close.shift(1) - 1})
    for k in (5, 10, 15, 20, 25, 30): f[f"ma{k}"] = g.close.transform(lambda s: s.rolling(k).mean()) / df.close - 1
    f["date"] = df.date.values; f["stock_id"] = df.stock_id.values; f["y"] = df.y_up.values; f["close"] = df.close.values; feats = [c for c in f.columns if c not in ("date", "stock_id", "y", "close")]
    days = np.sort(f.date.unique()); meta = {}
    for yr in TEST_YEARS:
        t0 = pd.Timestamp(f"{yr}-01-01"); v0 = pd.Timestamp(f"{yr - 1}-01-01"); t1 = pd.Timestamp(f"{yr + 1}-01-01"); dd = days[days < t1]; dd = np.concatenate([dd, days[days >= t1][:1]]); trm = f.date < v0   # + the first day of the next year: it carries the label of the last test row (see ALIGNMENT)
        mu = f.loc[trm, feats].mean(); sd = f.loc[trm, feats].std().replace(0, 1); d = OUT / str(yr) / "ourpped"; d.mkdir(parents=True, exist_ok=True)
        for sid, gg in f.groupby("stock_id"):
            gg = gg.set_index("date").reindex(dd); X = ((gg[feats] - mu) / sd).clip(-10, 10); ys = gg.y.shift(1); lab = np.where(ys == 1, 1.0, np.where(ys == 0, -1.0, 0.0)); arr = np.column_stack([X.values, lab, gg.close.values]); bad = np.isnan(arr).any(1); arr[bad] = -123321.0; arr[:, -2] = lab
            # ALIGNMENT: the official loader predicts label[d] from rows d-seq..d-1 (their label on day d is Close(d) > Close(d-1)); our y_up on row t is Close(t+1) > Close(t),
            # so the label is written one row later (row t+1 carries y_up[t]) and the window then ends ON the feature day t. Label 0 = no instance.
            np.savetxt(d / f"{sid}.csv", arr, delimiter=",", fmt="%.6f")
        pd.Series([pd.Timestamp(x).strftime("%Y-%m-%d") for x in dd]).to_csv(OUT / str(yr) / "trading_dates.csv", index=False, header=False)
        tes = f[(f.date >= t0) & (f.date < t1) & f.y.notna()]; meta[str(yr)] = dict(tra=pd.Timestamp(dd[0]).strftime("%Y-%m-%d"), val=pd.Timestamp(dd[dd >= v0][1]).strftime("%Y-%m-%d"), tes=pd.Timestamp(dd[dd >= t0][1]).strftime("%Y-%m-%d"), majority_acc=float(max(tes.y.mean(), 1 - tes.y.mean())), n_test=int(len(tes)))
        print(yr, meta[str(yr)], flush=True)
    json.dump(meta, open(OUT / "meta.json", "w"))
else:
    meta = json.load(open(OUT / "meta.json")); res = {}
    for lg in sorted(glob.glob(str(RES / "log_28_adv_*.txt"))):
        m = re.search(r"log_28_adv_(\d{4})_(\w+)\.txt", lg); txt = open(lg).read(); b = re.search(r"Best Test performance: \{'acc': ([\d.]+), 'mcc': ([-\d.e]+)\}", txt); v = re.search(r"Best Valid performance: \{'acc': ([\d.]+)", txt)
        if b: res.setdefault(m.group(2), {})[m.group(1)] = dict(test_acc=float(b.group(1)), test_mcc=float(b.group(2)), val_acc=float(v.group(1)) if v else None, majority_acc=meta[m.group(1)]["majority_acc"], n_test=meta[m.group(1)]["n_test"])
    for mdl in res: res[mdl]["mean"] = {k: float(np.mean([res[mdl][str(y)][k] for y in TEST_YEARS if str(y) in res[mdl]] or [np.nan])) for k in ("test_acc", "test_mcc", "majority_acc")}
    XR = json.load(open(RES / "external_runs.json")); XR["adv_alstm_official_tw50"] = dict(res, note="official code, KDD17 hyper-parameters, walk-forward by year (validation = previous year), label = our y_up (+1 up / -1 non-up), 11 'ourpped' features"); json.dump(XR, open(RES / "external_runs.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(res, indent=1))
