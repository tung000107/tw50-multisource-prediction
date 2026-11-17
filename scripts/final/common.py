"""Shared utilities for the final study. Every number in the final deck is produced by scripts in this folder.
Protocol (identical for all models / arms):
  * expanding-window walk-forward, test years 2023, 2024, 2025, 2026 (to 2026-09-23)
  * train = all rows with date_next < first test date (1-trading-day purge: label horizon = 1 day)
  * inner validation = last 12 months of the training window (purged); used ONLY for early stopping,
    decision threshold (argmax MCC) and any hyper-parameter choice; the test year is never touched
  * imputation / scaling fitted on the training window only; LightGBM / XGBoost use native missing handling
  * inference: trading-day moving-block bootstrap (5-day blocks, B=1000) for every CI; paired on the same resamples
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
                             average_precision_score, brier_score_loss, log_loss)
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "final"; FIG = ROOT / "figures" / "final"; RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "processed" / "dataset_final.parquet"

SRC = {
 "S1": ["ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ma5_ratio", "ma20_ratio", "ma60_ratio", "vol20", "volume_ratio20", "hl_range", "gap_open", "rsi14", "spread"],
 "S2": ["foreign_net_ratio", "foreign_net_ratio5", "trust_net_ratio", "trust_net_ratio5", "dealer_net_ratio", "dealer_net_ratio5", "inst_net_ratio", "margin_chg", "margin_chg5", "short_chg"],
 "S3": ["taiex_ret1", "taiex_ret5", "taiex_ma20_ratio", "taiex_vol_ratio20", "taiex_vol20", "rel_ret1"],
 "S4": ["us_sp500_ret", "us_nasdaq_ret", "us_sox_ret", "us_tsm_adr_ret", "us_dxy_ret", "us_nikkei_ret", "us_vix_level", "us_vix_chg", "us_usdtwd_chg", "us_10y_chg", "us_stale"],
 "S5": ["weekday", "month", "is_month_end"]}
SRC_NAME = {"S1": "price / technical", "S2": "institutional flows & margin", "S3": "Taiwan market-wide", "S4": "overnight U.S. / global", "S5": "calendar"}
ALL = sum(SRC.values(), []); TW = SRC["S1"] + SRC["S2"] + SRC["S3"]
TEST_YEARS = [2023, 2024, 2025, 2026]
ARMS = {"Full": ALL, "S4-only": SRC["S4"], "TW-only": TW, "TW+S5": TW + SRC["S5"], **{f"{k}-only": v for k, v in SRC.items()},
        **{f"Full-{k}": [c for c in ALL if c not in SRC[k]] for k in SRC}, **{f"S4+{k}": SRC["S4"] + SRC[k] for k in ["S1", "S2", "S3"]}}
LGB_PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=4)

def load():
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"]); df["date_next"] = pd.to_datetime(df["date_next"])
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)

def folds(df, test_years=TEST_YEARS, inner_months=12):
    """yield (test_year, idx_train, idx_inner_train, idx_inner_val, idx_test); all purged by label horizon."""
    for yr in test_years:
        t0 = df.loc[df.year == yr, "date"].min()
        test = df.index[df.year == yr]
        train = df.index[(df.date < t0) & (df.date_next < t0)]                       # purge: label known before first test prediction
        v0 = t0 - pd.DateOffset(months=inner_months)
        inner_val = df.index[(df.date >= v0) & (df.date < t0) & (df.date_next < t0)]
        inner_train = df.index[(df.date < v0) & (df.date_next < v0)]
        yield yr, train, inner_train, inner_val, test

def metrics(y, prob, thr=0.5):
    y = np.asarray(y); prob = np.asarray(prob); p = (prob >= thr).astype(int)
    out = dict(n=int(len(y)), auroc=roc_auc_score(y, prob), pr_auc=average_precision_score(y, prob), acc=accuracy_score(y, p),
               bal_acc=balanced_accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"), mcc=matthews_corrcoef(y, p),
               brier=brier_score_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)), logloss=log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)), thr=float(thr))
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}

def best_threshold(y, prob):
    cands = np.quantile(prob, np.linspace(0.2, 0.8, 61)); best, bt = -9, 0.5
    for t in cands:
        m = matthews_corrcoef(y, (prob >= t).astype(int))
        if m > best: best, bt = m, float(t)
    return bt

# ---------- inference: trading-day moving-block bootstrap ----------
class DayBootstrap:
    """Resample whole trading days in moving blocks of `block` consecutive days (handles cross-sectional and short-range serial dependence).
    The same resample indices are reused for every statistic -> paired comparisons."""
    def __init__(self, dates, B=1000, block=5, seed=2026):
        self.dates = pd.to_datetime(pd.Series(dates)).values; self.u = np.sort(np.unique(self.dates))
        self.groups = pd.Series(np.arange(len(self.dates))).groupby(self.dates).indices
        rng = np.random.default_rng(seed); n = len(self.u); nb = int(np.ceil(n / block))
        self.samples = []
        for _ in range(B):
            starts = rng.integers(0, n - block + 1, nb); days = np.concatenate([self.u[s:s + block] for s in starts])[:n]
            self.samples.append(np.concatenate([self.groups[d] for d in days]))
    def ci(self, stat, *arrays, alpha=0.05):
        v = np.array([stat(*[a[j] for a in arrays]) for j in self.samples])
        return [float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))]
    def ci_diff(self, stat, y, pa, pb, alpha=0.05):
        v = np.array([stat(y[j], pb[j]) - stat(y[j], pa[j]) for j in self.samples])
        return [float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))), float((v <= 0).mean())]

def auc_stat(y, p):
    try: return roc_auc_score(y, p)
    except ValueError: return np.nan

# ---------- DeLong test for paired AUROCs ----------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out
def delong(y, p1, p2):
    """returns (auc1, auc2, z, p-value two-sided) — DeLong, DeLong & Clarke-Pearson (1988) via Sun & Xu (2014) fast algorithm."""
    y = np.asarray(y).astype(int); order = np.argsort(-y); y = y[order]; P = np.vstack([p1, p2])[:, order]
    m = int(y.sum()); n = len(y) - m; k = P.shape[0]
    tx = np.array([_midrank(P[r, :m]) for r in range(k)]); ty = np.array([_midrank(P[r, m:]) for r in range(k)]); tz = np.array([_midrank(P[r, :]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n; d = np.array([1, -1]); var = d @ S @ d
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else np.nan
    from scipy.stats import norm
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan

def dump(obj, name):
    (RES / name).write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    print("saved", RES / name)


def news_feature_day(ts, days):
    """Feature day of a headline stamped `ts` (Taiwan time). The decision for feature day i is taken at 08:30 on the next trading day, so the headline belongs to
    the first trading day i whose decision time is >= ts: window (08:30 on day i, 08:30 on day i+1]. (Rule fixed 2026-11-13: the first version cut at the 09:00 open,
    which let headlines stamped 08:30-09:00 into the row, and pushed daytime headlines of weekends / holidays one row late.) Returns positions into `days`; -1 = after the last decision."""
    days = pd.to_datetime(np.asarray(days)); dec = np.append(days[1:].values, (days[-1] + pd.Timedelta(days=1)).to_datetime64()) + np.timedelta64(8 * 60 + 30, "m")
    pos = np.searchsorted(dec, np.asarray(ts, dtype="datetime64[ns]"), side="left"); return np.where(pos < len(days), pos, -1)
