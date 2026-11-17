"""Re-verify the Background correlation numbers (p3 / p4) with returns instead of index levels.

Alignment (identical to the modelling pipeline, scripts/build_dataset.py): row date t carries the latest U.S. session
dated <= t (merge_asof backward), which closes ~05:00 Taiwan time on t+1, i.e. before the TWSE open on t+1.
The overnight pair for TWSE day d = t+1 is therefore (us_sox_ret on row t, r_TW(d)).
r_TW(d) = ln(TAIEX_close(d) / TAIEX_close(previous TWSE trading day)).
Stale rows: the same U.S. session reused on consecutive TWSE rows (U.S. holiday) are excluded from all statistics.
Outputs: figures/bg_level.png, figures/bg_rollcorr.png, figures/bg_market_linkage.png, figures/bg_corr_check.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figures"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 11})
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781"
W = 252

# ---- S4 features as aligned by the pipeline (one row per TWSE date) ----
full = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet", columns=["date", "us_date", "us_sox_ret", "us_sp500_ret", "us_stale"])
s4 = full.drop_duplicates("date").sort_values("date").set_index("date")
assert full.groupby("date")[["us_date", "us_sox_ret"]].nunique().max().max() == 1, "S4 must be identical across stocks on a date"
# ---- TAIEX log returns from the same source as the original figure ----
tx = pd.read_parquet(ROOT / "data/raw/finmind/TAIEX.parquet"); tx["date"] = pd.to_datetime(tx.date); tx = tx.sort_values("date").set_index("date").close
r_tw = np.log(tx / tx.shift(1)).rename("r_tw")
d = pd.DataFrame({"r_tw": r_tw}).join(s4, how="inner")
d["r_tw_next"] = d.r_tw.shift(-1)                      # r_TW(d) for the next TWSE trading day d = t+1
d["sox_reused"] = d.us_date.eq(d.us_date.shift(1))      # same U.S. session as the previous TWSE row -> stale
start, end = d.index.min().date(), d.index.max().date()
n_days = int(d.r_tw_next.notna().sum())
m_flag = int(d.us_stale.sum()); m_reused = int(d.sox_reused.sum()); m_either = int((d.us_stale.eq(1) | d.sox_reused).sum())
keep = d[(~d.sox_reused) & d.us_stale.eq(0)].dropna(subset=["r_tw_next", "us_sox_ret", "r_tw"])

def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))
def stats(x, y):
    m = x.notna() & y.notna(); n = int(m.sum()); r = float(pearsonr(x[m], y[m])[0]); lo, hi = fisher_ci(r, n); rho = float(spearmanr(x[m], y[m])[0])
    return dict(pearson=round(r, 4), ci95=[round(lo, 4), round(hi, 4)], n=n, spearman=round(rho, 4))
sox = stats(keep.us_sox_ret, keep.r_tw_next); own = stats(keep.r_tw, keep.r_tw_next); sp = stats(keep.us_sp500_ret, keep.r_tw_next)
roll_sox = keep.r_tw_next.rolling(W).corr(keep.us_sox_ret); roll_sp = keep.r_tw_next.rolling(W).corr(keep.us_sp500_ret); roll_own = keep.r_tw_next.rolling(W).corr(keep.r_tw)
rng = lambda s: [round(float(s.min()), 2), round(float(s.max()), 2)]
# for the explanation: the original figure's method (250-day window, simple returns, stale rows kept)
d2 = d.dropna(subset=["r_tw_next", "us_sox_ret"]); tx_s = tx.pct_change(); nx = tx_s.shift(-1).reindex(d2.index)
old_sox = nx.rolling(250).corr(d2.us_sox_ret); old_own = nx.rolling(250).corr(tx_s.reindex(d2.index))
out = {"period": [str(start), str(end)], "n_tw_days_with_next": n_days, "excluded_us_stale_flag": m_flag, "excluded_reused_us_session": m_reused, "excluded_total": m_either,
       "n_pairs_used": len(keep), "full_corr_sox_t_taiex_t1": sox, "full_corr_sp500_t_taiex_t1": sp, "full_corr_taiex_t_taiex_t1": own,
       "rolling252_sox": rng(roll_sox), "rolling252_sp500": rng(roll_sp), "rolling252_own": rng(roll_own),
       "original_method_rolling250_simple_returns_stale_kept": {"sox": rng(old_sox), "own": rng(old_own)}}
json.dump(out, open(FIG / "bg_corr_check.json", "w"), indent=1); print(json.dumps(out, indent=1))

# ---- figures ----
us = pd.read_parquet(ROOT / "data/raw/us/us_close.parquet"); us.index = pd.to_datetime(us.index)
lv = pd.concat([tx.rename("TAIEX"), us[["SOX", "SP500", "TSM_ADR"]]], axis=1, sort=True).ffill().loc["2019-01-02":].dropna(); lv = lv / lv.iloc[0] * 100
def level_panel(ax):
    for col, c, lab in [("TAIEX", BLUE, "TAIEX"), ("SOX", ORANGE, "SOX (U.S. semiconductors)"), ("SP500", AQUA, "S&P 500"), ("TSM_ADR", YELLOW, "TSMC ADR (NYSE: TSM)")]:
        ax.plot(lv.index, lv[col], color=c, lw=1.8, label=lab); ax.text(lv.index[-1], lv[col].iloc[-1], f" {lv[col].iloc[-1]:.0f}", color=c, fontsize=8, va="center")
    ax.set_yscale("log"); ax.set_title("Price level, indexed to 100 on 2019-01-02 (log scale)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
def roll_panel(ax):
    for s, c, lab in [(roll_sox, ORANGE, "overnight SOX return (US session t)"), (roll_sp, AQUA, "overnight S&P 500 return"), (roll_own, BLUE, "TAIEX own return on day t")]:
        ax.plot(s.index, s, color=c, lw=1.8, label=lab)
    ax.axhline(0, color=GRAY, ls="--", lw=1); ax.set_ylim(-0.3, 0.8)
    ax.set_title("Rolling 252-day correlation with TAIEX log return on day t+1"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.99, 0.03, "Pearson; stale U.S. rows excluded", transform=ax.transAxes, ha="right", fontsize=7.5, color=GRAY)
fig, ax = plt.subplots(figsize=(6.5, 4.2)); level_panel(ax); fig.tight_layout(); fig.savefig(FIG / "bg_level.png", bbox_inches="tight"); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.5, 4.2)); roll_panel(ax); fig.tight_layout(); fig.savefig(FIG / "bg_rollcorr.png", bbox_inches="tight"); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); level_panel(ax[0]); roll_panel(ax[1]); fig.tight_layout(); fig.savefig(FIG / "bg_market_linkage.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
