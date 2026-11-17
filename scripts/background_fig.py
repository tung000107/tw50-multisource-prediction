"""Background figure: TAIEX vs U.S. co-movement and rolling next-day correlation. Output: figures/bg_market_linkage.png"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figures"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 11})
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781"
tx = pd.read_parquet(ROOT / "data/raw/finmind/TAIEX.parquet"); tx["date"] = pd.to_datetime(tx.date); tx = tx.set_index("date").close.rename("TAIEX")
us = pd.read_parquet(ROOT / "data/raw/us/us_close.parquet"); us.index = pd.to_datetime(us.index)
# panel A: indexed levels
lv = pd.concat([tx, us[["SOX", "SP500", "TSM_ADR"]]], axis=1).ffill().loc["2019-01-02":].dropna()
lv = lv / lv.iloc[0] * 100
# panel B: rolling corr of TAIEX next-day return with latest overnight US return vs. with own previous-day return
r = tx.pct_change().rename("tw"); d = pd.DataFrame({"tw": r}); d["tw_next"] = d.tw.shift(-1)
usr = us[["SOX", "SP500"]].pct_change().dropna(); usr.index.name = "us_date"; usr = usr.reset_index()
d = d.reset_index().rename(columns={"index": "date"}); d["date"] = d["date"].astype("datetime64[ns]"); usr["us_date"] = usr["us_date"].astype("datetime64[ns]")
d = pd.merge_asof(d.sort_values("date"), usr.sort_values("us_date"), left_on="date", right_on="us_date").set_index("date").dropna()
W = 250
c_sox = d.tw_next.rolling(W).corr(d.SOX); c_sp = d.tw_next.rolling(W).corr(d.SP500); c_own = d.tw_next.rolling(W).corr(d.tw)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for col, c, lab in [("TAIEX", BLUE, "TAIEX"), ("SOX", ORANGE, "SOX (U.S. semiconductors)"), ("SP500", AQUA, "S&P 500"), ("TSM_ADR", YELLOW, "TSMC ADR (NYSE: TSM)")]:
    ax[0].plot(lv.index, lv[col], color=c, lw=1.8, label=lab); ax[0].text(lv.index[-1], lv[col].iloc[-1], f" {lv[col].iloc[-1]:.0f}", color=c, fontsize=8, va="center")
ax[0].set_yscale("log"); ax[0].set_title("Price level, indexed to 100 on 2019-01-02 (log scale)"); ax[0].legend(frameon=False, fontsize=8, loc="upper left")
corr_level = np.corrcoef(np.log(lv.TAIEX), np.log(lv.SOX))[0, 1]
ax[0].text(0.98, 0.04, f"corr(log TAIEX, log SOX) = {corr_level:.2f}", transform=ax[0].transAxes, ha="right", fontsize=8.5, bbox=dict(fc="white", ec="#e1e0d9"))
for s, c, lab in [(c_sox, ORANGE, "overnight SOX return (US session t)"), (c_sp, AQUA, "overnight S&P 500 return"), (c_own, BLUE, "TAIEX own return on day t")]:
    ax[1].plot(s.index, s, color=c, lw=1.8, label=lab)
ax[1].axhline(0, color=GRAY, ls="--", lw=1); ax[1].set_ylim(-0.3, 0.8)
ax[1].set_title("Rolling 1-year correlation with TAIEX return on day t+1"); ax[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "bg_market_linkage.png", bbox_inches="tight"); plt.close(fig)
print("full-sample corr: SOX", round(d.tw_next.corr(d.SOX), 3), "SP500", round(d.tw_next.corr(d.SP500), 3), "own", round(d.tw_next.corr(d.tw), 3), "| level corr", round(corr_level, 3))
print("rolling SOX corr range", round(c_sox.min(), 2), round(c_sox.max(), 2), "| own range", round(c_own.min(), 2), round(c_own.max(), 2))
print("indexed end:", lv.iloc[-1].round(0).to_dict())
