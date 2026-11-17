"""Source-interaction EDA: next-day up-rate on the joint grid of overnight SOX direction x Taiwan-side signals. Output: figures/eda_08_interaction.png + json"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figures"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 10.5})
d = pd.read_parquet(ROOT / "data/processed/dataset.parquet"); d = d[d.us_stale == 0].copy()
d["US"] = np.where(d.us_sox_ret > 0, "SOX up overnight", "SOX down overnight")
q = lambda s: pd.qcut(s, 3, labels=["low tercile", "mid", "high tercile"])
panels = [("foreign_net_ratio", "S2  Foreign net buy / volume (day t)"), ("margin_chg", "S2  Margin balance change (day t)"), ("taiex_ret1", "S3  TAIEX return (day t)"), ("ret5", "S1  Own 5-day return")]
out = {}; fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True)
for ax, (col, title) in zip(axes, panels):
    d["bin"] = q(d[col]); t = d.groupby(["bin", "US"], observed=True).y_up.agg(["mean", "size"]).reset_index()
    out[col] = {f"{r.bin}|{r.US}": [round(float(r['mean']), 4), int(r['size'])] for _, r in t.iterrows()}
    xs = np.arange(3); w = 0.38
    for j, (u, c) in enumerate([("SOX down overnight", "#eb6834"), ("SOX up overnight", "#2a78d6")]):
        v = t[t.US == u].set_index("bin")["mean"].reindex(["low tercile", "mid", "high tercile"]).values
        ax.bar(xs + (j - 0.5) * w, v, w, color=c, label=u)
        for x, val in zip(xs + (j - 0.5) * w, v): ax.text(x, val + 0.008, f"{val:.0%}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(["low", "mid", "high"]); ax.set_title(title); ax.axhline(d.y_up.mean(), color="#898781", ls="--", lw=1); ax.set_ylim(0.25, 0.68)
    ax.set_xlabel("tercile of the Taiwan-side signal")
axes[0].set_ylabel("P(up next day)"); axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Sources interact: the same Taiwan-side signal means different things depending on the overnight U.S. move", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "eda_08_interaction.png", bbox_inches="tight")
# spread of up-rate across terciles within each US state
spread = {}
for col in out:
    for u in ["SOX up overnight", "SOX down overnight"]:
        vals = [out[col][f"{b}|{u}"][0] for b in ["low tercile", "mid", "high tercile"]]; spread[f"{col}|{u}"] = [vals[0], vals[2], round(vals[2] - vals[0], 4)]
json.dump({"cells": out, "low_high_spread": spread}, open(FIG / "eda_interaction.json", "w"), indent=1); print(json.dumps(spread, indent=1))
