"""Redraw the multi-source ablation figure from figures/source_ablation.json."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG = Path(__file__).resolve().parents[1] / "figures"; J = json.load(open(FIG / "source_ablation.json"))
R = J["results"]; K = "LightGBM"
C2 = json.load(open(FIG / "review_stats2.json"))["auroc_diff_dayclustered_ci95"]
T = {**{f"drop {k}": C2[f"drop {k}"] for k in ["S1", "S2", "S3", "S4", "S5"]}, "S4+S1+S2+S3 vs S4 only": C2["S1-S3 joint, add on S4"]}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 11})
C = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#eda100", "S5": "#4a3aa7"}
L = {"S1": "S1 price & technical", "S2": "S2 inst. flows & margin", "S3": "S3 TW market-wide", "S4": "S4 overnight U.S./global", "S5": "S5 calendar"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ks = list(L); allv = R[f"{K}|All (S1–S5)"]["auroc"]
v = [R[f"{K}|{k} only"]["auroc"] for k in ks]
ax[0].barh([L[k] for k in ks][::-1], v[::-1], color=[C[k] for k in ks][::-1], height=0.6)
for i, x in enumerate(v[::-1]): ax[0].text(x + 0.002, i, f"{x:.3f}", va="center", fontsize=9)
ax[0].axvline(0.5, color="#898781", ls="--", lw=1); ax[0].axvline(allv, color="#0b0b0b", ls=":", lw=1.3)
ax[0].text(allv - 0.002, -0.75, f"all five sources = {allv:.3f}", fontsize=8.5, ha="right"); ax[0].set_xlim(0.48, 0.62); ax[0].set_ylim(-0.95, 4.5)
ax[0].set_title("A. Each source alone (AUROC)")
g = [T[f"drop {k}"] for k in ks]
ax[1].barh([f"without {L[k]}" for k in ks][::-1], [x[0] for x in g][::-1], color=[C[k] for k in ks][::-1], height=0.6,
           xerr=np.array([[x[0] - x[1], x[2] - x[0]] for x in g][::-1]).T, capsize=3, error_kw=dict(lw=0.9, ecolor="#52514e"))
for i, x in enumerate(g[::-1]): ax[1].text(max(x[2], 0) + 0.0015, i, f"−{x[0]:.4f}" if x[0] >= 0 else f"+{-x[0]:.4f}", va="center", fontsize=9)
ax[1].axvline(0, color="#898781", lw=1); ax[1].set_xlim(-0.005, 0.085); ax[1].set_title("B. AUROC lost when one source is removed")
names = ["S4 only", "S4 + S1", "S4 + S2", "S4 + S3", "S4 + S1 + S2 + S3"]; lab = ["S4 overnight U.S. only", "+ S1 price", "+ S2 flows", "+ S3 market", "+ S1 + S2 + S3"]
v = [R[f"{K}|{n}"]["auroc"] for n in names]; cols = [C["S4"], C["S1"], C["S2"], C["S3"], "#0b0b0b"]
ax[2].barh(lab[::-1], v[::-1], color=cols[::-1], height=0.6)
for i, x in enumerate(v[::-1]): ax[2].text(x + 0.0008, i, f"{x:.3f}", va="center", fontsize=9)
ax[2].set_xlim(0.56, 0.595); ax[2].set_title("C. Taiwan sources added on top of overnight U.S.")
ax[2].text(0.5605, -0.75, f"gain {T['S4+S1+S2+S3 vs S4 only'][0]:+.4f} (95% CI {T['S4+S1+S2+S3 vs S4 only'][1]:+.4f} to {T['S4+S1+S2+S3 vs S4 only'][2]:+.4f})", fontsize=8.5); ax[2].set_ylim(-0.95, 4.5)
fig.suptitle("Multi-source ablation — LightGBM, walk-forward test 2023–2026, 45,000 out-of-sample stock-days (whiskers: 95 % CI, bootstrap over trading days)", fontsize=11.5, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "prelim_sources.png", bbox_inches="tight"); print("ok")
