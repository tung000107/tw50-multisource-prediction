"""Redraw f21 from results/final/final_model_checks.json (no model fitting). See 34_final_model_checks.py."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import *
out = json.load(open(RES / "final_model_checks.json")); B = out["B_which_day_vs_which_stock"]
LF, LA, EF, EA = "LightGBM Full (S1-S5)", "LightGBM All9 (S1-S9)", "Ensemble LightGBM+LSTM (S1-S5)", "Ensemble LightGBM+LSTM All9 (ours, final)"
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
# panel (c): the S4-vs-S8 ablation repeated over seeds (robustness.json, 38_robustness.py); the first version of this panel showed seed 42 only
a = ax[2]; RBc = json.load(open(RES / "robustness.json"))["C_S4_vs_S8_over_seeds"]; ar = ["TW+S4", "TW+S8", "TW+S4+S8"]; x = np.arange(len(ar)); cols = {"y_up": "#4a3aa7", "y_gap": "#eb6834"}; nm2 = {"y_up": "Close→Close", "y_gap": "Close→Open (gap)"}
for j, l in enumerate(["y_up", "y_gap"]):
    sm = RBc[l]["seed_mean"]; v = {"TW+S4": sm["S8_arm_minus_S4_arm"]["a"], "TW+S8": sm["S8_arm_minus_S4_arm"]["b"], "TW+S4+S8": sm["S4_on_top_of_S8"]["b"]}; off = (j - 0.5) * 0.36
    a.bar(x + off, [v[k] for k in ar], 0.34, color=cols[l], label=nm2[l] + " (bar: mean of 4 seeds)")
    for i, k in enumerate(ar):
        ps = [RBc[l]["per_seed_auroc"][str(sd)][k] for sd in RBc[l]["seeds"]]; a.scatter([i + off] * len(ps), ps, s=14, color="black", zorder=3, label="single seeds" if (j == 0 and i == 0) else None); a.text(i + off, max(max(ps), v[k]) + 0.006, f"{v[k]:.3f}", ha="center", fontsize=8)
a.axhline(0.5, color="#999", lw=0.8); a.set_ylim(0.55, 0.84); a.set_xticks(x); a.set_xticklabels(["TW + S4\n(U.S.)", "TW + S8\n(TX night)", "TW + S4 + S8\n(both)"], fontsize=8.5); a.legend(fontsize=7.5, frameon=False, loc="upper left"); a.set_ylabel("AUROC, LightGBM (axis starts at 0.55)")
a.set_title("(c) U.S. block vs TX night session vs both, 4 seeds", fontsize=10, fontweight="bold")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "figures/final/f21_final_model_checks.png", dpi=200); print("figure saved; done")
