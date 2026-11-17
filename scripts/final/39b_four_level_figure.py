"""Redraw f24_four_level_deep.png from results/final/four_level_deep.json (no refit). See 39_four_level_deep.py."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import *
out = json.load(open(RES / "four_level_deep.json")); A, B, C = out["A_macro_f1"], out["B_per_class"], out["C_predicted_class_outcomes"]; NAMES = ["strong down", "mild down", "mild up", "strong up"]; ARMS = ["TW-only", "S4-only", "Full", "All9"]
# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10}); fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.0), gridspec_kw={"width_ratios": [1.05, 1.0, 1.15]}); col = {"TW-only": "#8f96a8", "S4-only": "#eda100", "Full": "#4a3aa7", "All9": "#1baf7a"}; nm = {"TW-only": "Taiwan only", "S4-only": "U.S. only", "Full": "five sources", "All9": "nine sources"}
a = ax[0]; x = np.arange(4); w = 0.2
for j, arm in enumerate(ARMS):
    v = [B[arm]["f1"][n] for n in NAMES]; lo = [v[i] - B[arm]["f1_ci"][n][0] for i, n in enumerate(NAMES)]; hi = [B[arm]["f1_ci"][n][1] - v[i] for i, n in enumerate(NAMES)]
    a.bar(x + (j - 1.5) * w, v, w, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], capsize=2, color=col[arm], label=f"{nm[arm]} (Macro-F1 {A[arm]['macro_f1_seed_mean_prob']:.3f})")
a.set_xticks(x); a.set_xticklabels(NAMES); a.set_ylabel("F1 of the class (95 % trading-day block CI)"); a.set_ylim(0, 0.62); a.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=2); a.set_title("(a) Which class can be predicted, by information set", fontsize=11, fontweight="bold")
a = ax[1]; cmn = np.array(B["Full"]["confusion_row_normalised"]); im = a.imshow(cmn, cmap="Purples", vmin=0, vmax=0.6)
for i in range(4):
    for j in range(4): a.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", fontsize=11, color="white" if cmn[i, j] > 0.35 else "#222")
a.set_xticks(range(4)); a.set_xticklabels(NAMES, fontsize=9); a.set_yticks(range(4)); a.set_yticklabels(NAMES, fontsize=9); a.set_xlabel("predicted class"); a.set_ylabel("true class"); a.set_title("(b) Five-source model: where each true class goes (row = 100 %)", fontsize=11, fontweight="bold")
a = ax[2]; w = 0.38
for j, arm in enumerate(["Full", "All9"]):
    v = [C[arm][n].get("mean_return", np.nan) * 100 for n in NAMES]; lo = [(C[arm][n]["mean_return"] - C[arm][n]["mean_return_ci"][0]) * 100 if "mean_return" in C[arm][n] else 0 for n in NAMES]; hi = [(C[arm][n]["mean_return_ci"][1] - C[arm][n]["mean_return"]) * 100 if "mean_return" in C[arm][n] else 0 for n in NAMES]
    a.bar(x + (j - 0.5) * w, v, w, yerr=[lo, hi], capsize=3, color=col[arm], label=nm[arm])
    for i, n in enumerate(NAMES):
        if "up_rate" in C[arm][n]: a.text(i + (j - 0.5) * w, (max(v[i] + hi[i], [C[q][n]["mean_return_ci"][1] * 100 for q in ("Full", "All9")][0]) + 0.03 + j * 0.13) if v[i] >= 0.05 else (min(v[i] - lo[i], C["Full"][n]["mean_return_ci"][0] * 100) - 0.03 - j * 0.13), f"up-rate {C[arm][n]['up_rate']:.0%}\nn = {C[arm][n]['n']:,}", ha="center", va="bottom" if v[i] >= 0.05 else "top", fontsize=7.5)
a.axhline(C["Full"]["all"]["mean_return"] * 100, color="#d7263d", ls="--", lw=0.9, label="all stock-days"); a.axhline(0, color="#999", lw=0.8); a.set_xticks(x); a.set_xticklabels(["predicted\n" + n for n in NAMES]); a.set_ylabel("realised mean next-day return, % (95 % CI)"); a.set_ylim(-1.0, 1.32); a.legend(fontsize=8.5, frameon=False, loc="upper left"); a.set_title("(c) What a predicted class is worth in realised returns", fontsize=11, fontweight="bold")
for a in (ax[0], ax[2]): a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "f24_four_level_deep.png", dpi=150); print("figure saved; done")
