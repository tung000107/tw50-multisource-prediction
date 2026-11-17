"""Figures redrawn from the multi-seed checks (results/final/robustness.json, 38_robustness.py), so that the charts on the slides show the
same evidence as the text.  No model fitting.
 f04b_label_decomposition_4seed.png   R4: TW-only / U.S.-only / five sources on six targets, seed-mean of four seeds (the first chart was seed 42)
 f14b_source_study_4seed.png          R12: what each added source adds to the five, four targets, four seeds (+ the nine-source Shapley, one seed, labelled)
 f20b_sota_headtohead_ens.png         R10c: the head-to-head chart with the ALSTM ensemble (four runs' scores averaged) next to the single run"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
RB = json.load(open(RES / "robustness.json")); SS = json.load(open(RES / "source_study.json")); HH = json.load(open(RES / "sota_headtohead.json")); plt.rcParams.update({"font.size": 10})
# ---------------------------------------------------------------- f04b
I, N = RB["I_label_variants_over_seeds"], RB["N_more_label_variants_over_seeds"]; T = [("y_up", N, "Close→Close\n(primary)"), ("y_tr", N, "Close→Close\ndividend-adj."), ("y_gap", I, "Close→Open\n(overnight gap)"), ("y_intra", I, "Open→Close\n(tradable session)"), ("y_rel", I, "beats TAIEX\n(relative label)"), ("y_up_noflat", N, "Close→Close\nflat days excluded")]
fig, a = plt.subplots(figsize=(11.5, 4.9)); x = np.arange(len(T)); col = {"TW-only": "#8f96a8", "S4-only": "#eda100", "Full": "#4a3aa7"}; nm = {"TW-only": "Taiwan only", "S4-only": "U.S. only (S4)", "Full": "five sources"}
for j, arm in enumerate(["TW-only", "S4-only", "Full"]):
    v = [d[k]["seed_mean_auroc"][arm]["auroc"] for k, d, _ in T]; lo = [v[i] - d[k]["seed_mean_auroc"][arm]["ci"][0] for i, (k, d, _) in enumerate(T)]; hi = [d[k]["seed_mean_auroc"][arm]["ci"][1] - v[i] for i, (k, d, _) in enumerate(T)]
    a.bar(x + (j - 1) * 0.27, v, 0.26, yerr=[lo, hi], capsize=2.5, color=col[arm], label=nm[arm])
    for i, t in enumerate(v): a.text(i + (j - 1) * 0.27, t + hi[i] + 0.004, f"{t:.3f}", ha="center", fontsize=8)
a.axhline(0.5, color="#999", lw=0.9); a.set_ylim(0.44, 0.78); a.set_xticks(x); a.set_xticklabels([t[2] for t in T], fontsize=9); a.set_ylabel("AUROC, mean of 4 seeds, 95 % block CI\n(axis starts at 0.44)"); a.legend(frameon=False, ncol=3, loc="upper right")
a.set_title("Where does each source matter? Same features, six definitions of the target (LightGBM, four seeds)", fontsize=11, fontweight="bold"); a.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(FIG / "f04b_label_decomposition_4seed.png", dpi=150); plt.close()
# ---------------------------------------------------------------- f14b
J, O = RB["J_addon_sources_over_seeds"], RB["O_addon_sources_other_targets_over_seeds"]; tg = [("Close→Close", J), ("Close→Open", O["y_gap"]), ("Open→Close", O["y_intra"]), ("beats TAIEX", O["y_rel"])]; src = ["S6", "S7", "S8", "S9"]
cl = {"S6": "#e87ba4", "S7": "#1baf7a", "S8": "#2a78d6", "S9": "#eda100"}; ln = {"S6": "S6 news headlines (cnyes)", "S7": "S7 cross-stock / sector / leader", "S8": "S8 TX futures night session, positioning", "S9": "S9 day-trading, lending, revenue"}
fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1.45, 1]}); a = ax[0]; x = np.arange(len(tg))
for j, k in enumerate(src):
    p = [d["paired"][f"Full+{k} minus Full"] for _, d in tg]; v = [q["diff"] for q in p]; a.bar(x + (j - 1.5) * 0.2, v, 0.19, yerr=[[q["diff"] - q["ci"][0] for q in p], [q["ci"][1] - q["diff"] for q in p]], capsize=2.5, color=cl[k], label=ln[k])
a.axhline(0, color="#999", lw=0.9); a.set_xticks(x); a.set_xticklabels([t for t, _ in tg]); a.set_ylabel("ΔAUROC when added to the five sources\n(mean of 4 seeds, 95 % block CI)"); a.legend(frameon=False, fontsize=9, loc="upper right"); a.set_title("(a) What each added source adds — four seeds", fontsize=11, fontweight="bold")
a = ax[1]; ph = SS["shapley9_y_up"]["phi"]; ks = [f"S{i}" for i in range(1, 10)]; cc = ["#8f96a8"] * 3 + ["#eda100", "#8f96a8", cl["S6"], cl["S7"], cl["S8"], cl["S9"]]; a.bar(range(9), [ph[k] for k in ks], color=cc)
for i, k in enumerate(ks): a.text(i, ph[k] + (0.0015 if ph[k] >= 0 else -0.004), f"{ph[k]:+.3f}", ha="center", fontsize=8.5)
a.axhline(0, color="#999", lw=0.9); a.set_xticks(range(9)); a.set_xticklabels(ks); a.set_ylabel("AUROC attributed (Close→Close)"); a.set_title("(b) Exact Shapley over nine sources — ONE seed, no interval", fontsize=11, fontweight="bold")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "f14b_source_study_4seed.png", dpi=150); plt.close()
# ---------------------------------------------------------------- f20b
R = HH["rows"]; G = RB["G_official_alstm_reruns"]; rows = [(k, v["auroc"], v["ci"], ("" if k.startswith("SOX") else f"acc {v['acc']:.3f}  MCC {v['mcc']:.3f}"), "final" if k == HH["final_model"] else ("sota" if k in HH["sota_models"] else "ours")) for k, v in R.items()]
best = "ALSTM official (all nine sources, 74 features)"; rows = [(k + ("  — one run" if k == best else ""), a_, c, t, g) for k, a_, c, t, g in rows]
rows.append((f"ALSTM official, nine sources — ensemble of {len(G['per_run_auroc'])} runs (scores averaged)", G["mean_of_runs_scores"]["auroc"], G["mean_of_runs_scores"]["ci"], f"single runs {' / '.join(f'{x:.3f}' for x in G['per_run_auroc'].values())}", "sota_ens"))
rows.sort(key=lambda r: r[1]); fig, a = plt.subplots(figsize=(18.5, 6.9)); y = np.arange(len(rows)); cmap = {"final": "#3a2c9c", "ours": "#a99be0", "sota": "#eb6834", "sota_ens": "#b8431a"}
a.barh(y, [r[1] - 0.47 for r in rows], left=0.47, color=[cmap[r[4]] for r in rows], xerr=[[r[1] - r[2][0] for r in rows], [r[2][1] - r[1] for r in rows]], error_kw=dict(lw=0.9, ecolor="#444"), height=0.72)
for i, r in enumerate(rows): a.text(r[2][1] + 0.003, i, f"{r[1]:.3f}   {r[3]}", va="center", fontsize=8.6)
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=9); a.axvline(0.5, color="#999", lw=0.9); a.set_xlim(0.47, 0.70); a.set_xlabel("AUROC, pooled 2023–2026 (95 % trading-day block-bootstrap CI; axis starts at 0.47)")
import matplotlib.patches as mp; a.legend(handles=[mp.Patch(color=cmap["sota"], label="published architecture, one run (price-only inputs, or given our inputs)"), mp.Patch(color=cmap["sota_ens"], label="published architecture, four runs averaged — built like our final model"), mp.Patch(color=cmap["ours"], label="our models"), mp.Patch(color=cmap["final"], label="our final model (7 seeds averaged)")], frameon=False, fontsize=9, loc="lower right")
a.set_title("Same data, same walk-forward split, same label: published architectures vs our multi-source models (Taiwan 50)", fontsize=11.5, fontweight="bold"); a.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(FIG / "f20b_sota_headtohead_ens.png", dpi=150); plt.close(); print("figures saved")
