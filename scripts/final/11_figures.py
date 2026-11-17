"""All final-deck figures from results/final/*.json -> figures/final/*.png (consistent style, 200 dpi)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from common import *
for f in ["Noto Sans CJK TC", "DejaVu Sans"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3dc",
                     "axes.edgecolor": "#c3c2b7", "xtick.color": "#3d3c39", "ytick.color": "#3d3c39", "axes.facecolor": "white", "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9.5})
C = {"Full": "#4a3aa7", "S4-only": "#eda100", "TW-only": "#7f8fa6", "S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#eda100", "S5": "#9b59b6", "rule": "#b8b5ad", "ok": "#1baf7a", "bad": "#d7263d"}
J = lambda n: json.load(open(RES / n))
M = J("main_results.json"); E = J("explain.json"); L = J("labels.json"); A = J("audits.json"); SN = J("sota_stocknet.json"); SD = J("sota_stocknet_deep.json"); Y = J("sota_yang.json"); B = J("backtest.json")
def save(fig, name): fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig); print("fig", name)
arm = M["arms"]
def A_(g): return arm[g]["seed_mean_prob"]
# ---------------- F1 main binary ----------------
models = [("LR", {"TW-only": "LR|S1+S2+S3", "S4-only": "LR|S4", "Full": "LR|S1+S2+S3+S4+S5"}), ("LightGBM", {"TW-only": "LightGBM|TW-only", "S4-only": "LightGBM|S4-only", "Full": "LightGBM|Full"}),
          ("XGBoost", {"TW-only": "XGBoost|S1", "S4-only": "XGBoost|S4-only", "Full": "XGBoost|Full"}), ("LSTM (W=5)", {"TW-only": "LSTM-W5|TW-only", "S4-only": "LSTM-W5|S4-only", "Full": "LSTM-W5|Full"}), ("Transformer (W=5)", {"TW-only": "Transformer-W5|TW-only", "S4-only": "Transformer-W5|S4-only", "Full": "Transformer-W5|Full"})]
models[2][1]["TW-only"] = None   # XGBoost TW-only arm was not run as a 3-source set
fig, ax = plt.subplots(figsize=(11, 4.4)); x = np.arange(len(models)); w = 0.26
for j, a in enumerate(["TW-only", "S4-only", "Full"]):
    v, lo, hi = [], [], []
    for _, mm in models:
        g = mm[a]
        if g is None: v.append(np.nan); lo.append(0); hi.append(0); continue
        s = A_(g); v.append(s["pooled"]["auroc"]); ci = s["ci_auroc"] or [s["pooled"]["auroc"]] * 2; lo.append(s["pooled"]["auroc"] - ci[0]); hi.append(ci[1] - s["pooled"]["auroc"])
    ax.bar(x + (j - 1) * w, v, w, yerr=[lo, hi], color=C[a], capsize=2, error_kw=dict(lw=0.8), label=a)
    for xi, vi in zip(x + (j - 1) * w, v):
        if not np.isnan(vi): ax.text(xi, vi + 0.012, f"{vi:.3f}", ha="center", fontsize=7.5)
rule = A_("Rule|SOX return score")["pooled"]["auroc"]; ax.axhline(rule, color=C["rule"], ls="--", lw=1.2); ax.text(-0.45, rule + 0.004, f"overnight SOX return alone {rule:.3f}", fontsize=8, color="#555", ha="left")
ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.47, 0.64); ax.set_xticks(x); ax.set_xticklabels([m for m, _ in models]); ax.set_ylabel("AUROC, pooled 2023–2026 (seed-mean probability)")
ax.set_title("Primary task (next-day Close(t+1) > Close(t)): every model, three information sets; whiskers = 95 % trading-day block-bootstrap CI"); ax.legend(frameon=False, ncol=3, loc="upper left")
save(fig, "f01_main_binary.png")
# ---------------- F2 by year ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.8)); yrs = [str(y) for y in TEST_YEARS]
for a, g, mk in [("Full", "LightGBM|Full", "o"), ("S4-only", "LightGBM|S4-only", "s"), ("TW-only", "LightGBM|TW-only", "^")]:
    ax.plot(yrs, [A_(g)["by_year"][y]["auroc"] for y in yrs], marker=mk, lw=2, color=C[a], label=f"LightGBM {a}")
ax.plot(yrs, [A_("Rule|SOX return score")["by_year"][y]["auroc"] for y in yrs], ls="--", color=C["rule"], lw=1.5, label="overnight SOX return alone")
ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.48, 0.64); ax.set_ylabel("AUROC"); ax.set_xlabel("walk-forward test year (2026: Jan–Sep 23)"); ax.set_title("The pattern holds in every test year"); ax.legend(frameon=False, fontsize=8)
save(fig, "f02_by_year.png")
# ---------------- F3 Shapley ----------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, kind in zip(axes[:2], ["LightGBM", "LR"]):
    sh = E[f"shapley_{kind}"]; v = [sh["phi"][s]["value"] for s in SRC]; ci = np.array([sh["phi"][s]["ci"] for s in SRC])
    ax.bar(list(SRC), v, color=[C[s] for s in SRC], yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], capsize=3, error_kw=dict(lw=0.8))
    for i, vi in enumerate(v): ax.text(i, vi + (0.003 if vi >= 0 else -0.008), f"{vi:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="#999", lw=0.8); ax.set_title(f"Exact source Shapley value, {kind}\n(AUROC above 0.5 split over 5 sources; sum = {sh['sum_phi']:+.3f})"); ax.set_ylabel("ΔAUROC attributed")
ax = axes[2]; sh = E["shapley_LightGBM"]; x = np.arange(5); w = 0.38
ax.bar(x - w / 2, [sh["single"][s] - 0.5 for s in SRC], w, color=[C[s] for s in SRC], alpha=0.55, label="alone (AUROC − 0.5)")
ax.bar(x + w / 2, [sh["leave_one_out"][s] for s in SRC], w, color=[C[s] for s in SRC], label="loss when removed from Full")
ax.set_xticks(x); ax.set_xticklabels(list(SRC)); ax.axhline(0, color="#999", lw=0.8); ax.legend(frameon=False, fontsize=8); ax.set_title("Single-source vs. leave-one-out (LightGBM)")
save(fig, "f03_shapley.png")
# ---------------- F4 label decomposition ----------------
labs = [("y_up", "Close→Close\n(primary)"), ("y_tr", "Close→Close\ndividend-adj."), ("y_gap", "Close→Open\n(overnight gap)"), ("y_intra", "Open→Close\n(tradable intraday)"), ("y_rel", "beats TAIEX\n(relative)")]
fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(len(labs)); w = 0.26
for j, a in enumerate(["TW-only", "S4-only", "Full"]):
    v, lo, hi = [], [], []
    for lab, _ in labs:
        if lab == "y_up": s = A_(f"LightGBM|{a}"); vv = s["pooled"]["auroc"]; ci = s["ci_auroc"]
        else: vv = L["binary_variants"][lab][a]["auroc"]; ci = L["binary_variants"][lab][a]["ci"]
        v.append(vv); lo.append(vv - ci[0]); hi.append(ci[1] - vv)
    ax.bar(x + (j - 1) * w, v, w, yerr=[lo, hi], color=C[a], capsize=2, error_kw=dict(lw=0.8), label=a)
    for xi, vi in zip(x + (j - 1) * w, v): ax.text(xi, vi + 0.012, f"{vi:.3f}", ha="center", fontsize=7.5)
ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.44, 0.78); ax.set_xticks(x); ax.set_xticklabels([l for _, l in labs]); ax.set_ylabel("AUROC (LightGBM, pooled 2023–2026)")
ax.set_title("Where does each source matter? The same features, five definitions of 'tomorrow'"); ax.legend(frameon=False, ncol=3)
save(fig, "f04_label_decomposition.png")
# ---------------- F5 four-level / magnitude ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (key, ttl) in zip(axes, [("raw", "k = median |r| of the training window (0.83–0.86 %)"), ("volnorm", "volatility-normalised: k = median |r / vol20|")]):
    r = L["four_level"][key]; x = np.arange(3); w = 0.26
    for j, (met, lab) in enumerate([("macro_f1", "four-level Macro-F1"), ("bal_acc", "four-level balanced acc."), ("mag", "|move| > k AUROC")]):
        v = [r[f"magnitude_{a}"]["auroc"] if met == "mag" else r[f"four_{a}"][met] for a in ["TW-only", "S4-only", "Full"]]
        ax.bar(x + (j - 1) * w, v, w, color=["#7f8fa6", "#eda100", "#4a3aa7"], alpha=[0.45, 0.7, 1.0][j], label=lab)
        for xi, vi in zip(x + (j - 1) * w, v): ax.text(xi, vi + 0.01, f"{vi:.3f}", ha="center", fontsize=7.5)
    ax.axhline(r["majority_macro_f1"], color="#999", ls=":", lw=1); ax.text(2.4, r["majority_macro_f1"] + 0.005, "majority-class Macro-F1", fontsize=7.5, ha="right", color="#666")
    ax.set_xticks(x); ax.set_xticklabels(["TW-only", "S4-only", "Full"]); ax.set_ylim(0, 0.95); ax.set_title(ttl, fontsize=10); ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3)
fig.suptitle("Secondary four-level task (strong / mild × down / up): magnitude is a Taiwan-side signal", fontweight="bold")
save(fig, "f05_four_level.png")
# ---------------- F6 audits ----------------
fig, ax = plt.subplots(figsize=(8.5, 3.8))
items = [("random stratified\n5-fold (shuffled rows)", A["random_5fold_full"]["auroc_test_years_only"], C["bad"]), ("walk-forward\n(ours)", A["walk_forward_full"]["auroc"], C["Full"]), ("walk-forward,\nU.S. stale days excluded", A["exclude_us_stale"]["auroc"], C["Full"]),
         ("training labels permuted\nin 20-day blocks (null)", float(np.mean(A["block_permutation_null_auroc"])), C["rule"]), ("U.S. session of t+1\ninjected (deliberate leak)", A["future_shift_full"], C["bad"])]
ax.bar(range(len(items)), [v for _, v, _ in items], color=[c for _, _, c in items], width=0.6)
for i, (_, v, _) in enumerate(items): ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.45, 0.76); ax.set_xticks(range(len(items))); ax.set_xticklabels([n for n, _, _ in items], fontsize=8.5); ax.set_ylabel("AUROC, LightGBM Full, test rows 2023–2026")
ax.set_title(f"Leakage audit: a random split inflates AUROC ×{A['leakage_ratio']:.2f}; nulls and deliberate leaks behave as expected")
save(fig, "f06_audits.png")
# ---------------- F7 SOTA reproduction ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]; pub = SN["published_pulsekan_table1_stocknet"]
names = list(pub) + ["ours: LR", "ours: LightGBM", "ours: XGBoost", "ours: ALSTM-style\n(official inputs)"]
accs = [pub[k][0] for k in pub] + [SN["LR"]["acc"], SN["LightGBM"]["mean"]["acc"], SN["XGBoost"]["mean"]["acc"], SD["stocknet_band"]["mean"]["acc"]]
errs = [0] * len(pub) + [0, SN["LightGBM"]["std"]["acc"], SN["XGBoost"]["std"]["acc"], SD["stocknet_band"]["std"]["acc"]]
cols = ["#b8b5ad"] * len(pub) + ["#4a3aa7"] * 4
ax.barh(range(len(names)), accs, color=cols, xerr=errs, capsize=2); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
for i, v in enumerate(accs): ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
ax.axvline(SN["majority_acc_test"], color="#999", ls="--", lw=1); ax.set_xlim(0.48, 0.60); ax.set_xlabel("accuracy on the StockNet test split (2015-10 → 2015-12, 3,744 samples)")
ax.set_title("StockNet benchmark: published (PULSE-KAN Table 1, grey) vs. our runs (purple)", fontsize=10); ax.grid(axis="x"); ax.grid(False, axis="y")
ax = axes[1]
rs = Y["random_split"]; wf = Y["walk_forward_pooled_mean"]
labels = ["raw indicators\n(random split)", "+ predicted ind.\n(random split)", "oracle: true next\nclose (random split)", "raw indicators\n(walk-forward)", "+ predicted ind.\n(walk-forward)"]
vals = [[rs["raw indicators (Table 4 analogue)"][m]["acc"], rs["+ predicted indicators (Table 5 analogue)"][m]["acc"], rs["oracle: indicators with the true next close"][m]["acc"], wf["raw"][m]["acc"], wf["+ predicted"][m]["acc"]] for m in ["RF", "SVM", "LightGBM"]]
x = np.arange(5); w = 0.26
for j, (m, col) in enumerate(zip(["RF", "SVM", "LightGBM"], ["#2a78d6", "#eb6834", "#4a3aa7"])):
    ax.bar(x + (j - 1) * w, vals[j], w, color=col, label=m)
    for xi, vi in zip(x + (j - 1) * w, vals[j]): ax.text(xi, vi + 0.006, f"{vi:.2f}", ha="center", fontsize=7)
ax.axhline(0.67, color="#d7263d", ls="--", lw=1); ax.text(4.3, 0.675, "published all-day avg 0.67 (TSMC, SVM) [3]", fontsize=7.5, color="#d7263d", ha="right")
ax.axhline(0.86, color="#d7263d", ls=":", lw=1); ax.text(4.3, 0.865, "published best, filtered days 0.86 [3]", fontsize=7.5, color="#d7263d", ha="right")
ax.set_ylim(0.45, 1.04); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylabel("accuracy, Taiwan 50 2019–2026"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Yang et al. (2024) pipeline re-implemented on our data", fontsize=10)
save(fig, "f07_sota_reproduction.png")
# ---------------- F8 sector / stock ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]; sec = E["by_sector"]; order = sorted(sec, key=lambda s: -sec[s]["n"])
x = np.arange(len(order)); w = 0.26
for j, (a, key) in enumerate([("TW-only", "auroc_TW"), ("S4-only", "auroc_S4"), ("Full", "auroc_Full")]):
    ax.bar(x + (j - 1) * w, [sec[s][key] for s in order], w, color=C[a], label=a)
for i, s in enumerate(order):
    d = sec[s]["full_minus_S4"]; ax.text(i, max(sec[s]["auroc_Full"], sec[s]["auroc_S4"]) + 0.01, f"Δ{d[0]:+.3f}\n[{d[1]:+.3f}, {d[2]:+.3f}]" if False else f"Full−S4 {sec[s]['auroc_Full'] - sec[s]['auroc_S4']:+.3f}\n[{d[0]:+.3f}, {d[1]:+.3f}]", ha="center", fontsize=6.8)
ax.set_xticks(x); ax.set_xticklabels([s.replace("/", "/\n") + f"\n(n={sec[s]['n']:,})" for s in order], fontsize=7.5); ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.45, 0.70); ax.set_ylabel("AUROC (LightGBM)"); ax.legend(frameon=False, ncol=3, fontsize=8); ax.set_title("By sector (test 2023–2026)")
ax = axes[1]; stk = pd.DataFrame(E["by_stock"]); stk["lo"] = stk.ci.str[0]; stk["hi"] = stk.ci.str[1]; stk = stk.sort_values("diff")
colors = np.where(stk.lo > 0, C["ok"], np.where(stk.hi < 0, C["bad"], "#b8b5ad"))
ax.barh(range(len(stk)), stk["diff"], color=colors, xerr=[stk["diff"] - stk.lo, stk.hi - stk["diff"]], error_kw=dict(lw=0.4, ecolor="#777"))
ax.set_yticks(range(len(stk))); ax.set_yticklabels(stk.stock, fontsize=5.5); ax.axvline(0, color="#999", lw=0.8); ax.set_xlabel("Full − S4-only AUROC per stock (95 % CI)")
s = E["by_stock_summary"]; ax.set_title(f"Per stock: {s['n_better']}/50 positive; {s['n_sig_up']} significantly up, {s['n_sig_down']} down", fontsize=10); ax.grid(axis="x"); ax.grid(False, axis="y")
save(fig, "f08_sector_stock.png")
# ---------------- F9 grouped SHAP ----------------
gs = E["grouped_shap"]; fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]; yrs = list(gs); bottom = np.zeros(len(yrs))
for s in SRC:
    v = np.array([gs[y]["share"][s] for y in yrs]); ax.bar(yrs, v, bottom=bottom, color=C[s], label=f"{s} {SRC_NAME[s]}"); 
    for i, (b, vi) in enumerate(zip(bottom, v)):
        if vi > 0.06: ax.text(i, b + vi / 2, f"{vi:.0%}", ha="center", va="center", fontsize=8, color="white")
    bottom += v
ax.set_ylim(0, 1); ax.set_ylabel("share of mean |grouped SHAP|"); ax.set_title("Inside the Full LightGBM: attribution by source, per test year"); ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, -0.12), ncol=3)
ax = axes[1]; top = pd.Series(gs[yrs[-1]]["top_features"]).sort_values()
ax.barh(top.index, top.values, color=[C[next(k for k in SRC if f in SRC[k])] for f in top.index]); ax.set_title(f"Top-10 features by mean |SHAP|, test year {yrs[-1]}"); ax.grid(axis="x"); ax.grid(False, axis="y"); ax.set_xlabel("mean |SHAP| (log-odds)")
save(fig, "f09_grouped_shap.png")
# ---------------- F10 backtest ----------------
bd = pd.read_parquet(RES / "backtest_daily.parquet"); fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
for key, lab, col, ls in [("p_oc_Full|k10", "Full, open→close model, top-10, net", C["Full"], "-"), ("p_oc_TW-only|k10", "TW-only, open→close model, top-10, net", C["TW-only"], "-"), ("p_oc_S4-only|k10", "S4-only, open→close model, top-10, net", C["S4-only"], "-"),
                          ("p_cc_Full|k10", "Full, close→close model, top-10, net", C["Full"], "--"), ("p_rand|k10", "random top-10, net", "#b8b5ad", ":")]:
    r = bd[key]["long_net"]; ax.plot(pd.to_datetime(r.index), (1 + r).cumprod() - 1, color=col, ls=ls, lw=1.6, label=lab)
ew = pd.Series(B["benchmarks"]["equal_weight_50_intraday_net"]["mean_daily"], index=bd.index); 
ax.axhline(0, color="#999", lw=0.8); ax.set_ylabel("cumulative return (intraday long-only, after 0.435 % round-trip cost)"); ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.set_title("Cost-aware backtest, 2023–2026: buy top-k at the t+1 open, sell at the t+1 close")
ax = axes[1]; rows = [("Full oc top-10", "p_oc_Full|top10"), ("Full oc top-5", "p_oc_Full|top5"), ("TW oc top-10", "p_oc_TW-only|top10"), ("S4 oc top-10", "p_oc_S4-only|top10"), ("Full cc top-10", "p_cc_Full|top10"), ("SOX rule top-10", "p_sox|top10"), ("random top-10", "p_rand|top10")]
tab = [[n, f"{B['strategies'][k]['long_only_net']['ann_return']:+.1%}", f"{B['strategies'][k]['long_only_net']['sharpe']:+.2f}", f"{B['strategies'][k]['long_only_net']['max_drawdown']:.1%}", f"{B['strategies'][k]['long_only_gross']['sharpe']:+.2f}"] for n, k in rows]
tab += [["EW-50 intraday (net)", f"{B['benchmarks']['equal_weight_50_intraday_net']['ann_return']:+.1%}", f"{B['benchmarks']['equal_weight_50_intraday_net']['sharpe']:+.2f}", f"{B['benchmarks']['equal_weight_50_intraday_net']['max_drawdown']:.1%}", "—"],
        ["0050 ETF buy&hold", f"{B['benchmarks']['etf_0050_buy_hold_close_to_close']['ann_return']:+.1%}", f"{B['benchmarks']['etf_0050_buy_hold_close_to_close']['sharpe']:+.2f}", f"{B['benchmarks']['etf_0050_buy_hold_close_to_close']['max_drawdown']:.1%}", "—"]]
ax.axis("off"); t = ax.table(cellText=tab, colLabels=["strategy", "ann. return (net)", "Sharpe (net)", "max DD", "Sharpe (gross)"], loc="center", cellLoc="center", colWidths=[0.34, 0.2, 0.15, 0.14, 0.17]); t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.35)
save(fig, "f10_backtest.png")
# ---------------- F11 calibration + within-day ----------------
oos = pd.read_parquet(RES / "oos_tabular.parquet"); pf = oos[[c for c in oos if c.startswith("LightGBM|Full|")]].mean(axis=1).values; y = oos.y_up.values
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
ax = axes[0]; bins = np.linspace(0.2, 0.8, 13); idx = np.digitize(pf, bins); xs, ys, ns = [], [], []
for b in range(1, len(bins)):
    m = idx == b
    if m.sum() > 200: xs.append(pf[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
ax.plot([0.2, 0.8], [0.2, 0.8], "--", color="#999", lw=1, label="perfect calibration"); ax.plot(xs, ys, "o-", color=C["Full"], label="LightGBM Full (seed-mean)"); ax.axhline(y.mean(), color="#bbb", ls=":", lw=1)
bs = metrics(y, pf); ax.set_xlabel("predicted P(up)"); ax.set_ylabel("observed up-rate in bin"); ax.set_title(f"Calibration: Brier {bs['brier']:.4f} vs base-rate {y.mean()*(1-y.mean()):.4f} (BSS {1 - bs['brier'] / (y.mean()*(1-y.mean())):+.1%})", fontsize=10); ax.legend(frameon=False, fontsize=8)
ax = axes[1]; dc = pd.read_parquet(RES / "daily_crosssection.parquet"); wd = E["within_day"]
ax.bar(["S4-only", "stock\nbase-rate", "TW-only", "Full"], [wd["S4-only"]["mean"], wd["stock base-rate"]["mean"], wd["TW-only"]["mean"], wd["Full"]["mean"]], color=[C["S4-only"], "#b8b5ad", C["TW-only"], C["Full"]], width=0.6)
for i, k in enumerate(["S4-only", "stock base-rate", "TW-only", "Full"]): ax.text(i, wd[k]["mean"] + 0.003, f"{wd[k]['mean']:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.47, 0.58); ax.set_ylabel(f"mean within-day AUROC ({E['within_day']['Full']['n_days']} test days)"); ax.set_title(f"Which stock, not which day: within-day ranking (Full−S4 t = {wd['Full_minus_S4_days']['t']:.1f}; rank IC {wd['IC Full']['mean']:.3f}, >0 on {wd['IC Full']['share_gt']:.0%} of days)", fontsize=9.5)
save(fig, "f11_calibration_withinday.png")

# ---------------- F12 CausalStock (no news) re-implementation ----------------
if (RES / "causalstock_lite.json").exists():
    CS = J("causalstock_lite.json"); KT = J("kdd17_tabular.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]; pub = CS["kdd17"]["published_table1"]
    names = [f"published: {k}" for k in pub] + ["ours: CausalStock-lite\n(no news, 5 seeds)", "ours: w/o causal graph\n(self-loops)", "ours: LightGBM (55 inputs)", "ours: LR"]
    accs = [pub[k][0] / 100 for k in pub] + [CS["kdd17"]["CausalStock-lite (no news)"]["mean"]["acc"], CS["kdd17"]["w/o TCD (self-loops only)"]["mean"]["acc"], KT["LightGBM"]["mean"]["acc"], KT["LR"]["acc"]]
    errs = [0] * len(pub) + [CS["kdd17"]["CausalStock-lite (no news)"]["std"]["acc"], CS["kdd17"]["w/o TCD (self-loops only)"]["std"]["acc"], KT["LightGBM"]["std"]["acc"], 0]
    cols = ["#b8b5ad"] * len(pub) + ["#4a3aa7", "#7f8fa6", "#2a78d6", "#2a78d6"]
    ax.barh(range(len(names)), accs, color=cols, xerr=errs, capsize=2); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    for i, v in enumerate(accs): ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    maj = max(KT["n"]["test_up_rate"], 1 - KT["n"]["test_up_rate"]); ax.axvline(maj, color="#d7263d", ls="--", lw=1); ax.text(maj + 0.002, len(names) - 0.6, f"majority class {maj:.3f}", fontsize=7.5, color="#d7263d")
    ax.set_xlim(0.46, 0.58); ax.set_xlabel(f"accuracy, KDD17 test 2016 ({KT['n']['test']:,} labelled instances)"); ax.set_title("KDD17 (price-only task): published CausalStock Table 1 vs. our runs", fontsize=10); ax.grid(axis="x"); ax.grid(False, axis="y")
    ax = axes[1]; yrs = [str(y) for y in TEST_YEARS]; tw = CS["tw50"]["seed_mean_by_year"]
    for name, col, mk in [("CausalStock-lite (no news)", "#4a3aa7", "o"), ("w/o TCD (self-loops only)", "#7f8fa6", "s")]:
        ax.plot(yrs, [tw[name][y]["auroc"] for y in yrs], marker=mk, color=col, lw=2, label=f"{name} (seed mean)")
    ax.plot(yrs, [A_("LightGBM|S1")["by_year"][y]["auroc"] for y in yrs], ls="--", color="#2a78d6", lw=1.5, label="LightGBM S1-only (price)")
    ax.plot(yrs, [A_("LightGBM|S4-only")["by_year"][y]["auroc"] for y in yrs], ls=":", color="#eda100", lw=1.5, label="LightGBM S4-only (overnight U.S.)")
    ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.47, 0.63); ax.set_ylabel("AUROC"); ax.set_title("Taiwan 50, same 11 price features, our walk-forward", fontsize=10); ax.legend(frameon=False, fontsize=7.5)
    save(fig, "f12_causalstock.png")

# ---------------- F13 CausalStock v2 (faithful re-implementation): label artifact ----------------
if (RES / "causalstock_v2.json").exists():
    V = J("causalstock_v2.json"); fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]; a = V["acl18_dclose"]; b = V["acl18_mov"]
    names = ["paper: CausalStock (with news)", "paper: w/o news", "paper: w/o TCD", "Hwang re-impl.: full (10 seeds)", "ours v2, alternative target (Δclose): with news", "ours v2, alternative target (Δclose): w/o news", "ours v2, alternative target (Δclose): w/o TCD",
             "ours v2, paper's up/down label: with news", "ours v2, paper's up/down label: w/o news", "ours v2, paper's up/down label: w/o TCD"]
    vals = [a["published"]["CausalStock"][0] / 100, a["published"]["w/o news"][0] / 100, a["published"]["w/o TCD"][0] / 100, a["hwang_reimplementation"]["full 10 seeds"][0] / 100,
            a["CausalStock v2 with news (cached DNE scores)"]["mean"]["acc"], a["CausalStock v2 w/o news"]["mean"]["acc"], a["w/o TCD"]["mean"]["acc"],
            b["CausalStock v2 with news (cached DNE scores)"]["mean"]["acc"], b["CausalStock v2 w/o news"]["mean"]["acc"], b["w/o TCD"]["mean"]["acc"]]
    errs = [0, 0, 0, 0, a["CausalStock v2 with news (cached DNE scores)"]["std"]["acc"], a["CausalStock v2 w/o news"]["std"]["acc"], a["w/o TCD"]["std"]["acc"], b["CausalStock v2 with news (cached DNE scores)"]["std"]["acc"], b["CausalStock v2 w/o news"]["std"]["acc"], b["w/o TCD"]["std"]["acc"]]
    cols = ["#b8b5ad"] * 3 + ["#7f8fa6"] + ["#d7263d"] * 3 + ["#4a3aa7"] * 3
    ax.barh(range(len(names)), vals, color=cols, xerr=errs, capsize=2); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.2); ax.invert_yaxis()
    for i, v in enumerate(vals): ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.axvline(b["majority_acc_test"], color="#999", ls="--", lw=1); ax.set_xlim(0.46, 0.70); ax.set_xlabel(f"accuracy, ACL18 test 2015-10 → 2016-01 ({b['n_test_instances']:,} instances)"); ax.set_title("ACL18: 0.64 is reached only with an alternative target (Δclose)", fontsize=10); ax.grid(axis="x"); ax.grid(False, axis="y")
    ax = axes[1]; k = V["kdd17"]; KT = J("kdd17_tabular.json")
    names = ["paper: CausalStock", "paper: DTML", "paper: Adv-ALSTM", "ours v2: price-only", "ours v2: w/o TCD", "ours: LightGBM (55 inputs)"]
    vals = [k["published"]["CausalStock"][0] / 100, k["published"]["DTML"][0] / 100, k["published"]["Adv-ALSTM"][0] / 100, k["CausalStock v2 (price-only)"]["mean"]["acc"], k["w/o TCD"]["mean"]["acc"], KT["LightGBM"]["mean"]["acc"]]
    errs = [0, 0, 0, k["CausalStock v2 (price-only)"]["std"]["acc"], k["w/o TCD"]["std"]["acc"], KT["LightGBM"]["std"]["acc"]]; cols = ["#b8b5ad"] * 3 + ["#4a3aa7"] * 2 + ["#2a78d6"]
    ax.barh(range(len(names)), vals, color=cols, xerr=errs, capsize=2); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    for i, v in enumerate(vals): ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(k["majority_acc_test"], color="#d7263d", ls="--", lw=1); ax.text(k["majority_acc_test"] + 0.002, 5.4, "majority", fontsize=7.5, color="#d7263d"); ax.set_xlim(0.46, 0.60); ax.set_xlabel("accuracy, KDD17 test 2016 (official labels)"); ax.set_title("KDD17 (price-only benchmark, labels verified)", fontsize=10); ax.grid(axis="x"); ax.grid(False, axis="y")
    ax = axes[2]; yrs = [str(y) for y in TEST_YEARS]; tw = V["tw50"]["seed_mean_by_year"]
    for name, col, mk in [("CausalStock v2 (price-only)", "#4a3aa7", "o"), ("w/o TCD", "#7f8fa6", "s")]:
        ax.plot(yrs, [tw[name][y]["auroc"] for y in yrs], marker=mk, color=col, lw=2, label=f"{name} (seed mean)")
    ax.plot(yrs, [A_("LightGBM|S1")["by_year"][y]["auroc"] for y in yrs], ls="--", color="#2a78d6", lw=1.5, label="LightGBM S1-only (price)")
    ax.plot(yrs, [A_("LightGBM|S4-only")["by_year"][y]["auroc"] for y in yrs], ls=":", color="#eda100", lw=1.5, label="LightGBM S4-only (overnight U.S.)")
    ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.47, 0.63); ax.set_ylabel("AUROC"); ax.set_title("Taiwan 50: same model and recipe, our walk-forward", fontsize=10); ax.legend(frameon=False, fontsize=7.5)
    save(fig, "f13_causalstock_v2.png")
# ---------------- F14 extended source study ----------------
if (RES / "source_study.json").exists():
    SS = J("source_study.json"); fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]; labs = [("y_up", "Close→Close"), ("y_gap", "Close→Open"), ("y_intra", "Open→Close"), ("y_rel", "beats TAIEX")]; news = ["S6", "S7", "S8", "S9"]; x = np.arange(len(labs)); w = 0.2
    for j, k in enumerate(news):
        v = [SS[l][f"Full+{k} minus Full"] for l, _ in labs]; ax.bar(x + (j - 1.5) * w, [a[0] + 0 for a in [[ (SS[l][f"Full+{k}"]["auroc"] - SS[l]["Full"]["auroc"]) ] for l, _ in labs]], w, yerr=[[ (SS[l][f"Full+{k}"]["auroc"] - SS[l]["Full"]["auroc"]) - SS[l][f"Full+{k} minus Full"][0] for l, _ in labs], [SS[l][f"Full+{k} minus Full"][1] - (SS[l][f"Full+{k}"]["auroc"] - SS[l]["Full"]["auroc"]) for l, _ in labs]], color=["#e87ba4", "#1baf7a", "#2a78d6", "#eda100"][j], capsize=2, error_kw=dict(lw=0.8), label=f"{k} {SS['sources'][k]}")
    ax.axhline(0, color="#999", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([n for _, n in labs]); ax.set_ylabel("ΔAUROC when added to Full (S1–S5)"); ax.set_title("Which external dataset adds to the five original sources? (LightGBM, 95 % block-bootstrap CI)", fontsize=9.5); ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    if "shapley9_y_up" in SS:
        ph = SS["shapley9_y_up"]["phi"]; ks = list(ph); ax.bar(ks, [ph[k] for k in ks], color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9b59b6", "#e87ba4", "#1baf7a", "#2a78d6", "#eda100"][:len(ks)])
        for i, k in enumerate(ks): ax.text(i, ph[k] + (0.002 if ph[k] >= 0 else -0.006), f"{ph[k]:+.3f}", ha="center", fontsize=7.5)
        ax.axhline(0, color="#999", lw=0.8); ax.set_title(f"Exact Shapley over 9 sources, Close→Close (sum = {sum(ph.values()):+.3f}, 511 subsets)", fontsize=9.5); ax.set_ylabel("ΔAUROC attributed")
    save(fig, "f14_source_study.png")

# ---------------- F16 Taiwan causal graph + news ----------------
if (RES / "causal_news_tw.json").exists() and (RES / "graph_features.json").exists():
    CN = J("causal_news_tw.json"); GF = J("graph_features.json"); TS = J("tw_structure.json"); full_lgb = J("main_results.json")["arms"]["LightGBM|Full"]["seed_mean_prob"]["pooled"]["auroc"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1.25, 0.85]})
    ax = axes[0]; P = CN["pooled_seed_mean"]; names = list(P); vals = [P[n]["auroc"] for n in names]; lo = [P[n]["auroc"] - P[n]["ci"][0] for n in names]; hi = [P[n]["ci"][1] - P[n]["auroc"] for n in names]
    cols = ["#2a78d6" if n.startswith("learned") else ("#eb6834" if n.startswith("prior") else "#1baf7a") for n in names]
    ax.barh(range(len(names)), vals, xerr=[lo, hi], color=cols, capsize=3, height=0.6); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
    for i, v in enumerate(vals): ax.text(0.472, i, f"{v:.3f}", va="center", fontsize=7.5, color="white")
    ax.axvline(0.5, color="#999", lw=0.8); ax.axvline(full_lgb, color="#d7263d", ls="--", lw=1, label=f"LightGBM, five sources: {full_lgb:.3f}"); ax.set_xlim(0.47, 0.62)
    ax.set_xlabel("pooled AUROC 2023–2026 (seed-mean probability, 95 % block-bootstrap CI)", fontsize=8); ax.set_title("(a) CausalStock v2 on Taiwan 50: graph source × news (3 seeds × 4 folds)", fontsize=9.5); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax = axes[1]; labs = [("y_up", "Close→Close"), ("y_gap", "Close→Open"), ("y_intra", "Open→Close"), ("y_rel", "beats TAIEX")]
    keys = [("Full+S10g minus Full", "Full + graph-price (S10g)", "#2a78d6"), ("Full+S10n minus Full", "Full + graph-news (S10n)", "#eb6834"), ("Full+S6+S10n minus Full+S6", "Full+S6 + graph-news (vs Full+S6)", "#eda100"), ("All9+S10 minus All9", "All nine + graph (vs All nine)", "#1baf7a")]
    x = np.arange(len(labs)); w = 0.2
    for j, (k, nm, c) in enumerate(keys):
        pt = []; e0 = []; e1 = []
        for l, _ in labs:
            a, b = k.split(" minus "); d = GF[l][a]["auroc"] - GF[l][b]["auroc"]; ci = GF[l][k]; pt.append(d); e0.append(d - ci[0]); e1.append(ci[1] - d)
        ax.bar(x + (j - 1.5) * w, pt, w, yerr=[e0, e1], color=c, capsize=2, label=nm)
    ax.axhline(0, color="#999", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([n for _, n in labs]); ax.set_ylabel("ΔAUROC"); ax.set_title("(b) Graph-derived features inside LightGBM (95 % block-bootstrap CI)", fontsize=9.5); ax.legend(frameon=False, fontsize=7)
    ax = axes[2]; yrs = sorted(CN["graph_stability"]); gl = [CN["graph_stability"][y]["mean_lift_vs_prior"] for y in yrs]; ll = [TS["overlap"][f"lasso_{y}"]["lift"] for y in yrs]; lg = [TS["overlap"][f"leadlag_{y}"]["lift"] if TS["overlap"][f"leadlag_{y}"].get("n", 0) else np.nan for y in yrs]
    x = np.arange(len(yrs)); ax.bar(x - 0.27, gl, 0.27, color="#2a78d6", label="learned TCD graph"); ax.bar(x, ll, 0.27, color="#1baf7a", label="training-window Lasso"); ax.bar(x + 0.27, lg, 0.27, color="#9b59b6", label="significant lead-lag pairs")
    ax.axhline(1, color="#999", lw=0.8, ls="--"); ax.set_xticks(x); ax.set_xticklabels(yrs); ax.set_ylabel("share of edges inside the prior graph ÷ chance"); ax.set_title("(c) Do data-driven graphs recover known business ties?", fontsize=9.5); ax.legend(frameon=False, fontsize=7)
    for i, y in enumerate(yrs): ax.text(i - 0.27, gl[i] + 0.03, f"{CN['graph_stability'][y]['mean_edges']:.0f}e", ha="center", fontsize=6.5); ax.text(i, ll[i] + 0.03, f"{TS['lasso']['n_edges'][y]}e", ha="center", fontsize=6.5)
    save(fig, "f16_causal_graph_tw.png")

# ---------------- F19 LLM look-ahead ----------------
if (RES / "llm_lookahead.json").exists():
    LK = J("llm_lookahead.json"); u = LK["univariate_by_year"]; yrs = list(u); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(yrs)); ax[0].bar(x - 0.2, [u[y]["auroc_llm"] for y in yrs], 0.4, color="#eb6834", label="LLM score (Qwen3-4B)"); ax[0].bar(x + 0.2, [u[y]["auroc_tfidf"] for y in yrs], 0.4, color="#2a78d6", label="TF-IDF score (fold-trained)")
    ax[0].axhline(0.5, color="#999", lw=0.8); ax[0].axvline(1.5, color="#d7263d", ls="--", lw=1); ax[0].text(1.55, 0.545, "LLM cut-off ≈ Mar 2025", color="#d7263d", fontsize=8); ax[0].set_xticks(x); ax[0].set_xticklabels(yrs); ax[0].set_ylim(0.45, 0.56); ax[0].set_ylabel("univariate AUROC, Close→Close (stock-days with news)"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("(a) Does the LLM read news better than a bag of n-grams?", fontsize=9.5)
    P1 = "up to 2025-03-31 (on or before the stated knowledge cut-off)"; P2 = "from 2025-04-30 (after the weights were released)"; labs = [("y_up", "Close→Close"), ("y_gap", "Close→Open"), ("y_intra", "Open→Close")]; x = np.arange(len(labs))
    for j, (P, nm, c) in enumerate([(P1, "to 2025-03-31 (≤ cut-off)", "#eb6834"), (P2, "from 2025-04-30 (post release)", "#1baf7a")]):
        d = [LK[l][P]["Full+LLM"] - LK[l][P]["Full"] for l, _ in labs]; ci = [LK[l][P]["Full+LLM minus Full"] for l, _ in labs]
        ax[1].bar(x + (j - 0.5) * 0.38, d, 0.38, yerr=[[a - b[0] for a, b in zip(d, ci)], [b[1] - a for a, b in zip(d, ci)]], capsize=3, color=c, label=nm)
    ax[1].axhline(0, color="#999", lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels([n for _, n in labs]); ax[1].set_ylabel("ΔAUROC: Full + LLM score − Full"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("(b) Is the LLM's gain larger inside its pre-training window?", fontsize=9.5)
    save(fig, "f19_llm_lookahead.png")

print("done")
