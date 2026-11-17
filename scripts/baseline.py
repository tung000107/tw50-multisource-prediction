"""Preliminary walk-forward experiment for the proposal.
Source ablation S1 -> +S2 -> +S3 -> +S4 -> +S5 with Logistic Regression and LightGBM,
rule baselines, per-year metrics, McNemar test (TW-only vs TW+US), bootstrap AUROC CI, sector breakdown.
Outputs: figures/prelim_ablation.png, figures/prelim_sector.png, figures/prelim_results.md, figures/prelim_results.json
"""
import json, warnings, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from scipy.stats import chi2
import lightgbm as lgb
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figures"
for f in ["Noto Sans CJK TC", "Noto Sans CJK HK"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.titleweight": "bold"})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]; GRAY = "#898781"
rng = np.random.default_rng(42)

df = pd.read_parquet(ROOT / "data/processed/dataset.parquet")
S1 = ["ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ma5_ratio", "ma20_ratio", "ma60_ratio", "vol20", "volume_ratio20", "hl_range", "gap_open", "rsi14", "spread"]
S2 = ["foreign_net_ratio", "foreign_net_ratio5", "trust_net_ratio", "trust_net_ratio5", "dealer_net_ratio", "dealer_net_ratio5", "inst_net_ratio", "margin_chg", "margin_chg5", "short_chg"]
S3 = ["taiex_ret1", "taiex_ret5", "taiex_ma20_ratio", "taiex_vol_ratio20", "taiex_vol20", "rel_ret1"]
S4 = ["us_sp500_ret", "us_nasdaq_ret", "us_sox_ret", "us_tsm_adr_ret", "us_dxy_ret", "us_nikkei_ret", "us_vix_level", "us_vix_chg", "us_usdtwd_chg", "us_10y_chg", "us_stale"]
S5 = ["weekday", "month", "is_month_end"]
CONFIGS = {"S1 price only": S1, "S1+S2 (+flows)": S1 + S2, "S1+S2+S3 (TW-only)": S1 + S2 + S3, "S1..S4 (TW+US)": S1 + S2 + S3 + S4, "S1..S5 (all)": S1 + S2 + S3 + S4 + S5, "S4 US only": S4}
TEST_YEARS = [2023, 2024, 2025, 2026]

def metrics(y, p, prob):
    return dict(acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p), f1=f1_score(y, p, average="macro"), auroc=roc_auc_score(y, prob), mcc=matthews_corrcoef(y, p))

def make_model(kind):
    if kind == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=42, n_jobs=4)

def walk_forward(feats, kind):
    """expanding-window: train on all years < test year; returns out-of-sample prob/pred aligned to df index."""
    prob = pd.Series(np.nan, index=df.index)
    for yr in TEST_YEARS:
        tr, te = df[df.year < yr], df[df.year == yr]
        m = make_model(kind).fit(tr[feats], tr.y_up)
        prob.loc[te.index] = m.predict_proba(te[feats])[:, 1]
    return prob

results, oos = {}, {}
for kind in ["LR", "LightGBM"]:
    for name, feats in CONFIGS.items():
        prob = walk_forward(feats, kind); oos[(kind, name)] = prob
        te = df[df.year.isin(TEST_YEARS)]; p = prob.loc[te.index]
        row = {"pooled": metrics(te.y_up, (p > 0.5).astype(int), p)}
        for yr in TEST_YEARS:
            t = te[te.year == yr]; row[str(yr)] = metrics(t.y_up, (p.loc[t.index] > 0.5).astype(int), p.loc[t.index])
        results[f"{kind} | {name}"] = row; print(f"{kind:8s} {name:22s} acc={row['pooled']['acc']:.4f} auroc={row['pooled']['auroc']:.4f} mcc={row['pooled']['mcc']:.4f}", flush=True)
# rule baselines
te = df[df.year.isin(TEST_YEARS)]
rules = {"Majority class (train)": np.zeros(len(te), int), "Previous-day sign": (te.ret1 > 0).astype(int).values, "Overnight SOX sign": (te.us_sox_ret > 0).astype(int).values, "Overnight S&P 500 sign": (te.us_sp500_ret > 0).astype(int).values}
for name, p in rules.items():
    results[f"Rule | {name}"] = {"pooled": dict(acc=accuracy_score(te.y_up, p), bal_acc=balanced_accuracy_score(te.y_up, p), f1=f1_score(te.y_up, p, average="macro"), auroc=float("nan"), mcc=matthews_corrcoef(te.y_up, p))}
    print(f"Rule     {name:22s} acc={results[f'Rule | {name}']['pooled']['acc']:.4f}")

# McNemar TW-only vs TW+US (LightGBM & LR), bootstrap AUROC difference
stats = {}
for kind in ["LR", "LightGBM"]:
    pa, pb = oos[(kind, "S1+S2+S3 (TW-only)")].loc[te.index], oos[(kind, "S1..S4 (TW+US)")].loc[te.index]
    ca, cb = ((pa > 0.5).astype(int) == te.y_up).values, ((pb > 0.5).astype(int) == te.y_up).values
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum()); stat = (abs(b - c) - 1) ** 2 / (b + c); pval = float(chi2.sf(stat, 1))
    y = te.y_up.values; diffs = []
    for _ in range(500):
        idx = rng.integers(0, len(y), len(y)); diffs.append(roc_auc_score(y[idx], pb.values[idx]) - roc_auc_score(y[idx], pa.values[idx]))
    stats[kind] = dict(mcnemar_b_only_TW_correct=b, mcnemar_c_only_TWUS_correct=c, mcnemar_chi2=float(stat), mcnemar_p=pval, auroc_gain=float(roc_auc_score(y, pb) - roc_auc_score(y, pa)), auroc_gain_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
    print(kind, stats[kind])

# sector breakdown (LightGBM)
pa, pb = oos[("LightGBM", "S1+S2+S3 (TW-only)")].loc[te.index], oos[("LightGBM", "S1..S4 (TW+US)")].loc[te.index]
sec = []
from universe import SECTOR_EN
for s, g in te.groupby(te.sector.map(SECTOR_EN)):
    sec.append(dict(sector=s, n=len(g), acc_tw=accuracy_score(g.y_up, (pa.loc[g.index] > 0.5).astype(int)), acc_twus=accuracy_score(g.y_up, (pb.loc[g.index] > 0.5).astype(int)), auroc_tw=roc_auc_score(g.y_up, pa.loc[g.index]), auroc_twus=roc_auc_score(g.y_up, pb.loc[g.index])))
sec = pd.DataFrame(sec).sort_values("auroc_twus", ascending=False); print(sec.round(4).to_string())

# ---------- figures ----------
cfgs = list(CONFIGS); x = np.arange(len(cfgs)); w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, met, lab, base in [(axes[0], "auroc", "AUROC (pooled 2023–2026 out-of-sample)", 0.5), (axes[1], "acc", "Accuracy (pooled 2023–2026 out-of-sample)", None)]:
    for j, kind in enumerate(["LR", "LightGBM"]):
        v = [results[f"{kind} | {c}"]["pooled"][met] for c in cfgs]
        ax.bar(x + (j - 0.5) * w, v, w, color=C[j], label=kind)
        for xi, vi in zip(x + (j - 0.5) * w, v): ax.text(xi, vi + 0.002, f"{vi:.3f}", ha="center", fontsize=7.2)
    if met == "acc":
        for k, (nm, key) in enumerate([("majority", "Majority class (train)"), ("prev-day sign", "Previous-day sign"), ("SOX sign", "Overnight SOX sign")]):
            v = results[f"Rule | {key}"]["pooled"]["acc"]; ax.axhline(v, color=[GRAY, C[3], C[2]][k], ls="--", lw=1, label=f"rule: {nm} = {v:.3f}")
        ax.set_ylim(0.44, 0.66)
    else:
        ax.axhline(0.5, color=GRAY, ls="--", lw=1); ax.set_ylim(0.45, 0.70)
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=15, ha="right", fontsize=8.5); ax.set_title(lab, fontsize=10.5); ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
fig.suptitle("Preliminary source-ablation results (walk-forward, expanding window, test = 2023, 2024, 2025, 2026)", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "prelim_ablation.png", bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
yrs = [str(y) for y in TEST_YEARS]
for ax, met, lab in [(axes[0], "auroc", "AUROC by test year (LightGBM)"), (axes[1], "mcc", "MCC by test year (LightGBM)")]:
    for j, c in enumerate(["S1 price only", "S1+S2+S3 (TW-only)", "S1..S4 (TW+US)", "S4 US only"]):
        ax.plot(yrs, [results[f"LightGBM | {c}"][y][met] for y in yrs], marker="o", lw=2, color=C[j], label=c)
    ax.axhline(0.5 if met == "auroc" else 0, color=GRAY, ls="--", lw=1); ax.set_title(lab, fontsize=10.5); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "prelim_by_year.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4)); xs = np.arange(len(sec))
ax.bar(xs - 0.19, sec.auroc_tw, 0.38, color=C[1], label="TW-only (S1+S2+S3)"); ax.bar(xs + 0.19, sec.auroc_twus, 0.38, color=C[0], label="TW + US (S1..S4)")
for i, r in enumerate(sec.itertuples()): ax.text(i, max(r.auroc_tw, r.auroc_twus) + 0.006, f"Δ {r.auroc_twus - r.auroc_tw:+.3f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(sec.sector); ax.set_ylim(0.45, 0.72); ax.axhline(0.5, color=GRAY, ls="--", lw=1); ax.set_ylabel("AUROC (2023–2026 OOS)")
ax.set_title("Gain from overnight US information by sector (LightGBM)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "prelim_sector.png", bbox_inches="tight"); plt.close(fig)

# ---------- markdown table ----------
lines = ["| Model | Feature set | Acc | Bal. Acc | Macro-F1 | AUROC | MCC |", "|---|---|---|---|---|---|---|"]
for k, r in results.items():
    m = r["pooled"]; kind, name = k.split(" | "); lines.append(f"| {kind} | {name} | {m['acc']:.4f} | {m['bal_acc']:.4f} | {m['f1']:.4f} | {'–' if np.isnan(m['auroc']) else f'{m['auroc']:.4f}'} | {m['mcc']:.4f} |")
lines += ["", "Per-year AUROC (LightGBM):", "", "| Feature set | " + " | ".join(yrs) + " |", "|---|" + "---|" * len(yrs)]
for c in cfgs: lines.append(f"| {c} | " + " | ".join(f"{results[f'LightGBM | {c}'][y]['auroc']:.4f}" for y in yrs) + " |")
lines += ["", "Sector breakdown (LightGBM, AUROC TW-only → TW+US):", "", sec.round(4).to_markdown(index=False), "", "Tests (TW-only vs TW+US):", "", "```", json.dumps(stats, indent=1), "```"]
(FIG / "prelim_results.md").write_text("\n".join(lines)); (FIG / "prelim_results.json").write_text(json.dumps({"results": results, "stats": stats, "sector": sec.to_dict("records")}, indent=1, default=float))
print("done")
