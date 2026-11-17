"""Multi-source ablation: single-source, leave-one-source-out, add-to-US, with bootstrap CIs.
Same walk-forward protocol as baseline.py (expanding window, test 2023-2026). Output: figures/source_ablation.json, figures/prelim_sources.png"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
import lightgbm as lgb
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
df = pd.read_parquet(ROOT / "data/processed/dataset.parquet")
df["sector_code"] = df.sector.astype("category").cat.codes
SRC = {
 "S1": ["ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ma5_ratio", "ma20_ratio", "ma60_ratio", "vol20", "volume_ratio20", "hl_range", "gap_open", "rsi14", "spread"],
 "S2": ["foreign_net_ratio", "foreign_net_ratio5", "trust_net_ratio", "trust_net_ratio5", "dealer_net_ratio", "dealer_net_ratio5", "inst_net_ratio", "margin_chg", "margin_chg5", "short_chg"],
 "S3": ["taiex_ret1", "taiex_ret5", "taiex_ma20_ratio", "taiex_vol_ratio20", "taiex_vol20", "rel_ret1"],
 "S4": ["us_sp500_ret", "us_nasdaq_ret", "us_sox_ret", "us_tsm_adr_ret", "us_dxy_ret", "us_nikkei_ret", "us_vix_level", "us_vix_chg", "us_usdtwd_chg", "us_10y_chg", "us_stale"],
 "S5": ["weekday", "month", "is_month_end"]}
ALL = sum(SRC.values(), [])
CONFIGS = {f"{k} only": v for k, v in SRC.items()}
CONFIGS["All (S1–S5)"] = ALL
for k in SRC: CONFIGS[f"All − {k}"] = [c for c in ALL if c not in SRC[k]]
for k in ["S1", "S2", "S3"]: CONFIGS[f"S4 + {k}"] = SRC["S4"] + SRC[k]
CONFIGS["S4 + S1 + S2 + S3"] = SRC["S4"] + SRC["S1"] + SRC["S2"] + SRC["S3"]
TEST = [2023, 2024, 2025, 2026]; te_mask = df.year.isin(TEST); y = df.loc[te_mask, "y_up"].values

def model(kind):
    if kind == "LR": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=42, n_jobs=4)
def wf(feats, kind):
    p = pd.Series(np.nan, index=df.index)
    for yr in TEST:
        tr, te = df[df.year < yr], df[df.year == yr]
        p.loc[te.index] = model(kind).fit(tr[feats], tr.y_up).predict_proba(te[feats])[:, 1]
    return p.loc[te_mask].values
res, P = {}, {}
for kind in ["LR", "LightGBM"]:
    for name, f in CONFIGS.items():
        p = wf(f, kind); P[(kind, name)] = p
        res[f"{kind}|{name}"] = dict(auroc=roc_auc_score(y, p), acc=accuracy_score(y, p > 0.5), mcc=matthews_corrcoef(y, p > 0.5), n_feat=len(f))
        print(f"{kind:8s} {name:20s} AUROC {res[f'{kind}|{name}']['auroc']:.4f} ACC {res[f'{kind}|{name}']['acc']:.4f}", flush=True)
rng = np.random.default_rng(42); idx = [rng.integers(0, len(y), len(y)) for _ in range(300)]
def ci(kind, a, b):
    pa, pb = P[(kind, a)], P[(kind, b)]; d = [roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]) for i in idx]
    return [round(roc_auc_score(y, pb) - roc_auc_score(y, pa), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
tests = {}
for kind in ["LR", "LightGBM"]:
    tests[kind] = {
        "All vs S4 only": ci(kind, "S4 only", "All (S1–S5)"),
        "S4+S1+S2+S3 vs S4 only": ci(kind, "S4 only", "S4 + S1 + S2 + S3"),
        **{f"drop {k}": ci(kind, f"All − {k}", "All (S1–S5)") for k in SRC},
        **{f"S4+{k} vs S4": ci(kind, "S4 only", f"S4 + {k}") for k in ["S1", "S2", "S3"]}}
    print(kind, json.dumps(tests[kind]))
json.dump({"results": res, "tests_auroc_gain_ci95": tests}, open(FIG / "source_ablation.json", "w"), indent=1)

# ---- figure ----
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 11})
C = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#eda100", "S5": "#4a3aa7"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 1]})
K = "LightGBM"; base_all = res[f"{K}|All (S1–S5)"]["auroc"]
# A: single source
names = ["S1 only", "S2 only", "S3 only", "S4 only", "S5 only"]; v = [res[f"{K}|{n}"]["auroc"] for n in names]
ax[0].barh(names[::-1], v[::-1], color=[C[n[:2]] for n in names[::-1]], height=0.6); ax[0].axvline(0.5, color="#898781", ls="--", lw=1); ax[0].axvline(base_all, color="#0b0b0b", ls=":", lw=1.2)
for i, x in enumerate(v[::-1]): ax[0].text(x + 0.002, i, f"{x:.3f}", va="center", fontsize=9)
ax[0].text(base_all, 4.45, f" all sources {base_all:.3f}", fontsize=8); ax[0].set_xlim(0.48, 0.62); ax[0].set_title("A. Each source alone (AUROC)")
# B: leave-one-out drop
names = [f"All − {k}" for k in SRC]; v = [tests[K][f"drop {k}"] for k in SRC]
ax[1].barh([f"remove {k}" for k in SRC][::-1], [x[0] for x in v][::-1], xerr=np.array([[x[0] - x[1], x[2] - x[0]] for x in v][::-1]).T, color=[C[k] for k in list(SRC)[::-1]], height=0.6, capsize=3, error_kw=dict(lw=0.8, ecolor="#52514e"))
ax[1].axvline(0, color="#898781", lw=1)
for i, x in enumerate(v[::-1]): ax[1].text(max(x[2], 0) + 0.002, i, f"{x[0]:+.4f}", va="center", fontsize=9)
ax[1].set_title("B. AUROC lost when one source is removed"); ax[1].set_xlim(-0.01, 0.075)
# C: add Taiwan sources on top of US
names = ["S4 only", "S4 + S1", "S4 + S2", "S4 + S3", "S4 + S1 + S2 + S3"]; v = [res[f"{K}|{n}"]["auroc"] for n in names]
cols = ["#eda100", "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b"]
ax[2].barh(names[::-1], v[::-1], color=cols[::-1], height=0.6)
for i, x in enumerate(v[::-1]): ax[2].text(x + 0.001, i, f"{x:.3f}", va="center", fontsize=9)
ax[2].set_xlim(0.56, 0.60); ax[2].set_title("C. Adding Taiwan sources to overnight U.S. (AUROC)")
fig.suptitle("Multi-source ablation, LightGBM, walk-forward test 2023–2026 (45,000 out-of-sample stock-days; whiskers = bootstrap 95 % CI)", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "prelim_sources.png", bbox_inches="tight"); print("figure ok")
