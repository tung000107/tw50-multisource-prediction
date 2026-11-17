"""Four-level task in depth (strong down / mild down / mild up / strong up; k = median |r| of each fold's training window).
The first study (06_labels.py) reported Macro-F1 for three arms with one seed. This script asks what the four classes actually look like:
 A  three seeds x four arms (TW-only, S4-only, Full, All9 = nine sources): Macro-F1 mean and spread, paired block-bootstrap differences
    on seed-mean probabilities, per-year sign
 B  per-class precision / recall / F1 with CIs and the pooled confusion matrix (which class is predictable?)
 C  what a predicted class means in returns: realised mean next-day return, up-rate and share of |r| > k by predicted class, with CIs
 D  direct four-class model vs composing two binary models, P(up) x P(big move) (is a multiclass model needed?)
 E  two questions inside the four classes: direction (classes 2,3 vs 0,1) and size (classes 0,3 vs 1,2) AUROC per arm
Output: results/final/four_level_deep.json, results/final/oos_four_level_deep.parquet, figures/final/f24_four_level_deep.png"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
from joblib import Parallel, delayed
from common import *
KEY = ["date", "stock_id"]; NAMES = ["strong down", "mild down", "mild up", "strong up"]; SEEDS = [42, 1, 2]
df = load(); ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); ex["date"] = pd.to_datetime(ex["date"]); nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); nf["date"] = pd.to_datetime(nf["date"])
base_cols = set(df.columns); df = df.merge(ex, on=KEY, how="left").merge(nf, on=KEY, how="left")
NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"],
       "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"],
       "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
EXT = sum(NEW.values(), [])   # the same nine-source feature list as 33_sota_headtohead.py; the two text scores are fold-trained, so they are set inside each fold
def fold_text(yr): df["text_score"] = df[f"text_score_{yr}"]; df["market_text_score"] = df[f"market_text_score_{yr}"]
fold_text(TEST_YEARS[0]); assert all(c in df.columns for c in EXT), [c for c in EXT if c not in df.columns]
FOLDS = list(folds(df)); te = df.year.isin(TEST_YEARS); TW = SRC["S1"] + SRC["S2"] + SRC["S3"]
ARMS = {"TW-only": TW, "S4-only": SRC["S4"], "Full": ALL, "All9": ALL + EXT}; print("nine-source arm:", len(ARMS["All9"]), "features", flush=True)
def four_level(r, k): return np.select([r <= -k, r <= 0, r <= k], [0, 1, 2], 3)
K = {yr: float(np.nanmedian(np.abs(df.loc[tr, "ret_next"]))) for yr, tr, itr, iva, tst in FOLDS}
LAB = {yr: pd.Series(np.where(df.ret_next.notna(), four_level(df.ret_next.values, K[yr]), np.nan), index=df.index) for yr in K}
y4 = pd.Series(np.nan, index=df.index)
for yr, tr, itr, iva, tst in FOLDS: y4.loc[tst] = LAB[yr].loc[tst]
def fit4(arm, seed):
    pr = np.full((len(df), 4), np.nan); P = dict(LGB_PARAMS, n_jobs=6)
    for yr, tr, itr, iva, tst in FOLDS:
        lab = LAB[yr]; itr2, iva2, tr2, te2 = [i[lab.loc[i].notna().values] for i in (itr, iva, tr, tst)]
        X = df[ARMS[arm]]
        if arm == "All9": X = X.assign(text_score=df[f"text_score_{yr}"].values, market_text_score=df[f"market_text_score_{yr}"].values)   # this fold's text scores, without touching the shared frame
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, random_state=seed, **P).fit(X.loc[itr2], lab.loc[itr2].astype(int), eval_set=[(X.loc[iva2], lab.loc[iva2].astype(int))], callbacks=[lgb.early_stopping(100, verbose=False)])
        pr[df.index.get_indexer(te2)] = lgb.LGBMClassifier(objective="multiclass", n_estimators=int(m.best_iteration_ or 1), random_state=seed, **P).fit(X.loc[tr2], lab.loc[tr2].astype(int)).predict_proba(X.loc[te2])
    return arm, seed, pr
def fitbin(target, seed=42):   # two binary models on the Full arm: direction (up vs non-up... here classes 2,3) and size (classes 0,3), per-fold labels
    prob = pd.Series(np.nan, index=df.index); P = dict(LGB_PARAMS, n_jobs=6)
    for yr, tr, itr, iva, tst in FOLDS:
        lab4 = LAB[yr]; lab = (lab4 >= 2).astype(float) if target == "dir" else lab4.isin([0, 3]).astype(float); lab[lab4.isna()] = np.nan
        itr2, iva2, tr2, te2 = [i[lab.loc[i].notna().values] for i in (itr, iva, tr, tst)]
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **P).fit(df.loc[itr2, ALL], lab.loc[itr2], eval_set=[(df.loc[iva2, ALL], lab.loc[iva2])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        prob.loc[te2] = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=seed, **P).fit(df.loc[tr2, ALL], lab.loc[tr2]).predict_proba(df.loc[te2, ALL])[:, 1]
    return target, prob
res = Parallel(n_jobs=4, prefer="threads")(delayed(fit4)(a, s) for a in ARMS for s in SEEDS); PR = {(a, s): p for a, s, p in res}
bins = dict(Parallel(n_jobs=2, prefer="threads")(delayed(fitbin)(t) for t in ["dir", "size"]))
mk = (te & y4.notna()).values; y = y4[mk].astype(int).values; D = df.loc[mk, KEY + ["year", "ret_next"]].reset_index(drop=True); bs = DayBootstrap(D.date.values, B=1000); yrs = D.year.values
mf = lambda t, p: f1_score(t, p.argmax(1), average="macro"); out = {"k_by_test_year": {str(k): v for k, v in K.items()}, "n_test": int(mk.sum()), "class_share_test": [float((y == c).mean()) for c in range(4)], "seeds": SEEDS, "n_features": {a: len(f) for a, f in ARMS.items()}}
SM = {a: np.mean([PR[(a, s)][mk] for s in SEEDS], axis=0) for a in ARMS}
# ---- A
A = {}
for a in ARMS:
    per = [float(mf(y, PR[(a, s)][mk])) for s in SEEDS]; ci = bs.ci(mf, y, SM[a])
    A[a] = dict(macro_f1_per_seed=per, seed_mean=float(np.mean(per)), seed_std=float(np.std(per, ddof=1)), macro_f1_seed_mean_prob=float(mf(y, SM[a])), ci=ci[:2], by_year={int(t): float(mf(y[yrs == t], SM[a][yrs == t])) for t in TEST_YEARS},
                qwk=float(cohen_kappa_score(y, SM[a].argmax(1), weights="quadratic")), accuracy=float((SM[a].argmax(1) == y).mean()))
def pdiff(a, b):
    ci = bs.ci_diff(mf, y, SM[a], SM[b]); by = {int(t): float(mf(y[yrs == t], SM[b][yrs == t]) - mf(y[yrs == t], SM[a][yrs == t])) for t in TEST_YEARS}; d = float(mf(y, SM[b]) - mf(y, SM[a]))
    return dict(diff=d, ci=ci[:2], by_year=by, same_sign_years=int(sum(np.sign(v) == np.sign(d) for v in by.values())), per_seed=[float(mf(y, PR[(b, s)][mk]) - mf(y, PR[(a, s)][mk])) for s in SEEDS])
A["paired"] = {"Full minus S4-only": pdiff("S4-only", "Full"), "Full minus TW-only": pdiff("TW-only", "Full"), "All9 minus Full": pdiff("Full", "All9")}
A["majority_macro_f1"] = float(f1_score(y, np.full(len(y), np.bincount(y).argmax()), average="macro")); out["A_macro_f1"] = A
print("A", {a: (round(A[a]["seed_mean"], 4), round(A[a]["seed_std"], 4)) for a in ARMS}, {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"]) for k, v in A["paired"].items()}, flush=True)
# ---- B per-class metrics + confusion matrix
B = {}
for a in ARMS:
    pred = SM[a].argmax(1); p_, r_, f_, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], zero_division=0); cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    B[a] = dict(precision=dict(zip(NAMES, map(float, p_))), recall=dict(zip(NAMES, map(float, r_))), f1=dict(zip(NAMES, map(float, f_))), predicted_share=dict(zip(NAMES, [float((pred == c).mean()) for c in range(4)])),
                confusion_rows_true_cols_pred=cm.tolist(), confusion_row_normalised=(cm / cm.sum(1, keepdims=True)).round(4).tolist(),
                f1_ci={NAMES[c]: bs.ci(lambda t, p, c=c: f1_score(t == c, p.argmax(1) == c, zero_division=0), y, SM[a])[:2] for c in range(4)})
out["B_per_class"] = B; print("B", {a: {k: round(v, 3) for k, v in B[a]["f1"].items()} for a in ARMS}, flush=True)
# ---- C what a predicted class means
C = {}; r = D.ret_next.values; kk = np.array([K[int(t)] for t in yrs])
for a in ["Full", "All9", "S4-only", "TW-only"]:
    pred = SM[a].argmax(1); C[a] = {}
    for c in range(4):
        m = pred == c
        if m.sum() < 50: C[a][NAMES[c]] = dict(n=int(m.sum())); continue
        b2 = DayBootstrap(D.date.values[m], B=1000)
        C[a][NAMES[c]] = dict(n=int(m.sum()), share=float(m.mean()), mean_return=float(r[m].mean()), mean_return_ci=b2.ci(lambda x: float(x.mean()), r[m])[:2], up_rate=float((r[m] > 0).mean()), up_rate_ci=b2.ci(lambda x: float((x > 0).mean()), r[m])[:2],
                               big_move_rate=float((np.abs(r[m]) > kk[m]).mean()), mean_abs_return=float(np.abs(r[m]).mean()), true_class_share=[float((y[m] == j).mean()) for j in range(4)])
    C[a]["all"] = dict(mean_return=float(r.mean()), up_rate=float((r > 0).mean()), big_move_rate=float((np.abs(r) > kk).mean()), mean_abs_return=float(np.abs(r).mean()))
    su, sd = pred == 3, pred == 0
    if su.sum() > 50 and sd.sum() > 50:
        ds = pd.DataFrame({"date": D.date.values, "r": r, "su": su, "sd": sd}); g = ds.groupby("date"); sp = (g.apply(lambda x: x.r[x.su].mean()) - g.apply(lambda x: x.r[x.sd].mean())).dropna()
        C[a]["strong_up_minus_strong_down_daily"] = dict(n_days=int(len(sp)), mean=float(sp.mean()), t=float(sp.mean() / sp.std(ddof=1) * np.sqrt(len(sp)))) if len(sp) > 1 else dict(n_days=int(len(sp)), mean=None, t=None)   # U.S.-only gives one class per day: no day has both
out["C_predicted_class_outcomes"] = C; print("C", {a: {n: (C[a][n].get("n"), round(C[a][n].get("mean_return", np.nan) * 100, 3), round(C[a][n].get("up_rate", np.nan), 3), round(C[a][n].get("big_move_rate", np.nan), 3)) for n in NAMES} for a in C}, flush=True)
# ---- D direct four-class vs two binary models composed
pu, pb = bins["dir"][mk].values, bins["size"][mk].values; comp = np.stack([(1 - pu) * pb, (1 - pu) * (1 - pb), pu * (1 - pb), pu * pb], axis=1); direct = PR[("Full", 42)][mk]
out["D_direct_vs_composed"] = dict(note="Full arm, seed 42 for both; composed = P(direction) x P(size) from two binary LightGBM models, assuming the two are independent given the features",
    direct_macro_f1=float(mf(y, direct)), composed_macro_f1=float(mf(y, comp)), composed_minus_direct=(lambda ci: dict(diff=float(mf(y, comp) - mf(y, direct)), ci=ci[:2]))(bs.ci_diff(mf, y, direct, comp)),
    composed_per_class_f1=dict(zip(NAMES, map(float, precision_recall_fscore_support(y, comp.argmax(1), labels=[0, 1, 2, 3], zero_division=0)[2]))), direct_per_class_f1=dict(zip(NAMES, map(float, precision_recall_fscore_support(y, direct.argmax(1), labels=[0, 1, 2, 3], zero_division=0)[2]))))
print("D", out["D_direct_vs_composed"]["direct_macro_f1"], out["D_direct_vs_composed"]["composed_macro_f1"], out["D_direct_vs_composed"]["composed_minus_direct"], flush=True)
# ---- E the two questions inside the four classes
E = {}
for a in ARMS:
    p = SM[a]; E[a] = dict(direction_auroc=float(auc_stat((y >= 2).astype(int), p[:, 2] + p[:, 3])), direction_ci=bs.ci(auc_stat, (y >= 2).astype(int), p[:, 2] + p[:, 3])[:2], size_auroc=float(auc_stat(np.isin(y, [0, 3]).astype(int), p[:, 0] + p[:, 3])), size_ci=bs.ci(auc_stat, np.isin(y, [0, 3]).astype(int), p[:, 0] + p[:, 3])[:2])
out["E_direction_and_size"] = E; print("E", {a: (round(v["direction_auroc"], 4), round(v["size_auroc"], 4)) for a, v in E.items()}, flush=True)
dump(out, "four_level_deep.json")
o = D.copy(); o["y4"] = y
for a in ARMS:
    for c in range(4): o[f"{a}|p{c}"] = SM[a][:, c]
o.to_parquet(RES / "oos_four_level_deep.parquet")
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
