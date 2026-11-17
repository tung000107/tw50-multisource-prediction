"""Alternative targets with LightGBM (Full / S4-only / TW-only), same protocol:
  y_tr (dividend-adjusted), y_intra (open-to-close), y_gap, y_rel (relative to TAIEX), flat-excluded y_up,
  magnitude task |r| > k, four-level (k = median |r| in the training window), volatility-normalised four-level (z = r / vol20).
Output: results/final/labels.json, results/final/oos_labels.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score
from joblib import Parallel, delayed
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
ARM = {"Full": ALL, "S4-only": SRC["S4"], "TW-only": TW}
te_mask = df.year.isin(TEST_YEARS); bs = DayBootstrap(df.loc[te_mask, "date"].values, B=1000)
out = {"binary_variants": {}, "magnitude": {}, "four_level": {}}; oos = df.loc[te_mask, ["date", "stock_id", "year", "y_up"]].copy()
# ---- binary variants ----
df["y_up_noflat"] = df.y_up.where(df.ret_next != 0)
def one(lab, arm):
    p, _ = run_lgb(ARM[arm], 42, label=lab); return lab, arm, p
res = Parallel(n_jobs=6, prefer="threads")(delayed(one)(lab, arm) for lab in ["y_tr", "y_intra", "y_gap", "y_rel", "y_up_noflat"] for arm in ARM)
P = {}
for lab, arm, p in res: P[(lab, arm)] = p; oos[f"{lab}|{arm}"] = p.loc[te_mask].values
for lab in ["y_tr", "y_intra", "y_gap", "y_rel", "y_up_noflat"]:
    m = te_mask & df[lab].notna(); y = df.loc[m, lab].values; b = DayBootstrap(df.loc[m, "date"].values, B=1000)
    r = {"rate": float(y.mean()), "n": int(m.sum())}
    for arm in ARM:
        pa = P[(lab, arm)].loc[m].values; r[arm] = dict(auroc=auc_stat(y, pa), ci=b.ci(auc_stat, y, pa))
    r["Full_minus_S4"] = b.ci_diff(auc_stat, y, P[(lab, "S4-only")].loc[m].values, P[(lab, "Full")].loc[m].values)
    r["Full_minus_TW"] = b.ci_diff(auc_stat, y, P[(lab, "TW-only")].loc[m].values, P[(lab, "Full")].loc[m].values)
    out["binary_variants"][lab] = r; print(lab, {k: (round(v["auroc"], 4) if isinstance(v, dict) else v) for k, v in r.items() if k in ARM}, r["Full_minus_S4"], flush=True)
# ---- magnitude & four-level (fold-dependent k) ----
def four_level(r, k): return np.select([r <= -k, r <= 0, r <= k], [0, 1, 2], 3)
for variant in ["raw", "volnorm"]:
    df["_r"] = df.ret_next if variant == "raw" else df.ret_next / df.vol20.replace(0, np.nan)
    # FIX 2026-11-14: the labels of each fold are built inside that fold with that fold's own k (median |r| of its training window).
    # The first version wrote every fold's labels into one shared column before training, so later folds overwrote earlier ones and
    # all folds ended up with the k of the last fold (which had seen 2023-2025). The four k values differ by < 0.03 pp, but the rule is now what we state.
    klist = {str(yr): float(np.nanmedian(np.abs(df.loc[tr, "_r"]))) for yr, tr, itr, iva, te in FOLDS}
    def fold_labels(yr):
        k = klist[str(yr)]; r = df["_r"]; ok = r.notna()
        return pd.Series(np.where(ok, four_level(r.values, k), np.nan), index=df.index), pd.Series(np.where(ok, (r.abs().values > k).astype(float), np.nan), index=df.index)
    FL = {yr: fold_labels(yr) for yr, *_ in FOLDS}; y4 = pd.Series(np.nan, index=df.index); ymag = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, te in FOLDS: y4.loc[te] = FL[yr][0].loc[te]; ymag.loc[te] = FL[yr][1].loc[te]      # test labels: each test year with its own fold's k
    df["y4"] = y4; df["ymag"] = ymag
    res4 = {"k_by_test_year": klist}
    # magnitude binary (per-fold label)
    def run_mag(feats):
        prob = pd.Series(np.nan, index=df.index)
        for yr, tr, itr, iva, te in FOLDS:
            lab = FL[yr][1]; itr, iva, tr, te = [i[lab.loc[i].notna().values] for i in (itr, iva, tr, te)]
            m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(df.loc[itr, feats], lab.loc[itr], eval_set=[(df.loc[iva, feats], lab.loc[iva])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
            prob.loc[te] = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(df.loc[tr, feats], lab.loc[tr]).predict_proba(df.loc[te, feats])[:, 1]
        return prob
    for arm in ARM:
        p = run_mag(ARM[arm]); m = te_mask & df.ymag.notna(); y = df.loc[m, "ymag"].values
        res4[f"magnitude_{arm}"] = dict(auroc=auc_stat(y, p.loc[m].values), ci=bs.ci(auc_stat, y, p.loc[m].values)); oos[f"ymag_{variant}|{arm}"] = p.loc[te_mask].values
    # four-level multiclass LightGBM (early stopping on inner val, multi_logloss)
    probs4 = {}
    for arm in ARM:
        pr = np.full((len(df), 4), np.nan)
        for yr, tr, itr, iva, te in FOLDS:
            lab4 = FL[yr][0]; itr2, iva2, tr2, te2 = [i[lab4.loc[i].notna().values] for i in (itr, iva, tr, te)]
            m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, random_state=42, **LGB_PARAMS)
            m.fit(df.loc[itr2, ARM[arm]], lab4.loc[itr2].astype(int), eval_set=[(df.loc[iva2, ARM[arm]], lab4.loc[iva2].astype(int))], callbacks=[lgb.early_stopping(100, verbose=False)])
            f = lgb.LGBMClassifier(objective="multiclass", n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(df.loc[tr2, ARM[arm]], lab4.loc[tr2].astype(int))
            pr[te2] = f.predict_proba(df.loc[te2, ARM[arm]])
        probs4[arm] = pr
        m = te_mask & df.y4.notna(); y = df.loc[m, "y4"].astype(int).values; pm = pr[m.values]; pred = pm.argmax(1)
        d = {"macro_f1": f1_score(y, pred, average="macro"), "bal_acc": balanced_accuracy_score(y, pred), "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
             "binary_auroc_from_4": auc_stat((y >= 2).astype(int), pm[:, 2:].sum(1)), "macro_f1_ci": bs.ci(lambda a, b: f1_score(a, b.argmax(1), average="macro"), y, pm)}
        # strong vs mild within direction
        up = y >= 2; dn = y <= 1
        d["auroc_strong_vs_mild_up"] = auc_stat((y[up] == 3).astype(int), pm[up, 3] / (pm[up, 2] + pm[up, 3])); d["auroc_strong_vs_mild_down"] = auc_stat((y[dn] == 0).astype(int), pm[dn, 0] / (pm[dn, 0] + pm[dn, 1]))
        res4[f"four_{arm}"] = d; print(variant, arm, {k: round(v, 4) for k, v in d.items() if not isinstance(v, list)}, flush=True)
        for c in range(4): oos[f"y4_{variant}|{arm}|p{c}"] = pr[te_mask.values, c]
    m = te_mask & df.y4.notna(); y = df.loc[m, "y4"].astype(int).values
    res4["majority_macro_f1"] = f1_score(y, np.full(len(y), pd.Series(y).mode()[0]), average="macro")
    res4["Full_minus_S4_macro_f1"] = bs.ci_diff(lambda a, b: f1_score(a, b.argmax(1), average="macro"), y, probs4["S4-only"][m.values], probs4["Full"][m.values])
    res4["Full_minus_S4_qwk"] = bs.ci_diff(lambda a, b: cohen_kappa_score(a, b.argmax(1), weights="quadratic"), y, probs4["S4-only"][m.values], probs4["Full"][m.values])
    res4["class_share_test"] = pd.Series(y).value_counts(normalize=True).sort_index().round(4).tolist()
    oos[f"y4_{variant}"] = df.loc[te_mask, "y4"].values; out["four_level"][variant] = res4; print(variant, res4["Full_minus_S4_macro_f1"], flush=True)
oos.to_parquet(RES / "oos_labels.parquet"); dump(out, "labels.json")
# ---- per-year AUROC of the label variants (moved here from 29_review_fixes.py so that re-running this script keeps the fields)
L = json.load(open(RES / "labels.json")); lab_ = pd.read_parquet(RES / "oos_labels.parquet"); d0 = load()
for v in ["y_gap", "y_intra", "y_rel", "y_tr"]:
    m_ = lab_.merge(d0[["date", "stock_id", v]], on=["date", "stock_id"], how="left"); by = {}
    for yr in TEST_YEARS:
        mm = m_[(m_.year == yr) & m_[v].notna()]; by[str(yr)] = {a: auc_stat(mm[v].values, mm[f"{v}|{a}"].values) for a in ["Full", "S4-only", "TW-only"]}
    L["binary_variants"][v]["by_year"] = by; L["binary_variants"][v]["Full_minus_S4_same_sign_years"] = int(sum(np.sign(by[str(yr)]["Full"] - by[str(yr)]["S4-only"]) == np.sign(L["binary_variants"][v]["Full"]["auroc"] - L["binary_variants"][v]["S4-only"]["auroc"]) for yr in TEST_YEARS))
    print(v, L["binary_variants"][v]["Full_minus_S4_same_sign_years"])
dump(L, "labels.json"); print("by-year fields written")
