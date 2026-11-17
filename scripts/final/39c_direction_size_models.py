"""Is a four-class model needed, or is 'direction' x 'size' better handled as two questions?  (follow-up of 39_four_level_deep.py, part D,
which used one seed).  Five-source arm, three seeds, same folds and the same per-fold labels (k = median |r| of the fold's training window).
 direct        the four-class LightGBM of 39 (seed-mean probabilities, read from oos_four_level_deep.parquet)
 independent   P(up) x P(big move): two binary models, multiplied as if direction and size were independent given the features
 conditional   P(up) x P(big | up) and P(down) x P(big | down): the size model is fitted separately on up-days and on down-days,
               so no independence is assumed
Output: results/final/direction_size_models.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, precision_recall_fscore_support
from joblib import Parallel, delayed
from common import *
KEY = ["date", "stock_id"]; NAMES = ["strong down", "mild down", "mild up", "strong up"]; SEEDS = [42, 1, 2]
df = load(); FOLDS = list(folds(df)); te = df.year.isin(TEST_YEARS)
def four_level(r, k): return np.select([r <= -k, r <= 0, r <= k], [0, 1, 2], 3)
K = {yr: float(np.nanmedian(np.abs(df.loc[tr, "ret_next"]))) for yr, tr, itr, iva, tst in FOLDS}
LAB = {yr: pd.Series(np.where(df.ret_next.notna(), four_level(df.ret_next.values, K[yr]), np.nan), index=df.index) for yr in K}
def fitbin(target, seed):
    """target: 'dir' = up classes vs down classes; 'size' = strong vs mild on all rows; 'size_up' / 'size_dn' = strong vs mild fitted on up-days / down-days only"""
    prob = pd.Series(np.nan, index=df.index); P = dict(LGB_PARAMS, n_jobs=5)
    for yr, tr, itr, iva, tst in FOLDS:
        l4 = LAB[yr]
        if target == "dir": lab = (l4 >= 2).astype(float); keep = l4.notna()
        elif target == "size": lab = l4.isin([0, 3]).astype(float); keep = l4.notna()
        elif target == "size_up": lab = (l4 == 3).astype(float); keep = l4 >= 2
        else: lab = (l4 == 0).astype(float); keep = l4 <= 1
        itr2, iva2, tr2 = [i[keep.loc[i].values] for i in (itr, iva, tr)]
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **P).fit(df.loc[itr2, ALL], lab.loc[itr2], eval_set=[(df.loc[iva2, ALL], lab.loc[iva2])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        prob.loc[tst] = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=seed, **P).fit(df.loc[tr2, ALL], lab.loc[tr2]).predict_proba(df.loc[tst, ALL])[:, 1]   # scored on every test row
    return (target, seed), prob
jobs = [(t, s) for t in ["dir", "size", "size_up", "size_dn"] for s in SEEDS]; R = dict(Parallel(n_jobs=4, prefer="threads")(delayed(fitbin)(t, s) for t, s in jobs))
D = pd.read_parquet(RES / "oos_four_level_deep.parquet"); D["date"] = pd.to_datetime(D.date); D["stock_id"] = D.stock_id.astype(str); base = df.loc[te, KEY].copy(); base["stock_id"] = base.stock_id.astype(str)
for (t, s), p in R.items(): base[f"{t}|{s}"] = p.loc[te].values
D = D.merge(base, on=KEY, how="left"); y = D.y4.astype(int).values; bs = DayBootstrap(D.date.values, B=1000); yrs = D.year.values
def indep(s): pu, pb = D[f"dir|{s}"].values, D[f"size|{s}"].values; return np.stack([(1 - pu) * pb, (1 - pu) * (1 - pb), pu * (1 - pb), pu * pb], axis=1)
def cond(s): pu, bu, bd = D[f"dir|{s}"].values, D[f"size_up|{s}"].values, D[f"size_dn|{s}"].values; return np.stack([(1 - pu) * bd, (1 - pu) * (1 - bd), pu * (1 - bu), pu * bu], axis=1)
PR = {"direct four-class": D[[f"Full|p{c}" for c in range(4)]].values, "independent: P(up) x P(big)": np.mean([indep(s) for s in SEEDS], axis=0), "conditional: P(up) x P(big | direction)": np.mean([cond(s) for s in SEEDS], axis=0)}
mf = lambda t, p: f1_score(t, p.argmax(1), average="macro"); out = dict(arm="five sources (S1-S5)", seeds=SEEDS, n_test=int(len(D)), methods={})
for nm, p in PR.items():
    f = precision_recall_fscore_support(y, p.argmax(1), labels=[0, 1, 2, 3], zero_division=0)[2]
    out["methods"][nm] = dict(macro_f1=float(mf(y, p)), ci=bs.ci(mf, y, p)[:2], per_class_f1=dict(zip(NAMES, map(float, f))), predicted_share=dict(zip(NAMES, [float((p.argmax(1) == c).mean()) for c in range(4)])), by_year={int(t): float(mf(y[yrs == t], p[yrs == t])) for t in TEST_YEARS},
                              per_seed=[float(mf(y, (indep if nm.startswith("indep") else cond)(s))) for s in SEEDS] if not nm.startswith("direct") else None)
def pdiff(a, b):
    ci = bs.ci_diff(mf, y, PR[a], PR[b]); by = {int(t): float(mf(y[yrs == t], PR[b][yrs == t]) - mf(y[yrs == t], PR[a][yrs == t])) for t in TEST_YEARS}; d = float(mf(y, PR[b]) - mf(y, PR[a]))
    return dict(diff=d, ci=ci[:2], by_year=by, same_sign_years=int(sum(np.sign(v) == np.sign(d) for v in by.values())))
ks = list(PR); out["paired"] = {"independent minus direct": pdiff(ks[0], ks[1]), "conditional minus direct": pdiff(ks[0], ks[2]), "conditional minus independent": pdiff(ks[1], ks[2])}
# does the size of the move depend on direction once the features are known? compare the two conditional size scores
out["size_given_direction"] = dict(corr_p_big_given_up_vs_given_down=float(np.corrcoef(np.mean([D[f"size_up|{s}"].values for s in SEEDS], axis=0), np.mean([D[f"size_dn|{s}"].values for s in SEEDS], axis=0))[0, 1]),
    auroc_big_among_true_up_days={"P(big | up) model": float(auc_stat((y[y >= 2] == 3).astype(int), np.mean([D[f"size_up|{s}"].values for s in SEEDS], axis=0)[y >= 2])), "pooled P(big) model": float(auc_stat((y[y >= 2] == 3).astype(int), np.mean([D[f"size|{s}"].values for s in SEEDS], axis=0)[y >= 2]))},
    auroc_big_among_true_down_days={"P(big | down) model": float(auc_stat((y[y <= 1] == 0).astype(int), np.mean([D[f"size_dn|{s}"].values for s in SEEDS], axis=0)[y <= 1])), "pooled P(big) model": float(auc_stat((y[y <= 1] == 0).astype(int), np.mean([D[f"size|{s}"].values for s in SEEDS], axis=0)[y <= 1]))})
print({k: (round(v["macro_f1"], 4), [round(x, 4) for x in v["ci"]], {a: round(b, 3) for a, b in v["per_class_f1"].items()}, v["per_seed"]) for k, v in out["methods"].items()})
print({k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]], v["same_sign_years"]) for k, v in out["paired"].items()}); print(out["size_given_direction"])
dump(out, "direction_size_models.json")
