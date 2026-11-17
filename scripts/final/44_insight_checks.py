"""Cross-result checks behind the 'Key observations' / 'Insights' slides (no model fitting except one small logistic regression in E).
All on the 44,973 walk-forward test rows unless stated; intervals = trading-day block bootstrap (5-day blocks, B = 1,000).
 A  'which day' vs 'which stock': split every score into its day mean (the same value for all 50 stocks that day) and the deviation from it;
    pooled AUROC of each part; ceiling: AUROC of the realised share of up-stocks that day (leave-one-stock-out version too);
    share of the variance of y_up that lies between days; R^2 of the final model's day part on the night return and the SOX return
 B  within-day AUROC by year (final model, nine-source LightGBM) and the hold-out value
 C  continuation / reversal split (realised gap and intraday leg same / opposite sign) for the night-return rule, the SOX rule and the final model
 D  final model vs night rule where they disagree, with cut-offs matched to the same share of 'up' calls
 E  can a reversal day be predicted before the open? walk-forward logistic regression (train on earlier test years, score 2024-2026)
 F  'under-calls up': mean probability vs up-rate; recall of up / non-up at 0.5 and at the cut-off that calls 'up' as often as it happens (five vs nine)
 G  hold-out (31 days): day / stock parts for the five-source, Taiwan-only and five + S7 + S8 models
These are cuts chosen after seeing the results (exploratory), except A-B which restate the R1 / R5 numbers in another form.
Output: results/final/insight_checks.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from common import *
KEY = ["date", "stock_id"]; out = {}
FIN, L5, L9, E5 = "Ensemble LightGBM+LSTM All9 (ours, final)", "LightGBM Full (S1-S5)", "LightGBM All9 (S1-S9)", "Ensemble LightGBM+LSTM (S1-S5)"
P = pd.read_parquet(RES / "oos_sota_headtohead.parquet"); P["date"] = pd.to_datetime(P.date); P["stock_id"] = P.stock_id.astype(str)
df = load(); df["stock_id"] = df.stock_id.astype(str)
ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); ex["date"] = pd.to_datetime(ex.date); ex["stock_id"] = ex.stock_id.astype(str)
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); nf["date"] = pd.to_datetime(nf.date); nf["stock_id"] = nf.stock_id.astype(str)
P = P.merge(df[KEY + ["y_gap", "y_intra", "us_sox_ret", "us_vix_level", "vol20", "ret1", "weekday"]], on=KEY, how="left").merge(ex[KEY + ["tx_night_ret"]], on=KEY, how="left").merge(nf[KEY + ["n_news", "market_n_news"]], on=KEY, how="left")
P = P.sort_values(KEY).reset_index(drop=True); P["night rule"] = P.tx_night_ret; P["SOX rule"] = P.us_sox_ret
y = P.y_up.values.astype(int); bs = DayBootstrap(P.date.values, B=1000); g = P.groupby("date")
def aci(s): return dict(auroc=float(auc_stat(y, s)), ci=bs.ci(auc_stat, y, np.asarray(s)))
# ---------------------------------------------------------------- A
A = {}
for c in [FIN, L9, L5, "night rule", "SOX rule"]:
    day = g[c].transform("mean").values; dev = P[c].values - day
    A[c] = dict(full=aci(P[c].values), day_part=aci(day), stock_part=aci(dev) if np.nanstd(dev) > 0 else None)
    A[c]["day_share_of_gain"] = float((A[c]["day_part"]["auroc"] - 0.5) / (A[c]["full"]["auroc"] - 0.5))
share = g.y_up.transform("mean").values; n = g.y_up.transform("size").values; loo = (share * n - y) / (n - 1)
A["ceiling: realised up-share of the day"] = aci(share); A["ceiling: realised up-share of the other 49 stocks"] = aci(loo)
A["between_day_share_of_variance"] = float(np.var(share) / np.var(y))
dm = g[[FIN, "tx_night_ret", "us_sox_ret"]].mean().dropna()
def r2(cols): X = np.c_[np.ones(len(dm)), dm[cols].values]; b = np.linalg.lstsq(X, dm[FIN].values, rcond=None)[0]; return float(1 - ((dm[FIN].values - X @ b) ** 2).sum() / ((dm[FIN].values - dm[FIN].mean()) ** 2).sum())
A["final_day_part_R2"] = dict(night=r2(["tx_night_ret"]), sox=r2(["us_sox_ret"]), both=r2(["tx_night_ret", "us_sox_ret"]))
A["random_split_auroc_test_years"] = json.load(open(RES / "audits.json"))["random_5fold_full"]["auroc_test_years_only"]
out["A_day_vs_stock"] = A
print("A", {k: (round(v["full"]["auroc"], 3), round(v["day_part"]["auroc"], 3), v["stock_part"] and round(v["stock_part"]["auroc"], 3), round(v["day_share_of_gain"], 2)) for k, v in A.items() if isinstance(v, dict) and "full" in v})
print("  ceiling", round(A["ceiling: realised up-share of the day"]["auroc"], 3), round(A["ceiling: realised up-share of the other 49 stocks"]["auroc"], 3), "between-day var", round(A["between_day_share_of_variance"], 3), A["final_day_part_R2"], "random split", round(A["random_split_auroc_test_years"], 3))
# ---------------------------------------------------------------- B
def within(c, sub=None):
    q = P if sub is None else P[sub]; v = q.groupby("date").apply(lambda t: auc_stat(t.y_up.values, t[c].values) if t.y_up.nunique() > 1 else np.nan, include_groups=False).dropna(); return v
B = {}
for c in [FIN, L9, L5]:
    v = within(c); yr = pd.to_datetime(v.index).year
    B[c] = dict(mean=float(v.mean()), n_days=int(len(v)), by_year={str(t): float(v[yr == t].mean()) for t in TEST_YEARS})
H = pd.read_parquet(RES / "oos_fresh_holdout.parquet"); H["date"] = pd.to_datetime(H.date); hy = H.y_up.values.astype(int); hg = H.groupby("date")
B["holdout"] = {c: float(hg.apply(lambda t: auc_stat(t.y_up.values, t[c].values) if t.y_up.nunique() > 1 else np.nan, include_groups=False).dropna().mean()) for c in ["Full", "Full+S7+S8", "TW-only"]}
out["B_within_day_by_year"] = B; print("B", {k: (round(v["mean"], 3), {a: round(b, 3) for a, b in v["by_year"].items()}) if "mean" in v else v for k, v in B.items()})
# ---------------------------------------------------------------- C
P = P.merge(df[KEY + ["ret_gap", "ret_intra"]], on=KEY, how="left"); _sg = np.sign(P.ret_gap) * np.sign(P.ret_intra); rev = _sg < 0; con = _sg > 0   # strict signs; rows with a flat leg are in neither group
C = {"reversal_share": float(rev.mean())}
for c in [FIN, "night rule", "SOX rule"]: C[c] = dict(continuation=float(auc_stat(y[con.values], P.loc[con, c].values)), reversal=float(auc_stat(y[rev.values], P.loc[rev, c].values)))
out["C_continuation_reversal"] = C; print("C", C)
# ---------------------------------------------------------------- D
up_share = y.mean(); cut = lambda s: np.quantile(s, 1 - up_share)
mf, mr = P[FIN].values > cut(P[FIN].values), P["night rule"].values > cut(P["night rule"].values); dis = mf != mr
right_m = (mf[dis] == y[dis].astype(bool)); yrs = P.year.values
D = dict(disagree_share=float(dis.mean()), model_right_share=float(right_m.mean()), by_year={str(t): float(right_m[yrs[dis] == t].mean()) for t in TEST_YEARS})
q = P.drop_duplicates("date").set_index("date").tx_night_ret.abs(); t1 = q.quantile(1 / 3); D["share_of_disagreements_on_quiet_nights"] = float((P.loc[dis, "date"].map(q) <= t1).mean())
out["D_model_vs_night_rule_disagreements"] = D; print("D", D)
# ---------------------------------------------------------------- E
F_ = ["abs_night", "night", "us_sox_ret", "us_vix_level", "weekday", "ret1", "n_news", "market_n_news", "vol20"]; P["abs_night"] = P.tx_night_ret.abs(); P["night"] = P.tx_night_ret
Q = P[con | rev].copy(); Q["rev"] = rev[con | rev].astype(int).values; E = {"single_feature_auroc": {f: float(auc_stat(Q.rev.values, Q[f].fillna(Q[f].median()).values)) for f in F_}}
pr = pd.Series(np.nan, index=Q.index)
for t in [2024, 2025, 2026]:
    tr, te_ = Q.year < t, Q.year == t; X = Q[F_].fillna(Q.loc[tr, F_].median()); mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
    pr[te_] = LogisticRegression(max_iter=2000).fit(((X[tr] - mu) / sd).values, Q.rev[tr].values).predict_proba(((X[te_] - mu) / sd).values)[:, 1]
m = pr.notna().values; bq = DayBootstrap(Q.date.values[m], B=1000); E["walk_forward_logit_2024_2026"] = dict(auroc=float(auc_stat(Q.rev.values[m], pr.values[m])), ci=bq.ci(auc_stat, Q.rev.values[m], pr.values[m]), n=int(m.sum()))
E["single_feature_range"] = [min(max(v, 1 - v) for v in E["single_feature_auroc"].values()), max(max(v, 1 - v) for v in E["single_feature_auroc"].values())]
out["E_reversal_predictability"] = E; print("E", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in E.items() if k != "single_feature_auroc"})
# ---------------------------------------------------------------- F
Fo = {}
for nm, c in [("final nine-source model", FIN), ("five-source ensemble", E5), ("five-source LightGBM", L5)]:
    s = P[c].values; d05 = s > 0.5; dm_ = s > cut(s)
    rc = lambda d: dict(up=float(d[y == 1].mean()), non_up=float((~d)[y == 0].mean()), predicted_up_share=float(d.mean()))
    Fo[nm] = dict(mean_probability=float(s.mean()), up_rate=float(up_share), at_0_5=rc(d05), at_matched_cutoff=rc(dm_))
out["F_cutoff_and_recall"] = Fo; print("F", Fo)
# ---------------------------------------------------------------- G
G = {}
for c in ["Full", "TW-only", "Full+S7+S8", "S4-only"]:
    dayp = hg[c].transform("mean").values; dev = H[c].values - dayp
    G[c] = dict(full=float(auc_stat(hy, H[c].values)), day_part=float(auc_stat(hy, dayp)), stock_part=float(auc_stat(hy, dev)) if np.nanstd(dev) > 0 else None)
out["G_holdout_day_vs_stock"] = G; print("G", G)
dump(out, "insight_checks.json")
