"""Tables asked for by the course's 'common issues' checklist (EDA completeness, split comparison, class-wise results, failure modes).
 A  split comparison: for each walk-forward fold, the training window, the inner validation and the test year side by side
    (rows, days, up-rate, flat share, median |r|, volatility, U.S.-TW sign agreement, missing cells)
 B  missingness: share of missing cells per source and the features with the most missing values, original and extension sources,
    training rows vs test rows
 C  class-wise results on the binary task: confusion matrix, precision / recall / F1 of 'up' and 'non-up' at threshold 0.5,
    for five-source LightGBM and the final nine-source model (seed-mean probabilities)
 D  where the final model fails: AUROC and accuracy by year, sector, size of the overnight signal, VIX level, U.S.-holiday days,
    and 'reversal' rows where the overnight gap and the intraday leg have opposite signs; what the most confident errors look like
Output: results/final/eda_and_errors.json   (no model fitting)"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from common import *
KEY = ["date", "stock_id"]; out = {}
df = load(); df["stock_id"] = df.stock_id.astype(str); ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); ex["date"] = pd.to_datetime(ex.date); ex["stock_id"] = ex.stock_id.astype(str)
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); nf["date"] = pd.to_datetime(nf.date); nf["stock_id"] = nf.stock_id.astype(str); d = df.merge(ex, on=KEY, how="left").merge(nf, on=KEY, how="left")
# S6 has seven model features: five counts plus two text scores that are fitted per fold (a row uses the score of its own test year; rows before 2023 use the 2023 fit)
for c in ["text_score", "market_text_score"]:
    d[c] = d[f"{c}_2023"]
    for yr in TEST_YEARS: d.loc[d.year == yr, c] = d.loc[d.year == yr, f"{c}_{yr}"]
NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"], "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"], "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
GR = {**SRC, **NEW}; FOLDS = list(folds(df)); te = df.year.isin(TEST_YEARS)
# ---------------------------------------------------------------- A split comparison
def desc(ix):
    g = d.loc[ix]; day = g.drop_duplicates("date")
    return dict(first=str(g.date.min().date()), last=str(g.date.max().date()), rows=int(len(g)), days=int(g.date.nunique()), up_rate=float(g.y_up.mean()), flat_share=float((g.ret_next == 0).mean()), median_abs_ret=float(g.ret_next.abs().median()),
                mean_vol20=float(g.vol20.mean()), us_tw_sign_agreement=float((np.sign(day.us_sox_ret) == np.sign(day.taiex_ret_next if "taiex_ret_next" in day else day.ret_next)).mean()) if "us_sox_ret" in day else None,
                missing_cells_S1_S5=float(g[ALL].isna().mean().mean()), missing_cells_S6_S9=float(g[sum(NEW.values(), [])].isna().mean().mean()))
A = {}
for yr, tr, itr, iva, tst in FOLDS: A[str(yr)] = dict(train=desc(tr), inner_validation=desc(iva), test=desc(tst))
out["A_split_comparison"] = A; print("A", {y: (v["train"]["rows"], v["test"]["rows"], round(v["train"]["up_rate"], 3), round(v["test"]["up_rate"], 3), round(v["train"]["median_abs_ret"], 4), round(v["test"]["median_abs_ret"], 4)) for y, v in A.items()})
# ---------------------------------------------------------------- B missingness
B = {"by_source": {}, "top_features": []}
for s, fs in GR.items():
    B["by_source"][s] = dict(n_features=len(fs), all_rows=float(d[fs].isna().mean().mean()), train_rows_before_2023=float(d.loc[~te.values, fs].isna().mean().mean()), test_rows=float(d.loc[te.values, fs].isna().mean().mean()), rows_with_any_missing=float(d[fs].isna().any(axis=1).mean()))
m = d[sum(GR.values(), [])].isna().mean().sort_values(ascending=False); src_of = {f: s for s, fs in GR.items() for f in fs}
B["top_features"] = [dict(feature=f, source=src_of[f], missing_share=float(v), test_rows=float(d.loc[te.values, f].isna().mean())) for f, v in m.head(8).items()]
B["handling"] = "tree models use LightGBM / XGBoost native missing handling; LR uses training-window medians; sequence models standardise with training-window statistics and set missing values to the training mean (all fitted inside each training window)"
out["B_missingness"] = B; print("B", {s: round(v["all_rows"], 4) for s, v in B["by_source"].items()}, [(x["feature"], round(x["missing_share"], 3)) for x in B["top_features"][:5]])
# ---------------------------------------------------------------- C / D
P = pd.read_parquet(RES / "oos_sota_headtohead.parquet"); P["stock_id"] = P.stock_id.astype(str); P["date"] = pd.to_datetime(P.date)
cols = ["sector_en", "y_gap", "y_intra", "ret_next", "us_sox_ret", "us_vix_level", "us_stale", "tx_night_ret", "vol20"]; P = P.merge(d[KEY + cols], on=KEY, how="left")
# five vs nine with the same model (mean of LightGBM and LSTM); the first version compared a five-source LightGBM with the nine-source ensemble
FIN, LF5 = "Ensemble LightGBM+LSTM All9 (ours, final)", "Ensemble LightGBM+LSTM (S1-S5)"; y = P.y_up.values.astype(int)
C = {}
for nm, c in [("five-source ensemble", LF5), ("final nine-source model", FIN), ("five-source LightGBM", "LightGBM Full (S1-S5)")]:
    pred = (P[c].values > 0.5).astype(int); cm = confusion_matrix(y, pred, labels=[0, 1]); p_, r_, f_, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    C[nm] = dict(threshold=0.5, confusion_rows_true_cols_pred={"true non-up": cm[0].tolist(), "true up": cm[1].tolist()}, predicted_up_share=float(pred.mean()), true_up_share=float(y.mean()), accuracy=float((pred == y).mean()),
                 non_up=dict(precision=float(p_[0]), recall=float(r_[0]), f1=float(f_[0])), up=dict(precision=float(p_[1]), recall=float(r_[1]), f1=float(f_[1])), probability_range=[float(np.percentile(P[c], 1)), float(np.percentile(P[c], 99))])
out["C_class_wise_binary"] = C; print("C", {k: (round(v["accuracy"], 3), v["non_up"], v["up"], round(v["predicted_up_share"], 3)) for k, v in C.items()})
def grp(mask, c=FIN):
    mask = np.asarray(mask) & P[c].notna().values; g = P[mask]
    if len(g) < 200 or g.y_up.nunique() < 2: return dict(n=int(len(g)))
    bs = DayBootstrap(g.date.values, B=500); return dict(n=int(len(g)), days=int(g.date.nunique()), auroc=float(auc_stat(g.y_up.values, g[c].values)), ci=bs.ci(auc_stat, g.y_up.values, g[c].values)[:2], accuracy=float(((g[c].values > 0.5).astype(int) == g.y_up.values).mean()), up_rate=float(g.y_up.mean()))
D = {"by_year": {str(t): grp(P.year == t) for t in TEST_YEARS}, "by_sector": {s: grp(P.sector_en == s) for s in sorted(P.sector_en.dropna().unique())}}
an = P.tx_night_ret.abs(); q = an.quantile([1 / 3, 2 / 3]).values; D["by_size_of_night_move"] = {"small |night return|": grp(an <= q[0]), "medium": grp((an > q[0]) & (an <= q[1])), "large |night return|": grp(an > q[1])}; D["night_move_cut_points"] = [float(x) for x in q]
vq = P.us_vix_level.quantile([1 / 3, 2 / 3]).values; D["by_vix"] = {"low VIX": grp(P.us_vix_level <= vq[0]), "mid VIX": grp((P.us_vix_level > vq[0]) & (P.us_vix_level <= vq[1])), "high VIX": grp(P.us_vix_level > vq[1])}
D["by_us_session"] = {"fresh U.S. session": grp(P.us_stale != 1), "no new U.S. session (U.S. holiday)": grp(P.us_stale == 1)}
# strict signs (fixed 2026-11-17): the first version compared the up / non-up codings, so a flat gap or a flat intraday leg counted as a sign change
P = P.merge(d[KEY + ["ret_gap", "ret_intra"]], on=KEY, how="left"); sg_ = np.sign(P.ret_gap) * np.sign(P.ret_intra)
rev = (sg_ < 0).values; con = (sg_ > 0).values; zero = (sg_ == 0).values
D["gap_vs_intraday"] = {"continuation (gap and intraday leg same sign)": grp(con), "reversal (opposite signs)": grp(rev)}; D["reversal_share"] = float(rev.mean()); D["continuation_share"] = float(con.mean()); D["zero_leg_share"] = float(zero.mean())
D["five_source_same_groups"] = {"continuation": grp(con, LF5), "reversal": grp(rev, LF5)}
# the most confident errors of the final model
conf = (P[FIN] - 0.5).abs(); top = conf >= conf.quantile(0.8); wrong = ((P[FIN] > 0.5).astype(int) != P.y_up); tw = top & wrong
D["most_confident_fifth"] = dict(n=int(top.sum()), error_rate=float(wrong[top].mean()), share_of_errors_that_are_reversal_rows=float(rev[tw].mean()), reversal_share_among_all_rows=float(rev.mean()),
    errors_called_up=float((P.loc[tw, FIN] > 0.5).mean()), mean_abs_next_return_of_errors=float(P.loc[tw, "ret_next"].abs().mean()), mean_abs_next_return_all=float(P.ret_next.abs().mean()),
    worst_days=[dict(date=str(k.date()), errors=int(v), night_return=float(P.loc[P.date == k, "tx_night_ret"].iloc[0]), mean_next_return=float(P.loc[P.date == k, "ret_next"].mean())) for k, v in P[tw].groupby("date").size().sort_values(ascending=False).head(5).items()])
out["D_failure_modes_final_model"] = D
print("D year", {k: (v.get("auroc") and round(v["auroc"], 3), v.get("accuracy") and round(v["accuracy"], 3)) for k, v in D["by_year"].items()}); print("D sector", {k: v.get("auroc") and round(v["auroc"], 3) for k, v in D["by_sector"].items()})
print("D night", {k: (v["n"], round(v["auroc"], 3), round(v["accuracy"], 3)) for k, v in D["by_size_of_night_move"].items()}); print("D vix", {k: round(v["auroc"], 3) for k, v in D["by_vix"].items()}); print("D us", {k: (v["n"], v.get("auroc") and round(v["auroc"], 3)) for k, v in D["by_us_session"].items()})
print("D rev", {k: (v["n"], round(v["auroc"], 3), round(v["accuracy"], 3)) for k, v in D["gap_vs_intraday"].items()}, D["reversal_share"], D["most_confident_fifth"])
dump(out, "eda_and_errors.json")
