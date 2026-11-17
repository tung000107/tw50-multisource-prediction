"""EDA of the external / extension sources S6-S9 (the first EDA covered S1-S5 only; the course checklist asks for EDA of every dataset used).
 A  per feature: missing share, zero share, percentiles, extreme values; Spearman rho with next-day direction (y_up) and with the size of
    the move (|ret_next|), computed on all rows and, for comparison, on the rows before 2023 only (training side)
 B  up-rate by quintile of the main feature of each source (TX night return next to the SOX return; news count; fold-trained text score on
    its own test year; foreign futures position change; day-trading share; cross-sectional rank; revenue YoY)
 C  coverage: news headlines by year and by stock; revenue availability by year; drift of each source between training rows and test rows
 D  redundancy: day-level correlation of the TX night return with every S4 series; for every new feature the largest |rho| with any S1-S5 feature
 E  extreme values: rows whose one-day return exceeds the +-10 % price limit (unadjusted corporate actions) and what they do to S7
Output: results/final/external_eda.json, figures/final/f25_external_eda.png   (no model fitting)"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from common import *
KEY = ["date", "stock_id"]; out = {}
df = load(); df["stock_id"] = df.stock_id.astype(str)
ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); ex["date"] = pd.to_datetime(ex.date); ex["stock_id"] = ex.stock_id.astype(str)
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); nf["date"] = pd.to_datetime(nf.date); nf["stock_id"] = nf.stock_id.astype(str)
d = df.merge(ex, on=KEY, how="left").merge(nf, on=KEY, how="left"); te = d.year.isin(TEST_YEARS).values
# the text score of a row is the one fitted for its own test year (out of sample); rows before 2023 have no out-of-sample score
d["text_score"] = np.nan; d["market_text_score"] = np.nan
for yr in TEST_YEARS: m = (d.year == yr).values; d.loc[m, "text_score"] = d.loc[m, f"text_score_{yr}"]; d.loc[m, "market_text_score"] = d.loc[m, f"market_text_score_{yr}"]
NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"], "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"], "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
NAMES = {"S6": "S6 news headlines", "S7": "S7 cross-stock", "S8": "S8 TX futures", "S9": "S9 day-trading, lending, revenue"}; FEATS = sum(NEW.values(), []); src_of = {f: s for s, fs in NEW.items() for f in fs}
OOS_ONLY = {"text_score", "market_text_score"}
def rho(a, b):
    m = np.isfinite(a) & np.isfinite(b); return float(spearmanr(a[m], b[m]).statistic) if m.sum() > 100 and np.nanstd(a[m]) > 0 else None
# ---------------------------------------------------------------- A
A = {}; absr = d.ret_next.abs().values; yu = d.y_up.values.astype(float)
for f in FEATS:
    rows = te if f in OOS_ONLY else np.ones(len(d), bool); x = d[f].values.astype(float); xs = x[rows]; q = np.nanpercentile(xs, [1, 25, 50, 75, 99])
    A[f] = dict(source=src_of[f], rows_used="test rows 2023-2026 (out-of-sample score)" if f in OOS_ONLY else "all rows", n=int(np.isfinite(xs).sum()), missing_share=float(np.isnan(xs).mean()), zero_share=float((xs == 0).mean()),
                p1=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p99=float(q[4]), min=float(np.nanmin(xs)), max=float(np.nanmax(xs)), mean=float(np.nanmean(xs)), std=float(np.nanstd(xs)),
                rho_direction=rho(x[rows], yu[rows]), rho_size=rho(x[rows], absr[rows]), rho_direction_train_rows=None if f in OOS_ONLY else rho(x[~te], yu[~te]), rho_direction_test_rows=rho(x[te], yu[te]))
out["A_feature_summary"] = A
ref = {f: rho(d[f].values.astype(float), yu) for f in SRC["S4"]}; out["A_reference_S4_rho_direction"] = ref
tw = {f: rho(d[f].values.astype(float), yu) for f in SRC["S1"] + SRC["S2"] + SRC["S3"]}; out["A_reference_S1_S3_max_abs_rho_direction"] = float(max(abs(v) for v in tw.values() if v is not None))
print("A  rho with direction:", {f: round(v["rho_direction"], 3) for f, v in A.items()}); print("   rho with size:", {f: round(v["rho_size"], 3) for f, v in A.items()})
# ---------------------------------------------------------------- B
def by_bins(x, bins, labels, rows=None):
    g = d if rows is None else d[rows]; b = pd.cut(x if rows is None else x[rows], bins, labels=labels, include_lowest=True)
    t = g.groupby(b, observed=True).agg(n=("y_up", "size"), up_rate=("y_up", "mean"), median_abs_ret=("ret_next", lambda s: s.abs().median())); return {str(k): dict(n=int(r.n), up_rate=float(r.up_rate), median_abs_ret=float(r.median_abs_ret)) for k, r in t.iterrows()}
def quint(f, rows=None, day_level=False):
    g = d if rows is None else d[rows]; base = g.drop_duplicates("date")[f] if day_level else g[f]; e = base.quantile([0.2, 0.4, 0.6, 0.8]).values
    if len(np.unique(e)) < 4: return None
    return dict(edges=[float(v) for v in e], bins=by_bins(d[f], [-np.inf, *e, np.inf], ["Q1", "Q2", "Q3", "Q4", "Q5"], rows))
B = {"tx_night_ret (S8), day-level quintiles": quint("tx_night_ret", day_level=True), "us_sox_ret (S4, reference), day-level quintiles": quint("us_sox_ret", day_level=True),
     "n_news (S6)": by_bins(d.n_news, [-0.5, 0.5, 1.5, 3.5, 1e9], ["0", "1", "2-3", "4+"]), "text_score (S6), test rows with a headline, own-year score": quint("text_score", rows=te & (d.has_news == 1).values),
     "foreign_fut_net_oi_chg (S8), day-level quintiles": quint("foreign_fut_net_oi_chg", day_level=True), "daytrade_share (S9)": quint("daytrade_share"), "ret1_rank (S7)": quint("ret1_rank"), "rev_yoy (S9)": quint("rev_yoy"), "lend_log (S9)": quint("lend_log")}
out["B_up_rate_by_bin"] = B
for k, v in B.items(): print("B", k, [round(x["up_rate"], 3) for x in (v["bins"] if v and "bins" in v else v).values()] if v else None)
# ---------------------------------------------------------------- C
C = {"news_by_year": {}, "revenue_available_by_year": {}, "drift_train_vs_test": {}}
for yr, g in d.groupby("year"):
    C["news_by_year"][str(yr)] = dict(stock_days=int(len(g)), share_with_headline=float(g.has_news.mean()), mean_headlines_per_stock_day=float(g.n_news.mean()), mean_market_headlines_per_day=float(g.drop_duplicates("date").market_n_news.mean()))
    C["revenue_available_by_year"][str(yr)] = float(g.rev_yoy.notna().mean())
ps = d.groupby("stock_id").agg(share=("has_news", "mean"), mean_n=("n_news", "mean"), name=("name", "first")).sort_values("share")
C["news_by_stock"] = dict(min=float(ps.share.min()), p25=float(ps.share.quantile(0.25)), median=float(ps.share.median()), p75=float(ps.share.quantile(0.75)), max=float(ps.share.max()), stocks_below_20pct=int((ps.share < 0.2).sum()), stocks_above_80pct=int((ps.share > 0.8).sum()),
                           lowest=[dict(stock_id=k, share=float(r.share), mean_headlines=float(r.mean_n)) for k, r in ps.head(5).iterrows()], highest=[dict(stock_id=k, share=float(r.share), mean_headlines=float(r.mean_n)) for k, r in ps.tail(5).iterrows()])
C["news_test_rows_share_with_headline"] = float(d.loc[te, "has_news"].mean()); C["news_all_rows_share_with_headline"] = float(d.has_news.mean())
top5 = float(d.groupby("stock_id").n_news.sum().sort_values(ascending=False).head(5).sum() / d.n_news.sum()); C["news_share_of_headlines_in_top5_stocks"] = top5
for f in FEATS:
    if f in OOS_ONLY: continue
    a, b = d.loc[~te, f], d.loc[te, f]; sd = float(d[f].std()); C["drift_train_vs_test"][f] = dict(train_mean=float(a.mean()), test_mean=float(b.mean()), train_std=float(a.std()), test_std=float(b.std()), shift_in_pooled_sd=float((b.mean() - a.mean()) / sd) if sd > 0 else None)
out["C_coverage_and_drift"] = C; print("C news by year", {k: (round(v["share_with_headline"], 3), round(v["mean_headlines_per_stock_day"], 2)) for k, v in C["news_by_year"].items()}); print("C news by stock", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in C["news_by_stock"].items() if k not in ("lowest", "highest")}, "top5 share", round(top5, 3))
print("C revenue", {k: round(v, 3) for k, v in C["revenue_available_by_year"].items()}); print("C drift > 0.3 sd", {k: round(v["shift_in_pooled_sd"], 2) for k, v in C["drift_train_vs_test"].items() if v["shift_in_pooled_sd"] is not None and abs(v["shift_in_pooled_sd"]) > 0.3})
# ---------------------------------------------------------------- D
day = d.drop_duplicates("date").set_index("date"); D = {"night_vs_S4_day_level": {}}
for f in SRC["S4"]:
    if f == "us_stale": continue
    m = day[[f, "tx_night_ret"]].dropna(); D["night_vs_S4_day_level"][f] = dict(spearman=float(spearmanr(m[f], m.tx_night_ret).statistic), pearson=float(np.corrcoef(m[f], m.tx_night_ret)[0, 1]), n_days=int(len(m)))
m = day[["us_sox_ret", "tx_night_ret"]].dropna(); D["night_vs_sox_sign_agreement"] = float((np.sign(m.us_sox_ret) == np.sign(m.tx_night_ret)).mean()); fresh = day.us_stale != 1
D["night_vs_sox_spearman_fresh_us_days"] = float(spearmanr(day.loc[fresh, "us_sox_ret"], day.loc[fresh, "tx_night_ret"]).statistic); D["night_vs_sox_spearman_stale_us_days"] = float(spearmanr(day.loc[~fresh, "us_sox_ret"], day.loc[~fresh, "tx_night_ret"]).statistic); D["stale_us_days"] = int((~fresh).sum())
gap = d.groupby("date").ret_gap.mean(); mm = pd.concat([day.tx_night_ret, day.us_sox_ret, gap], axis=1).dropna(); D["day_mean_open_gap_vs"] = dict(tx_night_ret=float(spearmanr(mm.tx_night_ret, mm.ret_gap).statistic), us_sox_ret=float(spearmanr(mm.us_sox_ret, mm.ret_gap).statistic))
R = d[ALL + [f for f in FEATS if f not in OOS_ONLY]].sample(30000, random_state=0).rank().corr(); D["max_abs_rho_with_S1_S5"] = {}
for f in FEATS:
    if f in OOS_ONLY: continue
    s = R.loc[f, ALL].abs().sort_values(ascending=False); D["max_abs_rho_with_S1_S5"][f] = dict(feature=s.index[0], source=[k for k, v in SRC.items() if s.index[0] in v][0], abs_rho=float(s.iloc[0]))
out["D_redundancy"] = D; print("D night vs S4", {k: round(v["spearman"], 3) for k, v in D["night_vs_S4_day_level"].items()}, "sign agreement", round(D["night_vs_sox_sign_agreement"], 3), "gap", D["day_mean_open_gap_vs"])
print("D max |rho| with S1-S5", {k: (v["feature"], round(v["abs_rho"], 2)) for k, v in D["max_abs_rho_with_S1_S5"].items()})
# ---------------------------------------------------------------- E
big = d[d.ret1.abs() > 0.105]; E = dict(price_limit=0.10, rows_with_abs_ret1_above_10_5pct=int(len(big)), share_of_rows=float(len(big) / len(d)), in_test_rows=int((big.year.isin(TEST_YEARS)).sum()), largest=[dict(date=str(r.date.date()), stock_id=r.stock_id, ret1=float(r.ret1)) for r in big.reindex(big.ret1.abs().sort_values(ascending=False).index).head(8).itertuples()],
         max_abs_ret1_minus_sector=float(d.ret1_minus_sector.abs().max()), rows_abs_ret1_minus_sector_above_0_2=int((d.ret1_minus_sector.abs() > 0.2).sum()), rows_abs_sector_ret1_ex_above_0_12=int((d.sector_ret1_ex.abs() > 0.12).sum()),
         tx_night_volume_ratio_above_5=int((d.drop_duplicates("date").tx_night_volume_ratio > 5).sum()), rev_yoy_abs_above_5=int((d.rev_yoy.abs() > 5).sum()), note="one-day returns beyond the +-10 % limit come from unadjusted corporate actions (ex-dividend, capital reduction); they enter the S1 return features and the S7 features built from them; tree models are insensitive to the size of an outlier, LR / sequence inputs are scaled with training-window statistics")
out["E_extreme_values"] = E; print("E", {k: v for k, v in E.items() if k not in ("largest", "note")}); print("  largest", E["largest"][:5])
dump(out, "external_eda.json")
# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 11.5}); CL = {"S6": "#e87ba4", "S7": "#1baf7a", "S8": "#2a78d6", "S9": "#eda100"}
fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.7), gridspec_kw={"width_ratios": [1.0, 1.0, 1.75, 1.0]})
a = ax[0]; qn = ["Q1", "Q2", "Q3", "Q4", "Q5"]; vn = [B["tx_night_ret (S8), day-level quintiles"]["bins"][q]["up_rate"] * 100 for q in qn]; vs = [B["us_sox_ret (S4, reference), day-level quintiles"]["bins"][q]["up_rate"] * 100 for q in qn]; x = np.arange(5)
a.bar(x - 0.2, vn, 0.38, color=CL["S8"], label="TX night return (S8)"); a.bar(x + 0.2, vs, 0.38, color="#b9b4c9", label="SOX return (S4)")
for i in range(5): a.text(i - 0.2, vn[i] + 0.6, f"{vn[i]:.0f}", ha="center", fontsize=8.5); a.text(i + 0.2, vs[i] + 0.6, f"{vs[i]:.0f}", ha="center", fontsize=8.5)
a.axhline(d.y_up.mean() * 100, color="#777", lw=0.9, ls="--"); a.set_ylim(25, 70); a.set_xticks(x); a.set_xticklabels(["lowest", "Q2", "Q3", "Q4", "highest"]); a.set_ylabel("next-day up-rate, % (axis starts at 25)"); a.legend(frameon=False, fontsize=8.5, loc="upper left")
a.set_title("(a) Up-rate by quintile of the overnight signal", fontsize=10.2, fontweight="bold")
a = ax[1]; yrs = sorted(C["news_by_year"]); sh = [C["news_by_year"][y]["share_with_headline"] * 100 for y in yrs]; a.bar([y[2:] for y in yrs], sh, color=CL["S6"]); a.set_xlabel("year (20xx)")
for i, v in enumerate(sh): a.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=8.5)
a.set_ylim(0, 100); a.set_ylabel("stock-days with ≥ 1 tagged headline, %"); a.set_title("(b) News coverage by year (S6)", fontsize=10.2, fontweight="bold")
a.text(0.02, 0.97, f"per stock: {C['news_by_stock']['min'] * 100:.0f} % – {C['news_by_stock']['max'] * 100:.0f} % (median {C['news_by_stock']['median'] * 100:.0f} %)\ntop-5 stocks hold {top5 * 100:.0f} % of the headlines", transform=a.transAxes, va="top", fontsize=8.5, color="#333")
a = ax[2]; fs = FEATS; v = [A[f]["rho_direction"] for f in fs]; a.bar(range(len(fs)), v, color=[CL[src_of[f]] for f in fs]); a.axhline(0, color="#999", lw=0.9)
mx = max(abs(x) for x in ref.values() if x is not None); a.axhline(mx, color="#555", lw=0.9, ls="--"); a.axhline(-mx, color="#555", lw=0.9, ls="--"); a.text(len(fs) - 0.4, mx + 0.004, f"strongest S4 feature: |ρ| = {mx:.2f}", ha="right", fontsize=8.5, color="#333")
a.axhspan(-0.05, 0.05, color="#eee", zorder=0); a.set_xticks(range(len(fs))); a.set_xticklabels(fs, rotation=60, ha="right", fontsize=7.4); a.set_ylabel("Spearman ρ with next-day direction"); a.set_title("(c) Each new feature vs next-day direction (grey band: |ρ| < 0.05)", fontsize=10.2, fontweight="bold")
for s in NEW: a.bar([0], [0], color=CL[s], label=NAMES[s])
a.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, fontsize=8.8, ncol=1, loc="upper left")
a = ax[3]; m = day[["us_sox_ret", "tx_night_ret"]].dropna(); a.scatter(m.us_sox_ret * 100, m.tx_night_ret * 100, s=5, alpha=0.35, color=CL["S8"]); a.axhline(0, color="#bbb", lw=0.7); a.axvline(0, color="#bbb", lw=0.7)
a.set_xlim(-8, 8); a.set_ylim(-5, 5); a.set_xlabel("overnight SOX return, %"); a.set_ylabel("TX night-session return, %"); a.set_title("(d) Night session vs SOX, per day", fontsize=10.2, fontweight="bold")
a.text(0.03, 0.97, f"Spearman ρ = {D['night_vs_S4_day_level']['us_sox_ret']['spearman']:.2f}\nsame sign on {D['night_vs_sox_sign_agreement'] * 100:.0f} % of days", transform=a.transAxes, va="top", fontsize=9, color="#333")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "f25_external_eda.png", dpi=170, facecolor="white"); print("figure ok")
