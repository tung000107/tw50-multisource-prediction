"""Why is the corrected TX night-session return so much stronger than the U.S. series (and than the first, misaligned version)?
Market-day level analysis (1,818 days): correlations with the next overnight gap / next-day return / next intraday leg / today's own return,
R2 of the gap on the night return vs on five U.S. series, what each explains that the other does not, sign agreement, and rule AUROCs on days
with a fresh U.S. session vs days that reuse an older U.S. session (U.S. holidays).   Output: results/final/night_session.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from common import *
df = load(); ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); ex["date"] = pd.to_datetime(ex.date); old = pd.read_parquet(ROOT / "data/processed/extra_features_v1_misaligned_night.parquet"); old["date"] = pd.to_datetime(old.date)
d = df.merge(ex[["date", "stock_id", "tx_night_ret"]], on=["date", "stock_id"]).merge(old[["date", "stock_id", "tx_night_ret"]].rename(columns={"tx_night_ret": "night_first_version"}), on=["date", "stock_id"])
day = d.groupby("date").agg(night=("tx_night_ret", "first"), night_first_version=("night_first_version", "first"), sox=("us_sox_ret", "first"), sp500=("us_sp500_ret", "first"), nasdaq=("us_nasdaq_ret", "first"), tsm_adr=("us_tsm_adr_ret", "first"), vix_chg=("us_vix_chg", "first"),
                            next_ret=("ret_next", "mean"), gap=("ret_gap", "mean"), intraday=("ret_intra", "mean"), today=("ret1", "mean")).dropna()
US = ["sox", "sp500", "nasdaq", "tsm_adr", "vix_chg"]; r2 = lambda X, y: float(LinearRegression().fit(X, y).score(X, y)); out = {"n_days": int(len(day))}
out["correlation"] = {c: dict(next_overnight_gap=float(day[c].corr(day.gap)), next_day_return=float(day[c].corr(day.next_ret)), next_intraday_leg=float(day[c].corr(day.intraday)), todays_own_return=float(day[c].corr(day.today))) for c in ["night", "night_first_version", "sox", "sp500", "nasdaq", "tsm_adr"]}
out["r2_of_next_overnight_gap"] = dict(night=r2(day[["night"]], day.gap), five_us_series=r2(day[US], day.gap), both=r2(day[US + ["night"]], day.gap)); out["r2_of_next_day_return"] = dict(night=r2(day[["night"]], day.next_ret), five_us_series=r2(day[US], day.next_ret), both=r2(day[US + ["night"]], day.next_ret))
out["r2_of_night_on_five_us_series"] = r2(day[US], day.night)
rn = day.night - LinearRegression().fit(day[US], day.night).predict(day[US]); rs = day.sox - LinearRegression().fit(day[["night"]], day.sox).predict(day[["night"]])
out["what_each_adds"] = dict(part_of_night_not_explained_by_us=dict(corr_gap=float(rn.corr(day.gap)), corr_next_day_return=float(rn.corr(day.next_ret))), part_of_sox_not_explained_by_night=dict(corr_gap=float(rs.corr(day.gap)), corr_next_day_return=float(rs.corr(day.next_ret))))
out["sign_agreement_with_next_day_market_return"] = dict(night=float((np.sign(day.night) == np.sign(day.next_ret)).mean()), sox=float((np.sign(day.sox) == np.sign(day.next_ret)).mean()))
full = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet"); full["date"] = pd.to_datetime(full.date); z = full.drop_duplicates("date").sort_values("date").set_index("date"); reuse = d.date.map(z.us_date == z.us_date.shift(1)).fillna(False).values
te = d.year.isin(TEST_YEARS).values; y = d.y_up.values; bs = DayBootstrap(d.loc[te, "date"].values, B=1000)
out["rule_auroc_test_years"] = dict(night_return_as_score=dict(auroc=float(auc_stat(y[te], d.tx_night_ret.values[te])), ci=bs.ci(auc_stat, y[te], d.tx_night_ret.values[te])[:2]), sox_return_as_score=dict(auroc=float(auc_stat(y[te], d.us_sox_ret.fillna(0).values[te])), ci=bs.ci(auc_stat, y[te], d.us_sox_ret.fillna(0).values[te])[:2]),
    night_minus_sox=(lambda ci: dict(diff=float(auc_stat(y[te], d.tx_night_ret.values[te]) - auc_stat(y[te], d.us_sox_ret.fillna(0).values[te])), ci=ci[:2]))(bs.ci_diff(auc_stat, y[te], d.us_sox_ret.fillna(0).values[te], d.tx_night_ret.values[te])),
    first_version_as_score=float(auc_stat(y[te], d.night_first_version.values[te])))
out["by_us_session"] = {nm: dict(n=int(m.sum()), night=float(auc_stat(y[m], d.tx_night_ret.values[m])), sox=float(auc_stat(y[m], d.us_sox_ret.fillna(0).values[m]))) for nm, m in [("fresh U.S. session", te & ~reuse), ("U.S. session reused (U.S. holiday)", te & reuse)]}
# final nine-source model against the one-number night rule: same rows, same day bootstrap (the strongest single-source comparator)
HP = pd.read_parquet(RES / "oos_sota_headtohead.parquet"); HP["stock_id"] = HP.stock_id.astype(str); HP["date"] = pd.to_datetime(HP.date); FIN, LA9 = "Ensemble LightGBM+LSTM All9 (ours, final)", "LightGBM All9 (S1-S9)"
dd = d.assign(stock_id=d.stock_id.astype(str))[["date", "stock_id", "tx_night_ret"]].merge(HP[["date", "stock_id", "y_up", FIN, LA9]], on=["date", "stock_id"]).dropna(subset=[FIN, LA9]); dd["tx_night_ret"] = dd.tx_night_ret.fillna(0)
b2 = DayBootstrap(dd.date.values, B=1000); yy = dd.y_up.values; nr = dd.tx_night_ret.values; dm = dd.groupby("date")[FIN].transform("mean").values
out["final_model_vs_night_rule"] = dict(n=int(len(dd)), night_rule=float(auc_stat(yy, nr)),
    **{k: (lambda ci, v: dict(auroc=float(auc_stat(yy, v)), minus_night_rule=float(auc_stat(yy, v) - auc_stat(yy, nr)), ci=ci[:2], p=ci[2]))(b2.ci_diff(auc_stat, yy, nr, v), v) for k, v in [("final_model", dd[FIN].values), ("lightgbm_all9", dd[LA9].values), ("final_model_day_mean_only", dm)]})
print("final vs night rule", out["final_model_vs_night_rule"], flush=True)
dump(out, "night_session.json"); print({k: v for k, v in out.items() if k != "correlation"}); print("done")
# ---------------------------------------------------------------- figures f22 (night session vs U.S.) and f23 (fresh hold-out)
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3)); C = out["correlation"]; ks = ["night", "sox", "nasdaq", "sp500", "tsm_adr", "night_first_version"]; nm = ["TX night\nsession", "SOX", "NASDAQ", "S&P 500", "TSM ADR", "first version\n(previous night)"]; x = np.arange(len(ks))
a = ax[0]; a.bar(x - 0.2, [C[k]["next_overnight_gap"] for k in ks], 0.38, color="#eb6834", label="with the next overnight gap"); a.bar(x + 0.2, [C[k]["next_day_return"] for k in ks], 0.38, color="#4a3aa7", label="with the next-day return (close→close)")
for i, k in enumerate(ks): a.text(i - 0.2, max(C[k]["next_overnight_gap"], 0) + 0.015, f"{C[k]['next_overnight_gap']:.2f}", ha="center", fontsize=8); a.text(i + 0.2, max(C[k]["next_day_return"], 0) + 0.015, f"{C[k]['next_day_return']:.2f}", ha="center", fontsize=8)
a.axhline(0, color="#999", lw=0.8); a.set_xticks(x); a.set_xticklabels(nm, fontsize=8); a.set_ylim(-0.08, 0.95); a.legend(fontsize=8, frameon=False, loc="upper right"); a.set_ylabel("correlation (market-day level, 1,818 days)"); a.set_title("(a) What lines up with tomorrow's market move", fontsize=10, fontweight="bold")
a = ax[1]; R2 = out["r2_of_next_overnight_gap"]; v = [R2["five_us_series"], R2["night"], R2["both"]]; a.bar(range(3), v, color=["#eda100", "#eb6834", "#8a8a8a"], width=0.6)
for i, t in enumerate(v): a.text(i, t + 0.012, f"{t:.2f}", ha="center", fontsize=9)
W = out["what_each_adds"]; a.set_xticks(range(3)); a.set_xticklabels(["five U.S. series\n(SOX, S&P, NASDAQ,\nTSM ADR, VIX chg)", "TX night return\n(one number)", "both"], fontsize=8); a.set_ylim(0, 1.0); a.set_ylabel("R² of the next overnight gap")
a.text(0.02, 0.97, f"correlation with the gap of the part that\nthe other side does not explain:\nnight return, U.S. removed: {W['part_of_night_not_explained_by_us']['corr_gap']:.2f}\nSOX return, night removed: {W['part_of_sox_not_explained_by_night']['corr_gap']:.2f}", transform=a.transAxes, va="top", fontsize=8.5, color="#333"); a.set_title("(b) How much of the overnight gap each one explains", fontsize=10, fontweight="bold")
a = ax[2]; B = out["by_us_session"]; g = list(B); x = np.arange(len(g)); a.bar(x - 0.2, [B[k]["night"] for k in g], 0.38, color="#eb6834", label="TX night return as the score"); a.bar(x + 0.2, [B[k]["sox"] for k in g], 0.38, color="#eda100", label="SOX return as the score")
for i, k in enumerate(g): a.text(i - 0.2, B[k]["night"] + 0.004, f"{B[k]['night']:.3f}", ha="center", fontsize=9); a.text(i + 0.2, B[k]["sox"] + 0.004, f"{B[k]['sox']:.3f}", ha="center", fontsize=9)
a.axhline(0.5, color="#999", lw=0.8); a.set_ylim(0.47, 0.68); a.set_xticks(x); a.set_xticklabels([f"{k}\n(n = {B[k]['n']:,} rows)" for k in g], fontsize=8.5); a.legend(fontsize=8, frameon=False, loc="upper right"); a.set_ylabel("AUROC of a one-number rule, 2023–2026"); a.set_title("(c) It keeps working when the U.S. is closed", fontsize=10, fontweight="bold")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "figures/final/f22_night_session.png", dpi=200); plt.close(fig)
if (RES / "fresh_holdout.json").exists():
    H = json.load(open(RES / "fresh_holdout.json")); fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3)); arms = ["LightGBM|TW-only", "LightGBM|S4-only", "LightGBM|Full"] + (["LightGBM|Full+S7+S8"] if "LightGBM|Full+S7+S8" in H["labels"]["y_up"]["arms"] else [])
    lab = {"LightGBM|TW-only": "TW-only", "LightGBM|S4-only": "U.S.-only", "LightGBM|Full": "five sources", "LightGBM|Full+S7+S8": "five + S7 + S8"}; ttl = {"y_up": "Close→Close (primary)", "y_gap": "Close→Open (overnight gap)", "y_rel": "beats the TAIEX (which stock)"}
    SS = json.load(open(RES / "source_study.json"))
    for a, l in zip(ax, ["y_up", "y_gap", "y_rel"]):
        R = H["labels"][l]; x = np.arange(len(arms)); v = [R["arms"][k]["auroc"] for k in arms]; lo = [R["arms"][k]["auroc"] - R["arms"][k]["ci"][0] for k in arms]; hi = [R["arms"][k]["ci"][1] - R["arms"][k]["auroc"] for k in arms]
        a.bar(x, v, yerr=[lo, hi], capsize=4, color=["#8a93a8", "#eda100", "#4a3aa7", "#1baf7a"][:len(arms)], width=0.6, label=f"fresh hold-out ({H['n_days']} days)")
        wf = [R["walk_forward_2023_2026"].get(k.split("|")[1]) if k != "LightGBM|Full+S7+S8" else (SS[l]["Full+S7+S8"]["auroc"] if "Full+S7+S8" in SS[l] else SS.get("fusion_vs_us_only", {}).get("Full+S7+S8", {}).get("auroc") if l == "y_up" else None) for k in arms]
        for i, w in enumerate(wf):
            if w is not None: a.plot([i - 0.33, i + 0.33], [w, w], color="#d62839", lw=2.2, label="walk-forward 2023–2026" if i == 0 else None)
        for i, t in enumerate(v): a.text(i, R["arms"][arms[i]]["ci"][1] + 0.006, f"{t:.3f}", ha="center", fontsize=9)
        a.axhline(0.5, color="#999", lw=0.8); a.set_xticks(x); a.set_xticklabels([lab[k] for k in arms], fontsize=9); a.set_ylim(0.38, 0.84); a.set_title(ttl[l], fontsize=10, fontweight="bold"); a.set_ylabel("AUROC (95 % trading-day block CI)"); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=8.5, frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(ROOT / "figures/final/f23_fresh_holdout.png", dpi=200); plt.close(fig)
print("figures saved")
