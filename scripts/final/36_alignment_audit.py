"""Systematic audit of time alignment: every feature (S1-S9), the labels, the folds, the sequence windows and the SOTA inputs.
Two alignment errors were found by accident on 2026-11-13 (SOTA windows ended one day early; the TX night-session feature used the previous night).
This script checks every input on purpose, with tests that do not depend on reading the code:
 T1 leak test      a legitimate feature of row t (known before the t+1 open) may predict the overnight gap Close(t)->Open(t+1) but must NOT predict the
                   intraday leg Open(t+1)->Close(t+1) much better than chance; univariate AUROC for y_gap / y_intra / y_up, and the same feature taken one row
                   LATER (what a one-day look-ahead would look like) for scale
 T2 staleness test each series must line up with a reference that is known to be contemporaneous: correlation at lag -1 / 0 / +1 must peak at lag 0
 T3 structure      label identities, fold purge, U.S. merge direction, sequence window end, SOTA window end, news assignment times
Output: results/final/alignment_audit.json"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import *
KEY = ["date", "stock_id"]; out = {}
df = load(); ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet")
for e in (ex, nf): e["date"] = pd.to_datetime(e["date"])
df = df.merge(ex, on=KEY, how="left").merge(nf, on=KEY, how="left").sort_values(["stock_id", "date"]).reset_index(drop=True)
NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score_2026", "market_text_score_2026"],
       "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"],
       "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
SRC_ALL = {**SRC, **NEW}; te = df.year.isin(TEST_YEARS)
# ---------------------------------------------------------------- T1 leak test
def ua(y, x):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100 or np.nanstd(x[m]) == 0: return None
    a = auc_stat(y[m], x[m]); return float(max(a, 1 - a))
rows = []; g = df.groupby("stock_id")
for s, fs in SRC_ALL.items():
    for f in fs:
        x = df[f].astype(float); lead = g[f].shift(-1).astype(float)
        rows.append(dict(source=s, feature=f, auroc_y_up=ua(df.y_up.values, x.values), auroc_y_gap=ua(df.y_gap.values, x.values), auroc_y_intra=ua(df.y_intra.values, x.values),
                         auroc_y_up_if_one_row_later=ua(df.y_up.values, lead.values), auroc_y_intra_if_one_row_later=ua(df.y_intra.values, lead.values)))
T1 = pd.DataFrame(rows); T1["flag_predicts_intraday"] = T1.auroc_y_intra > 0.53
out["T1_leak_test"] = dict(rule="flag if a feature alone predicts the intraday leg with AUROC > 0.53 (all rows 2019-2026; orientation-free)", n_features=int(len(T1)), n_flagged=int(T1.flag_predicts_intraday.sum()),
    flagged=T1[T1.flag_predicts_intraday].to_dict("records"), max_intraday_auroc=float(T1.auroc_y_intra.max()), top10_by_y_up=T1.sort_values("auroc_y_up", ascending=False).head(10).to_dict("records"),
    look_ahead_scale=dict(note="what one row of look-ahead does to a return feature, for scale", ret1_one_row_later_auroc_y_up=float(T1.loc[T1.feature == "ret1", "auroc_y_up_if_one_row_later"].iloc[0]),
                          hl_range_one_row_later_auroc_y_intra=float(T1.loc[T1.feature == "hl_range", "auroc_y_intra_if_one_row_later"].iloc[0])), table=T1.to_dict("records"))
print("T1 flagged:", T1[T1.flag_predicts_intraday][["feature", "auroc_y_intra"]].to_dict("records"), "| max intraday AUROC", round(T1.auroc_y_intra.max(), 4), flush=True)
print(T1.sort_values("auroc_y_up", ascending=False).head(8)[["feature", "auroc_y_up", "auroc_y_gap", "auroc_y_intra"]].round(3).to_string(), flush=True)
# ---------------------------------------------------------------- T2 staleness: lag profile against a contemporaneous reference
day = df.drop_duplicates("date").set_index("date").sort_index(); xs = df.groupby("date"); mret = xs.ret1.mean(); mabs = xs.ret1.apply(lambda s: s.abs().mean()); gapn = xs.ret_gap.mean() if "ret_gap" in df else None
tx = pd.read_parquet(ROOT / "data/raw/finmind/TAIEX.parquet"); tx["date"] = pd.to_datetime(tx.date); tx = tx.set_index("date").sort_index(); taiex_ret = tx.close.pct_change(); taiex_next = tx.close.shift(-1) / tx.close - 1
def lagprof(x, ref):
    x, ref = x.align(ref, join="inner"); return {f"lag{k:+d}": float(x.shift(k).corr(ref)) for k in (-1, 0, 1)}   # lag+1 = the series one day EARLIER against today's reference
def peak0(p): return bool(abs(p["lag+0"]) >= max(abs(p["lag-1"]), abs(p["lag+1"])))
T2 = {}
def add(name, x, ref, refname, expect="lag+0"):
    p = lagprof(x, ref); k = max(p, key=lambda q: abs(p[q])); T2[name] = dict(reference=refname, **p, peak=k, ok=bool(k == expect))
add("taiex_ret1 (S3)", day.taiex_ret1, mret, "mean stock return of the same day")
add("us_nikkei_ret (S4)", day.us_nikkei_ret, taiex_ret, "TAIEX return of the same day (Tokyo and Taipei trade the same hours)")
for c in ["us_sp500_ret", "us_nasdaq_ret", "us_sox_ret", "us_tsm_adr_ret", "us_vix_chg", "us_10y_chg", "us_dxy_ret"]:
    add(f"{c} (S4)", day[c], gapn, "mean overnight gap Close(t)->Open(t+1) of the 50 stocks (a U.S. session dated t must line up with the gap that follows day t)")
add("us_usdtwd_chg (S4)", day.us_usdtwd_chg, taiex_ret, "TAIEX return of the same day (TWD strengthens when Taiwan stocks rise)")
add("tx_night_ret (S8)", day.tx_night_ret, gapn, "mean overnight gap that follows day t")
add("tx_night_ret (S8) vs S&P 500", day.tx_night_ret, day.us_sp500_ret, "U.S. session dated t (trades during the same night)")
add("tx_day_basis (S8)", day.tx_day_basis.diff(), taiex_ret, "TAIEX return of the same day (change of the basis)")
add("foreign_fut_net_oi_chg (S8)", day.foreign_fut_net_oi_chg, taiex_ret, "TAIEX return of the same day")
add("total_margin_chg (S8)", day.total_margin_chg, taiex_ret, "TAIEX return of the same day")
add("xs_dispersion (S7)", day.xs_dispersion, mabs, "mean |stock return| of the same day")
add("leader_ret1 (S7)", day.leader_ret1, taiex_ret, "TAIEX return of the same day")
# stock-level series: pooled correlation with the stock's own same-day return / range at lags
def add_stock(name, f, ref, refname):
    p = {f"lag{k:+d}": float(g[f].shift(k).corr(df[ref])) for k in (-1, 0, 1)}; k = max(p, key=lambda q: abs(p[q])); T2[name] = dict(reference=refname, **p, peak=k, ok=bool(k == "lag+0"))
for f in ["foreign_net_ratio", "trust_net_ratio", "dealer_net_ratio", "margin_chg", "short_chg"]: add_stock(f"{f} (S2)", f, "ret1", "the stock's own return on the same day")
add_stock("daytrade_share (S9)", "daytrade_share", "hl_range", "the stock's own high-low range on the same day"); add_stock("lend_log (S9)", "lend_log", "volume_ratio20", "the stock's own relative volume on the same day")
add_stock("log_news (S6)", "log_news", "hl_range", "the stock's own high-low range on the same day"); add_stock("ret1_rank (S7)", "ret1_rank", "ret1", "the stock's own return on the same day"); add_stock("sector_ret1_ex (S7)", "sector_ret1_ex", "ret1", "the stock's own return on the same day")
T2["foreign_fut_net_oi_chg (S8)"]["external_check"] = "not peaking at lag 0, so the dates were checked against the exchange's own page (taifex.com.tw futContractsDate, TX, foreign investors): open interest long / short 7,218 / 106,791 on 2026-11-12 and 8,086 / 112,357 on 2026-11-11, identical to the FinMind rows of the same dates. The series is dated correctly; the lag profile is behaviour (positions are adjusted the day after a move), not a date shift."; T2["foreign_fut_net_oi_chg (S8)"]["ok_after_external_check"] = True
out["T2_staleness_test"] = dict(rule="correlation with a contemporaneous reference at lag -1 / 0 / +1 must peak at lag 0", n_series=len(T2), n_not_peaking_at_lag0=int(sum(not v["ok"] for v in T2.values())), series=T2)
print("T2 not peaking at lag 0:", {k: {q: round(v[q], 3) for q in ("lag-1", "lag+0", "lag+1")} for k, v in T2.items() if not v["ok"]}, flush=True)
# ---------------------------------------------------------------- T3 structure
T3 = {}
T3["label_identities"] = dict(y_up_equals_close_next_gt_close=float((df.y_up == (df.close_next > df.close)).mean()), y_gap_equals_open_next_gt_close=float((df.y_gap == (df.open_next > df.close)).mean()),
    y_intra_equals_close_next_gt_open_next=float((df.loc[df.y_intra.notna(), "y_intra"] == (df.close_next > df.open_next)[df.y_intra.notna()]).mean()),
    ret1_of_next_row_equals_ret_next=float(np.isclose(g.ret1.shift(-1), df.ret_next, atol=1e-9)[g.ret1.shift(-1).notna()].mean()))
alld = np.sort(df.date.unique()); nxt = pd.Series(alld[1:], index=alld[:-1]); T3["label_next_day"] = dict(share_rows_where_date_next_is_not_the_next_market_day=float((df.date_next != df.date.map(nxt)).mean()),
    note="a stock halted on the next market day is labelled with its next traded day; the U.S. features are then one or more sessions older than the label day")
d2 = load(); T3["fold_purge"] = {str(yr): dict(last_train_label_day=str(d2.loc[tr, "date_next"].max().date()), first_test_feature_day=str(d2.loc[tst, "date"].min().date()), ok=bool(d2.loc[tr, "date_next"].max() < d2.loc[tst, "date"].min()),
                                               inner_val_before_test=bool(d2.loc[iva, "date_next"].max() < d2.loc[tst, "date"].min()), inner_train_before_inner_val=bool(d2.loc[itr, "date_next"].max() < d2.loc[iva, "date"].min())) for yr, tr, itr, iva, tst in folds(d2)}
full = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet"); full["date"] = pd.to_datetime(full.date); full["us_date"] = pd.to_datetime(full.us_date)
T3["us_merge"] = dict(max_days_us_date_after_tw_date=int((full.us_date - full.date).dt.days.max()), share_same_calendar_day=float((full.us_date == full.date).mean()), share_stale_gt3d=float(((full.date - full.us_date).dt.days > 3).mean()),
                      share_tw_days_reusing_previous_us_session=float((lambda z: (z.us_date == z.us_date.shift(1)).mean())(full.drop_duplicates("date").sort_values("date"))))
# news: time of day of the headlines assigned to a feature day, relative to the 08:30 decision on the next trading day
rows = []
for f in sorted(glob.glob(str(ROOT / "data/raw/cnyes/*.jsonl"))):
    for line in open(f):
        for x in json.loads(line)["rows"]: rows.append(pd.Timestamp(int(x["publishAt"]), unit="s", tz="Asia/Taipei").tz_localize(None))
ts = pd.Series(sorted(rows)); days = pd.to_datetime(alld); ts = ts[ts > days[0] + pd.Timedelta(hours=8, minutes=30)].reset_index(drop=True); fpos = news_feature_day(ts.values, days); ok = fpos >= 0; ts = ts[ok].reset_index(drop=True); fpos = fpos[ok]
decision = np.append(days[1:].values, (days[-1] + pd.Timedelta(days=1)).to_datetime64()) + np.timedelta64(510, "m"); prev_dec = np.append(np.datetime64("1900-01-01"), decision[:-1])
T3["news_assignment"] = dict(n_headlines=int(len(ts)), share_published_after_the_0830_decision_they_feed=float((ts.values > decision[fpos]).mean()), share_that_could_have_fed_an_earlier_decision=float((ts.values <= prev_dec[fpos]).mean()),
    first_version=dict(share_stamped_0830_0900_counted_before_the_decision=0.0340, share_weekend_holiday_daytime_headlines_one_row_late=0.0432), note="rule common.news_feature_day since 2026-11-13: window (08:30 on day t, 08:30 on the next trading day]; both shares must be 0")
# sequence windows end on the row itself
src = open(Path(__file__).parent / "03_sequence.py").read().split("if __name__")[0]; gq = {"__name__": "sq", "__file__": str(Path(__file__).parent / "03_sequence.py")}; exec(src, gq); wi = gq["window_index"](5); dq = gq["df"]
okw = (wi >= 0).all(1); T3["sequence_windows"] = dict(last_column_is_the_row_itself=bool((wi[okw, -1] == np.arange(len(dq))[okw]).all()), strictly_increasing_dates_within_window=bool(all((dq.date.values[wi[okw, j]] < dq.date.values[wi[okw, j + 1]]).all() for j in range(4))),
                                      same_stock_within_window=bool(all((dq.stock_id.values[wi[okw, j]] == dq.stock_id.values[wi[okw, -1]]).all() for j in range(4))))
# SOTA inputs: after the fix the window of length L ends on the feature day whose y_up is the label
L = 5; Ypv = d2.pivot(index="date", columns="stock_id", values="y_up"); Ysh = Ypv.shift(1); dd = Ypv.index[L - 1:-1]; lab = Ysh.values[L:]; ref = Ypv.reindex(dd).values; m = ~np.isnan(lab) & ~np.isnan(ref)
T3["sota_windows_causalstock"] = dict(label_of_window_equals_y_up_of_last_window_day=float((lab[m] == ref[m]).mean()), windows_checked=int(m.sum()))
meta = json.load(open(ROOT / "data/raw/tw50_advalstm45/meta.json")); a = np.genfromtxt(ROOT / "data/raw/tw50_advalstm45/2026/ourpped/2330.csv", delimiter=","); dts = np.genfromtxt(ROOT / "data/raw/tw50_advalstm45/2026/trading_dates.csv", dtype=str)
r30 = d2[d2.stock_id == "2330"].set_index("date").y_up; chk = [(a[i, -2] > 0) == (r30.get(pd.Timestamp(str(dts[i - 1]))) == 1) for i in range(1, len(dts)) if abs(a[i, -2]) > 1e-8 and pd.Timestamp(str(dts[i - 1])) in r30.index]
T3["sota_windows_official_advalstm"] = dict(label_on_file_row_d_equals_y_up_of_day_d_minus_1=float(np.mean(chk)), rows_checked=int(len(chk)), note="the official loader predicts the label on row d from rows d-seq..d-1, so the window ends on the feature day d-1")
fx = pd.read_parquet(ROOT / "data/raw/finmind_extra/TaiwanFuturesDaily.parquet"); fx["date"] = pd.to_datetime(fx.date); fx = fx[(fx.futures_id == "TX") & (fx.volume > 0) & (~fx.contract_date.str.contains("/"))].sort_values(["date", "contract_date"])
fr = fx.groupby(["date", "trading_session"]).first().reset_index(); dy = fr[fr.trading_session == "position"].set_index("date"); ng = fr[fr.trading_session == "after_market"].set_index("date")
T3["tx_night_session_date_convention"] = dict(median_abs_gap_night_open_of_NEXT_record_vs_day_close=float((ng.open.shift(-1).reindex(dy.index) / dy.close - 1).abs().median()), median_abs_gap_night_open_of_SAME_record_vs_day_close=float((ng.open.reindex(dy.index) / dy.close - 1).abs().median()),
    reading="the after-hours record dated t+1 opens at the day-t close, so it is the session that follows day t (15:00 t -> 05:00 t+1); the feature uses that record since 2026-11-13")
out["T3_structure"] = T3; print("T3", json.dumps({k: v for k, v in T3.items() if k != "fold_purge"}, default=str)[:2400], flush=True); print("fold purge ok:", {k: v["ok"] and v["inner_val_before_test"] and v["inner_train_before_inner_val"] for k, v in T3["fold_purge"].items()})
dump(out, "alignment_audit.json"); print("done")
