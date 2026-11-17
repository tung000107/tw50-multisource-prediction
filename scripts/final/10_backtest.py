"""Cost-aware backtest (supplementary, not classification). Rules fixed ex ante:
  * every test day: rank the 50 stocks by the model's probability (seed-mean LightGBM); buy the top-k at the t+1 open, equal weight,
    sell at the t+1 close (intraday round trip). k in {5, 10} (reported both; no selection on the test set).
  * costs: commission 0.1425 % x 2 + day-trade securities tax 0.15 % = 0.435 % per round trip (list rates; broker discounts ignored)
  * signals: Full / S4-only / TW-only close-to-close models (what the primary task predicts) and the open-to-close models (y_intra)
  * long-short (top-k minus bottom-k) reported as a theoretical bound only (short-sale constraints ignored; tax 0.3 % on the short leg)
  * benchmarks: 0050 ETF buy-and-hold (close-to-close), equal-weight 50-stock intraday (open->close) portfolio with the same costs, random top-k
Metrics: mean daily net return (+ day-block bootstrap CI), annualised return, volatility, Sharpe (252 d), max drawdown, hit rate, PSR vs 0.
Output: results/final/backtest.json, results/final/backtest_daily.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
from common import *
oos = pd.read_parquet(RES / "oos_tabular.parquet"); lab = pd.read_parquet(RES / "oos_labels.parquet")
for arm in ["Full", "S4-only", "TW-only"]:
    oos[f"p_cc_{arm}"] = oos[[c for c in oos if c.startswith(f"LightGBM|{arm}|")]].mean(axis=1).values
    oos[f"p_oc_{arm}"] = lab[f"y_intra|{arm}"].values
oos["p_rand"] = np.random.default_rng(0).random(len(oos)); oos["p_sox"] = oos["Rule|SOX return score|0"]
_tie = np.random.default_rng(1).random(len(oos)) * 1e-9   # random tie-break: a signal that is identical across stocks (S4-only, SOX) selects stocks at random
for c in [c for c in oos if c.startswith("p_")]: oos[c] = oos[c] + _tie
COST_RT = 0.001425 * 2 + 0.0015; COST_LS = 0.001425 * 2 + 0.003; COST_DISC = 0.001425 * 0.4 * 2 + 0.0015   # 60 % commission discount scenario
etf = pd.read_parquet(ROOT / "data/raw/finmind_extra/etf_0050.parquet"); etf["date"] = pd.to_datetime(etf["date"]); etf = etf.set_index("date").sort_index()
# 2026-11-17: the unadjusted 0050 close has a 4-for-1 split on 2025-06-18 (one-day "return" of -75 %); a jump beyond +-30 % in an index ETF is
# a share split, so the price before it is divided by the (integer) split ratio. Without this the benchmark showed a -77 % drawdown.
_c = etf["close"].astype(float).copy(); _r = _c.pct_change()
for _d in _r.index[_r.abs() > 0.3]:
    _k = round(_c.shift(1).loc[_d] / _c.loc[_d]) if _r.loc[_d] < 0 else 1 / round(_c.loc[_d] / _c.shift(1).loc[_d]); _c.loc[_c.index < _d] = _c.loc[_c.index < _d] / _k; print("0050 split adjusted on", _d.date(), "ratio", _k)
etf_ret = _c.pct_change().reindex(pd.to_datetime(sorted(oos.date_next.unique()))).fillna(0)
# the same for the 50 stocks: close-to-close returns beyond the +-10 % price limit are ex-dividend days, splits or capital reductions, not
# market moves; they are removed from the close-to-close benchmark and from the 'signal only' column (the intraday legs are unaffected)
oos["ret_next_clean"] = oos["ret_next"].where(oos["ret_next"].abs() <= 0.105)
def stats(r):
    r = pd.Series(r).dropna(); n = len(r); mu, sd = r.mean(), r.std(ddof=1); sr = mu / sd * np.sqrt(252) if sd > 0 else np.nan
    eq = (1 + r).cumprod(); mdd = float((eq / eq.cummax() - 1).min())
    sr_d = mu / sd if sd > 0 else 0; psr = float(norm.cdf(sr_d * np.sqrt(n - 1) / np.sqrt(1 - skew(r) * sr_d + (kurtosis(r, fisher=False) - 1) / 4 * sr_d ** 2))) if sd > 0 else np.nan
    return dict(n_days=int(n), mean_daily=float(mu), ann_return=float((1 + mu) ** 252 - 1), ann_vol=float(sd * np.sqrt(252)), sharpe=float(sr), max_drawdown=mdd, hit_rate=float((r > 0).mean()), psr_vs_0=psr, total_return=float(eq.iloc[-1] - 1))
daily = {}; out = {"cost_round_trip": COST_RT, "cost_long_short": COST_LS, "cost_discounted": COST_DISC, "strategies": {}}
days = oos.groupby("date")
for sig in ["p_cc_Full", "p_cc_S4-only", "p_cc_TW-only", "p_oc_Full", "p_oc_S4-only", "p_oc_TW-only", "p_sox", "p_rand"]:
    for k in [5, 10]:
        rows = []
        for d, g in days:
            g = g.dropna(subset=["ret_intra"]); s = g.sort_values(sig, ascending=False)
            top = s.head(k); bot = s.tail(k)
            rows.append(dict(date=d, date_next=g.date_next.iloc[0], long_net=top.ret_intra.mean() - COST_RT, long_gross=top.ret_intra.mean(),
                             ls_net=(top.ret_intra.mean() - bot.ret_intra.mean()) - COST_RT - COST_LS, cc_signal_only=top.ret_next_clean.mean(), long_disc=top.ret_intra.mean() - COST_DISC))
        r = pd.DataFrame(rows).set_index("date"); daily[f"{sig}|k{k}"] = r
        b = DayBootstrap(r.index.values, B=1000); mn = r.long_net.values
        out["strategies"][f"{sig}|top{k}"] = dict(long_only_net=stats(r.long_net), long_only_gross=stats(r.long_gross), long_short_net=stats(r.ls_net), close_to_close_signal_only_gross=stats(r.cc_signal_only), long_only_discounted_cost=stats(r.long_disc), break_even_round_trip_cost=float(r.long_gross.mean()),
                                                 mean_daily_net_ci=b.ci(lambda x: x.mean(), mn), by_year={str(y): stats(r.loc[r.index.year == y, "long_net"]) for y in TEST_YEARS})
        print(f"{sig:14s} k={k:2d} net ann {out['strategies'][f'{sig}|top{k}']['long_only_net']['ann_return']:+.3f} Sharpe {out['strategies'][f'{sig}|top{k}']['long_only_net']['sharpe']:+.2f} MDD {out['strategies'][f'{sig}|top{k}']['long_only_net']['max_drawdown']:.3f} | gross Sharpe {out['strategies'][f'{sig}|top{k}']['long_only_gross']['sharpe']:+.2f}", flush=True)
ew = days.ret_intra.mean() - COST_RT; out["benchmarks"] = {"equal_weight_50_intraday_net": stats(ew), "equal_weight_50_intraday_gross": stats(days.ret_intra.mean()), "etf_0050_buy_hold_close_to_close": stats(etf_ret.values),
                                                           "equal_weight_50_close_to_close_gross": stats(days.ret_next_clean.mean())}
print("benchmarks", {k: (round(v["ann_return"], 3), round(v["sharpe"], 2)) for k, v in out["benchmarks"].items()})
pd.concat({k: v[["long_net", "long_gross", "ls_net", "cc_signal_only", "long_disc"]] for k, v in daily.items()}, axis=1).to_parquet(RES / "backtest_daily.parquet"); dump(out, "backtest.json")
