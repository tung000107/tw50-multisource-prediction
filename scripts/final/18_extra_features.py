"""Additional sources for the extended source study (all public, all keyed to 'known before the t+1 open'):
 S7 cross-stock (from our own prices; the 'causal-graph' information in feature form):
    sector_ret1 / sector_ret5 (equal-weight sector return excl. the stock), leader_ret1 / leader_ret5 (TSMC 2330), rank of ret1 within the day,
    cross-sectional dispersion of ret1, stock-minus-sector ret1, 5-day lagged return of the most correlated peer in the training window is
    deliberately NOT used (fold-dependent); learned graphs are evaluated by the CausalStock-lite model instead.
 S8 derivatives & market positioning (FinMind, TAIFEX / TWSE):
    tx_night_ret (close of the TX after-hours session that follows the day-t close, i.e. 15:00 t -> 05:00 t+1, vs the day-t close), tx_day_basis (TX close - TAIEX close)/TAIEX,
    tx_night_volume_ratio, foreign futures net open interest (long-short OI, change), dealer net OI, total margin balance change, total short balance change
 S9 stock-level flows (FinMind / TWSE): day-trading share of volume, 5-day mean; securities-lending volume (log), 5-day sum;
    monthly revenue YoY and MoM (last published; FinMind 'date' = publication month, usable from the 11th of that month)
Output: data/processed/extra_features.parquet"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import ROOT, load
df = load()[["date", "stock_id", "sector_en", "ret1", "ret5", "volume", "close"]].copy(); R = ROOT / "data/raw/finmind_extra"
# ---- S7 cross-stock ----
g = df.groupby(["date", "sector_en"])
for k in ("ret1", "ret5"):
    s = g[k].transform("sum"); n = g[k].transform("count"); df[f"sector_{k}_ex"] = (s - df[k]) / (n - 1).replace(0, np.nan)
lead = df[df.stock_id == "2330"].set_index("date")[["ret1", "ret5"]].rename(columns={"ret1": "leader_ret1", "ret5": "leader_ret5"}); df = df.merge(lead, on="date", how="left")
df["ret1_rank"] = df.groupby("date").ret1.rank(pct=True); df["xs_dispersion"] = df.groupby("date").ret1.transform("std"); df["ret1_minus_sector"] = df.ret1 - df.sector_ret1_ex
S7 = ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"]
# ---- S8 derivatives ----
fx = pd.read_parquet(R / "TaiwanFuturesDaily.parquet"); fx["date"] = pd.to_datetime(fx["date"]); fx = fx[(fx.futures_id == "TX") & (fx.volume > 0) & (~fx.contract_date.str.contains("/"))]
fx = fx.sort_values(["date", "contract_date"]); front = fx.groupby(["date", "trading_session"]).first().reset_index()           # nearest contract
day = front[front.trading_session == "position"].set_index("date"); night = front[front.trading_session == "after_market"].set_index("date")
tx = pd.DataFrame({"tx_day_close": day.close, "tx_day_vol": day.volume}); tx["tx_night_close"] = night.close; tx["tx_night_vol"] = night.volume
# ALIGNMENT (fixed 2026-11-13): TAIFEX books the after-hours session that starts at 15:00 on day t and ends at 05:00 on t+1 under the NEXT trading day, and FinMind keeps that date.
# The night session that follows the day-t close is therefore the 'after_market' record dated t+1 (its open equals the day-t close: median gap 0.07 %). The first version divided the
# record dated t (the PREVIOUS night, which ended at 05:00 on day t) by the day-t close: no look-ahead, but not the overnight move the feature was meant to be.
# FIX 2026-11-17: on a settlement day t the front day contract expires, so the night record dated t+1 belongs to the next month: the return
# then included the calendar spread (94 days). The night return now uses the SAME contract on both legs: the contract of the night record
# dated t+1, divided by that contract's day-session close on t.
allday = fx[fx.trading_session == "position"].set_index(["date", "contract_date"]).close
nxt = night[["contract_date", "close", "volume"]].copy(); nxt["prev_date"] = pd.Series(tx.index, index=tx.index).shift(1).reindex(nxt.index)
nxt["day_close_same"] = [allday.get((d, c), np.nan) if pd.notna(d) else np.nan for d, c in zip(nxt.prev_date, nxt.contract_date)]
tx["tx_night_ret"] = (nxt.close / nxt.day_close_same - 1).reindex(tx.index).shift(-1); tx["tx_night_volume_ratio"] = tx.tx_night_vol.shift(-1) / tx.tx_day_vol
nxt["roll"] = nxt.contract_date.values != day.contract_date.reindex(nxt.prev_date).values; print("night return: same-contract legs on", int(nxt.day_close_same.notna().sum()), "nights; contract roll on", int(nxt.roll.sum()), "nights")
taiex = pd.read_parquet(ROOT / "data/raw/finmind/TAIEX.parquet"); taiex["date"] = pd.to_datetime(taiex["date"]); taiex = taiex.set_index("date").close
tx["tx_day_basis"] = tx.tx_day_close / taiex.reindex(tx.index) - 1
fi = pd.read_parquet(R / "TaiwanFuturesInstitutionalInvestors.parquet"); fi["date"] = pd.to_datetime(fi["date"])
for who, nm in [("外資", "foreign"), ("自營商", "dealer")]:
    x = fi[fi.institutional_investors == who].set_index("date"); tx[f"{nm}_fut_net_oi"] = (x.long_open_interest_balance_volume - x.short_open_interest_balance_volume)
    tx[f"{nm}_fut_net_oi_chg"] = tx[f"{nm}_fut_net_oi"].diff(); tx[f"{nm}_fut_net_oi"] = tx[f"{nm}_fut_net_oi"] / 1e4; tx[f"{nm}_fut_net_oi_chg"] = tx[f"{nm}_fut_net_oi_chg"] / 1e4
mg = pd.read_parquet(R / "TaiwanStockTotalMarginPurchaseShortSale.parquet"); mg["date"] = pd.to_datetime(mg["date"])
for nm, col in [("MarginPurchaseMoney", "total_margin_chg"), ("ShortSale", "total_short_chg")]:
    x = mg[mg.name == nm].set_index("date").TodayBalance.astype(float); tx[col] = x.pct_change()
S8 = ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"]
df = df.merge(tx[S8].reset_index().rename(columns={"index": "date"}), on="date", how="left")
# ---- S9 stock-level flows ----
dt = pd.read_parquet(R / "TaiwanStockDayTrading.parquet"); dt["date"] = pd.to_datetime(dt["date"]); dt = dt.rename(columns={"Volume": "dt_volume"})[["stock_id", "date", "dt_volume"]]
df = df.merge(dt, on=["stock_id", "date"], how="left"); df["daytrade_share"] = (df.dt_volume / df.volume).clip(0, 1).fillna(0)
df = df.sort_values(["stock_id", "date"]); df["daytrade_share_5d"] = df.groupby("stock_id").daytrade_share.transform(lambda s: s.rolling(5, min_periods=1).mean())
sl = pd.read_parquet(R / "TaiwanStockSecuritiesLending.parquet"); sl["date"] = pd.to_datetime(sl["date"]); sl = sl.groupby(["stock_id", "date"]).volume.sum().rename("lend_vol").reset_index()
df = df.merge(sl, on=["stock_id", "date"], how="left"); df["lend_vol"] = df.lend_vol.fillna(0); df["lend_log"] = np.log1p(df.lend_vol); df["lend_log_5d"] = df.groupby("stock_id").lend_log.transform(lambda s: s.rolling(5, min_periods=1).sum())
rv = pd.read_parquet(R / "TaiwanStockMonthRevenue.parquet"); rv["pub"] = pd.to_datetime(rv["date"]) + pd.Timedelta(days=10)      # usable from the 11th of the publication month
rv = rv.sort_values(["stock_id", "pub"]); rv["rev_yoy"] = rv.groupby("stock_id").revenue.pct_change(12); rv["rev_mom"] = rv.groupby("stock_id").revenue.pct_change(1)
rv = rv[["stock_id", "pub", "rev_yoy", "rev_mom"]].rename(columns={"pub": "date"}).dropna(); rv["date"] = rv["date"].astype("datetime64[ns]"); df["date"] = df["date"].astype("datetime64[ns]")
df = pd.merge_asof(df.sort_values("date"), rv.sort_values("date"), on="date", by="stock_id", direction="backward")
S9 = ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]
out = df[["date", "stock_id"] + S7 + S8 + S9].sort_values(["date", "stock_id"])
out.to_parquet(ROOT / "data/processed/extra_features.parquet"); print(out.shape); print(out[S7 + S8 + S9].describe().T[["mean", "std", "min", "max"]].round(4).to_string()); print("missing %", (out[S7 + S8 + S9].isna().mean() * 100).round(2).to_dict())
