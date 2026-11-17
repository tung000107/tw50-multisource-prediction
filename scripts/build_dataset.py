"""Merge TWSE price / institutional / margin / TAIEX (FinMind mirror) with US overnight data
into one stock-day table with next-day label. Output: data/processed/dataset.parquet

Information-set convention: every feature on row (stock i, date t) is known before the
TWSE open on t+1 (TW close 13:30 on t, TWSE flows published ~16:00 on t, US session dated t
closes ~04:00-05:00 Taiwan time on t+1). Label = 1 if close_{t+1} > close_t.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import TAIWAN50, SECTOR

ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data" / "raw"; OUT = ROOT / "data" / "processed"; OUT.mkdir(exist_ok=True)

# ---------- S1 price & technicals ----------
px = pd.read_parquet(RAW / "finmind" / "TaiwanStockPrice.parquet")
px["date"] = pd.to_datetime(px["date"]); px = px.sort_values(["stock_id", "date"])
px = px.rename(columns={"Trading_Volume": "volume", "Trading_money": "amount", "max": "high", "min": "low", "Trading_turnover": "n_trades"})
px = px[px["volume"] > 0].copy()
g = px.groupby("stock_id")
px["ret1"] = g["close"].pct_change()
for k in (2, 3, 5, 10, 20):
    px[f"ret{k}"] = g["close"].pct_change(k)
for w in (5, 20, 60):
    px[f"ma{w}_ratio"] = px["close"] / g["close"].transform(lambda s: s.rolling(w).mean()) - 1
px["vol20"] = g["ret1"].transform(lambda s: s.rolling(20).std())
px["volume_ratio20"] = px["volume"] / g["volume"].transform(lambda s: s.rolling(20).mean())
px["hl_range"] = (px["high"] - px["low"]) / px["close"]
px["gap_open"] = px["open"] / g["close"].shift(1) - 1
delta = g["close"].diff(); up = delta.clip(lower=0); dn = -delta.clip(upper=0)
rs = up.groupby(px["stock_id"]).transform(lambda s: s.rolling(14).mean()) / dn.groupby(px["stock_id"]).transform(lambda s: s.rolling(14).mean())
px["rsi14"] = 100 - 100 / (1 + rs)
# label
px["close_next"] = g["close"].shift(-1); px["date_next"] = g["date"].shift(-1)
px["ret_next"] = px["close_next"] / px["close"] - 1
px["y_up"] = (px["ret_next"] > 0).astype("float"); px.loc[px["ret_next"].isna(), "y_up"] = np.nan

# ---------- S2 institutional & margin ----------
ins = pd.read_parquet(RAW / "finmind" / "TaiwanStockInstitutionalInvestorsBuySell.parquet")
ins["date"] = pd.to_datetime(ins["date"]); ins["net"] = ins["buy"] - ins["sell"]
piv = ins.pivot_table(index=["stock_id", "date"], columns="name", values="net", aggfunc="sum").fillna(0)
dealer = piv.get("Dealer_self", 0) + piv.get("Dealer_Hedging", 0)
dealer = dealer.where(dealer != 0, piv.get("Dealer", 0))
inst = pd.DataFrame({"foreign_net": piv.get("Foreign_Investor", 0) + piv.get("Foreign_Dealer_Self", 0),
                     "trust_net": piv.get("Investment_Trust", 0), "dealer_net": dealer}).reset_index()
mg = pd.read_parquet(RAW / "finmind" / "TaiwanStockMarginPurchaseShortSale.parquet")
mg["date"] = pd.to_datetime(mg["date"])
mg = mg[["stock_id", "date", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]].rename(
    columns={"MarginPurchaseTodayBalance": "margin_bal", "ShortSaleTodayBalance": "short_bal"})
df = px.merge(inst, on=["stock_id", "date"], how="left").merge(mg, on=["stock_id", "date"], how="left")
g = df.groupby("stock_id")
for c in ("foreign_net", "trust_net", "dealer_net"):
    df[f"{c}_ratio"] = df[c] / df["volume"]
    df[f"{c}_ratio5"] = g[f"{c}_ratio"].transform(lambda s: s.rolling(5).sum())
df["inst_net_ratio"] = df[["foreign_net_ratio", "trust_net_ratio", "dealer_net_ratio"]].sum(axis=1)
df["margin_chg"] = g["margin_bal"].pct_change().replace([np.inf, -np.inf], np.nan)
df["short_chg"] = g["short_bal"].pct_change().replace([np.inf, -np.inf], np.nan)
df["margin_chg5"] = g["margin_bal"].pct_change(5).replace([np.inf, -np.inf], np.nan)

# ---------- S3 market (TAIEX) ----------
tx = pd.read_parquet(RAW / "finmind" / "TAIEX.parquet"); tx["date"] = pd.to_datetime(tx["date"]); tx = tx.sort_values("date")
tx["taiex_ret1"] = tx["close"].pct_change(); tx["taiex_ret5"] = tx["close"].pct_change(5)
tx["taiex_ma20_ratio"] = tx["close"] / tx["close"].rolling(20).mean() - 1
tx["taiex_vol_ratio20"] = tx["Trading_money"] / tx["Trading_money"].rolling(20).mean()
tx["taiex_vol20"] = tx["taiex_ret1"].rolling(20).std()
df = df.merge(tx[["date", "taiex_ret1", "taiex_ret5", "taiex_ma20_ratio", "taiex_vol_ratio20", "taiex_vol20"]], on="date", how="left")
df["rel_ret1"] = df["ret1"] - df["taiex_ret1"]

# ---------- S4 US / global overnight ----------
us = pd.read_parquet(RAW / "us" / "us_close.parquet").sort_index(); us.index = pd.to_datetime(us.index)
# FIX 2026-11-17: us_close.parquet has a row for every weekday with NaN on U.S. (or Tokyo) holidays. In pandas 3 pct_change() no longer pads
# over NaN, so the session AFTER each holiday got a NaN return and was dropped below; Taiwan rows of that day then reused the session before
# the holiday (60 days, 2,998 rows). Each series' change is now computed on its own trading days and the U.S. session calendar is the S&P 500's.
us = us[us["SP500"].notna()]
chg = lambda s, f: f(s.dropna()).reindex(s.index)
usr = pd.DataFrame(index=us.index)
for c in ("SP500", "NASDAQ", "SOX", "TSM_ADR", "DXY", "NIKKEI"):
    usr[f"us_{c.lower()}_ret"] = chg(us[c], lambda x: x.pct_change())
usr["us_vix_level"] = us["VIX"]; usr["us_vix_chg"] = chg(us["VIX"], lambda x: x.pct_change())
usr["us_usdtwd_chg"] = chg(us["USDTWD"], lambda x: x.pct_change()); usr["us_10y_chg"] = chg(us["US10Y"], lambda x: x.diff())
usr.index.name = "us_date"; usr = usr.dropna(subset=["us_sp500_ret"]).reset_index()
usr["us_date"] = pd.to_datetime(usr["us_date"]).astype("datetime64[ns]"); df["date"] = df["date"].astype("datetime64[ns]")
# For TW date t use the latest US session dated <= t (its close happens ~05:00 Taiwan time on t+1, before TW open).
df = pd.merge_asof(df.sort_values("date"), usr.sort_values("us_date"), left_on="date", right_on="us_date", direction="backward")
# US holiday flag: no new U.S. session on the TW date, so the row reuses the previous row's U.S. session.
# (Changed 2026-11-17: the old rule 'session > 3 days old' only fired because of the holiday bug above; after the fix it was 0 on every row.)
df["us_stale"] = (df["date"] > df["us_date"]).astype(int)
# ---------- S5 calendar ----------
df["weekday"] = df["date"].dt.weekday; df["month"] = df["date"].dt.month
df["is_month_end"] = (df["date"].dt.is_month_end | (df["date_next"].dt.month != df["date"].dt.month)).astype(int)
df["sector"] = df["stock_id"].map(SECTOR); df["name"] = df["stock_id"].map(TAIWAN50); df["year"] = df["date"].dt.year
df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
df.to_parquet(OUT / "dataset_full.parquet")
# modeling-ready: drop warm-up rows (60-day indicators) and rows without label
model = df.dropna(subset=["ma60_ratio", "rsi14", "y_up", "foreign_net_ratio5", "us_sp500_ret"]).copy()
model.to_parquet(OUT / "dataset.parquet")
print("full rows", len(df), "| model-ready rows", len(model), "| stocks", model["stock_id"].nunique(),
      "| period", model["date"].min().date(), "->", model["date"].max().date(), "| features", model.shape[1])
print("label up-rate", round(model["y_up"].mean(), 4), "| flat (0) share", round((model["ret_next"] == 0).mean(), 4))
print(model.groupby("year").agg(n=("y_up", "size"), up=("y_up", "mean")).round(3).to_string())
