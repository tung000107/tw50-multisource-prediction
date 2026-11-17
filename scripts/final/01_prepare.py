"""Build data/processed/dataset_final.parquet = modelling table (90,851 rows, 45 features) + extra labels.
Labels (all for stock i, feature day t; decision at 08:30 on t+1):
  y_up    : Close(t+1) > Close(t)                                  [primary, as in proposal / progress]
  y_tr    : total-return version: on ex-dividend day t+1 compare Close(t+1) with the TWSE reference price
            (= Close(t) adjusted for the dividend) instead of Close(t)  [robustness: dividend-adjusted]
  y_intra : Close(t+1) > Open(t+1)   (open-to-close: tradable after the overnight gap is known)
  y_gap   : Open(t+1)  > Close(t)    (the overnight gap itself)
  y_rel   : stock return(t->t+1) > TAIEX return(t->t+1)  (relative-to-market: cross-sectional question)
  ret_next, open_next, taiex_ret_next kept for four-level labels (fold-dependent cut-offs) and the backtest.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
df = pd.read_parquet(ROOT / "data/processed/dataset.parquet")
full = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet").sort_values(["stock_id", "date"])
full["open_next"] = full.groupby("stock_id")["open"].shift(-1)
df = df.merge(full[["stock_id", "date", "open_next"]], on=["stock_id", "date"], how="left")
tx = pd.read_parquet(ROOT / "data/raw/finmind/TAIEX.parquet"); tx["date"] = pd.to_datetime(tx["date"]); tx = tx.sort_values("date")
tx["taiex_ret_next"] = tx["close"].shift(-1) / tx["close"] - 1
df["date"] = pd.to_datetime(df["date"]); df = df.merge(tx[["date", "taiex_ret_next"]], on="date", how="left")
# dividend-adjusted (total-return) label
dv = pd.read_parquet(ROOT / "data/raw/finmind_extra/dividend_result.parquet")
dv["date"] = pd.to_datetime(dv["date"]); dv = dv.rename(columns={"date": "date_next"})[["stock_id", "date_next", "before_price", "after_price", "reference_price"]]
df["date_next"] = pd.to_datetime(df["date_next"]); df = df.merge(dv, on=["stock_id", "date_next"], how="left")
ex = df.reference_price.notna()
ok = ex & np.isclose(df.before_price, df.close, rtol=1e-3)
print("ex-dividend next-days:", int(ex.sum()), "| before_price == close(t):", int(ok.sum()))
df["ret_next_tr"] = df["ret_next"]
df.loc[ex, "ret_next_tr"] = df.loc[ex, "close_next"] / df.loc[ex, "reference_price"] - 1
df["y_tr"] = (df.ret_next_tr > 0).astype(float)
df["y_intra"] = (df.close_next > df.open_next).astype(float); df["y_gap"] = (df.open_next > df.close).astype(float)
df.loc[df.open_next.isna(), ["y_intra", "y_gap"]] = np.nan
df["y_rel"] = (df.ret_next > df.taiex_ret_next).astype(float); df.loc[df.taiex_ret_next.isna(), "y_rel"] = np.nan
df["ret_intra"] = df.close_next / df.open_next - 1; df["ret_gap"] = df.open_next / df.close - 1
df["sector_en"] = df["sector"].map({"半導體": "Semiconductor", "電子/硬體": "Electronics/HW", "金融": "Financials", "航運": "Shipping", "傳產/民生": "Traditional/Consumer", "電信": "Telecom"})
df = df.drop(columns=["before_price", "after_price", "reference_price"]).sort_values(["date", "stock_id"]).reset_index(drop=True)
df.to_parquet(ROOT / "data/processed/dataset_final.parquet")
print(df.shape, "labels:", {c: round(float(df[c].mean()), 4) for c in ["y_up", "y_tr", "y_intra", "y_gap", "y_rel"]})
print("rows where y_tr != y_up:", int((df.y_tr != df.y_up).sum()), "| missing y_intra:", int(df.y_intra.isna().sum()), "| missing y_rel:", int(df.y_rel.isna().sum()))
