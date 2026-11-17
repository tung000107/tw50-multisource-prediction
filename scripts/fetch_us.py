"""US / global overnight information via Yahoo Finance."""
from pathlib import Path
import pandas as pd, yfinance as yf
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "us"; RAW.mkdir(parents=True, exist_ok=True)
TICKERS = {"^GSPC": "SP500", "^IXIC": "NASDAQ", "^SOX": "SOX", "^VIX": "VIX", "TSM": "TSM_ADR",
           "TWD=X": "USDTWD", "DX-Y.NYB": "DXY", "^TNX": "US10Y", "^N225": "NIKKEI"}
raw = yf.download(list(TICKERS), start="2018-12-01", end="2026-09-27", auto_adjust=False, progress=False, group_by="ticker")
close = pd.concat({TICKERS[t]: raw[t]["Close"] for t in TICKERS}, axis=1)
close.index.name = "date"; close.to_parquet(RAW / "us_close.parquet"); print(close.tail(3)); print(close.notna().sum())
