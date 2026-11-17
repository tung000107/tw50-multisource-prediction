"""Example of the daily output (deck slide 'Deployment'): the final nine-source model's out-of-sample probabilities for the last test day,
top five and bottom five of the 50 stocks, with the return that was realised afterwards.
(The first version of this table was written on 2026-11-03 from the five-source LightGBM and had no script; this file replaces it.)
Input: results/final/oos_sota_headtohead.parquet (walk-forward predictions, no fitting here).  Output: results/final/example_ranking.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from common import *
FIN = "Ensemble LightGBM+LSTM All9 (ours, final)"
P = pd.read_parquet(RES / "oos_sota_headtohead.parquet"); P["stock_id"] = P.stock_id.astype(str); P["date"] = pd.to_datetime(P.date)
df = load(); df["stock_id"] = df.stock_id.astype(str); d = P[P.date == P.date.max()].merge(df[["date", "stock_id", "name", "sector_en", "ret_next", "date_next", "us_sox_ret"]], on=["date", "stock_id"], how="left").sort_values(FIN, ascending=False)
row = lambda r: dict(stock=r.stock_id, name=r["name"], sector=r.sector_en, p_up=round(float(r[FIN]), 3), realised=float(r.ret_next))
out = dict(model=FIN, feature_date=str(d.date.iloc[0].date()), trade_date=str(pd.to_datetime(d.date_next.iloc[0]).date()), sox_overnight=float(d.us_sox_ret.iloc[0]), n_stocks=int(len(d)),
           top=[row(r) for _, r in d.head(5).iterrows()], bottom=[row(r) for _, r in d.tail(5).iterrows()], top5_mean_realised=float(d.head(5).ret_next.mean()), bottom5_mean_realised=float(d.tail(5).ret_next.mean()), all_mean_realised=float(d.ret_next.mean()))
print(out["feature_date"], out["trade_date"], [(x["stock"], x["p_up"], round(x["realised"], 4)) for x in out["top"]], [(x["stock"], x["p_up"], round(x["realised"], 4)) for x in out["bottom"]]); dump(out, "example_ranking.json")
