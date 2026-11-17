"""Additional public sources (FinMind free tier) for the extended source study:
  market level : TaiwanFuturesDaily (TX, day + after_market sessions), TaiwanFuturesInstitutionalInvestors (TX), TaiwanStockTotalMarginPurchaseShortSale
  stock level  : TaiwanStockDayTrading, TaiwanStockSecuritiesLending, TaiwanStockMonthRevenue (50 stocks each)
Output: data/raw/finmind_extra/<dataset>.parquet
"""
import sys, time
from pathlib import Path
import pandas as pd, requests
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts")); from universe import TAIWAN50, START, END
RAW = ROOT / "data/raw/finmind_extra"; API = "https://api.finmindtrade.com/api/v4/data"
def get(dataset, data_id=None, start=START, end=END):
    for i in range(6):
        p = dict(dataset=dataset, start_date=start, end_date=end); 
        if data_id: p["data_id"] = data_id
        r = requests.get(API, params=p, timeout=90)
        if r.status_code == 200 and r.json().get("status") == 200: return pd.DataFrame(r.json()["data"])
        print(f"  {dataset} {data_id}: {r.status_code} {r.text[:100]}; retry", file=sys.stderr, flush=True); time.sleep(30 * (i + 1))
    raise RuntimeError(f"failed {dataset} {data_id}")
def yearly(dataset, data_id=None):
    frames = []
    for y in range(2019, 2027):
        frames.append(get(dataset, data_id, f"{y}-01-01", f"{y}-12-31")); time.sleep(0.6)
    return pd.concat(frames, ignore_index=True)
for ds, did in [("TaiwanFuturesDaily", "TX"), ("TaiwanFuturesInstitutionalInvestors", "TX"), ("TaiwanStockTotalMarginPurchaseShortSale", None)]:
    out = RAW / f"{ds}.parquet"
    if out.exists(): print("skip", ds); continue
    d = yearly(ds, did); d.to_parquet(out); print(ds, d.shape, flush=True)
for ds in ["TaiwanStockDayTrading", "TaiwanStockSecuritiesLending", "TaiwanStockMonthRevenue"]:
    out = RAW / f"{ds}.parquet"
    if out.exists(): print("skip", ds); continue
    frames = []
    for i, sid in enumerate(TAIWAN50):
        frames.append(get(ds, sid)); time.sleep(0.7)
        if (i + 1) % 10 == 0: print(ds, i + 1, flush=True)
    d = pd.concat(frames, ignore_index=True); d.to_parquet(out); print(ds, d.shape, flush=True)
print("done")
