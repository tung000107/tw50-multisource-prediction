"""Extra raw data for the final study (all free, public):
  - FinMind TaiwanStockDividendResult for the 50 stocks (ex-dividend reference prices -> total-return label)
  - FinMind TaiwanStockPrice for 0050 ETF (buy-and-hold benchmark for the backtest)
Output: data/raw/finmind_extra/dividend_result.parquet, data/raw/finmind_extra/etf_0050.parquet
"""
import sys, time
from pathlib import Path
import pandas as pd, requests
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts"))
from universe import TAIWAN50, START, END
RAW = ROOT / "data" / "raw" / "finmind_extra"; RAW.mkdir(parents=True, exist_ok=True)
API = "https://api.finmindtrade.com/api/v4/data"
def get(dataset, data_id, start=START, end=END):
    for i in range(5):
        r = requests.get(API, params=dict(dataset=dataset, data_id=data_id, start_date=start, end_date=end), timeout=60)
        if r.status_code == 200 and r.json().get("status") == 200:
            return pd.DataFrame(r.json()["data"])
        print(f"  {dataset} {data_id}: {r.status_code} {r.text[:100]}; retry", file=sys.stderr); time.sleep(30 * (i + 1))
    raise RuntimeError(f"failed {dataset} {data_id}")
out = RAW / "dividend_result.parquet"
if not out.exists():
    frames = []
    for i, sid in enumerate(TAIWAN50):
        d = get("TaiwanStockDividendResult", sid); frames.append(d); time.sleep(0.7)
        if (i + 1) % 10 == 0: print("dividend", i + 1, flush=True)
    df = pd.concat(frames, ignore_index=True); df.to_parquet(out); print("dividend_result", df.shape)
out = RAW / "etf_0050.parquet"
if not out.exists():
    get("TaiwanStockPrice", "0050").to_parquet(out); print("0050 ok")
print("done")
