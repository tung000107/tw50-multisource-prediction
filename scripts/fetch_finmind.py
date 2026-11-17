"""Fast fetch of the same TWSE datasets through the FinMind open mirror (for EDA turnaround).
Datasets: TaiwanStockPrice, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockMarginPurchaseShortSale,
plus TAIEX daily via TaiwanStockPrice(data_id=TAIEX).
"""
import sys, time
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import TAIWAN50, START, END

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "finmind"; RAW.mkdir(parents=True, exist_ok=True)
API = "https://api.finmindtrade.com/api/v4/data"
DATASETS = ["TaiwanStockPrice", "TaiwanStockInstitutionalInvestorsBuySell", "TaiwanStockMarginPurchaseShortSale"]

def get(dataset, data_id):
    for i in range(4):
        r = requests.get(API, params=dict(dataset=dataset, data_id=data_id, start_date=START, end_date=END), timeout=60)
        if r.status_code == 200 and r.json().get("status") == 200:
            return pd.DataFrame(r.json()["data"])
        print(f"  {dataset} {data_id}: status {r.status_code} {r.text[:120]}; retry", file=sys.stderr)
        time.sleep(20 * (i + 1))
    raise RuntimeError(f"failed {dataset} {data_id}")

for ds in DATASETS:
    out = RAW / f"{ds}.parquet"
    if out.exists():
        print("skip", ds); continue
    frames = []
    for i, sid in enumerate(TAIWAN50):
        frames.append(get(ds, sid)); time.sleep(0.6)
        if (i + 1) % 10 == 0: print(f"{ds}: {i+1}/50", flush=True)
    df = pd.concat(frames, ignore_index=True); df.to_parquet(out); print(ds, df.shape)
if not (RAW / "TAIEX.parquet").exists():
    get("TaiwanStockPrice", "TAIEX").to_parquet(RAW / "TAIEX.parquet"); print("TAIEX ok")
