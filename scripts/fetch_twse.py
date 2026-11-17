"""Resumable crawler for official TWSE daily endpoints.

Per trading day it stores raw JSON for:
  MI_INDEX  : all-stock OHLCV, TAIEX & sub-indices, market totals, advance/decline counts
  T86       : institutional investors (foreign / trust / dealer) net buy-sell per stock
  MI_MARGN  : margin purchase / short sale balances per stock
Usage: uv run python scripts/fetch_twse.py [--start 2019-01-01] [--end 2026-09-26] [--sleep 1.5]
"""
import argparse, json, sys, time
from datetime import date, timedelta
from pathlib import Path
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "twse"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
      "Referer": "https://www.twse.com.tw/"}
ENDPOINTS = {
    "MI_INDEX": "https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={d}&type=ALLBUT0999",
    "T86":      "https://www.twse.com.tw/fund/T86?response=json&date={d}&selectType=ALLBUT0999",
    "MI_MARGN": "https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={d}&selectType=ALL",
}

def fetch(url, retries=5, backoff=10):
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    pass
            print(f"  http {r.status_code}, retry {i+1}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  error {e}, retry {i+1}", file=sys.stderr)
        time.sleep(backoff * (i + 1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01"); ap.add_argument("--end", default="2026-09-26")
    ap.add_argument("--sleep", type=float, default=1.5)
    a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    d = d0; n_req = 0; t0 = time.time()
    while d <= d1:
        if d.weekday() < 5:  # skip weekends; holidays return stat != OK and are recorded as such
            ds = d.strftime("%Y%m%d")
            for name, tmpl in ENDPOINTS.items():
                out = RAW / name / f"{ds}.json"
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                js = fetch(tmpl.format(d=ds))
                if js is None:
                    print(f"{ds} {name}: giving up for now", file=sys.stderr); continue
                out.write_text(json.dumps(js, ensure_ascii=False))
                n_req += 1
                if n_req % 100 == 0:
                    print(f"{n_req} requests, at {ds}, {time.time()-t0:.0f}s elapsed", flush=True)
                time.sleep(a.sleep)
        d += timedelta(days=1)
    print(f"done: {n_req} new requests in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
