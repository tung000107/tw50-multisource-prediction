"""Fetch Taiwan stock news headlines (title, publish time, tagged stock codes) from the public cnyes newslist API, day by day,
2018-12-01 .. 2026-09-30, category tw_stock. Resumable: one JSONL per month in data/raw/cnyes/. Titles and metadata only.
"""
import json, time, sys, datetime as dt
from pathlib import Path
import requests
OUT = Path(__file__).resolve().parents[2] / "data/raw/cnyes"; OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"; H = {"User-Agent": "Mozilla/5.0 (academic research; titles only)"}
start, end = dt.date(2018, 12, 1), dt.date(2026, 9, 30)
if len(sys.argv) == 3: start, end = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2])
def fetch_day(d):
    t0 = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()); t1 = t0 + 86399
    page, rows = 1, []
    while True:
        for k in range(5):
            try:
                r = requests.get(API, params=dict(startAt=t0, endAt=t1, limit=30, page=page), headers=H, timeout=30)
                if r.status_code == 200: break
            except Exception as e: pass
            time.sleep(5 * (k + 1))
        else: raise RuntimeError(f"failed {d} page {page}")
        it = r.json().get("items", {}); data = it.get("data", [])
        for x in data: rows.append(dict(newsId=x.get("newsId"), publishAt=x.get("publishAt"), title=x.get("title"), summary=(x.get("summary") or "")[:200], codes=sorted({m.get("code") for m in (x.get("market") or []) if m.get("code")}), category=x.get("categoryName")))
        if page >= int(it.get("last_page") or 1): break
        page += 1; time.sleep(0.25)
    return rows
d = start
while d <= end:
    f = OUT / f"news_{d:%Y-%m}.jsonl"; done = set()
    if f.exists():
        for line in open(f): done.add(json.loads(line)["day"])
    if f"{d}" not in done:
        rows = fetch_day(d)
        with open(f, "a") as fh: fh.write(json.dumps(dict(day=f"{d}", n=len(rows), rows=rows), ensure_ascii=False) + "\n")
        if d.day == 1: print(d, len(rows), flush=True)
        time.sleep(0.3)
    d += dt.timedelta(days=1)
print("done")
