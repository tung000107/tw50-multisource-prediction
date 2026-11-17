"""Cross-check the FinMind mirror against official TWSE JSON already crawled (MI_INDEX close/volume, T86 foreign net, MI_MARGN margin balance)."""
import json, sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from universe import TAIWAN50
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data/raw"
num = lambda s: pd.to_numeric(str(s).replace(",", "").replace("--", "").strip(), errors="coerce")
rows_px, rows_t86, rows_mg = [], [], []
for f in sorted((RAW / "twse/MI_INDEX").glob("*.json")):
    d = json.loads(f.read_text());
    if d.get("stat") != "OK": continue
    for t in d.get("tables", []):
        if "每日收盤行情" in t.get("title", ""):
            for r in t["data"]:
                if r[0].strip() in TAIWAN50: rows_px.append((f.stem, r[0].strip(), num(r[2]), num(r[8])))
for f in sorted((RAW / "twse/T86").glob("*.json")):
    d = json.loads(f.read_text())
    if d.get("stat") != "OK": continue
    fields = d["fields"]; i_net = fields.index("外陸資買賣超股數(不含外資自營商)"); i_all = fields.index("三大法人買賣超股數")
    for r in d["data"]:
        if r[0].strip() in TAIWAN50: rows_t86.append((f.stem, r[0].strip(), num(r[i_net]), num(r[i_all])))
for f in sorted((RAW / "twse/MI_MARGN").glob("*.json")):
    d = json.loads(f.read_text())
    if d.get("stat") != "OK": continue
    for t in d.get("tables", []):
        if "融資融券彙總" in t.get("title", ""):
            for r in t["data"]:
                if r[0].strip() in TAIWAN50: rows_mg.append((f.stem, r[0].strip(), num(r[6]), num(r[12])))
off_px = pd.DataFrame(rows_px, columns=["date", "stock_id", "volume", "close"]); off_px["date"] = pd.to_datetime(off_px.date)
off_t86 = pd.DataFrame(rows_t86, columns=["date", "stock_id", "foreign_net_official", "inst_net_official"]); off_t86["date"] = pd.to_datetime(off_t86.date)
off_mg = pd.DataFrame(rows_mg, columns=["date", "stock_id", "margin_bal_official", "short_bal_official"]); off_mg["date"] = pd.to_datetime(off_mg.date)
px = pd.read_parquet(RAW / "finmind/TaiwanStockPrice.parquet"); px["date"] = pd.to_datetime(px.date)
ins = pd.read_parquet(RAW / "finmind/TaiwanStockInstitutionalInvestorsBuySell.parquet"); ins["date"] = pd.to_datetime(ins.date); ins["net"] = ins.buy - ins.sell
fm_for = ins[ins.name == "Foreign_Investor"].groupby(["stock_id", "date"]).net.sum().rename("foreign_net_mirror").reset_index()
fm_all = ins.groupby(["stock_id", "date"]).net.sum().rename("inst_net_mirror").reset_index()
mg = pd.read_parquet(RAW / "finmind/TaiwanStockMarginPurchaseShortSale.parquet"); mg["date"] = pd.to_datetime(mg.date)
m1 = off_px.merge(px[["date", "stock_id", "close", "Trading_Volume"]], on=["date", "stock_id"], suffixes=("_official", "_mirror"))
m2 = off_t86.merge(fm_for, on=["date", "stock_id"]).merge(fm_all, on=["date", "stock_id"])
m3 = off_mg.merge(mg[["date", "stock_id", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]], on=["date", "stock_id"])
rep = {
 "period_checked": [str(off_px.date.min().date()), str(off_px.date.max().date())],
 "n_trading_days_official": int(off_px.date.nunique()),
 "close": {"n": len(m1), "exact_match_rate": float((m1.close_official == m1.close_mirror).mean()), "max_abs_diff": float((m1.close_official - m1.close_mirror).abs().max())},
 "volume": {"n": len(m1), "exact_match_rate": float((m1.volume == m1.Trading_Volume).mean())},
 "foreign_net": {"n": len(m2), "exact_match_rate": float((m2.foreign_net_official == m2.foreign_net_mirror).mean())},
 "inst_net_total": {"n": len(m2), "exact_match_rate": float((m2.inst_net_official == m2.inst_net_mirror).mean())},
 "margin_balance": {"n": len(m3), "exact_match_rate": float((m3.margin_bal_official == m3.MarginPurchaseTodayBalance).mean())},
 "short_balance": {"n": len(m3), "exact_match_rate": float((m3.short_bal_official == m3.ShortSaleTodayBalance).mean())},
 "official_only_days": int(len(set(off_px.date) - set(px.date))), "mirror_only_days_in_period": int(len(set(px[(px.date >= off_px.date.min()) & (px.date <= off_px.date.max())].date) - set(off_px.date))),
}
(ROOT / "figures/crosscheck.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(json.dumps(rep, indent=1, ensure_ascii=False))
if rep["foreign_net"]["exact_match_rate"] < 1: print(m2[m2.foreign_net_official != m2.foreign_net_mirror].head())
