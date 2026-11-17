# group_A_part_1_team_04_dataset — README（草稿，12/22 繳交用）

## 1. 內容
| 檔案 | 說明 |
|---|---|
| `dataset_final.parquet` / `.csv` | 90,851 筆 stock-day（Taiwan 50 成分股 × 2019-04-10 ～ 2026-09-23），77 欄：45 個模型特徵（S1–S5）、識別欄（date, stock_id, name, sector）、原始價量（open/high/low/close/volume/amount）、標籤（y_up, y_tr, y_intra, y_gap, y_rel, ret_next, …） |
| `feature_dictionary.md` | 每個欄位的定義、來源、公開時間（資訊時間表） |
| `raw/` | 原始 JSON／parquet：TWSE 官方端點（MI_INDEX、T86、MI_MARGN、TWTB4U）、FinMind 鏡像、Yahoo Finance（yfinance）、FinMind TaiwanStockDividendResult、0050 ETF |
| `scripts/` | `fetch_*.py`、`build_dataset.py`、`scripts/final/01_prepare.py`（一行指令重建） |

## 2. 來源與引用
- TWSE 官方交易資訊（個股日成交、三大法人、融資融券、當沖統計）[1][13][14]
- FinMind 開源鏡像（交叉驗證：收盤價 99.99%、成交量／外資／融資餘額 100% 一致）
- Yahoo Finance via `yfinance`（S&P 500、NASDAQ、SOX、VIX、TSM ADR、USD/TWD、DXY、US10Y、Nikkei 225）[10]
- 除權息參考價：FinMind `TaiwanStockDividendResult`（TWSE 公告）

## 3. 前處理摘要
93,879 原始 stock-day → 刪 8 筆零成交量 → 刪 50 筆無次日標籤 → 刪 2,950 筆 60 日指標暖機與 20 筆法人缺漏 → **90,851**。
缺值：S2 1.1%、S4 1.3%（最大：short_chg 8.0%、us_nikkei_ret 11.1%）；樹模型原生處理，LR 以訓練窗中位數補值。
所有特徵皆為 t+1 開盤前可得（TW 13:30 收盤、法人收盤後公布、美股台灣時間約 05:00 收盤）。

## 4. 標籤定義
| 欄位 | 定義 |
|---|---|
| `y_up` | Close(t+1) > Close(t)（主任務；平盤 7.8% 歸 0） |
| `y_tr` | 除息日以 TWSE 參考價取代 Close(t)（只影響 170 筆） |
| `y_intra` | Close(t+1) > Open(t+1) |
| `y_gap` | Open(t+1) > Close(t) |
| `y_rel` | 個股次日報酬 > 加權指數次日報酬 |
| 四分類 | 依訓練窗 \|r\| 中位數 k（0.83–0.86%）切 strong/mild × down/up，於建模腳本內計算 |

## 5. 重建
```
uv run python scripts/fetch_finmind.py && uv run python scripts/fetch_us.py && uv run python scripts/build_dataset.py
uv run python scripts/final/00_fetch_extra.py && uv run python scripts/final/01_prepare.py
```
