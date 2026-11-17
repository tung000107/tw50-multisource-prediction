# Multi-Source Next-Day Taiwan 50 Stock Trend Classification

Data Science Project, Fall 2026 — Group A, Team 4.

台灣 50 成分股次日收盤方向預測：比較價格、法人籌碼、台股大盤、隔夜美股、日曆，以及新聞、跨股票結構、台指期夜盤、個股籌碼等九個資料來源。

## 目錄

| 路徑 | 內容 |
|---|---|
| `scripts/final/` | 期末所有實驗的程式（編號即執行順序；`common.py` 是共用協定） |
| `scripts/` | 資料抓取與建表（`fetch_*.py`、`build_dataset.py`）及期中以前的程式 |
| `results/final/*.json` | 每個實驗的數字結果（報告與簡報的所有數字都來自這裡） |
| `figures/final/` | 簡報與報告用的圖 |
| `final_present/06_實驗總表_設定數據分析結論.md` | **先讀這份**：每個實驗的設定、數據、分析、結論、比較 |
| `final_present/05_總整理_全部實驗與結論.md` | 完整總整理（由 `scripts/final/13_full_report.py` 產生） |
| `final_present/01_查核報告_Progress_Update.md` | Progress Update 數字查核 |
| `final_present/03_完成報告與待辦.md` | 第 2 節為重跑步驟 |
| `final_present/dataset_README.md` | 資料集說明 |

## 未包含

- `data/`（2.7 GB）：原始與處理後資料，用 `scripts/fetch_*.py` 與 `scripts/build_dataset.py` 重建。
- `external/`：第三方官方程式（StockNet、Adv-ALSTM、CausalStock 重現），請從原作者的 repo 取得。
- 預測值的 parquet 檔與執行 log。

## 環境

```sh
uv sync                      # 依 pyproject.toml / uv.lock 建立 .venv
.venv/bin/python scripts/final/02_tabular.py
```

部分 shell 腳本（`30_*.sh`、`38_run_alstm_reruns.sh`）寫死了原機器的絕對路徑與 TF1 conda 環境，在其他機器上需要修改。
