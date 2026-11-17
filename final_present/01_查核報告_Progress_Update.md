# 期末簡報前置查核報告（Double Check）

**查核日期：** 2026-11-03　**查核對象：** `progress_update/…Progress Update.pdf`（80 頁，Canva 匯出）、`proposal_plan.md`、`docs/*.md`、`figures/*.json`、`scripts/*.py`
**隊伍：** Group A・Team 4（314554052 梁欣童、313554063 劉霈琳、314554038 林冠宇）
**方法：** 每個數字回溯到 (1) 專案資料重算、(2) 論文全文 / Crossref / 官方網頁，標記「✓ 一致」「✗ 錯誤」「△ 不一致／口徑不明」「？ 無法重現」。沒有來源的數字一律不採用。

---

## 0. 結論摘要（先看這裡）

| 類別 | 件數 | 影響 |
|---|---|---|
| ✗ 明確錯誤（必改） | 6 | 引用貼錯、同一份簡報前後矛盾、常數規則的 AUROC 不可能是 0.491 |
| △ 口徑不一致（必統一） | 9 | 同一個量在不同頁用不同方法算（Pearson vs Spearman、250 日 vs 252 日、33.7 vs 33.8%） |
| ？ 無程式可重現（必重算） | 14 組數字 | Progress 簡報 p.41、p.69–73 的所有新數字，在 `scripts/` 與各 session 暫存區都找不到程式 |
| 作業規範落差（期末必補） | 7 | Final Presentation 大綱 A–H 與助教「共通問題」要求 |
| 需使用者決定 | 4 | Part 1 或 Part 2、AI 使用規範原文、分工表、是否保留 Streamlit |

**最重要的一件事：** Progress Update 內部自相矛盾。p.9 Motivation 與 p.12 Contributions 仍寫「融合比最強單一來源好 +0.014（95% CI +0.004～+0.024）」，但 p.70 已更正為「Full − S4-only = −0.001 [−0.008, +0.006]，+0.014 來自固定 400 棵樹的過擬合」。期末簡報只能保留一個版本，而且要說明為什麼改。

---

## 1. 作業要求確認（`DSP-2026Fall-Term-Project-Final_Presentation.pdf`）

| 項目 | 規定 | 我們的狀態 |
|---|---|---|
| 影片長度 | 約 20 分鐘，**超過 20 分鐘全隊 50%** | 講稿要控制在 18:30–19:30 |
| 必備章節 | A 封面、B Introduction（Background / Motivation / Research Aims / Challenges / Contributions）、C Problem Setting（IPO、Statement、Metrics）、D Proposed Methods（五步驟、挑戰對應）、E Experimental Setting（Datasets + Checklist、Baseline/SOTA、Evaluation Strategy、Environment）、F Results（含 insights）、G Conclusion & Limitation、**H Work distribution chart（分工表）** | Progress 簡報沒有 H；D 必須把「每個 challenge 由哪一步解決」標出來 |
| 檔名 | `group_A_part_{1 or 2}_team_04_final.pptx / .mp4` | **Part 1 或 2 未確認**（見第 6 節） |
| 期限 | Part 1：11/21（六）中午；Part 2：11/28（六）中午 | 現場 Q&A：Part 1 11/24（二）、Part 2 12/01（二） |
| 其他 | 全員都要在影片與現場發言；不得後製加速；繳錯區 0 分 | 分工表要和講者分配一致 |
| 期末報告 | 12/22 23:59，依模板；資料集 zip + README 同日 | 不在本次範圍，但資料集 README 可一併先寫 |

**助教「Proposal 共通問題」對期末的要求（`DSP_2026_Proposal_Common_Issues.pdf`）：**
1. 「Research Aims」改稱「**Project Aims**」；Aims / Challenges / Contributions / Method 要逐項對應。
2. Challenges 必須是研究困難，資料清理與防洩漏不算（Progress 的 6 個 challenge 符合，但 D 章要明示對應）。
3. Problem Definition 要寫清楚 input / output / 預測時間點；主題必須是分類。
4. 類別不平衡不可只看 Accuracy / AUROC，要補 F1、PR-AUC。→ 期末加 **PR-AUC** 與 Macro-F1。
5. **資料集出現的頁面就要附 citation**（IEEE/APA）；External data 要說明整合與時間對齊方式。
6. Train/Val/Test 用途要明確、不得用 test 調參；前處理只能用訓練資料擬合；workflow 要標先後順序。
7. Baseline / SOTA 要有依據、公平比較（同資料、同切分、同指標）；不能只列分數，要指出限制。
8. 加目錄頁與頁碼；避免大段文字；術語前後一致。

---

## 2. ✗ 明確錯誤（必改）

| # | 位置 | 內容 | 證據 | 修正方法 |
|---|---|---|---|---|
| E1 | p.9、p.12 vs p.70 | p.9「+0.014 AUROC when Taiwan sources are fused … 95% CI +0.004 to +0.024」、p.12「fusion beats the best single source by AUROC +0.014; Taiwan sources jointly significant」；p.70 卻說 Full − S4-only = **−0.001 [−0.008, +0.006]**，+0.014 來自固定 400 棵樹過擬合 | 兩組數字分別對應 `figures/review_stats2.json`（舊：+0.0139 [0.0044, 0.0238]）與 p.70 的 v1 early-stopping 版本（無程式） | 期末一律採用**重建後的單一協定**（inner validation early stopping + 固定調參預算），Motivation / Contribution / Results 全部用同一組數字；在 Results 用一頁說明「為什麼 proposal 的 +0.014 不成立」（這本身是好的 insight） |
| E2 | p.24 | Jeng et al. (2024) 的文獻頁，標題下方貼的是 **CMIN（Luo et al., ACL 2023）的引用文字** | 原頁文字「D. Luo, W. Liao, S. Li … ACL … doi: 10.18653/v1/2023.acl-long.679」 | 改為「S.-L. Jeng, C.-H. Liu, and C. M. Guan, "Stock trend prediction assisted by automatically defined polarity scores of news articles," *J. Inf. Sci. Eng.*, vol. 40, no. 4, pp. 717–743, 2024, doi: 10.6688/JISE.202407_40(4).0003.」（Airiti 官方頁已核對；作者拼法 **Chiau-Hsuan Liu**，p.16 寫 Chiau Hsuan 缺連字號） |
| E3 | p.69 | 「Majority class (rule) pooled AUROC 0.491 [0.477, 0.505]」 | 常數預測的 AUROC 恆為 0.500；逐年欄位也寫 0.500，pooled 卻 0.491 | 刪掉該格或改 0.500；若 pooled 用的是「各訓練窗多數類」仍是常數，不可能有 CI |
| E4 | p.69、p.63 | 「Overnight SOX sign (rule) AUROC 0.576」、p.63「a sign rule alone turns into AUROC 0.58」 | 資料重算：以 **連續 SOX 報酬**當分數 AUROC = 0.5765；sign 規則本身的 AUROC = balanced accuracy = 0.553 | 寫法改成「overnight SOX return as a single score: AUROC 0.577; sign rule accuracy 0.553 / BA 0.557」 |
| E5 | p.11 Challenge 2 | 「U.S.–TW sign agreement 53.8–62.3 % by year」 | 資料重算（SOX 方向 vs 次日個股方向）：2019–2026 為 **53.2–60.7%**；用 S&P 500 為 54.0–60.7%；沒有任何定義算得出 62.3 | 重算後改數字並註明定義（SOX 隔夜方向 = 個股次日方向的比例，2019–2026 逐年） |
| E6 | p.11 Challenge 5 | 「Per-stock up-rate 40.5–51.6 %」 | 資料重算：最低為中華電 2412 **39.0%**，最高奇鋐 3017 51.6% | 改為 39.0–51.6% |

---

## 3. △ 口徑不一致（必統一）

| # | 位置 | 內容 | 說明 | 建議 |
|---|---|---|---|---|
| I1 | p.11 / p.78 vs p.41 | S&P 500–NASDAQ 相關 0.93、NASDAQ–SOX 0.83 vs 0.95、0.86 | 重算：Pearson 0.949 / 0.860，Spearman 0.934 / 0.829。兩者都對，但沒標方法 | 全簡報統一用 Spearman（與 p.78 相關矩陣一致）並標「Spearman ρ」 |
| I2 | p.7 vs `docs/user_deck_fix_guide.md` | 外資成交占比 33.8% vs 33.7% | TWSE Fact Book 2025 Table 3.04（2024）：外資法人 33.72%，外資自然人 0.04%，合計 33.76% | 用「33.7 % (foreign juridical investors, 2024) [13]」 |
| I3 | p.6 / p.7 | rolling corr 0.27–0.62（p.7）；`docs/bg_corr_report.md` 已更正為 252 日對數報酬、排除沿用列；proposal 版是 0.26–0.59 | 新方法已有 `scripts/bg_corr_check.py`，可重現 | 期末沿用 bg_corr_check 結果：全期 0.49（CI 0.45–0.52，n=1,752）、rolling 0.27–0.62；自相關 0.03、−0.20～0.29 |
| I4 | p.62 | 「average pairwise correlation 0.30 (2020–2026), 0.42 in 2020, 0.34 in 2025; PC1 34 %」 | 靜態重算：0.28 / 0.446 / 0.325 / PC1 31.2%。簡報值可能是 rolling 252 日平均，但無程式 | 期末重新計算並寫清楚方法（rolling 252 日平均 vs 區間靜態） |
| I5 | p.18 | 「Ours: ✓ 14 features (S1)」 | 程式 S1 = 15 個（含 spread） | 改 15 |
| I6 | p.55 / p.56 / p.10 | 特徵數「57」（資料表欄位數）與模型實際用 45 個 | 57 含 open/high/low/close/amount 等不進模型的欄位 | 期末統一寫「57 columns, **45 model features** (S1 15 / S2 10 / S3 6 / S4 11 / S5 3)」 |
| I7 | p.4、p.65 | 「1-trading-day purge at every boundary」vs `docs/qa_20_answers.md` Q8 已說明依時間線其實不需要 purge | 兩種說法都有人講過 | 期末保留 purge（保守），但說法改成「purge 1 day as a conservative safeguard; information timing already prevents overlap」 |
| I8 | p.65 | 「5 seeds per model, mean ± seed std」 | Progress 的結果實際上是單一種子（`random_state=42`） | 期末真的跑 5 個種子，或改寫成「single seed; seed sensitivity reported separately」 |
| I9 | p.53 | Environment 列 XGBoost、PyTorch、Optuna、pandas-ta | 當時環境根本沒裝（本次已裝：torch 2.6+cu124、xgboost 3.4、shap 0.54、optuna 5.0） | 期末只列真正用到的套件與版本 |

---

## 4. ？ 無程式可重現的數字（必重算）

以下數字在 `scripts/`、`figures/*.json`、`/tmp/claude-1014/*` 各 session 暫存區均**找不到產生程式**（只找到 `q20.py`，對應 proposal Q&A 的 +0.014 版本）。期末簡報不可以直接沿用，必須由新 pipeline 重新產生；數值可能略有差異，屬正常。

| 頁 | 數字 | 用途 |
|---|---|---|
| p.4、p.69、p.70 | LightGBM v1（early stopping）：Full 0.597、S4-only 0.598、TW-only 0.528；Full − S4 = −0.001 [−0.008, +0.006]；Full − TW = +0.070 [+0.052, +0.087] | 主結果 |
| p.70 | 「fixed 400 trees +0.012 [+0.001, +0.023]; early stopping only +0.001 [−0.007, +0.009]」 | 解釋 +0.014 消失 |
| p.71 | 單一來源 S1 0.524、S2 0.526、S3 0.521、S5 0.519；leave-one-out −0.074 / ≤ +0.004；early stopping 樹數 2–43 vs 46–104 | 消融 |
| p.72 | 四分類 Macro-F1 0.354 vs 0.336、BA 0.371、QWK 0.166、AUROC strong-vs-mild 0.715 / 0.710 | 次任務 |
| p.41 | Macro-F1 +0.019 [0.013, 0.024]、QWK +0.009 [−0.004, 0.021] | 顯著性 |
| p.41 | LSTM W = 1/5/10/20/60 AUROC 0.589–0.592，60 日 10 倍時間 | 模型比較 |
| p.41 | 隨機 5-fold 0.714 vs walk-forward 0.597；label-permutation null 0.506 | 洩漏稽核 |
| p.41 | per-stock：3 檔顯著變好、3 檔變差；半導體 +0.008 [+0.001, +0.015] | 異質性 |
| p.41 | SOX + TSM ADR 精簡 S4 比完整 S4 差 0.020 [0.007, 0.032] | 共線性 |
| p.41 | 「move size AUROC 0.709，S1 單獨 0.707，籌碼 +0.001」 | 幅度 |
| p.73 | Brier 0.2419 vs 0.2495（BSS +3.0%）；rank IC 0.085（t = 12.1，65.6% 天數 > 0） | 校準／排序 |
| p.62 | 橫斷面相關 0.30 / PC1 34% | EDA |
| p.69 | 各規則與 LR 的逐年 AUROC 與 Brier | 主表 |
| p.56 | 「− 2,970 warm-up + missing」 | 已由 q20.json 證實：2,950 暖機 + 50 無標籤 + 20 法人缺漏 = 3,020 ✓ |

**可重現、可直接沿用的數字：** `crosscheck.json`（99.99% / 100%）、`data_quality.json`（93,879 → 90,851、缺值比例、漲跌停比例）、`eda_summary.json`、`fourclass_k.json`（k = 0.853 / 0.830 / 0.850 / 0.865%）、`bg_corr_check.json`、`daytrade_2025_sample.json`（36.67%）、`ta_followup_data.md` 的四分類分布表。

---

## 5. 文獻與外部來源核對（本次逐筆查證）

### 5.1 五篇 Related Works + SOTA

| 編號 | 論文 | 核對方式 | 結果 |
|---|---|---|---|
| [3] Yang et al. 2024, *Appl. Artif. Intell.* 38(1) 2429188 | Crossref ✓；全文（CC-BY）已於 `docs/literature_notes.md`、`ta_followup_data.md` 3-5 核對 | ✓ 67%（2330 SVM 全日平均 Table 5）、86%（1434 篩選日 Table 9）、82/86（Table 10）、隨機切分 |
| [18] Jeng, Liu & Guan 2024, *JISE* 40(4) 717–743 | Airiti 官方頁 ✓（DOI 10.6688 可解析至 Airiti，非 Crossref） | ✓ MARS 0.653 vs LSTM 0.505；p.27 Table 6 細項（0.5085 / 0.6442 / 0.6507 / 0.6532 等）**只在全文**，本次無法再開全文核對 → 保留但標「from Table 6 of [18]」 |
| [19] Zhang & Li 2026, *Mathematics* 14(9) 1494 | Crossref ✓；本機 PDF `exports/PULSE-KAN_Mathematics_2026.pdf` Table 1 ✓ | ✓ 0.5756 / 0.1509（adv, StockNet, 5 seeds） |
| [20] Luo et al. 2023, ACL, pp. 12164–12176 | Crossref ✓；原文站台擋機器人，改由 CausalStock Table 1 交叉核對 | ✓ 62.69 / 0.2090（ACL18）、53.43 / 0.0460、55.28 / 0.1110；p.35 的消融數字（52.88 / 0.039 等）無法再核，標示來源 Table 3 |
| [21] Li et al. 2024, NeurIPS 37 | arXiv 2411.06391 全文 ✓（本次下載解析） | ✓ ACL18 63.42 ± 0.0039 / 0.2172；CMIN-US 54.64 / 0.0481；KDD17 56.09 / 0.1235；w/o TCD 51.08 / 0.0102；w/o news 58.10 / 0.1421；10 runs（news）/ 5 runs；Sharpe 0.369 vs CMIN 0.357 vs market 0.107；新聞打分用 GPT-3.5 ✓。**p.36 引用缺頁碼**：NeurIPS 37 正式頁碼需查 proceedings（fix guide 寫 47432–47454，未核實，先用 arXiv 編號） |

### 5.2 其他引用（Crossref 全數通過）
[2] Wu et al. 2021 *Appl. Intell.* 51(11) 8119–8131 ✓；[4] Su et al. 2023 ✓；[5] Huang et al. 2026 BDCC 10(4) 109 ✓；[6] Chung et al. 2024 JCMSE 24(3) 2017–2035 ✓；[7] Chen et al. 2025 LNCS 15804 102–111 ✓；[8] Chen & Hsu 2025 *Discov. Comput.* 28(1) ✓（p.79 寫 Art. no. 248，Crossref 未回傳文章號，請保留 DOI 即可）；[9] Bui et al. 2023 ✓；[15] Chun et al. 2024 ✓；[16] Campisi et al. 2024 40(3) 869–880 ✓；[17] Huang, Tung & Lin 2023 ✓。

### 5.3 背景數字
| 數字 | 來源 | 結果 |
|---|---|---|
| 台積電 43.8%、前三大 52%（2026 年 5 月） | Taipei Times, N. P. Daly, 2026-07-31 | ✓ 原文「By May, TSMC accounted for 43.8 percent…」「the three represented 52 percent」；外資持有 47.42% ✓（原文未標日期） |
| 外資持有 45.41%（2024 年底） | TWSE Market Insights, J. Lin, 2025-02-21 | ✓ |
| 散戶成交 54.1%、外資 33.7%（2024） | TWSE Fact Book 2025 Table 3.04 | ✓ 54.07%；外資法人 33.72%（+自然人 0.04%） |
| 當沖 36.7%（2025，51 個週三抽樣） | 自算，`figures/daytrade_2025_sample.json` | ✓ 平均 36.67%，範圍 30.78–41.59% |
| 5 日 CNN 訊號 −0.54%、Sharpe −1.3 [7] | `ta_followup_data.md` 3-1 | ✓（Table 1 等權 buy-only） |
| 90.6%（32 日輸入窗口）[8] | `ta_followup_data.md` 3-2 | ✓；注意 32 日是輸入窗口不是測試期 |

### 5.4 Dataset Checklist 的「Q1 期刊」風險（仍未解決）
規定：資料須「published in related papers (conference proceedings or **Q1** journals)」。登記的 [2] *Applied Intelligence* 是否 Q1 未確認。建議期末 checklist 同時列 [9] *Pacific-Basin Finance Journal*（使用 TWSE 資料）與 [3]，並在備註寫明「TWSE official data; used in [2], [3], [9]」。

---

## 6. 需要使用者決定的事項

1. **Part 1 還是 Part 2？** 決定檔名（`group_A_part_1_team_04_final` 或 `part_2`）、截止日（11/21 或 11/28）與現場日期。**本報告與後續產出暫以 Part 1（11/21）為準**，因為較早的期限比較安全；若是 Part 2 只需改封面與檔名。
2. **課程 AI 使用規範原文**：專案目錄內所有 PDF 都找不到相關條文。本次採保守做法：所有 AI 協助產生的文字與程式都標示來源、每個數字可回溯到程式與資料、文獻全部經 Crossref / 原文核對、簡報附錄加一頁「AI 工具使用聲明」。請提供課程規範原文以便對照。
3. **分工表（H）**：需要三位成員實際負責項目。暫擬：梁欣童（資料管線、文獻、Introduction／Conclusion 講者）、劉霈琳（問題定義、方法設計、Problem Setting／Methods 講者）、林冠宇（實驗、統計檢定、Results 講者）。**這是依 Progress 講稿與 `qa_prep_zh.md` 0.1 的分配推估，必須由團隊確認。**
4. **Deployment（optional）**：Progress 規劃 Streamlit 每日排名 demo。建議期末**不做**，改以「每日 08:30 推論流程圖 + 一張範例排名表」交代，把時間留給 SOTA 重現與回測。

---

## 7. 期末簡報的證據生產計畫（接下來要做的事）

所有數字由 `scripts/final/` 下的新腳本產生，輸出到 `results/final/*.json` 與 `figures/final/*.png`，一行指令可重跑。

| 步驟 | 內容 | 對應章節 / Challenge |
|---|---|---|
| F1 資料與標籤 | 原有 45 特徵；新增標籤：open-to-close、gap、relative-to-market、**除權息調整（FinMind TaiwanStockDividendResult，已確認可免費取得）**、四分類（k = 訓練窗 \|r\| 中位數）、**波動度正規化四分類** | D-Preparation；C3（標籤雜訊）、C6 |
| F2 表格模型 | Rules、LR、LightGBM、XGBoost；**31 個來源子集全部跑**（exact source Shapley）；inner validation（訓練窗最後 12 個月）做 early stopping 與門檻；5 個種子 | D-Modeling；C1、C4、RQ1–RQ2 |
| F3 序列模型 | LSTM、Transformer（W = 5；S4 作靜態輸入），GPU，3 個種子 | RQ3 模型比較 |
| F4 統計推論 | 交易日 5 日區塊 bootstrap（1,000 次）、DeLong、逐日配對 t 檢定（HAC）、Holm 校正 | C1（低訊噪） |
| F5 洩漏稽核 | 隨機 5-fold vs walk-forward 比值、區塊標籤置換 null、future-shift 測試、特徵時間表 | D-Evaluation |
| F6 SOTA 重現 | (a) **Yang et al. 2024 流程**（LSTM 預測技術指標 → SVM/RF/LightGBM）在 Taiwan 50 上，分別用原文的隨機切分與我們的 walk-forward；(b) **StockNet 公開基準**（88 檔美股，2014–2016，官方時間切分與標籤規則）上跑我們的 LR / LightGBM 價格特徵，與 ALSTM 0.561 / Adv-ALSTM 0.562 / PULSE-KAN 0.576、CausalStock KDD17 56.09 對照；(c) 不重現 CausalStock（需新聞與 GPT-3.5 打分，說明理由） | E-Baseline、F-Results |
| F7 可解釋性 | grouped SHAP by source（逐年、逐產業）、per-sector / per-stock ΔAUROC 與 CI | RQ2、C5 |
| F8 回測 | top-k（k 由 inner validation 決定）long-only，t+1 開盤進、收盤出；成本 0.1425%×2 + 0.15%（當沖）/ 0.3%；對照 0050 buy & hold；年化報酬、Sharpe、MDD、turnover、PSR | C6 |
| F9 新聞來源 | **評估結論：不可行。** FinMind `TaiwanStockNews` 每次只回傳一天、一檔（50 檔 × 1,878 日 ≈ 94,000 次請求，免費額度不夠）；GDELT 無個股對應；自行爬鉅亨網／PTT 的時間與版權風險超出期末範圍。期末以「Limitation / Future work」交代，並引用 [8]、[18]、[21] 說明新聞可能的增益範圍 | G |
| F10 簡報與講稿 | 依 A–H 大綱產出 pptx（pptxgenjs，含頁碼、目錄、每頁引用）、中文講稿（三人分段、含秒數）、分工表、附錄參考文獻（IEEE） | 全部 |
