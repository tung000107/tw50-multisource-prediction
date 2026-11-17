"""Generate the Chinese speech script (three speakers, ~19 min) from results/final/*.json so that every number matches the deck.
Output: final_present/02_講稿_期末簡報.md"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, ROOT, SRC
J = lambda n: json.load(open(RES / n))
M, E, L, A, SN, SD, Y, B = [J(n) for n in ["main_results.json", "explain.json", "labels.json", "audits.json", "sota_stocknet.json", "sota_stocknet_deep.json", "sota_yang.json", "backtest.json"]]
HH = J("sota_headtohead.json") if (RES / "sota_headtohead.json").exists() else None; CV = J("causalstock_v2.json") if (RES / "causalstock_v2.json").exists() else None; XR = J("external_runs.json") if (RES / "external_runs.json").exists() else None; CN = J("causal_news_tw.json") if (RES / "causal_news_tw.json").exists() else None; GF = J("graph_features.json") if (RES / "graph_features.json").exists() else None; TS = J("tw_structure.json") if (RES / "tw_structure.json").exists() else None; NA = J("news_analysis.json") if (RES / "news_analysis.json").exists() else None; LK = J("llm_lookahead.json") if (RES / "llm_lookahead.json").exists() else None; SS = J("source_study.json") if (RES / "source_study.json").exists() else None
auc = lambda g: M["arms"][g]["seed_mean_prob"]["pooled"]["auroc"]; cmp = lambda k: M["comparisons"][k]
f3 = lambda x: f"{x:.3f}"; sg = lambda x: f"{x:+.3f}"; pc = lambda x, d=1: f"{100*x:.{d}f}%"
D = lambda k: f"{sg(cmp(k)['diff'])}（95% CI {sg(cmp(k)['ci'][0])}～{sg(cmp(k)['ci'][1])}）"
v = L["binary_variants"]; r4 = L["four_level"]["raw"]; sh = E["shapley_LightGBM"]; wd = E["within_day"]; st = B["strategies"]
full, s4, tw = auc("LightGBM|Full"), auc("LightGBM|S4-only"), auc("LightGBM|TW-only"); sox = auc("Rule|SOX return score")
lstm, trf, lr, xgb = auc("LSTM-W5|Full"), auc("Transformer-W5|Full"), auc("LR|S1+S2+S3+S4+S5"), auc("XGBoost|Full")
seed_std = M["arms"]["LightGBM|Full"]["auroc_seed_std"]
SO = XR.get("stocknet_official_acl18") if XR else None
if SO and SO.get("hedge"):
    _h = SO["hedge"]["mean"]; _lvl = "也沒到文獻水準" if _h["test_acc"] < 0.56 else "接近文獻"
    SO_TXT = f"官方程式我們也跑了 {_h['n_runs']} 次：HedgeFundAnalyst 平均 {f3(_h['test_acc'])}、MCC {f3(_h['test_mcc'])}，{_lvl}；而且官方程式的資料過濾只留 {SO['counts']['test']['n']:,} 個測試樣本，多數類就有 {f3(SO['test_majority_acc'])}。"
else: SO_TXT = "沒有重現出來。"
out = f"""# 期末簡報中文講稿（Group A Team 4，目標 19 分 00 秒 ± 30 秒）

- 本講稿由 `scripts/final/12_speech.py` 從 `results/final/*.json` 產生，數字與簡報完全一致；重跑實驗後請重新產生。
- 三位講者：**梁欣童**（封面、Introduction、Conclusion、分工表，約 6 分）、**劉霈琳**（Problem Setting、Proposed Methods，約 5.5 分）、**林冠宇**（Experimental Setting、Results、Insights，約 7.5 分）。
- 專有名詞（AUROC、MCC、walk-forward、LightGBM、Shapley、bootstrap）維持英文念法。粗體句是該頁結論，放慢講。
- 影片規定：約 20 分鐘，**超過 20 分鐘全隊只得 50%**；不得後製加速；錄完務必確認總長。

---

## 講者一：梁欣童（p.1–p.10）

### p.1 封面｜25 秒
大家好，我們是 Group A 第 4 組，我是梁欣童。題目是「結合多來源資訊之台灣股票次日漲跌趨勢分類」。Proposal 的時候我們問：台灣自己的籌碼、大盤資料，能不能比前一晚的美股多告訴我們一點明天漲還是跌？今天我們用一套完整、可重現的實驗回答這個問題，而且答案分成兩半：**美股決定「哪一天」，台灣資料決定「哪一檔」和「漲多少」**。

### p.2 Outline｜20 秒
報告依規定分成 A 到 H：Introduction、Problem Setting、Proposed Methods、Experimental Setting、Experimental Results、Conclusion and Limitation，最後是分工表。頁碼都標在目錄上，方便老師助教對照。

### p.3 分隔頁｜5 秒
先從背景開始。

### p.4 Background：台股市場結構｜50 秒
三個數字：台積電一家佔上市市值 43.8%，前三大科技股佔 52%，外資持有 45 到 47%。右下角的 0.49 是前一晚費半報酬和隔天加權指數報酬的相關係數；加權指數自己的前一日報酬只有 0.03。左圖可見加權指數幾乎跟著費半走。**所以台股的價格同時被個股交易、本地法人籌碼、大盤、還有隔夜的全球科技股推動；哪一個真的有預測力，要量，不能假設。**

### p.5 Background：五個資訊來源與時間｜45 秒
開盤前可用的公開資訊有五類：S1 個股價量，一點半收盤就有；S2 法人籌碼與融資券，收盤後公布；S3 大盤；S4 美股與全球市場，台灣時間清晨五點收盤；S5 日曆。**圖的右下角：單獨看只有 S4 強；台灣這幾個弱來源合起來有沒有價值，就是多來源的問題。**

### p.6 Background：文獻與缺口｜45 秒
文獻分五類，準確率從 0.67 報到 0.86，但協定各異；另類資料的 SOTA 是 NeurIPS 2024 的 CausalStock。**缺口是：沒有人在同一套 walk-forward 協定下，把籌碼、大盤、美股對個股次日方向的邊際價值分開量，而且附上不確定性。這就是我們做的事。**

### p.7 Motivation｜50 秒
三個動機。第一，文獻的數字沒辦法比：我們自己量，同一個 LightGBM 隨機切分 AUROC {f3(A['random_5fold_full']['auroc_test_years_only'])}，照時間順序只有 {f3(A['walk_forward_full']['auroc'])}，差了 {A['leakage_ratio']:.2f} 倍。第二，台灣資料單獨只有 {f3(tw)}，美股單獨 {f3(s4)}；proposal 時的「融合贏 0.014」在正式協定下變成 {D('RQ1 Full vs S4-only (LightGBM)')}。第三，來回成本 0.435 到 0.585%，可是每天漲跌幅中位數才 0.85%，**準不等於賺，所以最後一定做含成本的回測。**

### p.8 Project Aims｜50 秒
主目標不變：融合五個來源做台灣 50 個股的次日漲跌分類，並量化每個來源的貢獻。RQ1 問融合有沒有顯著贏過最強單一來源，這次還在跳空、開盤後、相對大盤、幅度四種定義上問。RQ2 用 31 個來源子集算 exact Shapley 值。RQ3 看來源有沒有交互作用，以及線性、樹、序列模型誰用得到。期中之後的延伸也掛在同樣三個問題下：多加四個來源和台股關係圖回答 RQ1、RQ2，和已發表架構在同一份資料上比較回答 RQ3。

### p.9 Challenges｜55 秒
六個挑戰各有資料證據。C1 訊噪比低，換個種子 AUROC 就差 {seed_std:.3f}，所以多種子、交易日 bootstrap。C2 來源不獨立，所以 31 個子集算 Shapley。C3「明天」混了跳空和盤中，所以拆標籤。C4 五十檔同一天一起動，有效樣本 900 天，所以信賴區間以交易日為單位。C5 協定比模型更影響分數，所以做洩漏稽核。C6 利潤薄，所以含成本回測。

### p.10 Contributions｜45 秒
四個貢獻：一套可重現的資料管線，官方資料逐筆核對；六種模型家族乘 31 個來源子集的同協定 benchmark；量化來源貢獻，而且把「明天」拆開來看；以及協定與文獻可轉移性的證據，包含在同一份台灣資料上和已發表架構的直接比較。

### p.11 Problem Setting 分隔頁｜5 秒
接下來由霈琳說明問題定義和方法。

---

## 講者二：劉霈琳（p.11–p.21）

### p.12 Problem Description｜45 秒
大家好，我是劉霈琳。輸入是交易日 t 每檔台灣 50 成分股的五組資訊；輸出是隔天收盤高於今天收盤的機率。問題陳述：給定 t+1 開盤前所有可觀測資訊，預測 Close(t+1) 是否大於 Close(t)，並量化每個來源的貢獻。平盤和下跌都是 0。

### p.13 資訊時間與五種「明天」｜50 秒
左圖是推論時間線：決策點在 t+1 早上八點半，台股 t 日的資料和 t 日的美股盤都已經知道，t+1 的美股盤還沒發生，所以不會偷看未來。右表是五種標籤：收盤到收盤、跳空、開盤到收盤、相對大盤、四分類。**重點是：收盤到收盤等於跳空加盤中，拆開來才知道哪個來源推動哪一段。**

### p.14 Evaluation Metrics 與驗證協定｜55 秒
主指標是 AUROC，輔助 MCC、Macro-F1、Brier。驗證是 expanding-window walk-forward，四個 fold 測 2023 到 2026 年 9 月；inner validation 是訓練窗最後 12 個月，負責 early stopping、門檻和調參；前處理只用訓練窗；交界空一天。**因為 50 檔同一天共享同一個衝擊，有效樣本接近 900 天，所以每個信賴區間都是以連續 5 個交易日為一塊重抽 1,000 次的 block bootstrap。**

### p.15 Methods 分隔頁｜5 秒
方法依五個步驟。

### p.16 Methods 總覽｜45 秒
五步驟對應六個挑戰：Understanding 處理 C1、C4，Preparation 處理 C3、C5，Modeling 處理 C1、C2，Evaluation 處理 C2、C4、C5、C6。固定順序：依時間切分、訓練窗上擬合前處理、inner validation 調參、refit、預測測試年、滾到下一年。

### p.17 Data Understanding｜50 秒
官方端點逐筆對 FinMind 鏡像，99.99% 一致。再看訊號多弱：台灣端特徵的相關都小於 0.05；隔夜費半從最差到最好五分位，上漲率從 34.6% 升到 57.5%。50 檔日報酬兩兩相關 0.28，所以信賴區間用交易日重抽。

### p.18 Data Preparation｜50 秒
93,879 筆原始資料，去掉零成交量、無標籤、暖機和缺漏，剩 90,851 筆。45 個特徵都是比率和報酬，原始價格不進模型。四分類門檻 k 是訓練窗 |r| 中位數，0.83 到 0.86%，另做一個除以 20 日波動度的版本。

### p.19 Data Modeling｜55 秒
六個模型家族：規則、L2 邏輯迴歸、LightGBM、XGBoost、LSTM、Transformer。相同 fold、驗證規則和種子，不做逐臂調參。最終模型是 LightGBM 和 LSTM 在九個來源上的機率平均；已發表的架構則用官方程式原樣跑在同樣的列上。**關鍵設計是 S1 到 S5 的 31 個非空組合全部跑，所以每個來源的 Shapley 值和交互作用是精確算的，不是近似。**

### p.20 Evaluation 設計｜55 秒
推論用交易日 block bootstrap，DeLong、Holm 一起報，宣稱規則是 CI 不含 0 且四年至少三年同號；歸因用 exact Shapley 和 grouped SHAP；稽核有隨機切分對比、標籤置換、灌入未來；回測每天買前 5 或 10 名，開盤進收盤出，成本 0.435%。

### p.21 Deployment（選配）｜25 秒
部署只做原型：一行指令重建整個研究，加上一個每天八點半打分數的腳本。右邊是最後一個測試日的樣本外排名，同一天 50 檔的機率差距很小，這就是為什麼「選股」訊號天生很弱。接下來由冠宇說明實驗設定與結果。

---

## 講者三：林冠宇（p.22–p.47）

### p.22 分隔頁｜5 秒
大家好，我是林冠宇。

### p.23 Datasets 描述｜45 秒
主資料是證交所官方日資料；外部資料是 Yahoo Finance 的九個美股與全球序列。台灣 50 的 50 檔、六個產業，2019 到 2026 年 9 月，90,851 筆、45 個特徵；測試集 900 個交易日，沒有用來做任何選擇。

### p.24 Dataset Checklist｜30 秒
六項規定逐一確認：Finance and Economics；真實資料；公開來源，且同樣的台灣 50 日資料在三篇期刊論文用過，其中 Applied Intelligence 是否 Q1 會再跟助教確認；90,851 筆；特徵都有金融定義；二元分類加四分類次任務。資料頁的引用都標在頁面上。

### p.25 EDA｜35 秒
左邊：每年類別接近平衡；四分類的 strong up 逐年從 24% 升到 35%。右邊相關矩陣：跟標籤相關超過 0.05 的特徵全在 S4，且 S4 內部高度共線，所以歸因在來源層級做。

### p.26 Baseline 與 SOTA｜55 秒
文獻 SOTA 是 CausalStock，ACL18 上 63.42%、MCC 0.217；沒有公開程式碼，我們自己重新實作，在它的基準和台灣 50 上都跑；Adv-ALSTM 有官方程式，我們原樣執行。台灣 baseline 是 Yang 等人，流程在台灣 50 上重做。**公平比較原則：只在同資料、同切分、同標籤規則上比數字；所以我們唯一說「贏」的地方，是台灣 50 上的同資料比較。**

### p.27 Evaluation Strategy｜35 秒
為什麼不用 stratified k-fold：隨機 fold 會把 t+1 放進訓練、t 放進測試，模型學到的是測試期的 regime，等一下會看到差 {A['leakage_ratio']:.2f} 倍。取而代之的是年度 fold、inner validation、多種子、block bootstrap、DeLong、Holm，比較家族事先定好。**一個要坦白的限制：2023 到 2026 在 proposal 階段就看過，協定是確認性的，但歷史不是；乾淨的測試是 2026 年 10 月以後的新資料。**

### p.28 Environment｜20 秒
Ubuntu 22.04、24 核、93 GB、RTX 4090；Python 3.13 用 uv 鎖版本；LightGBM 4.7、XGBoost 3.4、PyTorch 2.6、scikit-learn 1.9、SHAP 0.54；簡報由 pptxgenjs 從結果 JSON 產生。

### p.29 Results 分隔頁｜5 秒
結果分四組：主任務、拆解「明天」、稽核與回測，以及外部資料、因果圖與新聞。

### p.30 R1 主任務｜60 秒
主結果，五種模型、三種資訊集。第一，收盤到收盤方向上融合和只用美股打平，LightGBM Full 減 S4-only 是 {D('RQ1 Full vs S4-only (LightGBM)')}，其他四種模型都一樣，種子標準差 {seed_std:.4f} 跟這個差距一樣大。第二，台灣資料單獨 {f3(tw)}，幾乎等於個股基本上漲率規則的 {f3(auc('Rule|stock base-rate'))}。第三，五種模型家族全部打平。**所以 RQ1 在主任務上的答案是：不成立。**

### p.31 R2 逐年與配對檢定｜45 秒
左圖：每一年美股資訊的臂都在 0.58 到 0.61，台灣資料 {min(x['auroc'] for x in M['arms']['LightGBM|TW-only']['seed_mean_prob']['by_year'].values()):.2f} 到 {max(x['auroc'] for x in M['arms']['LightGBM|TW-only']['seed_mean_prob']['by_year'].values()):.2f}，沒有例外。右表是事先定好的比較，主要看 block bootstrap 的 CI；DeLong 把 4 萬 5 千筆當獨立樣本所以過度樂觀。依宣稱規則，融合對美股、以及模型家族的差異都不算真的。

### p.32 R3 來源 Shapley｜50 秒
LightGBM 的 Shapley 值：S4 {sg(sh['phi']['S4']['value'])}，佔總共 {sg(sh['sum_phi'])} 的絕大部分；S1 {sg(sh['phi']['S1']['value'])}、S2 {sg(sh['phi']['S2']['value'])} 很小但 CI 在 0 以上，S3、S5 含 0；LR 一樣。兩兩交互作用比 S4 小一個數量級，來源接近可加，「籌碼會改變美股效果」的假設沒有被支持。

### p.33 R4 拆解「明天」｜70 秒
**這是最重要的一頁。**同樣的特徵，換五種目標。跳空：美股單獨 {f3(v['y_gap']['S4-only']['auroc'])}，融合 {f3(v['y_gap']['Full']['auroc'])}，多 {sg(v['y_gap']['Full']['auroc'] - v['y_gap']['S4-only']['auroc'])}，顯著：台灣資料告訴你每一檔會跳多少。開盤到收盤：美股被開盤價吸收後就沒用了，S4 只有 {f3(v['y_intra']['S4-only']['auroc'])}；只剩台灣資料有一點訊號 {f3(v['y_intra']['TW-only']['auroc'])}。相對大盤：融合 {f3(v['y_rel']['Full']['auroc'])} 對美股 {f3(v['y_rel']['S4-only']['auroc'])}，差 {sg(v['y_rel']['Full']['auroc'] - v['y_rel']['S4-only']['auroc'])}，四年同號。**所以 RQ1 的完整答案是：融合在「台灣資料答得出來的問題」上有幫助：美股決定哪一天，台灣資料決定哪一檔。**

### p.34 R5 同一天內排序與校準｜40 秒
美股給 50 檔同一個分數，所以同一天內的 AUROC 必然是 0.5；個股基本上漲率 {f3(wd['stock base-rate']['mean'])}，台灣資料 {f3(wd['TW-only']['mean'])}，融合 {f3(wd['Full']['mean'])}，配對 t 值 {wd['Full_minus_S4_days']['t']:.1f}。rank IC 只有 {f3(wd['IC Full']['mean'])}，{pc(wd['IC Full']['share_gt'], 0)} 的日子為正；校準上，模型說 55% 時真的大約 55%。

### p.35 R6 四分類與幅度｜45 秒
幅度是台灣端的訊號：明天是不是大漲大跌日，台灣資料 {f3(r4['magnitude_TW-only']['auroc'])}，美股只有 {f3(r4['magnitude_S4-only']['auroc'])}。四分類 Macro-F1 融合 {f3(r4['four_Full']['macro_f1'])}、美股 {f3(r4['four_S4-only']['macro_f1'])}，差距的 CI 是 {sg(r4['Full_minus_S4_macro_f1'][0])} 到 {sg(r4['Full_minus_S4_macro_f1'][1])}；用波動度正規化後幅度訊號縮小，因為除掉了預測它的那個特徵，**也就是說「漲跌幅度大不大」主要就是「這檔最近波動大不大」。**

### p.36 R7 產業與個股｜35 秒
美股訊號在半導體和電子最強、電信最弱；融合的增益在每個產業都小，只有電信的 CI 不含 0，而且還沒做多重比較校正。逐檔來看，50 檔裡 {E['by_stock_summary']['n_better']} 檔為正，{E['by_stock_summary']['n_sig_up']} 檔顯著變好、{E['by_stock_summary']['n_sig_down']} 檔顯著變差，50 次 5% 檢定本來就會給 2 到 3 個假陽性，所以這離隨機不遠，也沒有產業型態。

### p.37 R8 grouped SHAP｜30 秒
模型內部的歸因：S4 佔大約 {pc(list(E['grouped_shap'].values())[-1]['share']['S4'], 0)}，台灣特徵佔剩下的 {pc(1 - list(E['grouped_shap'].values())[-1]['share']['S4'], 0)}，逐年穩定。**模型「用了」台灣特徵，不等於它們帶來樣本外訊號，這就是為什麼 Shapley over subsets 才是主要歸因，SHAP 是交叉檢查。**

### p.38 R9 協定稽核｜40 秒
同一個模型、同樣的列，只換協定。隨機 5-fold {f3(A['random_5fold_full']['auroc_test_years_only'])} 對 walk-forward {f3(A['walk_forward_full']['auroc'])}，{A['leakage_ratio']:.2f} 倍。訓練標籤 20 天區塊置換，剩 {' / '.join(f3(x) for x in A['block_permutation_null_auroc'])}。故意灌入 t+1 的美股盤，升到 {f3(A['future_shift_full'])}，代表這套流程抓得到一天的洩漏。

### p.39 R10 重現文獻｜60 秒
在文獻自己的條件上重測。StockNet 官方切分：我們自己的模型 0.50 到 0.52，文獻 0.56 到 0.58。{SO_TXT}Yang 的流程放到台灣 50：隨機切分約 {f3(Y['random_split']['raw indicators (Table 4 analogue)']['RF']['acc'])}，預測指標幾乎沒加分。

### p.40 R10b SOTA 重現｜70 秒
{("" if CV is None else f"""SOTA 重現是我們最意外的發現。CausalStock 沒有公開程式碼，我們依論文附錄自己實作。在 ACL18 上，照該重現者的方式讀標籤，也就是把價格檔第 6 欄當收盤價，我們得到 {CV['acl18_dclose']['CausalStock v2 w/o news']['mean']['acc']:.3f}，跟論文的 0.634 和重現者的 0.637 幾乎一樣，而且**不用新聞也一樣**。但第 6 欄其實是「收盤價日變化量」，這個標籤等於問「今天漲幅是否比昨天大」。改用論文定義的正確標籤，同樣的程式、同樣的種子，掉到 {CV['acl18_mov']['CausalStock v2 w/o news']['mean']['acc']:.3f}，多數類是 {CV['acl18_mov']['majority_acc_test']:.3f}。搬到台灣 50，和只用價格的 LightGBM 一樣。我們還用重現倉庫的原始程式再驗證：他們的標籤 {f3(XR['hwang_repo_own_code_acl18_no_news']['their label (column 6 = daily dollar change)']['test_acc'])}，只改一行標籤定義變成 {f3(XR['hwang_repo_own_code_acl18_no_news']['corrected label (movement > 0)']['test_acc'])}。官方 Adv-ALSTM 程式只給價格跑在台灣 50，準確率只比多數類高不到一個百分點。**我們能宣稱的是：0.634 只有在標籤錯誤時才重現得出來；原作者有沒有犯同樣的錯，沒有程式碼無法判斷。**""")}

### p.41 R10c 同資料 head-to-head｜45 秒
{("" if HH is None else f"""這一頁回答「我們有沒有贏 SOTA」。做法是把已發表的架構放到我們的台灣 50 資料上，同樣的列、同樣的切分、同樣的標籤；每個架構跑兩次，一次只給價格，一次給我們的 45 個特徵。只給價格時，Adv-ALSTM {f3(HH['rows']['AdvALSTM official (11 price features)']['auroc'])}、CausalStock {f3(HH['rows']['CausalStock v2 (price-only, 11)']['auroc'])}；給了我們的特徵，Adv-ALSTM 升到 {f3(HH['rows']['AdvALSTM official (our 45 features)']['auroc'])}。我們的最終模型是 {f3(HH['rows'][HH['final_model']]['auroc'])}，比最好的 SOTA 高 {sg(HH['paired'][HH['final_model'] + ' minus ALSTM official (our 45 features)']['diff'])}，信賴區間不含零，四年同號。要誠實說的是：特徵完全相同時，官方 ALSTM 和我們的 LightGBM 打平。**所以我們贏在資訊來源和模型平均，不是架構；這也再次說明，決定結果的是餵進去的資訊。**""")}

### p.42 R11 回測｜50 秒
誠實的結果：扣掉牌告成本每個策略都賠。最好的訊號是台灣資料的開盤到收盤排名，前五名損益兩平成本每天 {pc(st['p_oc_TW-only|top5']['break_even_round_trip_cost'], 3)}，但要付 {pc(B['cost_round_trip'], 3)}。毛報酬上排序有價值：TW-only 前五名 Sharpe {st['p_oc_TW-only|top5']['long_only_gross']['sharpe']:.2f}。**背景是：這 50 檔收盤到收盤一年賺 {pc(B['benchmarks']['equal_weight_50_close_to_close_gross']['ann_return'], 0)}，開盤到收盤卻是 {pc(B['benchmarks']['equal_weight_50_intraday_gross']['ann_return'], 0)}，市場的報酬都在隔夜，剛好是美股訊號所在、也是當沖進不去的地方。**

### p.43 R12 外部資料來源比較｜60 秒
{("" if SS is None else f"""依老師建議我們加了四個外部來源：S6 鉅亨網新聞標題，自帶個股標籤，文本分數在訓練窗內自己訓練、不用 LLM；S7 跨股票與產業、龍頭特徵；S8 台指期夜盤與法人期貨部位；S9 當沖、借券、月營收。表格是加到原本五個來源上的 AUROC 變化。{"有顯著增益的來源：" + "、".join(k for k in ["S6","S7","S8","S9"] if any(SS[l][f"Full+{k} minus Full"][0] > 0 for l in ["y_up","y_gap","y_intra","y_rel"])) if any(SS[l][f"Full+{k} minus Full"][0] > 0 for k in ["S6","S7","S8","S9"] for l in ["y_up","y_gap","y_intra","y_rel"]) else "沒有任何一個來源在任何標籤上有 CI 不含 0 的增益"}。{("九個來源的 exact Shapley：美股 " + f"{SS['shapley9_y_up']['phi']['S4']:+.3f}" + "，新來源 " + "、".join(f"{k} {SS['shapley9_y_up']['phi'][k]:+.3f}" for k in ["S6","S7","S8","S9"]) + "。") if "shapley9_y_up" in SS else ""}""")}

### p.44 R13 台股因果圖加新聞｜60 秒
{("" if (CN is None or GF is None or TS is None) else f"""我們分三步建台股因果圖。第一步先看結構：台灣 50 同一天的連動很強，同產業相關 {TS['contemporaneous']['mean_corr_within_sector']:.2f}、跨產業 {TS['contemporaneous']['mean_corr_cross_sector']:.2f}；但「今天誰動、明天誰跟」很弱，{TS['leadlag']['n_pairs_tested']} 對股票只有 {TS['leadlag']['n_significant_holm05']} 對顯著。第二步把三種圖放進 CausalStock：學出來的圖、我們用產業和供應鏈關係手建的先驗圖、Lasso 圖，再各自加上我們自己訓練的新聞分數。**結果全部在 {min(x['auroc'] for x in CN['pooled_seed_mean'].values()):.3f} 到 {max(x['auroc'] for x in CN['pooled_seed_mean'].values()):.3f} 之間，LightGBM 五來源是 {full:.3f}**；加新聞的差是 {CN['pooled_seed_mean']['learned | +news']['auroc'] - CN['pooled_seed_mean']['learned | price']['auroc']:+.3f}，信賴區間包含零。第三步把圖做成特徵進 LightGBM，對收盤到收盤沒有增益，只對盤中任務有 {GF['y_intra']['Full+S10']['auroc'] - GF['y_intra']['Full']['auroc']:+.3f}。結論：台股的關係圖是真的，但它解釋的是「今天誰一起動」，不是「明天誰會動」；這個共同因子已經被美股和台指期夜盤帶進來了。""")}

### p.45 R14 新聞的其他用途｜55 秒
{("" if NA is None else f"""第一，新聞預測「會不會大動」：新聞則數預測明天漲跌幅是否超過中位數，AUROC 各年 {" / ".join(f"{NA['univariate_news_vs_magnitude'][y]['n_news']['auroc_magnitude']:.2f}" for y in NA['univariate_news_vs_magnitude'])}，預測方向只有 0.5；但 20 日波動度更會做這件事，所以加新聞沒有增益。第二，關鍵字事件：報導已發生價格事件的標題當天就反映完，「漲停」標題當天 {100*NA['keyword_events']['limit-up reported']['same_day_ret_mean']:.1f}%、隔天 {100*NA['keyword_events']['limit-up reported']['next_mean']:.2f}%；帶新資訊的標題隔天還有漂移，分析師調升目標價隔天平均 {100*NA['keyword_events']['target price raised / upgrade']['next_mean']:.2f}%、上漲機率 {100*NA['keyword_events']['target price raised / upgrade']['up_rate']:.0f}%，減資 {100*NA['keyword_events']['capital reduction']['next_mean']:.2f}%，但這類事件只占不到一成的股票日。第三，AI 新聞最多的日子，AI 概念股隔天的漲跌幅和上漲機率都沒有放大。第四，營收標題在收盤後發布，隔天反應和年增率成正比，年增超過五成的上漲機率 {100*NA['revenue_headlines']['> 50 %']['up_rate']:.0f}%，年減的只有 {100*NA['revenue_headlines']['-20..0 %']['up_rate']:.0f}%。**結論：新聞在「帶有新資訊的事件」和「幅度」上有用；對所有股票日的漲或不漲，仍然是美股隔夜盤在決定。**""")}

### p.46 R15 LLM 新聞分數與前視偏誤｜50 秒
{("" if LK is None else f"""助教擔心用新的大模型打新聞分數會「背到」測試期的新聞，我們把這個疑慮做成實驗：本機的開源模型 Qwen3-4B，2025 年 4 月釋出，給 {LK['n_scored_headlines']:,} 則標題打分。加進五來源後，在模型的知識截止日 2025 年 3 月底以前、也就是它可能看過的期間，增益是 {LK['y_up']["up to 2025-03-31 (on or before the stated knowledge cut-off)"]['Full+LLM'] - LK['y_up']["up to 2025-03-31 (on or before the stated knowledge cut-off)"]['Full']:+.3f}；它釋出之後的期間，是 {LK['y_up']['from 2025-04-30 (after the weights were released)']['Full+LLM'] - LK['y_up']['from 2025-04-30 (after the weights were released)']['Full']:+.3f}。{"前段比後段大，這就是背答案的痕跡，誠實的估計要看後段。" if (LK['y_up']["up to 2025-03-31 (on or before the stated knowledge cut-off)"]['Full+LLM'] - LK['y_up']["up to 2025-03-31 (on or before the stated knowledge cut-off)"]['Full']) - (LK['y_up']['from 2025-04-30 (after the weights were released)']['Full+LLM'] - LK['y_up']['from 2025-04-30 (after the weights were released)']['Full']) > 0.005 else "兩段差不多，用 4B 模型加標題看不出背答案的痕跡，但誠實的估計仍然只能看後段。"}""")}

### p.47 Insights 總表｜60 秒
文獻的九個說法逐條檢查：外部資訊有用，成立且更極端；籌碼對個股次日方向，在知道美股後沒有；融合有幫助，看問題；簡單模型不輸深度模型，成立；短回看夠用，成立；分數取決於協定，成立；增益因股票而異，接近假陽性數量；方向訊號可交易，在散戶成本下不行；新聞情緒能預測方向，只在幅度和帶日期的事件上成立。接下來由欣童做結論。

---

## 講者一：梁欣童（p.48–p.51）

### p.49 Conclusion｜60 秒
RQ1：在規定的收盤到收盤任務上，融合沒有贏過只用美股，所有模型都在 0.59 附近；但在跳空、盤中、相對大盤、幅度這些台灣資料答得出來的問題上，融合有顯著幫助。RQ2：Shapley 顯示美股幾乎是全部的方向訊號。但加上台指期夜盤和跨股票結構後，融合比只用美股多 {SS['fusion_vs_us_only']['Full+S7+S8']['auroc'] - SS['fusion_vs_us_only']['S4-only']['auroc']:+.3f}，{SS['fusion_vs_us_only']['Full+S7+S8 minus S4-only same-sign years']} 年同號；新聞、因果圖、LLM 都改變不了方向的答案。RQ3：來源接近可加，模型家族打平；真正改變數字的是標籤定義和評估協定。和已發表架構在同一份資料上比，我們的最終模型領先，但領先來自來源和模型平均，不是架構。**一句話：美股決定哪一天，台灣資料決定哪一檔和多少，而協定決定你能宣稱什麼。**

### p.50 Limitations and Future Work｜55 秒
限制：歷史不乾淨，乾淨的測試是 2026 年 10 月以後的新資料；存活者偏誤；特徵端沒有除權息還原；6.6% 的列重複使用前一天的美股盤；回測沒有滑價，也沒回測隔夜持有；部署只有設計；已發表架構用的是論文的超參數，沒有替它們做同等的調參。新聞只用了標題；全文、中文金融語言模型、PTT 情緒都還沒試。

### p.51 分工表｜20 秒
分工表如投影片，三個人在影片和現場都會發言，各自負責自己講的章節。以上是第 4 組的期末報告，謝謝大家。

---

### 時間總計（估）
請以本檔最後一行的字數估算為準。錄完請以播放器確認總長 < 20:00。
"""
Path(ROOT / "final_present" / "02_講稿_期末簡報.md").write_text(out); print("speech written", len(out))
