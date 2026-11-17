"""Generate the one-document summary of the whole final study: every experiment, how it was compared, conclusions, opinions,
and the comparison with the Progress Update (new / completed / contradicted). Every number is read from results/final/*.json,
so the document stays consistent with the deck (deck/build.js) and the speech (12_speech.py).
Static facts that are not in results/final (Progress Update page numbers, EDA cross-check figures, literature values) are marked
[PU] (Progress Update deck), [EDA] (figures/*.json of the data-understanding step) or [lit] (paper tables verified in 01_查核報告).
Output: final_present/05_總整理_全部實驗與結論.md"""
import json, sys, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, ROOT
J = lambda n: json.load(open(RES / n)) if (RES / n).exists() else None
M, E, L, A, SN, SD, Y, B = [J(n) for n in ["main_results.json", "explain.json", "labels.json", "audits.json", "sota_stocknet.json", "sota_stocknet_deep.json", "sota_yang.json", "backtest.json"]]
CV, XR, SS, TS, CN, GF, GP, MF, NA, SC, LK = [J(n) for n in ["causalstock_v2.json", "external_runs.json", "source_study.json", "tw_structure.json", "causal_news_tw.json", "graph_features.json", "gap_fills.json", "more_fills.json", "news_analysis.json", "tw_supply_chain.json", "llm_lookahead.json"]]
SO = XR.get("stocknet_official_acl18") if XR else None

f3 = lambda x: "—" if x is None else f"{x:.3f}"; sg = lambda x: "—" if x is None else f"{x:+.3f}"; pc = lambda x, d=1: "—" if x is None else f"{100*x:.{d}f}%"
ci = lambda c: f"[{sg(c[0])}, {sg(c[1])}]"; cia = lambda c: f"[{f3(c[0])}, {f3(c[1])}]"
auc = lambda g: M["arms"][g]["seed_mean_prob"]["pooled"]["auroc"]; cmp = lambda k: M["comparisons"][k]
D = lambda k: f"{sg(cmp(k)['diff'])} {ci(cmp(k)['ci'])}"
v = L["binary_variants"]; r4 = L["four_level"]["raw"]; vn = L["four_level"]["volnorm"]; sh = E["shapley_LightGBM"]; shl = E["shapley_LR"]; wd = E["within_day"]; st = B["strategies"]
full, s4, tw = auc("LightGBM|Full"), auc("LightGBM|S4-only"), auc("LightGBM|TW-only")
seed_std = M["arms"]["LightGBM|Full"]["auroc_seed_std"]
yrs = ["2023", "2024", "2025", "2026"]
byyear = lambda g: " / ".join(f3(M["arms"][g]["seed_mean_prob"]["by_year"][y]["auroc"]) for y in yrs)
today = datetime.date.today().isoformat()

def row(*cells): return "| " + " | ".join(str(c) for c in cells) + " |"
def table(header, rows): return "\n".join([row(*header), "|" + "---|" * len(header)] + [row(*r) for r in rows])

# ---------- 1. main results table ----------
fams = [("規則：費半隔夜報酬當分數", "Rule|SOX return score"), ("規則：分股訓練期上漲率", "Rule|stock base-rate"), ("LR", "LR|S1+S2+S3+S4+S5"), ("LightGBM", "LightGBM|Full"), ("XGBoost", "XGBoost|Full"), ("LSTM (W = 5)", "LSTM-W5|Full"), ("Transformer (W = 5)", "Transformer-W5|Full")]
main_rows = []
for name, g in fams:
    if g.startswith("Rule"): main_rows.append([name, f3(auc(g)), "—", "—", "—"]); continue
    fam = g.split("|")[0]; s4g = {"LR": "LR|S4", "LightGBM": "LightGBM|S4-only", "XGBoost": "XGBoost|S4-only", "LSTM-W5": "LSTM-W5|S4-only", "Transformer-W5": "Transformer-W5|S4-only"}[fam]
    twg = {"LR": "LR|S1+S2+S3", "LightGBM": "LightGBM|TW-only", "XGBoost": "XGBoost|TW-only", "LSTM-W5": "LSTM-W5|TW-only", "Transformer-W5": "Transformer-W5|TW-only"}[fam]
    ck = {"LR": "RQ1 Full vs S4-only (LR)", "LightGBM": "RQ1 Full vs S4-only (LightGBM)", "XGBoost": "RQ1 Full vs S4-only (XGBoost)", "LSTM-W5": "RQ1 Full vs S4-only (LSTM)", "Transformer-W5": "RQ1 Full vs S4-only (Transformer)"}[fam]
    main_rows.append([name, f3(auc(g)), f3(auc(s4g)), f3(auc(twg)) if twg in M["arms"] else "—", D(ck) if ck in M["comparisons"] else "—"])
main_tbl = table(["模型", "Full（S1–S5）", "S4-only（美股）", "TW-only（S1+S2+S3）", "Full − S4（95% CI）"], main_rows)

lab_rows = [["y_up 收盤→收盤（主任務）", pc(1 - 0.5244, 1) if False else "47.6%", f3(full), f3(s4), f3(tw), D("RQ1 Full vs S4-only (LightGBM)")]]
for k, nm in [("y_tr", "y_tr 除息還原"), ("y_gap", "y_gap 隔夜跳空 Open(t+1) > Close(t)"), ("y_intra", "y_intra 開盤→收盤 Close(t+1) > Open(t+1)"), ("y_rel", "y_rel 相對加權指數"), ("y_up_noflat", "y_up 去掉平盤日")]:
    x = v[k]; lab_rows.append([nm, pc(x["rate"], 1), f3(x["Full"]["auroc"]), f3(x["S4-only"]["auroc"]), f3(x["TW-only"]["auroc"]), f"{sg(x['Full_minus_S4'][0])} 到 {sg(x['Full_minus_S4'][1])}"])
lab_rows.append(["幅度 |r| > k（k = 訓練窗 |r| 中位數 ≈ 0.83–0.86%）", "50%", f3(r4["magnitude_Full"]["auroc"]), f3(r4["magnitude_S4-only"]["auroc"]), f3(r4["magnitude_TW-only"]["auroc"]), "—"])
lab_rows.append(["幅度，波動率正規化 r / vol20", "50%", f3(vn["magnitude_Full"]["auroc"]), f3(vn["magnitude_S4-only"]["auroc"]), f3(vn["magnitude_TW-only"]["auroc"]), "—"])
lab_tbl = table(["標籤", "正類比例", "Full", "S4-only", "TW-only", "Full − S4（CI）"], lab_rows)

four_tbl = table(["四分類（strong down / mild down / mild up / strong up）", "Macro-F1", "CI"], [
    ["Full", f3(r4["four_Full"]["macro_f1"]), cia(r4["four_Full"]["macro_f1_ci"])], ["S4-only", f3(r4["four_S4-only"]["macro_f1"]), cia(r4["four_S4-only"]["macro_f1_ci"])], ["TW-only", f3(r4["four_TW-only"]["macro_f1"]), cia(r4["four_TW-only"]["macro_f1_ci"])],
    ["多數類", f3(r4["majority_macro_f1"]), "—"], ["Full − S4", f"{sg(r4['Full_minus_S4_macro_f1'][0])} 到 {sg(r4['Full_minus_S4_macro_f1'][1])}", f"QWK 差 {sg(r4['Full_minus_S4_qwk'][0])} 到 {sg(r4['Full_minus_S4_qwk'][1])}（含 0）"]])

shap_tbl = table(["來源", "LightGBM φ（CI）", "LR φ（CI）", "單獨 AUROC（LightGBM）", "移除該來源後 AUROC 的變化（負值 = 變差）"], [
    [s, f"{sg(sh['phi'][s]['value'])} {ci(sh['phi'][s]['ci'])}", f"{sg(shl['phi'][s]['value'])} {ci(shl['phi'][s]['ci'])}", f3(sh["single"][s]), sg(-sh["leave_one_out"][s])] for s in ["S1", "S2", "S3", "S4", "S5"]] + [["合計 φ（= Full − 0.5）", sg(sh["sum_phi"]), sg(shl["sum_phi"]), "", ""]])
inter = sorted(sh["interaction"].items(), key=lambda kv: -abs(kv[1]))
inter_txt = "、".join(f"{k} {sg(x)}" for k, x in inter[:4])

sec_tbl = table(["產業", "S4-only AUROC", "Full AUROC", "Full − S4（CI）"], [[k, f3(x["auroc_S4"]), f3(x["auroc_Full"]), f"{sg(x['full_minus_S4'][0])} 到 {sg(x['full_minus_S4'][1])}"] for k, x in E["by_sector"].items()])
gs_last = E["grouped_shap"][yrs[-1]]
gs_txt = "、".join(f"{s} {pc(gs_last['share'][s], 0)}" for s in ["S1", "S2", "S3", "S4", "S5"])

audit_tbl = table(["稽核", "數字", "解讀"], [
    ["walk-forward，LightGBM Full", f"{f3(A['walk_forward_full']['auroc'])} {cia(A['walk_forward_full']['ci'])}", "誠實的數字"],
    ["同樣的列改隨機分層 5-fold", f"{f3(A['random_5fold_full']['auroc_test_years_only'])}（含訓練年 {f3(A['random_5fold_full']['auroc_all_years'])}）", f"協定本身膨脹 {A['leakage_ratio']:.2f} 倍"],
    ["固定 400 棵樹的 walk-forward", f3(A["walk_forward_full_fixed400"]), "早停不是差距的原因"],
    ["訓練標籤 20 日區塊置換（3 次）", " / ".join(f3(x) for x in A["block_permutation_null_auroc"]), f"明顯下降，接近分股基準率規則 {f3(auc('Rule|stock base-rate'))}（其中兩次略高）；只做 3 次，是健全性檢查，不能當成「沒有洩漏」的證明"],
    ["故意灌入 t+1 的美股盤", f3(A["future_shift_full"]), "流程抓得到一天的洩漏"],
    ["排除過期美股列", f"{f3(A['exclude_us_stale']['auroc'])}（n = {A['exclude_us_stale']['n']:,}）", "不變"]])

# ---------- SOTA ----------
sn_rows = [["我們的 LR（價格特徵）", f3(SN["LR"]["acc"]), f3(SN["LR"]["mcc"])], ["我們的 LightGBM（5 種子平均）", f3(SN["LightGBM"]["mean"]["acc"]), f3(SN["LightGBM"]["mean"]["mcc"])], ["我們的 XGBoost（5 種子平均）", f3(SN["XGBoost"]["mean"]["acc"]), f3(SN["XGBoost"]["mean"]["mcc"])],
           ["我們的 ALSTM 類模型，官方正規化輸入，StockNet 標籤", f"{f3(SD['stocknet_band']['mean']['acc'])} ± {f3(SD['stocknet_band']['std']['acc'])}", f3(SD["stocknet_band"]["mean"]["mcc"])], ["同上，單純正負號標籤", f3(SD["sign"]["mean"]["acc"]), f3(SD["sign"]["mean"]["mcc"])],
           ["多數類（測試集 3,744 筆）", f3(SN["majority_acc_test"]), "0"]]
for k, (a, m) in SN["published_pulsekan_table1_stocknet"].items(): sn_rows.append([f"[lit] {k}（PULSE-KAN 論文 Table 1）", f3(a), f3(m)])
sn_tbl = table(["StockNet 基準（88 檔美股，測試 2015-10 到 2015-12）", "ACC", "MCC"], sn_rows)

so_tbl = ""
if SO:
    rows = []
    for vkey, nm, pub in [("hedge", "HedgeFundAnalyst（文字 + 價格）", "HedgeFundAnalyst"), ("tech", "TechnicalAnalyst（只有價格）", "TechnicalAnalyst"), ("fund", "FundamentalAnalyst（只有推文）", "FundamentalAnalyst")]:
        if vkey not in SO: continue
        runs = [SO[vkey][r] for r in SO[vkey] if r != "mean"]; m = SO[vkey]["mean"]
        rows.append([nm, ", ".join(f3(r["test_acc"]) for r in runs), f3(m["test_acc"]), ", ".join("未定義" if r["test_mcc"] is None else f3(r["test_mcc"]) for r in runs), f3(SO["published"][pub]["acc"]), f3(SO["published"][pub]["mcc"])])
    rows.append(["多數類（過濾後測試集）", "", f3(SO["test_majority_acc"]), "", "", ""])
    so_tbl = table(["官方 StockNet 程式", "各 run ACC", "平均 ACC", "各 run MCC", "論文 ACC", "論文 MCC"], rows)

cs_tbl = ""
if CV:
    a, b, k = CV["acl18_dclose"], CV["acl18_mov"], CV["kdd17"]
    cs_tbl = table(["CausalStock 重現", "我們的 v2 實作 ACC", "MCC", "論文 / 重現者", "多數類"], [
        ["ACL18，重現者的標籤（第 6 欄 = 日變化量），無新聞", f"{f3(a['CausalStock v2 w/o news']['mean']['acc'])} ± {f3(a['CausalStock v2 w/o news']['std']['acc'])}", f3(a["CausalStock v2 w/o news"]["mean"]["mcc"]), f"論文 0.634 / 0.217（含新聞）；論文無新聞 0.581；重現者 0.637", f3(a["majority_acc_test"])],
        ["ACL18，同上，含快取 LLM 新聞分數", f3(a["CausalStock v2 with news (cached DNE scores)"]["mean"]["acc"]), f3(a["CausalStock v2 with news (cached DNE scores)"]["mean"]["mcc"]), "", ""],
        ["ACL18，同上，拿掉因果圖 TCD", f3(a["w/o TCD"]["mean"]["acc"]), f3(a["w/o TCD"]["mean"]["mcc"]), "論文 w/o TCD 0.511", ""],
        ["ACL18，同上，只用昨天的美元變化量做邏輯迴歸", f3(a["artifact_check_logreg_on_prev_dollar_change_acc"]), "", "證明該標籤有均值回歸 artifact", ""],
        ["ACL18，論文定義的正確標籤（movement % > 0），無新聞", f"{f3(b['CausalStock v2 w/o news']['mean']['acc'])} ± {f3(b['CausalStock v2 w/o news']['std']['acc'])}", f3(b["CausalStock v2 w/o news"]["mean"]["mcc"]), "", f3(b["majority_acc_test"])],
        ["ACL18，正確標籤，含新聞", f3(b["CausalStock v2 with news (cached DNE scores)"]["mean"]["acc"]), f3(b["CausalStock v2 with news (cached DNE scores)"]["mean"]["mcc"]), "", ""],
        ["KDD17，官方標籤，只有價格", f"{f3(k['CausalStock v2 (price-only)']['mean']['acc'])} ± {f3(k['CausalStock v2 (price-only)']['std']['acc'])}", f3(k["CausalStock v2 (price-only)"]["mean"]["mcc"]), "論文 0.561 / 0.124", f3(k["majority_acc_test"])],
        ["台灣 50，我們的 walk-forward，AUROC 逐年", " / ".join(f3(x["auroc"]) for x in CV["tw50"]["seed_mean_by_year"]["CausalStock v2 (price-only)"].values()), "", f"LightGBM 五來源 {f3(full)}", ""]])
HH = J("sota_headtohead.json") if (RES / "sota_headtohead.json").exists() else None; FC, FH, NS, AU = [J(n) for n in ["final_model_checks.json", "fresh_holdout.json", "night_session.json", "alignment_audit.json"]]; BESTS = max(HH["sota_models"], key=lambda k: HH["rows"][k]["auroc"]) if HH else None; qd = lambda x: f"{sg(x['diff'])} [{sg(x['ci'][0])}, {sg(x['ci'][1])}]"; RB = J("robustness.json"); DSM = J("direction_size_models.json"); EE = J("eda_and_errors.json"); EC9 = EE["C_class_wise_binary"]["final nine-source model"]; EC5 = EE["C_class_wise_binary"]["five-source LightGBM"]; ED = EE["D_failure_modes_final_model"]; FL = J("four_level_deep.json"); FLA = FL["A_macro_f1"]; FLB = FL["B_per_class"]["Full"]; FLC = FL["C_predicted_class_outcomes"]; FLE = FL["E_direction_and_size"]; pp2 = lambda x: f"{x*100:+.2f}%"; RC = RB["C_S4_vs_S8_over_seeds"]["y_up"]; RG = RB["G_official_alstm_reruns"]; RH = RB["H_backtest_gross_vs_random"]; pq = lambda x: f"{sg(x['diff'])} [{sg(x['ci'][0])}, {sg(x['ci'][1])}]"; NRD = (lambda z: f"{sg(z['minus_night_rule'])} [{sg(z['ci'][0])}, {sg(z['ci'][1])}]")(NS["final_model_vs_night_rule"]["final_model"]); hh_sec = "（未執行）"
if HH:
    R_, PR_, FM_ = HH["rows"], HH["paired"], HH["final_model"]; order = sorted(R_, key=lambda k: -R_[k]["auroc"])
    hh_tbl = table(["模型（同一份台灣 50 資料、同一個 walk-forward、同一個標籤）", "類別", "AUROC [95% CI]", "ACC", "MCC", "最終模型 − 它（ΔAUROC [CI]，同號年數）"], [
        [k, "SOTA 架構" if k in HH["sota_models"] else "我們", f"{f3(R_[k]['auroc'])} [{f3(R_[k]['ci'][0])}, {f3(R_[k]['ci'][1])}]", f3(R_[k]["acc"]), f3(R_[k]["mcc"]),
         (lambda d: f"{sg(d['diff'])} [{sg(d['ci'][0])}, {sg(d['ci'][1])}]，{d['same_sign_years']}/4" if d else "—")(PR_.get(f"{FM_} minus {k}"))] for k in order])
    q = lambda a, b_: (lambda d: f"{sg(d['diff'])} [{sg(d['ci'][0])}, {sg(d['ci'][1])}]")(PR_[f"{a} minus {b_}"])
    hh_sec = hh_tbl + f"""

- **做法**：把已發表的架構放到我們的資料上，同樣的列、同樣的年度切分、同樣的標籤（Close(t+1) > Close(t)）。每個架構跑兩次：只給價格特徵（論文原本的用法），以及給我們的 45 個特徵。CausalStock 另跑九來源與一個調參版（較寬的編碼器、lr 1e-3）。Adv-ALSTM / ALSTM / LSTM 用官方程式與 KDD17 超參數，CausalStock 用我們的重現。
- **結果一（我們贏）**：最終模型（LightGBM + LSTM 平均，九個來源）AUROC {f3(R_[FM_]['auroc'])}，高於所有 SOTA 架構；對其中最好的「ALSTM 官方程式 + 我們的 45 特徵」差 {q(FM_, 'ALSTM official (our 45 features)')}，區間不含 0。
- **結果二（要誠實講）**：在完全相同的 45 個特徵上，官方 ALSTM 與我們的 LightGBM 打平（LightGBM − ALSTM = {q('LightGBM Full (S1-S5)', 'ALSTM official (our 45 features)')}，區間含 0）。所以我們贏的原因是多加了四個來源並平均兩種模型，不是架構比較好。
- **結果三（資訊比架構重要）**：同一個架構，從只給價格到給我們的特徵，Adv-ALSTM {f3(R_['AdvALSTM official (11 price features)']['auroc'])} → {f3(R_['AdvALSTM official (our 45 features)']['auroc'])}、CausalStock {f3(R_['CausalStock v2 (price-only, 11)']['auroc'])} → {f3(R_['CausalStock v2 + S1-S5']['auroc'])}。CausalStock 在 45 特徵上不同種子差異大，給九來源反而變差（{f3(R_['CausalStock v2 + All9']['auroc'])}，調參版 {f3(R_['CausalStock v2 + All9, tuned']['auroc'])}）。
- **對齊修正（2026-11-13）**：先前 SOTA 在台灣 50 的輸入視窗只到 t−1，卻預測 t→t+1，比我們自己的模型少看當天的資料，對 SOTA 不公平。已改成視窗結束在第 t 天並全部重跑；4.4、4.5 的台灣 50 數字是修正後的。ACL18、KDD17 的重現用論文原本的標籤定義，不受影響。`14_causalstock_lite.py` 只改了程式沒有重跑，其台灣 50 輸出已不再被引用。
- **限制**：SOTA 用的是論文／官方的超參數，只有 CausalStock 多試了一組；各架構沒有做同等的調參搜尋。每對比較只用兩個模型都有分數的列。"""
xr_tbl = ""
if XR:
    h = XR["hwang_repo_own_code_acl18_no_news"]; ak = XR["adv_alstm_official_kdd17"]; at = XR["adv_alstm_official_tw50"]
    xr_tbl = table(["外部官方／重現程式（安全審核後執行）", "ACC", "MCC", "對照"], [
        ["重現倉庫原始碼，ACL18 無新聞，他們的標籤", f3(h["their label (column 6 = daily dollar change)"]["test_acc"]), f3(h["their label (column 6 = daily dollar change)"]["test_mcc"]), "只加一行標籤開關"],
        ["同上，改成正確標籤", f3(h["corrected label (movement > 0)"]["test_acc"]), f3(h["corrected label (movement > 0)"]["test_mcc"]), ""],
        ["Adv-ALSTM 官方程式，KDD17：LSTM", f3(ak["LSTM (official code)"]["test_acc"]), f3(ak["LSTM (official code)"]["test_mcc"]), f"CausalStock 表列 {ak['LSTM (official code)']['published_in_causalstock_table1'][0]/100:.3f}"],
        ["Adv-ALSTM 官方程式，KDD17：ALSTM", f3(ak["ALSTM (official code, hyperparameters file)"]["test_acc"]), f3(ak["ALSTM (official code, hyperparameters file)"]["test_mcc"]), f"表列 {ak['ALSTM (official code, hyperparameters file)']['published_in_causalstock_table1'][0]/100:.3f}"],
        ["Adv-ALSTM 官方程式，KDD17：Adv-ALSTM", f3(ak["Adv-ALSTM (official code, saved ALSTM init)"]["test_acc"]), f3(ak["Adv-ALSTM (official code, saved ALSTM init)"]["test_mcc"]), "原論文 0.531 / 0.052；我們的 KDD17 設定能重現官方基線"],
        ["Adv-ALSTM 官方程式，台灣 50 walk-forward：LSTM / ALSTM / Adv-ALSTM", " / ".join(f3(at[m]["mean"]["test_acc"]) for m in ["LSTM", "ALSTM", "AdvALSTM"]), " / ".join(f3(at[m]["mean"]["test_mcc"]) for m in ["LSTM", "ALSTM", "AdvALSTM"]), f"多數類 {f3(at['AdvALSTM']['mean']['majority_acc'])}；我們的 LightGBM Full 準確率 {f3(GP['mcnemar_Full_vs_S4']['acc_full']) if GP else '—'}"]])
yang_tbl = ""
if Y:
    rs, wf = Y["random_split"], Y["walk_forward_pooled_mean"]
    yang_tbl = table(["Yang et al. 2024 流程在台灣 50", "RF", "SVM", "LightGBM"], [
        ["他們的隨機 80/20 切分，原始指標", *[f3(rs["raw indicators (Table 4 analogue)"][m]["acc"]) for m in ["RF", "SVM", "LightGBM"]]],
        ["隨機切分，加 LSTM 預測指標", *[f3(rs["+ predicted indicators (Table 5 analogue)"][m]["acc"]) for m in ["RF", "SVM", "LightGBM"]]],
        ["隨機切分，oracle：用真實次日收盤算指標", *[f3(rs["oracle: indicators with the true next close"][m]["acc"]) for m in ["RF", "SVM", "LightGBM"]]],
        ["我們的 walk-forward，原始指標", *[f3(wf["raw"][m]["acc"]) for m in ["RF", "SVM", "LightGBM"]]],
        ["我們的 walk-forward，加預測指標", *[f3(wf["+ predicted"][m]["acc"]) for m in ["RF", "SVM", "LightGBM"]]],
        ["[lit] 論文：全日平均（2330 SVM）/ 篩選日最高", "0.67 / 0.86", "", f"LSTM 單獨方向準確率：論文 0.52，我們 {f3(Y['lstm_direction_acc_random_split'])}"]])

# ---------- backtest ----------
def bt(k):
    s = st[k]; return [k.replace("p_cc_", "收盤→收盤訊號 ").replace("p_oc_", "開盤→收盤訊號 ").replace("p_sox", "費半規則").replace("p_rand", "隨機").replace("|top", "，前 "), pc(s["long_only_gross"]["ann_return"], 0), f"{s['long_only_gross']['sharpe']:.2f}", pc(s["long_only_net"]["ann_return"], 0), f"{s['long_only_net']['sharpe']:.2f}", pc(s["break_even_round_trip_cost"], 3)]
bt_tbl = table(["策略（t+1 開盤買、收盤賣，等權）", "毛年化", "毛 Sharpe", "淨年化", "淨 Sharpe", "損益兩平成本"], [bt(k) for k in ["p_cc_Full|top5", "p_cc_S4-only|top5", "p_cc_TW-only|top5", "p_oc_Full|top5", "p_oc_TW-only|top5", "p_sox|top5", "p_rand|top5"]])
bm = B["benchmarks"]
bm_txt = f"0050 買進持有年化 {pc(bm['etf_0050_buy_hold_close_to_close']['ann_return'], 0)}；等權 50 檔收盤→收盤毛年化 {pc(bm['equal_weight_50_close_to_close_gross']['ann_return'], 0)}（Sharpe {bm['equal_weight_50_close_to_close_gross']['sharpe']:.2f}）；等權 50 檔開盤→收盤毛年化 {pc(bm['equal_weight_50_intraday_gross']['ann_return'], 0)}，扣成本 {pc(bm['equal_weight_50_intraday_net']['ann_return'], 0)}。"

# ---------- sources S6-S9 ----------
ss_tbl = ""
if SS:
    rows = []
    for s, nm in [("S6", "S6 新聞標題（鉅亨網；訓練窗內 TF-IDF 分數、則數、有無新聞）"), ("S7", "S7 跨股票 / 產業 / 龍頭報酬"), ("S8", "S8 台指期夜盤、法人期貨部位"), ("S9", "S9 當沖、借券、月營收")]:
        rows.append([f"{nm}（{SS['n_features'][s]} 個特徵）", f3(SS["y_up"][f"{s}-only"]["auroc"])] + [f"{sg(SS[l][f'Full+{s} minus Full'][0])} 到 {sg(SS[l][f'Full+{s} minus Full'][1])}" for l in ["y_up", "y_gap", "y_intra", "y_rel"]])
    rows.append(["九來源全用 All9", f3(SS["y_up"]["All9"]["auroc"]), f"{sg(SS['fusion_vs_us_only']['All9 minus Full'][0])} 到 {sg(SS['fusion_vs_us_only']['All9 minus Full'][1])}（對 Full）", "", "", ""])
    ss_tbl = table(["新來源", "單獨 AUROC（y_up）", "加到 Full：y_up", "y_gap", "y_intra", "y_rel"], rows)
    ph9 = SS["shapley9_y_up"]["phi"]; sh9_txt = "、".join(f"{k} {sg(ph9[k])}" for k in sorted(ph9, key=lambda k: -ph9[k]))
    fu = SS["fusion_vs_us_only"]

# ---------- causal graph ----------
cg_txt = ""
if TS and CN and GF:
    ll, co, la, pr = TS["leadlag"], TS["contemporaneous"], TS["lasso"], TS["prior"]
    cn_tbl = table(["CausalStock v2 在台灣 50（3 種子 × 4 年）", "AUROC（CI）", "ACC", "對學習圖只用價格的差"], [[k.replace(" | ", " · "), f"{f3(x['auroc'])} {cia(x['ci'])}", f3(x["acc"]), "—" if x["diff_vs_learned_price"] is None else f"{sg(x['diff_vs_learned_price'][0])} 到 {sg(x['diff_vs_learned_price'][1])}"] for k, x in CN["pooled_seed_mean"].items()])
    gf_tbl = table(["圖特徵 S10 進 LightGBM", "y_up 收盤→收盤", "y_gap 跳空", "y_intra 盤中", "y_rel 相對大盤"], [
        ["Full", *[f3(GF[l]["Full"]["auroc"]) for l in ["y_up", "y_gap", "y_intra", "y_rel"]]],
        ["Full + S10", *[f3(GF[l]["Full+S10"]["auroc"]) for l in ["y_up", "y_gap", "y_intra", "y_rel"]]],
        ["Full + S10 − Full（CI）", *[f"{sg(GF[l]['Full+S10 minus Full'][0])} 到 {sg(GF[l]['Full+S10 minus Full'][1])}" for l in ["y_up", "y_gap", "y_intra", "y_rel"]]],
        ["All9 + S10 − All9（CI）", *[f"{sg(GF[l]['All9+S10 minus All9'][0])} 到 {sg(GF[l]['All9+S10 minus All9'][1])}" for l in ["y_up", "y_gap", "y_intra", "y_rel"]]]])
    gst = CN["graph_stability"]
    cg_txt = f"""
### 5.1 結構：同一天一起動，但不領先彼此（`20_tw_structure.py`，`tw_structure.json`）
- **同日相關**（第一個訓練窗，{co['n_days']} 天，市場調整後）：同產業 {co['mean_corr_within_sector']:.2f}、跨產業 {co['mean_corr_cross_sector']:.2f}、全部 {co['mean_corr_all']:.2f}。資料驅動的分群和產業標籤只有弱對應（ARI {co['ari_clusters_vs_sectors']:.2f}）。
- **領先落後**：corr(r_i,t, r_j,t+1) 對全部 {ll['n_pairs_tested']:,} 組有序對，Holm 5% 校正後只有 **{ll['n_significant_holm05']} 組顯著**（{ll['n_positive']} 組為正），|corr| 中位數 {ll['abs_corr_quantiles']['0.5']:.3f}、第 99 百分位 {ll['abs_corr_quantiles']['0.99']:.3f}。個股自身一階自相關平均 {ll['self_autocorr_mean']:.3f}，{ll['self_autocorr_significant']} 檔顯著。
- **先驗圖**：同產業 + {pr['n_supply_chain_ties']} 條公開供應鏈關係，共 {pr['n_undirected_edges']} 條無向邊（密度 {pr['density']:.2f}）。顯著的領先落後邊落在先驗圖內的比例 {pc(TS['overlap']['leadlag_sig_first_window']['share_in_prior'], 0)}，是隨機的 {TS['overlap']['leadlag_sig_first_window']['lift']:.1f} 倍：關係是真的，只是幾乎不帶隔日預測力。
- **Lasso 圖**（每折用訓練窗內的時間序列 CV 選 alpha）：樣本外 R² 全為負（{' / '.join(f"{x:.3f}" for x in la['oos_r2_mean'].values())}），相鄰年份邊集合的 Jaccard 只有 {' / '.join(f"{x:.2f}" for x in la['jaccard_consecutive'].values())}。資料裡的領先落後訊號小到連稀疏回歸都抓不穩。

### 5.2 把圖放進 CausalStock（`21_causal_news_tw.py`，`causal_news_tw.json`）
{cn_tbl}

- **學習出來的圖是退化的**：每個種子 × 年份的邊數平均 {' / '.join(f"{x['mean_edges']:.0f}" for x in gst.values())}（全圖 2,450 條），同一年不同種子的 Jaccard {' / '.join(f"{x['jaccard_seeds']:.2f}" for x in gst.values())}，也就是不是全開就是全關；邊機率全部擠在同一個值附近，TCD 的稀疏先驗在這份資料上選不出圖。
- 加新聞的增益（圖固定不變，只比較有無新聞）：學習圖 {qd(CN['news_effect_same_graph']['learned graph'])}，先驗圖 {qd(CN['news_effect_same_graph']['prior graph'])}。學習圖的區間不含 0，是一個很小但測得到的增益；先驗圖沒有增益、方向相反。兩種圖不一致，幅度都在 0.005 以內，所以不能說新聞有穩定的加分。先前版本把「先驗圖 + 新聞」拿去和「學習圖、只用價格」比，同時改了圖和輸入，已更正。絕對水準 {f3(CN['pooled_seed_mean']['prior fixed | +news']['auroc'])} 對 LightGBM 五來源 {f3(full)}。

### 5.3 圖做成特徵進 LightGBM（`22_graph_features.py`，`graph_features.json`）
S10g = 先驗圖鄰居的 1 日 / 5 日報酬、Lasso 父節點預測、顯著領先者的報酬；S10n = 鄰居的新聞分數與則數。
{gf_tbl}

### 5.4 供應鏈方向與產業領先（`26_supply_chain.py`，`tw_supply_chain.json`）
- {SC['pooled']['n_ties']} 條有方向的供應鏈關係，市場調整後：上游 t → 下游 t+1 相關 {sg(SC['pooled']['daily: upstream t -> downstream t+1']['corr'])} {ci(SC['pooled']['daily: upstream t -> downstream t+1']['ci'])}，下游 t → 上游 t+1 {sg(SC['pooled']['daily: downstream t -> upstream t+1']['corr'])} {ci(SC['pooled']['daily: downstream t -> upstream t+1']['ci'])}，兩個方向一樣小，遠低於同日相關 {SC['pooled']['contemporaneous daily (mean over ties)']:.3f}；週頻上游→下游 {sg(SC['pooled']['weekly: upstream w -> downstream w+1']['corr'])}（不顯著）。
- Cohen–Frazzini 式客戶動量：客戶過去 20 日報酬對供應商未來 20 日市場調整報酬的 Spearman IC {sg(SC['customer_momentum']['ic_customer_to_supplier'])}（t = {SC['customer_momentum']['t']:.1f}，{SC['customer_momentum']['n_months']} 個月），反方向 {sg(SC['customer_momentum']['ic_supplier_to_customer'])}（t = {SC['customer_momentum']['t_rev']:.1f}）。方向和文獻一致，但很微弱。
- 產業領先大盤（Hong et al. 2007 式）：六個產業本週報酬對大盤下週報酬的 HAC t 值全在 ±1 內（{SC['sector_leadlag']['n_weeks']} 週），不成立。
- 概念股群內相關：AI 伺服器鏈 {SC['concepts']['AI-server / Nvidia chain']['within_corr']:.2f}、蘋果鏈 {SC['concepts']['Apple chain']['within_corr']:.2f}、金融 {SC['concepts']['Financials']['within_corr']:.2f}、台塑集團 {SC['concepts']['Formosa group']['within_corr']:.2f}；AI 鏈對隔夜費半的 beta {SC['concepts']['AI-server / Nvidia chain']['beta_to_overnight_sox']:.2f}，金融只有 {SC['concepts']['Financials']['beta_to_overnight_sox']:.2f}。

### 5.5 結論與看法
台灣 50 的關係圖**是真的**（同產業同日相關接近 0.5、顯著邊集中在先驗圖內），但它描述的是「今天誰一起動」，不是「明天誰會動」。這個共同因子已經被 S8 台指期夜盤（以及 S4 美股）帶進模型，所以不管是 CausalStock 的學習圖、我們手建的先驗圖還是 Lasso 圖，對收盤→收盤方向都沒有增益；唯一有用的地方是開盤→收盤（{sg(GF['y_intra']['Full+S10 minus Full'][0])} 到 {sg(GF['y_intra']['Full+S10 minus Full'][1])}），那是美股資訊被開盤價吸收後、台股內部結構還剩一點訊號的地方。我的看法：在日頻、單一市場、高度同步的 50 檔大型股上，「因果圖」這個模型假設和資料不匹配，與其繼續調圖學習器，不如承認資料本身沒有可學的領先落後結構。
"""

# ---------- news ----------
news_txt = ""
if NA:
    ke = NA["keyword_events"]; rv = NA["revenue_headlines"]; um = NA["univariate_news_vs_magnitude"]
    def kev(k): x = ke[k]; return [k, f"{x['n_events']:,}", pc(x["same_day_ret_mean"], 2), f"{pc(x['next_mean'], 2)} [{pc(x['next_mean_ci'][0], 2)}, {pc(x['next_mean_ci'][1], 2)}]", pc(x["up_rate"], 1)]
    kw_tbl = table(["關鍵字事件（2019–2026 全樣本）", "事件數", "當天報酬", "隔天報酬（CI）", "隔天上漲率"], [kev(k) for k in ["limit-up reported", "foreign net buy", "target price raised / upgrade", "revenue record high", "EPS / profit", "dividend", "capital reduction", "loss", "target cut / downgrade"] if k in ke])
    rv_tbl = table(["月營收標題，年增率分組", "n", "隔天報酬", "隔天上漲率"], [[k, rv[k]["n"], pc(rv[k]["next_mean"], 2), pc(rv[k]["up_rate"], 1)] for k in ["YoY < -20 %", "-20..0 %", "0..20 %", "20..50 %", "> 50 %"]])
    ai = NA["concept_themes"]["AI / Nvidia"]
    news_txt = f"""
- **方向 vs 幅度**（`univariate_news_vs_magnitude`）：個股新聞則數預測「明天 |r| 是否超過中位數」的 AUROC 逐年 {' / '.join(f3(um[y]['n_news']['auroc_magnitude']) for y in um)}，預測方向只有 {' / '.join(f3(um[y]['n_news']['auroc_direction']) for y in um)}；但 20 日波動率預測幅度 {' / '.join(f3(um[y]['vol20']['auroc_magnitude']) for y in um)}，更強且已在模型內，所以幅度任務 Full+S6 − Full = {sg(NA['magnitude_task']['Full+S6 minus Full'][0])} 到 {sg(NA['magnitude_task']['Full+S6 minus Full'][1])}；四分類 Macro-F1 {f3(NA['four_level_task']['Full']['macro_f1'])} → {f3(NA['four_level_task']['Full+S6']['macro_f1'])}（{sg(NA['four_level_task']['Full+S6 minus Full'][0])} 到 {sg(NA['four_level_task']['Full+S6 minus Full'][1])}，剛好顯著）。
- **關鍵字事件研究**：報導「已發生的價格事件」的標題當天就反映完；帶新資訊的事件（調升目標價、營收創高、EPS、股利、減資）有隔日漂移，但這些事件少：調升目標價、營收創高、EPS、股利四類的股票日數相加約占全部的 {100 * sum(NA['keyword_events'][k]['n_stock_days'] for k in ['target price raised / upgrade', 'revenue record high', 'EPS / profit', 'dividend']) / NA['baseline_all_stock_days']['n']:.1f}%（各類直接相加，同一天多類會重複計算）。

{kw_tbl}

- **概念股主題**：AI 新聞最密集的 10% 日子，AI 概念成員隔天 |r| {pc(ai['abs_next_return']['members_high'], 2)} 對平常 {pc(ai['abs_next_return']['members_normal'], 2)}、上漲率 {pc(ai['up_rate']['members_high'], 0)} 對 {pc(ai['up_rate']['members_normal'], 0)}：主題強度**不會**放大概念股。
- **月營收標題**（{rv['n_events']} 則可解析年增率，{pc(rv['share_positive_yoy'], 0)} 為正）：收盤後發布，隔天反應與年增率成正比（Spearman {sg(rv['spearman_yoy_vs_next_return'])}；發布前一天 {sg(rv['spearman_yoy_vs_same_day_return'])}，所以不是提前洩漏）。

{rv_tbl}
"""

llm_txt = ""
if LK:
    a, b_, p = LK["y_up"]["up to 2025-03-31 (on or before the stated knowledge cut-off)"], LK["y_up"]["from 2025-04-30 (after the weights were released)"], LK["y_up"]["pooled"]
    ub = LK["univariate_by_year"]
    llm_txt = f"""
- 本機 vLLM 的 Qwen3-4B（開源權重，2025-04 釋出）給 {LK['n_scored_headlines']:,} 則獨特標題打 −1 到 1 的「對該公司股價的隱含方向」分數，覆蓋 {pc(LK['coverage_stock_days_with_llm'], 0)} 的股票日；對齊方式與 TF-IDF 分數完全相同。
- 單變量 AUROC（有新聞的股票日，逐年）：LLM {' / '.join(f3(ub[y]['auroc_llm']) for y in ub)}，TF-IDF {' / '.join(f3(ub[y]['auroc_tfidf']) for y in ub)}。
- 加到五來源（y_up）：2025-03-31（我們假設的知識截止日，比釋出早一個月；查不到模型方的正式說明）以前 Full+LLM − Full = {sg(a['Full+LLM minus Full'][0])} 到 {sg(a['Full+LLM minus Full'][1])}（{f3(a['Full'])} → {f3(a['Full+LLM'])}）；2025-04-30（權重釋出後）以後 {sg(b_['Full+LLM minus Full'][0])} 到 {sg(b_['Full+LLM minus Full'][1])}（{f3(b_['Full'])} → {f3(b_['Full+LLM'])}）。前段沒有比後段大，**看不到「背答案」的痕跡**（4B 模型、只給標題）。跳空任務 pooled {sg(LK['y_gap']['pooled']['Full+LLM minus Full'][0])} 到 {sg(LK['y_gap']['pooled']['Full+LLM minus Full'][1])}，顯著但很小。
- Full+LLM 比 Full+S6（TF-IDF）好 {sg(p['Full+LLM minus Full+S6'][0])} 到 {sg(p['Full+LLM minus Full+S6'][1])}，主要是因為 S6 本身把 Full 拉低（{sg(p['Full+S6 minus Full'][0])} 到 {sg(p['Full+S6 minus Full'][1])}）。
- 看法：這是把助教「新模型吃到測試期新聞」的疑慮變成可量化的實驗。結果是在這個規模下量不到前視偏誤，但原則不變：任何在測試期之後釋出的 LLM 都不可能事後證明無洩漏，所以主要新聞分數仍用訓練窗內自己訓練的模型，LLM 只當敏感度分析。
"""

# ---------- gap fills ----------
gap_txt = ""
if GP and MF:
    ar, df, mc = GP["arms"], GP["diffs"], GP["mcnemar_Full_vs_S4"]; t3 = MF["tertiary_3class"]; tu = MF["tuning"]; en = MF["ensemble"]; rg = MF["regimes"]; mg = MF["magnitude_gated_backtest"]; hol = MF["us_holiday_subsample"]["stale (U.S. closed the night before)"]; dm = MF["diebold_mariano_logloss"]
    gap_tbl = table(["Progress 承諾過的項目", "結果", "對 Full 0.590 的差（CI）", "結論"], [
        ["A3 精簡 S4（SOX、VIX、VIX 變化、TSM ADR）單獨", f3(ar["S4-reduced-only"]["auroc"]), f"{sg(df['S4-reduced-only minus Full'][0])} 到 {sg(df['S4-reduced-only minus Full'][1])}", f"比完整 S4 {f3(ar['S4-only']['auroc'])} 差；11 個 S4 特徵不冗餘"],
        ["A3 Full 但 S4 換成精簡版", f3(ar["Full with reduced S4"]["auroc"]), f"{sg(df['Full with reduced S4 minus Full'][0])} 到 {sg(df['Full with reduced S4 minus Full'][1])}", "含 0"],
        ["A2 Full + 產業指標", f3(ar["Full + sector indicators"]["auroc"]), f"{sg(df['Full + sector indicators minus Full'][0])} 到 {sg(df['Full + sector indicators minus Full'][1])}", "含 0"],
        ["A2 分產業各一個 LightGBM", f3(ar["Per-sector LightGBM"]["auroc"]), f"{sg(df['Per-sector LightGBM minus Full'][0])} 到 {sg(df['Per-sector LightGBM minus Full'][1])}", "含 0：異質性不需要分開建模"],
        ["A2 分股各一個 LR", f3(ar["Per-stock LR"]["auroc"]), f"{sg(df['Per-stock LR minus Full'][0])} 到 {sg(df['Per-stock LR minus Full'][1])}", "含 0"],
        ["A1 McNemar Full vs S4-only（內層驗證閾值）", f"Full 對 S4 錯 {mc['full_right_s4_wrong']:,}；S4 對 Full 錯 {mc['s4_right_full_wrong']:,}", f"p = {mc['p_exact']:.3f}", f"**S4-only 準確率略高**（{f3(mc['acc_s4'])} 對 {f3(mc['acc_full'])}）；列間相依，p 偏樂觀"],
        ["A5 Diebold–Mariano 逐日 log-loss，Full vs S4", f"DM {dm['Full vs S4-only']['dm_stat']:.2f}", f"p = {dm['Full vs S4-only']['p_two_sided']:.3f}", "Full 的機率品質顯著較好；和 McNemar 不矛盾，一個看硬分類、一個看機率"],
        ["A5 DM Full vs TW", f"DM {dm['Full vs TW-only']['dm_stat']:.1f}", f"p < 0.001", ""],
        ["A4 三分類 ±0.5%（跌 / 平 / 漲）Macro-F1", f"Full {f3(t3['Full']['macro_f1'])}、S4 {f3(t3['S4-only']['macro_f1'])}、TW {f3(t3['TW-only']['macro_f1'])}", f"Full − S4 {sg(t3['Full minus S4 macro_f1'][0])} 到 {sg(t3['Full minus S4 macro_f1'][1])}", f"多數類 {f3(t3['majority_macro_f1'])}；**一有「平盤」類，融合就贏**，台股資料認得出不會動的日子"],
        ["A7 LightGBM 27 組超參數網格，內層驗證挑選", f3(tu["tuned_auroc"]), f"{sg(tu['tuned_minus_default_seed42'][0])} 到 {sg(tu['tuned_minus_default_seed42'][1])}", f"比固定預設 {f3(tu['default_auroc'])} 差：一年的驗證期太短，調參在過擬合驗證集"],
        ["A6 集成：LightGBM 與 LSTM 機率平均", f3(en["mean_prob"]), f"{sg(en['mean_prob_minus_lightgbm'][0])} 到 {sg(en['mean_prob_minus_lightgbm'][1])}", f"兩者相關 {en['corr_lightgbm_lstm']:.2f}；小而顯著"],
        ["D7 Regime：VIX 高 / 中 / 低", f"Full {f3(rg['VIX tercile']['high VIX']['Full'])} / {f3(rg['VIX tercile']['mid VIX']['Full'])} / {f3(rg['VIX tercile']['low VIX']['Full'])}", f"TW-only {f3(rg['VIX tercile']['high VIX']['TW-only'])} / {f3(rg['VIX tercile']['mid VIX']['TW-only'])} / {f3(rg['VIX tercile']['low VIX']['TW-only'])}", "美股訊號在高波動期最強；台股資料在每個 regime 都接近 0.52"],
        ["D7 Regime：加權指數在 MA20 之上 / 之下", f"Full {f3(rg['TAIEX trend']['TAIEX above MA20']['Full'])} / {f3(rg['TAIEX trend']['TAIEX below MA20']['Full'])}", "", "空頭段更好預測"],
        ["A8 幅度閘門回測（只在預測大動的日子交易前 10 名）", f"選中股票的絕對盤中變動 {pc(mg['mean_abs_intraday_move']['gated_top10'], 2)} 對全部 {pc(mg['mean_abs_intraday_move']['all_50'], 2)}", f"多空毛 Sharpe {mg['gated_ls_gross']['sharpe']:.2f}；純多毛 {mg['gated_long_gross']['sharpe']:.2f}", f"找得到會大動的股票，但方向判斷弱：純多扣成本後年化 {pc(mg['gated_long_net']['ann_return'], 0)}（多空只算了毛報酬，沒有扣成本的版本）"],
        ["D6 美股休市隔天（{n} 天）".format(n=MF["us_holiday_subsample"]["stale (U.S. closed the night before)"]["n_days"]), f"Full {f3(hol['Full']['auroc'])} {cia(hol['Full']['ci'])}", f"TW+S6 {f3(hol['TW+S6']['auroc'])}", "樣本太少，無法下結論"]])
    gap_txt = gap_tbl

# ---------- insights ----------
ins_rows = [
    ["外部 / 跨市場資訊有用 [15][18][21]", f"S4 的 Shapley {sg(sh['phi']['S4']['value'])}，占總共 {sg(sh['sum_phi'])}；TW-only {f3(tw)}", "成立，而且比文獻更極端：對收盤→收盤方向，隔夜美股幾乎就是全部"],
    ["法人籌碼有資訊 [4][5]", f"S2 單獨 {f3(sh['single']['S2'])}；φ(S2) {sg(sh['phi']['S2']['value'])}；S2 × S4 交互 {sg(sh['interaction']['S2xS4'])}", "對個股次日方向，在知道美股之後沒有；文獻的證據是指數層級、20 日視野"],
    ["融合多來源有幫助 [4][20][21]", f"y_up {D('RQ1 Full vs S4-only (LightGBM)')}；y_gap {sg(v['y_gap']['Full']['auroc'] - v['y_gap']['S4-only']['auroc'])}；y_rel {sg(v['y_rel']['Full']['auroc'] - v['y_rel']['S4-only']['auroc'])}；幅度 TW {f3(r4['magnitude_TW-only']['auroc'])} 對 S4 {f3(r4['magnitude_S4-only']['auroc'])}", "看問題：「哪一天」靠美股，「哪一檔、動多大」靠台股"],
    ["簡單模型不輸深度模型 [18][19]", f"LightGBM {f3(full)}、LSTM {f3(auc('LSTM-W5|Full'))}、Transformer {f3(auc('Transformer-W5|Full'))}、LR {f3(auc('LR|S1+S2+S3+S4+S5'))}", "成立；LightGBM + LSTM 平均多 +0.008（Appendix D）"],
    ["短回看就夠（T = 5）[19][20][21]", f"LSTM W = 1 / 5 / 10 / 20：{' / '.join(f3(M['lstm_window'][k]['auroc']) for k in ['LSTM-W1|Full|0', 'LSTM-W5|Full|0', 'LSTM-W10|Full|0', 'LSTM-W20|Full|0'])}", "成立"],
    ["報告的準確率取決於協定 [3][8]", f"隨機切分 ×{A['leakage_ratio']:.2f}；StockNet 文獻 0.56–0.58 我們 0.50–0.52" + (f"，官方程式 {f3(SO['hedge']['mean']['test_acc'])} 對多數類 {f3(SO['test_majority_acc'])}" if SO else "") + "；[3] 流程在我們資料 0.52–0.54", "成立；台灣文獻的準確率無法轉移到全日 walk-forward 設定"],
    ["增益因股票而異 [18]", f"50 檔中 {E['by_stock_summary']['n_sig_up']} 檔顯著變好、{E['by_stock_summary']['n_sig_down']} 檔顯著變差（95%）", "接近假陽性數量；50 檔共享同一天，個股層級的宣稱需要考慮相依性"],
    ["方向訊號可交易", f"最佳策略損益兩平成本 {pc(st['p_oc_TW-only|top5']['break_even_round_trip_cost'], 2)} 對實付 {pc(B['cost_round_trip'], 3)}", "在散戶成本下不行；隔夜報酬無法用盤中規則捕捉"],
    ["新聞情緒預測次日方向 [8][18][21]", (f"S6 加到 Full {sg(SS['y_up']['Full+S6 minus Full'][0])} 到 {sg(SS['y_up']['Full+S6 minus Full'][1])}" if SS else "") + (f"；LLM 分數 {sg(LK['y_up']['pooled']['Full+LLM minus Full'][0])} 到 {sg(LK['y_up']['pooled']['Full+LLM minus Full'][1])}" if LK else ""), "不是對所有股票日的方向；新聞帶的是幅度和有日期的事件；新聞 SOTA 的增益在正確標籤下消失"]]
ins_tbl = table(["文獻說法", "我們的數字", "判定"], ins_rows)

# ---------- Progress Update comparison ----------
pu_tbl = table(["數字", "Progress Update [PU]", "期末重算", "差異來源"], [
    ["LightGBM Full / S4-only / TW-only AUROC", "0.597 / 0.598 / 0.528（p.69–70）", f"{f3(full)} / {f3(s4)} / {f3(tw)}", "PU 數字無程式可重現；新流程 5 種子、內層驗證早停"],
    ["Full − S4-only", "p.9、p.12：**+0.014 [+0.004, +0.024]**；p.70：**−0.001 [−0.008, +0.006]**（自相矛盾）", D("RQ1 Full vs S4-only (LightGBM)"), (lambda g: f"已用期末程式驗證：固定 400 棵樹時 Full {f3(g['fixed400']['full'])}、只用美股 {f3(g['fixed400']['s4'])}，差 {sg(g['fixed400']['full'] - g['fixed400']['s4'])} [{sg(g['fixed400']['diff_ci'][0])}, {sg(g['fixed400']['diff_ci'][1])}]，重現了期中的 +0.014；同一個種子改用早停後，只用美股升到 {f3(g['early_stopping']['s4'])}，差距縮成 {sg(g['early_stopping']['full'] - g['early_stopping']['s4'])} [{sg(g['early_stopping']['diff_ci'][0])}, {sg(g['early_stopping']['diff_ci'][1])}]。也就是說 400 棵樹對只有 11 個特徵的美股組太多、把它壓低了，不是融合真的比較好。期末結論：含 0")(A["fusion_gain_fixed400_vs_early_stopping"])],
    ["Full − TW-only", "+0.070 [+0.052, +0.087]", D("RQ1 Full vs TW-only (LightGBM)"), "方向一致"],
    ["隨機 5-fold vs walk-forward", "0.714 vs 0.597", f"{f3(A['random_5fold_full']['auroc_test_years_only'])} vs {f3(A['walk_forward_full']['auroc'])}（×{A['leakage_ratio']:.2f}）", "重算後一致"],
    ["標籤置換 null", "0.506", " / ".join(f3(x) for x in A["block_permutation_null_auroc"]), "期末改用 20 日區塊置換；結果下降到接近分股基準率規則（0.518），而不是 0.5，因為各股上漲率不同這件事置換後仍在"],
    ["四分類 Macro-F1 Full vs S4", "0.354 vs 0.336（p.72）", f"{f3(r4['four_Full']['macro_f1'])} vs {f3(r4['four_S4-only']['macro_f1'])}", "差距比 PU 大得多；PU 的 +0.019 無程式"],
    ["幅度 AUROC", "0.709（p.41）", f"TW-only {f3(r4['magnitude_TW-only']['auroc'])}、Full {f3(r4['magnitude_Full']['auroc'])}", "一致；期末新增波動率正規化版本"],
    ["精簡 S4 比完整 S4", "差 0.020 [0.007, 0.032]", f"{sg(GP['diffs']['S4-reduced-only minus S4-only'][0])} 到 {sg(GP['diffs']['S4-reduced-only minus S4-only'][1])}" if GP else "—", "方向一致，期末 CI 含 0 邊緣"],
    ["LSTM 視窗 1 / 5 / 10 / 20", "0.589–0.592", f"{min(x['auroc'] for x in M['lstm_window'].values()):.3f}–{max(x['auroc'] for x in M['lstm_window'].values()):.3f}", "一致"],
    ["rank IC", "0.085（t = 12.1）", f"{f3(wd['IC Full']['mean'])}（t = {wd['IC Full']['t']:.1f}）", "PU 數字無程式；期末小很多"],
    ["逐股顯著變好 / 變差", "3 / 3", f"{E['by_stock_summary']['n_sig_up']} / {E['by_stock_summary']['n_sig_down']}", ""],
    ["多數類規則 AUROC", "0.491 [0.477, 0.505]（p.69）", "0.500（常數預測恆為 0.5）", "PU 錯誤 E3"],
    ["費半規則 AUROC", "0.576「sign rule」", f"費半報酬當分數 {f3(auc('Rule|SOX return score'))}；正負號規則 AUROC = 平衡準確率 0.553 [EDA]", "PU 把連續分數的 AUROC 寫成正負號規則（E4）"],
    ["美台同向比例", "53.8–62.3%", "53.2–60.7%（費半方向 vs 個股次日方向，2019–2026）[EDA]", "PU 錯誤 E5，沒有任何定義算得出 62.3"],
    ["分股上漲率範圍", "40.5–51.6%", "39.0–51.6%（最低中華電）[EDA]", "PU 錯誤 E6"],
    ["Jeng et al. 2024 文獻頁引用", "貼成 CMIN（Luo et al. ACL 2023）的引用文字（p.24）", "已改正", "PU 錯誤 E2"],
    ["S1 特徵數", "14", "15（含 spread）", "口徑 I5"], ["特徵數 57 vs 45", "混用", "57 欄位、45 個模型特徵", "口徑 I6"],
    ["種子數", "寫 5 種子，實際單一種子 42", f"真的跑 {M['arms']['LightGBM|Full']['n_seeds']}–5 個種子，std {seed_std:.4f}", "口徑 I8"],
    ["環境", "列了未安裝的 XGBoost、PyTorch、Optuna", "只列真正用到的套件與版本", "口徑 I9"]])

new_tbl = table(["期末新增（Progress Update 沒有）", "放哪裡", "一句話結果"], [
    ["五種「明天」標籤（y_tr、y_gap、y_intra、y_rel）+ 去平盤版本", "R4", "融合在跳空、相對大盤顯著贏；盤中只有台股資料有訊號"],
    ["六種模型家族 × 31 個來源子集的同協定 benchmark、exact Shapley、交互作用", "R1–R3", "S4 占八成以上；來源接近可加"],
    ["同一天內橫斷面 AUROC、rank IC、校準", "R5", f"Full 同日 AUROC {f3(wd['Full']['mean'])}、IC {f3(wd['IC Full']['mean'])}"],
    ["洩漏稽核：區塊置換、未來灌入、固定樹數、排除過期列", "R9", "流程不洩漏也不遲鈍"],
    ["文獻在自己條件上重測：StockNet（我們的模型 + 官方程式）、Yang 2024、CausalStock（ACL18 / KDD17 / 台灣 50）、Adv-ALSTM 官方程式（KDD17 / 台灣 50）", "R10、R10b", "三篇在正確條件下都沒有重現；CausalStock 只在標籤錯誤時重現"],
    ["含成本回測 + 幅度閘門回測", "R11、Appendix D", "牌告成本下全部虧損；報酬都在隔夜"],
    ["四個新來源 S6–S9 與九來源 Shapley", "R12", "台指期夜盤是最大的來源（修正對齊後）；跨股票有小增益；新聞標題為負"],
    ["台股因果圖（結構、CausalStock 接三種圖、圖特徵）、供應鏈方向、產業領先", "R13", "關係圖是真的但不領先；只對盤中有用"],
    ["新聞事件研究、概念股主題、營收年增率", "R14", "新聞帶幅度與有日期的事件，不帶方向"],
    ["LLM 新聞分數與前視偏誤實驗（本機 Qwen3-4B）", "R15", "看不到背答案的痕跡；增益很小"],
    ["Progress 承諾過的 11 個項目補齊（McNemar、DM、分產業 / 分股、精簡 S4、三分類、調參、集成、regime、幅度回測、美股休市）", "Appendix D", "見第 7 節"],
    ["資料官方端點逐列核對、除息還原標籤、dataset README", "Data Understanding / Preparation", "資料是 TWSE 公布的"]])

promise_tbl = table(["Progress 承諾", "狀態"], [
    ["McNemar（p.11 挑戰 1）", "完成（Appendix D）"], ["Pooled vs per-stock vs per-sector（p.11 挑戰 5）", "完成"], ["精簡 S4（p.11 挑戰 4）", "完成"], ["三分類 ±0.5%（p.10，optional）", "完成"],
    ["Diebold–Mariano / Model Confidence Set（p.70）", "DM 完成；MCS 未做（bootstrap 已夠）"], ["Stacking / ensemble（p.70）", "完成（LightGBM + LSTM 平均）"], ["超參數調整", "完成（27 組網格，結果是調參反而變差）"],
    ["四分類 / 幅度的回測（p.12）", "完成（幅度閘門回測）"], ["Streamlit 每日排名儀表板（提案 6.2，optional）", "未做；簡報改寫為「流程設計與示範輸出」"], ["GitHub 釋出", "未做（需要帳號）"], ["Dataset zip + README（12/22）", "README 草稿完成，zip 未做"],
    ["新聞來源", "完成（鉅亨網 17 萬則標題；PU 時判定不可行的 FinMind 路線放棄）"]])

# ---------- assemble ----------
out = f"""# 期末研究總整理：所有實驗、結論、看法，與 Progress Update 的對照

> 產生方式：`scripts/final/13_full_report.py` 從 `results/final/*.json` 讀取；日期 {today}。標 [PU] 的是 Progress Update 簡報上的數字（無程式可重現，僅供對照），標 [EDA] 的是資料理解階段 `figures/*.json` 的數字，標 [lit] 的是論文表格（已於 `01_查核報告` 逐筆核對）。其餘數字全部來自 `results/final/`，與簡報 `group_A_part_1_team_04_final.pptx`、講稿 `02_講稿_期末簡報.md` 同源。

## 0. 一頁結論

1. **主任務（台灣 50 個股次日收盤→收盤方向）**：LightGBM 五來源 Full AUROC {f3(full)}，只用隔夜美股 S4 {f3(s4)}，只用台灣資料 {f3(tw)}。Full − S4 = {D('RQ1 Full vs S4-only (LightGBM)')}，CI 含 0；五種模型家族全在 {min(auc(g) for _, g in fams[2:]):.3f}–{max(auc(g) for _, g in fams[2:]):.3f}。**沒有足夠證據顯示五來源融合贏過只用美股（不等於證明兩者相同）；跨日期的方向訊號主要來自美股。**
2. **拆開「明天」之後**：跳空 Full {f3(v['y_gap']['Full']['auroc'])} 對 S4 {f3(v['y_gap']['S4-only']['auroc'])}（顯著）；開盤→收盤 S4 {f3(v['y_intra']['S4-only']['auroc'])}、TW {f3(v['y_intra']['TW-only']['auroc'])}；相對大盤 Full {f3(v['y_rel']['Full']['auroc'])} 對 S4 {f3(v['y_rel']['S4-only']['auroc'])}（四年同號）；幅度 TW {f3(r4['magnitude_TW-only']['auroc'])} 對 S4 {f3(r4['magnitude_S4-only']['auroc'])}；三分類（含平盤）Macro-F1 Full {f3(MF['tertiary_3class']['Full']['macro_f1']) if MF else '—'} 對 S4 {f3(MF['tertiary_3class']['S4-only']['macro_f1']) if MF else '—'}。**台灣資料決定「哪一檔、動多大、會不會動」。**
3. **來源貢獻**：exact Shapley S4 {sg(sh['phi']['S4']['value'])}，占總共 {sg(sh['sum_phi'])}；S1 {sg(sh['phi']['S1']['value'])}、S2 {sg(sh['phi']['S2']['value'])} 的 CI 大於 0 但很小；交互作用小一個數量級。
4. **協定決定數字**：同樣的列改隨機 5-fold 從 {f3(A['walk_forward_full']['auroc'])} 變 {f3(A['random_5fold_full']['auroc_test_years_only'])}（×{A['leakage_ratio']:.2f}）。這說明切分方式本身就能把分數推高，但不能據此認定每篇文獻的高分都出於同一原因。
5. **SOTA 重現**：CausalStock 的 63.4% 只有在把價格檔第 6 欄（日變化量）當收盤價時才重現得出（我們 {f3(CV['acl18_dclose']['CausalStock v2 w/o news']['mean']['acc']) if CV else '—'}，不用新聞也一樣）；正確標籤下 {f3(CV['acl18_mov']['CausalStock v2 w/o news']['mean']['acc']) if CV else '—'}（多數類 {f3(CV['acl18_mov']['majority_acc_test']) if CV else '—'}）。StockNet 官方程式 HedgeFundAnalyst {f3(SO['hedge']['mean']['test_acc']) if SO else '—'} 對過濾後多數類 {f3(SO['test_majority_acc']) if SO else '—'}（論文 0.582）。Adv-ALSTM 官方程式在 KDD17 重現官方基線、在台灣 50 等於多數類。
6. **擴充來源（期中後新增，夜盤對齊於 11-13 修正）**：九來源 {f3(SS['y_up']['All9']['auroc'])} 對五來源 {f3(full)}，增益 {sg(SS['y_up']['All9 minus Full'][0])} 到 {sg(SS['y_up']['All9 minus Full'][1])}。**最大的來源是台指期夜盤（S8）**：九來源 Shapley S8 {sg(SS['shapley9_y_up']['phi']['S8'])}、美股 S4 {sg(SS['shapley9_y_up']['phi']['S4'])}；光是「夜盤漲跌」一個數字當分數就有 {f3(NS['rule_auroc_test_years']['night_return_as_score']['auroc'])}（費半規則 {f3(NS['rule_auroc_test_years']['sox_return_as_score']['auroc'])}）；最終模型只比這個規則高 {NRD}，區間包含 0，模型多出的價值在同一天內選股。已知夜盤後，美股仍有額外貢獻：4 個 seed 平均 {pq(RC['seed_mean']['S4_on_top_of_S8'])}，四年同號（11-13 版只用 seed 42 得到 {sg(FC['C_S4_vs_S8']['y_up']['S4_on_top_of_S8']['diff'])}，寫成「不再有額外貢獻」，11-14 用多個 seed 重做後更正）。新聞標題為負；台股因果圖不領先；LLM 新聞分數沒有加分，也看不到前視偏誤。
7. **回測**：牌告成本 {pc(B['cost_round_trip'], 3)} 每趟之下全部虧損；最佳訊號損益兩平只有 {pc(st['p_oc_TW-only|top5']['break_even_round_trip_cost'], 2)}。市場報酬都在隔夜（等權 50 檔收盤→收盤毛年化 {pc(bm['equal_weight_50_close_to_close_gross']['ann_return'], 0)}，開盤→收盤 {pc(bm['equal_weight_50_intraday_gross']['ann_return'], 0)}）。
8. **與 Progress Update 相比**：五來源的結論方向一致（美股是五個來源中唯一強的），但所有數字重算；PU 宣稱的融合增益 +0.014 不成立（PU 自己 p.70 已寫 −0.001）；6 項明確錯誤、9 項口徑不一致全部修正。

9. **最終模型與 SOTA**：最終模型（九來源）{f3(HH['rows'][HH['final_model']]['auroc'])}；已發表架構給同樣九來源後最好的是 {f3(HH['rows'][BESTS]['auroc'])}，差 {qd(HH['paired'][HH['final_model'] + ' minus ' + BESTS])}。九來源下 LightGBM 單獨就有 {f3(HH['rows']['LightGBM All9 (S1-S9)']['auroc'])}，模型平均沒有再加分。
10. **乾淨新資料**：2026-09-24 之後 {FH['n_days']} 個交易日（模型和我們都沒看過）：五來源 {f3(FH['labels']['y_up']['arms']['LightGBM|Full']['auroc'])}，歷史測試 {f3(FH['labels']['y_up']['walk_forward_2023_2026']['Full'])}，一致。
11. **時間對齊**：11-13 查出三個對齊錯誤（兩個是用到較舊的資料；新聞那一個除了週末新聞排晚一列之外，另有 3.4% 的標題晚於 08:30 決策時間、最多 30 分鐘，屬於輕微偷看，但仍在開盤前），全部修正重跑；之後的系統審計 {AU['T1_leak_test']['n_features']} 個特徵無一偷看未來。

**一句話：隔夜資訊告訴我們「哪一天」，其中台灣自己的期貨夜盤帶的最多、美股再補上一部分；台灣資料告訴我們「哪一檔」和「漲跌多少」；協定決定你能宣稱什麼。**（其中「哪一檔」的證據最弱：同一天內的排序 AUROC {f3(E['within_day']['Full']['mean'])}，只比「照各股過去上漲率排」的 {f3(E['within_day']['stock base-rate']['mean'])} 高一點，差異 t = {E['within_day']['Full_minus_baserate_days']['t']:.1f}。）

---

## 0b. 怎麼讀這份文件（先看這一節）

**這份研究在問什麼**：預測台灣 50 成分股「明天」的表現時，把股價、法人、大盤、美股、日曆五種資料一起交給模型，會不會比只看其中一種更準？各種資料各貢獻多少？換模型有沒有差？

**幾個一定要先分清楚的觀念**

- **AUROC 不是準確率。** AUROC 看的是模型能不能把「會漲」的樣本排在「不會漲」的前面；0.5 等於亂猜，1.0 等於完全分開。AUROC 0.590 不代表猜對 59%。
- **信賴區間包含 0，代表不能說有差。** 例如五來源比只用美股高 +0.004，區間是 −0.002 到 +0.011；資料還容許「其實沒有改善」，所以不能因為 +0.004 是正的就說融合有效。本文說「顯著」時，指的是區間整段在 0 的同一側，而且四個測試年至少三年同方向。區間端點顯示成 +0.000 或 −0.000 的，一律當作「含 0」看待。
- **4.5 萬列不是 4.5 萬次獨立測試。** 同一天的 50 檔股票會一起受大盤影響，所以信賴區間是以「交易日」為單位、連續 5 天為一塊重抽 1,000 次算出來的。真正的樣本單位是 900 個交易日。
- **「模型沒看過」和「人沒看過」是兩件事。** 模型訓練、選超參數、選門檻都只用更早的資料，所以沒有資料洩漏。但我們人在 proposal 和期中階段已經用 2023–2026 跑過實驗，才決定期末要做什麼，所以分數可能略偏樂觀。
- **預測得到，不等於賺得到。** 美股收盤已經是台灣隔天清晨，這時今天的台股收盤價早就過去了，沒辦法回頭用那個價格買進。隔夜的漲跌大多直接反映在隔天開盤價上。
- **「融合」的意思**：把不同來源的特徵放進同一張表，交給同一個模型。不是每個來源各訓練一個模型再投票。另外做的「LightGBM 和 LSTM 機率平均」才是模型集成。
- **同一個量在不同表可能差 0.001。** 主結果表的 Full 0.590 是 4 個種子的平均預測；Shapley 那組為了跑完 31 個子集只用種子 42，所以 Full 是 0.589、S4 是 0.585。兩者是不同實驗口徑，不是算錯。差值欄是用未四捨五入的數字算的，所以偶爾和表上兩個數相減差 0.001。

---

## 1. 先要懂的基礎

### 1.1 資料
- 台灣 50 成分股（2026 Q3 名單，6 個產業），2019-01-02 → 2026-09-24，1,878 個交易日；原始 93,879 個股票日，扣停牌 8、每檔最後一天 50、60 日暖機與法人缺漏 2,970，剩 **90,851 列**可建模（2019-04-10 → 2026-09-23，1,818 天）[EDA]。
- 來源：TWSE 官方端點（MI_INDEX、T86、MI_MARGN）逐日爬取，與 FinMind 鏡像逐列比對：收盤價 99.99%、成交量 100%、外資買賣 100%、融資融券 100% 相同 [EDA]。美股與全球序列來自 yfinance，對齊規則是「最新一個收盤日期 ≤ t 的美股盤」，超過 3 天視為過期並加旗標（2.5% 的列）。
- 測試集：2023 → 2026-09-23，44,973 列、900 個交易日，模型沒有看過（訓練、選超參數、選門檻都只用更早的資料，所以不是 data leakage）；但我們人看過（proposal 與 Progress Update 階段已用這段期間跑過實驗，才決定期末的特徵與模型），所以分數可能略偏樂觀，不算一次性的乾淨測試（見第 10 節限制）。

### 1.2 五個來源（45 個特徵，全是比率或報酬率）
| 來源 | 內容 | 可得時間 | 特徵數 |
|---|---|---|---|
| S1 價格技術 | ret1–20、MA5/20/60 比、vol20、量比、振幅、跳空、RSI14、價差 | t 日 13:30 | 15 |
| S2 法人籌碼 | 外資 / 投信 / 自營 淨買 ÷ 成交量（1、5 日）、融資融券變化 | t 日收盤後 | 10 |
| S3 台股大盤 | 加權指數報酬、MA20 比、波動、量能、個股相對大盤 | t 日 13:30 | 6 |
| S4 隔夜美股 / 全球 | S&P 500、NASDAQ、SOX、TSM ADR、DXY、日經、VIX 水準與變化、USD/TWD、美債 10 年、過期旗標 | 台灣時間約 05:00 | 11 |
| S5 日曆 | 星期、月份、月底 | 已知 | 3 |

### 1.3 EDA 的五個事實（決定了設計）[EDA]
1. 標籤接近平衡：上漲 47.6%，不漲 52.4%（平盤 7.8%、下跌 44.6%）；每年 45.7–49.8%；每檔 39.0–51.6%。所以主指標用 AUROC，輔以 MCC、Macro-F1、PR-AUC、Brier。
2. 台灣端每個特徵與次日方向的 |Spearman ρ| < 0.05；外資淨買、RSI、融資變化分十等分，上漲率都在 46–49%。只有 S4 有感：隔夜費半報酬最低五分之一上漲率 34.6%，最高五分之一 57.5%。費半隔夜報酬與加權指數次日報酬相關 0.49，自身落後一日 0.03。
3. S4 內部共線：S&P 500–NASDAQ 0.93、NASDAQ–SOX 0.83（Spearman）。所以歸因做來源級，不做特徵級。
4. 50 檔同日高度相關：平均兩兩 0.28（2020 年 0.45），PC1 解釋 31%。44,973 列測試資料實際上約 900 個獨立的「天」，所以所有 CI 以交易日為單位重抽樣。
5. 美台同向比例逐年 53–61%，2026 斜率與 2023 不同，所以逐年重訓、逐年報告。

### 1.4 共用協定（`common.py`）
- 擴展式 walk-forward，測試年 2023 / 2024 / 2025 / 2026；訓練 = 所有 t+1 早於測試年首日的列，隔一天 purge。
- 內層驗證 = 訓練窗最後 12 個月，只用於早停、決策閾值（argmax MCC）、任何超參數選擇。
- 填補、標準化、閾值、樹數、四分類切點都在訓練窗內估。
- 五個標籤：y_up（主）、y_tr（除息還原，440 個除息日 170 個標籤改變）、y_gap、y_intra、y_rel；四分類 k = 訓練窗 |r| 中位數（{', '.join(pc(x, 2) for x in r4['k_by_test_year'].values())}）。
- 推論：5 日 moving-block bootstrap B = 1000，成對比較用同一組重抽；AUROC 差另報 DeLong p、逐日準確率差 t 檢定；RQ1 / RQ2 / RQ3 各自 Holm 校正。宣稱規則：CI 不含 0 且四年至少三年同號。
- 模型：規則、L2 LR（C = 0.1）、LightGBM（lr 0.03、31 葉、min 200 列 / 葉、80% 行列抽樣、L2 = 5、早停）、XGBoost（深度 5，同政策）、LSTM(64)、2 層 Transformer（d = 64，4 頭）W = 5；LightGBM 多分類。同折、同臂、同早停規則、同種子，不做分臂調參。

### 1.5 融合怎麼做
把來源的特徵**串接成一張表餵同一個模型**。沒有加權投票、沒有分來源子模型。每個「臂」只差輸入哪些欄位，差異只能來自資訊本身。31 個非空子集 × LR / LightGBM 全部訓練（31 × 4 折 × 2 = 248 次），才能算精確 Shapley 與交互作用；關鍵臂再跑 4–5 個種子。

---

## 2. 主結果（R1–R3；`02_tabular.py`、`03_sequence.py`、`04_stats.py`、`09_explain.py`）

### 2.1 五種模型、三種資訊集（pooled AUROC，種子平均機率）
{main_tbl}

- 逐年 LightGBM Full {byyear('LightGBM|Full')}；S4-only {byyear('LightGBM|S4-only')}；TW-only {byyear('LightGBM|TW-only')}。
- 種子標準差 {seed_std:.4f}，和 Full − S4 的差一樣大。DeLong p = {cmp('RQ1 Full vs S4-only (LightGBM)')['delong_p']:.4f} 看起來顯著，但它把 44,973 列當獨立樣本；交易日 bootstrap 的 CI 含 0，逐日準確率差 t = {cmp('RQ1 Full vs S4-only (LightGBM)')['day_t']:.2f}。依宣稱規則：**不成立**。
- Full 對費半報酬規則 {D('Full vs SOX-return rule')}；S4-only 對費半規則 {D('S4-only LightGBM vs SOX-return rule')}。模型比一條免費規則多的不多。
- RQ3 模型家族：LightGBM Full 對 LR {D('RQ3 LightGBM Full vs LR Full')}；LSTM 對 LightGBM {D('RQ3 LSTM Full vs LightGBM Full')}；Transformer 對 LightGBM {D('RQ3 Transformer Full vs LightGBM Full')}。全部含 0。

### 2.2 來源 Shapley（RQ2；價值 = Full AUROC − 0.5，31 個子集精確計算）
{shap_tbl}

- 兩兩交互作用（最大四個）：{inter_txt}。比 S4 主效果小一個數量級；「籌碼會改變美股效果」的假設沒有被支持。
- 模型內部 grouped SHAP（{yrs[-1]}）：{gs_txt}。模型「用了」台灣特徵約三成的歸因，但它們幾乎不帶樣本外訊號，所以 Shapley over subsets 才是主要歸因，SHAP 只當交叉檢查。

### 2.3 看法
RQ1 在規定的主任務上是「否」，而且是五種模型一致的否。我認為這不是模型沒學好，而是收盤→收盤方向的可預測部分幾乎全部來自隔夜美股這個共同因子；台灣資料是在這個共同因子之外的個股雜訊，單日方向上沒有可學的結構。

---

## 3. 拆開「明天」（R4–R6；`06_labels.py`）

{lab_tbl}

{four_tbl}

- 跳空：Full − S4 {sg(v['y_gap']['Full_minus_S4'][0])} 到 {sg(v['y_gap']['Full_minus_S4'][1])}，{v['y_gap']['Full_minus_S4_same_sign_years']} 年同號。台灣資料告訴你每一檔會跳多少（個股 beta、流動性）。
- 開盤→收盤：美股被開盤價吸收後就沒用了（S4 {f3(v['y_intra']['S4-only']['auroc'])} 低於 0.5）；只剩台灣資料一點訊號。
- 相對大盤：Full − S4 {sg(v['y_rel']['Full_minus_S4'][0])} 到 {sg(v['y_rel']['Full_minus_S4'][1])}，四年同號。這是「哪一檔」的問題。
- 幅度：除以 vol20 之後 TW 從 {f3(r4['magnitude_TW-only']['auroc'])} 掉到 {f3(vn['magnitude_TW-only']['auroc'])}，因為除掉了預測它的那個特徵：「漲跌幅度大不大」主要就是「這檔最近波動大不大」。strong-vs-mild 在方向內的 AUROC：Full {f3(r4['four_Full']['auroc_strong_vs_mild_up'])} / {f3(r4['four_Full']['auroc_strong_vs_mild_down'])}。
- 同一天內（`explain.json['within_day']`）：S4 給 50 檔同一個分數，同日 AUROC 必為 0.5；分股基準率 {f3(wd['stock base-rate']['mean'])}、TW-only {f3(wd['TW-only']['mean'])}、Full {f3(wd['Full']['mean'])}（Full − S4 逐日 t = {wd['Full_minus_S4_days']['t']:.1f}，Full − 基準率 t = {wd['Full_minus_baserate_days']['t']:.1f}）；rank IC {f3(wd['IC Full']['mean'])}，{pc(wd['IC Full']['share_gt'], 0)} 的日子為正。

### 3.1 產業與個股（R7）
{sec_tbl}

50 檔中 {E['by_stock_summary']['n_better']} 檔 Full − S4 為正，{E['by_stock_summary']['n_sig_up']} 檔顯著變好、{E['by_stock_summary']['n_sig_down']} 檔顯著變差（95%）；50 次 5% 檢定本來就會給 2–3 個假陽性。沒有產業型態。

### 3.2 看法
這一節是整個研究最重要的轉折。同樣的 45 個特徵，只是換了「明天」的定義，RQ1 的答案就從「否」變成「是」。文獻裡「融合有沒有用」的爭論，很大一部分是因為大家問的不是同一個問題。

---

## 4. 稽核、文獻重測與回測（R9–R11）

### 4.1 協定稽核（`05_audits.py`）
{audit_tbl}

### 4.2 StockNet 基準：我們的模型（`07_sota_stocknet.py`、`07b_sota_stocknet_deep.py`）
協定照文獻：87 檔（88 檔中有價格者）、訓練 2014-01 → 2015-07、驗證 → 2015-09、測試 2015-10 → 2015-12；標籤 ≥ +0.55% 漲、≤ −0.5% 跌、中間去掉；回看 5 天。
{sn_tbl}

### 4.3 StockNet 官方程式（`30_run_stocknet_official.sh`、`30_stocknet_official.py`；{'完成' if SO else '未完成'}）
{so_tbl if SO else '（尚未執行）'}

{('- 官方程式（yumoxu/stocknet-code，安全審核後執行）+ 官方資料集 + GloVe twitter 50d；Python 3 移植只改語法；config.yml 用論文設定（15 epochs、batch 32、alpha 0.5、測最後一個 checkpoint）；程式沒有固定種子。兩張 4090 被 vLLM 占滿且 TF 1.15 的 cuDNN GRU 在 4090 上失敗，所以在 CPU 跑，五個 run 並行約 6 小時。' + chr(10) + f"- **關鍵發現**：官方 DataPipe 除了 ±0.5% 中性帶，還丟掉 5 天視窗內任一天沒有推文的樣本，實際訓練 / 驗證 / 測試只有 {SO['counts']['train']['n']:,} / {SO['counts']['dev']['n']:,} / {SO['counts']['test']['n']:,} 筆（論文 {SO['published']['paper_sample_counts']['train']:,} / {SO['published']['paper_sample_counts']['dev']:,} / {SO['published']['paper_sample_counts']['test']:,}）。三種變體都低於多數類，MCC 全為負；一個 run 整個測試集只預測一類。驗證集 acc 整個訓練過程在 0.47–0.57 擺盪，沒有收斂跡象。" + chr(10) + '- 看法：官方程式在它自己看得到的資料上連多數類都沒達到，而論文的 58.23% 是在 3,720 筆上報的，兩邊樣本集不同。我們能說的是「用官方程式與官方資料，照論文設定，重現不出論文數字」，不能說論文造假；可能的原因包括資料集釋出版本與論文用的不同、程式釋出版本有過濾 bug、論文測的是驗證集最佳 checkpoint 等，這些都無法在沒有作者回應的情況下判斷。') if SO else ''}

### 4.4 CausalStock（NeurIPS 2024，無官方程式碼；`14_causalstock_v2.py`）
{cs_tbl if CV else '（未執行）'}

- v1 依論文文字實作，在 KDD17 與台灣 50 都低於多數類，判定實作不到位後重寫 v2，訓練配方取自獨立重現倉庫的文件（lr 1e-4、BCE 權重 1.0、KL warm-up、梯度裁剪、best-of-3 restarts）。
- 第 6 欄已對照原始 AAPL.csv 驗證確實是 close_T − close_T−1 的美元變化量。該標籤等於問「今天漲幅是否比昨天大」，均值回歸就能答對一半以上。
- **可宣稱**：63.4% 只有在標籤錯誤時才重現得出來；重現倉庫的作者犯了這個錯；原論文作者有沒有犯，沒有程式碼無法判斷。

### 4.5 外部官方程式交叉驗證（`external/`，`external_runs.json`）
{xr_tbl if XR else '（未執行）'}

安全審核：兩個倉庫皆無網路外連（除需金鑰的 OpenAI 呼叫）、無子程序外呼、無 eval / exec / pickle、無混淆碼。

### 4.5b 台灣 50 同資料 head-to-head（`33_sota_headtohead.py`、`33_run_advalstm.sh`；`sota_headtohead.json`）
{hh_sec}

### 4.6 台灣 baseline：Yang et al. 2024（`08_sota_yang.py`）
{yang_tbl if Y else ''}

簡化（簡報上有寫）：一個 pooled LSTM 而不是每檔一個；線性 SVM 在 20k 列子樣本；指標集是我們的。oracle 列顯示「預測指標」要值多少錢，必須把次日收盤預測到什麼程度。

### 4.7 含成本回測（`10_backtest.py`）
規則事前固定：每個測試日用種子平均機率排名，買前 k 名，t+1 開盤買、收盤賣，等權；成本 = 手續費 0.1425% × 2 + 當沖證交稅 0.15% = {pc(B['cost_round_trip'], 3)}（折扣後 {pc(B['cost_discounted'], 3)}）；多空只當理論上界（空方稅 {pc(B['cost_long_short'] - B['cost_round_trip'], 2)} 額外）。
{bt_tbl}

- 基準：{bm_txt}
- 注意：S4-only 模型與費半規則給同一天 50 檔**同一個分數**（S4 特徵是市場層級），所以它們的前 k 名是平手後任意排序，兩列數字相同不是巧合；「選股」只能靠台灣資料。
- 看法：方向訊號真實存在（毛 Sharpe {st['p_oc_TW-only|top5']['long_only_gross']['sharpe']:.2f}；重抽 {RH['random_top5_sharpe']['n_draws']} 次隨機前 5 名的 Sharpe 是 {RH['random_top5_sharpe']['p2_5']:.2f} 到 {RH['random_top5_sharpe']['p97_5']:.2f}，模型高於其中 {pc(RH['strategies']['TW-only (Open->Close model)']['share_of_random_draws_below'], 0)}。原本只抽一次的隨機基準 {st['p_rand|top5']['long_only_gross']['sharpe']:.2f} 剛好落在這個範圍的上緣，低估了差距），但報酬發生在隔夜，而隔夜正是美股訊號所在、也是當沖進不去的地方。這不是回測參數能救的問題。

### 4.8 看法：文獻可轉移性
三篇文獻在正確條件下都沒重現，一篇官方程式（Adv-ALSTM）重現了但水準只有多數類。我不認為這代表文獻造假；更可能的解釋是：隨機切分（Yang）、標籤讀取錯誤（CausalStock 重現者）、資料集版本差異（StockNet）、以及小測試集上的模型選擇（3,744 筆上一個百分點只有約 1.2 個標準誤）。但對我們的研究而言結論一致：**文獻的準確率數字不能當作可達到的水準來引用**。

---

## 5. 台股因果圖（R13；`20`–`22`、`26`）
{cg_txt if cg_txt else '（未執行）'}

---

## 6. 新聞與擴充來源（R12、R14、R15）

### 6.1 四個新來源（`19_source_study.py`，`source_study.json`）
{ss_tbl if SS else '（未執行）'}

{(f"- 九來源 exact Shapley（511 子集 × 4 折）：{sh9_txt}。" + chr(10) + f"- 融合對只用美股：Full+S7+S8 {f3(fu['Full+S7+S8']['auroc'])} − S4-only {f3(fu['S4-only']['auroc'])} = {sg(fu['Full+S7+S8 minus S4-only'][0])} 到 {sg(fu['Full+S7+S8 minus S4-only'][1])}，{fu['Full+S7+S8 minus S4-only same-sign years']} 年同號；All9 − S4-only {sg(fu['All9 minus S4-only'][0])} 到 {sg(fu['All9 minus S4-only'][1])}。**加了台指期夜盤與跨股票結構之後，融合才在主任務上贏過只用美股**，這是 Conclusion 的 RQ2 補充。" + chr(10) + f"- 新聞覆蓋：{pc(SS['coverage']['share_stock_days_with_news'], 0)} 的股票日有至少一則個股新聞。S6 單獨 {f3(SS['y_up']['S6-only']['auroc'])}，加到 Full 反而變差，表示訓練窗內 TF-IDF 分數帶進了雜訊。" + chr(10) + "- 限制：這張表與九來源 Shapley 只用一個種子（42），Shapley 沒有 bootstrap CI（511 子集 × 4 折已是 2,044 次訓練）。九來源整體的結果另用 4 個種子重做（4.5b），並在新資料上檢查（10d）。這一節的 S8 數字是夜盤對齊修正後的（見 10c）；修正前 S8 單獨只有 0.509、加到五來源只多 +0.011。") if SS else ''}

### 6.1b 台指期夜盤為什麼比美股指數強（`37_night_session_analysis.py`，`night_session.json`；`34_final_model_checks.py` 的 C 部分）

**在問什麼**：九來源比五來源好，主要來自 S8。S8 裡的夜盤漲跌到底提供了什麼？它只是把美股的消息換個形式再說一次，還是有美股沒有的資訊？

**夜盤是什麼**：台指期的盤後交易，從下午 15:00 交易到隔天清晨 05:00，和美股開盤時間重疊，比我們 08:30 的預測時點早 3.5 小時。特徵定義是「這一晚夜盤的收盤價 ÷ 當天日盤收盤價 − 1」。

**結果**（以 {NS['n_days']:,} 個交易日的市場平均來看）

| 訊號 | 和隔天開盤跳空的相關 | 和隔天收盤漲跌的相關 | 和隔天開盤後漲跌的相關 |
|---|---|---|---|
| 台指期夜盤 | {NS['correlation']['night']['next_overnight_gap']:.2f} | {NS['correlation']['night']['next_day_return']:.2f} | {NS['correlation']['night']['next_intraday_leg']:.2f} |
| 費半 | {NS['correlation']['sox']['next_overnight_gap']:.2f} | {NS['correlation']['sox']['next_day_return']:.2f} | {NS['correlation']['sox']['next_intraday_leg']:.2f} |
| NASDAQ | {NS['correlation']['nasdaq']['next_overnight_gap']:.2f} | {NS['correlation']['nasdaq']['next_day_return']:.2f} | {NS['correlation']['nasdaq']['next_intraday_leg']:.2f} |
| S&P 500 | {NS['correlation']['sp500']['next_overnight_gap']:.2f} | {NS['correlation']['sp500']['next_day_return']:.2f} | {NS['correlation']['sp500']['next_intraday_leg']:.2f} |
| 台積電 ADR | {NS['correlation']['tsm_adr']['next_overnight_gap']:.2f} | {NS['correlation']['tsm_adr']['next_day_return']:.2f} | {NS['correlation']['tsm_adr']['next_intraday_leg']:.2f} |
| 修正前的夜盤（其實是前一晚） | {NS['correlation']['night_first_version']['next_overnight_gap']:.2f} | {NS['correlation']['night_first_version']['next_day_return']:.2f} | {NS['correlation']['night_first_version']['next_intraday_leg']:.2f} |

- **一個數字就勝過五個美股指標**：五個美股序列合起來能解釋隔天跳空的 {pc(NS['r2_of_next_overnight_gap']['five_us_series'], 0)}，夜盤一個數字能解釋 {pc(NS['r2_of_next_overnight_gap']['night'], 0)}，兩邊一起用也只到 {pc(NS['r2_of_next_overnight_gap']['both'], 0)}。
- **夜盤包含美股的資訊，反過來不成立**：夜盤裡美股解釋不了的部分，和隔天跳空的相關仍有 {NS['what_each_adds']['part_of_night_not_explained_by_us']['corr_gap']:.2f}；費半裡夜盤解釋不了的部分，只有 {NS['what_each_adds']['part_of_sox_not_explained_by_night']['corr_gap']:.2f}。
- **當成規則直接用**：夜盤漲跌當分數，測試期 AUROC {f3(NS['rule_auroc_test_years']['night_return_as_score']['auroc'])} [{f3(NS['rule_auroc_test_years']['night_return_as_score']['ci'][0])}, {f3(NS['rule_auroc_test_years']['night_return_as_score']['ci'][1])}]；費半當分數 {f3(NS['rule_auroc_test_years']['sox_return_as_score']['auroc'])}；差 {qd(NS['rule_auroc_test_years']['night_minus_sox'])}。這個單一規則已經接近九來源模型的 {f3(HH['rows'][HH['final_model']]['auroc'])}。在同一批資料上直接比較：最終模型減夜盤規則 {NRD}，區間包含 0。也就是說，在整體 AUROC 上，九來源模型沒有顯著贏過「夜盤漲跌」這一個數字；模型多出來的價值在同一天內排序個股（夜盤規則對所有股票給同一個分數，做不到）。
- **放進模型**（LightGBM，4 個 seed；`robustness.json`）：台灣資料 + S8 比台灣資料 + 美股高 {pq(RC['seed_mean']['S8_arm_minus_S4_arm'])}（{RC['seed_mean']['S8_arm_minus_S4_arm']['same_sign_years']}/4 年同號）。已有美股再加 S8：{pq(RC['seed_mean']['S8_on_top_of_S4'])}。已有 S8 再加美股：{pq(RC['seed_mean']['S4_on_top_of_S8'])}，{RC['seed_mean']['S4_on_top_of_S8']['same_sign_years']}/4 年同號，各 seed 分別是 {' / '.join(sg(x) for x in RC['S4_on_top_of_S8_per_seed'])}。**更正**：11-13 的版本只跑 seed 42，得到 {sg(FC['C_S4_vs_S8']['y_up']['S4_on_top_of_S8']['diff'])}，因此寫成「已知夜盤後美股沒有幫助」。這個結論是單一 seed 造成的，不成立。正確說法是：夜盤帶的訊號比較多，美股仍補上一部分。對開盤後的漲跌，兩者都沒有幫助。
- **美股休市時差別最大**：沿用舊美股盤的日子（{NS['by_us_session']['U.S. session reused (U.S. holiday)']['n']:,} 列），費半規則 {f3(NS['by_us_session']['U.S. session reused (U.S. holiday)']['sox'])}，夜盤規則 {f3(NS['by_us_session']['U.S. session reused (U.S. holiday)']['night'])}。夜盤照常交易，所以原本列為限制的「6.6% 的日子沿用舊美股盤」由它補上。

**怎麼解讀**：夜盤是台灣市場自己把隔夜消息換算成「台股該漲還是該跌」的結果。美股指數是間接的：費半漲 2% 不代表台股漲 2%，還要看權重、匯率和台灣自己的消息。夜盤裡多出來的那部分從哪裡來（亞洲與歐洲盤、匯率、本地盤後消息），我們沒有逐項驗證，這是推論。

**沒有改變的事**：這個訊號預測的是隔夜跳空，開盤時價格已經反映完，所以「預測得到、開盤後買不到」的結論不變。它也沒有偷看未來：對開盤後漲跌的相關是 {NS['correlation']['night']['next_intraday_leg']:.2f}。

### 6.1c 最終模型的檢查（`34_final_model_checks.py`，`final_model_checks.json`）

**進步來自哪裡**（成對比較，交易日區塊 bootstrap）

| 比較 | ΔAUROC [95% CI] | 同號年數 |
|---|---|---|
| 多加四個來源（LightGBM，九 − 五） | {qd(FC['A_decomposition']['more_sources_single_model'])} | {FC['A_decomposition']['more_sources_single_model']['same_sign_years']}/4 |
| 多加四個來源（集成，九 − 五） | {qd(FC['A_decomposition']['more_sources_ensemble'])} | {FC['A_decomposition']['more_sources_ensemble']['same_sign_years']}/4 |
| 模型平均（五來源，集成 − LightGBM） | {qd(FC['A_decomposition']['ensemble_on_five_sources'])} | {FC['A_decomposition']['ensemble_on_five_sources']['same_sign_years']}/4 |
| 模型平均（九來源，集成 − LightGBM） | {qd(FC['A_decomposition']['ensemble_on_nine_sources'])} | {FC['A_decomposition']['ensemble_on_nine_sources']['same_sign_years']}/4 |

進步幾乎全部來自資料來源。九來源時模型平均沒有額外幫助，LightGBM 單獨就和最終模型一樣好。

**是更會判斷哪一天，還是更會挑股票**

| 模型 | 整體 AUROC | 只看「哪一天」 | 同一天內排序 | rank IC |
|---|---|---|---|---|
| LightGBM 五來源 | {f3(FC['B_which_day_vs_which_stock']['LightGBM Full (S1-S5)']['pooled_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['LightGBM Full (S1-S5)']['which_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['LightGBM Full (S1-S5)']['within_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['LightGBM Full (S1-S5)']['rank_ic'])} |
| LightGBM 九來源 | {f3(FC['B_which_day_vs_which_stock']['LightGBM All9 (S1-S9)']['pooled_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['LightGBM All9 (S1-S9)']['which_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['LightGBM All9 (S1-S9)']['within_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['LightGBM All9 (S1-S9)']['rank_ic'])} |
| 最終模型（集成，九來源） | {f3(FC['B_which_day_vs_which_stock']['Ensemble LightGBM+LSTM All9 (ours, final)']['pooled_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['Ensemble LightGBM+LSTM All9 (ours, final)']['which_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['Ensemble LightGBM+LSTM All9 (ours, final)']['within_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['Ensemble LightGBM+LSTM All9 (ours, final)']['rank_ic'])} |
| 各股歷史上漲率 | {f3(FC['B_which_day_vs_which_stock']['stock base-rate']['pooled_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['stock base-rate']['which_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['stock base-rate']['within_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['stock base-rate']['rank_ic'])} |
| 費半規則 | {f3(FC['B_which_day_vs_which_stock']['SOX-return rule']['pooled_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['SOX-return rule']['which_day_auroc'])} | {f3(FC['B_which_day_vs_which_stock']['SOX-return rule']['within_day_auroc'])} | — |

「只看哪一天」是把同一天所有股票的分數換成當天平均再算 AUROC。兩邊都有進步：判斷哪一天從 {f3(FC['B_which_day_vs_which_stock']['LightGBM Full (S1-S5)']['which_day_auroc'])} 到 {f3(FC['B_which_day_vs_which_stock']['Ensemble LightGBM+LSTM All9 (ours, final)']['which_day_auroc'])}；同一天內排序，最終模型比「各股歷史上漲率」高 {FC['B_which_day_vs_which_stock']['paired_within_day']['final minus stock base-rate']['mean']:+.3f}（逐日 t = {FC['B_which_day_vs_which_stock']['paired_within_day']['final minus stock base-rate']['t']:.1f}），五來源時這個差只有 t = {FC['B_which_day_vs_which_stock']['paired_within_day']['LightGBM Full minus stock base-rate']['t']:.1f}。挑股票的能力變得測得到了，但仍然很小。

**模型的信心可不可信**：把測試期的預測依「離 0.5 多遠」分成五等份，準確率依序是 {' / '.join(f3(d['accuracy']) for d in FC['D_confidence']['accuracy_by_confidence_quintile'])}。越有把握越準。這是事後的描述，沒有用測試期選門檻。

**回測**（隔天開盤買前五名、收盤賣）：扣成本前 Sharpe {FC['B_backtest_final_score_intraday']['strategies']['Ensemble LightGBM+LSTM All9 (ours, final)|top5']['gross']['sharpe']:.2f}，損益兩平成本每趟 {pc(FC['B_backtest_final_score_intraday']['strategies']['Ensemble LightGBM+LSTM All9 (ours, final)|top5']['break_even_round_trip_cost'], 3)}，實際成本 {pc(FC['B_backtest_final_score_intraday']['cost_round_trip'], 3)}，扣成本後年化 {pc(FC['B_backtest_final_score_intraday']['strategies']['Ensemble LightGBM+LSTM All9 (ours, final)|top5']['net']['ann_return'], 0)}。最終模型比五來源好，但仍然賠錢。

**和已發表架構在同樣九來源上比**：官方 ALSTM {f3(HH['rows']['ALSTM official (all nine sources, 74 features)']['auroc'])}、Adv-ALSTM {f3(HH['rows']['AdvALSTM official (all nine sources, 74 features)']['auroc'])}，最終模型 {f3(HH['rows'][HH['final_model']]['auroc'])}，差 {qd(HH['paired'][HH['final_model'] + ' minus ALSTM official (all nine sources, 74 features)'])}，四年同號。每個架構拿到更多資訊都會變好；資訊相同時我們仍領先 0.015。

### 6.2 新聞還能做什麼（`25_news_analysis.py`，`news_analysis.json`）
{news_txt if news_txt else '（未執行）'}

### 6.3 LLM 新聞分數與前視偏誤（`27_llm_news_lookahead.py`，`llm_lookahead.json`）
{llm_txt if llm_txt else '（未執行）'}

### 6.4 看法
新聞在這個問題上的角色很清楚：它帶「幅度」和「有日期的事件」，不帶「所有股票日的方向」。文獻裡新聞 SOTA 的方向增益，在我們能檢查的那一篇（CausalStock）上，是標籤錯誤的產物。對台股而言，最值得做的外部資料不是新聞，而是台指期夜盤和跨股票結構，兩者都是「同一個全球共同因子」的另一種讀法。

---

## 7. Progress Update 承諾過的項目（Appendix D；`23_gap_fills.py`、`24_more_fills.py`）

{gap_txt if gap_txt else '（未執行）'}

看法：這一組裡最有訊息量的三個是 (1) McNemar 與 DM 方向相反，說明「融合」改善的是機率品質而不是硬分類，(2) 三分類一加入「平盤」類融合就大贏，再次證明台灣資料的價值在「會不會動」，(3) 內層驗證調參反而變差，提醒一年的驗證期太短，這也是為什麼全研究不做分臂調參。

---

## 8. Insights：文獻的九個說法逐條檢查

{ins_tbl}

---

## 9. 與 Progress Update 的對照

### 9.1 Progress Update 的錯誤、矛盾與重算後的數字
{pu_tbl}

Progress Update 的 14 組新數字（p.41、p.69–73）在所有程式與暫存區都找不到產生程式，期末全部由新流程重算；查核細節在 `01_查核報告_Progress_Update.md`。

### 9.2 期末新增的工作
{new_tbl}

### 9.3 Progress Update 承諾的項目
{promise_tbl}

---

## 10. 對結論的信心、限制與待決定

### 10.1 我對各結論的信心
- **高**：美股主導收盤→收盤方向；融合在主任務上不贏；協定膨脹 ×{A['leakage_ratio']:.2f}；成本下無法交易；CausalStock 標籤 artifact（兩套程式、兩種標籤一致）。
- **中**：融合在跳空 / 相對大盤的增益（CI 不含 0、四年同號，但絕對量小）；S7 + S8 讓融合贏過美股（已用 4 個種子重做，新資料上方向一致）；新聞事件漂移（事件少）。
- **低 / 需更多資料**：美股休市日子樣本；LLM 前視偏誤「量不到」（4B 模型、只有標題）；StockNet 官方程式數字的解釋（樣本集與論文不同）。

### 10.2 限制（簡報 p.49）
歷史不是乾淨的測試（2023–2026 在 proposal 階段已被我們人看過並據此做了設計決定；模型本身沒有用到，不是 leakage，乾淨測試是 2026-10 以後的新資料）；存活者偏誤（2026 Q3 名單）；特徵端未除權息還原（只有 y_tr 標籤有）；6.6% 的列沿用前一天的美股盤；回測無滑價、未回測隔夜持有；部署只有設計；新聞只用標題；外部來源研究單一種子。

### 10.3 沒做、為什麼
- Streamlit 儀表板：改以流程設計與示範輸出交代。
- GitHub 釋出、dataset zip：需要團隊帳號與期末報告時程。
- Model Confidence Set：bootstrap + DM 已足夠。
- StockNet 官方程式的「驗證集最佳 checkpoint」、更多種子、關掉推文過濾：都是偏離官方程式的改法，留待之後。

### 10.4 待團隊決定
Part 1（11/21）或 Part 2（11/28）；課程 AI 使用規範原文（Appendix E 措辭）；分工表百分比；PowerPoint 開啟後逐頁視覺檢查（此機器無 LibreOffice）；錄影總長必須 < 20:00。

---

## 10b. 外部 review 列出的待回查項目：逐條答覆（2026-11-13）

一份外部 review 把原稿裡定義不清或說得太滿的地方列了出來。下面逐條回答；「已修正」表示程式或文字已改，數字以本文為準。

| 待回查項目 | 答覆 | 處理 |
|---|---|---|
| 美股資料的日期對齊 | 每個台股交易日 t 取「日期 ≤ t 的最新一個美股交易日」。美股日期 t 的盤在台灣時間 t+1 清晨收盤，早於 t+1 的 08:30 決策時點，所以沒有用到未來資料；但它晚於台股 t 日收盤，這正是「預測得到、買不到」的原因。 | 說明 |
| S4 各欄位的可得時間 | 美股指數、VIX、TSM ADR、美債殖利率在台灣時間約 04:00–05:00（依日光節約時間）定案；日經在台灣時間 t 日下午收盤；DXY 與 USD/TWD 用的是 yfinance 的日線收盤，確切的收盤時刻沒有逐欄驗證。全部都在 t+1 的 08:30 之前。 | 說明；匯率欄位的時刻列為限制 |
| 過期 2.5% 與沿用前盤 6.6% | 兩個不同定義，已用資料重算確認：2.5% 是「最新美股盤距今超過 3 天」（us_stale 旗標）；6.6% 是「和前一個台股交易日用到同一個美股盤」（美股單日休市）。 | 說明 |
| Full 0.590／0.589、S4 0.586／0.585 | 0.590／0.586 是 4 個種子的平均預測；0.589／0.585 是 Shapley 那組的單一種子（42）。 | 已在 0b 節說明 |
| Shapley 表最後一欄的符號 | 欄名已改成「移除該來源後 AUROC 的變化（負值 = 變差）」。 | 已修正 |
| AUROC 差值和表值相減不一致 | 差值用未四捨五入的數字、在同一批重抽樣上算，和表上兩個三位小數相減可能差 0.001。 | 說明 |
| 20 日區塊置換「掉到天花板」 | 說法太滿。三次結果是 0.519／0.532／0.544，接近但不等於分股基準率規則的 0.518，而且只有三次。已改寫為健全性檢查，不再說「沒有洩漏」。 | 已修正（本文 4.1、簡報 R9） |
| 「台灣資料決定哪一檔」 | 證據確實最弱：同日排序 AUROC {f3(E['within_day']['Full']['mean'])} 對分股基準率 {f3(E['within_day']['stock base-rate']['mean'])}，t = {E['within_day']['Full_minus_baserate_days']['t']:.1f}。成立的是「只用美股完全無法排序（0.500），要排序就需要台灣資料」；不成立的是「台灣資料能明顯選出好股票」。 | 已在一頁結論與簡報 Conclusion 加註 |
| 跳空方向 vs 跳空幅度 | y_gap 只測方向。本文沒有宣稱能預測跳空多少。 | 說明 |
| 三分類的「平」 | 是 ±0.5% 的中性區間，不是收盤價完全不變。 | 說明 |
| 強弱 AUROC 0.718／0.707 的順序 | 已回查結果檔：{f3(L['four_level']['raw']['four_Full']['auroc_strong_vs_mild_up'])} 是「已知上漲時分強漲／弱漲」，{f3(L['four_level']['raw']['four_Full']['auroc_strong_vs_mild_down'])} 是「已知下跌時分強跌／弱跌」。 | 已補 |
| 因果模型「先驗圖 + 新聞」的比較對象 | 原本拿它和「學習圖、只用價格」比，同時改了圖和輸入。改成圖固定、只比有無新聞後：學習圖 {qd(CN['news_effect_same_graph']['learned graph'])}，先驗圖 {qd(CN['news_effect_same_graph']['prior graph'])}，學習圖的區間不含 0（很小的增益），先驗圖沒有，方向不一致、幅度都很小。原本「先驗圖 + 新聞統計上為正」的說法撤回。 | 已修正（本文第 5 節、簡報 R13） |
| 幅度閘門回測的 −74% | 是「純多、扣成本後」的年化報酬。多空只算了毛報酬。 | 已修正文字 |
| LLM 的 2025–2026 分組 | 原本按年份切，2025 年 1–4 月其實在模型釋出之前。已改按日期切：2025-03-31（我們假設的知識截止日，比釋出早一個月；查不到模型方的正式說明）以前，Full+LLM − Full = {sg(LK['y_up']['up to 2025-03-31 (on or before the stated knowledge cut-off)']['Full+LLM minus Full'][0])} 到 {sg(LK['y_up']['up to 2025-03-31 (on or before the stated knowledge cut-off)']['Full+LLM minus Full'][1])}；2025-04-30（權重釋出後）以後 {sg(LK['y_up']['from 2025-04-30 (after the weights were released)']['Full+LLM minus Full'][0])} 到 {sg(LK['y_up']['from 2025-04-30 (after the weights were released)']['Full+LLM minus Full'][1])}。兩段都含 0，前段沒有比後段大。知識截止日是模型方的說法，不是我們驗證過的訓練資料邊界。 | 已重跑（`27_llm_news_lookahead.py`） |
| LLM 總表區間 −0.001 到 +0.008 | 那是四年合併的區間；分期的兩個區間是各自子樣本的。 | 說明 |
| 四分類 0.345（新聞那組）vs 0.343（主結果） | 主結果是多種子，新聞那組是單一種子 42 的 Full。不同實驗口徑。 | 說明 |
| 新聞事件覆蓋率 | 原稿寫「約 1–5%」沒有根據。四類事件的股票日數相加約占 {100 * sum(NA['keyword_events'][k]['n_stock_days'] for k in ['target price raised / upgrade', 'revenue record high', 'EPS / profit', 'dividend']) / NA['baseline_all_stock_days']['n']:.1f}%（未去重）。 | 已修正 |
| 「六種模型 × 31 子集」 | 說法錯誤。31 個子集只有 LR 和 LightGBM 全跑；XGBoost、LSTM、Transformer 只跑關鍵組合。 | 已修正（簡報 Contributions） |
| 九來源結果只有一個種子 | 九來源的 Shapley（511 個子集）確實只有種子 42、沒有信賴區間。但「九來源比五來源好」已用 4 個種子重做：LightGBM 九來源 {f3(HH['rows']['LightGBM All9 (S1-S9)']['auroc'])}，LSTM、Transformer 九來源各 3 個種子（見 4.5b），新資料上也重現（見 10d）。 | 已補實驗 |
| 6 項錯誤、9 項口徑不一致的完整清單 | 完整清單在 `01_查核報告_Progress_Update.md`，本文第 9 節只列主要項目。 | 指向文件 |
| Sharpe 的算法 | 每日報酬的平均 ÷ 標準差 × √252，無風險利率當 0。 | 已補 |
| 「自身落後一日相關 0.03」 | 指加權指數今天的報酬和它自己明天的報酬的相關，用來對照「隔夜費半對加權指數明天」的 0.49。 | 說明 |
| 區間端點顯示 ±0.000 | 一律當作含 0，不稱顯著（見 0b 節）。 | 已訂規則 |

**review 裡已經過時的數字**（它看的是修正前的版本）：台灣 50 上的 CausalStock 與官方 Adv-ALSTM 數字，因為 11-13 修正了「輸入少看一天」的錯誤而更新，見 4.4、4.5、4.5b。修正後官方 Adv-ALSTM 只給價格時準確率比多數類高不到一個百分點；給我們的 45 個特徵後 AUROC 到 0.584。

---

## 10c. 時間對齊審計（`36_alignment_audit.py`，`alignment_audit.json`；2026-11-13）

**為什麼做**：11-13 一天之內查出三個時間對齊的錯，而且都是做別的事時順便發現的。所以改成有系統地查：用資料本身來驗證每一個輸入，不靠讀程式碼。

**查到並已修正的三個錯**（第 1、2 個是用到「比預期更舊」的資料；第 3 個有一部分是輕微偷看：標題晚於 08:30 決策時間最多 30 分鐘，但仍在 09:00 開盤前，沒有用到要預測的價格）

| # | 錯在哪 | 影響 | 修正後 |
|---|---|---|---|
| 1 | SOTA 架構在台灣 50 的輸入視窗只到 t−1，卻預測 t→t+1 | 讓已發表架構吃虧 | 視窗結束在第 t 天；全部重跑 |
| 2 | 台指期夜盤漲跌用到前一晚。期交所把 15:00 開始、隔天 05:00 結束的夜盤記在下一個交易日名下，原本直接取同日紀錄 | 夜盤特徵幾乎沒用（當分數 AUROC {f3(NS['rule_auroc_test_years']['first_version_as_score'])}），低估了這個來源 | 取下一個交易日名下的夜盤紀錄；當分數 AUROC {f3(NS['rule_auroc_test_years']['night_return_as_score']['auroc'])} |
| 3 | 新聞以 09:00 開盤為界，但決策時點是 08:30 | {pc(AU['T3_structure']['news_assignment']['first_version']['share_stamped_0830_0900_counted_before_the_decision'], 1)} 的標題在 08:30–09:00 之間仍被算入；另有 {pc(AU['T3_structure']['news_assignment']['first_version']['share_weekend_holiday_daytime_headlines_one_row_late'], 1)} 的假日白天標題被晚算一列 | 統一規則：08:30 到下一個交易日 08:30 的標題歸當天 |

**審計怎麼查、結果如何**

- **有沒有偷看未來**：第 t 列的特徵可以預測隔夜跳空，但不該能預測隔天開盤後的漲跌。{AU['T1_leak_test']['n_features']} 個特徵單獨預測盤中漲跌的 AUROC 最高只有 {f3(AU['T1_leak_test']['max_intraday_auroc'])}，沒有任何一個被標記。（對照：如果把「明天的報酬」當特徵，也就是偷看一天，預測收盤方向的 AUROC 是 {f3(AU['T1_leak_test']['look_ahead_scale']['ret1_one_row_later_auroc_y_up'])}。）
- **有沒有用到過期資料**：{AU['T2_staleness_test']['n_series']} 個序列各自和「確定是同一天」的參考值比對，相關性必須在同一天最高。{AU['T2_staleness_test']['n_series'] - AU['T2_staleness_test']['n_not_peaking_at_lag0']} 個通過；1 個（外資期貨部位變化）高峰不在同一天，我們直接對照期交所官網 2026-11-11、11-12 兩天的數字，和資料逐筆相同，日期沒有錯，那是外資隔天才調整部位的行為。
- **結構**：標籤定義一致率 {f3(AU['T3_structure']['label_identities']['y_up_equals_close_next_gt_close'])}；四個年度的訓練標籤都早於測試第一天；美股日期從不晚於台股日期（最大差 {AU['T3_structure']['us_merge']['max_days_us_date_after_tw_date']} 天）；序列模型與 SOTA 的視窗都結束在特徵日；修正後沒有任何標題晚於它所屬的 08:30 決策時點。
- **無法由我們驗證的**：Yahoo 日線對美元指數與美元兌台幣的確切收盤時刻。把這兩欄拿掉重跑，五來源 AUROC 變化 {sg(FC['E_drop_fx_features']['diff'])} [{sg(FC['E_drop_fx_features']['ci'][0])}, {sg(FC['E_drop_fx_features']['ci'][1])}]，區間含 0，所以沒有任何結論依賴它們。

**對結論的影響**：錯 1、3 沒有改變任何結論。錯 2 改變了擴充來源的結論（見 6.0）。五來源的結果（第 2、3 節）不使用這些輸入，沒有變。修正前的結果另存在 `results/final/before_s8_fix/`。

## 10d. 乾淨新資料的測試（`35_fresh_holdout.py`，`fresh_holdout.json`；2026-11-13）

**在問什麼**：測試期（2023–2026-09）雖然模型沒看過，但我們人在 proposal 和期中階段看過。那麼在模型和人都沒看過的資料上，結果還一樣嗎？

**怎麼測**：把管線凍結（特徵、超參數、切分規則都不動），抓 2026-09-24 到 11-12 的新資料，照原本的規則再往前走一步。分析計畫在看任何結果之前就寫在程式開頭。重建出來的資料表在 09-23 以前和凍結版逐格相同（{FH['build']['reproduces_frozen_table']['frozen_rows']:,} 列，最大差異 {FH['build']['reproduces_frozen_table']['max_abs_feature_diff']}）。

**結果**（{FH['n_days']} 個交易日、{FH['n_rows']:,} 列）

| 任務 | 新資料 | 歷史測試 | 區間是否包含歷史值 |
|---|---|---|---|
| 收盤方向，五來源 | {f3(FH['labels']['y_up']['arms']['LightGBM|Full']['auroc'])} [{f3(FH['labels']['y_up']['arms']['LightGBM|Full']['ci'][0])}, {f3(FH['labels']['y_up']['arms']['LightGBM|Full']['ci'][1])}] | {f3(FH['labels']['y_up']['walk_forward_2023_2026']['Full'])} | 是 |
| 收盤方向，只用美股 | {f3(FH['labels']['y_up']['arms']['LightGBM|S4-only']['auroc'])} [{f3(FH['labels']['y_up']['arms']['LightGBM|S4-only']['ci'][0])}, {f3(FH['labels']['y_up']['arms']['LightGBM|S4-only']['ci'][1])}] | {f3(FH['labels']['y_up']['walk_forward_2023_2026']['S4-only'])} | 是 |
| 收盤方向，只用台灣資料 | {f3(FH['labels']['y_up']['arms']['LightGBM|TW-only']['auroc'])} [{f3(FH['labels']['y_up']['arms']['LightGBM|TW-only']['ci'][0])}, {f3(FH['labels']['y_up']['arms']['LightGBM|TW-only']['ci'][1])}] | {f3(FH['labels']['y_up']['walk_forward_2023_2026']['TW-only'])} | 是 |
| 收盤方向，五來源 − 只用美股 | {qd(FH['labels']['y_up']['paired']['Full minus S4-only'])} | +0.004 | 含 0，和歷史一樣沒有證據 |
| 跳空，五來源 − 只用美股 | {qd(FH['labels']['y_gap']['paired']['Full minus S4-only'])} | +0.059 | 不含 0，重現 |
| 相對大盤，五來源 − 只用美股 | {qd(FH['labels']['y_rel']['paired']['Full minus S4-only'])} | +0.031 | 不含 0，重現 |
| 收盤方向，五來源 + S7 + S8（修正後夜盤） | {f3(FH['labels']['y_up']['arms']['LightGBM|Full+S7+S8']['auroc'])} [{f3(FH['labels']['y_up']['arms']['LightGBM|Full+S7+S8']['ci'][0])}, {f3(FH['labels']['y_up']['arms']['LightGBM|Full+S7+S8']['ci'][1])}] | {f3(SS['fusion_vs_us_only']['Full+S7+S8']['auroc'])} | 是 |

**怎麼解讀**：只有 {FH['n_days']} 天，區間很寬，單靠它不能證明什麼；但新資料上沒有任何一項和歷史測試矛盾。要誠實說明的一點：最後一列（加 S7、S8）是在夜盤對齊修正之後才跑的，不在事先寫好的計畫內；其餘各列都是計畫內的。

## 10e. 穩健性檢查與第二方審查（`38_robustness.py`，`robustness.json`；2026-11-14）

**為什麼做**：先前有幾個結論只靠單一數字（單一 seed、單一次訓練、單一次隨機抽樣）。這一節把主要結論各自用「跨年份、跨 seed、跨模型、新資料」重新檢查，並請另一個模型（OpenAI Codex）獨立審查 reference 與圖。

| 結論 | 主要數字 | 是否重現 | 強度 |
|---|---|---|---|
| 夜盤規則比費半規則好 | {pq(RB['A_night_rule_vs_sox_rule']['walk_forward'])} | {RB['A_night_rule_vs_sox_rule']['walk_forward']['same_sign_years']}/4 年；新資料 31 天 {pq(RB['A_night_rule_vs_sox_rule']['fresh_holdout'])}（同方向，含 0） | 強（測試期）；新資料太短 |
| 夜盤在美股之上還有貢獻 | {pq(RC['seed_mean']['S8_on_top_of_S4'])} | 4/4 年，4 個 seed 都為正 | 強 |
| 美股在夜盤之上還有貢獻 | {pq(RC['seed_mean']['S4_on_top_of_S8'])} | 4/4 年；各 seed {' / '.join(sg(x) for x in RC['S4_on_top_of_S8_per_seed'])} | 中（受 seed 影響；單一 seed 曾得到相反結論） |
| 九來源比五來源好 | {pq(RB['B_final_model']['lightgbm_nine_vs_five'])} | 4/4 年；seed 標準差 {f3(RB['D_seed_spread_ours']['LightGBM All9']['std'])} | 強 |
| 最終模型領先官方 ALSTM（同樣九來源） | 對 ALSTM 四次平均 {pq(RG['final_vs_mean_of_runs'])} | 對每一次單獨訓練 {' / '.join(sg(x['diff']) for x in RG['final_vs_each_run'].values())}，區間都不含 0；ALSTM 四次 {' / '.join(f3(x) for x in RG['per_run_auroc'].values())} | 中：贏每一次單獨訓練，但對方也取平均後區間碰到 0 |
| 最終模型贏過夜盤單一規則 | {pq(RB['B_final_model']['vs_night_rule'])} | {RB['B_final_model']['vs_night_rule']['same_sign_years']}/4 年 | 未確立 |
| 同一天內選股優於各股基準率 | {sg(RB['F_within_day']['final_minus_base_rate']['mean'])} [{sg(RB['F_within_day']['final_minus_base_rate']['ci'][0])}, {sg(RB['F_within_day']['final_minus_base_rate']['ci'][1])}] | 四年都為正，2026 年只有 {sg(RB['F_within_day']['by_year_final_minus_base']['2026'])} | 中 |
| 扣成本前，選前 5 名優於隨機 | 每天比等權籃子多 {RH['strategies']['TW-only (Open->Close model)']['mean_daily_excess_over_basket']*1e4:.1f} bp [{RH['strategies']['TW-only (Open->Close model)']['excess_ci'][0]*1e4:.1f}, {RH['strategies']['TW-only (Open->Close model)']['excess_ci'][1]*1e4:.1f}] | {RH['strategies']['TW-only (Open->Close model)']['positive_years']}/4 年；高於 {pc(RH['strategies']['TW-only (Open->Close model)']['share_of_random_draws_below'], 0)} 的隨機組合 | 中（毛報酬）；扣成本後全部賠錢（強） |

**這一輪查出並更正的錯**

1. 「已知夜盤後美股沒有幫助」是單一 seed 的結果，更正如上。
2. 「贏過 SOTA +0.015」只和 ALSTM 的一次訓練比，而我們是 7 個 seed 平均。補跑三次後改為上表的說法。
3. 四分類門檻 k：程式讓後面的 fold 蓋掉前面的標籤，等於所有年份都用最後一個 fold 的 k，和「每個 fold 只用自己訓練期」的說法不符（Codex 發現）。已改成每個 fold 各自建標籤並重跑；k 值本來就幾乎相同，Macro-F1 從 0.343 變成 {f3(r4['four_Full']['macro_f1'])}，結論不變。
4. 回測的隨機基準只抽一次，改抽 2,000 次。
5. Reference [32] 的 DOI 錯誤（指到另一篇論文），已更正；[8] 補文章編號、[21] 改成正式論文集的 DOI；新增方法文獻 [36]–[43]。
6. 簡報文字：φ(S2) 寫成「區間含 0」但實際不含 0；y_up = y_gap ⊕ y_intra 的寫法數學上不對；費半圖是十分位、文字卻寫五分位；多張圖被拉伸變形。皆已更正。

## 10f. 四分類深入分析（`39_four_level_deep.py`，`four_level_deep.json`；2026-11-14）

四個類別：大跌、小跌、小漲、大漲。門檻 k 是每個 fold 訓練期內 |報酬| 的中位數（約 0.85%）。這一節用 3 個 seed、四組資訊來源重做，並回答「哪一類猜得準」「模型說大漲時實際漲多少」。

**1. 融合在四分類上明顯最好，而且不受 seed 影響。** Macro-F1（3 個 seed 的平均 ± 標準差）：只用台灣 {f3(FLA['TW-only']['seed_mean'])} ± {f3(FLA['TW-only']['seed_std'])}，只用美股 {f3(FLA['S4-only']['seed_mean'])} ± {f3(FLA['S4-only']['seed_std'])}，五來源 {f3(FLA['Full']['seed_mean'])} ± {f3(FLA['Full']['seed_std'])}，九來源 {f3(FLA['All9']['seed_mean'])} ± {f3(FLA['All9']['seed_std'])}（全猜同一類 {f3(FLA['majority_macro_f1'])}）。五來源減美股 {pq(FLA['paired']['Full minus S4-only'])}，九來源減五來源 {pq(FLA['paired']['All9 minus Full'])}，都是四年同號。

**2. 大漲、大跌、小跌猜得到，小漲幾乎猜不到。** 五來源各類的 F1：大漲 {f3(FLB['f1']['strong up'])}、小跌 {f3(FLB['f1']['mild down'])}、大跌 {f3(FLB['f1']['strong down'])}、小漲 {f3(FLB['f1']['mild up'])}。模型只在 {pc(FLB['predicted_share']['mild up'], 0)} 的股票日預測「小漲」；真正小漲的日子有 {pc(FLB['confusion_row_normalised'][2][1], 0)} 被判成小跌。原因是整體上漲率只有 47.6%，幅度又小，猜小跌比較保險。

**3. 模型比較知道「會不會大動」，比較不知道「往哪邊動」。** 真正大跌的股票日，{pc(FLB['confusion_row_normalised'][0][0], 0)} 被判成大跌，但有 {pc(FLB['confusion_row_normalised'][0][3], 0)} 被判成大漲。把四分類機率拆成兩個問題來看：

| 來源 | 猜方向（AUROC） | 猜幅度大小（AUROC） |
|---|---|---|
| 只用台灣 | {f3(FLE['TW-only']['direction_auroc'])} | {f3(FLE['TW-only']['size_auroc'])} |
| 只用美股 | {f3(FLE['S4-only']['direction_auroc'])} | {f3(FLE['S4-only']['size_auroc'])} |
| 五來源 | {f3(FLE['Full']['direction_auroc'])} | {f3(FLE['Full']['size_auroc'])} |
| 九來源 | {f3(FLE['All9']['direction_auroc'])} | {f3(FLE['All9']['size_auroc'])} |

幅度來自個股自己最近的波動，加再多隔夜資訊也不會變好；方向來自隔夜資訊。融合贏，是因為只有它兩樣都有。

**4. 模型預測的類別，對應到實際報酬是有順序的。** 五來源：預測大跌的股票日隔天平均 {pp2(FLC['Full']['strong down']['mean_return'])}（上漲率 {pc(FLC['Full']['strong down']['up_rate'], 0)}），預測小跌 {pp2(FLC['Full']['mild down']['mean_return'])}，預測小漲 {pp2(FLC['Full']['mild up']['mean_return'])}，預測大漲 {pp2(FLC['Full']['strong up']['mean_return'])}（上漲率 {pc(FLC['Full']['strong up']['up_rate'], 0)}）；全部股票日平均 {pp2(FLC['Full']['all']['mean_return'])}。九來源時差距更大（大跌 {pp2(FLC['All9']['strong down']['mean_return'])}，大漲 {pp2(FLC['All9']['strong up']['mean_return'])}）。

**兩個要誠實講的限制**：(a) 這是收盤到收盤的報酬，包含隔夜跳空，開盤時買不到。(b) 這個順序大部分來自「哪一天」，不是「哪一檔」：同一天內，預測大漲的股票只比預測大跌的多 {pp2(FLC['Full']['strong_up_minus_strong_down_daily']['mean'])}（t = {FLC['Full']['strong_up_minus_strong_down_daily']['t']:.1f}），證據弱。

**5. 不一定需要四分類模型。** 用兩個二元模型相乘（上漲機率 × 大幅波動機率）得到 Macro-F1 {f3(FL['D_direct_vs_composed']['composed_macro_f1'])}，直接做四分類是 {f3(FL['D_direct_vs_composed']['direct_macro_f1'])}，差 {pq(FL['D_direct_vs_composed']['composed_minus_direct'])}。這個比較兩邊都只用一個 seed，所以只說「分成兩個問題做至少一樣好」。

## 10g. 第二方審查「結論對數據」後補做的檢查（Codex，28 項；`38_robustness.py more`；2026-11-14）

Codex 逐頁核對後指出：許多延伸實驗只跑 seed 42，卻寫成確定的結論。下面是補跑 4 個 seed（42、1、2、3）後的結果。

| 原本的說法（只有 seed 42） | 4 個 seed 之後 | 現在的說法 |
|---|---|---|
| 跨股票結構 S7 有一點幫助 | {pq(RB['J_addon_sources_over_seeds']['paired']['Full+S7 minus Full'])}，各 seed {' / '.join(sg(x) for x in RB['J_addon_sources_over_seeds']['paired']['Full+S7 minus Full']['per_seed'])} | 沒有測到影響 |
| 新聞 S6 讓收盤方向變差 | {pq(RB['J_addon_sources_over_seeds']['paired']['Full+S6 minus Full'])} | 沒有測到影響 |
| 個股籌碼 S9 讓收盤方向變差 | {pq(RB['J_addon_sources_over_seeds']['paired']['Full+S9 minus Full'])} | 沒有測到影響 |
| 夜盤 S8 有幫助 | {pq(RB['J_addon_sources_over_seeds']['paired']['Full+S8 minus Full'])}，4 個 seed、4 年都為正 | 成立 |
| 開盤後漲跌：美股低於亂猜（0.483） | 美股 {f3(RB['I_label_variants_over_seeds']['y_intra']['seed_mean_auroc']['S4-only']['auroc'])}，區間含 0.5；五來源 {f3(RB['I_label_variants_over_seeds']['y_intra']['seed_mean_auroc']['Full']['auroc'])}，區間含 0.5；只用台灣 {f3(RB['I_label_variants_over_seeds']['y_intra']['seed_mean_auroc']['TW-only']['auroc'])}，區間高於 0.5 | 只有台灣資料有一點訊號；融合沒有贏過只用台灣（{pq(RB['I_label_variants_over_seeds']['y_intra']['paired']['Full minus TW-only'])}） |
| 跳空：融合贏美股 | {pq(RB['I_label_variants_over_seeds']['y_gap']['paired']['Full minus S4-only'])}，4 個 seed、4 年都為正 | 成立 |
| 相對大盤：融合贏美股 | {pq(RB['I_label_variants_over_seeds']['y_rel']['paired']['Full minus S4-only'])}，4 個 seed、4 年都為正 | 成立 |

**其他依審查修正的地方**

- 移除一個來源的損失：原本拿「4 個 seed 平均的完整模型」和「單一 seed 的少一個來源模型」比，不對等。改成同一個 seed 比：移除美股 {pq(RB['K_matched_leave_one_out_seed42']['drop S4'])}，移除其他來源最多 {sg(max(RB['K_matched_leave_one_out_seed42'][f'drop {k}']['diff'] for k in ['S1','S2','S3','S5']))}，區間都含 0。結論不變。
- 籌碼與美股的交互作用：補上區間，LightGBM {sg(RB['L_interaction_index_ci']['LightGBM']['S2xS4']['value'])} [{sg(RB['L_interaction_index_ci']['LightGBM']['S2xS4']['ci'][0])}, {sg(RB['L_interaction_index_ci']['LightGBM']['S2xS4']['ci'][1])}]，LR {sg(RB['L_interaction_index_ci']['LR']['S2xS4']['value'])} [{sg(RB['L_interaction_index_ci']['LR']['S2xS4']['ci'][0])}, {sg(RB['L_interaction_index_ci']['LR']['S2xS4']['ci'][1])}]，兩個模型都為負且區間不含 0。
- 超參數調整：原本的差值是和 seed 1 的預設模型比（欄位名稱卻寫 seed 42）。改成和 seed 42 比：{sg(MF['tuning']['tuned_minus_default_seed42'][0])} 到 {sg(MF['tuning']['tuned_minus_default_seed42'][1])}。
- 回測：全部策略扣成本後年化 −70% 到 −62%（原本寫 −62% 到 −69%）；損益兩平成本最高的是五來源盤中前 5 名 0.055%（原本寫 0.050%，那是只用台灣的版本）。
- R1 表格的 MCC 是門檻 0.5（seed 平均機率），不是驗證集調出來的門檻；已標明。
- XGBoost 只用台灣資料其實有跑（0.520），簡報原本寫「沒跑」。
- 「同一組 seed」的說法不對：LightGBM 用 1–4、XGBoost 用 42 與 1–4、序列模型 0–2；子集、標籤變體、延伸實驗大多只有 seed 42。已照實寫。
- 「最終模型贏夜盤規則」的「哪一天」比較，改用當天平均分數來比：+0.002，區間 [−0.005, +0.009]。
- 用語：「tie」「adds nothing」「irrelevant」「useless」「exactly」等一律改成「沒有測到差異」這類說法；單一 seed 的結果標明 seed 42、探索性。

**仍然只有單一 seed、沒有補跑的**（簡報上已標示）：圖特徵（R13）、LLM 新聞分數（R15）、三分類、分產業模型、LSTM 視窗長度、九來源 Shapley。個股與產業的異質性沒有做聯合檢定。歷史成分股的影響沒有測。

## 10h. 第三份外部審查（GPT，針對 80 頁版）後的修正（2026-11-14）

審查的主要意見：文字已經改成新結果，但部分圖表、摘要、結論還停在上一輪的說法；另有幾個推論和比較口徑要修正。逐項核對後都成立，處理如下。

**圖表改成和文字同一批證據**（`40_multiseed_figures.py`，資料來自 `robustness.json`）

- R4 的圖與表改用 4 個 seed：跳空 {pq(RB['I_label_variants_over_seeds']['y_gap']['paired']['Full minus S4-only'])}、相對大盤 {pq(RB['I_label_variants_over_seeds']['y_rel']['paired']['Full minus S4-only'])}、開盤後 {pq(RB['I_label_variants_over_seeds']['y_intra']['paired']['Full minus S4-only'])}、除權息調整 {pq(RB['N_more_label_variants_over_seeds']['y_tr']['paired']['Full minus S4-only'])}、排除平盤 {pq(RB['N_more_label_variants_over_seeds']['y_up_noflat']['paired']['Full minus S4-only'])}。
- R12 的圖與表改用 4 個 seed、四個目標。區間不含 0 的只有：S8 對收盤方向與跳空；S9 對跳空（{pq(RB['O_addon_sources_other_targets_over_seeds']['y_gap']['paired']['Full+S9 minus Full'])}，非常小）；S6 對相對大盤（{pq(RB['O_addon_sources_other_targets_over_seeds']['y_rel']['paired']['Full+S6 minus Full'])}，邊界）。九來源 Shapley 仍是單一 seed，圖上已標明，且說明它和 4 seed 的增量比較是不同實驗。
- R10c 的圖加上 ALSTM 四次平均（{f3(RG['mean_of_runs_scores']['auroc'])}）。這個數字是「先把四次的分數逐列平均，再算一次 AUROC」，和我們最終模型的做法相同；四個 AUROC 直接平均是 {f3(RG['mean_of_runs_scores']['mean_of_the_four_aurocs'])}。

**推論修正**

- 「特徵預測不了盤中漲跌，所以沒有偷看未來」這個推論不成立。改成：沒有偷看的依據是時間本身（發布時間、對齊方式、新聞截止、輸入視窗、purge）；盤中檢查只是一個粗略篩檢，抓得到明顯的洩漏，不能證明沒有洩漏。
- 證據強度表：「五來源沒有贏美股」不再標「強」，改成「一致地沒有測到差異；不是證明沒有增益，區間仍容許約 +0.01」。表下說明：只有 R2 事先指定的 RQ1／RQ3 比較做了 Holm 校正，其餘都是探索性比較，區間沒有做多重比較校正。
- 「贏大盤」標籤不等於「同一天內選股」。前者是 pooled 標籤，後者要看同日 AUROC（最終模型比各股基準率高 {sg(RB['F_within_day']['final_minus_base_rate']['mean'])}）。簡報用語已分開。
- LLM：改成「這個模型、提示方式與標題資料下，沒有測到額外增益；這個實驗無法判斷是否有記憶效應」。
- 交易：08:30 才有的預測不能回到前一天收盤買進。刪掉「隔夜持有就能交易」的說法，只保留「開盤買、收盤賣」這個測過的策略的結論。

**比較口徑**

- 「美股再加 +0.014」的基準是「台灣資料 + 夜盤」的模型，不是夜盤單一規則。
- 五來源到九來源：LightGBM 是 +0.040，最終 ensemble 是 +0.033，分開寫。
- 新資料上的 +0.003 是「五來源 + S7 + S8 的 LightGBM」對夜盤規則，不是最終九來源模型；而且 S7 + S8 這一組是第一次跑完新資料之後才加的，不在事先寫好的計畫內。
- 損益兩平成本：R11 的五來源訊號最高 0.055%，R16 的九來源分數是 0.058%，兩個數字各自標明範圍。
- 費半規則是報酬率不是機率，不再報它的 MCC。
- 附錄 B 補上 S6–S9 的 29 個特徵與各自可取得的時間。
- 評估指標頁註明 AUROC 0.63 不是「63% 準確率」。

**新增分析**

- 方向 × 大小要不要分開建模（`39c_direction_size_models.py`，3 個 seed）：直接四分類 {f3(DSM['methods']['direct four-class']['macro_f1'])}；兩個二元模型相乘 {f3(DSM['methods']['independent: P(up) x P(big)']['macro_f1'])}（{pq(DSM['paired']['independent minus direct'])}）；不假設兩者獨立的版本 {f3(DSM['methods']['conditional: P(up) x P(big | direction)']['macro_f1'])}（{pq(DSM['paired']['conditional minus direct'])}）。相乘的版本分數較高，是因為它比較常猜「小漲」（F1 {f3(DSM['methods']['independent: P(up) x P(big)']['per_class_f1']['mild up'])} 對 {f3(DSM['methods']['direct four-class']['per_class_f1']['mild up'])}），同時「大漲」變差；這是決策規則不同，不是機率變準。結論：不需要四分類模型，分成兩個問題做沒有損失。
- 依用途選模型：同一天內排序，九來源 LightGBM（{f3(RB['P_within_day_lightgbm_nine_minus_final']['lightgbm_nine'])}）比最終 ensemble（{f3(RB['P_within_day_lightgbm_nine_minus_final']['final'])}）好 {sg(RB['P_within_day_lightgbm_nine_minus_final']['mean'])} [{sg(RB['P_within_day_lightgbm_nine_minus_final']['ci'][0])}, {sg(RB['P_within_day_lightgbm_nine_minus_final']['ci'][1])}]。整體 AUROC 最高的模型不是最會選股的模型。

**結論頁重做**：改成四項結果加一項主要限制；原本很長的 RQ1–RQ3 答案保留為下一頁的詳細版。

## 10i. 對照課程「Progress Review 共通問題」清單的檢查與補強（2026-11-16）

依助教發的 `DSP_2026_Progress_Review_Common_Issues.pdf` 八大項逐項對照。原本沒有或不夠的，補了七頁並重畫流程圖。

| 清單項目 | 檢查結果 | 處理 |
|---|---|---|
| 一、Progress Update：獨立進度頁、Completed／In Progress／Planned、對應頁碼 | 原本沒有獨立的進度頁 | 新增「Progress since the Progress Update」，13 項工作各標狀態、證據、頁碼（頁碼由程式自動填） |
| 二、Experimental Setting：切分、洩漏、特徵可取得時間、設定一致 | 切分、前處理只在訓練期擬合、對齊審計都有。不一致處：seed 的說法、附錄只列 45 個特徵 | 上一輪已改；這次再把腳本編號、JSON 清單、已發表模型用的特徵數同步 |
| 三、Datasets／EDA：類別分布、缺值、不同切分的差異；外部資料的來源、整合、是否有效 | 類別分布有；缺值只有一句；沒有訓練期對測試期的比較；外部資料散在各頁 | 新增「EDA：訓練期對測試年、缺值」與「外部資料」兩頁 |
| 四、Methods：流程圖、標註 Challenge、區分已實作／已評估、ablation | 流程圖還是五來源版本，沒有 S6–S9、最終模型、已發表模型，也沒標 Challenge | 重畫（`31_diagram_ipo.py`）：六組輸入、三個處理模組各標 C1–C6、註明全部已實作並評估 |
| 五、Results：簡單基準、公平比較、多 seed、各類別表現、錯誤案例 | 基準、公平比較、多 seed 都有；二元任務沒有各類別表現，也沒有錯誤分析 | 新增 R18 錯誤分析 |
| 六、Related Works：可比性、區分文獻數字與重現結果、啟示 | 內容都有但分散在 R10／R10b | 新增「R10 overview」重現狀態總表 |
| 七、Research Alignment：目標—方法—結果對應、預期對實際、不符的原因 | 沒有一頁把這些對起來 | 新增「Research alignment」 |
| 八、整體一致性 | 流程圖、腳本編號、檔案清單、章節頁是舊的 | 已同步 |

**新算出的內容**（`41_eda_and_errors.py` → `eda_and_errors.json`）

- 訓練期對測試年：上漲比例很穩定（差距在 2.4 個百分點內），但漲跌幅度不穩定。2026 年的 |報酬| 中位數是 {pc(EE['A_split_comparison']['2026']['test']['median_abs_ret'], 2)}，它的訓練期是 {pc(EE['A_split_comparison']['2026']['train']['median_abs_ret'], 2)}。這就是四分類門檻要每個 fold 各自算的原因。
- 缺值：最多的是日經報酬 {pc(EE['B_missingness']['top_features'][0]['missing_share'], 1)}（東京休市）、月營收 {pc(EE['B_missingness']['top_features'][1]['missing_share'], 1)}（暖機期）、融券變動 {pc(EE['B_missingness']['top_features'][3]['missing_share'], 1)}。
- 二元任務各類別（門檻 0.5，最終模型）：「非上漲」recall {f3(EC9['non_up']['recall'])}，「上漲」recall {f3(EC9['up']['recall'])}。模型偏向猜非上漲：只在 {pc(EC9['predicted_up_share'], 0)} 的股票日猜上漲，實際是 {pc(EC9['true_up_share'], 0)}。九來源主要改善的是「上漲」這一類（recall {f3(EC5['up']['recall'])} → {f3(EC9['up']['recall'])}）。
- 模型在哪裡失敗：
  - **夜盤沒什麼動的日子**：夜盤漲跌最小的三分之一，AUROC {f3(ED['by_size_of_night_move']['small |night return|']['auroc'])}；最大的三分之一 {f3(ED['by_size_of_night_move']['large |night return|']['auroc'])}。
  - **開盤後反轉的日子**：跳空和盤中同方向時 AUROC {f3(ED['gap_vs_intraday']['continuation (gap and intraday leg same sign)']['auroc'])}；開盤後反向走（佔 {pc(ED['reversal_share'], 0)}）時 {f3(ED['gap_vs_intraday']['reversal (opposite signs)']['auroc'])}，等於亂猜。最有把握的五分之一預測裡，猜錯的有 {pc(ED['most_confident_fifth']['share_of_errors_that_are_reversal_rows'], 0)} 是這種反轉日。
  - 白話：模型預測的其實是「開盤會往哪邊跳」，不是「開盤之後怎麼走」。
  - 產業：半導體最好（{f3(ED['by_sector']['Semiconductor']['auroc'])}），電信最差（{f3(ED['by_sector']['Telecom']['auroc'])}）。
- 重現狀態：在論文自己的資料上做了五個重現，只有官方 Adv-ALSTM 成功（KDD17：0.519 對論文表列 0.517）。Yang 等人那一項不算重現，因為資料不同，只是把流程搬到我們的資料上跑。

## 11. 檔案索引
- 腳本：`scripts/final/00–12`（主流程）、`14`（CausalStock v2）、`15–19`（新聞與新來源）、`20–22`、`26`（因果圖與供應鏈）、`23–24`（承諾項目）、`25`（新聞分析）、`27`（LLM）、`28`（Adv-ALSTM 台灣 50）、`30`（官方 StockNet）、`31`（p.12 IPO 圖）、`12_speech.py`（講稿）、`13_full_report.py`（本文件）。
- 結果：`results/final/*.json`；圖：`figures/final/f01–f19.png`。
- 外部程式（安全審核後執行）：`external/stocknet-code`、`external/stocknet-dataset`、`external/Adv-ALSTM`、`external/causalstock-reproduction`。
- 文件：`01_查核報告_Progress_Update.md`（PU 查核）、`02_講稿_期末簡報.md`、`03_完成報告與待辦.md`、`04_覆核_缺口與待辦.md`（承諾 / 完成 / 失敗 / 想法，權威清單）、`dataset_README.md`。
- 簡報：`group_A_part_1_team_04_final.pptx`（57 頁，`deck/build.js` 產生，`bun build.js`）。
- 重跑：見 `03_完成報告與待辦.md` 第 2 節；本文件 `.venv/bin/python scripts/final/13_full_report.py`。
"""
Path(ROOT / "final_present" / "05_總整理_全部實驗與結論.md").write_text(out)
print("report written", len(out), "chars")
