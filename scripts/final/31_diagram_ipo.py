"""Input-Process-Output diagram for the final deck (p.12), describing what was actually run (replaces the proposal-era figures/diagram_ipo.png,
which still listed Random Forest / GRU / 1D-CNN / stacking / McNemar and 60-100 features). Output: figures/final/diagram_ipo_final.png"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
FIG = Path(__file__).resolve().parents[2] / "figures" / "final"
from matplotlib import font_manager
for f in ["Noto Sans CJK TC", "Noto Sans CJK HK", "DejaVu Sans"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]; INK = "#0b0b0b"; MUTED = "#52514e"

def box(ax, x, y, w, h, title, lines, color, title_size=10, size=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=color, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec=color, lw=2))
    ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color="white")
    for i, l in enumerate(lines): ax.text(x + 0.15, y + h - 0.62 - i * 0.30, l, fontsize=size, va="center", color=INK)
def arrow(ax, x0, y0, x1, y1, color=MUTED): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color))

# 2026-11-16: redrawn so that the diagram matches what was finally implemented -- the extension sources S6-S9 (74 features), the final model,
# the published architectures -- and so that every module carries the challenge(s) C1-C6 it answers (course checklist: challenge mapping, version consistency).
def tag(ax, x, y, txt): ax.text(x, y, txt, ha="right", va="center", fontsize=7.6, fontweight="bold", color="#3a2c9c", bbox=dict(boxstyle="round,pad=0.25", fc="#efeafc", ec="#b9a8f0", lw=0.8))
fig, ax = plt.subplots(figsize=(15.6, 9.4)); ax.set_xlim(0, 15.6); ax.set_ylim(-0.45, 8.9); ax.axis("off")
ax.text(0.3, 8.62, "INPUT  (all known before the 08:30 decision on t+1)", fontsize=11, fontweight="bold", color=INK)
ax.text(7.2, 8.62, "PROCESS", fontsize=11, fontweight="bold", color=INK); ax.text(12.6, 8.62, "OUTPUT", fontsize=11, fontweight="bold", color=INK)
src = [("S1  Price / technical (TWSE, 15)", ["returns 1-20 d, MA ratios, vol20, volume ratio,", "H-L range, opening gap, RSI14, spread"], C[0]),
       ("S2  Institutional & margin (TWSE, 10)", ["foreign / trust / dealer net buy ÷ volume (1 d, 5 d),", "margin and short-balance changes"], C[1]),
       ("S3  Market-wide (TWSE, 6)", ["TAIEX return, trend, volatility, turnover;", "stock return minus TAIEX"], C[2]),
       ("S4  Overnight U.S. / global (Yahoo, 11)", ["S&P 500, NASDAQ, SOX, VIX, TSM ADR, USD/TWD,", "DXY, US 10Y, Nikkei; stale flag"], C[3]),
       ("S5  Calendar (3)", ["weekday, month, month-end flag"], C[5]),
       ("Extension  S6-S9 (29)  — added after the progress review", ["S6 news (cnyes) · S7 cross-stock · S8 TX night futures", "& positioning (TAIFEX) · S9 day-trading, lending, revenue"], C[4])]
for i, (t, ls, c) in enumerate(src): box(ax, 0.3, 7.05 - i * 1.36, 4.9, 1.24, t, ls, c, 9.2, 7.9)
box(ax, 6.0, 6.15, 4.3, 2.15, "1. Data preparation", ["time alignment to the 08:30 decision; 1-day purge", "medians / scaling fitted inside the training window", "expanding walk-forward, test years 2023-2026",
    "inner validation = last 12 months of the window", "five label definitions + four-level (k per fold)"], C[0]); tag(ax, 10.25, 8.08, "C3  C5")
box(ax, 6.0, 3.55, 4.3, 2.3, "2. Classification models", ["rules: majority, previous sign, SOX / night-return rule", "LR · LightGBM · XGBoost · LSTM · Transformer",
    "final model = mean of LightGBM and LSTM, nine sources", "published architectures on the same rows:", "official Adv-ALSTM / ALSTM, CausalStock (re-impl.)"], C[1]); tag(ax, 10.25, 5.63, "C1  C2")
box(ax, 6.0, 0.55, 4.3, 2.7, "3. Attribution, ablations, audits", ["31 source subsets → exact Shapley, interactions", "leave-one-out and add-on ablations (S6-S9)", "grouped SHAP inside the fitted model",
    "protocol audit (random split, nulls, forced leak)", "time-alignment audit; SOTA re-tests", "robustness: repeated seeds, fresh hold-out"], C[2]); tag(ax, 10.25, 3.03, "C2  C4  C5")
box(ax, 11.1, 5.1, 4.2, 3.2, "Prediction", ["p = P(Close(t+1) > Close(t)), per stock and day", "y = 1: up   |   y = 0: down or flat  (primary)", "other labels: overnight gap, intraday leg,",
    "   beats TAIEX, dividend-adjusted", "secondary: four levels", "   (strong / mild × down / up)", "daily ranking of the 50 stocks"], C[3]); tag(ax, 15.25, 8.08, "C3")
box(ax, 11.1, 0.55, 4.2, 4.2, "Evaluation", ["AUROC (primary), per year and pooled", "95 % CI: trading-day block bootstrap", "paired differences; DeLong; Holm", "MCC, Macro-F1, per-class F1, Brier",
    "within-day AUROC, daily rank IC", "class-wise results and failure modes", "evidence grading (seeds / years / hold-out)", "cost-aware top-k backtest (supplementary)"], C[2]); tag(ax, 15.25, 4.53, "C4  C6")
for i in range(6): arrow(ax, 5.25, 7.65 - i * 1.36, 5.95, 7.2)
arrow(ax, 8.15, 6.15, 8.15, 5.88); arrow(ax, 8.15, 3.55, 8.15, 3.28); arrow(ax, 10.35, 4.7, 11.05, 6.4); arrow(ax, 10.35, 1.9, 11.05, 2.6); arrow(ax, 13.2, 5.1, 13.2, 4.78)
ax.text(7.8, -0.32, "five proposal sources = 45 features;  nine sources = 74 features;  every module shown was implemented and evaluated", ha="center", fontsize=8.5, color=MUTED, style="italic")
ax.text(7.8, 0.02, "Challenges:  C1 low signal · C2 dependent sources · C3 ambiguous 'tomorrow' · C4 non-stationarity and day clustering · C5 protocol / leakage · C6 costs", ha="center", fontsize=8.8, color="#3a2c9c")
fig.savefig(FIG / "diagram_ipo_final.png", bbox_inches="tight", facecolor="white"); plt.close(fig); print("diagram ok")

# ---------- inference-phase diagram (p.13), consistent with the final protocol ----------
fig, ax = plt.subplots(figsize=(15, 4.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis("off")
ax.text(1.4, 4.35, "INPUT  (collected 08:30 on day t+1)", fontsize=12, fontweight="bold", color=INK)
ax.text(6.4, 4.35, "PROCESS  (frozen walk-forward model)", fontsize=12, fontweight="bold", color=INK)
ax.text(12.4, 4.35, "OUTPUT", fontsize=12, fontweight="bold", color=INK)
box(ax, 0.3, 2.25, 4.4, 1.8, "TWSE data of day t (after close)", ["S1 OHLCV, technical indicators", "S2 foreign / trust / dealer net buy, margin", "S3 TAIEX return, trend, volatility", "S5 calendar flags"], C[0])
box(ax, 0.3, 0.25, 4.4, 1.8, "Overnight: U.S. session dated t, TX night futures", ["S4 S&P 500, NASDAQ, SOX, VIX, TSM ADR, FX, ...", "stale flag if the U.S. was closed", "extension: TX night session (ends 05:00), news to 08:30"], C[3])
box(ax, 5.6, 2.25, 4.0, 1.8, "1. Feature pipeline (as in training)", ["align by date, compute the features (45 / 74)", "medians / scaler fitted on the training window", "no re-fitting at inference"], C[0])
box(ax, 5.6, 0.25, 4.0, 1.8, "2. Trained classifier", ["p_i = P(Close(t+1) > Close(t))  per stock", "threshold: inner validation (0.5 if seed-averaged)", "rank the 50 stocks by p_i"], C[1])
box(ax, 10.7, 2.25, 4.0, 1.8, "Next-day prediction, per stock", ["label y_hat in {1: up, 0: down / flat}", "probability p_i for ranking", "available at 08:30, before the 09:00 open"], C[3])
box(ax, 10.7, 0.25, 4.0, 1.8, "Evaluation after the t+1 close", ["compare with the realised Close(t+1)", "AUROC (CI), MCC, Bal. acc., Macro-F1", "supplementary: cost-aware top-k P&L"], C[2])
arrow(ax, 4.75, 3.15, 5.55, 3.15); arrow(ax, 4.75, 1.15, 5.55, 2.9); arrow(ax, 7.6, 2.25, 7.6, 2.1); arrow(ax, 9.65, 1.15, 10.65, 3.0); arrow(ax, 12.7, 2.25, 12.7, 2.1)
fig.savefig(FIG / "diagram_inference_final.png", bbox_inches="tight", facecolor="white"); plt.close(fig); print("inference diagram ok")

# ---------- information-timing diagram for all nine sources (replaces figures/diagram_timing.png, which showed S1-S5 only) ----------
fig, ax = plt.subplots(figsize=(15, 4.3)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.3); ax.axis("off")
ax.plot([0.5, 14.5], [2.15, 2.15], color="#b9b6c9", lw=3, solid_capstyle="round"); ax.axvspan(11.55, 12.25, ymin=0.08, ymax=0.92, color="#fde3e3", zorder=0)
EV = [(1.2, "13:30  (day t)", "TWSE close", ["S1 price / technical", "S3 market-wide", "S7 cross-stock"], C[0], 1), (3.1, "after the close  (day t)", "TWSE / TAIFEX publish", ["S2 flows & margin", "S9 day-trading, lending", "S8 futures positioning"], C[1], -1),
      (5.0, "15:00  (day t)", "TX night session opens", [], C[5], 1), (6.7, "21:30  (day t)", "U.S. market opens", [], C[3], -1), (8.7, "04:00–05:00  (t+1)", "U.S. market closes", ["S4 overnight U.S. / global"], C[3], 1),
      (10.2, "05:00  (t+1)", "TX night session closes", ["S8 night-session return"], C[5], -1), (11.9, "08:30  (t+1)", "DECISION", ["all S1–S9 features fixed", "S6 news headlines up to 08:30"], "#d7263d", 1), (13.6, "09:00  (t+1)", "TWSE opens", [], C[0], -1)]
for x, when, what, feats, col, side in EV:
    ax.plot([x, x], [2.15, 2.15 + side * 0.42], color=col, lw=1.8); ax.scatter([x], [2.15], s=90, color=col, zorder=3)
    y0 = 2.15 + side * 0.62; va = "bottom" if side > 0 else "top"; lines = [what] + feats
    ax.text(x, y0 + (side > 0) * (0.30 * len(lines)) - (side < 0) * 0.0, when, ha="center", va=va, fontsize=10, fontweight="bold", color=col)
    for i, l in enumerate(lines):
        yy = y0 + (0.30 * (len(lines) - 1 - i)) if side > 0 else y0 - 0.30 * (i + 1)
        ax.text(x, yy, l, ha="center", va=va, fontsize=9.2 if i else 9.6, color=INK if i == 0 else MUTED, fontweight="bold" if (i == 0 and what == "DECISION") else "normal")
ax.annotate("", xy=(10.2, 2.42), xytext=(5.0, 2.42), arrowprops=dict(arrowstyle="-", color=C[5], lw=2.2, alpha=0.55)); ax.text(7.6, 2.5, "TX night futures session", ha="center", va="bottom", fontsize=8.8, color=C[5], style="italic")
ax.text(14.9, 0.12, "S5 calendar flags are known in advance.  Taiwan time; U.S. times shift by one hour with daylight saving (open 21:30 or 22:30, close 04:00 or 05:00).", ha="right", fontsize=8.6, color=MUTED, style="italic")
fig.savefig(FIG / "diagram_timing_final.png", bbox_inches="tight", facecolor="white"); plt.close(fig); print("timing diagram ok")
