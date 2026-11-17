"""Input-Process-Output flow chart and information-timing diagram for the proposal slides."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
FIG = Path(__file__).resolve().parents[1] / "figures"
for f in ["Noto Sans CJK TC", "Noto Sans CJK HK", "Droid Sans Fallback"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]; INK = "#0b0b0b"; MUTED = "#52514e"

def box(ax, x, y, w, h, title, lines, color, title_size=10, size=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=color, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec=color, lw=2))
    ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color="white")
    for i, l in enumerate(lines): ax.text(x + 0.15, y + h - 0.62 - i * 0.30, l, fontsize=size, va="center", color=INK)
def arrow(ax, x0, y0, x1, y1, color=MUTED): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color))

# ---------- Figure A: Input - Process - Output ----------
fig, ax = plt.subplots(figsize=(15, 8.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 8.1); ax.axis("off")
ax.text(2.2, 7.85, "INPUT  (known before TWSE open on day t+1)", fontsize=12, fontweight="bold", color=INK)
ax.text(7.5, 7.85, "PROCESS", fontsize=12, fontweight="bold", color=INK); ax.text(12.6, 7.85, "OUTPUT", fontsize=12, fontweight="bold", color=INK)
src = [("S1  Price & volume (TWSE)", ["OHLCV, #trades, monthly avg price", "returns lag 1-20, MA5/20/60, RSI,", "MACD, KD, Bollinger, ATR"], C[0]),
       ("S2  Institutional & margin (TWSE)", ["Foreign / trust / dealer net buy", "÷ volume, 1-day & 5-day sums", "margin & short balance changes"], C[1]),
       ("S3  Market-wide (TWSE)", ["TAIEX return, trend, volatility,", "market turnover, stock return vs. TAIEX,", "breadth (advance / decline): planned"], C[2]),
       ("S4  US / global overnight", ["S&P 500, NASDAQ, SOX, VIX,", "TSM ADR, USD/TWD, DXY, US10Y,", "Nikkei (all dated t, close < TW open t+1)"], C[3]),
       ("S5  Calendar (optional)", ["weekday, month-end, quarter-end,", "days to ex-dividend"], C[5])]
for i, (t, ls, c) in enumerate(src): box(ax, 0.3, 6.15 - i * 1.5, 4.4, 1.42, t, ls, c, 9.5, 8)
box(ax, 5.6, 5.0, 4.0, 2.3, "1. Data preparation", ["time alignment (t → t+1 label), no look-ahead", "dividend adjustment, missing / halted days",
    "feature engineering (≈ 60-100 features)", "scaling fitted on training window only", "walk-forward yearly re-training splits"], C[0])
box(ax, 5.6, 2.35, 4.0, 2.3, "2. Classification models", ["baselines: majority, previous sign, LR (S1)", "tabular: Random Forest, LightGBM, XGBoost",
    "sequence (60-day window): LSTM / GRU,", "   1D-CNN, Transformer", "stacking ensemble"], C[1])
box(ax, 5.6, 0.25, 4.0, 1.75, "3. Source ablation (RQ1 / RQ2)", ["S1 → +S2 → +S3 → +S4 → +S5", "TW-only  vs.  TW + US overnight",
    "SHAP per source · McNemar / DeLong tests"], C[2])
box(ax, 10.7, 4.55, 4.0, 2.55, "Prediction  (binary label)", ["p = P(close_{t+1} > close_t)", "y = 1: up   |   y = 0: down or flat",
    "per stock i, per trading day t", "(3-class ±0.5% variant: optional)"], C[3])
box(ax, 10.7, 1.55, 4.0, 2.55, "Evaluation", ["Accuracy · Balanced Acc · Macro-F1", "AUROC · MCC  (walk-forward, yearly)",
    "source-contribution table (ablation)", "supplementary: cost-aware backtest", "   (return, Sharpe, max drawdown)"], C[2])
for i in range(5): arrow(ax, 4.75, 6.85 - i * 1.5, 5.55, 6.15)
arrow(ax, 7.6, 5.0, 7.6, 4.68); arrow(ax, 7.6, 2.35, 7.6, 2.03); arrow(ax, 9.65, 3.5, 10.65, 5.4); arrow(ax, 9.65, 1.1, 10.65, 2.6); arrow(ax, 12.7, 4.55, 12.7, 4.13)
fig.savefig(FIG / "diagram_ipo.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------- Figure B: information timing (Taiwan time) ----------
fig, ax = plt.subplots(figsize=(14, 4.2)); ax.set_xlim(-0.5, 34); ax.set_ylim(-2.45, 2.4); ax.axis("off")
ax.plot([0, 33], [0, 0], color="#c3c2b7", lw=3)
ev = [(0, "09:00", "TWSE open\n(day t)", C[0], 1), (4.5, "13:30", "TWSE close\n(day t)\nclose_t known", C[0], 1),
      (7.5, "after close (t)", "TWSE publishes\ninstitutional flows\n& margin balances (t)", C[1], -1), (12.5, "21:30", "US market\nopens (t)", C[3], 1),
      (19, "04:00–05:00 (t+1)", "US market closes (t)\nSOX, S&P 500, TSM ADR\novernight returns known", C[3], -1),
      (23, "08:30–08:59 (t+1)", "PREDICTION TIME\nall S1–S5 features fixed", "#e34948", 1), (24.5, "09:00 (t+1)", "TWSE open\n(day t+1)", C[0], -1),
      (33, "13:30 (t+1)", "TWSE close (t+1)\nlabel: close_{t+1} > close_t ?", C[2], 1)]
for x, tm, lab, c, side in ev:
    ax.plot([x, x], [0, 0.55 * side], color=c, lw=1.6); ax.scatter([x], [0], s=70, color=c, zorder=3, edgecolors="white", linewidths=1.5)
    ax.text(x, 0.65 * side, tm, ha="center", va="bottom" if side > 0 else "top", fontsize=8.5, fontweight="bold", color=c)
    ax.text(x, (0.65 + 0.32) * side, lab, ha="center", va="bottom" if side > 0 else "top", fontsize=7.8, color=INK, linespacing=1.25)
ax.axvspan(23, 24.4, color="#e34948", alpha=0.08)
ax.annotate("", xy=(33, -2.05), xytext=(4.5, -2.05), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
ax.text(18.7, -2.22, "information set used for one prediction  →  everything observable between close_t and open_{t+1}", ha="center", fontsize=8.5, color=MUTED)
ax.text(-0.3, 2.25, "Information timing (Taiwan time) - when each source becomes available; none of them leaks the t+1 label", fontsize=11, fontweight="bold", color=INK)
fig.savefig(FIG / "diagram_timing.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("diagrams ok")
