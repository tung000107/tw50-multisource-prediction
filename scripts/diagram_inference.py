"""Inference-phase (daily prediction) diagram, paired with diagram_ipo.png (training phase)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
FIG = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.family": "DejaVu Sans"})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]; INK = "#0b0b0b"; MUTED = "#52514e"
def box(ax, x, y, w, h, title, lines, color, ts=10, s=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=color, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec=color, lw=2))
    ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", fontsize=ts, fontweight="bold", color="white")
    for i, l in enumerate(lines): ax.text(x + 0.15, y + h - 0.62 - i * 0.30, l, fontsize=s, va="center", color=INK)
def arrow(ax, x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=MUTED))
fig, ax = plt.subplots(figsize=(15, 4.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis("off")
ax.text(1.4, 4.35, "INPUT  (collected 08:30 on day t+1)", fontsize=12, fontweight="bold", color=INK)
ax.text(6.4, 4.35, "PROCESS  (frozen model)", fontsize=12, fontweight="bold", color=INK)
ax.text(12.4, 4.35, "OUTPUT", fontsize=12, fontweight="bold", color=INK)
box(ax, 0.3, 2.25, 4.4, 1.8, "TWSE data of day t (after close)", ["S1 OHLCV, technical indicators", "S2 foreign / trust / dealer net buy, margin", "S3 TAIEX return, trend, turnover", "S5 calendar flags"], C[0])
box(ax, 0.3, 0.25, 4.4, 1.8, "US session dated t (closes ~05:00)", ["S4 S&P 500, NASDAQ, SOX, VIX", "TSM ADR, USD/TWD, DXY, US10Y, Nikkei", "stale flag if US holiday"], C[3])
box(ax, 5.6, 2.25, 4.0, 1.8, "1. Feature pipeline (as in training)", ["align by date, compute 57-100 features", "scaler fitted on training window", "no re-fitting at inference"], C[0])
box(ax, 5.6, 0.25, 4.0, 1.8, "2. Trained classifier", ["p_i = P(close_{t+1} > close_t)  per stock", "threshold 0.5 -> label (up / not up)", "optional: rank 50 stocks by p_i"], C[1])
box(ax, 10.7, 2.25, 4.0, 1.8, "Next-day prediction, per stock", ["label y_hat in {1: up, 0: down / flat}", "probability p_i for ranking", "delivered before 09:00 open"], C[3])
box(ax, 10.7, 0.25, 4.0, 1.8, "Evaluation after the t+1 close", ["compare with realized close_{t+1}", "Accuracy, Bal. Acc, F1, AUROC, MCC", "supplementary: cost-aware PnL"], C[2])
arrow(ax, 4.75, 3.15, 5.55, 3.15); arrow(ax, 4.75, 1.15, 5.55, 2.9); arrow(ax, 7.6, 2.25, 7.6, 2.1); arrow(ax, 9.65, 1.15, 10.65, 3.0); arrow(ax, 12.7, 2.25, 12.7, 2.1)
fig.savefig(FIG / "diagram_inference.png", bbox_inches="tight", facecolor="white"); print("inference diagram ok")
