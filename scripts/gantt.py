"""Tentative schedule Gantt chart for the proposal (section F.3). Output: figures/gantt.png"""
from pathlib import Path
from datetime import date
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib import font_manager
FIG = Path(__file__).resolve().parents[1] / "figures"
for f in ["Noto Sans CJK TC", "Noto Sans CJK HK", "Droid Sans Fallback"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "axes.unicode_minus": False})
C = {"Understanding": "#2a78d6", "Preparation": "#eb6834", "Modeling": "#1baf7a", "Evaluation": "#eda100", "Report": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
D = lambda m, d, y=2026: date(y, m, d)
tasks = [  # (label, start, end, phase)
    ("Proposal slides + video",                  D(9, 28),  D(10, 3),  "Report"),
    ("Proposal on-site Q&A (A 10/06, B 10/13)",   D(10, 6),  D(10, 13), "Report"),
    ("Literature survey → Survey Report",     D(10, 6),  D(11, 2),  "Understanding"),
    ("Official TWSE data, dividend adj., features",     D(10, 6),  D(11, 2),  "Preparation"),
    ("Walk-forward splits, data quality checks",         D(10, 27), D(11, 9),  "Preparation"),
    ("Baselines & tabular models (LR/RF/LGBM/XGB)", D(10, 20), D(11, 16), "Modeling"),
    ("Source ablation v1 (S1→+S2→+S3→+S4)",          D(11, 3),  D(11, 23), "Modeling"),
    ("Sequence models (LSTM/CNN/Transformer)",    D(11, 10), D(12, 7),  "Modeling"),
    ("Source-ablation tests, SHAP, interactions",               D(11, 24), D(12, 14), "Evaluation"),
    ("Cost-aware backtest",                          D(12, 1),  D(12, 14), "Evaluation"),
    ("Final report, slides, demo (optional)",      D(12, 8),  D(12, 31), "Report"),
    ("Final presentation (date TBA)",                  D(1, 4, 2027), D(1, 15, 2027), "Report"),
]
milestones = [(D(10, 3), "10/03 Proposal due"), (D(10, 6), "10/06 Q&A (A)"), (D(10, 13), "10/13 Q&A (B)"), (D(12, 31), "12/31 Final report draft")]

fig, ax = plt.subplots(figsize=(14, 6.2)); ax.set_facecolor("#fcfcfb"); fig.patch.set_facecolor("#fcfcfb")
for i, (lab, s, e, ph) in enumerate(tasks):
    y = len(tasks) - 1 - i
    ax.barh(y, (e - s).days + 1, left=mdates.date2num(s), height=0.58, color=C[ph], edgecolor="#fcfcfb", linewidth=1.5)
    ax.text(mdates.date2num(e) + 1.2, y, f"{s:%m/%d}–{e:%m/%d}", ha="left", va="center", fontsize=7.5, color=MUTED)
for k, (d, lab) in enumerate(milestones):
    x = mdates.date2num(d); ax.axvline(x, color="#e34948", lw=1.1, ls=(0, (4, 3)), zorder=0)
    ax.text(x + 0.8, len(tasks) - 0.35 + 0.5 * (k % 3), lab, va="bottom", ha="left", fontsize=7.8, color="#e34948")
ax.axvline(mdates.date2num(date.today()), color=MUTED, lw=1, zorder=0); ax.text(mdates.date2num(date.today()), -0.9, "today", ha="center", fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in tasks][::-1], fontsize=9, color=INK); ax.tick_params(axis="y", length=0)
ax.set_ylim(-1.1, len(tasks) + 1.4)
ax.set_xlim(mdates.date2num(D(9, 21)), mdates.date2num(D(1, 24, 2027)))
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0)); ax.grid(True, axis="x", which="minor", color=GRID, lw=0.6); ax.grid(True, axis="x", which="major", color="#c3c2b7", lw=1)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in C.values()]
ax.legend(handles, ["Data Understanding", "Data Preparation", "Data Modeling", "Evaluation", "Report / Presentation"], loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5, frameon=False, fontsize=8.5, handlelength=1.2)
ax.set_title("Tentative Schedule (Gantt Chart) — 2026-09-28 to 2027-01-15", loc="left", fontsize=12, fontweight="bold", color=INK, pad=14)
fig.subplots_adjust(left=0.27, right=0.95, top=0.90, bottom=0.14)
fig.savefig(FIG / "gantt.png", bbox_inches="tight", facecolor="#fcfcfb"); print("gantt ok")
