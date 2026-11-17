"""Redraw f18_news_analysis.png from results/final/news_analysis.json (no refit): larger fonts, no clipped titles, sample sizes on the bars."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
N = json.load(open(RES / "news_analysis.json")); ev, th, rvr, uni, BASE = N["keyword_events"], N["concept_themes"], N["revenue_headlines"], N["univariate_news_vs_magnitude"], N["baseline_all_stock_days"]
plt.rcParams.update({"font.size": 11}); fig, ax = plt.subplots(1, 4, figsize=(24, 6.2), gridspec_kw={"width_ratios": [1.55, 1.15, 1, 1]})
ks = sorted(ev, key=lambda k: ev[k]["next_mean"]); yv = np.arange(len(ks)); a = ax[0]
a.barh(yv - 0.2, [ev[k]["same_day_ret_mean"] * 100 for k in ks], 0.4, color="#bbb", label="same-day return (what the headline reports)")
a.barh(yv + 0.2, [ev[k]["next_mean"] * 100 for k in ks], 0.4, color="#2a78d6", xerr=[[(ev[k]["next_mean"] - ev[k]["next_mean_ci"][0]) * 100 for k in ks], [(ev[k]["next_mean_ci"][1] - ev[k]["next_mean"]) * 100 for k in ks]], capsize=2, label="next-day return (95 % CI)")
a.axvline(BASE["next_mean"] * 100, color="#d7263d", ls="--", lw=0.9, label="all stock-days"); a.set_yticks(yv); a.set_yticklabels([f"{k} (n={ev[k]['n_events']})" for k in ks], fontsize=10); a.set_xlabel("return, %"); a.legend(fontsize=9.5, frameon=False, loc="upper left"); a.set_title("(a) Keyword events, 2019–2026", fontsize=13, fontweight="bold")
a = ax[1]; tn = list(th); x = np.arange(len(tn)); hi = [th[k]["abs_next_return"]["members_high"] * 100 for k in tn]; no = [th[k]["abs_next_return"]["members_normal"] * 100 for k in tn]
a.bar(x - 0.2, hi, 0.4, color="#eb6834", label="concept members, top-10 % theme-news days"); a.bar(x + 0.2, no, 0.4, color="#f7c6b3", label="concept members, other days")
a.set_ylim(0, max(hi + no) * 1.28); a.set_xticks(x); a.set_xticklabels(tn, rotation=35, ha="right", fontsize=10); a.set_ylabel("mean |next-day return| (%)"); a.legend(fontsize=9.5, frameon=False, loc="upper left"); a.set_title("(b) Theme news vs size of the move", fontsize=13, fontweight="bold")
a = ax[2]; bk = [k for k in rvr if isinstance(rvr[k], dict)]; x = np.arange(len(bk)); a.bar(x - 0.2, [rvr[k]["up_rate"] for k in bk], 0.4, color="#2a78d6", label="next-day up-rate"); a.bar(x + 0.2, [rvr[k]["four"]["strong_up"] for k in bk], 0.4, color="#1baf7a", label="strong-up share")
a.axhline(BASE["up_rate"], color="#d7263d", ls="--", lw=0.9, label="up-rate, all stock-days"); a.set_ylim(0, 0.78); a.set_xticks(x); a.set_xticklabels([f"{k.replace('YoY ', '')}\n(n={rvr[k]['n']})" for k in bk], fontsize=10); a.set_xlabel("revenue YoY bucket"); a.legend(fontsize=9.5, frameon=False, loc="upper left"); a.set_title("(c) Revenue headlines vs next day", fontsize=13, fontweight="bold")
a = ax[3]; yrs = list(uni); feats = [("n_news", "headline count (stock)"), ("market_n_news", "headline count (market)"), ("vol20", "20-day volatility")]; x = np.arange(len(yrs)); w = 0.25
for j, (f, nm) in enumerate(feats): a.bar(x + (j - 1) * w, [uni[y][f]["auroc_magnitude"] for y in yrs], w, label=nm)
a.axhline(0.5, color="#999", lw=0.9); a.set_xticks(x); a.set_xticklabels(yrs); a.set_ylim(0.45, 0.80); a.set_ylabel("AUROC for |r_next| > k (axis starts at 0.45)"); a.legend(fontsize=9.5, frameon=False, loc="upper left"); a.set_title("(d) What predicts a big move", fontsize=13, fontweight="bold")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "f18_news_analysis.png", dpi=150); print("figure saved")
