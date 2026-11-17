"""Basic EDA figures for the proposal. Output: figures/eda_*.png + figures/eda_summary.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
for f in ["Noto Sans CJK TC", "Noto Sans CJK HK", "Droid Sans Fallback"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#898781", "ytick.color": "#898781", "grid.color": "#e1e0d9", "axes.grid": True, "axes.grid.axis": "y",
    "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 11})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # dataviz categorical order
BLUE, ORANGE, AQUA, YELLOW = C[:4]; GRAY = "#898781"

df = pd.read_parquet(ROOT / "data/processed/dataset.parquet"); full = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet")
S = {}
def save(fig, name): fig.tight_layout(); fig.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.close(fig)
def bar_style(ax): ax.set_axisbelow(True)

# ---- Fig 1: coverage & sample size ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
yr = df.groupby("year").size(); axes[0].bar(yr.index.astype(str), yr.values, color=BLUE, width=0.7)
for x, v in zip(yr.index.astype(str), yr.values): axes[0].text(x, v + 150, f"{v:,}", ha="center", fontsize=8, color="#52514e")
axes[0].set_title("Model-ready stock-day samples per year"); axes[0].set_ylabel("samples"); axes[0].set_ylim(0, yr.max() * 1.15)
axes[0].text(0.02, 0.95, f"Total = {len(df):,} (50 stocks, {df.date.min():%Y-%m} to {df.date.max():%Y-%m})", transform=axes[0].transAxes, va="top", fontsize=9)
ups = df.groupby("stock_id")["y_up"].mean().sort_values(); names = [str(k) for k in ups.index]
axes[1].barh(range(len(ups)), ups.values, color=BLUE, height=0.7); axes[1].axvline(df.y_up.mean(), color=ORANGE, ls="--", lw=1.2)
axes[1].set_yticks(range(len(ups))); axes[1].set_yticklabels(names, fontsize=5.6); axes[1].set_xlim(0.34, 0.54); axes[1].grid(True, axis="x"); axes[1].grid(False, axis="y")
axes[1].set_title(f"Up-rate per stock (each has {full.groupby('stock_id').size().min()}-{full.groupby('stock_id').size().max()} trading days)"); axes[1].set_xlabel("P(up next day)  (dashed = pooled mean)")
save(fig, "eda_01_coverage"); S["n_samples"] = int(len(df)); S["n_stocks"] = int(df.stock_id.nunique())

# ---- Fig 2: label balance ----
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
up = df.groupby("year")["y_up"].mean(); flat = df.groupby("year").apply(lambda g: (g.ret_next == 0).mean())
axes[0].bar(up.index.astype(str), up.values, color=BLUE, width=0.7); axes[0].axhline(0.5, color=GRAY, ls="--", lw=1)
for x, v in zip(up.index.astype(str), up.values): axes[0].text(x, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
axes[0].set_ylim(0, 0.7); axes[0].set_title("Share of 'up' days (y=1) per year"); axes[0].set_ylabel("P(close_{t+1} > close_t)")
cls = pd.Series({"Up (>0)": (df.ret_next > 0).mean(), "Flat (=0)": (df.ret_next == 0).mean(), "Down (<0)": (df.ret_next < 0).mean()})
axes[1].bar(cls.index, cls.values, color=[BLUE, GRAY, ORANGE], width=0.6)
for x, v in cls.items(): axes[1].text(x, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
axes[1].set_ylim(0, 0.6); axes[1].set_title("Next-day outcome distribution (all samples)")
save(fig, "eda_02_label_balance"); S["up_rate"] = round(float(df.y_up.mean()), 4); S["flat_rate"] = round(float(cls["Flat (=0)"]), 4); S["down_rate"] = round(float(cls["Down (<0)"]), 4)

# ---- Fig 3: next-day return distribution ----
fig, ax = plt.subplots(figsize=(7.5, 3.8)); r = df.ret_next.clip(-0.1, 0.1) * 100
ax.hist(r, bins=120, color=BLUE, alpha=0.9); ax.axvspan(-0.5, 0.5, color=YELLOW, alpha=0.18)
ax.set_title("Distribution of next-day return (clipped at ±10%)"); ax.set_xlabel("next-day return (%)"); ax.set_ylabel("count")
d = df.ret_next * 100
txt = f"mean {d.mean():.3f}%  median {d.median():.3f}%  std {d.std():.2f}%\nskew {d.skew():.2f}  kurtosis {d.kurt():.1f}\n|ret| < 0.5%: {(d.abs() < 0.5).mean():.1%} of days (shaded)"
ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, bbox=dict(fc="white", ec="#e1e0d9"))
save(fig, "eda_03_return_dist"); S["ret_std_pct"] = round(float(d.std()), 3); S["ret_kurt"] = round(float(d.kurt()), 1); S["small_move_share"] = round(float((d.abs() < 0.5).mean()), 4)

# ---- Fig 4: feature bins vs up-rate (Taiwan sources) ----
def binned(ax, col, title, xlabel, q=10, ylim=(0.40, 0.56)):
    s = df[[col, "y_up"]].dropna(); s["bin"] = pd.qcut(s[col], q, labels=False, duplicates="drop")
    m = s.groupby("bin")["y_up"].agg(["mean", "size"]); se = np.sqrt(m["mean"] * (1 - m["mean"]) / m["size"])
    ax.bar(m.index, m["mean"], yerr=1.96 * se, color=BLUE, width=0.7, capsize=2, error_kw=dict(lw=0.8, ecolor=GRAY))
    ax.axhline(df.y_up.mean(), color=ORANGE, ls="--", lw=1.2); ax.set_ylim(*ylim); ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("P(up next day)")
    ax.set_xticks(m.index); ax.set_xticklabels([f"D{i+1}" for i in m.index], fontsize=8); return m["mean"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
m1 = binned(axes[0, 0], "foreign_net_ratio", "Foreign investors net buy / volume (day t)", "decile (D1 = heaviest net sell)")
m2 = binned(axes[0, 1], "foreign_net_ratio5", "Foreign net buy / volume, 5-day sum", "decile")
m3 = binned(axes[1, 0], "rsi14", "RSI(14)", "decile (D1 = most oversold)")
m4 = binned(axes[1, 1], "margin_chg5", "Margin balance 5-day change", "decile (D1 = largest decrease)")
axes[0, 0].text(0.02, 0.95, "dashed = overall up-rate", transform=axes[0, 0].transAxes, fontsize=8, color=ORANGE, va="top")
fig.suptitle("Conditional next-day up-rate by feature decile (95% CI)", fontweight="bold", y=1.0)
save(fig, "eda_04_tw_features_vs_label"); S["foreign_d1_d10"] = [round(float(m1.iloc[0]), 4), round(float(m1.iloc[-1]), 4)]; S["rsi_d1_d10"] = [round(float(m3.iloc[0]), 4), round(float(m3.iloc[-1]), 4)]

# ---- Fig 5: US overnight information vs TW next-day (RQ2) ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
fresh = df[df.us_stale == 0]
m5 = binned(axes[0], "us_sox_ret", "TW up-rate vs. overnight SOX return", "SOX overnight return decile (D1 = worst)", ylim=(0.25, 0.70))
for i, v in enumerate(m5.values): axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=7.5)
# direction agreement per year: TW next-day sign vs overnight SOX sign / SP500 sign / previous-day own sign
def agree(a, b): return (np.sign(a) == np.sign(b))[(a != 0) & (b != 0)].mean()
yrs = sorted(fresh.year.unique()); rows = []
for y in yrs:
    g = fresh[fresh.year == y]
    rows.append({"year": y, "vs SOX (overnight)": agree(g.ret_next, g.us_sox_ret), "vs S&P 500 (overnight)": agree(g.ret_next, g.us_sp500_ret),
                 "vs TSM ADR (overnight)": agree(g.ret_next, g.us_tsm_adr_ret), "vs own previous day": agree(g.ret_next, g.ret1)})
ag = pd.DataFrame(rows).set_index("year")
for i, c in enumerate(ag.columns):
    axes[1].plot(ag.index, ag[c], marker="o", ms=5, lw=2, color=C[i], label=c)
axes[1].axhline(0.5, color=GRAY, ls="--", lw=1); axes[1].set_ylim(0.4, 0.75); axes[1].set_title("Direction agreement rate with TW next-day move, by year")
axes[1].set_xlim(yrs[0] - 0.3, yrs[-1] + 0.3); axes[1].set_ylabel("share of days with same sign"); axes[1].legend(fontsize=7, loc="lower left", frameon=False)
# TSMC vs TSM ADR scatter
t = fresh[fresh.stock_id == "2330"]; x = t.us_tsm_adr_ret * 100; y = t.ret_next * 100
axes[2].scatter(x, y, s=6, alpha=0.35, color=BLUE, edgecolors="none"); k, b = np.polyfit(x.dropna(), y[x.notna()], 1)
xx = np.linspace(-10, 10, 50); axes[2].plot(xx, k * xx + b, color=ORANGE, lw=2)
rho = t[["us_tsm_adr_ret", "ret_next"]].corr(method="spearman").iloc[0, 1]
axes[2].set_xlim(-10, 10); axes[2].set_ylim(-10, 10); axes[2].set_title("TSMC (2330) next-day vs. TSM ADR overnight return")
axes[2].set_xlabel("TSM ADR return, US session t (%)"); axes[2].set_ylabel("2330 return on t+1 (%)"); axes[2].grid(True, axis="x")
axes[2].text(0.02, 0.95, f"Spearman ρ = {rho:.2f}\nslope = {k:.2f}\nsame-sign days = {agree(t.ret_next, t.us_tsm_adr_ret):.1%}", transform=axes[2].transAxes, va="top", fontsize=8.5, bbox=dict(fc="white", ec="#e1e0d9"))
save(fig, "eda_05_us_vs_tw"); S["agree_overall"] = {c: round(float(agree(fresh.ret_next, fresh[k])), 4) for c, k in [("sox", "us_sox_ret"), ("sp500", "us_sp500_ret"), ("tsm_adr", "us_tsm_adr_ret"), ("own_prev", "ret1")]}
S["sox_d1_d10"] = [round(float(m5.iloc[0]), 4), round(float(m5.iloc[-1]), 4)]; S["tsmc_adr_rho"] = round(float(rho), 3)

# ---- Fig 6: sector x US direction (RQ2 heterogeneity) ----
fig, ax = plt.subplots(figsize=(9, 4))
from universe import SECTOR_EN
sec = fresh.assign(sox_up=np.where(fresh.us_sox_ret > 0, "SOX up overnight", "SOX down overnight"), sector=fresh.sector.map(SECTOR_EN))
tab = sec.groupby(["sector", "sox_up"])["y_up"].mean().unstack(); tab["gap"] = tab["SOX up overnight"] - tab["SOX down overnight"]; tab = tab.sort_values("gap", ascending=False)
xs = np.arange(len(tab)); w = 0.38
ax.bar(xs - w / 2, tab["SOX up overnight"], w, color=BLUE, label="SOX up overnight"); ax.bar(xs + w / 2, tab["SOX down overnight"], w, color=ORANGE, label="SOX down overnight")
for i, (_, r) in enumerate(tab.iterrows()): ax.text(i, max(r.iloc[0], r.iloc[1]) + 0.012, f"Δ {r.gap:+.1%}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(tab.index); ax.set_ylim(0.3, 0.7); ax.axhline(0.5, color=GRAY, ls="--", lw=1)
ax.set_ylabel("P(up next day)"); ax.set_title("Next-day up-rate by sector, conditioned on overnight SOX direction"); ax.legend(frameon=False, fontsize=8, loc="upper right")
save(fig, "eda_06_sector_us"); S["sector_gap"] = {k: round(float(v), 4) for k, v in tab["gap"].items()}

# ---- Fig 7: correlation heatmap ----
feats = ["ret1", "ret5", "ma20_ratio", "rsi14", "vol20", "volume_ratio20", "gap_open", "foreign_net_ratio", "foreign_net_ratio5", "trust_net_ratio",
         "dealer_net_ratio", "inst_net_ratio", "margin_chg", "margin_chg5", "short_chg", "taiex_ret1", "taiex_ma20_ratio", "taiex_vol_ratio20",
         "us_sp500_ret", "us_nasdaq_ret", "us_sox_ret", "us_tsm_adr_ret", "us_vix_chg", "us_usdtwd_chg", "us_10y_chg", "us_nikkei_ret", "y_up"]
corr = df[feats].corr(method="spearman")
fig, ax = plt.subplots(figsize=(11, 9.5)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=8); ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8); ax.grid(False)
for i in range(len(feats)):
    for j in range(len(feats)):
        v = corr.iloc[i, j]
        if abs(v) >= 0.3 or i == len(feats) - 1 or j == len(feats) - 1: ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.6 else "#0b0b0b")
for s, e, lab in [(0, 7, "S1 price/technical"), (7, 15, "S2 flows"), (15, 18, "S3 market"), (18, 26, "S4 US/global")]:
    ax.add_patch(plt.Rectangle((s - 0.5, s - 0.5), e - s, e - s, fill=False, ec="#0b0b0b", lw=1.2)); ax.text(s - 0.3, s - 0.7, lab, fontsize=8, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman ρ"); ax.set_title("Spearman correlation between features and next-day label (last row/col = y_up)", pad=22)
save(fig, "eda_07_correlation")
S["corr_with_label"] = {k: round(float(v), 4) for k, v in corr["y_up"].drop("y_up").sort_values(key=abs, ascending=False).head(8).items()}
S["max_abs_within_us"] = round(float(corr.loc[["us_sp500_ret", "us_nasdaq_ret", "us_sox_ret"], ["us_sp500_ret", "us_nasdaq_ret", "us_sox_ret"]].where(~np.eye(3, dtype=bool)).abs().max().max()), 3)
S["us_stale_share"] = round(float(df.us_stale.mean()), 4)
(FIG / "eda_summary.json").write_text(json.dumps(S, indent=2, ensure_ascii=False)); print(json.dumps(S, indent=1, ensure_ascii=False))
