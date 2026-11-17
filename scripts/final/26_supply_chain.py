"""Economic links among the Taiwan 50: do upstream parts suppliers lead their customers (Cohen & Frazzini 2008; Menzly & Ozbas 2010), do sectors lead
the market (Hong, Torous & Valkanov 2007), and how do 'concept' groups (AI-server chain, Apple chain) co-move?
 (a) directed supply-chain ties (public business relationships): corr(r_up,t, r_down,t+1) vs corr(r_down,t, r_up,t+1), daily and weekly (5-day
     non-overlapping), market-adjusted returns, pooled across ties with trading-day block bootstrap; per-tie values
 (b) customer momentum: Spearman IC between the customer's past-20-day return and the supplier's next-20-day market-adjusted return (monthly, across ties)
 (c) sector lead-lag: six equal-weight sector indices + TAIEX, lag-1 daily and weekly correlations; TAIEX next-week on each sector's this-week (HAC t)
 (d) concept groups: within-group correlation, response to the overnight SOX, and whether the group leads the rest of the market
Output: results/final/tw_supply_chain.json, figures/final/f17_supply_chain.png"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
from common import ROOT, RES, FIG, TEST_YEARS, load, dump, DayBootstrap
sys.path.insert(0, str(ROOT / "scripts")); from universe import TAIWAN50, SECTOR, SECTOR_EN
df = load(); S = sorted(TAIWAN50); sec = {k: SECTOR_EN[SECTOR[k]] for k in S}
wide = df.pivot(index="date", columns="stock_id", values="ret1")[S]; taiex = df.groupby("date").taiex_ret1.first().reindex(wide.index); sox = df.groupby("date").us_sox_ret.first().reindex(wide.index)
radj = wide.sub(taiex, axis=0)
DIRECTED = [("3037", "2330", "Unimicron ABF substrates → TSMC advanced packaging"), ("2330", "3711", "TSMC wafers → ASE packaging/testing"), ("2330", "2454", "TSMC foundry → MediaTek"), ("2330", "3034", "TSMC/UMC foundry → Novatek"),
            ("2330", "2379", "TSMC foundry → Realtek"), ("2330", "3661", "TSMC advanced nodes → Alchip ASICs"), ("2303", "3034", "UMC foundry → Novatek"), ("2303", "2379", "UMC foundry → Realtek"),
            ("2383", "3037", "EMC copper-clad laminate → Unimicron PCB"), ("3037", "2382", "Unimicron PCB → Quanta servers"), ("3037", "6669", "Unimicron PCB → Wiwynn servers"), ("3017", "2382", "AVC cooling → Quanta"), ("3017", "6669", "AVC cooling → Wiwynn"), ("3017", "2317", "AVC cooling → Hon Hai"),
            ("2308", "2382", "Delta power → Quanta"), ("2308", "6669", "Delta power → Wiwynn"), ("2301", "2382", "Lite-On power → Quanta"), ("2327", "2317", "Yageo passives → Hon Hai"), ("2330", "2317", "TSMC-made accelerators → Hon Hai AI servers"), ("2330", "2382", "TSMC-made accelerators → Quanta AI servers"), ("2330", "6669", "TSMC-made accelerators → Wiwynn AI servers"),
            ("2345", "6669", "Accton switches ↔ Wiwynn racks (same data-centre build)"), ("6505", "1301", "Formosa Petrochemical naphtha → Formosa Plastics"), ("6505", "1326", "naphtha → Formosa Chemicals"), ("1301", "1303", "Formosa Plastics → Nan Ya plastics"), ("1216", "2912", "Uni-President goods → PCSC stores"), ("2002", "1101", "steel ↔ cement (construction demand)")]
CONCEPTS = {"AI-server / Nvidia chain": ["2330", "2317", "2382", "3231", "6669", "3017", "2308", "2383", "3037", "2345", "3661", "2454"], "Apple chain": ["2330", "2317", "2382", "3231", "2308", "2327", "3037", "3711", "2301", "2383"],
            "Financials": [k for k in S if sec[k] == "Financials"], "Shipping": ["2603", "2609", "2615"], "Formosa group": ["1301", "1303", "1326", "6505"], "Telecom": ["2412", "3045", "4904"]}
out = {"directed_ties": [], "concepts": {}}
def ll(x, y):  # corr(x_t, y_{t+1}) with n and t
    a = pd.concat([x.shift(1), y], axis=1).dropna(); r = a.iloc[:, 0].corr(a.iloc[:, 1]); n = len(a); return float(r), float(r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-9))), int(n)
wk = radj.resample("W-FRI").sum(); wk = wk[(radj.resample("W-FRI").count() >= 3).all(1)]
rows = []
for up, dn, why in DIRECTED:
    d1 = ll(radj[up], radj[dn]); d2 = ll(radj[dn], radj[up]); w1 = ll(wk[up], wk[dn]); w2 = ll(wk[dn], wk[up])
    rows.append(dict(up=up, up_name=TAIWAN50[up], down=dn, down_name=TAIWAN50[dn], why=why, daily_up_leads=d1[0], daily_down_leads=d2[0], weekly_up_leads=w1[0], weekly_down_leads=w2[0], t_daily_up=d1[1], t_weekly_up=w1[1], contemporaneous=float(radj[up].corr(radj[dn]))))
out["directed_ties"] = rows; R = pd.DataFrame(rows)
# pooled: stack (up_t, down_t+1) pairs and bootstrap over days
def pooled(A, B, lag=1):
    X = []; Y = []; D = []
    for up, dn, _ in DIRECTED:
        a = pd.concat([A[up].shift(lag), B[dn]], axis=1).dropna(); X.append(a.iloc[:, 0].values); Y.append(a.iloc[:, 1].values); D.append(a.index.values)
    X, Y, D = np.concatenate(X), np.concatenate(Y), np.concatenate(D); bs = DayBootstrap(D, B=1000, block=5 if lag == 1 else 2)
    f = lambda x, y: float(np.corrcoef(x, y)[0, 1]); return dict(corr=f(X, Y), ci=bs.ci(f, X, Y), n_pairs=int(len(X)))
def pooled_rev(A, lag=1):
    X = []; Y = []; D = []
    for up, dn, _ in DIRECTED:
        a = pd.concat([A[dn].shift(lag), A[up]], axis=1).dropna(); X.append(a.iloc[:, 0].values); Y.append(a.iloc[:, 1].values); D.append(a.index.values)
    X, Y, D = np.concatenate(X), np.concatenate(Y), np.concatenate(D); bs = DayBootstrap(D, B=1000, block=5 if lag == 1 else 2); f = lambda x, y: float(np.corrcoef(x, y)[0, 1]); return dict(corr=f(X, Y), ci=bs.ci(f, X, Y), n_pairs=int(len(X)))
out["pooled"] = {"daily: upstream t -> downstream t+1": pooled(radj, radj), "daily: downstream t -> upstream t+1": pooled_rev(radj), "weekly: upstream w -> downstream w+1": pooled(wk, wk), "weekly: downstream w -> upstream w+1": pooled_rev(wk),
                 "contemporaneous daily (mean over ties)": float(R.contemporaneous.mean()), "n_ties": len(DIRECTED), "n_ties_daily_up_leads_t>2": int((R.t_daily_up.abs() > 2).sum()), "n_ties_weekly_up_leads_t>2": int((R.t_weekly_up.abs() > 2).sum())}
# (b) customer momentum (monthly): supplier next-20d market-adjusted return vs customer past-20d return
r20 = (1 + wide).rolling(20).apply(np.prod, raw=True) - 1; ra20 = (1 + radj).rolling(20).apply(np.prod, raw=True) - 1; ics = []; ics_rev = []
for t in range(20, len(wide) - 20, 20):
    d0 = wide.index[t]; d1 = wide.index[t + 20]; xs = []; ys = []; xr = []; yr_ = []
    for up, dn, _ in DIRECTED:
        xs.append(r20.loc[d0, dn]); ys.append(ra20.loc[d1, up]); xr.append(r20.loc[d0, up]); yr_.append(ra20.loc[d1, dn])
    if np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(xr).all() and np.isfinite(yr_).all(): ics.append(stats.spearmanr(xs, ys)[0]); ics_rev.append(stats.spearmanr(xr, yr_)[0])
ics = np.array(ics); ics_rev = np.array(ics_rev)
out["customer_momentum"] = dict(n_months=int(len(ics)), ic_customer_to_supplier=float(ics.mean()), t=float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))), ic_supplier_to_customer=float(ics_rev.mean()), t_rev=float(ics_rev.mean() / ics_rev.std(ddof=1) * np.sqrt(len(ics_rev))))
# (c) sector lead-lag
secs = sorted(set(sec.values())); SI = pd.DataFrame({s: wide[[k for k in S if sec[k] == s]].mean(1) for s in secs}); SI["TAIEX"] = taiex
LL1 = pd.DataFrame({b: {a: ll(SI[a], SI[b])[0] for a in SI} for b in SI}); SIw = SI.resample("W-FRI").sum(); SIw = SIw[(SI.resample("W-FRI").count() >= 3).all(1)]; LLw = pd.DataFrame({b: {a: ll(SIw[a], SIw[b])[0] for a in SIw} for b in SIw})
def hac_t(x, y, lags=4):
    a = pd.concat([x.shift(1), y], axis=1).dropna(); X = np.column_stack([np.ones(len(a)), a.iloc[:, 0].values]); Y = a.iloc[:, 1].values; b = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ b; n = len(Y)
    S_ = sum(((1 - k / (lags + 1)) if k else 1) * (X[k:] * e[k:, None]).T @ (X[:n - k] * e[:n - k, None]) * (1 if k == 0 else 2) for k in range(lags + 1)) / n if False else None
    XtX = X.T @ X; Om = (X * e[:, None]).T @ (X * e[:, None])
    for k in range(1, lags + 1): G = (X[k:] * e[k:, None]).T @ (X[:n - k] * e[:n - k, None]); Om += (1 - k / (lags + 1)) * (G + G.T)
    V = np.linalg.inv(XtX) @ Om @ np.linalg.inv(XtX); return float(b[1]), float(b[1] / np.sqrt(V[1, 1]))
out["sector_leadlag"] = dict(lag1_daily=LL1.round(4).to_dict(), lag1_weekly=LLw.round(4).to_dict(), sector_leads_TAIEX_weekly_HAC={s: dict(zip(("beta", "t"), hac_t(SIw[s], SIw["TAIEX"]))) for s in secs}, TAIEX_leads_sector_weekly_HAC={s: dict(zip(("beta", "t"), hac_t(SIw["TAIEX"], SIw[s]))) for s in secs}, n_weeks=int(len(SIw)))
# (d) concept groups
for name, ks in CONCEPTS.items():
    g = wide[ks].mean(1); rest = wide[[k for k in S if k not in ks]].mean(1); C = wide[ks].corr().values; iu = np.triu_indices(len(ks), 1)
    out["concepts"][name] = dict(members=[f"{k} {TAIWAN50[k]}" for k in ks], n=len(ks), within_corr=float(C[iu].mean()), corr_with_rest=float(g.corr(rest)), beta_to_overnight_sox=float(np.polyfit(sox.fillna(0), g, 1)[0]), corr_overnight_sox=float(g.corr(sox)),
                                 own_autocorr_daily_adj=float((g - taiex).autocorr(1)), group_leads_TAIEX_t1=ll(g - taiex, taiex)[0])
dump(out, "tw_supply_chain.json"); print(json.dumps({k: out[k] for k in ("pooled", "customer_momentum")}, indent=1)); print({k: {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)} for k, v in out["concepts"].items()}); print("sector→TAIEX weekly", out["sector_leadlag"]["sector_leads_TAIEX_weekly_HAC"])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
R2 = R.sort_values("weekly_up_leads"); yl = [f"{r.up_name}→{r.down_name}" for r in R2.itertuples()]; yy = np.arange(len(R2))
ax[0].barh(yy - 0.2, R2.daily_up_leads, 0.4, color="#2a78d6", label="daily: upstream t → downstream t+1"); ax[0].barh(yy + 0.2, R2.weekly_up_leads, 0.4, color="#eb6834", label="weekly: upstream w → downstream w+1")
ax[0].set_yticks(yy); ax[0].set_yticklabels(yl, fontsize=7); ax[0].axvline(0, color="#999", lw=0.8); ax[0].legend(fontsize=7, frameon=False); ax[0].set_title(f"(a) Directed supply-chain lead-lag, market-adjusted returns ({len(DIRECTED)} public ties, 2019–2026)", fontsize=9)
im = ax[1].imshow(LLw.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25); ax[1].set_xticks(range(len(LLw))); ax[1].set_xticklabels(LLw.columns, rotation=45, ha="right", fontsize=7); ax[1].set_yticks(range(len(LLw))); ax[1].set_yticklabels(LLw.index, fontsize=7); ax[1].set_title("(b) Weekly lead-lag corr(sector_w, column_{w+1})", fontsize=9); plt.colorbar(im, ax=ax[1], fraction=0.046)
names = list(CONCEPTS); ax[2].bar(np.arange(len(names)) - 0.2, [out["concepts"][n]["within_corr"] for n in names], 0.4, color="#2a78d6", label="within-group corr (daily)"); ax[2].bar(np.arange(len(names)) + 0.2, [out["concepts"][n]["corr_overnight_sox"] for n in names], 0.4, color="#eda100", label="corr with overnight SOX")
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=7); ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("(c) Concept groups: cohesion and response to the U.S. session", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "f17_supply_chain.png", dpi=150); print("figure saved")
