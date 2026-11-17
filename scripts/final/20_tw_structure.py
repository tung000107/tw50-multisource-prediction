"""Step 1 of the Taiwan causal-graph study: what structure is there among the Taiwan 50, and who leads whom?
(a) contemporaneous correlation of daily returns on the first training window (< 2023); clustering vs. sector tags
(b) lead-lag: corr(r_i,t , r_j,t+1) for all ordered pairs on market-adjusted returns (ret1 - taiex_ret1), Holm-corrected over 2500 pairs;
    leaders = out-degree, followers = in-degree; within- vs cross-sector share
(c) per-fold data-driven sparse graph: Lasso of r_j,t+1 on all 50 r_i,t (+ TAIEX), standardised, alpha by time-series CV inside the training
    window only; parents per stock per test year; stability across folds (Jaccard)
(d) prior graph from public business relationships (sector membership + supply-chain / group ties), hand-coded with the reason for each edge
(e) overlap of the data-driven edges with the prior graph vs. chance
Output: results/final/tw_structure.json, data/processed/tw_graphs.json, figures/final/f15_tw_structure.png
"""
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from common import ROOT, RES, FIG, TEST_YEARS, load, dump
sys.path.insert(0, str(ROOT / "scripts")); from universe import TAIWAN50, SECTOR, SECTOR_EN
df = load(); S = sorted(TAIWAN50); sec = {k: SECTOR_EN[SECTOR[k]] for k in S}
wide = df.pivot(index="date", columns="stock_id", values="ret1")[S]; taiex = df.groupby("date").taiex_ret1.first().reindex(wide.index)
radj = wide.sub(taiex, axis=0)
out = {"n_stocks": len(S), "sectors": sec}
# ---------- (a) contemporaneous structure (first training window) ----------
w0 = wide[wide.index < "2023-01-01"].dropna(); C0 = w0.corr()
secs = np.array([sec[k] for k in S]); same = secs[:, None] == secs[None, :]; off = ~np.eye(len(S), dtype=bool)
out["contemporaneous"] = dict(n_days=int(len(w0)), mean_corr_within_sector=float(C0.values[same & off].mean()), mean_corr_cross_sector=float(C0.values[~same].mean()),
    mean_corr_all=float(C0.values[off].mean()), by_sector={s: float(C0.values[(secs[:, None] == s) & (secs[None, :] == s) & off].mean()) for s in sorted(set(secs))})
Z = linkage(C0.values[np.triu_indices(len(S), 1)] * -1 + 1, "average"); cl = fcluster(Z, 6, "maxclust")
out["contemporaneous"]["ari_clusters_vs_sectors"] = float(adjusted_rand_score(secs, cl)); out["contemporaneous"]["clusters"] = {int(c): [k for k, cc in zip(S, cl) if cc == c] for c in sorted(set(cl))}
# ---------- (b) lead-lag on market-adjusted returns ----------
def leadlag(R):
    R = R.dropna(); X = R.values[:-1]; Y = R.values[1:]; n = len(X)
    Xs = (X - X.mean(0)) / X.std(0); Ys = (Y - Y.mean(0)) / Y.std(0); C = Xs.T @ Ys / n            # C[i, j] = corr(r_i,t, r_j,t+1)
    t = C * np.sqrt((n - 2) / np.clip(1 - C ** 2, 1e-9, None)); p = 2 * stats.t.sf(np.abs(t), n - 2); return C, p, n
C1, p1, n1 = leadlag(radj[radj.index < "2023-01-01"])
pv = p1.ravel(); order = np.argsort(pv); m = len(pv); holm = np.minimum.accumulate((np.maximum.accumulate(pv[order] * (m - np.arange(m))))[::-1])[::-1]; sig = np.zeros(m, bool); sig[order] = holm < 0.05; sig = sig.reshape(p1.shape)
sig_off = sig & off; edges = [(S[i], S[j], float(C1[i, j])) for i, j in zip(*np.where(sig_off))]; edges.sort(key=lambda e: -abs(e[2]))
outdeg = pd.Series(sig_off.sum(1), index=S); indeg = pd.Series(sig_off.sum(0), index=S)
out["leadlag"] = dict(n_days=int(n1), n_pairs_tested=int(off.sum()), n_significant_holm05=int(sig_off.sum()), n_positive=int((sig_off & (C1 > 0)).sum()),
    share_within_sector=float((sig_off & same).sum() / max(sig_off.sum(), 1)), prior_share_within_sector=float((same & off).sum() / off.sum()),
    self_autocorr_mean=float(np.mean(np.diag(C1))), self_autocorr_significant=int(np.diag(sig).sum()),
    abs_corr_quantiles={q: float(np.quantile(np.abs(C1[off]), q)) for q in (0.5, 0.9, 0.99)},
    top_edges=[dict(leader=a, leader_name=TAIWAN50[a], follower=b, follower_name=TAIWAN50[b], corr=round(c, 4)) for a, b, c in edges[:25]],
    leaders=[dict(stock=k, name=TAIWAN50[k], sector=sec[k], out_degree=int(v)) for k, v in outdeg.sort_values(ascending=False).head(10).items()],
    followers=[dict(stock=k, name=TAIWAN50[k], sector=sec[k], in_degree=int(v)) for k, v in indeg.sort_values(ascending=False).head(10).items()])
# raw returns for comparison (market autocorrelation dominates)
Cr, pr, _ = leadlag(wide[wide.index < "2023-01-01"]); out["leadlag"]["raw_returns_n_significant_bonf"] = int(((pr < 0.05 / m) & off).sum())
# lead-lag by sector pair (mean corr)
sl = sorted(set(secs)); out["leadlag"]["sector_pair_mean_corr"] = {a: {b: float(C1[np.ix_(secs == a, secs == b)][off[np.ix_(secs == a, secs == b)]].mean()) for b in sl} for a in sl}
# ---------- (c) per-fold Lasso graph ----------
graphs = {"stocks": S, "lasso": {}, "leadlag_sig": {}}; r2 = {}
for yr in TEST_YEARS:
    t0 = pd.Timestamp(f"{yr}-01-01"); R = radj[radj.index < t0 - pd.Timedelta(days=1)].dropna(); X = R.values[:-1]; Y = R.values[1:]
    Xf = np.column_stack([X, taiex.reindex(R.index).values[:-1]]); mu, sd = Xf.mean(0), Xf.std(0); Xs = (Xf - mu) / sd
    Rt = radj[(radj.index >= t0) & (radj.index < pd.Timestamp(f"{yr + 1}-01-01"))].dropna(); Xt = np.column_stack([Rt.values[:-1], taiex.reindex(Rt.index).values[:-1]]); Xt = (Xt - mu) / sd; Yt = Rt.values[1:]
    par = {}; r2y = []
    for j, s in enumerate(S):
        y = Y[:, j]; ys = y.std(); las = LassoCV(cv=TimeSeriesSplit(5), alphas=30, max_iter=5000, n_jobs=8).fit(Xs, y / ys)
        coef = las.coef_; nz = [(S[i] if i < len(S) else "TAIEX", float(coef[i])) for i in np.where(np.abs(coef) > 1e-6)[0] if i != j]
        par[s] = sorted(nz, key=lambda e: -abs(e[1])); yhat = las.predict(Xt) * ys; r2y.append(1 - ((Yt[:, j] - yhat) ** 2).sum() / ((Yt[:, j] - Yt[:, j].mean()) ** 2).sum())
    graphs["lasso"][str(yr)] = par; r2[str(yr)] = float(np.mean(r2y))
    Cy, py, ny = leadlag(R); pvy = py.ravel(); o = np.argsort(pvy); h = np.minimum.accumulate((np.maximum.accumulate(pvy[o] * (m - np.arange(m))))[::-1])[::-1]; sg = np.zeros(m, bool); sg[o] = h < 0.05; sg = sg.reshape(py.shape) & off
    graphs["leadlag_sig"][str(yr)] = [(S[i], S[j], float(Cy[i, j])) for i, j in zip(*np.where(sg))]
    print("fold", yr, "lasso edges", sum(len(v) for v in par.values()), "oos R2", round(r2[str(yr)], 4), "leadlag sig", int(sg.sum()), flush=True)
def eset(par): return {(i, j) for j, lst in par.items() for i, _ in lst if i != "TAIEX"}
ls = {y: eset(graphs["lasso"][y]) for y in graphs["lasso"]}; yrs = sorted(ls)
out["lasso"] = dict(n_edges={y: len(ls[y]) for y in yrs}, oos_r2_mean=r2, jaccard_consecutive={f"{a}->{b}": float(len(ls[a] & ls[b]) / max(len(ls[a] | ls[b]), 1)) for a, b in zip(yrs[:-1], yrs[1:])},
    parents_per_stock_mean={y: float(np.mean([len(v) for v in graphs["lasso"][y].values()])) for y in yrs},
    taiex_parent_share={y: float(np.mean(["TAIEX" in [i for i, _ in v] for v in graphs["lasso"][y].values()])) for y in yrs},
    top_parents_2023=[dict(stock=k, name=TAIWAN50[k], n_children=int(v)) for k, v in pd.Series([i for j, lst in graphs["lasso"]["2023"].items() for i, _ in lst if i != "TAIEX"]).value_counts().head(10).items()])
# ---------- (d) prior graph (public business relationships) ----------
PRIOR = []  # (a, b, reason) undirected
for s in sl:
    ks = [k for k in S if sec[k] == s]
    for a, b in itertools.combinations(ks, 2): PRIOR.append((a, b, f"same sector: {s}"))
TIES = [("2330", "3711", "TSMC -> ASE: back-end packaging/testing of TSMC wafers (OSAT)"), ("2330", "2454", "MediaTek is a leading fabless customer of TSMC"),
        ("2330", "3034", "Novatek: fabless display-driver IC customer of TSMC/UMC"), ("2330", "2379", "Realtek: fabless customer of TSMC"), ("2330", "3661", "Alchip: ASIC design house on TSMC advanced nodes"),
        ("2330", "3037", "Unimicron: ABF substrates for advanced packaging (CoWoS chain)"), ("2330", "2317", "Hon Hai assembles AI servers using TSMC-made GPUs; both in the Nvidia chain"),
        ("2317", "2382", "Hon Hai / Quanta: AI-server & notebook EMS/ODM peers"), ("2317", "3231", "Hon Hai / Wistron: AI-server EMS/ODM peers"), ("2382", "3231", "Quanta / Wistron: server ODM peers"),
        ("3231", "6669", "Wiwynn is a Wistron subsidiary (cloud/AI server ODM)"), ("2382", "6669", "Quanta / Wiwynn: hyperscaler server ODM peers"),
        ("3017", "2382", "AVC: liquid/air cooling for AI servers built by Quanta et al."), ("3017", "6669", "AVC cooling modules in Wiwynn AI servers"), ("3017", "2317", "AVC cooling modules in Hon Hai AI servers"),
        ("2308", "2382", "Delta: power supplies for AI servers"), ("2308", "6669", "Delta: power supplies for AI servers"), ("2301", "2382", "Lite-On: server power supplies"),
        ("2383", "3037", "EMC copper-clad laminate -> Unimicron PCB"), ("2383", "6669", "EMC CCL for AI-server boards"), ("3037", "2382", "Unimicron PCB/substrate for servers"),
        ("2345", "6669", "Accton: data-centre switches alongside AI servers"), ("2345", "2382", "Accton switches / Quanta servers: same data-centre capex"),
        ("2327", "2317", "Yageo passives into Hon Hai products"), ("2357", "2382", "Asus / Quanta: PC supply chain"), ("2357", "2317", "Asus / Hon Hai: PC supply chain"), ("2395", "2357", "Advantech / Asus: PC & industrial PC"),
        ("2303", "3034", "UMC: foundry for Novatek"), ("2303", "2379", "UMC: foundry for Realtek"), ("2454", "2379", "MediaTek / Realtek: IC-design peers"), ("2454", "3034", "MediaTek / Novatek: IC-design peers"),
        ("1216", "2912", "PCSC (7-Eleven) is a Uni-President subsidiary"), ("1301", "6505", "Formosa Petrochemical: Formosa Plastics Group"), ("1303", "6505", "Formosa Petrochemical: Formosa Plastics Group"), ("1326", "6505", "Formosa Petrochemical: Formosa Plastics Group"),
        ("2002", "1101", "China Steel / Taiwan Cement: construction & commodity cycle"), ("5871", "2881", "Chailease leasing / Fubon: credit cycle"), ("2603", "2609", "Evergreen / Yang Ming: container shipping"), ("2603", "2615", "Evergreen / Wan Hai: container shipping"),
        ("2412", "3045", "Chunghwa / Taiwan Mobile: telecom"), ("2330", "2303", "TSMC / UMC: foundry peers"), ("2330", "2408", "Nanya DRAM: semiconductor cycle")]
seen = {frozenset((a, b)) for a, b, _ in PRIOR}
for a, b, r in TIES:
    if frozenset((a, b)) not in seen: PRIOR.append((a, b, r)); seen.add(frozenset((a, b)))
    else: PRIOR[[i for i, e in enumerate(PRIOR) if frozenset((e[0], e[1])) == frozenset((a, b))][0]] = (a, b, PRIOR[[i for i, e in enumerate(PRIOR) if frozenset((e[0], e[1])) == frozenset((a, b))][0]][2] + "; " + r)
graphs["prior"] = [dict(a=a, b=b, reason=r) for a, b, r in PRIOR]; pset = {frozenset((a, b)) for a, b, _ in PRIOR}
out["prior"] = dict(n_undirected_edges=len(PRIOR), n_supply_chain_ties=len(TIES), density=float(len(PRIOR) / (len(S) * (len(S) - 1) / 2)), degree_top=[dict(stock=k, name=TAIWAN50[k], degree=int(v)) for k, v in pd.Series([x for a, b, _ in PRIOR for x in (a, b)]).value_counts().head(8).items()])
# ---------- (e) overlap ----------
def overlap(E):  # E: set of directed (i, j)
    if not E: return dict(n=0)
    inp = sum(frozenset(e) in pset for e in E); return dict(n=len(E), in_prior=int(inp), share_in_prior=float(inp / len(E)), chance=out["prior"]["density"], lift=float(inp / len(E) / out["prior"]["density"]))
out["overlap"] = {"leadlag_sig_first_window": overlap({(a, b) for a, b, _ in edges}), **{f"lasso_{y}": overlap(ls[y]) for y in yrs}, **{f"leadlag_{y}": overlap({(a, b) for a, b, _ in graphs["leadlag_sig"][y]}) for y in yrs}}
dump(out, "tw_structure.json"); json.dump(graphs, open(ROOT / "data/processed/tw_graphs.json", "w"))
print(json.dumps({k: out[k] for k in ("contemporaneous", "leadlag", "lasso", "prior", "overlap")}, ensure_ascii=False, indent=1, default=str)[:6000])
# ---------- figure ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(19, 6)); order_idx = np.argsort(secs, kind="stable"); So = [S[i] for i in order_idx]
im = ax[0].imshow(C0.values[np.ix_(order_idx, order_idx)], cmap="RdBu_r", vmin=-0.6, vmax=0.6); ax[0].set_title(f"(a) Contemporaneous corr of daily returns (< 2023, n={len(w0)})\nordered by sector; within {out['contemporaneous']['mean_corr_within_sector']:.2f} vs cross {out['contemporaneous']['mean_corr_cross_sector']:.2f}", fontsize=10)
bounds = np.cumsum([0] + [int((secs == s).sum()) for s in sl]); [ax[0].axhline(b - 0.5, color="k", lw=0.6) or ax[0].axvline(b - 0.5, color="k", lw=0.6) for b in bounds[1:-1]]
ax[0].set_xticks((bounds[:-1] + bounds[1:]) / 2); ax[0].set_xticklabels(sl, rotation=45, ha="right", fontsize=8); ax[0].set_yticks((bounds[:-1] + bounds[1:]) / 2); ax[0].set_yticklabels(sl, fontsize=8); plt.colorbar(im, ax=ax[0], fraction=0.046)
im2 = ax[1].imshow(C1[np.ix_(order_idx, order_idx)], cmap="RdBu_r", vmin=-0.15, vmax=0.15); ax[1].set_title(f"(b) Lead-lag corr(r_i,t, r_j,t+1), market-adjusted\n{out['leadlag']['n_significant_holm05']} of 2450 ordered pairs significant (Holm 5%)", fontsize=10)
[ax[1].axhline(b - 0.5, color="k", lw=0.6) or ax[1].axvline(b - 0.5, color="k", lw=0.6) for b in bounds[1:-1]]; ax[1].set_xticks((bounds[:-1] + bounds[1:]) / 2); ax[1].set_xticklabels(sl, rotation=45, ha="right", fontsize=8); ax[1].set_yticks((bounds[:-1] + bounds[1:]) / 2); ax[1].set_yticklabels(sl, fontsize=8)
ax[1].set_ylabel("leader i (day t)"); ax[1].set_xlabel("follower j (day t+1)"); plt.colorbar(im2, ax=ax[1], fraction=0.046)
ld = outdeg.sort_values(ascending=False).head(12); ax[2].barh([f"{k} {sec[k][:6]}" for k in ld.index][::-1], ld.values[::-1], color="#4472c4"); ax[2].set_title("(c) Leaders: number of stocks significantly led (out-degree)", fontsize=10); ax[2].set_xlabel("out-degree (Holm 5%)")
plt.tight_layout(); plt.savefig(FIG / "f15_tw_structure.png", dpi=150); print("figure saved")
