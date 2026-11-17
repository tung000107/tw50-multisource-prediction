"""Taiwan causal-graph + news study: CausalStock v2 (our own code, 14_causalstock_v2.py) on the Taiwan 50, walk-forward 2023-2026,
3 seeds x best-of-3 restarts, validation = last 12 months of each training window. Variants:
  learned graph, price only             TCD exactly as in the paper (re-run here so the learned graph can be saved)
  learned graph + news (5-dim)          per stock-day: fold-trained TF-IDF text score, market text score, log count, has-news, 5-day count (S6)
  learned graph + news (score only)     only the text score
  prior graph (fixed) [+ news]          adjacency = sector membership + public business ties (20_tw_structure.py), symmetric + self-loops
  lasso graph (fixed, fold-specific)    parents from the training-window Lasso (20_tw_structure.py)
Learned graphs (eval-mode hard threshold) are saved and compared with the prior / lasso graphs.
Output: results/final/causal_news_tw.json, results/final/oos_causal_news_tw.parquet
"""
import sys, json, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
_argv = list(sys.argv); sys.argv = [sys.argv[0], "tw50"]; exec(open(Path(__file__).parent / "14_causalstock_v2.py").read().split("out = json.load(")[0]); sys.argv = _argv
from common import ROOT, RES, TEST_YEARS, dump, DayBootstrap, auc_stat
sys.path.insert(0, str(ROOT / "scripts")); from universe import TAIWAN50
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
def pick_dev():
    if not torch.cuda.is_available(): return torch.device("cpu")
    try:
        free = [int(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()]
        i = int(np.argmax(free)); return torch.device(f"cuda:{i}") if free[i] > 4000 else torch.device("cpu")
    except Exception: return torch.device("cpu")
dev = pick_dev(); print("device", dev, flush=True)
MAKE = None
def fit(Xtr, Ytr, Mtr, Ctr, Xva, Yva, Mva, Cva, D, F, seed, use_news=False, use_tcd=True, lag_dep=True, lr=1e-4, bce_w=1.0, max_ep=100, patience=10, warm=10):
    torch.manual_seed(seed); np.random.seed(seed); net = MAKE(D, F, use_news, use_tcd, lag_dep).to(dev); opt = torch.optim.Adam(net.parameters(), lr)
    def ev(X, Y, M, C):
        net.eval(); out = []
        with torch.no_grad():
            for b in range(0, len(X), 64): out.append(net(X[b:b + 64], None if C is None else C[b:b + 64])[0])
        p = torch.cat(out); m = M.bool(); return p, accuracy_score(Y[m].cpu().numpy(), (p[m] >= 0.5).cpu().numpy())
    best, state, bad, ep = -1, None, 0, 0
    for ep in range(max_ep):
        net.train(); beta = min(1.0, ep / warm) if warm > 0 else 1.0; perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(perm), 32):
            j = perm[b:b + 32]; f, sigma, G = net(Xtr[j], None if Ctr is None else Ctr[j]); m = Mtr[j]
            fm = torch.where(m.bool(), f, 0.5 * torch.ones_like(f)); ym = torch.where(m.bool(), Ytr[j], 0.5 * torch.ones_like(f))
            lo = loss_fn(fm, ym, sigma, G, Fn.softplus(net.log_sigma), beta, bce_w); opt.zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        _, va = ev(Xva, Yva, Mva, Cva)
        if va > best + 1e-5: best, bad, state = va, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else: bad += 1
        if bad >= patience: break
    net.load_state_dict(state); return net, best, ep + 1, ev
def run_restarts(Xtr, Ytr, Mtr, Ctr, Xva, Yva, Mva, Cva, Xte, Yte, Mte, Cte, D, F, seed, **kw):
    cands = []
    for r in range(3):
        net, va, eps, ev = fit(Xtr, Ytr, Mtr, Ctr, Xva, Yva, Mva, Cva, D, F, seed * 100 + r, **kw); cands.append((va, net, eps, ev))
    va, net, eps, ev = max(cands, key=lambda c: c[0]); p, acc = ev(Xte, Yte, Mte, Cte); m = Mte.bool(); y = Yte[m].cpu().numpy(); pp = p[m].cpu().numpy()
    return dict(acc=accuracy_score(y, pp >= 0.5), mcc=matthews_corrcoef(y, pp >= 0.5), auroc=roc_auc_score(y, pp), val_acc=float(va), epochs=eps, restart_val_accs=[float(c[0]) for c in cands]), pp, y, net
class FixedGraph(CausalStockV2):
    def __init__(s, D, F, mask, *a, **kw):
        super().__init__(D, F, *a, **kw); s.register_buffer("fixed", torch.tensor(mask, dtype=torch.float32))
    def graph(s): return s.fixed, s.fixed.clamp(1e-4, 1 - 1e-4)
# ---------------- data ----------------
df = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet").sort_values(["stock_id", "date"]); df["date"] = pd.to_datetime(df["date"]); g = df.groupby("stock_id")
f = pd.DataFrame({"c_open": df.open / df.close - 1, "c_high": df.high / df.close - 1, "c_low": df.low / df.close - 1, "n_close": df.close / g.close.shift(1) - 1, "n_adj": df.close / g.close.shift(1) - 1})
for k in (5, 10, 15, 20, 25, 30): f[f"ma{k}"] = g.close.transform(lambda s: s.rolling(k).mean()) / df.close - 1
f["date"] = df.date.values; f["stock_id"] = df.stock_id.values; f["y"] = df.y_up.values
piv = {c: f.pivot(index="date", columns="stock_id", values=c) for c in f.columns if c not in ("date", "stock_id", "y")}; days = piv["c_open"].index; S = list(piv["c_open"].columns); D = len(S); sidx = {s: i for i, s in enumerate(S)}
F3 = np.stack([piv[c].values for c in piv], -1); miss = np.isnan(F3).any(-1); F3 = np.nan_to_num(F3); Ypv = f.pivot(index="date", columns="stock_id", values="y").shift(1); Y2 = np.nan_to_num(Ypv.values); M2 = ~np.isnan(Ypv.values)   # label shifted one row: the window ends on the feature day t (see 14_causalstock_v2.py)
validwin = np.stack([~miss[t - L:t].any(0) for t in range(L, len(days))], 0)
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); nf["date"] = pd.to_datetime(nf["date"])
def news_tensor(yr, score_only=False):
    cols = {"ts": f"text_score_{yr}", "ms": f"market_text_score_{yr}", "ln": "log_news", "hn": "has_news", "n5": "n_news_5d"}
    P = {k: nf.pivot(index="date", columns="stock_id", values=v).reindex(index=days, columns=S).fillna(0.5 if k in ("ts", "ms") else 0).values for k, v in cols.items()}
    C = np.stack([(P["ts"] - 0.5) * 10, (P["ms"] - 0.5) * 10, P["ln"], P["hn"], P["n5"] / 5], -1).astype(np.float32)   # (T, D, 5)
    if score_only: C[..., 1:] = 0
    return C[:, :, None, :]                                                                                             # (T, D, 1, 5)
graphs = json.load(open(ROOT / "data/processed/tw_graphs.json")); prior_A = np.eye(D, dtype=np.float32)
for e in graphs["prior"]:
    if e["a"] in sidx and e["b"] in sidx: prior_A[sidx[e["a"]], sidx[e["b"]]] = prior_A[sidx[e["b"]], sidx[e["a"]]] = 1
pset = {frozenset((e["a"], e["b"])) for e in graphs["prior"]}
def lasso_mask(yr):
    A = np.eye(D, dtype=np.float32)
    for child, lst in graphs["lasso"][str(yr)].items():
        for par, _ in lst:
            if par in sidx: A[sidx[par], sidx[child]] = 1                                                          # mask[l, source, target]
    return np.repeat(A[None], L, 0)
def graph_summary(net):
    net.eval()
    with torch.no_grad(): G, sigma = net.graph()
    A = (G.any(0) if G.dtype == torch.bool else (G > 0.5).any(0)).cpu().numpy(); np.fill_diagonal(A, False); E = {(S[i], S[j]) for i, j in zip(*np.where(A))}
    inp = sum(frozenset(e) in pset for e in E); dens = float(len(E) / (D * (D - 1)))
    sg = sigma.max(0).values.cpu().numpy(); np.fill_diagonal(sg, 0); top = np.dstack(np.unravel_index(np.argsort(-sg.ravel())[:15], sg.shape))[0]
    return dict(n_edges=len(E), density=dens, share_in_prior=float(inp / max(len(E), 1)), lift_vs_prior=float(inp / max(len(E), 1) / (len(pset) / (D * (D - 1) / 2))), edges=sorted(E),
                top_sigma=[dict(src=S[i], dst=S[j], sigma=float(sg[i, j])) for i, j in top])
VARIANTS = [("learned | price", dict(graph="learned", news=None)), ("learned | +news", dict(graph="learned", news="full")), ("learned | +news score only", dict(graph="learned", news="score")),
            ("prior fixed | price", dict(graph="prior", news=None)), ("prior fixed | +news", dict(graph="prior", news="full")), ("lasso fixed | price", dict(graph="lasso", news=None))]
out = {"variants": {}, "graphs": {}, "n_seeds": NSEED}; oos = []
for yr in TEST_YEARS:
    t0 = pd.Timestamp(f"{yr}-01-01"); v0 = t0 - pd.DateOffset(months=12); trm = (days < v0)
    Fz = np.clip(zscore(F3, trm), -10, 10); d = days[L - 1:-1]; itr, iva, ite = (d < v0), (d >= v0) & (d < t0), (d >= t0) & (d < pd.Timestamp(f"{yr + 1}-01-01"))
    Cs = {None: None, "full": news_tensor(yr), "score": news_tensor(yr, True)}
    for name, cfg in VARIANTS:
        X, Y, Mk, C = windows(Fz, Y2, M2, Cs[cfg["news"]]); Mk = Mk * validwin; use_news = cfg["news"] is not None
        mask = None if cfg["graph"] == "learned" else (np.repeat(prior_A[None], L, 0) if cfg["graph"] == "prior" else lasso_mask(yr))
        def MAKE(D_, F_, un, ut, ld, mask=mask):
            net = CausalStockV2(D_, F_, un, ut, ld) if mask is None else FixedGraph(D_, F_, mask, un, ut, ld)
            if un: net.nscale.fill_(1.0)
            return net
        for seed in range(NSEED):
            t1 = time.time(); Ct = (lambda i: T(C[i])) if use_news else (lambda i: None)
            r, pp, yy, net = run_restarts(T(X[itr]), T(Y[itr]), T(Mk[itr]), Ct(itr), T(X[iva]), T(Y[iva]), T(Mk[iva]), Ct(iva), T(X[ite]), T(Y[ite]), T(Mk[ite]), Ct(ite), D, 11, seed, use_news=use_news)
            r["seconds"] = round(time.time() - t1, 1); out["variants"].setdefault(name, {}).setdefault(str(yr), []).append(r)
            mte = Mk[ite].astype(bool); dates = np.repeat(d[ite].values, D).reshape(-1, D)[mte]; stocks = np.tile(np.array(S), mte.shape[0]).reshape(-1, D)[mte]
            oos.append(pd.DataFrame({"variant": name, "seed": seed, "year": yr, "date": dates, "stock_id": stocks, "y": yy, "p": pp}))
            if cfg["graph"] == "learned": out["graphs"].setdefault(name, {}).setdefault(str(yr), []).append(graph_summary(net))
            print("TW50", yr, name, seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "restart_val_accs"}, flush=True)
            del net; torch.cuda.empty_cache() if dev.type == "cuda" else None
    pd.concat(oos).to_parquet(RES / "oos_causal_news_tw.parquet"); dump(out, "causal_news_tw.json")
# ---------------- pooled metrics, seed-mean probabilities, paired bootstrap vs. learned|price ----------------
o = pd.concat(oos); sm = o.groupby(["variant", "date", "stock_id"]).agg(y=("y", "first"), p=("p", "mean")).reset_index()
base = sm[sm.variant == "learned | price"].set_index(["date", "stock_id"]); pooled = {}
for name, _ in VARIANTS:
    v = sm[sm.variant == name].set_index(["date", "stock_id"]).reindex(base.index); y = base.y.values; bs = DayBootstrap(base.index.get_level_values(0).values, B=1000)
    pooled[name] = dict(auroc=float(roc_auc_score(y, v.p.values)), acc=float(accuracy_score(y, v.p.values >= 0.5)), mcc=float(matthews_corrcoef(y, v.p.values >= 0.5)), ci=bs.ci(auc_stat, y, v.p.values),
                        diff_vs_learned_price=None if name == "learned | price" else bs.ci_diff(auc_stat, y, base.p.values, v.p.values),
                        by_year={str(yr): {m: float(np.mean([r[m] for r in out["variants"][name][str(yr)]])) for m in ("acc", "mcc", "auroc")} for yr in TEST_YEARS})
out["pooled_seed_mean"] = pooled
# effect of news with the graph held fixed (prior | +news vs prior | price): the comparison against learned | price changes graph AND input at once
def _pair(a_, b_):
    A_ = sm[sm.variant == a_].set_index(["date", "stock_id"]); B_ = sm[sm.variant == b_].set_index(["date", "stock_id"]).reindex(A_.index); bs_ = DayBootstrap(A_.index.get_level_values(0).values, B=1000)
    return dict(diff=float(roc_auc_score(A_.y.values, B_.p.values) - roc_auc_score(A_.y.values, A_.p.values)), ci=bs_.ci_diff(auc_stat, A_.y.values, A_.p.values, B_.p.values))
out["news_effect_same_graph"] = {"learned graph": _pair("learned | price", "learned | +news"), "prior graph": _pair("prior fixed | price", "prior fixed | +news")}
# graph stability across seeds / folds
gs = out["graphs"].get("learned | price", {})
def jac(a, b): A, B = set(map(tuple, a["edges"])), set(map(tuple, b["edges"])); return len(A & B) / max(len(A | B), 1)
out["graph_stability"] = {yr: dict(mean_edges=float(np.mean([g["n_edges"] for g in L_])), jaccard_seeds=float(np.mean([jac(L_[a], L_[b]) for a in range(len(L_)) for b in range(a + 1, len(L_))] or [np.nan])), mean_lift_vs_prior=float(np.mean([g["lift_vs_prior"] for g in L_]))) for yr, L_ in gs.items()}
lasso_sets = {yr: {(p, c) for c, lst in graphs["lasso"][yr].items() for p, _ in lst if p != "TAIEX"} for yr in graphs["lasso"]}
out["learned_vs_lasso"] = {yr: float(np.mean([len(set(map(tuple, g["edges"])) & lasso_sets[yr]) / max(len(lasso_sets[yr]), 1) for g in L_])) for yr, L_ in gs.items()}
dump(out, "causal_news_tw.json"); print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "by_year"} for k, v in pooled.items()}, indent=1)); print("done")
