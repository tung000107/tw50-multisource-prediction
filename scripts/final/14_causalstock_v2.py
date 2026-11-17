"""CausalStock re-implementation v2 (our own code), following the paper (Sec. 4, App. C.4) and the design decisions documented by the
independent re-implementation of Hwang (HUFS, 2026; github.com/peanutdanny/causalstock-reproduction, docs/reproduction-report.md), which
reports ACL18 ACC 0.6374 +- 0.0042 vs the paper's 0.6342 with these deviations: lr 1e-4 (paper 1e-5), BCE weight 1.0 (paper 0.01),
KL warm-up over 10 epochs, gradient clipping 1.0, best-of-3 random restarts per seed selected on validation accuracy.
Architecture: price encoder Linear(F->4); TCD with free edge parameters U,V (L,D,D), lag-dependency MLP h_u/h_v on (u_l, u_{l-1}),
sigma = sigmoid(u-v), Gumbel-softmax sample (tau 1) in training / hard threshold at eval, learnable causal weight G_hat; FCM with shared
3-layer MLP ell (hidden 332), news branch psi on the mean of the day's 5-dim news scores (news_per_day 10), per-stock 3-layer heads zeta_i,
additive-noise Gaussian likelihood with learnable sigma_i; loss = (-ELBO + bce_w * BCE * D) / D with sparsity prior lambda_s = 1.
Datasets: ACL18 (85 tickers covering the window, 6 price features, label close_T > close_{T-1}; news scores from the re-implementation's
DNE cache, data/processed/dne_acl18.parquet), KDD17 (11 official features, labels +-1), Taiwan 50 (11 features, y_up, walk-forward; the window ends on the feature day t, label = Close(t+1) > Close(t)).
Output: results/final/causalstock_v2.json
"""
import sys, glob, os, time, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from common import RES, ROOT, TEST_YEARS, dump
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); L = 5; NPD = 10
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"; LABEL = sys.argv[2] if len(sys.argv) > 2 else "mov"   # ACL18 label: "mov" = movement% > 0 (correct); "dclose" = column-6 artifact (close_T - close_{T-1} > previous day's change)

def mlp3(i, o, h=332):
    m = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
    for x in m:
        if isinstance(x, nn.Linear): nn.init.xavier_uniform_(x.weight); nn.init.zeros_(x.bias)
    return m
class Heads(nn.Module):
    """D parallel 3-layer MLPs (zeta_i) implemented with batched matmuls."""
    def __init__(s, D, i, h=332):
        super().__init__(); s.W1 = nn.Parameter(torch.empty(D, i, h)); s.b1 = nn.Parameter(torch.zeros(D, h)); s.W2 = nn.Parameter(torch.empty(D, h, h)); s.b2 = nn.Parameter(torch.zeros(D, h)); s.W3 = nn.Parameter(torch.empty(D, h, 1)); s.b3 = nn.Parameter(torch.zeros(D, 1))
        for w in (s.W1, s.W2, s.W3):
            for d in range(D): nn.init.xavier_uniform_(w[d])
    def forward(s, x):  # x: (B, D, i) -> (B, D)
        h = torch.relu(torch.einsum("bdi,dih->bdh", x, s.W1) + s.b1); h = torch.relu(torch.einsum("bdh,dhk->bdk", h, s.W2) + s.b2); return (torch.einsum("bdh,dho->bdo", h, s.W3) + s.b3).squeeze(-1)
class CausalStockV2(nn.Module):
    def __init__(s, D, F, use_news=False, use_tcd=True, lag_dep=True, hidden=332, d_p=4, d_m=64):
        super().__init__(); s.D, s.use_news, s.use_tcd, s.lag_dep = D, use_news, use_tcd, lag_dep
        s.penc = nn.Linear(F, d_p); nn.init.xavier_uniform_(s.penc.weight); nn.init.zeros_(s.penc.bias)
        s.U = nn.Parameter(torch.empty(L, D, D)); s.V = nn.Parameter(torch.empty(L, D, D)); nn.init.xavier_uniform_(s.U); nn.init.xavier_uniform_(s.V)
        with torch.no_grad(): s.U.mul_(0.1); s.V.mul_(0.1)
        s.hu = nn.Linear(2, 1); s.hv = nn.Linear(2, 1); s.G_hat = nn.Parameter(torch.empty(L, D, D)); nn.init.xavier_uniform_(s.G_hat)
        s.ell = mlp3(d_p, hidden, hidden); zin = hidden
        if use_news: s.nenc = nn.Linear(5, d_m); s.psi = mlp3(d_m, hidden, hidden); zin = 2 * hidden; s.register_buffer("nscale", torch.tensor([10.0, 1.0, 10.0, 10.0, 10.0]))
        s.zeta = Heads(D, zin, hidden); s.log_sigma = nn.Parameter(torch.zeros(D))
    def lagdep(s, raw, m):
        prev = torch.zeros_like(raw); prev[1:] = raw[:-1]; return m(torch.stack([raw, prev], -1)).squeeze(-1)
    def graph(s):
        u, v = (s.lagdep(s.U, s.hu), s.lagdep(s.V, s.hv)) if s.lag_dep else (s.U, s.V); sigma = torch.sigmoid(u - v)
        if s.training: G = Fn.gumbel_softmax(torch.stack([u, v], -1), tau=1.0, hard=False, dim=-1)[..., 0]
        else: G = (sigma > 0.5).float()
        return G, sigma
    def forward(s, P, C=None):
        B = P.shape[0]; feat = s.ell(s.penc(P))                                                   # (B, D, L, H)
        if s.use_news: feat = torch.cat([feat, s.psi(s.nenc(C / s.nscale).mean(-2))], -1)        # C: (B, D, L, l, 5)
        if s.use_tcd: G, sigma = s.graph(); mask = G * s.G_hat
        else: G = torch.ones(L, s.D, s.D, device=P.device); sigma = torch.full((L, s.D, s.D), 0.5, device=P.device); mask = G
        agg = torch.einsum("bjlf,lji->bif", feat, mask); return torch.sigmoid(s.zeta(agg)), sigma, G

def loss_fn(f, y, sigma, G, sig_noise, beta, bce_w, lam_s=1.0, eps=1e-7):
    D = f.shape[-1]; var = (sig_noise ** 2).clamp_min(eps)[None]; z = y - f
    ll = (-0.5 * z ** 2 / var - 0.5 * torch.log(2 * math.pi * var)).sum(-1).mean()
    prior = -lam_s * (G ** 2).sum() / f.shape[0]; p = sigma.clamp(eps, 1 - eps); ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).sum() / f.shape[0]
    bce = Fn.binary_cross_entropy(f.clamp(eps, 1 - eps), y); return (-(ll + beta * (prior + ent)) + bce_w * bce * D) / D

def fit(Xtr, Ytr, Mtr, Ctr, Xva, Yva, Mva, Cva, D, F, seed, use_news=False, use_tcd=True, lag_dep=True, lr=1e-4, bce_w=1.0, max_ep=100, patience=10, warm=10):
    torch.manual_seed(seed); np.random.seed(seed); net = CausalStockV2(D, F, use_news, use_tcd, lag_dep).to(dev); opt = torch.optim.Adam(net.parameters(), lr)
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
            fm = torch.where(m.bool(), f, 0.5 * torch.ones_like(f)); ym = torch.where(m.bool(), Ytr[j], 0.5 * torch.ones_like(f))   # masked rows contribute nothing
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
    return dict(acc=accuracy_score(y, pp >= 0.5), mcc=matthews_corrcoef(y, pp >= 0.5), auroc=roc_auc_score(y, pp), val_acc=float(va), epochs=eps, restart_val_accs=[float(c[0]) for c in cands]), pp, y
T = lambda a: torch.tensor(a, device=dev)
def windows(F3, Y2, M2, C4=None):
    Tn = F3.shape[0]; X = np.stack([F3[t - L:t] for t in range(L, Tn)], 0).transpose(0, 2, 1, 3).astype(np.float32)
    C = None if C4 is None else np.stack([C4[t - L:t] for t in range(L, Tn)], 0).transpose(0, 2, 1, 3, 4).astype(np.float32)
    return X, Y2[L:].astype(np.float32), M2[L:].astype(np.float32), C
def zscore(F3, mask_train):
    mu = F3[mask_train].reshape(-1, F3.shape[-1]).mean(0); sd = F3[mask_train].reshape(-1, F3.shape[-1]).std(0); sd[sd < 1e-8] = 1; return (F3 - mu) / sd
out = json.load(open(RES / "causalstock_v2.json")) if (RES / "causalstock_v2.json").exists() else {}
# ---------------- ACL18 ----------------
if MODE in ("all", "acl18"):
    files = sorted(glob.glob(str(ROOT / "data/raw/stocknet/preprocessed/*.txt"))); px = {}
    for f in files:
        d = pd.read_csv(f, sep="\t", header=None, names=["date", "mov", "open", "high", "low", "close", "volume"]); d["date"] = pd.to_datetime(d["date"]); px[Path(f).stem] = d.sort_values("date").set_index("date")
    keep = sorted(k for k, d in px.items() if d.index.min() <= pd.Timestamp("2013-12-03") and d.index.max() >= pd.Timestamp("2016-01-01"))
    dates = sorted(set.intersection(*[set(px[k].index) for k in keep])); dates = [d for d in dates if pd.Timestamp("2013-12-01") <= d <= pd.Timestamp("2016-01-01")]
    D = len(keep); F3 = np.stack([px[k].loc[dates, ["mov", "open", "high", "low", "close", "volume"]].values for k in keep], 1).astype(np.float32)
    # StockNet preprocessed columns (verified against raw/AAPL.csv): movement% = adj_close_T/adj_close_{T-1}-1; cols 3-5 = open/high/low relative to the
    # previous adjusted close; col 6 = close_T - close_{T-1} in dollars (NOT the close); col 7 = volume. Correct label: movement% > 0.
    mov = np.stack([px[k].loc[dates, "mov"].values for k in keep], 1); dcl = np.stack([px[k].loc[dates, "close"].values for k in keep], 1); prevd = np.vstack([dcl[:1], dcl[:-1]])
    Y2 = (mov > 0).astype(float) if LABEL == "mov" else (dcl > prevd).astype(float); M2 = np.ones_like(Y2); dts = np.array(dates, dtype="datetime64[ns]")
    trm = (dts >= np.datetime64("2014-01-02")) & (dts <= np.datetime64("2015-08-02")); F3 = zscore(F3, trm); F3 = np.clip(F3, -10, 10)
    dne = pd.read_parquet(ROOT / "external/causalstock-reproduction/data/processed/dne_acl18.parquet"); dne["date"] = pd.to_datetime(dne["date"])
    cache = {(r.ticker, r.date): np.asarray(r.scores, dtype=np.float32).reshape(20, 5)[:NPD] for r in dne.itertuples()}
    C4 = np.zeros((len(dates), D, NPD, 5), dtype=np.float32)
    for ti, d in enumerate(dates):
        for si, k in enumerate(keep):
            if (k, d) in cache: C4[ti, si] = cache[(k, d)]
    X, Y, Mk, C = windows(F3, Y2, M2, C4); d = dts[L:]
    itr = (d >= np.datetime64("2014-01-02")) & (d <= np.datetime64("2015-08-02")); iva = (d >= np.datetime64("2015-08-03")) & (d <= np.datetime64("2015-09-30")); ite = (d >= np.datetime64("2015-10-01")) & (d <= np.datetime64("2016-01-01"))
    print("ACL18 stocks", D, "days train/val/test", itr.sum(), iva.sum(), ite.sum(), "news-scored stock-days", len(cache), flush=True)
    acl = {}
    for name, kw in [("CausalStock v2 w/o news", dict(use_news=False)), ("CausalStock v2 with news (cached DNE scores)", dict(use_news=True)), ("w/o TCD", dict(use_news=True, use_tcd=False))]:
        runs = []
        for seed in range(5):
            t0 = time.time(); r, _, _ = run_restarts(T(X[itr]), T(Y[itr]), T(Mk[itr]), T(C[itr]) if kw.get("use_news") else None, T(X[iva]), T(Y[iva]), T(Mk[iva]), T(C[iva]) if kw.get("use_news") else None, T(X[ite]), T(Y[ite]), T(Mk[ite]), T(C[ite]) if kw.get("use_news") else None, D, 6, seed, **kw)
            r["seconds"] = round(time.time() - t0); runs.append(r); print("ACL18", name, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        acl[name] = dict(mean={m: float(np.mean([r[m] for r in runs])) for m in ["acc", "mcc", "auroc"]}, std={m: float(np.std([r[m] for r in runs])) for m in ["acc", "mcc", "auroc"]}, runs=runs)
    acl["published"] = {"CausalStock": [63.42, 0.2172], "w/o news": [58.10, 0.1421], "w/o TCD": [51.08, 0.0102], "CMIN": [62.69, 0.2090]}; acl["hwang_reimplementation"] = {"full 10 seeds": [63.74, 0.2740], "w/o news 1 seed": [62.11, None], "w/o TCD 1 seed": [54.03, None]}
    acl["n_test_instances"] = int(Mk[ite].sum()); acl["majority_acc_test"] = float(max(Y[ite].mean(), 1 - Y[ite].mean())); acl["label"] = LABEL
    # artifact check: how predictable is each label from yesterday's own dollar change alone (logistic regression)?
    from sklearn.linear_model import LogisticRegression
    xa = X[:, :, -1, 4].reshape(-1, 1); ya = Y.reshape(-1); tri = np.repeat(itr, D); tei = np.repeat(ite, D)
    lr_ = LogisticRegression().fit(xa[tri], ya[tri]); acl["artifact_check_logreg_on_prev_dollar_change_acc"] = float(accuracy_score(ya[tei], lr_.predict(xa[tei])))
    out[f"acl18_{LABEL}"] = acl; dump(out, "causalstock_v2.json")
# ---------------- KDD17 ----------------
if MODE in ("all", "kdd17"):
    files = sorted(glob.glob(str(ROOT / "data/raw/kdd17/ourpped/*.csv"))); dates = np.genfromtxt(ROOT / "data/raw/kdd17/trading_dates.csv", dtype=str)
    A = np.stack([np.genfromtxt(f, delimiter=",") for f in files], 1); F3 = A[:, :, :11].copy(); lab = A[:, :, -2]; miss = (A[:, :, :11] < -123320).any(-1); F3[miss] = 0
    trm = dates < "2015-01-02"; F3 = np.clip(zscore(F3, trm), -10, 10); Y2 = (lab > 0).astype(float); M2 = ((lab == 1) | (lab == -1)) & ~miss
    validwin = np.stack([~miss[t - L:t].any(0) for t in range(L, len(dates))], 0); X, Y, Mk, _ = windows(F3, Y2, M2); Mk = Mk * validwin; d = dates[L:]
    itr, iva, ite = (d >= "2007-01-03") & (d < "2015-01-02"), (d >= "2015-01-02") & (d < "2016-01-04"), d >= "2016-01-04"; D = X.shape[1]
    kd = {}
    for name, kw in [("CausalStock v2 (price-only)", dict()), ("w/o TCD", dict(use_tcd=False))]:
        runs = []
        for seed in range(5):
            t0 = time.time(); r, _, _ = run_restarts(T(X[itr]), T(Y[itr]), T(Mk[itr]), None, T(X[iva]), T(Y[iva]), T(Mk[iva]), None, T(X[ite]), T(Y[ite]), T(Mk[ite]), None, D, 11, seed, **kw)
            r["seconds"] = round(time.time() - t0); runs.append(r); print("KDD17", name, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        kd[name] = dict(mean={m: float(np.mean([r[m] for r in runs])) for m in ["acc", "mcc", "auroc"]}, std={m: float(np.std([r[m] for r in runs])) for m in ["acc", "mcc", "auroc"]}, runs=runs)
    kd["published"] = {"CausalStock": [56.09, 0.1235], "DTML": [53.53, 0.0733], "Adv-ALSTM": [51.69, 0.0333]}; kd["majority_acc_test"] = float(max(Y[ite][Mk[ite] > 0].mean(), 1 - Y[ite][Mk[ite] > 0].mean())); out["kdd17"] = kd; dump(out, "causalstock_v2.json")
# ---------------- Taiwan 50 ----------------
if MODE in ("all", "tw50"):
    df = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet").sort_values(["stock_id", "date"]); df["date"] = pd.to_datetime(df["date"]); g = df.groupby("stock_id")
    f = pd.DataFrame({"c_open": df.open / df.close - 1, "c_high": df.high / df.close - 1, "c_low": df.low / df.close - 1, "n_close": df.close / g.close.shift(1) - 1, "n_adj": df.close / g.close.shift(1) - 1})
    for k in (5, 10, 15, 20, 25, 30): f[f"ma{k}"] = g.close.transform(lambda s: s.rolling(k).mean()) / df.close - 1
    f["date"] = df.date.values; f["stock_id"] = df.stock_id.values; f["y"] = df.y_up.values
    piv = {c: f.pivot(index="date", columns="stock_id", values=c) for c in f.columns if c not in ("date", "stock_id", "y")}; days = piv["c_open"].index
    F3 = np.stack([piv[c].values for c in piv], -1); miss = np.isnan(F3).any(-1); F3 = np.nan_to_num(F3); Ypv = f.pivot(index="date", columns="stock_id", values="y").shift(1); Y2 = np.nan_to_num(Ypv.values); M2 = ~np.isnan(Ypv.values)   # ALIGNMENT: our y_up on row t is Close(t+1) > Close(t), the paper's label on day T is Close(T) > Close(T-1) with inputs T-L..T-1 -> shift the label one row so the window ends ON the feature day t (inclusive)
    cols = list(Ypv.columns)
    validwin = np.stack([~miss[t - L:t].any(0) for t in range(L, len(days))], 0); tw = {}; oos = []
    for yr in TEST_YEARS:
        t0 = pd.Timestamp(f"{yr}-01-01"); v0 = t0 - pd.DateOffset(months=12); trm = (days < v0)
        Fz = np.clip(zscore(F3, trm), -10, 10); X, Y, Mk, _ = windows(Fz, Y2, M2); Mk = Mk * validwin; d = days[L - 1:-1]   # d = date of the last feature row (= our row date)
        itr, iva, ite = (d < v0), (d >= v0) & (d < t0), (d >= t0) & (d < pd.Timestamp(f"{yr + 1}-01-01")); D = X.shape[1]
        pairs = [(day, s_) for day, mrow in zip(d[ite], Mk[ite]) for s_, mk in zip(cols, mrow) if mk > 0]; PP = []
        for name, kw in [("CausalStock v2 (price-only)", dict()), ("w/o TCD", dict(use_tcd=False))]:
            for seed in range(3):
                r, pp, yy = run_restarts(T(X[itr]), T(Y[itr]), T(Mk[itr]), None, T(X[iva]), T(Y[iva]), T(Mk[iva]), None, T(X[ite]), T(Y[ite]), T(Mk[ite]), None, D, 11, seed, **kw)
                tw.setdefault(name, {}).setdefault(str(yr), []).append(r); print("TW50", yr, name, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
                if name.startswith("CausalStock"):
                    PP.append(pp)
                    if seed == 0: oos.append(pd.DataFrame({"year": yr, "y": yy, "p": pp}))
        assert len(pairs) == len(yy); oos2 = (oos2 if "oos2" in dir() else []) + [pd.DataFrame({"date": [p_[0] for p_ in pairs], "stock_id": [str(p_[1]) for p_ in pairs], "y_up": yy, "CausalStock v2 (price-only, 11)": np.mean(PP, 0)})]
    pd.concat(oos2).to_parquet(RES / "oos_causalstock_v2_tw50.parquet")
    o = pd.concat(oos); out["tw50"] = {"seed_mean_by_year": {n: {y: {m: float(np.mean([r[m] for r in v])) for m in ["acc", "mcc", "auroc"]} for y, v in d2.items()} for n, d2 in tw.items()}, "pooled_seed0": dict(auroc=float(roc_auc_score(o.y, o.p)), acc=float(accuracy_score(o.y, o.p >= 0.5)), mcc=float(matthews_corrcoef(o.y, o.p >= 0.5)))}
    dump(out, "causalstock_v2.json")
print("done")
