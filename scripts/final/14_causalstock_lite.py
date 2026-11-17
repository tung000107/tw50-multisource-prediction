"""Re-implementation of CausalStock *without news* (Li et al., NeurIPS 2024), following Sec. 4.1/4.3/4.4 and Appendix C.4:
  price encoder: 1-layer MLP, hidden 4, on the 11 temporal price features of each stock and lag (L = 5)
  lag-dependent temporal causal discovery: q(G_l) = Bernoulli(sigmoid(f(u_j, v_i, lag_l))) with 1-layer MLPs h_v, h_u on node
     embeddings; Gumbel-sigmoid relaxation while training, expectation at test; KL to a sparse Bernoulli prior (lambda_s = 1)
  functional causal model: z_i = zeta_i( psi( sum_l sum_j G_l[j,i] * ell(e_{j,l}) ) ) with 3-layer MLPs (hidden 332); zeta_i via stock embedding
  loss = BCE + 0.01 * KL ; Adam ; batch = 32 trading days ; early stopping on validation ACC+MCC ; 5 seeds
No official code exists; this is our reading of the paper (stated on the slide). Two datasets:
  kdd17   : official Adv-ALSTM preprocessing (11 features, labels +1/-1, 0 = excluded), split 2007-01-03 / 2015-01-02 / 2016-01-04 (Table 3)
  tw50    : the same 11 features computed from our Taiwan 50 prices, label y_up, our walk-forward (4 folds, inner validation), 3 seeds
Ablation 'w/o TCD': graph fixed to self-loops only.
Output: results/final/causalstock_lite.json
"""
import sys, glob, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from common import RES, ROOT, TEST_YEARS, dump
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
L = 5; FEAT = 11

class CausalStockLite(nn.Module):
    def __init__(s, D, F=FEAT, L=L, h_enc=4, h=332, d_node=16, self_only=False):
        super().__init__(); s.D, s.L, s.self_only = D, L, self_only
        s.enc = nn.Sequential(nn.Linear(F, h_enc), nn.ReLU())
        s.U = nn.Parameter(torch.randn(D, d_node) * 0.1); s.V = nn.Parameter(torch.randn(D, d_node) * 0.1); s.lag = nn.Parameter(torch.randn(L, d_node) * 0.1)
        s.hu = nn.Linear(d_node, d_node); s.hv = nn.Linear(d_node, d_node); s.g = nn.Linear(3 * d_node, 1)
        s.ell = nn.Sequential(nn.Linear(h_enc, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        s.psi = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        s.stock = nn.Parameter(torch.randn(D, d_node) * 0.1); s.zeta = nn.Sequential(nn.Linear(h + d_node, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
        for m in s.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def graph_logits(s):
        u = torch.tanh(s.hu(s.U)); v = torch.tanh(s.hv(s.V))                       # (D, d)
        uj = u[None, :, None, :].expand(s.L, s.D, s.D, -1); vi = v[None, None, :, :].expand(s.L, s.D, s.D, -1); lg = s.lag[:, None, None, :].expand(s.L, s.D, s.D, -1)
        return s.g(torch.cat([uj, vi, lg], -1)).squeeze(-1)                            # (L, D_from, D_to)
    def forward(s, x, train=True, tau=0.5):
        # x: (B, D, L, F) -> e: (B, D, L, h_enc); lag index 0 = oldest
        B = x.shape[0]; e = s.enc(x); m = s.ell(e)                                      # (B, D, L, h)
        if s.self_only: G = torch.eye(s.D, device=x.device)[None].expand(s.L, -1, -1); kl = torch.tensor(0.0, device=x.device)
        else:
            logits = s.graph_logits(); p = torch.sigmoid(logits)
            if train:
                u = torch.rand_like(p).clamp(1e-6, 1 - 1e-6); G = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / tau)
            else: G = p
            p0 = 0.1; kl = (p * torch.log(p / p0 + 1e-9) + (1 - p) * torch.log((1 - p) / (1 - p0) + 1e-9)).mean()
        # aggregate parents: agg[b, i] = sum_l sum_j G[l, j, i] * m[b, j, l]
        agg = torch.einsum("lji,bjlh->bih", G, m) / s.L
        hid = s.psi(agg); z = s.zeta(torch.cat([hid, s.stock[None].expand(B, -1, -1)], -1)).squeeze(-1)   # (B, D)
        return z, kl

def train_eval(Xtr, Ytr, Mtr, Xva, Yva, Mva, Xte, Yte, Mte, D, seed, self_only=False, lr=1e-4, max_ep=100, patience=10):
    torch.manual_seed(seed); np.random.seed(seed); net = CausalStockLite(D, self_only=self_only).to(dev); opt = torch.optim.Adam(net.parameters(), lr)
    T = lambda a: torch.tensor(a, device=dev)
    Xtr, Ytr, Mtr, Xva, Yva, Mva, Xte, Yte, Mte = map(T, (Xtr, Ytr, Mtr, Xva, Yva, Mva, Xte, Yte, Mte))
    def evaluate(X, Y, M):
        net.eval(); out = []
        with torch.no_grad():
            for b in range(0, len(X), 64): z, _ = net(X[b:b + 64], train=False); out.append(torch.sigmoid(z))
        p = torch.cat(out); m = M.bool(); y = Y[m].cpu().numpy(); pp = p[m].cpu().numpy()
        return dict(acc=accuracy_score(y, pp >= 0.5), mcc=matthews_corrcoef(y, pp >= 0.5), auroc=roc_auc_score(y, pp)), pp, y
    best, state, bad, ep = -9, None, 0, 0
    for ep in range(max_ep):
        net.train(); perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(perm), 32):
            j = perm[b:b + 32]; z, kl = net(Xtr[j], train=True); m = Mtr[j].bool()
            loss = Fn.binary_cross_entropy_with_logits(z[m], Ytr[j][m]) + 0.01 * kl; opt.zero_grad(); loss.backward(); opt.step()
        v, _, _ = evaluate(Xva, Yva, Mva); score = v["acc"] + v["mcc"]
        if score > best + 1e-4: best, bad, state = score, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else: bad += 1
        if bad >= patience: break
    net.load_state_dict(state); te, pp, y = evaluate(Xte, Yte, Mte); te["epochs"] = ep + 1; te["val_acc_plus_mcc"] = float(best)
    return te, pp, y

def windows(F3, Y2, M2):
    """F3: (T, D, F) features by day; Y2/M2: (T, D). Returns X (T-L, D, L, F) windows of days t-L..t-1 for label day t."""
    T = F3.shape[0]; X = np.stack([F3[t - L:t] for t in range(L, T)], 0).transpose(0, 2, 1, 3)   # (T-L, D, L, F)
    return X.astype(np.float32), Y2[L:].astype(np.float32), M2[L:].astype(np.float32)

GRID = len(sys.argv) > 1 and sys.argv[1] == "grid"
out = {}
# ---------------- KDD17 (official preprocessing) ----------------
files = sorted(glob.glob(str(ROOT / "data/raw/kdd17/ourpped/*.csv"))); dates = np.genfromtxt(ROOT / "data/raw/kdd17/trading_dates.csv", dtype=str)
A = np.stack([np.genfromtxt(f, delimiter=",") for f in files], 1)            # (T, D, 13)
F3 = A[:, :, :FEAT].copy(); lab = A[:, :, -2]; miss = (A[:, :, :FEAT] < -123320).any(-1)
F3[miss] = 0.0; F3 = np.clip(F3, -50, 50) / 10.0                               # percent features; scale down
Y2 = (lab > 0).astype(float); M2 = ((lab == 1) | (lab == -1)) & ~miss
# a window is valid if the label is +-1 and the last L days of that stock have no missing rows
validwin = np.stack([~miss[t - L:t].any(0) for t in range(L, len(dates))], 0)
X, Y, Mk = windows(F3, Y2, M2); Mk = Mk * validwin
d = dates[L:]; itr, iva, ite = (d >= "2007-01-03") & (d < "2015-01-02"), (d >= "2015-01-02") & (d < "2016-01-04"), d >= "2016-01-04"
print("KDD17 windows train/val/test days", itr.sum(), iva.sum(), ite.sum(), "| labelled test instances", int(Mk[ite].sum()), flush=True)
runs = {"CausalStock-lite (no news)": [], "w/o TCD (self-loops only)": []}
if GRID:
    grid = {}
    for lr in [1e-3, 1e-4, 1e-5]:
        for seed in range(3):
            t0 = time.time(); r, _, _ = train_eval(X[itr], Y[itr], Mk[itr], X[iva], Y[iva], Mk[iva], X[ite], Y[ite], Mk[ite], X.shape[1], seed, lr=lr, max_ep=300, patience=25)
            r["seconds"] = round(time.time() - t0); grid.setdefault(str(lr), []).append(r); print("GRID lr", lr, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    dump({"kdd17_lr_grid": {lr: dict(mean={m: float(np.mean([r[m] for r in v])) for m in ["acc", "mcc", "auroc", "val_acc_plus_mcc"]}, runs=v) for lr, v in grid.items()}}, "causalstock_lite_grid.json"); sys.exit(0)
for name, so in [("CausalStock-lite (no news)", False), ("w/o TCD (self-loops only)", True)]:
    for seed in range(5):
        t0 = time.time(); r, _, _ = train_eval(X[itr], Y[itr], Mk[itr], X[iva], Y[iva], Mk[iva], X[ite], Y[ite], Mk[ite], X.shape[1], seed, self_only=so)
        r["seconds"] = round(time.time() - t0); runs[name].append(r); print("KDD17", name, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
out["kdd17"] = {k: dict(mean={m: float(np.mean([r[m] for r in v])) for m in ["acc", "mcc", "auroc"]}, std={m: float(np.std([r[m] for r in v])) for m in ["acc", "mcc", "auroc"]}, runs=v) for k, v in runs.items()}
out["kdd17"]["published_table1"] = {"LSTM": [51.18, 0.0187], "ALSTM": [51.66, 0.0316], "StockNet": [51.93, 0.0335], "Adv-ALSTM": [51.69, 0.0333], "DTML": [53.53, 0.0733], "CausalStock": [56.09, 0.1235]}
out["kdd17"]["n_test_instances"] = int(Mk[ite].sum()); dump(out, "causalstock_lite.json")
# ---------------- Taiwan 50 (same 11 features from our prices; label y_up; walk-forward) ----------------
df = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet").sort_values(["stock_id", "date"]); df["date"] = pd.to_datetime(df["date"])
g = df.groupby("stock_id")
f = pd.DataFrame({"c_open": df.open / df.close - 1, "c_high": df.high / df.close - 1, "c_low": df.low / df.close - 1, "n_close": df.close / g.close.shift(1) - 1, "n_adj": df.close / g.close.shift(1) - 1})
for k in (5, 10, 15, 20, 25, 30): f[f"ma{k}"] = g.close.transform(lambda s: s.rolling(k).mean()) / df.close - 1
f = f * 100 / 10.0; f["date"] = df.date.values; f["stock_id"] = df.stock_id.values; f["y"] = df.y_up.values
piv = {c: f.pivot(index="date", columns="stock_id", values=c) for c in f.columns if c not in ("date", "stock_id", "y")}
days = piv["c_open"].index; stocks = piv["c_open"].columns
F3 = np.stack([piv[c].values for c in piv], -1); miss = np.isnan(F3).any(-1); F3 = np.nan_to_num(np.clip(F3, -5, 5))
Ypv = f.pivot(index="date", columns="stock_id", values="y").shift(1); Y2 = np.nan_to_num(Ypv.values); M2 = ~np.isnan(Ypv.values)   # label shifted one row so the window ends on the feature day t (see 14_causalstock_v2.py)
validwin = np.stack([~miss[t - L:t].any(0) for t in range(L, len(days))], 0)
X, Y, Mk = windows(F3, Y2, M2); Mk = Mk * validwin; d = days[L - 1:-1]
tw = {}; oos_rows = []
for yr in TEST_YEARS:
    t0 = pd.Timestamp(f"{yr}-01-01"); v0 = t0 - pd.DateOffset(months=12)
    itr, iva, ite = (d < v0), (d >= v0) & (d < t0), (d >= t0) & (d < pd.Timestamp(f"{yr + 1}-01-01"))
    for name, so in [("CausalStock-lite (no news)", False), ("w/o TCD (self-loops only)", True)]:
        for seed in range(3):
            r, pp, yy = train_eval(X[itr], Y[itr], Mk[itr], X[iva], Y[iva], Mk[iva], X[ite], Y[ite], Mk[ite], X.shape[1], seed, self_only=so)
            tw.setdefault(name, {}).setdefault(str(yr), []).append(r); print("TW50", yr, name, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
            if name.startswith("CausalStock") and seed == 0:
                dd = np.repeat(d[ite].values, X.shape[1])[Mk[ite].astype(bool).ravel()]; ss = np.tile(np.array(stocks), int(ite.sum()))[Mk[ite].astype(bool).ravel()]
                oos_rows.append(pd.DataFrame({"date": dd, "stock_id": ss, "y": yy, "p": pp}))
oos = pd.concat(oos_rows); oos.to_parquet(RES / "oos_causalstock_tw50.parquet")
pooled = {name: dict(auroc=float(roc_auc_score(oos.y, oos.p)), acc=float(accuracy_score(oos.y, oos.p >= 0.5)), mcc=float(matthews_corrcoef(oos.y, oos.p >= 0.5))) for name in ["CausalStock-lite (no news)"]}
out["tw50"] = {"by_year_runs": tw, "pooled_seed0": pooled, "seed_mean_by_year": {name: {yr: {m: float(np.mean([r[m] for r in v])) for m in ["acc", "mcc", "auroc"]} for yr, v in d2.items()} for name, d2 in tw.items()}}
dump(out, "causalstock_lite.json"); print("done")
