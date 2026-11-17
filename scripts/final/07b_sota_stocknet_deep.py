"""ALSTM-style sequence baseline on StockNet with the OFFICIAL preprocessed inputs (price/preprocessed/*.txt:
date, movement %, normalised open/high/low/close, volume), look-back T = 5, StockNet split. Two label rules:
 (a) StockNet rule: movement >= +0.55 % up, <= -0.5 % down, rest dropped;  (b) plain sign of the next-day movement.
Model: 1-layer LSTM(64) + temporal attention pooling (ALSTM without adversarial training), AdamW, early stopping on validation MCC.
5 seeds. Output: results/final/sota_stocknet_deep.json
"""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
from common import *
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RAW = ROOT / "data/raw/stocknet/preprocessed"; frames = []
for f in sorted(glob.glob(str(RAW / "*.txt"))):
    d = pd.read_csv(f, sep="\t", header=None, names=["date", "mov", "open", "high", "low", "close", "volume"]); d["date"] = pd.to_datetime(d["date"]); d["stock"] = Path(f).stem; frames.append(d.sort_values("date"))
px = pd.concat(frames, ignore_index=True); g = px.groupby("stock")
px["volume"] = np.log1p(px["volume"]); px["volume"] = (px["volume"] - g["volume"].transform("mean")) / g["volume"].transform("std")
px["mov_next"] = g["mov"].shift(-1)
T = 5; F = ["mov", "open", "high", "low", "close", "volume"]
for k in range(1, T):
    for c in F: px[f"{c}_l{k}"] = g[c].shift(k)
px = px.dropna(subset=[f"{c}_l{T-1}" for c in F] + ["mov_next"])
def seq(d): return np.stack([np.stack([d[f"{c}_l{k}" if k else c].values for c in F], 1) for k in range(T - 1, -1, -1)], 1).astype(np.float32)   # (n, T, 6), oldest first
class ALSTM(nn.Module):
    def __init__(s, d=6, h=64): super().__init__(); s.rnn = nn.LSTM(d, h, batch_first=True); s.att = nn.Linear(h, 1); s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * h, 1))
    def forward(s, x):
        o, _ = s.rnn(x); a = torch.softmax(s.att(o).squeeze(-1), 1); ctx = (a.unsqueeze(-1) * o).sum(1); return s.head(torch.cat([ctx, o[:, -1]], 1)).squeeze(-1)
out = {}
for rule in ["stocknet_band", "sign"]:
    d = px[(px.date >= "2014-01-01") & (px.date < "2016-01-01")].copy()
    d["y"] = np.where(d.mov_next >= 0.0055, 1, np.where(d.mov_next <= -0.005, 0, np.nan)) if rule == "stocknet_band" else (d.mov_next > 0).astype(float)
    d = d.dropna(subset=["y"]); tr = d[d.date < "2015-08-01"]; va = d[(d.date >= "2015-08-01") & (d.date < "2015-10-01")]; te = d[d.date >= "2015-10-01"]
    Xtr, Xva, Xte = [torch.tensor(seq(x), device=dev) for x in (tr, va, te)]; ytr = torch.tensor(tr.y.values, dtype=torch.float32, device=dev)
    runs = []
    for seed in range(5):
        torch.manual_seed(seed); net = ALSTM().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
        best, state, bad = -9, None, 0
        for ep in range(100):
            net.train(); perm = torch.randperm(len(Xtr), device=dev)
            for b in range(0, len(perm), 256):
                j = perm[b:b + 256]; opt.zero_grad(); lossf(net(Xtr[j]), ytr[j]).backward(); opt.step()
            net.eval()
            with torch.no_grad(): pv = torch.sigmoid(net(Xva)).cpu().numpy()
            m = matthews_corrcoef(va.y, pv >= 0.5)
            if m > best + 1e-4: best, bad, state = m, 0, {k: v.clone() for k, v in net.state_dict().items()}
            else: bad += 1
            if bad >= 10: break
        net.load_state_dict(state); net.eval()
        with torch.no_grad(): pt = torch.sigmoid(net(Xte)).cpu().numpy()
        runs.append(dict(acc=accuracy_score(te.y, pt >= 0.5), mcc=matthews_corrcoef(te.y, pt >= 0.5), auroc=roc_auc_score(te.y, pt), val_mcc=float(best), epochs=ep + 1))
    out[rule] = dict(n=dict(train=len(tr), val=len(va), test=len(te), test_up=float(te.y.mean())), mean={k: float(np.mean([r[k] for r in runs])) for k in ["acc", "mcc", "auroc"]}, std={k: float(np.std([r[k] for r in runs])) for k in ["acc", "mcc", "auroc"]}, runs=runs)
    print(rule, out[rule]["n"], out[rule]["mean"], out[rule]["std"], flush=True)
dump(out, "sota_stocknet_deep.json")
