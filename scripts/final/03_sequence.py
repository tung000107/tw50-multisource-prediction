"""Deep sequence models (LSTM, Transformer encoder) under the same walk-forward protocol.
Input for row (i, t): the arm's features on the W most recent trading days of stock i (t-W+1 .. t), standardised with
training-window statistics (NaN -> 0 after standardisation). Early stopping + threshold on the purged inner validation;
the inner-train model is used for the test year (no refit, as is standard for NNs). 3 seeds. GPU.
Output: results/final/oos_sequence.parquet, results/final/sequence_info.json
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch, torch.nn as nn
from common import *
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); torch.backends.cudnn.benchmark = True
df = load(); LABEL = "y_up"; FOLDS = list(folds(df))
df["_pos"] = df.groupby("stock_id").cumcount()                        # position within stock (sorted by date inside load())
stock_rows = {s: g.index.values for s, g in df.sort_values(["stock_id", "date"]).groupby("stock_id")}

def window_index(W):
    """for each row return the W row indices ending at that row within the same stock (-1 if unavailable)."""
    idx = np.full((len(df), W), -1, dtype=np.int64)
    for s, rows in stock_rows.items():
        for j in range(W):
            idx[rows[j:], W - 1 - j] = rows[:len(rows) - j] if j else rows
    return idx

class LSTMNet(nn.Module):
    def __init__(self, d, h=64):
        super().__init__(); self.rnn = nn.LSTM(d, h, num_layers=1, batch_first=True); self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 1))
    def forward(self, x): o, _ = self.rnn(x); return self.head(o[:, -1]).squeeze(-1)
class TransNet(nn.Module):
    def __init__(self, d, h=64, W=5):
        super().__init__(); self.inp = nn.Linear(d, h); self.pos = nn.Parameter(torch.zeros(1, W, h))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 128, dropout=0.2, batch_first=True), 2); self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 1))
    def forward(self, x): z = self.enc(self.inp(x) + self.pos); return self.head(z[:, -1]).squeeze(-1)

def fit_predict(kind, feats, W, seed, label=LABEL):
    torch.manual_seed(seed); np.random.seed(seed)
    X = df[feats].values.astype(np.float32); widx = window_index(W); ok = (widx >= 0).all(axis=1)
    prob = pd.Series(np.nan, index=df.index); info = {}
    for yr, tr, itr, iva, te in FOLDS:
        itr, iva, te = [i[ok[i] & df.loc[i, label].notna().values] for i in (itr, iva, te)]
        mu = np.nanmean(X[tr], axis=0); sd = np.nanstd(X[tr], axis=0) + 1e-9
        Z = np.nan_to_num((X - mu) / sd, nan=0.0).astype(np.float32); Z = np.clip(Z, -8, 8)
        def tens(ix): return torch.tensor(Z[widx[ix]], device=dev), torch.tensor(df.loc[ix, label].values, dtype=torch.float32, device=dev)
        Xtr, ytr = tens(itr); Xva, yva = tens(iva); Xte, _ = tens(te)
        net = (LSTMNet(len(feats)) if kind == "LSTM" else TransNet(len(feats), W=W)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad, ep = -1, None, 0, 0
        for ep in range(60):
            net.train(); perm = torch.randperm(len(Xtr), device=dev)
            for b in range(0, len(perm), 1024):
                j = perm[b:b + 1024]; opt.zero_grad(); lossf(net(Xtr[j]), ytr[j]).backward(); opt.step()
            net.eval()
            with torch.no_grad(): pv = torch.sigmoid(torch.cat([net(Xva[b:b + 8192]) for b in range(0, len(Xva), 8192)])).cpu().numpy()
            a = auc_stat(yva.cpu().numpy(), pv)
            if a > best + 1e-4: best, bad, best_state = a, 0, {k: v.clone() for k, v in net.state_dict().items()}
            else: bad += 1
            if bad >= 8: break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(torch.cat([net(Xva[b:b + 8192]) for b in range(0, len(Xva), 8192)])).cpu().numpy()
            pt = torch.sigmoid(torch.cat([net(Xte[b:b + 8192]) for b in range(0, len(Xte), 8192)])).cpu().numpy()
        prob.loc[te] = pt; info[str(yr)] = dict(best_val_auc=float(best), epochs=ep + 1, thr=best_threshold(yva.cpu().numpy(), pv))
    return prob, info

if __name__ == "__main__":
    runs = [(k, a, 5, s) for k in ["LSTM", "Transformer"] for a in ["Full", "S4-only", "TW-only"] for s in [0, 1, 2]]
    runs += [("LSTM", "Full", W, 0) for W in [1, 10, 20]]
    ARM = {"Full": ALL, "S4-only": SRC["S4"], "TW-only": TW}
    te_mask = df.year.isin(TEST_YEARS); res = df.loc[te_mask, ["date", "stock_id", "year", "y_up"]].copy(); info = {}
    for kind, arm, W, seed in runs:
        t = time.time(); p, inf = fit_predict(kind, ARM[arm], W, seed); name = f"{kind}-W{W}|{arm}|{seed}"
        res[name] = p.loc[te_mask].values; inf["seconds"] = round(time.time() - t, 1); info[name] = inf
        y = res.loc[res[name].notna(), "y_up"].values; print(f"{name:28s} auroc={auc_stat(y, res.loc[res[name].notna(), name].values):.4f} ({time.time()-t:.0f}s)", flush=True)
        res.to_parquet(RES / "oos_sequence.parquet"); dump(info, "sequence_info.json")
