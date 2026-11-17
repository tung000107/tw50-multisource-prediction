"""Re-implementation of the baseline paper Yang et al. (2024), Appl. Artif. Intell. 38(1), on our Taiwan 50 data.
Their pipeline: (1) an LSTM regresses the next-day close from a look-back window of OHLCV (44 days); (2) technical indicators are
re-computed with the predicted close appended ("predicted indicators"); (3) RF / SVM / LightGBM classify next-day up/down from
raw + predicted indicators (+ three-institution net buys); (4) data "randomly divided into training and testing" (no walk-forward).
We run their classifier stage under (a) their random 80/20 split and (b) our walk-forward protocol, with and without the predicted indicators.
Simplifications (stated on the slide): one pooled LSTM over the 50 stocks (standardised inputs) instead of one per asset; linear SVM on a
20k-row subsample for speed; indicators: %K, %D, MA5/10/20, EMA12/26, BIAS, RSI, Williams %R, CCI, MACD, PPO, ROC, Momentum.
Output: results/final/sota_yang.json
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from common import *
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); torch.manual_seed(0); np.random.seed(0)
full = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet").sort_values(["stock_id", "date"]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
# ---------- technical indicators from a close series (vectorised per stock) ----------
def indicators(g):
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    out = pd.DataFrame(index=g.index)
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max(); out["K"] = 100 * (c - lo14) / (hi14 - lo14); out["D"] = out["K"].rolling(3).mean()
    for w in (5, 10, 20): out[f"MA{w}"] = c.rolling(w).mean() / c - 1
    e12, e26 = c.ewm(span=12).mean(), c.ewm(span=26).mean(); out["EMA12"] = e12 / c - 1; out["EMA26"] = e26 / c - 1
    out["BIAS"] = (c - c.rolling(20).mean()) / c.rolling(20).mean() * 100
    d = c.diff(); up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean(); out["RSI"] = 100 - 100 / (1 + up / dn)
    out["WR"] = -100 * (hi14 - c) / (hi14 - lo14)
    tp = (h + l + c) / 3; out["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True))
    out["MACD"] = (e12 - e26) / c; out["PPO"] = (e12 - e26) / e26 * 100; out["ROC"] = c.pct_change(10) * 100; out["MOM"] = c - c.shift(10)
    out["MOM"] = out["MOM"] / c
    return out
IND = ["K", "D", "MA5", "MA10", "MA20", "EMA12", "EMA26", "BIAS", "RSI", "WR", "CCI", "MACD", "PPO", "ROC", "MOM"]
ind = pd.concat([indicators(g) for _, g in full.groupby("stock_id")]); full = full.join(ind)
# ---------- stage 1: pooled LSTM regressing next-day log-return of close from 44 days of OHLCV ----------
W = 44; feats = ["open", "high", "low", "close", "volume"]
full["lr_next"] = np.log(full.close_next / full.close)
X = np.stack([np.log(full[c].values + 1e-9) for c in feats], axis=1).astype(np.float32)
rows = {s: g.index.values for s, g in full.groupby("stock_id")}; widx = np.full((len(full), W), -1, dtype=np.int64)
for s, r in rows.items():
    for j in range(W): widx[r[j:], W - 1 - j] = r[:len(r) - j] if j else r
ok = (widx >= 0).all(axis=1) & full.lr_next.notna().values
def seqs(ix):
    Z = X[widx[ix]]; Z = Z - Z[:, -1:, :]          # log-prices relative to the last day (scale-free), volume likewise
    return torch.tensor(Z, device=dev)
class Net(nn.Module):
    def __init__(s): super().__init__(); s.rnn = nn.LSTM(5, 64, batch_first=True); s.head = nn.Linear(64, 1)
    def forward(s, x): o, _ = s.rnn(x); return s.head(o[:, -1]).squeeze(-1)
def lstm_predict(train_idx, pred_idx, epochs=15):
    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); Xtr = seqs(train_idx); ytr = torch.tensor(full.lr_next.values[train_idx], dtype=torch.float32, device=dev)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(perm), 512):
            j = perm[b:b + 512]; opt.zero_grad(); nn.functional.mse_loss(net(Xtr[j]), ytr[j]).backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(pred_idx), 8192): out.append(net(seqs(pred_idx[b:b + 8192])).cpu().numpy())
    return np.concatenate(out)
# ---------- predicted indicators: recompute indicators with the predicted next close appended ----------
def predicted_indicators(pred_lr):
    f2 = full.copy(); f2["pred_close"] = f2.close * np.exp(pred_lr)
    out = []
    for s, g in f2.groupby("stock_id"):
        gg = g.copy(); gg2 = gg.copy(); gg2["close"] = gg2["pred_close"]; gg2["high"] = np.maximum(gg2["high"], gg2["close"]); gg2["low"] = np.minimum(gg2["low"], gg2["close"])
        # indicator at t computed as if the (predicted) next close were already observed: shift the series by one and recompute
        shifted = gg2[["open", "high", "low", "close", "volume"]].shift(-1); shifted.iloc[-1] = gg2[["open", "high", "low", "close", "volume"]].iloc[-1]
        comb = gg[["open", "high", "low", "close", "volume"]].copy()
        comb_next = pd.concat([comb, shifted.iloc[[-1]]]).reset_index(drop=True)
        # simpler, exact: compute indicators on the series with one appended predicted bar, then take the value aligned to the appended bar
        # vectorised approximation: indicators on series where close_t is replaced by pred_close_t -> value "as of predicted t+1"
        pi = indicators(gg2.assign(close=gg2.pred_close)); pi.columns = ["p" + c for c in pi.columns]; out.append(pi)
    return pd.concat(out)
PIND = ["p" + c for c in IND]
INST = ["foreign_net", "trust_net", "dealer_net"]; full["inst_total"] = full[INST].sum(axis=1)
for c in INST + ["inst_total"]: full[c + "_r"] = full[c] / full.volume
BASE = IND + [c + "_r" for c in INST + ["inst_total"]]
res = {}
# ---------- (a) random 80/20 split on all rows (their protocol) ----------
idx_all = full.index[ok & full[IND].notna().all(axis=1) & full.y_up.notna()].values
tr_i, te_i = train_test_split(idx_all, test_size=0.2, random_state=42, shuffle=True)
t = time.time(); pred = np.full(len(full), np.nan); pred[idx_all] = lstm_predict(tr_i, idx_all); print("LSTM random fit", round(time.time() - t), "s; direction acc of LSTM alone (test):", round(float(((pred[te_i] > 0) == (full.lr_next.values[te_i] > 0)).mean()), 4), flush=True)
pi = predicted_indicators(pred); full2 = full.join(pi)
def classify(trn, tst, feats, label="y_up"):
    out = {}
    Xtr, ytr = full2.loc[trn, feats], full2.loc[trn, label]; Xte, yte = full2.loc[tst, feats], full2.loc[tst, label]
    imp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(Xtr); A, B = imp.transform(Xtr), imp.transform(Xte)
    rf = RandomForestClassifier(300, min_samples_leaf=20, n_jobs=16, random_state=0).fit(A, ytr); out["RF"] = metrics(yte, rf.predict_proba(B)[:, 1])
    sub = np.random.default_rng(0).choice(len(A), min(20000, len(A)), replace=False)
    svm = CalibratedClassifierCV(LinearSVC(C=0.1), cv=3).fit(A[sub], ytr.values[sub]); out["SVM"] = metrics(yte, svm.predict_proba(B)[:, 1])
    lg = lgb.LGBMClassifier(n_estimators=400, random_state=0, **LGB_PARAMS).fit(A, ytr); out["LightGBM"] = metrics(yte, lg.predict_proba(B)[:, 1])
    return out
res["random_split"] = {"raw indicators (Table 4 analogue)": classify(tr_i, te_i, BASE), "+ predicted indicators (Table 5 analogue)": classify(tr_i, te_i, BASE + PIND)}
print("random split", {k: {m: round(v["acc"], 4) for m, v in d.items()} for k, d in res["random_split"].items()}, flush=True)
# oracle check: indicators computed with the TRUE next close (upper bound / leakage illustration)
pi_true = predicted_indicators(full.lr_next.values); full2 = full.join(pi_true.rename(columns=lambda c: "o" + c[1:]))
res["random_split"]["oracle: indicators with the true next close"] = classify(tr_i, te_i, BASE + ["o" + c for c in IND])
full2 = full.join(pi)
# ---------- (b) our walk-forward protocol (LSTM refit per fold on the training window) ----------
fw = full.loc[idx_all]; wf = {}
for yr in TEST_YEARS:
    t0 = fw.loc[fw.date.dt.year == yr, "date"].min(); trn = fw.index[(fw.date_next < t0)].values; tst = fw.index[fw.date.dt.year == yr].values
    p = np.full(len(full), np.nan); p[np.concatenate([trn, tst])] = lstm_predict(trn, np.concatenate([trn, tst])); full2 = full.join(predicted_indicators(p))
    wf[str(yr)] = {"raw": classify(trn, tst, BASE), "+ predicted": classify(trn, tst, BASE + PIND)}; print("walk-forward", yr, {k: {m: round(v["acc"], 4) for m, v in d.items()} for k, d in wf[str(yr)].items()}, flush=True)
res["walk_forward_by_year"] = wf
res["walk_forward_pooled_mean"] = {k: {m: {met: float(np.mean([wf[y][k][m][met] for y in wf])) for met in ["acc", "auroc", "mcc"]} for m in ["RF", "SVM", "LightGBM"]} for k in ["raw", "+ predicted"]}
res["published"] = {"Table 5 all-day avg, 2330 SVM": 0.67, "Table 5 all-day avg, 0050 SVM": 0.65, "Table 6 our-method avg/max": [0.69, 0.72], "Table 9 filtered days 1434 SVM max": 0.86, "Table 10 filtered avg/max": [0.82, 0.86], "LSTM direction alone": 0.52, "split": "random, ratio not reported"}
dump(res, "sota_yang.json")
