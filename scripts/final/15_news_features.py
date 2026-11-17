"""Leakage-safe news source (S6) for the Taiwan 50 panel from cnyes headlines (data/raw/cnyes/*.jsonl).
Assignment rule (fixed 2026-11-13, common.news_feature_day): a headline published after 08:30 on day t and up to 08:30 on the next trading day belongs to feature day t (the decision time).
First version, kept for the record: a headline published after the TWSE close of day t-1 (13:30) and before the open of day t+1 (09:00) belongs to
feature day t of every stock it is tagged with (cnyes 'market' codes); untagged headlines form a market-wide stream.
Features per (stock, day t):   n_news (count), n_news_5d, has_news, log1p(n_news), market_n_news;
   text score (fold-trained, no external LLM): inside each walk-forward training window, a TF-IDF (char 1-3-gram) + logistic
   regression is fitted on headlines -> next-day up of the tagged stock; the test-year score is the mean P(up) of that stock-day's
   headlines (0.5 if none). The same for the market stream -> market_text_score. Scores for the training rows are produced
   out-of-fold inside the window (5-fold by date blocks) so the downstream model does not see in-sample text fits.
Output: data/processed/news_features.parquet (one row per stock-day with fold-specific score columns)
"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import ROOT, TEST_YEARS, load, news_feature_day
sys.path.insert(0, str(ROOT / "scripts")); from universe import TAIWAN50
df = load(); days = np.sort(df.date.unique()); day_idx = {d: i for i, d in enumerate(days)}
rows = []
for f in sorted(glob.glob(str(ROOT / "data/raw/cnyes/*.jsonl"))):
    for line in open(f):
        d = json.loads(line)
        for x in d["rows"]:
            ts = pd.Timestamp(int(x["publishAt"]), unit="s", tz="Asia/Taipei").tz_localize(None); rows.append((ts, x["title"] or "", tuple(x["codes"])))
news = pd.DataFrame(rows, columns=["ts", "title", "codes"]).sort_values("ts"); print("headlines", len(news), news.ts.min(), news.ts.max())
# map timestamp -> feature day t: the first trading day t such that ts <= t 09:00 ... i.e. window (t-1 13:30, t+1 09:00] -> t. Equivalent: t = last trading day whose 13:30 close is <= ts, shifted: ts after close(t) and before open(t+1) -> t.
closes = pd.Series(days) + pd.Timedelta(hours=13, minutes=30)
news = news[news.ts > pd.Timestamp(days[0]) + pd.Timedelta(hours=8, minutes=30)].copy(); pos = news_feature_day(news.ts.values, days); news = news[pos >= 0].copy(); pos = pos[pos >= 0]; news["fday"] = pd.to_datetime(days[pos])   # shared rule, see common.news_feature_day
tw = set(TAIWAN50); ex = news.explode("codes"); tagged = ex[ex.codes.isin(tw)].rename(columns={"codes": "stock_id"}); market = news[news.codes.apply(lambda c: len(c) == 0)]
print("tagged Taiwan-50 headlines", len(tagged), "| market-wide (untagged)", len(market), "| stock-days with news", tagged.groupby(["stock_id", "fday"]).ngroups)
cnt = tagged.groupby(["stock_id", "fday"]).size().rename("n_news").reset_index().rename(columns={"fday": "date"})
out = df[["date", "stock_id", "year", "y_up"]].merge(cnt, on=["date", "stock_id"], how="left"); out["n_news"] = out.n_news.fillna(0)
out = out.sort_values(["stock_id", "date"]); out["n_news_5d"] = out.groupby("stock_id").n_news.transform(lambda s: s.rolling(5, min_periods=1).sum()); out["has_news"] = (out.n_news > 0).astype(float); out["log_news"] = np.log1p(out.n_news)
mc = market.groupby("fday").size().rename("market_n_news"); out["market_n_news"] = out.date.map(mc).fillna(0)
# ---- fold-trained text scores ----
lab = df.set_index(["stock_id", "date"]).y_up
tagged = tagged.merge(df[["stock_id", "date", "y_up"]].rename(columns={"date": "fday"}), on=["stock_id", "fday"], how="inner")
mk_lab = df.groupby("date").y_up.mean(); market = market.assign(y=market.fday.map(mk_lab)).dropna(subset=["y"]); market["y_up"] = (market.y > 0.5).astype(float)
def fit_score(train_df, apply_df, key):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=5, max_features=200000, sublinear_tf=True)
    Xtr = vec.fit_transform(train_df.title); clf = LogisticRegression(C=0.5, max_iter=2000).fit(Xtr, train_df.y_up)
    if len(apply_df) == 0: return pd.Series(dtype=float)
    p = clf.predict_proba(vec.transform(apply_df.title))[:, 1]; return pd.Series(p, index=apply_df.index)
# FIX 2026-11-17 (purge): a headline of feature day d is labelled with the close of the NEXT trading day, so a training headline must have
# d before the last trading day of the previous year; the old cut (d < 31 Dec) let the last trading day's headlines (labelled with the first
# test day's close) into training in the 2023 and 2024 folds (118 headlines).
_TD = np.sort(df.date.unique()); _last_td = lambda t0: pd.Timestamp(_TD[_TD < np.datetime64(t0)][-1])
for yr in TEST_YEARS:
    t0 = pd.Timestamp(f"{yr}-01-01"); trn = tagged[tagged.fday < _last_td(t0)]; tst = tagged[(tagged.fday >= t0) & (tagged.fday < pd.Timestamp(f"{yr + 1}-01-01"))]
    sc = pd.Series(np.nan, index=tagged.index)
    sc.loc[tst.index] = fit_score(trn, tst, "stock")                       # test-year headlines scored by a model fitted strictly before the test year
    blocks = pd.qcut(trn.fday.rank(method="first"), 5, labels=False)      # out-of-fold scores for training headlines (date blocks)
    for b in range(5): sc.loc[trn.index[blocks == b]] = fit_score(trn[blocks != b], trn[blocks == b], "stock")
    tagged[f"score_{yr}"] = sc
    trm = market[market.fday < _last_td(t0)]; tsm = market[(market.fday >= t0) & (market.fday < pd.Timestamp(f"{yr + 1}-01-01"))]
    scm = pd.Series(np.nan, index=market.index); scm.loc[tsm.index] = fit_score(trm, tsm, "mkt")
    blocks = pd.qcut(trm.fday.rank(method="first"), 5, labels=False)
    for b in range(5): scm.loc[trm.index[blocks == b]] = fit_score(trm[blocks != b], trm[blocks == b], "mkt")
    market[f"mscore_{yr}"] = scm; print("scored fold", yr, "tagged train/test", len(trn), len(tst), flush=True)
for yr in TEST_YEARS:
    s = tagged.groupby(["stock_id", "fday"])[f"score_{yr}"].mean().rename(f"text_score_{yr}").reset_index().rename(columns={"fday": "date"})
    out = out.merge(s, on=["stock_id", "date"], how="left"); out[f"text_score_{yr}"] = out[f"text_score_{yr}"].fillna(0.5)
    m = market.groupby("fday")[f"mscore_{yr}"].mean(); out[f"market_text_score_{yr}"] = out.date.map(m).fillna(0.5)
out.drop(columns=["year", "y_up"]).to_parquet(ROOT / "data/processed/news_features.parquet"); print(out.describe().T[["mean", "max"]].head(12)); print("saved")
