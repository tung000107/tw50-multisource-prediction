"""LLM news scores and look-ahead bias (the TA's concern made measurable).
A local open-weight LLM (Qwen3-4B, served by vLLM on this machine; released 2025-04-29, so its pre-training may contain 2023-2024 news) scores every
cnyes headline tagged to a Taiwan-50 stock (and the untagged market stream): "implied direction for the tagged company's share price, one number in [-1, 1]".
Scores are aligned exactly like the fold-trained TF-IDF score (15_news_features.py). We then compare, year by year, the univariate AUROC of the LLM score
vs the leakage-safe TF-IDF score, and the gain of Full+LLM over Full. If the LLM "remembers" the period, its edge should be larger in 2023-2024 than in 2025-2026.
Stage 1 (argv[1] == "score"): score headlines (resumable, data/processed/llm_scores.parquet). Stage 2 (default): features + LightGBM. Output: results/final/llm_lookahead.json"""
import sys, json, glob, re, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import *
sys.path.insert(0, str(ROOT / "scripts")); from universe import TAIWAN50
URLS = ["http://127.0.0.1:1053/v1/chat/completions", "http://127.0.0.1:1054/v1/chat/completions"]; MODEL = "Qwen/Qwen3-4B"; OUT = ROOT / "data/processed/llm_scores.parquet"
PROMPT = "你是台股新聞分析師。請判斷下列新聞標題對「標題所指公司（若無特定公司則指台股大盤）」股價的隱含方向。只輸出一個介於 -1 到 1 之間的小數（-1 極負面、0 中性或無關、1 極正面），不要輸出其他文字。\n標題："
rows = []
for f in sorted(glob.glob(str(ROOT / "data/raw/cnyes/*.jsonl"))):
    for line in open(f):
        d = json.loads(line)
        for x in d["rows"]: rows.append((pd.Timestamp(int(x["publishAt"]), unit="s", tz="Asia/Taipei").tz_localize(None), x["title"] or "", tuple(x["codes"])))
news = pd.DataFrame(rows, columns=["ts", "title", "codes"]).sort_values("ts"); tw = set(TAIWAN50)
news["tagged"] = news.codes.apply(lambda c: any(k in tw for k in c)); news["market"] = news.codes.apply(lambda c: len(c) == 0); news = news[news.tagged | news.market]
titles = sorted(set(news.title)); print("unique headlines to score", len(titles), flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "score":
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["title", "score"]); have = set(done.title); todo = [t for t in titles if t not in have]; print("already", len(have), "todo", len(todo), flush=True)
    def ask(i_t):
        i, t = i_t; body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT + t}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(URLS[i % 2], data=body, headers={"Content-Type": "application/json"}); r = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
                m = re.search(r"-?\d+(?:\.\d+)?", r); return t, (float(np.clip(float(m.group()), -1, 1)) if m else np.nan)
            except Exception as e: time.sleep(2)
        return t, np.nan
    t0 = time.time(); buf = []
    with ThreadPoolExecutor(48) as ex:
        for k, (t, s) in enumerate(ex.map(ask, enumerate(todo))):
            buf.append((t, s))
            if (k + 1) % 5000 == 0 or k + 1 == len(todo):
                done = pd.concat([done, pd.DataFrame(buf, columns=["title", "score"])]); buf = []; done.to_parquet(OUT); print(k + 1, "scored", round(time.time() - t0), "s", "nan share", round(float(done.score.isna().mean()), 4), flush=True)
    sys.exit(0)
# ---------------- stage 2 ----------------
sc = pd.read_parquet(OUT).drop_duplicates("title").set_index("title").score; news["llm"] = news.title.map(sc); print("scored share", float(news.llm.notna().mean()))
df = load(); days = np.sort(df.date.unique()); closes = pd.Series(days) + pd.Timedelta(hours=13, minutes=30)
news = news[news.ts > pd.Timestamp(days[0]) + pd.Timedelta(hours=8, minutes=30)].copy(); pos = news_feature_day(news.ts.values, days); news = news[pos >= 0].copy(); pos = pos[pos >= 0]; news["fday"] = pd.to_datetime(days[pos])   # shared rule, see common.news_feature_day
ex_ = news[news.tagged].explode("codes"); ex_ = ex_[ex_.codes.isin(tw)].rename(columns={"codes": "stock_id", "fday": "date"})
st = ex_.groupby(["stock_id", "date"]).llm.agg(llm_score="mean", llm_pos="max", llm_neg="min").reset_index(); mk = news[news.market].groupby("fday").llm.mean().rename("market_llm_score")
df = df.merge(st, on=["stock_id", "date"], how="left"); df["market_llm_score"] = df.date.map(mk); df[["llm_score", "llm_pos", "llm_neg", "market_llm_score"]] = df[["llm_score", "llm_pos", "llm_neg", "market_llm_score"]].fillna(0)
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); df = df.merge(nf, on=["date", "stock_id"], how="left")
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0].split("df = load()")[0])   # imports only
FOLDS = list(folds(df)); LLM = ["llm_score", "llm_pos", "llm_neg", "market_llm_score"]; S6 = ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"]
def run(feats, label):
    prob = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, te in FOLDS:
        d = df.copy(); d["text_score"] = d[f"text_score_{yr}"]; d["market_text_score"] = d[f"market_text_score_{yr}"]
        itr, iva, tr, te = [i[d.loc[i, label].notna()] for i in (itr, iva, tr, te)]
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(d.loc[itr, feats], d.loc[itr, label], eval_set=[(d.loc[iva, feats], d.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(d.loc[tr, feats], d.loc[tr, label]); prob.loc[te] = f.predict_proba(d.loc[te, feats])[:, 1]
    return prob
te = df.year.isin(TEST_YEARS); out = {"n_scored_headlines": int(sc.notna().sum()), "coverage_stock_days_with_llm": float((df.loc[te, "llm_score"] != 0).mean())}
# univariate, by year: LLM score vs leakage-safe TF-IDF score (rows with at least one headline)
uni = {}
for yr in TEST_YEARS:
    m = te & (df.year == yr) & (df.n_news > 0); y = df.loc[m, "y_up"].values
    uni[str(yr)] = dict(n=int(m.sum()), auroc_llm=auc_stat(y, df.loc[m, "llm_score"].values), auroc_tfidf=auc_stat(y, df.loc[m, f"text_score_{yr}"].values), auroc_llm_market=auc_stat(y, df.loc[m, "market_llm_score"].values))
out["univariate_by_year"] = uni; print(uni, flush=True)
arms = {"Full": ALL, "Full+S6": ALL + S6, "Full+LLM": ALL + LLM, "Full+S6+LLM": ALL + S6 + LLM, "S4+LLM": SRC["S4"] + LLM, "LLM-only": LLM}
for label in ["y_up", "y_gap", "y_intra"]:
    res = dict(Parallel(n_jobs=3, prefer="threads")(delayed(lambda a: (a, run(arms[a], label)))(a) for a in arms)); r = {}
    CUT, REL = pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-29")   # stated knowledge cut-off / weight release; rows between the two belong to neither group
    for part, sel in [("pooled", df.year.isin(TEST_YEARS)), ("2023-2024 (inside the LLM's likely pre-training window)", df.year.isin([2023, 2024])), ("2025-2026 (after its release)", df.year.isin([2025, 2026])),
                      ("up to 2025-03-31 (on or before the stated knowledge cut-off)", df.date <= CUT), ("from 2025-04-30 (after the weights were released)", df.date > REL)]:
        m = te & sel & df[label].notna(); y = df.loc[m, label].values; bs = DayBootstrap(df.loc[m, "date"].values, B=1000)
        r[part] = {a: auc_stat(y, res[a].loc[m].values) for a in arms}
        r[part].update({"Full+LLM minus Full": bs.ci_diff(auc_stat, y, res["Full"].loc[m].values, res["Full+LLM"].loc[m].values), "Full+S6 minus Full": bs.ci_diff(auc_stat, y, res["Full"].loc[m].values, res["Full+S6"].loc[m].values), "Full+LLM minus Full+S6": bs.ci_diff(auc_stat, y, res["Full+S6"].loc[m].values, res["Full+LLM"].loc[m].values)})
    r["by_year"] = {str(yr): {a: auc_stat(df.loc[te & (df.year == yr) & df[label].notna(), label].values, res[a].loc[te & (df.year == yr) & df[label].notna()].values) for a in arms} for yr in TEST_YEARS}
    out[label] = r; print(label, json.dumps({k: v for k, v in r.items() if k != "by_year"}, default=float)[:1500], flush=True); dump(out, "llm_lookahead.json")
print("done")
