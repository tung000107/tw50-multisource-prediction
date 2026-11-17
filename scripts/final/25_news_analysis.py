"""What else do the news headlines say? (all headline timestamps aligned to the feature day exactly as in 15_news_features.py)
 (1) Direction vs magnitude: univariate AUROC of news-count features for y_up and for |r_next| > k; LightGBM magnitude task Full vs Full+S6 vs Full+S9;
     four-level task (k = training-window median |r|, as in 06_labels.py) Full vs Full+S6 vs Full+S6+S9 (macro-F1, block-bootstrap CI)
 (2) Keyword event study on stock-tagged headlines: same-day return (what the headline reports) vs next-day return / up-rate / four-level shares
 (3) Concept themes (AI / Nvidia, Apple, CoWoS, tariffs, copper, shipping rates, Fed, FX, memory): daily theme intensity -> next-day return and |return|
     of the concept group vs the other stocks on high-intensity days; theme intensities x membership as LightGBM features (S11)
 (4) Monthly-revenue headlines: YoY parsed from the headline ("年增 x %" / "年減 x %") -> next-day return, up-rate, four-level shares by bucket
Output: results/final/news_analysis.json, figures/final/f18_news_analysis.png"""
import sys, json, glob, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
sys.path.insert(0, str(ROOT / "scripts")); from universe import TAIWAN50, SECTOR, SECTOR_EN
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); df = df.merge(nf, on=["date", "stock_id"], how="left"); ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); df = df.merge(ex, on=["date", "stock_id"], how="left")
S6 = ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"]; S9 = ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]
te_mask = df.year.isin(TEST_YEARS); out = {}
# ---------- headlines aligned to feature days ----------
rows = []
for f in sorted(glob.glob(str(ROOT / "data/raw/cnyes/*.jsonl"))):
    for line in open(f):
        for x in json.loads(line)["rows"]: rows.append((pd.Timestamp(int(x["publishAt"]), unit="s", tz="Asia/Taipei").tz_localize(None), x["title"] or "", tuple(x["codes"])))
news = pd.DataFrame(rows, columns=["ts", "title", "codes"]).sort_values("ts"); days = np.sort(df.date.unique()); closes = pd.Series(days) + pd.Timedelta(hours=13, minutes=30)
news = news[news.ts > pd.Timestamp(days[0]) + pd.Timedelta(hours=8, minutes=30)].copy(); pos = news_feature_day(news.ts.values, days); news = news[pos >= 0].copy(); pos = pos[pos >= 0]; news["fday"] = pd.to_datetime(days[pos])   # shared rule, see common.news_feature_day
tw = set(TAIWAN50); tagged = news.explode("codes"); tagged = tagged[tagged.codes.isin(tw)].rename(columns={"codes": "stock_id", "fday": "date"})
base = df[["date", "stock_id", "year", "ret1", "ret_next", "y_up", "vol20"]].copy(); k_all = float(base.ret_next.abs().median()); base["y4"] = np.select([base.ret_next <= -k_all, base.ret_next <= 0, base.ret_next <= k_all], [0, 1, 2], 3)
def four_shares(y4): return {n: float((y4 == i).mean()) for i, n in enumerate(["strong_down", "mild_down", "mild_up", "strong_up"])}
BASE = dict(n=int(len(base)), next_mean=float(base.ret_next.mean()), up_rate=float(base.y_up.mean()), abs_next=float(base.ret_next.abs().mean()), four=four_shares(base.y4)); out["baseline_all_stock_days"] = BASE; out["k_for_descriptive_four_level"] = k_all
# ---------- (1) direction vs magnitude ----------
L = json.load(open(RES / "labels.json")); kby = L["four_level"]["raw"]["k_by_test_year"]
df["_abs"] = df.ret_next.abs(); uni = {}
for yr in TEST_YEARS:
    m = te_mask & (df.year == yr); k = kby[str(yr)]; ymag = (df.loc[m, "_abs"] > k).astype(int).values; yu = df.loc[m, "y_up"].values
    uni[str(yr)] = {c: dict(auroc_magnitude=auc_stat(ymag, df.loc[m, c].values), auroc_direction=auc_stat(yu, df.loc[m, c].values)) for c in ["n_news", "n_news_5d", "market_n_news", "vol20"]}
    uni[str(yr)]["text_score"] = dict(auroc_magnitude=auc_stat(ymag, df.loc[m, f"text_score_{yr}"].values), auroc_direction=auc_stat(yu, df.loc[m, f"text_score_{yr}"].values))
out["univariate_news_vs_magnitude"] = uni; print("univariate", {y: {c: round(v["auroc_magnitude"], 3) for c, v in r.items()} for y, r in uni.items()}, flush=True)
def fold_cols(d, yr): d = d.copy(); d["text_score"] = d[f"text_score_{yr}"]; d["market_text_score"] = d[f"market_text_score_{yr}"]; return d
def run_bin(feats, label_fn):
    prob = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, te in FOLDS:
        d = fold_cols(df, yr); k = float(d.loc[tr, "_abs"].median()); d["_y"] = label_fn(d, k)
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(d.loc[itr, feats], d.loc[itr, "_y"], eval_set=[(d.loc[iva, feats], d.loc[iva, "_y"])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(d.loc[tr, feats], d.loc[tr, "_y"]); prob.loc[te] = f.predict_proba(d.loc[te, feats])[:, 1]
    return prob
def run4(feats):
    pred = pd.Series(np.nan, index=df.index); truth = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, te in FOLDS:
        d = fold_cols(df, yr); k = float(d.loc[tr, "_abs"].median()); d["_y"] = np.select([d.ret_next <= -k, d.ret_next <= 0, d.ret_next <= k], [0, 1, 2], 3)
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, random_state=42, **LGB_PARAMS).fit(d.loc[itr, feats], d.loc[itr, "_y"], eval_set=[(d.loc[iva, feats], d.loc[iva, "_y"])], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(objective="multiclass", n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(d.loc[tr, feats], d.loc[tr, "_y"]); pred.loc[te] = f.predict_proba(d.loc[te, feats]).argmax(1); truth.loc[te] = d.loc[te, "_y"]
    return pred, truth
MAG = {"Full": ALL, "TW-only": TW, "Full+S6": ALL + S6, "Full+S9": ALL + S9, "Full+S6+S9": ALL + S6 + S9, "S6-only": S6}
pm = dict(Parallel(n_jobs=3, prefer="threads")(delayed(lambda a: (a, run_bin(MAG[a], lambda d, k: (d._abs > k).astype(int))))(a) for a in MAG))
ymag = pd.Series(np.nan, index=df.index)
for yr, tr, itr, iva, te in FOLDS: ymag.loc[te] = (df.loc[te, "_abs"] > float(df.loc[tr, "_abs"].median())).astype(int)
yy = ymag.loc[te_mask].values; bs = DayBootstrap(df.loc[te_mask, "date"].values, B=1000)
out["magnitude_task"] = {a: dict(auroc=auc_stat(yy, pm[a].loc[te_mask].values), ci=bs.ci(auc_stat, yy, pm[a].loc[te_mask].values)) for a in MAG}
for a in ["Full+S6", "Full+S9", "Full+S6+S9"]: out["magnitude_task"][f"{a} minus Full"] = bs.ci_diff(auc_stat, yy, pm["Full"].loc[te_mask].values, pm[a].loc[te_mask].values)
print("magnitude", {a: round(v["auroc"], 4) if "auroc" in v else [round(x, 4) for x in v] for a, v in out["magnitude_task"].items()}, flush=True)
F4 = {"Full": ALL, "Full+S6": ALL + S6, "Full+S6+S9": ALL + S6 + S9}; p4 = dict(Parallel(n_jobs=3, prefer="threads")(delayed(lambda a: (a, run4(F4[a])))(a) for a in F4)); y4 = p4["Full"][1].loc[te_mask].values.astype(int)
mf = lambda t, p: f1_score(t, p, average="macro"); out["four_level_task"] = {a: dict(macro_f1=float(mf(y4, p4[a][0].loc[te_mask].values.astype(int))), ci=bs.ci(mf, y4, p4[a][0].loc[te_mask].values.astype(int))) for a in F4}
for a in ["Full+S6", "Full+S6+S9"]: out["four_level_task"][f"{a} minus Full"] = bs.ci_diff(mf, y4, p4["Full"][0].loc[te_mask].values.astype(int), p4[a][0].loc[te_mask].values.astype(int))
print("four-level", out["four_level_task"], flush=True); dump(out, "news_analysis.json")
# ---------- (2) keyword event study ----------
KW = {"revenue record high": r"營收.*(創新高|新高|創高)", "orders / big order": r"訂單|大單|接單", "target price raised / upgrade": r"調升|上修|升評|喊到", "target cut / downgrade": r"調降|下修|降評", "earnings call": r"法說", "buyback": r"庫藏股", "capital reduction": r"減資",
      "limit-up reported": r"漲停", "limit-down reported": r"跌停", "foreign net buy": r"外資.*買超|買超.*外資", "foreign net sell": r"外資.*賣超|賣超.*外資", "loss": r"虧損|轉虧", "tariff": r"關稅", "capacity expansion / new plant": r"擴產|新廠|擴廠", "partnership": r"合作|結盟|攜手", "dividend": r"股利|配息|殖利率", "EPS / profit": r"EPS|獲利"}
ev = {}; bsd = {}
for name, pat in KW.items():
    t = tagged[tagged.title.str.contains(pat, regex=True)]; e = t.groupby(["stock_id", "date"]).size().rename("n").reset_index().merge(base, on=["stock_id", "date"], how="inner")
    if len(e) < 30: continue
    b = DayBootstrap(e.date.values, B=500); ev[name] = dict(n_events=int(len(e)), n_stock_days=int(e.groupby(["stock_id", "date"]).ngroups), same_day_ret_mean=float(e.ret1.mean()), next_mean=float(e.ret_next.mean()), next_mean_ci=b.ci(lambda x: float(x.mean()), e.ret_next.values), up_rate=float(e.y_up.mean()), up_rate_ci=b.ci(lambda x: float(x.mean()), e.y_up.values), abs_next=float(e.ret_next.abs().mean()), four=four_shares(e.y4))
out["keyword_events"] = ev; print("events", {k: (v["n_events"], round(v["same_day_ret_mean"] * 100, 2), round(v["next_mean"] * 100, 3), round(v["up_rate"], 3)) for k, v in ev.items()}, flush=True); dump(out, "news_analysis.json")
# ---------- (3) concept themes ----------
THEMES = {"AI / Nvidia": (r"AI|人工智慧|輝達|NVIDIA|Nvidia|GPU|算力|AI伺服器", ["2330", "2317", "2382", "3231", "6669", "3017", "2308", "2383", "3037", "2345", "3661", "2454"]), "Apple": (r"蘋果|Apple|iPhone", ["2330", "2317", "2382", "3231", "2308", "2327", "3037", "3711", "2301", "2383"]),
          "CoWoS / advanced packaging": (r"CoWoS|先進封裝", ["2330", "3711", "3037", "3661", "2454"]), "tariff": (r"關稅", [k for k in TAIWAN50 if SECTOR[k] in ("半導體", "電子/硬體")]), "copper / materials": (r"銅價|銅", ["2383", "3037", "2308", "2301", "2327"]), "shipping rates": (r"運價|貨櫃|航運|SCFI", ["2603", "2609", "2615"]),
          "Fed / rates": (r"升息|降息|Fed|聯準會", [k for k in TAIWAN50 if SECTOR[k] == "金融"]), "TWD / FX": (r"台幣|匯率|新台幣", [k for k in TAIWAN50 if SECTOR[k] == "金融"]), "memory / DRAM": (r"DRAM|記憶體|HBM", ["2408", "2330"])}
inten = pd.DataFrame(index=pd.to_datetime(days)); th = {}
for name, (pat, members) in THEMES.items():
    c = news[news.title.str.contains(pat, regex=True)].groupby("fday").size(); inten[name] = c.reindex(inten.index).fillna(0)
    d = base.copy(); d["I"] = d.date.map(inten[name]); d["member"] = d.stock_id.isin(members); d["hi"] = d.groupby("year").I.transform(lambda s: s >= s.quantile(0.9))
    g = d.groupby(["date", "member", "hi"]).agg(r=("ret_next", "mean"), a=("ret_next", lambda s: s.abs().mean()), u=("y_up", "mean")).reset_index()
    def cmp(col):
        a = g[g.member & g.hi]; b = g[g.member & ~g.hi]; c_ = g[~g.member & g.hi]; return dict(members_high=float(a[col].mean()), members_normal=float(b[col].mean()), others_high=float(c_[col].mean()), diff_members_high_minus_normal_ci=[float(x) for x in np.percentile([np.random.default_rng(i).choice(a[col].values, len(a)).mean() - b[col].mean() for i in range(500)], [2.5, 97.5])])
    th[name] = dict(n_headlines=int(inten[name].sum()), n_high_days=int(g[g.hi].date.nunique()), members=[f"{k} {TAIWAN50[k]}" for k in members], next_return=cmp("r"), abs_next_return=cmp("a"), up_rate=cmp("u"), corr_intensity_vs_member_abs_return=float(d[d.member].groupby("date").agg(I=("I", "first"), a=("ret_next", lambda s: s.abs().mean())).corr().iloc[0, 1]))
out["concept_themes"] = th; print("themes", {k: (v["n_high_days"], round(v["abs_next_return"]["members_high"] * 100, 2), round(v["abs_next_return"]["members_normal"] * 100, 2), round(v["up_rate"]["members_high"], 3), round(v["up_rate"]["members_normal"], 3)) for k, v in th.items()}, flush=True)
S11 = []
for name, (pat, members) in THEMES.items():
    c = f"theme_{re.sub('[^A-Za-z]', '_', name)}"; df[c] = np.log1p(df.date.map(inten[name]).fillna(0)); df[c + "_x"] = df[c] * df.stock_id.isin(members); S11 += [c, c + "_x"]
NEW = {"S6": S6, "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"], "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"], "S9": S9}; ALL9 = ALL + sum(NEW.values(), [])
arms = {"Full": ALL, "Full+S11": ALL + S11, "All9": ALL9, "All9+S11": ALL9 + S11, "S11-only": S11}; th_res = {}
for label, fn in [("y_up", lambda d, k: d.y_up), ("y_intra", lambda d, k: d.y_intra), ("magnitude", lambda d, k: (d._abs > k).astype(int))]:
    res = dict(Parallel(n_jobs=3, prefer="threads")(delayed(lambda a: (a, run_bin(arms[a], fn)))(a) for a in arms)); t = ymag.loc[te_mask].values if label == "magnitude" else df.loc[te_mask, label].values; ok = ~np.isnan(t.astype(float)); tt = t[ok].astype(int); b = DayBootstrap(df.loc[te_mask, "date"].values[ok], B=1000)
    th_res[label] = {a: auc_stat(tt, res[a].loc[te_mask].values[ok]) for a in arms}; th_res[label]["Full+S11 minus Full"] = b.ci_diff(auc_stat, tt, res["Full"].loc[te_mask].values[ok], res["Full+S11"].loc[te_mask].values[ok]); th_res[label]["All9+S11 minus All9"] = b.ci_diff(auc_stat, tt, res["All9"].loc[te_mask].values[ok], res["All9+S11"].loc[te_mask].values[ok])
    print("S11", label, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in th_res[label].items()}, flush=True)
out["theme_features_S11"] = th_res; dump(out, "news_analysis.json")
# ---------- (4) revenue headlines ----------
rv = tagged[tagged.title.str.contains("營收")].copy(); yoy = rv.title.str.extract(r"年(增|減)(?:率)?[^\d\-]{0,6}(\d+(?:\.\d+)?)\s*[%％]")
rv["yoy"] = np.where(yoy[0] == "增", 1, -1) * pd.to_numeric(yoy[1], errors="coerce"); rv = rv.dropna(subset=["yoy"]); e = rv.groupby(["stock_id", "date"]).yoy.mean().reset_index().merge(base, on=["stock_id", "date"], how="inner")
e["bucket"] = pd.cut(e.yoy, [-np.inf, -20, 0, 20, 50, np.inf], labels=["YoY < -20 %", "-20..0 %", "0..20 %", "20..50 %", "> 50 %"]); rvr = {"n_events": int(len(e)), "share_positive_yoy": float((e.yoy > 0).mean())}
for bkt, g in e.groupby("bucket", observed=True):
    b = DayBootstrap(g.date.values, B=500); rvr[str(bkt)] = dict(n=int(len(g)), same_day_ret=float(g.ret1.mean()), next_mean=float(g.ret_next.mean()), next_mean_ci=b.ci(lambda x: float(x.mean()), g.ret_next.values), up_rate=float(g.y_up.mean()), four=four_shares(g.y4))
rvr["spearman_yoy_vs_next_return"] = float(e[["yoy", "ret_next"]].corr(method="spearman").iloc[0, 1]); rvr["spearman_yoy_vs_same_day_return"] = float(e[["yoy", "ret1"]].corr(method="spearman").iloc[0, 1])
out["revenue_headlines"] = rvr; print("revenue", {k: (v if not isinstance(v, dict) else (v["n"], round(v["same_day_ret"] * 100, 2), round(v["next_mean"] * 100, 3), round(v["up_rate"], 3))) for k, v in rvr.items()}, flush=True); dump(out, "news_analysis.json")
# ---------- figure ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(22, 5), gridspec_kw={"width_ratios": [1.5, 1.1, 1, 1]})
ks = sorted(ev, key=lambda k: ev[k]["next_mean"]); yv = np.arange(len(ks)); a = ax[0]
a.barh(yv - 0.2, [ev[k]["same_day_ret_mean"] * 100 for k in ks], 0.4, color="#bbb", label="same-day return (what the headline reports)"); a.barh(yv + 0.2, [ev[k]["next_mean"] * 100 for k in ks], 0.4, color="#2a78d6", xerr=[[(ev[k]["next_mean"] - ev[k]["next_mean_ci"][0]) * 100 for k in ks], [(ev[k]["next_mean_ci"][1] - ev[k]["next_mean"]) * 100 for k in ks]], capsize=2, label="next-day return (95 % CI)")
a.axvline(BASE["next_mean"] * 100, color="#d7263d", ls="--", lw=0.8, label="all stock-days"); a.set_yticks(yv); a.set_yticklabels([f"{k} (n={ev[k]['n_events']})" for k in ks], fontsize=7.5); a.set_xlabel("%"); a.legend(fontsize=7, frameon=False); a.set_title("(a) Keyword events in stock-tagged headlines, 2019–2026", fontsize=9.5)
a = ax[1]; tn = list(th); x = np.arange(len(tn)); a.bar(x - 0.2, [th[k]["abs_next_return"]["members_high"] * 100 for k in tn], 0.4, color="#eb6834", label="concept members, high-intensity days"); a.bar(x + 0.2, [th[k]["abs_next_return"]["members_normal"] * 100 for k in tn], 0.4, color="#f7c6b3", label="concept members, other days")
a.set_xticks(x); a.set_xticklabels(tn, rotation=35, ha="right", fontsize=7.5); a.set_ylabel("mean |next-day return| (%)"); a.legend(fontsize=7, frameon=False); a.set_title("(b) Theme-news intensity and the size of the next-day move", fontsize=9.5)
a = ax[2]; bk = [k for k in rvr if isinstance(rvr[k], dict)]; x = np.arange(len(bk)); a.bar(x - 0.2, [rvr[k]["up_rate"] for k in bk], 0.4, color="#2a78d6", label="next-day up-rate"); a.bar(x + 0.2, [rvr[k]["four"]["strong_up"] for k in bk], 0.4, color="#1baf7a", label="strong-up share (four-level)")
a.axhline(BASE["up_rate"], color="#d7263d", ls="--", lw=0.8); a.set_xticks(x); a.set_xticklabels([f"{k}\n(n={rvr[k]['n']})" for k in bk], fontsize=7.5); a.legend(fontsize=7, frameon=False); a.set_title("(c) Monthly-revenue headlines: YoY bucket vs next day", fontsize=9.5)
a = ax[3]; yrs = list(uni); feats = ["n_news", "market_n_news", "vol20"]; x = np.arange(len(yrs)); w = 0.25
for j, f in enumerate(feats): a.bar(x + (j - 1) * w, [uni[y][f]["auroc_magnitude"] for y in yrs], w, label=f)
a.axhline(0.5, color="#999", lw=0.8); a.set_xticks(x); a.set_xticklabels(yrs); a.set_ylim(0.45, 0.75); a.set_ylabel("AUROC for |r_next| > k"); a.legend(fontsize=7, frameon=False); a.set_title("(d) News count predicts the size of tomorrow's move (univariate)", fontsize=9.5)
plt.tight_layout(); plt.savefig(FIG / "f18_news_analysis.png", dpi=150); print("figure saved"); print("done")
