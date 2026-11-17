"""Numbers requested by the independent review: (a) news coverage of test stock-days; (b) fusion with the new sources vs U.S.-only
(Full+S7+S8 and All9 vs S4-only, pooled CI and per-year AUROC) -> source_study.json['fusion_vs_us_only']; (c) per-year AUROC of the
label variants (y_gap / y_intra / y_rel / y_tr) for Full vs S4-only -> labels.json['binary_variants'][label]['by_year']."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from common import *
exec(open(Path(__file__).parent / "02_tabular.py").read().split("if __name__")[0])
ex = pd.read_parquet(ROOT / "data/processed/extra_features.parquet"); df = df.merge(ex, on=["date", "stock_id"], how="left")
nf = pd.read_parquet(ROOT / "data/processed/news_features.parquet"); df = df.merge(nf, on=["date", "stock_id"], how="left")
NEW = {"S6": ["n_news", "n_news_5d", "has_news", "log_news", "market_n_news", "text_score", "market_text_score"], "S7": ["sector_ret1_ex", "sector_ret5_ex", "leader_ret1", "leader_ret5", "ret1_rank", "xs_dispersion", "ret1_minus_sector"],
       "S8": ["tx_night_ret", "tx_night_volume_ratio", "tx_day_basis", "foreign_fut_net_oi", "foreign_fut_net_oi_chg", "dealer_fut_net_oi", "dealer_fut_net_oi_chg", "total_margin_chg", "total_short_chg"], "S9": ["daytrade_share", "daytrade_share_5d", "lend_log", "lend_log_5d", "rev_yoy", "rev_mom"]}
def run(feats, label="y_up"):
    prob = pd.Series(np.nan, index=df.index)
    for yr, tr, itr, iva, te in FOLDS:
        d = df.copy(); d["text_score"] = d[f"text_score_{yr}"]; d["market_text_score"] = d[f"market_text_score_{yr}"]
        m = lgb.LGBMClassifier(n_estimators=3000, random_state=42, **LGB_PARAMS).fit(d.loc[itr, feats], d.loc[itr, label], eval_set=[(d.loc[iva, feats], d.loc[iva, label])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        f = lgb.LGBMClassifier(n_estimators=int(m.best_iteration_ or 1), random_state=42, **LGB_PARAMS).fit(d.loc[tr, feats], d.loc[tr, label]); prob.loc[te] = f.predict_proba(d.loc[te, feats])[:, 1]
    return prob
te = df.year.isin(TEST_YEARS); SS = json.load(open(RES / "source_study.json")); SS["coverage"] = dict(share_stock_days_with_news=float((df.loc[te, "n_news"] > 0).mean()), share_all_stock_days_with_news=float((df.n_news > 0).mean()))
arms = {"Full": ALL, "S4-only": SRC["S4"], "Full+S7+S8": ALL + NEW["S7"] + NEW["S8"], "All9": ALL + sum(NEW.values(), [])}
res = dict(Parallel(n_jobs=4, prefer="threads")(delayed(lambda a: (a, run(arms[a])))(a) for a in arms)); y = df.loc[te, "y_up"].values; bs = DayBootstrap(df.loc[te, "date"].values, B=1000)
fv = {a: dict(auroc=auc_stat(y, res[a].loc[te].values), by_year={str(yr): auc_stat(df.loc[te & (df.year == yr), "y_up"].values, res[a].loc[te & (df.year == yr)].values) for yr in TEST_YEARS}) for a in arms}
for a in ["Full+S7+S8", "All9"]:
    fv[f"{a} minus S4-only"] = bs.ci_diff(auc_stat, y, res["S4-only"].loc[te].values, res[a].loc[te].values); fv[f"{a} minus S4-only same-sign years"] = int(sum(np.sign(fv[a]["by_year"][str(yr)] - fv["S4-only"]["by_year"][str(yr)]) == np.sign(fv[a]["auroc"] - fv["S4-only"]["auroc"]) for yr in TEST_YEARS))
    fv[f"{a} minus Full"] = bs.ci_diff(auc_stat, y, res["Full"].loc[te].values, res[a].loc[te].values)
SS["fusion_vs_us_only"] = fv; dump(SS, "source_study.json"); print(json.dumps(fv, indent=0)[:1500])
L = json.load(open(RES / "labels.json")); lab = pd.read_parquet(RES / "oos_labels.parquet"); d0 = load()
for v in ["y_gap", "y_intra", "y_rel", "y_tr"]:
    m = lab.merge(d0[["date", "stock_id", v]], on=["date", "stock_id"], how="left"); by = {}
    for yr in TEST_YEARS:
        mm = m[(m.year == yr) & m[v].notna()]; by[str(yr)] = {a: auc_stat(mm[v].values, mm[f"{v}|{a}"].values) for a in ["Full", "S4-only", "TW-only"]}
    L["binary_variants"][v]["by_year"] = by; L["binary_variants"][v]["Full_minus_S4_same_sign_years"] = int(sum(np.sign(by[str(yr)]["Full"] - by[str(yr)]["S4-only"]) == np.sign(L["binary_variants"][v]["Full"]["auroc"] - L["binary_variants"][v]["S4-only"]["auroc"]) for yr in TEST_YEARS))
    print(v, by, L["binary_variants"][v]["Full_minus_S4_same_sign_years"])
dump(L, "labels.json"); print("done")
