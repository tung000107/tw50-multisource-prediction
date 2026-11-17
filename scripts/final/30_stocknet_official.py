"""Official StockNet (Xu & Cohen 2018) run on its own dataset — bookkeeping.
'count'  (run with the tf1 env python, from external/stocknet-code/src): count the samples the released DataPipe actually yields
         per phase after its filters (|movement| band drop + every day in the window must have >= 1 tweet) and the test up-share
         -> data/processed/stocknet_official_counts.json
'collect' (project .venv): parse results/final/log_30_stocknet_official_<variant>_<run>.txt (dev acc trajectory, epoch train acc,
         TEST_RESULT) -> results/final/external_runs.json['stocknet_official_acl18']"""
import sys, json, re, glob, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; RES = ROOT / "results/final"; CNT = ROOT / "data/processed/stocknet_official_counts.json"
if sys.argv[1] == "count":
    import numpy as np; sys.path.insert(0, os.getcwd())
    from DataPipe import DataPipe
    p = DataPipe(); out = {}
    for ph in ("train", "dev", "test"):
        n = 0; up = 0
        for b in p.batch_gen_by_stocks(ph):
            n += b["batch_size"]; T = b["T_batch"]; y = b["y_batch"]; up += sum(float(y[i, T[i] - 1, 1]) for i in range(b["batch_size"]))
        out[ph] = dict(n=int(n), up_share=up / n)
    out["note"] = "samples yielded by the released DataPipe (batch_gen_by_stocks, no partial-batch drop); filters: movement in [-0.5%, 0.55%) dropped, window days without tweets dropped"
    json.dump(out, open(CNT, "w"), indent=1); print(json.dumps(out, indent=1))
else:
    cnt = json.load(open(CNT)); runs = {}
    for lg in sorted(glob.glob(str(RES / "log_30_stocknet_official_*_r*.txt"))):
        m = re.search(r"official_(\w+?)_(r\d+)\.txt", lg); v, r = m.group(1), m.group(2); txt = open(lg).read()
        dev = [float(x) for x in re.findall(r"Eval, eval loss: [\d.]+, acc: ([\d.]+)\s*$", txt, re.M)]  # dev evals only (the final test eval line carries ", mcc: ...")
        ep = [float(x) for x in re.findall(r"Epoch: loss: [\d.]+, acc: ([\d.]+)", txt)]
        t = re.search(r"TEST_RESULT \{'loss': ([\d.]+), 'acc': ([\d.]+), 'mcc': ([-\d.e]+|None)\}", txt); ex = re.search(r"^exit (\d+)", txt, re.M)
        if not t: continue
        runs.setdefault(v, {})[r] = dict(test_acc=float(t.group(2)), test_mcc=None if t.group(3) == "None" else float(t.group(3)), test_loss=float(t.group(1)), n_epochs_done=len(ep), last_train_epoch_acc=ep[-1] if ep else None, dev_acc_last=dev[-1] if dev else None, dev_acc_max=max(dev) if dev else None, exit_code=int(ex.group(1)) if ex else None)
    import numpy as np
    for v in runs:
        rs = [runs[v][r] for r in runs[v] if r != "mean"]
        mc = [x["test_mcc"] for x in rs if x["test_mcc"] is not None]
        runs[v]["mean"] = dict(test_acc=float(np.mean([x["test_acc"] for x in rs])), test_mcc=float(np.mean(mc)) if mc else None, n_runs=len(rs), n_mcc_defined=len(mc),
                               note="test_mcc=None means the run predicted a single class on the whole test set (MCC undefined); mean MCC is over the defined runs only")
    XR = json.load(open(RES / "external_runs.json"))
    XR["stocknet_official_acl18"] = dict(runs, counts=cnt, test_majority_acc=max(cnt["test"]["up_share"], 1 - cnt["test"]["up_share"]),
        published={"HedgeFundAnalyst": {"acc": 0.5823, "mcc": 0.080796}, "FundamentalAnalyst": {"acc": 0.5823, "mcc": 0.071704}, "TechnicalAnalyst": {"acc": 0.5496, "mcc": 0.016456}, "IndependentAnalyst": {"acc": 0.5754, "mcc": 0.036610}, "DiscriminativeAnalyst": {"acc": 0.5615, "mcc": 0.056493}, "HAN": {"acc": 0.5764, "mcc": 0.0518}, "RAND": {"acc": 0.5089, "mcc": -0.002266}, "paper_sample_counts": {"train": 20339, "dev": 2555, "test": 3720}, "source": "Xu & Cohen 2018, Table 1 and Sec. 3 (verified from the PDF, homepages.inf.ed.ac.uk/scohen/acl18stock.pdf)"},
        note="official code (external/stocknet-code) + official dataset (external/stocknet-dataset), paper config.yml unchanged (hedge = HedgeFundAnalyst, alpha 0.5, 15 epochs, last checkpoint tested as in Main.py); Python-3 ports: file()->open, range->list, PEP479 return, 'is not'->'!='; our additions: MCC on test, per-run dirs, CPU thread cap. No seed in the code: runs differ by shuffle/init.")
    json.dump(XR, open(RES / "external_runs.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(XR["stocknet_official_acl18"], indent=1)[:3000])
