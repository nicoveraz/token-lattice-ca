"""Phase 2 hardening: T x r phase sweep with >=5 independent seeds per condition,
reporting mean +/- std error bars on the order parameter and activity. Each seed
is a fully independent replica (different init, update order, and uniform stream).

Word-level by default; pass --bpe to drive the BPE model (ckpt_bpe, data_bpe).
Writes results/sweep_multiseed[_bpe].jsonl.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time, argparse
import numpy as np
import ca
from model import load
from ca import run, metrics

TS = [0.3, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5]     # finer near the drop than the pilot
RS = [1, 2, 4, 8, 16]
SEEDS = [11, 22, 33, 44, 55]                   # 5 independent replicas
B, N, SWEEPS = 16, 48, 100


def condition(params, r, T):
    bo, ac, di = [], [], []
    for sd in SEEDS:
        out = run(params, B=B, N=N, r=r, T=T, sweeps=SWEEPS, mode="async",
                  init="random", seed=sd)
        m = metrics(out["final"])
        bo.append(m["bigram_overlap"]); di.append(m["distinct"])
        ac.append(float(out["activity"][-10:].mean()))
    a = np.array
    return dict(r=r, T=T, n_seeds=len(SEEDS),
                bigram_mean=float(a(bo).mean()), bigram_std=float(a(bo).std()),
                act_mean=float(a(ac).mean()), act_std=float(a(ac).std()),
                distinct_mean=float(a(di).mean()), distinct_std=float(a(di).std()),
                bigram_seeds=[round(x, 4) for x in bo])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", action="store_true")
    args = ap.parse_args()
    if args.bpe:
        ca.DATA_DIR, ca.VOCAB, ca.INIT_LO = "data_bpe", 4096, 1
        params = load("ckpt_bpe/final.npz")
        out_path = "results/sweep_multiseed_bpe.jsonl"
    else:
        params = load("ckpt/final.npz")
        out_path = "results/sweep_multiseed.jsonl"
    open(out_path, "w").close()
    t0 = time.time()
    for r in RS:
        for T in TS:
            rec = condition(params, r, T)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"r={r} T={T}: order={rec['bigram_mean']:.3f}+/-{rec['bigram_std']:.3f} "
                  f"act={rec['act_mean']:.3f}+/-{rec['act_std']:.3f}", flush=True)
    print(f"MULTISEED SWEEP DONE ({time.time()-t0:.0f}s)", flush=True)
