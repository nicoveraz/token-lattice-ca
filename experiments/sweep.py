"""Coarse phase sweep: T x r (async, random init) + one sync row at r=2."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time, sys
import numpy as np
from model import load
from ca import run, metrics

TS = [0.3, 0.7, 1.0, 1.5, 2.5]
RS = [1, 2, 4, 8, 16]
B, N, SWEEPS = 8, 48, 120

def one(params, r, T, mode, seed):
    t0 = time.time()
    out = run(params, B=B, N=N, r=r, T=T, sweeps=SWEEPS, mode=mode,
              init="random", seed=seed)
    per_sweep = [metrics(s) for s in out["snaps"]]
    act = out["activity"]  # (sweeps, B)
    frozen = float((act[-10:].mean(axis=0) == 0).mean())
    rec = dict(r=r, T=T, mode=mode,
               act_final=float(act[-10:].mean()),
               entropy_final=per_sweep[-1]["entropy"],
               distinct_final=per_sweep[-1]["distinct"],
               bigram_final=per_sweep[-1]["bigram_overlap"],
               frozen_frac=frozen, secs=round(time.time() - t0, 1))
    np.savez_compressed(f"results/sweep_{mode}_r{r}_T{T}.npz",
                        snaps=out["snaps"].astype(np.int16),
                        activity=act,
                        entropy=[m["entropy"] for m in per_sweep],
                        distinct=[m["distinct"] for m in per_sweep],
                        bigram=[m["bigram_overlap"] for m in per_sweep])
    with open("results/summary.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

if __name__ == "__main__":
    params = load("ckpt/final.npz")
    for r in RS:
        for T in TS:
            one(params, r, T, "async", seed=100 + r)
    for T in TS:
        one(params, 2, T, "sync", seed=200)
    print("SWEEP DONE", flush=True)
