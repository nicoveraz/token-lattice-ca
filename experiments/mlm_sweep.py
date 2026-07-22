"""Phase 3: T x r phase sweep for a real MLM. Records the order parameter (bigram
overlap vs WikiText proxy), k-gram overlap for k=2,3,4 (the radius test: does
longer-range corpus structure appear at larger r, or is equilibrium radius-blind
like the toy?), activity, entropy, distinct.  Usage: mlm_sweep.py --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import argparse, json, time
import numpy as np
import mlm_ca
from mlm_ca import MLMRule, run
from mlm_lib import MODELS, RESDIR, load_ref, ref_kgram_sets, kgram_overlap, order_param, ensure_resdir

TS = [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]
RS = [1, 2, 4, 8, 16]


def main(tag, B, N, sweeps, scheme):
    ensure_resdir()
    rule = MLMRule(MODELS[tag])
    ref_bi = mlm_ca.ref_bigrams(load_ref())
    ksets = ref_kgram_sets(4)
    recs, snaps_store = [], {}
    t0 = time.time()
    for r in RS:
        for T in TS:
            tc = time.time()
            out = run(rule, B=B, N=N, r=r, T=T, sweeps=sweeps, scheme=scheme,
                      init="random", seed=100 + r)
            lat = out["final"]
            op, _ = order_param(lat, ref_bi)
            ko = kgram_overlap(lat, ksets)
            B_, N_ = lat.shape
            ent = float(np.mean([-(lambda p: (p * np.log2(p)).sum())(
                np.unique(lat[b], return_counts=True)[1] / N_) for b in range(B_)]))
            rec = dict(tag=tag, r=r, T=T, scheme=scheme,
                       order=op, k2=ko[2], k3=ko[3], k4=ko[4],
                       act_final=float(out["activity"][-5:].mean()),
                       distinct=float(np.mean([len(np.unique(lat[b])) / N_ for b in range(B_)])),
                       entropy=ent, secs=round(time.time() - tc, 1))
            recs.append(rec)
            if r == 2 and T in (0.5, 0.8, 1.0, 2.0):
                snaps_store[f"snaps_T{T}"] = out["snaps"][:, 0, :].astype(np.int32)
            print(f"[{tag}] r={r:>2} T={T}: order={op:.3f} k2={ko[2]:.3f} k3={ko[3]:.3f} "
                  f"k4={ko[4]:.3f} act={rec['act_final']:.3f} ({rec['secs']}s)", flush=True)
    json.dump(recs, open(f"{RESDIR}/{tag}_sweep.json", "w"), indent=1)
    if snaps_store:
        np.savez_compressed(f"{RESDIR}/{tag}_spacetime.npz", **snaps_store)
    print(f"[{tag}] SWEEP DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--scheme", default="cls_sep")
    a = ap.parse_args()
    main(a.model, a.B, a.N, a.sweeps, a.scheme)
