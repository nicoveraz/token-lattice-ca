"""Phase B rigor (F23): the finite-size Lyapunov exponent lambda -- the CA-canonical
chaos measure -- from the early exponential growth of the CRN twin separation. lambda
= slope of log(number of damaged sites) vs sweep over the ballistic window. lambda>0
= perturbation amplified (chaos / edge-of-chaos); lambda<=0 = healed. Tests the
edge-of-chaos frame quantitatively: does lambda climb toward positive with model
CAPACITY (tiny -> mini -> base)? Usage: lyapunov.py --backend mlm --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, argparse, json, time
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from mlm_lib import RESDIR, ensure_resdir

RS = [1, 2, 4, 8, 16]
TS = [0.7, 0.9]
SEEDS = [21, 22]


def lyap_from_cone(cone, N):
    """cone (sweeps+1, N) per-site damage prob -> finite-size Lyapunov (per sweep)."""
    d = np.maximum(cone.sum(axis=1), 1e-6)               # expected damaged sites
    dmax = d.max()
    if dmax >= 3.5:                                       # grew beyond the ~3-site seed
        end = int(np.argmax(d >= 0.5 * dmax)); end = max(3, min(end, len(d) - 1))
    else:                                                 # healed: fit the early decay
        end = min(8, len(d) - 1)
    ts = np.arange(end + 1)
    lam = float(np.polyfit(ts, np.log(d[:end + 1]), 1)[0])
    return lam, float(dmax / N)


def main(backend, tag, B, N, sweeps):
    ensure_resdir()
    if backend == "mlm":
        from mlm_ca import MLMRule
        from mlm_damage import block_damage
        from mlm_lib import MODELS
        rule = MLMRule(MODELS[tag]); scheme = "cls_sep"
        key = tag
    else:
        from ar_probe import block_damage, MODELS
        from ar_ca import ARRule
        rule = ARRule(MODELS[tag]); scheme = "none"
        key = tag
    res = {"backend": backend, "model": tag, "RS": RS, "TS": TS}
    lam = {r: {} for r in RS}
    t0 = time.time()
    for r in RS:
        for T in TS:
            ls = []
            for sd in SEEDS:
                d = block_damage(rule, T, r, block=3, B=B, N=N, settle=12,
                                 sweeps=sweeps, seed=sd, scheme=scheme)
                ls.append(lyap_from_cone(d["cone"], N)[0])
            lam[r][T] = round(float(np.mean(ls)), 4)
            print(f"[{key}] lambda r={r:>2} T={T}: {lam[r][T]:+.4f}/sweep", flush=True)
    res["lambda"] = {str(r): {str(T): lam[r][T] for T in TS} for r in RS}
    res["lambda_max"] = round(float(max(lam[r][T] for r in RS for T in TS)), 4)
    res["lambda_max_at"] = [(r, T) for r in RS for T in TS if lam[r][T] == max(lam[r2][T2] for r2 in RS for T2 in TS)][0]
    json.dump(res, open(f"{RESDIR}/lyapunov_{backend}_{tag}.json", "w"), indent=1)
    print(f"[{key}] lambda_max={res['lambda_max']:+.4f} at r,T={res['lambda_max_at']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlm", choices=["mlm", "ar"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=25)
    a = ap.parse_args()
    main(a.backend, a.model, a.B, a.N, a.sweeps)
