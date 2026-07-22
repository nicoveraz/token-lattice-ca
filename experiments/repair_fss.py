"""Phase B rigor: finite-size check of the D_norm(r) trend (the deferred N-scan). If
the diversity-controlled r-trend (and tiny's r=16 dip) is a ring-geometry artifact it
must change with N; if it holds across N it is a real conditioning-radius effect. Runs
on tiny (the trend is geometric, so one model suffices; tiny is where the dip appeared).
Usage: repair_fss.py --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import argparse, json, time
import numpy as np
from mlm_ca import MLMRule
from mlm_damage import block_damage, drift_floor
from mlm_lib import MODELS, RESDIR, ensure_resdir

NS = [48, 96, 192]
RS = [2, 4, 8, 16]
SEEDS = [21, 22]
T = 0.7


def main(tag, B):
    ensure_resdir()
    rule = MLMRule(MODELS[tag])
    res = {"model": tag, "T": T, "NS": NS, "RS": RS}
    t0 = time.time()
    for N in NS:
        sw = max(25, int(N / 8) + 10)
        prof = {}
        for r in RS:
            dn = []
            for sd in SEEDS:
                d = block_damage(rule, T, r, block=3, B=B, N=N, settle=12,
                                 sweeps=sw, seed=sd, scheme="cls_sep", tail=8)
                d0, _ = drift_floor(rule, T, r, B=B, N=N, settle=12, sweeps=sw,
                                    seed=sd, scheme="cls_sep", tail=8)
                dn.append(d["mean_damage"] / max(d0, 1e-3))
            prof[str(r)] = round(float(np.mean(dn)), 4)
            print(f"[{tag}] N={N:>3} r={r:>2}: D_norm={prof[str(r)]:.3f}", flush=True)
        res[f"N{N}"] = prof
    json.dump(res, open(f"{RESDIR}/repair_fss_{tag}.json", "w"), indent=1)
    print(f"[{tag}] REPAIR-FSS DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", choices=list(MODELS))
    ap.add_argument("--B", type=int, default=24)
    a = ap.parse_args()
    main(a.model, a.B)
