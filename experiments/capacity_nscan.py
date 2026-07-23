"""Finite-size check of the capacity claim (F23): is tiny << mini (the significant part)
N-robust? Measures D_norm at the discriminating radius r=2, T=0.7, for tiny/mini/base at
N in {48,96}, 2 seeds. Cheap (r=2 windows are small). Usage: capacity_nscan.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time
import numpy as np
from mlm_ca import MLMRule
from mlm_damage import block_damage, drift_floor
from mlm_lib import MODELS, RESDIR, ensure_resdir

NS, SEEDS, R, T = [48, 96], [21, 22], 2, 0.7


def main():
    ensure_resdir()
    res = {"r": R, "T": T, "NS": NS}
    for tag in ["tiny", "mini", "base"]:
        rule = MLMRule(MODELS[tag])
        res[tag] = {}
        for N in NS:
            sw = max(25, int(N / 8) + 10)
            dn = []
            for sd in SEEDS:
                d = block_damage(rule, T, R, block=3, B=24, N=N, settle=12, sweeps=sw, seed=sd)
                d0, _ = drift_floor(rule, T, R, B=24, N=N, settle=12, sweeps=sw, seed=sd)
                dn.append(d["mean_damage"] / max(d0, 1e-3))
            res[tag][f"N{N}"] = round(float(np.mean(dn)), 4)
            print(f"[{tag}] N={N}: D_norm(r=2,T=0.7)={res[tag][f'N{N}']:.3f}", flush=True)
        del rule
    json.dump(res, open(f"{RESDIR}/capacity_nscan.json", "w"), indent=1)
    print("verdict per N: tiny < mini < base ?")
    for N in NS:
        vals = {t: res[t][f"N{N}"] for t in ["tiny", "mini", "base"]}
        print(f"  N={N}: {vals}  tiny<mini={vals['tiny']<vals['mini']}")
    print("CAPACITY NSCAN DONE")


if __name__ == "__main__":
    main()
