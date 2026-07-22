"""Phase A2: kill-or-confirm the damage-velocity ceiling (F16).

The 11.5 sites/sweep plateau at r=8,16 on N=48 may be finite-size wraparound (max
front half-width ~ N/2 = 24). Finite-size scan N in {48,96,192,384} at r in {4,8,16}:
  - if the plateau LIFTS with N -> velocity ~ r continues, the ceiling was an artifact
  - if it HOLDS across N -> the saturation is a real velocity bound
Velocity is model-invariant (F16), so this runs on tiny by default (fast, clean);
--model mini adds a model-invariance check. base at N=384 is skipped as too slow
(logged). Usage: phaseA_velocity.py --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import argparse, json, time
import numpy as np
from mlm_ca import MLMRule
from mlm_damage import block_damage
from mlm_lib import MODELS, RESDIR, cone_front_velocity, ensure_resdir

NS = [48, 96, 192, 384]
RS = [4, 8, 16]


def main(tag, B, sweeps, ns, rs):
    ensure_resdir()
    rule = MLMRule(MODELS[tag])
    res = {"model": tag, "B": B, "sweeps": sweeps}
    t0 = time.time()
    for N in ns:
        # give the front room+time: sweeps scaled so the cone can cross N/2 before saturating
        sw = max(sweeps, int(N / 8) + 10)
        for r in rs:
            tc = time.time()
            d = block_damage(rule, 1.0, r, block=3, B=B, N=N, settle=15, sweeps=sw, scheme="cls_sep")
            v = cone_front_velocity(d["cone"])
            res[f"N{N}_r{r}"] = dict(velocity=round(v["velocity_sites_per_sweep"], 2),
                                     saturate_sweep=v["saturate_sweep"],
                                     mean_damage=round(d["mean_damage"], 3),
                                     N_over_2=N // 2, sweeps=sw)
            print(f"[{tag}] N={N:>3} r={r:>2}: v={v['velocity_sites_per_sweep']:.2f} sites/sweep "
                  f"(N/2={N//2}, sat@{v['saturate_sweep']}) ({time.time()-tc:.0f}s)", flush=True)
    json.dump(res, open(f"{RESDIR}/phaseA_velocity_{tag}.json", "w"), indent=1)
    print(f"[{tag}] PHASE-A VELOCITY DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", choices=list(MODELS))
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--sweeps", type=int, default=30)
    ap.add_argument("--ns", default="48,96,192,384")
    ap.add_argument("--rs", default="4,8,16")
    a = ap.parse_args()
    main(a.model, a.B, a.sweeps, [int(x) for x in a.ns.split(",")],
         [int(x) for x in a.rs.split(",")])
