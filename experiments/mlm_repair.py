"""Phase B: measure the self-correction length xi_repair.

Turns F17's binary "no healing" into a curve with a characteristic scale. For each
model, block-flip CRN twins over a (radius r, temperature T) grid, measuring the
asymptotic damage D(r,T) = fraction of sites still differing at long time (mean of
the last `tail` sweeps), with ignition probability reported separately from
conditional spread (F13), >=3 seeds.

  xi_repair(T) = the conditioning radius where D(r) crosses 0.5 (half-max), interp
                 in log2(r); a measured stability length.
  T_c(r)       = the temperature where D(T) crosses 0.5 at fixed r (the heal/spread
                 boundary); B2 shows whether it rises with r (more context -> stable
                 to higher T).
Usage: mlm_repair.py --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import argparse, json, time
import numpy as np
from mlm_ca import MLMRule
from mlm_damage import block_damage
from mlm_lib import MODELS, RESDIR, ensure_resdir

RS = [1, 2, 4, 8, 16]
TS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [21, 22, 23]


def cross(xs, ys, level=0.5):
    """First x where y crosses `level` (linear interp). None if no crossing."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if (a - level) * (b - level) <= 0 and a != b:
            f = (a - level) / (a - b)
            return float(xs[i] + f * (xs[i + 1] - xs[i]))
    return None


def main(tag, B, N, sweeps):
    ensure_resdir()
    rule = MLMRule(MODELS[tag])
    D = {r: {T: [] for T in TS} for r in RS}     # per-seed asymptotic damage
    IG = {r: {T: [] for T in TS} for r in RS}    # ignition prob
    SP = {r: {T: [] for T in TS} for r in RS}    # conditional spread
    t0 = time.time()
    for r in RS:
        for T in TS:
            for sd in SEEDS:
                d = block_damage(rule, T, r, block=3, B=B, N=N, settle=12,
                                 sweeps=sweeps, seed=sd, scheme="cls_sep", tail=8)
                D[r][T].append(d["mean_damage"]); IG[r][T].append(d["ignition_prob"])
                SP[r][T].append(d["cond_spread"])
            dm = np.mean(D[r][T])
            print(f"[{tag}] r={r:>2} T={T}: D={dm:.3f}±{np.std(D[r][T]):.3f} "
                  f"P_ign={np.mean(IG[r][T]):.2f} spread={np.mean(SP[r][T]):.2f}", flush=True)

    def grid(M):
        return {str(r): {str(T): dict(mean=round(float(np.mean(M[r][T])), 4),
                                      std=round(float(np.std(M[r][T])), 4)) for T in TS} for r in RS}
    # xi_repair(T): radius where D(r) crosses 0.5 (log2 r); T_c(r): T where D(T) crosses 0.5
    xi = {}
    for T in TS:
        ys = [np.mean(D[r][T]) for r in RS]
        c = cross(np.log2(RS), ys, 0.5)
        xi[str(T)] = round(2 ** c, 3) if c is not None else ("all_heal" if max(ys) < 0.5 else "none_heal")
    Tc = {}
    for r in RS:
        ys = [np.mean(D[r][T]) for T in TS]
        c = cross(TS, ys, 0.5)
        Tc[str(r)] = round(c, 3) if c is not None else ("all_heal" if max(ys) < 0.5 else "none_heal")
    res = dict(model=tag, RS=RS, TS=TS, seeds=SEEDS, N=N,
               D=grid(D), ignition=grid(IG), spread=grid(SP),
               xi_repair_by_T=xi, T_c_by_r=Tc)
    json.dump(res, open(f"{RESDIR}/repair_{tag}.json", "w"), indent=1)
    print(f"[{tag}] xi_repair(T)={xi}")
    print(f"[{tag}] T_c(r)={Tc}")
    print(f"[{tag}] REPAIR DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=40)
    a = ap.parse_args()
    main(a.model, a.B, a.N, a.sweeps)
