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
from mlm_damage import block_damage, drift_floor
from mlm_lib import MODELS, RESDIR, ensure_resdir

RS = [1, 2, 4, 8, 16]
TS = [0.5, 0.7, 0.9]              # drop 0.3 (all-heal); runs doubled by the drift floor
SEEDS = [21, 22]


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
    M = {k: {r: {T: [] for T in TS} for r in RS}
         for k in ("D", "D0", "Dn", "distinct", "ig", "sp")}
    t0 = time.time()
    for r in RS:
        for T in TS:
            for sd in SEEDS:
                d = block_damage(rule, T, r, block=3, B=B, N=N, settle=12,
                                 sweeps=sweeps, seed=sd, scheme="cls_sep", tail=8)
                d0, dist = drift_floor(rule, T, r, B=B, N=N, settle=12,
                                       sweeps=sweeps, seed=sd, scheme="cls_sep", tail=8)
                M["D"][r][T].append(d["mean_damage"]); M["D0"][r][T].append(d0)
                M["Dn"][r][T].append(d["mean_damage"] / max(d0, 1e-3))
                M["distinct"][r][T].append(dist)
                M["ig"][r][T].append(d["ignition_prob"]); M["sp"][r][T].append(d["cond_spread"])
            print(f"[{tag}] r={r:>2} T={T}: D={np.mean(M['D'][r][T]):.3f} "
                  f"D0(floor)={np.mean(M['D0'][r][T]):.3f} Dnorm={np.mean(M['Dn'][r][T]):.3f} "
                  f"distinct={np.mean(M['distinct'][r][T]):.2f}", flush=True)

    def grid(k):
        return {str(r): {str(T): dict(mean=round(float(np.mean(M[k][r][T])), 4),
                                      std=round(float(np.std(M[k][r][T])), 4)) for T in TS} for r in RS}

    def crossings(key, lv):     # xi_repair(T) [radius] and T_c(r) [temperature]
        xi, Tc = {}, {}
        for T in TS:
            ys = [np.mean(M[key][r][T]) for r in RS]
            c = cross(np.log2(RS), ys, lv)
            xi[str(T)] = round(2 ** c, 3) if c is not None else ("all_below" if max(ys) < lv else "none")
        for r in RS:
            ys = [np.mean(M[key][r][T]) for T in TS]
            c = cross(TS, ys, lv)
            Tc[str(r)] = round(c, 3) if c is not None else ("all_below" if max(ys) < lv else "none")
        return xi, Tc

    # raw D uses 0.5; normalized D_norm uses 0.5 (half of full decorrelation)
    xi_D, Tc_D = crossings("D", 0.5)
    xi_Dn, Tc_Dn = crossings("Dn", 0.5)
    res = dict(model=tag, RS=RS, TS=TS, seeds=SEEDS, N=N,
               D=grid("D"), D0_floor=grid("D0"), D_norm=grid("Dn"),
               distinct=grid("distinct"), ignition=grid("ig"), spread=grid("sp"),
               xi_repair_rawD=xi_D, T_c_rawD=Tc_D,
               xi_repair_normD=xi_Dn, T_c_normD=Tc_Dn)
    json.dump(res, open(f"{RESDIR}/repair_{tag}.json", "w"), indent=1)
    print(f"[{tag}] raw   xi(T)={xi_D}  T_c(r)={Tc_D}")
    print(f"[{tag}] norm  xi(T)={xi_Dn}  T_c(r)={Tc_Dn}")
    print(f"[{tag}] REPAIR DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=40)
    a = ap.parse_args()
    main(a.model, a.B, a.N, a.sweeps)
