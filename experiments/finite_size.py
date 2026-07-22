"""Phase 2 hardening: finite-size scan at the transition.

N in {48, 96, 192}, r=2, fine T grid across the order-parameter drop, >=5 seeds.
Question (matters for any 'phase transition' claim): does the order-parameter
drop SHARPEN with N (-> genuine transition, correlation length diverging) or stay
the same width (-> smooth crossover)?

Signatures reported per N:
  order_mean(T)      the drop itself
  susceptibility(T)  variance of the per-lattice order parameter across the
                     ensemble; at a continuous transition this peaks at Tc and
                     the peak grows/sharpens with N.
  width_75_25        T-interval over which order goes 0.75 -> 0.25 (interp).
                     Shrinking with N => sharpening.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time
import numpy as np
import ca
from model import load
from ca import run

NS = [48, 96, 192]
TS = [0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
SEEDS = [11, 22, 33, 44, 55]
R, B, SWEEPS, TAIL = 2, 16, 120, 20


def per_lattice_order(lat):
    """Fraction of ring bigrams present in the corpus, per lattice (B,)."""
    bi = ca.corpus_bigrams()
    out = []
    for b in range(lat.shape[0]):
        pairs = list(zip(lat[b, :-1].tolist(), lat[b, 1:].tolist())) + \
                [(int(lat[b, -1]), int(lat[b, 0]))]
        out.append(np.mean([pr in bi for pr in pairs]))
    return np.array(out)


def order_at(params, N, T):
    """Pool the per-lattice order parameter over all seeds x B, averaged over the
    last TAIL snapshots (quasi-stationary)."""
    vals = []
    for sd in SEEDS:
        out = run(params, B=B, N=N, r=R, T=T, sweeps=SWEEPS, mode="async",
                  init="random", seed=sd, record_every=1)
        for s in out["snaps"][-TAIL:]:
            vals.append(per_lattice_order(s))
    v = np.concatenate(vals)                              # (TAIL*SEEDS*B,)
    return v


def width_75_25(ts, order):
    """T interval where order crosses 0.75 down to 0.25 (linear interpolation)."""
    ts, order = np.asarray(ts), np.asarray(order)
    def cross(level):
        for i in range(len(ts) - 1):
            a, b = order[i], order[i + 1]
            if (a - level) * (b - level) <= 0 and a != b:
                f = (a - level) / (a - b)
                return ts[i] + f * (ts[i + 1] - ts[i])
        return float("nan")
    return cross(0.25) - cross(0.75)


if __name__ == "__main__":
    params = load("ckpt/final.npz")
    res = {}
    t0 = time.time()
    for N in NS:
        om, sus = [], []
        for T in TS:
            v = order_at(params, N, T)
            om.append(float(v.mean())); sus.append(float(v.var()))
            print(f"N={N:3d} T={T}: order={v.mean():.3f} chi(var)={v.var():.4f}", flush=True)
        res[str(N)] = dict(T=TS, order_mean=om, susceptibility=sus,
                           width_75_25=round(float(width_75_25(TS, om)), 4),
                           chi_peak=round(max(sus), 5),
                           chi_peak_T=TS[int(np.argmax(sus))])
        print(f"  -> N={N}: width(0.75->0.25)={res[str(N)]['width_75_25']}  "
              f"chi_peak={res[str(N)]['chi_peak']} at T={res[str(N)]['chi_peak_T']}", flush=True)
    json.dump(res, open("results/finite_size.json", "w"), indent=1)
    print(f"FINITE-SIZE DONE ({time.time()-t0:.0f}s)", flush=True)
