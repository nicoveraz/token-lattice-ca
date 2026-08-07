"""Advance two damage twins in LOCKSTEP so a coupling can be chosen. (#coupling ladder, rung 1)

WHY THIS CANNOT BE A SAMPLER. `lattice.run` advances one lattice and exposes `sampler(probs, u)`,
which sees ONE replica's distribution. The production CRN protocol runs the twins as two SEPARATE
`run()` calls sharing a `u_stream`, and that is exactly what makes monotone coupling expressible:
each twin applies inverse-CDF to its own conditional against the same uniform, independently. A
maximal coupling cannot be written that way -- its draw is a function of BOTH twins' distributions
at the same site at the same moment -- so the twins have to be advanced together.

WHY `lattice.run` IS NOT MODIFIED. It is the single shared loop, and a golden-file regression
asserts it stays bit-identical. Adding a paired mode there would put every existing result behind a
change made for one experiment. This mirrors its `mode="async", order="shared"` path instead, and
then EARNS the right to be trusted by reproducing it -- see `rung1_reproduces_production`.

WHAT MUST HOLD, AND WHY IT IS A REAL TEST DESPITE LOOKING TRIVIAL. Under the monotone coupler this
loop should reproduce the production twin protocol EXACTLY -- not approximately, and not up to a
tolerance. Both consume the same `u_stream` in the same order, both draw the same permutation from
the same seed, and inverse-CDF against a shared uniform is a deterministic function of (probs, u).
So a mismatch of even one cell means the paired loop consumes randomness in a different order or
slices the window differently, and every subsequent comparison against a maximal arm would be
measuring that bug rather than the coupling. It looks trivial because it should hold by
construction; it is worth asserting because "should hold by construction" is what F45 and F46 both
said.

Usage:
    .venv/bin/python experiments/paired_runner.py     # runs rung 1 on the toy backend
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"), str(_ROOT / "gatecheck" / "src")]

import numpy as np


def paired_run(rule, coupler, *, B, N, r, T, sweeps, state_a, state_b, seed, u_stream,
               extra_uniforms=None):
    """Advance two lattices in lockstep under `coupler`. Mirrors lattice.run async/shared order.

    `coupler(p, q, u, *extra)` returns (x, y): the tokens drawn for twin A and twin B at this site.
    A coupler needing more than one uniform per site takes them from `extra_uniforms`, which is
    consumed in the same lockstep so a run stays reproducible from its seeds -- as a PAIR, which is
    the property maximal coupling costs and monotone coupling does not.

    Returns (snaps_a, snaps_b), each (sweeps+1, B, N), matching lattice.run's `snaps`.
    """
    rng = np.random.default_rng(seed)
    a = np.array(state_a, copy=True)
    b = np.array(state_b, copy=True)
    snaps_a, snaps_b = [a.copy()], [b.copy()]
    ui = 0
    xi = 0

    for _ in range(sweeps):
        # The permutation is drawn from `rng` exactly as lattice.run does, and ONCE -- both twins
        # are visited in the same order, which is what CRN requires and what two separate run()
        # calls achieve by sharing a seed.
        for i in rng.permutation(N):
            idx = rule.window(i, r, N)
            u = u_stream[ui:ui + B]; ui += B
            # NO dtype coercion. The production sampler cumsums whatever `rule.probs` returns,
            # which is float32 on the JAX backend. Forcing float64 here changes the rounding of the
            # CDF and moves `searchsorted` across bin boundaries -- 163 of 1296 cells, when this
            # was first written. The coupler must see exactly the array the production path sees.
            p = np.asarray(rule.probs(a[:, idx], T))
            q = np.asarray(rule.probs(b[:, idx], T))
            if extra_uniforms is None:
                x, y = coupler(p, q, u)
            else:
                e = extra_uniforms[:, xi:xi + B]; xi += B
                x, y = coupler(p, q, u, *e)
            a[:, i] = x
            b[:, i] = y
        snaps_a.append(a.copy())
        snaps_b.append(b.copy())
    return np.array(snaps_a), np.array(snaps_b)


# ------------------------------------------------------------------ rung 1

def rung1_reproduces_production(B=6, N=24, r=2, T=0.7, sweeps=8, seed=21):
    """The paired loop under MONOTONE coupling must equal two separate run() calls, cell for cell.

    Uses the toy JAX backend, because the assertion is about the LOOP and the RNG consumption
    order, not about which model supplies the conditionals -- and the toy one is the backend whose
    golden files pin `lattice.run` in the first place.
    """
    import ca
    from model import load
    from lattice import run as lattice_run
    from coupling_ladder import couple_monotone

    params = load("ckpt/final.npz")
    settled = lattice_run(ca.ToyRule(params, "random"), B=B, N=N, r=r, T=T, sweeps=6,
                          mode="async", init="random", seed=seed)["final"]
    flipped = settled.copy()
    rngf = np.random.default_rng(seed + 5)
    flipped[:, N // 2] = rngf.integers(ca.INIT_LO, ca._vocab(), size=B)

    u = np.random.default_rng(seed + 1).random(sweeps * N * B)

    # production: two separate runs sharing u_stream (this is the CRN protocol verbatim)
    pa = lattice_run(ca.ToyRule(params, "random"), B=B, N=N, r=r, T=T, sweeps=sweeps,
                     mode="async", init_state=settled, seed=seed + 2, u_stream=u)["snaps"]
    pb = lattice_run(ca.ToyRule(params, "random"), B=B, N=N, r=r, T=T, sweeps=sweeps,
                     mode="async", init_state=flipped, seed=seed + 2, u_stream=u)["snaps"]

    # paired: one loop, both twins, monotone coupler
    qa, qb = paired_run(ca.ToyRule(params, "random"), couple_monotone, B=B, N=N, r=r, T=T,
                        sweeps=sweeps, state_a=settled, state_b=flipped, seed=seed + 2,
                        u_stream=u)

    mism_a = int((pa != qa).sum())
    mism_b = int((pb != qb).sum())
    prod_damage = (pa != pb)
    pair_damage = (qa != qb)
    mism_field = int((prod_damage != pair_damage).sum())
    return dict(cells=int(pa.size), mismatch_a=mism_a, mismatch_b=mism_b,
                mismatch_damage_field=mism_field,
                production_damage_cells=int(prod_damage.sum()),
                passes=bool(mism_a == 0 and mism_b == 0 and mism_field == 0))


def main():
    print("RUNG 1 -- the paired loop must reproduce the production twin protocol exactly\n")
    try:
        r = rung1_reproduces_production()
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return 0
    print(f"  cells compared        : {r['cells']}")
    print(f"  mismatching, twin A   : {r['mismatch_a']}")
    print(f"  mismatching, twin B   : {r['mismatch_b']}")
    print(f"  mismatching in the DAMAGE FIELD: {r['mismatch_damage_field']}")
    print(f"  (production damage cells: {r['production_damage_cells']} -- a nonzero field, so this "
          f"is not passing on an all-zero comparison)")
    print(f"\n  -> {'PASS' if r['passes'] else 'FAIL'}")
    if not r["passes"]:
        print("     The paired loop consumes randomness in a different order or slices the window\n"
              "     differently. Any maximal-coupling comparison built on it would measure that\n"
              "     bug rather than the coupling. Fix the loop, not the comparison.")
    return 0 if r["passes"] else 1


if __name__ == "__main__":
    _sys.exit(main())
