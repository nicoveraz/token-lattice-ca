"""Logistic-map / CML Lyapunov: a UNIT TEST of growth-rate arithmetic in the SMOOTH LIMIT.

WHAT THIS IS (and is not)
-------------------------
This script does **not** validate the token-lattice instrument. It validates the
arithmetic of estimating an exponential growth rate from a separating pair of
trajectories, in the limit of an *infinitesimal* perturbation on a *continuous* state
space. That is a regime the token instrument can never occupy.

`lyap_tangent_fd` renormalizes the twin separation back to `d0` after every step, along
the same orbit as `lyap_exact`. Therefore

    log(d_{t+1} / d0) = log|f'(x_t)| + O(d0),

i.e. it is a **finite-difference evaluation of the analytic derivative** it is being
compared against. It agrees with `lyap_exact` to machine precision at d0=1e-9 because it
*is* the same quantity, not because a damage-spreading measurement was independently
confirmed. The agreement is real for what it measures; see
`experiments/logistic_epsilon_sweep.py`, which shows the error is O(d0) and vanishes only
in the d0 -> 0 limit.

Why this matters for the instrument: a token flip is **O(1)** in a discrete alphabet.
There is no epsilon -> 0 limit in token space, so the smooth-limit agreement transfers
nothing. The regime the instrument actually lives in is the *finite* perturbation with no
renormalization -- see `--finite-perturbation` below, and `lyap_from_cone` in
`experiments/lyapunov.py`, which is a STRUCTURALLY DIFFERENT estimator (a windowed
polyfit on a saturating discrete damage count). The two share zero code.

The weight-bearing discrete rungs of the validation ladder are `eca_calib.py`
(deterministic discrete) and `dk_calib.py` (stochastic discrete, published boundary).

Usage:
  reproduce_lyapunov.py                        # smooth-limit unit test (default paths)
  reproduce_lyapunov.py --finite-perturbation  # the honest finite-d0, NO-renormalization regime
"""
import pathlib, json, argparse
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def lyap_exact(r, n=20000, burn=2000, seed=0):
    """Analytic time-average <ln|f'(x)|> along the orbit. The ground truth."""
    rng = np.random.default_rng(seed); x = rng.random()
    for _ in range(burn):
        x = r * x * (1 - x)
    s = 0.0
    for _ in range(n):
        s += np.log(abs(r * (1 - 2 * x)) + 1e-300)
        x = r * x * (1 - x)
    return s / n


def lyap_tangent_fd(r, n=20000, burn=2000, d0=1e-9, seed=0):
    """Finite-difference TANGENT-GROWTH estimate; exact in the d0 -> 0 limit.

    NOT the instrument's estimator. The twin is re-anchored to the reference orbit every
    step (`xp = x + d0*sign(...)`), so this measures |f'(x)| by finite difference and
    converges to `lyap_exact` as d0 -> 0 (error is O(d0)). Retained as a unit test of the
    growth-rate arithmetic, not as evidence about damage spreading in discrete systems.
    """
    rng = np.random.default_rng(seed); x = rng.random()
    for _ in range(burn):
        x = r * x * (1 - x)
    xp = x + d0
    s = 0.0
    for _ in range(n):
        x = r * x * (1 - x)
        xp = r * xp * (1 - xp)
        d = abs(xp - x)
        s += np.log(d / d0 + 1e-300)
        xp = x + d0 * np.sign(xp - x)   # <-- re-anchors to the reference orbit: the circularity
    return s / n


# Backwards-compatible alias so any stale reference keeps working; prefer the honest name.
lyap_crn = lyap_tangent_fd


def lyap_finite_perturbation(r, d0=1e-3, n=200, burn=2000, seed=0, sat_frac=0.1,
                             min_steps=3, max_steps=40):
    """FINITE perturbation, NO renormalization -- the regime the token instrument occupies.

    Perturb once by a finite d0, evolve both orbits freely, and fit the growth rate of
    log|xp - x| over the pre-saturation window (the continuous analogue of `lyap_from_cone`,
    which fits a window on a saturating discrete damage count). Because the separation
    saturates at the attractor diameter O(1) instead of being renormalized away, the
    estimate is BIASED relative to the analytic lambda -- that bias is the quantity of
    interest, and it is what a discrete O(1) token flip inherits.

    The perturbed point is kept inside the invariant interval [0,1] (perturbing outward past
    1 would leave the logistic map's domain, where orbits diverge to -inf and the estimate is
    meaningless rather than merely biased).

    Returns (lambda_hat, fit_end, d_max).
    """
    rng = np.random.default_rng(seed); x = rng.random()
    for _ in range(burn):
        x = r * x * (1 - x)
    xp = x + d0 if x + d0 <= 1.0 else x - d0        # stay in the invariant interval
    ds = [abs(xp - x)]
    for _ in range(n):
        x = r * x * (1 - x)
        xp = r * xp * (1 - xp)
        ds.append(abs(xp - x))                      # NO renormalization: free divergence
    d = np.asarray(ds)
    if not np.all(np.isfinite(d)):
        return float("nan"), 0, float("nan")
    dmax = float(d.max())
    if dmax <= 0:
        return float("nan"), 0, dmax
    # window ends when the separation first reaches sat_frac of its eventual maximum
    above = np.nonzero(d >= sat_frac * dmax)[0]
    sat = int(above[0]) if above.size else max_steps
    end = int(max(min_steps, min(sat, max_steps, len(d) - 1)))
    t = np.arange(end + 1)
    lam = float(np.polyfit(t, np.log(d[:end + 1] + 1e-300), 1)[0])
    return lam, end, dmax


def cml_lyap(r, eps_c, N=64, n=4000, burn=1000, d0=1e-9, seed=0):
    """Coupled-map-lattice maximal Lyapunov by renormalized twin separation.

    NOTE (same caveat as `lyap_tangent_fd`): the twin is rescaled back to `d0` every step,
    so this is a tangent-space (infinitesimal) estimate, not a finite damage-spreading
    measurement. Ground truth for the eps>0 values is supplied separately by a
    Benettin/QR Jacobian-product computation (see `experiments/cml_benettin.py`).
    """
    rng = np.random.default_rng(seed)
    f = lambda v: r * v * (1 - v)

    def step(v):
        fv = f(v)
        return (1 - eps_c) * fv + eps_c * 0.5 * (np.roll(fv, 1) + np.roll(fv, -1))
    x = rng.random(N)
    for _ in range(burn):
        x = step(x)
    xp = x.copy(); xp[N // 2] += d0
    s = 0.0
    for _ in range(n):
        x = step(x); xp = step(xp)
        d = np.linalg.norm(xp - x)
        s += np.log(d / d0 + 1e-300)
        xp = x + (xp - x) * (d0 / (d + 1e-300))   # <-- also renormalizes (tangent-space)
    return s / n


def run_finite_perturbation(out_path):
    """The honest regime: finite d0, no renormalization. Reports departure from analytic."""
    rs = np.linspace(2.8, 4.0, 61)
    ex = np.array([lyap_exact(r) for r in rs])
    print("=== FINITE perturbation, NO renormalization (the instrument's regime) ===")
    print(f"{'d0':>8s} {'mean|err|':>10s} {'max|err|':>10s} {'mean lam_fp':>12s} {'mean lam_ex':>12s}")
    rows = {}
    for d0 in [1e-6, 1e-3, 1e-2, 1e-1]:
        lam = np.array([lyap_finite_perturbation(r, d0=d0)[0] for r in rs])
        e = np.abs(ex - lam)
        rows[f"{d0:g}"] = dict(mean_abs_err=round(float(e.mean()), 4),
                               max_abs_err=round(float(e.max()), 4),
                               mean_lambda=round(float(lam.mean()), 4),
                               lambda_at_r4=round(float(lam[-1]), 4))
        print(f"{d0:8.0e} {e.mean():10.4f} {e.max():10.4f} {lam.mean():12.4f} {ex.mean():12.4f}")
    out = dict(kind="finite_perturbation_no_renormalization",
               note=("No renormalization: the separation saturates at the attractor diameter, "
                     "so lambda_hat is biased relative to the analytic value. This is the "
                     "regime a discrete O(1) token flip inherits; the smooth-limit agreement "
                     "of lyap_tangent_fd does NOT transfer here."),
               ln2=round(float(np.log(2)), 4), by_d0=rows)
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite-perturbation", action="store_true",
                    help="run the no-renormalization finite-d0 variant (the instrument's regime)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.finite_perturbation:
        run_finite_perturbation(a.out or str(ROOT / "results" / "logistic_finite_perturbation.json"))
        return

    rs = np.linspace(2.8, 4.0, 61)
    ex = np.array([lyap_exact(r) for r in rs])
    cr = np.array([lyap_tangent_fd(r) for r in rs])
    err = np.abs(ex - cr)
    print("=== logistic map: SMOOTH-LIMIT unit test (tangent finite-difference) ===")
    print(f"  points={len(rs)}  mean|exact-tangent_fd|={err.mean():.6f}  max={err.max():.6f}")
    print("  NOTE: agreement is O(d0) by construction -- see logistic_epsilon_sweep.py")
    for r in [3.0, 3.2, 3.5, 3.57, 3.83, 4.0]:
        i = int(np.argmin(abs(rs - r)))
        print(f"  r={rs[i]:.3f}: exact={ex[i]:+.4f}  tangent_fd={cr[i]:+.4f}")
    print("\n=== coupled map lattice (tangent-space; ground truth via Benettin) ===")
    cml = {}
    for ec in [0.0, 0.1, 0.2, 0.3, 0.4]:
        lam = np.mean([cml_lyap(4.0, ec, seed=s) for s in range(3)])
        cml[ec] = round(float(lam), 4)
        print(f"  eps_coupling={ec}: lattice lambda={lam:+.4f}  (single-site ln2={np.log(2):.4f})")
    out = dict(kind="smooth_limit_unit_test",
               note=("lyap_tangent_fd renormalizes to the reference orbit each step, so it is a "
                     "finite-difference derivative, exact as d0->0. NOT a damage-spreading "
                     "validation of the token instrument."),
               logistic=dict(r=list(np.round(rs, 3)), exact=list(np.round(ex, 4)),
                             tangent_fd=list(np.round(cr, 4)),
                             mean_abs_err=round(float(err.mean()), 6),
                             max_abs_err=round(float(err.max()), 6)),
               cml=cml, ln2=round(float(np.log(2)), 4))
    dest = a.out or str(ROOT / "results" / "logistic_smooth_limit_unit_test.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
