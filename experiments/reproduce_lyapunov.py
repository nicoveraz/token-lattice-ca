"""Reproduce a KNOWN metric with the CA/damage-spreading approach.

The Lyapunov exponent of the logistic map x -> r x (1-x) is exactly known:
  lambda_exact(r) = < ln |f'(x)| > = < ln |r (1 - 2x)| >   (time-average along the orbit),
giving lambda = ln 2 at r=4, negative in the periodic windows, the textbook bifurcation curve.

We recompute it with the SAME primitive the LM instrument uses: CRN twin trajectories
(two orbits sharing the deterministic dynamics, differing by a tiny epsilon) whose
log-separation growth rate, measured by renormalized perturbation tracking (a 1-D damage-
spreading measurement), estimates lambda. If lambda_crn(r) matches lambda_exact(r) across the
bifurcation diagram, the damage-spreading estimator is validated against an exact benchmark.
CPU-only. Also does a coupled-map lattice (a genuine spatially-extended CA) as a lattice check.
"""
import pathlib, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def lyap_exact(r, n=20000, burn=2000, seed=0):
    rng = np.random.default_rng(seed); x = rng.random()
    for _ in range(burn):
        x = r * x * (1 - x)
    s = 0.0
    for _ in range(n):
        s += np.log(abs(r * (1 - 2 * x)) + 1e-300)
        x = r * x * (1 - x)
    return s / n


def lyap_crn(r, n=20000, burn=2000, d0=1e-9, seed=0):
    """CRN twin-separation (damage-spreading) estimate: renormalized perturbation growth."""
    rng = np.random.default_rng(seed); x = rng.random()
    for _ in range(burn):
        x = r * x * (1 - x)
    xp = x + d0                                    # twin, perturbed by epsilon (CRN: same map)
    s = 0.0
    for _ in range(n):
        x = r * x * (1 - x)
        xp = r * xp * (1 - xp)
        d = abs(xp - x)
        s += np.log(d / d0 + 1e-300)
        xp = x + d0 * np.sign(xp - x)              # renormalize separation back to d0
    return s / n


def cml_lyap(r, eps_c, N=64, n=4000, burn=1000, d0=1e-9, seed=0):
    """Coupled map lattice (a genuine spatially-extended CA): each cell a logistic map,
    diffusively coupled. CRN twin lattices differing by a single-site perturbation ->
    finite-size Lyapunov via total-separation growth (the lattice damage-spreading measure)."""
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
        xp = x + (xp - x) * (d0 / (d + 1e-300))   # renormalize
    return s / n


def main():
    rs = np.linspace(2.8, 4.0, 61)
    ex = np.array([lyap_exact(r) for r in rs])
    cr = np.array([lyap_crn(r) for r in rs])
    err = np.abs(ex - cr)
    print("=== logistic map: reproduce known Lyapunov via CRN damage-spreading ===")
    print(f"  points={len(rs)}  mean|exact-crn|={err.mean():.4f}  max={err.max():.4f}")
    for r in [3.0, 3.2, 3.5, 3.57, 3.83, 4.0]:
        i = int(np.argmin(abs(rs - r)))
        print(f"  r={rs[i]:.3f}: exact={ex[i]:+.4f}  crn={cr[i]:+.4f}  (ln2={np.log(2):.4f} expected at r=4)")
    # coupled map lattice: reproduce the CML finite-size Lyapunov (known to be >0 in chaotic regime)
    print("\n=== coupled map lattice (spatially-extended CA), r=4, coupling sweep ===")
    cml = {}
    for ec in [0.0, 0.1, 0.2, 0.3, 0.4]:
        lam = np.mean([cml_lyap(4.0, ec, seed=s) for s in range(3)])
        cml[ec] = round(float(lam), 4)
        print(f"  eps_coupling={ec}: lattice lambda={lam:+.4f}  (single-site ln2={np.log(2):.4f})")
    out = dict(logistic=dict(r=list(np.round(rs, 3)), exact=list(np.round(ex, 4)), crn=list(np.round(cr, 4)),
                             mean_abs_err=round(float(err.mean()), 4), max_abs_err=round(float(err.max()), 4)),
               cml=cml, ln2=round(float(np.log(2)), 4))
    json.dump(out, open(str(ROOT / "results" / "reproduce_lyapunov.json"), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.3))
        ax.plot(rs, ex, "-", color="#2c6fbb", lw=2, label="known (analytic)  ⟨ln|f'(x)|⟩")
        ax.plot(rs, cr, "o", color="#b0413e", ms=3, label="CA damage-spreading (CRN twins)")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.axhline(np.log(2), color="green", lw=0.8, ls="--", label="ln 2 (expected at r=4)")
        ax.set_xlabel("logistic parameter r"); ax.set_ylabel("Lyapunov exponent λ")
        ax.set_title(f"Reproducing a known metric with the CA approach\nmean |exact−CRN| = {err.mean():.4f}")
        ax.legend(fontsize=9)
        fig.tight_layout(); fp = str(ROOT / "fig" / "reproduce_lyapunov.png")
        fig.savefig(fp, bbox_inches="tight"); print("wrote", fp)
    except Exception as ex_:
        print("plot skipped:", ex_)


if __name__ == "__main__":
    main()
