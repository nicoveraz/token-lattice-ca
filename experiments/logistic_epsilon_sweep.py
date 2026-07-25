"""The epsilon-sweep: why the logistic 'validation' agrees perfectly, and what that means.

`lyap_tangent_fd` renormalizes the twin separation back to d0 every step along the same
orbit as `lyap_exact`, so log(d/d0) = log|f'(x)| + O(d0): it is a finite-difference
evaluation of the analytic derivative it is compared against. This script makes that
explicit by sweeping d0 and showing the error is O(d0) -- perfect agreement is a property
of the infinitesimal limit and nothing else.

Why publish it: a token flip is O(1) in a discrete alphabet, so there is NO epsilon -> 0
limit in token space. The rung the logistic map validates is exactly the one regime the
instrument can never occupy. Showing the scaling converts an over-claim into a precise
statement about the estimator's domain of validity.

Also reports, at each d0, the FINITE-perturbation (no renormalization) estimate, which is
the regime the instrument actually lives in -- and which does NOT converge to the analytic
value.

Writes results/logistic_epsilon_sweep.json and fig/logistic_epsilon.png. CPU, ~1 min.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from reproduce_lyapunov import lyap_exact, lyap_tangent_fd, lyap_finite_perturbation

ROOT = pathlib.Path(__file__).resolve().parents[1]
D0S = [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]
RS = np.linspace(2.8, 4.0, 61)


def main():
    ex = np.array([lyap_exact(r) for r in RS])
    res = {"d0_grid": D0S, "n_r_points": len(RS), "r_min": float(RS[0]), "r_max": float(RS[-1]),
           "ln2": round(float(np.log(2)), 4), "tangent_renormalized": {}, "finite_no_renorm": {}}
    print("=== TANGENT (renormalized every step) -- error should be O(d0) ===")
    print(f"{'d0':>10s} {'mean|err|':>12s} {'max|err|':>12s}")
    tan_mean, tan_max = [], []
    for d0 in D0S:
        lam = np.array([lyap_tangent_fd(r, d0=d0) for r in RS])
        e = np.abs(ex - lam)
        tan_mean.append(float(e.mean())); tan_max.append(float(e.max()))
        res["tangent_renormalized"][f"{d0:g}"] = dict(
            mean_abs_err=round(float(e.mean()), 8), max_abs_err=round(float(e.max()), 8))
        print(f"{d0:10.0e} {e.mean():12.6f} {e.max():12.6f}")

    print("\n=== FINITE perturbation (NO renormalization) -- the instrument's regime ===")
    print(f"{'d0':>10s} {'mean|err|':>12s} {'max|err|':>12s}")
    fin_mean = []
    for d0 in D0S:
        lam = np.array([lyap_finite_perturbation(r, d0=d0)[0] for r in RS])
        e = np.abs(ex - lam)
        fin_mean.append(float(e.mean()))
        res["finite_no_renorm"][f"{d0:g}"] = dict(
            mean_abs_err=round(float(e.mean()), 6), max_abs_err=round(float(e.max()), 6))
        print(f"{d0:10.0e} {e.mean():12.6f} {e.max():12.6f}")

    # log-log slope of the tangent error vs d0 (expect ~1 => O(d0))
    lg = np.polyfit(np.log10(D0S), np.log10(np.maximum(tan_mean, 1e-12)), 1)
    res["tangent_error_loglog_slope"] = round(float(lg[0]), 3)
    res["interpretation"] = (
        "Tangent error scales as O(d0) (log-log slope ~1): perfect agreement at d0=1e-9 is the "
        "infinitesimal limit, not an independent confirmation of damage spreading. The finite, "
        "un-renormalized estimator does NOT converge to the analytic value -- and a token flip "
        "is O(1), so the token instrument inherits the finite regime, never the tangent one.")
    out = str(ROOT / "results" / "logistic_epsilon_sweep.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"\ntangent error log-log slope = {res['tangent_error_loglog_slope']} (1.0 == O(d0))")
    print("wrote", out)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "figure.dpi": 200})
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
        ax[0].loglog(D0S, np.maximum(tan_mean, 1e-12), "o-", color="#2c6fbb", lw=2,
                     label=f"tangent (renormalized), slope={res['tangent_error_loglog_slope']}")
        ax[0].loglog(D0S, np.maximum(fin_mean, 1e-12), "s--", color="#b0413e", lw=2,
                     label="finite perturbation (no renorm)")
        ax[0].set_xlabel(r"perturbation size $d_0$"); ax[0].set_ylabel(r"mean $|\lambda_{exact}-\hat\lambda|$")
        ax[0].set_title("Error vs perturbation size\ntangent error is $O(d_0)$ — the agreement is a limit")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
        for d0, c in zip([1e-9, 1e-3, 1e-1], ["#2c6fbb", "#c78a1e", "#b0413e"]):
            lam = np.array([lyap_tangent_fd(r, d0=d0) for r in RS])
            ax[1].plot(RS, lam, "-", color=c, lw=1.4, label=f"tangent $d_0$={d0:g}")
        ax[1].plot(RS, ex, "k:", lw=1.6, label="analytic")
        ax[1].axhline(0, color="gray", lw=0.7, ls=":")
        ax[1].set_xlabel("logistic $r$"); ax[1].set_ylabel(r"$\lambda$")
        ax[1].set_title("Bifurcation curve degrades as $d_0$ grows")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fp = str(ROOT / "fig" / "logistic_epsilon.png")
        fig.savefig(fp, bbox_inches="tight"); print("wrote", fp)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
