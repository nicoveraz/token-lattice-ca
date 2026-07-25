"""Issue #23 -- a real ground-truth reference for the coupled-map-lattice rung.

The CML rung currently has ground truth only at eps=0 (ln 2); its eps>0 values
{0.1: 0.4495, 0.2: 0.3648, 0.3: 0.3658, 0.4: 0.3688} are compared against nothing.

This computes the maximal Lyapunov exponent by the standard **Benettin** method: evolve the
state, evolve a tangent vector under the EXACT analytic Jacobian, renormalize each step and
accumulate the log growth. For this CML the Jacobian is exact, so this is a genuine
reference rather than another estimate.

    x_i' = (1-e) f(x_i) + (e/2)[ f(x_{i-1}) + f(x_{i+1}) ],   f(x) = r x (1-x)
    dx_i'/dx_j = C_ij f'(x_j),   f'(x) = r(1-2x)
    C = circulant with (1-e) on the diagonal and e/2 on both neighbours

WHAT THIS DOES AND DOES NOT SETTLE (F30/F31). Benettin is a TANGENT-space computation, so it
is the correct ground truth for the quantity `cml_lyap` estimates -- but agreement between
them does NOT rehabilitate the CML as a validation of the token instrument. Both live in the
infinitesimal limit; the instrument does not. This closes the "compared against nothing" gap
and nothing more. The rung stays labelled a smooth-limit arithmetic check.

Writes results/cml_benettin.json.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from reproduce_lyapunov import cml_lyap

ROOT = pathlib.Path(__file__).resolve().parents[1]
R, N = 4.0, 64
EPS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = [0, 1, 2, 3, 4]
N_STEPS, BURN = 20000, 2000


def coupling_matrix(N, eps):
    C = np.zeros((N, N))
    idx = np.arange(N)
    C[idx, idx] = 1.0 - eps
    C[idx, (idx - 1) % N] += eps / 2.0
    C[idx, (idx + 1) % N] += eps / 2.0
    return C


def benettin(r, eps, N=N, n_steps=N_STEPS, burn=BURN, seed=0):
    """Maximal Lyapunov exponent via tangent-vector growth under the exact Jacobian."""
    rng = np.random.default_rng(seed)
    C = coupling_matrix(N, eps)
    f = lambda v: r * v * (1 - v)
    fp = lambda v: r * (1 - 2 * v)

    x = rng.random(N)
    for _ in range(burn):
        x = C @ f(x)
    v = rng.standard_normal(N); v /= np.linalg.norm(v)
    acc = 0.0
    for _ in range(n_steps):
        J_v = C @ (fp(x) * v)          # J v  = C diag(f'(x)) v, formed without building J
        x = C @ f(x)
        nrm = np.linalg.norm(J_v)
        if nrm <= 0 or not np.isfinite(nrm):
            return float("nan")
        acc += np.log(nrm)
        v = J_v / nrm                  # renormalize (Benettin)
    return acc / n_steps


def main():
    print(f"CML Benettin reference: r={R}, N={N}, {N_STEPS} steps, {len(SEEDS)} seeds")
    print(f"{'eps':>5} {'Benettin (exact J)':>20} {'cml_lyap (finite-diff)':>24} {'|diff|':>8}")
    rows = {}
    for e in EPS:
        b = np.array([benettin(R, e, seed=s) for s in SEEDS])
        c = np.array([cml_lyap(R, e, N=N, seed=s) for s in SEEDS])
        diff = abs(b.mean() - c.mean())
        rows[f"{e:g}"] = dict(
            benettin_mean=round(float(b.mean()), 4), benettin_sd=round(float(b.std(ddof=1)), 4),
            cml_lyap_mean=round(float(c.mean()), 4), cml_lyap_sd=round(float(c.std(ddof=1)), 4),
            abs_diff=round(float(diff), 4))
        print(f"{e:5.1f} {b.mean():13.4f} ± {b.std(ddof=1):.4f} "
              f"{c.mean():17.4f} ± {c.std(ddof=1):.4f} {diff:8.4f}")

    ln2 = float(np.log(2))
    e0 = rows["0"]["benettin_mean"]
    print(f"\n  anchor: at eps=0 the lattice decouples -> single-map ln2={ln2:.4f}; "
          f"Benettin gives {e0:.4f} (|err|={abs(e0-ln2):.4f})")
    worst = max(v["abs_diff"] for v in rows.values())
    print(f"  max |Benettin - cml_lyap| across eps = {worst:.4f}")

    out = dict(
        note=("Maximal Lyapunov of the CML by Benettin (exact analytic Jacobian, renormalized "
              "tangent vector) as ground truth for the eps>0 values, which previously had "
              "none. NOTE: Benettin is a TANGENT-space computation, so agreement does not "
              "rehabilitate the CML as a validation of the token instrument -- both live in "
              "the infinitesimal limit and the instrument does not (F30/F31). This closes the "
              "'compared against nothing' gap only; the rung stays a smooth-limit check."),
        r=R, N=N, n_steps=N_STEPS, seeds=SEEDS, ln2=round(ln2, 4),
        eps0_error_vs_ln2=round(abs(e0 - ln2), 4),
        max_abs_diff=round(float(worst), 4), by_eps=rows)
    dest = ROOT / "results" / "cml_benettin.json"
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
