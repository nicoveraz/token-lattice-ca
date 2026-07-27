"""Phase B rigor (F23): the finite-size Lyapunov exponent lambda -- the CA-canonical
chaos measure -- from the early exponential growth of the CRN twin separation. lambda
= slope of log(number of damaged sites) vs sweep over the ballistic window. lambda>0
= perturbation amplified (chaos / edge-of-chaos); lambda<=0 = healed. Tests the
edge-of-chaos frame quantitatively: does lambda climb toward positive with model
CAPACITY (tiny -> mini -> base)? Usage: lyapunov.py --backend mlm --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, argparse, json, time
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from mlm_lib import RESDIR, ensure_resdir

RS = [1, 2, 4, 8, 16]
TS = [0.7, 0.9]
SEEDS = [21, 22]

# --- the estimator's numerical floor ---------------------------------------------------
# `lyap_from_cone` clamps the damage count at DAMAGE_CLAMP before taking a log. When damage
# dies immediately -- one damaged site at t=0 and nothing after -- the fitted sequence is
# [1, eps, eps, ...] and the least-squares slope over the default 9-point window is a
# CONSTANT, independent of rule, seed, model and lattice size:
#
#     DEAD_DAMAGE_FLOOR = -0.4 * ln(10) = -0.9210340371976...
#
# It is a sentinel meaning "damage never ignited", NOT an estimated exponent. Five of the
# seven ordered rules in results/eca_calib_hardened.json sit exactly on it with a zero-width
# bootstrap interval, which is why the ordered-group mean lambda must never be reported as a
# measured exponent (see findings F40). Use `is_dead_damage_floor` to detect and exclude it;
# the DP-class order parameter for those runs is the ignition probability, not lambda.
DAMAGE_CLAMP = 1e-6
DEAD_DAMAGE_FLOOR = -0.4 * np.log(10)


def is_dead_damage_floor(lam, tol=1e-9):
    """True if `lam` is the never-ignited sentinel rather than a measured exponent.

    NOTE: this catches only ONE specific value. It is NOT a general test for "damage never
    ignited" -- see `is_unignited`, which is. F40 named this constant; F42 found the general
    case it does not cover.
    """
    return bool(np.isfinite(lam) and abs(float(lam) - DEAD_DAMAGE_FLOOR) < tol)


def is_unignited(mean_damage=None, D_norm=None):
    """True if the run's damage never ignited, so lambda_ca is UNDEFINED for it (F42).

    lambda_ca is the growth rate of a damage cone. If no damage survives there is no cone
    and no rate -- but `lyap_from_cone` returns a finite number anyway, fitted against the
    DAMAGE_CLAMP floor, and that number is WILDLY UNSTABLE for physically identical runs:

        Phase 3 (96 runs): 1 unignited, N=96 step256 seed22 -> lambda = -0.1649
        N=192 run 1/24   : same zero final damage           -> lambda = -1.7130

    Same physical outcome, an order of magnitude apart, and `is_dead_damage_floor` catches
    NEITHER (both are far from -0.9210).

    ON THE MECHANISM, stated carefully. It is tempting to say the magnitude grows with the
    lattice size. VERIFIED: it does not -- `lyap_from_cone` is N-INDEPENDENT for a fixed
    cone (a 3-site seed that dies immediately returns -0.9943 at N=48, 96 and 192 alike;
    the fit window is min(max_sweeps=8, len(d)-1), with no N in it, and N enters only the
    second return value dmax/N). The spread therefore comes from the CONES differing --
    how gradually the damage decays before vanishing -- not from N entering the estimator.
    Whether unignited runs become more common or more extreme at larger N is an open
    empirical question, not an established mechanism, and must not be asserted as one.

    The reason to exclude these runs does not depend on the mechanism: lambda is undefined
    without a cone, and the emitted numbers span an order of magnitude for the same physical
    outcome. Averaging one into a cell mean is arbitrary -- a single unignited run displaces
    a 16-run pre mean by ~-0.108, which is 73% of N=96's entire pre->plateau gap.

    Keyed on `mean_damage`, the raw quantity, not on lambda's sign or magnitude -- a run can
    be negative AND ignited (N=192 seed23: lam=-0.2197, D_norm=0.0250) and that is a real
    measurement which must be kept.

    `D_norm` is accepted as a fallback for records written before `mean_damage` was stored.
    It is sound at the configurations used here: the smallest nonzero mean_damage is
    1/(tail*N*B), which yields D_norm >= 2.7e-4 for every (N, B) in this project, so a
    stored D_norm of exactly 0.0 can only have come from mean_damage == 0. Verify this
    before reusing the fallback at a new (N, B, tail).
    """
    if mean_damage is not None:
        return bool(float(mean_damage) <= 0.0)
    if D_norm is not None:
        return bool(float(D_norm) == 0.0)
    raise ValueError("is_unignited needs mean_damage (preferred) or D_norm (fallback)")


def lyap_from_cone(cone, N, sat_threshold=3.5, frac_of_max=0.5, max_sweeps=8, min_sweeps=3,
                   fit_window=None):
    """cone (sweeps+1, N) per-site damage prob -> finite-size Lyapunov (per sweep).

    The estimator fits the slope of log(expected damaged sites) over an early window. The
    window is chosen by a DATA-DEPENDENT branch whose constants were previously hard-coded;
    they are now explicit keyword arguments so that (a) they can be reported, and (b) a
    headline result can be made independent of them.

    Parameters
    ----------
    sat_threshold : float
        If the damage count ever exceeds this, the perturbation is treated as having GROWN
        beyond the ~3-site seed; otherwise it is treated as HEALED and the early decay is fit.
    frac_of_max : float
        In the grown branch, the window ends when the count first reaches this fraction of
        its maximum (i.e. before saturation flattens the slope).
    max_sweeps, min_sweeps : int
        Clamp on the fitted window length.
    fit_window : (start, end) or None
        If given, BYPASSES the branch entirely and fits exactly this window. Use a
        PRE-REGISTERED window for headline results so that no claim can be an artifact of
        the data-dependent branch.

    Returns (lambda_per_sweep, dmax_fraction_of_N).

    NOTE: a returned lambda equal to DEAD_DAMAGE_FLOOR is the never-ignited sentinel, not a
    measurement -- test it with `is_dead_damage_floor` before averaging.
    """
    d = np.maximum(cone.sum(axis=1), DAMAGE_CLAMP)       # expected damaged sites
    dmax = d.max()
    if fit_window is not None:                            # pre-registered: no data-dependence
        start, end = fit_window
        start = max(0, int(start)); end = int(min(end, len(d) - 1))
        if end <= start:
            end = min(start + 1, len(d) - 1)
    else:
        start = 0
        if dmax >= sat_threshold:                         # grew beyond the ~3-site seed
            end = int(np.argmax(d >= frac_of_max * dmax))
            end = max(min_sweeps, min(end, len(d) - 1))
        else:                                             # healed: fit the early decay
            end = min(max_sweeps, len(d) - 1)
    ts = np.arange(start, end + 1)
    lam = float(np.polyfit(ts, np.log(d[start:end + 1]), 1)[0])
    return lam, float(dmax / N)


def main(backend, tag, B, N, sweeps):
    ensure_resdir()
    if backend == "mlm":
        from mlm_ca import MLMRule
        from mlm_damage import block_damage
        from mlm_lib import MODELS
        rule = MLMRule(MODELS[tag]); scheme = "cls_sep"
        key = tag
    else:
        from ar_probe import block_damage, MODELS
        from ar_ca import ARRule
        rule = ARRule(MODELS[tag]); scheme = "none"
        key = tag
    res = {"backend": backend, "model": tag, "RS": RS, "TS": TS}
    lam = {r: {} for r in RS}
    t0 = time.time()
    for r in RS:
        for T in TS:
            ls = []
            for sd in SEEDS:
                d = block_damage(rule, T, r, block=3, B=B, N=N, settle=12,
                                 sweeps=sweeps, seed=sd, scheme=scheme)
                ls.append(lyap_from_cone(d["cone"], N)[0])
            lam[r][T] = round(float(np.mean(ls)), 4)
            print(f"[{key}] lambda r={r:>2} T={T}: {lam[r][T]:+.4f}/sweep", flush=True)
    res["lambda"] = {str(r): {str(T): lam[r][T] for T in TS} for r in RS}
    res["lambda_max"] = round(float(max(lam[r][T] for r in RS for T in TS)), 4)
    res["lambda_max_at"] = [(r, T) for r in RS for T in TS if lam[r][T] == max(lam[r2][T2] for r2 in RS for T2 in TS)][0]
    json.dump(res, open(f"{RESDIR}/lyapunov_{backend}_{tag}.json", "w"), indent=1)
    print(f"[{key}] lambda_max={res['lambda_max']:+.4f} at r,T={res['lambda_max_at']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlm", choices=["mlm", "ar"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=25)
    a = ap.parse_args()
    main(a.backend, a.model, a.B, a.N, a.sweeps)

# ---------------------------------------------------------------- F42 at the RUN level (#63)
def run_ignited(run):
    """Did this run's damage ignite? The single definition; do not re-implement it.

    `is_unignited` takes a VALUE; every caller needs a RUN-record adapter that picks which field
    to pass, because older records predate `mean_damage` and need the `D_norm` fallback. That
    adapter was hand-written thirteen times across experiments/ and tests/, and got written
    wrongly twice -- once applying the filter to BOTH metrics, which inflated D_norm's N=192
    plateau 0.1393 -> 0.1592, a 14% error on the quantity whose size scaling was the entire point
    of that run.
    """
    if "mean_damage" in run:
        return not is_unignited(mean_damage=run["mean_damage"])
    return not is_unignited(D_norm=run["D_norm"])


def lambda_of(runs):
    """lambda_ca over IGNITED runs only -- lambda is UNDEFINED without a cone (F42)."""
    return [r["lambda_ca"] for r in runs if run_ignited(r)]


def dnorm_of(runs):
    """D_norm over ALL runs -- zero damage is a TRUE ZERO, not a missing value (F42).

    The asymmetry between this and `lambda_of` is the rule, not an oversight: it is why the two
    live here as a pair rather than as one filter applied everywhere. Reading them side by side
    is the point.
    """
    return [r["D_norm"] for r in runs]

