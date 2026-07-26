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
    """True if `lam` is the never-ignited sentinel rather than a measured exponent."""
    return bool(np.isfinite(lam) and abs(float(lam) - DEAD_DAMAGE_FLOOR) < tol)


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
