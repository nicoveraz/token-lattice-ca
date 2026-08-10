"""Is F117's compliance signal just general capability wearing an IFEval label?

THE DEFLATION THIS RUNS. F117 found the attractor share loads on IFEval and on no capability
benchmark, and used model SIZE as its control. Size is the wrong control for the sharpest objection,
which is not "big models score well" but "IFEval is a capability benchmark like the others, so a
probe correlating with capability will correlate with it." That objection is answered by partialling
out a general-capability index and asking whether anything survives.

WHY IT COULD FAIL. If the probe reads general quality, then conditioning on quality should collapse
rho(top1, IFEval) toward zero. If IFEval is itself mostly a capability measure, the same happens for
a different reason. Either outcome would deflate F117 to "the probe is a weak capability correlate",
which is the reading F119's audit makes it important to rule out explicitly rather than assume.

THE QUALITY INDEX is the mean RANK across the five correctness benchmarks (BBH, GPQA, MUSR,
MMLU-PRO, MATH Lvl 5). Ranks rather than raw scores because the benchmarks have different scales and
saturation points, and a mean of raw percentages would be dominated by whichever has the widest
spread. This is the same COMPLIANCE / CORRECTNESS split F117 registered, used as a covariate instead
of as a comparison column.

PRE-REGISTERED:
  RUNG      rho(top1@0.7, IFEval) must reproduce F117's stored value in
            `compliance_selectivity.json -> matrix`. That pins the model set, the readout and the
            ranking to the run F117 reported; a mismatch means this is a different comparison and
            nothing below is read.
  PRIMARY   partial rho(top1@0.7, IFEval | quality index), with an EXACT permutation p over the
            readout. Registered reading: if the partial falls below half the raw rho, F117 is
            substantially a capability correlate and must be restated.
  SECONDARY rho(IFEval, quality) and rho(top1, quality) -- the two components that decide whether
            the partial is a suppression or a confound.
  BOUNDARY  n = 10, base models, benchmark scores downloaded rather than measured. A partial
            correlation with one covariate on 10 points has 7 effective degrees of freedom; this is
            a deflation check, not an independent confirmation.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import itertools, json

import numpy as np
from ranking import rank
from provenance import stamp, rel

SRC = str(_ROOT / "results" / "band_greedy.json")
F117 = str(_ROOT / "results" / "compliance_selectivity.json")
OUT = str(_ROOT / "results" / "compliance_vs_quality.json")
READOUT = "0.7"                                   # F117's strongest, and the paper's operating point
COMPLIANCE = "IFEval"
CORRECTNESS = ["BBH", "GPQA", "MUSR", "MMLU-PRO", "MATH Lvl 5"]
RUNG_TOL = 0.005
COLLAPSE_FRAC = 0.5                               # registered: partial below half the raw deflates


def _r(a, b):
    ra, rb = rank(a), rank(b)
    if not (np.isfinite(ra).all() and np.isfinite(rb).all()):
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def partial(x, y, z):
    """Rank partial correlation of x and y given z, plus the three pairwise rhos it is built from."""
    rxy, rxz, ryz = _r(x, y), _r(x, z), _r(y, z)
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (float("nan") if denom == 0 else (rxy - rxz * ryz) / denom), rxy, rxz, ryz


def analyse(res):
    a, parts = res["stats"], []
    ok = abs(a["rho_top1_ifeval"] - a["f117_stored"]) <= RUNG_TOL
    parts.append(
        f"RUNG (pins this to F117's comparison): rho(top1@{READOUT}, {COMPLIANCE}) = "
        f"{a['rho_top1_ifeval']:+.4f} against F117's stored {a['f117_stored']:+.4f}. "
        + ("Reproduced, so the model set, readout and ranking match."
           if ok else "NOT reproduced -- a different comparison, so nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False)
        res["verdict"] = " ".join(parts)
        return
    pr, raw = a["partial"], a["rho_top1_ifeval"]
    collapsed = abs(pr) < COLLAPSE_FRAC * abs(raw)
    parts.append(
        f"PRIMARY: partial rho(top1, {COMPLIANCE} | quality) = {pr:+.4f} (exact permutation "
        f"p = {a['perm_p']:.4f}, n = {res['n']}), against the raw {raw:+.4f}. "
        + (f"The signal COLLAPSES once general capability is held fixed, so F117 is substantially a "
           f"capability correlate and must be restated." if collapsed else
           f"The signal SURVIVES conditioning on general capability -- it does not shrink, so F117's "
           f"selectivity is not general quality wearing an IFEval label."))
    parts.append(
        f"SECONDARY, which decides why: rho({COMPLIANCE}, quality) = {a['rho_ifeval_quality']:+.4f} "
        f"and rho(top1, quality) = {a['rho_top1_quality']:+.4f}. "
        + ("Both are weak, so quality is not a common cause and the partial is a SUPPRESSION -- "
           "removing a small shared component slightly strengthens the association."
           if abs(a["rho_ifeval_quality"]) < 0.4 and abs(a["rho_top1_quality"]) < 0.4 else
           "At least one is substantial, so quality is a genuine third variable here."))
    parts.append(
        f"BOUNDARY: n = {res['n']}, base models, benchmark scores downloaded from the Open LLM "
        f"Leaderboard rather than measured here. A partial with one covariate on {res['n']} points "
        f"has {res['n'] - 3} effective degrees of freedom. rho({COMPLIANCE}, quality) near zero is a "
        f"property of THIS model set, not a general claim about the benchmark. This is a deflation "
        f"check on F117, not an independent confirmation of it.")
    res["analysis"] = dict(rung_passes=True, collapsed=bool(collapsed), **a)
    res["verdict"] = " ".join(parts)


def main():
    cells = json.load(open(SRC))["cells"]
    rows = [c for c in cells.values()
            if isinstance(c.get("top1"), dict) and READOUT in c["top1"] and "scores" in c
            and all(b in c["scores"] for b in [COMPLIANCE] + CORRECTNESS)]
    x = [c["top1"][READOUT] for c in rows]
    y = [c["scores"][COMPLIANCE] for c in rows]
    q = list(np.array([rank([c["scores"][b] for c in rows]) for b in CORRECTNESS]).mean(axis=0))

    pr, rxy, rxz, ryz = partial(x, y, q)

    # EXACT NULL, VECTORISED. Ranking a permutation of x is the same as permuting rank(x), so the
    # three rank vectors are computed ONCE and only the readout's ranks are shuffled. A first
    # version re-ranked inside the loop -- 10! = 3.6M permutations x 3 rankdata calls -- and had to
    # be killed. Standardising each vector turns every correlation into a dot product / n.
    def _z(v):
        v = np.asarray(rank(v), float)
        return (v - v.mean()) / v.std()

    a_, b_, c_ = _z(x), _z(y), _z(q)
    nn = len(a_)
    perms = np.array(list(itertools.permutations(range(nn))))
    A = a_[perms]                                       # (10!, n) every relabelling of the readout
    p_xy = A @ b_ / nn
    p_xz = A @ c_ / nn
    den = np.sqrt((1 - p_xz ** 2) * (1 - ryz ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        null = np.where(den == 0, np.nan, (p_xy - p_xz * ryz) / den)
    null = null[np.isfinite(null)]

    res = dict(_preregistration=dict(
        source=rel(SRC), rung_source=rel(F117), readout=f"top1@{READOUT}",
        compliance=COMPLIANCE, correctness=CORRECTNESS,
        quality_index="mean RANK across the five correctness benchmarks",
        rung=f"rho(top1@{READOUT},{COMPLIANCE}) must reproduce F117's stored value +/- {RUNG_TOL}",
        primary="partial rho(top1, IFEval | quality), exact permutation p over the readout",
        reading=f"a partial below {COLLAPSE_FRAC:g}x the raw rho deflates F117 to a capability "
                f"correlate",
    ))
    res["n"] = len(rows)
    res["models"] = [c["model"] for c in rows]
    res["stats"] = dict(
        f117_stored=json.load(open(F117))["matrix"][f"top1@{READOUT}"][COMPLIANCE],
        rho_top1_ifeval=round(rxy, 4), rho_top1_quality=round(rxz, 4),
        rho_ifeval_quality=round(ryz, 4), partial=round(pr, 4),
        perm_p=round(float(np.mean(np.abs(null) >= abs(pr) - 1e-12)), 4),
        n_perm=int(len(null)), null="exact")
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    s = res["stats"]
    print(f"\n  n = {res['n']} models")
    print(f"  rho(top1@{READOUT}, {COMPLIANCE})     = {s['rho_top1_ifeval']:+.4f}")
    print(f"  rho(top1@{READOUT}, quality)    = {s['rho_top1_quality']:+.4f}")
    print(f"  rho({COMPLIANCE}, quality)        = {s['rho_ifeval_quality']:+.4f}")
    print(f"  PARTIAL (top1, {COMPLIANCE} | quality) = {s['partial']:+.4f}  p = {s['perm_p']:.4f}")
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
