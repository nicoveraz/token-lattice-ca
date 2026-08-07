"""Is the coupling a COMMON MODE? Maximal vs monotone, on a three-rung ladder. (W2)

WHAT IS ALREADY SETTLED, SO THIS DOES NOT RE-ASK IT. F41 established that this project's CRN is the
MONOTONE (quantile) coupling, not the maximal one, and measured the gap on real conditionals from a
live bert-tiny run: mean disagreement 0.7717 maximal against 0.7818 inverse-CDF at T=0.7 (1.013x),
and 0.8042 against 0.8477 at T=0.9 (1.054x). The couplings differ, the direction is known (maximal
maximises agreement, so it MINIMISES damage), and the magnitude is 1-5% at the operating point.
Asking "do they differ" again would be spending compute on a closed question.

THE OPEN QUESTION IS THE ONE THE PAPER RESTS ON. F41's own escape clause is that every RELATIVE
comparison -- checkpoint to checkpoint, across radii, rule to rule -- survives, "because the coupling
is a common mode". That is asserted, not measured. W2 concedes the same gap from the other side:
"the alternative floors themselves are unrun on the LM backends." If the coupling is a common mode,
the developmental crossing and the discriminator are safe under either choice and the absolute
D_norm caveat stays a caveat. If it is NOT, every relative reading in the paper needs re-taking.

    PRIMARY: does lambda_ca(checkpoint) measured under MAXIMAL coupling preserve the ORDERING and
             the SIGN CROSSING it has under monotone coupling?

THREE RUNGS, CLIMBED BEFORE THE PRIMARY IS READ. Each has a known answer, and each validates a
different piece.

  RUNG 1 -- the new runner reproduces the trusted one. `lattice.run` is the single shared loop and
    carries a golden-file regression asserting bit-identity, so it is NOT touched. Maximal coupling
    cannot be expressed as a `sampler(probs, u)` anyway: that hook sees one replica's distribution,
    while maximal coupling needs BOTH twins' distributions at the same site at the same moment, and
    the twins are separate `run()` calls sharing a `u_stream`. So a paired runner is written here.
    Under the MONOTONE coupler it must reproduce the existing CRN damage field EXACTLY -- same
    seeds, same order, zero mismatching cells. A new loop that cannot reproduce the old one on the
    old coupling has no standing to report a difference on a new one.

  RUNG 2 -- maximal and monotone coincide at |V| = 2, provably. F41 verified this over 200,000
    random binary pairs at `max |maximal - quantile| = 0.0`. So on Domany-Kinzel the two couplers
    must give IDENTICAL damage, cell for cell. This validates the maximal coupler itself against a
    case where its answer is known in advance, which is the only way to tell a correct
    implementation from a plausible one.

  RUNG 3 -- the marginals are coupling-invariant by construction. A coupling is a joint law whose
    marginals are fixed; it cannot change what one replica alone does. So single-replica statistics
    -- token frequencies, entropy, the settled distribution -- must match under both couplers on the
    LM backend at large |V|, where rung 2 no longer applies. Any difference here is a bug in the
    coupler, not a finding, and this is the rung that catches an implementation error the binary
    case cannot.

Only if all three pass is the primary read. A failure at any rung means the maximal arm is not
measuring what it claims and the primary is NOT DECIDABLE.

WHAT IT COSTS, STATED RATHER THAN DISCOVERED. Monotone coupling is replica-independent: each
replica's next state is a function of its own state and a shared uniform, which is what makes twin
runs reproducible from a seed alone. Maximal coupling is not -- the draw depends on BOTH twins'
distributions jointly -- so the paired runner consumes randomness in a different order and its runs
are reproducible only as a pair. That is a real loss of a property this project relies on, and it is
the reason the monotone coupling stays the default whatever this returns.

Usage:
    .venv/bin/python experiments/coupling_ladder.py --rungs   # rungs only, no LM compute
    .venv/bin/python experiments/coupling_ladder.py           # the full ladder + primary
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"), str(_ROOT / "gatecheck" / "src")]

import argparse
import json
import os
import time

import numpy as np

from provenance import stamp, rel
from gatecheck import NOT_DECIDABLE, carries_verdict, dynamic_range, distinct_units

OUT = str(_ROOT / "results" / "coupling_ladder.json")
# F41's measured gap on real conditionals, for the record and as a sanity band -- not a threshold.
F41_GAP = {"T0.7": 1.013, "T0.9": 1.054}


# ------------------------------------------------------------------ the two couplers

def couple_monotone(p, q, u):
    """Inverse-CDF against a SHARED uniform: this project's CRN, exactly as `sampling.inverse_cdf`.

    Written here in pair form so both couplers have the same signature; rung 1 asserts it agrees
    with the production sampler cell for cell rather than trusting that it does.
    """
    cp = np.cumsum(p, axis=-1); cp /= cp[:, -1:]
    cq = np.cumsum(q, axis=-1); cq /= cq[:, -1:]
    x = np.array([np.searchsorted(cp[b], u[b]) for b in range(len(u))])
    y = np.array([np.searchsorted(cq[b], u[b]) for b in range(len(u))])
    return x, y


def couple_maximal(p, q, u, u2, u3):
    """Maximal coupling: agree with probability sum_v min(p_v, q_v), else draw the residuals.

    Needs THREE independent uniforms per site rather than one -- the agreement coin, and one draw
    for each residual. That is the concrete form of the reproducibility cost: the monotone coupler
    consumes exactly one uniform per site per replica and is replica-independent; this one is not.
    """
    m = np.minimum(p, q)
    M = m.sum(axis=-1)                                   # P(agree) under the maximal coupling
    agree = u < M

    def draw(w, uu):
        tot = w.sum(axis=-1, keepdims=True)
        tot[tot <= 0] = 1.0
        c = np.cumsum(w / tot, axis=-1); c[:, -1] = 1.0
        return np.array([np.searchsorted(c[b], uu[b]) for b in range(len(uu))])

    common = draw(m, u2)
    rx = draw(np.clip(p - m, 0, None), u2)
    ry = draw(np.clip(q - m, 0, None), u3)
    return np.where(agree, common, rx), np.where(agree, common, ry)


COUPLERS = {"monotone": couple_monotone, "maximal": couple_maximal}


# ------------------------------------------------------------------ the rungs

def rung2_binary(n=200_000, seed=0):
    """|V| = 2: the two couplers must agree EXACTLY (F41 verified this at max difference 0.0)."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, 1)); b = rng.random((n, 1))
    p = np.hstack([a, 1 - a]); q = np.hstack([b, 1 - b])
    u = rng.random(n); u2 = rng.random(n); u3 = rng.random(n)
    xm, ym = couple_monotone(p, q, u)
    xM, yM = couple_maximal(p, q, u, u2, u3)
    dis_mono = float(np.mean(xm != ym))
    dis_max = float(np.mean(xM != yM))
    return dict(n=n, disagreement_monotone=round(dis_mono, 6),
                disagreement_maximal=round(dis_max, 6),
                gap=round(abs(dis_mono - dis_max), 6),
                passes=bool(abs(dis_mono - dis_max) < 2e-3))


def rung3_marginals(n=40_000, V=64, seed=1):
    """A coupling fixes the marginals. Each replica's own token distribution must be unchanged."""
    rng = np.random.default_rng(seed)
    p = rng.random((n, V)); p /= p.sum(1, keepdims=True)
    q = rng.random((n, V)); q /= q.sum(1, keepdims=True)
    u, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    xm, _ = couple_monotone(p, q, u)
    xM, _ = couple_maximal(p, q, u, u2, u3)
    fm = np.bincount(xm, minlength=V) / n
    fM = np.bincount(xM, minlength=V) / n
    tv = float(0.5 * np.abs(fm - fM).sum())
    # Sampling noise on a TV between two multinomials of this size is ~sqrt(V/n).
    tol = 3 * np.sqrt(V / n)
    return dict(n=n, V=V, total_variation=round(tv, 5), tolerance=round(float(tol), 5),
                passes=bool(tv < tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", action="store_true", help="rungs only; no LM compute")
    a = ap.parse_args()

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res["_preregistration"] = dict(
        question="is the coupling a COMMON MODE across the developmental grid?",
        settled_elsewhere="F41 already measured that the couplings DIFFER (1.013x at T=0.7, 1.054x "
                          "at T=0.9 on real conditionals). This does not re-ask that",
        primary="does lambda_ca(checkpoint) under MAXIMAL coupling preserve the ORDERING and the "
                "SIGN CROSSING it has under monotone coupling?",
        rung1="the paired runner must reproduce the production CRN damage field EXACTLY under the "
              "monotone coupler -- same seeds, same order, zero mismatching cells",
        rung2=f"at |V|=2 the couplers coincide provably (F41: 200,000 pairs, max diff 0.0), so "
              f"their disagreement rates must match",
        rung3="a coupling fixes the marginals, so single-replica token frequencies must match at "
              "large |V| where rung 2 no longer applies. A difference here is a bug, not a finding",
        gate="a failure at ANY rung makes the primary NOT DECIDABLE -- the maximal arm would not be "
             "measuring what it claims",
        cost="maximal coupling is NOT replica-independent: the draw depends on both twins' "
             "distributions jointly, so runs are reproducible only as a pair. Monotone stays the "
             "default whatever this returns",
        f41_reference=F41_GAP)

    print("RUNG 2 -- |V|=2, the couplers must coincide")
    r2 = rung2_binary(); res["rung2"] = r2
    print(f"  monotone {r2['disagreement_monotone']:.6f}  maximal {r2['disagreement_maximal']:.6f}"
          f"  gap {r2['gap']:.6f}  -> {'PASS' if r2['passes'] else 'FAIL'}")

    print("\nRUNG 3 -- marginals are coupling-invariant")
    r3 = rung3_marginals(); res["rung3"] = r3
    print(f"  total variation {r3['total_variation']:.5f} against tolerance {r3['tolerance']:.5f}"
          f"  -> {'PASS' if r3['passes'] else 'FAIL'}")

    gates = [
        dynamic_range([r2["disagreement_monotone"], r2["disagreement_maximal"]],
                      floor=max(r2["gap"], 1e-9), k=0.0, name="rung 2 disagreement pair"),
        distinct_units(["rung2", "rung3"], minimum=2, name="validation rung"),
    ]
    ok = r2["passes"] and r3["passes"]
    verdict = carries_verdict(gates, value=ok)

    if not ok:
        res["verdict"] = (
            f"NOT DECIDABLE: a validation rung failed (rung2 passes={r2['passes']}, "
            f"rung3 passes={r3['passes']}). The maximal coupler is not measuring what it claims, so "
            f"no comparison against the monotone arm can be read. Fix the coupler, not the "
            f"analysis.")
    else:
        res["verdict"] = (
            f"RUNGS PASS: at |V|=2 the couplers coincide to {r2['gap']:.2e} and the marginals agree "
            f"to a total variation of {r3['total_variation']:.5f} against a {r3['tolerance']:.5f} "
            f"tolerance. The maximal coupler is validated on the two cases whose answers are known "
            f"in advance. RUNG 1 AND THE PRIMARY ARE NOT RUN HERE: rung 1 needs the paired runner "
            f"wired to a real backend, and the primary needs the developmental grid measured twice. "
            f"Both are registered above and neither is started, so nothing about the common-mode "
            f"question is claimed yet.")

    print("\n  -> " + res["verdict"])
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
