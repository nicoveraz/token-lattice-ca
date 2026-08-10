"""Calibrating F122: is colliding-front sub-additivity generic, or is the LM's number larger?

WHAT F122 LEFT UNEARNED. Two damage clouds superpose sub-additively on the LM ring (-2.52, -1.17,
-0.51 damaged sites at separations 6, 12, 24, against a causally-disconnected control at
-0.016 +/- 0.010) while the underlying two-token response is additive (F114). F122 states that as
"the lattice is not reducible to its local response", which is correct. The MECHANISM label --
competition for sites versus shared healing -- is not earned, and neither is the implicit novelty.

THE PRIOR, REGISTERED BEFORE THE NUMBER BECAUSE IT IS THE LIKELY ANSWER. Colliding fronts in
absorbing-state systems should be sub-additive classically: two fires burn less than twice one fire,
because the overlap region can only be damaged once and both fronts compete for the same sites. So
the EXPECTED verdict is "generic collision, construction-level", which DEFLATES F122's novelty.
Recording that expectation up front is what makes the alternative outcome -- an LM interaction
clearly exceeding DK's -- worth anything if it occurs. The rung earns its keep either way; it only
earns it honestly if the prior is on record first.

WHY DK IS THE RIGHT HOST. Domany-Kinzel is this project's known-answer system: single-seed damage is
already bit-exact against it (F38), it is in the directed-percolation class the LM ring was tested
against (F57-F61), and it is pure numpy -- no model, no GPU, no tokenizer.

FROZEN BEFORE THE RUN, all four:
  STATISTIC  interaction magnitude NORMALISED by the single-cone area at matched separation, with
             separations expressed in units of CONE WIDTH rather than sites. Frozen so that
             "comparable to DK" versus "exceeds DK" is decided by a criterion predating the data.
             Raw site counts are not comparable across systems with different cone geometries.
  GEOMETRY   N = 4096, far larger than the LM ring, so every separation sits inside the
             pre-collision window and the wraparound that broke damage_geometry twice (F119, 7a)
             cannot enter. The LM's separations are mapped proportionally, in cone widths.
  GATE       F42-style: BOTH seeds must ignite in a replica or that replica is UNDEFINED and
             dropped, not counted as zero interaction. An unignited seed contributes no cone, and
             averaging its zero into the interaction would manufacture sub-additivity out of
             extinction.
  PRIOR      sub-additive is the expected classical outcome (above).

PRE-REGISTERED READING:
  RUNG      single-seed DK damage must reproduce the known behaviour at the calibration point
            before any two-seed number is read.
  PRIMARY   normalised interaction on DK across separations. If the LM's normalised interaction sits
            inside DK's range, F122 is generic collision and its mechanism label stays open but its
            novelty is gone. If it sits clearly outside, the LM lattice does something DK does not.
  BOUNDARY  DK is a binary-alphabet, synchronous, two-parameter automaton; the LM ring is
            multi-state and asynchronous. A match in normalised interaction does not mean the same
            mechanism, only that the magnitude needs no special explanation.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, time

import numpy as np
from dk import dk_step
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "dk_interaction.json")
LM = str(_ROOT / "results" / "damage_interaction.json")
N = 4096
SWEEPS = 64
B = 64
# P1 CHOSEN AGAINST THE IGNITION GATE, MEASURED BEFORE THE RUN, NOT TUNED TO AN ANSWER. DK's
# DAMAGE-SPREADING line is not its density transition: at p1=0.72 only 15.6% of replicas still carry
# damage at sweep 48 (cone width 7.8), which is why the first version measured a 0.7-site cone and
# collapsed every separation to the floor. Scanned: 0.65->6%, 0.72->16%, 0.80->59%, 0.85->84%,
# 0.90->97%, 0.95->98%. 0.90 is the first point where the F42-style gate is comfortably satisfied.
P1, P2 = 0.90, 0.0
SEPS_WIDTHS = [0.25, 0.5, 1.0, 2.0]   # separations in units of cone width, mapped from the LM's
SEEDS = [11, 12, 13, 14]
EXPECTED = "sub-additive (negative): colliding absorbing-state fronts compete for the same sites"


def cones(rng, sep, sweeps=SWEEPS):
    """Damage sets for seed A alone, seed B alone, and both together, under COMMON noise.

    Returns per-replica boolean damage masks over (sweeps, N) for the three arms. Common random
    numbers across all four runs (base + three perturbed) is what makes the set comparison exact
    rather than statistical -- the same discipline as the LM side.
    """
    u = rng.random((sweeps, B, N)).astype(np.float32)
    s0 = (rng.random((B, N)) < 0.5).astype(np.int8)
    a, b = N // 2 - sep // 2, N // 2 + sep - sep // 2
    states = {}
    for name, flips in (("base", []), ("A", [a]), ("B", [b]), ("AB", [a, b])):
        s = s0.copy()
        for j in flips:
            s[:, j] ^= 1
        acc = np.zeros((sweeps, B, N), dtype=bool)
        states[name] = (s, acc)
    for t in range(sweeps):
        for name, (s, acc) in states.items():
            states[name] = (dk_step(s, u[t], P1, P2), acc)
            states[name][1][t] = states[name][0].astype(bool)
    base = states["base"][1]
    return {k: (states[k][1] ^ base) for k in ("A", "B", "AB")}


def measure(sep, seed):
    rng = np.random.default_rng(seed)
    d = cones(rng, sep)
    # F42-STYLE GATE: a replica where either single seed fails to ignite is UNDEFINED, not zero.
    igA, igB = d["A"].any(axis=(0, 2)), d["B"].any(axis=(0, 2))
    live = igA & igB
    if not live.any():
        return None
    A, Bm, AB = d["A"][:, live], d["B"][:, live], d["AB"][:, live]
    union = A | Bm
    inter = AB.sum(axis=(0, 2)).astype(float) - union.sum(axis=(0, 2)).astype(float)
    area = 0.5 * (A.sum(axis=(0, 2)) + Bm.sum(axis=(0, 2))).astype(float)
    return dict(n_live=int(live.sum()), n_total=int(B),
                interaction=float(inter.mean()), interaction_sd=float(inter.std(ddof=1)),
                single_area=float(area.mean()),
                normalised=float((inter / np.maximum(area, 1e-12)).mean()),
                normalised_sd=float((inter / np.maximum(area, 1e-12)).std(ddof=1)))


def analyse(res):
    cells, parts = res["cells"], []
    live = {k: v for k, v in cells.items() if v}
    ok = bool(live) and all(v["n_live"] > 0.5 * v["n_total"] for v in live.values())
    parts.append(
        f"RUNG (ignition, F42-style): {len(live)} of {len(cells)} cells have both seeds igniting in "
        f"a majority of replicas"
        + ("; damage is live at this DK point so the two-seed comparison is defined."
           if ok else " -- NOT met, so the interaction is measured on extinguishing seeds and "
                      "nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False)
        res["verdict"] = " ".join(parts); return
    by_w = {}
    for v in live.values():
        by_w.setdefault(v["sep_widths"], []).append(v["normalised"])
    dk_range = (min(min(x) for x in by_w.values()), max(max(x) for x in by_w.values()))
    parts.append(
        f"REGISTERED PRIOR, on record before the number: {EXPECTED}. A sub-additive DK result is "
        f"therefore the EXPECTED outcome and deflates F122's novelty rather than confirming a "
        f"discovery.")
    parts.append(
        "PRIMARY, normalised interaction (interaction / single-cone area) by separation in cone "
        "widths: "
        + ", ".join(f"{w}w={np.mean(vs):+.4f}" for w, vs in sorted(by_w.items()))
        + f". DK range [{dk_range[0]:+.4f}, {dk_range[1]:+.4f}].")
    try:
        lm = json.load(open(LM))
        rows = lm.get("analysis", {}).get("rows") or {}
        parts.append(
            "CONTRAST with the LM ring (F122): its raw interactions were -2.52, -1.17, -0.51 "
            "damaged sites at separations 6, 12, 24 on N=96. Normalising those against its own "
            "single-cone areas is required before the comparison is meaningful, and that "
            "normalisation is NOT computed here -- damage_interaction stored per-replica "
            "interaction but the single-arm areas needed for the denominator are not in its "
            "results file. So this run establishes the DK reference and the LM side needs a "
            "re-analysis to land on the same axis. Stated rather than approximated.")
    except Exception:
        pass
    parts.append(
        f"BOUNDARY: DK is binary-alphabet, synchronous and two-parameter (p1={P1}, p2={P2}); the LM "
        f"ring is multi-state and asynchronous. Matching normalised interaction would mean the "
        f"MAGNITUDE needs no special explanation, not that the mechanism is the same. N={N}, "
        f"{SWEEPS} sweeps, B={B}, {len(SEEDS)} seeds.")
    res["analysis"] = dict(rung_passes=True, by_width={str(k): v for k, v in by_w.items()},
                           dk_range=list(dk_range))
    res["verdict"] = " ".join(parts)


def main():
    res = {"cells": {}, "_preregistration": dict(
        N=N, sweeps=SWEEPS, B=B, p1=P1, p2=P2, seps_widths=SEPS_WIDTHS, seeds=SEEDS,
        expected=EXPECTED,
        statistic="interaction / single-cone area, separations in units of cone width -- frozen "
                  "before the run so 'comparable to DK' vs 'exceeds DK' predates the data",
        geometry=f"N={N}, chosen so every separation sits inside the pre-collision window and the "
                 f"wraparound that broke damage_geometry twice cannot enter",
        gate="both seeds must ignite in a replica or that replica is UNDEFINED, not zero",
        lm_reference=rel(LM))}
    # cone width at this DK point, measured once, so separations can be set in width units
    rng = np.random.default_rng(SEEDS[0])
    probe = cones(rng, 8)
    # LIVE REPLICAS ONLY. Counting an extinguished replica as width 0 is the same error as counting
    # it as zero interaction -- it averages an absence into a measurement.
    ws = [np.ptp(np.where(probe["A"][-1, b])[0]) for b in range(min(B, 64))
          if probe["A"][-1, b].any()]
    if not ws:
        raise RuntimeError(f"no damage survives at p1={P1}: the cone width is undefined")
    width = float(np.mean(ws))
    res["cone_width"] = width
    print(f"  measured cone width at p1={P1}, {SWEEPS} sweeps: {width:.1f} sites")
    for w in SEPS_WIDTHS:
        sep = max(2, int(round(w * width)))
        for sd in SEEDS:
            key = f"w{w}|s{sd}"
            t0 = time.time()
            m = measure(sep, sd)
            if m:
                m.update(sep_sites=sep, sep_widths=w, seed=sd, secs=round(time.time() - t0, 1))
            res["cells"][key] = m
            print(f"  {key:<12} sep={sep:<5} " +
                  (f"interaction={m['interaction']:+.3f} area={m['single_area']:.1f} "
                   f"normalised={m['normalised']:+.4f} live={m['n_live']}/{m['n_total']} "
                   f"({m['secs']:.0f}s)" if m else "NO LIVE REPLICAS"), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
