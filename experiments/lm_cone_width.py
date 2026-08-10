"""Measure the LM cone width the way DK's was measured, so F131's ratio means something.

WHAT F131 LEFT UNRESOLVED. The DK calibration compared normalised interaction at separations
expressed in CONE WIDTHS, and the two sides were not measured the same way: DK's width was measured
(71.9 sites, from the damaged-site extent of live replicas at the final sweep), while the LM's was a
THEORETICAL bound, `r * sweeps` = 44. F119 established that bound is wrong for this construction --
asynchronous random-order updating lets a site damaged early in a sweep pass damage down a chain of
later-visited neighbours, so the reach inside one sweep is set by the visit order, not by r. The
measured front reached offset 24 by sweep 8 where r*t was 16.

WHY THE DIRECTION SURVIVES ANYWAY AND THE RATIO DOES NOT. A larger true width moves the LM's
separations to SMALLER values in width units, i.e. toward where DK's interaction is strongest -- so
correcting the width can only make the LM look weaker relative to DK, never stronger. F131's
conclusion (the LM's interaction is sub-generic, not super-generic) is therefore safe under this
correction, and only the 8-50x figure moves. This script replaces that figure with a measured one.

SAME GEOMETRY AS F122, exactly: damage_interaction's own N, r, T, block and sweeps, read from its
preregistration rather than restated here, so the width belongs to the run it is used to normalise.

PRE-REGISTERED:
  RUNG      the measured width must EXCEED the synchronous bound r*sweeps. If it does not, either
            the asynchronous reach argument (F119) is wrong at this geometry or the measurement is
            not finding the front, and the correction is not applied.
  PRIMARY   the width, and F131's LM separations re-expressed in it.
  BOUNDARY  width is measured on live replicas at the final sweep, the same definition DK used; a
            different definition (10-90% flank, second moment) would give a different number, and
            the point is that BOTH sides use one definition rather than that this one is canonical.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, time

import numpy as np
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "lm_cone_width.json")
LM = str(_ROOT / "results" / "damage_interaction.json")
DK = str(_ROOT / "results" / "dk_interaction.json")
SEEDS = [21, 22, 23, 24]


def main():
    src = json.load(open(LM))
    pre = src["_preregistration"]
    MODEL = pre["model"]
    R, T, N, B = pre["r"], pre["T"], pre["N"], pre["B"]
    SETTLE, SWEEPS, BLOCK = pre.get("settle", 12), pre["sweeps"], pre["block"]
    STEPS = pre["steps"]
    res = {"cells": {}, "_preregistration": dict(
        source=rel(LM), dk_reference=rel(DK), model=MODEL, r=R, T=T, N=N, B=B,
        sweeps=SWEEPS, block=BLOCK, seeds=SEEDS,
        definition="max minus min damaged-site offset at the final sweep, live replicas only -- "
                   "the same definition dk_interaction used, so the two sides are comparable",
        rung="the measured width must exceed the synchronous bound r*sweeps; if not, the "
             "asynchronous-reach argument fails here and no correction is applied",
        synchronous_bound=R * SWEEPS)}
    from ar_ca import ARRule, run
    rule = ARRule(MODEL, revision=f"step{STEPS[-1]}")
    widths = []
    for sd in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(sd)
        base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none", init="random",
                   seed=sd)["final"]
        c = N // 2
        idx = [c + k for k in range(-(BLOCK // 2), BLOCK - BLOCK // 2)]
        fl = base.copy()
        for j in idx:
            fl[:, j] = rng.choice(rule.init_pool, size=B)
        u = np.random.default_rng(sd + 1).random(SWEEPS * N * B)
        u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
        c2 = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                 init_state=np.concatenate([base, fl], axis=0), seed=sd + 2, u_stream=u2)
        s = c2["snaps"]
        diff = (s[:, :B] != s[:, B:])
        off = (np.arange(N) - c + N // 2) % N - N // 2
        final = diff[-1]
        ws = [float(np.ptp(off[final[b]])) for b in range(B) if final[b].any()]
        w = float(np.mean(ws)) if ws else float("nan")
        res["cells"][f"s{sd}"] = dict(seed=sd, width=w, n_live=len(ws), n_total=B,
                                      secs=round(time.time() - t0, 1))
        widths.append(w)
        print(f"  seed {sd}: width={w:.1f} sites over {len(ws)}/{B} live replicas "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    live = [w for w in widths if np.isfinite(w)]
    mean_w = float(np.mean(live)) if live else float("nan")
    bound = R * SWEEPS
    ok = np.isfinite(mean_w) and mean_w > bound
    parts = [f"RUNG: measured cone width {mean_w:.1f} sites against the synchronous bound "
             f"r*sweeps = {bound}. "
             + ("The measured reach EXCEEDS the bound, confirming F119's asynchronous-reach "
                "argument at this geometry, so the correction below is applied."
                if ok else
                "The measured reach does NOT exceed the bound, so F119's argument does not hold "
                "here and F131's width mapping is left as it was.")]
    if ok:
        seps = sorted({c["sep"] for c in src["cells"].values() if isinstance(c, dict)
                       and "sep" in c})
        parts.append(
            "PRIMARY: F131 expressed the LM's separations against the bound (44 sites); measured, "
            "they are "
            + ", ".join(f"sep {s} = {s / mean_w:.2f}w (was {s / bound:.2f}w)" for s in seps)
            + ". Every separation moves to a SMALLER width fraction, i.e. toward the regime where "
              "DK's interaction is strongest, so the LM's interaction is weaker relative to DK than "
              "F131 reported -- its direction is unchanged and its 8-50x figure is a floor.")
    parts.append(
        f"BOUNDARY: width is max-minus-min damaged offset at the final sweep over live replicas, the "
        f"definition dk_interaction used. A different definition (10-90% flank, second moment) gives "
        f"a different number; the point is that both sides now use ONE definition, not that this one "
        f"is canonical. {len(live)} of {len(SEEDS)} seeds contributed.")
    res["analysis"] = dict(measured_width=mean_w, synchronous_bound=bound, rung_passes=bool(ok),
                           per_seed=widths)
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
