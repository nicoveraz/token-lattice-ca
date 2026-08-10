"""Can a WIDER window pay for a SMALLER vocabulary? The r-ladder F94's mean field makes precise.

THE TRADE. F94 puts damage criticality at s = 1/r, and F110 established that the branching ratio is
literally the SUM of the per-position contributions, so the criterion is `sum_pos s_pos >= 1`.
Restricting the vocabulary lowers the per-position s; adding window positions adds terms. Whether
more tokens can buy back what a smaller alphabet costs is therefore a question with an exact target
rather than a metaphor.

WHY THIS RUNS AFTER F123/F124 AND NOT BEFORE. The obvious version of this experiment -- ladder the
semantic alphabets and see when they cross 1 -- would have been aimed at the wrong target. F123
showed the sub-alphabet lattice is subcritical because of the STATE IT SETTLES INTO, not because the
alphabet is small, and that the SELECTION RULE moves s_far by up to 0.588 at fixed size. So the
ladder is run on all three selection modes: if widening the window only rescues the arms that were
already near criticality, the trade is not general.

WHAT DECIDES IT. Per-position s does not have to decay slowly. If s_pos falls off geometrically with
distance the sum converges below 1 and NO window size reaches criticality; if it decays slowly the
sum crosses. F110 measured slow decay on the FULL vocabulary (0.579 at i-2, 0.535 at i-3, an 8%
drop). Whether that survives restriction is exactly what is unmeasured.

PRE-REGISTERED:
  RUNG      at r = 2 the per-position values must reproduce `selection_mode.json`'s stored
            s_far/s_near within RUNG_TOL for every arm. That pins model, geometry, estimator and
            settle to F123's; a mismatch stops the read.
  PRIMARY   for each arm, the smallest r at which sum_pos s_pos >= 1 on the SETTLED pool -- the
            pool the lattice actually occupies. Reported as a crossing radius, or None.
  SECONDARY the decay of s_pos with distance, fitted as a ratio between consecutive positions. A
            ratio near 1 means the sum grows without bound; well below 1 means it converges and the
            crossing radius may not exist.
  CONTROL   the same on the uniform pool, which F123 showed differs by 1.6-3.5x on semantic arms.
  BOUNDARY  one model, one revision, one temperature. This measures the CONSTRUCTION, not a model
            property, and a lattice at large r is drifting toward ordinary conditional generation --
            where the object stops deserving the name CA is a judgement this cannot make.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, time

import numpy as np, torch
from subalphabet import make_sampler, sub_init
from subalphabet_why import MODEL, REV, B, N, SETTLE, N_CTX
from meanfield_lambda import s_crn
from provenance import stamp, rel

SRC = str(_ROOT / "results" / "selection_mode.json")
OUT = str(_ROOT / "results" / "window_ladder.json")
T = 0.7
RADII = [2, 3, 4, 6]
SEED = 20260810
RUNG_TOL = 0.10


def s_at(rule, ids, pool, r, pos, rng, n_ctx=N_CTX):
    """Exact mean CRN disagreement when window position `pos` is flipped, at radius r."""
    ids = np.asarray(ids, dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    rows = []
    for _ in range(n_ctx):
        w = [int(x) for x in rng.choice(pool, size=r)]
        a = list(w)
        while a[pos] == w[pos]:
            a[pos] = int(rng.choice(pool))
        rows += [w, a]
    with torch.no_grad():
        lg = rule.model(input_ids=torch.tensor(rows, device=rule.device)
                        ).logits[:, -1].float().cpu().double().numpy()
    P = lg[:, ids]
    P = np.exp((P - P.max(axis=1, keepdims=True)) / T)
    P = P / P.sum(axis=1, keepdims=True)
    return float(np.mean([s_crn(P[2 * i], P[2 * i + 1]) for i in range(n_ctx)]))


def analyse(res):
    cells, parts = res["cells"], []
    ref = json.load(open(SRC))["cells"]
    errs = []
    for key, c in cells.items():
        if c["r"] != 2:
            continue
        r0 = ref.get(c["arm"])
        if r0:
            errs.append(max(abs(c["s_pos_settled"][0] - r0["s_far"]),
                            abs(c["s_pos_settled"][1] - r0["s_near"])))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst <= RUNG_TOL
    parts.append(
        f"RUNG: at r=2 the per-position values reproduce selection_mode's stored s_far/s_near to "
        f"within {worst:.4f} across {len(errs)} arms (tolerance {RUNG_TOL}). "
        + ("Same lattice as F123, so the ladder extends it rather than replacing it."
           if ok else "NOT reproduced -- a different lattice, so nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst_err=worst)
        res["verdict"] = " ".join(parts); return
    arms = sorted({c["arm"] for c in cells.values()})
    cross, decay = {}, {}
    for arm in arms:
        byr = {c["r"]: c for c in cells.values() if c["arm"] == arm}
        br = {r: byr[r]["branching_settled"] for r in sorted(byr)}
        hit = [r for r in sorted(br) if br[r] >= 1.0]
        cross[arm] = dict(branching={str(k): round(v, 4) for k, v in br.items()},
                          crossing_r=(hit[0] if hit else None))
        big = byr[max(byr)]
        sp = big["s_pos_settled"]
        # ratio between consecutive FAR positions: does influence decay slowly or geometrically?
        ratios = [sp[i] / sp[i + 1] for i in range(len(sp) - 2) if sp[i + 1] > 1e-9]
        decay[arm] = round(float(np.mean(ratios)), 4) if ratios else None
    ncross = sum(1 for v in cross.values() if v["crossing_r"] is not None)
    parts.append(
        "PRIMARY, branching by radius on the settled pool: "
        + "; ".join(f"{a} " + "/".join(f"r{r}={v:.3f}" for r, v in cross[a]["branching"].items())
                    for a in arms) + ". "
        + f"{ncross} of {len(arms)} arms reach the criticality threshold of 1 within r <= "
          f"{max(RADII)}"
        + (": crossing radii " + ", ".join(f"{a}={cross[a]['crossing_r']}" for a in arms
                                           if cross[a]["crossing_r"] is not None) + ". "
           if ncross else ". ")
        + ("Widening the window DOES buy back what a smaller alphabet costs, and the mean-field "
           "trade s = 1/r is usable as a design rule."
           if ncross > len(arms) / 2 else
           "Widening the window does NOT generally rescue a restricted lattice, so the mean-field "
           "trade is not available as a design rule at these sizes."))
    parts.append(
        "SECONDARY, decay of s_pos with distance (ratio between consecutive positions at the "
        "largest radius): "
        + ", ".join(f"{a}={decay[a]}" for a in arms)
        + ". A ratio near 1 means the sum grows with r; well above 1 means it converges and no "
          "window size crosses.")
    parts.append(
        f"BOUNDARY: {MODEL} {REV}, T = {T}, N = {N}, {N_CTX} windows per position. This measures "
        f"the CONSTRUCTION, not a model property. At large r the lattice drifts toward ordinary "
        f"conditional generation and stops deserving the name CA; where that line sits is not "
        f"something this run can decide.")
    res["analysis"] = dict(rung_passes=True, rung_worst_err=worst, crossings=cross, decay=decay,
                           n_crossing=ncross, n_arms=len(arms))
    res["verdict"] = " ".join(parts)


def main():
    src = json.load(open(SRC))["cells"]
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"cells": {}}
    res["_preregistration"] = dict(
        model=MODEL, revision=REV, T=T, radii=RADII, N=N, B=B, settle=SETTLE, n_ctx=N_CTX,
        seed=SEED, source=rel(SRC), rung_tol=RUNG_TOL,
        rung="at r=2 the per-position values must reproduce selection_mode's stored s_far/s_near",
        primary="smallest r at which sum_pos s_pos >= 1 on the settled pool, per arm",
        target="F94's criticality, which F110 showed is sum_pos s_pos = 1")
    from ar_ca import ARRule, run
    rule = ARRule(MODEL, revision=REV)
    for arm, c in src.items():
        ids = np.array(c["ids"], dtype=np.int64)
        for r in RADII:
            key = f"{arm}|r{r}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            rng = np.random.default_rng(SEED)
            smp = make_sampler(ids, None)
            settled = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme="none",
                          init_state=sub_init(ids, B, N, rng), seed=SEED, sampler=smp)["final"]
            pool = settled.reshape(-1)
            sp = [s_at(rule, ids, pool, r, p, np.random.default_rng(SEED + p)) for p in range(r)]
            su = [s_at(rule, ids, ids, r, p, np.random.default_rng(SEED + p)) for p in range(r)]
            res["cells"][key] = dict(
                arm=arm, alphabet=c["alphabet"], mode=c["mode"], k=c["k"], r=r,
                s_pos_settled=[round(v, 5) for v in sp],
                s_pos_uniform=[round(v, 5) for v in su],
                branching_settled=round(float(sum(sp)), 5),
                branching_uniform=round(float(sum(su)), 5),
                distinct=int(len(set(pool.tolist()))), secs=round(time.time() - t0, 1))
            print(f"  {key:<28} branching(settled)={sum(sp):.4f}  (uniform)={sum(su):.4f}  "
                  f"s_pos={[round(v, 3) for v in sp]}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
