"""One source of truth for the DP calibration gate (#82; the fix for F56).

WHY THIS IS A MODULE AND NOT A COPY. F56 was a calibration measured at one lattice geometry and
applied to another: the tolerance came from Domany-Kinzel at N=512 over 200 sweeps, the language
model ran at N=96 over 40, and the resulting verdict ("not in the DP class") rejected directed
percolation using a threshold that rejects DK itself. The repair is that every DP run measures its
own bias, at its own geometry, with the same code. The moment that code is pasted into a second
script the two can drift, and a drifted gate is indistinguishable from the defect it exists to
prevent -- so there is exactly one implementation and the scripts import it.

WHAT THE GATE ASSERTS. Before any statement about the language model is licensed, the identical
estimator must be shown to recover KNOWN exponents at the SAME (N, sweeps, replicas, fit_from) on
Domany-Kinzel, whose p2=0 line is textbook 1+1D DP and whose simulator F38 established is
bit-exact against an independent implementation. Any error is therefore the fit's.

Three things the gate does that a naive version would not:

  * It is evaluated on DK ALONE, blind to the LM numbers, so it cannot be tuned to the answer.
  * It averages over many seeds. At 5 seeds the recommendation was noise -- N=192/80 sweeps landed
    inside the gate at 5 seeds and outside it at 20 -- and a multi-hour compute decision was
    resting on that. DK is pure numpy and free; there is no reason to be stingy here.
  * It requires the deviation PLUS its own seed-to-seed spread to clear tolerance. A mean that
    lands inside by less than its own scatter has demonstrated nothing, and the first version of
    this gate failed by 0.7 points against a 9.6-point spread -- reporting a coin flip as a
    decision.

WHICH CRITICAL POINT. The DK p2=0 damage-spreading line is disputed -- 0.801(2) (Zebende & Penna)
vs 0.8087(5) (Hinrichsen et al.), a disagreement the paper reports and declines to adjudicate. The
bias is measured at both and the KINDER one kept: if even the most favourable p_c fails the gate,
the failure is robust to that dispute rather than an artifact of picking a side.

No model, no GPU, no network.
"""
import numpy as np

# Jensen's 1+1D directed percolation exponents
DP = dict(delta=0.159464, theta=0.313686, z=1.580745)

CAL_PC = [0.8087, 0.801]                    # the two disputed DK p2=0 values
CAL_TOL = 20.0                              # percent; the tolerance dp_pipeline_validation set
CAL_SEEDS = list(range(1000, 21000, 1000))  # 20; five proved too noisy to gate a decision on


def slope(t, y):
    """Log-log slope and R^2 -- the estimator every DP script in this project shares."""
    ok = y > 0
    if ok.sum() < 4:
        return None, None
    lt, ly = np.log(t[ok]), np.log(y[ok])
    c = np.polyfit(lt, ly, 1)
    r2 = 1 - np.sum((ly - np.polyval(c, lt)) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-12)
    return float(c[0]), float(r2)


def dk_exponents(p1, n, sweeps, replicas, fit_from, seed):
    """Fit delta and theta on Domany-Kinzel with the LM scripts' geometry and conventions.

    Same single-site seed, same `slope`, same fit_from, and the initial state is dropped so the
    time axis runs 1..sweeps exactly as the LM `trajectory` functions return it.
    """
    from dk import dk_run
    P = np.zeros(sweeps); Nt = np.zeros(sweeps); done = 0
    while done < replicas:
        b = min(512, replicas - done)
        s0 = np.zeros((b, n), dtype=np.int8); s0[:, n // 2] = 1
        u = np.random.default_rng(seed + done).random(sweeps * n * b)
        a = np.asarray(dk_run(s0, u, p1=p1, p2=0.0, sweeps=sweeps))[1:]   # drop t=0
        c = a.sum(axis=2)
        P += (c > 0).sum(axis=1); Nt += c.sum(axis=1); done += b
    P /= replicas; Nt /= replicas
    t = np.arange(1, sweeps + 1, dtype=float); m = t >= fit_from
    sd, _ = slope(t[m], P[m]); st, _ = slope(t[m], Nt[m])
    if sd is None or st is None:
        return None
    return dict(delta=round(-sd, 4), theta=round(st, 4),
                delta_pct=round(abs(-sd - DP["delta"]) / DP["delta"] * 100, 1),
                theta_pct=round(abs(st - DP["theta"]) / DP["theta"] * 100, 1))


def bias_at(n, sweeps, replicas, fit_from):
    """Seed-averaged bias at one geometry, at whichever disputed p_c is kinder to the estimator."""
    cands = []
    for p in CAL_PC:
        cs = [c for c in (dk_exponents(p, n, sweeps, replicas, fit_from, seed=s)
                          for s in CAL_SEEDS) if c]
        if not cs:
            continue
        d = [c["delta"] for c in cs]; t = [c["theta"] for c in cs]
        cands.append(dict(
            p_c=p, seeds=len(cs),
            delta=round(float(np.mean(d)), 4), theta=round(float(np.mean(t)), 4),
            delta_sd=round(float(np.std(d)), 4), theta_sd=round(float(np.std(t)), 4),
            delta_pct=round(abs(float(np.mean(d)) - DP["delta"]) / DP["delta"] * 100, 1),
            theta_pct=round(abs(float(np.mean(t)) - DP["theta"]) / DP["theta"] * 100, 1),
            delta_sd_pct=round(float(np.std(d)) / DP["delta"] * 100, 1),
            theta_sd_pct=round(float(np.std(t)) / DP["theta"] * 100, 1)))
    return min(cands, key=lambda c: max(c["delta_pct"], c["theta_pct"])) if cands else None


def decides(c):
    """Is a geometry DEMONSTRABLY adequate -- error PLUS its own spread inside tolerance?

    A bare `pct <= CAL_TOL` is not enough: the first gate failed by 0.7 points while the
    seed-to-seed spread on theta was 9.6 points, which is a coin flip reported as a decision.
    Licensing a claim about the LM requires showing the estimator works at that geometry, and a
    margin swamped by its own noise shows nothing either way.
    """
    return bool(c and c["delta_pct"] + c["delta_sd_pct"] <= CAL_TOL
                and c["theta_pct"] + c["theta_sd_pct"] <= CAL_TOL)


def calibrate(n, sweeps, replicas, fit_from, grid, secs_per_run, n_runs):
    """Bias at this run's geometry, plus a ladder saying which geometry WOULD decide, and its cost.

    `secs_per_run` is measured from the run's own recorded timings, and cost is modelled as
    sequential in sweeps*N -- which is what the measurement is: sweeps*N forward passes whose
    order cannot be parallelised. The ladder exists so "get more compute" is a number rather than
    a gesture.
    """
    here = bias_at(n, sweeps, replicas, fit_from)
    if here:
        here = dict(here, N=n, sweeps=sweeps)     # stated, so a test can catch the two drifting
    out, unit = {}, secs_per_run / (n * sweeps)
    for gn, gsw in grid:
        c = bias_at(gn, gsw, replicas, fit_from)
        if not c:
            continue
        out[f"N{gn}_sw{gsw}"] = dict(
            c, N=gn, sweeps=gsw, passes=decides(c),
            projected_hours=round(unit * gn * gsw * n_runs / 3600, 1))
    passing = sorted((c for c in out.values() if c["passes"]),
                     key=lambda c: c["projected_hours"])
    return dict(at_run_geometry=here, tolerance_pct=CAL_TOL, p_c_candidates=CAL_PC,
                calibration_seeds=CAL_SEEDS, replicas=replicas, fit_from=fit_from, grid=out,
                gate="mean deviation PLUS its seed-to-seed spread must clear the tolerance",
                geometry_decides=decides(here),
                cheapest_passing=(passing[0] if passing else None))


def print_ladder(cal, n, sweeps):
    """The shared rendering, so two scripts cannot disagree about what the ladder means."""
    print(f"\n=== the SAME estimator on Domany-Kinzel, where the exponents are known ===")
    print(f"  deviation from Jensen, mean +/- seed spread over {len(CAL_SEEDS)} seeds")
    print(f"  {'geometry':>16} {'delta dev':>16} {'theta dev':>16} {'LM hours':>9}")
    cp = cal["cheapest_passing"]
    for k, c in cal["grid"].items():
        mark = "  <- this run" if (c["N"], c["sweeps"]) == (n, sweeps) else \
               ("  <- cheapest that decides" if cp and k == f"N{cp['N']}_sw{cp['sweeps']}" else "")
        print(f"  N={c['N']:<4} sweeps={c['sweeps']:<4} "
              f"{c['delta_pct']:>8.1f} +/-{c['delta_sd_pct']:>4.1f}% "
              f"{c['theta_pct']:>8.1f} +/-{c['theta_sd_pct']:>4.1f}% "
              f"{c['projected_hours']:>9.1f}{mark}")
    h = cal["at_run_geometry"]
    print(f"\n  Jensen: delta={DP['delta']}, theta={DP['theta']}")
    print(f"  pipeline bias AT THIS GEOMETRY: delta {h['delta_pct']}+/-{h['delta_sd_pct']}%, "
          f"theta {h['theta_pct']}+/-{h['theta_sd_pct']}%")
    print(f"  gate ({cal['gate']}, tol {CAL_TOL:.0f}%) -> {cal['geometry_decides']}")
