"""Phase 2.2 -- the Domany-Kinzel rung: an exact anchor plus two published critical points.

The ladder's other rungs are all the wrong regime for the token instrument. The logistic
map and the CML are smooth and infinitesimal (F30/F31/F37); the ECA rung is discrete but
deterministic. DK is the only rung that is stochastic AND discrete -- the instrument's own
regime -- and it comes with published numbers.

Four parts, in descending order of how much they prove:

  A. THE EXACT IDENTITY (the reason this rung is worth having). On the p2=0 line the damage
     field between CRN twins is itself a DK automaton at the same p1 (Kohring &
     Schreckenberg 1992). Predicted independently and compared bit for bit -- no error bar.
     `tests/test_dk_damage_identity.py` runs this through `lattice.run`, the loop that
     produces every LM number; here it is repeated at scale.

  B. DAMAGE survival vs ACTIVITY survival on p2=0. A corollary of A: the two curves must be
     statistically identical. This checks the identity survives the estimator, not just the
     arithmetic.

  C. The DP transition on the p2=0 (Wolfram-18) line, against the published value -- which
     the literature genuinely disagrees about: 0.801(2) (Zebende & Penna 1994) vs 0.8087(5)
     (Hinrichsen, Weitz & Domany 1997). Both are reported; neither is picked.

  D. The DP transition on the site-DP line p1=p2, against 0.705489(4) -- a seven-digit
     number, and the tightest anchor anywhere in this project.

C and D locate p_c by the standard local-slope method: at criticality the seed survival
probability decays as P(t) ~ t^-delta with delta = 0.159464, so the effective exponent
crosses that value at p_c. This is a ~1%-accurate method at these sizes and is reported as
such -- it is a calibration, not a measurement of DP exponents.

Writes results/dk_calib.json and fig/dk_ladder.png. Pure numpy, no model, no GPU.
Usage:  .venv/bin/python experiments/dk_calib.py
"""
import sys, pathlib, json, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from provenance import rel
import numpy as np

from dk import (ANCHORS, DP_DELTA, dk_run, dk_step, survival_from_seed,
                damage_survival_from_seed)

OUT = ROOT / "results" / "dk_calib.json"
FIG = ROOT / "fig" / "dk_ladder.png"

# survival-simulation size. N > 2*STEPS keeps the light cone off the periodic boundary.
N_TRIALS, STEPS, N_RING = 2000, 512, 1100
W18_GRID = [0.76, 0.78, 0.79, 0.80, 0.805, 0.81, 0.82, 0.84]
SITE_GRID = [0.68, 0.69, 0.695, 0.70, 0.705, 0.71, 0.715, 0.73]


# ------------------------------------------------------------------ A: the exact identity
def part_a_exact_identity(N=4096, steps=1500, p1s=(0.2, 0.5, 0.75, 0.8087, 0.95, 1.0),
                          seeds=(0, 1, 2)):
    """Damage field vs an independent DK run, at scale. Any nonzero mismatch is a failure."""
    print("A. exact damage identity on p2=0 (Kohring-Schreckenberg)")
    rows, worst = {}, 0
    for p1 in p1s:
        mism, dens = 0, []
        for sd in seeds:
            rng = np.random.default_rng(1000 + sd)
            a = rng.integers(0, 2, size=(1, N)).astype(np.int8)
            b = a.copy(); b[:, N // 2] ^= 1
            d = a ^ b
            for _ in range(steps):
                u = rng.random((1, N))              # ONE stream, shared (CRN)
                a = dk_step(a, u, p1, 0.0)
                b = dk_step(b, u, p1, 0.0)
                d = dk_step(d, u, p1, 0.0)          # predicted damage, evolved independently
                mism += int(((a ^ b) != d).sum())
            dens.append(float((a ^ b).mean()))
        worst = max(worst, mism)
        rows[f"{p1:g}"] = dict(mismatching_cells=mism,
                               final_damage_density=round(float(np.mean(dens)), 5))
        print(f"   p1={p1:<7g} mismatching cells over {steps} steps x {len(seeds)} seeds = "
              f"{mism:<3d}  final damage density={np.mean(dens):.4f}")

    # control: off the line the identity must break, or part A proves nothing
    ctrl = 0
    rng = np.random.default_rng(99)
    a = rng.integers(0, 2, size=(1, N)).astype(np.int8)
    b = a.copy(); b[:, N // 2] ^= 1
    d = a ^ b
    for _ in range(steps):
        u = rng.random((1, N))
        a, b, d = (dk_step(a, u, 0.6, 0.5), dk_step(b, u, 0.6, 0.5), dk_step(d, u, 0.6, 0.5))
        ctrl += int(((a ^ b) != d).sum())
    print(f"   control (p1=0.6, p2=0.5, off the line): {ctrl} mismatching cells "
          f"-- must be > 0, else the exact test is vacuous")
    return dict(by_p1=rows, max_mismatch=worst, control_offline_mismatch=ctrl,
                N=N, steps=steps, seeds=list(seeds))


# ------------------------------------------------------------------ p_c by local slope
def delta_eff(P, t_lo, t_hi):
    """Effective decay exponent of P(t) between t_lo and t_hi. At p_c this equals DP_DELTA."""
    if P[t_lo] <= 0 or P[t_hi] <= 0:
        return float("inf")                    # died out: steeper than any power law
    return -np.log(P[t_hi] / P[t_lo]) / np.log(t_hi / t_lo)


def locate_pc(grid, deltas):
    """Linear interpolation in p to where delta_eff crosses DP_DELTA. None if unbracketed."""
    for (pa, da), (pb, db) in zip(zip(grid, deltas), zip(grid[1:], deltas[1:])):
        if np.isfinite(da) and np.isfinite(db) and (da - DP_DELTA) * (db - DP_DELTA) <= 0:
            if da == db:
                return float(0.5 * (pa + pb))
            return float(pa + (DP_DELTA - da) * (pb - pa) / (db - da))
    return None


def survival_scan(grid, p2_of, label, damage=False):
    """Survival curves over a p grid; returns (p_c estimate, per-p records)."""
    kind = "damage" if damage else "activity"
    print(f"\n{label}  ({kind} survival, {N_TRIALS} seeds, {STEPS} steps, ring {N_RING})")
    fn = damage_survival_from_seed if damage else survival_from_seed
    recs, deltas = {}, []
    for p in grid:
        t0 = time.time()
        P = fn(p, p2_of(p), n_trials=N_TRIALS, N=N_RING, steps=STEPS, seed=int(p * 1e6))
        de = delta_eff(P, STEPS // 4, STEPS)
        deltas.append(de)
        recs[f"{p:g}"] = dict(P_final=float(P[-1]), delta_eff=None if not np.isfinite(de)
                              else round(float(de), 4),
                              P_curve=[round(float(x), 5) for x in P[::16]])
        print(f"   p={p:<7g} P({STEPS})={P[-1]:.4f}  delta_eff={de:7.4f}  "
              f"({time.time() - t0:.1f}s)")
    pc = locate_pc(grid, deltas)
    print(f"   -> p_c estimate (delta_eff = {DP_DELTA}): "
          f"{'%.4f' % pc if pc is not None else 'NOT BRACKETED by this grid'}")
    return pc, recs, deltas


def main():
    t_start = time.time()
    out = {}

    out["part_a_exact_identity"] = part_a_exact_identity()

    pc_w18, rec_w18, _ = survival_scan(W18_GRID, lambda p: 0.0,
                                       "C. Wolfram-18 line (p2=0): DP transition")
    pc_dmg, rec_dmg, _ = survival_scan(W18_GRID, lambda p: 0.0,
                                       "B. Wolfram-18 line (p2=0): DAMAGE transition",
                                       damage=True)
    pc_site, rec_site, _ = survival_scan(SITE_GRID, lambda p: p,
                                         "D. site-DP line (p1=p2): DP transition")

    out["part_c_w18_activity"] = dict(grid=W18_GRID, pc_estimate=pc_w18, by_p=rec_w18)
    out["part_b_w18_damage"] = dict(grid=W18_GRID, pc_estimate=pc_dmg, by_p=rec_dmg)
    out["part_d_site_dp"] = dict(grid=SITE_GRID, pc_estimate=pc_site, by_p=rec_site)

    print("\n=== calibration against published values ===")
    checks = {}
    def report(name, est, anchors):
        if est is None:
            print(f"  {name:22s} NOT BRACKETED -- no estimate, no claim")
            checks[name] = dict(estimate=None, verdict="unbracketed")
            return
        parts = []
        for key in anchors:
            a = ANCHORS[key]
            rel = abs(est - a["p1"]) / a["p1"] * 100
            parts.append(dict(anchor=key, published=a["p1"], published_err=a["err"],
                              rel_error_pct=round(float(rel), 2), ref=a["ref"]))
            print(f"  {name:22s} est={est:.4f}  vs {key} {a['p1']}({a['err']}) "
                  f"-> {rel:.2f}% off")
        checks[name] = dict(estimate=round(float(est), 4), against=parts)

    report("W18 activity", pc_w18, ["w18_zp", "w18_hwd"])
    report("W18 damage", pc_dmg, ["w18_zp", "w18_hwd"])
    report("site DP", pc_site, ["site_dp"])

    if pc_w18 is not None and pc_dmg is not None:
        gap = abs(pc_w18 - pc_dmg)
        checks["damage_vs_activity_gap_on_p2_zero"] = round(float(gap), 4)
        print(f"\n  corollary of the exact identity: damage and activity transitions on "
              f"p2=0 must coincide -> |gap| = {gap:.4f}")
    out["calibration"] = checks

    out["_note"] = (
        "Domany-Kinzel rung. Part A is the load-bearing result: on the p2=0 line the CRN "
        "damage field is EXACTLY a DK automaton at the same p1 (Kohring & Schreckenberg, "
        "J. Phys. I France 2, 2033 (1992)), so the damage machinery is verified bit-exactly "
        "rather than to within a critical point. Parts B-D are ~1%-accurate calibrations by "
        "the local-slope method at modest sizes; they are NOT measurements of DP exponents. "
        "The p2=0 anchor is reported against BOTH published values, which disagree: "
        "0.801(2) (Zebende & Penna 1994) and 0.8087(5) (Hinrichsen, Weitz & Domany 1997). "
        "On this BINARY alphabet CRN (one shared uniform, inverse CDF) coincides with HWD's "
        "maximal-correlation coupling, so the damage numbers in THIS file are a lower bound "
        "over admissible couplings. That does NOT extend to the language-model backends, "
        "where |V|>2 and inverse-CDF is the monotone coupling, not the maximal one; see "
        "experiments/coupling_gap.py (W2).")
    out["_runtime_s"] = round(time.time() - t_start, 1)
    out["_config"] = dict(n_trials=N_TRIALS, steps=STEPS, ring=N_RING,
                          dp_delta=DP_DELTA)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}  ({out['_runtime_s']}s)")
    make_figure(out)


def make_figure(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator, NullFormatter
    except Exception as e:                     # a missing plotting lib must not lose results
        print(f"(figure skipped: {e})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    t = np.arange(0, STEPS + 1, 16)
    for ax, key, title, anchors in (
            (axes[0], "part_c_w18_activity", r"$p_2=0$ (Wolfram-18)", ["w18_zp", "w18_hwd"]),
            (axes[1], "part_d_site_dp", r"site DP ($p_1=p_2$)", ["site_dp"])):
        for p, rec in out[key]["by_p"].items():
            y = np.array(rec["P_curve"], dtype=float)
            m = (y > 0) & (t > 0)
            ax.loglog(t[m], y[m], lw=1.0, label=f"$p$={p}")
        est = out[key]["pc_estimate"]
        sub = f"est $p_c$={est:.4f}" if est is not None else "no bracket"
        pub = ", ".join(f"{ANCHORS[a]['p1']:g}" for a in anchors)
        ax.set_title(f"{title}\n{sub}  (published {pub})", fontsize=9)
        ax.set_xlabel("t"); ax.set_ylabel("P(t)")
        # decade ticks only: matplotlib's default log minor labels collide on this aspect
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10))
            axis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1)))
            axis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=5.5, ncol=2, frameon=False, loc="lower left")
    fig.suptitle("Domany-Kinzel rung: seed survival probability, the DP order parameter",
                 fontsize=10)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=170)
    print("wrote", FIG)


if __name__ == "__main__":
    # --figure-only redraws from the saved JSON: the sweeps cost minutes, the plot does not
    if "--figure-only" in sys.argv:
        make_figure(json.load(open(OUT)))
    else:
        main()
