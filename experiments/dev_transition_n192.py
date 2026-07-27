"""A third lattice size for the developmental transition: is lambda_ca intensive, and is
D_norm 1/N?

F39 found a split at two sizes: lambda_ca's effect and level are size-stable (95% retention,
plateau levels +0.1683 vs +0.1686) while D_norm's halve (53% retention, 0.5689 vs 0.3062).
Two points cannot distinguish "intensive vs 1/N" from "any decreasing function", and
`paper.tex` names a third size as its own next requirement. This adds one.

The repo already demonstrates the same dichotomy for a different observable pair, which is
why the prediction below is sharp: `results/finite_size.json` has N=48/96/192 at T=0.7 with
order_mean flat (0.9610 / 0.9608 / 0.9616) and susceptibility falling
9.75e-4 -> 4.78e-4 -> 2.37e-4, i.e. ratios 0.490 and 0.495 -- 1/N to half a percent over two
doublings. So the instrument can separate an intensive observable from a 1/N one at three
sizes; this asks the same question of lambda_ca and D_norm.

MECHANISM UNDER TEST. lambda_ca is fitted on an UNSATURATED cone-growth slope
(FIT_KW: max_sweeps=8, frac_of_max=0.5), so it is a rate -- intensive by construction.
D_norm is a density ratio whose numerator (damage) stays localised inside a cone while its
denominator D0 (the independent-noise floor) is delocalised over every site, so D/D0 should
fall as 1/N. Measured at two sizes the level ratio is 1.858 and the gap ratio 1.881, both
just under the 2.000 of pure 1/N.

PRE-REGISTERED PREDICTIONS (written before running, 2026-07-26):
  * D_norm plateau at N=192, if 1/N:        0.142--0.153   (scaling from N=48 and N=96)
  * D_norm plateau at N=192, if intensive:  >= 0.25
    Per-cell SE is ~0.02, so these are separated by roughly 5 sigma. A value between
    0.17 and 0.24 falsifies both and is reported as "neither" rather than rounded to one.
  * lambda_ca plateau at N=192:             0.168 +- 0.02  (if intensive, as F39 indicates)
  * If lambda_ca's plateau lands outside that interval, F39's size-robustness claim is
    DOWNGRADED from "invariant across a 4x range" to "stable across 48->96 only".

Checkpoints: the pre-registered PRE set {256, 512} plus step143000 as the plateau. Using the
final checkpoint alone for the plateau is a deliberate economy -- it is the most conservative
single member (F39 shows no drift among 2000/8000/143000) and it keeps this to 24 runs.

Protocol identical to F39 by construction: `measure` imported, not reimplemented.
Incremental save + resume. Writes results/dev_transition_n192.json.
Usage:  caffeinate -i .venv/bin/python experiments/dev_transition_n192.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

from dev_transition_phase3 import measure          # identical protocol, not a copy
from provenance import stamp, rel
from lyapunov import run_ignited      # F42: lambda is undefined without a cone

STEPS = ["step256", "step512", "step143000"]
PRE = {"step256", "step512"}
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 192, 4                                      # B quartered from N=48 for 16 GB
OUT = str(_ROOT / "results" / "dev_transition_n192.json")

PREDICT = dict(
    D_norm_if_one_over_N=[0.142, 0.153],
    D_norm_if_intensive_at_least=0.25,
    lambda_ca_expected=[0.148, 0.188],
    prior_levels=dict(N48_lambda=0.1683, N96_lambda=0.1686,
                      N48_D_norm=0.5689, N96_D_norm=0.3062))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(N=N, B=B, steps=STEPS, seeds=SEEDS, predictions=PREDICT)
    runs = res["runs"]
    todo = [(st, sd) for st in STEPS for sd in SEEDS]
    print(f"N=192 developmental cell: {len(todo)} runs ({len(STEPS)} ckpts x {len(SEEDS)} seeds)")
    print(f"PRE-REGISTERED: D_norm plateau {PREDICT['D_norm_if_one_over_N']} if 1/N, "
          f">={PREDICT['D_norm_if_intensive_at_least']} if intensive; "
          f"lambda_ca {PREDICT['lambda_ca_expected']}", flush=True)
    for k, (st, sd) in enumerate(todo, 1):
        key = f"N192_{st}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        lam, dn, md, ig = measure(st, N, B, sd)
        runs[key] = dict(N=N, step=int(st.replace("step", "")), seed=sd,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} "
              f"({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    done = [v for v in runs.values() if "lambda_ca" in v]
    if len(done) < len(todo):
        print(f"partial: {len(done)}/{len(todo)}"); json.dump(res, open(OUT, "w"), indent=1); return

    def unignited(v):
        return not run_ignited(v)

    def sel(steps, m):
        """F42 asymmetry, matching dev_transition_shape.py exactly.

        lambda_ca: zero damage means NO CONE -> undefined -> drop the run.
        D_norm   : zero damage means the ratio is GENUINELY ZERO -> keep it.

        Applying the filter to BOTH (as the first version of this script did) silently biases
        D_norm upward by discarding its true zeros -- here it moved the plateau 0.1393 -> 0.1592,
        a 14% inflation of the quantity whose SIZE SCALING is the whole point of the run.
        """
        rows = [v for v in done if f"step{v['step']}" in steps]
        if m == "lambda_ca":
            rows = [v for v in rows if not unignited(v)]
        return np.array([v[m] for v in rows])

    # --- F42: ignition fraction is its own observable, reported before any lambda mean ---
    out = {"ignition": {}}
    print("\n=== ignition per cell (F42): lambda is UNDEFINED for unignited runs ===")
    for st in STEPS:
        rows = [v for v in done if f"step{v['step']}" == st]
        dead = [v for v in rows if unignited(v)]
        out["ignition"][st] = dict(
            n=len(rows), n_unignited=len(dead), n_ignited=len(rows) - len(dead),
            frac_ignited=round(1 - len(dead) / max(len(rows), 1), 4),
            unignited_lambdas=[v["lambda_ca"] for v in dead])
        print(f"  {st:>12}: {len(rows)-len(dead)}/{len(rows)} ignited"
              + (f"   DISCARDED lambdas {[v['lambda_ca'] for v in dead]}" if dead else ""))

    print("\n=== N=192 result vs the pre-registered predictions (IGNITED runs only) ===")
    for m in ("lambda_ca", "D_norm"):
        pre, post = sel(PRE, m), sel({"step143000"}, m)
        if len(pre) < 2 or len(post) < 2:
            print(f"  {m:>10}: too few ignited runs to summarise"); continue
        out[m] = dict(pre_mean=round(float(pre.mean()), 4), n_pre=len(pre),
                      plateau_mean=round(float(post.mean()), 4), n_plateau=len(post),
                      plateau_sd=round(float(post.std(ddof=1)), 4),
                      plateau_se=round(float(post.std(ddof=1) / np.sqrt(len(post))), 4),
                      gap=round(float(post.mean() - pre.mean()), 4),
                      basis=("ignited runs only (F42)" if m == "lambda_ca"
                             else "ALL runs -- zero damage is a true zero (F42 asymmetry)"))
        print(f"  {m:>10}  pre {pre.mean():+.4f} (n={len(pre)}) -> plateau "
              f"{post.mean():+.4f} (n={len(post)}, sd {post.std(ddof=1):.4f}, "
              f"se {post.std(ddof=1)/np.sqrt(len(post)):.4f})")
        # the rank test is immune to the discarded magnitudes, so it uses ALL runs
        from scipy import stats as _st
        rows_pre = [v[m] for v in done if f"step{v['step']}" in PRE]
        rows_post = [v[m] for v in done if v["step"] == 143000]
        u = _st.mannwhitneyu(rows_post, rows_pre, alternative="two-sided")
        out[m]["mannwhitney_all_runs_p"] = float(u.pvalue)
        print(f"{'':>14}rank test on ALL runs (ranks do not depend on a dead run's "
              f"magnitude): p={u.pvalue:.2e}")

    if "D_norm" not in out or "lambda_ca" not in out:
        print("insufficient ignited runs for a verdict"); json.dump(res, open(OUT, "w"), indent=1); return
    dn = out["D_norm"]["plateau_mean"]
    lo, hi = PREDICT["D_norm_if_one_over_N"]
    if lo <= dn <= hi:
        v = "1/N CONFIRMED"
    elif dn >= PREDICT["D_norm_if_intensive_at_least"]:
        v = "INTENSIVE -- 1/N falsified"
    else:
        v = "NEITHER prediction -- reported as such, not rounded to one"
    lam = out["lambda_ca"]["plateau_mean"]
    llo, lhi = PREDICT["lambda_ca_expected"]
    lv = ("lambda_ca INVARIANT across a 4x size range" if llo <= lam <= lhi else
          "lambda_ca OUTSIDE the predicted interval -- size-robustness DOWNGRADED to 48->96")
    print(f"\n  D_norm  = {dn:.4f}  -> {v}")
    print(f"  lambda_ca = {lam:.4f}  -> {lv}")
    # three-size ratios, the same test finite_size.json passes for order vs susceptibility
    p = PREDICT["prior_levels"]
    print(f"\n  three-size levels  lambda: {p['N48_lambda']:.4f} / {p['N96_lambda']:.4f} / {lam:.4f}")
    print(f"                     D_norm: {p['N48_D_norm']:.4f} / {p['N96_D_norm']:.4f} / {dn:.4f}"
          f"   ratios {p['N48_D_norm']/p['N96_D_norm']:.3f}, {p['N96_D_norm']/max(dn,1e-9):.3f}"
          f"   (1/N would be 2.000, 2.000)")
    res["analysis"] = out
    res["verdict"] = dict(D_norm=v, lambda_ca=lv)
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Third lattice size for the developmental transition. Predictions were "
                    "written into _preregistration before the run. lambda_ca is expected to "
                    "be intensive (a cone-growth RATE fitted before saturation) and D_norm "
                    "to fall as 1/N (localised numerator over a delocalised floor).")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
