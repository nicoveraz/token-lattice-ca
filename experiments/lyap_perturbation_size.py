"""Is lambda_ca a Lyapunov exponent, or a property of the perturbation we happened to use? (#81)

WHAT #81 ASKED FOR, AND WHY IT NEEDED RESTATING. The issue asked for the transverse Lyapunov
exponent Lambda, whose sign separates directed percolation (Lambda < 0) from the
multiplicative-noise / bounded-KPZ class (Lambda = 0) in the synchronization framework of Munoz
& Pastor-Satorras (cond-mat/0301059). It also flagged the obstacle: the token alphabet is
DISCRETE, so there is no infinitesimal separation and therefore no tangent space in which a
transverse exponent is literally defined. The issue said: state how you take the limit, because
the DP-vs-MN verdict rests entirely on it.

Taking that seriously changes the experiment. Two facts about this codebase decide it:

  1. lambda_ca is ALREADY the annealed damage growth rate. `lyap_from_cone` fits "the slope of
     log(expected damaged sites)", and the cone is averaged over the batch INCLUDING lattices
     whose damage died. So the ensemble-mean quantity Lambda would be defined on is the quantity
     lambda_ca already measures. Measuring "Lambda" as a separate number would re-measure
     lambda_ca and report it under a second name.

  2. The limit that is actually missing is not infinitesimal STATE separation -- unreachable on a
     discrete alphabet -- but LINEAR RESPONSE: independence of the perturbation's size. A
     Lyapunov exponent must not depend on how hard you kicked the system. Our headline uses a
     3-site block, which is O(1), and nothing in the project has ever checked that lambda is
     invariant to that choice.

So the well-posed version of #81 is: **does lambda depend on the size of the initial
perturbation?** If it does not, linear response holds, lambda_ca is a bona fide exponent, and its
sign carries the DP-vs-MN reading the issue wanted. If it does, lambda_ca is a property of the
(model, coupling, PERTURBATION) triple, "Lyapunov exponent" is the wrong name for it, and the
universality-class program needs a different observable to target.

A HAZARD THAT WOULD HAVE MANUFACTURED THE ANSWER. `lyap_from_cone`'s default window is chosen by
a data-dependent branch on `sat_threshold=3.5`, documented in its own source as "grew beyond the
~3-site seed". That constant is tuned to block=3. Sweeping block with the default branch active
would let the ESTIMATOR change its fitting window with the perturbation and return a
block-dependence that is pure estimator artifact. Every headline number here therefore uses an
explicit, PRE-REGISTERED `fit_window`, which the estimator's own docstring recommends for exactly
this reason. The default-branch result is recorded alongside as a secondary, so the size of that
artifact is visible rather than assumed away.

A SECOND CONFOUND, RECORDED NOT HIDDEN. A larger seed starts closer to saturation, and
saturation flattens a log-slope mechanically. On N=48 a 5-site seed has ~10x headroom to the
ring, a 1-site seed ~48x. So a monotone decrease of lambda with block might be saturation rather
than nonlinearity. `max_damage_fraction` is reported per cell so the two readings can be told
apart, and the pre-registration below says which is which.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: is lambda independent of block over {1, 2, 3, 5}, at a FIXED fit window, at each of
    the two checkpoints? Tested by Kruskal-Wallis across blocks, run-level, per checkpoint.
      - independent            -> linear response holds; lambda_ca is an exponent; its sign is
                                  the DP-vs-MN reading and the universality program proceeds.
      - varies with block      -> lambda_ca is perturbation-size dependent. "Lyapunov exponent"
                                  must be qualified in the paper, and #82's exponent-extraction
                                  target changes.
  * Distinguishing nonlinearity from saturation: if lambda falls monotonically with block AND
    max_damage_fraction rises toward saturation, the reading is SATURATION and the follow-up is
    the same scan at N=96, where headroom doubles. Recorded as a conditional, not decided here.
  * Sign: at the plateau checkpoint, is lambda significantly > 0? That is the super-critical
    reading the paper already makes; here it is re-checked at every perturbation size.
  * lambda statistics exclude unignited runs (F42) with n stated.
  * A NULL -- lambda independent of block -- is the outcome that VALIDATES existing usage and is
    the more useful result. It is not a failed experiment.

Writes results/lyap_perturbation_size.json. Protocol constants match F39/C20 exactly except for
`block`, which is the variable under test.
Usage:  caffeinate -i .venv/bin/python experiments/lyap_perturbation_size.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from scipy import stats

from lyapunov import lyap_from_cone, is_unignited
from provenance import stamp, rel

BASE = "EleutherAI/pythia-410m"
BLOCKS = [1, 2, 3, 5]                       # the variable under test; 3 is the headline's value
STEPS = ["step256", "step143000"]           # the two ends of the developmental curve
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B, R, T = 48, 16, 2, 0.7
SETTLE, SWEEPS = 12, 22
# PRE-REGISTERED fit window, identical for every block. Chosen before running as the early,
# pre-saturation stretch the estimator's own defaults target (min_sweeps=3, max_sweeps=8)
# without letting the sat_threshold branch pick a different window per block.
FIT_WINDOW = (0, 6)
OUT = str(_ROOT / "results" / "lyap_perturbation_size.json")


def measure(revision, block, seed):
    """One run at a given perturbation size. Identical to F39's path except for `block`."""
    from ar_ca import ARRule
    from ar_probe import block_damage
    rule = ARRule(BASE, revision=revision)
    try:
        d = block_damage(rule, T, R, block=block, B=B, N=N, settle=SETTLE, sweeps=SWEEPS,
                         seed=seed, scheme="none")
        lam_fixed, dmax = lyap_from_cone(d["cone"], N, fit_window=FIT_WINDOW)
        lam_branch, _ = lyap_from_cone(d["cone"], N)          # default data-dependent branch
        return dict(lambda_fixed_window=float(lam_fixed),
                    lambda_default_branch=float(lam_branch),
                    max_damage_fraction=float(dmax),
                    mean_damage=float(d["mean_damage"]),
                    ignition_prob=float(d["ignition_prob"]))
    finally:
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        blocks=BLOCKS, steps=STEPS, seeds=SEEDS, N=N, B=B, r=R, T=T,
        fit_window=list(FIT_WINDOW),
        primary="is lambda independent of perturbation size at a fixed fit window?",
        null_validates_existing_usage=True,
        saturation_confound="if lambda falls monotonically with block AND max_damage_fraction "
                            "rises, the reading is saturation; follow up at N=96",
        estimator_hazard="lyap_from_cone's default branch keys on sat_threshold=3.5, tuned to a "
                         "3-site seed; the fixed window exists so the estimator cannot change "
                         "its window with the variable under test")
    runs = res["runs"]
    todo = [(st, bl, sd) for st in STEPS for bl in BLOCKS for sd in SEEDS]
    print(f"perturbation-size scan: {len(todo)} runs "
          f"(blocks {BLOCKS} x {len(STEPS)} checkpoints x {len(SEEDS)} seeds, N={N})", flush=True)
    print(f"fixed fit window {FIT_WINDOW} for every block; default branch recorded alongside",
          flush=True)
    for k, (st, bl, sd) in enumerate(todo, 1):
        key = f"{st}_b{bl}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        m = measure(st, bl, sd)
        runs[key] = dict(step=int(st.replace("step", "")), block=bl, seed=sd, **m,
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={m['lambda_fixed_window']:+.4f} "
              f"(branch {m['lambda_default_branch']:+.4f}) dmax={m['max_damage_fraction']:.3f} "
              f"({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    done = [v for v in runs.values() if "lambda_fixed_window" in v]
    if len(done) < len(todo):
        print(f"partial: {len(done)}/{len(todo)}")
        json.dump(res, open(OUT, "w"), indent=1); return
    analyse(res, done)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res, done):
    def ignited(v):
        return not is_unignited(mean_damage=v["mean_damage"])

    out = {}
    for st in STEPS:
        s = int(st.replace("step", ""))
        print(f"\n=== {st}: lambda vs perturbation size (fixed window {FIT_WINDOW}) ===")
        print(f"  {'block':>6} {'n_ign':>6} {'lambda':>10} {'sd':>8} "
              f"{'branch':>10} {'dmax/N':>8}")
        by_block, groups = {}, []
        for bl in BLOCKS:
            cells = [v for v in done if v["step"] == s and v["block"] == bl]
            ign = [v for v in cells if ignited(v)]
            if not ign:
                print(f"  {bl:>6} {0:>6}  all runs unignited"); continue
            lam = np.array([v["lambda_fixed_window"] for v in ign])
            br = np.array([v["lambda_default_branch"] for v in ign])
            dmx = np.array([v["max_damage_fraction"] for v in ign])
            groups.append(lam)
            by_block[str(bl)] = dict(
                n=len(cells), n_ignited=len(ign),
                lambda_mean=round(float(lam.mean()), 4), lambda_sd=round(float(lam.std(ddof=1)), 4),
                lambda_default_branch_mean=round(float(br.mean()), 4),
                max_damage_fraction_mean=round(float(dmx.mean()), 4))
            print(f"  {bl:>6} {len(ign):>6} {lam.mean():>+10.4f} {lam.std(ddof=1):>8.4f} "
                  f"{br.mean():>+10.4f} {dmx.mean():>8.3f}")

        entry = {"by_block": by_block}
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            h, p = stats.kruskal(*groups)
            entry["kruskal_H"] = round(float(h), 4)
            entry["kruskal_p"] = float(p)
            indep = p >= 0.05
            print(f"\n  Kruskal-Wallis across blocks: H={h:.3f}, p={p:.4f}  "
                  f"-> {'INDEPENDENT of perturbation size' if indep else 'DEPENDS on size'}")
            # saturation vs nonlinearity, as pre-registered
            lams = [by_block[str(b)]["lambda_mean"] for b in BLOCKS if str(b) in by_block]
            dmxs = [by_block[str(b)]["max_damage_fraction_mean"] for b in BLOCKS if str(b) in by_block]
            mono_down = all(x >= y for x, y in zip(lams, lams[1:]))
            sat_up = all(x <= y for x, y in zip(dmxs, dmxs[1:]))
            entry["lambda_monotone_decreasing_in_block"] = bool(mono_down)
            entry["headroom_shrinks_with_block"] = bool(sat_up)
            if not indep and mono_down and sat_up:
                entry["reading"] = ("SATURATION rather than nonlinearity -- lambda falls "
                                    "monotonically while headroom shrinks; follow up at N=96")
            elif not indep:
                entry["reading"] = ("NONLINEAR RESPONSE -- lambda varies with perturbation size "
                                    "in a way saturation does not explain")
            else:
                entry["reading"] = ("LINEAR RESPONSE holds at this checkpoint; lambda_ca is not "
                                    "an artifact of the 3-site seed")
            print(f"  -> {entry['reading']}")
        out[st] = entry

    plateau = out.get("step143000", {}).get("by_block", {})
    if plateau:
        signs = {b: v["lambda_mean"] > 0 for b, v in plateau.items()}
        print(f"\n=== sign at the plateau, per perturbation size ===")
        for b, pos in signs.items():
            print(f"  block {b}: lambda = {plateau[b]['lambda_mean']:+.4f} "
                  f"({'positive' if pos else 'NOT positive'})")
        out["plateau_sign_robust_to_block"] = bool(all(signs.values()))

    res["analysis"] = out
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Perturbation-size dependence of lambda_ca (#81). The issue asked for the transverse "
        "Lyapunov exponent Lambda; on a discrete alphabet there is no tangent space, and "
        "lambda_ca is ALREADY the annealed damage growth rate that Lambda would be defined on -- "
        "lyap_from_cone fits log(expected damaged sites) over a batch that includes extinguished "
        "lattices. The limit genuinely missing is linear response: independence of the "
        "perturbation's SIZE, which a Lyapunov exponent requires and which nothing here had "
        "checked. Every headline number uses a PRE-REGISTERED fixed fit window, because "
        "lyap_from_cone's default branch keys on sat_threshold=3.5, explicitly tuned to a 3-site "
        "seed, and would otherwise change its fitting window with the variable under test. The "
        "default-branch value is recorded alongside so that artifact is measurable. "
        "max_damage_fraction is reported per cell because a larger seed starts closer to "
        "saturation, which flattens a log-slope mechanically and can imitate nonlinearity.")
    res["_config"] = dict(model=BASE, blocks=BLOCKS, steps=STEPS, seeds=SEEDS,
                          N=N, B=B, r=R, T=T, settle=SETTLE, sweeps=SWEEPS,
                          fit_window=list(FIT_WINDOW))
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
