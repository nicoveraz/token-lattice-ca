"""Width scan, on a grid that actually contains the transition (#87; amends #83).

WHY THERE IS A SECOND GRID, AND WHY THAT IS NOT GRID-SHOPPING. #83 ran 14m/31m/70m over
128..4000 -- the grid C20 used -- and returned NOT DECIDABLE: a crossing bracket was located for
only one of three widths. The ignition counts say why, and they say it is a design misalignment
rather than a null:

    d_model     128    256    512   1000   2000   4000      (ignited / 8)
        128     8/8    8/8   *4/8*   8/8    8/8    8/8
        256    *0/8*   7/8    8/8    8/8    8/8    8/8
        512     8/8    8/8    8/8    8/8    8/8    8/8

14m's apparent crossing at 512->1000 sits on a cell where HALF the runs died, so its mean is over
the four that happened to ignite -- a biased sample, not a transition. And 31m at step128 has ZERO
ignition: no lambda exists there at all.

C20's grid begins at step128 because it was designed for 410m, whose learning rate is 3x smaller.
At LR = 1.0e-3 these three models have already passed the transition before that window opens.

Extending a grid after a null is what p-hacking looks like, so the distinction has to be stated:
this is not searching for significance inside a window that already contains the phenomenon. It
is moving a window that provably does not contain it -- 31m's step128 cell has no ignited runs at
all, so there is nothing there to find at any n. The #83 result stands as reported; this does not
overwrite it.

PRE-REGISTERED BEFORE RUNNING (second registration, superseding neither):
  * Grid: steps {1, 2, 4, 8, 16, 32, 64}, verified present on all three models before writing
    this. Combined with #83's 128..4000 these give 13 checkpoints spanning 1 -> 4000.
  * Primary: with depth (6 layers), learning rate (1.0e-3), batch (2M tokens) and data order all
    fixed, is the crossing bracket the SAME across d_model 128/256/512?
      - identical        -> no width effect at fixed LR. C20's timing shift is then attributable
                            to learning rate, and the paper's "ordering, not a size effect" hedge
                            is vindicated rather than merely cautious.
      - moves with width -> a size effect measured AT FIXED LR, which C20 cannot supply.
  * Ignition counts are reported per cell. A bracket whose lower cell has < 4 ignited runs is
    reported as UNRELIABLE rather than used, because that is exactly the artifact that made
    #83's 14m bracket meaningless.
  * A null remains publishable, for the reason #83 gave: related work reads emergence trends
    straight across the LR discontinuity without flagging it.

ESTIMATOR NOTE. `measure` is imported from dev_transition_phase3 unchanged, so lambda here is
comparable to #83, C20 and the paper. #81 established that the damage-range estimator gives
larger magnitudes at the plateau (1.38x at block=3) because the default branch still fits partway
into the bend -- but this experiment is about the SIGN and where it changes, and swapping
estimators mid-comparison would confound the thing being measured. The corrected estimator
belongs to #82, where magnitudes are the point.

Incremental save + resume (safe to kill). Writes results/dev_transition_width_early.json.
Usage:  caffeinate -i .venv/bin/python experiments/dev_transition_width_early.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from scipy import stats

from dev_transition_phase3 import measure, bh_fdr     # identical protocol, not a copy
from provenance import stamp, rel
from lyapunov import run_ignited

MODELS = [("EleutherAI/pythia-14m", 14, 128),
          ("EleutherAI/pythia-31m", 31, 256),
          ("EleutherAI/pythia-70m", 70, 512)]
STEPS = ["step1", "step2", "step4", "step8", "step16", "step32", "step64"]
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 48, 16
LR = "1.0e-3"
MIN_IGNITED = 4        # a bracket resting on fewer ignited runs than this is not usable
OUT = str(_ROOT / "results" / "dev_transition_width_early.json")
LATE = str(_ROOT / "results" / "dev_transition_width.json")


def _n(k):
    return int(str(k).replace("step", ""))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        steps=STEPS, seeds=SEEDS, N=N, B=B, learning_rate_all_three=LR, layers_all_three=6,
        d_model=[d for _, _, d in MODELS], min_ignited_for_a_usable_bracket=MIN_IGNITED,
        amends="#83 -- its grid began at step128, which is past the transition at this LR",
        why_not_grid_shopping="31m had ZERO ignited runs at step128; the old window provably "
                              "does not contain the phenomenon, so this moves it rather than "
                              "searching inside it",
        estimator="dev_transition_phase3.measure unchanged, for comparability with #83/C20")
    runs = res["runs"]
    todo = [(m, tag, st, sd) for (m, tag, _) in MODELS for st in STEPS for sd in SEEDS]
    print(f"EARLY width grid at fixed depth (6) and fixed LR ({LR}): {len(todo)} runs "
          f"(3 widths x {len(STEPS)} ckpts x {len(SEEDS)} seeds, N={N})", flush=True)
    print("Amends #83, whose grid started past the transition for these models.", flush=True)
    for k, (model, tag, st, sd) in enumerate(todo, 1):
        key = f"m{tag}_{st}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        try:
            lam, dn, md, ig = measure(st, N, B, sd, base=model)
        except Exception as e:
            print(f"[{k}/{len(todo)}] {key}: FAILED ({type(e).__name__}: {e})", flush=True)
            runs[key] = dict(model=model, size_m=tag, step=_n(st), seed=sd,
                             failed=f"{type(e).__name__}: {e}")
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        runs[key] = dict(model=model, size_m=tag, step=_n(st), seed=sd,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} ign={ig:.2f} "
              f"({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _cells(runs, sz):
    """(step -> (mean lambda over ignited, n_ignited, n_total)) merging the early and late grids.

    Merged on (model, step, seed) with an agreement assert, because the two files overlap at no
    step today but nothing stops a future grid from overlapping -- and silently double-counting
    is the defect that inflated every n in loss_baseline.py until it was caught.
    """
    seen, out = {}, {}
    src = list(runs.values())
    if os.path.exists(LATE):
        src += list(json.load(open(LATE))["runs"].values())
    for v in src:
        if not (isinstance(v, dict) and v.get("size_m") == sz and "lambda_ca" in v):
            continue
        k = (v["step"], v["seed"])
        if k in seen:
            assert seen[k] == v["lambda_ca"], f"grids disagree about size {sz} at {k}"
            continue
        seen[k] = v["lambda_ca"]
        out.setdefault(v["step"], []).append(v)
    return {s: (float(np.mean([r["lambda_ca"] for r in rs if run_ignited(r)]))
                if any(run_ignited(r) for r in rs) else None,
                sum(1 for r in rs if run_ignited(r)), len(rs))
            for s, rs in out.items()}


def analyse(res):
    dm = {t: d for _, t, d in MODELS}
    per_model, tests = {}, []
    all_steps = sorted({s for sz in dm for s in _cells(res["runs"], sz)})
    print(f"\n=== mean lambda_ca over the MERGED grid (early + #83), ignited runs only ===")
    print("  d_model " + "".join(f"{s:>9}" for s in all_steps))
    for sz, d in dm.items():
        c = _cells(res["runs"], sz)
        row = f"  {d:>7}  "
        for s in all_steps:
            m = c.get(s, (None, 0, 0))[0]
            row += f"{m:>+9.4f}" if m is not None else f"{'--':>9}"
        print(row)
        print("          " + "".join(f"{c.get(s,(None,0,0))[1]:>4}/{c.get(s,(None,0,0))[2]:<5}"
                                     for s in all_steps))

        ordered = [s for s in sorted(c) if c[s][0] is not None]
        cross, unreliable = None, False
        for a, b in zip(ordered, ordered[1:]):
            if c[a][0] < 0 <= c[b][0]:
                cross = (a, b)
                unreliable = c[a][1] < MIN_IGNITED or c[b][1] < MIN_IGNITED
                break
        per_model[f"{sz}m"] = dict(
            d_model=d, layers=6, learning_rate=LR,
            step_means={str(s): (None if c[s][0] is None else round(c[s][0], 4)) for s in sorted(c)},
            ignited={str(s): c[s][1] for s in sorted(c)},
            crossing_interval=list(cross) if cross else None,
            crossing_unreliable_low_ignition=bool(unreliable))

    print(f"\n=== PRIMARY: crossing bracket per width, depth and LR fixed ===")
    usable = []
    for sz, d in dm.items():
        e = per_model[f"{sz}m"]
        ci = e["crossing_interval"]
        tag = "" if not e["crossing_unreliable_low_ignition"] else \
              f"  <- UNRELIABLE (<{MIN_IGNITED} ignited in a bracket cell)"
        print(f"  {d:>5} wide ({sz}m): "
              f"{'step%d -> step%d' % tuple(ci) if ci else 'no crossing located'}{tag}")
        if ci and not e["crossing_unreliable_low_ignition"]:
            usable.append(tuple(ci))

    if len(usable) < 2:
        verdict = (f"NOT DECIDABLE: only {len(usable)} reliable bracket(s) across three widths.")
    elif len(set(usable)) == 1:
        verdict = (f"NO WIDTH EFFECT AT FIXED LR: all {len(usable)} reliable brackets identical "
                   f"(step{usable[0][0]} -> step{usable[0][1]}) across a 4x width range with "
                   f"depth, learning rate, batch and data order held fixed. C20's timing shift "
                   f"is therefore not attributable to width.")
    else:
        verdict = (f"WIDTH EFFECT AT FIXED LR: {len(set(usable))} distinct brackets with "
                   f"everything but width held fixed -- a size effect C20 could not measure.")
    print(f"\n  -> {verdict}")

    res["per_model"] = per_model
    res["primary_verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Early-checkpoint width grid (#87), amending #83 whose grid began at step128 -- past the "
        "transition for LR=1.0e-3 models, as its own ignition counts showed (31m had ZERO ignited "
        "runs at step128). Steps 1..64 verified present on all three models before running. "
        "Analysed on the MERGED early+late grid so the curve spans step1..step4000, deduplicated "
        "on (model, step, seed) with an agreement assert. A bracket whose cells carry fewer than "
        f"{MIN_IGNITED} ignited runs is flagged UNRELIABLE rather than used, because that is "
        "exactly the artifact that made #83's 14m bracket meaningless. lambda comes from "
        "dev_transition_phase3.measure unchanged, for comparability with #83/C20; #81's corrected "
        "damage-range estimator gives larger plateau magnitudes but belongs to #82, where "
        "magnitudes rather than sign changes are the point.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
