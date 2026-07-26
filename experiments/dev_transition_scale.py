"""Is the developmental transition's TIMING universal across model scale?

F39 established, at 8 seeds and two lattice sizes with BH-FDR, that Pythia-410m's token-space
Lyapunov exponent goes from seeds disagreeing about its sign to every plateau run being
positive (0/48 negative), with the cell-mean crossing between steps 256 and 512. That is one model. The obvious question the result raises -- and the paper's
stated main limitation -- is whether the *step at which it happens* is a property of the
training run (data order, optimizer schedule, tokenizer) or of the model's capacity.

WHAT THIS IS NOT. This is NOT the retracted capacity->sensitivity axis (W1). That claim was
"lambda / D_norm measured at a fixed checkpoint increases with model size", it was
pseudoreplicated at n=2 seeds, and it stays retracted. The question here is a different
object: WHEN, in training, does the sign change occur, and is that step the same across
sizes? A transition that lands at the same step for a 70m and a 1b model says something about
the training process; one that moves with size says something about capacity. Either answer
is publishable and neither resurrects the retracted claim. The post-transition LEVEL is also
recorded, but it is reported as EXPLORATORY and is explicitly not a capacity-axis claim --
see the pre-registration below.

PRE-REGISTERED BEFORE RUNNING (2026-07-26):
  * Primary: for each model, locate the crossing interval -- the adjacent checkpoint pair
    (s_i, s_{i+1}) with mean lambda_ca(s_i) < 0 <= mean lambda_ca(s_{i+1}). Claim under test:
    the crossing interval is the SAME across the four model sizes.
  * Secondary: per model, post-transition vs pre-transition lambda_ca (run-level
    Mann-Whitney), BH-FDR over the family of all such tests reported here.
  * Exploratory, NOT a capacity claim: the post-transition plateau level vs model size.
    Reported with its Spearman rho and p, and labelled exploratory whatever it shows.
  * If a model shows no crossing on this grid, that is reported as "no crossing located",
    not as an absent transition -- the grid may simply miss it.

Protocol is IDENTICAL to F39's by construction: `measure` is imported from
dev_transition_phase3.py, not reimplemented. N=48 throughout, which F39 licenses for a
lambda_ca question specifically -- lambda_ca was shown size-robust (95% retention; plateau
levels agree within +-14%, 95% CI), unlike D_norm, whose absolute scale moves with N and
which is therefore recorded here but not used for the primary.

Checkpoint grid is finer around the crossing than F39's: 128/256/512/1000/2000/4000.

Incremental save + resume (safe to kill). Writes results/dev_transition_scale.json.
Usage:  caffeinate -i .venv/bin/python experiments/dev_transition_scale.py
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
from lyapunov import is_unignited                     # F42

MODELS = [("EleutherAI/pythia-70m", 70), ("EleutherAI/pythia-160m", 160),
          ("EleutherAI/pythia-410m", 410), ("EleutherAI/pythia-1b", 1000)]
STEPS = ["step128", "step256", "step512", "step1000", "step2000", "step4000"]
PRE = {"step128", "step256", "step512"}
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 48, 16
OUT = str(_ROOT / "results" / "dev_transition_scale.json")


def _step_num(k):
    return int(str(k).replace("step", ""))


def crossing_interval(step_means):
    """First CHRONOLOGICALLY adjacent pair where the mean crosses zero upward.

    The keys are strings like "step128", "step1000". Sorting them directly is LEXICOGRAPHIC --
    it orders 1000 before 128 before 2000 before 256 -- so the "adjacent" pairs were not
    adjacent in training time and the reported crossing interval was meaningless. Sort by the
    integer step. (Same class of defect as F39 and F42: a helper whose declared behaviour and
    actual behaviour differed, with nothing asserting the difference.)
    """
    steps = sorted(step_means, key=_step_num)
    for a, b in zip(steps, steps[1:]):
        if step_means[a] < 0 <= step_means[b]:
            return (a, b)
    return None


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    runs = res["runs"]
    todo = [(m, tag, st, sd) for (m, tag) in MODELS for st in STEPS for sd in SEEDS]
    print(f"Developmental TIMING across scale: {len(todo)} runs "
          f"({len(MODELS)} models x {len(STEPS)} ckpts x {len(SEEDS)} seeds, N={N})",
          flush=True)
    print("NOT the retracted capacity axis -- the question is WHEN the sign change happens.",
          flush=True)
    for k, (model, tag, st, sd) in enumerate(todo, 1):
        key = f"m{tag}_{st}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        try:
            lam, dn, md, ig = measure(st, N, B, sd, base=model)
        except Exception as e:                     # a missing revision must not lose the run
            print(f"[{k}/{len(todo)}] {key}: FAILED ({type(e).__name__}: {e})", flush=True)
            runs[key] = dict(model=model, size_m=tag, step=int(st.replace("step", "")),
                             seed=sd, failed=f"{type(e).__name__}: {e}")
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        runs[key] = dict(model=model, size_m=tag, step=int(st.replace("step", "")), seed=sd,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} "
              f"({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", OUT)


def analyse(res):
    runs = [v for v in res["runs"].values() if "lambda_ca" in v]
    if not runs:
        print("no completed runs yet"); return
    sizes = sorted({v["size_m"] for v in runs})
    per_model, tests, plateau_by_size = {}, [], {}

    print("\n=== mean lambda_ca by (model, step) ===")
    hdr = "  size   " + "".join(f"{s.replace('step',''):>9}" for s in STEPS)
    print(hdr)
    for sz in sizes:
        means = {}
        row = f"  {sz:>4}m  "
        for st in STEPS:
            v = np.array([r["lambda_ca"] for r in runs
                          if r["size_m"] == sz and r["step"] == int(st.replace("step", ""))])
            if len(v):
                means[st] = float(v.mean())
                row += f"{v.mean():>+9.4f}"
            else:
                row += f"{'--':>9}"
        print(row)
        ci = crossing_interval(means) if len(means) == len(STEPS) else None
        pre = [r["lambda_ca"] for r in runs
               if r["size_m"] == sz and f"step{r['step']}" in PRE]
        post = [r["lambda_ca"] for r in runs
                if r["size_m"] == sz and f"step{r['step']}" not in PRE]
        entry = dict(step_means={k: round(v, 4) for k, v in means.items()},
                     crossing_interval=list(ci) if ci else None,
                     n_pre=len(pre), n_post=len(post))
        if len(pre) >= 3 and len(post) >= 3:
            u = stats.mannwhitneyu(post, pre, alternative="two-sided")
            entry.update(pre_mean=round(float(np.mean(pre)), 4),
                         post_mean=round(float(np.mean(post)), 4))
            tests.append((f"m{sz}_lambda_post_vs_pre", float(u.pvalue)))
            plateau_by_size[sz] = float(np.mean(post))
        per_model[f"{sz}m"] = entry

    print("\n=== PRIMARY: crossing interval per model ===")
    intervals = []
    for sz in sizes:
        ci = per_model[f"{sz}m"]["crossing_interval"]
        print(f"  {sz:>4}m : {' -> '.join(ci) if ci else 'no crossing located on this grid'}")
        if ci:
            intervals.append(tuple(ci))
    # Distinguish the three cases the pre-registration cares about. "No crossing located" is
    # NOT the same as "intervals differ" -- it means the transition is outside this grid for
    # that model, which is itself informative and must not be pooled into a disagreement.
    n_missing = len(sizes) - len(intervals)
    distinct = len(set(intervals))
    if n_missing == 0 and distinct == 1:
        verdict = "UNIVERSAL: every size crosses in the same interval"
    elif distinct <= 1 and n_missing:
        verdict = (f"CONSISTENT where located, but {n_missing} size(s) have no crossing on this "
                   f"grid -- their transition lies outside it")
    else:
        verdict = (f"SIZE-DEPENDENT: {distinct} distinct crossing intervals across the "
                   f"{len(intervals)} size(s) where one was located"
                   + (f"; {n_missing} size(s) had none on this grid" if n_missing else ""))
    print(f"  -> {verdict}")

    if tests:
        names, praw = zip(*tests)
        padj = bh_fdr(praw)
        res["tests"] = [dict(name=n, p_raw=round(pr, 6), p_bh=round(pa, 6),
                             significant_bh_05=bool(pa < 0.05))
                        for n, pr, pa in zip(names, praw, padj)]
        print("\n=== per-model post-vs-pre, BH-FDR across the family ===")
        for t in res["tests"]:
            print(f"  {t['name']:26s} p_raw={t['p_raw']:.5f}  p_BH={t['p_bh']:.5f}  "
                  f"{'SURVIVES' if t['significant_bh_05'] else 'n.s.'}")

    if len(plateau_by_size) >= 3:
        szs = sorted(plateau_by_size)
        rho, p = stats.spearmanr(szs, [plateau_by_size[s] for s in szs])
        # With n sizes there are only n! orderings, so the smallest attainable permutation
        # p-value is 1/n! -- 0.167 at n=3. scipy's asymptotic p (which returns 0.0 here) is
        # not valid at this n and must not be quoted.
        from math import factorial
        p_floor = 1.0 / factorial(len(szs))
        res["exploratory_plateau_vs_size"] = dict(
            by_size={str(s): round(plateau_by_size[s], 4) for s in szs},
            spearman_rho=round(float(rho), 3),
            p_exact_floor=round(p_floor, 4),
            p_scipy_INVALID_AT_THIS_N=round(float(p), 4),
            label=("EXPLORATORY. This is NOT the retracted capacity->sensitivity axis (W1); "
                   "it is the post-transition level by size, reported for completeness. It is "
                   "not corrected for multiplicity and must not be quoted as an axis. Note the "
                   "n: with this many sizes the smallest attainable permutation p is "
                   "1/n!, so a 'significant' Spearman here is arithmetically impossible."))
        print(f"\n=== exploratory (NOT a capacity claim): plateau level vs size ===")
        print("  " + "  ".join(f"{s}m={plateau_by_size[s]:+.4f}" for s in szs))
        print(f"  Spearman rho={rho:+.3f}  -- exploratory. With {len(szs)} sizes the smallest "
              f"attainable permutation p is 1/{len(szs)}! = {p_floor:.3f}, so this cannot be "
              f"significant however it comes out; scipy's {p:.3f} is an invalid asymptotic.")

    res["per_model"] = per_model
    res["primary_verdict"] = verdict
    res["_note"] = (
        "Timing of the developmental transition across Pythia sizes, N=48, 8 seeds, protocol "
        "imported verbatim from dev_transition_phase3.measure. PRIMARY is the crossing "
        "interval, not the level. The post-transition level by size is EXPLORATORY and is "
        "explicitly not the retracted capacity->sensitivity axis (W1).")
    res["_preregistration"] = dict(models=[m for m, _ in MODELS], steps=STEPS,
                                   pre=sorted(PRE), seeds=SEEDS, N=N, B=B)


if __name__ == "__main__":
    main()
