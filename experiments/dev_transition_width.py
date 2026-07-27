"""Does the transition's timing move with WIDTH at fixed depth and fixed learning rate? (#83)

THE CONFOUND THIS BREAKS. C20 found the transition's timing moves later with model size across
70m/160m/410m/1b -- but learning rate falls as size rises in that suite (1.0e-3, 6.0e-4, 3.0e-4,
3.0e-4) and the crossing order tracks the LR order EXACTLY, tie included. The paper therefore
reports the shift as an ordering and not as a size effect, which is honest but weak.

Architecture makes the confound worse rather than better. Pythia-410m is 24 layers x 1024 while
Pythia-1b is 16 layers x 2048, so 1b is SHALLOWER than 410m. If depth drove the timing, 410m
should pair with 1.4b (also 24 layers), not with 1b. The only thing 410m and 1b share is
LR = 3.0e-4.

The three smallest models break it cleanly. VERIFIED against the EleutherAI model card before
writing this, not assumed -- the issue's own instruction, since 14m and 31m were added after the
initial Pythia release and are absent from Table 1 of Biderman et al.:

    model   layers  d_model  heads  batch   learning rate
    14M       6       128      4     2M       1.0e-3
    31M       6       256      8     2M       1.0e-3
    70M       6       512      8     2M       1.0e-3
    160M     12       768     12     2M       6.0e-4     <- depth AND LR both change here

Fixed depth, fixed learning rate, fixed batch, 4x width, 16x non-embedding parameters. And the
card states: "All models were trained on the exact same data, in the exact same order." So a
timing difference across this trio cannot be data order, cannot be LR, and cannot be depth.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: is the crossing bracket the SAME across 14m/31m/70m?
      - identical      -> no width effect at fixed LR. C20's timing shift is then attributable
                          to learning rate, and the paper's "ordering, not a size effect" hedge
                          is vindicated rather than merely cautious.
      - moves with width -> a size effect MEASURED AT FIXED LR, which is the number C20 cannot
                          supply. This is the publishable positive.
      - no crossing on this grid for some model -> reported as "not located", not as absent.
  * Secondary: per model, post- vs pre-transition lambda_ca (run-level Mann-Whitney), BH-FDR
    over the family of tests reported here.
  * lambda statistics exclude unignited runs (F42) with n stated; rank tests keep every run.
  * A NULL IS THE LIKELY OUTCOME AND IS PUBLISHABLE. Related work reads emergence trends
    straight across the LR discontinuity without flagging it, so "width alone does not move the
    timing" is a result, not a failed experiment.

BUILT-IN REPLICATION CHECK. 70m x these six checkpoints x these eight seeds is already measured
in dev_transition_scale.json under an identical protocol. This script re-runs those 48 cells
rather than importing them -- costing ~35 minutes -- and asserts they reproduce. That converts a
redundancy into a determinism check across independent invocations, and it keeps this file
self-contained so no cross-file pseudoreplication question can arise (the defect found in
loss_baseline.py, where 32 rows were the same run in two files).

Protocol is IDENTICAL to F39/C20 by construction: `measure` is imported from
dev_transition_phase3, not reimplemented.

Incremental save + resume (safe to kill). Writes results/dev_transition_width.json.
Usage:  caffeinate -i .venv/bin/python experiments/dev_transition_width.py
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
from lyapunov import is_unignited

# (hf name, tag, layers, d_model) -- layers/d_model recorded so the results file states the
# design it claims to hold fixed, rather than leaving it to a docstring.
MODELS = [("EleutherAI/pythia-14m", 14, 6, 128),
          ("EleutherAI/pythia-31m", 31, 6, 256),
          ("EleutherAI/pythia-70m", 70, 6, 512)]
STEPS = ["step128", "step256", "step512", "step1000", "step2000", "step4000"]
PRE = {"step128", "step256", "step512"}
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 48, 16
LR = "1.0e-3"          # identical for all three, per the EleutherAI model card
OUT = str(_ROOT / "results" / "dev_transition_width.json")
SCALE = str(_ROOT / "results" / "dev_transition_scale.json")


def _step_num(k):
    return int(str(k).replace("step", ""))


def crossing_interval(step_means):
    steps = sorted(step_means, key=_step_num)
    for a, b in zip(steps, steps[1:]):
        if step_means[a] < 0 <= step_means[b]:
            return (a, b)
    return None


def _ignited(r):
    return not (is_unignited(mean_damage=r["mean_damage"]) if "mean_damage" in r
                else is_unignited(D_norm=r["D_norm"]))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=[m for m, _, _, _ in MODELS], steps=STEPS, seeds=SEEDS, N=N, B=B,
        learning_rate_all_three=LR, layers_all_three=6,
        d_model=[d for _, _, _, d in MODELS],
        primary="is the crossing bracket identical across the three widths?",
        null_is_publishable=True,
        source_of_design="EleutherAI model card; verified before running")
    runs = res["runs"]
    todo = [(m, tag, st, sd) for (m, tag, _, _) in MODELS for st in STEPS for sd in SEEDS]
    print(f"WIDTH scan at fixed depth (6) and fixed LR ({LR}): {len(todo)} runs "
          f"({len(MODELS)} widths x {len(STEPS)} ckpts x {len(SEEDS)} seeds, N={N})", flush=True)
    print("Breaks C20's LR/size confound: nothing but width varies here.", flush=True)
    for k, (model, tag, st, sd) in enumerate(todo, 1):
        key = f"m{tag}_{st}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        try:
            lam, dn, md, ig = measure(st, N, B, sd, base=model)
        except Exception as e:
            print(f"[{k}/{len(todo)}] {key}: FAILED ({type(e).__name__}: {e})", flush=True)
            runs[key] = dict(model=model, size_m=tag, step=_step_num(st), seed=sd,
                             failed=f"{type(e).__name__}: {e}")
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        runs[key] = dict(model=model, size_m=tag, step=_step_num(st), seed=sd,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} "
              f"({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def replication_check(runs):
    """70m here must reproduce 70m in dev_transition_scale.json, cell for cell.

    Same model, checkpoints, seeds, N, B and code path, so the values are deterministic and any
    disagreement means the protocol is not what one of the two files claims.
    """
    if not os.path.exists(SCALE):
        return {"status": "scale file absent"}
    other = {(v["step"], v["seed"]): v["lambda_ca"]
             for v in json.load(open(SCALE))["runs"].values()
             if isinstance(v, dict) and v.get("size_m") == 70 and "lambda_ca" in v}
    mine = {(v["step"], v["seed"]): v["lambda_ca"]
            for v in runs.values() if v.get("size_m") == 70 and "lambda_ca" in v}
    shared = sorted(set(mine) & set(other))
    bad = [(k, mine[k], other[k]) for k in shared if mine[k] != other[k]]
    print(f"\n=== replication check: 70m vs dev_transition_scale.json ===")
    print(f"  {len(shared)} shared cells, {len(bad)} disagree")
    for k, a, b in bad[:5]:
        print(f"    step{k[0]} seed{k[1]}: here {a:+.5f} vs scale {b:+.5f}")
    return dict(shared_cells=len(shared), disagreeing=len(bad),
                reproduces=bool(shared and not bad))


def analyse(res):
    runs = [v for v in res["runs"].values() if "lambda_ca" in v]
    if not runs:
        print("no completed runs yet"); return
    sizes = sorted({v["size_m"] for v in runs})
    n_drop = len(runs) - sum(1 for r in runs if _ignited(r))
    if n_drop:
        print(f"\nF42: {n_drop} of {len(runs)} runs never ignited; excluded from lambda means, "
              f"kept in the rank tests")

    print(f"\n=== mean lambda_ca by (width, step) --- depth and LR held fixed ===")
    print("  d_model " + "".join(f"{s.replace('step',''):>10}" for s in STEPS))
    per_model, tests = {}, []
    dmodel = {t: d for _, t, _, d in MODELS}
    for sz in sizes:
        means, row = {}, f"  {dmodel[sz]:>7}  "
        for st in STEPS:
            v = np.array([r["lambda_ca"] for r in runs
                          if r["size_m"] == sz and r["step"] == _step_num(st) and _ignited(r)])
            if len(v):
                means[st] = float(v.mean()); row += f"{v.mean():>+10.4f}"
            else:
                row += f"{'--':>10}"
        print(row)
        ci = crossing_interval(means) if len(means) == len(STEPS) else None
        pre = [r["lambda_ca"] for r in runs if r["size_m"] == sz and f"step{r['step']}" in PRE]
        post = [r["lambda_ca"] for r in runs if r["size_m"] == sz and f"step{r['step']}" not in PRE]
        pre_i = [r["lambda_ca"] for r in runs
                 if r["size_m"] == sz and f"step{r['step']}" in PRE and _ignited(r)]
        post_i = [r["lambda_ca"] for r in runs
                  if r["size_m"] == sz and f"step{r['step']}" not in PRE and _ignited(r)]
        e = dict(d_model=dmodel[sz], layers=6, learning_rate=LR,
                 step_means={k: round(v, 4) for k, v in means.items()},
                 crossing_interval=list(ci) if ci else None,
                 n_pre=len(pre), n_post=len(post),
                 n_pre_ignited=len(pre_i), n_post_ignited=len(post_i),
                 lambda_means_exclude_unignited=True)
        if len(pre) >= 3 and len(post) >= 3:
            u = stats.mannwhitneyu(post, pre, alternative="two-sided")
            e.update(pre_mean=round(float(np.mean(pre_i)), 4),
                     post_mean=round(float(np.mean(post_i)), 4))
            tests.append((f"m{sz}_lambda_post_vs_pre", float(u.pvalue)))
        per_model[f"{sz}m"] = e

    print(f"\n=== PRIMARY: crossing bracket per width (depth and LR fixed) ===")
    intervals = []
    for sz in sizes:
        ci = per_model[f"{sz}m"]["crossing_interval"]
        print(f"  {dmodel[sz]:>5} wide ({sz}m): "
              f"{' -> '.join(ci) if ci else 'no crossing located on this grid'}")
        if ci:
            intervals.append(tuple(ci))
    located = len(intervals)
    distinct = len(set(intervals))
    if located < 2:
        verdict = (f"NOT DECIDABLE: a crossing was located for only {located} of {len(sizes)} "
                   f"widths on this grid, so the brackets cannot be compared.")
    elif distinct == 1:
        verdict = (f"NO WIDTH EFFECT AT FIXED LR: all {located} located brackets are identical "
                   f"({intervals[0][0]}->{intervals[0][1]}) across a {max(dmodel.values())//min(dmodel.values())}x "
                   f"width range at fixed depth and fixed learning rate. C20's timing shift is "
                   f"therefore not attributable to width, which leaves learning rate as the "
                   f"live explanation.")
    else:
        verdict = (f"WIDTH EFFECT AT FIXED LR: {distinct} distinct brackets across the three "
                   f"widths with depth, learning rate, batch and data order all held fixed. "
                   f"This is a size effect C20 could not measure.")
    print(f"\n  -> {verdict}")

    if tests:
        names, praw = zip(*tests)
        padj = bh_fdr(praw)
        print(f"\n=== per-width post-vs-pre, BH-FDR over this family ===")
        for n_, pr, pa in zip(names, praw, padj):
            print(f"  {n_:>26} p_raw={pr:.5f}  p_BH={pa:.5f}  "
                  f"{'SURVIVES' if pa < 0.05 else 'n.s.'}")
            per_model[n_.split('_')[0][1:] + "m"]["p_bh"] = float(pa)

    res["per_model"] = per_model
    res["primary_verdict"] = verdict
    res["replication_check_70m"] = replication_check(res["runs"])
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Width scan at FIXED depth (6 layers), FIXED learning rate (1.0e-3) and FIXED batch "
        "(2M tokens), across d_model 128/256/512 -- a 4x width and 16x non-embedding-parameter "
        "range. Verified against the EleutherAI model card before running, because 14m and 31m "
        "were added after the initial Pythia release. The card also states all models saw the "
        "exact same data in the exact same order, so a timing difference here cannot be data "
        "order, learning rate, or depth. This is the confound C20 could not break: there, LR "
        "falls as size rises and the crossing order tracks the LR order exactly, tie included. "
        "A null is publishable and was pre-registered as such. lambda statistics exclude "
        "unignited runs (F42) with n stated; rank tests keep every run. The 70m cells duplicate "
        "dev_transition_scale.json by design and are re-run rather than imported, turning the "
        "redundancy into a determinism check across independent invocations.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
