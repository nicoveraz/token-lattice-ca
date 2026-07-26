"""Separate lattice size from batch size for the ignition question (issue #39, F44).

F44 found that the unignited fraction rises with N -- 0/16, 1/16, 5/16 at N=48/96/192 at matched
checkpoints, Fisher p=0.022 -- and then showed the rise is fully explained by BATCH size, because
the design halves B as N doubles (16/8/4) and a run counts as unignited only if ALL B lattices
die. A single constant per-lattice death probability with no N dependence fit all three sizes
(d=0.690, chi-square p=0.912).

But that is a MODEL fit, not a manipulation. Every N in the existing data comes with its own B, so
nothing in it can distinguish the two. This runs the missing cell of the 2x2 -- **N=48 at B=4** --
which breaks the confound directly:

           B=16        B=4
  N=48     0/16       <- THIS RUN
  N=192     --        5/16

PRE-REGISTERED PREDICTIONS (written before running):
  * If the effect is BATCH SIZE (F44's conclusion): N=48 at B=4 should look like N=192 at B=4,
    i.e. roughly 5/16 unignited, and NOTHING like N=48 at B=16 (0/16).
  * If the effect is LATTICE SIZE: N=48 at B=4 should look like N=48 at B=16, i.e. near 0/16.
  * These are separated by about 5 unignited runs out of 16 -- a Fisher exact on the two N=48
    cells is decisive at this n if the batch-size story is right.
  * A result strictly between (2-3 of 16) means BOTH contribute and neither pure story holds;
    that is reported as such, not rounded to whichever is closer.

Also recorded, and not available in the older data: `ignition_prob`, the per-lattice ignited
fraction that `block_damage` already computes. It gives the per-lattice death probability d
DIRECTLY, rather than inferring it from a d^B fit, so the F44 model can be checked against a
measurement instead of only against its own residuals.

Cheap: B=4 at N=48 is a quarter of the Phase 3 forward passes, so ~16 runs in minutes.
Protocol imported from dev_transition_phase3. Writes results/ignition_nb.json.
Usage:  caffeinate -i .venv/bin/python experiments/ignition_nb.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from scipy import stats

from dev_transition_phase3 import measure
from lyapunov import is_unignited
from provenance import stamp

STEPS = ["step256", "step512"]          # the checkpoints F44 compared at
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 48, 4                            # the missing cell: small lattice, small batch
OUT = str(_ROOT / "results" / "ignition_nb.json")

PREDICT = dict(
    if_batch_size_effect="~5/16 unignited, like N=192 B=4",
    if_lattice_size_effect="~0/16 unignited, like N=48 B=16",
    if_both="2-3/16, reported as 'both contribute', not rounded to the nearer story",
    reference_cells={"N48_B16": "0/16", "N96_B8": "1/16", "N192_B4": "5/16"})


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(N=N, B=B, steps=STEPS, seeds=SEEDS, predictions=PREDICT)
    runs = res["runs"]
    todo = [(st, sd) for st in STEPS for sd in SEEDS]
    print(f"N={N} B={B}: the missing cell of the N-vs-B 2x2 ({len(todo)} runs)")
    print(f"PRE-REGISTERED: batch-size story -> ~5/16 unignited; lattice-size story -> ~0/16",
          flush=True)
    for k, (st, sd) in enumerate(todo, 1):
        key = f"N{N}B{B}_{st}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        lam, dn, md, ig = measure(st, N, B, sd)
        runs[key] = dict(N=N, B=B, step=int(st.replace("step", "")), seed=sd,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} "
              f"ignition_prob={ig:.3f} ({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    done = [v for v in runs.values() if "lambda_ca" in v]
    if len(done) < len(todo):
        print(f"partial: {len(done)}/{len(todo)}")
        json.dump(res, open(OUT, "w"), indent=1); return

    dead = [v for v in done if is_unignited(mean_damage=v["mean_damage"])]
    k_new, n_new = len(dead), len(done)
    print(f"\n=== RESULT: N=48 B=4 -> {k_new}/{n_new} unignited ===")
    print(f"  reference cells: N=48 B=16 -> 0/16   N=96 B=8 -> 1/16   N=192 B=4 -> 5/16")

    # decisive comparison: the two N=48 cells differ ONLY in B
    tbl = [[k_new, n_new - k_new], [0, 16]]          # new cell vs Phase 3's N=48 B=16
    odds_b, p_b = stats.fisher_exact(tbl)
    # and the two B=4 cells differ ONLY in N
    tbl2 = [[k_new, n_new - k_new], [5, 11]]         # new cell vs N=192 B=4
    odds_n, p_n = stats.fisher_exact(tbl2)
    print(f"\n  vs N=48 B=16 (B changes, N fixed):  Fisher p={p_b:.4f}")
    print(f"  vs N=192 B=4 (N changes, B fixed):  Fisher p={p_n:.4f}")

    if p_b < 0.05 and p_n >= 0.05:
        verdict = ("BATCH SIZE: the new cell differs from same-N/different-B and matches "
                   "same-B/different-N. F44's conclusion is confirmed by manipulation.")
    elif p_b >= 0.05 and p_n < 0.05:
        verdict = ("LATTICE SIZE: the new cell matches same-N/different-B and differs from "
                   "same-B/different-N. F44's batch-size explanation is REFUTED.")
    elif p_b < 0.05 and p_n < 0.05:
        verdict = "BOTH contribute -- the new cell differs from both reference cells."
    else:
        verdict = ("NOT RESOLVED at this n -- the new cell is distinguishable from neither "
                   "reference cell.")
    print(f"\n  -> {verdict}")

    # per-lattice death probability, MEASURED rather than fit
    ip = np.array([v["ignition_prob"] for v in done])
    d_meas = float(1 - ip.mean())
    print(f"\n  measured per-lattice ignition prob = {ip.mean():.4f}  -> death prob d = {d_meas:.4f}")
    print(f"  F44 fit d from the d^B model (no N dependence) = 0.6897")
    print(f"  predicted unignited at B={B}: d^B = {d_meas**B:.4f} -> {n_new*d_meas**B:.2f} of {n_new}"
          f"   (observed {k_new})")

    res["analysis"] = dict(
        unignited=k_new, n=n_new, frac_unignited=round(k_new / n_new, 4),
        fisher_vs_sameN_diffB=dict(p=round(float(p_b), 5), cell="N=48 B=16 -> 0/16"),
        fisher_vs_sameB_diffN=dict(p=round(float(p_n), 5), cell="N=192 B=4 -> 5/16"),
        measured_ignition_prob=round(float(ip.mean()), 4),
        measured_death_prob=round(d_meas, 4),
        f44_fitted_death_prob=0.6897,
        predicted_unignited_from_measured_d=round(float(n_new * d_meas ** B), 2),
        verdict=verdict)
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Missing cell of the N-vs-B 2x2 for the ignition question (issue #39). Every N in the "
        "existing data came with its own B, so F44 could only FIT a batch-size model, not "
        "manipulate it. This holds N at 48 and drops B to 4. Also records ignition_prob, giving "
        "the per-lattice death probability directly rather than inferring it from a d^B fit.")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
