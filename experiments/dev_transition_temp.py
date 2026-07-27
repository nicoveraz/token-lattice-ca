"""Is the developmental transition a property of the model, or of T=0.7? (issue #17)

The headline (F39/F45/F46) is measured at a single sampling temperature, T=0.7. Temperature is
known to be a strong axis in this project -- F12 showed the "temperature phase transition" is a
finite-size crossover, and the cross-level negative found the temperature axis confounded by a
common cause -- so a transition observed at one T is not automatically a property of the model.

Issue #17 asks the sharp version: as T falls the rule becomes deterministic (greedy), so if
token-space criticality is a SAMPLING phenomenon it should weaken or vanish at low T; if it is
structural it should persist. This tests the developmental claim specifically rather than the
static criticality, because the developmental claim is what the paper rests on.

DESIGN. The two extreme checkpoints only -- step256 (pre) and step143000 (plateau) -- at
T in {0.3, 0.5, 0.9, 1.1}, 8 seeds, N=48, B=16. T=0.7 is already measured (Phase 3), so this
gives FIVE temperatures spanning near-greedy to high-entropy at both ends of training.

WIDENED FOR ISSUE #73. The first pass ran T in {0.3, 1.1} only. Three points on the temperature
axis is thin for the paper's most attackable scope limit: it supports "the effect appears at one
temperature and pegs at the two we tried outside it", and the ceiling/floor mechanism offered to
explain the pegs is then a story about three points. T=0.5 and T=0.9 turn that into "the effect
appears across a RANGE and pegs outside it", which is a categorically different claim for the
same cost. The BH-FDR correction is recomputed over the FULL five-temperature family rather than
appended to the old one -- a correction that grows by accretion is not a correction.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: does the pre -> plateau sign change in lambda_ca persist at each T?
    Tested per temperature with a run-level Mann-Whitney, BH-FDR over the family of temperatures
    reported here.
  * If it survives at 0.5, 0.7 and 0.9 but not at 0.3 and 1.1, the ceiling/floor reading is
    confirmed by a range rather than asserted from two endpoints, and the paper says
    "at intermediate temperature".
  * If it survives ONLY at 0.7, the scope limit is real and narrow: the paper must say "at
    T=0.7" and not "at intermediate temperature". That is a successful outcome of this
    experiment, not a failure, and the sentence gets rewritten.
  * Ignition fraction is recorded at every temperature, because it is what makes the
    ceiling/floor reading testable rather than post-hoc.
  * If it persists at both, the headline is NOT a T=0.7 artifact and the paper should say so.
  * If it vanishes at low T, the claim is a sampling phenomenon and must be restated as such --
    that is a successful outcome of this experiment, not a failure.
  * If it vanishes at HIGH T only, the likely reading is that high-temperature dynamics are
    already super-critical everywhere, i.e. a ceiling rather than a refutation. Reported as such.
  * lambda statistics exclude unignited runs per F42, with n stated; the rank test keeps all runs.

Writes results/dev_transition_temp.json. Protocol imported from dev_transition_phase3.
Usage:  caffeinate -i .venv/bin/python experiments/dev_transition_temp.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from scipy import stats

from dev_transition_phase3 import measure, bh_fdr
from lyapunov import is_unignited, run_ignited
from provenance import stamp, rel

TEMPS = [0.3, 0.5, 0.9, 1.1]             # T=0.7 comes from Phase 3
STEPS = ["step256", "step143000"]        # the two ends of the developmental curve
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 48, 16
OUT = str(_ROOT / "results" / "dev_transition_temp.json")
PHASE3 = str(_ROOT / "results" / "dev_transition_phase3.json")


def unignited(v):
    return not run_ignited(v)


def phase3_reference():
    """The T=0.7 cells at the same two checkpoints and N, for comparison."""
    d = json.load(open(PHASE3))
    out = {}
    for st in (256, 143000):
        rows = [v for v in d["runs"].values()
                if isinstance(v, dict) and "lambda_ca" in v and v["N"] == N and v["step"] == st]
        ig = [v for v in rows if not unignited(v)]
        out[st] = dict(n=len(rows), n_ignited=len(ig),
                       lambda_mean=round(float(np.mean([v["lambda_ca"] for v in ig])), 4))
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(temps=TEMPS, steps=STEPS, seeds=SEEDS, N=N, B=B,
                                   reference_T=0.7, source_of_reference="dev_transition_phase3")
    runs = res["runs"]
    todo = [(t, st, sd) for t in TEMPS for st in STEPS for sd in SEEDS]
    print(f"developmental transition vs temperature: {len(todo)} runs "
          f"(T in {TEMPS} x {len(STEPS)} checkpoints x {len(SEEDS)} seeds); T=0.7 from Phase 3",
          flush=True)
    for k, (t, st, sd) in enumerate(todo, 1):
        key = f"T{t}_{st}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        lam, dn, md, ig = measure(st, N, B, sd, temp=t)
        runs[key] = dict(T=t, step=int(st.replace("step", "")), seed=sd, N=N, B=B,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} "
              f"ign={ig:.2f} ({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    done = [v for v in runs.values() if "lambda_ca" in v]
    if len(done) < len(todo):
        print(f"partial: {len(done)}/{len(todo)}")
        json.dump(res, open(OUT, "w"), indent=1); return

    ref = phase3_reference()
    print(f"\n=== lambda_ca: pre (step256) -> plateau (step143000), by temperature ===")
    print(f"  {'T':>5} {'pre':>10} {'n':>4} {'plateau':>10} {'n':>4} {'ignited':>9} {'p_raw':>10}")
    print(f"  {0.7:>5} {ref[256]['lambda_mean']:>+10.4f} {ref[256]['n_ignited']:>4} "
          f"{ref[143000]['lambda_mean']:>+10.4f} {ref[143000]['n_ignited']:>4} "
          f"{'(Phase 3)':>9} {'--':>10}")
    tests, summary = [], {"0.7": dict(source="dev_transition_phase3", **{
        "pre_mean": ref[256]["lambda_mean"], "plateau_mean": ref[143000]["lambda_mean"]})}
    for t in TEMPS:
        pre_all = [v for v in done if v["T"] == t and v["step"] == 256]
        post_all = [v for v in done if v["T"] == t and v["step"] == 143000]
        pre = [v["lambda_ca"] for v in pre_all if not unignited(v)]
        post = [v["lambda_ca"] for v in post_all if not unignited(v)]
        n_ig = len(pre) + len(post)
        u = stats.mannwhitneyu([v["lambda_ca"] for v in post_all],
                               [v["lambda_ca"] for v in pre_all], alternative="two-sided")
        tests.append((f"T{t}", float(u.pvalue)))
        summary[str(t)] = dict(
            pre_mean=round(float(np.mean(pre)), 4) if pre else None, n_pre_ignited=len(pre),
            plateau_mean=round(float(np.mean(post)), 4) if post else None,
            n_plateau_ignited=len(post),
            n_unignited=len(pre_all) + len(post_all) - n_ig, p_raw=float(u.pvalue))
        pm = f"{np.mean(pre):+.4f}" if pre else "n/a"
        qm = f"{np.mean(post):+.4f}" if post else "n/a"
        print(f"  {t:>5} {pm:>10} {len(pre):>4} {qm:>10} {len(post):>4} "
              f"{n_ig:>9} {u.pvalue:>10.2e}")

    names, praw = zip(*tests)
    padj = bh_fdr(praw)
    print(f"\n=== BH-FDR over the temperatures tested here ===")
    for n_, pr, pa in zip(names, praw, padj):
        print(f"  {n_:>6} p_raw={pr:.5f}  p_BH={pa:.5f}  "
              f"{'SURVIVES' if pa < 0.05 else 'n.s.'}")
        summary[n_.replace("T", "")]["p_bh"] = float(pa)

    survives = [n_ for n_, pa in zip(names, padj) if pa < 0.05]
    if len(survives) == len(TEMPS):
        verdict = ("PERSISTS at every temperature tested -- the developmental transition is not "
                   "an artifact of T=0.7")
    elif not survives:
        verdict = ("VANISHES at both tested temperatures -- the claim is temperature-specific "
                   "and must be restated as such")
    else:
        verdict = (f"PARTIAL: survives at {survives} but not at the others -- see the per-"
                   f"temperature means before reading this either way")
    print(f"\n  -> {verdict}")

    res["reference_T07"] = ref
    res["summary"] = summary
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Robustness of the developmental transition to sampling temperature (issue #17). The "
        "headline is measured at T=0.7 only, and temperature is a known-strong axis here (F12; "
        "the cross-level temperature confound), so this tests the two ends of the developmental "
        "curve at T=0.3 and T=1.1 against the Phase 3 T=0.7 cells. lambda statistics exclude "
        "unignited runs (F42) with n stated; the rank test keeps all runs, since ranks do not "
        "depend on a dead run's magnitude.")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}")


if __name__ == "__main__":
    main()
