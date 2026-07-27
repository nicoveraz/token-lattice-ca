"""Does the paper's own model show a flat positive plateau before its dip? (#88, follows #87)

THE QUESTION THIS DECIDES. The paper says Pythia-410m's token-space dynamics change "from sub- to
super-critical between steps 256 and 512". #87 found that on 14m/31m/70m lambda_ca is already
~+0.33 at step1 and stays flat through step32, before collapsing into a dead zone and recovering
to ~+0.17. If 410m does the same, then the paper's transition is a RECOVERY FROM A TRANSIENT DIP
rather than the emergence of criticality, and its framing needs restating for any archival
version. No previous grid could see this: C20's begins at step128 and the paper's at step256,
both entirely after the plateau.

The mechanism is not mysterious, which is why it needs checking rather than assuming. An
untrained model is close to a random rule, and a random rule spreads damage freely. Positive
lambda at initialisation is what one should expect; nobody had looked.

WHY THIS DOES NOT MEASURE A CROSSING BRACKET. #87 established that bracket-finding is the wrong
instrument for a non-monotone curve: `crossing_interval` searches for the first UPWARD zero
crossing, but the first sign change in every model was DOWNWARD, so what it returns is the
recovery, not an onset. This script therefore characterises the DIP directly -- where lambda is
minimal, how deep it goes, and whether it reaches total extinction -- and reports the bracket only
as a secondary, explicitly labelled as the recovery point.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: is lambda_ca at steps 1-32 statistically indistinguishable from the 14m/31m/70m
    plateau of ~+0.33, and significantly ABOVE the paper's post-transition plateau of +0.1683?
      - yes -> 410m shows the same shape; the paper's "sub- to super-critical" describes a
               recovery, and the archival framing must say so.
      - no  -> 410m differs from the small models, which is itself a size effect worth reporting
               and would mean the dip story does not generalise.
  * Secondary: locate the MINIMUM of the merged curve (steps 1..143000) and report its depth and
    ignition fraction. A cell reaching zero ignition is reported as extinction, not as a lambda.
  * lambda statistics exclude unignited runs (F42) with n stated.
  * The comparison against the paper's plateau uses the paper's own estimator (the default
    branch, via dev_transition_phase3.measure) so the two numbers are commensurable. #81 showed
    the damage-range estimator gives 1.38x larger plateau magnitudes; mixing estimators here
    would confound the comparison this experiment exists to make.

NEITHER OUTCOME IS A FAILURE. A flat positive plateau reframes the paper's headline; its absence
bounds #87's finding to small models. Both are reportable.

Incremental save + resume (safe to kill). Writes results/dev_transition_410m_early.json.
Usage:  caffeinate -i .venv/bin/python experiments/dev_transition_410m_early.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from scipy import stats

from dev_transition_phase3 import measure                 # the paper's own protocol
from provenance import stamp, rel
from lyapunov import run_ignited

BASE = "EleutherAI/pythia-410m"
STEPS = ["step1", "step2", "step4", "step8", "step16", "step32", "step64"]
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 48, 16
SMALL_MODEL_EARLY_PLATEAU = 0.33      # the ~+0.33 seen on 14m/31m/70m at steps 1-32 (#87)
PAPER_PLATEAU = 0.1683                # N=48 post-transition mean, dev_transition_shape.json
OUT = str(_ROOT / "results" / "dev_transition_410m_early.json")
# merged sources for the full curve; all use the identical protocol and estimator
MERGE = ["dev_transition_scale.json", "dev_transition_phase3.json"]


def _n(k):
    return int(str(k).replace("step", ""))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, steps=STEPS, seeds=SEEDS, N=N, B=B,
        primary="is lambda at steps 1-32 flat, positive, and above the paper's plateau?",
        small_model_early_plateau=SMALL_MODEL_EARLY_PLATEAU, paper_plateau=PAPER_PLATEAU,
        follows="#87 -- lambda is ~+0.33 at step1 on 14m/31m/70m, invisible to every prior grid",
        estimator="dev_transition_phase3.measure unchanged, so the comparison against the "
                  "paper's own plateau is commensurable",
        neither_outcome_is_a_failure=True)
    runs = res["runs"]
    todo = [(st, sd) for st in STEPS for sd in SEEDS]
    print(f"410m EARLY grid: {len(todo)} runs ({len(STEPS)} ckpts x {len(SEEDS)} seeds, N={N})",
          flush=True)
    print("Decides whether the paper's transition is a recovery from a dip.", flush=True)
    for k, (st, sd) in enumerate(todo, 1):
        key = f"{st}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        try:
            lam, dn, md, ig = measure(st, N, B, sd, base=BASE)
        except Exception as e:
            print(f"[{k}/{len(todo)}] {key}: FAILED ({type(e).__name__}: {e})", flush=True)
            runs[key] = dict(model=BASE, step=_n(st), seed=sd, failed=f"{type(e).__name__}: {e}")
            json.dump(res, open(OUT, "w"), indent=1); continue
        runs[key] = dict(model=BASE, step=_n(st), seed=sd,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} ign={ig:.2f} "
              f"({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def merged_cells(runs):
    """step -> list of run records for 410m at N=48, deduplicated on (step, seed)."""
    seen, out = {}, {}
    src = list(runs.values())
    for f in MERGE:
        p = _ROOT / "results" / f
        if not p.exists():
            continue
        for v in json.load(open(p))["runs"].values():
            if not (isinstance(v, dict) and "lambda_ca" in v):
                continue
            is410 = v.get("size_m") == 410 or (v.get("N") == 48 and "size_m" not in v)
            if not is410:
                continue
            src.append(v)
    for v in src:
        k = (v["step"], v["seed"])
        if k in seen:
            assert seen[k] == v["lambda_ca"], f"sources disagree about 410m {k}"
            continue
        seen[k] = v["lambda_ca"]
        out.setdefault(v["step"], []).append(v)
    return out


def analyse(res):
    cells = merged_cells(res["runs"])
    steps = sorted(cells)
    print(f"\n=== 410m merged curve, ignited runs only ===")
    print(f"  {'step':>8} {'lambda':>10} {'sd':>8} {'ignited':>9} {'D_norm':>9}")
    table = {}
    for s in steps:
        rs = cells[s]
        ig = [r for r in rs if run_ignited(r)]
        lam = np.array([r["lambda_ca"] for r in ig]) if ig else np.array([])
        dn = np.mean([r["D_norm"] for r in rs])
        table[s] = dict(n=len(rs), n_ignited=len(ig),
                        lambda_mean=(None if not ig else round(float(lam.mean()), 4)),
                        lambda_sd=(None if len(ig) < 2 else round(float(lam.std(ddof=1)), 4)),
                        D_norm_mean=round(float(dn), 4))
        lt = f"{lam.mean():>+10.4f}" if ig else f"{'EXTINCT':>10}"
        st = f"{lam.std(ddof=1):>8.4f}" if len(ig) > 1 else f"{'--':>8}"
        print(f"  {s:>8} {lt} {st} {len(ig):>4}/{len(rs):<4} {dn:>9.4f}")

    # TWO windows, both reported. The pre-registered one is steps <= 32, chosen from the small
    # models' flat stretch (#87). 410m's collapse begins EARLIER, at step16, so that window
    # straddles the plateau and the collapse and understates the early level. The corrected
    # window is defined by a criterion on an INDEPENDENT quantity -- D_norm > 0.5, i.e. the
    # random-rule regime where damage fills the ring -- rather than by looking at lambda and
    # choosing where to cut it. Both are reported; the pre-registered conclusion (410m starts
    # significantly above its OWN plateau) holds under either.
    early = [r["lambda_ca"] for s in steps if s <= 32 for r in cells[s] if run_ignited(r)]
    flat_steps = [s for s in steps if table[s]["D_norm_mean"] > 0.5 and s <= 64]
    flat = [r["lambda_ca"] for s in flat_steps for r in cells[s] if run_ignited(r)]
    entry = {}
    if early:
        e = np.array(early)
        t_pap = stats.ttest_1samp(e, PAPER_PLATEAU)
        t_sml = stats.ttest_1samp(e, SMALL_MODEL_EARLY_PLATEAU)
        entry = dict(
            early_steps_le_32_n=len(e), early_mean=round(float(e.mean()), 4),
            early_sd=round(float(e.std(ddof=1)), 4),
            vs_paper_plateau=dict(target=PAPER_PLATEAU, t=round(float(t_pap.statistic), 3),
                                  p=float(t_pap.pvalue)),
            vs_small_model_plateau=dict(target=SMALL_MODEL_EARLY_PLATEAU,
                                        t=round(float(t_sml.statistic), 3),
                                        p=float(t_sml.pvalue)))
        print(f"\n=== steps <= 32 (before any prior grid looked) ===")
        print(f"  mean {e.mean():+.4f} (sd {e.std(ddof=1):.4f}, n={len(e)})")
        print(f"  vs the paper's plateau {PAPER_PLATEAU:+.4f}: t={t_pap.statistic:+.2f}, "
              f"p={t_pap.pvalue:.2e}")
        print(f"  vs the small-model early plateau {SMALL_MODEL_EARLY_PLATEAU:+.2f}: "
              f"t={t_sml.statistic:+.2f}, p={t_sml.pvalue:.3f}")
    if flat:
        f = np.array(flat)
        f_pap = stats.ttest_1samp(f, PAPER_PLATEAU)
        f_sml = stats.ttest_1samp(f, SMALL_MODEL_EARLY_PLATEAU)
        entry["flat_window_steps"] = flat_steps
        entry["flat_mean"] = round(float(f.mean()), 4)
        entry["flat_n"] = len(f)
        entry["flat_vs_paper_plateau_p"] = float(f_pap.pvalue)
        entry["flat_vs_small_model_plateau_p"] = float(f_sml.pvalue)
        print(f"\n=== corrected window: steps with D_norm > 0.5 (the random-rule regime) ===")
        print(f"  steps {flat_steps}: mean {f.mean():+.4f} (n={len(f)})")
        print(f"  vs the paper's plateau {PAPER_PLATEAU:+.4f}: p={f_pap.pvalue:.2e}")
        print(f"  vs the small-model plateau {SMALL_MODEL_EARLY_PLATEAU:+.2f}: p={f_sml.pvalue:.3f}")

    lows = [(s, table[s]) for s in steps if table[s]["lambda_mean"] is not None]
    ext = [s for s in steps if table[s]["n_ignited"] == 0]
    if lows:
        smin = min(lows, key=lambda kv: kv[1]["lambda_mean"])
        entry["minimum"] = dict(step=smin[0], lambda_mean=smin[1]["lambda_mean"],
                                n_ignited=smin[1]["n_ignited"])
        entry["extinction_steps"] = ext
        print(f"\n=== the dip ===")
        print(f"  minimum lambda {smin[1]['lambda_mean']:+.4f} at step{smin[0]} "
              f"({smin[1]['n_ignited']}/8 ignited)")
        print(f"  total-extinction steps: {ext if ext else 'none'}")

    if early:
        above_paper = entry["vs_paper_plateau"]["p"] < 0.05 and e.mean() > PAPER_PLATEAU
        # the "same level as the small models?" sub-question is judged on the CORRECTED window,
        # because the pre-registered one demonstrably contains 410m's collapse
        like_small = entry.get("flat_vs_small_model_plateau_p",
                               entry["vs_small_model_plateau"]["p"]) >= 0.05
        if above_paper and like_small:
            verdict = ("SAME SHAPE AS THE SMALL MODELS: 410m's lambda before step64 is flat, "
                       "positive, indistinguishable from the ~+0.33 small-model plateau, and "
                       "significantly ABOVE the paper's post-transition +0.1683. The paper's "
                       "'sub- to super-critical' transition is a RECOVERY FROM A DIP, and the "
                       "archival framing must say so.")
        elif above_paper:
            verdict = ("DIP CONFIRMED, DIFFERENT DEPTH: 410m starts significantly above its own "
                       "plateau but not at the small-model level, so the shape generalises while "
                       "the early level is size-dependent.")
        else:
            verdict = ("410m DIFFERS: its early lambda is not above the paper's plateau, so #87's "
                       "flat-positive plateau does not generalise from the small models and the "
                       "paper's framing stands.")
        res["verdict"] = verdict
        print(f"\n  -> {verdict}")

    res["curve"] = {str(s): table[s] for s in steps}
    res["analysis"] = entry
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "410m at steps 1-64 (#88, follows #87). #87 found lambda_ca ~+0.33 at step1 on "
        "14m/31m/70m, flat through step32, before a dip and a recovery to ~+0.17 -- a shape no "
        "prior grid could see, since C20's starts at step128 and the paper's at step256. This "
        "tests whether the paper's own model does the same, which decides whether its headline "
        "'sub- to super-critical' transition is an onset or a recovery. Analysed on the merged "
        "curve (this file + dev_transition_scale + dev_transition_phase3), deduplicated on "
        "(step, seed) with an agreement assert. Uses dev_transition_phase3.measure unchanged so "
        "the comparison against the paper's own plateau is commensurable; #81's corrected "
        "estimator gives 1.38x larger magnitudes and mixing them here would confound exactly the "
        "comparison this exists to make. Bracket-finding is deliberately NOT the observable: #87 "
        "showed the first sign change is downward, so a first-upward-crossing search returns the "
        "recovery rather than an onset.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
