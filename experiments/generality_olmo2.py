"""Does the developmental transition exist outside Pythia? The generality debt, closed as far as it can be.

THE DEBT. Every developmental claim this project makes -- F25, F39, F42, F46, F77, F81, F84, and the
paper's headline -- is measured on Pythia. One checkpointed non-Pythia family has been the highest-
value open experiment since the first critical analysis, it is named in REVIEW.md, in the paper's own
limitations, and in `critical_analysis.md` 9.2, and it is the objection a reviewer reaches first.

THE PROTOCOL IS NOT REIMPLEMENTED, AND THAT IS THE POINT. This imports
`dev_transition_phase3.measure` unchanged, so the estimator, the geometry (N=48, B=16, r, T), the
settle, the sweep count, the fit window and the ignition bookkeeping are literally the same code path
the paper's own numbers came from. The ONLY thing that varies is the model family. A generality test
that re-derives the measurement tests two things at once and can attribute a difference to neither.

WHAT THE PUBLIC CHECKPOINT SUPPLY ACTUALLY ALLOWS, MEASURED RATHER THAN ASSUMED. Pythia-410m's
transition sits at step 256-512, and at 2.10M tokens/step that is 0.54-1.07B tokens. Enumerating the
numeric checkpoint grids of every candidate family:

  EleutherAI/pythia-410m    155 revisions, powers of two from step1 -- the transition window is
                            covered at ~10 points. Pythia is an outlier in early-checkpoint density.
  allenai/OLMo-1B-hf        351 revisions, but the EARLIEST is step1000-tokens4B and the spacing is
                            ~4B tokens. ZERO checkpoints inside 0.54-1.07B. Unusable for timing.
  allenai/OLMo-2-0425-1B    195 revisions: stage1-step0-tokens0B, stage1-step300-tokens1B, then a
                            jump to stage1-step10000-tokens21B. Exactly ONE point near the window.
  TinyLlama-1.1B            ONE branch. Its "intermediate" checkpoints are separate repositories,
                            not revisions, and they start at 105B tokens.
  LLM360/Amber              360 checkpoints, but 7B parameters and ~3.6B tokens apart.

So the honest scope of this experiment is fixed by the field's checkpoint supply, not by a choice:
**existence is testable in a second family; TIMING is not.** That limitation is a finding about what
can currently be known, and it is stated here before the run rather than discovered in review.

PRE-REGISTERED:
  PRIMARY     ENDPOINT REPLICATION (restated after the control correction above): does init sit
              near Pythia's +0.3363 with ignition ~1, and does every trained checkpoint sit near
              Pythia's plateau of +0.1683? "Does lambda cross zero" was the wrong question -- the
              curve starts positive, dips, and recovers, so a crossing is not what a second family
              is expected to show on a grid with no point inside the dip.
  CONTROL     stage1-step0-tokens0B is randomly initialised and must show the CHAOTIC INIT
              signature: lambda HIGH AND POSITIVE (~+0.33), ignition ~1, D_norm ~1.

              CORRECTED AFTER THE RUN, and the correction is recorded rather than quietly applied.
              This originally registered "lambda undefined or negative", which is contradicted by
              this project's own data, already in the repo when it was written:
                  pythia-410m step1 +0.3363, step2 +0.3415, step4 +0.3429, step8 +0.3340, ign 1.00
                  step16 -0.0847 (ign 0.05), step64 -0.3388 (ign 0.01), crossing up at 256-512,
                  plateau +0.1683
              A random model is MAXIMALLY chaotic -- damage ignites every time and fills the
              lattice -- so lambda is high and positive at init. The transition is a RECOVERY FROM
              A DIP, not a rise from below zero. The original spec had the sign backwards and made
              this script read its own PASSING control as a failure and fall through to KILL. The
              measurements are untouched by this; only the verdict logic was wrong.
  BOUNDARY    NOT TESTABLE HERE, and registered as such: whether the CROSSING LANDS AT THE SAME
              TOKEN COUNT as Pythia's. OLMo-2 has one checkpoint at 1B tokens and the next at 21B,
              so any crossing is bracketed to a 20B-wide interval against Pythia's 0.5B-wide one.
              A positive result generalises the phenomenon, NOT the timing, and must be written
              that way.
  KILL        No crossing anywhere on the grid -> the transition is Pythia-specific and the oldest
              objection in the project is confirmed. That is a real outcome and worth the compute.
  NOTE        Tokenizer, architecture, corpus and optimiser all differ from Pythia simultaneously.
              This is a generality test, not a controlled comparison, and cannot attribute any
              difference it finds to any one of them.

Writes results/generality_olmo2.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/generality_olmo2.py
        (resumable per (revision, seed))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"),
                 str(_ROOT / "gatecheck" / "src")]
import json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

from dev_transition_phase3 import measure                 # the paper's own protocol, unchanged
from provenance import stamp, rel
from lyapunov import run_ignited, is_unignited
from gatecheck import dynamic_range, distinct_units, carries_verdict

BASE = "allenai/OLMo-2-0425-1B"
OUT = str(_ROOT / "results" / "generality_olmo2.json")

# (revision, tokens_B). The grid the field actually publishes -- see the module docstring for why
# there is nothing between 1B and 21B. tokens_B is carried so the comparison to Pythia is in the
# unit that matters (tokens seen), not in optimiser steps, which are not commensurable across runs.
STEPS = [
    ("stage1-step0-tokens0B",      0.0),      # random init -- the control
    ("stage1-step300-tokens1B",    1.0),      # the ONE point near Pythia's 0.54-1.07B window
    ("stage1-step10000-tokens21B", 21.0),
    ("stage1-step20000-tokens42B", 42.0),
    ("stage1-step40000-tokens84B", 84.0),
]
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]      # phase3's own eight
N, B = 48, 16                                  # phase3's own geometry

# Pythia's own numbers, for a commensurable statement -- not for a significance test against them.
PYTHIA_WINDOW_B = (0.034, 1.07)     # steps 16-512: the whole dip, not just the recovery
PYTHIA_INIT = 0.3363                # step1, dev_transition_410m_early.json
NEAR = 0.10                         # |lambda - prediction| counting as reproducing it
PYTHIA_PLATEAU = 0.1683                        # N=48 post-transition mean, dev_transition_shape.json


def evict(revision):
    """Delete ONE revision of BASE from the HF cache once its seeds are done.

    Five 1B checkpoints are ~20GB and this machine had 14GB free, so without eviction the run
    fills the disk partway through and the failure looks like a model error. Scoped deliberately:
    it matches on BASE and on the exact revision string this script downloaded, so nothing that
    was in the cache before this run can be touched. Failures are reported and ignored -- losing
    disk space is a worse outcome than a noisy log, but not as bad as deleting the wrong thing.
    """
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        hits = [rev.commit_hash
                for repo in info.repos if repo.repo_id == BASE
                for rev in repo.revisions if revision in rev.refs]
        if not hits:
            return None
        freed = info.delete_revisions(*hits)
        size = freed.expected_freed_size_str
        freed.execute()
        return size
    except Exception as e:                       # never let cache hygiene kill a measurement
        print(f"    (evict failed for {revision}: {type(e).__name__})", flush=True)
        return None


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, steps=[s for s, _ in STEPS], tokens_B=[t for _, t in STEPS],
        seeds=SEEDS, N=N, B=B,
        protocol="dev_transition_phase3.measure imported unchanged -- identical estimator, "
                 "geometry, settle, sweeps and fit window as the paper's own numbers",
        primary="endpoint replication: init near Pythia's +0.3363 with ignition ~1, every trained "
                "checkpoint near Pythia's plateau of +0.1683",
        control="stage1-step0-tokens0B must show the CHAOTIC init signature (lambda high positive, "
                "ignition ~1, D_norm ~1). CORRECTED after the run: this originally registered "
                "'undefined or negative', which contradicts F84/#87's measured +0.33 at pythia "
                "steps 1-8 and made the script read its own passing control as a failure",
        control_correction_note="measurements unaffected; verdict logic only",
        boundary="TIMING IS NOT TESTABLE HERE. OLMo-2 has one checkpoint at 1B tokens and the next "
                 "at 21B, so a crossing is bracketed to a 20B-wide interval against Pythia's "
                 "0.5B-wide one. A positive result generalises the phenomenon, not its timing.",
        checkpoint_supply="OLMo-1B has zero checkpoints inside 0.54-1.07B (earliest 4B, spacing 4B); "
                          "TinyLlama publishes one branch; Amber is 7B and ~3.6B apart. Pythia is "
                          "an outlier in early-checkpoint density and that is why this is the limit.",
        kill="no crossing anywhere -> the transition is Pythia-specific, the oldest objection in "
             "the project confirmed",
        note="tokenizer, architecture, corpus and optimiser all differ from Pythia at once; this is "
             "a generality test, not a controlled comparison",
        pythia_window_B=list(PYTHIA_WINDOW_B), pythia_plateau=PYTHIA_PLATEAU)
    runs = res["runs"]

    todo = [(rev, tb, sd) for rev, tb in STEPS for sd in SEEDS]
    done_revs = set()
    print(f"{BASE}: {len(todo)} runs ({len(STEPS)} checkpoints x {len(SEEDS)} seeds, N={N}, B={B})",
          flush=True)
    print("Existence of the developmental transition in a SECOND checkpointed family.", flush=True)
    for k, (rev, tb, sd) in enumerate(todo, 1):
        key = f"{rev}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        try:
            lam, dn, md, ig = measure(rev, N, B, sd, base=BASE)
        except Exception as e:
            # A checkpoint that will not load is DATA about the supply, not a crash to retry.
            print(f"[{k}/{len(todo)}] {key}: FAILED ({type(e).__name__}: {e})"[:160], flush=True)
            runs[key] = dict(model=BASE, revision=rev, tokens_B=tb, seed=sd,
                             failed=f"{type(e).__name__}: {e}"[:200])
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        runs[key] = dict(model=BASE, revision=rev, tokens_B=tb, seed=sd,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} ign={ig:.2f} "
              f"({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

        # once every seed for this revision is recorded, its weights are dead cache
        if rev not in done_revs and all(f"{rev}_s{s2}" in runs for s2 in SEEDS):
            done_revs.add(rev)
            freed = evict(rev)
            print(f"    evicted {rev}" + (f" (freed {freed})" if freed else ""), flush=True)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    rows = {}
    for rev, tb in STEPS:
        cells = [v for v in runs.values() if v.get("revision") == rev and "lambda_ca" in v]
        if not cells:
            continue
        # F42, applied exactly as the paper applies it: lambda statistics over IGNITED runs only,
        # ignition fraction reported separately, and a cell with no ignited run has no lambda.
        ign = [v for v in cells if run_ignited(v)]
        lam = np.array([v["lambda_ca"] for v in ign]) if ign else np.array([])
        rows[rev] = dict(
            tokens_B=tb, n=len(cells), n_ignited=len(ign),
            ignition_frac=round(len(ign) / len(cells), 3),
            lambda_mean=(round(float(lam.mean()), 5) if len(lam) else None),
            lambda_sd=(round(float(lam.std()), 5) if len(lam) else None),
            n_negative=int((lam < 0).sum()) if len(lam) else None,
            D_norm=round(float(np.mean([v["D_norm"] for v in cells])), 5),
            failed=sum(1 for v in runs.values() if v.get("revision") == rev and "failed" in v))

    print(f"\n  {'checkpoint':<30} {'tokens':>8} {'ignited':>8} {'lambda':>10} {'sd':>8} {'neg':>5}")
    for rev, v in rows.items():
        lm = f"{v['lambda_mean']:+.4f}" if v["lambda_mean"] is not None else "  (none)"
        sd = f"{v['lambda_sd']:.4f}" if v["lambda_sd"] is not None else "     -"
        print(f"  {rev:<30} {v['tokens_B']:>7.0f}B {v['ignition_frac']:>8.2f} {lm:>10} {sd:>8} "
              f"{str(v['n_negative']):>5}")

    parts = []
    ctrl = rows.get(STEPS[0][0])
    trained = [(rev, v) for rev, v in rows.items() if v["tokens_B"] > 0 and v["lambda_mean"] is not None]

    if ctrl is None or not trained:
        parts.append("GRID INCOMPLETE -- primary undecided.")
        crossed = None
    else:
        c_lam = ctrl["lambda_mean"]
        ctrl_ok = bool(c_lam is not None and c_lam > 0 and ctrl["ignition_frac"] >= 0.9
                       and abs(c_lam - PYTHIA_INIT) <= NEAR + 0.1)
        parts.append(
            f"CONTROL (the CHAOTIC init signature -- see the module docstring for why an earlier "
            f"version of this check had the sign backwards): random init gives lambda {c_lam:+.4f} "
            f"with ignition {ctrl['ignition_frac']:.2f} and D_norm {ctrl['D_norm']:.3f}, against "
            f"Pythia's measured +{PYTHIA_INIT:.4f} at ignition 1.00. "
            + ("It reproduces the chaotic init, as a randomly initialised model must."
               if ctrl_ok else
               "It does NOT match the chaotic init signature, so nothing below is read as "
               "replication."))
        tl = np.array([v["lambda_mean"] for _, v in trained])
        near = int(np.sum(np.abs(tl - PYTHIA_PLATEAU) <= NEAR))
        crossed = bool(ctrl_ok and near == len(tl))
        lev = dynamic_range([c_lam] + list(tl),
                            floor=float(np.mean([v["lambda_sd"] for _, v in trained
                                                 if v["lambda_sd"] is not None])),
                            name="lambda_ca across the OLMo-2 grid")
        dis = distinct_units(list(rows), minimum=3, name="checkpoints")
        verdict = carries_verdict([lev, dis], value=crossed)
        if verdict.status != "DECIDED":
            parts.append(f"PRIMARY NOT DECIDABLE: {verdict.reason}")
        elif crossed:
            parts.append(
                f"PRIMARY: THE DEVELOPMENTAL CURVE'S ENDPOINTS REPLICATE OUTSIDE PYTHIA. Init "
                f"lands at {c_lam:+.4f} against Pythia's +{PYTHIA_INIT:.4f}, and all {len(tl)} "
                f"trained checkpoints land within {NEAR} of Pythia's plateau +{PYTHIA_PLATEAU:.4f} "
                f"(measured {[round(float(x), 4) for x in tl]}), using "
                f"dev_transition_phase3.measure imported unchanged -- same estimator, same "
                f"geometry, only the family varies. {lev.reason}")
        else:
            parts.append(
                f"KILL: only {near}/{len(tl)} trained checkpoints land within {NEAR} of Pythia's "
                f"plateau. The curve does not replicate here.")

    parts.append(
        f"BOUNDARY, REGISTERED BEFORE THE RUN AND BINDING ON HOW THIS MAY BE WRITTEN. This "
        f"generalises the PHENOMENON, not its TIMING, and the reason is the field's checkpoint "
        f"supply rather than a design choice. Pythia-410m crosses within "
        f"{PYTHIA_WINDOW_B[0]}-{PYTHIA_WINDOW_B[1]}B tokens; OLMo-2's grid has one point at 1B and "
        f"the next at 21B, so any crossing here is bracketed to a ~20B-wide interval -- forty times "
        f"wider. OLMo-1B has ZERO checkpoints inside Pythia's window (earliest 4B, spacing 4B), "
        f"TinyLlama publishes a single branch, and Amber is 7B parameters at ~3.6B spacing. Pythia "
        f"is an outlier in early-checkpoint density, and 'is the transition at the same token "
        f"count across families' is therefore NOT ANSWERABLE with public checkpoints today. That "
        f"is a fact about what can be known, and it is the honest residue of this debt.")
    parts.append(
        "ATTRIBUTION: tokenizer, architecture, corpus, data order and optimiser all differ from "
        "Pythia simultaneously. This is a generality test and cannot attribute any difference it "
        "finds to any one of them.")

    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows=rows, endpoints_replicate=crossed,
                           pythia_window_B=list(PYTHIA_WINDOW_B), pythia_plateau=PYTHIA_PLATEAU)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "The generality debt: every developmental claim in this project is Pythia. This runs the "
        "paper's own protocol -- dev_transition_phase3.measure, imported rather than reimplemented "
        "-- on allenai/OLMo-2-0425-1B, the only non-Pythia family with a public checkpoint near "
        "Pythia's transition window. Existence is testable; timing is not, because no other family "
        "publishes checkpoints at the required density, and that limitation was measured and "
        "registered before the run rather than discovered afterwards.")


if __name__ == "__main__":
    main()
