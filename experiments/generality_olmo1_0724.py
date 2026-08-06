"""Second non-Pythia family: does the developmental curve's SHAPE replicate twice?

WHY A SECOND FAMILY. `generality_olmo2.py` measured OLMo-2-0425-1B and found both endpoints of
Pythia's developmental curve reproduced: init lambda ~ +0.36 against Pythia's +0.334..+0.343, and a
plateau ~ +0.18/+0.19 against Pythia's +0.1683. That is a real result resting on ONE family, which is
exactly the position the generality debt was opened to escape. Two independent families make the
endpoint replication substantially harder to attribute to a coincidence of one training recipe.

THE CONTROL IS SPECIFIED CORRECTLY HERE, AND IT WAS NOT IN THE FIRST SCRIPT. generality_olmo2.py
registered "the untrained anchor must look untrained: lambda undefined or negative". That is
contradicted by this project's own data, which was already in the repo when it was written:

    pythia-410m  step1 +0.3363  step2 +0.3415  step4 +0.3429  step8 +0.3340   ignition 1.00
                 step16 -0.0847 (ignition 0.05)      step64 -0.3388 (ignition 0.01)
                 crossing back up at step256-512     plateau +0.1683

A randomly initialised model is MAXIMALLY CHAOTIC -- damage ignites every time and fills the lattice,
so lambda sits high and positive (~+0.33, F84/#87) with D_norm ~ 1. The transition is a RECOVERY FROM
A DIP, not a rise from below zero. The first script had the sign backwards and would have reported
its own passing control as a failure. The measurements were unaffected; only the verdict logic was
wrong, and it is fixed here rather than repeated.

WHAT IS AND IS NOT TESTABLE, MEASURED ACROSS ~4000 BRANCHES IN SIX FAMILIES. Pythia's dip spans
steps 16-512 = 0.034-1.07B tokens. Checkpoints strictly inside that window:

  allenai/OLMo-2-0425-1B    0B, then 1B (step300)                    one boundary point
  allenai/OLMo-1B-0724-hf   0B, then 2B (step1000); 1446 checkpoints  NONE inside
  allenai/OLMo-2-1124-7B    1B (step150), then 3B                     one boundary point, 7B params
  LLM360/CrystalCoder       250 checkpoints, 1500-step spacing over 1.4T
  LLM360/K2                 141 checkpoints, 65B parameters
  stablelm-2, SmolLM2, bloom-1b1, neo, open_llama    a single branch each

The window is EMPTY for every public non-Pythia family. So the dip itself is not observable by
anyone today, and this script does not pretend otherwise: its trained checkpoints all sit past the
recovery, which makes the prediction sharp and cheap to falsify.

PRE-REGISTERED:
  PRIMARY     ENDPOINT REPLICATION. Does init sit near Pythia's +0.33 with ignition ~1, and does
              every trained checkpoint sit near Pythia's plateau of +0.1683? Both are quantitative
              predictions with the numbers fixed before the run.
  CONTROL     step0-tokens0B must show the CHAOTIC init signature -- lambda high and positive,
              ignition near 1, D_norm near 1. If it looks like the plateau instead, the revision is
              not what its name says.
  RISKY       Because the earliest trained checkpoint is 2B tokens and Pythia has recovered by
              ~1.07B, EVERY trained point here should already be at plateau. A negative lambda at
              any of them would mean this family's dip extends far later than Pythia's -- an
              interesting failure, and the only way this design can see a dip at all.
  BOUNDARY    TIMING IS NOT TESTABLE. Nothing inside 0.034-1.07B exists to measure.
  KILL        Trained lambdas scattered far from +0.1683, or an init that does not look chaotic ->
              the curve does not replicate and the endpoint result from OLMo-2 was family-specific.
  NOTE        Tokenizer, architecture, corpus, data order and optimiser all differ from Pythia at
              once. Generality test, not a controlled comparison.

Writes results/generality_olmo1_0724.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/generality_olmo1_0724.py
        (resumable per (revision, seed); evicts each checkpoint from the HF cache when done)
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
from lyapunov import run_ignited
from gatecheck import dynamic_range, distinct_units, carries_verdict

BASE = "allenai/OLMo-1B-0724-hf"
OUT = str(_ROOT / "results" / "generality_olmo1_0724.json")

STEPS = [
    ("step0-tokens0B",      0.0),      # random init -- the chaotic control
    ("step1000-tokens2B",   2.0),      # earliest trained checkpoint that exists
    ("step2000-tokens4B",   4.0),
    ("step4000-tokens8B",   8.0),
    ("step10000-tokens20B", 20.0),
]
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]
N, B = 48, 16

# Pythia's own measured values, frozen here BEFORE the run as the quantitative prediction.
PYTHIA_INIT = 0.3363          # step1, dev_transition_410m_early.json
PYTHIA_PLATEAU = 0.1683       # N=48 post-transition mean, dev_transition_shape.json
PYTHIA_DIP_WINDOW_B = (0.034, 1.07)
NEAR = 0.10                   # |lambda - prediction| within this counts as reproducing it


def evict(revision):
    """Drop ONE revision of BASE from the HF cache once its seeds are done.

    Deliberately a local copy rather than an import from generality_olmo2: that script was RUNNING
    when this one was written, and adding a `base` parameter to its evict() would have meant editing
    a live job -- the #38 stale-analysis hazard. Scoped to BASE and to the exact revision string
    this script downloaded, so nothing pre-existing can be touched.
    """
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        hits = [rev.commit_hash
                for repo in info.repos if repo.repo_id == BASE
                for rev in repo.revisions if revision in rev.refs]
        if not hits:
            return None
        strat = info.delete_revisions(*hits)
        size = strat.expected_freed_size_str
        strat.execute()
        return size
    except Exception as e:                       # cache hygiene must never kill a measurement
        print(f"    (evict failed for {revision}: {type(e).__name__})", flush=True)
        return None


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, steps=[s for s, _ in STEPS], tokens_B=[t for _, t in STEPS],
        seeds=SEEDS, N=N, B=B, near=NEAR,
        protocol="dev_transition_phase3.measure imported unchanged",
        primary="endpoint replication: init near Pythia's +0.3363 with ignition ~1, every trained "
                "checkpoint near Pythia's plateau of +0.1683, both fixed before the run",
        control="step0-tokens0B must show the CHAOTIC init signature (lambda high positive, "
                "ignition ~1, D_norm ~1) -- NOT negative. generality_olmo2.py registered the "
                "opposite and had the sign backwards; F84/#87 measured +0.33 at pythia steps 1-8",
        risky="the earliest trained checkpoint is 2B tokens and Pythia recovers by ~1.07B, so every "
              "trained point should ALREADY be at plateau; a negative lambda at any would mean this "
              "family's dip extends far later than Pythia's",
        boundary="timing not testable: no public non-Pythia family has a checkpoint strictly inside "
                 "0.034-1.07B tokens (~4000 branches enumerated across six families)",
        kill="trained lambdas far from the plateau, or a non-chaotic init -> the curve does not "
             "replicate and OLMo-2's endpoint result was family-specific",
        pythia_init=PYTHIA_INIT, pythia_plateau=PYTHIA_PLATEAU,
        pythia_dip_window_B=list(PYTHIA_DIP_WINDOW_B))
    runs = res["runs"]

    todo = [(rev, tb, sd) for rev, tb in STEPS for sd in SEEDS]
    print(f"{BASE}: {len(todo)} runs ({len(STEPS)} checkpoints x {len(SEEDS)} seeds, N={N}, B={B})",
          flush=True)
    print(f"Predicts: init ~ {PYTHIA_INIT:+.4f}, every trained point ~ {PYTHIA_PLATEAU:+.4f}.",
          flush=True)
    done_revs = set()
    for k, (rev, tb, sd) in enumerate(todo, 1):
        key = f"{rev}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        try:
            lam, dn, md, ig = measure(rev, N, B, sd, base=BASE)
        except Exception as e:
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
        ign = [v for v in cells if run_ignited(v)]          # F42: lambda over ignited runs only
        lam = np.array([v["lambda_ca"] for v in ign]) if ign else np.array([])
        rows[rev] = dict(
            tokens_B=tb, n=len(cells), n_ignited=len(ign),
            ignition_frac=round(len(ign) / len(cells), 3),
            lambda_mean=(round(float(lam.mean()), 5) if len(lam) else None),
            lambda_sd=(round(float(lam.std()), 5) if len(lam) else None),
            n_negative=int((lam < 0).sum()) if len(lam) else None,
            D_norm=round(float(np.mean([v["D_norm"] for v in cells])), 5))

    print(f"\n  {'checkpoint':<24} {'tokens':>8} {'ign':>6} {'lambda':>10} {'sd':>8} "
          f"{'D_norm':>8} {'neg':>5}")
    for rev, v in rows.items():
        lm = f"{v['lambda_mean']:+.4f}" if v["lambda_mean"] is not None else " (none)"
        sd = f"{v['lambda_sd']:.4f}" if v["lambda_sd"] is not None else "    -"
        print(f"  {rev:<24} {v['tokens_B']:>7.0f}B {v['ignition_frac']:>6.2f} {lm:>10} {sd:>8} "
              f"{v['D_norm']:>8.4f} {str(v['n_negative']):>5}")

    parts = []
    ctrl = rows.get(STEPS[0][0])
    trained = [(r, v) for r, v in rows.items() if v["tokens_B"] > 0 and v["lambda_mean"] is not None]

    if ctrl is None or not trained:
        parts.append("GRID INCOMPLETE -- primary undecided.")
        replicates = None
    else:
        c = ctrl["lambda_mean"]
        ctrl_ok = bool(c is not None and c > 0 and ctrl["ignition_frac"] >= 0.9
                       and abs(c - PYTHIA_INIT) <= NEAR + 0.1)
        parts.append(
            f"CONTROL (the CHAOTIC init signature, not a negative one): random init gives lambda "
            f"{c:+.4f} with ignition {ctrl['ignition_frac']:.2f} and D_norm {ctrl['D_norm']:.3f}, "
            f"against Pythia's measured +{PYTHIA_INIT:.4f} at ignition 1.00. "
            + ("It reproduces the chaotic init, as a randomly initialised model must -- damage "
               "ignites every time and fills the lattice."
               if ctrl_ok else
               "IT DOES NOT match the chaotic init signature, so either the revision is not what "
               "its name says or the init regime differs between families; nothing below is read "
               "as replication."))
        tl = np.array([v["lambda_mean"] for _, v in trained])
        near = int(np.sum(np.abs(tl - PYTHIA_PLATEAU) <= NEAR))
        any_neg = int(np.sum(tl < 0))
        replicates = bool(ctrl_ok and near == len(tl))
        lev = dynamic_range([c] + list(tl),
                            floor=float(np.mean([v["lambda_sd"] for _, v in trained
                                                 if v["lambda_sd"] is not None])),
                            name="lambda_ca across the grid")
        dis = distinct_units(list(rows), minimum=3, name="checkpoints")
        verdict = carries_verdict([lev, dis], value=replicates)
        if verdict.status != "DECIDED":
            parts.append(f"PRIMARY NOT DECIDABLE: {verdict.reason}")
        elif replicates:
            parts.append(
                f"PRIMARY: THE DEVELOPMENTAL CURVE'S ENDPOINTS REPLICATE IN A SECOND NON-PYTHIA "
                f"FAMILY. Init lands at {c:+.4f} against Pythia's +{PYTHIA_INIT:.4f}, and all "
                f"{len(tl)} trained checkpoints land within {NEAR} of Pythia's plateau "
                f"+{PYTHIA_PLATEAU:.4f} (measured {[round(float(x), 4) for x in tl]}). With "
                f"OLMo-2-0425-1B this is two independent families, both measured by "
                f"dev_transition_phase3.measure imported unchanged -- same estimator, same "
                f"geometry, only the family varies. {lev.reason}")
        else:
            parts.append(
                f"KILL: {near}/{len(tl)} trained checkpoints land within {NEAR} of Pythia's "
                f"plateau, and {any_neg} are negative. The curve does not replicate here, so "
                f"OLMo-2's endpoint agreement was family-specific rather than general.")
        if any_neg:
            parts.append(
                f"NOTE, and it is the risky prediction paying off in the interesting direction: "
                f"{any_neg} trained checkpoint(s) are NEGATIVE despite sitting past 2B tokens, "
                f"where Pythia has long recovered. That would put this family's dip later than "
                f"Pythia's in token terms and is the only way this design can observe a dip at all.")

    parts.append(
        f"BOUNDARY, BINDING ON HOW THIS MAY BE WRITTEN. This replicates the curve's ENDPOINTS, not "
        f"its TIMING. Pythia's dip spans {PYTHIA_DIP_WINDOW_B[0]}-{PYTHIA_DIP_WINDOW_B[1]}B tokens "
        f"and NO public non-Pythia family has a checkpoint strictly inside it -- ~4000 branches "
        f"enumerated across OLMo-1B-0724 (1446, earliest trained 2B), OLMo-2-0425-1B (0B then 1B), "
        f"OLMo-2-1124-7B, CrystalCoder, K2, and five single-branch repositories. The dip is not "
        f"observable by anyone today, and that is a fact about the field's checkpoint supply "
        f"rather than a limitation of this design.")
    parts.append(
        "ATTRIBUTION: tokenizer, architecture, corpus, data order and optimiser all differ from "
        "Pythia simultaneously; this cannot attribute any difference to any one of them.")

    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows=rows, replicates=replicates, pythia_init=PYTHIA_INIT,
                           pythia_plateau=PYTHIA_PLATEAU,
                           pythia_dip_window_B=list(PYTHIA_DIP_WINDOW_B), near=NEAR)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Second non-Pythia family for the generality debt. Runs the paper's own protocol unchanged "
        "on allenai/OLMo-1B-0724-hf (1446 checkpoints, earliest trained at 2B tokens). Predicts, "
        "with Pythia's numbers frozen beforehand, a chaotic init near +0.3363 and every trained "
        "checkpoint near the +0.1683 plateau. The control is specified against F84/#87's MEASURED "
        "init signature -- lambda high and positive with full ignition -- correcting "
        "generality_olmo2.py, which registered the opposite sign and would have read its own "
        "passing control as a failure.")


if __name__ == "__main__":
    main()
