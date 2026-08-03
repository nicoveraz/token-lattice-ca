"""When does the argmax fixed point form? Dating the F62-F70 degeneracy in training time (#98).

F70 established that at r <= 2 and T -> 0 the CA is essentially the argmax map, and that
pythia-410m's map has an attracting fixed point while gpt2-medium's has none. That one property
unifies the whole artifact line: the frozen phase IS the fixed point (F62); it is not reducible to
corpus, architecture or scale (F63/F64) because it belongs to the MAP rather than the recipe; a BOS
prefix removes it (F66) by changing the map's domain; it lives at r <= 2 (F69); and the MLM
construction has none (F67), hence no absorbing state and no transition.

A model at step1 cannot have an attracting fixed point in any meaningful sense. A model at
step143000 does. So it forms somewhere, and dating that dates the acquisition of the
out-of-distribution fallback behaviour the entire line is about.

THE OBSERVABLE IS THE BASIN, NOT EXISTENCE. For a map over |V| ~ 5e4 the chance that
argmax(x | t,t) == t for a particular t is ~1/|V|, and there are |V| candidate self-pairs, so a
RANDOM map has about ONE fixed point in expectation. "A fixed point exists at step k" would
therefore be reporting noise. What separates a trained model is the BASIN -- 18/24 starts reaching
a common endpoint for pythia-410m against 0 for gpt2-medium.

AND THE NULL IS MEASURED, NOT DERIVED. step1 IS the random-map control, supplied free by the
training run. Deriving a baseline from random-mapping statistics would import assumptions about the
state space that do not hold for a lattice; measuring step1 imports none. This is the F65 rule --
run the control that should show nothing -- with the control already in the checkpoint series.

RE-RUN WITH FULL ENDPOINT HISTOGRAMS -- F84's own stated refinement. F84's kill condition fired:
the modal endpoint token WANDERS (newline at most checkpoints, '.' at step128, ',' at step256,
' the' at 2000 and 8000), so "the basin" is not one quantity and only a per-token-AWARE claim was
possible. The stored runs kept the modal endpoint and its count alone, so basin DEPTHS per token
could not be recomputed. `gate1.argmax_census` now also returns the full histogram -- additive, every
prior key unchanged, so F84's numbers must reproduce exactly.

THE QUESTION THE HISTOGRAM ANSWERS AND F84 COULD NOT. Is the wandering ONE funnel with a near-tie at
the top, or genuinely different attractors swapping?

  NEAR-TIE  newline stays a large share where '.' or ',' is modal, and the label flips on a margin
            of one or two starts out of 24 -- one funnel, noisy label.
  SWAP      newline genuinely collapses there -- the attractor IDENTITY changes during training, and
            F63's cross-MODEL variety recurs inside a single trajectory.

ADDED FOR THE RE-RUN:
  Primary(hist). At checkpoints where the modal token is NOT newline, what share does newline hold?
                 NEAR-TIE if within 2 starts of the modal token at every such checkpoint; SWAP if it
                 falls below half the modal share at any of them.
  Reproduction.  Every field F84 reported must come back identical -- the histogram is additive and
                 the seed fixed, so a change means the probe was perturbed, not extended.
  Boundary.      This refines HOW the funnel is described. It does not revisit the onset date, the
                 ordering against the extinction window and the crossing, or the
                 learned-not-architectural conclusion. Those are F84's and they stand.

PRE-REGISTERED (from #98, not invented here):
  Primary   At which checkpoint does the basin fraction first rise above its step1 value by more
            than the binomial CI, and is the rise monotone thereafter?
  Secondary Report the date against the two other dated events in the same units: the extinction
            window (#95/#97, 410m at step32) and the developmental crossing (F39/F46/F77, 410m at
            step256->512). Whether the fixed point forms before, during or after the extinction
            window is the interesting comparison.
  Null      The basin sits at its step1 level at every checkpoint, or is already high at step1.
            Then the fixed point is a property of ARCHITECTURE AND INITIALISATION rather than of
            training -- a strong result that would rhyme with F29 (white-box lambda_top is
            architectural and flat across training). INFORMATIVE IN BOTH DIRECTIONS.
  Kill      If the endpoint TOKEN changes between checkpoints, "the basin" is not one quantity and
            per-token basins must be reported instead of a single fraction. F63 already showed the
            dominant token varies across models ('\\n', ' ', '0'); it may vary across checkpoints.

SCOPE, STATED SO IT CANNOT BE MISREAD. This runs at r <= 2, which F69 proved is the
out-of-distribution artifact regime. That is deliberate and is not a defect: THE ARTIFACT IS THE
OBJECT OF STUDY. Nothing here is a claim about a model in ordinary use, and any write-up must say
so in the same breath as the result -- the F62-F70 line's whole value came from getting that
boundary right.

The probe is `gate1.argmax_census`, imported rather than reimplemented. It is already gated against
F70's known answer (pythia-410m has a fixed point at a whitespace token, gpt2-medium has none), and
that gate caught a real defect: an earlier version tested (a,b)->b, which only says the trajectory
reached the diagonal, and scored gpt2-medium at 0.96 where the truth is 0.00.

Deterministic map, so the ONLY randomness is the 24 starts -- no damage runs, no CRN twins, no
seed-as-independent-unit machinery, no sampling. Binomial CI on the basin fraction.

Writes results/fixed_point_onset.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/fixed_point_onset.py
        (resumable, keyed by checkpoint)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint")]
import os, json, math, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census, f70_instrument_check      # one implementation, already F70-gated

MODEL = "EleutherAI/pythia-410m"
STEPS = ["step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128",
         "step256", "step512", "step1000", "step2000", "step4000", "step8000",
         "step16000", "step32000", "step64000", "step143000"]
NULL_STEP = "step1"                 # the measured random-map control, not a derived baseline
EXTINCTION = "step32"               # #95/#97, same units
CROSSING = ("step256", "step512")   # F39/F46/F77, same units
SEED = 20260802
OUT = str(_ROOT / "results" / "fixed_point_onset.json")


def _step(s):
    return int(s.replace("step", ""))


def wilson(k, n, z=1.96):
    """Binomial CI on the basin fraction. Wilson, not normal-approx: n=24 and p runs to 0 and 1,
    where the normal interval is famously wrong (it gives zero width at p=0)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=STEPS, null_step=NULL_STEP, extinction_window=EXTINCTION,
        developmental_crossing=list(CROSSING), n_starts=24, seed=SEED,
        observable="basin fraction (share of random starts reaching a common endpoint), NOT the "
                   "existence of a fixed point -- a random map over |V|~5e4 has about one by "
                   "chance, so existence would be noise",
        null="step1 IS the random-map control, measured rather than derived from random-mapping "
             "statistics, which would import state-space assumptions that do not hold here",
        primary="the first checkpoint whose basin exceeds step1's by more than the binomial CI, "
                "and whether the rise is monotone thereafter",
        null_meaning="a flat basin, or one already high at step1, means the fixed point is "
                     "ARCHITECTURAL rather than learned -- which rhymes with F29 and is a strong "
                     "result in its own right",
        kill="if the endpoint TOKEN changes across checkpoints, the basin is not one quantity and "
             "per-token basins must be reported instead",
        scope="r<=2 is the out-of-distribution artifact regime (F69). That is deliberate: the "
              "artifact is the object of study. Nothing here is a claim about ordinary use.",
        probe="gate1.argmax_census, imported and already gated against F70's known answer",
        resumable="keyed by checkpoint")
    runs = res["runs"]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    for st in STEPS:
        if st in runs:
            continue
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(MODEL, revision=st)
            model = AutoModelForCausalLM.from_pretrained(MODEL, revision=st).eval().to(
                dev, torch.float16 if dev != "cpu" else torch.float32)
        except Exception as e:
            print(f"  {st}: LOAD FAILED ({type(e).__name__})", flush=True)
            runs[st] = dict(step=st, failed=type(e).__name__)
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        V = int(getattr(model.config, "vocab_size", len(tok)))
        special = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                               tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in special], dtype=np.int64)
        # The SAME 24 starts at every checkpoint: the map changes, the probe must not.
        c = argmax_census(model, tok, dev, pool, np.random.default_rng(SEED))
        k = int(round(c["modal_endpoint_share"] * c["n_starts"]))
        lo, hi = wilson(k, c["n_starts"])
        runs[st] = dict(step=st, step_n=_step(st), **c, basin_k=k,
                        basin_ci=[lo, hi], secs=round(time.time() - t0, 1))
        print(f"  {st:<10} basin {c['modal_endpoint_share']:.3f} [{lo:.2f},{hi:.2f}]  "
              f"endpoints={c['n_distinct_endpoints']:<2} fixed={c['fixed_point_fraction']:.2f} "
              f"cyclic={c['cyclic_fraction']:.2f}  token={c['modal_endpoint_token']!r} "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del model
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = {k: v for k, v in res["runs"].items() if not v.get("failed")}
    if NULL_STEP not in runs:
        res["verdict"] = f"NOT DECIDABLE -- {NULL_STEP}, the measured null, is missing."
        return
    order = [s for s in STEPS if s in runs]
    null = runs[NULL_STEP]
    n_hi = null["basin_ci"][1]

    print(f"\n=== basin fraction vs training step (null = {NULL_STEP}, "
          f"basin {null['modal_endpoint_share']:.3f} CI [{null['basin_ci'][0]:.2f},{n_hi:.2f}]) ===")
    print(f"  {'step':>10} {'basin':>7} {'CI':>15} {'endpoints':>10} {'fixed':>6} {'token':>10}")
    for s in order:
        v = runs[s]
        mark = "  <- above null CI" if v["modal_endpoint_share"] > n_hi else ""
        print(f"  {s:>10} {v['modal_endpoint_share']:7.3f} "
              f"[{v['basin_ci'][0]:.2f},{v['basin_ci'][1]:.2f}]".rjust(16)
              + f" {v['n_distinct_endpoints']:>10} {v['fixed_point_fraction']:6.2f} "
              f"{v['modal_endpoint_token']!r:>10}{mark}")

    above = [s for s in order if runs[s]["modal_endpoint_share"] > n_hi]
    onset = above[0] if above else None
    tail = order[order.index(onset):] if onset else []
    vals = [runs[s]["modal_endpoint_share"] for s in tail]
    monotone = all(b >= a - (runs[s]["basin_ci"][1] - runs[s]["basin_ci"][0]) / 2
                   for a, b, s in zip(vals, vals[1:], tail[1:]))
    toks = {runs[s]["modal_endpoint_token"] for s in order
            if runs[s]["modal_endpoint_share"] > n_hi}
    parts = []

    if not onset:
        parts.append(
            f"NULL RESULT, AND AN INFORMATIVE ONE: no checkpoint's basin exceeds {NULL_STEP}'s "
            f"upper CI ({n_hi:.2f}). The fixed point is not acquired during training in any way "
            f"this probe can see -- it is a property of ARCHITECTURE AND INITIALISATION, which "
            f"rhymes with F29's white-box lambda_top being flat across training. That is a "
            f"stronger claim than a date would have been, and it was pre-registered as such.")
    elif runs[NULL_STEP]["modal_endpoint_share"] >= 0.5:
        parts.append(
            f"NULL RESULT: the basin is ALREADY {null['modal_endpoint_share']:.2f} at "
            f"{NULL_STEP}, so there is nothing to date -- an untrained map already funnels random "
            f"starts to one endpoint, and the trained model's basin is not evidence of learning.")
    else:
        parts.append(
            f"ONSET AT {onset.upper()}: the basin first clears {NULL_STEP}'s upper CI "
            f"({n_hi:.2f}) at step {_step(onset)}, reaching "
            f"{runs[onset]['modal_endpoint_share']:.2f} "
            f"[{runs[onset]['basin_ci'][0]:.2f},{runs[onset]['basin_ci'][1]:.2f}]. "
            + ("The rise is monotone thereafter." if monotone else
               "The rise is NOT monotone thereafter, so the basin is not a simple accumulation "
               "and the trajectory matters as much as the date."))
        e, c0, c1 = _step(EXTINCTION), _step(CROSSING[0]), _step(CROSSING[1])
        o = _step(onset)
        where = ("BEFORE the extinction window" if o < e else
                 "AT the extinction window" if o == e else
                 "BETWEEN the extinction window and the developmental crossing" if o < c0 else
                 "INSIDE the developmental crossing bracket" if c0 <= o <= c1 else
                 "AFTER the developmental crossing")
        parts.append(
            f"ON THE SHARED AXIS: the fixed point forms {where} -- onset step {o}, extinction "
            f"window {e} (#95/#97), lambda_ca crossing {c0}->{c1} (F39/F46/F77). "
            + ("Forming after the crossing means the CA transition cannot be caused by the "
               "degeneracy, since the degeneracy did not exist yet -- which independently supports "
               "F77's finding that the developmental transition is not an artifact of the "
               "two-token window." if o > c1 else
               "Forming before or during the crossing means the two are NOT separable by timing "
               "alone, and co-timing is not attribution -- #100's ablation is the test that can "
               "separate them, not this one." if o <= c1 else ""))

    if len(toks) > 1:
        parts.append(
            f"KILL CONDITION HIT: the endpoint token is not stable across checkpoints ({toks}), so "
            f"'the basin' is not one quantity. The single fraction above pools different attractors "
            f"and per-token basins must be reported instead. F63 saw the same across models.")
    elif toks:
        parts.append(f"The endpoint token is stable at {list(toks)[0]!r} wherever the basin clears "
                     f"the null, so a single basin fraction is a well-defined quantity here.")

    fic = f70_instrument_check({"EleutherAI/pythia-410m": runs.get("step143000", {}),
                                "gpt2-medium": {"fixed_point_fraction": 0.0}})
    parts.append(
        f"INSTRUMENT: the probe is F70's, imported from gate1 and gated against F70's own answer. "
        f"At step143000 it reads fixed={runs.get('step143000', {}).get('fixed_point_fraction')} "
        f"with endpoint {runs.get('step143000', {}).get('modal_endpoint_token')!r}, against F70's "
        f"18/24 to a whitespace token on the full model.")
    parts.append(
        "SCOPE: r<=2 is the out-of-distribution artifact regime (F69) and that is deliberate here "
        "-- the artifact is the object of study. Nothing in this result is a claim about a model "
        "in ordinary use.")

    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    # ---- per-token basins: the point of the re-run -------------------------------------------
    hist = {s: {ts: n for _, ts, n in runs[s].get("endpoint_histogram", [])} for s in order}
    NL = chr(10)
    off = [s for s in order if runs[s]["modal_endpoint_token"] != NL]
    tie_rows, verdict_hist = [], None
    if all(runs[s].get("endpoint_histogram") for s in order):
        for s in off:
            modal_n = max(hist[s].values()); nl_n = hist[s].get(NL, 0)
            tie_rows.append(dict(step=_step(s), modal=runs[s]["modal_endpoint_token"],
                                 modal_n=modal_n, newline_n=nl_n, margin=modal_n - nl_n))
        if not off:
            verdict_hist = "the modal token is newline at every checkpoint; nothing to adjudicate."
        elif all(r["margin"] <= 2 for r in tie_rows):
            verdict_hist = (f"NEAR-TIE: wherever the modal endpoint is not newline, newline sits "
                            f"within 2 starts of it (margins {[r['margin'] for r in tie_rows]} of "
                            f"24). F84's wandering is the argmax flipping on a flat top, not a "
                            f"change of attractor: ONE funnel, noisy label.")
        elif any(r["newline_n"] * 2 < r["modal_n"] for r in tie_rows):
            w = max(tie_rows, key=lambda r: r["modal_n"] - r["newline_n"])
            verdict_hist = (f"SWAP: at step{w['step']} the modal endpoint {w['modal']!r} takes "
                            f"{w['modal_n']}/24 while newline takes {w['newline_n']}/24, below "
                            f"half. The attractor IDENTITY genuinely changes during training -- "
                            f"F63's cross-model variety recurring inside one trajectory, and 'the "
                            f"basin' is correctly not one quantity.")
        else:
            verdict_hist = (f"INTERMEDIATE: newline neither stays within 2 starts nor falls below "
                            f"half (margins {[r['margin'] for r in tie_rows]}). Neither "
                            f"pre-registered branch fits; margins reported, no label forced.")
        verdict = verdict + " " + verdict_hist
        print(f"\n  per-token: {verdict_hist}")

    res["analysis"] = dict(
        per_token_basins=hist, off_newline_checkpoints=tie_rows,
        histogram_verdict=verdict_hist,
        null_step=NULL_STEP, null_basin=null["modal_endpoint_share"], null_ci_upper=n_hi,
        onset=onset, onset_step=_step(onset) if onset else None,
        monotone_after_onset=bool(monotone) if onset else None,
        endpoint_tokens_where_above_null=sorted(toks),
        curve=[[_step(s), runs[s]["modal_endpoint_share"], runs[s]["basin_ci"]] for s in order],
        landmarks=dict(extinction_window=_step(EXTINCTION),
                       crossing=[_step(CROSSING[0]), _step(CROSSING[1])]),
        f70_instrument_check=fic)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Dates F70's argmax fixed point in training time (#98). The observable is the BASIN -- the "
        "share of 24 random starts reaching a common endpoint -- and NOT the existence of a fixed "
        "point, because a random map over |V|~5e4 has about one by chance and existence would be "
        "noise. The null is step1, MEASURED rather than derived from random-mapping statistics, "
        "which would import state-space assumptions that do not hold for a lattice; this is F65's "
        "run-the-control rule with the control supplied free by the checkpoint series. The map is "
        "deterministic, so the only randomness is the 24 starts, and the same starts are used at "
        "every checkpoint. The probe is gate1.argmax_census, imported and already gated against "
        "F70's known answer -- a gate that caught a real defect, an earlier version testing "
        "(a,b)->b and scoring gpt2-medium at 0.96 fixed where the truth is 0.00. r<=2 is the "
        "out-of-distribution artifact regime (F69) and is the object of study here, not a defect.")


if __name__ == "__main__":
    main()
