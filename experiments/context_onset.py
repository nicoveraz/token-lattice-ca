"""When does the model start USING context, and does that explain the developmental transition?

THE GAP THIS ADDRESSES. `critical_analysis.md` §3: the flagship (F39/F46/F77) is a *when*, not a
*what*. lambda_ca crosses between step256 and step512 and nothing connects that to an independently
measurable internal event, so it is "a detector without an explanandum".

WHAT TODAY'S DATA ALREADY CONSTRAINS. F77 added two facts nobody has used:

  * the crossing bracket is RADIUS-INVARIANT -- step256->512 at r=2, 3 and 4 -- while the lambda
    LEVEL rises monotonically with r;
  * the plateau is FLAT TO THREE DECIMALS from step1000 to step143000, a 143x span of training in
    which the quantity does not move while the model's capabilities change enormously.

So the event is not about window size (its timing does not shift with r) but its magnitude scales
with how much context is visible, and it COMPLETES EARLY AND THEN STOPS. That is the profile of
"the model learns to use local context at all": a near-binary that finishes in the first billion
tokens. Deflationary, and testable with forward passes alone.

THE MEASUREMENT, IMPORTED NOT COPIED. `evidence_falloff.py` already computes, on real text, the
total-variation distance between p(x | k real tokens) and the model's marginal p(x | BOS). Small TV
means context is barely moving the model off its prior; large TV means it is being used. That
script runs the measurement ACROSS MODELS. This runs the identical measurement ACROSS CHECKPOINTS
of the model the flagship is measured on, reusing `next_probs`, `top1_and_entropy`, `LENGTHS`,
`N_CONTEXTS` and `CORPUS` so the two are directly comparable.

WHY TV AND NOT LOSS. Loss falls monotonically through training and would show an elbow wherever the
data happens to put one; #84 tracks that separately. TV to the marginal asks a sharper question with
a meaningful zero: it is ~0 for a model that ignores its input regardless of how good its loss is.

PRE-REGISTERED:
  Primary.    Does TV(p(x | k), marginal) rise across step256->512 -- the same bracket lambda_ca
              crosses at every radius (F77) -- rather than somewhere else?
  Secondary.  Does TV then SATURATE by step1000 and stay flat to step143000, matching lambda_ca's
              flat plateau? A shared saturation is a stronger signature than a shared onset,
              because it is a second coincidence in the same series.
  Null.       TV rises smoothly with no feature at 256->512, or its feature sits elsewhere. Then
              context-use onset does NOT explain the transition and the explanandum is still open.
              A NULL IS A GOOD RESULT: it eliminates the leading candidate cheaply.
  Kill.       If TV is already at its plateau value by step1 -- i.e. an untrained model's
              conditional is already far from its marginal -- the quantity is not measuring
              context USE and cannot answer the question. Report NOT DECIDABLE.
  Boundary.   Co-timing is CORRELATION. Even a perfect match does not show lambda_ca measures
              context use; it shows the two events coincide in one model. Only an ablation
              (route 3) would attribute one to the other, and this script must never be written up
              as though it had.

Writes results/context_onset.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/context_onset.py
        (safe to interrupt and re-run -- resumes, keyed by checkpoint)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
# The identical measurement, imported rather than copied, so this series and the across-model
# series in evidence_falloff.json are directly comparable.
from evidence_falloff import next_probs, top1_and_entropy, LENGTHS, N_CONTEXTS, CORPUS

BASE = "EleutherAI/pythia-410m"
# Spans all three regimes the ledger knows about: the extinction dip (step16-64, #88), the
# lambda_ca crossing (step256->512, F39/F77), and the flat plateau (step1000+).
STEPS = ["step1", "step8", "step16", "step32", "step64", "step128",
         "step256", "step512", "step1000", "step2000", "step8000", "step143000"]
CROSSING = ("step256", "step512")            # F39/F77, radius-invariant
PLATEAU = ["step1000", "step2000", "step8000", "step143000"]
K_REPORT = 8                                 # the length the headline reads; all of LENGTHS stored
OUT = str(_ROOT / "results" / "context_onset.json")


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        base=BASE, steps=STEPS, lengths=list(LENGTHS), n_contexts=N_CONTEXTS,
        crossing=list(CROSSING), plateau=PLATEAU, k_report=K_REPORT,
        measurement="TV(p(x | k real tokens), p(x | BOS)) -- imported from evidence_falloff",
        primary="does TV rise across step256->512, the bracket lambda_ca crosses at every radius?",
        secondary="does TV then saturate by step1000 and stay flat to step143000, like lambda_ca?",
        null="TV rises smoothly, or its feature sits elsewhere -> context-use onset does NOT "
             "explain the transition. A NULL IS A GOOD RESULT; it eliminates the leading "
             "candidate cheaply",
        kill="TV already at plateau by step1 -> not measuring context USE -> NOT DECIDABLE",
        boundary="CO-TIMING IS CORRELATION. A match does not show lambda_ca measures context use; "
                 "only an ablation would. This must never be written up as though it had",
        resumable="keyed by checkpoint")
    runs = res["runs"]
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(BASE)
    ids = tok(CORPUS, return_tensors=None)["input_ids"]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    print(f"corpus: {len(ids)} tokens; {len([s for s in STEPS if s not in runs])} checkpoints "
          f"to run\n", flush=True)

    for step in STEPS:
        if step in runs:
            continue
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(BASE, revision=step).eval()
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(dev, torch.float16 if dev != "cpu" else torch.float32)

        marg = next_probs(model, [bos], dev)
        m_top1, m_ent = top1_and_entropy(marg)
        rec = dict(step=step, n_tokens=int(str(step).replace("step", "")) * 1024 * 2048,
                   marginal_top1=round(m_top1, 4), marginal_entropy=round(m_ent, 4),
                   marginal_token=tok.decode([int(np.argmax(marg))]), by_length={})
        rng = np.random.default_rng(0)               # same contexts at every checkpoint
        for k in LENGTHS:
            starts = list(range(k, len(ids) - 1))
            if not starts:
                continue
            pick = rng.choice(starts, size=min(N_CONTEXTS, len(starts)), replace=False)
            t1s, ents, tvs = [], [], []
            for i in pick:
                p = next_probs(model, ids[i - k:i], dev)
                a, e = top1_and_entropy(p)
                t1s.append(a); ents.append(e)
                tvs.append(0.5 * float(np.abs(p - marg).sum()))
            rec["by_length"][str(k)] = dict(
                tv_to_marginal=round(float(np.mean(tvs)), 4),
                tv_sd=round(float(np.std(tvs)), 4),
                top1=round(float(np.mean(t1s)), 4),
                entropy=round(float(np.mean(ents)), 4), n=len(pick))
        rec["secs"] = round(time.time() - t0, 1)
        runs[step] = rec
        b = rec["by_length"].get(str(K_REPORT), {})
        print(f"  {step:>10}  TV@k={K_REPORT}={b.get('tv_to_marginal', float('nan')):.4f}  "
              f"top1={b.get('top1', float('nan')):.4f}  marg_top1={m_top1:.4f} "
              f"({rec['marginal_token']!r})  {rec['secs']:.0f}s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

        del model
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _tv(runs, step, k=K_REPORT):
    r = runs.get(step)
    return None if not r else r["by_length"].get(str(k), {}).get("tv_to_marginal")


def analyse(res):
    runs = res["runs"]
    have = [s for s in STEPS if s in runs]
    if len(have) < len(STEPS):
        res["analysis"] = dict(complete=False, have=len(have), need=len(STEPS))
        res["verdict"] = (f"INCOMPLETE -- {len(have)}/{len(STEPS)} checkpoints. Absence of data is "
                          f"not absence of effect; this file is a checkpoint, not a result.")
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}")
        return

    series = {s: _tv(runs, s) for s in STEPS}
    lo, hi = series[STEPS[0]], series[STEPS[-1]]
    span = (hi - lo) if (lo is not None and hi is not None) else None

    # step-to-step rises, so "where is the feature" is read off the data rather than assumed
    rises = [(STEPS[i], STEPS[i + 1], series[STEPS[i + 1]] - series[STEPS[i]])
             for i in range(len(STEPS) - 1)]
    biggest = max(rises, key=lambda x: x[2])
    at_crossing = next(d for a, b, d in rises if (a, b) == CROSSING)
    frac_at_crossing = (at_crossing / span) if span else None

    plat = [series[s] for s in PLATEAU]
    plateau_spread = max(plat) - min(plat)

    print(f"\n  {'step':>10} {'tokens':>9} {'TV@k=8':>8} {'rise':>8}")
    for i, s in enumerate(STEPS):
        rise = "" if i == 0 else f"{series[s]-series[STEPS[i-1]]:+8.4f}"
        print(f"  {s:>10} {runs[s]['n_tokens']/1e6:>8.0f}M {series[s]:>8.4f} {rise:>8}")

    kill = lo is not None and span is not None and span > 0 and (lo / max(hi, 1e-9)) > 0.9
    if kill:
        verdict = (f"NOT DECIDABLE: TV is already {lo:.4f} at {STEPS[0]} against {hi:.4f} at the "
                   f"end, so an untrained model's conditional is already essentially as far from "
                   f"its marginal as a trained one's. This quantity is not measuring context USE "
                   f"and cannot answer the question.")
    elif biggest[:2] == CROSSING:
        verdict = (f"THE ONSET CO-TIMES WITH THE TRANSITION: the largest single rise in TV@k="
                   f"{K_REPORT} is {CROSSING[0]}->{CROSSING[1]} ({at_crossing:+.4f}, "
                   f"{frac_at_crossing*100:.0f}% of the total span), which is the SAME bracket "
                   f"lambda_ca crosses at r=2, 3 and 4 (F77). Plateau spread "
                   f"{PLATEAU[0]}..{PLATEAU[-1]} is {plateau_spread:.4f}"
                   + (f", so TV saturates like lambda_ca does -- a second coincidence in the same "
                      f"series." if plateau_spread < 0.05 else
                      f", so TV does NOT saturate where lambda_ca does; the onset matches but the "
                      f"saturation does not.")
                   + " CO-TIMING IS CORRELATION: this does not show lambda_ca measures context "
                     "use, only that the two events coincide in one model. Attribution needs an "
                     "ablation.")
    else:
        verdict = (f"NULL, AND IT IS A CLEAN ONE: the largest rise in TV@k={K_REPORT} is "
                   f"{biggest[0]}->{biggest[1]} ({biggest[2]:+.4f}), NOT the "
                   f"{CROSSING[0]}->{CROSSING[1]} bracket where lambda_ca crosses at every radius "
                   f"(that bracket contributes {at_crossing:+.4f}, "
                   f"{(frac_at_crossing or 0)*100:.0f}% of the span). Context-use onset does not "
                   f"explain the developmental transition, and the leading candidate is "
                   f"eliminated at the cost of forward passes.")

    print(f"\n  -> {verdict}")
    res["analysis"] = dict(complete=True, tv_by_step=series, rises=[list(r) for r in rises],
                           biggest_rise=list(biggest), rise_at_crossing=at_crossing,
                           frac_of_span_at_crossing=frac_at_crossing,
                           plateau_spread=plateau_spread, span=span)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Route 1 of the explanandum programme (critical_analysis.md §3). Does the model's use of "
        "context -- TV between p(x | k real tokens) and its marginal -- turn on in the same "
        "bracket lambda_ca crosses? Measured across pythia-410m checkpoints with the measurement "
        "imported from evidence_falloff.py so the across-checkpoint and across-model series are "
        "comparable. Forward passes only: no ring dynamics, no damage runs. CO-TIMING IS "
        "CORRELATION and is not attribution; only an ablation would show lambda_ca measures this.")


if __name__ == "__main__":
    main()
