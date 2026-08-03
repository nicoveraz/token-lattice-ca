"""What is the model doing during the extinction window? (#97)

THE WINDOW. F81 (#95) measured the dip directly and found the bottom moves earlier with width --
14m step512, 31m step128, 70m step64 -- with 31m and 70m reaching TRUE extinction (D_norm exactly
0.0000, 0/8 ignited), matching #88's 410m at step32. A model whose damage cannot propagate AT ALL
is in a categorically frozen regime, and nothing in the ledger says what it is doing there.

THE OBVIOUS HYPOTHESIS IS DEAD ALREADY, BY ARITHMETIC (#97's body). Induction heads form at ~step
1000 (arXiv:2502.14010, 7 Pythia models, checked under #70); the dip sits at step32-512. The
extinction window precedes it by one to two orders of magnitude.

THE LEADING CANDIDATE, AND IT IS TESTABLE WITH FORWARD PASSES. The earliest thing a language model
learns is the token frequency distribution. A conditional that has collapsed onto the MARGINAL is
nearly context-independent -- so flipping a neighbour changes nothing, damage dies, and D_norm -> 0.
That predicts the TV between p(x | k real tokens) and p(x | BOS) reaches its MINIMUM in the same
window the dip does.

F78 ALREADY SUPPLIES ONE OF THE FOUR POINTS. `context_onset.py` ran this measurement across
pythia-410m and found TV minimum at **step16** against #88's extinction at **step32** -- adjacent
sampled checkpoints. One model is a coincidence; the test is whether it tracks the dip ACROSS the
width ladder, where F81 showed the dip itself moves by a factor of 8.

    model   D_norm bottom (F81)   TV minimum
     14m         step512              ?
     31m         step128              ?
     70m         step64               ?
    410m         step32            step16      (F78, adjacent)

THE MEASUREMENT IS IMPORTED TWICE OVER: from `evidence_falloff` (the TV/top-1/entropy probe) and
via the same grid `context_onset.py` used, so the width series and the 410m series are directly
comparable. Forward passes only -- no ring, no damage runs.

REFERENCE IS READ, NOT HARDCODED. The D_norm minima come from the same two results files F81
analysed, so if those are ever re-run this script follows them.

PRE-REGISTERED:
  Primary.   Does the TV minimum coincide with the D_norm minimum WITHIN ONE SAMPLED CHECKPOINT,
             on each of the three width models? The Pythia grid is powers of two, so "within one
             step" is the finest resolution that exists (F81).
  Secondary. Does the TV minimum MOVE with width in the same direction the dip does? F81 has the
             dip moving 512 -> 128 -> 64. A TV minimum that tracks that ordering is far stronger
             than three independent coincidences, because the ordering spans a factor of 8.
  Null.      The TV minimum sits elsewhere, or does not move with width. Then conditional collapse
             does NOT explain the extinction window, the leading candidate is eliminated, and the
             window stays open with two candidates removed (induction heads by arithmetic, this by
             measurement). A NULL IS A GOOD RESULT.
  Kill.      If TV has no interior minimum for a model -- monotone across the whole grid -- the
             observable does not apply there and that model is reported as NOT DECIDABLE rather
             than having its argmin read off a monotone curve.
  Boundary.  Coincidence is CORRELATION. Even a clean four-model match shows the two events
             co-occur, not that conditional collapse CAUSES the extinction. F79/F80 closed the
             attribution route for lambda_ca and nothing here reopens it.

Writes results/context_onset_width.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/context_onset_width.py
        (safe to interrupt and re-run -- resumes, keyed by (model, checkpoint))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time, statistics
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from evidence_falloff import next_probs, top1_and_entropy, LENGTHS, N_CONTEXTS, CORPUS
from lyapunov import run_ignited

MODELS = ["EleutherAI/pythia-14m", "EleutherAI/pythia-31m", "EleutherAI/pythia-70m"]
STEPS = ["step1", "step2", "step4", "step8", "step16", "step32", "step64",
         "step128", "step256", "step512", "step1000", "step2000", "step4000"]
K_REPORT = 8
# F81's arms, read from the same files it analysed rather than hardcoded.
DIP_SOURCES = ["dev_transition_width_early.json", "dev_transition_width.json"]
OUT = str(_ROOT / "results" / "context_onset_width.json")


def dip_minima():
    """{model: (bottom_step, D_norm, ignited, n)} -- F81's measurement, recomputed from source."""
    import collections
    runs = []
    for f in DIP_SOURCES:
        p = _ROOT / "results" / f
        if p.exists():
            runs += list(json.loads(p.read_text())["runs"].values())
    by = collections.defaultdict(list)
    for v in runs:
        by[(v["model"], v["step"])].append(v)
    out = {}
    for m in {k[0] for k in by}:
        cells = [(s, statistics.median([r["D_norm"] for r in by[(m, s)]]),
                  sum(1 for r in by[(m, s)] if run_ignited(r)), len(by[(m, s)]))
                 for (mm, s) in by if mm == m for s in [s] if mm == m]
        cells = sorted({c[0]: c for c in cells}.values())
        if cells:
            out[m] = min(cells, key=lambda c: c[1])
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    dips = dip_minima()
    res["_preregistration"] = dict(
        models=MODELS, steps=STEPS, lengths=list(LENGTHS), n_contexts=N_CONTEXTS,
        k_report=K_REPORT, dip_reference={m: v[0] for m, v in dips.items()},
        dip_source=DIP_SOURCES,
        measurement="TV(p(x | k real tokens), p(x | BOS)) -- imported from evidence_falloff",
        primary="does the TV minimum coincide with the D_norm minimum WITHIN ONE SAMPLED "
                "checkpoint, per model? (powers of two is the finest resolution that exists)",
        secondary="does the TV minimum MOVE with width in the same direction the dip does "
                  "(F81: 512 -> 128 -> 64, a factor of 8)?",
        null="TV minimum elsewhere or not moving with width -> conditional collapse does NOT "
             "explain the extinction window. A NULL IS A GOOD RESULT; it eliminates the leading "
             "candidate after induction heads were eliminated by arithmetic",
        kill="no interior TV minimum for a model -> NOT DECIDABLE for that model rather than "
             "reading an argmin off a monotone curve",
        boundary="coincidence is CORRELATION; F79/F80 closed attribution for lambda_ca and "
                 "nothing here reopens it",
        prior_point="F78: pythia-410m TV minimum step16 against extinction step32 (adjacent)",
        resumable="keyed by (model, checkpoint)")
    runs = res["runs"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    todo = [(m, s) for m in MODELS for s in STEPS if f"{m}|{s}" not in runs]
    print(f"{len(runs)} cached, {len(todo)} cells\n", flush=True)
    print("  dip minima read from F81's sources:", {m.split('-')[-1]: v[0]
                                                    for m, v in dips.items()}, "\n", flush=True)

    for model in MODELS:
        need = [s for s in STEPS if f"{model}|{s}" not in runs]
        if not need:
            continue
        tok = AutoTokenizer.from_pretrained(model)
        ids = tok(CORPUS, return_tensors=None)["input_ids"]
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        for step in need:
            t0 = time.time()
            m = AutoModelForCausalLM.from_pretrained(model, revision=step).eval()
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            m = m.to(dev, torch.float16 if dev != "cpu" else torch.float32)
            marg = next_probs(m, [bos], dev)
            rec = dict(model=model, step=step, by_length={},
                       marginal_token=tok.decode([int(np.argmax(marg))]))
            rng = np.random.default_rng(0)
            for k in LENGTHS:
                starts = list(range(k, len(ids) - 1))
                if not starts:
                    continue
                pick = rng.choice(starts, size=min(N_CONTEXTS, len(starts)), replace=False)
                tvs, t1s = [], []
                for i in pick:
                    p = next_probs(m, ids[i - k:i], dev)
                    tvs.append(0.5 * float(np.abs(p - marg).sum()))
                    t1s.append(top1_and_entropy(p)[0])
                rec["by_length"][str(k)] = dict(
                    tv_to_marginal=round(float(np.mean(tvs)), 4),
                    top1=round(float(np.mean(t1s)), 4), n=len(pick))
            rec["secs"] = round(time.time() - t0, 1)
            runs[f"{model}|{step}"] = rec
            print(f"  {model.split('-')[-1]:>5} {step:>9}  "
                  f"TV@k={K_REPORT}={rec['by_length'][str(K_REPORT)]['tv_to_marginal']:.4f}  "
                  f"{rec['secs']:.0f}s", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            del m
            try: torch.mps.empty_cache()
            except Exception: pass
            gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    dips = dip_minima()
    out, parts = {}, []
    idx = {s: i for i, s in enumerate(STEPS)}

    print(f"\n  {'model':>5} {'TV min':>9} {'dip min':>9} {'gap':>5}  TV series (k=8)")
    for model in MODELS:
        series = {s: runs[f"{model}|{s}"]["by_length"][str(K_REPORT)]["tv_to_marginal"]
                  for s in STEPS if f"{model}|{s}" in runs}
        if len(series) < len(STEPS):
            out[model] = dict(complete=False, have=len(series))
            continue
        vals = [series[s] for s in STEPS]
        tv_min_step = min(STEPS, key=lambda s: series[s])
        interior = 0 < STEPS.index(tv_min_step) < len(STEPS) - 1
        dip_step = f"step{dips[model][0]}" if model in dips else None
        gap = (abs(idx[tv_min_step] - idx[dip_step]) if dip_step in idx else None)
        out[model] = dict(complete=True, tv_series=series, tv_min=tv_min_step,
                          tv_min_interior=interior, dip_min=dip_step, gap_in_steps=gap,
                          dip_D_norm=dips[model][1] if model in dips else None,
                          dip_ignited=f"{dips[model][2]}/{dips[model][3]}" if model in dips else None)
        print(f"  {model.split('-')[-1]:>5} {tv_min_step:>9} {str(dip_step):>9} "
              f"{('--' if gap is None else gap):>5}  "
              + " ".join(f"{v:.2f}" for v in vals))

    done = [m for m in MODELS if out.get(m, {}).get("complete")]
    if len(done) < len(MODELS):
        res["analysis"] = out
        res["verdict"] = f"INCOMPLETE -- {len(done)}/{len(MODELS)} models."
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}")
        return

    monotone_kill = [m for m in done if not out[m]["tv_min_interior"]]
    gaps = [out[m]["gap_in_steps"] for m in done if out[m]["gap_in_steps"] is not None]
    # `x or 99` would treat a gap of ZERO as missing (0 is falsy). No gap was 0 in this run so
    # the verdict was unaffected, but the idiom is wrong and it DID bite in
    # conditional_sensitivity.py -- fixed here too rather than left as a latent duplicate.
    coincide = [m for m in done if out[m]["gap_in_steps"] is not None
                and out[m]["gap_in_steps"] <= 1]
    tv_order = [STEPS.index(out[m]["tv_min"]) for m in MODELS if m in done]
    dip_order = [STEPS.index(out[m]["dip_min"]) for m in MODELS if m in done]
    tracks = tv_order == sorted(tv_order, reverse=True) and dip_order == sorted(dip_order,
                                                                               reverse=True)

    if monotone_kill:
        verdict = (f"NOT DECIDABLE for {', '.join(m.split('-')[-1] for m in monotone_kill)}: TV has "
                   f"no interior minimum, so its argmin sits on a grid edge and cannot be read as a "
                   f"feature. The observable does not apply for those models.")
    elif len(coincide) == len(done):
        verdict = (f"THE CONDITIONAL COLLAPSE CO-TIMES WITH THE EXTINCTION WINDOW on all "
                   f"{len(done)} width models: the TV minimum is within one sampled checkpoint of "
                   f"the D_norm minimum in every case (gaps {gaps}), and F78 already had "
                   f"pythia-410m at TV step16 against extinction step32. "
                   + (f"The TV minimum also MOVES with width in the same direction the dip does, "
                      f"which spans a factor of 8 -- three coincidences plus a shared ordering. "
                      if tracks else
                      f"The TV minimum does NOT track the width ordering, so the per-model "
                      f"coincidences are not supported by a shared trend. ")
                   + "BOUNDARY: this is CORRELATION. It does not show conditional collapse causes "
                     "the extinction, and F79/F80 closed the attribution route.")
    else:
        verdict = (f"NULL, AND IT IS A CLEAN ONE: the TV minimum coincides with the D_norm minimum "
                   f"on only {len(coincide)} of {len(done)} width models (gaps in sampled steps: "
                   f"{gaps}). Conditional collapse onto the marginal does NOT explain the "
                   f"extinction window. Two candidates are now eliminated -- induction heads by "
                   f"arithmetic (#70: ~step1000 against a dip at step32-512) and this by "
                   f"measurement -- and the window stays open.")

    print(f"\n  -> {verdict}")
    res["analysis"] = dict(per_model=out, coincide=coincide, gaps=gaps, tracks_width=tracks)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "#97. Does the conditional collapsing onto the marginal explain the extinction window F81 "
        "measured? TV(p(x|k), marginal) across the iso-LR width ladder, with the dip minima read "
        "from F81's own source files rather than hardcoded. The measurement is imported from "
        "evidence_falloff and uses the same grid context_onset.py used for 410m, so the width "
        "series and F78's 410m series are comparable. Forward passes only. Coincidence is "
        "correlation: F79/F80 closed attribution for lambda_ca and nothing here reopens it.")


if __name__ == "__main__":
    main()
