"""Does the conditional stop responding to its context exactly where damage goes extinct? (#97)

THE NARROWER HYPOTHESIS, after F82's null. #97 asked whether conditional collapse explains the
DIP. It does not: TV minima span step16-64 (4x) while dip minima span step32-512 (16x), so a single
mechanism cannot move both. But a sharper pattern survived that test:

    model   TV min   dip/extinct   gap   reaches TRUE extinction?
     410m   step16      step32      1    yes (0/8 ignited, #88)
      70m   step32      step64      1    yes (0/8, F81)
      31m   step64      step128     1    yes (0/8, F81)
      14m   step64      step512     3    NO  (4/8, D_norm 0.0066)

Every model that reaches TRUE extinction has its TV minimum exactly one sampled checkpoint before
it; the only exception is the one model that never extincts. So the coincidence may track
EXTINCTION rather than the dip -- a narrower claim, and one #97 did not pre-register.

TV-TO-MARGINAL IS THE WRONG OBSERVABLE FOR IT, AND THAT IS THIS FILE'S POINT. It is a proxy for
"does the model use context at all". The MECHANISM of extinction is narrower and directly
measurable: damage propagates in the ring only if resampling a site with a PERTURBED neighbourhood
yields a different token. If flipping one context token does not change what the model emits, damage
cannot spread -- that IS extinction, one step of it.

So measure that, not a proxy:

    argmax_flip_rate   P(argmax p(x|ctx') != argmax p(x|ctx))  for ctx' = ctx with ONE token
                       replaced by a random non-special token. THE DIRECT SINGLE-STEP ANALOGUE OF
                       DAMAGE PROPAGATION, and the quantity the CA's dynamics actually depend on at
                       low temperature, where sampling is essentially argmax (F70).
    tv_sensitivity     mean TV(p(x|ctx), p(x|ctx')). The distributional version, for models whose
                       argmax is stable while the distribution still moves.

Reference extinction/dip steps are READ from F81's own source files and #88's, never hardcoded.

PRE-REGISTERED:
  Primary.   For the three models that reach true extinction (31m, 70m, 410m), does
             `argmax_flip_rate` reach its MINIMUM at the extinction checkpoint, within one sampled
             step? The Pythia grid is powers of two, so one step is the finest resolution there is.
  Secondary. For 14m, which never extincts, is the minimum SHALLOWER -- i.e. does the observable
             separate extincting from non-extincting models rather than merely having a minimum
             everywhere? Report the minimum VALUE, not just its location.
  Null.      The minimum sits elsewhere, or 14m looks the same as the others. Then conditional
             insensitivity does not explain extinction either, THREE candidates are eliminated
             (induction heads by arithmetic, conditional collapse vs the dip by F82, and this),
             and the window closes as genuinely open. A NULL IS A GOOD RESULT.
  Kill.      If `argmax_flip_rate` is pinned near 0 or near 1 at every checkpoint for a model, it
             has no dynamic range there and that model is NOT DECIDABLE rather than having an
             argmin read off a flat curve.
  Boundary.  Coincidence is still CORRELATION. A perfect match would show the conditional stops
             responding exactly where damage stops spreading -- which is close to a mechanism, but
             the two are measured on the same forward pass and one does not cause the other by
             being adjacent to it. F79/F80 closed attribution for lambda_ca; this does not reopen it.

Writes results/conditional_sensitivity.json. Forward passes only.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/conditional_sensitivity.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time, statistics, collections
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from evidence_falloff import next_probs, CORPUS
from lyapunov import run_ignited

MODELS = ["EleutherAI/pythia-14m", "EleutherAI/pythia-31m",
          "EleutherAI/pythia-70m", "EleutherAI/pythia-410m"]
STEPS = ["step1", "step2", "step4", "step8", "step16", "step32", "step64",
         "step128", "step256", "step512", "step1000", "step2000", "step4000"]
CTX_LEN = 8                 # the k F78/F82 reported at, so the series are comparable
N_PAIRS = 200               # (ctx, ctx') pairs per checkpoint
DIP_SOURCES = ["dev_transition_width_early.json", "dev_transition_width.json",
               "dev_transition_410m_early.json"]
OUT = str(_ROOT / "results" / "conditional_sensitivity.json")


def dip_reference():
    """{model: dict(bottom_step, D_norm, ignited, n, extinct_step)} from the source files."""
    runs = []
    for f in DIP_SOURCES:
        p = _ROOT / "results" / f
        if not p.exists():
            continue
        for v in json.loads(p.read_text())["runs"].values():
            v = dict(v)
            v.setdefault("model", "EleutherAI/pythia-410m")     # 410m file records no model field
            runs.append(v)
    by = collections.defaultdict(list)
    for v in runs:
        by[(v["model"], v["step"])].append(v)
    out = {}
    for m in {k[0] for k in by}:
        cells = []
        for s in sorted({s for (mm, s) in by if mm == m}):
            rs = by[(m, s)]
            cells.append(dict(step=s, D_norm=statistics.median([r["D_norm"] for r in rs]),
                              ignited=sum(1 for r in rs if run_ignited(r)), n=len(rs)))
        bot = min(cells, key=lambda c: c["D_norm"])
        ext = [c["step"] for c in cells if c["ignited"] == 0]
        out[m] = dict(bottom_step=f"step{bot['step']}", D_norm=round(bot["D_norm"], 4),
                      ignited=f"{bot['ignited']}/{bot['n']}",
                      extinct_step=(f"step{ext[0]}" if ext else None))
    return out


@torch.no_grad()
def sensitivity(model, ids, pool, dev, rng):
    """Flip ONE context token; how often does the emitted token change, and how far does p move?"""
    flips, tvs = 0, []
    starts = list(range(CTX_LEN, len(ids) - 1))
    for _ in range(N_PAIRS):
        i = int(rng.choice(starts))
        ctx = list(ids[i - CTX_LEN:i])
        p0 = next_probs(model, ctx, dev)
        j = int(rng.integers(CTX_LEN))                       # which position to perturb
        alt = list(ctx)
        while True:                                          # a genuine flip, not a no-op
            t = int(rng.choice(pool))
            if t != ctx[j]:
                alt[j] = t
                break
        p1 = next_probs(model, alt, dev)
        flips += int(np.argmax(p0) != np.argmax(p1))
        tvs.append(0.5 * float(np.abs(p0 - p1).sum()))
    return flips / N_PAIRS, float(np.mean(tvs))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    ref = dip_reference()
    res["_preregistration"] = dict(
        models=MODELS, steps=STEPS, ctx_len=CTX_LEN, n_pairs=N_PAIRS,
        dip_reference={m: ref[m] for m in MODELS if m in ref}, dip_source=DIP_SOURCES,
        measurement="argmax_flip_rate = P(argmax changes | ONE context token replaced); the "
                    "direct single-step analogue of damage propagation. tv_sensitivity is the "
                    "distributional version",
        why_not_tv_to_marginal="F82 showed TV-to-marginal cannot explain the DIP (minima span 4x "
                               "against the dip's 16x). It is a proxy for 'uses context at all'; "
                               "this measures the quantity damage propagation actually depends on",
        primary="for the models reaching TRUE extinction, does argmax_flip_rate bottom AT the "
                "extinction checkpoint within one sampled step?",
        secondary="for 14m (never extincts), is the minimum SHALLOWER? report the VALUE not just "
                  "the location",
        null="minimum elsewhere, or 14m indistinguishable -> conditional insensitivity does not "
             "explain extinction either; three candidates eliminated. A NULL IS A GOOD RESULT",
        kill="argmax_flip_rate pinned near 0 or 1 across the whole grid for a model -> no dynamic "
             "range -> NOT DECIDABLE for that model",
        boundary="coincidence is CORRELATION; adjacency on the same forward pass is not causation, "
                 "and F79/F80 closed attribution for lambda_ca",
        resumable="keyed by (model, checkpoint)")
    runs = res["runs"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("  dip/extinction reference, read from source:")
    for m in MODELS:
        if m in ref:
            print(f"    {m.split('-')[-1]:>5}  bottom {ref[m]['bottom_step']:>8} "
                  f"D_norm={ref[m]['D_norm']:.4f} ign {ref[m]['ignited']}  "
                  f"extinct={ref[m]['extinct_step']}")
    print()

    for model in MODELS:
        need = [s for s in STEPS if f"{model}|{s}" not in runs]
        if not need:
            continue
        tok = AutoTokenizer.from_pretrained(model)
        ids = tok(CORPUS, return_tensors=None)["input_ids"]
        special = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                               tok.unk_token_id) if i is not None}
        for step in need:
            t0 = time.time()
            m = AutoModelForCausalLM.from_pretrained(model, revision=step).eval()
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            m = m.to(dev, torch.float16 if dev != "cpu" else torch.float32)
            V = m.get_output_embeddings().weight.shape[0]
            pool = np.array([i for i in range(min(V, len(tok))) if i not in special],
                            dtype=np.int64)
            rng = np.random.default_rng(0)          # same contexts and flips at every checkpoint
            flip, tv = sensitivity(m, ids, pool, dev, rng)
            runs[f"{model}|{step}"] = dict(model=model, step=step, ctx_len=CTX_LEN,
                                           n_pairs=N_PAIRS, argmax_flip_rate=round(flip, 4),
                                           tv_sensitivity=round(tv, 4),
                                           secs=round(time.time() - t0, 1))
            print(f"  {model.split('-')[-1]:>5} {step:>9}  flip={flip:.4f}  tv={tv:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            del m
            try: torch.mps.empty_cache()
            except Exception: pass
            gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs, ref = res["runs"], dip_reference()
    idx = {s: i for i, s in enumerate(STEPS)}
    out, rows = {}, []

    for model in MODELS:
        series = {s: runs[f"{model}|{s}"]["argmax_flip_rate"]
                  for s in STEPS if f"{model}|{s}" in runs}
        if len(series) < len(STEPS):
            out[model] = dict(complete=False, have=len(series))
            continue
        vals = list(series.values())
        lo = min(STEPS, key=lambda s: series[s])
        spread = max(vals) - min(vals)
        r = ref.get(model, {})
        target = r.get("extinct_step") or r.get("bottom_step")
        gap = abs(idx[lo] - idx[target]) if target in idx else None
        out[model] = dict(complete=True, series=series, flip_min_step=lo,
                          flip_min_value=series[lo], spread=round(spread, 4),
                          interior=0 < idx[lo] < len(STEPS) - 1,
                          target_step=target, extincts=bool(r.get("extinct_step")),
                          gap_in_steps=gap)
        rows.append((model, out[model]))

    if len(rows) < len(MODELS):
        res["analysis"] = out
        res["verdict"] = f"INCOMPLETE -- {len(rows)}/{len(MODELS)} models."
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}"); return

    print(f"\n  {'model':>6} {'flip min':>9} {'value':>7} {'target':>9} {'gap':>4} "
          f"{'spread':>7} {'extincts':>9}")
    for m, d in rows:
        print(f"  {m.split('-')[-1]:>6} {d['flip_min_step']:>9} {d['flip_min_value']:>7.4f} "
              f"{str(d['target_step']):>9} {str(d['gap_in_steps']):>4} {d['spread']:>7.4f} "
              f"{str(d['extincts']):>9}")

    dead = [m for m, d in rows if d["spread"] < 0.05 or not d["interior"]]
    ext = [(m, d) for m, d in rows if d["extincts"]]
    non = [(m, d) for m, d in rows if not d["extincts"]]
    # `x or 99` treats a gap of ZERO -- a PERFECT hit -- as missing, because 0 is falsy in
    # Python. That excluded pythia-70m, whose flip minimum lands exactly on its extinction
    # checkpoint, and turned "2 of 3" into "1 of 3" in the emitted verdict. Same failure family as
    # F74's denominator degeneracy: a guard clause firing on a legitimate extreme value.
    hit = [m for m, d in ext if d["gap_in_steps"] is not None and d["gap_in_steps"] <= 1]
    deeper = (statistics.fmean([d["flip_min_value"] for _, d in ext])
              < statistics.fmean([d["flip_min_value"] for _, d in non]) if ext and non else None)

    if dead:
        verdict = (f"NOT DECIDABLE for {', '.join(m.split('-')[-1] for m in dead)}: "
                   f"argmax_flip_rate has no interior minimum or under 0.05 of dynamic range, so "
                   f"its argmin cannot be read as a feature.")
    elif len(hit) == len(ext) and ext:
        verdict = (f"THE CONDITIONAL STOPS RESPONDING EXACTLY WHERE DAMAGE STOPS SPREADING: for "
                   f"all {len(ext)} models that reach TRUE extinction, argmax_flip_rate bottoms "
                   f"within one sampled checkpoint of the extinction step "
                   f"({', '.join(m.split('-')[-1] for m, _ in ext)}). "
                   + (f"And the non-extincting model (14m) has a SHALLOWER minimum "
                      f"({non[0][1]['flip_min_value']:.4f} against a mean of "
                      f"{statistics.fmean([d['flip_min_value'] for _, d in ext]):.4f}), so the "
                      f"observable separates the two classes rather than merely having a minimum "
                      f"everywhere. " if deeper else
                      f"But the non-extincting model's minimum is NOT shallower, so the observable "
                      f"does not separate extincting from non-extincting models and the "
                      f"coincidence is weaker than it looks. ")
                   + "BOUNDARY: this is CORRELATION -- both quantities come off the same forward "
                     "pass, and adjacency is not causation.")
    else:
        verdict = (f"NULL, AND IT IS A CLEAN ONE: argmax_flip_rate bottoms at the extinction "
                   f"checkpoint for only {len(hit)} of {len(ext)} extincting models. Conditional "
                   f"insensitivity does not explain extinction either. Three candidates are now "
                   f"eliminated -- induction heads by arithmetic (#70), conditional collapse "
                   f"against the dip (F82), and this against extinction -- and the window is "
                   f"genuinely open rather than merely unexamined.")

    print(f"\n  -> {verdict}")
    res["analysis"] = dict(per_model=out, extincting_hits=hit,
                           non_extincting_shallower=deeper)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "#97's narrower hypothesis, after F82. Measures the DIRECT single-step analogue of damage "
        "propagation -- P(argmax changes when one context token is flipped) -- rather than "
        "TV-to-marginal, which F82 showed cannot explain the dip. Reference extinction steps are "
        "read from F81's and #88's source files, never hardcoded. Forward passes only. Coincidence "
        "here is still correlation: both quantities come off the same forward pass.")


if __name__ == "__main__":
    main()
