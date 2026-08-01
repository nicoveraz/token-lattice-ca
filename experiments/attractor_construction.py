"""Is the degeneracy the CA, or the model? Compare CA constructions (tests F65).

WHAT F65 ESTABLISHED, AND THE HOLE IT LEAVES. The frozen phase that the whole universality
programme rests on exists only at r=2 and is carried by a single token: banning `'\\n'` takes
pythia-410m from 74% to 15%, the control's baseline, without relocating. So it is a property of
the construction rather than of language-model dynamics.

But "the construction" was never varied. Every number in this project comes from ONE rule: an
autoregressive model asked for p(x_i | x_{i-r..i-1}) with r=2. That is a **two-token prompt**, and
it is severely out of distribution for a model trained on thousands of tokens of context. Falling
back on the highest-frequency filler token is precisely what an OOD prompt should produce. If that
is the mechanism, the degeneracy should weaken or vanish under a rule the model is actually
trained for.

THREE CONSTRUCTIONS, SAME MEASUREMENT:

  ar-none   p(x_i | x_{i-2}, x_{i-1}), no prefix     the project's rule; the baseline
  ar-bos    p(x_i | BOS, x_{i-2}, x_{i-1})           one token less out of distribution. `ARRule`
                                                     already implements this as scheme="bos"; it
                                                     has simply never been used for this question
  mlm       p(x_i | x_{i-r..i+r}, centre masked)     a masked LM doing its NATIVE task. Symmetric
                                                     context, and infilling a masked centre is
                                                     exactly what BERT was trained on

The MLM arm is the real test. It is not a variation on the AR rule -- it is a different CA, with
bidirectional context, on models whose training objective *is* this operation. If the single-token
degeneracy is an artifact of asking an AR model to work from two tokens, it should be absent here.
If it appears anyway, the degeneracy is a property of iterated resampling on token lattices in
general, which is a much broader and more interesting statement.

PRE-REGISTERED:
  * Primary: does the MLM construction show a single-token attractor at low T, by the same
    threshold the nine-model screen fixed (top-1 >= 40% AND distinct/N <= 30%)?
      - no  -> the degeneracy belongs to the AR two-token rule. Phase 3's MLM results, which used
               this construction, are on sounder ground than the AR universality programme, and a
               second paper should be built on the MLM path.
      - yes -> it is a property of iterated resampling itself, not of the AR prompt being short.
               That would make F65's "artifact" reading broader, not narrower.
  * Secondary: does `ar-bos` differ from `ar-none`? A single BOS token is a small step toward the
    training distribution; a large change from it would implicate OOD-ness directly.
  * The MLM arm reports its dominant token decoded, so "is it whitespace/filler again?" is
    answerable rather than assumed. BERT's tokenizer has no newline token, which is itself
    informative: if the attractor needs a newline, it cannot form here.

NOTE ON COMPARABILITY. The AR and MLM arms use different tokenizers, vocabularies and models, so
the top-1 *shares* are not directly comparable as numbers. What is comparable is the binary the
screen already uses -- does a single token dominate the settled lattice -- which is exactly the
question F65 answered for the AR rule.

Writes results/attractor_construction.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/attractor_construction.py
        (safe to interrupt and re-run -- it resumes)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time, collections
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel

# (arm, backend, model, revision, scheme, role)
ARMS = [
    ("ar-none",  "ar",  "EleutherAI/pythia-410m", "step143000", "none",    "baseline (F65: 74%)"),
    ("ar-bos",   "ar",  "EleutherAI/pythia-410m", "step143000", "bos",     "one token less OOD"),
    ("ar-none",  "ar",  "gpt2-medium",            None,         "none",    "control (F65: 15%)"),
    ("mlm",      "mlm", "bert-base-uncased",      None,         "cls_sep", "NATIVE task"),
    ("mlm",      "mlm", "prajjwal1/bert-medium",  None,         "cls_sep", "NATIVE task, smaller"),
]
RADII = [2, 4]
TEMPS = [0.02, 0.436, 0.70]            # lowest; F58's T_c; the submitted paper's
N, B, SETTLE = 96, 8, 12
TOP1_HIGH, DISTINCT_LOW = 0.40, 0.30   # the screen's threshold, unchanged
OUT = str(_ROOT / "results" / "attractor_construction.json")


def composition(backend, rule, T, r, scheme):
    """Settled-lattice composition under whichever CA this arm uses."""
    run = (__import__("ar_ca").run if backend == "ar" else __import__("mlm_ca").run)
    s = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme=scheme,
            init="random", seed=5, order="per_replica")["final"]
    distinct, top1, toks = [], [], collections.Counter()
    for row in s:
        c = collections.Counter(row.tolist())
        distinct.append(len(c) / N)
        top1.append(c.most_common(1)[0][1] / N)
        toks.update(c)
    tid, _ = toks.most_common(1)[0]
    return dict(distinct_frac=round(float(np.mean(distinct)), 4),
                top1_share=round(float(np.mean(top1)), 4),
                dominant_token=rule.tok.decode([tid]),
                has_attractor=bool(np.mean(top1) >= TOP1_HIGH
                                   and np.mean(distinct) <= DISTINCT_LOW))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        arms=[dict(arm=a, model=m, scheme=sc, role=ro) for a, _, m, _, sc, ro in ARMS],
        radii=RADII, temps=TEMPS, N=N, B=B, settle=SETTLE,
        threshold=f"top-1 >= {TOP1_HIGH} AND distinct/N <= {DISTINCT_LOW} -- the screen's, unchanged",
        primary="does the MLM construction (native masked-centre infilling) show a single-token "
                "attractor? no -> the degeneracy belongs to the AR two-token rule; yes -> it is a "
                "property of iterated resampling on token lattices generally",
        secondary="does ar-bos differ from ar-none? a large change implicates OOD-ness directly",
        comparability="AR and MLM arms use different tokenizers and models, so top-1 SHARES are "
                      "not comparable as numbers; the binary (does one token dominate) is",
        note="BERT's tokenizer has no newline token -- if the attractor requires one, it cannot "
             "form in the MLM arm, and that is itself the answer",
        resumable="keyed by (arm, model, r, T)")
    runs = res["runs"]

    for arm, backend, model, rev, scheme, role in ARMS:
        keys = [f"{arm}|{model}|r{r}|T{T}" for r in RADII for T in TEMPS]
        if all(k in runs for k in keys):
            print(f"  {arm}/{model}: already complete", flush=True); continue
        t0 = time.time()
        if backend == "ar":
            from ar_ca import ARRule
            rule = ARRule(model, revision=rev) if rev else ARRule(model)
        else:
            from mlm_ca import MLMRule
            rule = MLMRule(model)
        print(f"\n  {arm} / {model} ({role}) loaded in {time.time()-t0:.0f}s", flush=True)
        for r in RADII:
            for T in TEMPS:
                key = f"{arm}|{model}|r{r}|T{T}"
                if key in runs:
                    continue
                try:
                    c = composition(backend, rule, T, r, scheme)
                except Exception as e:
                    print(f"     r={r} T={T}: FAILED ({type(e).__name__}: {str(e)[:70]})", flush=True)
                    runs[key] = dict(arm=arm, model=model, r=r, T=T,
                                     failed=f"{type(e).__name__}: {str(e)[:120]}")
                    json.dump(res, open(OUT, "w"), indent=1); continue
                runs[key] = dict(arm=arm, model=model, backend=backend, scheme=scheme,
                                 role=role, r=r, T=T, **c)
                print(f"     r={r} T={T:<6} distinct={c['distinct_frac']*100:>5.1f}%  "
                      f"top1={c['top1_share']*100:>5.1f}%  dominant={c['dominant_token']!r}  "
                      f"attractor={c['has_attractor']}", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = {k: v for k, v in res["runs"].items() if "top1_share" in v}
    print(f"\n=== does the degeneracy survive a change of CA? (r=2, the project's window) ===")
    print(f"  {'arm':>9} {'model':>24} {'T':>6} {'distinct':>9} {'top1':>7} "
          f"{'dominant':>10} {'attractor':>10}")
    for T in TEMPS:
        for arm, _, model, _, _, _ in ARMS:
            v = runs.get(f"{arm}|{model}|r2|T{T}")
            if not v: continue
            print(f"  {arm:>9} {model:>24} {T:>6} {v['distinct_frac']*100:>8.1f}% "
                  f"{v['top1_share']*100:>6.1f}% {v['dominant_token']!r:>10} "
                  f"{str(v['has_attractor']):>10}")
        print()

    T0 = TEMPS[0]
    mlm = [v for k, v in runs.items() if v.get("arm") == "mlm" and v["r"] == 2 and v["T"] == T0]
    base = runs.get(f"ar-none|EleutherAI/pythia-410m|r2|T{T0}")
    bos = runs.get(f"ar-bos|EleutherAI/pythia-410m|r2|T{T0}")
    parts = []
    if mlm:
        any_attr = any(v["has_attractor"] for v in mlm)
        rng = f"{min(v['top1_share'] for v in mlm)*100:.1f}-{max(v['top1_share'] for v in mlm)*100:.1f}%"
        if not any_attr:
            parts.append(
                f"THE MLM CONSTRUCTION HAS NO SINGLE-TOKEN ATTRACTOR ({len(mlm)} models, top-1 "
                f"{rng}), against the AR rule's {base['top1_share']*100:.1f}% on the same "
                f"threshold. The degeneracy belongs to the AR two-token rule, not to iterated "
                f"resampling on token lattices. Phase 3's MLM results used this construction and "
                f"are not undermined by F65; a second paper should be built on the MLM path rather "
                f"than the AR universality programme.")
        else:
            parts.append(
                f"THE MLM CONSTRUCTION SHOWS IT TOO (top-1 {rng}, dominant "
                f"{[v['dominant_token'] for v in mlm]}). The degeneracy is a property of iterated "
                f"resampling on token lattices in general, not of the AR prompt being two tokens "
                f"long. F65's artifact reading gets BROADER: every construction in this project "
                f"drives the lattice into a single-token state at low temperature.")
    if base and bos:
        d = bos["top1_share"] - base["top1_share"]
        parts.append(f"ar-bos vs ar-none: {bos['top1_share']*100:.1f}% vs "
                     f"{base['top1_share']*100:.1f}% ({d*100:+.1f} points). "
                     + ("A single BOS token materially changes it, which implicates the prompt "
                        "being out of distribution." if abs(d) >= 0.15 else
                        "One BOS token does not rescue it, so the effect is not merely a missing "
                        "prefix."))
    verdict = " ".join(parts) if parts else "insufficient data"
    print(f"  -> {verdict}")

    res["analysis"] = {k: v for k, v in runs.items()}
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "F65 showed the frozen phase is carried by one token and exists only at r=2, making it a "
        "property of the construction. But the construction itself had never been varied: every "
        "number in this project comes from an AR model asked for p(x_i | x_{i-2}, x_{i-1}), a "
        "two-token prompt that is far out of distribution for a model trained on long contexts. "
        "This compares three CAs on the same measurement -- the AR rule, the AR rule with a BOS "
        "prefix, and the MASKED-LM rule, where symmetric masked-centre infilling is the model's "
        "native training objective. The MLM arm is the real test: if the degeneracy is an artifact "
        "of a short AR prompt it should be absent there, and if it appears anyway the artifact "
        "reading applies to iterated token-lattice resampling generally. BERT's tokenizer has no "
        "newline token, so an attractor requiring one cannot form in that arm.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
