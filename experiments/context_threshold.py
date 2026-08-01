"""How many tokens of context before the CA stops emitting filler? (#91; refines F65/F66)

WHAT IS ALREADY KNOWN, AND WHAT IS NOT. F65 swept the conditioning window over r in {2, 4, 8, 16}
and found the single-token degeneracy at r=2, gone by r=4, and back at r=16 -- but the r=16 return
appears in the CONTROL too, so it is a generic long-context effect and not the family-distinguishing
phenomenon. F66 identified the mechanism: a two-token prompt is far outside anything a model trained
on thousands of tokens has seen, and a model handed almost no context falls back on its most common
filler token.

What that leaves open is the boundary. Between r=2 (degenerate) and r=4 (not) there is one
untested value, and the coarse grid cannot say whether the recovery is a sharp threshold or a
gradual climb. This runs r = 1, 2, 3, 4, 5, 6, 8, 12, 16.

WHY THE ANSWER IS WORTH HAVING
  * It is the **minimum viable context** for this construction, per model -- an operational number
    anyone reusing the iterated-resampling probe would need, and one nobody has measured.
  * If the recovery is SHARP (degenerate at r=2, fine at r=3), the failure is specific to the
    smallest possible window and easy to state as a caveat. If it is GRADUAL, every radius carries
    some contamination and the caveat is much broader -- including for the r=2 numbers the whole
    universality programme used.
  * The control tells us whether "recovery" is the model behaving well or merely the concentration
    metric losing power at larger windows.

MEASUREMENT: TWO KINDS, DELIBERATELY
  * **Quantitative** -- the concentration statistics the nine-model screen fixed: top-1 share and
    distinct-token fraction, with the same 40%/30% threshold. No new criterion is introduced.
  * **Qualitative** -- the settled ring is DECODED and stored verbatim at every cell. Concentration
    is a proxy for "is this degenerate"; the decoded text is the thing itself, and a reader can
    check the proxy against it rather than taking it on trust. Cheap to store and the most
    convincing exhibit the finding has.

PRE-REGISTERED:
  * Primary: the smallest r at which top-1 share falls below the threshold, per model and
    temperature. Reported as a number, not a narrative.
  * The r=16 rebound is expected to reappear in BOTH models (F65); if it appears only in the
    treatment this time, F65's control reading was wrong and must be revisited.
  * T=0.70 is included because it is the submitted paper's operating point. If the degeneracy is
    absent there at every radius, that is direct evidence the paper is clear of it, replacing an
    inference drawn from the newline share with a measurement.

Writes results/context_threshold.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/context_threshold.py
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

MODELS = [("EleutherAI/pythia-410m", "step143000", "treatment -- strongest attractor measured"),
          ("gpt2-medium",            None,         "control -- no attractor at r=2")]
RADII = [1, 2, 3, 4, 5, 6, 8, 12, 16]
TEMPS = [0.02, 0.436, 0.70]            # strongest effect; F58's T_c; the paper's operating point
N, B, SETTLE = 96, 8, 12
TOP1_HIGH, DISTINCT_LOW = 0.40, 0.30   # the screen's threshold, unchanged
SAMPLE_CHARS = 220
OUT = str(_ROOT / "results" / "context_threshold.json")


def composition(rule, T, r):
    """Concentration statistics AND the decoded ring, so the proxy can be checked against the text."""
    from ar_ca import run
    s = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme="none",
            init="random", seed=5, order="per_replica")["final"]
    distinct, top1, toks = [], [], collections.Counter()
    for row in s:
        c = collections.Counter(row.tolist())
        distinct.append(len(c) / N); top1.append(c.most_common(1)[0][1] / N); toks.update(c)
    tid, _ = toks.most_common(1)[0]
    return dict(distinct_frac=round(float(np.mean(distinct)), 4),
                top1_share=round(float(np.mean(top1)), 4),
                dominant_token=rule.tok.decode([tid]),
                has_attractor=bool(np.mean(top1) >= TOP1_HIGH
                                   and np.mean(distinct) <= DISTINCT_LOW),
                sample=rule.tok.decode(s[0].tolist())[:SAMPLE_CHARS])


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=[dict(name=m, role=ro) for m, _, ro in MODELS],
        radii=RADII, temps=TEMPS, N=N, B=B, settle=SETTLE,
        threshold=f"top-1 >= {TOP1_HIGH} AND distinct/N <= {DISTINCT_LOW} -- the screen's, unchanged",
        primary="the smallest r at which top-1 falls below threshold, per model and temperature",
        sharp_vs_gradual="a sharp recovery makes the caveat specific to r=2; a gradual one means "
                         "every radius carries contamination, including the r=2 numbers the "
                         "universality programme used",
        r16_expectation="the F65 rebound should appear in BOTH models; if only in the treatment, "
                        "F65's control reading was wrong and must be revisited",
        why_T070="the submitted paper's operating point -- absence there at every radius replaces "
                 "an inference from the newline share with a measurement",
        qualitative="the settled ring is decoded and stored verbatim, so the concentration proxy "
                    "can be checked against the text rather than trusted",
        resumable="keyed by (model, r, T)")
    runs = res["runs"]
    from ar_ca import ARRule

    for name, rev, role in MODELS:
        keys = [f"{name}|r{r}|T{T}" for r in RADII for T in TEMPS]
        if all(k in runs for k in keys):
            print(f"  {name}: already complete", flush=True); continue
        t0 = time.time()
        rule = ARRule(name, revision=rev) if rev else ARRule(name)
        print(f"\n  {name} ({role}) loaded in {time.time()-t0:.0f}s", flush=True)
        for T in TEMPS:
            print(f"  -- T={T} --", flush=True)
            for r in RADII:
                key = f"{name}|r{r}|T{T}"
                if key in runs: continue
                c = composition(rule, T, r)
                runs[key] = dict(model=name, role=role, r=r, T=T, **c)
                print(f"     r={r:<3} top1={c['top1_share']*100:>5.1f}%  "
                      f"distinct={c['distinct_frac']*100:>5.1f}%  "
                      f"dom={c['dominant_token']!r:<8} attr={str(c['has_attractor']):<5} "
                      f"| {c['sample'][:60]!r}", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    print(f"\n=== top-1 share vs conditioning radius ===")
    out = {}
    for name, _, role in MODELS:
        for T in TEMPS:
            pts = [(r, runs[f"{name}|r{r}|T{T}"]) for r in RADII if f"{name}|r{r}|T{T}" in runs]
            if not pts: continue
            line = "  ".join(f"r{r}:{v['top1_share']*100:.0f}%" for r, v in pts)
            first_clean = next((r for r, v in pts if not v["has_attractor"]), None)
            out[f"{name}|T{T}"] = dict(
                points=[[r, v["top1_share"], v["has_attractor"]] for r, v in pts],
                first_clean_radius=first_clean)
            print(f"  {name:>24} T={T:<6} {line}")
            print(f"  {'':>24}          smallest clean radius: {first_clean}")
        print()

    a, c = MODELS[0][0], MODELS[1][0]
    T0 = TEMPS[0]
    ta, tc = out.get(f"{a}|T{T0}"), out.get(f"{c}|T{T0}")
    parts = []
    if ta:
        fc = ta["first_clean_radius"]
        deg = [r for r, t, h in ta["points"] if h]
        if fc == 3 and deg == [1, 2]:
            parts.append(f"SHARP: {a} is degenerate at r=1 and r=2 and clean from r=3 onward. One "
                         f"extra token of context is the whole difference, so the caveat is "
                         f"specific to the smallest windows rather than general.")
        elif fc is not None:
            shape = ("sharp" if len(deg) <= 2 else
                     "GRADUAL, so every small radius carries contamination -- including the r=2 "
                     "the universality programme used")
            parts.append(f"The smallest clean radius for {a} at T={T0} is r={fc}; degenerate at "
                         f"r in {deg}. Recovery is {shape}.")
        else:
            parts.append(f"{a} is degenerate at EVERY radius tested at T={T0}, which contradicts "
                         f"F65's r=4 recovery and must be reconciled before either is used.")
    if ta and tc:
        rebound_a = [r for r, t, h in ta["points"] if h and r >= 12]
        rebound_c = [r for r, t, h in tc["points"] if h and r >= 12]
        if rebound_a and rebound_c:
            parts.append(f"The large-radius rebound appears in BOTH models (treatment r={rebound_a}, "
                         f"control r={rebound_c}), confirming F65: it is a generic long-context "
                         f"effect, not the family-distinguishing phenomenon.")
        elif rebound_a and not rebound_c:
            parts.append(f"The rebound appears ONLY in the treatment (r={rebound_a}), which "
                         f"contradicts F65's control reading. F65 must be revisited.")
    hi = out.get(f"{a}|T{TEMPS[-1]}")
    if hi:
        any_attr = any(h for _, _, h in hi["points"])
        parts.append(f"At the submitted paper's operating point T={TEMPS[-1]}, {a} shows "
                     + ("an attractor at some radius, which would need checking against the "
                        "paper's actual settings." if any_attr else
                        "NO attractor at any radius tested -- the paper is clear of this by "
                        "measurement rather than by inference from the newline share."))
    verdict = " ".join(parts) if parts else "insufficient data"
    print(f"  -> {verdict}")

    res["analysis"] = out
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Locates the boundary F65's coarse sweep (r in 2,4,8,16) left open: between r=2 "
        "(degenerate) and r=4 (not) sits one untested value, and the coarse grid cannot say "
        "whether recovery is a sharp threshold or a gradual climb. That distinction matters -- a "
        "sharp recovery confines the caveat to the smallest windows, a gradual one means every "
        "small radius carries contamination including the r=2 the universality programme used. "
        "Concentration uses the nine-model screen's threshold unchanged, and the settled ring is "
        "decoded verbatim so the proxy can be checked against the text. T=0.70 is included as the "
        "submitted paper's operating point, replacing an inference with a measurement.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
