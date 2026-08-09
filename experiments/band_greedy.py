"""Close the disjoint-model-set gap: greedy degeneration on the BAND-SCREEN models.

THE GAP THIS EXISTS TO CLOSE. F117 found the attractor share is selective for compliance failures
(IFEval) over correctness failures (BBH/GPQA/MUSR/MMLU-PRO), with model size as a passing negative
control. But it could not test T*, rep_4 or distinct_1 at all, because **the band-screen models and
the degeneration models are disjoint sets** -- zero overlap. So the readouts that actually predict
something external (F86: T* vs rep_4 at rho = 0.833 family-level) are precisely the ones F117 could
not ask about.

This runs the greedy protocol on the band-screen models so the two halves finally share an axis.

THE PROTOCOL IS IMPORTED, NOT REIMPLEMENTED. `rep_stats`, `PROMPTS`, `NEW_TOKENS` and `NGRAM` come
from degeneration_vs_tstar unchanged, so the rep_4 values produced here are commensurable with the
26 models already measured there and with F86's anchor. A reimplementation would make the comparison
meaningless in exactly the way F98 avoided by importing dev_transition_phase3.measure.

T* IS FREE. The band screen already measured top1 at [0.02, 0.2, 0.436, 0.7] for these models, and
T* is defined as the interpolated crossing of the same threshold on the same ladder. No new lattice
runs: only greedy generation is new.

PRE-REGISTERED:
  PRIMARY    is T* ALSO compliance-selective? Same statistic as F117 -- |rho(T*, IFEval)| minus
             max|rho(T*, capability)| against a readout-permutation null -- so the two are directly
             comparable. F117 found the attractor share selective at p = 0.002 (T=0.7).
  SECOND     does rep_4 correlate with IFEval, and is it selective? If degeneration itself is the
             compliance-selective quantity, F117's attractor result is mediated by it and
             consolidates rather than extends.
  ANCHOR     rep_4 vs T* on these models must reproduce F86's positive relation. F86 got rho = 0.833
             at family level over 8 pairs; a contradiction here means the protocols have drifted
             despite the import and nothing else is read.
  COHORT     2 of the 10 band-screen models are GATED (gemma-2-2b, Llama-3.2-3B) and were evicted
             from the local cache; they may not load. gatecheck.cohort reports the realised set
             against the declared one -- a shrunken denominator is disclosed, never silently used.
  BOUNDARY   n <= 10, base models, greedy decoding, and the benchmark scores remain downloaded
             rather than measured.

Writes results/band_greedy.json.  Resumable per model.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from ranking import rank as _rank
from provenance import stamp, rel
from degeneration_vs_tstar import rep_stats, PROMPTS, NEW_TOKENS, NGRAM, THRESH
from gatecheck.cohort import cohort_complete

OUT = str(_ROOT / "results" / "band_greedy.json")
TEMPS = [0.02, 0.2, 0.436, 0.7]
COMPLIANCE, CORRECTNESS = ["IFEval"], ["BBH", "GPQA", "MUSR", "MMLU-PRO", "MATH Lvl 5"]
N_PERM = 20000


def band_models():
    cov = json.load(open(_ROOT / "results" / "band_benchmark_range.json"))["covered"]
    runs = json.load(open(_ROOT / "results" / "band_screen.json"))["runs"]
    prof = {}
    for v in runs.values():
        if v.get("arm") == "temp" and "top1" in v:
            prof.setdefault((v["model"], v["T"]), []).append(v["top1"])
    top1 = {k: float(np.mean(x)) for k, x in prof.items()}
    out = []
    for c in cov.values():
        m = c["model"]
        if all((m, T) in top1 for T in TEMPS):
            out.append(dict(model=m, params=c["params_b"], scores=c["scores"],
                            top1={str(T): top1[(m, T)] for T in TEMPS}))
    return out


def t_star_from(top1):
    """The screen's own definition: interpolated crossing of THRESH on the TEMPS ladder."""
    pts = [(T, top1[str(T)]) for T in TEMPS]
    for (a, ya), (b, yb) in zip(pts, pts[1:]):
        if ya >= THRESH > yb:
            return round(a + (b - a) * (ya - THRESH) / (ya - yb), 4)
    return "censored_above" if pts[-1][1] >= THRESH else None


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    band = band_models()
    res["_preregistration"] = dict(
        models=[b["model"] for b in band], temps=TEMPS, prompts=len(PROMPTS),
        new_tokens=NEW_TOKENS, ngram=NGRAM, thresh=THRESH, n_perm=N_PERM,
        protocol="rep_stats/PROMPTS/NEW_TOKENS/NGRAM imported UNCHANGED from "
                 "degeneration_vs_tstar, so rep_4 here is commensurable with its 26 models and "
                 "with F86's anchor",
        primary="is T* ALSO compliance-selective, by F117's exact statistic?",
        second="does rep_4 correlate with IFEval and is it selective? If so F117's attractor result "
               "is mediated by degeneration and consolidates rather than extends",
        anchor="rep_4 vs T* must reproduce F86's positive relation; a contradiction means the "
               "protocols drifted despite the import and nothing else is read",
        cohort="gemma-2-2b and Llama-3.2-3B are gated and were evicted; a shrunken denominator is "
               "disclosed via gatecheck.cohort, never silently used",
        boundary="n <= 10, base models, greedy decoding, benchmark scores downloaded not measured")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for b in band:
        m = b["model"]
        if m in res["cells"]:
            continue
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(m)
            mdl = AutoModelForCausalLM.from_pretrained(m).eval().to(
                dev, torch.float16 if dev != "cpu" else torch.float32)
            # rep_stats takes ONE continuation's flat token ids and is averaged ACROSS prompts by
            # the caller -- see degeneration_vs_tstar's loop, which this must match exactly or the
            # comparison to F86 is not commensurable. A first version passed the whole list of
            # continuations, so the 4-grams became tuples-of-lists and every model died with
            # `unhashable type: 'list'`. A per-prompt try/except mirrors F86's too: one bad prompt
            # must not discard a model, and a model with NO usable generation is recorded as failed
            # rather than averaged from an empty list.
            stats = []
            for p in PROMPTS:
                try:
                    ids = tok(p, return_tensors="pt").input_ids.to(dev)
                    with torch.no_grad():
                        g = mdl.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False,
                                         pad_token_id=tok.eos_token_id or 0)
                    s = rep_stats(g[0, ids.shape[1]:].tolist())
                    if s:
                        stats.append(s)
                except Exception as e:
                    print(f"    prompt failed: {type(e).__name__}: {str(e)[:60]}", flush=True)
            del mdl
            if not stats:
                raise RuntimeError("no usable generations")
            st = dict(n_prompts=len(stats),
                      rep_4=round(float(np.mean([s["rep_4"] for s in stats])), 4),
                      distinct_1=round(float(np.mean([s["distinct_1"] for s in stats])), 4),
                      longest_loop=round(float(np.mean([s["longest_loop"] for s in stats])), 2))
        except Exception as e:
            print(f"  {m}: FAILED {type(e).__name__}: {str(e)[:70]}"[:130], flush=True)
            res["cells"][m] = dict(model=m, failed=repr(e)[:200])
            json.dump(res, open(OUT, "w"), indent=1); continue
        res["cells"][m] = dict(model=m, params=b["params"], scores=b["scores"], top1=b["top1"],
                               t_star=t_star_from(b["top1"]), secs=round(time.time() - t0, 1), **st)
        c = res["cells"][m]
        print(f"  {m:<32} rep_4={c.get('rep_4'):.3f} distinct_1={c.get('distinct_1'):.3f} "
              f"T*={c['t_star']} ({c['secs']:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _rk(x): return _rank(np.asarray(x, float)).astype(float)
def _rho(a, b): return float(np.corrcoef(_rk(a), _rk(b))[0, 1])


def _selectivity(x, rows, seed=0):
    benches = COMPLIANCE + CORRECTNESS
    r = {b: _rho(x, [q["scores"][b] for q in rows]) for b in benches}
    obs = max(abs(r[b]) for b in COMPLIANCE) - max(abs(r[b]) for b in CORRECTNESS)
    g = np.random.default_rng(seed); xa = np.asarray(x, float); null = []
    for _ in range(N_PERM):
        xp = g.permutation(xa)
        rp = {b: _rho(xp, [q["scores"][b] for q in rows]) for b in benches}
        null.append(max(abs(rp[b]) for b in COMPLIANCE) - max(abs(rp[b]) for b in CORRECTNESS))
    return r, round(obs, 4), round(float(np.mean(np.array(null) >= obs - 1e-12)), 4)


def analyse(res):
    ok = [c for c in res["cells"].values() if c.get("rep_4") is not None]
    declared = res["_preregistration"]["models"]
    coh = cohort_complete(declared, [c["model"] for c in ok], unit="model")
    parts = [f"COHORT: {coh.reason}"]
    if len(ok) < 6:
        res["analysis"] = dict(cohort=coh.block(), n=len(ok))
        res["verdict"] = " ".join(parts) + f" Only {len(ok)} models -- NOT DECIDABLE."
        res["_analysis_provenance"] = stamp(__file__); print(f"\n  -> {res['verdict']}"); return
    fin = [c for c in ok if isinstance(c["t_star"], (int, float))]
    print(f"\n  {'model':<32} {'T*':>8} {'rep_4':>7} {'IFEval':>7}")
    for c in sorted(ok, key=lambda q: -q["rep_4"]):
        print(f"  {c['model']:<32} {str(c['t_star']):>8} {c['rep_4']:>7.3f} "
              f"{c['scores']['IFEval']:>7.2f}")
    if len(fin) >= 5:
        r_anchor = _rho([c["t_star"] for c in fin], [c["rep_4"] for c in fin])
        parts.append(
            f"ANCHOR: rho(T*, rep_4) = {r_anchor:+.3f} over {len(fin)} models with a finite T*, "
            f"against F86's +0.833 at family level. "
            + ("Same sign, so the imported protocol reproduces the anchor and the comparison is "
               "commensurable." if r_anchor > 0 else
               "OPPOSITE SIGN to F86 -- the protocols have drifted despite the import and nothing "
               "below is read."))
        if r_anchor <= 0:
            res["analysis"] = dict(cohort=coh.block(), anchor=r_anchor, n=len(ok))
            res["verdict"] = " ".join(parts); res["_analysis_provenance"] = stamp(__file__)
            print(f"\n  -> {res['verdict']}"); return
    out = {}
    for name, vals, rows_ in (("T*", [c["t_star"] for c in fin], fin),
                              ("rep_4", [c["rep_4"] for c in ok], ok),
                              ("distinct_1", [c["distinct_1"] for c in ok], ok),
                              ("top1@0.7", [c["top1"]["0.7"] for c in ok], ok)):
        if len(rows_) < 6:
            continue
        r, s_, p_ = _selectivity(vals, rows_, seed=abs(hash(name)) % 1000)
        out[name] = dict(rhos={k: round(v, 4) for k, v in r.items()}, selectivity=s_, perm_p=p_,
                         n=len(rows_))
        print(f"  {name:<12} IFEval rho={r['IFEval']:+.3f}  selectivity={s_:+.3f}  p={p_:.4f} "
              f"(n={len(rows_)})")
    hits = [k for k, v in out.items() if v["perm_p"] < 0.05]
    ts = out.get("T*")
    parts.append(
        f"PRIMARY: T* selectivity = {ts['selectivity'] if ts else 'n/a'} at p = "
        f"{ts['perm_p'] if ts else 'n/a'} (n = {ts['n'] if ts else 0}), against the attractor "
        f"share's +0.55 at p = 0.002 (F117, T=0.7). "
        + ("T* IS also compliance-selective, so the selectivity is a property of the attractor "
           "FAMILY of readouts rather than of one statistic -- and it now attaches to the one "
           "reading that already predicts something external (F86)."
           if ts and ts["perm_p"] < 0.05 else
           "T* is NOT compliance-selective on this evidence. The selectivity F117 found does not "
           "extend to the project's externally-predictive reading, which bounds it considerably: "
           "the attractor SHARE is selective, the melting TEMPERATURE is not."))
    rp4 = out.get("rep_4")
    if rp4:
        parts.append(
            f"SECOND: rep_4 vs IFEval = {rp4['rhos']['IFEval']:+.3f}, selectivity "
            f"{rp4['selectivity']:+.3f} at p = {rp4['perm_p']:.4f}. "
            + ("Degeneration is itself compliance-selective, so F117's attractor result is "
               "plausibly MEDIATED by it -- consolidating the two rather than extending them."
               if rp4["perm_p"] < 0.05 else
               "Degeneration is not compliance-selective, so F117's attractor result is not "
               "mediated by it and the two are separate."))
    parts.append(
        f"BOUNDARY: n <= 10, base models, greedy decoding, and benchmark scores remain downloaded "
        f"from the Open LLM Leaderboard rather than measured here. Selective readouts: "
        f"{hits or 'none'}.")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(cohort=coh.block(), selectivity=out, hits=hits, n=len(ok))
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Closes the disjoint-model-set gap that blocked F117 from testing T*, rep_4 and "
                    "distinct_1. Greedy protocol imported unchanged from degeneration_vs_tstar; T* "
                    "interpolated from the band screen's existing four-temperature ladder.")


if __name__ == "__main__":
    main()
