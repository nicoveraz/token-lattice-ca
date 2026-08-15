"""Is F147's kind contrast a property of PROSE, or of ONE paragraph? M3a.

WHY THIS GATES EVERYTHING ELSE. F147's headline is that at matched token count the prose-vs-template
contrast flips sign between models -- gemma 1.000 vs 0.000, Falcon3 0.000 vs 0.839. But every model
saw the SAME prose: the first n tokens of gate1's CORPUS, one academic paragraph. gemma's n is NINE.
So the claim "nine tokens of ordinary prose build a funnel more perfect than the unconditioned model
has" currently rests on nine SPECIFIC tokens, drawn once. If other prose of the same length scatters
across the range, the correct statement is not "kind matters" but "different texts of the same length
do different things" -- a weaker and much duller claim that would REPLACE F147's contrast, not
qualify it. Nothing downstream in M3 can be read until this is settled.

THE SAMPLES ARE SELECTED BY OFFSET, NEVER BY CONTENT. Choosing texts that behave interestingly is
exactly the bias this run exists to rule out, so no sample is read or picked by hand:
  corpus{j}  CORPUS tokens [j*n : (j+1)*n] -- disjoint chunks of the SAME paragraph, same register
             as F147. corpus0 is bit-identically F147's text_matched prefix, which is the RUNG.
  shak{j}    data/shakespeare.txt at three FIXED fractional offsets, snapped forward to a line
             boundary, first n tokens. Different source and different register.
Two strata answer two different questions: corpus{j} asks "is it THAT text, within a register", and
shak{j} asks "does it survive leaving the register at all". A result that holds across both is much
harder to explain away than one that holds only within CORPUS.

PRE-REGISTERED:
  RUNG       corpus0 at census seed CENSUS_SEEDS[0] must reproduce domain_gradient's text_matched
             cell EXACTLY for the same model. Same prefix, same seed, same census -- any difference
             means this is not F147's measurement and nothing below is read.
  ANTI-VACUITY  a model whose prose values are ALL at the floor (< FLOOR) cannot distinguish "stable
             across samples" from "cannot move at all", and zero spread among floored values would
             read as a FALSE confirmation of stability. This is the mirror of the trap F147 fell
             into three times. Such models are excluded from the PRIMARY and named.
  PRIMARY    per readable model, the spread of fixed_point_fraction ACROSS prose samples, judged
             against two references: its own across-seed noise, and the prose-vs-template GAP that
             F147 reported for it. Registered readings:
               spread <= NOISE_FACTOR * seed_noise AND spread < 0.5 * gap
                 -> prose behaves as a CATEGORY at fixed length; F147's contrast survives, and the
                    sign-flip is about kind rather than about which paragraph was drawn.
               spread >= 0.5 * gap
                 -> F147's contrast is a SINGLE-DRAW artefact. The kind claim collapses to
                    "different texts of the same length produce different fixed-point structure",
                    which replaces it. F147 would need amending, not extending.
               otherwise -> NOT_DECIDABLE, both numbers reported.
  DECISIVE   gemma-1.1-2b-it is called out by name BEFORE the run: it carries F147's most extreme
             value (prose 1.000 at 9 tokens, both seeds, zero noise). If its samples scatter, the
             headline dies regardless of what the cohort average does.
  SECONDARY  the same analysis on n_distinct_endpoints -- and this time it is PRE-REGISTERED, with
             its threshold fixed here before any value is seen. F147's endpoint arm was exploratory
             because it was added after seeing the floor; this run is where that debt is paid.
  BOUNDARY   two prose sources, three chunks each at most, one length per model (that model's own
             template length). "Prose" remains two registers, not a survey of text.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census, CORPUS
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_instruct import PAIRS
from argmax_census_templated import _Prefixed, template_ids

OUT = str(_ROOT / "results" / "prose_samples.json")
GRAD = _ROOT / "results" / "domain_gradient.json"
SHAKE = _ROOT / "data" / "shakespeare.txt"

MODELS = [i for _b, i, _k in PAIRS]
N_CORPUS_CHUNKS = 3                      # disjoint chunks of CORPUS, as many as its length allows
SHAKE_OFFSETS = (0.10, 0.40, 0.70)       # fixed fractions of the file, chosen before looking at it
FLOOR = 4.0 / N_STARTS                   # below this a model cannot show spread
NOISE_FACTOR = 2.0
GAP_FRACTION = 0.5
DECISIVE = "google/gemma-1.1-2b-it"


def prose_samples(tok, n):
    """The prefixes for one model, keyed by sample name. Offset-selected, never content-selected."""
    out = {}
    if n <= 0:
        return out
    cids = tok(CORPUS, add_special_tokens=False)["input_ids"]
    for j in range(N_CORPUS_CHUNKS):
        chunk = cids[j * n:(j + 1) * n]
        if len(chunk) == n:
            out[f"corpus{j}"] = [int(t) for t in chunk]
    raw = SHAKE.read_text(encoding="utf-8", errors="replace")
    for j, frac in enumerate(SHAKE_OFFSETS):
        p = int(len(raw) * frac)
        nl = raw.find("\n", p)                       # snap forward to a line boundary
        seg = raw[(nl + 1 if nl != -1 else p):][:4000]
        ids = tok(seg, add_special_tokens=False)["input_ids"]
        if len(ids) >= n:
            out[f"shak{j}"] = [int(t) for t in ids[:n]]
    return out


def _mean(runs, m, s):
    v = [runs[f"{m}|s{cs}|{s}"]["fixed_point_fraction"] for cs in CENSUS_SEEDS]
    e = [runs[f"{m}|s{cs}|{s}"]["n_distinct_endpoints"] for cs in CENSUS_SEEDS]
    return float(np.mean(v)), float(abs(v[0] - v[1])), float(np.mean(e)), float(abs(e[0] - e[1]))


def analyse(res):
    runs, parts, analysis = res["runs"], [], {}
    grad = json.load(open(GRAD))["runs"] if GRAD.exists() else {}

    errs = []
    for m in MODELS:
        a = runs.get(f"{m}|s{CENSUS_SEEDS[0]}|corpus0")
        b = grad.get(f"{m}|s{CENSUS_SEEDS[0]}|text_matched")
        if a and b:
            errs.append((m, abs(a["fixed_point_fraction"] - b["fixed_point_fraction"])))
    worst = max((e for _m, e in errs), default=float("inf"))
    ok = bool(errs) and worst == 0.0
    parts.append(
        f"RUNG (corpus0 reproduces domain_gradient's text_matched): {len(errs)} models compared, "
        f"worst error {worst:.2e}. "
        + ("Identical, so this is F147's measurement with only the prose sample varying."
           if ok else "NOT reproduced -- this is not F147's measurement and nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    _kc = json.load(open(GRAD))["analysis"].get("kind_contrast", []) if GRAD.exists() else []
    gap_by_model = {r["model"]: abs(r["text"] - r["template"]) for r in _kc}
    tmpl_by_model = {r["model"]: r["template"] for r in _kc}
    rows = {}
    for m in MODELS:
        names = sorted({k.split("|")[2] for k in runs
                        if k.startswith(f"{m}|") and len(k.split("|")) == 3
                        and k.split("|")[2] not in ("rawcheck",)})
        have = [s for s in names
                if all(f"{m}|s{cs}|{s}" in runs for cs in CENSUS_SEEDS)]
        if len(have) < 3:
            continue
        vals, noise, eps, enoise = {}, {}, {}, {}
        for s in have:
            vals[s], noise[s], eps[s], enoise[s] = _mean(runs, m, s)
        gap = gap_by_model.get(m)
        tv = tmpl_by_model.get(m)
        # TWO DIFFERENT QUESTIONS, kept apart because they have different answers.
        #  (a) is F147's CONTRAST robust to which prose was drawn? -> how close does the CLOSEST
        #      prose sample come to the template value. This is what M3a was built to ask.
        #  (b) is the prose VALUE itself sample-independent? -> spread against seed noise.
        # Judging (a) by (b)'s yardstick demands the samples be identical to within census noise,
        # which is a far stronger claim than "the contrast survives", and conflating them produced a
        # verdict that contradicted its own PRIMARY.
        margin = None if tv is None else float(min(abs(v - tv) for v in vals.values()))
        rows[m] = dict(samples=have, fix={k: round(v, 4) for k, v in vals.items()},
                       endpoints={k: round(v, 1) for k, v in eps.items()},
                       seed_noise=round(float(np.mean(list(noise.values()))), 4),
                       endpoint_seed_noise=round(float(np.mean(list(enoise.values()))), 2),
                       spread=round(float(max(vals.values()) - min(vals.values())), 4),
                       endpoint_spread=round(float(max(eps.values()) - min(eps.values())), 1),
                       f147_gap=None if gap is None else round(gap, 4),
                       f147_template=None if tv is None else round(tv, 4),
                       worst_margin=None if margin is None else round(margin, 4))
    analysis["rows"] = rows
    if not rows:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " No model has three complete prose samples yet."
        return

    # TWO GATES, because (a) and (b) are vacuous under DIFFERENT conditions. Using one gate for both
    # excluded the model that answers (a) most decisively (Falcon3: prose floored, template 0.839,
    # so every sample is 0.83 clear of it) while admitting one whose entire F147 "contrast" was half
    # a census start (Qwen, gap 0.005) -- exactly backwards, and the R8 defect: the most informative
    # object discarded while a vacuous one is read.
    #  (a) needs the F147 GAP to be real; a contrast that never existed cannot be robust or fragile.
    #  (b) needs the prose values to be off the floor; zero spread among floored values would read
    #      as a FALSE confirmation of sample-independence.
    flat = [m for m, r in rows.items() if max(r["fix"].values()) < FLOOR]
    readable_b = [m for m in rows if m not in flat]
    no_gap = [m for m, r in rows.items() if r["f147_gap"] is None or r["f147_gap"] < FLOOR]
    readable = [m for m in rows if m not in no_gap]          # the (a) set
    analysis["excluded_flat"] = flat
    analysis["excluded_no_gap"] = no_gap
    parts.append(
        f"ANTI-VACUITY, applied SEPARATELY to the two questions. For (a): {len(no_gap)} of "
        f"{len(rows)} models had an F147 prose-vs-template gap below {FLOOR:.3f} (one census start "
        f"is {1.0/N_STARTS:.4f})"
        + (f" -- {[m.split('/')[-1] for m in no_gap]}. Excluded: a contrast that was never larger "
           f"than census noise cannot be shown robust OR fragile, and testing it either way would "
           f"be vacuous. "
           if no_gap else ", so every model has a real contrast to test. ")
        + f"For (b): {len(flat)} of {len(rows)} models have EVERY prose sample below the floor"
        + (f" -- {[m.split('/')[-1] for m in flat]}. Excluded there, because zero spread among "
           f"floored values cannot distinguish 'stable across samples' from 'cannot move'."
           if flat else ", so every model can show spread."))

    if not readable:
        parts.append(
            "PRIMARY (a) NOT_DECIDABLE: no model's F147 gap was larger than census noise, so there "
            "is no contrast whose robustness could be tested.")
        verdicts = {}
    else:
        verdicts = {}
        for m in readable:
            r = rows[m]
            gp, mg = r["f147_gap"], r["worst_margin"]
            if gp is None or mg is None:
                verdicts[m] = "no_f147_reference"
            elif mg >= GAP_FRACTION * gp:
                verdicts[m] = "contrast_robust"      # even the closest sample stays clear of it
            elif mg <= NOISE_FACTOR * max(r["seed_noise"], 1.0 / N_STARTS):
                verdicts[m] = "single_draw"          # some sample lands ON the template value
            else:
                verdicts[m] = "not_decidable"
        analysis["verdicts"] = verdicts
        n_stab = sum(1 for v in verdicts.values() if v == "contrast_robust")
        n_art = sum(1 for v in verdicts.values() if v == "single_draw")
        parts.append(
            "PRIMARY (a) -- is F147's CONTRAST robust to the draw? Per model, how close the CLOSEST "
            "prose sample comes to that model's template value: "
            + "; ".join(
                "{} closest sample is {} from template {} (F147 gap {}) -> {}".format(
                    m.split("/")[-1],
                    "n/a" if rows[m]["worst_margin"] is None else f"{rows[m]['worst_margin']:.3f}",
                    "n/a" if rows[m]["f147_template"] is None else f"{rows[m]['f147_template']:.3f}",
                    "n/a" if rows[m]["f147_gap"] is None else f"{rows[m]['f147_gap']:.3f}",
                    verdicts[m]) for m in readable)
            + f". Contrast robust on {n_stab}, single-draw on {n_art}, of {len(readable)} readable. "
            + ("At least one model has a prose sample landing ON its template value, so there the "
               "contrast is a SINGLE-DRAW artefact and F147 needs amending rather than extending."
               if n_art else
               "On every readable model, EVERY prose sample stays at least half the F147 gap away "
               "from the template value, so the contrast is about kind and not about which "
               "paragraph was drawn."))
    parts.append(
        "PRIMARY (b) -- is the prose VALUE itself sample-independent? A different and stronger "
        "question, on its own gate, and it has a different answer: "
        + ("; ".join("{} spread {:.3f} vs seed noise {:.3f} -> {}".format(
            m.split("/")[-1], rows[m]["spread"], rows[m]["seed_noise"],
            "sample-independent" if rows[m]["spread"] <= NOISE_FACTOR * max(
                rows[m]["seed_noise"], 1.0 / N_STARTS) else "TEXT-DEPENDENT")
            for m in readable_b)
           + ". Where a value is text-dependent, any claim quoting a single prose number must name "
             "its text -- even where (a) shows the contrast itself survives."
           if readable_b else
           "no model has prose values off the floor, so NOT_DECIDABLE -- a floored value cannot "
           "show text-dependence."))
    if DECISIVE in rows:
        r = rows[DECISIVE]
        parts.append(
            f"DECISIVE (named before the run), {DECISIVE.split('/')[-1]} carries F147's most "
            f"extreme value -- prose 1.000 at its own template length of "
            f"{runs[f'{DECISIVE}|s{CENSUS_SEEDS[0]}|corpus0']['n_prefix_tokens']} tokens. "
            f"Samples: "
            + ", ".join(f"{s}={r['fix'][s]:.3f}" for s in r["samples"])
            + f" (spread {r['spread']:.3f}; closest to its template value of "
            + ("n/a" if r["f147_template"] is None else f"{r['f147_template']:.3f}")
            + (" is n/a" if r["worst_margin"] is None else f" is {r['worst_margin']:.3f}")
            + "). "
            + (("Every sample stays clear of the template value, so the perfect prose funnel is "
                "NOT a property of those nine tokens and F147's contrast survives varying the "
                "text -- across registers, not just across chunks of one paragraph."
                if verdicts.get(DECISIVE) == "contrast_robust" else
                "A prose sample lands on the template value, so F147's 1.000 belongs to that "
                "one paragraph and the headline does not survive varying the text.")
               if DECISIVE in verdicts else
               "Excluded from PRIMARY (a) by anti-vacuity, so it cannot answer this."))
    else:
        parts.append(f"DECISIVE model {DECISIVE.split('/')[-1]} not complete yet.")

    ep_rows = {m: r for m, r in rows.items()}
    ep_stab = [m for m, r in ep_rows.items()
               if r["endpoint_spread"] <= NOISE_FACTOR * max(r["endpoint_seed_noise"], 1.0)]
    parts.append(
        "SECONDARY (PRE-REGISTERED here, unlike F147's exploratory endpoint arm -- this run is where "
        "that debt is paid), distinct greedy endpoints across the same prose samples, read for every "
        "model including the floored ones: "
        + "; ".join("{} [{}] spread {:.1f} vs seed noise {:.1f}".format(
            m.split("/")[-1], ", ".join(f"{s}:{r['endpoints'][s]:.1f}" for s in r["samples"]),
            r["endpoint_spread"], r["endpoint_seed_noise"]) for m, r in ep_rows.items())
        + f". Stable across samples on {len(ep_stab)} of {len(ep_rows)}. "
        + ("Endpoint counts do not depend on which prose was drawn."
           if len(ep_stab) == len(ep_rows) else
           "On the rest, which paragraph was drawn changes where the trajectories land, so endpoint "
           "claims at fixed length must name their text."))
    parts.append(
        f"BOUNDARY: {len(rows)} models, up to {N_CORPUS_CHUNKS} disjoint CORPUS chunks plus "
        f"{len(SHAKE_OFFSETS)} Shakespeare offsets, every sample selected by OFFSET and never by "
        f"content, {N_STARTS} starts, {len(CENSUS_SEEDS)} census seeds, one length per model (its "
        f"own template length). 'Prose' here is two registers, not a survey of text, and the "
        f"Shakespeare stratum changes source and register together.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=MODELS, n_corpus_chunks=N_CORPUS_CHUNKS, shake_offsets=list(SHAKE_OFFSETS),
        n_starts=N_STARTS, census_seeds=CENSUS_SEEDS, floor=FLOOR, noise_factor=NOISE_FACTOR,
        gap_fraction=GAP_FRACTION, decisive_model=DECISIVE,
        selection="samples chosen by OFFSET only -- disjoint CORPUS chunks and fixed fractional "
                  "positions in shakespeare.txt; no sample was read or picked by hand",
        rung="corpus0 must reproduce domain_gradient's text_matched cell exactly",
        primary="spread of fixed_point_fraction across prose samples vs seed noise and vs F147's "
                "prose-vs-template gap",
        secondary="the same on n_distinct_endpoints, PRE-REGISTERED (F147's endpoint arm was added "
                  "after seeing the floor and was exploratory; this pays that debt)",
        why="F147's kind contrast used ONE paragraph for every model, and gemma's share of it was "
            "NINE tokens; if other prose of the same length scatters, the contrast is a single draw")
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in MODELS:
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
            except Exception as e:
                print(f"  {m}: TOK FAILED ({type(e).__name__})", flush=True)
                continue
            tids, _txt = template_ids(tok)
            n = len(tids) if tids else 0
            want = [f"{m}|s{cs}|{s}" for s in prose_samples(tok, n) for cs in CENSUS_SEEDS]
            if want and all(k in res["runs"] for k in want):
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            try:
                model = AutoModelForCausalLM.from_pretrained(m).eval().to(
                    dev, torch.float16 if dev != "cpu" else torch.float32)
            except Exception as e:
                res["runs"][f"{m}|failed"] = dict(model=m, error=type(e).__name__)
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED ({type(e).__name__})", flush=True)
                continue
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            samples = prose_samples(tok, n)
            print(f"  {m:<40} n={n} tok, {len(samples)} samples: {sorted(samples)}", flush=True)
            for s, pre in samples.items():
                for cs in CENSUS_SEEDS:
                    k = f"{m}|s{cs}|{s}"
                    if k in res["runs"]:
                        continue
                    c = argmax_census(_Prefixed(model, pre), tok, dev, pool,
                                      np.random.default_rng(cs), n_starts=N_STARTS)
                    c.update(cls=classify(c), model=m, census_seed=cs, sample=s,
                             n_prefix_tokens=n)
                    res["runs"][k] = c
                    json.dump(res, open(OUT, "w"), indent=1)
                    print(f"  {m:<40} {s:<9} s={cs} cls={c['cls']:<11} "
                          f"fix={c['fixed_point_fraction']:.3f} ep={c['n_distinct_endpoints']}",
                          flush=True)
            done += 1
            print(f"  {m:<40} model done in {time.time()-t0:.0f}s", flush=True)
            del model
            gc.collect()
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
