"""Is the domain effect monotone in PREFIX LENGTH, or does the KIND of prefix matter? M2.

WHAT THREE RUNS HAVE SHOWN AND WHAT THEY CANNOT SETTLE. The domain moves every readout this project
has: a class (F144), a model ranking (F145, F146), a scalar value (F135). And the sizes do not line
up with the lengths:

    F146   1 BOS token        share shift 0.3103   on base models
    F145   ~35-token template share shift 0.1696   on instruct models
    F144   9-token template   class destroyed;  30-token template class destroyed;
           11-token template  class REINFORCED

One token doing more than thirty-five is not what a length effect looks like. But every comparison
so far confounds LENGTH with KIND (a special token vs structured chat markup vs prose) and with
COHORT. This holds the model and the estimator fixed and varies the domain along both axes.

FOUR DOMAINS, TWO OF THEM ALREADY MEASURED:

    raw             0 tokens                          from argmax_census_instruct   (have)
    bos             1 token, a SPECIAL token                                        (new)
    text_matched    ordinary prose, truncated to EXACTLY the model's template length (new)
    chat_template   the model's own template, 9-35 tokens                           (have)

THE CONTROL IS `text_matched`, and it is the point of the design. It has the same token count as
that model's chat template and none of its structure, so `text_matched` vs `chat_template` isolates
KIND at matched LENGTH, while raw -> bos -> matched traces LENGTH at roughly fixed kind. Without it,
"templates change the class" cannot be distinguished from "any prefix of that length changes it",
and the second is a much duller claim.

PRE-REGISTERED:
  RUNG      the raw arm must reproduce `argmax_census_instruct`'s stored class and
            fixed_point_fraction for every model, or this is not F144's measurement.
  PRIMARY   is `fixed_point_fraction` MONOTONE along raw -> bos -> text_matched -> chat_template,
            ordered by prefix length? Registered reading: monotone in most models means the domain
            effect is a length effect and can be reasoned about as one; non-monotone means the kind
            of prefix matters and the axis cannot be summarised by a token count.
  CONTRAST  `text_matched` vs `chat_template` at IDENTICAL length. A difference is KIND; no
            difference means the template's structure is doing nothing a prefix of prose would not.
  STABILITY both census seeds must agree on a model's class before that class is read, as F143/F144
            required.
  BOUNDARY  six instruction-tuned models, one prose sample, one template per model. "Kind" here is
            three points, not a taxonomy of prefixes.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census, CORPUS
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_instruct import PAIRS
from argmax_census_templated import _Prefixed, template_ids   # same wrapper, no shared-code edit

OUT = str(_ROOT / "results" / "domain_gradient.json")
RAW = _ROOT / "results" / "argmax_census_instruct.json"
TMPL = _ROOT / "results" / "argmax_census_templated.json"

MODELS = [i for _b, i, _k in PAIRS]
NEW_DOMAINS = ("bos", "text_matched")      # raw and chat_template are already measured

# ANTI-VACUITY floor for the PRIMARY. The census has N_STARTS trajectories, so the fixed-point
# fraction is quantised at 1/N_STARTS and a model that moves by one or two starts across the whole
# domain axis has not moved. Four starts is the smallest span worth calling a shape.
MIN_RANGE = 4.0 / N_STARTS
NOISE_FACTOR = 2.0


def domain_prefix(tok, kind, n_match):
    """The prefix ids for one domain. `n_match` is the model's own template length."""
    if kind == "bos":
        b = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        return ([int(b)], 1) if b is not None else (None, 0)
    if kind == "text_matched":
        # ordinary prose, truncated to EXACTLY the template's token count -- length matched, kind
        # different. gate1's CORPUS is reused so the text is not chosen for this experiment.
        ids = tok(CORPUS, add_special_tokens=False)["input_ids"]
        if len(ids) < n_match or n_match == 0:
            return None, 0
        return list(ids[:n_match]), n_match
    raise ValueError(kind)


def analyse(res):
    raw = json.load(open(RAW))["runs"] if RAW.exists() else {}
    tmpl = json.load(open(TMPL))["runs"] if TMPL.exists() else {}
    runs = res["runs"]
    parts, rows = [], {}

    checks, bad = 0, []
    for m in MODELS:
        k = f"{m}|s{CENSUS_SEEDS[0]}|rawcheck"
        r = raw.get(f"{m}|s{CENSUS_SEEDS[0]}")
        if k in runs and r:
            checks += 1
            for f in ("cls", "fixed_point_fraction"):
                if runs[k].get(f) != r.get(f):
                    bad.append(f"{m}:{f} {r.get(f)} -> {runs[k].get(f)}")
    ok = checks > 0 and not bad
    parts.append(
        f"RUNG (the raw arm reproduces argmax_census_instruct): {checks} models compared. "
        + ("Identical." if ok else f"MISMATCH {bad[:3]} -- nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, mismatches=bad)
        res["verdict"] = " ".join(parts)
        return

    for m in MODELS:
        pt = {}
        rk = [f"{m}|s{cs}" for cs in CENSUS_SEEDS]
        if all(k in raw for k in rk):
            pt["raw"] = dict(n=0, fix=[raw[k]["fixed_point_fraction"] for k in rk],
                             cls=[raw[k]["cls"] for k in rk],
                             ep=[raw[k]["n_distinct_endpoints"] for k in rk])
        for d in NEW_DOMAINS:
            ks = [f"{m}|s{cs}|{d}" for cs in CENSUS_SEEDS]
            if all(k in runs for k in ks):
                pt[d] = dict(n=runs[ks[0]].get("n_prefix_tokens"),
                             fix=[runs[k]["fixed_point_fraction"] for k in ks],
                             cls=[runs[k]["cls"] for k in ks],
                             ep=[runs[k]["n_distinct_endpoints"] for k in ks])
        tk = [f"{m}|s{cs}|tmpl" for cs in CENSUS_SEEDS]
        if all(k in tmpl for k in tk):
            pt["chat_template"] = dict(n=tmpl[tk[0]].get("n_prefix_tokens"),
                                       fix=[tmpl[k]["fixed_point_fraction"] for k in tk],
                                       cls=[tmpl[k]["cls"] for k in tk],
                                       ep=[tmpl[k]["n_distinct_endpoints"] for k in tk])
        if len(pt) == 4:
            rows[m] = pt
    if not rows:
        res["analysis"] = dict(rung_passes=True, rows={})
        res["verdict"] = " ".join(parts) + " No model has all four domains yet."
        return

    mono, unstable, flat = [], [], []
    for m, pt in rows.items():
        for d, v in pt.items():
            if v["cls"][0] != v["cls"][1]:
                unstable.append(f"{m.split('/')[-1]}/{d}")
        order = sorted(pt, key=lambda d: pt[d]["n"])
        seq = [float(np.mean(pt[d]["fix"])) for d in order]
        # ANTI-VACUITY, per model. A model whose fixed-point fraction barely moves across the whole
        # domain axis cannot be monotone OR non-monotone: the shape is census noise. Scoring it
        # either way is the defect this project keeps rediscovering -- a shape criterion applied to
        # a quantity with no room to vary. Such a model is EXCLUDED from the denominator and named.
        rng = float(max(seq) - min(seq))
        noise = max(float(abs(pt[d]["fix"][0] - pt[d]["fix"][1])) for d in order)
        if rng < max(MIN_RANGE, NOISE_FACTOR * noise):
            flat.append(dict(model=m, span=round(rng, 4), seed_noise=round(noise, 4), seq=seq))
            continue
        # Monotonicity is judged UP TO the census's own resolution. A step of one trajectory in 96
        # is not a reversal, and letting a single start flip the verdict would repeat the vacuity
        # defect one level up: the tolerance is the larger of the quantum and this model's noise.
        tol = max(1.0 / N_STARTS, noise)
        inc = all(b >= a - tol for a, b in zip(seq, seq[1:]))
        dec = all(b <= a + tol for a, b in zip(seq, seq[1:]))
        mono.append((m, order, seq, inc or dec, round(tol, 4)))
    n_mono = sum(1 for _m, _o, _s, ok, _t in mono if ok)
    analysis = dict(rung_passes=True, rows=rows, n_monotone=n_mono, n_models=len(mono),
                    n_with_all_domains=len(rows), unstable=unstable, excluded_flat=flat)
    flat_desc = "; ".join("{} (span {:.3f}, seed noise {:.3f})".format(
        x["model"].split("/")[-1], x["span"], x["seed_noise"]) for x in flat)
    parts.append(
        f"ANTI-VACUITY: {len(flat)} of {len(rows)} models move less than "
        f"max({MIN_RANGE:.3f}, {NOISE_FACTOR:g}x their own seed noise) across the ENTIRE domain axis"
        + (f" -- {flat_desc}. Excluded from the PRIMARY: a flat sequence is neither monotone nor "
           f"non-monotone, and scoring it either way would be vacuous."
           if flat else ", so every model below can carry a shape."))
    # SECONDARY, and it is read for EVERY model including the floor ones. fixed_point_fraction can
    # be pinned at zero while the map still reorganises: a model with no fixed points still lands
    # its 96 trajectories on some number of distinct endpoints, and that count is free to move.
    # Reading only the floored quantity would discard the floor cohort's entire signal.
    ep_rows, ep_movers = [], []
    for m, pt in rows.items():
        order = sorted(pt, key=lambda d: pt[d]["n"])
        seq = [float(np.mean(pt[d]["ep"])) for d in order]
        noise = max(float(abs(pt[d]["ep"][0] - pt[d]["ep"][1])) for d in order)
        span = float(max(seq) - min(seq))
        moved = span > NOISE_FACTOR * noise and span > 0
        ep_rows.append(dict(model=m, order=order, endpoints=[round(v, 1) for v in seq],
                            span=round(span, 1), seed_noise=round(noise, 1), moved=bool(moved)))
        if moved:
            ep_movers.append(m)
    analysis["endpoint_secondary"] = ep_rows
    parts.append(
        "SECONDARY (EXPLORATORY -- added mid-run, after the first two models showed "
        "fixed_point_fraction pinned at the floor, so it is NOT pre-registered and its threshold "
        "was chosen with the first values already visible; it generates a hypothesis, it does not "
        "test one), distinct greedy endpoints along the same axis, read for every model including "
        "those the PRIMARY excludes -- a model pinned at zero fixed points can still reorganise: "
        + "; ".join("{} [{}]".format(r["model"].split("/")[-1],
                                     ", ".join(f"{d}({rows[r['model']][d]['n']}):{v:.1f}"
                                               for d, v in zip(r["order"], r["endpoints"])))
                    for r in ep_rows)
        + f". Moves beyond seed noise on {len(ep_movers)} of {len(ep_rows)} models. "
        + ("So the domain reaches models whose fixed-point fraction cannot show it, and the floor "
           "cohort is not evidence of domain-invariance."
           if ep_movers else
           "No model's endpoint count moves beyond noise either, which is a stronger null than the "
           "PRIMARY alone: two independent census statistics both fail to see the domain."
           if len(ep_rows) >= 3 else
           f"But only {len(ep_rows)} model(s) are complete and the endpoint count's seed noise is "
           f"estimated from {len(CENSUS_SEEDS)} censuses, so this is NOT_DECIDABLE rather than a "
           f"null -- too few models and too coarse a noise estimate to call."))
    if not mono:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + (
            " NOT_DECIDABLE: no model varies enough in fixed_point_fraction for monotonicity to "
            "mean anything, so the PRIMARY is not read. This is a floor, not a null -- it bounds "
            "the cohort, not the domain effect, and the SECONDARY above is where the read lives.")
        return
    parts.append(
        f"STABILITY: {len(unstable)} of {len(rows) * 4} (model, domain) censuses disagree across "
        f"seeds" + (f" -- {unstable}, not read as classes." if unstable else ", so every class below "
                    "is stable."))
    parts.append(
        "PRIMARY, is fixed_point_fraction monotone in prefix LENGTH (judged up to each model's own "
        "tolerance, the larger of one census start and its seed noise)? "
        + "; ".join(f"{m.split('/')[-1]} [" + ", ".join(f"{d}({pt[d]['n']}):{np.mean(pt[d]['fix']):.3f}"
                                                        for d in sorted(pt, key=lambda d: pt[d]['n']))
                    + f"] tol={tol:.3f} {'monotone' if ok else 'NOT monotone'}"
                    for m, _o, _s, ok, tol in mono for pt in [rows[m]])
        + f". Monotone in {n_mono} of {len(mono)} models that can carry a shape "
          f"({len(flat)} excluded above). "
        # The question is EXISTENTIAL before it is proportional. One model that is decisively
        # non-monotone refutes 'the domain effect is a length effect' as a general law; the count
        # only says how common the exception is, and with a handful of models a 'majority' verdict
        # would be the weaker claim resting on the smaller n.
        + (f"At least one model is non-monotone beyond its own tolerance, so the domain effect is "
           f"NOT a length effect in general: prefix length does not order the fixed-point structure, "
           f"and a length-based correction would be wrong. How COMMON the exception is needs more "
           f"models -- {n_mono} of {len(mono)} monotone here is a count, not a rate."
           if n_mono < len(mono) else
           f"Every model that can carry a shape is monotone up to its own tolerance. On this cohort "
           f"the domain effect tracks prefix length, though {len(flat)} models could not be read and "
           f"a monotone sample of {len(mono)} does not establish monotonicity in general."))
    # CONTRAST, and it inherits the same floor problem. Two arms that both sit at zero do not
    # "agree": neither can move. On fixed_point_fraction the contrast is read ONLY for models that
    # cleared anti-vacuity; every model is then read again on ENDPOINTS, which are not floored.
    mono_models = {m for m, _o, _s, _ok, _t in mono}
    diffs = []
    for m, pt in rows.items():
        a, b = float(np.mean(pt["text_matched"]["fix"])), float(np.mean(pt["chat_template"]["fix"]))
        ea, eb = float(np.mean(pt["text_matched"]["ep"])), float(np.mean(pt["chat_template"]["ep"]))
        diffs.append((m, pt["text_matched"]["n"], a, b, abs(a - b), ea, eb, abs(ea - eb)))
    analysis["kind_contrast"] = [
        dict(model=m, n=n, text=round(a, 4), template=round(b, 4), diff=round(d, 4),
             text_endpoints=round(ea, 1), template_endpoints=round(eb, 1), endpoint_diff=round(ed, 1),
             fix_readable=m in mono_models) for m, n, a, b, d, ea, eb, ed in diffs]
    readable = [x for x in diffs if x[0] in mono_models]
    big = [x for x in readable if x[4] >= 0.10]
    parts.append(
        "CONTRAST at IDENTICAL length -- prose vs the model's own chat template. On "
        "fixed_point_fraction, read only for the models that cleared anti-vacuity (two arms both "
        "pinned at the floor cannot disagree): "
        + ("; ".join(f"{m.split('/')[-1]} ({n} tok) text {a:.3f} vs template {b:.3f}"
                     for m, n, a, b, _d, _ea, _eb, _ed in readable) + ". "
           + (f"They differ by >= 0.10 on {len(big)} of {len(readable)}, so the KIND of prefix "
              f"matters beyond its length."
              if big else
              f"None of the {len(readable)} differs by more than 0.10, so on this statistic prose "
              f"and chat markup do the same thing at the same length.")
           if readable else "no model clears the floor, so this statistic cannot answer it. ")
        + " On ENDPOINTS (the exploratory arm, same caveat as the SECONDARY), read for every model: "
        + "; ".join(f"{m.split('/')[-1]} ({n} tok) text {ea:.1f} vs template {eb:.1f}"
                    for m, n, _a, _b, _d, ea, eb, _ed in diffs) + ".")
    parts.append(
        f"BOUNDARY: {len(rows)} instruction-tuned models, ONE prose sample (gate1's CORPUS, reused "
        f"rather than chosen for this run), one template per model, {N_STARTS} starts, two census "
        f"seeds. 'Kind' is three points -- special token, prose, chat markup -- not a taxonomy of "
        f"prefixes, and prose truncated mid-sentence is its own oddity.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=MODELS, new_domains=list(NEW_DOMAINS),
        reused=["raw from argmax_census_instruct", "chat_template from argmax_census_templated"],
        n_starts=N_STARTS, census_seeds=CENSUS_SEEDS,
        rung="the raw arm must reproduce argmax_census_instruct's class and fixed_point_fraction",
        primary="is fixed_point_fraction monotone along raw -> bos -> text_matched -> template, "
                "ordered by prefix length",
        contrast="text_matched vs chat_template at IDENTICAL token count isolates KIND from LENGTH",
        why="F146's one BOS token moved the share further than F145's 35-token template, which is "
            "not what a length effect looks like -- but every comparison so far confounds length "
            "with kind and with cohort",
        amendments=[
            "ANTI-VACUITY gate added mid-run, BEFORE any model with room to vary had completed: the "
            "first two models' entire domain span was one census start (0.010) with seed noise "
            "0.021, and the original code scored that as 'not monotone'. Models whose span is below "
            "max(4/N_STARTS, 2x own seed noise) are now excluded from the PRIMARY and named. "
            "Monotonicity is likewise judged up to a per-model tolerance rather than exactly, so a "
            "one-trajectory step is not a reversal.",
            "ENDPOINT arm (n_distinct_endpoints) added mid-run, AFTER seeing fixed_point_fraction "
            "floored on two models. EXPLORATORY, not pre-registered: its threshold was chosen with "
            "the first values visible. It exists because reading only a floored statistic would "
            "have recorded the floor cohort as 'no signal'. Any claim resting on it needs its own "
            "pre-registered run.",
            "CONTRAST on fixed_point_fraction restricted to models that clear anti-vacuity: two "
            "arms both pinned at zero cannot disagree, so counting them as agreement was the same "
            "vacuity defect a third time."])
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in MODELS:
            want = [f"{m}|s{cs}|{d}" for cs in CENSUS_SEEDS for d in NEW_DOMAINS]
            if all(k in res["runs"] for k in want):
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
                model = AutoModelForCausalLM.from_pretrained(m).eval().to(
                    dev, torch.float16 if dev != "cpu" else torch.float32)
            except Exception as e:
                res["runs"][f"{m}|failed"] = dict(model=m, error=type(e).__name__)
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED ({type(e).__name__})", flush=True)
                continue
            tids, _txt = template_ids(tok)
            n_match = len(tids) if tids else 0
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            rk = f"{m}|s{CENSUS_SEEDS[0]}|rawcheck"
            if rk not in res["runs"]:
                c = argmax_census(model, tok, dev, pool,
                                  np.random.default_rng(CENSUS_SEEDS[0]), n_starts=N_STARTS)
                c["cls"] = classify(c); c["model"] = m
                res["runs"][rk] = c
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<40} RUNG raw cls={c['cls']:<11} fix={c['fixed_point_fraction']:.3f}",
                      flush=True)
            for d in NEW_DOMAINS:
                pre, n = domain_prefix(tok, d, n_match)
                if pre is None:
                    print(f"  {m}: {d} unavailable -- skipped", flush=True)
                    continue
                for cs in CENSUS_SEEDS:
                    k = f"{m}|s{cs}|{d}"
                    if k in res["runs"]:
                        continue
                    c = argmax_census(_Prefixed(model, pre), tok, dev, pool,
                                      np.random.default_rng(cs), n_starts=N_STARTS)
                    c.update(cls=classify(c), model=m, census_seed=cs, domain=d,
                             n_prefix_tokens=n)
                    res["runs"][k] = c
                    json.dump(res, open(OUT, "w"), indent=1)
                    print(f"  {m:<40} {d:<13} s={cs} cls={c['cls']:<11} "
                          f"fix={c['fixed_point_fraction']:.3f} ({n} tok)", flush=True)
            done += 1
            del model
            gc.collect()
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
