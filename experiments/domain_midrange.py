"""The domain axis on models that can actually MOVE BOTH WAYS. F149's question, finally answerable.

WHY THIS RUN EXISTS. F149 asked whether the domain's direction is a property of the interaction or
just of where the model starts -- a model at raw 0.948 has nowhere to go but down, one at 0.000
nowhere but up. It could not answer: scoring the floor/ceiling baseline only where it COULD have been
wrong left one model and two units, because five of six models sat at an extreme. F150 then screened
for the missing instrument and found two models with genuine headroom in both directions:

    Qwen2.5-1.5B-Instruct   raw 0.573   funnel       zero seed noise across the screen's two seeds
    gemma-2-2b-it           raw 0.714   FRAGMENTED   seed noise 0.011

THE SECOND ONE MATTERS TWICE. Every model in F144-F150's domain work is `funnel` or `none`. The
taxonomy has four classes and the domain axis has never been run on a `fragmented` model, so F144's
"the class is a joint property of weights and domain" has only ever been tested on two of its four
classes. Whatever the PRIMARY returns, this is the first domain gradient on a fragmented model.

DELIBERATELY A NEW FILE. domain_gradient.py's model list is NOT edited: its provenance closure is
stamped into F147's and F148's stored results, and widening it in place would re-stamp and invalidate
them. The census, the wrapper and the prose sampler are IMPORTED unchanged so the quantity is
identical to F147's and no code drifts between the two runs.

PRE-REGISTERED:
  RUNG       the raw arm must reproduce midrange_screen's stored fixed_point_fraction EXACTLY at both
             census seeds for both models. Same census, same seeds -- any difference means this is
             not F150's measurement and nothing below is read.
  ANTI-VACUITY  by construction both models have room to move by more than tolerance in BOTH
             directions -- that is why they were screened. It is checked anyway rather than assumed:
             a model whose headroom on either side is below its own tolerance is excluded and named,
             because the PRIMARY would then be unanswerable for it exactly as in F149.
  PRIMARY    BIDIRECTIONALITY from a single raw value. Does either model move robustly UP under one
             domain and robustly DOWN under another, with weights and starting point held fixed and
             only the domain varying? Each shift is judged against ITS OWN seed noise (F149's lesson:
             a model-level tolerance hid that one prose sample was six times noisier than the rest).
             Registered readings:
               at least one bidirectional model -> the floor/ceiling account is REFUTED. Direction is
                 a joint property of weights and domain, and F147's model-specific direction is a
                 real interaction rather than a restatement of the raw value.
               neither -> the deflationary account SURVIVES, and now on models that COULD have
                 falsified it. That is a far stronger null than F149's untestable, and it would mean
                 the domain moves every model toward the same place regardless of where it started.
  SECONDARY  the kind contrast at matched length (F147's test) on two fresh models, and whether the
             FRAGMENTED class behaves like the funnels did.
  BOUNDARY   two models. The PRIMARY is an existence test either way -- it can refute a general
             floor/ceiling account or fail to, but two models cannot measure how common either is.
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
from gate1 import argmax_census
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_templated import _Prefixed, template_ids
from domain_gradient import domain_prefix           # unchanged, so the domains are F147's exactly
from prose_samples import prose_samples             # unchanged, so the texts are F148's exactly

OUT = str(_ROOT / "results" / "domain_midrange.json")
SCREEN = _ROOT / "results" / "midrange_screen.json"

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "google/gemma-2-2b-it"]
DOMAINS = ("bos", "text_matched")                   # chat_template is measured as its own arm below
MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0


def _pair(runs, m, key):
    ks = [f"{m}|s{cs}|{key}" for cs in CENSUS_SEEDS]
    if not all(k in runs for k in ks):
        return None
    v = [runs[k]["fixed_point_fraction"] for k in ks]
    return float(np.mean(v)), float(abs(v[0] - v[1])), [runs[k]["cls"] for k in ks]


def analyse(res):
    runs, parts, analysis = res["runs"], [], {}
    screen = json.load(open(SCREEN))["runs"] if SCREEN.exists() else {}

    errs = []
    for m in MODELS:
        for cs in CENSUS_SEEDS:
            a, b = runs.get(f"{m}|s{cs}|raw"), screen.get(f"{m}|s{cs}")
            if a and b:
                errs.append(abs(a["fixed_point_fraction"] - b["fixed_point_fraction"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst == 0.0
    parts.append(
        f"RUNG (raw arm reproduces midrange_screen): {len(errs)} cells compared, worst error "
        f"{worst:.2e}. "
        + ("Identical, so this is F150's measurement with only the domain varying."
           if ok else "NOT reproduced -- nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    rows = {}
    for m in MODELS:
        raw = _pair(runs, m, "raw")
        if raw is None:
            continue
        raw_v, raw_n, raw_cls = raw
        moves = {}
        for key in list(DOMAINS) + ["chat_template"]:
            p = _pair(runs, m, key)
            if p is None:
                continue
            v, n, cls = p
            tol = max(MIN_SHIFT, NOISE_FACTOR * max(n, raw_n))
            d = v - raw_v
            moves[key] = dict(value=round(v, 4), shift=round(d, 4), seed_noise=round(n, 4),
                              tol=round(tol, 4), cls=cls[0] if cls[0] == cls[1] else "UNSTABLE",
                              dir="up" if d > tol else ("down" if d < -tol else "flat"))
        # PROSE, per sample, each against its OWN noise -- F149's lesson.
        for s in sorted({k.split("|")[2] for k in runs
                         if k.startswith(f"{m}|") and k.split("|")[2].startswith(("corpus", "shak"))}):
            p = _pair(runs, m, s)
            if p is None:
                continue
            v, n, cls = p
            tol = max(MIN_SHIFT, NOISE_FACTOR * max(n, raw_n))
            d = v - raw_v
            moves[f"prose:{s}"] = dict(value=round(v, 4), shift=round(d, 4), seed_noise=round(n, 4),
                                       tol=round(tol, 4),
                                       cls=cls[0] if cls[0] == cls[1] else "UNSTABLE",
                                       dir="up" if d > tol else ("down" if d < -tol else "flat"))
        rows[m] = dict(raw=round(raw_v, 4), raw_seed_noise=round(raw_n, 4),
                       raw_cls=raw_cls[0] if raw_cls[0] == raw_cls[1] else "UNSTABLE", moves=moves)
    analysis["rows"] = rows
    if not rows:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " No model complete yet."
        return

    # ANTI-VACUITY: verified, not assumed.
    flat = []
    for m, r in rows.items():
        tol = max(MIN_SHIFT, NOISE_FACTOR * r["raw_seed_noise"])
        if min(r["raw"], 1.0 - r["raw"]) <= tol:
            flat.append((m, r["raw"], tol))
    readable = [m for m in rows if m not in {x[0] for x in flat}]
    parts.append(
        f"ANTI-VACUITY (checked, not assumed): {len(flat)} of {len(rows)} models lack room to move "
        f"both ways beyond tolerance"
        + (f" -- {[(m.split('/')[-1], round(v, 3), round(t, 3)) for m, v, t in flat]}, excluded."
           if flat else ", so both models can in principle show either direction."))

    bi = []
    for m in readable:
        ds = {v["dir"] for v in rows[m]["moves"].values()}
        if "up" in ds and "down" in ds:
            bi.append(m)
    analysis["bidirectional"] = bi
    parts.append(
        "PRIMARY, bidirectionality from a single raw value (weights and starting point FIXED, only "
        "the domain varying, each shift against its own seed noise): "
        + "; ".join(
            "{} raw {:.3f} [{}] -> ".format(m.split("/")[-1], rows[m]["raw"], rows[m]["raw_cls"])
            + ", ".join(f"{k} {v['shift']:+.3f} ({v['dir']}, tol {v['tol']:.3f}, {v['cls']})"
                        for k, v in rows[m]["moves"].items())
            for m in readable)
        + ". "
        + (f"{[m.split('/')[-1] for m in bi]} move BOTH ways from a single raw value, which no "
           f"floor/ceiling account can produce -- the starting point is identical for every arm. "
           f"The domain's direction is a JOINT property of weights and domain, and F147's "
           f"model-specific direction is a real interaction rather than a restatement of where the "
           f"model started. F149's question is answered."
           if bi else
           "NEITHER model moves both ways, and unlike F149 these models COULD have. That is a much "
           "stronger null: on models with genuine headroom the domain still moves everything one "
           "way, so the deflationary account survives a test designed to break it."))

    tmpl_contrast = []
    for m, r in rows.items():
        t = r["moves"].get("chat_template")
        pr = [v for k, v in r["moves"].items() if k.startswith("prose:")]
        if t and pr:
            lo, hi = min(v["value"] for v in pr), max(v["value"] for v in pr)
            margin = min(abs(v["value"] - t["value"]) for v in pr)
            tmpl_contrast.append((m, t["value"], lo, hi, margin))
    if tmpl_contrast:
        parts.append(
            "SECONDARY, the kind contrast at matched length on two FRESH models (F147's test, prose "
            "read as an ensemble per F148): "
            + "; ".join(f"{m.split('/')[-1]} template {tv:.3f} vs prose [{lo:.3f}, {hi:.3f}], "
                        f"closest prose {mg:.3f} away"
                        for m, tv, lo, hi, mg in tmpl_contrast)
            + ". A prose range that does not reach the template value extends F147's contrast to "
              "models it was not fitted on; one that straddles it bounds the contrast instead.")
    frag = [m for m, r in rows.items() if r["raw_cls"] == "fragmented"]
    if frag:
        parts.append(
            f"FIRST FRAGMENTED MODEL in any domain run ({[m.split('/')[-1] for m in frag]}): classes "
            f"across its domains are "
            + "; ".join(f"{m.split('/')[-1]} " + ", ".join(
                f"{k}={v['cls']}" for k, v in rows[m]["moves"].items()) for m in frag)
            + ". F144's class claim had only ever been tested on funnels and nones, so this is new "
              "coverage of the taxonomy regardless of what the PRIMARY returns.")
    parts.append(
        f"BOUNDARY: {len(rows)} models, {N_STARTS} starts, {len(CENSUS_SEEDS)} census seeds, one "
        f"prose ensemble per F148. The PRIMARY is an existence test either way: it can refute a "
        f"general floor/ceiling account or fail to, but two models cannot say how common either is.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=MODELS, domains=list(DOMAINS) + ["chat_template", "prose ensemble (F148's samples)"],
        n_starts=N_STARTS, census_seeds=CENSUS_SEEDS, min_shift=MIN_SHIFT, noise_factor=NOISE_FACTOR,
        rung="the raw arm must reproduce midrange_screen exactly at both seeds",
        primary="does either model move robustly UP under one domain and robustly DOWN under "
                "another, from a single raw value, each shift judged against its OWN seed noise",
        why="F149 could not test the floor/ceiling account because 5 of 6 models sat at an extreme; "
            "F150 screened for models with genuine headroom and found these two",
        note="a new file on purpose -- domain_gradient.py's model list is NOT edited, because its "
             "provenance closure is stamped into F147's and F148's stored results and widening it "
             "in place would invalidate them; the census, wrapper and prose sampler are imported "
             "unchanged so the quantity is identical")
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
            n_match = len(tids) if tids else 0
            samples = prose_samples(tok, n_match)
            want = (["raw", "chat_template"] + list(DOMAINS) + list(samples))
            if all(f"{m}|s{cs}|{k}" in res["runs"] for k in want for cs in CENSUS_SEEDS):
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
            print(f"  {m:<32} template={n_match} tok, {len(samples)} prose samples", flush=True)

            arms = {"raw": None, "chat_template": [int(t) for t in tids] if tids else None}
            for d in DOMAINS:
                pre, _n = domain_prefix(tok, d, n_match)
                arms[d] = pre
            arms.update({s: p for s, p in samples.items()})
            for key, pre in arms.items():
                if key != "raw" and pre is None:
                    print(f"  {m}: {key} unavailable -- skipped", flush=True)
                    continue
                for cs in CENSUS_SEEDS:
                    k = f"{m}|s{cs}|{key}"
                    if k in res["runs"]:
                        continue
                    target = model if pre is None else _Prefixed(model, pre)
                    c = argmax_census(target, tok, dev, pool, np.random.default_rng(cs),
                                      n_starts=N_STARTS)
                    c.update(cls=classify(c), model=m, census_seed=cs, arm=key,
                             n_prefix_tokens=0 if pre is None else len(pre))
                    res["runs"][k] = c
                    json.dump(res, open(OUT, "w"), indent=1)
                    print(f"  {m:<32} {key:<10} s={cs} cls={c['cls']:<11} "
                          f"fix={c['fixed_point_fraction']:.3f}", flush=True)
            done += 1
            print(f"  {m:<32} model done in {time.time()-t0:.0f}s", flush=True)
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
