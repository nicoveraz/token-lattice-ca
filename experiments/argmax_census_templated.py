"""Is the fixed-point class a property of the WEIGHTS or of the DOMAIN? The chat-template test.

WHY THIS RUNS, and it comes from F143's own prior-art gate rather than from an idea. That gate
turned up a threat inside this project's published paper: arXiv:2608.10986 shows the frozen fraction
is a property of the map's DOMAIN, not its parameters — prepending a single BOS token moves it from
**74.4% to 24.1%** with zero weight change. F143 then reported that the fixed-point class survives
instruction tuning on two clean pairs. Both cannot be read casually: an instruction-tuned model in
deployment is read through its own chat template, and a template is exactly the kind of domain
change that paper showed can move everything.

So F143's null is currently "invariance at fixed domain". This asks whether it is more than that.

THE COMPARISON IS WITHIN ONE MODEL, which is what makes it clean. The same weights, the same census,
the same seeds, the same 96 starts — only the tokens preceding the two-token state change, from
nothing to the model's own rendered chat template. Any class difference is therefore the domain and
nothing else: no pretraining difference, no size difference, no tuning difference.

  class unchanged under the template -> the invariance is about the WEIGHTS, and F143 strengthens
                                        from "at fixed domain" to a statement about the model
  class changes under the template    -> the class is DOMAIN-determined, F143's null describes the
                                        raw map only, and 2608.10986's BOS result generalises from
                                        one token to a realistic deployment prefix. That is the more
                                        interesting outcome, because it makes the domain the object.

Both are worth having and neither is the hoped-for one.

PRE-REGISTERED:
  RUNG      the UNWRAPPED census run from this script must reproduce the stored raw-domain census
            EXACTLY — same class, same fixed_point_fraction, same modal share — for every model
            already measured, or the templated arm is being compared against a different estimator
            and nothing is read. The domain is injected by wrapping the MODEL (see `_Prefixed`), so
            `gate1.argmax_census` is called unmodified in both arms and this rung checks the whole
            path end to end rather than one argument.
  PRIMARY   per model, does the class differ between raw domain and templated domain? Reported per
            census seed, with the same stability requirement F143 used: a class that is unstable
            across seeds is underpowered, not a change.
  SECONDARY the within-class quantities (fixed_point_fraction, modal share, distinct endpoints),
            because 2608.10986's BOS effect was a large move in fixed_point_fraction and the
            four-way label would not have shown it.
  BOUNDARY  six instruction-tuned models, one template each, 96 starts, two seeds. A template is one
            domain; this does not characterise domains in general, and the base models have no
            template so they cannot be run on this axis at all.

WHY THE DOMAIN IS A WRAPPER AND NOT AN ARGUMENT. The first version added `prefix=` to
`gate1.argmax_census`. It was additive, and a rung proved the default path byte-identical — and it
still invalidated SIX results files, because gate1 sits in their provenance import closure and the
staleness guard hashes that closure precisely so an edit there cannot pass unnoticed. The guard was
right and the change was the wrong shape. Wrapping the model touches no shared code, and it is the
better framing anyway: the domain belongs to the model's input pipeline, not to the probe.
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
from gate1 import argmax_census


from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_instruct import PAIRS

OUT = str(_ROOT / "results" / "argmax_census_templated.json")
RAW = _ROOT / "results" / "argmax_census_instruct.json"

MODELS = [i for _b, i, _k in PAIRS]


class _Prefixed:
    """The model, seen through a fixed prefix. The DOMAIN as a wrapper, not as a code change.

    A first version added an optional `prefix=` argument to `gate1.argmax_census`. It was additive
    and the rung proved the default path byte-identical -- and it still invalidated SIX results
    files, because gate1 sits in their provenance import closure and the staleness guard hashes
    that closure precisely so an edit there cannot pass unnoticed. The guard was right; the change
    was the wrong shape.

    Wrapping the model instead changes nothing shared. `argmax_census` calls
    `model(input_ids=...)` and reads `.logits`, so an object that prepends the prefix and delegates
    is indistinguishable from a model that lives in that domain -- which is also the honest
    conceptual framing: the domain belongs to the model's input pipeline, not to the probe.
    """

    def __init__(self, model, prefix_ids):
        self._m = model
        self._p = [int(t) for t in prefix_ids]
        self.config = model.config

    def __call__(self, input_ids=None, **kw):
        import torch as _t
        pre = _t.tensor([self._p] * input_ids.shape[0], device=input_ids.device,
                        dtype=input_ids.dtype)
        return self._m(input_ids=_t.cat([pre, input_ids], dim=1), **kw)


def template_ids(tok):
    """The model's own chat prefix, rendered and tokenised up to the generation point."""
    if not getattr(tok, "chat_template", None):
        return None, "no chat_template on this tokenizer"
    try:
        txt = tok.apply_chat_template([{"role": "user", "content": ""}],
                                      tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return None, f"apply_chat_template failed: {type(e).__name__}"
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    return ids, txt


def analyse(res):
    raw = json.load(open(RAW))["runs"] if RAW.exists() else {}
    runs = res["runs"]
    parts, rows = [], {}

    # RUNG: the prefix=None path must reproduce the stored raw census exactly.
    checks, bad = 0, []
    for k, v in runs.items():
        if not k.endswith("|noprefix"):
            continue
        base_k = k.replace("|noprefix", "")
        r = raw.get(base_k)
        if not r:
            continue
        checks += 1
        for field in ("cls", "fixed_point_fraction", "modal_endpoint_share",
                      "n_distinct_endpoints"):
            if v.get(field) != r.get(field):
                bad.append(f"{base_k}:{field} {r.get(field)} -> {v.get(field)}")
    ok = checks > 0 and not bad
    parts.append(
        f"RUNG (the unwrapped census reproduces the stored raw census): {checks} field-sets "
        f"compared. "
        + ("Identical, so both arms run the same unmodified estimator and the comparison below is "
           "between one estimator and itself, differing only in the model's input domain."
           if ok else
           f"MISMATCH: {bad[:4]} — the raw path does not reproduce, so nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, mismatches=bad)
        res["verdict"] = " ".join(parts)
        return

    for m in MODELS:
        ks = [f"{m}|s{cs}|tmpl" for cs in CENSUS_SEEDS]
        if not all(k in runs for k in ks):
            continue
        rk = [f"{m}|s{cs}" for cs in CENSUS_SEEDS]
        if not all(k in raw for k in rk):
            continue
        tcls = [runs[k]["cls"] for k in ks]
        rcls = [raw[k]["cls"] for k in rk]
        rows[m] = dict(
            raw_cls=rcls[0] if rcls[0] == rcls[1] else None,
            tmpl_cls=tcls[0] if tcls[0] == tcls[1] else None,
            tmpl_stable=tcls[0] == tcls[1],
            n_prefix_tokens=runs[ks[0]].get("n_prefix_tokens"),
            raw_fix=[raw[k]["fixed_point_fraction"] for k in rk],
            tmpl_fix=[runs[k]["fixed_point_fraction"] for k in ks],
            raw_modal=[raw[k]["modal_endpoint_share"] for k in rk],
            tmpl_modal=[runs[k]["modal_endpoint_share"] for k in ks],
            raw_endpoints=[raw[k]["n_distinct_endpoints"] for k in rk],
            tmpl_endpoints=[runs[k]["n_distinct_endpoints"] for k in ks])
        rows[m]["changed"] = (rows[m]["raw_cls"] is not None and rows[m]["tmpl_cls"] is not None
                              and rows[m]["raw_cls"] != rows[m]["tmpl_cls"])
    unstable = [m for m, r in rows.items() if not r["tmpl_stable"]]
    changed = [m for m, r in rows.items() if r["changed"]]
    res["analysis"] = dict(rung_passes=True, rows=rows, n=len(rows),
                           n_changed=len(changed), unstable=unstable)
    parts.append(
        f"STABILITY: {len(rows) - len(unstable)} of {len(rows)} templated censuses agree across "
        f"both seeds"
        + ("." if not unstable else f"; UNSTABLE and therefore not read: {unstable}."))
    parts.append(
        "PRIMARY (same weights, same seeds, only the domain changes): "
        + "; ".join(f"{m.split('/')[-1]} [{r['raw_cls']}] -> [{r['tmpl_cls']}]"
                    + ("  CHANGED" if r["changed"] else "")
                    for m, r in rows.items()) + ". "
        + (f"{len(changed)} of {len(rows)} models change class under their own chat template. The "
           f"fixed-point class is DOMAIN-determined, not a property of the weights alone — "
           f"2608.10986's BOS effect generalises from one token to a deployment prefix, and F143's "
           f"null describes the RAW map only."
           if changed else
           "NO model changes class under its own chat template. The class is a property of the "
           "weights rather than of the domain at this scale of prefix, which strengthens F143 from "
           "'invariance at fixed domain' to invariance across the domain a deployed instruct model "
           "actually runs in."))
    parts.append(
        "SECONDARY (the within-class quantities the label cannot carry): "
        + "; ".join(f"{m.split('/')[-1]} fix {r['raw_fix']}->{r['tmpl_fix']} "
                    f"endpoints {r['raw_endpoints']}->{r['tmpl_endpoints']}"
                    for m, r in rows.items()) + ".")
    parts.append(
        f"BOUNDARY: {len(rows)} instruction-tuned models, one template each "
        f"({min((r['n_prefix_tokens'] or 0) for r in rows.values())}-"
        f"{max((r['n_prefix_tokens'] or 0) for r in rows.values())} prefix tokens), "
        f"{N_STARTS} starts, two census seeds. One template is one domain; base models have no "
        f"template and cannot be run on this axis at all.")
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=MODELS, n_starts=N_STARTS, census_seeds=CENSUS_SEEDS,
        rung="the unwrapped census must reproduce the stored raw census exactly; the domain is "
             "injected by wrapping the MODEL, so gate1.argmax_census is called unmodified in both "
             "arms and no shared code changed",
        primary="does the class differ between raw domain and the model's own chat template, at "
                "fixed weights and fixed seeds",
        why="arXiv:2608.10986 showed the frozen fraction moves 74.4% -> 24.1% on ONE BOS token, so "
            "F143's invariance is currently 'at fixed domain' and this tests whether it is more",
        raw_reference=rel(str(RAW)))
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in MODELS:
            want = [f"{m}|s{cs}|tmpl" for cs in CENSUS_SEEDS] + [f"{m}|s{CENSUS_SEEDS[0]}|noprefix"]
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
            ids, txt = template_ids(tok)
            if ids is None:
                res["runs"][f"{m}|notemplate"] = dict(model=m, reason=txt)
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: NO TEMPLATE ({txt})", flush=True)
                del model; gc.collect(); continue
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            # RUNG first, on the cheapest seed: no prefix must reproduce the stored raw census
            k0 = f"{m}|s{CENSUS_SEEDS[0]}|noprefix"
            if k0 not in res["runs"]:
                c = argmax_census(model, tok, dev, pool,
                                  np.random.default_rng(CENSUS_SEEDS[0]), n_starts=N_STARTS)
                c["cls"] = classify(c); c["model"] = m
                res["runs"][k0] = c
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<40} RUNG noprefix cls={c['cls']:<11} fix={c['fixed_point_fraction']:.3f}",
                      flush=True)
            for cs in CENSUS_SEEDS:
                k = f"{m}|s{cs}|tmpl"
                if k in res["runs"]:
                    continue
                c = argmax_census(_Prefixed(model, ids), tok, dev, pool,
                                  np.random.default_rng(cs), n_starts=N_STARTS)
                c["n_prefix_tokens"] = len(ids)
                c["cls"] = classify(c); c["model"] = m; c["census_seed"] = cs
                c["template_text"] = txt[:200]
                res["runs"][k] = c
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<40} tmpl s={cs} cls={c['cls']:<11} "
                      f"fix={c['fixed_point_fraction']:.3f} modal={c['modal_endpoint_share']:.3f} "
                      f"endpts={c['n_distinct_endpoints']} (prefix {c['n_prefix_tokens']} tok)",
                      flush=True)
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
