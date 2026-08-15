"""Does the ATTRACTOR SHARE survive the domain axis? Paper 2's positive core, tested where it hurts.

WHY THIS RUNS NOW. F144 measured the domain and found it dominant: a nine-token chat template took
one model's fixed-point fraction from 0.948 to 0.000 and changed its class, while eleven tokens took
another's from 0.615 to 0.844 and did not. Same weights, same estimator, same seeds. That was the
argmax MAP. The lattice is a different object, and paper 2's reframed thesis rests on the lattice:
F130 establishes the attractor share as the instrument's model-attributable readout, and everything
that transfers is built on it.

Every one of those measurements was taken in the RAW domain — r tokens and nothing before them. So
the question F144 forces is whether F130's model ranking is a property of the models or of the raw
domain they were measured in.

WHAT IS ALREADY KNOWN AND WHY IT IS NOT ENOUGH. F135 measured a chat scaffold moving the share by
0.1327 against a relative gate of 0.0406 — the VALUES are contaminated — while reporting that the
model RANKING survived (rho = +1.000). But that was three BASE models with a hand-written scaffold,
and three cautions were recorded against the ranking claim at n = 3, including that rho = +1.000 on
three models is one ordering out of six. This is six instruction-tuned models behind their OWN
templates, which is both a larger n and the domain these models are actually deployed in.

  ranking survives -> the share is more robust than the fixed-point class, F130 holds on the axis
                      that broke the class, and paper 2's core is safe
  ranking breaks   -> the domain kills the share too, F130's model-attributability is a raw-domain
                      statement, and the reframed thesis needs revising again

THE DOMAIN IS A SUBCLASS, NOT AN EDIT. `ar_ca.ARRule` already carries a domain axis — `scheme="bos"`
prepends a BOS token, which is exactly 2608.10986's experiment. Extending it for a template by
editing `ar_ca.py` would invalidate the provenance closure of most of this repository, which is the
mistake made earlier today with `gate1`. Subclassing costs nothing and changes no shared byte.

PRE-REGISTERED:
  RUNG      the raw arm must reproduce `share_instruct`'s stored top1 for the same (model, T, seed)
            to within RUNG_TOL. Same geometry, same seeds, so a mismatch means the subclass is not
            the same rule and nothing below is read.
  PRIMARY   Spearman between the model rankings the two domains produce, per temperature. Registered
            reading: >= CONCORDANT at a majority of temperatures means the ranking survives the
            domain; below it means the share's model-attributability is raw-domain only.
  SECONDARY the shift in VALUES, which F135 already showed is large. Reported so the two runs are
            comparable, not because a value shift is news.
  SIGNAL    the templated arm must itself have across-model spread above across-seed noise, or its
            ranking is noise and the correlation is uninterpretable in either direction.
  BOUNDARY  six instruction-tuned models, one template each, one lattice geometry. Base models have
            no template and cannot be run on this axis at all.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from gatecheck import pack_state, has_state, spearman, STATE_KEY

OUT = str(_ROOT / "results" / "share_templated.json")
RAW = _ROOT / "results" / "share_instruct.json"
COHORT = _ROOT / "results" / "instruct_cohort.json"

# share_instruct's geometry, copied so top1@T is the same quantity
N, B, SETTLE, R = 96, 16, 16, 2
# TWO TEMPERATURES, NOT FOUR, and the reason is measured rather than assumed. The template adds
# 35 tokens to every forward pass and the lattice does N*sweeps*B of them, so a cell costs 1617s
# against the raw run's ~150s -- an 11x tax that turns a 4x2x5 grid into ~18 hours. The extremes
# are kept because F144 found the largest domain effects at the cold end and the share's own
# across-model spread is widest there (0.40 at T=0.02 against 0.24 at T=0.7). Reported as reduced
# coverage in the boundary rather than quietly dropped.
TEMPS = [0.02, 0.7]
SEEDS = [20260813, 20260814]
RUNG_TOL = 1e-9          # the raw arm should be BIT-identical: same rule, same seeds, same geometry
CONCORDANT = 0.6
NOISE_FACTOR = 2.0


def _templated_rule(model_name):
    """ARRule seen through the model's own chat template. A subclass, so no shared byte changes."""
    from ar_ca import ARRule

    class _Templated(ARRule):
        def __init__(self, name):
            super().__init__(name)
            self.prefix = None
            tmpl = getattr(self.tok, "chat_template", None)
            if tmpl:
                try:
                    txt = self.tok.apply_chat_template(
                        [{"role": "user", "content": ""}], tokenize=False,
                        add_generation_prompt=True)
                    self.prefix = self.tok(txt, add_special_tokens=False)["input_ids"]
                except Exception:
                    self.prefix = None

        @torch.no_grad()
        def center_probs(self, win, T, scheme="none", as_torch=False):
            if self.prefix is None or scheme == "raw_passthrough":
                return super().center_probs(win, T, scheme="none", as_torch=as_torch)
            win = np.asarray(win, dtype=np.int64)
            pre = np.tile(np.asarray(self.prefix, np.int64), (win.shape[0], 1))
            seq = np.concatenate([pre, win], axis=1)
            ids = torch.from_numpy(seq).to(self.device)
            logits = self.model(input_ids=ids).logits[:, -1, :].float()
            if self._forbid_t is not None:
                logits[:, self._forbid_t] = -1e9
            probs = torch.softmax(logits / T, dim=-1)
            return probs if as_torch else probs.cpu().numpy()

    return _Templated(model_name)


def cell(rule, T, seed, raw=False):
    from ar_ca import run
    if raw:
        saved, rule.prefix = rule.prefix, None      # the same object, domain switched off
    try:
        settled = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none", init="random",
                      seed=seed)["final"]
    finally:
        if raw:
            rule.prefix = saved
    pool = settled.reshape(-1)
    vals, cnt = np.unique(pool, return_counts=True)
    return dict(top1=float(cnt.max() / cnt.sum()), distinct=float(len(vals)),
                rep2=float(np.mean(settled[:, :-1] == settled[:, 1:])),
                **{STATE_KEY: pack_state(settled, stride_axis=0,
                                         note="settled lattice, (replica, site)")})


def analyse(res):
    cells = res["cells"]
    raw_ref = json.load(open(RAW))["cells"] if RAW.exists() else {}
    models = sorted({c["model"] for c in cells.values()})
    parts, analysis = [], {}

    errs = []
    for k, v in cells.items():
        if not k.endswith("|rawcheck"):
            continue
        base = k.replace("|rawcheck", "")
        r = raw_ref.get(base)
        if r:
            errs.append(abs(v["top1"] - r["top1"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst <= RUNG_TOL
    parts.append(
        f"RUNG (the subclass with its domain switched off reproduces share_instruct): worst error "
        f"{worst:.2e} across {len(errs)} cells (tolerance {RUNG_TOL:g}). "
        + ("Bit-identical, so the templated arm differs from the raw one in the DOMAIN and in "
           "nothing else."
           if ok else "NOT reproduced -- the subclass is not the same rule and nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    def col(T, s, suffix):
        out = []
        for m in models:
            v = cells.get(f"{m}|T{T}|s{s}{suffix}")
            if v is None:
                return None
            out.append(v["top1"])
        return out

    sig, rhos, shifts = [], {}, {}
    for T in TEMPS:
        a, b = col(T, SEEDS[0], "|tmpl"), col(T, SEEDS[1], "|tmpl")
        if a is None or b is None:
            continue
        spread = float(max(a) - min(a))
        noise = float(np.mean([abs(x - y) for x, y in zip(a, b)]))
        sig.append((T, spread, noise, noise > 0 and spread >= NOISE_FACTOR * noise))
        rawv = [float(np.mean([raw_ref[f"{m}|T{T}|s{s}"]["top1"] for s in SEEDS
                               if f"{m}|T{T}|s{s}" in raw_ref])) for m in models]
        tmplv = [float(np.mean([cells[f"{m}|T{T}|s{s}|tmpl"]["top1"] for s in SEEDS
                                if f"{m}|T{T}|s{s}|tmpl" in cells])) for m in models]
        rhos[T] = round(float(spearman(rawv, tmplv)), 4)
        shifts[T] = round(float(np.mean(np.abs(np.array(tmplv) - np.array(rawv)))), 4)
    n_sig = sum(1 for *_x, o in sig if o)
    analysis.update(signal=[dict(T=t, spread=round(s, 4), seed_noise=round(n, 4), passes=bool(o))
                            for t, s, n, o in sig],
                    rank_agreement=rhos, mean_abs_shift=shifts, models=models)
    parts.append(
        "SIGNAL in the templated arm: "
        + "; ".join(f"T={t}: spread {s:.3f} vs seed noise {n:.3f}" + (" OK" if o else " FAILS")
                    for t, s, n, o in sig)
        + f". {n_sig} of {len(sig)} constructions carry model signal above seed noise"
        + ("." if n_sig > len(sig) / 2 else
           " -- the templated ranking is mostly noise, so the agreement below is uninterpretable."))
    good = [t for t, r in rhos.items() if r >= CONCORDANT]
    parts.append(
        "PRIMARY, agreement between the RAW and TEMPLATED model rankings: "
        + ", ".join(f"T={t}: rho={r:+.3f}" for t, r in sorted(rhos.items())) + ". "
        + (f"At or above {CONCORDANT} on {len(good)} of {len(rhos)} temperatures: the share's model "
           f"ranking SURVIVES the domain that broke the fixed-point class (F144), so F130 holds on "
           f"this axis and paper 2's core is safe."
           if len(good) > len(rhos) / 2 else
           f"Below {CONCORDANT} on {len(rhos) - len(good)} of {len(rhos)} temperatures: the domain "
           f"reorders the models, so F130's model-attributability is a RAW-DOMAIN statement and the "
           f"reframed thesis needs revising again."))
    parts.append(
        "SECONDARY, mean |shift| in top1 between domains: "
        + ", ".join(f"T={t}: {v:.4f}" for t, v in sorted(shifts.items()))
        + ". F135 already established the VALUES do not transfer; this is reported for "
          "comparability, not as news.")
    parts.append(
        f"BOUNDARY: {len(models)} instruction-tuned models, one template each, N={N}, B={B}, "
        f"settle={SETTLE}, r={R}, and only {len(TEMPS)} of share_instruct's four temperatures "
        f"({TEMPS}) -- the templated cell costs 1617s against the raw run's 150s, so the full grid "
        f"was ~18 hours and the extremes were kept. Base models have no template and cannot be run "
        f"on this axis. One template is one domain, so this shows whether the ranking survives THIS "
        f"domain change, not domain changes in general.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    cohort = [r["model"] for r in json.load(open(COHORT))["cohort"]]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        cohort=cohort, N=N, B=B, settle=SETTLE, r=R, temps=TEMPS, seeds=SEEDS,
        rung_tol=RUNG_TOL, concordant=CONCORDANT, noise_factor=NOISE_FACTOR,
        geometry="copied from share_instruct so top1@T is the same quantity",
        domain="each model's own chat template, injected by SUBCLASSING ARRule -- ar_ca.py is "
               "unchanged, because editing it would invalidate most of this repo's provenance "
               "closure",
        rung="the subclass with its domain switched off must reproduce share_instruct bit-for-bit",
        primary="Spearman between the raw and templated model rankings, per temperature",
        why="F144 showed the domain dominates on the argmax map; F130's share is the lattice's "
            "model-attributable readout and was only ever measured raw")
    if "--analyse" not in _sys.argv:
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in cohort:
            need = [(T, s) for T in TEMPS for s in SEEDS
                    if not has_state(res["cells"].get(f"{m}|T{T}|s{s}|tmpl", {}))]
            if not need:
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            try:
                rule = _templated_rule(m)
            except Exception as e:
                print(f"  {m}: LOAD FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            if rule.prefix is None:
                print(f"  {m}: NO CHAT TEMPLATE -- skipped", flush=True)
                del rule; gc.collect(); continue
            # RUNG first, on one cheap cell: domain off must reproduce share_instruct exactly
            rk = f"{m}|T{TEMPS[0]}|s{SEEDS[0]}|rawcheck"
            if rk not in res["cells"]:
                c = cell(rule, TEMPS[0], SEEDS[0], raw=True)
                c.update(model=m, T=TEMPS[0], seed=SEEDS[0], arm="rawcheck")
                res["cells"][rk] = c
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<40} RUNG raw top1={c['top1']:.4f}", flush=True)
            for T, s in need:
                k = f"{m}|T{T}|s{s}|tmpl"
                t0 = time.time()
                c = cell(rule, T, s)
                c.update(model=m, T=T, seed=s, arm="tmpl", n_prefix_tokens=len(rule.prefix))
                res["cells"][k] = c
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {k:<58} top1={c['top1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            done += 1
            del rule
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
