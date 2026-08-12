"""Does the attractor share survive a TOP-K interface? The free half of the closed-model question.

WHY THIS RUNS BEFORE ANY API CALL. F130 established that the attractor share is the instrument's
model-attributable readout -- signal on 6 of 6 constructions, seed-stable ranking 0.848,
construction-invariance +0.752 -- and that it needs only a settle, no CRN twins and no full
distribution. That makes it the one quantity computable through a commercial API, where the most a
provider exposes is the top-k logprobs per step (OpenAI caps `top_logprobs` at 20).

Going straight to a frontier API would confound two things: the INTERFACE restriction and the model
being different. A disagreement would be uninterpretable. Restricting a local model's own
conditional to its top k isolates the interface, costs nothing, and has a known answer to check
against -- which is what makes it a rung rather than an exploration.

THE PRIOR REASON TO EXPECT TROUBLE, which is what gives the test teeth. F110 found that projecting
the conditional onto a small support does not dim it uniformly: it removes the LONG-RANGE
contribution specifically, collapsing the far window position to 0.061 against the near position's
0.801. Top-k is a restriction. If the same thing happens here, a top-k lattice is a different
dynamical object and the API route needs F125's remedy -- widen r, where 9 of 9 arms reached
criticality by r <= 6 -- rather than a smaller alphabet.

WHAT IS AND IS NOT SIMULATED. Truncating to the top k of the model's own conditional is exactly
what an API exposes. What is NOT simulated: tokenizer access (assumed), rate limits, and the
provider's own sampling implementation. Those are engineering, not measurement.

PRE-REGISTERED:
  RUNG      the full-vocabulary arm must reproduce share_invariance's stored top1 for the same
            (model, construction) within RUNG_TOL. Same geometry, same seeds, so a mismatch means
            this script is not measuring F130's quantity and nothing below is read.
  PRIMARY   does the top-k share preserve F130's MODEL RANKING? Reported as Spearman between the
            full-vocabulary and top-k model orderings, per construction and per k. Registered
            reading: >= PRESERVES means the API route is viable on this readout; <= BREAKS means
            the interface destroys the ranking and the route needs a different construction.
  SECONDARY the absolute shift in top1, since a preserved ranking on shifted values still means
            cross-study numbers are not comparable to full-vocabulary ones.
  BOUNDARY  ten local models, none of them frontier-scale. A preserved ranking here says the
            INTERFACE is survivable; it says nothing about whether the share means the same thing
            on a model an order of magnitude larger.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, time

import numpy as np, torch
from ranking import spearman
from provenance import stamp, rel
from gatecheck import pack_state, STATE_KEY

OUT = str(_ROOT / "results" / "topk_ablation.json")
REF = str(_ROOT / "results" / "share_invariance.json")
MODELS = ["EleutherAI/pythia-31m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "gpt2", "gpt2-medium", "gpt2-large",
          "facebook/opt-350m", "bigscience/bloom-560m",
          "state-spaces/mamba-130m-hf", "RWKV/rwkv-4-169m-pile"]
# 20 is OpenAI's top_logprobs cap; 5 is a stricter provider; 100 is a generous one. 0 = full vocab.
KS = [0, 5, 20, 100]
CONSTRUCTIONS = [(2, 0.02), (2, 0.2), (2, 0.7), (3, 0.02)]
N, B, SETTLE = 48, 16, 30
SEEDS = [20260810, 20260811]
RUNG_TOL = 0.05
PRESERVES, BREAKS = 0.6, 0.3


def topk_sampler(k):
    """Inverse-CDF over the top-k of the conditional -- exactly what an API's top_logprobs gives.

    The renormalisation is the interface's, not ours: a provider hands back k (token, logprob)
    pairs and a caller who wants to sample must normalise over them. Returning real token ids keeps
    the lattice in the model's own vocabulary so every downstream tool works unchanged.
    """
    def sampler(probs, u):
        p = np.asarray(probs, dtype=np.float64)
        idx = np.argpartition(-p, kth=min(k, p.shape[1] - 1), axis=1)[:, :k]
        vals = np.take_along_axis(p, idx, axis=1)
        order = np.argsort(-vals, axis=1)                 # CDF walks in descending probability
        idx, vals = np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)
        s = vals.sum(axis=1, keepdims=True)
        vals = np.where(s > 0, vals / np.where(s > 0, s, 1.0), 1.0 / k)
        cdf = np.cumsum(vals, axis=1)
        cdf = cdf / cdf[:, -1:]
        pick = (cdf < np.asarray(u, dtype=np.float64)[:, None]).sum(axis=1)
        return np.take_along_axis(idx, np.clip(pick, 0, k - 1)[:, None], axis=1).ravel()
    sampler.k = k
    return sampler


def cell(rule, r, T, seed, k):
    from ar_ca import run
    kw = dict(B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme="none", init="random", seed=seed)
    if k:
        kw["sampler"] = topk_sampler(k)
    settled = run(rule, **kw)["final"]
    pool = settled.reshape(-1)
    vals, cnt = np.unique(pool, return_counts=True)
    # rep2 and the lattice itself: `top1` alone cannot separate a one-token attractor from a
    # short periodic orbit, which is 1/p by arithmetic (F136). Storing the state is what lets
    # share_periodicity.py ask that question of these cells without a re-run.
    rep2 = float(np.mean(settled[:, :-1] == settled[:, 1:]))
    return dict(top1=float(cnt.max() / cnt.sum()), distinct=float(len(vals)), rep2=rep2,
                **{STATE_KEY: pack_state(settled, stride_axis=0,
                                         note="settled lattice, (replica, site)")})


def ranks(cells, k, con, seed):
    out = []
    for m in MODELS:
        c = cells.get(f"{m}|k{k}|{con}|s{seed}")
        if c is None:
            return None
        out.append(c["top1"])
    return out


def analyse(res):
    cells, parts = res["cells"], []
    ref = json.load(open(REF))["cells"]
    errs = []
    for con in {c["construction"] for c in cells.values()}:
        for m in MODELS:
            a = cells.get(f"{m}|k0|{con}|s{SEEDS[0]}")
            b = ref.get(f"{m}|{con}|s{SEEDS[0]}")
            if a and b:
                errs.append(abs(a["top1"] - b["top1"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst <= RUNG_TOL
    parts.append(
        f"RUNG: the full-vocabulary arm reproduces share_invariance's stored top1 to within "
        f"{worst:.4f} across {len(errs)} cells (tolerance {RUNG_TOL}). "
        + ("Same quantity as F130, so the top-k arms below are a restriction of it."
           if ok else "NOT reproduced -- this is not F130's measurement and nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst_err=worst)
        res["verdict"] = " ".join(parts); return
    rho, shift = {}, {}
    for k in [x for x in KS if x]:
        rs, sh = [], []
        for con in sorted({c["construction"] for c in cells.values()}):
            full, cut = ranks(cells, 0, con, SEEDS[0]), ranks(cells, k, con, SEEDS[0])
            if full and cut:
                v = spearman(full, cut)
                if np.isfinite(v):
                    rs.append(v)
                sh.append(float(np.mean(np.abs(np.array(cut) - np.array(full)))))
        rho[k] = round(float(np.mean(rs)), 4) if rs else None
        shift[k] = round(float(np.mean(sh)), 4) if sh else None
    live = {k: v for k, v in rho.items() if v is not None}
    worstk = min(live, key=lambda k: live[k]) if live else None
    parts.append(
        "PRIMARY, agreement between the full-vocabulary and top-k model rankings: "
        + ", ".join(f"k={k}: rho={v:+.3f}" for k, v in sorted(live.items())) + ". "
        + (f"At or above the registered {PRESERVES} even at k={worstk}, so the interface preserves "
           f"the ordering and the API route is viable on this readout."
           if live and min(live.values()) >= PRESERVES else
           f"Below {PRESERVES} at k={worstk} (rho={live[worstk]:+.3f}), so a top-k interface does "
           f"not preserve F130's ranking at that width. The remedy F125 supplies is to widen r "
           f"rather than to accept the smaller support."
           if live else "no readable comparison."))
    parts.append(
        "SECONDARY, mean absolute shift in top1: "
        + ", ".join(f"k={k}: {v:+.4f}" for k, v in sorted(shift.items()) if v is not None)
        + ". A preserved RANKING on shifted VALUES still means top-k numbers are not comparable to "
          "full-vocabulary ones across studies, and any published figure must name its k.")
    parts.append(
        f"BOUNDARY: {len(MODELS)} local models, none frontier-scale, {len(CONSTRUCTIONS)} "
        f"constructions, N={N}. A preserved ranking says the INTERFACE is survivable; it says "
        f"nothing about whether the share means the same thing on a model an order of magnitude "
        f"larger. Tokenizer access, rate limits and the provider's own sampler are not simulated.")
    res["analysis"] = dict(rung_passes=True, rung_worst_err=worst, rho_by_k=rho, shift_by_k=shift)
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, ks=KS, constructions=[f"r{r}.T{T}" for r, T in CONSTRUCTIONS],
        N=N, B=B, settle=SETTLE, seeds=SEEDS, rung_tol=RUNG_TOL,
        preserves=PRESERVES, breaks=BREAKS, reference=rel(REF),
        rung="the k=0 (full vocabulary) arm must reproduce share_invariance's stored top1",
        primary="Spearman between the full-vocabulary and top-k model rankings, per k",
        prior="F110 found restriction removes the LONG-RANGE contribution specifically, so a "
              "top-k lattice may be a different dynamical object; F125 gives the remedy (widen r)")
    from ar_ca import ARRule
    for m in MODELS:
        try:
            rule = ARRule(m)
        except Exception as e:
            print(f"  {m}: LOAD FAILED {type(e).__name__}", flush=True); continue
        for k in KS:
            for r, T in CONSTRUCTIONS:
                for sd in SEEDS:
                    key = f"{m}|k{k}|r{r}.T{T}|s{sd}"
                    if key in res["cells"]:
                        continue
                    t0 = time.time()
                    try:
                        c = cell(rule, r, T, sd, k)
                    except Exception as e:
                        print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
                        continue
                    c.update(model=m, k=k, construction=f"r{r}.T{T}", r=r, T=T, seed=sd,
                             secs=round(time.time() - t0, 1))
                    res["cells"][key] = c
                    print(f"  {key:<48} top1={c['top1']:.4f} distinct={c['distinct']:.0f}",
                          flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)
        del rule
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
