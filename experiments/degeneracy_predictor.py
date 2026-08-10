"""WHICH sub-alphabet constructions freeze, and can that be predicted before running one?

THE OBSERVATION. Several constructions in `construction_invariance` collapse to a single token:
binary|freq_matched settles at top1 = 1.000 (distinct = 1) and branching 0.002 -- a lattice that
does nothing. A frozen ring is not a weak measurement, it is no measurement: there is no
perturbation to apply within its occupied support, which is why `s_at` returns nan there. Knowing in
advance which constructions freeze decides which are worth running at all.

WHY THIS IS A CLEANER TEST THAN F124. F124's H_gap correlates with s_far partly BY CONSTRUCTION --
if the conditional does not depend on the far token then both are zero identically, so some of that
correlation is guaranteed rather than discovered. Here the predictor is a STATIC property of the
conditional (measured on uniform windows, no CA) and the outcome is a DYNAMICAL one (what the ring
settles into after 12 sweeps). Nothing forces them to agree. A high correlation would be a real
prediction; a low one cannot be explained away.

THE CANDIDATE PREDICTORS, all from one batch of forward passes:
  p_dom    mean of max_v p(v | window), the renormalised conditional's dominant-token mass. The
           direct reading: if one token already owns most of the mass under a uniform window, the
           ring has nowhere else to go.
  h_cond   entropy of the same distribution, in bits. F124 found this does NOT predict s_far; this
           asks whether it predicts DEGENERACY instead, which is a different question and the one
           entropy is actually shaped for.
  k        alphabet size. The null-ish competitor: maybe small alphabets simply freeze.

PRE-REGISTERED:
  RUNG      every (model, alphabet, mode) in construction_invariance.json must be found and
            measured, and the observed top1 must be READ from that file rather than recomputed, so
            the outcome cannot drift toward the predictor.
  PRIMARY   rho(p_dom, top1) across all constructions. Registered reading: >= NAMES predicts
            freezing; <= 0.3 eliminates p_dom.
  SECONDARY rho(h_cond, top1) and rho(k, top1), so a positive primary has to beat the obvious
            alternatives rather than merely exist.
  BOUNDARY  freezing is measured at one temperature and one settle length; a construction that
            freezes in 12 sweeps at T = 0.7 need not freeze at another operating point.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, time

import numpy as np, torch
from ranking import spearman
from provenance import stamp, rel

SRC = str(_ROOT / "results" / "construction_invariance.json")
OUT = str(_ROOT / "results" / "degeneracy_predictor.json")
T = 0.7
N_CTX = 256
SEED = 20260810
NAMES = 0.6


def static_props(rule, ids, r, rng, n_ctx=N_CTX):
    """(p_dom, h_cond) of the renormalised conditional on UNIFORM windows over the alphabet."""
    ids = np.asarray(ids, dtype=np.int64)
    rows = [[int(x) for x in rng.choice(ids, size=r)] for _ in range(n_ctx)]
    with torch.no_grad():
        lg = rule.model(input_ids=torch.tensor(rows, device=rule.device)
                        ).logits[:, -1].float().cpu().double().numpy()
    P = lg[:, ids]
    P = np.exp((P - P.max(axis=1, keepdims=True)) / T)
    P = P / P.sum(axis=1, keepdims=True)
    h = -(P * np.log2(np.clip(P, 1e-300, None))).sum(axis=1)
    return float(P.max(axis=1).mean()), float(h.mean())


def analyse(res):
    cells, parts = res["cells"], []
    ok = len(cells) == res["n_expected"] and all("top1" in c for c in cells.values())
    parts.append(
        f"RUNG: {len(cells)} of {res['n_expected']} constructions measured, with top1 READ from "
        f"construction_invariance.json rather than recomputed so the outcome cannot drift toward "
        f"the predictor. " + ("" if ok else "INCOMPLETE -- nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False)
        res["verdict"] = " ".join(parts); return
    top1 = [c["top1"] for c in cells.values()]
    rho = {p: round(spearman([c[p] for c in cells.values()], top1), 4)
           for p in ("p_dom", "h_cond", "k")}
    frozen = sum(1 for c in cells.values() if c["top1"] >= 0.999)
    names = abs(rho["p_dom"]) >= NAMES
    beats = abs(rho["p_dom"]) > max(abs(rho["h_cond"]), abs(rho["k"]))
    parts.append(
        f"PRIMARY: rho(p_dom, top1) = {rho['p_dom']:+.3f} over {len(cells)} constructions, of which "
        f"{frozen} froze completely (top1 >= 0.999). "
        + (f"At or above the registered {NAMES}, so the dominant-token mass of the conditional "
           f"predicts whether the lattice will freeze -- from one batch of forward passes, with no "
           f"CA run. "
           if names else
           f"Below the registered {NAMES}, so p_dom does NOT predict freezing and the cause is "
           f"elsewhere. ")
        + ("It also beats both alternatives."
           if names and beats else
           "It does NOT beat the alternatives below, so it is not the operative property."
           if names else ""))
    parts.append(
        f"SECONDARY: rho(h_cond, top1) = {rho['h_cond']:+.3f} -- entropy, which F124 found does not "
        f"predict s_far, asked here about the degeneracy it is actually shaped for. "
        f"rho(k, top1) = {rho['k']:+.3f} -- alphabet size, the obvious competitor.")
    parts.append(
        f"BOUNDARY: freezing measured at T = {T} and one settle length, so a construction that "
        f"freezes here need not freeze at another operating point. Unlike F124's H_gap the "
        f"predictor is STATIC and the outcome DYNAMICAL, so nothing forces them to agree and this "
        f"correlation is not structural.")
    res["analysis"] = dict(rung_passes=True, rho=rho, n_frozen=frozen, names=bool(names),
                           beats_alternatives=bool(beats))
    res["verdict"] = " ".join(parts)


def main():
    src = json.load(open(SRC))["cells"]
    # one row per (model, alphabet.mode.r); top1 averaged over the seeds measured there
    groups = {}
    for c in src.values():
        groups.setdefault((c["model"], c["construction"]), []).append(c)
    res = {"cells": {}, "n_expected": len(groups), "_preregistration": dict(
        source=rel(SRC), T=T, n_ctx=N_CTX, seed=SEED, names=NAMES,
        predictors=["p_dom", "h_cond", "k"],
        primary="rho(p_dom, top1) across constructions; >= 0.6 predicts freezing",
        note="predictor is STATIC (uniform windows, no CA), outcome is DYNAMICAL (settled ring), so "
             "unlike F124's H_gap the correlation cannot be structural")}
    from ar_ca import ARRule
    cur, rule = None, None
    for (model, con), cs in sorted(groups.items()):
        if model != cur:
            del rule
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            rule, cur = ARRule(model), model
        ids = cs[0].get("ids")
        if ids is None:
            print(f"  SKIP {model}|{con}: no ids stored", flush=True)
            continue
        r = int(con.rsplit(".r", 1)[1])
        t0 = time.time()
        pd_, hc = static_props(rule, np.array(ids, dtype=np.int64), r,
                               np.random.default_rng(SEED))
        res["cells"][f"{model}|{con}"] = dict(
            model=model, construction=con, k=int(len(ids)), r=r,
            p_dom=round(pd_, 5), h_cond=round(hc, 5),
            top1=float(np.mean([c["top1"] for c in cs])),
            distinct=float(np.mean([c["distinct"] for c in cs])),
            frozen=bool(np.mean([c["top1"] for c in cs]) >= 0.999),
            secs=round(time.time() - t0, 1))
        print(f"  {model}|{con:<26} p_dom={pd_:.4f} h_cond={hc:.4f} "
              f"top1={res['cells'][f'{model}|{con}']['top1']:.4f}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
