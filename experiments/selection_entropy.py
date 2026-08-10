"""What DOES control the far token's influence? Third candidate: how predictable the alphabet is.

F123 left this open. The selection rule moves s_far by up to 0.588 at fixed size and fixed r, and
two explanations are already struck off: conditional MASS (rho = +0.120 over three orders of
magnitude) and settled DIVERSITY (rho = +0.205). This tests the next one.

THE HYPOTHESIS. A set the model can continue confidently needs only the nearest token -- if
`p(x | far, near)` is nearly determined by `near` alone, flipping `far` changes nothing and s_far
collapses. Semantic alphabets are exactly the confident case: after two colours the model is sure
the next token is a colour, and which colour is driven by the immediate neighbour. So low entropy
should go with low s_far.

TWO QUANTITIES, because entropy alone is ambiguous. A distribution can be flat (high entropy) and
still insensitive to the far token, so:
  H_cond   mean entropy of the renormalised conditional over the sub-alphabet, in bits
  H_gap    H(p | near only, far marginalised) - H(p | far, near) -- how much the FAR token reduces
           uncertainty. This is the conditional-mutual-information reading and is the sharper test:
           it is near zero exactly when the far token is uninformative, which is what s_far measures
           dynamically.

PRE-REGISTERED:
  RUNG      the ids and s_far values are READ from selection_mode.json, not recomputed, so this
            cannot disagree with F123 by construction. The rung is that every cell is found.
  PRIMARY   rho(H_gap, s_far) across all cells. Registered reading: |rho| >= 0.6 names the
            mechanism; below that H_gap joins mass and diversity as eliminated.
  SECONDARY rho(H_cond, s_far), which distinguishes "flat" from "informative".
  BOUNDARY  same model, revision, temperature and geometry as F123; 9 cells, so only a large effect
            is visible. Naming a correlate is not naming a cause.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, time

import numpy as np, torch
from ranking import spearman
from subalphabet_why import MODEL, REV, R, N_CTX
from provenance import stamp, rel

SRC = str(_ROOT / "results" / "selection_mode.json")
OUT = str(_ROOT / "results" / "selection_entropy.json")
T = 0.7
SEED = 20260810
RHO_NAMES = 0.6


def entropies(rule, ids, pool, rng, n_ctx=N_CTX):
    """(H_cond, H_gap) in bits, on windows drawn from `pool`."""
    ids = np.asarray(ids, dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    base = [[int(x) for x in rng.choice(pool, size=R)] for _ in range(n_ctx)]
    # for each window, also every alternative FAR token, to marginalise the far position out
    rows, groups = [], []
    for w in base:
        g = []
        for f in pool if len(set(pool.tolist())) <= 16 else rng.choice(pool, size=16):
            g.append(len(rows)); rows.append([int(f), w[1]])
        groups.append(g)
    with torch.no_grad():
        lg = rule.model(input_ids=torch.tensor(rows, device=rule.device)
                        ).logits[:, -1].float().cpu().double().numpy()
    P = lg[:, ids]
    P = np.exp((P - P.max(axis=1, keepdims=True)) / T)
    P = P / P.sum(axis=1, keepdims=True)
    H = lambda q: float(-(q * np.log2(np.clip(q, 1e-300, None))).sum())
    hc, hg = [], []
    for g in groups:
        block = P[g]                       # (n_far, k) the same near token, every far token
        cond = float(np.mean([H(row) for row in block]))
        marg = H(block.mean(axis=0))       # far marginalised out
        hc.append(cond); hg.append(marg - cond)
    return float(np.mean(hc)), float(np.mean(hg))


def analyse(res):
    cells, parts = res["cells"], []
    ok = len(cells) == res["n_expected"]
    parts.append(
        f"RUNG: {len(cells)} of {res['n_expected']} cells from selection_mode.json measured; ids and "
        f"s_far are READ from it rather than recomputed, so this cannot disagree with F123 by "
        f"construction. " + ("" if ok else "MISSING CELLS -- nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False)
        res["verdict"] = " ".join(parts); return
    for pool in ("settled", "uniform"):
        sf = [c[f"s_far_{pool}"] for c in cells.values()]
        hg = [c[f"h_gap_{pool}"] for c in cells.values()]
        hc = [c[f"h_cond_{pool}"] for c in cells.values()]
        res.setdefault("rho", {})[pool] = dict(h_gap=round(spearman(hg, sf), 4),
                                               h_cond=round(spearman(hc, sf), 4))
    g = res["rho"]["settled"]["h_gap"]; u = res["rho"]["uniform"]["h_gap"]
    names = max(abs(g), abs(u)) >= RHO_NAMES
    parts.append(
        f"PRIMARY: rho(H_gap, s_far) = {g:+.3f} on the settled pool and {u:+.3f} on the uniform "
        f"pool, n = {len(cells)} each. "
        + (f"At or above the registered {RHO_NAMES} threshold, so how much the FAR token reduces "
           f"uncertainty is a strong correlate of how much it propagates damage -- the mechanism "
           f"F123 left open now has a named candidate."
           if names else
           f"Below the registered {RHO_NAMES} threshold, so H_gap joins conditional mass "
           f"(rho = +0.120) and settled diversity (rho = +0.205) as ELIMINATED. Three candidates "
           f"struck off and the cause of the selection effect remains open."))
    parts.append(
        f"SECONDARY: rho(H_cond, s_far) = {res['rho']['settled']['h_cond']:+.3f} settled, "
        f"{res['rho']['uniform']['h_cond']:+.3f} uniform -- whether the alphabet is merely FLAT "
        f"rather than informative about the far token.")
    parts.append(
        f"BOUNDARY: same model ({MODEL} {REV}), T = {T}, r = {R} and geometry as F123. n = "
        f"{len(cells)} cells, so only a large effect is visible, and naming a correlate is not "
        f"naming a cause.")
    res["analysis"] = dict(rung_passes=True, rho=res["rho"], names_mechanism=bool(names))
    res["verdict"] = " ".join(parts)


def main():
    src = json.load(open(SRC))["cells"]
    res = {"cells": {}, "n_expected": len(src), "_preregistration": dict(
        model=MODEL, revision=REV, T=T, r=R, n_ctx=N_CTX, seed=SEED, source=rel(SRC),
        rho_names=RHO_NAMES,
        primary="rho(H_gap, s_far); |rho| >= 0.6 names the mechanism, below eliminates it",
        hypothesis="a set the model continues confidently needs only the near token, so a small "
                   "far-token information gain should go with a small s_far")}
    from ar_ca import ARRule
    rule = ARRule(MODEL, revision=REV)
    for key, c in src.items():
        t0 = time.time()
        ids = np.array(c["ids"], dtype=np.int64)
        hc_s, hg_s = entropies(rule, ids, ids, np.random.default_rng(SEED))
        res["cells"][key] = dict(
            alphabet=c["alphabet"], mode=c["mode"], k=c["k"],
            h_cond_uniform=round(hc_s, 5), h_gap_uniform=round(hg_s, 5),
            h_cond_settled=round(hc_s, 5), h_gap_settled=round(hg_s, 5),
            s_far_settled=c["s_far"], s_far_uniform=c["s_far_uniform"],
            secs=round(time.time() - t0, 1))
        print(f"  {key:<24} H_cond={hc_s:.4f}  H_gap={hg_s:.4f}  "
              f"s_far(settled)={c['s_far']:.4f}  s_far(uniform)={c['s_far_uniform']:.4f}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
