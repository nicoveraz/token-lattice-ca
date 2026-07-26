"""How far is this project's CRN coupling from the MAXIMAL coupling? Measure it, don't hedge.

Background. `src/dk.py` argued that drawing one uniform per site and thresholding it against
every probability yields Hinrichsen--Weitz--Domany's *maximal-correlation* member of the
admissible replica-coupling family, hence that every damage number here is a lower bound over
that family. **That argument is correct only on a binary alphabet.** The samplers in this
project use inverse-CDF sampling from a shared uniform, which is the MONOTONE (quantile)
coupling. On |V|=2 the monotone and maximal couplings coincide -- which is exactly why the
Domany--Kinzel identity in F38 is exact and why that rung is unaffected. On |V|>2 they come
apart: p=(0.5,0.5,0), q=(0,0.5,0.5) gives maximal agreement 0.5 and quantile agreement 0.

The direction matters. Maximal coupling MAXIMISES agreement, so it MINIMISES damage. Our
quantile coupling therefore reports *at least as much* damage as the maximal one -- so the
LM damage numbers are NOT a lower bound over the family. The old claim had the inequality
backwards for |V|>2 and is retracted.

This script replaces the hedge with a number, measured on real conditionals at the actual
operating point rather than on synthetic distributions:

    A_max   = sum_v min(p_v, q_v)          agreement under the maximal coupling (= 1 - TV)
    A_quant = sum_v |[F_p(v-1),F_p(v)) ^ [F_q(v-1),F_q(v))|   agreement under inverse-CDF
    gap     = A_max - A_quant >= 0         excess disagreement our coupling introduces

Pairs (p, q) are taken from a live CRN damage run: twin lattices, one block-flipped, sampled
at every site after the lattice has settled. Both twins are advanced with the SAME uniform
stream, so these are the conditionals the instrument actually compares.

Writes results/coupling_gap.json. Small model, CPU-friendly, seconds.
Usage:  .venv/bin/python experiments/coupling_gap.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

MODEL = "prajjwal1/bert-tiny"
B, N, R, SWEEPS, BLOCK = 8, 48, 2, 20, 3
TS = [0.7, 0.9]
SEED = 21
OUT = _ROOT / "results" / "coupling_gap.json"


def agreements(p, q):
    """(maximal, quantile) agreement probabilities for one pair of distributions."""
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / p.sum(); q = q / q.sum()
    a_max = float(np.minimum(p, q).sum())                      # = 1 - TV
    Fp = np.concatenate([[0.0], np.cumsum(p)])
    Fq = np.concatenate([[0.0], np.cumsum(q)])
    lo = np.maximum(Fp[:-1], Fq[:-1]); hi = np.minimum(Fp[1:], Fq[1:])
    a_quant = float(np.clip(hi - lo, 0.0, None).sum())
    return a_max, a_quant


def collect(rule, T):
    """Run CRN twins with a block flip; return (p, q) conditionals at every site, settled."""
    from mlm_ca import run as mlm_run
    rng = np.random.default_rng(SEED)
    init = rule.init_pool[rng.integers(0, len(rule.init_pool), size=(B, N))]
    flipped = init.copy()
    c = N // 2
    for j in range(c - BLOCK // 2, c - BLOCK // 2 + BLOCK):
        flipped[:, j % N] = rule.init_pool[rng.integers(0, len(rule.init_pool), size=B)]
    u = np.random.default_rng(SEED + 1).random(SWEEPS * N * B)
    kw = dict(B=B, N=N, r=R, T=T, sweeps=SWEEPS, mode="async", scheme="cls_sep",
              seed=71, u_stream=u)
    a = mlm_run(rule, init_state=init, **kw)["final"]
    b = mlm_run(rule, init_state=flipped, **kw)["final"]

    # go through the same adapter the loop uses, so these are the instrument's own conditionals
    from mlm_ca import _MLMAdapter
    ad = _MLMAdapter(rule, "cls_sep", None)
    rows = []
    for i in range(N):
        idx = ad.window(i, R, N)
        pa = np.asarray(_to_np(ad.probs(a[:, idx], T)), dtype=np.float64)
        pb = np.asarray(_to_np(ad.probs(b[:, idx], T)), dtype=np.float64)
        for bi in range(B):
            rows.append(agreements(pa[bi], pb[bi]))
    return np.array(rows), int((a != b).sum()), a.shape


def _to_np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    from mlm_ca import MLMRule
    rule = MLMRule(MODEL)
    print(f"coupling gap on {MODEL}: |V|={rule.V}, N={N}, B={B}, r={R}, "
          f"{SWEEPS} sweeps, block flip {BLOCK}")
    out = {}
    for T in TS:
        rows, ndiff, shape = collect(rule, T)
        a_max, a_quant = rows[:, 0], rows[:, 1]
        tv = 1.0 - a_max
        gap = a_max - a_quant
        # the region that decides propagation is where the twins nearly agree
        near = tv < 0.05
        out[f"T={T}"] = dict(
            n_pairs=int(len(rows)),
            n_differing_cells=ndiff,
            mean_TV=round(float(tv.mean()), 5),
            mean_disagree_maximal=round(float((1 - a_max).mean()), 5),
            mean_disagree_quantile=round(float((1 - a_quant).mean()), 5),
            mean_gap=round(float(gap.mean()), 5),
            median_gap=round(float(np.median(gap)), 5),
            max_gap=round(float(gap.max()), 5),
            inflation_factor=(None if (1 - a_max).mean() <= 0 else
                              round(float((1 - a_quant).mean() / (1 - a_max).mean()), 3)),
            near_agreement_subset=dict(
                definition="TV < 0.05 -- the region ahead of the damage front",
                n=int(near.sum()),
                mean_disagree_maximal=(None if not near.any() else
                                       round(float((1 - a_max)[near].mean()), 5)),
                mean_disagree_quantile=(None if not near.any() else
                                        round(float((1 - a_quant)[near].mean()), 5)),
                inflation_factor=(None if not near.any() or (1 - a_max)[near].mean() <= 0 else
                                  round(float((1 - a_quant)[near].mean()
                                              / (1 - a_max)[near].mean()), 2))),
        )
        r = out[f"T={T}"]
        print(f"\n  T={T}: {r['n_pairs']} real (p,q) pairs, {ndiff} differing cells")
        print(f"    mean TV                        {r['mean_TV']:.5f}")
        print(f"    mean disagreement, maximal     {r['mean_disagree_maximal']:.5f}")
        print(f"    mean disagreement, inverse-CDF {r['mean_disagree_quantile']:.5f}"
              f"   ({r['inflation_factor']}x)")
        n = r["near_agreement_subset"]
        print(f"    near-agreement subset (TV<0.05, n={n['n']}): "
              f"{n['mean_disagree_maximal']} -> {n['mean_disagree_quantile']} "
              f"({n['inflation_factor']}x)")

    out["_note"] = (
        "Measured excess disagreement of this project's CRN coupling (inverse-CDF from a "
        "shared uniform = the MONOTONE/quantile coupling) relative to the MAXIMAL coupling, "
        "on real conditionals from a live damage run. Retracts the earlier claim that our "
        "coupling is HWD's maximal-correlation member: that holds only on a binary alphabet "
        "(which is why the Domany-Kinzel rung, F38, is unaffected and still exact). Because "
        "maximal coupling minimises damage, our LM damage numbers are NOT a lower bound over "
        "the admissible family -- they sit inside it. The justified property of inverse-CDF "
        "is REPLICA-INDEPENDENCE: each replica's next state is a function of (its own state, "
        "the shared noise) alone, with no reference to its twin, so it extends consistently "
        "to >2 replicas and to a self-consistent damage field; a maximal coupling is defined "
        "only pairwise. The exact-zero null is untouched (p == q gives agreement 1 under any "
        "of these), as are all RELATIVE comparisons, where the coupling is a common mode.")
    out["_config"] = dict(model=MODEL, B=B, N=N, r=R, sweeps=SWEEPS, block=BLOCK,
                          temperatures=TS, seed=SEED)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
