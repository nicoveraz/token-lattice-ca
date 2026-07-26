"""Phase 4.0 -- read the completed developmental re-test and decide the SHAPE of the curve.

`dev_transition_phase3.py` answered the pre-registered question (is post > pre?) and its
BH-FDR table is in `logs/phase3_dev.log`. It did not ask what shape the rise has. The N=48
arm made clear that the curve is not a step: it overshoots at step 1000 and settles lower.

That matters for what may be quoted, AT BOTH ENDS. Reporting step 1000 as the post value
takes the transition's peak for its level. Reporting step 256 alone as the pre value does
the same thing at the other end -- and by more: it inflates lambda_ca's effect by 1.7x
against 1.4x for the peak. The first version of this script fixed the post end and committed
the pre-end error in the same edit, while declaring `pre: [256, 512]` in the JSON it wrote.
Both ends now use the PRE-REGISTERED sets. The unregistered variants are retained under
explicit _INFLATED / _UNREGISTERED keys so the difference stays auditable.

Because any effect size depends on which checkpoints are called "pre", the script also
reports a statement that needs no such choice: the SIGN AGREEMENT across all 96 runs.

Tests whether the step-1000 peak is separable from the plateau, and reports the N=48 vs
N=96 comparison objection W9 asks for -- as an equivalence BOUND, not a null p-value.

Reads results/dev_transition_phase3.json (never writes it); writes
results/dev_transition_shape.json. Pure numpy/scipy, seconds.
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
import numpy as np
from scipy import stats

SRC = ROOT / "results" / "dev_transition_phase3.json"
OUT = ROOT / "results" / "dev_transition_shape.json"
PRE = {256, 512}
PEAK = {1000}
PLATEAU = {2000, 8000, 143000}          # "settled": everything after the overshoot
STEPS = [256, 512, 1000, 2000, 8000, 143000]
METRICS = ("lambda_ca", "D_norm")


def load():
    d = json.load(open(SRC))
    rows = d if isinstance(d, list) else d.get("runs", d)
    if isinstance(rows, dict):
        rows = list(rows.values())
    rows = [r for r in rows if isinstance(r, dict) and "lambda_ca" in r]
    if len(rows) != 96:
        print(f"WARNING: expected 96 runs, found {len(rows)} -- results are partial")
    return rows


def cohens_d(a, b):
    """Standardized mean difference b - a, pooled SD."""
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / sp)


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, p[idx] * n / rank)
        adj[idx] = running
    return adj


def main():
    rows = load()
    sel = lambda N, steps, m: np.array([r[m] for r in rows
                                        if r["N"] == N and r["step"] in steps])
    out = {"cells": {}, "headline": {}, "peak_vs_plateau": {}, "size_scaling_W9": {},
           "variance": {}}

    print("=== per-cell summary ===")
    for m in METRICS:
        print(f"\n  {m}")
        print(f"  {'N':>4} {'step':>7} {'n':>3} {'mean':>9} {'sd':>8} {'min':>9} {'max':>9}")
        for N in (48, 96):
            for st in STEPS:
                v = sel(N, {st}, m)
                out["cells"][f"N{N}_step{st}_{m}"] = dict(
                    n=len(v), mean=round(float(v.mean()), 4),
                    sd=round(float(v.std(ddof=1)), 4),
                    min=round(float(v.min()), 4), max=round(float(v.max()), 4))
                print(f"  {N:>4} {st:>7} {len(v):>3} {v.mean():>+9.4f} "
                      f"{v.std(ddof=1):>8.4f} {v.min():>+9.4f} {v.max():>+9.4f}")

    print("\n=== HEADLINE: the PRE-REGISTERED pre set {256,512} vs the POOLED PLATEAU ===")
    print("    Both ends matter. Using the step-1000 PEAK as the post value inflates the")
    print("    effect; using step 256 ALONE as the pre value inflates it at the other end,")
    print("    and by MORE (up to 1.7x on lambda_ca). PRE is the pre-registered set.")
    praw, keys = [], []
    for N in (48, 96):
        for m in METRICS:
            a, b = sel(N, PRE, m), sel(N, PLATEAU, m)
            p = float(stats.mannwhitneyu(b, a, alternative="two-sided").pvalue)
            praw.append(p); keys.append(f"N{N}_{m}")
            d_lowest = cohens_d(sel(N, {256}, m), b)      # step256-only: NOT pre-registered
            out["headline"][f"N{N}_{m}"] = dict(
                pre_mean=round(float(a.mean()), 4), plateau_mean=round(float(b.mean()), 4),
                n_pre=len(a), n_plateau=len(b), cohens_d=round(cohens_d(a, b), 2),
                p_raw=p,
                cohens_d_vs_peak_INFLATED=round(cohens_d(a, sel(N, PEAK, m)), 2),
                cohens_d_from_step256_only_UNREGISTERED=round(d_lowest, 2))
            print(f"  N={N} {m:>10}  {a.mean():+.4f} -> {b.mean():+.4f}   "
                  f"d={cohens_d(a, b):.2f}   p={p:.2e}"
                  f"   [step256-only d would be {d_lowest:.2f}]")

    # --- the pre-set-free statement of the same result -------------------------------
    # Effect sizes depend on which checkpoints are called "pre". This does not: it uses
    # every run, and it is what panel C of the figure draws.
    print("\n=== SIGN AGREEMENT (uses all 96 runs; independent of the pre/post split) ===")
    out["sign_agreement"] = {}
    for N in (48, 96):
        pre, post = sel(N, PRE, "lambda_ca"), sel(N, PLATEAU, "lambda_ca")
        rec = dict(n_pre=len(pre), pre_negative=int((pre < 0).sum()),
                   pre_min=round(float(pre.min()), 4), pre_max=round(float(pre.max()), 4),
                   n_plateau=len(post), plateau_negative=int((post < 0).sum()),
                   plateau_min=round(float(post.min()), 4),
                   plateau_cv_pct=round(float(100 * post.std(ddof=1) / post.mean()), 1))
        out["sign_agreement"][f"N{N}"] = rec
        print(f"  N={N}: pre {rec['pre_negative']}/{rec['n_pre']} negative "
              f"(range {rec['pre_min']:+.3f} to {rec['pre_max']:+.3f})  |  "
              f"plateau {rec['plateau_negative']}/{rec['n_plateau']} negative "
              f"(min {rec['plateau_min']:+.4f}, CV {rec['plateau_cv_pct']}%)")
    allpost = np.concatenate([sel(48, PLATEAU, "lambda_ca"), sel(96, PLATEAU, "lambda_ca")])
    out["sign_agreement"]["pooled"] = dict(
        n=len(allpost), negative=int((allpost < 0).sum()),
        min=round(float(allpost.min()), 4))
    print(f"  pooled: {int((allpost < 0).sum())}/{len(allpost)} plateau runs negative, "
          f"min {allpost.min():+.4f}")
    for k, adj in zip(keys, bh_fdr(praw)):
        out["headline"][k]["p_bh"] = float(adj)

    print("\n=== is the step-1000 PEAK separable from the plateau? ===")
    pk_p, pk_keys = [], []
    for N in (48, 96):
        for m in METRICS:
            pk, pl = sel(N, PEAK, m), sel(N, PLATEAU, m)
            p = float(stats.mannwhitneyu(pk, pl, alternative="two-sided").pvalue)
            pk_p.append(p); pk_keys.append(f"N{N}_{m}")
            over = 100 * (pk.mean() - pl.mean()) / pl.mean()
            out["peak_vs_plateau"][f"N{N}_{m}"] = dict(
                peak=round(float(pk.mean()), 4), plateau=round(float(pl.mean()), 4),
                overshoot_pct=round(float(over), 1), cohens_d=round(cohens_d(pl, pk), 2),
                p_raw=p)
            print(f"  N={N} {m:>10}  {pk.mean():+.4f} vs {pl.mean():+.4f} "
                  f"({over:+5.1f}%)  d={cohens_d(pl, pk):>5.2f}  p={p:.4f}")
    for k, adj in zip(pk_keys, bh_fdr(pk_p)):
        out["peak_vs_plateau"][k]["p_bh"] = float(adj)
    surv = [k for k in pk_keys if out["peak_vs_plateau"][k]["p_bh"] < 0.05]
    print(f"  -> overshoot survives BH-FDR in {len(surv)}/4 cells: {surv or 'none'}")

    print("\n=== W9: does the effect fall with lattice size? ===")
    for m in METRICS:
        a48, b48 = sel(48, PRE, m), sel(48, PLATEAU, m)     # PRE, not the lowest checkpoint
        a96, b96 = sel(96, PRE, m), sel(96, PLATEAU, m)
        g48, g96 = float(b48.mean() - a48.mean()), float(b96.mean() - a96.mean())
        p_lvl = float(stats.mannwhitneyu(b48, b96, alternative="two-sided").pvalue)
        # equivalence BOUND, not a null p-value: a reviewer cannot object to a CI the way
        # they can to "p=0.91 therefore the same".
        se = float(np.sqrt(b48.var(ddof=1) / len(b48) + b96.var(ddof=1) / len(b96)))
        diff = float(b48.mean() - b96.mean())
        ci = (diff - 1.96 * se, diff + 1.96 * se)
        out["size_scaling_W9"][m] = dict(
            plateau_diff=round(diff, 4), plateau_diff_se=round(se, 4),
            plateau_diff_ci95=[round(ci[0], 4), round(ci[1], 4)],
            plateau_agree_within_pct=round(
                100 * max(abs(ci[0]), abs(ci[1])) / abs(float(b48.mean())), 1),
            gap_N48=round(g48, 4), gap_N96=round(g96, 4),
            retention_pct=round(100 * g96 / g48, 1),
            cohens_d_N48=round(cohens_d(a48, b48), 2),
            cohens_d_N96=round(cohens_d(a96, b96), 2),
            plateau_level_N48=round(float(b48.mean()), 4),
            plateau_level_N96=round(float(b96.mean()), 4),
            plateau_level_differs_p=p_lvl)
        print(f"  {m:>10}  gap {g48:+.4f} (N=48) -> {g96:+.4f} (N=96)   "
              f"retention {100 * g96 / g48:.0f}%   d {cohens_d(a48, b48):.2f} -> "
              f"{cohens_d(a96, b96):.2f}")
        print(f"{'':>14}plateau LEVEL {b48.mean():+.4f} vs {b96.mean():+.4f}  p={p_lvl:.2e}")

    print("\n=== variance collapse across the transition (observation, not pre-registered) ===")
    for N in (48, 96):
        pre, post = sel(N, PRE, "lambda_ca"), sel(N, PEAK | PLATEAU, "lambda_ca")
        lev = float(stats.levene(pre, post).pvalue)
        out["variance"][f"N{N}"] = dict(
            sd_pre=round(float(pre.std(ddof=1)), 4), sd_post=round(float(post.std(ddof=1)), 4),
            ratio=round(float(pre.std(ddof=1) / post.std(ddof=1)), 2), levene_p=lev)
        print(f"  N={N}: sd(lambda) {pre.std(ddof=1):.4f} -> {post.std(ddof=1):.4f} "
              f"({pre.std(ddof=1) / post.std(ddof=1):.1f}x)  Levene p={lev:.2e}")

    out["_note"] = (
        "Shape analysis of the completed developmental re-test. The pre-registered post>pre "
        "test and its BH-FDR table live in dev_transition_phase3.py; this file adds the shape. "
        "HEADLINE EFFECT SIZES ARE step256 vs the POOLED PLATEAU (2000/8000/143000), not vs "
        "the step-1000 peak: the curve overshoots and settles, so quoting the peak as the "
        "level inflates the effect. The vs-peak numbers are stored under "
        "cohens_d_vs_peak_INFLATED purely so the difference is auditable -- do not quote them. "
        "Variance collapse was NOT pre-registered and is reported as an observation.")
    out["_definitions"] = dict(pre=sorted(PRE), peak=sorted(PEAK), plateau=sorted(PLATEAU))
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
