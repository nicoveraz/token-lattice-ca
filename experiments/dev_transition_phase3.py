"""Phase 3 -- does the headline developmental transition survive proper statistics?

The paper's intended headline (F25/crosslevel_dev.json) is a black-box developmental phase
transition at training step ~1000: lambda_ca -0.08 -> +0.19 and D_norm 0.10 -> 0.71 across
Pythia checkpoints, detectable with NO weight access. It currently rests on 2 seeds at
N=48 -- exactly the pseudoreplication the audit flagged as most damaging (W1).

This re-runs it against the three cheap audit objections:
  W1  >= 8 INDEPENDENT seeds, per-seed values retained (never just means), and the run --
      not the lattice -- treated as the unit of analysis. Lattices within a run share an
      init and a uniform stream, so they are NOT independent samples.
  W9  a second lattice size (N=96) to test size-stability.
  W8  Benjamini-Hochberg FDR across the whole family of tests reported here, with raw and
      adjusted p-values side by side, and the family stated explicitly.

Pre-registered before running:
  * Primary claim: lambda_ca(step>=1000) > lambda_ca(step<=512), i.e. a sign change from
    healing to spreading, tested per lattice size with a two-sided Mann-Whitney U over
    RUN-level values (seeds as replicates).
  * Secondary: the same for D_norm.
  * If the primary fails at either N, the headline is DEMOTED. That is a successful
    outcome of this phase, not a failure.

Incremental save + resume (safe to kill). Writes results/dev_transition_phase3.json.
Usage:  caffeinate -i .venv/bin/python experiments/dev_transition_phase3.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from scipy import stats
from lyapunov import lyap_from_cone

BASE = "EleutherAI/pythia-410m"
# checkpoints spanning the claimed transition (pre/post step ~1000)
STEPS = ["step256", "step512", "step1000", "step2000", "step8000", "step143000"]
PRE = {"step256", "step512"}                 # pre-registered "before" set
SEEDS = [21, 22, 23, 24, 25, 26, 27, 28]     # 8 independent seeds (W1)
SIZES = [(48, 16), (96, 8)]                  # (N, B) -- B halved at N=96 for 16GB (W9 trade)
R, T = 2, 0.7
FIT_KW = dict(sat_threshold=3.5, frac_of_max=0.5, max_sweeps=8, min_sweeps=3)
OUT = str(_ROOT / "results" / "dev_transition_phase3.json")


def measure(revision, N, B, seed, base=None, r=None, temp=None):
    """One run -> (lambda_ca, D_norm). Run is the unit of analysis, not the lattice.

    `base`, `r` and `temp` default to the module constants; they are parameters so that
    `dev_transition_scale.py` can drive the IDENTICAL protocol on other model sizes.
    Cross-model comparison is only meaningful if the measurement path is literally the
    same code, so that script imports this function rather than copying it.
    """
    from ar_ca import ARRule
    from ar_probe import block_damage, drift_floor
    rr, tt = (R if r is None else r), (T if temp is None else temp)
    rule = ARRule(base or BASE, revision=revision)
    try:
        d = block_damage(rule, tt, rr, block=3, B=B, N=N, settle=12, sweeps=22,
                         seed=seed, scheme="none")
        lam = lyap_from_cone(d["cone"], N, **FIT_KW)[0]
        d0, _ = drift_floor(rule, tt, rr, B=B, N=N, settle=12, sweeps=22,
                            seed=seed, scheme="none")
        dn = d["mean_damage"] / max(d0, 1e-3)
        # F42: lambda is undefined when damage never ignites, so the raw damage and the
        # per-lattice ignition fraction must be RECORDED, not inferred from lambda later.
        # `ignition_prob` is already computed by block_damage -- it was being discarded.
        md, ig = d["mean_damage"], d["ignition_prob"]
    finally:
        rule.model = None; del rule; gc.collect()
        try: torch.mps.empty_cache()
        except Exception: pass
    return float(lam), float(dn), float(md), float(ig)


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return [float(x) for x in adj]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    runs = res["runs"]
    todo = [(N, B, st, sd) for (N, B) in SIZES for st in STEPS for sd in SEEDS]
    print(f"Phase 3 developmental transition: {len(todo)} runs "
          f"({len(STEPS)} ckpts x {len(SEEDS)} seeds x {len(SIZES)} sizes)", flush=True)
    for k, (N, B, st, sd) in enumerate(todo, 1):
        key = f"N{N}_{st}_s{sd}"
        if key in runs:
            continue
        t0 = time.time()
        lam, dn, md, ig = measure(st, N, B, sd)
        runs[key] = dict(N=N, step=int(st.replace("step", "")), seed=sd,
                         lambda_ca=round(lam, 5), D_norm=round(dn, 5),
                         mean_damage=md, ignition_prob=round(ig, 5),
                         secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: lam={lam:+.4f} D_norm={dn:.4f} "
              f"({runs[key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    # ---- analysis: run-level, per size, pre vs post ----
    tests, summary = [], {}
    for (N, _B) in SIZES:
        for metric in ("lambda_ca", "D_norm"):
            pre = [v[metric] for v in runs.values()
                   if v["N"] == N and f"step{v['step']}" in PRE]
            post = [v[metric] for v in runs.values()
                    if v["N"] == N and f"step{v['step']}" not in PRE]
            if len(pre) < 3 or len(post) < 3:
                continue
            u = stats.mannwhitneyu(post, pre, alternative="two-sided")
            summary[f"N{N}_{metric}"] = dict(
                pre_mean=round(float(np.mean(pre)), 4), pre_n=len(pre),
                post_mean=round(float(np.mean(post)), 4), post_n=len(post),
                pre_per_seed=[round(x, 4) for x in pre],
                post_per_seed=[round(x, 4) for x in post])
            tests.append((f"N{N}_{metric}_post_vs_pre", float(u.pvalue)))
    if tests:
        names, praw = zip(*tests)
        padj = bh_fdr(praw)
        res["family"] = ("all pre-vs-post tests reported here: "
                         "{lambda_ca, D_norm} x {N=48, N=96}")
        res["tests"] = [dict(name=n, p_raw=round(pr, 6), p_bh=round(pa, 6),
                             significant_bh_05=bool(pa < 0.05))
                        for n, pr, pa in zip(names, praw, padj)]
        res["summary"] = summary
        print("\n=== pre (step<=512) vs post (step>=1000), run-level ===")
        for k, v in summary.items():
            print(f"  {k:16s} pre {v['pre_mean']:+.4f} (n={v['pre_n']}) -> "
                  f"post {v['post_mean']:+.4f} (n={v['post_n']})")
        print("\n=== BH-FDR across the family ===")
        for t in res["tests"]:
            print(f"  {t['name']:28s} p_raw={t['p_raw']:.5f}  p_BH={t['p_bh']:.5f}  "
                  f"{'SURVIVES' if t['significant_bh_05'] else 'n.s.'}")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
