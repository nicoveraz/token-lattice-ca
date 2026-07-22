"""Damage spreading via twin runs with common random numbers (CRN).

Two probes:
  damage()        single-site flip -> damage cone (kept for backward compat and
                  the single-vs-block apparatus comparison). F8: single-site
                  ignition is all-or-nothing because the masked-center rule means
                  a flipped site never influences its own resampling.
  block_damage()  DEFAULT (Phase 2): flip a contiguous block of `block` sites,
                  ensemble B>=64, and report IGNITION PROBABILITY separately from
                  CONDITIONAL SPREAD (mean damage | ignited). Single-site probes
                  conflate a rare-ignition Bernoulli with the spread magnitude.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time
import numpy as np
import ca
from model import load
from ca import run


def _twin_cones(params, T, r, flip_idx, B, N, settle, sweeps, seed):
    """Settle from random soup, flip `flip_idx` sites at center, run CRN twins.
    Returns diff (sweeps+1, B, N) recentered on the block center."""
    rng = np.random.default_rng(seed)
    s = run(params, B=B, N=N, r=r, T=T, sweeps=settle, mode="async",
            init="random", seed=seed)
    base = s["final"]
    flipped = base.copy()
    V = ca._vocab()
    for c in flip_idx:
        flipped[:, c] = rng.integers(ca.INIT_LO, V, size=B)
    u = np.random.default_rng(seed + 1).random(sweeps * N * B)
    a = run(params, B=B, N=N, r=r, T=T, sweeps=sweeps, mode="async",
            init_state=base, seed=seed + 2, u_stream=u)
    b = run(params, B=B, N=N, r=r, T=T, sweeps=sweeps, mode="async",
            init_state=flipped, seed=seed + 2, u_stream=u)
    diff = (a["snaps"] != b["snaps"])                     # (sweeps+1, B, N) bool
    shift = N // 2 - flip_idx[len(flip_idx) // 2]
    return np.roll(diff, shift, axis=2)


def damage(params, T, r, B=16, N=48, sweeps=60, seed=21):
    """Single-site flip -> damage cone (sweeps+1, N). Backward compatible."""
    diff = _twin_cones(params, T, r, [N // 2], B, N, settle=30,
                       sweeps=sweeps, seed=seed)
    return diff.mean(axis=1)


def block_damage(params, T, r, block=3, B=64, N=48, settle=30, sweeps=60,
                 seed=21, ignite_thresh=0.05, tail=5):
    """3-site (default) block flip, B>=64. Separates ignition prob from spread."""
    c = N // 2
    flip_idx = [c + k for k in range(-(block // 2), block - block // 2)]
    diff = _twin_cones(params, T, r, flip_idx, B, N, settle, sweeps, seed)
    cone = diff.mean(axis=1)                              # (sweeps+1, N)
    final = diff[-tail:].mean(axis=(0, 2))               # (B,) per-lattice final damage
    ignited = final > ignite_thresh
    return dict(
        cone=cone,
        final_damage=final,
        ignition_prob=float(ignited.mean()),
        cond_spread=float(final[ignited].mean()) if ignited.any() else 0.0,
        mean_damage=float(final.mean()),                 # = the pilot's bdmg (unconditional)
        n_ignited=int(ignited.sum()), B=B,
    )


if __name__ == "__main__":
    params = load("ckpt/final.npz")
    TS, RS = [0.3, 0.7, 1.5], [1, 4, 16]
    B = 64
    cones, stats = {}, {}
    for T in TS:
        for r in RS:
            t0 = time.time()
            d = block_damage(params, T, r, block=3, B=B, seed=21)
            cones[f"T{T}_r{r}"] = d["cone"]
            stats[f"T{T}_r{r}"] = {k: d[k] for k in
                                   ("ignition_prob", "cond_spread", "mean_damage", "n_ignited", "B")}
            print(f"T{T} r{r}: ignition_prob={d['ignition_prob']:.3f} "
                  f"cond_spread={d['cond_spread']:.3f} mean={d['mean_damage']:.3f} "
                  f"({d['n_ignited']}/{B})  {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed("results/damage_block.npz", **cones)
    json.dump(stats, open("results/damage_block.json", "w"), indent=1)
    print("DAMAGE(block) DONE", flush=True)
