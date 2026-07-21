"""Damage spreading: twin runs with common random numbers, one-site flip."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time
import numpy as np
from model import load
from ca import run

def damage(params, T, r, B=16, N=48, sweeps=60, seed=21):
    rng = np.random.default_rng(seed)
    V = 2000
    init = rng.integers(2, V, size=(B, N)).astype(np.int32)
    # corpus-ish start: run 30 settle sweeps first, then flip
    settle = run(params, B=B, N=N, r=r, T=T, sweeps=30, mode="async",
                 init="random", seed=seed)
    base = settle["final"]
    flipped = base.copy()
    c = N // 2
    flipped[:, c] = rng.integers(2, V, size=B)
    n_up = sweeps * N * B
    u = np.random.default_rng(seed + 1).random(n_up)
    a = run(params, B=B, N=N, r=r, T=T, sweeps=sweeps, mode="async",
            init_state=base, seed=seed + 2, u_stream=u)
    b = run(params, B=B, N=N, r=r, T=T, sweeps=sweeps, mode="async",
            init_state=flipped, seed=seed + 2, u_stream=u)
    diff = (a["snaps"] != b["snaps"])          # (sweeps+1, B, N)
    # recenter on flip site
    diff = np.roll(diff, N // 2 - c, axis=2)
    cone = diff.mean(axis=1)                   # (sweeps+1, N) damage prob
    return cone

if __name__ == "__main__":
    params = load("ckpt/final.npz")
    cones = {}
    for T in [0.3, 0.7, 1.5]:
        for r in [1, 4, 16]:
            t0 = time.time()
            cones[f"T{T}_r{r}"] = damage(params, T, r)
            print(f"T{T} r{r} done {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed("results/damage.npz", **cones)
    print("DAMAGE DONE", flush=True)
