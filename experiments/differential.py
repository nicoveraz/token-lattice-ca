"""Differential CRN probing: twin lattices sharing init, update order, and
uniforms — differing in exactly ONE factor. Divergence is attributable to that
factor alone.  Arms: model-diff (checkpoints), null (nothing), apparatus
(CDF-ordering of the sampling coupling)."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json
import numpy as np
import jax
import ca
from model import init_params, load
from ca import run

def _sample_perm(perm):
    """Alternative coupling: same distribution, CDF built in permuted token order."""
    def f(probs, u):
        p = np.asarray(probs)[:, perm]
        cdf = np.cumsum(p, axis=-1); cdf /= cdf[:, -1:]
        idx = np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))])
        return perm[idx].astype(np.int32)
    return f

def coupled(pa, pb, T, sample_b=None, B=16, N=48, sweeps=40, seed=71):
    rng = np.random.default_rng(seed)
    init = rng.integers(2, 2000, size=(B, N)).astype(np.int32)
    u = np.random.default_rng(seed + 1).random(sweeps * N * B)
    a = run(pa, B=B, N=N, r=2, T=T, sweeps=sweeps, mode="async",
            init_state=init, seed=seed + 2, u_stream=u)
    orig = ca._sample
    if sample_b is not None:
        ca._sample = sample_b
    try:
        b = run(pb, B=B, N=N, r=2, T=T, sweeps=sweeps, mode="async",
                init_state=init, seed=seed + 2, u_stream=u)
    finally:
        ca._sample = orig
    d = (a["snaps"] != b["snaps"]).mean(axis=(1, 2))
    return d

if __name__ == "__main__":
    final = load("ckpt/final.npz")
    perm = np.random.default_rng(5).permutation(2000)
    out = {}
    for T in [0.3, 0.7]:
        # signal arm: model differs, everything else common-moded away
        for tag, path in [("step0", None), ("step1000", "ckpt/step1000.npz"),
                          ("step3000", "ckpt/step3000.npz"),
                          ("step5000", "ckpt/step5000.npz")]:
            pb = init_params(jax.random.PRNGKey(0)) if path is None else load(path)
            d = coupled(final, pb, T)
            out[f"model:{tag}_T{T}"] = [round(float(d[i]), 3) for i in (5, 20, -1)]
            print(f"T={T} model-diff {tag}: d5={d[5]:.3f} d20={d[20]:.3f} dEnd={d[-1]:.3f}", flush=True)
        # null test: identical everything -> must be exactly zero
        d = coupled(final, final, T)
        out[f"null_T{T}"] = float(d.max())
        print(f"T={T} NULL max divergence = {d.max():.6f}", flush=True)
        # apparatus arm: same model, same u, different sampling coupling
        d = coupled(final, final, T, sample_b=_sample_perm(perm))
        out[f"apparatus:cdfperm_T{T}"] = [round(float(d[i]), 3) for i in (5, 20, -1)]
        print(f"T={T} apparatus (CDF perm): d5={d[5]:.3f} d20={d[20]:.3f} dEnd={d[-1]:.3f}", flush=True)
    json.dump(out, open("results/differential.json", "w"), indent=1)
    print("DIFFERENTIAL DONE")
