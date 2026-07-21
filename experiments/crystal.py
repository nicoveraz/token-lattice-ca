"""Crystallization study: run the full instrument suite on every training checkpoint.
Watch attractor structure, order, and damage-healing FORM as the model learns."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time
import numpy as np
import jax
from model import init_params, load, save
from ca import run, metrics
from census import ngrams, validation, melt
from damage import damage
from train import batch_windows, eval_step, val_ids

CKPTS = [("0", None)] + [(str(s), f"ckpt/step{s}.npz") for s in
                         [1000, 2000, 3000, 4000, 5000, 6000]]

def probe(params, tag):
    out = {}
    # val masked-center accuracy (r=2)
    rng = np.random.default_rng(3)
    x, y = batch_windows(val_ids, 2, 768, rng)
    acc, ce = eval_step(params, np.asarray(x), np.asarray(y), 5)
    out["val_acc"], out["val_ce"] = float(acc), float(ce)
    # order parameters at T=0.3 and 0.7 (r=2, random init)
    for T in [0.3, 0.7]:
        o = run(params, B=8, N=48, r=2, T=T, sweeps=80, mode="async",
                init="random", seed=31)
        out[f"bigram_T{T}"] = metrics(o["final"])["bigram_overlap"]
        out[f"act_T{T}"] = float(o["activity"][-10:].mean())
    # census-lite at T=0.7
    ids = np.load("data/train_ids.npy").tolist()
    corpus_tri = ngrams([ids], 3)
    o = run(params, B=16, N=48, r=2, T=0.7, sweeps=100, mode="async",
            init="random", seed=37)
    tri = ngrams(o["snaps"][-30:].reshape(-1, 48).tolist(), 3)
    ov, rho, nsh = validation(tri, corpus_tri)
    out["census_overlap50"], out["census_spearman"] = ov, rho
    # damage healing at T=0.3, r=1
    cone = damage(params, 0.3, 1, B=8, N=48, sweeps=40, seed=41)
    out["damage_T0.3_r1"] = float(cone[-5:].mean())
    # melting retention at T=0.3
    keep = melt(params, 0.3, r=2, B=8, N=48, sweeps=40, seed=43)
    out["melt_retention"] = float(keep[-1])
    print(tag, json.dumps({k: round(v, 3) for k, v in out.items()}), flush=True)
    return out

if __name__ == "__main__":
    results = {}
    for tag, path in CKPTS:
        t0 = time.time()
        params = init_params(jax.random.PRNGKey(0)) if path is None else load(path)
        results[tag] = probe(params, tag)
        results[tag]["secs"] = round(time.time() - t0, 1)
    json.dump(results, open("results/crystal.json", "w"), indent=1)
    print("CRYSTAL DONE", flush=True)
