"""Attractor census + corpus-recovery validation + melting runs + cycle check."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time
from collections import Counter
import numpy as np
from model import load
from ca import run, metrics, decode

def ngrams(rows, n):
    c = Counter()
    for row in rows:
        for i in range(len(row) - n + 1):
            c[tuple(row[i:i + n])] += 1
    return c

def census_at(params, T, r=2, B=32, N=48, sweeps=150, seed=7):
    out = run(params, B=B, N=N, r=r, T=T, sweeps=sweeps, mode="async",
              init="random", seed=seed)
    tail = out["snaps"][-30:]                      # quasi-stationary window
    rows = tail.reshape(-1, N).tolist()
    tri = ngrams(rows, 3)
    # exact-state cycle check over last 60 recorded states, per lattice
    cyc = []
    for b in range(B):
        seen = {}
        for t, s in enumerate(out["snaps"][-60:]):
            h = s[b].tobytes()
            if h in seen:
                cyc.append(t - seen[h]); break
            seen[h] = t
    np.savez_compressed(f"results/census_T{T}_r{r}.npz",
                        snaps=out["snaps"].astype(np.int16))
    return tri, out["final"], cyc

def validation(tri_census, corpus_tri, k=50):
    top_c = [g for g, _ in tri_census.most_common(k)]
    top_k = [g for g, _ in corpus_tri.most_common(k)]
    ov = len(set(top_c) & set(top_k)) / k
    shared = set(tri_census) & set(corpus_tri)
    if len(shared) > 10:
        from scipy.stats import spearmanr
        a = [tri_census[g] for g in shared]
        b = [corpus_tri[g] for g in shared]
        rho = float(spearmanr(a, b).statistic)
    else:
        rho = float("nan")
    return ov, rho, len(shared)

def melt(params, T, r=2, B=16, N=48, sweeps=60, seed=11):
    out = run(params, B=B, N=N, r=r, T=T, sweeps=sweeps, mode="async",
              init="corpus", seed=seed)
    init = out["snaps"][0]
    keep = [(s == init).mean() for s in out["snaps"]]   # identity retention
    return keep

if __name__ == "__main__":
    itos = json.load(open("data/vocab.json"))
    params = load("ckpt/final.npz")
    ids = np.load("data/train_ids.npy").tolist()
    corpus_tri = ngrams([ids], 3)
    rand = np.random.default_rng(0)
    rand_rows = rand.integers(2, 2000, size=(960, 48)).tolist()  # baseline
    base_tri = ngrams(rand_rows, 3)

    results = {}
    for T in [0.3, 0.7, 1.0]:
        t0 = time.time()
        tri, final, cyc = census_at(params, T)
        ov, rho, nsh = validation(tri, corpus_tri)
        bov, brho, _ = validation(base_tri, corpus_tri)
        results[str(T)] = dict(overlap50=ov, spearman=rho, shared=nsh,
                               baseline_overlap50=bov, cycles=cyc,
                               secs=round(time.time() - t0, 1),
                               top_trigrams=[" ".join(itos[i] for i in g)
                                             for g, _ in tri.most_common(15)],
                               examples=[decode(final[b], itos) for b in range(4)])
        print(T, "done", results[str(T)]["secs"], "s", flush=True)

    melts = {str(T): melt(params, T) for T in [0.3, 0.7, 1.0, 1.5, 2.5]}
    json.dump(dict(census=results, melts=melts),
              open("results/census.json", "w"), indent=1)
    print("CENSUS DONE", flush=True)
