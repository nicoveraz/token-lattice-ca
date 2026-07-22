"""Phase C2 calibration: generate a synthetic Markov source with a KNOWN transition
matrix P. A tiny model trained on it, then censused, lets us measure corpus recovery
*quantitatively* against ground truth (the honest trained-model analog of a sampler
oracle) -- unlike the proxy census, here we know the exact bigram statistics.

Vocab: id 0=<mask>, 1=<unk> (unused), 2..K+1 = content symbols. First-order Markov
with a sparse random P (each state -> ~SUCC successors). -> data_markov/
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, os
import numpy as np

K = 60                 # content symbols
V = 64                 # total vocab (0=mask,1=unk, 2..61 content)
SUCC = 6               # successors per state
NTOK = 300_000
LO = 2                 # first content id


def build(seed=0, out="data_markov"):
    OUT = out
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = np.arange(LO, LO + K)
    P = np.zeros((V, V))
    for c in ids:
        succ = rng.choice(ids, size=SUCC, replace=False)
        w = rng.dirichlet(np.ones(SUCC) * 0.6)            # skewed, learnable
        P[c, succ] = w
    # generate the chain
    seq = np.empty(NTOK, dtype=np.int32)
    cur = int(rng.choice(ids))
    for t in range(NTOK):
        seq[t] = cur
        cur = int(rng.choice(V, p=P[cur]))
    n_val = int(0.05 * NTOK)
    np.save(f"{OUT}/train_ids.npy", seq[:-n_val])
    np.save(f"{OUT}/val_ids.npy", seq[-n_val:])
    np.save(f"{OUT}/P.npy", P)
    json.dump(["<mask>", "<unk>"] + [f"s{i}" for i in range(K)], open(f"{OUT}/vocab.json", "w"))
    # sanity: empirical bigram vs P
    emp = np.zeros((V, V))
    np.add.at(emp, (seq[:-1], seq[1:]), 1.0)
    row = emp.sum(1, keepdims=True); row[row == 0] = 1
    Q = emp / row
    tv = 0.5 * np.abs(Q[LO:LO+K] - P[LO:LO+K]).sum(1).mean()
    meta = dict(K=K, V=V, succ=SUCC, tokens=NTOK,
                empirical_vs_P_TV=round(float(tv), 4),
                entropy_bits=round(float(-(P[LO:LO+K] * np.log2(P[LO:LO+K] + 1e-12)).sum(1).mean()), 3))
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(json.dumps(meta))
    print("MARKOV CORPUS DONE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data_markov")
    a = ap.parse_args()
    build(a.seed, a.out)
