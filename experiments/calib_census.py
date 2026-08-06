"""Phase C2 calibration + discrimination. For each synthetic Markov source X (known
transition matrix P_X, a tiny model trained on it), run the CA to quasi-stationarity
and read off the empirical bigram transition Q_X. Then:
  recovery      TV(Q_X, P_X) -- quantitative: how close is the censused bigram
                distribution to ground truth?
  discrimination TV(Q_X, P_Y) for Y != X should be much larger -- model X's census
                matches ITS corpus and not the others.
This is the honest trained-model analog of a sampler oracle (cite/differentiate
2602.19619). Usage: calib_census.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json
import numpy as np
import ca
from model import load
from ca import run

SOURCES = ["a", "b", "c"]
V, LO, K = 64, 2, 60


def censused_Q(data_dir, ckpt, T=0.5, r=1, B=32, N=64, sweeps=120, tail=40, seed=7):
    # Scoped, not assigned (#25). This function is called once per corpus, so leaving the module
    # configured for `data_dir` leaked the last caller's vocabulary and init floor into whatever
    # ran next in the same process -- invisibly, because a wrong init floor still yields a
    # well-formed lattice drawn from the wrong support.
    with ca.using(data_dir=data_dir, vocab=V, init_lo=LO):
        params = load(ckpt)
        out = run(params, B=B, N=N, r=r, T=T, sweeps=sweeps, mode="async", init="random",
                  seed=seed)
        snaps = out["snaps"][-tail:].reshape(-1, N)
        Q = np.zeros((V, V))
        np.add.at(Q, (snaps[:, :-1].ravel(), snaps[:, 1:].ravel()), 1.0)
        row = Q.sum(1, keepdims=True); row[row == 0] = 1
        return Q / row


def tv(Q, P):
    """Mean row total-variation over content states that P actually uses."""
    used = P[LO:LO + K].sum(1) > 0
    return float((0.5 * np.abs(Q[LO:LO + K] - P[LO:LO + K]).sum(1))[used].mean())


if __name__ == "__main__":
    Ps = {x: np.load(f"data_markov_{x}/P.npy") for x in SOURCES}
    Qs = {}
    for x in SOURCES:
        Qs[x] = censused_Q(f"data_markov_{x}", f"ckpt_markov_{x}/final.npz")
        print(f"censused Q_{x}", flush=True)
    # baseline: a random-lattice bigram distribution
    rng = np.random.default_rng(0)
    rl = rng.integers(LO, V, size=(200, 64))
    B0 = np.zeros((V, V)); np.add.at(B0, (rl[:, :-1].ravel(), rl[:, 1:].ravel()), 1.0)
    row = B0.sum(1, keepdims=True); row[row == 0] = 1; B0 = B0 / row

    TVm = {x: {y: round(tv(Qs[x], Ps[y]), 4) for y in SOURCES} for x in SOURCES}
    base = {x: round(tv(B0, Ps[x]), 4) for x in SOURCES}
    recovery = {x: TVm[x][x] for x in SOURCES}
    off = np.mean([TVm[x][y] for x in SOURCES for y in SOURCES if x != y])
    res = dict(TV_matrix=TVm, self_recovery=recovery, baseline_TV=base,
               mean_self=round(float(np.mean(list(recovery.values()))), 4),
               mean_cross=round(float(off), 4),
               discriminates=bool(np.mean(list(recovery.values())) < 0.5 * off))
    json.dump(res, open("results/calib_census.json", "w"), indent=1)
    print("\nTV(Q_X, P_Y)  (rows=censused model, cols=true P; diagonal=self-recovery):")
    print("        " + "  ".join(f"P_{y}" for y in SOURCES) + "   baseline")
    for x in SOURCES:
        print(f"  Q_{x}: " + "  ".join(f"{TVm[x][y]:.3f}" for y in SOURCES) + f"   {base[x]:.3f}")
    print(f"\nmean self-recovery TV={res['mean_self']}  mean cross TV={res['mean_cross']}  "
          f"discriminates={res['discriminates']}")
    print("CALIB CENSUS DONE")
