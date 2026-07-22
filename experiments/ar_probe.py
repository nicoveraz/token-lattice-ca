"""Phase C1: run the instrument on an autoregressive model (Pythia). Replicates
the two load-bearing findings to test external validity beyond MLMs:
  velocity(r)   damage light-cone front velocity vs radius (F16/F21)
  D(r,T), r*    self-correction grid + instability radius (F23)
plus a coherence/census check vs the Pythia-tokenized WikiText proxy.
Usage: ar_probe.py --model pythia-160m
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, argparse, json, time
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from collections import Counter
import numpy as np
import ar_ca
from ar_ca import ARRule, run
from mlm_lib import cone_front_velocity, ensure_resdir, RESDIR

MODELS = {"pythia-160m": "EleutherAI/pythia-160m", "pythia-410m": "EleutherAI/pythia-410m"}
RS = [1, 2, 4, 8, 16]
TS = [0.5, 0.7, 0.9, 1.1]
SEEDS = [21, 22, 23]


def block_damage(rule, T, r, block=3, B=24, N=48, settle=12, sweeps=30, seed=21,
                 scheme="none", ignite_thresh=0.05, tail=8):
    rng = np.random.default_rng(seed)
    base = run(rule, B=B, N=N, r=r, T=T, sweeps=settle, scheme=scheme, init="random", seed=seed)["final"]
    c = N // 2
    idx = [c + k for k in range(-(block // 2), block - block // 2)]
    flipped = base.copy()
    for j in idx:
        flipped[:, j] = rng.choice(rule.init_pool, size=B)
    u = np.random.default_rng(seed + 1).random(sweeps * N * B)
    u2 = np.concatenate([u.reshape(sweeps * N, B)] * 2, axis=1).reshape(-1)
    init2 = np.concatenate([base, flipped], axis=0)      # batched CRN twins (2B)
    c2 = run(rule, B=2 * B, N=N, r=r, T=T, sweeps=sweeps, scheme=scheme,
             init_state=init2, seed=seed + 2, u_stream=u2)
    snaps = c2["snaps"]
    diff = (snaps[:, :B] != snaps[:, B:])
    cone = np.roll(diff, N // 2 - idx[len(idx) // 2], axis=2).mean(axis=1)
    final = diff[-tail:].mean(axis=(0, 2))
    ig = final > ignite_thresh
    return dict(cone=cone, mean_damage=float(final.mean()), ignition_prob=float(ig.mean()),
                cond_spread=float(final[ig].mean()) if ig.any() else 0.0)


def main(tag, B, N, sweeps):
    ensure_resdir()
    rule = ARRule(MODELS[tag])
    ref = np.load("data_ar/ref_ids.npy")
    ref_bi = set(zip(ref[:-1].tolist(), ref[1:].tolist()))
    res = {"model": tag, "RS": RS, "TS": TS, "N": N}
    t0 = time.time()

    # velocity vs r at a spreading T
    vel = {}
    for r in RS:
        d = block_damage(rule, 1.0, r, B=B, N=N, sweeps=sweeps, seed=21)
        v = cone_front_velocity(d["cone"])
        vel[r] = round(v["velocity_sites_per_sweep"], 2)
        print(f"[{tag}] velocity r={r:>2}: v={vel[r]} sites/sweep (mean_dmg={d['mean_damage']:.2f})", flush=True)
    res["velocity_vs_r"] = vel

    # D(r,T) grid (3 seeds)
    D = {r: {} for r in RS}
    for r in RS:
        for T in TS:
            ds = [block_damage(rule, T, r, B=B, N=N, sweeps=sweeps, seed=s)["mean_damage"] for s in SEEDS]
            D[r][T] = dict(mean=round(float(np.mean(ds)), 4), std=round(float(np.std(ds)), 4))
            print(f"[{tag}] D r={r:>2} T={T}: {D[r][T]['mean']:.3f}±{D[r][T]['std']:.3f}", flush=True)
    res["D"] = {str(r): {str(T): D[r][T] for T in TS} for r in RS}
    # r* = argmax_r mean-over-T damage
    meanD = {r: np.mean([D[r][T]["mean"] for T in TS]) for r in RS}
    res["r_star_max_D"] = int(max(meanD, key=meanD.get))

    # coherence/census at an ordered T
    o = run(rule, B=B, N=N, r=4, T=0.7, sweeps=40, scheme="none", init="random", seed=7)
    rows = o["snaps"][-15:].reshape(-1, N)
    bov = np.mean([[(int(row[i]), int(row[i+1])) in ref_bi for i in range(N-1)] for row in rows])
    res["order_bigram_overlap_T0.7_r4"] = round(float(bov), 4)
    res["example"] = rule.tok.decode(o["final"][0].tolist())
    print(f"[{tag}] order(bigram overlap, T=0.7 r=4)={bov:.3f}", flush=True)
    print(f"[{tag}] example: {res['example'][:180]!r}", flush=True)

    json.dump(res, open(f"{RESDIR}/ar_{tag}.json", "w"), indent=1)
    print(f"[{tag}] AR PROBE DONE ({time.time()-t0:.0f}s)  velocity={vel}  r*={res['r_star_max_D']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m", choices=list(MODELS))
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=30)
    a = ap.parse_args()
    main(a.model, a.B, a.N, a.sweeps)
