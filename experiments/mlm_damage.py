"""Phase 3: damage spreading for a real MLM. Two arms:
  (b) light cones: velocity of the damage front vs radius r at fixed T.
  (c) self-healing: mean damage & ignition probability vs T at fixed r, to locate
      the ordered (self-healing) phase relative to the slow/fast-mixing crossover
      tau~1.5-2 reported for full-context MLM-Glauber (arXiv:2605.16378). NOTE the
      setups differ: they use full-context conditionals, we use radius-windowed.
3-site block flips, CRN twins.  Usage: mlm_damage.py --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import argparse, json, time
import numpy as np
from mlm_ca import MLMRule, run
from mlm_lib import MODELS, RESDIR, cone_front_velocity, ensure_resdir


def block_damage(rule, T, r, block=3, B=32, N=48, settle=20, sweeps=40, seed=21,
                 scheme="cls_sep", ignite_thresh=0.05, tail=5):
    rng = np.random.default_rng(seed)
    base = run(rule, B=B, N=N, r=r, T=T, sweeps=settle, scheme=scheme,
               init="random", seed=seed)["final"]
    c = N // 2
    idx = [c + k for k in range(-(block // 2), block - block // 2)]
    flipped = base.copy()
    for j in idx:
        flipped[:, j] = rng.choice(rule.init_pool, size=B)
    # Batch the two CRN twins into one 2B run: stack [base; flipped]; tile the
    # per-site uniforms so lattice j and j+B share them (exact CRN). One forward +
    # one MPS sync per site instead of two -> ~1.5-1.7x faster, bit-identical result.
    u = np.random.default_rng(seed + 1).random(sweeps * N * B)
    u2 = np.concatenate([u.reshape(sweeps * N, B)] * 2, axis=1).reshape(-1)
    init2 = np.concatenate([base, flipped], axis=0)
    c2 = run(rule, B=2 * B, N=N, r=r, T=T, sweeps=sweeps, scheme=scheme,
             init_state=init2, seed=seed + 2, u_stream=u2)
    snaps = c2["snaps"]
    diff = (snaps[:, :B] != snaps[:, B:])
    cone = np.roll(diff, c - idx[len(idx) // 2] + (N // 2 - c), axis=2).mean(axis=1)
    final = diff[-tail:].mean(axis=(0, 2))
    ignited = final > ignite_thresh
    return dict(cone=cone, mean_damage=float(final.mean()),
                ignition_prob=float(ignited.mean()),
                cond_spread=float(final[ignited].mean()) if ignited.any() else 0.0,
                n_ignited=int(ignited.sum()), B=B)


def drift_floor(rule, T, r, B=32, N=48, settle=12, sweeps=30, seed=21,
                scheme="cls_sep", tail=8):
    """Diversity / mixing floor for the damage metric. Two twins from the SAME
    settled init but INDEPENDENT sampling noise (NO flip): how far the model
    decorrelates on its own -- the level the perturbed damage D saturates toward. A
    degenerate low-entropy model has a low floor (nowhere to differ); a diverse model
    a high one. Genuine self-correction is D relative to this floor, not D itself
    (else low D just means 'collapsed to mush', the stability analog of the A3
    repetition confound). Also returns the distinct-token fraction (diversity gauge)."""
    base = run(rule, B=B, N=N, r=r, T=T, sweeps=settle, scheme=scheme,
               init="random", seed=seed)["final"]
    ua = np.random.default_rng(seed + 1).random(sweeps * N * B)
    ub = np.random.default_rng(seed + 101).random(sweeps * N * B)   # INDEPENDENT noise
    u2 = np.concatenate([ua.reshape(sweeps * N, B), ub.reshape(sweeps * N, B)], axis=1).reshape(-1)
    init2 = np.concatenate([base, base], axis=0)                    # same init, no flip
    c2 = run(rule, B=2 * B, N=N, r=r, T=T, sweeps=sweeps, scheme=scheme,
             init_state=init2, seed=seed + 2, u_stream=u2)
    snaps = c2["snaps"]
    d0 = float((snaps[-tail:, :B] != snaps[-tail:, B:]).mean())
    last = snaps[-1]
    distinct = float(np.mean([len(np.unique(last[b])) / N for b in range(2 * B)]))
    return d0, distinct


def main(tag, B, N, sweeps, scheme):
    ensure_resdir()
    rule = MLMRule(MODELS[tag])
    cones, stats = {}, {"velocity_vs_r": {}, "healing_vs_T": {}}
    t0 = time.time()

    # (b) light cones: velocity vs r at T=1.0
    for r in [1, 2, 4, 8, 16]:
        tc = time.time()
        d = block_damage(rule, 1.0, r, B=B, N=N, sweeps=sweeps, scheme=scheme)
        cones[f"vel_T1.0_r{r}"] = d["cone"]
        v = cone_front_velocity(d["cone"])
        stats["velocity_vs_r"][r] = dict(velocity=v["velocity_sites_per_sweep"],
                                         mean_damage=d["mean_damage"],
                                         ignition_prob=d["ignition_prob"])
        print(f"[{tag}] cone T=1.0 r={r:>2}: v={v['velocity_sites_per_sweep']:.2f} sites/sweep "
              f"mean_dmg={d['mean_damage']:.3f} P_ign={d['ignition_prob']:.3f} ({time.time()-tc:.0f}s)",
              flush=True)

    # (c) self-healing vs T at r=4
    for T in [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]:
        tc = time.time()
        d = block_damage(rule, T, 4, B=B, N=N, sweeps=sweeps, scheme=scheme)
        cones[f"heal_r4_T{T}"] = d["cone"]
        stats["healing_vs_T"][T] = dict(mean_damage=d["mean_damage"],
                                        ignition_prob=d["ignition_prob"],
                                        cond_spread=d["cond_spread"])
        print(f"[{tag}] heal r=4 T={T}: mean_dmg={d['mean_damage']:.3f} "
              f"P_ign={d['ignition_prob']:.3f} spread={d['cond_spread']:.3f} ({time.time()-tc:.0f}s)",
              flush=True)

    np.savez_compressed(f"{RESDIR}/{tag}_damage.npz", **cones)
    json.dump(stats, open(f"{RESDIR}/{tag}_damage.json", "w"), indent=1)
    print(f"[{tag}] DAMAGE DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--scheme", default="cls_sep")
    a = ap.parse_args()
    main(a.model, a.B, a.N, a.sweeps, a.scheme)
