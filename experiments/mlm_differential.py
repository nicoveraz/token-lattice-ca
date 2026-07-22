"""Phase 3: differential CRN certification for real MLMs (the F9 protocol).

Twin lattices share init, update order, and uniforms; exactly one factor differs.
F9's lesson: trajectory-level divergence saturates under ANY difference (chaos
amplifies signal and apparatus alike), so we attribute at the STATISTICS level —
we compare the equilibrium order parameter of the two arms (delta_order), never
the endpoint trajectories. We report both to show the contrast.

Arms:
  null                 nothing differs           -> traj d = 0, delta_order = 0
  apparatus:order      different visitation order -> d > 0, delta_order ~ 0 (same dist)
  apparatus:cdfperm    CDF built in permuted order-> d > 0, delta_order ~ 0 (same dist)
  apparatus:scheme     cls_sep vs no-special      -> genuine apparatus sensitivity test
  model (--pair a,b)   different model            -> the signal: delta_order should move

Certification: a reading is model signal iff it NULLS under the distribution-
preserving apparatus swaps (order, cdfperm) and MOVES under model swaps.
Usage: mlm_differential.py --model tiny   |   mlm_differential.py --pair tiny,mini
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import argparse, json, time
import numpy as np
from mlm_ca import MLMRule, run
from mlm_lib import MODELS, RESDIR, load_ref, order_param, ensure_resdir
import mlm_ca

TAIL = 6


def _perm_sampler(perm):
    def f(probs, u):
        p = probs[:, perm]
        cdf = np.cumsum(p, axis=-1); cdf /= cdf[:, -1:]
        idx = np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))])
        return perm[idx].astype(np.int64)
    return f


def coupled(ruleA, ruleB, T, r, ref_bi, B=16, N=48, sweeps=30, seed=71,
            schemeA="cls_sep", schemeB="cls_sep", sampler_b=None, order_seed_b=None):
    """Run A and B sharing init + uniforms. order_seed_b differs -> different visit
    order. Returns dict(traj_d[last], delta_order, order_a, order_b)."""
    rng = np.random.default_rng(seed)
    init = ruleA.random_lattice(rng, B, N)
    u = np.random.default_rng(seed + 1).random(sweeps * N * B)
    a = run(ruleA, B=B, N=N, r=r, T=T, sweeps=sweeps, scheme=schemeA,
            init_state=init, seed=seed + 2, u_stream=u)
    b = run(ruleB, B=B, N=N, r=r, T=T, sweeps=sweeps, scheme=schemeB,
            init_state=init, seed=(seed + 2 if order_seed_b is None else order_seed_b),
            u_stream=u, sampler=sampler_b)
    traj = float((a["snaps"] != b["snaps"]).mean(axis=(1, 2))[-1])
    oa = np.mean([order_param(s, ref_bi)[0] for s in a["snaps"][-TAIL:]])
    ob = np.mean([order_param(s, ref_bi)[0] for s in b["snaps"][-TAIL:]])
    return dict(traj_d_end=round(traj, 4), order_a=round(float(oa), 4),
                order_b=round(float(ob), 4), delta_order=round(abs(float(oa - ob)), 4))


def apparatus_arms(tag, B, N, sweeps):
    rule = MLMRule(MODELS[tag])
    ref_bi = mlm_ca.ref_bigrams(load_ref())
    perm = np.random.default_rng(5).permutation(rule.V)
    out = {}
    for T in [0.8, 1.3]:
        for r in [2]:
            base = dict(ref_bi=ref_bi, B=B, N=N, sweeps=sweeps)
            out[f"null_T{T}_r{r}"] = coupled(rule, rule, T, r, **base)
            out[f"apparatus:order_T{T}_r{r}"] = coupled(rule, rule, T, r, order_seed_b=999, **base)
            out[f"apparatus:cdfperm_T{T}_r{r}"] = coupled(rule, rule, T, r, sampler_b=_perm_sampler(perm), **base)
            out[f"apparatus:scheme_T{T}_r{r}"] = coupled(rule, rule, T, r, schemeB="none", **base)
            for k, v in out.items():
                if k.endswith(f"T{T}_r{r}"):
                    print(f"[{tag}] {k}: traj_d={v['traj_d_end']} delta_order={v['delta_order']} "
                          f"(oA={v['order_a']} oB={v['order_b']})", flush=True)
    json.dump(out, open(f"{RESDIR}/{tag}_diff.json", "w"), indent=1)
    return out


def model_arm(a_tag, b_tag, B, N, sweeps):
    rA, rB = MLMRule(MODELS[a_tag]), MLMRule(MODELS[b_tag])
    ref_bi = mlm_ca.ref_bigrams(load_ref())
    out = {}
    for T in [0.8, 1.3]:
        v = coupled(rA, rB, T, 2, ref_bi, B=B, N=N, sweeps=sweeps)
        out[f"model:{a_tag}_vs_{b_tag}_T{T}"] = v
        print(f"[model] {a_tag} vs {b_tag} T={T}: traj_d={v['traj_d_end']} "
              f"delta_order={v['delta_order']} (o{a_tag}={v['order_a']} o{b_tag}={v['order_b']})", flush=True)
    path = f"{RESDIR}/model_arm_{a_tag}_{b_tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    return out


if __name__ == "__main__":
    ensure_resdir()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--pair", help="a,b model tags for the model arm")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    if a.pair:
        x, y = a.pair.split(",")
        model_arm(x, y, a.B, a.N, a.sweeps)
    else:
        apparatus_arms(a.model, a.B, a.N, a.sweeps)
    print(f"DIFFERENTIAL DONE ({time.time()-t0:.0f}s)", flush=True)
