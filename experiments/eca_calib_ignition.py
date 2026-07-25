"""Fix the ECA rung's ignition confound (diagnosed after F33).

Problem. `eca_calib.damage_cone` averages the damage cone over all B lattices before
fitting lambda. Single-site damage in a discrete CA is bimodal -- it either IGNITES or dies
outright -- so that average mixes ignited and dead runs, and lambda then measures the
MIXTURE, not the growth rate. Evidence from the 12-seed run (results/eca_calib_hardened.json):
  * Rule 90: 3/12 seeds sit exactly at the estimator floor (-0.9210, damage never ignited)
    while the other 9 average about +0.28. Its celebrated "correctly marginal" reading of
    -0.023 is therefore largely an AVERAGING ARTIFACT, not a clean marginal measurement.
  * Rule 30: 10/12 seeds in -0.13..+0.50 but two collapse runs at -2.30 and -2.19 drag the
    mean to -0.243, i.e. the canonical chaotic rule reads negative.

This is a lesson the LM path already learned: F8 ("ignition is a rare event and bimodal...
must report ignition probability separately from spread") and F13 (block-flip ignition
probability separated from conditional spread, via `block_damage`'s ignition_prob /
cond_spread). The ECA rung never got the same treatment. This script applies it.

Reports, per rule: ignition probability, and lambda CONDITIONAL on ignition (fit on the
cone averaged over ignited lattices only). Same pre-registered saturation-relative window.
Writes results/eca_calib_ignition.json. CPU, a few minutes.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from lyapunov import lyap_from_cone
from eca_calib import eca_table
from eca_calib_hardened import GROUPS, REFERENCE, FIT_KW, SEEDS, N_BOOT, RNG_BOOT, boot_ci

ROOT = pathlib.Path(__file__).resolve().parents[1]
IGNITE_TAIL = 5          # sweeps averaged at the end to decide "still damaged"
IGNITE_THRESH = 0.02     # fraction of sites still differing => ignited (matches LM path spirit)


def damage_per_lattice(rulenum, N=64, B=128, sweeps=20, settle=12, eta=0.0, seed=0):
    """Async CRN damage spreading, single-site flip -> per-lattice diff (sweeps+1, B, N).

    Identical dynamics to eca_calib.damage_cone; it simply does NOT average over lattices,
    so ignited and dead runs can be separated afterwards.
    """
    rng = np.random.default_rng(seed)
    tab = eca_table(rulenum)

    def update(X, idx, u):
        l = X[:, (idx - 1) % N]; c = X[:, idx]; r = X[:, (idx + 1) % N]
        b = tab[4 * l + 2 * c + r]
        X[:, idx] = np.where(u < eta, 1 - b, b)

    A = rng.integers(0, 2, size=(B, N), dtype=np.int8)
    for _ in range(settle):
        for idx in rng.permutation(N):
            update(A, idx, rng.random(B))
    Bl = A.copy(); Bl[:, N // 2] ^= 1
    diffs = [(A != Bl).astype(np.float64)]
    for _ in range(sweeps):
        for idx in rng.permutation(N):
            u = rng.random(B)                      # CRN: shared order + shared noise
            update(A, idx, u); update(Bl, idx, u)
        diffs.append((A != Bl).astype(np.float64))
    return np.asarray(diffs)                        # (sweeps+1, B, N)


def rule_stats(rulenum):
    """Per seed: ignition probability, unconditional lambda, and lambda | ignited."""
    ig, lam_all, lam_cond = [], [], []
    for s in SEEDS:
        d = damage_per_lattice(rulenum, seed=s)                  # (S+1, B, N)
        tail = d[-IGNITE_TAIL:].mean(axis=(0, 2))                # (B,) per-lattice residual
        ignited = tail > IGNITE_THRESH
        ig.append(float(ignited.mean()))
        lam_all.append(lyap_from_cone(d.mean(axis=1), 64, **FIT_KW)[0])   # old behaviour
        if ignited.any():
            lam_cond.append(lyap_from_cone(d[:, ignited].mean(axis=1), 64, **FIT_KW)[0])
    return (np.array(ig), np.array(lam_all),
            np.array(lam_cond) if lam_cond else np.array([np.nan]))


def main():
    all_rules = sorted({r for rs in GROUPS.values() for r in rs} | set(REFERENCE))
    print(f"ECA ignition fix: {len(all_rules)} rules x {len(SEEDS)} seeds\n")
    print(f"{'rule':>5} {'grp':<9} {'P(ignite)':>10} {'lam_all':>9} {'lam|ignited':>12}")
    per_rule = {}
    for rn in all_rules:
        ig, la, lc = rule_stats(rn)
        grp = next((g for g, rs in GROUPS.items() if rn in rs), "reference")
        ci = boot_ci(lc) if np.isfinite(lc).all() and len(lc) > 1 else (float("nan"),) * 2
        per_rule[rn] = dict(group=grp, ignition_prob=float(ig.mean()),
                            lambda_all=float(la.mean()), lambda_cond=float(np.nanmean(lc)),
                            lambda_cond_ci95=[round(ci[0], 4), round(ci[1], 4)],
                            n_seeds_ignited=int(np.isfinite(lc).sum()))
        print(f"{rn:5d} {grp:<9} {ig.mean():10.3f} {la.mean():+9.4f} {np.nanmean(lc):+12.4f}")

    rng = np.random.default_rng(RNG_BOOT)
    gd, gm = {}, {}
    for g, rs in GROUPS.items():
        v = np.array([per_rule[r]["lambda_cond"] for r in rs])
        gd[g] = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
        gm[g] = dict(mean=float(v.mean()),
                     ci95=[round(float(np.percentile(gd[g], 2.5)), 4),
                           round(float(np.percentile(gd[g], 97.5)), 4)])
    p_oc = float((gd["ordered"] >= gd["chaotic"]).mean())
    p_oe = float((gd["ordered"] >= gd["edge"]).mean())
    p_ec = float((gd["edge"] >= gd["chaotic"]).mean())
    print("\n=== group means using lambda | IGNITED ===")
    for g in ["ordered", "edge", "chaotic"]:
        print(f"  {g:8s} {gm[g]['mean']:+.4f}  CI95={gm[g]['ci95']}")
    print(f"\n  ordered < chaotic : p={p_oc:.4f}")
    print(f"  ordered < edge    : p={p_oe:.4f}")
    print(f"  edge    < chaotic : p={p_ec:.4f}   (was 0.1665 unconditional)")
    r90 = per_rule[90]
    print(f"\n  rule 90 (linear ref): P(ignite)={r90['ignition_prob']:.3f} "
          f"lam_all={r90['lambda_all']:+.4f} -> lam|ignited={r90['lambda_cond']:+.4f}")
    r30 = per_rule[30]
    print(f"  rule 30 (chaotic)   : P(ignite)={r30['ignition_prob']:.3f} "
          f"lam_all={r30['lambda_all']:+.4f} -> lam|ignited={r30['lambda_cond']:+.4f}")

    out = dict(note=("Separates ignition probability from conditional spread on the ECA rung, "
                     "matching the LM path (F8/F13). lambda_all reproduces the old, "
                     "ignition-confounded estimate for comparison."),
               ignite_thresh=IGNITE_THRESH, ignite_tail=IGNITE_TAIL, fit_kw=FIT_KW,
               seeds=SEEDS, per_rule={str(k): v for k, v in per_rule.items()},
               groups_conditional=gm,
               tests=dict(ordered_lt_chaotic_p=p_oc, ordered_lt_edge_p=p_oe,
                          edge_lt_chaotic_p=p_ec))
    dest = str(ROOT / "results" / "eca_calib_ignition.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
