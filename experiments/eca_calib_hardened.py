"""Phase 2.1 -- the hardened ECA rung: the ladder's weight-bearing calibration.

Methodologically this rung is the right one: discrete alphabet, finite single-site flip,
async CRN twins sharing visit order AND noise, no renormalization (unlike the logistic rung,
see F30). It was simply thin: 9 rules, 3 seeds. This raises it to 19 rules x 12 seeds with
bootstrap CIs.

PRE-REGISTRATION (fixed BEFORE running; do not tune after seeing results)
------------------------------------------------------------------------
1. Group assignment: Wolfram's ECA classification (Wolfram, "Universality and complexity in
   cellular automata", Physica D 10 (1984) 1-35; tables reproduced in A New Kind of Science,
   2002). Class I/II -> `ordered`, Class IV -> `edge`, Class III -> `chaotic`.
   HONEST CAVEAT: Wolfram's classes are not a crisp partition and a few rules are disputed
   in the literature (notably 62 and 106, variously placed II/III/IV). They are declared
   here in advance anyway, and the ordering test is reported both with and without them.
2. Fit window: the SATURATION-RELATIVE rule with `lyap_from_cone`'s default constants
   (sat_threshold=3.5, frac_of_max=0.5, max_sweeps=8, min_sweeps=3), frozen in advance.
   Justification (F32, results/lyap_fit_sensitivity.json): that rule recovers the ordering
   in 54/54 settings of its own constants, whereas a single global FIXED window does not
   generalise -- chaotic rules saturate fast, so any window past ~3 sweeps averages in the
   post-saturation plateau and INVERTS edge vs chaotic (fails at (0,5), (0,8), (1,6)).
   A fixed window is therefore the wrong pre-registration for this estimator.
3. Primary claim: `ordered` < `chaotic` (the robust separation).
   Secondary claim: `ordered` < `edge` < `chaotic` (F32 flags edge-vs-chaotic as fragile,
   margin as low as +0.023 -- reported with CIs, not asserted).
4. Rule 90 (linear) is a REFERENCE, excluded from the scored groups: it spreads
   ballistically but grows only marginally, so lambda_ca should read ~0 despite a wide cone.

Writes results/eca_calib_hardened.json. CPU, a few minutes.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from lyapunov import lyap_from_cone
from eca_calib import damage_cone

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEEDS = list(range(12))                     # >= 10, pre-registered
N_BOOT = 2000
RNG_BOOT = 20260725                          # fixed bootstrap seed

# Pre-registered groups (Wolfram 1984 classes; see docstring for the disputed cases).
GROUPS = {
    "ordered": [0, 8, 32, 128, 160, 232, 4],        # Class I / II
    "edge":    [110, 54, 106, 62],                   # Class IV (106, 62 disputed)
    "chaotic": [30, 45, 105, 126, 146, 150, 22],     # Class III
}
DISPUTED = [106, 62]
REFERENCE = {90: "linear (Class III): ballistic spread, marginal growth -- expect lambda~0"}
FIT_KW = dict(sat_threshold=3.5, frac_of_max=0.5, max_sweeps=8, min_sweeps=3)


def per_rule_lambda(rulenum):
    """lambda_ca per seed (one damage-spreading run each) under the pre-registered window."""
    out = []
    for s in SEEDS:
        cone = damage_cone(rulenum, seed=s)
        out.append(lyap_from_cone(cone, 64, **FIT_KW)[0])
    return np.asarray(out, dtype=float)


def boot_ci(x, n=N_BOOT, seed=RNG_BOOT, lo=2.5, hi=97.5):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def main():
    all_rules = sorted({r for rs in GROUPS.values() for r in rs} | set(REFERENCE))
    print(f"ECA hardened rung: {len(all_rules)} rules x {len(SEEDS)} seeds "
          f"(pre-registered saturation-relative window {FIT_KW})", flush=True)
    per_rule = {}
    for rn in all_rules:
        lam = per_rule_lambda(rn)
        ci = boot_ci(lam)
        per_rule[rn] = dict(mean=float(lam.mean()), sd=float(lam.std(ddof=1)),
                            ci95=[round(ci[0], 4), round(ci[1], 4)],
                            per_seed=[round(v, 4) for v in lam])
        grp = next((g for g, rs in GROUPS.items() if rn in rs), "reference")
        print(f"  rule {rn:3d} [{grp:9s}] lambda={lam.mean():+.4f} "
              f"CI95=[{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)

    # group-level: bootstrap over rules-within-group (rule is the unit, seeds averaged)
    rng = np.random.default_rng(RNG_BOOT)
    gstats, gdraws = {}, {}
    for g, rs in GROUPS.items():
        vals = np.array([per_rule[r]["mean"] for r in rs])
        draws = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
        gdraws[g] = draws
        gstats[g] = dict(mean=float(vals.mean()), n_rules=len(rs),
                         ci95=[round(float(np.percentile(draws, 2.5)), 4),
                               round(float(np.percentile(draws, 97.5)), 4)])
    p_ord_cha = float((gdraws["ordered"] >= gdraws["chaotic"]).mean())
    p_ord_edge = float((gdraws["ordered"] >= gdraws["edge"]).mean())
    p_edge_cha = float((gdraws["edge"] >= gdraws["chaotic"]).mean())

    # sensitivity: drop the disputed rules
    g2 = {g: [r for r in rs if r not in DISPUTED] for g, rs in GROUPS.items()}
    m2 = {g: float(np.mean([per_rule[r]["mean"] for r in rs])) for g, rs in g2.items()}

    print("\n=== group means (rule = unit of analysis) ===")
    for g in ["ordered", "edge", "chaotic"]:
        s = gstats[g]; print(f"  {g:8s} {s['mean']:+.4f}  CI95={s['ci95']}  ({s['n_rules']} rules)")
    print(f"\n  PRIMARY   ordered < chaotic : bootstrap p={p_ord_cha:.4f}")
    print(f"  secondary ordered < edge    : p={p_ord_edge:.4f}")
    print(f"  secondary edge    < chaotic : p={p_edge_cha:.4f}   <-- F32 flagged as fragile")
    print(f"  without disputed {DISPUTED}: " +
          "  ".join(f"{g} {v:+.4f}" for g, v in m2.items()))
    r90 = per_rule[90]
    print(f"\n  reference rule 90 (linear): lambda={r90['mean']:+.4f} CI95={r90['ci95']}")

    out = dict(
        preregistration=dict(source="Wolfram (1984) Physica D 10:1-35; NKS (2002) tables",
                             groups=GROUPS, disputed=DISPUTED, reference=REFERENCE,
                             fit_window_rule="saturation-relative (lyap_from_cone defaults)",
                             fit_kw=FIT_KW, seeds=SEEDS,
                             primary_claim="ordered < chaotic",
                             secondary_claim="ordered < edge < chaotic (fragile, see F32)"),
        per_rule={str(k): v for k, v in per_rule.items()},
        groups=gstats,
        tests=dict(ordered_lt_chaotic_p=p_ord_cha, ordered_lt_edge_p=p_ord_edge,
                   edge_lt_chaotic_p=p_edge_cha,
                   note="bootstrap p = P(group means out of order), rule as unit"),
        without_disputed=m2)
    dest = str(ROOT / "results" / "eca_calib_hardened.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
