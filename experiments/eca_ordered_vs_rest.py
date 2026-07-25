"""Issue #24 -- test the ECA class separation on IGNITION PROBABILITY, the right statistic.

Why this exists. F34 established that the ECA rung's discriminating quantity is the
ignition (survival) probability, not lambda: damage in a discrete CA is bimodal, and
lambda|ignited is *undefined* for ordered rules because they never ignite at all. That left
`eca_calib_ignition.json`'s `ordered < chaotic p=0.0000` as NaN-comparison artifacts, which
must not be quoted. This computes the test that can be.

P_ignite is also the physically correct object rather than a convenience: damage spreading
in a discrete CA is directed-percolation-class, and DP transitions are characterised by
survival probability. It is the order parameter the DK rung (#22) should share.

Statistics: the RULE is the unit of analysis (not the seed, not the lattice) -- the same
discipline that W1 flagged. Bootstrap over rules within group, two-sided.
Reads results/eca_calib_ignition.json; writes results/eca_ordered_vs_rest.json.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_BOOT, RNG = 20000, 20260725
SRC = ROOT / "results" / "eca_calib_ignition.json"


def main():
    d = json.load(open(SRC))["per_rule"]
    groups = {}
    for rn, v in d.items():
        groups.setdefault(v["group"], []).append((int(rn), v["ignition_prob"]))

    rng = np.random.default_rng(RNG)
    draws, stats_ = {}, {}
    for g, items in groups.items():
        vals = np.array([p for _, p in items])
        draws[g] = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
        stats_[g] = dict(mean=round(float(vals.mean()), 4), n_rules=len(vals),
                         ci95=[round(float(np.percentile(draws[g], 2.5)), 4),
                               round(float(np.percentile(draws[g], 97.5)), 4)],
                         rules={str(r): round(p, 4) for r, p in sorted(items)})

    print("=== ignition probability by class (rule = unit) ===")
    for g in ("ordered", "edge", "chaotic", "reference"):
        if g in stats_:
            s = stats_[g]
            print(f"  {g:9s} P_ignite={s['mean']:.4f}  CI95={s['ci95']}  ({s['n_rules']} rules)")

    # primary: ordered vs everything else pooled (the claim F33 showed is the robust one)
    rest = np.array([p for g in ("edge", "chaotic") for _, p in groups[g]])
    ordv = np.array([p for _, p in groups["ordered"]])
    d_rest = rng.choice(rest, size=(N_BOOT, len(rest)), replace=True).mean(axis=1)
    d_ord = rng.choice(ordv, size=(N_BOOT, len(ordv)), replace=True).mean(axis=1)
    p_primary = float((d_ord >= d_rest).mean())
    p_oe = float((draws["ordered"] >= draws["edge"]).mean())
    p_oc = float((draws["ordered"] >= draws["chaotic"]).mean())
    p_ec = float((draws["edge"] >= draws["chaotic"]).mean())
    # effect size: how separated, in units of the pooled spread
    sep = (rest.mean() - ordv.mean()) / np.sqrt(0.5 * (rest.var(ddof=1) + ordv.var(ddof=1)))

    print(f"\n  PRIMARY  ordered < rest    : p={p_primary:.4f}   Cohen's d={sep:.2f}")
    print(f"           ordered < edge    : p={p_oe:.4f}")
    print(f"           ordered < chaotic : p={p_oc:.4f}")
    print(f"           edge    < chaotic : p={p_ec:.4f}  (the fragile one, cf. F33/F34)")

    out = dict(
        note=("Class separation tested on IGNITION PROBABILITY, the DP-class order parameter. "
              "Supersedes the lambda-based tests, whose ordered-group p-values were "
              "NaN-comparison artifacts (lambda|ignited is undefined for rules that never "
              "ignite). Rule is the unit of analysis."),
        source=str(SRC.name), n_boot=N_BOOT, groups=stats_,
        tests=dict(ordered_lt_rest_p=p_primary, ordered_lt_edge_p=p_oe,
                   ordered_lt_chaotic_p=p_oc, edge_lt_chaotic_p=p_ec,
                   cohens_d_ordered_vs_rest=round(float(sep), 3)))
    dest = ROOT / "results" / "eca_ordered_vs_rest.json"
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
