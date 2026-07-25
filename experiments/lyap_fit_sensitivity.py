"""Phase 0.3: is the F27 ECA class ordering an artifact of lyap_from_cone's free parameters?

`lyap_from_cone` picks its fit window with a data-dependent branch governed by four
constants (sat_threshold, frac_of_max, max_sweeps, min_sweeps). If the headline ordering
ordered < edge < chaotic moves when those move, the ordering is an artifact of the
estimator rather than a property of the rules.

Sweeps the parameter grid (plus a set of explicit PRE-REGISTERED fixed windows, which
bypass the branch entirely) and reports whether the ordering survives each setting.
Writes results/lyap_fit_sensitivity.json. CPU, a few minutes.
"""
import sys, pathlib, json, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from lyapunov import lyap_from_cone
from eca_calib import RULES, damage_cone

ROOT = pathlib.Path(__file__).resolve().parents[1]
GROUPS = {"ordered": [128, 232, 4], "edge": [110, 54], "chaotic": [30, 150, 22]}
SEEDS = [0, 1, 2, 3, 4]

GRID = dict(sat_threshold=[2.5, 3.5, 5.0], frac_of_max=[0.3, 0.5, 0.7],
            max_sweeps=[5, 8, 12], min_sweeps=[2, 3])
FIXED_WINDOWS = [(0, 3), (0, 5), (0, 8), (1, 6)]


def group_means(cones, **kw):
    lam = {}
    for rn, cs in cones.items():
        lam[rn] = float(np.mean([lyap_from_cone(c, 64, **kw)[0] for c in cs]))
    gm = {g: float(np.mean([lam[r] for r in rs])) for g, rs in GROUPS.items()}
    return gm, lam


def main():
    # cache the cones once; the sweep is over the ESTIMATOR, not the simulation
    print("simulating ECA damage cones (cached across the whole sweep) ...", flush=True)
    cones = {}
    for rn, _ in RULES:
        if rn == 90:      # linear reference rule, excluded from the 3 scored groups
            continue
        cones[rn] = [damage_cone(rn, seed=s) for s in SEEDS]
    print(f"  {len(cones)} rules x {len(SEEDS)} seeds", flush=True)

    rows, n_ok = [], 0
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        kw = dict(zip(keys, combo))
        gm, _ = group_means(cones, **kw)
        ok = gm["ordered"] < gm["edge"] < gm["chaotic"]
        n_ok += ok
        rows.append(dict(params=kw, group_means={k: round(v, 4) for k, v in gm.items()},
                         ordering_recovered=bool(ok)))
    print(f"\nbranch-parameter grid: ordering recovered in {n_ok}/{len(rows)} settings")

    fixed = []
    for w in FIXED_WINDOWS:
        gm, lam = group_means(cones, fit_window=w)
        ok = gm["ordered"] < gm["edge"] < gm["chaotic"]
        fixed.append(dict(fit_window=list(w), group_means={k: round(v, 4) for k, v in gm.items()},
                          ordering_recovered=bool(ok)))
        print(f"  fixed window {w}: ord {gm['ordered']:+.3f} < edge {gm['edge']:+.3f} < "
              f"chaos {gm['chaotic']:+.3f}  -> {ok}")

    # Rule 90 nuance under the default estimator (feature, not bug: ballistic but marginal)
    r90 = [damage_cone(90, seed=s) for s in SEEDS]
    lam90 = float(np.mean([lyap_from_cone(c, 64)[0] for c in r90]))
    dmax90 = float(np.mean([lyap_from_cone(c, 64)[1] for c in r90]))

    out = dict(
        note=("Sensitivity of the F27 ECA class ordering to lyap_from_cone's free parameters. "
              "The simulation is cached, so the sweep isolates the ESTIMATOR."),
        seeds=SEEDS, groups=GROUPS,
        branch_grid=dict(n_settings=len(rows), n_ordering_recovered=n_ok,
                         fraction=round(n_ok / len(rows), 3), settings=rows),
        fixed_windows=fixed,
        rule90=dict(lambda_ca=round(lam90, 4), dmax_frac=round(dmax90, 4),
                    note="linear rule: ballistic spread with marginal exponential growth"))
    dest = str(ROOT / "results" / "lyap_fit_sensitivity.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nRule 90: lambda={lam90:+.4f} dmax_frac={dmax90:.3f} (marginal despite wide spread)")
    print("wrote", dest)


if __name__ == "__main__":
    main()
