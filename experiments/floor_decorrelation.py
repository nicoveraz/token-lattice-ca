"""How arbitrary is D_norm's denominator? Sweep the floor's decorrelation (issue #34, W2).

W2's standing objection is that D_norm's numerator and denominator use DIFFERENT couplings: the
numerator is CRN (twins share one uniform stream), the floor is independent noise. #34 originally
proposed fixing this by using a CRN-null floor or a maximal-coupling floor instead.

Both are structurally ZERO, and that is why this experiment is shaped the way it is. The floor is
twins from the same settled state with NO flip, so at every site the two conditionals are
IDENTICAL. Any coupling with P(X=Y)=1 when p==q keeps identical twins identical forever -- verified
at |V| = 2, 8 and 30522, where both maximal and monotone/CRN agreement are exactly 1.000000. So the
CRN floor is 0 by construction (it is the exact-zero null the suite already asserts) and the
maximal floor is 0 for the same reason. There is no third option inside the coupling family, and
the mismatch W2 objects to is therefore UNAVOIDABLE rather than a choice.

What CAN be varied is the DEGREE of decorrelation. Let alpha be the probability that the two
replicas draw the SAME uniform at a given site-update:

    alpha = 1  ->  full CRN            ->  floor identically 0
    alpha = 0  ->  independent noise   ->  the floor D_norm currently uses

Sweeping alpha traces D0(alpha) between those endpoints and shows how much of D_norm's value is an
artifact of having set alpha=0 by fiat. That is the honest form of the question #34 was reaching
for: not "which coupling should the denominator use" (none can), but "how sensitive is the reading
to where on this axis the denominator sits".

WHAT THIS CAN AND CANNOT SETTLE. It bounds the ARBITRARINESS of D_norm's absolute scale. It does
not rehabilitate that scale -- F39 already showed it moves as 1/N (F45: N^-1.02 over a 4x range)
and F41 showed the numerator's coupling is not extremal. Relative comparisons across checkpoints,
radii and rules are unaffected by any of this, because alpha is a common mode.

Writes results/floor_decorrelation.json. Usage:
  caffeinate -i .venv/bin/python experiments/floor_decorrelation.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp

MODEL = "EleutherAI/pythia-160m"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
SEEDS = [21, 22, 23, 24, 25]
N, B, R, T = 48, 8, 2, 0.7
SETTLE, SWEEPS, TAIL = 12, 22, 8
OUT = str(_ROOT / "results" / "floor_decorrelation.json")


def floor_at_alpha(rule, alpha, seed):
    """Drift floor when the twins share a fraction `alpha` of their uniform draws.

    alpha=0 reproduces `ar_probe.drift_floor` exactly (independent streams); alpha=1 makes the
    two streams identical, which is the CRN null and must give exactly 0.
    """
    from ar_ca import run
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
               init="random", seed=seed)["final"]
    n = SWEEPS * N * B
    ua = np.random.default_rng(seed + 1).random(n)
    ub = np.random.default_rng(seed + 101).random(n)
    share = np.random.default_rng(seed + 201).random(n) < alpha
    ub_mixed = np.where(share, ua, ub)                 # per-draw sharing, not per-run
    u2 = np.concatenate([ua.reshape(SWEEPS * N, B),
                         ub_mixed.reshape(SWEEPS * N, B)], axis=1).reshape(-1)
    init2 = np.concatenate([base, base], axis=0)
    snaps = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                init_state=init2, seed=seed + 2, u_stream=u2)["snaps"]
    return float((snaps[-TAIL:, :B] != snaps[-TAIL:, B:]).mean())


def main():
    from ar_ca import ARRule
    from ar_probe import block_damage, drift_floor
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    runs = res["runs"]
    rule = ARRule(MODEL)
    print(f"floor decorrelation sweep on {MODEL}: alpha in {ALPHAS}, {len(SEEDS)} seeds")
    print("  alpha=1 is full CRN (floor must be EXACTLY 0); alpha=0 is the current floor\n",
          flush=True)

    # the numerator is fixed -- only the denominator's alpha varies
    for sd in SEEDS:
        key_d = f"D_s{sd}"
        if key_d not in runs:
            d = block_damage(rule, T, R, block=3, B=B, N=N, settle=SETTLE, sweeps=SWEEPS,
                             seed=sd, scheme="none")
            runs[key_d] = dict(mean_damage=float(d["mean_damage"]),
                               ignition_prob=float(d["ignition_prob"]))
            json.dump(res, open(OUT, "w"), indent=1)
        for a in ALPHAS:
            key = f"floor_a{a}_s{sd}"
            if key in runs:
                continue
            t0 = time.time()
            d0 = floor_at_alpha(rule, a, sd)
            runs[key] = dict(alpha=a, seed=sd, D0=round(d0, 6),
                             secs=round(time.time() - t0, 1))
            print(f"  alpha={a:<5} seed={sd}: D0={d0:.6f}  ({runs[key]['secs']}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    del rule
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()

    # ---- analysis -------------------------------------------------------------------
    D = np.array([runs[f"D_s{s}"]["mean_damage"] for s in SEEDS])
    print(f"\n=== numerator (fixed): mean damage = {D.mean():.4f} ===")
    print(f"{'alpha':>7} {'D0 mean':>10} {'D0 sd':>9} {'D_norm = D/D0':>15}")
    table = {}
    for a in ALPHAS:
        f = np.array([runs[f"floor_a{a}_s{s}"]["D0"] for s in SEEDS])
        dn = np.array([D[i] / f[i] if f[i] > 0 else np.inf for i in range(len(SEEDS))])
        finite = dn[np.isfinite(dn)]
        table[str(a)] = dict(D0_mean=round(float(f.mean()), 6),
                             D0_sd=round(float(f.std(ddof=1)), 6),
                             D_norm_mean=(None if not len(finite)
                                          else round(float(finite.mean()), 4)),
                             n_undefined=int((~np.isfinite(dn)).sum()))
        shown = "undefined" if not len(finite) else f"{finite.mean():.4f}"
        print(f"{a:>7} {f.mean():>10.6f} {f.std(ddof=1):>9.6f} {shown:>15}")

    crn = table[str(1.0)]
    assert crn["D0_mean"] == 0.0, (
        f"alpha=1 gave a nonzero floor ({crn['D0_mean']}) -- the CRN null is broken, and every "
        f"damage number in the project depends on it being exactly zero")
    print(f"\n  alpha=1 floor is exactly {crn['D0_mean']} -- the CRN null, as required.")

    base = table["0.0"]["D_norm_mean"]
    span = [table[str(a)]["D_norm_mean"] for a in ALPHAS if table[str(a)]["D_norm_mean"]]
    print(f"  D_norm at the conventional alpha=0: {base}")
    print(f"  across alpha in [0, 0.9]: {min(span):.4f} to {max(span):.4f} "
          f"= a factor of {max(span)/min(span):.2f}")

    res["analysis"] = dict(
        numerator_mean_damage=round(float(D.mean()), 6), by_alpha=table,
        D_norm_at_conventional_alpha0=base,
        D_norm_span_excluding_crn=[round(float(min(span)), 4), round(float(max(span)), 4)],
        span_factor=round(float(max(span) / min(span)), 3))
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Sweep of the floor's DECORRELATION, not of the coupling. #34 proposed a CRN-null floor "
        "and a maximal-coupling floor; both are structurally zero, because the floor's twins are "
        "identical and any coupling with P(X=Y)=1 at p==q keeps them so. alpha is the probability "
        "that the two replicas share a uniform draw: alpha=1 is CRN (floor exactly 0, asserted "
        "here), alpha=0 is the conventional independent-noise floor. The span over alpha bounds "
        "how arbitrary D_norm's absolute scale is. It does not rehabilitate that scale -- F45 "
        "showed it also moves as 1/N and F41 that the numerator's coupling is not extremal. "
        "Relative comparisons are unaffected: alpha is a common mode.")
    res["_config"] = dict(model=MODEL, alphas=ALPHAS, seeds=SEEDS, N=N, B=B, r=R, T=T)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
