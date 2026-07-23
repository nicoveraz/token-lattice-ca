"""Issue #14: ground-truth calibration of the criticality instrument.

The black-box token-space Lyapunov lambda_ca is trusted to measure criticality but was
never checked against a rule whose criticality class is KNOWN. Here we drive the SAME
damage-spreading + Lyapunov estimator (lyap_from_cone) with classical Elementary CA rules
(k=2, radius 1) spanning the order->chaos axis, using the SAME protocol as the instrument:
async random-order updates, common-random-number (CRN) twins (shared visit order + shared
noise stream), single-site flip, damage cone -> lambda_ca.

Pre-registered success = lambda_ca recovers the known ordering
    ordered (Class I/II) < edge/complex (Class IV) < chaotic (Class III).
Rule 90 is a linear/ballistic reference. This is the criticality-side analog of the
attractor-census calibration (which recovers a known transition matrix). CPU-only.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json
import numpy as np
from lyapunov import lyap_from_cone

# rule number -> known class (Wolfram / damage-spreading literature)
RULES = [
    (128, "ordered (Class I, nucleating->0)"),
    (232, "ordered (Class II, majority/stable)"),
    (4,   "ordered (Class II, fixed)"),
    (110, "complex / edge (Class IV)"),
    (54,  "complex / edge (Class IV)"),
    (90,  "chaotic-linear (Class III, ballistic ref)"),
    (150, "chaotic-linear (Class III)"),
    (30,  "chaotic (Class III)"),
    (22,  "chaotic (Class III)"),
]


def eca_table(rulenum):
    # index = 4*left + 2*center + right  ->  output bit
    return np.array([(rulenum >> i) & 1 for i in range(8)], dtype=np.int8)


def damage_cone(rulenum, N=64, B=128, sweeps=20, settle=12, eta=0.0, seed=0):
    """Async CRN damage spreading from a single-site flip -> cone (sweeps+1, N)."""
    rng = np.random.default_rng(seed)
    tab = eca_table(rulenum)

    def update(X, idx, u):
        l = X[:, (idx - 1) % N]; c = X[:, idx]; r = X[:, (idx + 1) % N]
        b = tab[4 * l + 2 * c + r]
        X[:, idx] = np.where(u < eta, 1 - b, b)

    A = rng.integers(0, 2, size=(B, N), dtype=np.int8)
    for _ in range(settle):                                  # settle onto the rule's dynamics
        for idx in rng.permutation(N):
            update(A, idx, rng.random(B))
    Bl = A.copy(); Bl[:, N // 2] ^= 1                        # twin: single-site flip
    cone = [(A != Bl).mean(0)]
    for _ in range(sweeps):
        for idx in rng.permutation(N):
            u = rng.random(B)                                # CRN: shared order + shared noise
            update(A, idx, u); update(Bl, idx, u)
        cone.append((A != Bl).mean(0))
    return np.asarray(cone)


def main():
    seeds = [0, 1, 2]
    res = {}
    print(f"{'rule':>5s}  {'lambda_ca':>10s}  {'dmax/N':>7s}   class")
    for rulenum, cls in RULES:
        lams, dm = [], []
        for sd in seeds:
            cone = damage_cone(rulenum, seed=sd)
            lam, dmax = lyap_from_cone(cone, 64)
            lams.append(lam); dm.append(dmax)
        lam = float(np.mean(lams))
        res[str(rulenum)] = dict(rule=rulenum, cls=cls, lambda_ca=round(lam, 4),
                                 lambda_ca_se=round(float(np.std(lams) / len(lams) ** 0.5), 4),
                                 dmax_frac=round(float(np.mean(dm)), 3))
        print(f"{rulenum:5d}  {lam:+10.4f}  {np.mean(dm):7.3f}   {cls}")
    # verdict: does lambda_ca rank ordered < edge < chaotic?
    grp = {"ordered": [128, 232, 4], "edge": [110, 54], "chaotic": [90, 150, 30, 22]}
    means = {g: float(np.mean([res[str(r)]["lambda_ca"] for r in rs])) for g, rs in grp.items()}
    res["_group_means"] = {g: round(v, 4) for g, v in means.items()}
    ok = means["ordered"] < means["edge"] < means["chaotic"]
    res["_ordering_recovered"] = bool(ok)
    print(f"\ngroup-mean lambda_ca:  ordered={means['ordered']:+.3f}  "
          f"edge={means['edge']:+.3f}  chaotic={means['chaotic']:+.3f}")
    print(f"ordering ordered<edge<chaotic recovered: {ok}")
    out = str(_ROOT / "results" / "eca_calib.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
