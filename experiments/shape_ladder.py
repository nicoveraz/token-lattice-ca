"""Does cone SHAPE recover a distinction that ignition probability cannot? The ECA rung.

THE QUESTION. `damage_geometry.py` measures the damage cone as a field -- area, fill, front width,
and the two sum vectors -- where the project has only ever taken a growth rate and an edge slope
from it. But a new observable is worth nothing until it is shown to discriminate where the answer is
known, which is this project's founding rule.

The ECA rung supplies exactly that, and with real stakes. F33/F34/F36 established:
  * ordered-vs-rest SEPARATES on ignition probability: p = 0.0000, Cohen d = 3.03
  * edge-vs-chaotic DOES NOT: p = 0.47, and the 3-class ordering was demoted as a result
So there is a distinction the project's best existing statistic provably cannot make, on a system
whose classes are known independently. If cone shape makes it, shape is a strictly better observable
than anything here. If it does not, the 3-class ordering fails a fourth time and shape inherits the
same ceiling -- which is worth knowing before shape is quoted on a language model.

PRE-REGISTERED:
  CONTROL, read first: shape must separate ORDERED from the rest. That separation is known to exist
           and is easy; a shape measure that misses it is not a measure of dynamics and nothing
           else is read. This is the rung, not a formality.
  PRIMARY  does any shape scalar separate EDGE from CHAOTIC beyond the seed spread, where ignition
           probability gives p = 0.47?
  KILL     no shape scalar separates edge from chaotic -> the 3-class ordering is not recoverable
           from cone geometry either, and shape carries no more class information than ignition.
           That is a real outcome: it bounds what the new observable can be used for.
  BOUNDARY ECA is deterministic and binary; a language-model lattice is stochastic over a large
           vocabulary. A shape statistic that works here is licensed as a DYNAMICAL discriminator,
           not automatically as a model-facing one.

Writes results/shape_ladder.json.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import itertools, json
import numpy as np
from provenance import stamp, rel
from eca_calib import damage_cone

OUT = str(_ROOT / "results" / "shape_ladder.json")
SEEDS = list(range(12))
N, B, SWEEPS = 64, 128, 20


def shape_of(cone):
    """Area, fill, front width and the marginals, from a (sweeps, N) damage field."""
    sw, n = cone.shape
    c = n // 2
    off = (np.arange(n) - c + n // 2) % n - n // 2
    t = np.arange(sw)[:, None]
    area = float(cone.sum())
    # ECA is symmetric (3-cell neighbourhood), so the implied cone is |off| <= t
    inside = (np.abs(off)[None, :] <= t)
    fill = area / max(float(inside.sum()), 1.0)
    fin = cone[-1]
    rf = fin[off > 0][np.argsort(off[off > 0])]
    if rf.max() > 0:
        lo = int(np.argmax(rf <= 0.9 * rf.max())); hi = int(np.argmax(rf <= 0.1 * rf.max()))
        width = float(max(hi - lo, 0))
    else:
        width = 0.0
    tm = cone.sum(axis=1)
    # growth curvature: is the total-damage curve convex, linear or saturating?
    k = np.polyfit(np.arange(len(tm)), tm, 2)[0] if len(tm) > 3 else float("nan")
    return dict(area=round(area, 4), fill=round(fill, 5), front_width=round(width, 3),
                curvature=round(float(k), 5), final=round(float(cone[-1].mean()), 5))


def main():
    eca = json.load(open(_ROOT / "results" / "eca_ordered_vs_rest.json"))
    classes = {int(r): g for g in ("ordered", "edge", "chaotic")
               for r in eca.get("groups", {}).get(g, {}).get("rules", {})}
    res = {"cells": {}, "_preregistration": dict(
        rules=sorted(classes), classes=classes, seeds=SEEDS, N=N, B=B, sweeps=SWEEPS,
        control="shape must separate ORDERED from the rest, a separation known to exist "
                "(F36: p = 0.0000, d = 3.03). A shape measure that misses it is not a measure of "
                "dynamics and nothing else is read",
        primary="does any shape scalar separate EDGE from CHAOTIC beyond seed spread, where "
                "ignition probability gives p = 0.47 (F36)?",
        kill="no shape scalar separates edge from chaotic -> the 3-class ordering is not "
             "recoverable from cone geometry either and shape inherits ignition's ceiling",
        boundary="ECA is deterministic and binary; a shape statistic validated here is licensed as "
                 "a DYNAMICAL discriminator, not automatically as a model-facing one")}
    for rule in sorted(classes):
        rows = []
        for sd in SEEDS:
            rows.append(shape_of(np.asarray(damage_cone(rule, N=N, B=B, sweeps=SWEEPS, seed=sd))))
        agg = {k: round(float(np.mean([r[k] for r in rows])), 5) for k in rows[0]}
        agg.update({f"{k}_sd": round(float(np.std([r[k] for r in rows])), 5) for k in rows[0]})
        agg.update(rule=rule, group=classes[rule], n=len(rows))
        res["cells"][str(rule)] = agg
        print(f"  rule {rule:>3} {classes[rule]:<8} area={agg['area']:>8.1f} fill={agg['fill']:.3f} "
              f"width={agg['front_width']:>5.1f} curv={agg['curvature']:+.4f}", flush=True)
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _perm_p(a, b, seed=0, n=20000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); g = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        p = g.permutation(pool)
        if abs(p[:len(a)].mean() - p[len(a):].mean()) >= obs - 1e-12:
            hits += 1
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-12)
    return float(hits / n), float(d)


def analyse(res):
    cs = list(res["cells"].values())
    by = {}
    for c in cs:
        by.setdefault(c["group"], []).append(c)
    METRICS = ["area", "fill", "front_width", "curvature"]
    parts, ctrl, prim = [], {}, {}
    for m in METRICS:
        o = [c[m] for c in by.get("ordered", [])]
        rest = [c[m] for c in cs if c["group"] != "ordered"]
        if o and rest:
            p, d = _perm_p(o, rest)
            ctrl[m] = dict(p=round(p, 4), d=round(d, 3))
        e = [c[m] for c in by.get("edge", [])]
        ch = [c[m] for c in by.get("chaotic", [])]
        if e and ch:
            p, d = _perm_p(e, ch, seed=1)
            prim[m] = dict(p=round(p, 4), d=round(d, 3), n_edge=len(e), n_chaotic=len(ch))
    print(f"\n  {'metric':<14} {'ordered vs rest':>22} {'edge vs chaotic':>22}")
    for m in METRICS:
        c_, p_ = ctrl.get(m, {}), prim.get(m, {})
        print(f"  {m:<14} p={c_.get('p','-'):>8} d={c_.get('d','-'):>7}   "
              f"p={p_.get('p','-'):>8} d={p_.get('d','-'):>7}")
    passes = [m for m, v in ctrl.items() if v["p"] < 0.01]
    parts.append(
        f"CONTROL (the rung, read first): shape separates ORDERED from the rest on "
        f"{len(passes)} of {len(METRICS)} scalars ({passes}), against F36's ignition probability at "
        f"p = 0.0000, d = 3.03. "
        + ("Shape reproduces a separation that is known to exist, so it is measuring dynamics."
           if passes else
           "Shape MISSES a separation known to exist, so it is not measuring dynamics and nothing "
           "below is read."))
    if not passes:
        res["analysis"] = dict(control=ctrl, rung_passes=False)
        res["verdict"] = " ".join(parts); res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}"); return
    hits = [m for m, v in prim.items() if v["p"] < 0.05]
    parts.append(
        f"PRIMARY: edge vs chaotic separates on {len(hits)} of {len(METRICS)} shape scalars "
        f"({hits or 'none'}), where ignition probability gives p = 0.47. "
        + (f"Cone SHAPE recovers a class distinction the project's best existing statistic cannot, "
           f"on a system whose classes are known independently. That licenses shape as a dynamical "
           f"discriminator and reopens the 3-class ordering demoted by F33/F34/F36."
           if hits else
           f"No shape scalar separates them either. The 3-class ordering fails a fourth time, and "
           f"shape inherits ignition's ceiling -- it is not a finer discriminator of dynamical "
           f"class, which bounds what it can be used for on a language model."))
    parts.append(
        "BOUNDARY: ECA is deterministic and binary; a language-model lattice is stochastic over a "
        "large vocabulary. A shape statistic validated here is licensed as a DYNAMICAL "
        "discriminator, not automatically as a model-facing one.")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(control=ctrl, primary=prim, rung_passes=True,
                           control_hits=passes, primary_hits=hits)
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Tests whether damage-cone SHAPE discriminates dynamical class where ignition "
                    "probability cannot. F36 separates ordered-vs-rest at p=0.0000 but edge-vs-"
                    "chaotic at p=0.47; this asks whether cone geometry does better on the same "
                    "19 rules of known class.")


if __name__ == "__main__":
    main()
