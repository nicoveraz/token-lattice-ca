"""The damage cone has never been looked at. Area, shape, asymmetry, and the two sum vectors.

WHAT IS DISCARDED TODAY. `ar_probe.block_damage` builds a (sweeps, N) damage field and returns it,
and every consumer immediately collapses it to a scalar: lyap_from_cone takes a growth rate,
cone_front_velocity takes an edge slope, mean_damage takes the final level, ignition_prob takes a
yes/no. **No results file in this repository stores a cone.** The largest object the instrument
produces is reduced to four numbers and thrown away, and this session has now found six times that
a scalar summary hid the structure that mattered (F94, F96, F99, F109, F110, F115).

THE KNOWN-ANSWER RUNG, AND IT IS FREE. The AR construction's window is `np.arange(i - r, i)` --
strictly the r cells to the LEFT. So site i reads i-2 and i-1, which means site i can only influence
i+1 and i+2: **damage must spread rightward and cannot spread leftward at all.** The cone is
therefore predicted to be ONE-SIDED, exactly, as a matter of the update rule rather than of the
model. Any leftward mass beyond the injected block is a bug in the harness, not a property of a
language model. Nothing has ever checked this, and it gates everything else here.

WHAT IS MEASURED, none of which exists anywhere in the project:
  area          the integral of the damage field over (sweep, site) -- total damaged site-sweeps,
                the natural extensive quantity next to lambda's intensive one
  asymmetry     right-mass / (left+right) about the injection, whose predicted value is 1.0
  fill          area divided by the area of the light cone the front velocity implies -- a SOLID
                cone (fill near 1) and a HOLLOW one (front races ahead of a sparse interior) have
                the same velocity and the same lambda, and are different dynamics
  front_width   how many sites the edge takes to go from 10% to 90% damaged -- sharp versus diffuse
  t_marginal    damage summed over sites, per sweep: the growth curve lambda is fitted from
  x_marginal    damage summed over sweeps, per site: where damage spends its time

PRE-REGISTERED:
  RUNG        asymmetry must be 1.000 (up to the injected block's own width) on the AR backend. A
              failure stops the script -- it would mean the causal window is not being applied.
  PRIMARY     do area, fill and front_width vary across checkpoints in a way lambda does not? A
              scalar that moves where lambda is flat is a new observable; one that tracks lambda
              exactly is a re-derivation and says so.
  DEFLATION, frozen: if every shape scalar is a monotone function of lambda, the cone carries no
              information beyond it and this closes. Reported as rho against lambda for each.
  BOUNDARY    one family, one radius, one temperature. This is a geometry measurement, not a
              model-facing claim.

Writes results/damage_geometry.json, INCLUDING the cones themselves (downsampled) so the next
question about shape does not need a re-run.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, itertools, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from lyapunov import lyap_from_cone, is_unignited
from dev_transition_phase3 import FIT_KW

OUT = str(_ROOT / "results" / "damage_geometry.json")
MODEL = "EleutherAI/pythia-410m"
STEPS = [128, 256, 512, 1000, 2000, 4000]
R, T, N, B, SETTLE, SWEEPS, BLOCK = 2, 0.7, 48, 16, 12, 22, 3
SEEDS = [21, 22, 23, 24]


def cone_of(rule, seed):
    """block_damage's field, NOT rolled to centre -- the roll would destroy the asymmetry."""
    from ar_ca import run
    rng = np.random.default_rng(seed)
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none", init="random",
               seed=seed)["final"]
    c = N // 2
    idx = [c + k for k in range(-(BLOCK // 2), BLOCK - BLOCK // 2)]
    fl = base.copy()
    for j in idx:
        fl[:, j] = rng.choice(rule.init_pool, size=B)
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
    c2 = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
             init_state=np.concatenate([base, fl], axis=0), seed=seed + 2, u_stream=u2)
    s = c2["snaps"]
    diff = (s[:, :B] != s[:, B:])                       # (sweeps, B, N), NOT recentred
    return diff.mean(axis=1), idx                        # (sweeps, N)


def geometry(field, idx):
    """Area, asymmetry, fill, front width, and the two sum vectors. Injection at `idx`."""
    sw, n = field.shape
    c = int(np.mean(idx))
    off = (np.arange(n) - c + n // 2) % n - n // 2        # signed offset from injection, on the ring
    half = BLOCK // 2 + 1
    # WRAPAROUND. A left window means site i influences only i+1 and i+2, so the front advances
    # rightward at r sites per sweep and reaches r*sweeps = 44 sites on a ring of N = 48. Past
    # t = N/(2r) the cone wraps and RIGHTWARD mass reappears at negative offsets, making a
    # one-sided process look symmetric. That is F21's finite-size wraparound -- the same artifact
    # that retracted a velocity plateau in this project -- and the first version of this rung
    # walked into it, reporting asymmetry 0.39-0.70 and calling the harness broken.
    # The asymmetry is therefore measured only over sweeps the cone cannot have wrapped in.
    t_wrap = int(n // (2 * R))
    win = field[:t_wrap]
    right = float(win[:, off > half].sum())
    left = float(win[:, off < -half].sum())
    area = float(field.sum())
    tmarg = field.sum(axis=1)
    # the light cone the front velocity implies: |offset| <= r * t
    t = np.arange(sw)[:, None]
    inside = (np.abs(off)[None, :] <= R * t)
    fill = area / max(float(inside.sum()), 1.0)
    # front width: sites between the 10% and 90% damage level along the final sweep's right flank
    fin = field[-1]
    rf = fin[(off > 0)][np.argsort(off[off > 0])]
    if rf.max() > 0:
        lo = np.argmax(rf <= 0.9 * rf.max()); hi = np.argmax(rf <= 0.1 * rf.max())
        width = float(max(hi - lo, 0))
    else:
        width = float("nan")
    return dict(area=round(area, 4), right_mass=round(right, 4), left_mass=round(left, 4),
                t_wrap=int(t_wrap),
                asymmetry=round(right / max(right + left, 1e-12), 5),
                fill=round(fill, 5), front_width=round(width, 3),
                t_marginal=[round(float(v), 4) for v in tmarg],
                x_marginal=[round(float(v), 4) for v in field.sum(axis=0)])


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=STEPS, seeds=SEEDS, r=R, T=T, N=N, B=B, sweeps=SWEEPS, block=BLOCK,
        rung="the AR window is strictly left, so damage can only spread RIGHTWARD: asymmetry must "
             "be 1.000 up to the injected block's width, measured ONLY over the first N/(2r) "
             "sweeps -- past that the cone wraps the ring and rightward mass reappears at negative "
             "offsets (F21's finite-size wraparound). A failure stops the script",
        primary="do area, fill and front_width vary across checkpoints where lambda does not?",
        deflation="if every shape scalar is a monotone function of lambda, the cone carries nothing "
                  "beyond it and this closes; rho against lambda is reported for each",
        stores="the cones themselves, so the next shape question needs no re-run",
        boundary="one family, one radius, one temperature; a geometry measurement, not a "
                 "model-facing claim")
    from ar_ca import ARRule
    for st in STEPS:
        if all(f"step{st}|s{sd}" in res["cells"] for sd in SEEDS):
            continue
        rule = ARRule(MODEL, revision=f"step{st}")
        for sd in SEEDS:
            k = f"step{st}|s{sd}"
            if k in res["cells"]:
                continue
            t0 = time.time()
            field, idx = cone_of(rule, sd)
            g = geometry(field, idx)
            rolled = np.roll(field, N // 2 - int(np.mean(idx)), axis=1)
            g.update(step=st, seed=sd,
                     lambda_ca=round(float(lyap_from_cone(rolled, N, **FIT_KW)[0]), 5),
                     ignited=bool(not is_unignited(mean_damage=float(field[-1].mean()))),
                     cone=[[round(float(v), 3) for v in row] for row in field],
                     secs=round(time.time() - t0, 1))
            res["cells"][k] = g
            print(f"  {k:16s} area={g['area']:>7.1f} asym={g['asymmetry']:.4f} "
                  f"fill={g['fill']:.3f} width={g['front_width']:>5.1f} "
                  f"lam={g['lambda_ca']:+.4f} ({g['secs']:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _rho_p(a, b):
    a, b = np.array(a, float), np.array(b, float)
    rk = lambda x: np.argsort(np.argsort(x))
    r = float(np.corrcoef(rk(a), rk(b))[0, 1])
    null = [np.corrcoef(np.array(p), rk(b))[0, 1] for p in itertools.permutations(rk(a))]
    return r, float(np.mean(np.abs(np.array(null)) >= abs(r) - 1e-12))


def analyse(res):
    cs = [c for c in res["cells"].values() if "area" in c]
    parts = []
    asym = [c["asymmetry"] for c in cs]
    rung = bool(cs and min(asym) >= 0.999)
    parts.append(
        f"RUNG (the causal window, checked rather than assumed): damage asymmetry about the "
        f"injection is {min(asym):.4f}-{max(asym):.4f} across {len(cs)} runs. "
        + ("Exactly one-sided, as a strictly-left window requires -- the cone spreads rightward and "
           "carries no leftward mass at all."
           if rung else
           "NOT one-sided. A left-window update cannot propagate damage leftward, so this is a "
           "harness bug and nothing below is read."))
    if not rung:
        res["analysis"] = dict(rung_passes=False, asymmetry_range=[min(asym), max(asym)])
        res["verdict"] = " ".join(parts); res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}"); return
    rows = {}
    for st in STEPS:
        v = [c for c in cs if c["step"] == st and c["ignited"]]
        if v:
            rows[st] = {k: round(float(np.mean([c[k] for c in v])), 4)
                        for k in ("area", "fill", "front_width", "lambda_ca")}
    print(f"\n  {'step':>6} {'area':>9} {'fill':>8} {'front_width':>12} {'lambda':>9}")
    for st, v in rows.items():
        print(f"  {st:>6} {v['area']:>9.1f} {v['fill']:>8.3f} {v['front_width']:>12.1f} "
              f"{v['lambda_ca']:>+9.4f}")
    if len(rows) >= 5:
        lam = [rows[s]["lambda_ca"] for s in rows]
        det = {}
        for k in ("area", "fill", "front_width"):
            vals = [rows[s][k] for s in rows]
            r, p = _rho_p(vals, lam)
            det[k] = dict(rho=round(r, 3), perm_p=round(p, 4),
                          span=round(float(max(vals) - min(vals)), 4))
            print(f"  rho({k:<12}, lambda) = {r:+.3f}  p={p:.4f}")
        free = [k for k, v in det.items() if abs(v["rho"]) < 0.6]
        parts.append(
            f"PRIMARY: across checkpoints, area/fill/front_width correlate with lambda at "
            + ", ".join(f"{k} {v['rho']:+.3f}" for k, v in det.items()) + ". "
            + (f"{len(free)} of 3 move largely independently of lambda ({free}), so the cone's "
               f"SHAPE carries information its growth rate does not -- the first quantity in this "
               f"project extracted from the field rather than from a fit to it."
               if free else
               "Every shape scalar tracks lambda, so the cone carries nothing beyond its growth "
               "rate and this closes -- the registered deflation."))
        res["analysis"] = dict(rung_passes=True, rows=rows, vs_lambda=det,
                               independent=free, asymmetry_range=[min(asym), max(asym)])
    parts.append(
        "BOUNDARY: one family, one radius, one temperature. A geometry measurement of the "
        "construction, not a model-facing claim. The cones are stored, so the next shape question "
        "needs no re-run.")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("First measurement of the damage cone as a FIELD rather than as four scalars. "
                    "No results file in this repo had ever stored one. Includes the causal-window "
                    "rung: a strictly-left update must give exactly one-sided spread.")


if __name__ == "__main__":
    main()
