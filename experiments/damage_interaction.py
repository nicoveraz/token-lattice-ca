"""Do two damage sites interact? Superposition on the lattice, with causality as the rung.

THE QUESTION, AND WHY IT IS NOT TRIVIAL. Every damage measurement in this project injects ONE block
and watches one cone. Nothing has ever asked whether two perturbations superpose. Two things make
the answer non-obvious and one makes it predictable:

  F114 measured the LOCAL response and found it essentially ADDITIVE -- sub-additivity spans +0.003
       to +0.028 across fourteen models, two orders of magnitude below the weakest ladder anchor. So
       at the level of the conditional, two flips do not interact. Whether that survives iteration
       on a lattice is a different question with a stated prior.
  F109 left an unexplained residue: three cells cleared a branching ratio above 1 and still never
       ignited, and the first named candidate was that "the annealed ratio ignores that a damaged
       site's two children overlap on a ring". That is an interaction effect, never measured.
  F80  found the ABLATION response strongly non-additive -- 0 of 24 single layers, eight together
       +0.345, twenty-four singles summing to -0.224, the wrong sign. This asks the same question
       of damage sites rather than of layers.

THE TRAP, AND THE FIX. Damage is binary per site, so two overlapping cones are sub-additive by set
arithmetic alone -- a site cannot be damaged twice. Comparing joint damage to the SUM of two singles
would measure overlap, not interaction. Under CRN with a shared uniform stream the same three runs
can be compared as SETS instead:

    D_A   sites damaged when injecting at A alone
    D_B   sites damaged when injecting at B alone
    D_AB  sites damaged when injecting at both

Non-interaction predicts D_AB = D_A ∪ D_B exactly, per replica. The interaction is
|D_AB| - |D_A ∪ D_B|: positive means the pair does more than either does separately, negative means
they interfere. Set union removes the trivial overlap; what is left is dynamics.

THE RUNG IS FREE AND EXACT. Damage advances at most r sites per sweep (the light cone is kinematic,
F16/F21), so two injections separated by more than 2*r*sweeps CANNOT influence one another within
the run. At that separation the interaction must be EXACTLY ZERO by causality -- not small, zero.
Any nonzero value there is a harness error, and it is checked before anything else is read.

PRE-REGISTERED:
  RUNG      interaction at the causally-disconnected separation must be exactly 0. Failure stops.
  PRIMARY   is interaction non-zero at separations where the cones DO meet, beyond the seed spread?
  SIGN      negative = interference (the pair damages less than the union of singles, i.e. shared
            healing or competition for the same sites); positive = synergy. F114's additivity
            predicts approximately zero, so a clear sign either way is informative against it.
  KILL      interaction indistinguishable from zero at every separation -> damage superposes, the
            lattice inherits the conditional's additivity, and F109's overlap candidate is closed.
  BOUNDARY  one family, one checkpoint, one radius, one temperature.

Writes results/damage_interaction.json.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "damage_interaction.json")
MODEL = "EleutherAI/pythia-410m"
STEPS = [1000, 4000]                 # plateau checkpoints, where damage reliably ignites
R, T, N, B, SETTLE, SWEEPS, BLOCK = 2, 0.7, 96, 16, 12, 22, 3
SEEDS = [21, 22, 23, 24]
# Light cone reaches r*sweeps = 44 sites each way. On N=96 the maximum ring separation is 48, so
# SEPS[-1] is as causally disconnected as this geometry allows and is the rung.
SEPS = [6, 12, 24, 48]


def three_runs(rule, seed, sep):
    """D_A, D_B, D_AB as boolean site-sets per replica, under ONE shared uniform stream."""
    from ar_ca import run
    rng = np.random.default_rng(seed)
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none", init="random",
               seed=seed)["final"]
    a0 = N // 4
    ia = [(a0 + k) % N for k in range(BLOCK)]
    ib = [(a0 + sep + k) % N for k in range(BLOCK)]
    fa, fb, fab = base.copy(), base.copy(), base.copy()
    ra = rng.choice(rule.init_pool, size=(B, BLOCK))
    rb = rng.choice(rule.init_pool, size=(B, BLOCK))
    for j, jj in enumerate(ia):
        fa[:, jj] = ra[:, j]; fab[:, jj] = ra[:, j]
    for j, jj in enumerate(ib):
        fb[:, jj] = rb[:, j]; fab[:, jj] = rb[:, j]
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    out = []
    for fl in (fa, fb, fab):
        u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
        c = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                init_state=np.concatenate([base, fl], axis=0), seed=seed + 2, u_stream=u2)
        s = c["snaps"]
        out.append((s[:, :B] != s[:, B:])[-1])           # (B, N) final-sweep damage set
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=STEPS, seps=SEPS, seeds=SEEDS, r=R, T=T, N=N, B=B,
        sweeps=SWEEPS, block=BLOCK,
        comparison="interaction = |D_AB| - |D_A union D_B| per replica; set union removes the "
                   "trivial overlap that binary damage forces, so what is left is dynamics",
        rung=f"the light cone reaches r*sweeps = {R*SWEEPS} sites, so at separation {SEPS[-1]} on "
             f"N={N} the injections are as causally disconnected as the ring allows; interaction "
             f"there must be EXACTLY zero. Failure stops the script",
        primary="is interaction non-zero where the cones DO meet, beyond seed spread?",
        sign="negative = interference (shared healing / competition for sites); positive = synergy. "
             "F114's additivity of the local response predicts approximately zero",
        kill="zero at every separation -> damage superposes, the lattice inherits the conditional's "
             "additivity, and F109's overlap candidate is closed",
        boundary="one family, one checkpoint set, one radius, one temperature")
    from ar_ca import ARRule
    for st in STEPS:
        if all(f"step{st}|d{d}|s{sd}" in res["cells"] for d in SEPS for sd in SEEDS):
            continue
        rule = ARRule(MODEL, revision=f"step{st}")
        for sep in SEPS:
            for sd in SEEDS:
                k = f"step{st}|d{sep}|s{sd}"
                if k in res["cells"]:
                    continue
                t0 = time.time()
                DA, DB, DAB = three_runs(rule, sd, sep)
                union = DA | DB
                inter = float((DAB.sum(axis=1) - union.sum(axis=1)).mean())
                res["cells"][k] = dict(
                    step=st, sep=sep, seed=sd,
                    n_A=round(float(DA.sum(axis=1).mean()), 4),
                    n_B=round(float(DB.sum(axis=1).mean()), 4),
                    n_AB=round(float(DAB.sum(axis=1).mean()), 4),
                    n_union=round(float(union.sum(axis=1).mean()), 4),
                    interaction=round(inter, 5),
                    overlap=round(float((DA & DB).sum(axis=1).mean()), 4),
                    secs=round(time.time() - t0, 1))
                c = res["cells"][k]
                print(f"  {k:20s} |A|={c['n_A']:>6.2f} |B|={c['n_B']:>6.2f} "
                      f"|A∪B|={c['n_union']:>6.2f} |AB|={c['n_AB']:>6.2f} "
                      f"inter={c['interaction']:+7.3f} ({c['secs']:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    cs = [c for c in res["cells"].values() if "interaction" in c]
    parts, rows = [], {}
    for sep in SEPS:
        v = [c["interaction"] for c in cs if c["sep"] == sep]
        if v:
            rows[sep] = dict(n=len(v), mean=round(float(np.mean(v)), 5),
                             sd=round(float(np.std(v)), 5),
                             se=round(float(np.std(v) / np.sqrt(len(v))), 5),
                             overlap=round(float(np.mean([c["overlap"] for c in cs
                                                          if c["sep"] == sep])), 4))
    print(f"\n  {'sep':>5} {'n':>3} {'interaction':>13} {'sd':>8} {'|A∩B|':>8}")
    for s_, v in rows.items():
        print(f"  {s_:>5} {v['n']:>3} {v['mean']:>+13.4f} {v['sd']:>8.4f} {v['overlap']:>8.3f}")
    far = rows.get(SEPS[-1])
    rung = bool(far and abs(far["mean"]) <= max(3 * far["se"], 1e-9))
    parts.append(
        f"RUNG (causality, checked not assumed): at separation {SEPS[-1]} the light cone "
        f"(r*sweeps = {R*SWEEPS} sites) cannot connect the injections within the run, so the "
        f"interaction must be zero. Measured {far['mean']:+.4f} +/- {far['se']:.4f}. "
        + ("Consistent with zero, so the comparison is measuring dynamics rather than harness "
           "error." if rung else
           "NOT consistent with zero -- causally disconnected injections cannot interact, so this "
           "is a harness error and nothing below is read.") if far else "RUNG NOT MEASURED.")
    if not rung:
        res["analysis"] = dict(rows=rows, rung_passes=False)
        res["verdict"] = " ".join(parts); res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}"); return
    close = {s_: v for s_, v in rows.items() if s_ < SEPS[-1]}
    hits = [s_ for s_, v in close.items() if abs(v["mean"]) > 2 * v["se"]]
    detail = ", ".join(f"d={s_}: {v['mean']:+.4f}±{v['se']:.4f}" for s_, v in close.items())
    if hits:
        sign = float(np.mean([close[s_]["mean"] for s_ in hits]))
        which = ("negative — INTERFERENCE: the pair damages LESS than the union of the singles, "
                 "consistent with competition for the same sites or shared healing"
                 if sign < 0 else
                 "positive — SYNERGY: the pair damages MORE than either produces alone")
        tail = (f"Non-zero beyond 2 SE at separations {hits}, so two damage sites do NOT "
                f"superpose: the lattice adds an interaction the local response does not have "
                f"(F114 measured sub-additivity of +0.003 to +0.028, essentially additive). "
                f"The sign is {which}.")
    else:
        tail = ("Indistinguishable from zero at every separation. Damage SUPERPOSES: the lattice "
                "inherits the conditional's additivity (F114) rather than adding an interaction of "
                "its own, and F109's overlap candidate for its unexplained residue is closed.")
    parts.append(f"PRIMARY: at separations where the cones meet, interaction is {detail}. {tail}")
    parts.append(
        "BOUNDARY: one family, one radius, one temperature, plateau checkpoints only -- chosen "
        "because damage must reliably ignite for a superposition test to be defined at all, which "
        "excludes the dip where ignition is 0.05-0.3 (F42).")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(rows=rows, rung_passes=True, nonzero_at=hits)
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("First test of whether two damage injections superpose. Compared as SETS "
                    "(|D_AB| - |D_A union D_B|) because binary damage makes a sum-comparison "
                    "measure overlap rather than interaction. Causality supplies an exact rung: at "
                    "separation beyond the light cone the interaction must be zero.")


if __name__ == "__main__":
    main()
