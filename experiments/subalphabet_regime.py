"""Is there ANY live damage regime for a sub-alphabet lattice? The temperature sweep.

WHY THIS EXISTS. #105 (ordering), #106 (|V|=2) and #107 (successor) all ran and all returned
nothing usable at T=0.7, for one shared reason that took two wrong guesses to find. #106's
registered kill fired: the binary ring FREEZES (mean dominant-token share 0.978, and 1.000 at the
two earliest checkpoints), damage injected into it decays MONOTONICALLY from the injection and
coalesces to exactly zero within ~4 sweeps, so lambda_ca is undefined by F42 and #105 never had a
lambda to compute an ordering spread over.

WHY THE OBVIOUS DIAGNOSIS WAS WRONG, TWICE, AND WHAT IT COST. First guess: projecting onto a small
support destroys the window-dependence, so damage cannot start. Refuted directly -- projected s_crn
is 0.61 (colours), 0.69 (digits), 0.63 (binary) against 0.91 on the full vocabulary, and r*s = 1.26
predicts GROWTH. Second guess: the twins coalesce because the state space is small. Also not the
mechanism as stated. What the trajectory actually shows is that s was measured on RANDOM
sub-alphabet windows while the ring settles to a nearly homogeneous state where the twins' windows
coincide and s collapses. That is F96/F99's regime lesson for the third time: a sensitivity measured
on the wrong ensemble predicts the opposite of what the lattice does.

SO THE QUESTION IS NOT "does the construction work" BUT "is there a temperature where it is alive".
Freezing is a low-temperature phenomenon; the whole sub-alphabet family is untestable until a regime
exists where the ring is not frozen AND damage ignites. This screens for one before any of the three
issues is re-run, rather than re-running them hopefully at a guessed T.

PRE-REGISTERED:
  PRIMARY   is there any (alphabet, T) cell with ignition >= IGN_MIN AND settled top-share <
            FROZEN_MAX? That pair is the live regime the three issues need. Both thresholds are
            declared here, before the sweep.
  REPORTED  the damage TRAJECTORY per cell, not just its endpoint, so "never rose" is distinguished
            from "rose and died" by inspection rather than by inference. Getting that distinction
            wrong is what cost two turns.
  CONTROL   T=0.7 must reproduce the frozen, non-igniting behaviour already measured. If it does
            not, the sweep is not measuring the same construction.
  KILL      no live cell anywhere on the grid -> the sub-alphabet lattice has no usable regime, and
            #105/#106/#107 are closed as designed rather than left open. That is a real outcome:
            it says the restriction itself, not the choice of alphabet, is what kills the dynamics.
  BOUNDARY  one model, one checkpoint, one radius. A negative here bounds this construction, not
            token-lattice CAs in general.

Writes results/subalphabet_regime.json.  Resumable per (alphabet, T, seed).
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from subalphabet import pick_tokens, damage_on_sub, lambda_of, COLOURS, BINARY, DIGITS

OUT = str(_ROOT / "results" / "subalphabet_regime.json")
MODEL, REV = "EleutherAI/pythia-410m", "step4000"
R, B, N, SETTLE, SWEEPS = 2, 16, 48, 12, 22
TEMPS = [0.7, 1.0, 1.3, 1.6, 2.0, 2.5]
SEEDS = [21, 22]
ALPHABETS = [("binary", BINARY), ("colours", COLOURS), ("digits", DIGITS)]
IGN_MIN, FROZEN_MAX = 0.50, 0.90       # declared before the sweep


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        model=MODEL, revision=REV, temps=TEMPS, seeds=SEEDS, r=R, B=B, N=N,
        alphabets=[a for a, _ in ALPHABETS], ign_min=IGN_MIN, frozen_max=FROZEN_MAX,
        primary=f"any (alphabet, T) with ignition >= {IGN_MIN} AND settled top-share < {FROZEN_MAX}",
        reported="the damage trajectory per cell, so 'never rose' is distinguished from 'rose and "
                 "died' by inspection rather than inference",
        control="T=0.7 must reproduce the frozen, non-igniting behaviour already measured",
        kill="no live cell anywhere -> the sub-alphabet lattice has no usable regime and "
             "#105/#106/#107 close as designed",
        follows="#105/#106/#107 all returned nothing usable at T=0.7; #106's registered kill fired "
                "at mean top-share 0.978")
    from ar_ca import ARRule
    rule = ARRule(MODEL, revision=REV)
    for name, words in ALPHABETS:
        ids, kept, dropped = pick_tokens(rule.tok, words)
        for T in TEMPS:
            for sd in SEEDS:
                k = f"{name}|T{T}|s{sd}"
                if k in res["cells"]:
                    continue
                t0 = time.time()
                base, rolled = damage_on_sub(rule, ids, None, T=T, r=R, B=B, N=N,
                                             settle=SETTLE, sweeps=SWEEPS, seed=sd)
                traj = (rolled.sum(axis=(1, 2)) / rolled.shape[1]).tolist()
                vals, cnts = np.unique(base, return_counts=True)
                row = lambda_of(rolled, N)
                row.update(alphabet=name, T=T, seed=sd, k=len(ids),
                           top_share=round(float(cnts.max() / cnts.sum()), 4),
                           distinct=int(len(vals)),
                           traj=[round(float(x), 3) for x in traj],
                           peak=round(float(max(traj)), 3),
                           rose=bool(max(traj) > traj[0] + 0.5),
                           secs=round(time.time() - t0, 1))
                res["cells"][k] = row
                print(f"  {k:20s} top={row['top_share']:.3f} ign={row['ignition']:.2f} "
                      f"peak={row['peak']:.1f} rose={row['rose']} ({row['secs']:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    del rule
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    cells = list(res["cells"].values())
    grid = {}
    for c in cells:
        grid.setdefault((c["alphabet"], c["T"]), []).append(c)
    print(f"\n  {'alphabet':<9} {'T':>5} {'top share':>10} {'ignition':>9} {'peak dmg':>9} {'rose?':>6} live")
    live = []
    for (a, T), cs in sorted(grid.items()):
        top = float(np.mean([c["top_share"] for c in cs]))
        ign = float(np.mean([c["ignition"] for c in cs]))
        pk = float(np.mean([c["peak"] for c in cs]))
        rose = any(c["rose"] for c in cs)
        is_live = bool(ign >= IGN_MIN and top < FROZEN_MAX)
        if is_live:
            live.append((a, T, top, ign))
        print(f"  {a:<9} {T:>5.1f} {top:>10.3f} {ign:>9.2f} {pk:>9.1f} {str(rose):>6} "
              f"{'LIVE' if is_live else ''}")
    parts = []
    ctl = grid.get(("binary", 0.7))
    ctl_ok = bool(ctl and np.mean([c["top_share"] for c in ctl]) >= FROZEN_MAX
                  and np.mean([c["ignition"] for c in ctl]) < IGN_MIN)
    parts.append(
        "CONTROL: T=0.7 on the binary alphabet reproduces the frozen, non-igniting behaviour "
        "already measured." if ctl_ok else
        "CONTROL FAILS: T=0.7 does not reproduce the frozen non-igniting behaviour, so this sweep "
        "is not measuring the same construction and nothing below is read.")
    if ctl_ok and live:
        best = max(live, key=lambda x: x[3])
        parts.append(
            f"PRIMARY: a LIVE REGIME EXISTS. {len(live)} of {len(grid)} cells clear both thresholds "
            f"(ignition >= {IGN_MIN}, top-share < {FROZEN_MAX}); the strongest is {best[0]} at "
            f"T={best[1]} (top-share {best[2]:.3f}, ignition {best[3]:.2f}). #105 becomes askable "
            f"there, and #106's coupling rung has a temperature to run at. Every one of those "
            f"issues must be re-run AT that T rather than at the 0.7 inherited from the "
            f"full-vocabulary work.")
    elif ctl_ok:
        parts.append(
            f"KILL: NO live regime anywhere on the grid -- {len(grid)} cells across "
            f"{len(ALPHABETS)} alphabets and {len(TEMPS)} temperatures, none clearing both "
            f"ignition >= {IGN_MIN} and top-share < {FROZEN_MAX}. Raising temperature melts the "
            f"frozen ring but does not buy a spreading regime, so the failure is not the choice of "
            f"alphabet and not the temperature: it is the RESTRICTION ITSELF. Renormalising a "
            f"language model's conditional onto a small support removes whatever makes damage "
            f"spread on the full vocabulary. #105, #106 and #107 close as designed.")
    parts.append(
        "BOUNDARY: one model, one checkpoint, one radius, and the trajectory is reported per cell "
        "so 'never rose' is distinguished from 'rose and died' by inspection. A negative bounds "
        "THIS construction, not token-lattice CAs in general.")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(
        grid={f"{a}|T{T}": dict(top_share=round(float(np.mean([c['top_share'] for c in cs])), 4),
                                ignition=round(float(np.mean([c['ignition'] for c in cs])), 3),
                                peak=round(float(np.mean([c['peak'] for c in cs])), 3),
                                rose=bool(any(c["rose"] for c in cs)),
                                live=bool(np.mean([c['ignition'] for c in cs]) >= IGN_MIN
                                          and np.mean([c['top_share'] for c in cs]) < FROZEN_MAX))
              for (a, T), cs in sorted(grid.items())},
        live_cells=[[a, T] for a, T, _, _ in live], control_ok=ctl_ok,
        ign_min=IGN_MIN, frozen_max=FROZEN_MAX)
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Screens for a temperature where the sub-alphabet lattice is neither frozen nor "
                    "damage-dead, before #105/#106/#107 are re-run. At T=0.7 all three returned "
                    "nothing usable because the ring freezes (top-share 0.978) and injected damage "
                    "decays monotonically to zero. Trajectories are stored per cell.")


if __name__ == "__main__":
    main()
