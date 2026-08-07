"""#106: |V|=2 puts a real LM inside DK's exact coupling regime.

THE POINT. F41 established this project's CRN is the MONOTONE coupling, not the maximal one, and
that THEY COINCIDE AT |V|=2 -- which is exactly why the Domany-Kinzel rung stays bit-exact while
the LM numbers "sit inside the coupling family rather than at its damage-minimising edge". A
two-token sub-alphabet therefore puts a REAL LANGUAGE MODEL into the regime where monotone =
maximal, so F41's caveat LAPSES rather than being disclosed, and W2's standing coupling-mismatch
objection stops applying to this construction.

Nothing is re-derived: the coincidence is F41's result. What is new is running the LM lattice there.

PRE-REGISTERED:
  CONTROL    the exact-zero CRN null must hold: twins with no flip stay identical forever, at every
             checkpoint. This is asserted, not assumed, and a failure stops the script.
  PRIMARY    does the binary lattice show a developmental profile at all -- does lambda_ca move
             across checkpoints as it does on the full vocabulary?
  KILL, LIVE the binary lattice may simply manufacture a NEW fixed point, reproducing F62/F66 one
             level down with a different dominant token. F66's diagnosis predicts the degeneracy
             WEAKENS when short context is in-distribution; this measures whether it does. Either
             answer is worth having, and a fixed point here would be a clean negative.
  BOUNDARY   this is a RUNG, not a result about language models. It says the coupling caveat can be
             removed by construction; it says nothing about the full-vocabulary numbers.

Writes results/binary_dk_regime.json.  Resumable per (checkpoint, seed).
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from subalphabet import pick_tokens, damage_on_sub, lambda_of, sub_init, make_sampler, BINARY
from gatecheck import dynamic_range, carries_verdict
from gatecheck.cohort import cohort_complete

OUT = str(_ROOT / "results" / "binary_dk_regime.json")
MODEL = "EleutherAI/pythia-410m"
STEPS = ["step128", "step256", "step512", "step1000", "step2000", "step4000"]
T, R, B, N, SETTLE, SWEEPS = 0.7, 2, 16, 48, 12, 22
SEEDS = [21, 22, 23, 24]


def null_arm(rule, ids, seed):
    """The exact-zero CRN null: identical twins under a shared uniform must never diverge."""
    from ar_ca import run
    smp = make_sampler(ids)
    rng = np.random.default_rng(seed)
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
               init_state=sub_init(ids, B, N, rng), seed=seed, sampler=smp)["final"]
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
    c2 = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
             init_state=np.concatenate([base, base], axis=0), seed=seed + 2,
             u_stream=u2, sampler=smp)
    s = c2["snaps"]
    return int((s[:, :B] != s[:, B:]).sum())


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=STEPS, seeds=SEEDS, T=T, r=R, B=B, N=N, alphabet=BINARY,
        basis="F41: monotone and maximal coupling COINCIDE at |V|=2, so this construction removes "
              "the coupling caveat rather than disclosing it",
        control="exact-zero CRN null must hold at every checkpoint; a failure stops the script",
        primary="does lambda_ca move across checkpoints on the binary lattice?",
        kill="a new fixed point here reproduces F62/F66 one level down -- a clean negative",
        boundary="a RUNG, not a result about language models")
    from ar_ca import ARRule
    for rev in STEPS:
        if all(f"{rev}|s{sd}" in res["cells"] for sd in SEEDS):
            continue
        rule = ARRule(MODEL, revision=rev)
        ids, kept, dropped = pick_tokens(rule.tok, BINARY)
        if len(ids) != 2:
            print(f"  {rev}: alphabet is not size 2 ({kept} kept, {dropped} dropped) -- STOP")
            return
        nz = null_arm(rule, ids, SEEDS[0])
        res.setdefault("null", {})[rev] = nz
        if nz != 0:
            print(f"  {rev}: CRN NULL IS NOT ZERO ({nz} differing cells) -- STOP", flush=True)
            res["verdict"] = f"CRN null failed at {rev} ({nz} cells). Nothing measured."
            json.dump(res, open(OUT, "w"), indent=1)
            return
        for sd in SEEDS:
            k = f"{rev}|s{sd}"
            if k in res["cells"]:
                continue
            t0 = time.time()
            base, rolled = damage_on_sub(rule, ids, None, T=T, r=R, B=B, N=N,
                                         settle=SETTLE, sweeps=SWEEPS, seed=sd)
            row = lambda_of(rolled, N)
            vals, cnts = np.unique(base, return_counts=True)
            row.update(revision=rev, seed=sd, secs=round(time.time() - t0, 1),
                       top_share=round(float(cnts.max() / cnts.sum()), 4),
                       dominant=int(vals[int(np.argmax(cnts))]))
            res["cells"][k] = row
            print(f"  {k:20s} lam={row['lambda_ca']:+.4f} ign={row['ignition']:.2f} "
                  f"top={row['top_share']:.2f} ({row['secs']:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    parts = []
    nulls = res.get("null", {})
    ok = bool(nulls) and all(v == 0 for v in nulls.values())
    parts.append(
        f"CONTROL: the exact-zero CRN null holds at {sum(1 for v in nulls.values() if v == 0)}/"
        f"{len(nulls)} checkpoints. "
        + ("Identical twins never diverge on the binary lattice, so the coupling certification "
           "carries over to this construction." if ok else "IT FAILS -- nothing below is read."))
    rows = {}
    for rev in STEPS:
        cs = [c for c in res["cells"].values() if c.get("revision") == rev and c.get("ignited")]
        if cs:
            rows[rev] = dict(n=len(cs),
                             lam=round(float(np.mean([c["lambda_ca"] for c in cs])), 5),
                             sd=round(float(np.std([c["lambda_ca"] for c in cs])), 5),
                             ign=round(float(np.mean([c["ignition"] for c in cs])), 3),
                             top=round(float(np.mean([c["top_share"] for c in cs])), 3))
    print(f"\n  {'checkpoint':<12} {'n':>3} {'lambda':>9} {'sd':>8} {'ign':>6} {'top share':>10}")
    for r, v in rows.items():
        print(f"  {r:<12} {v['n']:>3} {v['lam']:>+9.4f} {v['sd']:>8.4f} {v['ign']:>6.2f} "
              f"{v['top']:>10.3f}")
    coh = cohort_complete(STEPS, list(rows), unit="checkpoint")
    parts.append(f"COHORT: {coh.reason}")
    if ok and len(rows) >= 3:
        lam = [v["lam"] for v in rows.values()]
        floor = float(np.mean([v["sd"] for v in rows.values()])) / np.sqrt(len(SEEDS))
        lev = dynamic_range(lam, floor=floor, name="lambda_ca across checkpoints (binary lattice)")
        v = carries_verdict([lev, coh], value=None) if coh.complete else None
        frozen = float(np.mean([r["top"] for r in rows.values()]))
        parts.append(
            f"PRIMARY: lambda_ca spans {max(lam)-min(lam):.4f} across {len(rows)} checkpoints "
            f"against a seed floor of {floor:.4f}. {lev.reason}")
        parts.append(
            f"FIXED-POINT CHECK (the registered live kill): mean dominant-token share on the "
            f"settled binary ring is {frozen:.3f}. "
            + (f"The lattice is FROZEN, so this construction reproduces F62/F66 one level down -- "
               f"a two-token alphabet manufactures its own degeneracy and the rung does not deliver "
               f"a usable regime." if frozen >= 0.9 else
               f"The lattice is NOT frozen, so restricting to an in-distribution two-token support "
               f"does not manufacture a fixed point -- which is what F66's diagnosis predicts, and "
               f"it makes |V|=2 a usable regime where F41's coupling caveat does not apply."))
    parts.append(
        "BOUNDARY: this is a RUNG. It shows the coupling caveat can be removed by construction; it "
        "says nothing about the full-vocabulary numbers, and must not be quoted as one.")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows=rows, null=nulls, null_ok=ok, cohort=coh.block())
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("#106. F41 proved monotone = maximal coupling at |V|=2, which is why the DK "
                    "rung is bit-exact. This runs a real LM lattice in that regime, so the "
                    "coupling caveat lapses by construction rather than being disclosed.")


if __name__ == "__main__":
    main()
