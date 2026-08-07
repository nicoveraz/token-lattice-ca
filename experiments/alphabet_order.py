"""#105: does lambda_ca depend on the VOCABULARY ORDER? The permutation test.

THE PREMISE. `ar_ca.sample_device` is `(cdf < u).sum()` -- inverse-CDF against a shared uniform, so
it is a functional of the ORDER the alphabet is laid out in. For BPE tokens that order is an
arbitrary artifact of merge frequency; nothing about the model or the lattice depends on it. F41
established this is the MONOTONE coupling and that maximal coupling is order-INVARIANT while
monotone is not. So an arbitrary implementation choice may carry variance in every lambda_ca here.

F101 decomposed lambda's seed spread into basin vs noise and licensed the floor as genuine noise.
ORDERING variance is a third component, orthogonal to both, and unmeasured.

PRE-REGISTERED:
  PRIMARY   spread of lambda_ca across index permutations of an UNORDERED sub-alphabet (colours),
            against the seed floor measured in the same run. Exceeding it means every published
            lambda_ca carries unmeasured implementation-choice variance.
  CONTROL   at |V|=2 there is one distinct ordering up to symmetry (tests/test_subalphabet.py
            proves the sampler-level identity), so the ordering spread there must be ~0. A large
            binary spread means the harness is wrong, not the coupling.
  GATE      lambda must span more than its own seed floor across permutations before any spread is
            quoted -- the defect class caught six times in this project.
  KILL      spread below the floor -> the coupling's arbitrariness does not propagate, and every
            published number stands. A null that LICENSES, like F101.

Writes results/alphabet_order.json.  Resumable per (alphabet, perm, seed).
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from subalphabet import pick_tokens, damage_on_sub, lambda_of, COLOURS, BINARY
from gatecheck import dynamic_range, carries_verdict

OUT = str(_ROOT / "results" / "alphabet_order.json")
MODEL, REV = "EleutherAI/pythia-410m", "step4000"
T, R, B, N, SETTLE, SWEEPS = 0.7, 2, 16, 48, 12, 22
SEEDS = [21, 22, 23, 24]
N_PERM = 6
SEED = 20260807


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    from ar_ca import ARRule
    rule = ARRule(MODEL, revision=REV)
    alphabets = {}
    for name, words in (("colours", COLOURS), ("binary", BINARY)):
        ids, kept, dropped = pick_tokens(rule.tok, words)
        alphabets[name] = ids
        print(f"  {name}: {len(ids)} single tokens {kept}" + (f"  DROPPED {dropped}" if dropped else ""),
              flush=True)
    res["_preregistration"] = dict(
        model=MODEL, revision=REV, T=T, r=R, B=B, N=N, seeds=SEEDS, n_perm=N_PERM, seed=SEED,
        alphabets={k: [int(x) for x in v] for k, v in alphabets.items()},
        primary="spread of lambda_ca across index permutations of the UNORDERED colour alphabet, "
                "against the seed floor measured in the same run",
        control="|V|=2 ordering spread must be ~0 (one distinct ordering up to symmetry)",
        kill="spread below the floor -> the coupling's arbitrariness does not propagate and every "
             "published lambda_ca stands")
    rng = np.random.default_rng(SEED)
    for name, ids in alphabets.items():
        k = len(ids)
        perms = [list(range(k))] + [list(rng.permutation(k)) for _ in range(N_PERM - 1)] \
            if k > 2 else [[0, 1], [1, 0]]
        res.setdefault("perms", {})[name] = [list(map(int, p)) for p in perms]
        for pi, perm in enumerate(perms):
            for sd in SEEDS:
                key = f"{name}|p{pi}|s{sd}"
                if key in res["cells"]:
                    continue
                t0 = time.time()
                _, rolled = damage_on_sub(rule, ids, perm, T=T, r=R, B=B, N=N,
                                          settle=SETTLE, sweeps=SWEEPS, seed=sd)
                row = lambda_of(rolled, N)
                row.update(alphabet=name, perm=pi, order=list(map(int, perm)), seed=sd,
                           secs=round(time.time() - t0, 1))
                res["cells"][key] = row
                print(f"  {key:22s} lam={row['lambda_ca']:+.4f} ign={row['ignition']:.2f} "
                      f"({row['secs']:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    del rule
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    parts, per = [], {}
    for name in ("binary", "colours"):
        cells = [c for c in res["cells"].values()
                 if c.get("alphabet") == name and c.get("ignited")]
        if len(cells) < 4:
            continue
        byp = {}
        for c in cells:
            byp.setdefault(c["perm"], []).append(c["lambda_ca"])
        means = {p: float(np.mean(v)) for p, v in byp.items() if len(v) >= 2}
        if len(means) < 2:
            continue
        seed_floor = float(np.mean([np.std(v) for v in byp.values() if len(v) >= 2])) / np.sqrt(len(SEEDS))
        spread = float(np.std(list(means.values())))
        per[name] = dict(n_perms=len(means), n_cells=len(cells),
                         perm_means={str(p): round(v, 5) for p, v in sorted(means.items())},
                         ordering_spread=round(spread, 5), seed_floor=round(seed_floor, 5),
                         ratio=round(spread / max(seed_floor, 1e-9), 3))
    print(f"\n  {'alphabet':<10} {'perms':>6} {'ordering spread':>16} {'seed floor':>11} {'ratio':>7}")
    for n, v in per.items():
        print(f"  {n:<10} {v['n_perms']:>6} {v['ordering_spread']:>16.5f} "
              f"{v['seed_floor']:>11.5f} {v['ratio']:>7.2f}")
    ctl = per.get("binary")
    if ctl:
        ok = ctl["ratio"] <= 1.0
        parts.append(
            f"CONTROL (|V|=2, one distinct ordering up to symmetry): ordering spread "
            f"{ctl['ordering_spread']:.5f} against a seed floor of {ctl['seed_floor']:.5f}, ratio "
            f"{ctl['ratio']:.2f}. "
            + ("At or below the floor, as the sampler-level identity requires."
               if ok else "ABOVE the floor, which the |V|=2 symmetry forbids -- the harness is "
                          "wrong and nothing below is read."))
    else:
        ok = False
        parts.append("CONTROL INCOMPLETE -- nothing below is read.")
    col = per.get("colours")
    if ok and col:
        lev = dynamic_range(list(col["perm_means"].values()), floor=col["seed_floor"],
                            name="lambda_ca across orderings")
        v = carries_verdict([lev], value=col["ratio"])
        if v.status != "DECIDED":
            parts.append(f"PRIMARY NOT DECIDABLE: {v.reason}")
        elif col["ratio"] > 1.0:
            parts.append(
                f"PRIMARY: lambda_ca DEPENDS ON THE VOCABULARY ORDER. Across {col['n_perms']} index "
                f"permutations of an unordered alphabet the spread is {col['ordering_spread']:.5f} "
                f"against a seed floor of {col['seed_floor']:.5f} ({col['ratio']:.2f}x). The model, "
                f"the lattice and the uniform stream are identical across these runs -- only the "
                f"arbitrary layout differs. Every lambda_ca in this project therefore carries an "
                f"unmeasured implementation-choice variance, and that must be disclosed.")
        else:
            parts.append(
                f"KILL, and it LICENSES rather than merely failing to impeach: ordering spread "
                f"{col['ordering_spread']:.5f} sits at {col['ratio']:.2f}x the seed floor, so the "
                f"monotone coupling's arbitrary layout does not propagate to lambda_ca. Every "
                f"published number stands, and the coupling's order-dependence is confined to the "
                f"sampler rather than the measurement.")
    parts.append(
        "BOUNDARY: one checkpoint, one temperature, one radius, and restricting the support is a "
        "real intervention that reopens F35's (model, construction) boundary -- the same one F65 "
        "validated when banning newline moved the attractor 74% -> 15%.")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(per_alphabet=per, control_ok=bool(ok))
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("#105. The CRN coupling is inverse-CDF and therefore a functional of the "
                    "vocabulary ORDER, which is arbitrary for BPE tokens. This measures whether "
                    "that arbitrariness reaches lambda_ca, with |V|=2 as a known-answer control.")


if __name__ == "__main__":
    main()
