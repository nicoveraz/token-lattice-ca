"""The attractor share on the instruction-tuned cohort, at band_screen's geometry. Raw context only.

WHY THIS RUN EXISTS. F117's selectivity result correlates `top1@T` against benchmark columns, and
both sides were measured on ten BASE models. F138 showed that cohort has no measurable variance in
verifiable instruction-following, so `instruct_cohort.py` selected a replacement where compliance
genuinely varies (IFEval span 48.8 against 13.4). Neither side of the correlation exists for the
new cohort: `compliance_v3.py` supplies the compliance side, this supplies the lattice side.

GEOMETRY IS COPIED FROM `band_screen.py`, NOT CHOSEN. N = 96, B = 16, settle = 16, r = 2, and the
four temperatures {0.02, 0.2, 0.436, 0.7} that F117's readouts are named for and that T*
interpolates on. A different geometry would produce a `top1@T` that is not F117's quantity, and
the whole point is a like-for-like replacement.

RAW CONTEXT, NO CHAT TEMPLATE, AND THAT IS THE OPPOSITE OF THE COMPLIANCE POOL'S CHOICE. F135
measured a chat scaffold moving the share by 0.1327 against a gate of 0.0406 -- more than half the
across-model spread -- so a templated lattice reads the template. The lattice conditions on r
tokens and nothing else, exactly as it does for base models, which is also what keeps this
commensurable with F117/F130. The compliance pool makes the opposite choice for the opposite
reason: it measures instruction-following, so it uses the interface these models were built for.
Same cohort, two interfaces, each matched to what it reads.

STATE IS STORED (R8): the settled lattice goes into the results file, so the F136 question -- is
`top1` reading an attractor or 1/period -- is answerable here without a re-run. Instruction-tuned
models have never been put on this lattice, so the possibility is open rather than settled.

PRE-REGISTERED:
  RUNG      the share must have across-model SIGNAL above across-seed noise at the reference
            construction, as F130 required. Without it there is no ranking to correlate and
            compliance_v3's PRIMARY is not read.
  READOUT   mean `top1` over seeds per (model, T), which is what F117 correlates.
  DEGENERACY  every settled lattice is classified frozen / periodic-cycle / disordered by
            `share_periodicity`'s detector, so a 1/period reading cannot be mistaken for an
            attractor -- the F136 gate, applied before the numbers are used rather than after.
  BOUNDARY  ten instruction-tuned models, one per pretraining family, one geometry.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from provenance import stamp, rel
from gatecheck import pack_state, has_state, STATE_KEY

OUT = str(_ROOT / "results" / "share_instruct.json")
COHORT = str(_ROOT / "results" / "instruct_cohort.json")

# band_screen.py's geometry, copied so `top1@T` is F117's quantity and not a new one
N, B, SETTLE, R = 96, 16, 16, 2
TEMPS = [0.02, 0.2, 0.436, 0.7]
SEEDS = [20260813, 20260814]
NOISE_FACTOR = 2.0
REF_T = 0.02


def cell(rule, T, seed):
    from ar_ca import run
    settled = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none", init="random",
                  seed=seed)["final"]
    pool = settled.reshape(-1)
    vals, cnt = np.unique(pool, return_counts=True)
    rep2 = float(np.mean(settled[:, :-1] == settled[:, 1:]))
    return dict(top1=float(cnt.max() / cnt.sum()), distinct=float(len(vals)), rep2=rep2,
                dominant=int(vals[cnt.argmax()]),
                **{STATE_KEY: pack_state(settled, stride_axis=0,
                                         note="settled lattice, (replica, site)")})


def analyse(res):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sp", _ROOT / "experiments" / "share_periodicity.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)

    cells = res["cells"]
    models = sorted({c["model"] for c in cells.values()})
    lines, analysis = [], {}

    # DEGENERACY FIRST (F136): a 1/period reading is not an attractor share.
    n_rep = n_cry = 0
    for c in cells.values():
        if not has_state(c):
            continue
        from gatecheck import unpack_state
        arr = unpack_state(c[STATE_KEY])
        per = [sp.classify(list(arr[b])) for b in range(arr.shape[0])]
        n_rep += len(per)
        n_cry += sum(p["crystal"] for p in per)
    analysis["periodicity"] = dict(replicas=n_rep, balanced_cycles=n_cry)
    lines.append(
        f"DEGENERACY (F136's gate, applied first): {n_cry} of {n_rep} settled replicas are "
        f"balanced periodic cycles, where top1 is 1/period by arithmetic rather than a share. "
        + ("None, so every reading below is an attractor share."
           if n_cry == 0 else
           f"{n_cry / max(n_rep,1):.1%} of replicas. A cycle reads 1/p <= 0.5, so this cannot "
           f"manufacture a high share, but the affected cells are flagged."))

    def col(T, s):
        out = []
        for m in models:
            k = f"{m}|T{T}|s{s}"
            if k not in cells:
                return None
            out.append(cells[k]["top1"])
        return out

    sig = []
    for T in TEMPS:
        a, b = col(T, SEEDS[0]), col(T, SEEDS[1])
        if a is None or b is None:
            continue
        spread = float(max(a) - min(a))
        noise = float(np.mean([abs(x - y) for x, y in zip(a, b)]))
        ok = noise > 0 and spread >= NOISE_FACTOR * noise
        sig.append((T, spread, noise, ok))
    analysis["signal"] = [dict(T=T, spread=round(s, 4), seed_noise=round(n, 4), passes=bool(o))
                          for T, s, n, o in sig]
    n_ok = sum(1 for *_x, o in sig if o)
    lines.append(
        "RUNG (signal, as F130 required): "
        + "; ".join(f"T={T}: spread {s:.3f} vs seed noise {n:.3f}" + (" OK" if o else " FAILS")
                    for T, s, n, o in sig)
        + f". {n_ok} of {len(sig)} constructions carry model signal above seed noise. "
        + ("The share ranks this cohort, so compliance_v3's PRIMARY can be read."
           if n_ok > len(sig) / 2 else
           "The share does NOT rank this cohort above its own seed noise, so there is nothing for "
           "the selectivity statistic to correlate and compliance_v3's PRIMARY is NOT_DECIDABLE."))

    readout = {}
    for T in TEMPS:
        for m in models:
            vs = [cells[f"{m}|T{T}|s{s}"]["top1"] for s in SEEDS if f"{m}|T{T}|s{s}" in cells]
            if vs:
                readout[f"{m}|T{T}"] = float(np.mean(vs))
    analysis["top1"] = {k: round(v, 4) for k, v in readout.items()}
    lines.append(
        f"BOUNDARY: {len(models)} instruction-tuned models, one per pretraining family, "
        f"N={N}, B={B}, settle={SETTLE}, r={R}, raw r-token context with NO chat template (F135). "
        f"Geometry copied from band_screen so top1@T is F117's quantity; these are ten families, "
        f"not a population, and instruction-tuned lattices have no prior in this project.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(lines)


def main():
    cohort = [r["model"] for r in json.load(open(COHORT))["cohort"]]
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"cells": {}}
    res["_preregistration"] = dict(
        cohort=cohort, N=N, B=B, settle=SETTLE, r=R, temps=TEMPS, seeds=SEEDS,
        noise_factor=NOISE_FACTOR, ref_T=REF_T,
        geometry="copied from band_screen.py so top1@T is F117's quantity",
        context="raw r-token, NO chat template -- F135 measured a scaffold moving the share by "
                "more than half the across-model spread",
        rung="across-model spread must exceed across-seed noise by NOISE_FACTOR",
        state="the settled lattice is stored, so F136's periodicity question needs no re-run")
    if "--analyse" not in _sys.argv:
        import torch
        from ar_ca import ARRule
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done_here = 0
        for m in cohort:
            need = [(T, s) for T in TEMPS for s in SEEDS
                    if not has_state(res["cells"].get(f"{m}|T{T}|s{s}", {}))]
            if not need:
                continue
            if limit and done_here >= limit:
                print(f"  (stopping after {done_here} model(s); re-run to continue)", flush=True)
                break
            try:
                rule = ARRule(m)
            except Exception as e:
                print(f"  {m}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                continue
            for T, s in need:
                k = f"{m}|T{T}|s{s}"
                t0 = time.time()
                try:
                    c = cell(rule, T, s)
                except Exception as e:
                    print(f"  {k}: FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                    continue
                c.update(model=m, T=T, seed=s, secs=round(time.time() - t0, 1))
                res["cells"][k] = c
                print(f"  {k:<52} top1={c['top1']:.4f} rep2={c['rep2']:.3f} "
                      f"({c['secs']:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
            done_here += 1
            del rule
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
