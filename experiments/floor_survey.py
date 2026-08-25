"""Does a usable precision floor exist at tau=0.5 for ANY model? Decides F185's tau=0.5 rung.

Registered in experiments/prereg_floor_survey.json (frozen `05bc8157...` before any new twin).

WHY. F185's ladder peaked at tau=0.5 and that rung was never promoted. Its registered replication
returned NOT DECIDABLE under KR3: the held-out floor reached agreement 1.0 by tau=0.5, leaving
nothing to resolve against. pythia-410m's reaches 0.995 at the same rung. Two models, both
degenerate. This asks the prior question -- is that a property of those two models, or of the
estimand? -- by measuring six more floors.

KF1 is registered as the EXPECTED outcome given two of two, and written down before the other six:
if every floor is degenerate at tau=0.5, the rung is RETIRED rather than merely unreplicated, because
the ratio has no stable denominator anywhere in the cohort.

AND IF A USABLE FLOOR EXISTS, THIS PREREG STILL DOES NOT RUN THE REPLICATION. A model chosen because
its floor turned out usable is a model chosen on the data. It names the model and stops.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from selfcont_set import DTYPE, BATCH, BATCH_BIG, BIG, out_path
from escape_confidence import top2

PREREG = "experiments/prereg_floor_survey.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "floor_survey.json"
CACHE = _ROOT / "results" / "floor_survey_cells.json"
CELLS = PR["cells"]
TAUS = PR["thresholds"]["tau_ladder"]
DEGEN = PR["thresholds"]["degenerate_floor"]
MIN_SRC = PR["thresholds"]["min_shared_sources"]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    failed = []
    for m in CELLS:
        for dt in ("fp32", "bf16"):
            key = f"{m}@{dt}"
            if key in cache:
                print(f"  {key:<40} cached", flush=True); continue
            src = out_path(m, "fp32")
            if not src.exists():
                failed.append(dict(model=m, error="no full-vocabulary cell")); break
            pid = np.array(json.load(open(src))["probe_token_ids"])
            idx = np.flatnonzero(pid >= 0)
            ids = [int(i) for i in pid[idx]]
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
                model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", DTYPE[dt])
            except Exception as e:
                failed.append(dict(model=m, dtype=dt, error=type(e).__name__))
                print(f"  {key:<40} LOAD FAILED {type(e).__name__}", flush=True); continue
            tid, tlg = top2(model, ids, "cpu", BATCH_BIG if m in BIG else BATCH)
            dec = [tok.decode([int(i)]) for i in tid[:, 0]]
            del model; gc.collect()
            cache[key] = dict(cell=key, model=m, dtype=dt, probe_positions=[int(i) for i in idx],
                              source_ids=ids, top1_str=dec,
                              conf_e4=[int(round(x)) for x in (tlg[:, 0] - tlg[:, 1]) * 1e4],
                              escapes=[bool(int(a) != int(b)) for a, b in zip(tid[:, 0], ids)],
                              secs=round(time.time() - t0, 1))
            json.dump(cache, open(CACHE, "w"))
            print(f"  {key:<40} n={len(ids):<6} ({cache[key]['secs']:.0f}s)", flush=True)
    _verdict(cache, failed)


def _verdict(cache, failed):
    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_floor_survey.sha256").read().split()[0],
               thresholds=PR["thresholds"], failed=failed, floors={})
    for m in CELLS:
        a, b = f"{m}@fp32", f"{m}@bf16"
        if a not in cache or b not in cache:
            continue
        ca = np.array(cache[a]["conf_e4"], np.float64) / 1e4
        cb = np.array(cache[b]["conf_e4"], np.float64) / 1e4
        sa = np.array(cache[a]["top1_str"], dtype=object)
        sb = np.array(cache[b]["top1_str"], dtype=object)
        ea = np.array(cache[a]["escapes"], bool); eb = np.array(cache[b]["escapes"], bool)
        rungs = []
        for tau in TAUS:
            keep = ea & eb & (ca >= tau) & (cb >= tau)
            n = int(keep.sum())
            rungs.append(dict(tau=tau, n=n,
                              agreement=None if n == 0 else round(float(np.mean(sa[keep] == sb[keep])), 4),
                              not_decidable=bool(n < MIN_SRC)))
        res["floors"][m] = rungs
    # the two already measured, reused rather than re-run
    res["reused_floors"] = PR["reused"]
    prior = {"EleutherAI/pythia-410m": 0.995, "EleutherAI/pythia-160m": 1.0}

    at_half = {}
    for m, rungs in res["floors"].items():
        r = next(x for x in rungs if x["tau"] == 0.5)
        at_half[m] = r
    for m, v in prior.items():
        at_half.setdefault(m, dict(tau=0.5, agreement=v, n=None, not_decidable=False,
                                   _source="reused"))
    usable = {m: r for m, r in at_half.items()
              if r["agreement"] is not None and r["agreement"] < DEGEN and not r["not_decidable"]}
    res["floor_at_tau_half"] = at_half
    res["usable_floors_at_tau_half"] = sorted(usable)
    res["KF1_rung_retired"] = bool(not usable)
    res["KF2_rung_is_testable"] = bool(usable)
    res["floor_spread_at_tau_zero"] = {
        m: next(x for x in rungs if x["tau"] == 0.0)["agreement"] for m, rungs in res["floors"].items()}

    p = [f"PRECISION-FLOOR SURVEY, registered in {PREREG} (sha256 {res['_prereg_sha256'][:12]}..., "
         f"frozen before any new twin). Each model measured at float32 and bfloat16 on the frozen "
         f"probe set, same estimator and same tau ladder as F185. "]
    if failed:
        p.append(f"KF4: {len(failed)} cell(s) NAMED: {[f['model'] for f in failed]}. ")
    p.append("FLOOR AT THE CONTESTED RUNG (tau=0.5): " + "; ".join(
        f"{m.split('/')[-1]} {r['agreement']}"
        + (" [reused]" if r.get("_source") else f" (n={r['n']})")
        + (", NOT DECIDABLE" if r.get("not_decidable") else "")
        for m, r in sorted(at_half.items())) + ". ")
    if res["KF1_rung_retired"]:
        p.append(f"KF1 FIRES -- THE TAU=0.5 RUNG IS RETIRED. Every floor in the cohort is at or above "
                 f"{DEGEN} at that rung, so the resolution ratio has no stable denominator anywhere. "
                 f"F185's 11.76x may never be quoted, and this is a stronger retirement than the "
                 f"replication's NOT DECIDABLE: the rung is not merely unreplicated on one pair, it "
                 f"is unresolvable on this estimand. Registered as the expected outcome before the "
                 f"six new floors were measured. ")
    else:
        p.append(f"KF2 FIRES: a usable floor exists at tau=0.5 on {res['usable_floors_at_tau_half']}. "
                 f"prereg_tau_replication therefore picked a degenerate control rather than the rung "
                 f"failing. THIS PREREG DOES NOT RUN THE REPLICATION -- a model chosen because its "
                 f"floor turned out usable is chosen on the data. It is named and the run stops. ")
    sp = res["floor_spread_at_tau_zero"]
    vals = [v for v in sp.values() if v is not None]
    p.append(f"AND THE SPREAD F187 FOUND, now measured across the cohort: at tau=0 the floors run "
             f"{min(vals)} to {max(vals)} across {len(vals)} models. Both earlier preregs treated one "
             f"model's floor as THE floor; it is not, and every per-model robustness statement in "
             f"the paper rests on this spread. ")
    p.append("REFUSALS: no p-value; no adjustment of the 0.99 bound, the ladder, or the source "
             "floor; no replication run here even under KF2; no claim that a bfloat16 floor bounds a "
             "quantized one -- 4-bit remains fatal per F187.")
    res["verdict"] = " ".join(p)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"]); print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
