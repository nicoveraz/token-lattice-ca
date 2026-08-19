"""Does a model's COPY STRENGTH predict which way its phi moves? The frozen prereg, run.

WHAT THIS TESTS, AND WHY IT IS STILL LIVE AFTER F166. F165 proposed two vectors; F166 killed half of
them -- the endpoint token is NOT a prefix property (best cross-model agreement on an arm's modal
token is 4 of 9). What SURVIVED, because it was verified directly rather than inferred, is that GIVEN
a shared endpoint token, whether it self-continues is model-specific: `SmolLM` sends 84 of 96
trajectories to '0' under p2 and phi stays 0.000, beside two models that reach '0' and raise to ~1.0.

H1 tests exactly that surviving half, with a quantity measured independently of any census: a model's
propensity to continue a pattern it has already seen. If a model copies strongly, a structured prefix
gives it something to latch onto and the self-loop forms or breaks accordingly.

THE ESTIMATOR IS BEHAVIOURAL AND DELIBERATELY NOT A CIRCUIT CLAIM. For K random token pairs (a, b),
build [a, b, <F filler tokens>, a] and ask whether argmax p(. | context) == b. That is the behaviour
the mechanism needs, at one forward pass per probe, with no attention inspected and no reference to
phi. It is named "induction-style" rather than "induction head" because the circuit claim belongs to
a literature this probe does not test -- and whose prior-art gate is OWED before any write-up.

FROZEN BEFORE THE RUN: experiments/prereg_copy_vs_repeat.json, with H2 marked BLOCKED ON F164 (the
bilinear loading v is not a licensed quantity at 0.790) and an amendment log dated before any copy
score existed. K4 gates the PREDICTOR for variance, which is F163's lesson pre-installed.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from provenance import stamp, rel
from gatecheck import balance_report                      # F163's gate, as code

OUT = str(_ROOT / "results" / "copy_vs_repeat.json")
PREREG = "experiments/prereg_copy_vs_repeat.json"

K, FILLER = 64, 8
SEEDS = [20260803, 990017]
CENSUS_SEEDS = [20260803, 990017]
MIN_SHIFT = 4.0 / 96
NOISE_FACTOR = 2.0
CEIL_HI, CEIL_LO = 0.95, 0.05


def copy_score(model, dev, pool, rng):
    """Fraction of K probes where argmax p(. | [a, b, filler..., a]) == b."""
    hits = 0
    for _ in range(K):
        a, b = (int(x) for x in rng.choice(pool, size=2, replace=False))
        filler = [int(x) for x in rng.choice(pool, size=FILLER, replace=False)]
        ids = torch.tensor([[a, b] + filler + [a]], device=dev)
        with torch.no_grad():
            lg = model(input_ids=ids).logits[0, -1]
        hits += int(int(torch.argmax(lg).item()) == b)
    return hits / K


def dphi_on(arm_runs, base_runs, m, arm):
    ka = [f"{m}|s{cs}|{arm}" for cs in CENSUS_SEEDS]
    kr = [f"{m}|s{cs}|raw" for cs in CENSUS_SEEDS]
    if not all(k in arm_runs for k in ka) or not all(k in base_runs for k in kr):
        return None
    va = [arm_runs[k]["fixed_point_fraction"] for k in ka]
    vr = [base_runs[k]["fixed_point_fraction"] for k in kr]
    mu, nu = float(np.mean(va)), float(abs(va[0] - va[1]))
    rw, nr = float(np.mean(vr)), float(abs(vr[0] - vr[1]))
    tol = max(MIN_SHIFT, NOISE_FACTOR * max(nu, nr))
    d = mu - rw
    return dict(phi_raw=round(rw, 4), phi_arm=round(mu, 4), dphi=round(d, 4),
                tol=round(tol, 4),
                direction="up" if d > tol else ("down" if d < -tol else "flat"),
                headroom_ok=bool((1 - rw if d > 0 else rw) > tol))


def main():
    res = {"_preregistration_file": PREREG,
           "_h2_status": "BLOCKED ON F164 -- not evaluated, no correlation against the bilinear "
                         "loading v is computed"}
    base = json.load(open(_ROOT / "results" / "domain_base.json"))["runs"]
    ti = json.load(open(_ROOT / "results" / "text_interaction.json"))["runs"]
    fill_p = _ROOT / "results" / "text_interaction_fill.json"
    if fill_p.exists():
        ti = dict(ti); ti.update(json.load(open(fill_p))["runs"])
    models = sorted({k.split("|")[0] for k in base if len(k.split("|")) == 3})

    # --reuse: the copy scores are the expensive half (CPU float32, K*2 passes per model). This
    # path recomputes ONLY the join and the verdict, so a gate added mid-run can be applied without
    # re-measuring the predictor. It refuses to invent scores: anything missing stays missing.
    if "--reuse" in _sys.argv and os.path.exists(OUT):
        prev = json.load(open(OUT))
        scores = prev.get("copy_scores", {})
        print(f"  reusing {sum(1 for v in scores.values() if 'copy_score' in v)} stored copy scores",
              flush=True)
        models = [m for m in models if m in scores]
        return _finish(res, scores, models, base, ti)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cpu"
    scores = {}
    for m in models:
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m).eval().to(dev, torch.float32)
        except Exception as e:
            scores[m] = dict(error=type(e).__name__)
            print(f"  {m}: LOAD FAILED {type(e).__name__}", flush=True)
            continue
        V = int(getattr(model.config, "vocab_size", len(tok)))
        sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                          tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
        per = [copy_score(model, dev, pool, np.random.default_rng(s)) for s in SEEDS]
        scores[m] = dict(copy_score=round(float(np.mean(per)), 4),
                         per_seed=[round(x, 4) for x in per],
                         seed_range=round(float(abs(per[0] - per[1])), 4),
                         secs=round(time.time() - t0, 1))
        print(f"  {m:<30} copy_score {np.mean(per):.3f}  (seeds {per})", flush=True)
        del model
        gc.collect()
    res["copy_scores"] = scores
    return _finish(res, scores, models, base, ti)


def _finish(res, scores, models, base, ti):
    res["copy_scores"] = scores

    rows = []
    for m in models:
        s = scores.get(m, {})
        if "copy_score" not in s:
            continue
        d = dphi_on(ti, base, m, "p1")
        if d is None:
            continue
        rows.append(dict(model=m, copy_score=s["copy_score"], **d))
    res["p1_join"] = rows

    parts = [f"COPY SCORE, K={K} probes x {len(SEEDS)} seeds, CPU, no attention inspected: "
             + "; ".join(f"{r['model'].split('/')[-1]} {r['copy_score']:.3f}" for r in rows) + ". "]

    # K4 first: does the PREDICTOR have room to vary? (F163's gate, as code)
    cs = [r["copy_score"] for r in rows]
    if cs and (all(c >= CEIL_HI for c in cs) or all(c <= CEIL_LO for c in cs)):
        parts.append(
            f"K4 FIRED: copy_score is saturated across every model (range "
            f"[{min(cs):.3f}, {max(cs):.3f}]). The predictor has no variance and cannot "
            f"discriminate: NOT DECIDABLE for predictor imbalance.")
        res["verdict"] = " ".join(parts)
    else:
        readable = [r for r in rows if r["direction"] != "flat" and r["headroom_ok"]]
        excluded = [r for r in rows if r not in readable]
        parts.append(
            f"ANTI-VACUITY: {len(readable)} of {len(rows)} models are readable on the p1 arm "
            f"(|dphi| beyond tolerance AND headroom on the side it moved)"
            + (f"; excluded: {[(r['model'].split('/')[-1], r['direction']) for r in excluded]}."
               if excluded else "."))
        ups = [r for r in readable if r["direction"] == "up"]
        downs = [r for r in readable if r["direction"] == "down"]
        rep = balance_report([r["direction"] for r in readable], name="p1 direction")
        res["balance"] = dict(readable=len(readable), n_up=len(ups), n_down=len(downs),
                              gate_readable=rep.readable, gate_reason=rep.reason)
        if not rep.readable or len(ups) < 2 or len(downs) < 2:
            parts.append(
                f"K3 / balance gate: {len(ups)} up and {len(downs)} down among {len(readable)} "
                f"readable models. {rep.reason} NOT DECIDABLE -- there is no contrast to separate, "
                f"and a copy_score split scored against this outcome would inherit its base rate.")
        else:
            lo_up, hi_up = min(r["copy_score"] for r in ups), max(r["copy_score"] for r in ups)
            lo_dn, hi_dn = min(r["copy_score"] for r in downs), max(r["copy_score"] for r in downs)
            disjoint = (lo_up > hi_dn) or (lo_dn > hi_up)
            # K5: disjointness must beat the PREDICTOR's own noise. At K=64 the binomial SE is
            # ~0.06, so raw min/max separation can be luck. Gap is judged against 2 * pooled SE.
            se = float(np.mean([abs(scores[r["model"]]["per_seed"][0]
                                    - scores[r["model"]]["per_seed"][1]) / 2
                                for r in readable]))
            gap = (lo_up - hi_dn) if lo_up > hi_dn else ((lo_dn - hi_up) if lo_dn > hi_up else 0.0)
            beats_noise = bool(disjoint and gap > 2 * se)
            res["separation"] = dict(up_range=[lo_up, hi_up], down_range=[lo_dn, hi_dn],
                                     disjoint=bool(disjoint), gap=round(gap, 4),
                                     pooled_half_seed_range=round(se, 4),
                                     beats_noise=beats_noise)
            parts.append(
                f"PRIMARY (K1/K2): copy_score on the UP models spans [{lo_up:.3f}, {hi_up:.3f}], on "
                f"the DOWN models [{lo_dn:.3f}, {hi_dn:.3f}]. "
                + (f"Ranges are DISJOINT by {gap:.3f} against a pooled noise scale of "
                   f"{2*se:.3f}: copy strength separates the direction of the p1 effect, and this "
                   f"is the first model-side quantity in the programme to predict a sign."
                   if beats_noise else
                   f"Ranges are disjoint by {gap:.3f} but that does NOT beat the predictor's own "
                   f"noise ({2*se:.3f} at K={K}): NOT DECIDABLE FOR PRECISION (K5). The fix is more "
                   f"probes, which is cheap, not a relaxed criterion."
                   if disjoint else
                   "Ranges OVERLAP: copy strength does not separate the direction of the p1 effect. "
                   "K1 fires and the copy account is dropped -- on a behavioural measurement rather "
                   "than on an argument."))
    parts.append(
        f"H2 is BLOCKED ON F164 and NOT evaluated: the bilinear loading v comes from a fit that "
        f"stands at 0.790 against a pre-registered 0.80 and was never licensed. BOUNDARY: one arm "
        f"(p1), {len(models)} models, K={K}, CPU float32. No p-value and no rank correlation are "
        f"computed -- both refusals were recorded before the numbers. The prior-art gate for "
        f"induction heads and repetition self-reinforcement remains OWED and gates any write-up.")
    res["verdict"] = res.get("verdict") or " ".join(parts)
    if "K4 FIRED" not in res["verdict"]:
        res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
