"""F167 at K=256: can the separation beat its own noise? And the direction, registered this time.

WHY A NEW FILE. `copy_vs_repeat.py` is unchanged: F167's stored result carries its sha256 in
`_analysis_provenance`, and editing it would invalidate a recorded finding. The estimator is imported,
not copied, so the measured quantity is provably the same one.

WHAT F167 LEFT. Ranges separated -- UP [0.125, 0.148], DOWN [0.203, 0.594] -- by 0.0547 against a
predictor noise scale of 0.069, so the verdict was NOT DECIDABLE FOR PRECISION (K5). Noise falls as
1/sqrt(K), and this quadruples the probes.

TWO THINGS ARE DONE DIFFERENTLY, BOTH BECAUSE F167 EXPOSED THEM.

1. THE DIRECTION IS REGISTERED. F167's kill conditions were direction-agnostic (registry R10): K2
   asked only for "disjoint ranges", so a separation in EITHER direction would have satisfied it while
   the hypothesis said "strong copiers raise". The data came back inverted. That inverted direction is
   now a hypothesis in its own right, stated with its sign BEFORE this run, and tested on FRESH probes
   (new K, new draws) rather than on the measurements that suggested it:

       H1' : models that RAISE phi under p1 have LOWER copy_score than models that FALL.

   It is labelled post-hoc in origin and pre-registered in test. Both halves of that are true and both
   matter.

2. THE NOISE CRITERION IS STRICTER, and it is the one F167 should have used. The gap is a DIFFERENCE
   between two measured endpoints -- max(UP) and min(DOWN) -- so its uncertainty combines both:
       SE(gap) = sqrt(SE_hi^2 + SE_lo^2),  SE_m = sqrt(p_m (1 - p_m) / N_m)
   with N_m = K * n_seeds pooled probes. F167 used a 2-point seed range, which is itself a very noisy
   estimator of noise. Both are computed here and the LARGER is used, which is the conservative choice
   and cannot be tuned after the fact.

ORDER OF EXECUTION IS DELIBERATE. Three models carry ~96% of the wall clock at float32 on CPU. The
gap depends on the BOUNDARY models -- max(UP) and min(DOWN) -- so those run FIRST. If the run is cut
short, the cells that decide the verdict already exist, and the rest only widen the interior.
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
from gatecheck import balance_report
from copy_vs_repeat import copy_score, dphi_on, FILLER, CENSUS_SEEDS   # estimator, unchanged

OUT = str(_ROOT / "results" / "copy_precision_k256.json")
K = 256
SEEDS = [20260803, 990017]
NOISE_FACTOR = 2.0

# boundary models first: they decide the gap, so a truncated run still answers the question
ORDER = ["tiiuae/Falcon3-1B-Base",            # max(UP) at K=64
         "HuggingFaceTB/SmolLM-1.7B",         # min(DOWN) at K=64
         "sapienzanlp/Minerva-3B-base-v1.0",  # the other UP
         "Qwen/Qwen1.5-1.8B", "EleutherAI/pythia-410m",
         "EleutherAI/pythia-410m-deduped", "bigcode/starcoder2-3b", "llm-jp/llm-jp-3-1.8b"]


def se_binom(p, n):
    return float(np.sqrt(max(p * (1 - p), 1e-12) / max(n, 1)))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"copy_scores": {}}
    # this file is loaded and re-saved across runs, so a key written by an EARLIER analysis survives
    # unless it is dropped. _truncated was the first version of the not-measured record and is
    # superseded by _not_measured below, which is recomputed every time.
    res.pop("_truncated", None)
    res["_preregistration"] = dict(
        supersedes="F167's K=64 measurement; copy_vs_repeat.py is unchanged and its result stands",
        K=K, seeds=SEEDS, filler=FILLER, device="cpu float32",
        H1_prime="models that RAISE phi under p1 have LOWER copy_score than models that FALL -- the "
                 "direction observed at K=64, POST-HOC IN ORIGIN and PRE-REGISTERED IN TEST, on "
                 "fresh probes rather than on the measurements that suggested it",
        why_direction_registered="F167's kill conditions were direction-agnostic (registry R10): a "
                                 "separation either way would have satisfied K2 while the hypothesis "
                                 "specified a sign. Registered here so that cannot recur.",
        noise_criterion="SE(gap) = sqrt(SE_hi^2 + SE_lo^2) with SE_m = sqrt(p(1-p)/N_m), N_m = K * "
                        "n_seeds pooled probes; also the 2-point seed range as in F167; the LARGER "
                        "is used. Conservative and not tunable after the fact.",
        kill_K1="ranges overlap -> the copy account is dropped",
        kill_K5="ranges separate but the gap does not exceed NOISE_FACTOR * SE(gap) -> NOT DECIDABLE "
                "FOR PRECISION again, and the remedy stays more probes, never a relaxed criterion",
        kill_direction="ranges separate with UP ABOVE DOWN -> H1' is refuted; that is the ORIGINAL "
                       "H1's direction and would mean the K=64 sign was noise",
        order="boundary models first, so a truncated run still decides the gap",
        refusals=["no p-value: 8 models cannot fail a significance test informatively",
                  "no rank correlation: below this project's ten-cluster floor (F149)"])

    # --analyse: read stored scores and compute the verdict without measuring. Used when the run is
    # STOPPED EARLY by design: the gap depends only on the BOUNDARY models (max UP, min DOWN), and
    # once those are measured the interior cannot change it. Which models were measured, and which
    # were deliberately not, are both recorded below so the truncation is visible rather than
    # inferred from a short list.
    analyse_only = "--analyse" in _sys.argv
    from transformers import AutoTokenizer, AutoModelForCausalLM
    base = json.load(open(_ROOT / "results" / "domain_base.json"))["runs"]
    ti = json.load(open(_ROOT / "results" / "text_interaction.json"))["runs"]
    fp = _ROOT / "results" / "text_interaction_fill.json"
    if fp.exists():
        ti = dict(ti); ti.update(json.load(open(fp))["runs"])

    # --skip-slow: exclude the models whose float32 CPU cost is measured in hours. Used after the
    # truncation left the p1 split at 2 up / 1 down, which the balance gate correctly refuses: the
    # BOUNDARY was decided by three models but a VERDICT needs >=2 a side. The remaining DOWN models
    # are the cheap ones (0.2-0.9 s/pass at K=64), so restoring the balance costs minutes.
    skip = {"bigcode/starcoder2-3b"} if "--skip-slow" in _sys.argv else set()
    if skip:
        res.setdefault("_skipped_slow", sorted(skip))
    for m in (() if analyse_only else [x for x in ORDER if x not in skip]):
        if m in res["copy_scores"]:
            continue
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", torch.float32)
        except Exception as e:
            res["copy_scores"][m] = dict(error=type(e).__name__)
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  {m}: LOAD FAILED {type(e).__name__}", flush=True)
            continue
        V = int(getattr(model.config, "vocab_size", len(tok)))
        sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                          tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
        per = []
        for s in SEEDS:
            import copy_vs_repeat as cvr
            old_k = cvr.K
            cvr.K = K                     # the estimator reads module-level K; restored below
            try:
                per.append(copy_score(model, "cpu", pool, np.random.default_rng(s)))
            finally:
                cvr.K = old_k
        p = float(np.mean(per))
        n_tot = K * len(SEEDS)
        res["copy_scores"][m] = dict(
            copy_score=round(p, 4), per_seed=[round(x, 4) for x in per],
            seed_half_range=round(float(abs(per[0] - per[1]) / 2), 4),
            se_binomial=round(se_binom(p, n_tot), 4), n_probes=n_tot,
            secs=round(time.time() - t0, 1))
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"  {m:<32} copy {p:.4f}  SE_binom {se_binom(p, n_tot):.4f}  "
              f"seed_half {abs(per[0]-per[1])/2:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        del model
        gc.collect()

    # WHAT WAS NOT MEASURED, recomputed here on every analysis rather than written once and left to
    # go stale. A truncated cohort has to be visible in the record; the earlier version of this block
    # was produced by a 3-model pass and survived a later run that measured four more, so the file
    # asserted a truncation that was no longer true.
    missing = [m for m in ORDER if "copy_score" not in res["copy_scores"].get(m, {})]
    if missing:
        res["_not_measured"] = dict(
            models=missing,
            why="cost. float32 CPU on a 16GB machine; these were the expensive cells and the gap is "
                "decided by the BOUNDARY models -- max(UP) and min(DOWN) -- which were run first.",
            can_they_move_the_boundary="no. Each unmeasured model scored >= 0.59 at K=64 (F167), and "
                "min(DOWN) here is 0.2324; moving the boundary would need a true score ~0.36 lower "
                "than its K=64 estimate, against a K=256 binomial SE of ~0.022. Stated as a bound, "
                "not as a guess: if any of them DID fall inside [0.1777, 0.2324] the verdict changes.")

    rows = []
    for m, v in res["copy_scores"].items():
        if "copy_score" not in v:
            continue
        d = dphi_on(ti, base, m, "p1")
        if d is None:
            continue
        rows.append(dict(model=m, **v, **d))
    res["p1_join"] = rows

    parts = [f"K={K} x {len(SEEDS)} seeds = {K*len(SEEDS)} probes per model, CPU float32. "
             + "; ".join(f"{r['model'].split('/')[-1]} {r['copy_score']:.4f}" for r in rows) + ". "]
    readable = [r for r in rows if r["direction"] != "flat" and r["headroom_ok"]]
    ups = [r for r in readable if r["direction"] == "up"]
    downs = [r for r in readable if r["direction"] == "down"]
    rep = balance_report([r["direction"] for r in readable], name="p1 direction")
    res["balance"] = dict(readable=len(readable), n_up=len(ups), n_down=len(downs),
                          excluded=[(r["model"].split("/")[-1], r["direction"], r["headroom_ok"])
                                    for r in rows if r not in readable],
                          gate_readable=bool(rep.readable), gate_reason=rep.reason)
    if len(ups) < 2 or len(downs) < 2 or not rep.readable:
        parts.append(f"NOT DECIDABLE: {len(ups)} up, {len(downs)} down among {len(readable)} "
                     f"readable. {rep.reason}")
    else:
        hi_up, lo_dn = max(ups, key=lambda r: r["copy_score"]), min(downs, key=lambda r: r["copy_score"])
        lo_up, hi_dn = min(ups, key=lambda r: r["copy_score"]), max(downs, key=lambda r: r["copy_score"])
        up_below = hi_up["copy_score"] < lo_dn["copy_score"]
        up_above = lo_up["copy_score"] > hi_dn["copy_score"]
        if up_below:
            gap = lo_dn["copy_score"] - hi_up["copy_score"]; a, b = hi_up, lo_dn
        elif up_above:
            gap = lo_up["copy_score"] - hi_dn["copy_score"]; a, b = hi_dn, lo_up
        else:
            gap, a, b = 0.0, None, None
        se_gap = (float(np.hypot(a["se_binomial"], b["se_binomial"])) if a else float("nan"))
        se_seed = (float(np.hypot(a["seed_half_range"], b["seed_half_range"])) if a else float("nan"))
        noise = max(se_gap, se_seed) if a else float("nan")
        res["separation"] = dict(up_range=[lo_up["copy_score"], hi_up["copy_score"]],
                                 down_range=[lo_dn["copy_score"], hi_dn["copy_score"]],
                                 direction=("up_below" if up_below else
                                            "up_above" if up_above else "overlap"),
                                 gap=round(gap, 4), se_gap_binomial=round(se_gap, 4),
                                 se_gap_seed=round(se_seed, 4), noise_used=round(noise, 4),
                                 beats_noise=bool(gap > NOISE_FACTOR * noise) if a else False)
        parts.append(
            f"UP [{lo_up['copy_score']:.4f}, {hi_up['copy_score']:.4f}], DOWN "
            f"[{lo_dn['copy_score']:.4f}, {hi_dn['copy_score']:.4f}]. ")
        if not a:
            parts.append("K1 FIRES: the ranges OVERLAP. Copy strength does not separate the "
                         "direction of the p1 effect, and the copy account is dropped -- on a "
                         "behavioural measurement, not an argument.")
        elif gap <= NOISE_FACTOR * noise:
            parts.append(
                f"NOT DECIDABLE FOR PRECISION again: gap {gap:.4f} against {NOISE_FACTOR}x noise "
                f"{NOISE_FACTOR*noise:.4f} (binomial {se_gap:.4f}, seed {se_seed:.4f}, larger used). "
                f"Quadrupling the probes did not resolve it; the remedy stays more measurement.")
        elif up_below:
            parts.append(
                f"H1' SUPPORTED: UP models sit BELOW DOWN models by {gap:.4f} against "
                f"{NOISE_FACTOR}x noise {NOISE_FACTOR*noise:.4f}. The direction registered before "
                f"this run is the direction observed, on fresh probes. Note it is the INVERSE of "
                f"F167's original H1.")
        else:
            parts.append(
                f"H1' REFUTED and the ORIGINAL H1's direction observed: UP sits ABOVE DOWN by "
                f"{gap:.4f}. That would mean the K=64 sign was noise, and neither hypothesis is "
                f"left standing without a third measurement.")
    # HOW GOOD IS THE NOISE ESTIMATOR ITSELF? The seed term is a TWO-POINT range, and the "larger of
    # the two" rule lets it set the gate. Simulate its sampling distribution under PURE count noise at
    # the observed p and this K, so the gate's own reliability is a measured quantity in the file
    # rather than an assumption in the prose. This changes no verdict; it explains one.
    rng = np.random.default_rng(20260820)
    ratios = {}
    for r in rows:
        p_hat = r["copy_score"]
        x = rng.binomial(K, p_hat, size=(40000, 2)) / K
        half = np.abs(x[:, 0] - x[:, 1]) / 2
        se_pool = float(np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / (K * len(SEEDS))))
        sim_ratio = half / se_pool
        obs = r["seed_half_range"] / se_pool if se_pool else float("nan")
        ratios[r["model"].split("/")[-1]] = dict(
            observed_ratio=round(float(obs), 3),
            expected_ratio=round(float(sim_ratio.mean()), 3),
            sd_of_ratio=round(float(sim_ratio.std()), 3),
            p_at_least_this_extreme=round(float(np.mean(sim_ratio >= obs)), 3))
    # The question the gate actually raises is NOT "could this model beat its own observation" -- that
    # is ~1 by construction and says nothing. It is: given 7 models, how often does the LARGEST ratio
    # reach the one that set the gate? That is a max over the cohort, evaluated at a single threshold.
    worst_obs = max(d["observed_ratio"] for d in ratios.values())
    per_model_at_worst, any_p = {}, 1.0
    for r in rows:
        p_hat = r["copy_score"]
        x = rng.binomial(K, p_hat, size=(40000, 2)) / K
        se_pool = float(np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / (K * len(SEEDS))))
        pi = float(np.mean((np.abs(x[:, 0] - x[:, 1]) / 2) / se_pool >= worst_obs))
        per_model_at_worst[r["model"].split("/")[-1]] = round(pi, 3)
        any_p *= (1.0 - pi)
    any_p = 1.0 - any_p
    res["noise_estimator_reliability"] = dict(
        what="seed_half_range / SE_binomial per model, against its distribution under PURE count "
             "noise. If the observed ratios sit inside the simulated spread, the across-seed term "
             "carries no information the binomial term lacks -- it is just a noisier estimate of the "
             "same thing, and the max() rule is then gated by estimator scatter.",
        per_model=ratios,
        threshold_that_set_the_gate=round(float(worst_obs), 3),
        p_each_model_reaches_that_threshold=per_model_at_worst,
        p_largest_of_cohort_reaches_that_threshold=round(any_p, 3),
        reading="a two-point range has sd ~0.6x its own mean, so across 7 models the largest ratio "
                "is expected to be roughly double the typical one BY CONSTRUCTION. More probes "
                "shrink the binomial term and the true seed variance but cannot stabilise a "
                "two-point ESTIMATOR of that variance; more SEEDS, or a paired design scoring the "
                "compared models on the SAME probe pairs, is the axis that would.")

    parts.append(
        f"BOUNDARY: one arm (p1), {len(rows)} models, K={K}, CPU float32, boundary models run first. "
        f"No p-value and no rank correlation. The prior-art gate for induction heads and repetition "
        f"self-reinforcement remains OWED and gates any write-up.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
