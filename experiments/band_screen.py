"""The 1.5-3B band screen (#101 keystone): one family-diverse run feeding three threads.

WHAT THIS RUN IS FOR, after its own gates. Gate 0 (band_family_census): 22 conservative families
with a base, ungated checkpoint in [1.4B, 3.6B]. Gate B (band_benchmark_range): the benchmark
correlation is NOT powered -- only 11 families have leaderboard coverage -- so the benchmark
PRIMARY is demoted to an exploratory secondary, declared here before the data exists. What the
run stands on is its RIDERS, none of which needs the leaderboard:

  T* x rep_4 (#90 / F68)   the melting temperature against greedy-decoding degeneration.
                           F68: rho=+0.552 at n=10 needed ~16; family-level n here can reach
                           ~16-20 PAIRS for the first time. This is the run's primary.
  F64 scale gate           is the attractor still scale-blind at 1.5-3B? Families screened at
                           small scale (pythia, gpt-neo via eleutherai-pile, qwen, granite,
                           smollm, bloom) have band members here; if the attractor tracks scale
                           in-family, F64 needs amending and #101's kill fires.
  corpus direction         top-1 share and dominant token per family at the frozen geometry --
                           the fingerprint programme's 2-family weakness, measured at 20+.
  battery riders           radius (r=2->3) and BOS drops, plus gate1's static census -- the
                           Gate-2 arms, carried so K1 stays evaluable at this scale.

EVERYTHING IS IMPORTED, NOTHING REIMPLEMENTED (hazard 1): t_star and rep_stats/PROMPTS from
degeneration_vs_tstar (same thresholds, same greedy protocol), argmax_census/conditional_stats
from gate1 (F70-gated), the settle through ar_ca.run with order="per_replica" (F57).

SEEDS. Two per cell. That is HALF gate2's four -- declared, not hidden -- and one MORE than the
original 26-model screen (single seed), whose numbers back F63/F64/F68. Between-family variance
is the quantity of interest at n=22; per-seed values are stored so thin cells are visible.

FAILURES ARE DATA. Several band families need trust_remote_code or bleeding-edge architectures
(bitnet, plamo, LFM2, possibly zamba2/jamba on this stack); loads are attempted with the same
transformers everything else used, failures recorded per model, and n_measured reported against
n_candidate. Silently shrinking n is how pseudoreplication hides (F68).

Writes results/band_screen.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/band_screen.py [--probe]
        (resumable per (model, arm, T, seed); safe to interrupt)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint")]
import collections, gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from ranking import rank as _rank
import torch

from provenance import stamp, rel
from degeneration_vs_tstar import t_star, rep_stats, PROMPTS, NEW_TOKENS
from gate1 import argmax_census, conditional_stats

OUT = str(_ROOT / "results" / "band_screen.json")
CENSUS = _ROOT / "results" / "band_family_census.json"
RANGE = _ROOT / "results" / "band_benchmark_range.json"

TEMPS = [0.02, 0.2, 0.436, 0.7]     # the screen's ladder -- T* interpolates on exactly these
SEEDS = [101, 102]
N, B, SETTLE, R, R_ALT = 96, 16, 16, 2, 3
TSTAR_THRESH = 0.40                  # the screen's threshold, unchanged (attractor binary)
CELL_BUDGET_S = 420                  # if ONE settle exceeds this, the model is too slow for this
                                     # stack (state-space archs fall back to a python-loop path on
                                     # MPS -- the probe measured Jamba2-3B at >5 min per cell) and
                                     # the remaining 11 settles would eat the night. One cell is
                                     # kept as data; the model is recorded too_slow, not hidden.
SLOW_ARCH = ("Jamba", "Mamba", "Zamba", "LFM", "BitNet", "Ouro", "Plamo")


def representatives():
    """One model per conservative family: the leaderboard-covered member where one exists
    (so the exploratory benchmark secondary shares its axis), else the most-downloaded usable
    in-band member. Deterministic, derived from the two gate files, never hand-listed."""
    cen = json.load(open(CENSUS))
    rng_ = json.load(open(RANGE))
    covered = {f: v["model"] for f, v in rng_["covered"].items()}
    by_model = {x["model"]: x for x in cen["in_band"]}
    reps = {}
    for fam, models in cen["curated_conservative"].items():
        if fam in covered and covered[fam] in by_model and by_model[covered[fam]]["usable"]:
            reps[fam] = covered[fam]
            continue
        usable = [m for m in models if by_model.get(m, {}).get("usable")]
        if usable:
            reps[fam] = max(usable, key=lambda m: by_model[m].get("downloads") or 0)
    # the ibm leaderboard member was added by Gate B and is absent from the census in_band list
    if "ibm" in covered and "ibm" not in reps:
        reps["ibm"] = covered["ibm"]
    return reps


def settle_top1(rule, T, r, scheme, seed):
    from ar_ca import run
    s = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme=scheme,
            init="random", seed=seed, order="per_replica")["final"]
    tops, toks = [], collections.Counter()
    for row in s:
        c = collections.Counter(row.tolist())
        tops.append(c.most_common(1)[0][1] / N)
        toks.update(c)
    return float(np.mean(tops)), int(toks.most_common(1)[0][0])


def measure_model(fam, name, res, probe=False):
    from ar_ca import ARRule
    from transformers import AutoTokenizer, AutoModelForCausalLM
    runs = res["runs"]
    if f"{name}|too_slow" in runs:
        print(f"  {fam}/{name}: previously marked too slow -- skipping", flush=True)
        return
    need = ([f"{name}|T{T}|s{s}" for T in TEMPS for s in SEEDS]
            + [f"{name}|r{R_ALT}|s{s}" for s in SEEDS]
            + [f"{name}|bos|s{s}" for s in SEEDS]
            + [f"{name}|rep", f"{name}|baseline"])
    if all(k in runs for k in need):
        print(f"  {fam}/{name}: already complete", flush=True)
        return
    t0 = time.time()
    try:
        rule = ARRule(name)
    except Exception as e:
        print(f"  {fam}/{name}: LOAD FAILED ({type(e).__name__}: {str(e)[:70]})", flush=True)
        runs[f"{name}|failed"] = dict(model=name, family=fam, error=type(e).__name__,
                                      detail=str(e)[:200])
        json.dump(res, open(OUT, "w"), indent=1)
        return
    print(f"\n  {fam}/{name} loaded in {time.time()-t0:.0f}s", flush=True)

    for T in TEMPS:
        for s in SEEDS:
            k = f"{name}|T{T}|s{s}"
            if k in runs: continue
            t1 = time.time()
            a, tk = settle_top1(rule, T, R, "none", s)
            cell_s = time.time() - t1
            runs[k] = dict(model=name, family=fam, arm="temp", T=T, seed=s,
                           top1=round(a, 4), dominant=tk, secs=round(cell_s, 1))
            json.dump(res, open(OUT, "w"), indent=1)
            if cell_s > CELL_BUDGET_S and f"{name}|too_slow" not in runs:
                runs[f"{name}|too_slow"] = dict(
                    model=name, family=fam, cell_secs=round(cell_s, 1),
                    note=f"one settle exceeded {CELL_BUDGET_S}s; remaining battery skipped so "
                         f"the run yields families instead of stalling")
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {fam}/{name}: TOO SLOW ({cell_s:.0f}s/cell) -- skipping rest",
                      flush=True)
                del rule
                try: torch.mps.empty_cache()
                except Exception: pass
                gc.collect()
                return
        vs = [runs[f"{name}|T{T}|s{s}"]["top1"] for s in SEEDS]
        print(f"     T={T:<6} top1={np.mean(vs):.3f}", flush=True)
        if probe:
            print(f"  PROBE: one temp cell ~{runs[f'{name}|T{T}|s{SEEDS[0]}']['secs']:.0f}s; "
                  f"battery = {len(need)-2} settles + rep + census", flush=True)
            return
    for arm, r_, scheme in ((f"r{R_ALT}", R_ALT, "none"), ("bos", R, "bos")):
        for s in SEEDS:
            k = f"{name}|{arm}|s{s}"
            if k in runs: continue
            # AN ARM MAY FAIL WITHOUT TAKING THE SCREEN WITH IT. helium-1 has no BOS token, so
            # scheme="bos" crashed on np.full(..., None) -- and because the supervisor restarts
            # on death, that ONE cell burned all 40 restart passes in a crash loop. A recorded
            # arm failure is data (the model HAS no BOS arm); an unhandled one is a spin lock.
            try:
                a, tk = settle_top1(rule, TEMPS[0], r_, scheme, s)
                runs[k] = dict(model=name, family=fam, arm=arm, T=TEMPS[0], seed=s,
                               top1=round(a, 4), dominant=tk)
            except Exception as e:
                runs[k] = dict(model=name, family=fam, arm=arm, seed=s,
                               failed=type(e).__name__, detail=str(e)[:120])
            json.dump(res, open(OUT, "w"), indent=1)

    if f"{name}|rep" not in runs:            # the #90 partner: greedy degeneration, imported
        tok, model = rule.tok, rule.model
        stats = []
        for p in PROMPTS:
            try:
                ids = tok(p, return_tensors="pt").input_ids.to(model.device)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False,
                                         pad_token_id=tok.eos_token_id or 0)
                st = rep_stats(out[0, ids.shape[1]:].tolist())
                if st: stats.append(st)
            except Exception as e:
                print(f"    rep prompt failed: {type(e).__name__}", flush=True)
        runs[f"{name}|rep"] = (dict(model=name, family=fam,
                                    rep_4=round(float(np.mean([s["rep_4"] for s in stats])), 4),
                                    n_prompts=len(stats))
                               if stats else dict(model=name, family=fam, failed="no generations"))
        json.dump(res, open(OUT, "w"), indent=1)

    if f"{name}|baseline" not in runs:       # gate1's static census, F70-gated machinery
        try:
            V = int(getattr(rule.model.config, "vocab_size", len(rule.tok)))
            sp = {i for i in (rule.tok.bos_token_id, rule.tok.eos_token_id,
                              rule.tok.pad_token_id, rule.tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(rule.tok))) if i not in sp], np.int64)
            rec = dict(model=name, family=fam)
            rec.update(argmax_census(rule.model, rule.tok, str(rule.model.device), pool,
                                     np.random.default_rng(20260803)))
            rec.update(conditional_stats(rule.model, rule.tok, str(rule.model.device), pool,
                                         np.random.default_rng(20260803)))
            runs[f"{name}|baseline"] = rec
        except Exception as e:
            runs[f"{name}|baseline"] = dict(model=name, family=fam, failed=type(e).__name__)
        json.dump(res, open(OUT, "w"), indent=1)

    del rule
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    print(f"  {fam}/{name}: done in {(time.time()-t0)/60:.1f} min", flush=True)


def spearman(a, b):
    ra = _rank(a); rb = _rank(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(res):
    runs = res["runs"]
    reps = res["_preregistration"]["representatives"]
    fam_of = {m: f for f, m in reps.items()}
    ok = {}
    for f, m in reps.items():
        prof = {}
        for T in TEMPS:
            vs = [runs[f"{m}|T{T}|s{s}"]["top1"] for s in SEEDS if f"{m}|T{T}|s{s}" in runs]
            if vs:
                prof[f"{m}@{T}"] = {"top1_share": float(np.mean(vs))}
        if len(prof) == len(TEMPS):
            ts = t_star(prof, m, thresh=TSTAR_THRESH)
            rep = runs.get(f"{m}|rep", {})
            ok[f] = dict(model=m,
                         top1_low=prof[f"{m}@{TEMPS[0]}"]["top1_share"],
                         tstar=ts if isinstance(ts, (int, float)) else None,
                         tstar_state=("finite" if isinstance(ts, (int, float)) else
                                      "censored_above" if ts == "censored_above" else "none"),
                         rep_4=rep.get("rep_4"))
    failed = sorted({v["model"] for k, v in runs.items() if k.endswith("|failed")})
    print(f"\n=== {len(ok)} families measured; {len(failed)} loads failed: {failed} ===")
    print(f"  {'family':24s} {'top1@0.02':>9} {'T*':>8} {'rep_4':>7}")
    for f in sorted(ok):
        v = ok[f]
        print(f"  {f:24s} {v['top1_low']:9.3f} {str(v['tstar'] or v['tstar_state']):>8} "
              f"{v['rep_4'] if v['rep_4'] is not None else float('nan'):7.3f}")

    parts = []
    # ---- rider 1: T* x rep_4, family-level, the #90 test at first real power ----
    pairs = [(v["tstar"], v["rep_4"]) for v in ok.values()
             if v["tstar"] is not None and v["rep_4"] is not None]
    if len(pairs) >= 3:
        a = np.array(pairs)
        rho = spearman(a[:, 0], a[:, 1])
        rng = np.random.default_rng(0)
        perm = sum(abs(spearman(rng.permutation(a[:, 0]), a[:, 1])) >= abs(rho)
                   for _ in range(10000)) / 10000
        parts.append(
            f"T*-rep_4 (#90): rho={rho:+.3f} over n={len(pairs)} FAMILIES with finite T*, "
            f"permutation p={perm:.4f}. F68's version was rho=+0.552 at n=10 (p=0.107), needing "
            f"~16; this is the first measurement at family-level power, and "
            + ("it now clears significance." if perm < 0.05 else
               "it still does not clear 0.05 -- report as measured, not massaged."))
        res.setdefault("analysis", {})["tstar_rep4"] = dict(
            rho=round(rho, 3), n_pairs=len(pairs), perm_p=perm)
    censored = [f for f, v in ok.items() if v["tstar_state"] == "censored_above"]
    none_ = [f for f, v in ok.items() if v["tstar_state"] == "none"]
    parts.append(f"T* states: {len(pairs)} finite, {len(censored)} censored_above "
                 f"({', '.join(censored) or '-'}), {len(none_)} no attractor.")

    # ---- rider 2: F64 scale gate, in-family where a small-scale twin was screened ----
    try:
        small = json.load(open(_ROOT / "results" / "attractor_corpus_screen.json"))["at_lowest_T"]
    except Exception:
        small = {}
    fam_small = {"eleutherai-pile": "EleutherAI/pythia-410m", "qwen": "Qwen/Qwen2.5-0.5B",
                 "ibm": "ibm-granite/granite-3.0-1b-a400m-base",
                 "bigscience/bloom": "bigscience/bloom-560m",
                 "hftb": "HuggingFaceTB/SmolLM2-360M",
                 "stability": "stabilityai/stablelm-2-1_6b"}
    scale_rows = []
    for f, sm in fam_small.items():
        if f in ok and sm in small:
            scale_rows.append((f, small[sm]["top1_share"], ok[f]["top1_low"]))
    if scale_rows:
        drifts = [abs(b - s) for _, s, b in scale_rows]
        binary_flips = [f for f, s, b in scale_rows
                        if (s >= TSTAR_THRESH) != (b >= TSTAR_THRESH)]
        parts.append(
            f"F64 SCALE GATE over {len(scale_rows)} in-family small-vs-band pairs: max top-1 "
            f"drift {max(drifts):.3f}, attractor-binary flips: "
            f"{', '.join(binary_flips) if binary_flips else 'NONE'}. "
            + ("The attractor stays scale-blind into the band; F64's ladder extends."
               if not binary_flips else
               "The binary FLIPS in-family across scale -- F64 needs amending and #101's kill "
               "condition fires; the corpus channel is no longer cleanly isolated."))
        res.setdefault("analysis", {})["scale_gate"] = dict(
            pairs=[(f, round(s, 3), round(b, 3)) for f, s, b in scale_rows],
            binary_flips=binary_flips)

    # ---- exploratory secondary: benchmark correlation on the covered subset ----
    rng_ = json.load(open(RANGE))
    cov = rng_["covered"]
    bench_ok = rng_["usable_benchmarks"]
    sec = {}
    for bch in bench_ok:
        xs = [(ok[f]["top1_low"], cov[f]["scores"][bch]) for f in ok
              if f in cov and cov[f]["model"] == ok[f]["model"]]
        if len(xs) >= 5:
            a = np.array(xs)
            sec[bch] = dict(rho=round(spearman(a[:, 0], a[:, 1]), 3), n=len(xs))
    if sec:
        parts.append(
            "EXPLORATORY (declared before data; Gate B failed on coverage, n<16): top1@0.02 vs "
            + "; ".join(f"{b} rho={v['rho']:+.2f} (n={v['n']})" for b, v in sec.items())
            + ". Hypothesis-generating only -- not powered, not corrected, not a claim.")
        res.setdefault("analysis", {})["benchmark_exploratory"] = sec

    parts.append(f"n_measured={len(ok)} of {len(reps)} candidates; failures are listed, not "
                 f"hidden (F68).")
    res["families"] = ok
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "The #101 band screen after its two gates: Gate 0 gave 22 conservative families at "
        "1.5-3.6B; Gate B failed the benchmark PRIMARY on leaderboard coverage (11<16), so the "
        "run stands on its riders -- T* x rep_4 at family-level power for the first time (#90), "
        "the F64 scale gate over in-family small-vs-band pairs, and the corpus/battery riders -- "
        "with the benchmark correlation demoted to a labeled exploratory secondary before the "
        "data existed. All estimators imported: t_star/rep_stats/PROMPTS from "
        "degeneration_vs_tstar, argmax_census/conditional_stats from gate1 (F70-gated), settles "
        "through ar_ca.run with per-replica orders (F57). Two seeds per cell -- half of gate2's "
        "four, double the original screen's one -- declared with per-seed values stored.")
    print(f"\n  -> {res['verdict']}")


def main(probe=False):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    reps = representatives()
    res["_preregistration"] = dict(
        representatives=reps, temps=TEMPS, seeds=SEEDS, N=N, B=B, settle=SETTLE,
        r=R, r_alt=R_ALT, tstar_thresh=TSTAR_THRESH,
        primary="T* x rep_4 at family level (the #90 anchor, first time near F68's n~16) and "
                "the F64 scale gate over in-family small-vs-band pairs",
        exploratory="benchmark correlation over the Gate-B-covered subset (n=11<16), declared "
                    "exploratory BEFORE the data exists",
        seeds_note="2 per cell: half gate2's 4, double the original screen's 1; per-seed stored",
        resumable="keyed per (model, arm, T, seed)")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"representatives ({len(reps)}):", flush=True)
    for f, m in sorted(reps.items()):
        print(f"  {f:26s} {m}", flush=True)
    # fast transformer architectures first, kernel-fallback architectures last, so the night
    # yields measured families even if the slow tail stalls
    cen = json.load(open(CENSUS))
    arch = {x["model"]: (x.get("arch") or "") for x in cen["in_band"]}
    def slow(m):
        return any(k.lower() in arch.get(m, m).lower() for k in SLOW_ARCH)
    ordered = sorted(reps.items(), key=lambda fm: (slow(fm[1]), fm[0]))
    for f, m in ordered:
        measure_model(f, m, res, probe=probe)
        if probe:
            return
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main(probe="--probe" in _sys.argv)
