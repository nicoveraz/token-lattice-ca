"""Are the domain effects an OUT-OF-DISTRIBUTION artefact? F66's warning, applied to F144-F159.

WHY THIS IS THE EXPERIMENT F159 CALLS FOR, AND WHY IT IS THE HIGHEST-STAKES ONE IN THE SEQUENCE.

F159 found the attention-sink account holds on REAL TEXT and fails on UNIFORMLY RANDOM tokens -- and
uniformly random tokens are exactly what our census feeds. That is a regime statement about the sink
mechanism, but it points at something larger, because this project already found the same thing about
its own readout once:

    F66: "the degeneracy is an out-of-distribution prompt artifact". A two-token context is far
    outside anything a model trained on thousands of tokens has seen, and one BOS token -- enough to
    make the context look like a document start rather than a fragment from nowhere -- removed two
    thirds of the effect (74.4% -> 24.1%). F66 read that collapse as the SIGNATURE of an OOD prompt,
    not of model dynamics.

Every finding F144-F159 runs on that same two-token census, and its starts are drawn UNIFORMLY FROM
THE VOCABULARY -- not merely short, but token sequences no corpus would ever contain. If the domain
effects are conditional on that, then paper 2 measures how models respond to nonsense under
conditioning, which is a real but much smaller claim than the one currently drafted.

THE TEST. Hold everything fixed except where the starts come from:
    random   two tokens drawn uniformly from the vocabulary  -- the existing census, F144-F159
    text     two ADJACENT tokens drawn from real text        -- in-distribution bigrams
and measure the BOS domain effect under each. Adjacency matters: drawing two independent tokens from
a text-derived pool would fix the unigram marginal but still produce bigrams no corpus contains, so
the starts are taken as adjacent pairs at offsets chosen by index.

PRE-REGISTERED:
  RUNG      this file implements its own census because `gate1.argmax_census` draws starts internally
            and cannot take them as an argument, and EDITING gate1 would invalidate every stamped
            result through the provenance import closure. The local implementation must therefore
            reproduce `argmax_census` BIT-IDENTICALLY when fed the starts that function would have
            drawn, on every model. Any difference and nothing below is read.
  PRIMARY   does the BOS domain effect survive in-distribution? Per model, compare
            d(phi) = phi(bos) - phi(raw) under RANDOM starts against the same under TEXT starts.
            Registered readings:
              d(phi) survives in sign and rough magnitude under text starts -> the domain effect is
                NOT an OOD artefact. F66's warning does not extend to this readout, and paper 2's
                claims stand as drafted.
              d(phi) collapses or reverses under text starts -> the domain effects of F144-F159 are
                CONDITIONAL ON OUT-OF-DISTRIBUTION INPUT. Paper 2 must say so in the abstract, not
                the limits: the subject becomes how models respond to conditioning when the loop is
                seeded with text no corpus contains. Given F66 this is a live outcome, and it is the
                reason to run this before submission rather than after.
              mixed across models -> report per model; the programme's own history says a factor that
                looks clean at n=6 usually is not.
  SECONDARY phi itself under text starts vs random starts, per model and arm. A large baseline shift
            would mean the two regimes are not measuring the same object at all, which bounds the
            PRIMARY's interpretation whichever way it goes.
  BOUNDARY  one text source, one prefix kind (BOS), two census seeds. In-distribution here means
            "adjacent pairs from one corpus", not "representative of use".
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import collections, gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census, MAX_STEPS, _next_logits
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_templated import _Prefixed

OUT = str(_ROOT / "results" / "in_distribution_census.json")
BASE = _ROOT / "results" / "domain_base.json"
SCREEN = _ROOT / "results" / "midrange_screen.json"

MODELS = ["tiiuae/Falcon3-1B-Base", "sapienzanlp/Minerva-3B-base-v1.0",
          "EleutherAI/pythia-410m-deduped", "Qwen/Qwen1.5-1.8B",
          "HuggingFaceTB/SmolLM-1.7B", "Qwen/Qwen2.5-1.5B-Instruct"]
START_KINDS = ("random", "text")
ARMS = ("raw", "bos")
PILE_ROWS = tuple(range(0, 40))       # fixed before the run; adjacent pairs are taken from these
MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0


def census_from_starts(model, tok, dev, starts):
    """A copy of gate1.argmax_census that takes STARTS explicitly.

    gate1 is NOT edited: its import closure is stamped into every stored result, and widening its
    signature would invalidate them. The RUNG below proves this reproduces it exactly."""
    endpoints, fixed, cyclic = [], 0, 0
    for s in starts:
        ctx = [int(s[0]), int(s[1])]
        seen, end = set(), ctx[-1]
        for _ in range(MAX_STEPS):
            nxt = int(torch.argmax(_next_logits(model, ctx, dev)))
            end = nxt
            if ctx[0] == ctx[1] == nxt:
                fixed += 1
                break
            state = (ctx[0], ctx[1])
            if state in seen:
                cyclic += 1
                break
            seen.add(state)
            ctx = [ctx[1], nxt]
        endpoints.append(end)
    n = len(starts)
    cnt = collections.Counter(endpoints)
    top_tok, top_n = cnt.most_common(1)[0]
    return dict(n_starts=n, n_distinct_endpoints=len(cnt),
                modal_endpoint_share=round(top_n / n, 4),
                modal_endpoint_token=tok.decode([int(top_tok)]),
                fixed_point_fraction=round(fixed / n, 4),
                cyclic_fraction=round(cyclic / n, 4),
                endpoint_histogram=[[int(t), tok.decode([int(t)]), int(c)]
                                    for t, c in cnt.most_common()])


def random_starts(pool, seed, n=None):
    """Exactly what gate1.argmax_census draws: n independent pairs via rng.choice(pool, size=2)."""
    rng = np.random.default_rng(seed)
    return [[int(x) for x in rng.choice(pool, size=2)] for _ in range(n or N_STARTS)]


def text_starts(tok, seed, n=None):
    """ADJACENT token pairs from real text -- in-distribution bigrams, not a text-derived marginal."""
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    ids = []
    for r in PILE_ROWS:
        ids.extend(tok(ds[int(r)]["text"][:4000], add_special_tokens=False)["input_ids"])
        if len(ids) > 20000:
            break
    rng = np.random.default_rng(seed)
    n = n or N_STARTS
    offs = rng.integers(0, len(ids) - 2, size=n)
    return [[int(ids[o]), int(ids[o + 1])] for o in offs]


def _stored_phi(m, arm):
    runs = json.load(open(BASE))["runs"] if BASE.exists() else {}
    ks = [f"{m}|s{cs}|{arm}" for cs in CENSUS_SEEDS]
    if all(k in runs for k in ks):
        return float(np.mean([runs[k]["fixed_point_fraction"] for k in ks]))
    return None


def analyse(res):
    cells, parts, analysis = res["cells"], [], {}

    rung = [v for k, v in cells.items() if k.endswith("|rung")]
    worst = max((v["abs_err"] for v in rung), default=float("inf"))
    ok = bool(rung) and worst == 0.0
    parts.append(
        f"RUNG (local census reproduces gate1.argmax_census on identical starts): {len(rung)} models "
        f"compared, worst error {worst:.2e}. "
        + ("Bit-identical, so the only thing that differs below is WHERE THE STARTS COME FROM."
           if ok else "NOT identical -- the local census is not gate1's measurement and nothing "
                      "below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    rows = {}
    for m in MODELS:
        got = {}
        for kind in START_KINDS:
            for arm in ARMS:
                ks = [f"{m}|{kind}|{arm}|s{cs}" for cs in CENSUS_SEEDS]
                if all(k in cells for k in ks):
                    v = [cells[k]["fixed_point_fraction"] for k in ks]
                    got[(kind, arm)] = (float(np.mean(v)), float(abs(v[0] - v[1])))
        if len(got) == 4:
            rows[m] = got
    analysis["rows"] = {m: {f"{k}|{a}": dict(phi=round(v, 4), noise=round(nz, 4))
                            for (k, a), (v, nz) in g.items()} for m, g in rows.items()}
    if not rows:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " No model has all four cells yet."
        return

    tab, survives, collapses, undecided = [], [], [], []
    for m, g in rows.items():
        (pr, nr), (pb, nb) = g[("random", "raw")], g[("random", "bos")]
        (tr, nt), (tb, nt2) = g[("text", "raw")], g[("text", "bos")]
        d_rand, d_text = pb - pr, tb - tr
        tol_r = max(MIN_SHIFT, NOISE_FACTOR * max(nr, nb))
        tol_t = max(MIN_SHIFT, NOISE_FACTOR * max(nt, nt2))
        rand_real = abs(d_rand) > tol_r
        text_real = abs(d_text) > tol_t
        row = dict(model=m, phi_rand_raw=round(pr, 4), phi_rand_bos=round(pb, 4),
                   d_random=round(d_rand, 4), tol_random=round(tol_r, 4),
                   phi_text_raw=round(tr, 4), phi_text_bos=round(tb, 4),
                   d_text=round(d_text, 4), tol_text=round(tol_t, 4))
        if not rand_real:
            row["verdict"] = "no effect to test under random starts"
            undecided.append(m)
        elif text_real and np.sign(d_text) == np.sign(d_rand):
            row["verdict"] = "survives"
            survives.append(m)
        elif not text_real:
            row["verdict"] = "COLLAPSES (in-distribution shift within noise)"
            collapses.append(m)
        else:
            row["verdict"] = "REVERSES"
            collapses.append(m)
        tab.append(row)
    analysis.update(table=tab, survives=survives, collapses=collapses, undecided=undecided)

    MIN_MODELS = 3    # "survives on 1 of 1" is not a finding -- this programme's own history
    if len(tab) < MIN_MODELS:
        analysis["primary_readable"] = False
        parts.append(
            "PRIMARY, per model so far: "
            + "; ".join("{}: random {:+.3f} vs text {:+.3f} -> {}".format(
                r["model"].split("/")[-1], r["d_random"], r["d_text"], r["verdict"]) for r in tab)
            + f". NOT READ: {len(tab)} of {MIN_MODELS} models required. A confident universal from a "
              f"handful of models is the defect this programme has hit four times (F151, F153, F154, "
              f"F156); the count gate is here so it cannot happen a fifth.")
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts)
        return
    analysis["primary_readable"] = True
    parts.append(
        "PRIMARY, does the BOS domain effect survive in-distribution starts? "
        + "; ".join("{}: random {:+.3f} (tol {:.3f}) vs text {:+.3f} (tol {:.3f}) -> {}".format(
            r["model"].split("/")[-1], r["d_random"], r["tol_random"], r["d_text"], r["tol_text"],
            r["verdict"]) for r in tab)
        + f". Survives on {len(survives)}, collapses or reverses on {len(collapses)}, "
          f"untestable on {len(undecided)} of {len(tab)}. "
        + ("The domain effect is NOT an out-of-distribution artefact on these models: it survives "
           "when the loop is seeded with real bigrams. F66's warning does not extend to this "
           "readout, and paper 2's claims stand as drafted."
           if survives and not collapses else
           "The domain effect COLLAPSES OR REVERSES in-distribution on every testable model. The "
           "effects of F144-F159 are CONDITIONAL ON OUT-OF-DISTRIBUTION INPUT, exactly as F66 found "
           "for the frozen fraction. Paper 2 must state this in the ABSTRACT, not the limits: its "
           "subject becomes how models respond to conditioning when seeded with text no corpus "
           "contains."
           if collapses and not survives else
           f"MIXED: survives on {[m.split('/')[-1] for m in survives]}, collapses or reverses on "
           f"{[m.split('/')[-1] for m in collapses]}. Report per model. This programme's history is "
           f"that a factor looking clean at this n usually is not, so do not summarise this as a "
           f"rate."))

    base = [(m, g[("random", "raw")][0], g[("text", "raw")][0]) for m, g in rows.items()]
    analysis["baseline_shift"] = [dict(model=m, phi_random=round(a, 4), phi_text=round(b, 4),
                                       shift=round(b - a, 4)) for m, a, b in base]
    big = [x for x in base if abs(x[2] - x[1]) >= 0.2]
    parts.append(
        "SECONDARY, phi under text starts vs random starts on the RAW arm: "
        + "; ".join(f"{m.split('/')[-1]} {a:.3f} -> {b:.3f} ({b - a:+.3f})" for m, a, b in base)
        + ". "
        + (f"{len(big)} of {len(base)} models shift by >= 0.20, so the two start distributions are "
           f"not probing the same object and the PRIMARY must be read as a comparison between two "
           f"regimes rather than a robustness check within one."
           if big else
           "No model shifts by >= 0.20, so the two start distributions probe comparable objects and "
           "the PRIMARY is a robustness check in the ordinary sense."))
    parts.append(
        f"BOUNDARY: {len(rows)} models, ONE text source (Pile rows {PILE_ROWS[0]}-{PILE_ROWS[-1]}), "
        f"ONE prefix kind (BOS), {len(CENSUS_SEEDS)} census seeds, {N_STARTS} starts. "
        f"'In-distribution' here means adjacent pairs from one corpus, not representative of use. "
        f"The local census is gate1's, proven by the RUNG rather than assumed.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, start_kinds=list(START_KINDS), arms=list(ARMS), pile_rows=list(PILE_ROWS),
        n_starts=N_STARTS, census_seeds=CENSUS_SEEDS, min_shift=MIN_SHIFT,
        noise_factor=NOISE_FACTOR,
        rung="the local census must reproduce gate1.argmax_census BIT-IDENTICALLY on the starts that "
             "function would have drawn; gate1 is not edited because its import closure is stamped "
             "into every stored result",
        primary="does d(phi) = phi(bos) - phi(raw) survive when starts are in-distribution bigrams",
        readings=dict(
            survives="the domain effect is not an OOD artefact; paper 2 stands as drafted",
            collapses="F144-F159 are CONDITIONAL ON OOD INPUT -- must go in the abstract, not the "
                      "limits",
            mixed="report per model; do not summarise as a rate"),
        secondary="phi under text vs random starts on the raw arm -- a large baseline shift means "
                  "the two regimes are not probing the same object",
        why="F66 already found this construction's degeneracy to be an out-of-distribution prompt "
            "artefact, and F159 found the sink account failing specifically on the uniformly random "
            "tokens our census feeds")
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in MODELS:
            want = [f"{m}|{k}|{a}|s{cs}" for k in START_KINDS for a in ARMS for cs in CENSUS_SEEDS]
            if all(k in res["cells"] for k in want) and f"{m}|rung" in res["cells"]:
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
                model = AutoModelForCausalLM.from_pretrained(m).eval().to(
                    dev, torch.float16 if dev != "cpu" else torch.float32)
            except Exception as e:
                res["cells"][f"{m}|failed"] = dict(model=m, error=type(e).__name__)
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED {type(e).__name__}", flush=True)
                continue
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            b = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

            rk = f"{m}|rung"
            if rk not in res["cells"]:
                cs = CENSUS_SEEDS[0]
                theirs = argmax_census(model, tok, dev, pool, np.random.default_rng(cs),
                                       n_starts=N_STARTS)
                mine = census_from_starts(model, tok, dev, random_starts(pool, cs))
                err = abs(theirs["fixed_point_fraction"] - mine["fixed_point_fraction"])
                res["cells"][rk] = dict(model=m, abs_err=float(err),
                                        theirs=theirs["fixed_point_fraction"],
                                        mine=mine["fixed_point_fraction"])
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<34} RUNG theirs={theirs['fixed_point_fraction']:.4f} "
                      f"mine={mine['fixed_point_fraction']:.4f} err={err:.2e}", flush=True)

            for kind in START_KINDS:
                for cs in CENSUS_SEEDS:
                    starts = (random_starts(pool, cs) if kind == "random"
                              else text_starts(tok, cs))
                    for arm in ARMS:
                        k = f"{m}|{kind}|{arm}|s{cs}"
                        if k in res["cells"]:
                            continue
                        target = model if arm == "raw" else _Prefixed(model, [int(b)])
                        c = census_from_starts(target, tok, dev, starts)
                        c.update(cls=classify(c), model=m, start_kind=kind, arm=arm, census_seed=cs)
                        res["cells"][k] = c
                        json.dump(res, open(OUT, "w"), indent=1)
                        print(f"  {m:<34} {kind:<6} {arm:<3} s={cs} cls={c['cls']:<11} "
                              f"phi={c['fixed_point_fraction']:.3f}", flush=True)
            done += 1
            print(f"  {m:<34} done in {time.time()-t0:.0f}s", flush=True)
            del model
            gc.collect()
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
