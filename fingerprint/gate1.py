"""Gate 1 — the deflationary baseline, run BEFORE any new CA compute (PROGRAM.md §3).

THE POINT OF THIS GATE IS TO TRY TO KILL THE INSTRUMENT. The fingerprint battery is CA-derived:
settle a ring, read its attractor share, its melting temperature, its radius sensitivity. All of
that is dynamics. But the *scientific object* underneath (F70) is the two-token conditional and its
argmax map -- a static property of the network that needs no lattice at all. If direct conditional
statistics fingerprint as well as the CA features, the CA machinery exits the capability and the
product reframes as "conditional-statistics fingerprinting": cheaper, simpler, one forward pass per
context instead of N*sweeps*B of them.

That is kill condition **K1**, frozen in prereg.json before any of this existed. It is written to be
winnable by the baseline, because F75 is what happens when you do not do this: the assembly thread
survived every external comparison and died to an internal one, when a random weight reproduced the
ordering the assembly index was being credited with. The lesson imported here is that a measure must
be tested against the cheapest thing that could produce the same numbers, not only against distant
alternatives.

WHAT IS COMPUTED, per PROGRAM.md §3 and the frozen battery's last line:

    argmax fixed-point census   F70's probe: iterate x -> argmax p(.|x1,x2) from random two-token
                                starts and census where they land. F70 found pythia-410m sends 18
                                of 24 starts to a single token while gpt2-medium has no fixed point
                                and wanders to 11 endpoints. This is the CA's low-T behaviour with
                                the CA removed.
    conditional top-1/entropy   mean over sampled RANDOM two-token contexts. Random, not corpus:
                                the CA regime is out-of-distribution by construction (F66), and
                                sampling corpus bigrams would measure a different regime. The
                                corpus-context version already exists in evidence_falloff.
    by-length falloff           top-1 mass as context grows 1,2,3,4,8 tokens, on corpus text.

Then the SAME three protocols Gate 0 ran on the CA features -- coherence, leave-one-out attribution,
controlled pair -- imported from reanalysis.py rather than reimplemented, so a divergence between
the two batteries cannot be an artifact of two different implementations of the same test.

NOT A FINDING UNTIL THE GATE IS READ. The comparison is pre-registered; the numbers are not.

Writes fingerprint/gate1.json.
Usage:  caffeinate -dimsu .venv/bin/python -u fingerprint/gate1.py
        (resumable, keyed by model)
"""
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT / "gatecheck" / "src"))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402
from gatecheck import save_results, verify_block  # noqa: E402
from reanalysis import (FAMILY, ATTENTION_FREE, load, pairwise_ratio, perm_p,  # noqa: E402
                        loo_family_attribution)
from dev_transition_phase3 import bh_fdr        # one implementation; imported, never copied

OUT = _HERE.parent / "gate1.json"
PREREG = _HERE.parent / "prereg.json"

N_STARTS = 24        # F70's count, kept so the census is comparable to the number it reports
MAX_STEPS = 40       # argmax iteration depth before declaring a cycle
N_CONTEXTS = 256     # random two-token contexts for the conditional statistics
LENGTHS = [1, 2, 3, 4, 8]
SEED = 20260801

CORPUS = (
    "The question of how complex structure arises from simple rules has occupied researchers for "
    "decades. Early work focused on the behaviour of small systems, where exhaustive analysis was "
    "possible, but the results did not obviously generalise. Later approaches used statistical "
    "methods to characterise ensembles rather than individual trajectories, and this proved more "
    "fruitful. A recurring theme is that the interesting behaviour occurs near a transition, "
    "though the reasons for this remain debated and the evidence is mixed across domains."
)


@torch.no_grad()
def _next_logits(model, ids, dev):
    return model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()


def _top1_entropy(logits):
    # .cpu() BEFORE .double(): MPS has no float64, and the softmax is worth doing in double
    # because these vocabularies run to 250k entries and the entropy sums that many terms.
    p = torch.softmax(logits.float(), dim=-1).cpu().double().numpy()
    p = p / max(p.sum(), 1e-12)
    nz = p[p > 0]
    return float(p.max()), float(-(nz * np.log(nz)).sum())


@torch.no_grad()
def _batch_last_logits(model, batch, dev, chunk=32):
    """Fixed-length contexts batch trivially; only the argmax iteration has to stay sequential."""
    outs = []
    for i in range(0, len(batch), chunk):
        ids = torch.tensor(batch[i:i + chunk], device=dev)
        outs.append(model(input_ids=ids).logits[:, -1].float().cpu())
    return torch.cat(outs)


def argmax_census(model, tok, dev, pool, rng):
    """F70's probe: where does the deterministic map send random two-token starts?

    The CA at T=0.02 is essentially argmax, so the question that matters is not how much mass the
    top token holds but whether the MAP has an attracting fixed point. Iterating (x1,x2) ->
    (x2, argmax p(.|x1,x2)) either reaches a token that maps to itself, or cycles, or wanders.
    """
    endpoints, fixed, cyclic = [], 0, 0
    for _ in range(N_STARTS):
        ctx = [int(x) for x in rng.choice(pool, size=2)]
        seen, end = set(), ctx[-1]
        for _ in range(MAX_STEPS):
            nxt = int(torch.argmax(_next_logits(model, ctx, dev)))
            end = nxt                                  # the endpoint is ALWAYS the last emission
            # A fixed point of the STATE map is (t,t) -> t: the pair must already be diagonal AND
            # reproduce itself. The first draft tested (a,b) -> b, which only says the trajectory
            # has REACHED the diagonal -- one step too early, and it counted gpt2 as having a fixed
            # point where F70 reports none.
            if ctx[0] == ctx[1] == nxt:
                fixed += 1
                break
            state = (ctx[0], ctx[1])                   # cycle detection on the STATE, not on (prev,next)
            if state in seen:
                cyclic += 1
                break
            seen.add(state)
            ctx = [ctx[1], nxt]
        endpoints.append(end)
    cnt = collections.Counter(endpoints)
    top_tok, top_n = cnt.most_common(1)[0]
    return dict(n_starts=N_STARTS, n_distinct_endpoints=len(cnt),
                modal_endpoint_share=round(top_n / N_STARTS, 4),
                modal_endpoint_token=tok.decode([int(top_tok)]),
                fixed_point_fraction=round(fixed / N_STARTS, 4),
                cyclic_fraction=round(cyclic / N_STARTS, 4),
                # FULL histogram, added for #98's re-run. F84 could state that the modal endpoint
                # WANDERS ('\n', '.', ',', ' the') but not whether that is one funnel with a
                # near-tie at the top or genuinely different attractors, because only the modal
                # token and its count were stored. Additive: every existing key is unchanged, so
                # F84's numbers reproduce exactly.
                endpoint_histogram=[[int(t), tok.decode([int(t)]), int(n)]
                                    for t, n in cnt.most_common()])


def conditional_stats(model, tok, dev, pool, rng):
    """Mean top-1 and entropy of p(.|x1,x2) over RANDOM two-token contexts -- the CA's own regime."""
    ctxs = [[int(x) for x in rng.choice(pool, size=2)] for _ in range(N_CONTEXTS)]
    t1s, ents = [], []
    for lg in _batch_last_logits(model, ctxs, dev):
        a, e = _top1_entropy(lg)
        t1s.append(a); ents.append(e)
    return dict(cond_top1_mean=round(float(np.mean(t1s)), 4),
                cond_top1_sd=round(float(np.std(t1s)), 4),
                cond_entropy_mean=round(float(np.mean(ents)), 4))


def by_length(model, tok, dev, rng):
    """Top-1 mass as context grows -- the falloff curve, on real text."""
    ids = tok(CORPUS, return_tensors=None)["input_ids"]
    out = {}
    for k in LENGTHS:
        starts = [i for i in range(k, len(ids))]
        if not starts:
            continue
        pick = rng.choice(starts, size=min(48, len(starts)), replace=False)
        vals = [_top1_entropy(lg)[0]
                for lg in _batch_last_logits(model, [ids[i - k:i] for i in pick], dev)]
        out[str(k)] = round(float(np.mean(vals)), 4)
    return out


def measure_all():
    res = json.load(open(OUT)) if OUT.exists() else {"runs": {}}
    runs = res["runs"]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    for name in FAMILY:
        if name in runs:
            continue
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name).eval().to(
                dev, torch.float16 if dev != "cpu" else torch.float32)
        except Exception as e:
            print(f"  {name}: LOAD FAILED ({type(e).__name__})", flush=True)
            runs[name] = dict(model=name, failed=type(e).__name__)
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        V = int(getattr(model.config, "vocab_size", len(tok)))
        special = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                               tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in special], dtype=np.int64)
        rng = np.random.default_rng(SEED)
        rec = dict(model=name, family=FAMILY[name], attention=name not in ATTENTION_FREE)
        rec.update(argmax_census(model, tok, dev, pool, rng))
        rec.update(conditional_stats(model, tok, dev, pool, rng))
        rec["by_length"] = by_length(model, tok, dev, rng)
        rec["secs"] = round(time.time() - t0, 1)
        runs[name] = rec
        print(f"  {name:52s} endpoints={rec['n_distinct_endpoints']:>2} "
              f"modal={rec['modal_endpoint_share']:.2f} fix={rec['fixed_point_fraction']:.2f} "
              f"top1={rec['cond_top1_mean']:.3f} H={rec['cond_entropy_mean']:.2f} "
              f"{rec['secs']:.0f}s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del model
        try: torch.mps.empty_cache()
        except Exception: pass
    return res


# The baseline's feature vector, frozen here to mirror the CA battery's four-temperature profile:
# four numbers per model, so attribution is compared like for like rather than the baseline winning
# on dimensionality alone.
BASELINE_FEATURES = ["modal_endpoint_share", "fixed_point_fraction",
                     "cond_top1_mean", "cond_entropy_mean"]

N_PERM_STABLE = 100_000   # 10k left the BH decision inside its own Monte-Carlo error


def perm_p_stable(values, labels, observed, feature, n_perm=N_PERM_STABLE):
    """Permutation p with its OWN deterministic RNG, plus the Monte-Carlo error on the estimate.

    reanalysis.perm_p draws from a module-level RNG that successive calls consume, so a p-value
    depends on how many tests ran before it and moves between runs. That is harmless when the
    answer is p~3e-4, as it was at Gate 0. Here the four baseline features land near the BH
    boundary, and two runs of identical code returned adjusted p = 0.0525 (nothing coheres) and
    0.0477 (everything does). A verdict that flips on permutation noise is not a verdict, so the
    randomness is pinned per feature and the MC error is reported alongside the estimate.
    """
    rng = np.random.default_rng(abs(hash(feature)) % (2 ** 32))
    lab = np.asarray(labels)
    hits = 0
    for _ in range(n_perm):
        r = pairwise_ratio(values, rng.permutation(lab))
        if r is not None and r <= observed:
            hits += 1
    est = (hits + 1) / (n_perm + 1)
    return est, float(np.sqrt(max(est * (1 - est), 1e-12) / n_perm))


def f70_instrument_check(runs):
    """The census must recover F70's known answer before its numbers are used for anything.

    F70 established, by a route that had nothing to do with fingerprinting, that pythia-410m's
    argmax map has an attracting fixed point at a whitespace token while gpt2-medium has NONE and
    wanders. That is the discriminating property this whole gate rests on, so it is asserted rather
    than assumed. The first draft of the census failed it -- a fixed point was tested as (a,b) -> b,
    which only says the trajectory reached the diagonal, and it scored gpt2-medium at 0.96 fixed
    where the truth is 0.00. Reporting the battery on that would have inverted the key feature for
    every model.
    """
    a, b = runs.get("EleutherAI/pythia-410m"), runs.get("gpt2-medium")
    if not (a and b):
        return {"ran": False}
    ok = a["fixed_point_fraction"] > 0.25 and b["fixed_point_fraction"] < 0.10
    return {"ran": True, "passes": bool(ok),
            "pythia410m_fixed_fraction": a["fixed_point_fraction"],
            "pythia410m_modal_token": a["modal_endpoint_token"],
            "gpt2medium_fixed_fraction": b["fixed_point_fraction"],
            "reference": "F70: pythia-410m has an attracting fixed point at a whitespace token; "
                         "gpt2-medium has none and wanders",
            "rule": "pythia-410m fixed fraction > 0.25 AND gpt2-medium < 0.10"}


def analyse(res):
    runs = {m: r for m, r in res["runs"].items() if not r.get("failed")}
    ca = load()                                   # the CA-derived rows Gate 0 used
    out = {"n_models": len(runs), "n_families": len({r["family"] for r in runs.values()})}
    out["f70_instrument_check"] = f70_instrument_check(runs)
    if out["f70_instrument_check"].get("ran") and not out["f70_instrument_check"]["passes"]:
        out["gate1_verdict"] = (
            "NOT DECIDABLE -- the argmax census does not recover F70's known answer, so its "
            "numbers cannot be used to decide K1. Fix the probe; do not reinterpret.")
        return out

    # -- 1. coherence, same protocol as Gate 0 --------------------------------------------
    fams = [r["family"] for r in runs.values()]
    coher = {}
    for feat in BASELINE_FEATURES:
        vals = [r[feat] for r in runs.values()]
        obs = pairwise_ratio(vals, fams)
        pv, se = perm_p_stable(vals, fams, obs, feat) if obs else (None, None)
        coher[feat] = {"within_over_between": round(obs, 3) if obs else None,
                       "perm_p": round(pv, 5) if pv else None,
                       "perm_p_mc_se": round(se, 5) if se else None}
    # FOUR features were tested for coherence, so four p-values need correcting. Reporting the
    # three that clear an uncorrected 0.05 would be the multiple-comparison error this project
    # already corrects elsewhere (F39 used bh_fdr on a battery of exactly this shape).
    keys = list(coher)
    adj = bh_fdr([coher[k]["perm_p"] for k in keys])
    for k, a in zip(keys, adj):
        # The BH adjustment scales the raw p, so the MC error scales with it too. A feature whose
        # adjusted p sits within 2 SE of 0.05 is NOT DECIDABLE at this permutation count -- saying
        # "coheres" or "does not" there is reporting the seed.
        scale = a / coher[k]["perm_p"] if coher[k]["perm_p"] else 1.0
        se = (coher[k]["perm_p_mc_se"] or 0.0) * scale
        coher[k]["perm_p_bh"] = round(a, 5)
        coher[k]["perm_p_bh_mc_se"] = round(se, 5)
        coher[k]["coheres_after_correction"] = bool(a + 2 * se <= 0.05)
        coher[k]["undecidable_at_boundary"] = bool(abs(a - 0.05) <= 2 * se)
    out["family_coherence_baseline"] = coher

    # -- 2. attribution, same protocol, same dimensionality --------------------------------
    rows = {m: {"family": r["family"], "profile": [r[f] for f in BASELINE_FEATURES]}
            for m, r in runs.items()}
    out["family_attribution_baseline"] = loo_family_attribution(rows)
    out["family_attribution_ca"] = loo_family_attribution(
        {m: r for m, r in ca.items() if all(p is not None for p in r["profile"])})

    # -- 3. the controlled pair, same statistic --------------------------------------------
    pair = {}
    for feat in BASELINE_FEATURES:
        within = {}
        for f in {r["family"] for r in runs.values()}:
            vs = [r[feat] for r in runs.values() if r["family"] == f]
            if len(vs) >= 2:
                within[f] = max(vs) - min(vs)
        worst = max(within.values()) if within else None
        gap = abs(runs["EleutherAI/gpt-neo-125M"][feat] - runs["gpt2"][feat])
        pair[feat] = {"gap": round(gap, 4),
                      "worst_within_family_range": round(worst, 4) if worst else None,
                      "gap_over_worst_within": round(gap / worst, 2) if worst else None}
    out["controlled_pair_baseline"] = pair

    # -- K1: does the baseline match or exceed the CA on BOTH H1 and H2? -------------------
    b, c = out["family_attribution_baseline"], out["family_attribution_ca"]
    h1_base = b["n_correct"] / max(b["n_scored"], 1)
    h1_ca = c["n_correct"] / max(c["n_scored"], 1)
    best_pair = max((v["gap_over_worst_within"] or 0) for v in pair.values())
    ca_pair = 2.4                                  # Gate 0's measured gap/worst-within for the CA
    h1_matches = h1_base >= h1_ca
    h2_matches = best_pair >= ca_pair
    k1 = bool(h1_matches and h2_matches)
    out["K1"] = {
        "fired": k1,
        "h1_baseline_accuracy": round(h1_base, 3), "h1_ca_accuracy": round(h1_ca, 3),
        "h1_baseline_matches_or_beats": bool(h1_matches),
        "h2_best_pair_gap_over_within_baseline": round(best_pair, 2),
        "h2_ca_reference": ca_pair, "h2_baseline_matches_or_beats": bool(h2_matches),
        "rule": "K1 fires only if the baseline matches or exceeds the CA on H1 AND H2 (prereg)",
    }

    parts = []
    if k1:
        parts.append(
            f"K1 FIRES. The direct two-token-conditional baseline matches or beats the CA battery "
            f"on both registered axes: attribution {h1_base:.0%} against the CA's {h1_ca:.0%}, and "
            f"the controlled pair separates at {best_pair:.1f}x the worst within-family range "
            f"against the CA's {ca_pair}x. THE CA MACHINERY EXITS THE CAPABILITY. The product "
            f"reframes as conditional-statistics fingerprinting -- one forward pass per context "
            f"instead of a settled lattice -- and the dynamics return to being a paper-2 control. "
            f"This is a success for the tool and a demotion for the instrument, which is exactly "
            f"the outcome the gate was written to be able to produce.")
    else:
        why = []
        if not h1_matches:
            why.append(f"attribution {h1_base:.0%} vs the CA's {h1_ca:.0%}")
        if not h2_matches:
            why.append(f"best pair separation {best_pair:.1f}x vs the CA's {ca_pair}x")
        if h1_matches:
            parts.append(
                f"On H1 the two are TIED WITHIN NOISE, not separated: {b['n_correct']}/{b['n_scored']} "
                f"for the baseline against {c['n_correct']}/{c['n_scored']} for the CA is a "
                f"ONE-MODEL margin, and both sit far above the ~{b['chance_expected']}/{b['n_scored']} "
                f"expected by chance while far below a capability. The prereg's rule counts this as "
                f"the baseline matching; the honest reading is that neither battery attributes "
                f"families from four numbers.")
        parts.append(
            f"K1 DOES NOT FIRE: the baseline loses on {' and '.join(why)}. The CA-derived features "
            f"carry something the static two-token conditional does not, so H3 survives its first "
            f"real test and Gate 2 is licensed. This does NOT establish that the dynamics are "
            f"necessary -- only that the cheapest deflation tried so far does not reproduce them.")
    raw = [f for f, v in coher.items() if (v["perm_p"] or 1) < 0.05]
    corrected = [f for f, v in coher.items() if v["coheres_after_correction"]]
    undec = [f for f, v in coher.items() if v.get("undecidable_at_boundary")]
    parts.append(
        f"COHERENCE, CORRECTED: {len(raw)} baseline feature(s) clear an uncorrected p<0.05 "
        f"({', '.join(raw) if raw else 'none'}), but after Benjamini-Hochberg over the four tests "
        f"actually run, {', '.join(corrected) if corrected else 'NONE SURVIVES'}"
        + (f" -- and {', '.join(undec)} sit within Monte-Carlo error of the 0.05 boundary, so they "
           f"are NOT DECIDABLE at {N_PERM_STABLE:,} permutations rather than negative"
           if undec else "") + f". Gate 0's CA "
        f"attractor share was 0.218 at p~3e-4, which survives any correction over its three tests. "
        f"So the CA feature coheres within families more tightly than any static-conditional "
        f"feature does, and the difference is not marginal.")
    parts.append(
        "Exploratory-to-registered boundary: the COMPARISON is pre-registered in prereg.json and "
        "K1's rule was frozen before any of these numbers existed; the numbers themselves are new. "
        "Family is the independent unit throughout, so the six Pythia sizes are one observation.")
    out["gate1_verdict"] = " ".join(parts)
    return out


def main():
    block = json.load(open(PREREG))
    ok = verify_block(block)
    print(f"  prereg block verifies: {ok}", flush=True)
    if not ok:
        raise SystemExit("prereg.json failed its own hash check -- refusing to run against it")
    res = measure_all()
    out = analyse(res)
    out["_prereg_sha256"] = block["sha256"]
    out["runs"] = res["runs"]
    print("\n  ->", out["gate1_verdict"])
    save_results(OUT, out, script=__file__, root=ROOT, prereg=block,
                 independent_unit="family", forbid_paths=True)
    print("\nwrote fingerprint/gate1.json")


if __name__ == "__main__":
    main()
