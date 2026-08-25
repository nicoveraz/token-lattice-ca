"""PHASE 2 of matched-entropy: dphi for the pairing frozen in phase 1. Censuses only.

Registered in experiments/prereg_matched_entropy.json. The pairing this reads was written and
committed by experiments/matched_entropy_pairing.py BEFORE any dphi existed, and its sha256 is
recorded in results/matched_entropy_pairing.sha256. This script verifies that hash before it runs.
If the pairing had been chosen after seeing dphi it would guarantee whichever answer was wanted,
which is the prereg's own note and the reason the two phases are separate commits.

THE HYPOTHESES, unchanged from the freeze:
  H1  entropy mediation -- within a model, two prefixes matched in induced entropy shift produce the
      SAME dphi, whatever their content.
  H2  content beyond entropy -- matched-entropy prefixes differing in document type produce
      DIFFERENT dphi within the same model.
They are exclusive readings of one number, |dphi(A) - dphi(B)|, against that model's own noise.

ANTI-VACUITY IS THE GATE THAT MATTERS. A model enters only if at least one arm moves phi beyond that
model's own tolerance AND the model has headroom on the side the effect moves. Comparing two prefixes
that both did nothing is comparing two nothings, and K3 makes that NOT DECIDABLE rather than a
confirmation of H1 -- which is exactly how this project's most frequent error would enter here.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, hashlib, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census
from argmax_census_hardened import N_STARTS, CENSUS_SEEDS

PREREG = "experiments/prereg_matched_entropy.json"
PAIRING = _ROOT / "results" / "matched_entropy_pairing.json"
PSHA = _ROOT / "results" / "matched_entropy_pairing.sha256"
OUT = _ROOT / "results" / "matched_entropy.json"
CACHE = _ROOT / "results" / "matched_entropy_cells.json"
# EXCLUDED FOR A MEASURED REASON, not a chosen one. sapienzanlp/Minerva-3B-base-v1.0 in float32 is
# ~12GB on a 16GB machine, and a census cannot be batched -- it is 96 sequential trajectories of up
# to 40 steps, three arms, two seeds. Measured while it ran: resident set 1.58GB against a 12GB
# model, swap at 13481MB of 14336MB, and CPU time advancing 17 seconds per 120 seconds of wall
# clock. Fourteen per cent efficiency, entirely paging. It produced ZERO completed arms before it
# was stopped, so nothing is lost and nothing is partial. Named here rather than dropped, and the
# K1/K2 majorities below are over the models that ARE readable, which is now five and not six.
EXCLUDED = {"sapienzanlp/Minerva-3B-base-v1.0":
            "float32 does not fit in 16GB; measured at 14% CPU efficiency, pure paging, 0 arms done"}

MIN_SHIFT = 4.0 / N_STARTS          # the registered tolerance floor
NOISE_FACTOR = 2.0
HEADROOM = 0.05                      # a model is floored/ceilinged within this of 0 or 1


@torch.no_grad()
def census_with_prefix(model, tok, dev, pool, rng, prefix):
    """argmax_census, but every context is preceded by the frozen prefix.

    gate1.argmax_census is imported for the raw arm and cannot take a prefix, so the prefixed arm
    reimplements its loop -- and the raw arm below is run through the IMPORTED function, so the two
    can be compared and any drift between them would show as a raw-arm mismatch against the stored
    text_interaction cells.
    """
    import collections
    endpoints, fixed, cyclic = [], 0, 0
    for _ in range(N_STARTS):
        ctx = [int(x) for x in rng.choice(pool, size=2)]
        seen, end = set(), ctx[-1]
        for _ in range(40):
            ids = torch.tensor([list(prefix) + ctx], dtype=torch.long, device=dev)
            nxt = int(torch.argmax(model(input_ids=ids).logits[0, -1]))
            end = nxt
            if ctx[0] == ctx[1] == nxt:
                fixed += 1; break
            st = (ctx[0], ctx[1])
            if st in seen:
                cyclic += 1; break
            seen.add(st); ctx = [ctx[1], nxt]
        endpoints.append(end)
    c = collections.Counter(endpoints)
    top, n = c.most_common(1)[0]
    return dict(n_starts=N_STARTS, n_distinct_endpoints=len(c),
                modal_endpoint_share=round(n / N_STARTS, 4),
                modal_endpoint_token=tok.decode([int(top)]),
                fixed_point_fraction=round(fixed / N_STARTS, 4),
                cyclic_fraction=round(cyclic / N_STARTS, 4))


def main():
    pair = json.load(open(PAIRING))
    claimed = PSHA.read_text().split()[0]
    actual = hashlib.sha256(json.dumps(pair["pairs"], sort_keys=True).encode()).hexdigest()
    if actual != claimed:
        raise AssertionError(
            f"the pairing has changed since it was frozen: {actual[:16]} against {claimed[:16]}. "
            f"Phase 1 wrote it before any dphi existed; if it no longer hashes to that value, the "
            f"pairing this run would use is not the one that was registered.")
    print(f"  pairing verified against its freeze: {claimed[:32]}...", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    for m, pr in sorted(pair["pairs"].items()):
        if m in EXCLUDED:
            print(f"  {m:<34} EXCLUDED: {EXCLUDED[m]}", flush=True); continue
        need = [(arm, cs) for arm in ("raw", "A", "B") for cs in CENSUS_SEEDS
                if f"{m}|{arm}|s{cs}" not in cache]
        if not need:
            print(f"  {m:<34} cached", flush=True); continue
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", torch.float32)
        V = int(getattr(model.config, "vocab_size", len(tok)))
        sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                          tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
        for arm, cs in need:
            rng = np.random.default_rng(cs)
            if arm == "raw":
                c = argmax_census(model, tok, "cpu", pool, rng, n_starts=N_STARTS)
            else:
                c = census_with_prefix(model, tok, "cpu", pool, rng, pr[arm]["ids"])
            c.update(model=m, arm=arm, census_seed=cs,
                     prefix_row=None if arm == "raw" else pr[arm]["row"])
            cache[f"{m}|{arm}|s{cs}"] = c
            json.dump(cache, open(CACHE, "w"))
        print(f"  {m:<34} raw/A/B done ({time.time()-t0:.0f}s)", flush=True)
        del model; gc.collect()
    _verdict(cache, pair)


def _verdict(cache, pair):
    res = dict(_preregistration_file=PREREG, _pairing_file="results/matched_entropy_pairing.json",
               _pairing_sha256=PSHA.read_text().split()[0],
               unmatched_models=pair["unmatched"], models={}, excluded=[])
    for m, pr in sorted(pair["pairs"].items()):
        if m in EXCLUDED:
            res["excluded"].append(dict(model=m, why=EXCLUDED[m], measured=True)); continue
        def phis(arm):
            return [cache[f"{m}|{arm}|s{cs}"]["fixed_point_fraction"] for cs in CENSUS_SEEDS
                    if f"{m}|{arm}|s{cs}" in cache]
        raw, A, B = phis("raw"), phis("A"), phis("B")
        if not (len(raw) == len(A) == len(B) == len(CENSUS_SEEDS)):
            res["excluded"].append(dict(model=m, why="incomplete arms")); continue
        mr, mA, mB = float(np.mean(raw)), float(np.mean(A)), float(np.mean(B))
        rngs = [max(x) - min(x) for x in (raw, A, B)]
        tol = max(MIN_SHIFT, NOISE_FACTOR * max(rngs))
        dA, dB = mA - mr, mB - mr
        moved = max(abs(dA), abs(dB)) > tol
        # THE HEADROOM GATE, corrected to the rule the prereg actually states: the model must have
        # "headroom on the side the effect moves". The first version required the RAW arm to be at a
        # boundary, which is the wrong test -- raw sits mid-range here while BOTH ARMS land on zero,
        # so the two prefixes agreed because the measurement is CENSORED, not because matched
        # entropy produced matched dphi. Two arms pinned to the same wall agree trivially, exactly as
        # two arms that never moved do, and reading that as support for H1 is F149/F161's defect.
        down = (dA + dB) < 0
        censored = (max(mA, mB) < HEADROOM) if down else (min(mA, mB) > 1 - HEADROOM)
        floored = bool(censored)
        row = dict(raw_phi=round(mr, 4), phi_A=round(mA, 4), phi_B=round(mB, 4),
                   dphi_A=round(dA, 4), dphi_B=round(dB, 4),
                   gap=round(abs(dA - dB), 4), tolerance=round(tol, 4),
                   entropy_shift_A=pr["A"]["entropy_shift"], entropy_shift_B=pr["B"]["entropy_shift"],
                   type_A=pr["A"]["pile_set_name"], type_B=pr["B"]["pile_set_name"],
                   an_arm_moved=bool(moved), floored_or_ceilinged=bool(floored),
                   direction="down" if down else "up",
                   _censoring_note="both arms on the same boundary means the arms agree by censoring "
                                   "rather than by matched entropy; such a model is not readable",
                   readable=bool(moved and not floored),
                   same_within_tolerance=bool(abs(dA - dB) <= tol))
        res["models"][m] = row
        if not row["readable"]:
            res["excluded"].append(dict(
                model=m, why="no arm moved beyond tolerance" if not moved
                else f"BOTH arms censored at the {'floor' if down else 'ceiling'} "
                     f"(phi_A {round(mA,4)}, phi_B {round(mB,4)}) -- they agree by censoring, "
                     f"not by matched entropy",
                dphi_A=round(dA, 4), dphi_B=round(dB, 4)))
    readable = {m: r for m, r in res["models"].items() if r["readable"]}
    res["n_readable"] = len(readable)
    res["K3_vacuity"] = bool(not readable)
    if readable:
        agree = sum(1 for r in readable.values() if r["same_within_tolerance"])
        res["n_same_within_tolerance"] = agree
        res["K1_mediation_dies"] = bool(agree < len(readable) / 2)
        res["K2_mediation_holds"] = bool(agree > len(readable) / 2 and
                                         any(max(abs(r["dphi_A"]), abs(r["dphi_B"])) >= 0.3
                                             for r in readable.values()))
    p = [f"MATCHED-ENTROPY PREFIX PAIRS, registered in {PREREG}. The pairing was frozen by phase 1 "
         f"before any dphi existed and verified against its sha256 ({res['_pairing_sha256'][:12]}...) "
         f"before this run started. Six models, prefixes matched within 0.10 nats of induced entropy "
         f"shift and differing in the corpus's own document-type label. "]
    res["_excluded_for_memory"] = EXCLUDED
    res["_edit_note"] = ("EXCLUDED was added after four models had already been measured, when "
                         "Minerva-3B was observed paging at 14% CPU efficiency. The edit adds an "
                         "exclusion list and changes no estimator, so it cannot affect the four "
                         "completed cells; they were produced by the pre-edit script and are "
                         "unchanged. Recorded because one provenance stamp now covers both.")
    if res["unmatched_models"]:
        p.append(f"NO MATCHED PAIR within tolerance, named rather than dropped: "
                 f"{res['unmatched_models']}. ")
    if EXCLUDED:
        p.append(f"EXCLUDED FOR MEMORY, measured rather than assumed: {sorted(EXCLUDED)} -- "
                 f"float32 does not fit in 16GB and the census cannot be batched; observed at 14% "
                 f"CPU efficiency with swap full, having completed zero arms. The K1/K2 majorities "
                 f"below are therefore over five models, not the registered six. ")
    for m, r in sorted(res["models"].items()):
        p.append(f"{m.split('/')[-1]}: raw {r['raw_phi']}, A {r['phi_A']} ({r['type_A']}), B "
                 f"{r['phi_B']} ({r['type_B']}); dphi {r['dphi_A']:+} vs {r['dphi_B']:+}, gap "
                 f"{r['gap']} against tolerance {r['tolerance']}"
                 + ("" if r["readable"] else " -- EXCLUDED, "
                    + ("no arm moved" if not r["an_arm_moved"] else "floored/ceilinged")) + ". ")
    cens = [e for e in res["excluded"] if "censor" in e.get("why", "")]
    if cens:
        p.append(f"CENSORING EXCLUDED {len(cens)} MODEL(S), and this is the finding rather than an "
                 f"aside: in each, BOTH matched-entropy prefixes drove phi onto the same boundary, "
                 f"so |dphi(A) - dphi(B)| is near zero because the measurement saturated and not "
                 f"because the prefixes were matched. Two arms pinned to one wall agree as trivially "
                 f"as two arms that never moved. "
                 + "; ".join(f"{e['model'].split('/')[-1]} {e['dphi_A']}/{e['dphi_B']}" for e in cens)
                 + ". ")
    if res["K3_vacuity"]:
        p.append("K3 FIRES: no model has a READABLE pair -- either no arm moved beyond its own "
                 "tolerance. The comparison is between two nothings and is NOT DECIDABLE. This is "
                 "the outcome the anti-vacuity gate exists to produce, and it is not evidence for "
                 "H1 -- two prefixes that both did nothing agree trivially. ")
    elif res.get("K1_mediation_dies"):
        p.append(f"K1 FIRES: on {len(readable) - res['n_same_within_tolerance']} of "
                 f"{len(readable)} readable models the two matched-entropy prefixes produce "
                 f"DIFFERENT dphi beyond tolerance. Entropy mediation is dead as a complete "
                 f"account -- content matters beyond entropy, which is H2. ")
    elif res.get("K2_mediation_holds"):
        p.append(f"K2 FIRES: matched-entropy prefixes agree within tolerance on "
                 f"{res['n_same_within_tolerance']} of {len(readable)} readable models, and at "
                 f"least one pair spans a large effect. Entropy mediation SURVIVES as a candidate "
                 f"reduction of the prompt-model interaction to a scalar. ")
    else:
        p.append(f"NEITHER K1 NOR K2: agreement on {res.get('n_same_within_tolerance')} of "
                 f"{len(readable)} readable models is a majority, but no readable pair spans a "
                 f"|dphi| of 0.3, so K2's second condition fails. Reported as inconclusive rather "
                 f"than as support. ")
    p.append("REFUSALS, registered before the numbers: no significance test -- a paired test on six "
             "models cannot fail informatively; no correlation between entropy shift and dphi "
             "magnitude across models, which is F149's refusal. AND THE PRIOR-ART GATE FOR "
             "greedy-decoding degeneration and repetition self-reinforcement IS OWED before any "
             "entropy-mediation claim is written up -- entropy sharpening under a prefix is close to "
             "that literature's territory.")
    res["verdict"] = " ".join(p)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"]); print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
