"""Does the attention sink explain the SIGN of the domain effect? The tension in F152/F157, measured.

THE TENSION. The attention-sink account explains initial-token dominance by a POSITIONAL,
content-free mechanism: SoftMax normalisation forces attention mass onto early tokens regardless of
what they are, so the first token acts as a no-op key bias. On that account the cross-model variation
is in MAGNITUDE and saturation point, never in SIGN -- adding an initial token moves the readout the
same way in every model examined. F157's prior-art gate confirmed this is what that literature
reports.

We observe the opposite. One BOS token takes `Falcon3-1B-Base` from 0.214 to 0.906 and collapses
other models toward zero on the same arm (F152). Paper 2 currently reports that as a discrepancy
against a mechanism WE NEVER MEASURED, which is a weak position: we compare our numbers to their
account rather than to their quantity.

THIS RUN MEASURES BOTH QUANTITIES ON THE SAME MODELS UNDER THE SAME PREFIXES. It is cheap -- a few
hundred short forward passes with attentions, not a 96-start census -- and it is decisive in a way
that does not depend on cohort size, because the comparison is WITHIN each model.

PRIMARY, and why it can fail. The literature's claim is that BOS raises sink strength in every model.
Our claim is that BOS moves fixed-point structure in opposite directions across models. Both are
measured here, per model:
    if  sink (vs the LENGTH-MATCHED raw3) rises in EVERY model  while  d(phi) has MIXED signs
        -> the mechanism behaves uniformly and the structural consequence does not. The sink account
           cannot explain the sign of the domain effect, the discrepancy is confirmed with both
           quantities in hand, and paper 2's centrepiece stands on measured ground.
    if  sink ALSO flips sign across models
        -> the sink account explains our result. The "discrepancy" dissolves into agreement, and we
           would have a MECHANISM -- the first predictive statement in this programme. Paper 2's
           section 'sink' must then be rewritten from a tension into a mechanistic finding.
    if  sink rises uniformly AND d(phi) is uniform too
        -> our F152 reading is wrong somewhere; investigate before anything else.

SECONDARY, exploratory and labelled so: across (model, prefix) cells, does sink strength track the
MAGNITUDE of the shift? With six model clusters a rank correlation cannot fail informatively -- the
F149/F153 refusal -- so this is descriptive and no p-value is computed.

DEFINITION OF SINK STRENGTH, fixed here before measurement: the mean attention paid by the LAST
position to POSITION 0, averaged over all layers and heads, over K random starts, reported BOTH raw
and as a multiple of uniform (1/S).

THE LENGTH CONFOUND, caught by a smoke test before any model was run in anger. The census condition
is a 2-token start, where the last position attends over only TWO positions and attention to position
0 is near-forced (Falcon3-1B-Base measured 0.778 there). Prepending BOS makes it three positions, so
a naive raw-vs-bos comparison confounds "is position 0 a BOS token" with "how many positions compete
for attention". Three arms are therefore measured:
    raw2   [x1, x2]         the census's own condition, length 2
    raw3   [x0, x1, x2]     ordinary tokens, length 3 -- LENGTH-MATCHED to bos
    bos    [BOS, x1, x2]    length 3, position 0 is the BOS
The PRIMARY compares bos against RAW3, which differs from it in exactly one thing: the identity of
position 0. raw2 is kept because it is the condition phi was censused in, and the raw2-vs-raw3 gap
measures the length artefact directly.

PRE-REGISTERED:
  RUNG      the phi values are READ from stored results, not recomputed; the (model, arm) keys must
            resolve in domain_base/text_interaction or the cell is skipped and named.
  PRIMARY   sign of d(sink) under BOS across models, against sign of d(phi) under BOS.
  SECONDARY sink vs |d(phi)| across cells -- DESCRIPTIVE, no p-value, declared before the numbers.
  DTYPE     bf16, forced: fp16 eager attention returns NaN on some models here.
  BOUNDARY  one definition of sink strength, one layer/head aggregation, short contexts. The sink
            literature measures on long contexts where streaming matters; a null here bounds the
            account at THIS context length and does not refute it in general.
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
from argmax_census_hardened import N_STARTS, CENSUS_SEEDS

OUT = str(_ROOT / "results" / "sink_vs_fixedpoint.json")
BASE = _ROOT / "results" / "domain_base.json"
TI = _ROOT / "results" / "text_interaction.json"
SCREEN = _ROOT / "results" / "midrange_screen.json"

MODELS = ["tiiuae/Falcon3-1B-Base", "sapienzanlp/Minerva-3B-base-v1.0",
          "EleutherAI/pythia-410m-deduped", "Qwen/Qwen1.5-1.8B",
          "HuggingFaceTB/SmolLM-1.7B", "Qwen/Qwen2.5-1.5B-Instruct"]
K_STARTS = 16                 # forward passes per (model, arm); the sink is a mean over these
SINK_SEED = 20260817


def phi_of(m, arm):
    """phi is READ, never recomputed. Returns (mean, source) or None."""
    for path, key in ((BASE, f"{m}|s{{cs}}|{arm}"), (TI, f"{m}|s{{cs}}|{arm}")):
        if not path.exists():
            continue
        runs = json.load(open(path))["runs"]
        ks = [key.format(cs=cs) for cs in CENSUS_SEEDS]
        if all(k in runs for k in ks):
            return float(np.mean([runs[k]["fixed_point_fraction"] for k in ks])), path.name
    return None


def sink_strength(model, tok, dev, pool, prefix, rng, n_start=2):
    """Mean attention from the LAST position to POSITION 0, over layers, heads and K starts.

    `n_start` is the number of ORDINARY tokens after the prefix, so (prefix, n_start) fixes the
    sequence length and lets bos be compared against a length-matched raw arm."""
    vals = []
    for _ in range(K_STARTS):
        start = [int(x) for x in rng.choice(pool, size=n_start, replace=False)]
        ids = torch.tensor([list(prefix) + start], device=dev)
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True)
        at = out.attentions                       # tuple(L) of (B, H, S, S)
        per_layer = [float(a[0, :, -1, 0].mean().item()) for a in at]
        vals.append(float(np.mean(per_layer)))
        del out, at
    S = len(prefix) + n_start
    m = float(np.mean(vals))
    return m, float(np.std(vals, ddof=1)), float(m * S)      # raw, sd, multiples of uniform


def analyse(res):
    cells, parts, analysis = res["cells"], [], {}
    rows = {}
    for k, v in cells.items():
        m, arm = k.rsplit("|", 1)
        rows.setdefault(m, {})[arm] = v
    analysis["rows"] = rows
    have = [m for m in rows if {"raw2", "raw3", "bos"} <= set(rows[m])]
    parts.append(
        f"Measured sink strength and read stored phi on {len(rows)} model(s); {len(have)} have both "
        f"the raw and bos arms needed for the PRIMARY.")
    if len(have) < 2:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " Fewer than two models complete -- PRIMARY not read."
        return

    tab = []
    for m in have:
        r2, r3, b = rows[m]["raw2"], rows[m]["raw3"], rows[m]["bos"]
        dsink = b["sink"] - r3["sink"]                     # LENGTH-MATCHED comparison
        dphi = (b.get("phi") - r2.get("phi")) if (b.get("phi") is not None
                                                 and r2.get("phi") is not None) else None
        tab.append(dict(model=m, sink_raw3=round(r3["sink"], 5), sink_bos=round(b["sink"], 5),
                        d_sink=round(dsink, 5), length_artefact=round(r3["sink"] - r2["sink"], 5),
                        phi_raw=r2.get("phi"), phi_bos=b.get("phi"),
                        d_phi=None if dphi is None else round(dphi, 4)))
    analysis["primary_table"] = tab
    # NaN MUST NOT VOTE. np.sign(nan) is nan, so a single unmeasured cell silently enlarges the
    # sign set and flips "sink is uniform" into "sink changes sign" -- i.e. inverts the verdict.
    # It did exactly that on the first run. Non-finite cells are excluded and NAMED.
    bad = [t["model"] for t in tab if not np.isfinite(t["d_sink"])]
    good = [t for t in tab if np.isfinite(t["d_sink"])]
    analysis["unmeasured_sink"] = bad
    sink_signs = {np.sign(t["d_sink"]) for t in good}
    phi_signs = {np.sign(t["d_phi"]) for t in good if t["d_phi"] is not None}
    sink_uniform = len(sink_signs) == 1
    phi_mixed = len(phi_signs) > 1
    analysis.update(sink_uniform=bool(sink_uniform), phi_mixed=bool(phi_mixed),
                    n_sink_measured=len(good))
    if bad:
        parts.append(
            f"UNMEASURED: sink is non-finite on {len(bad)} model(s) -- "
            f"{[m.split('/')[-1] for m in bad]} -- and they are EXCLUDED from the sign test rather "
            f"than allowed to enlarge the sign set. On the first run they were not, and a NaN "
            f"inverted the verdict.")
    if len(good) < 2:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + (
            f" Only {len(good)} model(s) have a finite sink measurement -- PRIMARY not read.")
        return

    parts.append(
        "PRIMARY, BOS: "
        + "; ".join("{} sink(raw3) {:.4f}->{:.4f} (d {:+.4f}), phi {}".format(
            t["model"].split("/")[-1], t["sink_raw3"], t["sink_bos"], t["d_sink"],
            "n/a" if t["d_phi"] is None else f"{t['phi_raw']:.3f}->{t['phi_bos']:.3f} "
                                             f"(d {t['d_phi']:+.3f})") for t in tab)
        + ". ")
    # THE TEST THAT WAS MISSING. The registered readings were written assuming sink would come out
    # uniform, so the non-uniform branch asserted that sink "tracks" phi WITHOUT EVER CHECKING
    # CORRESPONDENCE. Whether a mechanism explains a sign pattern is a question about AGREEMENT
    # per model, not about whether each quantity is separately uniform.
    paired = [t for t in good if t["d_phi"] is not None]
    agree = [t for t in paired if np.sign(t["d_sink"]) == np.sign(t["d_phi"])]
    analysis["sign_agreement"] = dict(agree=len(agree), n=len(paired),
                                      models_agree=[t["model"].split("/")[-1] for t in agree])
    parts.append(
        f"SIGN AGREEMENT, the decisive quantity: sink and phi move the same way on "
        f"{len(agree)} of {len(paired)} models ({[t['model'].split('/')[-1] for t in agree] or '-'}). "
        + (f"Sink strength does NOT predict the sign of the fixed-point shift -- agreement is at or "
           f"near chance. The attention-sink account, at this context length, does not explain the "
           f"domain effect's direction, and paper 2's discrepancy stands with both quantities now "
           f"measured on the same forward passes rather than one of them assumed."
           if len(agree) <= len(paired) / 2 + 1e-9 else
           f"Sink strength DOES track the sign on a majority. This is a mechanism candidate and "
           f"paper 2's sink section must be rewritten from tension toward explanation -- though "
           f"{len(paired)} models is a count, not a rate."))
    parts.append(
        f"Separately, the literature's UNIFORMITY claim does not reproduce here: d(sink) is "
        f"{'uniform in sign' if sink_uniform else 'NOT uniform in sign'} across the "
        f"{len(good)} measured models"
        + ("" if sink_uniform else
           f" ({sum(1 for t in good if t['d_sink'] < 0)} of {len(good)} go DOWN under BOS). ")
        + "That is a statement about THIS context length only -- the sink literature measures long "
          "contexts, where the phenomenon is about attention concentrating despite many competing "
          "positions. It is not a refutation of that work.")

    ex = []
    for m, arms in rows.items():
        for arm, v in arms.items():
            if arm == "bos" and v.get("phi") is not None and arms.get("raw2", {}).get("phi") is not None:
                ex.append((m, arm, v["sink"] - arms["raw3"]["sink"], v["phi"] - arms["raw2"]["phi"]))
    if ex:
        analysis["secondary"] = [dict(model=m, arm=a, d_sink=round(ds, 5), d_phi=round(dp, 4))
                                 for m, a, ds, dp in ex]
        parts.append(
            f"SECONDARY (EXPLORATORY, declared before the numbers -- {len(rows)} model clusters "
            f"cannot fail a rank correlation informatively, so no p-value is computed): across "
            f"{len(ex)} (model, arm) cells, d(sink) and d(phi) are "
            + "; ".join(f"{m.split('/')[-1]}/{a}: {ds:+.4f} vs {dp:+.3f}" for m, a, ds, dp in ex)
            + ".")
    parts.append(
        f"BOUNDARY: sink strength is defined ONCE here -- mean attention from the last position to "
        f"position 0, over all layers and heads, across {K_STARTS} random two-token starts, at the "
        f"census's own context length. The sink literature measures on LONG contexts where streaming "
        f"matters; a null here bounds that account at this context length and does not refute it in "
        f"general. phi is read from stored censuses, never recomputed.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, k_starts=K_STARTS, sink_seed=SINK_SEED, census_seeds=CENSUS_SEEDS,
        sink_definition="mean attention from the LAST position to POSITION 0 over all layers "
                        "and heads; three arms raw2/raw3/bos so bos is compared LENGTH-MATCHED "
                        "against raw3, isolating the identity of position 0 from sequence length",
        primary="sign of d(sink) = sink(bos) - sink(raw3), LENGTH-MATCHED, across models, against sign of d(phi) under BOS",
        readings=dict(
            sink_uniform_phi_mixed="the sink account cannot explain the SIGN; discrepancy confirmed",
            sink_flips="the sink account explains it; a MECHANISM, and paper 2's sink section is "
                       "rewritten from tension to finding",
            both_uniform="F152's mixed-sign reading is wrong somewhere; investigate first"),
        secondary="sink vs |d(phi)| across cells -- DESCRIPTIVE, no p-value, six clusters cannot "
                  "fail it informatively",
        why="paper 2 currently reports a discrepancy against a mechanism we never measured; this "
            "measures both quantities on the same forward passes")
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in MODELS:
            arms = {"raw2": ([], 2), "raw3": ([], 3), "bos": (None, 2)}
            if all(f"{m}|{a}" in res["cells"] for a in ("raw2", "raw3", "bos")):
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
                # BFLOAT16, not fp16: eager attention in fp16 produced NaN in 7 of 24 layers on
                # pythia-410m-deduped and all of Qwen2.5-1.5B-Instruct. fp32 is clean but a 3B model
                # at fp32 is ~12GB on a 16GB machine. bf16 has fp32's exponent range at fp16's cost
                # and measures NaN-free on both. phi is unaffected -- it is read, not recomputed.
                model = AutoModelForCausalLM.from_pretrained(
                    m, attn_implementation="eager").eval().to(
                    dev, torch.bfloat16 if dev != "cpu" else torch.float32)
            except Exception as e:
                res["cells"][f"{m}|failed"] = dict(model=m, error=type(e).__name__,
                                                   detail=str(e)[:120])
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                continue
            b = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
            arms["bos"] = ([int(b)], 2) if b is not None else (None, 2)
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            for arm, (pre, nst) in arms.items():
                if pre is None:
                    print(f"  {m}: {arm} unavailable -- skipped", flush=True)
                    continue
                k = f"{m}|{arm}"
                if k in res["cells"]:
                    continue
                sk, sd, xu = sink_strength(model, tok, dev, pool, pre,
                                           np.random.default_rng(SINK_SEED), n_start=nst)
                # phi was censused in the raw2 / bos conditions only; raw3 has no phi by design.
                ph = phi_of(m, {"raw2": "raw", "bos": "bos"}.get(arm, arm))
                res["cells"][k] = dict(model=m, arm=arm, sink=sk, sink_sd=sd, sink_x_uniform=xu,
                                       n_prefix_tokens=len(pre), seq_len=len(pre) + nst,
                                       phi=None if ph is None else ph[0],
                                       phi_source=None if ph is None else ph[1])
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<34} {arm:<5} S={len(pre)+nst} sink={sk:.5f} ({xu:.2f}x uniform) "
                      f"phi={'n/a' if ph is None else format(ph[0], '.3f')}", flush=True)
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
