"""Does the attention sink behave as its literature describes, in ITS regime rather than ours? F158's
unresolved boundary.

WHAT F158 LEFT OPEN. Measuring sink strength at the census's own context length -- three tokens --
two things happened. Sink and the fixed-point shift agreed in sign on 2 of 5 models, chance. And the
sink literature's own uniformity claim (initial-token effects vary across models in magnitude, never
in sign) did NOT reproduce: 2 of 6 models showed sink DECREASING under BOS. F158 recorded that as a
statement about three-token contexts and explicitly declined to read it as a refutation, because that
literature measures long contexts, where the phenomenon is about attention concentrating despite many
competing positions.

THE REGIME DIFFERS ON TWO AXES, NOT ONE, and F158 only named the first:
  LENGTH   the account is about hundreds to thousands of tokens; our probe runs at two or three.
  CONTENT  the account is measured on REAL TEXT; our probe draws UNIFORMLY RANDOM tokens, which is
           out-of-distribution input in a way real text is not.
Either could explain the non-reproduction, and they are cheap to separate. This run crosses both.

WHY THIS MATTERS FOR PAPER 2, and it may narrow rather than strengthen it. If uniformity EMERGES at
long context on real text, then the account holds in its own regime and F158's non-reproduction is a
regime artefact. The paper's claim then narrows -- correctly -- from "we contradict the account" to
"the account's regime is not our probe's regime, so it does not straightforwardly apply". That is a
weaker headline and a more defensible one. We would rather find this now.

PRE-REGISTERED:
  PRIMARY   at each (length, content) cell, is d(sink) = sink(bos) - sink(length-matched raw)
            UNIFORM in sign across models? Registered readings:
              uniformity emerges as length grows on REAL TEXT -> the account holds in its own regime;
                F158's non-reproduction is a short-context/random-token artefact, and paper 2 must
                narrow to "different regime" rather than "contradiction".
              non-uniformity persists at every length and both contents -> the uniformity prediction
                fails in the account's own regime too. A stronger claim, and one to state carefully.
              uniformity depends on CONTENT rather than LENGTH -> the relevant regime difference is
                out-of-distribution input, not context size; say so.
  SECONDARY the raw-arm sink CONCENTRATION (sink x sequence length, multiples of uniform) as a
            function of length. The account predicts concentration RISING with context length: more
            positions compete, and mass still lands on the first. NOTE the raw attention FRACTION
            necessarily falls as 1/S shrinks, so it is the normalised quantity that is registered
            here. If concentration does not rise, this measurement is not reproducing the phenomenon
            and nothing above should be read.
  BOUNDARY  phi is NOT measured here and cannot be: the census is defined at two-token starts, so
            there is no long-context fixed-point fraction to correlate against. This run is about
            whether the MECHANISM behaves as described, not about whether it explains our readout.
            It can narrow paper 2's claim; it cannot rescue it.
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

OUT = str(_ROOT / "results" / "sink_long_context.json")

MODELS = ["tiiuae/Falcon3-1B-Base", "sapienzanlp/Minerva-3B-base-v1.0",
          "EleutherAI/pythia-410m-deduped", "Qwen/Qwen1.5-1.8B",
          "HuggingFaceTB/SmolLM-1.7B", "Qwen/Qwen2.5-1.5B-Instruct"]
LENGTHS = (2, 8, 32, 128, 512)       # ordinary tokens after the prefix
CONTENTS = ("random", "text")        # our probe's input vs the account's
K_STARTS = 32                        # raised from 8: at K=8 the long-context cells were
                                     # UNDERPOWERED -- d(sink) sat inside its own standard
                                     # error on 4-5 of 6 models, which the noise gate correctly
                                     # refused to read. Not a null; too few draws.
SEED = 20260817
PILE_ROW = 0                         # a prose row, fixed before the run


def _pile_text():
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return ds[PILE_ROW]["text"]


def sink_at(model, tok, dev, pool, text_ids, content, n, use_bos, bos_id, rng):
    """Mean attention from the LAST position to POSITION 0, over layers, heads, K draws."""
    vals = []
    for _ in range(K_STARTS):
        if content == "random":
            body = [int(x) for x in rng.choice(pool, size=n, replace=False)]
        else:
            if len(text_ids) <= n:
                return None, None
            off = int(rng.integers(0, len(text_ids) - n))
            body = [int(x) for x in text_ids[off:off + n]]
        pre = [int(bos_id)] if use_bos else [int(rng.choice(pool))]   # length-matched control
        ids = torch.tensor([pre + body], device=dev)
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True)
        per_layer = [float(a[0, :, -1, 0].mean().item()) for a in out.attentions]
        vals.append(float(np.mean(per_layer)))
        del out
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def analyse(res):
    cells, parts, analysis = res["cells"], [], {}
    # A SIGN CAST BY NOISE IS NOT A SIGN. d(sink) values of ~0.001 were being counted as "down"
    # votes against uniformity. Each cell carries the across-draw sd, so the difference has a
    # standard error; a model votes only if |d| clears NOISE_FACTOR times it. Same anti-vacuity
    # rule the rest of the programme uses, applied to this quantity.
    NOISE_FACTOR = 2.0
    rows = {}
    for k, v in cells.items():
        if v.get("sink") is None:
            continue
        m, content, n, arm = k.split("|")
        rows.setdefault((content, int(n)), {}).setdefault(m, {})[arm] = (v["sink"], v.get("sink_sd"))
    MIN_MODELS_FOR_UNIFORMITY = 3   # "uniform across 1 model" is trivially true -- do not read it
    grid = {}
    for (content, n), per_model in sorted(rows.items()):
        ds, sub_noise = {}, []
        for m, a in per_model.items():
            if "bos" not in a or "raw" not in a:
                continue
            (vb, sb), (vr, sr) = a["bos"], a["raw"]
            dv = vb - vr
            if not np.isfinite(dv):
                continue
            se = float(np.hypot(sb or 0.0, sr or 0.0) / np.sqrt(K_STARTS))
            if abs(dv) <= NOISE_FACTOR * se:
                sub_noise.append(m)          # measured, but not resolved -- must not vote
                continue
            ds[m] = dv
        if not ds:
            continue
        signs = {np.sign(v) for v in ds.values()}
        grid[f"{content}@{n}"] = dict(
            n_models=len(ds), n_sub_noise=len(sub_noise),
            sub_noise=[m.split("/")[-1] for m in sub_noise],
            readable=bool(len(ds) >= MIN_MODELS_FOR_UNIFORMITY),
            uniform=bool(len(signs) == 1) if len(ds) >= MIN_MODELS_FOR_UNIFORMITY else None,
            n_down=int(sum(1 for v in ds.values() if v < 0)),
            mean_d=round(float(np.mean(list(ds.values()))), 5),
            per_model={m.split("/")[-1]: round(v, 5) for m, v in ds.items()})
    analysis["grid"] = grid
    if not grid:
        res["analysis"] = analysis
        res["verdict"] = "No complete cells yet."
        return

    parts.append(
        "PRIMARY, is d(sink) = sink(bos) - sink(length-matched raw) UNIFORM in sign across models, "
        "per (content, length) cell? "
        + "; ".join("{}: {} of {}, mean {:+.4f}".format(
            k, ("UNIFORM" if v["uniform"] else "mixed ({} down)".format(v["n_down"]))
               if v["readable"] else "n/a (needs {}+ resolved)".format(MIN_MODELS_FOR_UNIFORMITY),
            v["n_models"], v["mean_d"]) for k, v in grid.items())
        + ". Models whose d(sink) does not clear {}x its standard error are recorded as UNRESOLVED "
          "and do not vote: ".format(NOISE_FACTOR)
        + "; ".join("{} {}".format(k, v["sub_noise"] or "-") for k, v in grid.items()) + ". ")

    txt = {k: v for k, v in grid.items() if k.startswith("text@")}
    rnd = {k: v for k, v in grid.items() if k.startswith("random@")}
    long_txt = [v for k, v in txt.items() if int(k.split("@")[1]) >= 128 and v["readable"]]
    long_uniform = bool(long_txt) and all(v["uniform"] for v in long_txt)
    any_uniform = any(v["uniform"] for v in grid.values())
    parts.append(
        ("Uniformity EMERGES on real text at long context, so the account holds in its own regime and "
         "F158's non-reproduction is a regime artefact. Paper 2 must narrow its claim from "
         "'we contradict the account' to 'the account's regime is not our probe's regime, so it does "
         "not straightforwardly apply' -- weaker, and more defensible."
         if long_uniform else
         "Uniformity does NOT emerge on real text at long context. The account's own prediction is "
         "not reproduced even in the regime it is about, on this measurement. State this carefully: "
         "it is a claim about our implementation of their quantity as much as about their quantity."
         if txt else "No real-text cells complete yet."))
    if txt and rnd:
        tu = sum(1 for v in txt.values() if v["readable"] and v["uniform"])
        ru = sum(1 for v in rnd.values() if v["readable"] and v["uniform"])
        parts.append(
            f"CONTENT vs LENGTH: uniform on {tu} of {len(txt)} real-text cells and {ru} of "
            f"{len(rnd)} random-token cells. "
            + ("Content matters more than length here -- the relevant regime difference is "
               "out-of-distribution input, not context size."
               if tu != ru and abs(tu - ru) >= max(1, len(txt) // 2) else
               "Neither axis alone separates the regimes on this evidence."))

    seq = {}
    for k, v in cells.items():
        if v.get("sink") is None:
            continue
        m, content, n, arm = k.split("|")
        if arm == "raw":
            # NORMALISED, not raw. The raw fraction MUST fall as 1/S shrinks, so testing whether it
            # rises tests something no sink account claims. The phenomenon is concentration RELATIVE
            # to what is available: sink x S. Registering the raw fraction was our mis-specification.
            seq.setdefault(content, {}).setdefault(m, {})[int(n)] = v["sink"] * (int(n) + 1)
    rise = {}
    for content, per_model in seq.items():
        ups = 0
        for m, byn in per_model.items():
            ns = sorted(byn)
            if len(ns) >= 2 and byn[ns[-1]] > byn[ns[0]]:
                ups += 1
        rise[content] = (ups, len(per_model))
    analysis["raw_sink_rises_with_length"] = rise
    parts.append(
        "SECONDARY, does raw-arm sink concentration (sink x sequence length, i.e. multiples of uniform) RISE with context length -- the account's basic prediction? "
        + "; ".join(f"{c}: {u} of {n} models" for c, (u, n) in rise.items()) + ". "
        + ("It does on most models, so this measurement is reproducing the phenomenon and the "
           "PRIMARY above is interpretable."
           if all(u > n / 2 for u, n in rise.values() if n) else
           "It does NOT rise on most models, so this measurement may not be reproducing the "
           "phenomenon at all, and the PRIMARY above should not be read as being about the "
           "account's quantity."))
    parts.append(
        f"BOUNDARY: phi is NOT measured here and cannot be -- the census is defined at two-token "
        f"starts, so there is no long-context fixed-point fraction to correlate against. This run "
        f"asks whether the MECHANISM behaves as described, not whether it explains our readout. It "
        f"can narrow paper 2's claim; it cannot rescue it. Lengths {LENGTHS}, contents {CONTENTS}, "
        f"{K_STARTS} draws per cell, one Pile row for the text arm.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, lengths=list(LENGTHS), contents=list(CONTENTS), k_starts=K_STARTS,
        seed=SEED, pile_row=PILE_ROW,
        primary="is d(sink) uniform in sign across models, per (content, length) cell",
        readings=dict(
            uniformity_emerges_long_text="the account holds in its own regime; F158's "
                                         "non-reproduction is a regime artefact and paper 2 must "
                                         "NARROW to 'different regime', not 'contradiction'",
            persists="the prediction fails in its own regime too -- state carefully",
            content_not_length="the regime difference is out-of-distribution input, not size"),
        secondary="does raw-arm sink RISE with length -- if not, this is not reproducing the "
                  "phenomenon and the primary is uninterpretable",
        boundary="phi is not measurable at long context; this can narrow paper 2's claim, not "
                 "rescue it",
        why="F158 named ONE regime difference (length) but there are TWO -- the account is measured "
            "on real text and our probe draws uniformly random tokens")
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        raw_text = _pile_text()
        done = 0
        for m in MODELS:
            want = [f"{m}|{c}|{n}|{a}" for c in CONTENTS for n in LENGTHS for a in ("raw", "bos")]
            if all(k in res["cells"] for k in want):
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
                model = AutoModelForCausalLM.from_pretrained(
                    m, attn_implementation="eager").eval().to(
                    dev, torch.bfloat16 if dev != "cpu" else torch.float32)   # bf16: F158's NaN fix
            except Exception as e:
                res["cells"][f"{m}|failed|0|raw"] = dict(model=m, error=type(e).__name__, sink=None)
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED {type(e).__name__}", flush=True)
                continue
            b = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            tids = tok(raw_text, add_special_tokens=False)["input_ids"]
            print(f"  {m:<34} text arm has {len(tids)} tokens", flush=True)
            for content in CONTENTS:
                for n in LENGTHS:
                    for arm in ("raw", "bos"):
                        k = f"{m}|{content}|{n}|{arm}"
                        if k in res["cells"]:
                            continue
                        s, sd = sink_at(model, tok, dev, pool, tids, content, n,
                                        arm == "bos", b, np.random.default_rng(SEED))
                        res["cells"][k] = dict(model=m, content=content, n=n, arm=arm,
                                               sink=s, sink_sd=sd, seq_len=None if s is None else n + 1)
                        json.dump(res, open(OUT, "w"), indent=1)
                        print(f"  {m:<34} {content:<6} n={n:<4} {arm:<3} "
                              + ("skipped (text too short)" if s is None
                                 else f"sink={s:.5f} ({s*(n+1):.2f}x uniform)"), flush=True)
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
