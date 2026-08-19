"""Does the domain effect survive LONGER windows, or is it a two-token phenomenon? F159's gap.

THE OBJECTION THIS ANSWERS. Every finding F144-F160 uses one estimator: the argmax map with a
TWO-TOKEN window, x_{t+1} = argmax p(x | x_{t-1}, x_t). The first thing a reader will say is that no
model ever sees a two-token context, and this project has already been burned there once -- F66 found
a sibling readout to be an out-of-distribution prompt artefact. F160 answered half of that objection
by showing the domain effect survives when the loop is SEEDED with real bigrams rather than uniformly
random pairs. It did not answer the other half: the window itself is still two tokens, and F159's
boundary conceded that phi "is not measurable at long context -- the census is defined at two-token
starts", calling a longer-window probe a new construction rather than a fix. This is that
construction.

THE GENERALISATION, and it is the only place judgement enters. At window W the state is the last W
tokens and the map is (x_{t-W+1..t}) -> (x_{t-W+2..t}, argmax p(. | x_{t-W+1..t})). A fixed point of
that map is the diagonal state (t, t, ..., t) that reproduces t -- exactly the W=2 condition
`ctx[0] == ctx[1] == nxt` generalised, and the RUNG below proves the W=2 case is bit-identical to the
existing census. Cycle detection is on the full W-tuple.

STARTS ARE IN-DISTRIBUTION W-GRAMS, and F160 is what licenses that. Sixteen uniformly random tokens
is not a context any model has seen, so sweeping W with random starts would confound window length
with degree of out-of-distribution-ness. F160 established that at W=2 the domain effect is the same
under random pairs and under real bigrams, so real W-grams are the comparable choice as W grows. The
RUNG uses random pairs, because that is what the stored censuses used.

PRE-REGISTERED:
  RUNG      at W=2 with random starts this must reproduce gate1.argmax_census BIT-IDENTICALLY. Same
            reason as F160: gate1 is not edited, because its import closure is stamped into every
            stored result.
  ANTI-VACUITY  a model is read at W=16 only if that cell can show the DIRECTION its W=2 effect had.
            Headroom must NOT be judged against the observed direction -- that is circular, since an
            effect that can only move one way always shows room in that way. (Caught on Minerva,
            whose raw phi collapses to ~0.01 by W=4: its W=2 DOWNWARD effect becomes unmeasurable
            and the small upward one that remains was being scored as readable.) Persistence
            likewise means the W=2 SIGN survives, not merely that something exceeds tolerance.
  PRIMARY   per model, is the BOS domain effect still real (|d(phi)| above that cell's tolerance) at
            W = 16, and does |d(phi)| decay monotonically across W = 2, 4, 8, 16? Registered
            readings:
              the effect persists at W=16 on a majority of readable models -> the two-token window is
                a convenience, not a confound, and F144-F160 generalise beyond it. The paper's
                "one construction" limit weakens to "one estimator family".
              the effect decays to within tolerance by W=16 -> the programme measures a SHORT-CONTEXT
                phenomenon. That belongs in the abstract, not the limits, and the honest framing
                becomes what conditioning does to a model reading a fragment.
              mixed -> per model, and do not summarise as a rate; this programme has withdrawn four
                factors that looked clean at this n.
  SECONDARY raw-arm phi as a function of W. If phi itself collapses to 0 at large W on every model,
            the PRIMARY is unreadable there by anti-vacuity rather than by decay, and the two must
            not be confused.
  BOUNDARY  four window sizes, one prefix kind (BOS), one text source, two census seeds. A longer
            window is still not a long context in the streaming sense.
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
from in_distribution_census import random_starts, PILE_ROWS      # F160's, unchanged

OUT = str(_ROOT / "results" / "window_length_domain.json")

MODELS = ["tiiuae/Falcon3-1B-Base", "sapienzanlp/Minerva-3B-base-v1.0",
          "EleutherAI/pythia-410m-deduped", "Qwen/Qwen1.5-1.8B",
          "HuggingFaceTB/SmolLM-1.7B", "Qwen/Qwen2.5-1.5B-Instruct"]
WINDOWS = (2, 4, 8, 16)
ARMS = ("raw", "bos")
MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0
MIN_MODELS = 3


def census_window(model, tok, dev, starts, W):
    """The argmax map at window W. W=2 is gate1.argmax_census, proven by the RUNG."""
    endpoints, fixed, cyclic = [], 0, 0
    for s in starts:
        ctx = [int(x) for x in s][:W]
        seen, end = set(), ctx[-1]
        for _ in range(MAX_STEPS):
            nxt = int(torch.argmax(_next_logits(model, ctx, dev)))
            end = nxt
            # the diagonal state (t,...,t) reproducing t -- the W=2 condition generalised
            if len(set(ctx)) == 1 and ctx[0] == nxt:
                fixed += 1
                break
            state = tuple(ctx)
            if state in seen:
                cyclic += 1
                break
            seen.add(state)
            ctx = (ctx + [nxt])[-W:]
        endpoints.append(end)
    n = len(starts)
    cnt = collections.Counter(endpoints)
    top_tok, top_n = cnt.most_common(1)[0]
    return dict(n_starts=n, window=W, n_distinct_endpoints=len(cnt),
                modal_endpoint_share=round(top_n / n, 4),
                modal_endpoint_token=tok.decode([int(top_tok)]),
                fixed_point_fraction=round(fixed / n, 4),
                cyclic_fraction=round(cyclic / n, 4),
                endpoint_histogram=[[int(t), tok.decode([int(t)]), int(c)]
                                    for t, c in cnt.most_common()])


def text_ngrams(tok, seed, W, n=None):
    """ADJACENT W-token windows from real text -- in-distribution W-grams (F160's rationale)."""
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    ids = []
    for r in PILE_ROWS:
        ids.extend(tok(ds[int(r)]["text"][:4000], add_special_tokens=False)["input_ids"])
        if len(ids) > 20000:
            break
    rng = np.random.default_rng(seed)
    n = n or N_STARTS
    offs = rng.integers(0, len(ids) - W - 1, size=n)
    return [[int(ids[o + j]) for j in range(W)] for o in offs]


def analyse(res):
    cells, parts, analysis = res["cells"], [], {}

    rung = [v for k, v in cells.items() if k.endswith("|rung")]
    worst = max((v["abs_err"] for v in rung), default=float("inf"))
    ok = bool(rung) and worst == 0.0
    parts.append(
        f"RUNG (window-2 map reproduces gate1.argmax_census): {len(rung)} models, worst error "
        f"{worst:.2e}. "
        + ("Bit-identical, so the only thing varying below is the WINDOW."
           if ok else "NOT identical -- the generalised map is not gate1's at W=2, and nothing "
                      "below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    rows = {}
    for m in MODELS:
        per_w = {}
        for W in WINDOWS:
            got = {}
            for arm in ARMS:
                ks = [f"{m}|W{W}|{arm}|s{cs}" for cs in CENSUS_SEEDS]
                if all(k in cells for k in ks):
                    v = [cells[k]["fixed_point_fraction"] for k in ks]
                    got[arm] = (float(np.mean(v)), float(abs(v[0] - v[1])))
            if len(got) == 2:
                (pr, nr), (pb, nb) = got["raw"], got["bos"]
                tol = max(MIN_SHIFT, NOISE_FACTOR * max(nr, nb))
                d = pb - pr
                # HEADROOM MUST NOT BE COMPUTED FOR THE OBSERVED DIRECTION -- that is circular.
                # An effect that can only move one way always shows room in that way. Minerva's raw
                # phi collapses to ~0.01 by W=4, so its W=2 DOWNWARD effect becomes unmeasurable and
                # a small upward one is all that remains; scoring headroom on the observed (+) sign
                # called that cell readable. Record both sides, and judge against the direction
                # whose persistence is actually being tested.
                room_down, room_up = pr, 1.0 - pr
                per_w[W] = dict(phi_raw=round(pr, 4), phi_bos=round(pb, 4), d=round(d, 4),
                                tol=round(tol, 4), room_down=round(room_down, 4),
                                room_up=round(room_up, 4),
                                real=bool(abs(d) > tol),
                                free=bool(room_down > tol and room_up > tol))
        if per_w:
            rows[m] = per_w
    analysis["rows"] = rows
    if not rows:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " No model has a complete window yet."
        return

    parts.append(
        "Per model, |d(phi)| by window (* = below tolerance; + = one direction unavailable): "
        + "; ".join("{}: ".format(m.split("/")[-1])
                    + ", ".join("W{}={:.3f}{}".format(
                        W, abs(v["d"]),
                        ("" if v["real"] else "*") + ("" if v["free"] else "+"))
                        for W, v in sorted(per_w.items()))
                    for m, per_w in rows.items()) + ". ")

    biggest = max(WINDOWS)
    # The question is whether the W=2 effect PERSISTS, so the W=16 cell must be able to show the
    # W=2 DIRECTION. A model whose raw phi has collapsed can no longer move down however large the
    # domain effect is, and reporting that as "the effect reversed" or "persisted" would be reading
    # a forced sign.
    readable, excluded = [], []
    for m, pw in rows.items():
        if biggest not in pw or 2 not in pw or not pw[2]["real"]:
            excluded.append((m, "no real W=2 effect to test"))
            continue
        want_down = pw[2]["d"] < 0
        room = pw[biggest]["room_down"] if want_down else pw[biggest]["room_up"]
        if room <= pw[biggest]["tol"]:
            excluded.append((m, f"W={biggest} cannot show the W=2 direction "
                                f"({'down' if want_down else 'up'}); room {room:.3f} "
                                f"<= tol {pw[biggest]['tol']:.3f}"))
            continue
        readable.append(m)
    analysis["excluded_detail"] = [dict(model=m, reason=r) for m, r in excluded]
    parts.append(
        f"ANTI-VACUITY: {len(readable)} of {len(rows)} models can be read at W={biggest}. A model is "
        f"excluded if its W={biggest} cell cannot show the DIRECTION its W=2 effect had -- headroom "
        f"is judged against that direction, not against the observed one, because scoring the "
        f"observed sign is circular"
        + (f": {[(m.split('/')[-1], r) for m, r in excluded]}." if excluded else "."))
    if len(readable) < MIN_MODELS:
        analysis["primary_readable"] = False
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + (
            f" PRIMARY NOT READ: {len(readable)} of {MIN_MODELS} readable models required. Four "
            f"factors in this programme were withdrawn after being called from a handful of models "
            f"(F151, F153, F154, F156), so the count gate is not optional.")
        return
    analysis["primary_readable"] = True

    # persistence means the W=2 SIGN survives, not merely that something is above tolerance
    persists, decayed = [], []
    for m in readable:
        s2, sB = np.sign(rows[m][2]["d"]), np.sign(rows[m][biggest]["d"])
        (persists if (rows[m][biggest]["real"] and s2 == sB) else decayed).append(m)
    mono = []
    for m in readable:
        seq = [abs(rows[m][W]["d"]) for W in sorted(rows[m]) if W in WINDOWS]
        mono.append(all(b <= a + MIN_SHIFT for a, b in zip(seq, seq[1:])))
    analysis.update(persists=persists, decayed=decayed, n_monotone_decay=int(sum(mono)))
    parts.append(
        f"PRIMARY, is the domain effect still real at W={biggest}? Persists on "
        f"{len(persists)} of {len(readable)} readable models "
        f"({[m.split('/')[-1] for m in persists] or '-'}); decayed within tolerance on "
        f"{len(decayed)} ({[m.split('/')[-1] for m in decayed] or '-'}). "
        f"|d(phi)| decays monotonically in W on {sum(mono)} of {len(readable)}. "
        + (f"The effect survives a {biggest}-token window, so the two-token choice is a convenience "
           f"rather than a confound and F144-F160 generalise beyond it. The paper's 'one "
           f"construction' limit weakens to 'one estimator family'."
           if len(persists) > len(readable) / 2 else
           f"The effect decays to within tolerance by W={biggest} on a majority. The programme "
           f"measures a SHORT-CONTEXT phenomenon, and that belongs in the abstract rather than the "
           f"limits: the honest subject is what conditioning does to a model reading a fragment."
           if len(decayed) > len(readable) / 2 else
           f"Split {len(persists)}/{len(decayed)}. Report per model and do not summarise as a rate."))

    base = {}
    for m, pw in rows.items():
        base[m.split("/")[-1]] = {f"W{W}": v["phi_raw"] for W, v in sorted(pw.items())}
    analysis["raw_phi_by_window"] = base
    allzero = [m for m, pw in rows.items()
               if biggest in pw and pw[biggest]["phi_raw"] < MIN_SHIFT]
    parts.append(
        f"SECONDARY, raw-arm phi by window: "
        + "; ".join(f"{m} " + " ".join(f"{k}={v:.3f}" for k, v in d.items())
                    for m, d in base.items()) + ". "
        + (f"{len(allzero)} of {len(rows)} models have raw phi below {MIN_SHIFT:.3f} at "
           f"W={biggest} ({[m.split('/')[-1] for m in allzero]}), so on those the PRIMARY is "
           f"unreadable by ANTI-VACUITY rather than by decay -- the two must not be confused, and "
           f"the exclusion above is what keeps them apart."
           if allzero else
           f"Every model retains measurable raw phi at W={biggest}, so decay of the effect cannot be "
           f"attributed to the readout vanishing."))
    parts.append(
        f"BOUNDARY: {len(rows)} models, windows {WINDOWS}, ONE prefix kind (BOS), one text source, "
        f"{len(CENSUS_SEEDS)} census seeds, {N_STARTS} starts. Starts are in-distribution W-grams, "
        f"licensed by F160's finding that random and real starts give the same effect at W=2. A "
        f"{biggest}-token window is still not a long context in the streaming sense.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, windows=list(WINDOWS), arms=list(ARMS), n_starts=N_STARTS,
        census_seeds=CENSUS_SEEDS, min_shift=MIN_SHIFT, noise_factor=NOISE_FACTOR,
        min_models=MIN_MODELS,
        generalisation="at window W the state is the last W tokens; a fixed point is the diagonal "
                       "state (t,...,t) reproducing t, which is the W=2 condition generalised. "
                       "Cycle detection on the full W-tuple.",
        starts="in-distribution W-grams from real text, licensed by F160 (random and real starts "
               "give the same effect at W=2); the RUNG uses random pairs, as the stored censuses did",
        rung="W=2 with random starts must reproduce gate1.argmax_census bit-identically",
        primary="is the BOS domain effect still real at W=16, and does |d(phi)| decay in W",
        readings=dict(
            persists="the two-token window is a convenience, not a confound; F144-F160 generalise",
            decays="the programme measures a SHORT-CONTEXT phenomenon -- abstract, not limits",
            mixed="per model; do not summarise as a rate"),
        secondary="raw-arm phi by window -- if phi itself vanishes, the primary is unreadable by "
                  "anti-vacuity rather than by decay, and the two must not be confused",
        why="F159 conceded phi is not measurable at long context and called a longer-window probe a "
            "new construction; this is it, and it answers the objection F66 makes live")
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done = 0
        for m in MODELS:
            want = [f"{m}|W{W}|{a}|s{cs}" for W in WINDOWS for a in ARMS for cs in CENSUS_SEEDS]
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
                mine = census_window(model, tok, dev, random_starts(pool, cs), 2)
                err = abs(theirs["fixed_point_fraction"] - mine["fixed_point_fraction"])
                res["cells"][rk] = dict(model=m, abs_err=float(err),
                                        theirs=theirs["fixed_point_fraction"],
                                        mine=mine["fixed_point_fraction"])
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m:<34} RUNG W2 theirs={theirs['fixed_point_fraction']:.4f} "
                      f"mine={mine['fixed_point_fraction']:.4f} err={err:.2e}", flush=True)

            for W in WINDOWS:
                for cs in CENSUS_SEEDS:
                    starts = text_ngrams(tok, cs, W)
                    for arm in ARMS:
                        k = f"{m}|W{W}|{arm}|s{cs}"
                        if k in res["cells"]:
                            continue
                        target = model if arm == "raw" else _Prefixed(model, [int(b)])
                        c = census_window(target, tok, dev, starts, W)
                        c.update(cls=classify(c), model=m, arm=arm, census_seed=cs)
                        res["cells"][k] = c
                        json.dump(res, open(OUT, "w"), indent=1)
                        print(f"  {m:<34} W{W:<3} {arm:<3} s={cs} cls={c['cls']:<11} "
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
