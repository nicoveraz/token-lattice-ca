"""Does the settled state depend on what you seed it with? (#94)

THE QUESTION #93 CANNOT ANSWER. #93 measures novelty of the settled ring, seeded from RANDOM
tokens. That is novelty-from-noise. Real generation is prompted, and novelty relative to a prompt
is the quantity anyone actually cares about. Whether the two are the same thing depends entirely on
the basin structure, which nothing here has tested.

F70 showed pythia-410m's argmax map has an attracting fixed point and that 18 of 24 RANDOM starts
reach it. It said nothing about whether that basin is global. Three possibilities, meaning three
very different things:

    global attractor    text init also flows to the fixed point. The CA ERASES the prompt, and
                        novelty is a property of the map alone.
    init-dependent      text init stays near text. The CA EDITS the prompt, and novelty is
                        relative -- prompt-dependent, which is what generation actually is.
    partial retention   some of the seed survives. The interesting case, and the closest to what
                        generation does.

If the first holds, #93's numbers are the whole story. If the second or third holds, #93's "31% of
the way from real text to shuffled" is one point in a family, and specifically the point reached
from the least interesting starting condition.

THREE INITIAL CONDITIONS, ONE MEASUREMENT

    random        what #93 does now -- tokens drawn uniformly from the emission pool
    corpus        real text from the Pile sample, tokenised by the generator
    fixed-point   the ring seeded ENTIRELY with the model's own attractor token, to test whether
                  anything ESCAPES. Convergence is only half the question; the other half is
                  whether the attractor is a trap.

WHY r=2 IS INCLUDED HERE, HAVING BEEN EXCLUDED FROM #93. #93 avoided r=2 because F69 showed it is
degenerate and measuring "creativity" there would repeat F62-F66's mistake. This experiment is
about BASINS, and the fixed point lives at r=2 -- excluding it would remove the only regime where
the question has teeth. r=3 is run alongside precisely so the comparison shows whether basin
structure is specific to the degenerate window.

PRE-REGISTERED:
  * Primary: at each (construction, r, T), do the three initial conditions settle to the same
    composition? Measured as the spread in top-1 share and in retention across inits.
      - converge  -> the attractor is global, the prompt is erased, and #93's numbers stand as the
                     whole story.
      - diverge   -> basins are real, novelty is prompt-relative, and #93 measures one basin.
  * Secondary: does the fixed-point seed ESCAPE? If retention there stays near 1.0 the attractor is
    a trap; if it falls, the fixed point is unstable to noise at that temperature.
  * The MLM construction is the control. F67 showed it has no absorbing state, so its inits should
    converge trivially. If they do NOT, the basin structure is richer than "has a fixed point or
    does not", and F70's account is incomplete.

Writes results/basin_dependence.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/basin_dependence.py
        (safe to interrupt and re-run -- it resumes)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, re, time, collections
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel

GENERATORS = [("ar",  "EleutherAI/pythia-410m", "step143000", "has a fixed point (F70)"),
              ("mlm", "bert-base-uncased",      None,         "control -- no absorbing state (F67)")]
RADII = [2, 3]                         # r=2 deliberately: the fixed point lives there
TEMPS = [0.02, 0.50, 0.70]
INITS = ["random", "corpus", "fixed_point"]
SCORER = "gpt2-large"
N, B, SETTLE = 96, 16, 16
CORPUS_DOCS = 400
OUT = str(_ROOT / "results" / "basin_dependence.json")

_WORD = re.compile(r"[a-z0-9']+")


def words(t):
    return _WORD.findall(t.lower())


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        generators=[g for _, g, _, _ in GENERATORS], radii=RADII, temps=TEMPS, inits=INITS,
        scorer=SCORER, N=N, B=B, settle=SETTLE,
        primary="do the three inits settle to the same composition? converge -> global attractor, "
                "prompt erased, #93's numbers are the whole story; diverge -> basins are real and "
                "novelty is prompt-relative",
        secondary="does the fixed-point seed ESCAPE? retention near 1.0 means the attractor is a trap",
        control="the MLM construction has no absorbing state (F67), so its inits should converge "
                "trivially; if they do not, F70's account is incomplete",
        why_r2_here="#93 excluded r=2 because measuring creativity there would repeat F62-F66's "
                    "mistake. This is about BASINS and the fixed point lives at r=2, so excluding "
                    "it would remove the only regime where the question has teeth",
        resumable="keyed by (construction, r, T, init)")
    runs = res["runs"]

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    docs = [t for t in load_dataset("NeelNanda/pile-10k",
                                    split=f"train[:{CORPUS_DOCS}]")["text"] if t and t.strip()]
    ref2 = set()
    for t in docs:
        w = words(t); ref2.update(zip(w, w[1:]))
    print(f"  reference 2-grams: {len(ref2):,}", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    stok = AutoTokenizer.from_pretrained(SCORER)
    scorer = AutoModelForCausalLM.from_pretrained(SCORER).eval().to(
        dev, torch.float16 if dev != "cpu" else torch.float32)

    @torch.no_grad()
    def nll(text):
        ids = stok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
        return None if ids.shape[1] < 8 else float(scorer(input_ids=ids, labels=ids).loss.item())

    for kind, gen, rev, role in GENERATORS:
        keys = [f"{kind}|r{r}|T{T}|{i}" for r in RADII for T in TEMPS for i in INITS]
        if all(k in runs for k in keys):
            print(f"  {gen}: already complete", flush=True); continue
        if kind == "ar":
            from ar_ca import ARRule, run as carun
            rule = ARRule(gen, revision=rev) if rev else ARRule(gen)
            scheme = "none"
        else:
            from mlm_ca import MLMRule, run as carun
            rule = MLMRule(gen)
            scheme = "cls_sep"
        pool = np.array([i for i in range(rule.V) if i not in set(rule.forbidden.tolist())],
                        dtype=np.int64)
        print(f"\n  {kind} / {gen} ({role})", flush=True)

        # the model's own attractor token, measured rather than assumed
        probe = carun(rule, B=8, N=N, r=2, T=0.02, sweeps=SETTLE, scheme=scheme,
                      init="random", seed=3, order="per_replica")["final"]
        attractor_id = collections.Counter(probe.reshape(-1).tolist()).most_common(1)[0][0]
        print(f"    attractor token: {rule.tok.decode([attractor_id])!r}", flush=True)

        corpus_ids = rule.tok(" ".join(docs[:60]))["input_ids"]
        rng = np.random.default_rng(11)

        def make_init(which):
            if which == "random":
                return rng.choice(pool, size=(B, N))
            if which == "fixed_point":
                return np.full((B, N), attractor_id, dtype=np.int64)
            starts = rng.integers(0, max(1, len(corpus_ids) - N - 1), size=B)
            return np.array([corpus_ids[s:s + N] for s in starts], dtype=np.int64)

        for r in RADII:
            for T in TEMPS:
                for which in INITS:
                    key = f"{kind}|r{r}|T{T}|{which}"
                    if key in runs: continue
                    init = make_init(which)
                    s = carun(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme=scheme,
                              init_state=init.copy(), seed=7, order="per_replica")["final"]
                    text = " ".join(rule.tok.decode(row.tolist()) for row in s)
                    w = words(text); g = list(zip(w, w[1:]))
                    cnt = collections.Counter(s.reshape(-1).tolist())
                    runs[key] = dict(
                        kind=kind, model=gen, r=r, T=T, init=which,
                        attractor_token=rule.tok.decode([attractor_id]),
                        top1_share=round(float(cnt.most_common(1)[0][1] / s.size), 4),
                        distinct_frac=round(float(np.mean([len(set(row.tolist())) for row in s]) / N), 4),
                        retention=round(float((s == init).mean()), 4),
                        novel_2gram=round(sum(x not in ref2 for x in g) / len(g), 4) if g else None,
                        nll=nll(text), sample=text[:130])
                    v = runs[key]
                    print(f"    r={r} T={T:<5} {which:>12}  top1={v['top1_share']:.3f} "
                          f"retain={v['retention']:.3f} novel2={v['novel_2gram']} "
                          f"NLL={v['nll'] if v['nll'] is None else round(v['nll'],2)}", flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    del scorer
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    out, parts = {}, []
    for kind, gen, _, role in GENERATORS:
        print(f"\n=== {kind} / {gen} ({role}) ===")
        print(f"  {'r':>3} {'T':>5}   " + "  ".join(f"{i:>26}" for i in INITS))
        for r in RADII:
            for T in TEMPS:
                cells = [runs.get(f"{kind}|r{r}|T{T}|{i}") for i in INITS]
                if not all(cells): continue
                cols = [f"top1 {c['top1_share']:.2f} ret {c['retention']:.2f} nll "
                        f"{'--' if c['nll'] is None else format(c['nll'], '.2f')}" for c in cells]
                spread = max(c["top1_share"] for c in cells) - min(c["top1_share"] for c in cells)
                out[f"{kind}|r{r}|T{T}"] = dict(
                    top1_spread=round(float(spread), 4),
                    by_init={c["init"]: dict(top1=c["top1_share"], retention=c["retention"],
                                             nll=c["nll"], novel2=c["novel_2gram"]) for c in cells})
                print(f"  {r:>3} {T:>5}   " + "  ".join(f"{c:>26}" for c in cols)
                      + f"   spread={spread:.3f}")

        rows = [v for k, v in out.items() if k.startswith(f"{kind}|")]
        if rows:
            mx = max(v["top1_spread"] for v in rows)
            conv = mx < 0.10
            fp = [v["by_init"]["fixed_point"]["retention"] for v in rows
                  if "fixed_point" in v["by_init"]]
            trapped = [x for x in fp if x > 0.9]
            parts.append(
                f"{kind.upper()}: " +
                (f"INITS CONVERGE (max top-1 spread {mx:.3f} across all cells) -- the attractor is "
                 f"global, the seed is erased, and #93's novelty-from-noise is the whole story."
                 if conv else
                 f"INITS DIVERGE (max top-1 spread {mx:.3f}) -- basins are real, so novelty is "
                 f"PROMPT-RELATIVE and #93 measures one basin among several. Its 'between recall "
                 f"and noise' reading applies to random seeding only.") +
                (f" The fixed-point seed is a TRAP in {len(trapped)}/{len(fp)} cells "
                 f"(retention > 0.9)." if fp else ""))
    verdict = " ".join(parts) if parts else "insufficient data"
    print(f"\n  -> {verdict}")

    res["analysis"] = out
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "#93 measures novelty of the settled ring seeded from RANDOM tokens -- novelty-from-noise. "
        "Real generation is prompted, and whether the two coincide depends on basin structure, "
        "which nothing had tested. F70 showed pythia's argmax map has an attracting fixed point "
        "reached by 18 of 24 random starts, but said nothing about whether that basin is global. "
        "Three inits are compared: random, real corpus text, and the model's own attractor token "
        "filling the ring -- the last to test whether anything ESCAPES, since convergence is only "
        "half the question. r=2 is included here although #93 excluded it, because the fixed point "
        "lives at r=2 and the basin question has no teeth without it; r=3 is run alongside to show "
        "whether basin structure is specific to the degenerate window. The MLM construction is the "
        "control: F67 showed it has no absorbing state, so its inits should converge trivially.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
