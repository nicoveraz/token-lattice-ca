"""Does the CA's melting temperature T* predict a KNOWN failure mode? (#90; tests the F62-F66 line)

WHAT T* IS. The nineteen-model screen measured how concentrated the settled token lattice is at
several temperatures. Interpolating where the dominant token's share falls through 40% gives a
single scalar per model, T* -- the temperature at which the model's short-context conditional stops
being dominated by one token. Higher T* means a more peaked conditional when the model is handed
almost no context.

T* has two properties that make it worth a second look. It is far **tighter within a family** than
the raw share at any fixed temperature -- Pythia's top-1 at T=0.02 wanders over 74-98% across a 70x
size range while its T* sits at 0.52-0.58 -- and it **separates families the binary lumps
together**: Qwen melts at 0.30, granite near 0.50, Pythia near 0.55, GPT-Neo not even by 0.70,
though all four "have an attractor".

WHY THAT IS NOT ENOUGH ON ITS OWN. T* is measured in a regime the model was never trained for: a
two-token context. F66 established that this regime produces an artifact. So T* is, so far, a
well-defined property of *out-of-distribution behaviour* and nothing more. It becomes interesting
only if it predicts something measured **independently of the CA construction**.

THE ANCHOR. Neural text degeneration -- the collapse into repetition and blandness under greedy or
low-temperature decoding (Holtzman et al., "The Curious Case of Neural Text Degeneration") -- is a
known, independently studied failure mode. If a model's short-context conditional is sharply peaked
on filler, greedy decoding should fall into loops sooner. That is a testable prediction, and the
measurement shares no machinery with the CA: free generation from real text prompts, not iterated
resampling on a ring.

PRE-REGISTERED BEFORE RUNNING:
  * Primary: across the models with a finite T*, does T* correlate with repetition under greedy
    decoding? Reported as Spearman rho (rank, so it does not assume linearity) with n stated.
      - strong positive -> T* predicts a known failure mode. It stops being "a number our probe
        emits" and becomes a cheap scalar with external meaning, measurable in four settle runs.
      - null            -> T* is a property of the artifact and nothing more. Reportable, and it
        closes the question rather than leaving it open.
  * Models with NO finite T* (they never concentrate) are run too and reported separately. If they
    also show low repetition, that extends the association; if they show high repetition, T* is
    clearly not capturing degeneration.
  * Prompts are short but REAL -- fixed English sentence openings, tokenised per model. Not
    two-token fragments, which would make this a restatement of the CA measurement rather than an
    independent check.
  * Greedy decoding (no sampling), so the measurement has no temperature knob of its own and
    cannot be tuned to agree.

METRIC. `rep_4` = fraction of 4-grams in the continuation that are repeats of an earlier 4-gram.
This is the standard repetition statistic in that literature. `distinct_1` and the longest
immediate loop are recorded alongside so a single metric is not carrying the conclusion.

Writes results/degeneration_vs_tstar.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/degeneration_vs_tstar.py
        (safe to interrupt and re-run -- it resumes)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time, collections
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel

SCREEN = str(_ROOT / "results" / "attractor_corpus_screen.json")
SCREEN_TEMPS = [0.02, 0.20, 0.436, 0.70]
THRESH = 0.40                          # the screen's attractor threshold, unchanged

MODELS = [
    ("EleutherAI/pythia-14m", "step143000"), ("EleutherAI/pythia-31m", "step143000"),
    ("EleutherAI/pythia-70m", "step143000"), ("EleutherAI/pythia-160m", "step143000"),
    ("EleutherAI/pythia-410m", "step143000"), ("EleutherAI/pythia-1b", "step143000"),
    ("EleutherAI/gpt-neo-125M", None), ("Qwen/Qwen2.5-0.5B", None),
    ("ibm-granite/granite-3.0-1b-a400m-base", None), ("ibm-granite/granite-3.0-2b-base", None),
    ("gpt2", None), ("gpt2-medium", None), ("gpt2-large", None), ("gpt2-xl", None),
    ("bigscience/bloom-560m", None), ("allenai/OLMo-1B-hf", None),
    ("RWKV/rwkv-4-169m-pile", None), ("state-spaces/mamba-130m-hf", None),
    ("state-spaces/mamba-370m-hf", None),
]
PROMPTS = [
    "The first thing to understand about", "In the summer of 1994, a small",
    "Scientists have long suspected that the", "She opened the door and found",
    "The report concludes that there is", "According to the latest figures,",
    "One of the most difficult problems in", "After the war ended, many of the",
    "The company announced on Tuesday that", "It is often said that the best",
    "Researchers at the university have developed", "The old man sat by the window and",
]
NEW_TOKENS = 128
NGRAM = 4
OUT = str(_ROOT / "results" / "degeneration_vs_tstar.json")


def t_star(screen_runs, model, thresh=THRESH):
    """Temperature where top-1 share falls through `thresh`, linearly interpolated."""
    pts = []
    for T in SCREEN_TEMPS:
        v = screen_runs.get(f"{model}@{T}")
        if v and "top1_share" in v:
            pts.append((T, v["top1_share"]))
    if len(pts) < 2:
        return None
    for (a, ya), (b, yb) in zip(pts, pts[1:]):
        if ya >= thresh > yb:
            return round(a + (b - a) * (ya - thresh) / (ya - yb), 4)
    return None            # never crosses: either always below, or still above at the top


def rep_stats(ids):
    """Repetition statistics of one continuation, token ids only."""
    n = len(ids)
    if n < NGRAM + 1:
        return None
    grams = [tuple(ids[i:i + NGRAM]) for i in range(n - NGRAM + 1)]
    seen, rep = set(), 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    # longest immediately-repeating block, e.g. "a b a b a b" -> 3
    longest = 1
    for p in range(1, n // 2 + 1):
        run = 1
        for i in range(0, n - 2 * p + 1, p):
            if ids[i:i + p] == ids[i + p:i + 2 * p]:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
    return dict(rep_4=rep / len(grams), distinct_1=len(set(ids)) / n, longest_loop=longest)


def main():
    screen = json.load(open(SCREEN))["runs"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=[m for m, _ in MODELS], prompts=len(PROMPTS), new_tokens=NEW_TOKENS, ngram=NGRAM,
        decoding="greedy, no sampling -- no temperature knob to tune",
        t_star=f"interpolated from the screen at {SCREEN_TEMPS}, threshold {THRESH}",
        primary="does T* correlate with greedy-decoding repetition? Spearman rho, n stated",
        null_means="T* is a property of the out-of-distribution artifact and nothing more -- "
                   "reportable, and it closes the question",
        independence="prompts are short but REAL sentence openings, not two-token fragments; free "
                     "generation shares no machinery with the ring CA",
        resumable="keyed by model")
    runs = res["runs"]
    from transformers import AutoTokenizer, AutoModelForCausalLM

    for name, rev in MODELS:
        if name in runs:
            print(f"  {name}: already done", flush=True); continue
        t0 = time.time()
        try:
            kw = {"revision": rev} if rev else {}
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(dev, torch.float16 if dev != "cpu" else torch.float32)
        except Exception as e:
            print(f"  {name}: LOAD FAILED ({type(e).__name__}: {str(e)[:70]})", flush=True)
            runs[name] = dict(model=name, failed=f"{type(e).__name__}")
            json.dump(res, open(OUT, "w"), indent=1); continue
        stats = []
        for p in PROMPTS:
            try:
                ids = tok(p, return_tensors="pt").input_ids.to(dev)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False,
                                         pad_token_id=tok.eos_token_id or 0)
                cont = out[0, ids.shape[1]:].tolist()
                s = rep_stats(cont)
                if s: stats.append(s)
            except Exception as e:
                print(f"    prompt failed: {type(e).__name__}: {str(e)[:60]}", flush=True)
        if not stats:
            runs[name] = dict(model=name, failed="no usable generations")
        else:
            runs[name] = dict(
                model=name, n_prompts=len(stats), t_star=t_star(screen, name),
                rep_4=round(float(np.mean([s["rep_4"] for s in stats])), 4),
                distinct_1=round(float(np.mean([s["distinct_1"] for s in stats])), 4),
                longest_loop=round(float(np.mean([s["longest_loop"] for s in stats])), 2),
                secs=round(time.time() - t0, 1))
            r = runs[name]
            print(f"  {name:>38} T*={str(r['t_star']):>7}  rep_4={r['rep_4']:.3f}  "
                  f"distinct={r['distinct_1']:.3f}  loop={r['longest_loop']:.1f} "
                  f"({r['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del model, tok
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _spearman(x, y):
    """Rank correlation, and a permutation p-value -- scipy is not a dependency here."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(0)
    null = [abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) for _ in range(20000)]
    return rho, float((np.sum(np.array(null) >= abs(rho)) + 1) / (len(null) + 1))


def analyse(res):
    ok = [v for v in res["runs"].values() if "rep_4" in v]
    melt = [v for v in ok if v["t_star"] is not None]
    flat = [v for v in ok if v["t_star"] is None]
    print(f"\n=== does T* predict greedy-decoding repetition? ===")
    print(f"  {'model':>38} {'T*':>7} {'rep_4':>7} {'distinct':>9} {'loop':>6}")
    for v in sorted(melt, key=lambda v: -v["t_star"]):
        print(f"  {v['model']:>38} {v['t_star']:>7.3f} {v['rep_4']:>7.3f} "
              f"{v['distinct_1']:>9.3f} {v['longest_loop']:>6.1f}")
    print(f"  {'-- no finite T* (never concentrate) --':>38}")
    for v in sorted(flat, key=lambda v: -v["rep_4"]):
        print(f"  {v['model']:>38} {'--':>7} {v['rep_4']:>7.3f} "
              f"{v['distinct_1']:>9.3f} {v['longest_loop']:>6.1f}")

    out = {}
    if len(melt) >= 5:
        rho, p = _spearman([v["t_star"] for v in melt], [v["rep_4"] for v in melt])
        out["spearman_tstar_rep4"] = dict(rho=round(rho, 3), p=round(p, 4), n=len(melt))
        print(f"\n  Spearman rho(T*, rep_4) = {rho:+.3f}  (permutation p={p:.4f}, n={len(melt)})")
        strong = abs(rho) >= 0.6 and p < 0.05
        if strong and rho > 0:
            verdict = (f"T* PREDICTS A KNOWN FAILURE MODE: rho={rho:+.3f} (p={p:.4f}, n={len(melt)}) "
                       f"between the CA's melting temperature and repetition under greedy decoding "
                       f"-- a measurement that shares no machinery with the ring construction. T* "
                       f"stops being a number the probe emits and becomes a cheap scalar with "
                       f"external meaning, obtainable in four settle runs.")
        elif strong:
            verdict = (f"T* ANTI-CORRELATES with repetition: rho={rho:+.3f} (p={p:.4f}). A peaked "
                       f"short-context conditional goes with LESS greedy repetition, which is the "
                       f"opposite of the pre-registered prediction and needs an explanation before "
                       f"it is used for anything.")
        else:
            verdict = (f"NO ASSOCIATION: rho={rho:+.3f} (p={p:.4f}, n={len(melt)}). T* does not "
                       f"predict greedy-decoding repetition, so it remains a property of the "
                       f"out-of-distribution artifact and nothing more. This closes the question "
                       f"rather than leaving it open -- the pre-registered null.")
        if flat:
            fm = float(np.mean([v["rep_4"] for v in flat]))
            mm = float(np.mean([v["rep_4"] for v in melt]))
            verdict += (f" Models that never concentrate average rep_4={fm:.3f} against {mm:.3f} "
                        f"for those that do.")
    else:
        verdict = f"insufficient data: only {len(melt)} models have a finite T*"
    print(f"\n  -> {verdict}")

    res["melting"] = {v["model"]: v for v in melt}
    res["no_finite_tstar"] = {v["model"]: v for v in flat}
    res["analysis"] = out
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Tests whether T* -- the temperature at which the CA's settled lattice stops being "
        "dominated by one token, interpolated from the nineteen-model screen -- predicts neural "
        "text degeneration, an independently studied failure mode. T* is tighter within a family "
        "than the raw share at any fixed temperature and separates families the attractor binary "
        "lumps together, but it is measured in a two-token regime F66 showed to be an artifact, so "
        "on its own it is a property of out-of-distribution behaviour. The anchor shares no "
        "machinery with the ring CA: greedy decoding from short but REAL sentence openings, with "
        "repetition measured as the fraction of repeated 4-grams. Greedy so there is no "
        "temperature knob to tune. A null is pre-registered as closing the question.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
