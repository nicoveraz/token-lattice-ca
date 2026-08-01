"""Is the no-context failure the same thing as low-evidence behaviour? (#92)

THE QUESTION. F62-F69 established that a two-token context makes some models collapse onto a single
filler token, and that this is what the "phase transition" was. That is a statement about an extreme
regime. The natural next question is whether it is the endpoint of something continuous: when a
model has thin evidence -- a rare context, an unfamiliar domain, a short prompt -- does it drift
toward the same fallback?

If it does, the CA degeneracy stops being a curiosity about a broken probe and becomes the visible
end of a general property: **a model with insufficient evidence retreats toward its prior.**

THE MECHANISM THIS TESTS. A two-token context carries almost no information, so p(x | 2 tokens)
should be close to the model's MARGINAL distribution -- what it emits given nothing. If that is
right, then:

  * the CA degeneracy is not mysterious: it is the marginal, iterated;
  * F64's "attention is necessary and corpus determines" may reduce to something simpler -- some
    models have a peaked marginal and others do not, with the corpus setting the marginal and
    attention-free models approaching it differently;
  * and the same retreat should be visible at intermediate evidence, not only at the extreme.

FOUR MEASUREMENTS, EACH ON REAL TEXT EXCEPT WHERE STATED

  1. MARGINAL. p(x | BOS alone). Its top-1 mass and entropy. This is the model's prior.
  2. AGREEMENT. How close is the two-token conditional to that marginal? Reported as total variation
     distance, averaged over real two-token contexts. Small TV means the CA regime is simply
     sampling the prior.
  3. EVIDENCE LADDER. Top-1 mass and entropy of p(x | k real tokens) for k = 1, 2, 4, 8, 16, 32,
     taken from actual text rather than random tokens. This is the continuum: does confidence in
     filler decay smoothly as evidence accumulates, and at what k does it stop mattering?
  4. RARE VS COMMON. The same statistics split by context frequency. Contexts whose final bigram is
     RARE in a reference corpus stand in for "not enough training data". If the low-evidence retreat
     is real, rare contexts should sit further toward the marginal than common ones at the SAME
     length -- which separates "short" from "unfamiliar", two things length alone conflates.

PRE-REGISTERED:
  * Primary: is TV(p(x | 2 real tokens), marginal) small for the models that degenerate in the CA
    and large for those that do not? That would make the degeneracy a marginal-retreat phenomenon
    and give F64 a mechanism instead of a correlation.
  * Secondary: at fixed context length, do RARE contexts sit closer to the marginal than COMMON
    ones? A yes links the CA regime to genuine low-data behaviour, which is the question worth
    answering. A no means short-context collapse and rare-context behaviour are different things,
    and the CA result should not be generalised to data sparsity.
  * A null on either is informative and is reported as such. In particular a null on the secondary
    would mean the CA finding is about PROMPT LENGTH only, and must not be described as being about
    insufficient training data.

WHY THIS IS CHEAP. No generation and no iteration -- one forward pass per context. The expensive
part of every other experiment here was the ring dynamics, which this does not use at all.

Writes results/evidence_falloff.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/evidence_falloff.py
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

# a spread of degenerate and clean models, from the screen
MODELS = [
    ("EleutherAI/pythia-410m", "step143000"), ("EleutherAI/gpt-neo-125M", None),
    ("Qwen/Qwen2.5-0.5B", None), ("ibm-granite/granite-3.0-2b-base", None),
    ("gpt2-medium", None), ("bigscience/bloom-560m", None), ("allenai/OLMo-1B-hf", None),
    ("state-spaces/mamba-130m-hf", None),
]
LENGTHS = [1, 2, 4, 8, 16, 32]
N_CONTEXTS = 160                       # real contexts sampled per length per bin
RARE_QUANTILE = 0.25                   # bottom quartile of final-bigram frequency = "rare"
OUT = str(_ROOT / "results" / "evidence_falloff.json")

CORPUS = """The history of the city begins with a small settlement on the river. Over the following
centuries it grew into a centre of trade, and by the middle of the period its markets were known
across the region. Scientists studying the process have argued that several factors contributed,
though the relative importance of each remains disputed. In a report published last year, the
committee concluded that further work would be required before any firm recommendation could be
made. The building itself was completed in stages, with the eastern wing added considerably later
than the rest. Critics at the time described the design as austere, but opinion has shifted and it
is now regarded as one of the more successful examples of its style. Economic conditions during
those years were unusually volatile, and many of the smaller firms did not survive. Records from
the period are incomplete, which has made it difficult to establish exactly what happened. Modern
analysis suggests that the decline began earlier than previously thought, and that the immediate
causes were less important than the underlying structural weaknesses. Research into the mechanism
continues, with several groups pursuing different approaches, and no consensus has yet emerged."""


def top1_and_entropy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    nz = p[p > 0]
    return float(p.max()), float(-(nz * np.log(nz)).sum())


@torch.no_grad()
def next_probs(model, ids, dev):
    out = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
    return torch.softmax(out, dim=-1).cpu().numpy()


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=[m for m, _ in MODELS], lengths=LENGTHS, n_contexts=N_CONTEXTS,
        rare_quantile=RARE_QUANTILE,
        primary="is TV(p(x | 2 real tokens), marginal) small for CA-degenerate models and large "
                "for clean ones? that would give F64 a mechanism rather than a correlation",
        secondary="at FIXED context length, do rare contexts sit closer to the marginal than "
                  "common ones? yes links the CA regime to genuine low-data behaviour",
        null_meaning="a null on the secondary means the CA finding is about PROMPT LENGTH only "
                     "and must not be described as being about insufficient training data",
        method="one forward pass per context; no generation, no ring dynamics",
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
            print(f"  {name}: LOAD FAILED ({type(e).__name__}: {str(e)[:60]})", flush=True)
            runs[name] = dict(model=name, failed=f"{type(e).__name__}")
            json.dump(res, open(OUT, "w"), indent=1); continue

        ids = tok(CORPUS, return_tensors=None)["input_ids"]
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        marg = next_probs(model, [bos], dev)
        m_top1, m_ent = top1_and_entropy(marg)
        m_tok = tok.decode([int(np.argmax(marg))])

        # bigram frequency over the corpus, to split rare from common at fixed length
        bigrams = collections.Counter(zip(ids[:-1], ids[1:]))
        freqs = np.array([bigrams[(ids[i - 1], ids[i])] for i in range(1, len(ids))])
        cut = np.quantile(freqs, RARE_QUANTILE)

        rec = dict(model=name, marginal_top1=round(m_top1, 4), marginal_entropy=round(m_ent, 4),
                   marginal_token=m_tok, by_length={}, rare_vs_common={})
        rng = np.random.default_rng(0)
        for k in LENGTHS:
            starts = [i for i in range(k, len(ids) - 1)]
            if not starts: continue
            pick = rng.choice(starts, size=min(N_CONTEXTS, len(starts)), replace=False)
            t1s, ents, tvs, rare_t1, common_t1 = [], [], [], [], []
            for i in pick:
                p = next_probs(model, ids[i - k:i], dev)
                a, e = top1_and_entropy(p)
                t1s.append(a); ents.append(e)
                tvs.append(0.5 * float(np.abs(p - marg).sum()))
                (rare_t1 if freqs[i - 1] <= cut else common_t1).append(a)
            rec["by_length"][str(k)] = dict(
                top1=round(float(np.mean(t1s)), 4), entropy=round(float(np.mean(ents)), 4),
                tv_to_marginal=round(float(np.mean(tvs)), 4), n=len(pick))
            if rare_t1 and common_t1:
                rec["rare_vs_common"][str(k)] = dict(
                    rare_top1=round(float(np.mean(rare_t1)), 4),
                    common_top1=round(float(np.mean(common_t1)), 4),
                    gap=round(float(np.mean(rare_t1) - np.mean(common_t1)), 4),
                    n_rare=len(rare_t1), n_common=len(common_t1))
        rec["secs"] = round(time.time() - t0, 1)
        runs[name] = rec
        b2 = rec["by_length"].get("2", {})
        print(f"  {name:>38} marg_top1={m_top1:.3f} ({m_tok!r})  "
              f"TV@k=2={b2.get('tv_to_marginal', float('nan')):.3f}  "
              f"top1@k=2={b2.get('top1', float('nan')):.3f} ({rec['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del model, tok
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def t_star(screen_runs, model, thresh=0.40):
    pts = [(T, screen_runs[f"{model}@{T}"]["top1_share"]) for T in SCREEN_TEMPS
           if f"{model}@{T}" in screen_runs and "top1_share" in screen_runs[f"{model}@{T}"]]
    if len(pts) < 2: return None
    for (a, ya), (b, yb) in zip(pts, pts[1:]):
        if ya >= thresh > yb:
            return round(a + (b - a) * (ya - thresh) / (ya - yb), 4)
    return "censored_above" if pts[-1][1] >= thresh else None


def analyse(res):
    screen = json.load(open(SCREEN))["runs"] if os.path.exists(SCREEN) else {}
    ok = [v for v in res["runs"].values() if "by_length" in v]
    for v in ok:
        v["ca_degenerate"] = t_star(screen, v["model"]) is not None

    print(f"\n=== 1-2. is the two-token conditional just the MARGINAL? ===")
    print(f"  {'model':>38} {'CA deg':>7} {'marg top1':>10} {'marg tok':>10} {'TV@k=2':>8} {'TV@k=32':>8}")
    for v in sorted(ok, key=lambda v: -v["marginal_top1"]):
        b2, b32 = v["by_length"].get("2", {}), v["by_length"].get("32", {})
        print(f"  {v['model']:>38} {str(v['ca_degenerate']):>7} {v['marginal_top1']:>10.3f} "
              f"{v['marginal_token']!r:>10} {b2.get('tv_to_marginal', 0):>8.3f} "
              f"{b32.get('tv_to_marginal', 0):>8.3f}")

    print(f"\n=== 3. evidence ladder: top-1 mass vs context length (real text) ===")
    print(f"  {'model':>38} " + " ".join(f"k={k:<5}" for k in LENGTHS))
    for v in sorted(ok, key=lambda v: -v["by_length"].get("2", {}).get("top1", 0)):
        print(f"  {v['model']:>38} " +
              " ".join(f"{v['by_length'].get(str(k), {}).get('top1', 0):<7.3f}" for k in LENGTHS))

    print(f"\n=== 4. RARE vs COMMON contexts at the same length (top-1 mass) ===")
    print(f"  {'model':>38} " + " ".join(f"k={k:<6}" for k in LENGTHS))
    gaps_all = []
    for v in sorted(ok, key=lambda v: v["model"]):
        row = []
        for k in LENGTHS:
            g = v["rare_vs_common"].get(str(k), {}).get("gap")
            row.append(f"{g:+.3f} " if g is not None else "  --   ")
            if g is not None and k >= 2: gaps_all.append(g)
        print(f"  {v['model']:>38} " + " ".join(row))

    deg = [v for v in ok if v["ca_degenerate"]]
    cln = [v for v in ok if not v["ca_degenerate"]]
    parts = []
    if deg and cln:
        dm = float(np.mean([v["marginal_top1"] for v in deg]))
        cm = float(np.mean([v["marginal_top1"] for v in cln]))
        dt = float(np.mean([v["by_length"].get("2", {}).get("tv_to_marginal", np.nan) for v in deg]))
        ct = float(np.mean([v["by_length"].get("2", {}).get("tv_to_marginal", np.nan) for v in cln]))
        if dm > cm + 0.15:
            parts.append(f"THE MARGINAL EXPLAINS THE SPLIT: CA-degenerate models have a marginal "
                         f"top-1 of {dm:.3f} against {cm:.3f} for clean ones. The CA degeneracy is "
                         f"the model's PRIOR, iterated -- which turns F64's 'attention necessary, "
                         f"corpus determines' from a correlation into a mechanism, since the corpus "
                         f"sets the marginal.")
        else:
            parts.append(f"The marginal does NOT explain the split: {dm:.3f} vs {cm:.3f} top-1 for "
                         f"degenerate vs clean. Whatever makes a model collapse under two tokens is "
                         f"not simply that its prior is peaked.")
        parts.append(f"TV to the marginal at k=2 is {dt:.3f} (degenerate) vs {ct:.3f} (clean).")
    if gaps_all:
        mg = float(np.mean(gaps_all))
        pos = float(np.mean([g > 0 for g in gaps_all]))
        if mg > 0.02 and pos > 0.6:
            parts.append(f"RARE CONTEXTS SIT CLOSER TO THE FALLBACK: at matched context length, "
                         f"rare contexts carry {mg:+.3f} more top-1 mass than common ones "
                         f"({pos*100:.0f}% of cells positive). So the two-token collapse and "
                         f"thin-evidence behaviour are the same retreat, seen at different "
                         f"strengths -- the CA finding does generalise to data sparsity.")
        else:
            parts.append(f"NO RARE/COMMON EFFECT: the gap averages {mg:+.3f} with only "
                         f"{pos*100:.0f}% of cells positive. Short context and unfamiliar context "
                         f"are NOT the same thing here, so the CA result is about PROMPT LENGTH and "
                         f"must not be described as being about insufficient training data.")
    verdict = " ".join(parts) if parts else "insufficient data"
    print(f"\n  -> {verdict}")

    res["analysis"] = {v["model"]: v for v in ok}
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Asks whether the two-token collapse (F62-F69) is the extreme end of a general "
        "low-evidence retreat toward the model's prior, or a phenomenon of prompt length alone. "
        "Four measurements: the marginal p(x | BOS); the total-variation distance from the "
        "two-token conditional to it; an evidence ladder over real contexts of 1 to 32 tokens; and "
        "the same statistics split by context rarity, so 'short' is separated from 'unfamiliar' -- "
        "two things context length alone conflates. If rare contexts sit closer to the fallback at "
        "matched length, the CA finding generalises to data sparsity; if not, it is about prompt "
        "length only and must not be described otherwise. One forward pass per context, no "
        "generation and no ring dynamics.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
