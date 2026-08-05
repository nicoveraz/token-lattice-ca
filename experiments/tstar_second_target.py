"""Does T* predict a SECOND degeneration target, and is `modal` really dead? (F86/F92 follow-up)

WHY THIS IS THE ONE ITEM WORTH RUNNING. `paper/plan_paper3.md` recommends exactly this and nothing
else before the third-paper decision, because it is the only measurement that can change that
decision rather than decorate it. It now carries a second job as well:

  F86  rho(T*, rep_4) = +0.833, n = 8 families, p = 0.0137 -- the anchor, on ONE behavioural
       target measured under ONE decoder (greedy). A reviewer's first ask is a second target.
  F92  On those same 8 families the STATIC argmax census does not predict rep_4 (fix -0.12,
       cyc +0.12) while T* does -- the deflationary K1 test run on the anchor. But `modal`
       came in at rho = +0.595, p = 0.13: NOT dead, merely underpowered. If `modal` tracks a
       second target too, the cheap static probe replaces the ring on the axis that matters
       and F92's reading inverts.

So this run measures new targets and re-tests EVERY predictor against them on matched rows.

THE SECOND TARGET, chosen so it is not rep_4 wearing a hat. rep_4 is 4-gram repetition under
GREEDY decoding. Repeating it at another n-gram order would be a restatement. Two genuinely
different quantities are measured instead:

  nucleus_rep_4   the same repetition metric under NUCLEUS sampling (top-p 0.95, T = 1.0) --
                  a different DECODER. Holtzman et al. introduced nucleus sampling precisely to
                  fix greedy degeneration, so a T* that predicts repetition even under nucleus
                  is predicting something the standard mitigation does not remove. This is the
                  headline second target.
  distinct_4      distinct-4 over pooled continuations: a DIVERSITY measure rather than a
                  self-repetition measure. Low distinct-4 with low rep_4 means dull-but-varied
                  text, which repetition metrics cannot distinguish.
  selfbleu_sample the diversity ACROSS continuations from different prompts, not within one --
                  mode collapse rather than looping. Uses the same 4-gram machinery.

Every target is measured on the SAME models, SAME prompts, and (for the greedy arm) the SAME
protocol as `degeneration_vs_tstar`, whose `rep_stats` and `PROMPTS` are imported rather than
reimplemented, so the new numbers sit on the old axis without a reimplementation confound.

PRE-REGISTERED, before any number exists:
  Primary     rho(T*, nucleus_rep_4) over families with finite T*, permutation p. CORROBORATES
              if same sign and p < 0.05; DIRECTIONALLY CONSISTENT if same sign, p >= 0.05;
              FAILS if the sign flips. A failure demotes F86 to one-target-only and, per
              plan_paper3, makes branch B automatic.
  Deflation   the same correlation for `modal`, `fix`, `cyc` and `top1@0.02` on the SAME rows.
              If any static predictor matches or beats T* on the second target, F92's
              dissociation does not generalise and the ring's necessity is not established.
  Secondary   distinct_4 and self-BLEU as descriptive columns; no test is registered for them
              at this n, and none will be run post hoc.
  Multiplicity BH-FDR over the predictor family {T*, modal, fix, cyc, top1} on the primary
              target, via dev_transition_phase3.bh_fdr (one implementation, imported).

Writes results/tstar_second_target.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/tstar_second_target.py
        (resumable per model)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from degeneration_vs_tstar import rep_stats, PROMPTS, NEW_TOKENS, NGRAM   # one implementation
from dev_transition_phase3 import bh_fdr

OUT = str(_ROOT / "results" / "tstar_second_target.json")
BAND = _ROOT / "results" / "band_screen.json"
CENSUS = _ROOT / "results" / "argmax_census_hardened.json"

TOP_P, SAMPLE_T = 0.95, 1.0
N_SAMPLE = 4                 # nucleus continuations per prompt (greedy needs only one)
SEED = 20260805


def distinct_n(seqs, n=NGRAM):
    """Distinct-n over pooled continuations: diversity, not self-repetition."""
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def self_bleu_proxy(seqs, n=NGRAM):
    """Mean pairwise n-gram overlap ACROSS continuations -- mode collapse, not looping.

    A cheap symmetric Jaccard rather than corpus BLEU: the quantity of interest is whether
    different prompts produce the same material, and Jaccard on n-gram sets answers that with
    no smoothing conventions to get wrong.
    """
    sets = [set(tuple(s[i:i + n]) for i in range(len(s) - n + 1)) for s in seqs]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return None
    vals = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(vals))


def measure(name, res):
    runs = res["runs"]
    if name in runs:
        return
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).eval()
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(dev, torch.float16 if dev != "cpu" else torch.float32)
    except Exception as e:
        runs[name] = dict(model=name, failed=type(e).__name__)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"  {name}: LOAD FAILED ({type(e).__name__})", flush=True)
        return
    torch.manual_seed(SEED)
    greedy, nucleus = [], []
    for p in PROMPTS:
        try:
            ids = tok(p, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False,
                                   pad_token_id=tok.eos_token_id or 0)
                greedy.append(g[0, ids.shape[1]:].tolist())
                for _ in range(N_SAMPLE):
                    s = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=True,
                                       top_p=TOP_P, temperature=SAMPLE_T, top_k=0,
                                       pad_token_id=tok.eos_token_id or 0)
                    nucleus.append(s[0, ids.shape[1]:].tolist())
        except Exception as e:
            print(f"    prompt failed: {type(e).__name__}", flush=True)
    gs = [rep_stats(c) for c in greedy]
    ns = [rep_stats(c) for c in nucleus]
    gs = [x for x in gs if x]; ns = [x for x in ns if x]
    if not (gs and ns):
        runs[name] = dict(model=name, failed="no usable generations")
    else:
        runs[name] = dict(
            model=name,
            greedy_rep_4=round(float(np.mean([x["rep_4"] for x in gs])), 4),
            nucleus_rep_4=round(float(np.mean([x["rep_4"] for x in ns])), 4),
            distinct_4=round(distinct_n(nucleus), 4),
            self_bleu=round(self_bleu_proxy(nucleus) or float("nan"), 4),
            n_greedy=len(gs), n_nucleus=len(ns), secs=round(time.time() - t0, 1))
        r = runs[name]
        print(f"  {name:44s} greedy {r['greedy_rep_4']:.3f}  nucleus {r['nucleus_rep_4']:.3f}  "
              f"distinct4 {r['distinct_4']:.3f}  ({r['secs']:.0f}s)", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_p(a, b, rho, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    return (sum(abs(spearman(a, rng.permutation(b))) >= abs(rho) for _ in range(n)) + 1) / (n + 1)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    band = json.load(open(BAND))
    fams = band["families"]
    res["_preregistration"] = dict(
        models={f: v["model"] for f, v in fams.items()},
        top_p=TOP_P, sample_T=SAMPLE_T, n_sample=N_SAMPLE, new_tokens=NEW_TOKENS, ngram=NGRAM,
        primary="rho(T*, nucleus_rep_4) over finite-T* families; CORROBORATES if same sign and "
                "p<0.05, DIRECTIONALLY CONSISTENT if same sign p>=0.05, FAILS if the sign flips",
        deflation="the same correlation for modal/fix/cyc/top1 on the SAME rows; any static "
                  "predictor matching or beating T* means F92's dissociation does not generalise",
        secondary="distinct_4 and self-BLEU descriptive only; no test registered, none run post hoc",
        multiplicity="BH-FDR over the predictor family on the primary target",
        failure_meaning="a sign flip demotes F86 to one-target-only and makes branch B automatic")
    json.dump(res, open(OUT, "w"), indent=1)
    for f, v in sorted(fams.items()):
        measure(v["model"], res)
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    band = json.load(open(BAND))
    cen = json.load(open(CENSUS))["runs"]
    ms = {}
    for k, v in cen.items():
        if "|s" in k and "failed" not in k:
            ms.setdefault(k.split("|")[0], []).append(v)
    rows = []
    for f, v in band["families"].items():
        m = v["model"]
        r = res["runs"].get(m)
        if not r or r.get("failed") or m not in ms:
            continue
        rows.append(dict(
            fam=f, tstar=v["tstar"], greedy=v["rep_4"], nucleus=r["nucleus_rep_4"],
            distinct=r["distinct_4"], selfbleu=r["self_bleu"], top1=v["top1_low"],
            fix=float(np.mean([q["fixed_point_fraction"] for q in ms[m]])),
            cyc=float(np.mean([q["cyclic_fraction"] for q in ms[m]])),
            modal=float(np.mean([q["modal_endpoint_share"] for q in ms[m]]))))
    sub = [r for r in rows if r["tstar"] is not None]
    print(f"\n=== {len(rows)} families measured; {len(sub)} with finite T* (the primary's rows) ===")
    print(f"  {'family':22s} {'T*':>7} {'greedy':>7} {'nucleus':>8} {'distinct4':>10} {'selfBLEU':>9}")
    for r in sorted(rows, key=lambda x: -(x["nucleus"])):
        print(f"  {r['fam']:22s} {str(round(r['tstar'],3)) if r['tstar'] else '--':>7} "
              f"{r['greedy']:7.3f} {r['nucleus']:8.3f} {r['distinct']:10.3f} {r['selfbleu']:9.3f}")

    if len(sub) < 4:
        res["verdict"] = "NOT DECIDABLE -- fewer than 4 finite-T* families measured."
        print("\n  ->", res["verdict"]); return
    y = np.array([r["nucleus"] for r in sub])
    preds = {}
    for name in ("tstar", "modal", "fix", "cyc", "top1"):
        x = np.array([r[name] for r in sub])
        rho = spearman(x, y)
        preds[name] = dict(rho=round(rho, 3), p_raw=round(perm_p(x, y, rho), 4))
    keys = list(preds)
    for k, adj in zip(keys, bh_fdr([preds[k]["p_raw"] for k in keys])):
        preds[k]["p_bh"] = round(adj, 4)
    g = np.array([r["greedy"] for r in sub])
    rho_g = spearman(np.array([r["tstar"] for r in sub]), g)

    print(f"\n  predictors of NUCLEUS rep_4 (n={len(sub)}, same rows, BH over the family):")
    for k, v in sorted(preds.items(), key=lambda kv: -abs(kv[1]["rho"])):
        print(f"    {k:6s} rho={v['rho']:+.3f}  p={v['p_raw']:.4f}  p_BH={v['p_bh']:.4f}")
    print(f"  (T* vs GREEDY rep_4 on these same rows, for reference: rho={rho_g:+.3f})")

    t = preds["tstar"]
    same_sign = (t["rho"] > 0) == (rho_g > 0)
    word = ("CORROBORATES" if same_sign and t["p_bh"] < 0.05 else
            "DIRECTIONALLY CONSISTENT" if same_sign else "FAILS")
    best_static = max((k for k in preds if k != "tstar"), key=lambda k: abs(preds[k]["rho"]))
    bs = preds[best_static]
    deflated = abs(bs["rho"]) >= abs(t["rho"])

    parts = [
        f"SECOND TARGET {word}: rho(T*, nucleus_rep_4) = {t['rho']:+.3f}, p = {t['p_raw']:.4f} "
        f"(p_BH = {t['p_bh']:.4f}) over n = {len(sub)} finite-T* families, against "
        f"{rho_g:+.3f} for greedy rep_4 on the same rows. "
        + ("F86's anchor is two-legged: T* predicts degeneration under the decoder introduced to "
           "FIX greedy degeneration, so it is not an artifact of greedy decoding."
           if same_sign and t["p_bh"] < 0.05 else
           "The sign agrees but significance does not survive at this n -- reported as "
           "directionally consistent, which is weaker than F86 and must be stated that way."
           if same_sign else
           "THE SIGN FLIPS. F86 is a one-target result about greedy decoding specifically, not "
           "about degeneration, and per plan_paper3 branch B becomes automatic.")]
    parts.append(
        f"DEFLATION: the strongest static predictor is {best_static} at rho = {bs['rho']:+.3f} "
        f"(p_BH = {bs['p_bh']:.4f}). "
        + (f"It matches or beats T*, so F92's dissociation does NOT generalise to a second "
           f"target and the cheap static census is the better instrument on this axis -- the "
           f"ring's necessity is not established." if deflated else
           f"T* still leads, so F92's dissociation holds on a second, independent target: the "
           f"ring carries behavioural information the static conditional does not."))
    parts.append(
        "Secondary columns (distinct-4, self-BLEU) are descriptive; no test was registered for "
        "them and none was run. n is 8-ish by construction -- families without an attractor have "
        "no T* -- so this inherits F86's fragility and does not repair it.")
    res["analysis"] = dict(rows=rows, predictors=preds, n_primary=len(sub),
                           tstar_vs_greedy_same_rows=round(rho_g, 3),
                           verdict_word=word, best_static=best_static, deflated=bool(deflated))
    res["verdict"] = " ".join(parts)
    print(f"\n  -> {res['verdict']}")
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "The one measurement plan_paper3 recommends before the third-paper decision, doing two "
        "jobs: testing whether F86's anchor survives a SECOND behavioural target (nucleus "
        "sampling -- a different decoder, and the one introduced to fix greedy degeneration), and "
        "re-testing F92's deflationary dissociation, where `modal` survived at rho=+0.60 p=0.13 "
        "and could still invert the reading. rep_stats/PROMPTS/NEW_TOKENS imported from "
        "degeneration_vs_tstar so the new targets sit on the old axis; bh_fdr imported. Every "
        "predictor is compared on the SAME rows. A sign flip on the primary makes branch B "
        "automatic, which was registered before the data existed.")


if __name__ == "__main__":
    main()
