"""Is F117's "compliance" a CONSTRUCT, or is it IFEval? The second indicator paper 2 needs.

THE BLOCKER, STATED PLAINLY. F117 reports that the attractor share is compliance-selective:
selectivity +0.53 at p = 0.004, the share loading on IFEval (+0.73) and on none of BBH, GPQA, MUSR,
MMLU-PRO or MATH. F120 then showed T* is NOT selective, which sharpened it further. Every one of
those statements rests on COMPLIANCE being a single column. With one indicator there is no way to
tell "the share tracks compliance" from "the share tracks IFEval" -- a construct measured by one
instrument is not yet a construct, and this is the load-bearing weakness in paper 2's headline.

So: a SECOND compliance measure, built independently, run locally on the same ten models, and put
through the same statistic.

WHAT MAKES IT INDEPENDENT. The item pool is ours: twelve short open-ended prompts crossed with ten
programmatically verifiable output constraints, none of them drawn from IFEval's set. The
constraint FAMILY is deliberately the same -- verifiable instruction following is what "compliance"
means here, and a second measure of a different construct would answer a different question -- but
no prompt, no keyword and no phrasing is shared. Scoring is a Python predicate per item: no judge
model, no rubric, nothing that could import the first measure's idiosyncrasies through a back door.

THESE ARE BASE MODELS, WHICH IS THE WHOLE DIFFICULTY. None of the ten is instruction-tuned, so
strict compliance will be low across the board and the real risk is not a wrong answer but NO ROOM
TO VARY -- this project's own recurring defect class, arriving in the measure rather than in the
analysis. Three things address it, all registered before the run: a LOOSE score alongside the strict
one (IFEval has the same split for the same reason), a per-constraint-type breakdown so a pinned
type can be identified rather than averaged in, and an explicit dynamic-range gate that returns
NOT_DECIDABLE rather than a correlation if the measure turns out to be pinned.

PRE-REGISTERED:
  GATE 0    DYNAMIC RANGE, before anything else. The new measure's across-model span must exceed
            RANGE_K times its own binomial noise floor (sqrt(p(1-p)/n_items)). If the ten models do
            not separate on it, no correlation computed from it means anything and the run returns
            NOT_DECIDABLE. Reported per scoring mode (strict, loose) and per constraint type.
  RUNG      CONVERGENCE with IFEval: rho(new measure, IFEval) over the ten models must reach
            CONVERGE. This is what licenses calling it a compliance measure at all -- the same
            shape as F120's anchor rung, where an imported protocol had to reproduce F86's +0.833
            before its null was read. It is NOT circular: convergence establishes commensurability,
            and the PRIMARY below asks a different question of a different quantity.
  PRIMARY   REPLACE IFEval with the new measure and recompute F117's selectivity statistic,
            unchanged: |rho(readout, compliance)| - max|rho(readout, correctness)|, against the
            same null that permutes the READOUT across models. Registered reading: selectivity
            survives on an independent indicator -> compliance is a construct and F117 stands;
            selectivity vanishes -> F117 is a correlation with IFEval specifically, and paper 2's
            headline must be rewritten to say so.
  SECOND    ADD the new measure alongside IFEval (compliance = both), which is the two-indicator
            design paper 2 would actually use.
  CONTROL   params_b must not be selective, exactly as in F117. If it is, the statistic is not
            measuring selectivity here either and nothing is read.
  BOUNDARY  ten base models of 1.7-3.2B, one constraint family, greedy decoding, one prompt format.
            A measure that converges with IFEval here says compliance is a construct AT THIS SCALE
            AND IN THIS FAMILY; a base model's instruction following is not an instruction-tuned
            model's.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, os, re, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from ranking import rank as _rk
from provenance import stamp, rel
from gatecheck import resolves_units, independence_report

OUT = str(_ROOT / "results" / "compliance_second_measure.json")
BENCH = str(_ROOT / "results" / "band_benchmark_range.json")
SCREEN = str(_ROOT / "results" / "band_screen.json")

MODELS = ["bigcode/starcoder2-3b", "bigscience/bloom-3b", "EleutherAI/gpt-neo-2.7B",
          "google/gemma-2-2b", "HuggingFaceTB/SmolLM-1.7B", "kyutai/helium-1-preview-2b",
          "meta-llama/Llama-3.2-3B", "Qwen/Qwen1.5-1.8B", "stabilityai/stablelm-3b-4e1t",
          "tiiuae/Falcon3-1B-Base"]
CORRECTNESS = ["BBH", "GPQA", "MUSR", "MMLU-PRO", "MATH Lvl 5"]
TEMPS = [0.02, 0.2, 0.436, 0.7]
N_PERM = 10000
MAX_NEW = 96
RANGE_K = 2.0
CONVERGE = 0.5
# E[range] of k standard normals, k=10. The reference a span must be compared against: a gate
# that asks span >= 2 SD is passed BY PURE NOISE, since noise already delivers ~3.08 SD.
E_SPAN_K = 3.08

# Twelve short prompts a base model can continue without domain knowledge, none from IFEval.
PROMPTS = [
    "Describe the water cycle.",
    "Explain why the sky appears blue.",
    "Write a short paragraph about the history of paper.",
    "Summarise what a volcano is.",
    "Explain what causes the seasons on Earth.",
    "Describe how bread is made.",
    "Explain the purpose of a public library.",
    "Describe what happens during a thunderstorm.",
    "Explain how a bicycle works.",
    "Describe the life cycle of a butterfly.",
    "Explain what a desert is.",
    "Describe how honey is produced.",
]


def _words(t):
    return re.findall(r"[A-Za-z']+", t)


# Each constraint is (instruction text, predicate). The predicates are total functions of the
# response text: no model, no rubric, no tolerance parameter chosen after seeing an answer.
CONSTRAINTS = {
    "all_caps": ("Write your entire response in capital letters.",
                 lambda t: bool(re.search(r"[A-Za-z]", t)) and t.upper() == t),
    "lowercase": ("Write your entire response in lowercase letters only.",
                  lambda t: bool(re.search(r"[A-Za-z]", t)) and t.lower() == t),
    "no_commas": ("Do not use any commas in your response.",
                  lambda t: bool(t.strip()) and "," not in t),
    "include_word": ('Include the word "lantern" somewhere in your response.',
                     lambda t: "lantern" in t.lower()),
    "forbid_word": ('Do not use the word "the" anywhere in your response.',
                    lambda t: bool(t.strip()) and "the" not in [w.lower() for w in _words(t)]),
    "min_words": ("Write at least 50 words.", lambda t: len(_words(t)) >= 50),
    "end_phrase": ('Finish your response with the exact phrase "That is all."',
                   lambda t: t.strip().rstrip('"').rstrip().endswith("That is all.")),
    "three_bullets": ('Answer using exactly three bullet points, each line starting with "* ".',
                      lambda t: len([l for l in t.splitlines() if l.strip().startswith("* ")]) == 3),
    "wrap_quotes": ("Wrap your entire response in double quotation marks.",
                    lambda t: t.strip().startswith('"') and t.strip().endswith('"')
                    and len(t.strip()) > 2),
    "title_brackets": ("Begin your response with a title enclosed in double angular brackets, "
                       "like <<Title>>.",
                       lambda t: bool(re.match(r"\s*<<[^<>]+>>", t))),
}


def loosen(t):
    """IFEval's loose idea, implemented for this pool: strip wrappers a base model adds anyway.

    Removes markdown fences, a leading echo of the prompt scaffolding, and a first or last line
    that is pure boilerplate. It cannot rescue a response that ignored the constraint -- it only
    removes text the model wrapped AROUND a compliant answer, which is the failure mode that makes
    strict scoring pin at zero on base models.
    """
    s = t.strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    s = re.sub(r"^(Response|Answer|Output)\s*:\s*", "", s, flags=re.I)
    lines = [l for l in s.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def prompt_for(base, instr):
    return f"Instruction: {base} {instr}\nResponse:"


def generate(mdl, tok, texts, device):
    import torch
    out = []
    for p in texts:
        enc = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            g = mdl.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out.append(tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def score(responses):
    """Per-item pass/fail under both scoring modes, keyed by constraint type."""
    strict, loose = {}, {}
    for (ctype, _pi), text in responses.items():
        ok_s = CONSTRAINTS[ctype][1](text)
        ok_l = CONSTRAINTS[ctype][1](loosen(text))
        strict.setdefault(ctype, []).append(bool(ok_s))
        loose.setdefault(ctype, []).append(bool(ok_s or ok_l))
    return strict, loose


def _rho(a, b):
    ra, rb = _rk(a), _rk(b)
    if not (np.isfinite(ra).all() and np.isfinite(rb).all()):
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def selectivity(x, rows, compliance, correctness, rng):
    """F117's statistic, unchanged, so the two runs are commensurable."""
    m = {b: _rho(x, [r[b] for r in rows]) for b in compliance + correctness}
    comp = max(abs(m[b]) for b in compliance)
    corr = max(abs(m[b]) for b in correctness)
    obs = comp - corr
    xa = np.asarray(x, float)
    null = []
    for _ in range(N_PERM):
        xp = rng.permutation(xa)
        mp = {b: _rho(xp, [r[b] for r in rows]) for b in compliance + correctness}
        null.append(max(abs(mp[b]) for b in compliance)
                    - max(abs(mp[b]) for b in correctness))
    return dict(rhos={k: round(v, 4) for k, v in m.items()},
                compliance_max=round(comp, 4), correctness_max=round(corr, 4),
                selectivity=round(obs, 4),
                perm_p=round(float(np.mean(np.array(null) >= obs - 1e-12)), 4))


def analyse(res):
    parts = {}
    cov = {c["model"]: c for c in json.load(open(BENCH))["covered"].values()}
    runs = json.load(open(SCREEN))["runs"]
    prof = {}
    for v in runs.values():
        if v.get("arm") == "temp" and "top1" in v:
            prof.setdefault((v["model"], v["T"]), []).append(v["top1"])
    top1 = {k: float(np.mean(x)) for k, x in prof.items()}

    scored = {m: v for m, v in res["models"].items() if "strict" in v}
    n_items = len(PROMPTS) * len(CONSTRAINTS)
    lines, analysis = [], {}

    for mode in ("strict", "loose"):
        vals = {m: v[mode]["overall"] for m, v in scored.items()}
        if len(vals) < 4:
            lines.append(f"{mode.upper()}: only {len(vals)} models scored -- nothing read.")
            continue
        # GATE 0, CORRECTED. THE FIRST VERSION OF THIS GATE WAS ITSELF VACUOUS, which is this
        # project's own defect class reproduced inside the guard written to prevent it. It compared
        # the across-model SPAN to a single model's binomial SE and required 2x. The span of k
        # draws from PURE noise is about 3.08 SD at k=10, so that gate passes noise by
        # construction -- it can only fail if the measure is *more* degenerate than noise. It duly
        # "passed" at 2.27x on a measure that resolves nothing.
        #
        # Two things were wrong and both are fixed here:
        #   (i)  the reference must be the expected span UNDER NOISE (E_SPAN_K x SD), not one SD;
        #   (ii) the noise scale must be CLUSTER-aware. The 120 items are 10 constraint types x 12
        #        prompts, and within a type a model succeeds or fails near-uniformly, so the
        #        independent unit is the TYPE. gatecheck.units measures exactly this and was not
        #        used the first time; the design effect is 5-10x, so the item-level SE understates
        #        the noise by a factor of ~3 in SD.
        # The decisive quantity is RELIABILITY: the share of observed across-model variance that is
        # not noise. Reliability <= 0 means the models are not resolved at all.
        obs = np.array(list(vals.values()), float)
        per_model_se, icc_rows, icc_units = [], [], []
        for m in vals:
            bt = [scored[m][mode]["by_type"][t] for t in sorted(CONSTRAINTS)]
            per_model_se.append(float(np.std(bt, ddof=1) / np.sqrt(len(bt))))
            for t in sorted(CONSTRAINTS):
                icc_rows.append(scored[m][mode]["by_type"][t])
                icc_units.append(t)
        var_noise = float(np.mean(np.square(per_model_se)))
        var_obs = float(obs.var(ddof=1))
        # gatecheck.resolves_units now IS this check -- the incident below is the reason it exists,
        # so the script uses the package rather than keeping a private copy of the fix.
        rng_rep = resolves_units(list(vals.values()), noise_sd=per_model_se,
                                 name=f"the new measure ({mode})")
        noise_span = float(rng_rep.stats["expected_noise_span"])
        reliability = float(rng_rep.stats["reliability"])
        unit = independence_report(icc_rows, icc_units, unit_name="constraint type")
        analysis.setdefault("_gate0_detail", {})[mode] = dict(
            var_obs=round(var_obs, 8), var_noise_cluster=round(var_noise, 8),
            reliability=round(reliability, 4), observed_span=round(float(obs.max() - obs.min()), 4),
            expected_span_under_noise=round(noise_span, 4),
            icc=round(float(unit.icc), 4), effective_n=round(float(unit.effective_n), 1),
            n_obs=len(icc_rows))
        lines.append(
            f"{mode.upper()} -- GATE 0 (corrected): observed across-model span "
            f"{obs.max() - obs.min():.4f} against the span PURE NOISE would produce, {noise_span:.4f} "
            f"({E_SPAN_K:g} x the cluster-level SD). Reliability = "
            f"1 - var_noise/var_obs = {reliability:+.3f}; the constraint type is the independent "
            f"unit (ICC {unit.icc:.2f}, effective n {unit.effective_n:.0f} of {len(icc_rows)}). "
            + ("The measure does NOT resolve these models: its across-model variance is smaller "
               "than the noise it is made of, so every correlation computed from it is attenuated "
               "to zero and NOTHING below can be read -- including the convergence rung, whose "
               "failure therefore says nothing about whether compliance is a construct."
               if reliability <= 0 else
               "The measure resolves the models above its own clustered noise."))
        if not rng_rep.usable:
            analysis[mode] = dict(scores=vals, reliability=round(reliability, 4),
                                  gate0_passes=False,
                                  status="NOT_DECIDABLE: measure does not resolve the models")
            continue
        rows, x = [], []
        for m, v in vals.items():
            c = cov.get(m)
            if not c:
                continue
            rows.append(dict(model=m, params=c["params_b"], NEW=v, **c["scores"]))
            x.append(v)
        conv = _rho([r["NEW"] for r in rows], [r["IFEval"] for r in rows])
        analysis[mode] = dict(scores=vals, mean=round(p, 4), noise_floor=round(floor, 4),
                              range=rng_rep.block(), convergence_with_ifeval=round(conv, 4),
                              n=len(rows))
        lines.append(
            f"{mode.upper()} -- GATE 0 (dynamic range): {rng_rep.reason}. "
            + ("" if rng_rep.usable else
               "The measure is PINNED across models, so no correlation computed from it can carry "
               "a verdict and this mode returns NOT_DECIDABLE. ")
            + f"RUNG (convergence with IFEval): rho = {conv:+.3f} against a floor of {CONVERGE}. "
            + ("Convergent, so this is measuring the same construct and the comparison is licensed."
               if np.isfinite(conv) and conv >= CONVERGE else
               "NOT convergent -- either this is not a compliance measure or compliance is not a "
               "stable construct on base models at this scale. Either way the PRIMARY below is not "
               "read for this mode."))
        if not rng_rep.usable or not (np.isfinite(conv) and conv >= CONVERGE):
            continue

        g = np.random.default_rng(0)
        readouts = {}
        for T in TEMPS:
            xs = [top1.get((r["model"], T)) for r in rows]
            if any(v is None for v in xs):
                continue
            readouts[f"top1@{T}"] = xs
        readouts["params"] = [r["params"] for r in rows]
        prim, second = {}, {}
        for rd, xs in readouts.items():
            prim[rd] = selectivity(xs, rows, ["NEW"], CORRECTNESS, g)
            second[rd] = selectivity(xs, rows, ["NEW", "IFEval"], CORRECTNESS, g)
        analysis[mode]["primary_replace_ifeval"] = prim
        analysis[mode]["second_two_indicator"] = second
        ctrl = prim.get("params", {})
        hits = [rd for rd, v in prim.items() if rd != "params" and v["perm_p"] < 0.05]
        lines.append(
            f"{mode.upper()} PRIMARY (IFEval REPLACED by the new measure): "
            + "; ".join(f"{rd} sel={v['selectivity']:+.3f} p={v['perm_p']:.4f}"
                        for rd, v in prim.items() if rd != "params")
            + f". CONTROL params sel={ctrl.get('selectivity')} p={ctrl.get('perm_p')}. "
            + ("The negative control is itself selective, so the statistic is not measuring "
               "selectivity on this indicator and nothing is read."
               if ctrl.get("perm_p", 1) < 0.05 else
               f"Selectivity survives on an INDEPENDENT compliance indicator at {hits}: compliance "
               f"is a construct rather than an IFEval idiosyncrasy, and F117 stands."
               if hits else
               "Selectivity does NOT survive on an independent compliance indicator. F117's result "
               "is a correlation with IFEval specifically, and paper 2's headline has to say so."))
        lines.append(
            f"{mode.upper()} SECOND (two-indicator compliance, the design paper 2 would use): "
            + "; ".join(f"{rd} sel={v['selectivity']:+.3f} p={v['perm_p']:.4f}"
                        for rd, v in second.items() if rd != "params") + ".")

    # IS THE LOOSE ARM DOING ANYTHING? It was added because base models wrap a compliant answer in
    # boilerplate and strict scoring then pins at zero. If it never rescues an item, there is ONE
    # scoring mode here, not two, and reporting two identical columns would imply two measurements.
    # The loosener is NOT retuned in response to this: making it more aggressive after seeing the
    # data is how a noise floor gets converted into signal, and the predicate tests exist to stop
    # exactly that. The honest move is to report it as inert.
    flips = sum(int(v["loose"]["overall"] > v["strict"]["overall"]) for v in scored.values())
    ident = [m for m, v in scored.items()
             if abs(v["loose"]["overall"] - v["strict"]["overall"]) < 1e-12]
    analysis["loose_arm"] = dict(models_where_loose_exceeds_strict=flips,
                                 models_identical=len(ident), n_scored=len(scored))
    lines.append(
        f"THE LOOSE ARM IS {'INERT' if flips == 0 else 'ACTIVE'}: loose exceeds strict on "
        f"{flips} of {len(scored)} models"
        + (". Base models on this pool do not wrap a compliant answer in boilerplate -- they "
           "continue the text and ignore the constraint -- so the failure mode loose scoring was "
           "added to catch does not occur here. There is ONE scoring mode in this measure, not "
           "two, and the two columns below are the same measurement reported twice. The loosener "
           "was deliberately NOT retuned after seeing this: a more aggressive one would start "
           "rescuing genuine violations, which converts the measure's noise floor into signal."
           if flips == 0 else "."))

    by_type = {}
    for m, v in scored.items():
        for ctype in CONSTRAINTS:
            by_type.setdefault(ctype, []).append(v["loose"]["by_type"].get(ctype))
    pinned = [c for c, xs in by_type.items()
              if all(y is not None for y in xs) and (max(xs) - min(xs)) < 0.05]
    analysis["by_type_span"] = {c: round(float(max(xs) - min(xs)), 4)
                                for c, xs in by_type.items() if all(y is not None for y in xs)}
    lines.append(
        f"PER-CONSTRAINT RANGE (loose), reported because an aggregate hides a pinned component: "
        + ", ".join(f"{c}={analysis['by_type_span'][c]:.2f}"
                    for c in sorted(analysis.get("by_type_span", {})))
        + (f". PINNED (span < 0.05, carrying no model information): {pinned}."
           if pinned else ". No constraint type is pinned."))
    lines.append(
        f"BOUNDARY: {len(scored)} base models of 1.7-3.2B, one constraint family, greedy decoding, "
        f"{n_items} items, one prompt format. Convergence here says compliance is a construct AT "
        f"THIS SCALE AND IN THIS FAMILY -- a base model's instruction following is not an "
        f"instruction-tuned model's, and the ten are not ten independent families.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(lines)


def main():
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"models": {}}
    res["_preregistration"] = dict(
        models=MODELS, prompts=PROMPTS, constraints=list(CONSTRAINTS),
        n_items=len(PROMPTS) * len(CONSTRAINTS), max_new=MAX_NEW, decoding="greedy",
        correctness=CORRECTNESS, temps=TEMPS, n_perm=N_PERM,
        range_k=RANGE_K, converge=CONVERGE,
        gate0="the new measure must have across-model range above its own binomial noise floor",
        rung="rho(new measure, IFEval) >= CONVERGE licenses calling it a compliance measure",
        primary="F117's selectivity statistic with IFEval REPLACED by the new measure",
        why="F117's compliance column is a single benchmark; one indicator cannot distinguish "
            "'the share tracks compliance' from 'the share tracks IFEval'")
    if "--analyse" not in _sys.argv:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        items = [(c, i) for c in CONSTRAINTS for i in range(len(PROMPTS))]
        for m in MODELS:
            if m in res["models"] and res["models"][m].get("strict"):
                continue
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    m, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()
            except Exception as e:
                print(f"  {m}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                res["models"][m] = dict(error=f"{type(e).__name__}")
                json.dump(res, open(OUT, "w"), indent=1)
                continue
            texts = [prompt_for(PROMPTS[i], CONSTRAINTS[c][0]) for c, i in items]
            try:
                outs = generate(mdl, tok, texts, device)
            except Exception as e:
                print(f"  {m}: GEN FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                res["models"][m] = dict(error=f"gen:{type(e).__name__}")
                del mdl
                json.dump(res, open(OUT, "w"), indent=1)
                continue
            responses = {k: v for k, v in zip(items, outs)}
            st, lo = score(responses)
            res["models"][m] = dict(
                strict=dict(by_type={c: float(np.mean(v)) for c, v in st.items()},
                            overall=float(np.mean([y for v in st.values() for y in v]))),
                loose=dict(by_type={c: float(np.mean(v)) for c, v in lo.items()},
                           overall=float(np.mean([y for v in lo.values() for y in v]))),
                responses={f"{c}|{i}": t for (c, i), t in responses.items()},
                secs=round(time.time() - t0, 1))
            print(f"  {m:<38} strict={res['models'][m]['strict']['overall']:.3f} "
                  f"loose={res['models'][m]['loose']['overall']:.3f} "
                  f"({res['models'][m]['secs']:.0f}s)", flush=True)
            del mdl
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            json.dump(res, open(OUT, "w"), indent=1)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
