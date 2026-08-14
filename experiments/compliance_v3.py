"""The 40-type compliance pool on a cohort where compliance varies. F138's fix, applied.

WHAT CHANGED AND WHY IT IS THE COHORT. F137 could not resolve ten base models because the
instrument was coarse. F138 rebuilt the instrument to spec -- 40 constraint types, effective n 51.8
against 12.6, difficulty predictions tracking observed pass rates at +0.631 -- and still could not,
because the observed across-model variance (0.00334) sat below its own noise (0.00345). The true
between-model variance in verifiable instruction-following across ten BASE models is consistent
with zero. Instruction-following on models never trained to follow instructions is a construct at
its floor, so the cohort is the experiment now (`instruct_cohort.py`): ten instruction-tuned models,
one per pretraining family, IFEval spanning 48.8 points against the base cohort's 13.4.

THE PROMPT FORMAT IS A REGISTERED CHOICE, NOT A DEFAULT, and it differs from v2 on purpose.
Instruction-tuned models are trained behind a chat template and are evaluated behind one; giving
them the raw `Instruction: ... Response:` continuation format would measure them through an
interface they were not built for and would risk compressing exactly the variance this cohort was
chosen to supply. So the pool uses each model's own `chat_template` where the tokenizer has one,
and falls back to the v2 format where it does not -- recorded per model in the results file.

  THE COST OF THAT CHOICE, STATED. Scores here are NOT comparable to F138's base-cohort scores,
  because the interface differs as well as the models. That is acceptable because the comparison
  that matters is WITHIN this cohort -- the measure against the benchmarks -- and no claim is made
  across cohorts. It would not be acceptable if this were presented as "the same measurement on
  better models", and it is not.

  AND THE OPPOSITE CHOICE IS MADE FOR THE LATTICE. The attractor share must be measured on raw
  r-token context with NO template: F135 established that a chat scaffold moves the share by more
  than half the across-model spread. The pool measures instruction-following, so it uses the
  intended interface; the lattice measures the conditional, so it uses raw tokens. Same run, two
  interfaces, each matched to what it is reading.

PRE-REGISTERED, unchanged from v2 except where noted:
  GATE 0    `gatecheck.resolves_units` on the per-model scores, noise SD across constraint TYPES.
            Reliability >= MIN_RELIABILITY or the answer is NOT_DECIDABLE and nothing below is read.
  TRIM      types with across-model span < PIN_SPAN are named and excluded; cannot manufacture
            signal, since a zero-variance component only dilutes a mean.
  RUNG      convergence with IFEval >= CONVERGE. On THIS cohort the rung is a real test rather
            than a formality: IFEval has 48.8 points of range here, so a failure would mean two
            instruction-following measures genuinely disagree.
  PRIMARY   F117's selectivity statistic, imported from `compliance_v2` so it is literally the same
            code, with IFEval REPLACED by this measure.
  SECOND    two-indicator compliance (this AND IFEval), the design paper 2 would use.
  CONTROL   params_b must not be selective.
  BOUNDARY  ten instruction-tuned models, one constraint family, greedy decoding, chat templates.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from provenance import stamp, rel
from gatecheck import resolves_units, independence_report, spearman
from verifiable_constraints import CONSTRAINTS, PROMPTS_PER_TYPE, prompts_for
# the selectivity statistic is IMPORTED, not re-typed, so it cannot drift from F117's
from compliance_v2 import selectivity, _rho, CORRECTNESS, N_PERM, MAX_NEW, DIFF_RANK

OUT = str(_ROOT / "results" / "compliance_v3.json")
COHORT = str(_ROOT / "results" / "instruct_cohort.json")
SHARE = str(_ROOT / "results" / "share_instruct.json")

MIN_RELIABILITY = 0.5
PIN_SPAN = 0.05
CONVERGE = 0.5


def load_cohort():
    d = json.load(open(COHORT))
    return {r["model"]: r for r in d["cohort"]}


def build_prompt(tok, base, instr):
    """The model's own chat template where it has one; the v2 continuation format where it does not."""
    text = f"{base} {instr}"
    tmpl = getattr(tok, "chat_template", None)
    if tmpl:
        try:
            return tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False, add_generation_prompt=True), "chat"
        except Exception:
            pass
    return f"Instruction: {text}\nResponse:", "raw"


def items():
    out = []
    for name in sorted(CONSTRAINTS):
        for j, p in enumerate(prompts_for(name)):
            out.append((name, j, p))
    return out


def score(responses):
    by_type = {}
    for (name, _j), text in responses.items():
        by_type.setdefault(name, []).append(bool(CONSTRAINTS[name][1](text)))
    return {k: float(np.mean(v)) for k, v in by_type.items()}


def analyse(res):
    cohort = load_cohort()
    scored = {m: v for m, v in res["models"].items() if "by_type" in v}
    lines, analysis = [], {}
    if len(scored) < 6:
        res["analysis"] = dict(status="incomplete", n=len(scored))
        res["verdict"] = f"only {len(scored)} models scored -- nothing read."
        return
    types = sorted(CONSTRAINTS)
    spans = {t: float(max(scored[m]["by_type"][t] for m in scored)
                      - min(scored[m]["by_type"][t] for m in scored)) for t in types}
    pinned = sorted(t for t in types if spans[t] < PIN_SPAN)
    live = [t for t in types if t not in pinned]
    analysis["type_span"] = {t: round(v, 4) for t, v in sorted(spans.items())}
    analysis["pinned"] = pinned
    lines.append(
        f"TRIM: {len(pinned)} of {len(types)} types pinned (span < {PIN_SPAN}) and excluded: "
        f"{pinned}. {len(live)} carry model information. Widest: "
        + ", ".join(f"{t}={spans[t]:.2f}" for t in sorted(live, key=lambda t: -spans[t])[:5]) + ".")
    if len(live) < 4:
        res["analysis"] = analysis
        res["verdict"] = " ".join(lines) + " Too few live types to read anything."
        return

    vals = {m: float(np.mean([scored[m]["by_type"][t] for t in live])) for m in scored}
    noise_sd = [float(np.std([scored[m]["by_type"][t] for t in live], ddof=1) / np.sqrt(len(live)))
                for m in vals]
    gate0 = resolves_units(list(vals.values()), noise_sd=noise_sd,
                           min_reliability=MIN_RELIABILITY, name="the v3 compliance measure")
    flat = [scored[m]["by_type"][t] for m in scored for t in live]
    unit = independence_report(flat, [t for _m in scored for t in live], unit_name="constraint type")
    analysis["gate0"] = gate0.block()
    analysis["units"] = dict(icc=round(float(unit.icc), 4),
                             effective_n=round(float(unit.effective_n), 1), n_obs=len(flat))
    analysis["scores"] = {m: round(v, 4) for m, v in vals.items()}
    lines.append(
        f"GATE 0 (resolution): {gate0.reason}. Independent unit is the constraint type "
        f"(ICC {unit.icc:.2f}, effective n {unit.effective_n:.0f} of {len(flat)}). "
        + ("The measure resolves this cohort, which the base cohort never did, so the rung below "
           "is a real test."
           if gate0.usable else
           "NOT_DECIDABLE: the measure does not resolve even this cohort. Since the instrument is "
           "the same one that reached effective n 51.8 in F138 and this cohort has 48.8 points of "
           "IFEval range, that would point at the pool's construct rather than at either."))

    rows = [dict(model=m, params=cohort[m]["params_b"], NEW=vals[m], **cohort[m]["scores"])
            for m in vals if m in cohort]
    conv = _rho([r["NEW"] for r in rows], [r["IFEval"] for r in rows])
    analysis["convergence_with_ifeval"] = round(conv, 4)
    obs_rate = {t: float(np.mean([scored[m]["by_type"][t] for m in scored])) for t in types}
    cal = spearman([DIFF_RANK[CONSTRAINTS[t][3]] for t in types], [-obs_rate[t] for t in types])
    analysis["difficulty_calibration"] = round(float(cal), 4)

    if not gate0.usable:
        analysis["status"] = "NOT_DECIDABLE"
        lines.append(f"(Stored, NOT read: convergence with IFEval would be rho = {conv:+.3f}.)")
        lines.append(_tail(scored, live, cal, analysis))
        res["analysis"] = analysis
        res["verdict"] = " ".join(lines)
        return

    lines.append(
        f"RUNG (convergence with IFEval): rho = {conv:+.3f} against a floor of {CONVERGE}. "
        + ("Convergent: two independently built verifiable-instruction measures agree on this "
           "cohort, so compliance is a construct here and the comparison below is licensed."
           if np.isfinite(conv) and conv >= CONVERGE else
           "NOT convergent, with resolution established -- a real disagreement rather than noise. "
           "Two independently built instruction-following measures rank these ten differently, "
           "which bears directly on F117's single column."))
    if not (np.isfinite(conv) and conv >= CONVERGE):
        analysis["status"] = "RUNG_FAILED"
        lines.append(_tail(scored, live, cal, analysis))
        res["analysis"] = analysis
        res["verdict"] = " ".join(lines)
        return

    top1 = {}
    if _pathlib.Path(SHARE).exists():
        sh = json.load(open(SHARE)).get("cells", {})
        for c in sh.values():
            top1.setdefault((c["model"], c["T"]), []).append(c["top1"])
        top1 = {k: float(np.mean(v)) for k, v in top1.items()}
    if not top1:
        analysis["status"] = "AWAITING_SHARE"
        lines.append("PRIMARY not computed: results/share_instruct.json is absent, so the "
                     "attractor share has not been measured on this cohort yet. The compliance "
                     "side is ready and the lattice side is the remaining run.")
        lines.append(_tail(scored, live, cal, analysis))
        res["analysis"] = analysis
        res["verdict"] = " ".join(lines)
        return

    g = np.random.default_rng(0)
    temps = sorted({T for (_m, T) in top1})
    readouts = {}
    for T in temps:
        xs = [top1.get((r["model"], T)) for r in rows]
        if all(v is not None for v in xs):
            readouts[f"top1@{T}"] = xs
    readouts["params"] = [r["params"] for r in rows]
    prim = {rd: selectivity(xs, rows, ["NEW"], CORRECTNESS, g) for rd, xs in readouts.items()}
    second = {rd: selectivity(xs, rows, ["NEW", "IFEval"], CORRECTNESS, g)
              for rd, xs in readouts.items()}
    analysis["primary_replace_ifeval"] = prim
    analysis["second_two_indicator"] = second
    ctrl = prim.get("params", {})
    hits = [rd for rd, v in prim.items() if rd != "params" and v["perm_p"] < 0.05]
    analysis["status"] = "READ"
    lines.append(
        "PRIMARY (IFEval REPLACED by this measure): "
        + "; ".join(f"{rd} sel={v['selectivity']:+.3f} p={v['perm_p']:.4f}"
                    for rd, v in prim.items() if rd != "params")
        + f". CONTROL params sel={ctrl.get('selectivity')} p={ctrl.get('perm_p')}. "
        + ("The negative control is itself selective, so the statistic is not measuring "
           "selectivity here and nothing is read from it."
           if ctrl.get("perm_p", 1) < 0.05 else
           f"Selectivity SURVIVES on an independent compliance indicator at {hits}, on a cohort "
           f"where compliance genuinely varies: F117's result is about compliance rather than "
           f"about IFEval."
           if hits else
           "Selectivity does NOT survive on an independent compliance indicator. F117's result is "
           "a correlation with IFEval specifically, and paper 2's headline has to say so."))
    lines.append(
        "SECOND (two-indicator compliance): "
        + "; ".join(f"{rd} sel={v['selectivity']:+.3f} p={v['perm_p']:.4f}"
                    for rd, v in second.items() if rd != "params") + ".")
    lines.append(_tail(scored, live, cal, analysis))
    res["analysis"] = analysis
    res["verdict"] = " ".join(lines)


def _tail(scored, live, cal, analysis):
    fmt = {m: v.get("prompt_format") for m, v in scored.items()}
    n_chat = sum(1 for v in fmt.values() if v == "chat")
    return (
        f"CALIBRATION (diagnostic, not a gate): predicted vs observed difficulty rho = {cal:+.3f}. "
        + ("The pool behaves as its author expected."
           if np.isfinite(cal) and cal >= 0.4 else
           "The predictions do not track the data on this cohort, worth knowing whatever else "
           "holds.")
        + f" BOUNDARY: {len(scored)} instruction-tuned models, one per pretraining family, "
          f"{len(live)} live constraint types, {PROMPTS_PER_TYPE} prompts each, greedy decoding, "
          f"{n_chat} of {len(fmt)} run behind their own chat template. Scores are NOT comparable "
          f"to F138's base-cohort scores: the interface differs as well as the models, and no "
          f"cross-cohort claim is made. Ten families is ten, not a population.")


def main():
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"models": {}}
    cohort = load_cohort()
    res["_preregistration"] = dict(
        cohort=list(cohort), n_types=len(CONSTRAINTS), prompts_per_type=PROMPTS_PER_TYPE,
        n_items=len(CONSTRAINTS) * PROMPTS_PER_TYPE, max_new=MAX_NEW, decoding="greedy",
        correctness=CORRECTNESS, n_perm=N_PERM, min_reliability=MIN_RELIABILITY,
        pin_span=PIN_SPAN, converge=CONVERGE,
        prompt_format="each model's own chat template where present, else the v2 continuation "
                      "format; recorded per model, with one rendered exemplar stored",
        max_new_note="96 new tokens, imported from v2 so the two runs share a cap. ~70 words, so "
                     "min_words_50 and min_chars_300 remain satisfiable; the cap applies equally "
                     "to every model and is registered rather than tuned after seeing output",
        why_chat="instruction-tuned models are trained and evaluated behind a template, and the "
                 "raw format would measure them through an interface they were not built for",
        why_not_for_the_lattice="F135: a chat scaffold moves the attractor share by more than half "
                                "the across-model spread, so the lattice uses raw r-token context",
        not_comparable_to="compliance_v2.json -- different cohort AND different interface",
        supersedes="compliance_v2.py (F138): same instrument, base cohort, NOT_DECIDABLE")
    if "--analyse" not in _sys.argv:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        it = items()
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done_here = 0
        for m in cohort:
            if m in res["models"] and res["models"][m].get("by_type"):
                continue
            if res["models"].get(m, {}).get("load_failed") and "--retry-failed" not in _sys.argv:
                continue
            if limit and done_here >= limit:
                print(f"  (stopping after {done_here}; re-run to continue)", flush=True)
                break
            t0 = time.time()
            # TWO LOAD PATHS, and the fallback is the point. Several of these repos ship remote
            # modeling code written against transformers 4.x, which raises on 5.x
            # (EXAONE: create_causal_mask() unexpected kwarg; Phi-4-mini: cannot import
            # LossKwargs). transformers now supports many of those architectures NATIVELY, so
            # retrying with trust_remote_code=False rescues the model without pinning an old
            # library for the whole project. Which path loaded is recorded per model, because
            # "native" and "remote" are not guaranteed to be the same computation.
            mdl = tok = None
            load_path = None
            for remote in (True, False):
                try:
                    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=remote)
                    mdl = AutoModelForCausalLM.from_pretrained(
                        m, trust_remote_code=remote, torch_dtype=torch.float16).to(device).eval()
                    load_path = "remote" if remote else "native"
                    break
                except Exception as e:
                    last = f"{type(e).__name__}: {str(e)[:70]}"
                    mdl = tok = None
            if mdl is None:
                print(f"  {m}: LOAD FAILED both paths -- {last}", flush=True)
                res["models"][m] = dict(error=last, load_failed=True)
                json.dump(res, open(OUT, "w"), indent=1)
                continue
            texts, fmts = [], set()
            for c, _j, p in it:
                t, f = build_prompt(tok, p, CONSTRAINTS[c][0])
                texts.append(t); fmts.add(f)
            outs = []
            try:
                for i, p in enumerate(texts):
                    if i and i % 60 == 0:
                        print(f"      item {i}/{len(texts)}", flush=True)
                    enc = tok(p, return_tensors="pt").to(device)
                    with torch.no_grad():
                        gg = mdl.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
                    outs.append(tok.decode(gg[0][enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True))
            except Exception as e:
                print(f"  {m}: GEN FAILED [{load_path}] {type(e).__name__}: {str(e)[:70]}",
                      flush=True)
                res["models"][m] = dict(error=f"gen:{type(e).__name__}", load_failed=True,
                                        load_path=load_path)
                del mdl
                json.dump(res, open(OUT, "w"), indent=1)
                continue
            responses = {(c, j): t for (c, j, _p), t in zip(it, outs)}
            bt = score(responses)
            res["models"][m] = dict(
                by_type=bt, overall=float(np.mean(list(bt.values()))),
                # ONE RENDERED PROMPT, STORED. Several chat templates inject a system preamble and
                # some inject TODAY'S DATE (Llama-3.2 writes "Today Date: ..."), so the exact input
                # is not reconstructable from the script alone and the run is not byte-reproducible
                # across days. Keeping an exemplar makes the interface auditable, which is the
                # relevant property -- the constraint text and the task are identical regardless.
                prompt_example=texts[0],
                prompt_format=("chat" if fmts == {"chat"} else "raw" if fmts == {"raw"} else "mixed"),
                load_path=load_path,
                responses={f"{c}|{j}": t for (c, j), t in responses.items()},
                secs=round(time.time() - t0, 1))
            print(f"  {m:<44} overall={res['models'][m]['overall']:.3f} "
                  f"[{res['models'][m]['prompt_format']}] ({res['models'][m]['secs']:.0f}s)",
                  flush=True)
            done_here += 1
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
