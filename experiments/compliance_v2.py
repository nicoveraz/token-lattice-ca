"""A second compliance indicator that can actually resolve models. F137's attempt, respecified.

WHAT FAILED AND WHY, so this script's every choice traces to a measurement rather than a hunch.
F137 built a compliance measure from 10 constraint types x 12 prompts to test whether F117's
"compliance-selective" result is about COMPLIANCE or about IFEval specifically -- paper 2's
load-bearing weakness, since COMPLIANCE is a single borrowed column. It returned NOT_DECIDABLE:
reliability -12.4, an across-model span FOUR TIMES SMALLER than pure noise would produce. The
convergence rung's failure (rho = +0.360) was therefore uninformative -- an unresolving measure
cannot correlate with anything -- and must not be quoted as evidence about compliance.

The diagnosis was too few TYPES, not too few items: ICC by constraint type came in at 0.774.

    this design   40 types x  6 prompts = 240 items, deff 4.87, effective n ~ 49
    F137          10 types x 12 prompts = 120 items, deff 9.50, effective n ~ 12.6
    the naive fix 10 types x 48 prompts = 480 items, deff 37.4, effective n ~ 13

GATE 0 IS THE CORRECTED ONE FROM THE START. F137's anti-vacuity gate was itself vacuous: it
compared the across-model SPAN to one model's binomial SE at 2x, and the span of k=10 draws from
PURE NOISE is ~3.08 SD, so that gate passes noise ~93% of the time at this k. It duly reported
"2.27x -- passes" on a measure with reliability -12.4. `gatecheck.resolves_units` is that fix,
generalized into the package, and it is what runs here -- with the noise scale computed at the
level of the INDEPENDENT UNIT (the constraint type), not per item, because a per-item SE
understates the noise by the square root of the design effect.

PRE-REGISTERED, in this order, each with a NOT_DECIDABLE branch:
  GATE 0    RESOLUTION. `resolves_units` on the per-model scores, with the per-model noise SD taken
            across constraint types. Reliability must clear MIN_RELIABILITY. If the measure does
            not separate the ten models, nothing below is read and the answer is NOT_DECIDABLE --
            not a correlation, and specifically not a claim about compliance.
  TRIM      types whose across-model span is below PIN_SPAN are PINNED, named, and excluded from
            the aggregate. Registered in advance, and it cannot manufacture signal: a component
            with zero across-model variance contributes nothing but dilution to a mean.
  RUNG      CONVERGENCE with IFEval >= CONVERGE over the ten models. This licenses calling it a
            compliance measure -- the shape of F120's anchor rung. Not circular: convergence
            establishes commensurability; the PRIMARY asks a different question.
  PRIMARY   F117's selectivity statistic UNCHANGED, with IFEval REPLACED by this measure:
            |rho(readout, compliance)| - max|rho(readout, correctness)|, against the same null that
            permutes the readout across models. Reading, written before the run: selectivity
            survives on an independent indicator -> compliance is a construct and F117 stands;
            selectivity vanishes -> F117 is a correlation with IFEval specifically and paper 2's
            headline must say so.
  SECOND    the two-indicator design paper 2 would actually use (compliance = this AND IFEval).
  CONTROL   params_b must not be selective, as in F117.
  CALIBRATION each type carries a difficulty PREDICTION recorded before the run. Spearman between
            predicted difficulty and observed pass rate is reported as a diagnostic, not a gate: if
            it is near zero the author does not understand the measure, which is worth knowing even
            when the primary reads cleanly.
  BOUNDARY  ten base models of 1.7-3.2B, one constraint family, greedy decoding, one prompt format.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from ranking import rank as _rk
from provenance import stamp, rel
from gatecheck import resolves_units, independence_report, spearman
from verifiable_constraints import CONSTRAINTS, PROMPTS_PER_TYPE, prompts_for

OUT = str(_ROOT / "results" / "compliance_v2.json")
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
MIN_RELIABILITY = 0.5
PIN_SPAN = 0.05
CONVERGE = 0.5
DIFF_RANK = {"easy": 0, "medium": 1, "hard": 2}


def prompt_for(base, instr):
    return f"Instruction: {base} {instr}\nResponse:"


def items():
    """(constraint, prompt_index, prompt_text) for all 240 items, in a fixed order."""
    out = []
    for name in sorted(CONSTRAINTS):
        for j, p in enumerate(prompts_for(name)):
            out.append((name, j, p))
    return out


def generate(mdl, tok, texts, device):
    import torch
    out = []
    for i, p in enumerate(texts):
        if i and i % 60 == 0:                 # progress inside a model: a stall must be visible
            print(f"      item {i}/{len(texts)}", flush=True)
        enc = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            g = mdl.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out.append(tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def score(responses):
    by_type = {}
    for (name, _j), text in responses.items():
        by_type.setdefault(name, []).append(bool(CONSTRAINTS[name][1](text)))
    return {k: float(np.mean(v)) for k, v in by_type.items()}


def _rho(a, b):
    ra, rb = _rk(a), _rk(b)
    if not (np.isfinite(ra).all() and np.isfinite(rb).all()):
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def selectivity(x, rows, compliance, correctness, rng):
    """F117's statistic, unchanged, so this run and that one are commensurable."""
    m = {b: _rho(x, [r[b] for r in rows]) for b in compliance + correctness}
    comp = max(abs(m[b]) for b in compliance)
    corr = max(abs(m[b]) for b in correctness)
    obs = comp - corr
    xa = np.asarray(x, float)
    null = []
    for _ in range(N_PERM):
        xp = rng.permutation(xa)
        mp = {b: _rho(xp, [r[b] for r in rows]) for b in compliance + correctness}
        null.append(max(abs(mp[b]) for b in compliance) - max(abs(mp[b]) for b in correctness))
    return dict(rhos={k: round(v, 4) for k, v in m.items()},
                compliance_max=round(comp, 4), correctness_max=round(corr, 4),
                selectivity=round(obs, 4),
                perm_p=round(float(np.mean(np.array(null) >= obs - 1e-12)), 4))


def analyse(res):
    scored = {m: v for m, v in res["models"].items() if "by_type" in v}
    lines, analysis = [], {}
    if len(scored) < 4:
        res["analysis"] = dict(status="incomplete", n=len(scored))
        res["verdict"] = f"only {len(scored)} models scored -- nothing read."
        return
    types = sorted(CONSTRAINTS)

    # TRIM, registered in advance. A type with no across-model variance carries no model
    # information; dropping it cannot manufacture signal, only remove dilution.
    spans = {t: float(max(scored[m]["by_type"][t] for m in scored)
                      - min(scored[m]["by_type"][t] for m in scored)) for t in types}
    pinned = sorted(t for t in types if spans[t] < PIN_SPAN)
    live = [t for t in types if t not in pinned]
    analysis["type_span"] = {t: round(v, 4) for t, v in sorted(spans.items())}
    analysis["pinned"] = pinned
    lines.append(
        f"TRIM: {len(pinned)} of {len(types)} constraint types are PINNED (across-model span < "
        f"{PIN_SPAN}) and are excluded from the aggregate: {pinned}. "
        f"{len(live)} types carry model information. Widest: "
        + ", ".join(f"{t}={spans[t]:.2f}" for t in sorted(live, key=lambda t: -spans[t])[:5]) + ".")
    if len(live) < 4:
        res["analysis"] = analysis
        res["verdict"] = " ".join(lines) + " Too few live types to read anything."
        return

    vals = {m: float(np.mean([scored[m]["by_type"][t] for t in live])) for m in scored}
    noise_sd = [float(np.std([scored[m]["by_type"][t] for t in live], ddof=1) / np.sqrt(len(live)))
                for m in vals]
    gate0 = resolves_units(list(vals.values()), noise_sd=noise_sd,
                           min_reliability=MIN_RELIABILITY, name="the v2 compliance measure")
    flat = [scored[m]["by_type"][t] for m in scored for t in live]
    unit = independence_report(flat, [t for _m in scored for t in live], unit_name="constraint type")
    analysis["gate0"] = gate0.block()
    analysis["units"] = dict(icc=round(float(unit.icc), 4),
                             effective_n=round(float(unit.effective_n), 1), n_obs=len(flat))
    analysis["scores"] = {m: round(v, 4) for m, v in vals.items()}
    lines.append(
        f"GATE 0 (resolution): {gate0.reason}. The independent unit is the constraint type "
        f"(ICC {unit.icc:.2f}, effective n {unit.effective_n:.0f} of {len(flat)}). "
        + ("The measure resolves the ten models above its own clustered noise, so the rung below "
           "is worth asking."
           if gate0.usable else
           "NOT_DECIDABLE: the measure does not resolve the models, so every correlation from it "
           "is attenuated toward zero. Nothing below is read, and in particular a failed "
           "convergence rung would say nothing about whether compliance is a construct."))

    cov = {c["model"]: c for c in json.load(open(BENCH))["covered"].values()}
    rows = [dict(model=m, params=cov[m]["params_b"], NEW=vals[m], **cov[m]["scores"])
            for m in vals if m in cov]
    conv = _rho([r["NEW"] for r in rows], [r["IFEval"] for r in rows])
    analysis["convergence_with_ifeval"] = round(conv, 4)

    # CALIBRATION, a diagnostic and deliberately not a gate.
    obs_rate = {t: float(np.mean([scored[m]["by_type"][t] for m in scored])) for t in types}
    cal = spearman([DIFF_RANK[CONSTRAINTS[t][3]] for t in types], [-obs_rate[t] for t in types])
    analysis["difficulty_calibration"] = round(float(cal), 4)

    if not gate0.usable:
        analysis["status"] = "NOT_DECIDABLE"
        lines.append(f"(For the record and NOT as a result: convergence with IFEval would have "
                     f"read rho = {conv:+.3f}. It is not read.)")
        lines.append(_calibration_line(cal))
        lines.append(_boundary_line(scored, live))
        res["analysis"] = analysis
        res["verdict"] = " ".join(lines)
        return

    lines.append(
        f"RUNG (convergence with IFEval): rho = {conv:+.3f} against a floor of {CONVERGE}. "
        + ("Convergent: this measures the same construct IFEval does, so the comparison below is "
           "licensed."
           if np.isfinite(conv) and conv >= CONVERGE else
           "NOT convergent. The measure resolves the models -- Gate 0 passed -- so this is now a "
           "real disagreement rather than an artefact of noise: two independently built "
           "verifiable-instruction measures rank these base models differently. That bears on "
           "F117's single column, because it means 'compliance' at this scale is not one thing."))
    if not (np.isfinite(conv) and conv >= CONVERGE):
        analysis["status"] = "RUNG_FAILED"
        lines.append(_calibration_line(cal))
        lines.append(_boundary_line(scored, live))
        res["analysis"] = analysis
        res["verdict"] = " ".join(lines)
        return

    runs = json.load(open(SCREEN))["runs"]
    prof = {}
    for v in runs.values():
        if v.get("arm") == "temp" and "top1" in v:
            prof.setdefault((v["model"], v["T"]), []).append(v["top1"])
    top1 = {k: float(np.mean(x)) for k, x in prof.items()}
    g = np.random.default_rng(0)
    readouts = {}
    for T in TEMPS:
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
           f"Selectivity SURVIVES on an independent compliance indicator at {hits}: compliance is "
           f"a construct rather than an IFEval idiosyncrasy, and F117 stands."
           if hits else
           "Selectivity does NOT survive on an independent compliance indicator. F117's result is "
           "a correlation with IFEval specifically, and paper 2's headline has to say so."))
    lines.append(
        "SECOND (two-indicator compliance, the design paper 2 would use): "
        + "; ".join(f"{rd} sel={v['selectivity']:+.3f} p={v['perm_p']:.4f}"
                    for rd, v in second.items() if rd != "params") + ".")
    lines.append(_calibration_line(cal))
    lines.append(_boundary_line(scored, live))
    res["analysis"] = analysis
    res["verdict"] = " ".join(lines)


def _calibration_line(cal):
    return (
        f"CALIBRATION (diagnostic, not a gate): Spearman between the difficulty PREDICTED for each "
        f"type before the run and its observed pass rate is {cal:+.3f}. "
        + ("The predictions track the data, so the pool behaves as its author expected."
           if np.isfinite(cal) and cal >= 0.4 else
           "The predictions do NOT track the data, which means the pool's difficulty structure was "
           "not understood in advance -- worth knowing whatever the primary says."))


def _boundary_line(scored, live):
    return (
        f"BOUNDARY: {len(scored)} base models of 1.7-3.2B, {len(live)} live constraint types of "
        f"{len(CONSTRAINTS)}, {PROMPTS_PER_TYPE} prompts each, greedy decoding, one prompt format. "
        f"These are ten models, not ten families. A verifiable-instruction measure is one operational "
        f"reading of 'compliance'; agreement or disagreement with IFEval is evidence about that "
        f"family of measures, not about instruction following in general, and says nothing about "
        f"instruction-tuned models.")


def main():
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"models": {}}
    res["_preregistration"] = dict(
        models=MODELS, n_types=len(CONSTRAINTS), prompts_per_type=PROMPTS_PER_TYPE,
        n_items=len(CONSTRAINTS) * PROMPTS_PER_TYPE, max_new=MAX_NEW, decoding="greedy",
        correctness=CORRECTNESS, temps=TEMPS, n_perm=N_PERM,
        min_reliability=MIN_RELIABILITY, pin_span=PIN_SPAN, converge=CONVERGE,
        difficulty_prediction={t: CONSTRAINTS[t][3] for t in sorted(CONSTRAINTS)},
        gate0="gatecheck.resolves_units on per-model scores, noise SD taken ACROSS CONSTRAINT "
              "TYPES (the independent unit), reliability >= 0.5",
        trim="types with across-model span < 0.05 are pinned, named and excluded; this cannot "
             "manufacture signal because a zero-variance component only dilutes a mean",
        rung="rho(this measure, IFEval) >= 0.5 licenses calling it a compliance measure",
        primary="F117's selectivity statistic with IFEval replaced by this measure",
        supersedes="compliance_second_measure.py (F137): 10 types, effective n 12.6, "
                   "NOT_DECIDABLE on resolution")
    if "--analyse" not in _sys.argv:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        it = items()
        # ONE MODEL PER PROCESS by default. Holding several 3B models in one process exhausted this
        # machine: MPS allocations are wired and are not returned by `del` plus empty_cache(), so by
        # the third model the box was paging (59 MB free, 7.6 GB of 9.2 GB swap) and running at 17%
        # CPU duty -- a 3x slowdown that looks exactly like a hang. Process exit reclaims everything,
        # and the run is already resumable, so a shell loop over `--limit 1` is the whole fix.
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        done_here = 0
        for m in MODELS:
            if m in res["models"] and res["models"][m].get("by_type"):
                continue
            if limit and done_here >= limit:
                print(f"  (stopping after {done_here} model(s) this process; re-run to continue)",
                      flush=True)
                break
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    m, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()
            except Exception as e:
                print(f"  {m}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                res["models"][m] = dict(error=type(e).__name__)
                json.dump(res, open(OUT, "w"), indent=1)
                continue
            try:
                outs = generate(mdl, tok, [prompt_for(p, CONSTRAINTS[c][0]) for c, _j, p in it],
                                device)
            except Exception as e:
                print(f"  {m}: GEN FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                res["models"][m] = dict(error=f"gen:{type(e).__name__}")
                del mdl
                json.dump(res, open(OUT, "w"), indent=1)
                continue
            responses = {(c, j): t for (c, j, _p), t in zip(it, outs)}
            bt = score(responses)
            res["models"][m] = dict(
                by_type=bt, overall=float(np.mean(list(bt.values()))),
                responses={f"{c}|{j}": t for (c, j), t in responses.items()},
                secs=round(time.time() - t0, 1))
            print(f"  {m:<38} overall={res['models'][m]['overall']:.3f} "
                  f"({res['models'][m]['secs']:.0f}s)", flush=True)
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
