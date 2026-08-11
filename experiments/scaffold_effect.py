"""How much does a chat template move the attractor share? The gate on any API-based measurement.

WHY THIS EXISTS. A chat API cannot condition on r tokens alone: every request is wrapped in the
provider's template, so the model sees a system role, an instruction, formatting tokens AND our r
words. A first, coarse version of this check (N=24, one model, one seed) put the shift at 0.167 and
stopped the remote run before a call was made. That number had almost no resolution -- at N=24 with
a 6-word alphabet, top1 moves in steps of 1/24, so 0.500 vs 0.333 is four cells -- and it rested on
gpt2, the weakest model available and the one most easily pushed around by a long prefix.

THE THRESHOLD IS THE REAL FIX. A limit of 0.15 was picked by argument, which is the error this
project keeps recording. The scaffold matters only in proportion to the signal it would corrupt: the
ACROSS-MODEL SPREAD. A shift small against that spread leaves rankings intact; a shift comparable to
it makes the template a competing explanation for any ordering. So the gate is relative, and it is
the same shape as gatecheck's noise gate: signal must exceed the thing that would masquerade as it.

WHAT IS VARIED, so the answer is not one number from one place:
  scaffold   none (raw r-word context) | minimal (the sequence, no instruction) | full (the
             instruction prompt an API run would actually send). The minimal arm is the remedy
             candidate: if the effect is carried by the instruction text, a shorter prompt recovers
             the measurement.
  model      three cached models spanning 124M to 774M
  T          two temperatures, since the attractor is a low-T object
  seed       four, because the first version had one

PRE-REGISTERED:
  PRIMARY   mean |top1(scaffold) - top1(raw)| against the across-model spread measured on the raw
            arm. Registered reading: a scaffold passes if its mean shift is below GATE_FRAC of that
            spread, and the run reports which scaffolds pass rather than a single verdict.
  SECONDARY does the scaffold preserve the model RANKING? A shift that is large but uniform across
            models leaves rankings usable, which is what an API measurement actually needs.
  BOUNDARY  three small models, one alphabet, one lattice size. An instruction-tuned 70B may be far
            less scaffold-sensitive than gpt2 -- following the instruction is what it is trained
            for -- so a failure here bounds the naive design, not the remote route itself.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, torch
from ranking import spearman
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "scaffold_effect.json")
MODELS = ["gpt2", "gpt2-large", "EleutherAI/pythia-410m"]
WORDS = [" red", " green", " blue", " yellow", " black", " white"]
INSTR = ("Continue this sequence with exactly one more word from the list "
         "red, green, blue, yellow, black, white. Reply with the word only.\n")
SCAFFOLDS = {"none": "", "minimal": "Sequence:", "full": INSTR}
TEMPS = [0.2, 0.7]
SEEDS = [11, 12, 13, 14]
R, N, B, SETTLE = 2, 64, 8, 15
GATE_FRAC = 0.5


def settle(mdl, tok, ids, prefix, T, seed):
    """Async ring settle with a text prefix, B replicas advanced in parallel."""
    rng = np.random.default_rng(seed)
    ring = rng.integers(0, len(ids), size=(B, N))
    pre = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
    for _ in range(SETTLE):
        for i in rng.permutation(N):
            ctx = [[*pre, *[ids[ring[b, (i - j) % N]] for j in range(R, 0, -1)]] for b in range(B)]
            x = torch.tensor(ctx)
            with torch.no_grad():
                lg = mdl(input_ids=x).logits[:, -1].float()
            p = torch.softmax(lg[:, ids] / T, dim=-1).numpy().astype(np.float64)
            p = p / p.sum(axis=1, keepdims=True)
            u = rng.random(B)
            ring[:, i] = (np.cumsum(p, axis=1) < u[:, None]).sum(axis=1).clip(0, len(ids) - 1)
    out = []
    for b in range(B):
        _, cnt = np.unique(ring[b], return_counts=True)
        out.append(float(cnt.max() / cnt.sum()))
    return float(np.mean(out))


def analyse(res):
    cells, parts = res["cells"], []
    def val(m, s, T, sd):
        c = cells.get(f"{m}|{s}|T{T}|s{sd}")
        return c["top1"] if c else None
    raw_by_model = {m: np.mean([val(m, "none", T, sd) for T in TEMPS for sd in SEEDS
                                if val(m, "none", T, sd) is not None]) for m in MODELS}
    spread = float(max(raw_by_model.values()) - min(raw_by_model.values()))
    parts.append(
        "REFERENCE: on the raw arm the across-model spread in top1 is "
        f"{spread:.4f} ({', '.join(f'{m.split(chr(47))[-1]}={v:.3f}' for m, v in raw_by_model.items())}). "
        f"That is the signal any scaffold effect would compete with, and the gate is set against it "
        f"rather than against a number chosen by argument.")
    shifts, ranks = {}, {}
    for s in [k for k in SCAFFOLDS if k != "none"]:
        d = [abs(val(m, s, T, sd) - val(m, "none", T, sd))
             for m in MODELS for T in TEMPS for sd in SEEDS
             if val(m, s, T, sd) is not None and val(m, "none", T, sd) is not None]
        shifts[s] = (float(np.mean(d)), float(np.std(d, ddof=1)), len(d))
        a = [np.mean([val(m, "none", T, sd) for T in TEMPS for sd in SEEDS]) for m in MODELS]
        b = [np.mean([val(m, s, T, sd) for T in TEMPS for sd in SEEDS]) for m in MODELS]
        ranks[s] = float(spearman(a, b)) if all(x is not None for x in a + b) else float("nan")
    passed = [s for s, (mu, _, _) in shifts.items() if mu <= GATE_FRAC * spread]
    parts.append(
        "PRIMARY, mean |shift| in top1 against a gate of "
        f"{GATE_FRAC:g} x spread = {GATE_FRAC * spread:.4f}: "
        + "; ".join(f"{s} {mu:.4f} +/- {sd:.4f} (n={n})" for s, (mu, sd, n) in shifts.items()) + ". "
        + (f"PASSES: {passed}. A measurement through that scaffold is not dominated by the template."
           if passed else
           "NO scaffold passes. Every template tested moves the share by more than half the "
           "across-model spread, so an API measurement using one would be reading the prompt as "
           "much as the model."))
    parts.append(
        "SECONDARY, does the scaffold preserve the model RANKING: "
        + "; ".join(f"{s} rho={r:+.3f}" for s, r in ranks.items())
        + ". A large but UNIFORM shift leaves rankings usable, which is what an API measurement "
          "actually needs; a shift that reorders models does not.")
    parts.append(
        f"BOUNDARY: {len(MODELS)} models spanning 124M-774M, one alphabet, N={N}, r={R}. An "
        f"instruction-tuned 70B may be far less scaffold-sensitive than these -- following the "
        f"instruction is what it is trained for -- so a failure here bounds the naive design rather "
        f"than the remote route.")
    res["analysis"] = dict(raw_spread=spread, raw_by_model=raw_by_model,
                           shifts={k: v[0] for k, v in shifts.items()},
                           shift_sd={k: v[1] for k, v in shifts.items()},
                           rank_rho=ranks, gate=GATE_FRAC * spread, passed=passed)
    res["verdict"] = " ".join(parts)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, words=WORDS, scaffolds=list(SCAFFOLDS), temps=TEMPS, seeds=SEEDS,
        r=R, N=N, B=B, settle=SETTLE, gate_frac=GATE_FRAC,
        primary="mean |top1(scaffold) - top1(raw)| against GATE_FRAC x the across-model spread",
        why="the threshold is relative because the scaffold matters only in proportion to the "
            "signal it would corrupt; an absolute limit chosen by argument is the error this "
            "project keeps recording",
        supersedes="groq_share.py's coarse first version (N=24, one model, one seed, shift 0.167)")
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(m)
        mdl = AutoModelForCausalLM.from_pretrained(m).eval()
        ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
        for s, prefix in SCAFFOLDS.items():
            for T in TEMPS:
                for sd in SEEDS:
                    key = f"{m}|{s}|T{T}|s{sd}"
                    if key in res["cells"]:
                        continue
                    t0 = time.time()
                    v = settle(mdl, tok, ids, prefix, T, sd)
                    res["cells"][key] = dict(model=m, scaffold=s, T=T, seed=sd, top1=v,
                                             secs=round(time.time() - t0, 1))
                    print(f"  {key:<46} top1={v:.4f} ({time.time() - t0:.0f}s)", flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)
        del mdl
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
