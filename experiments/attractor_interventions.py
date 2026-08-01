"""Two interventions on the low-T attractor: conditioning radius, and token ablation (tests F62/F63).

WHY INTERVENE RATHER THAN SCREEN MORE MODELS. The screen across nine models established that the
attractor is bimodal -- top-1 share is 68-78% or 6-16%, with nothing between -- and eliminated
every observational axis tried. Not the corpus (F63: mamba is Pile with none, Qwen is non-Pile
with one). Not the architecture: granite-3.0's MoE and dense members agree to within 2 points
while differing by 2x in width, 1.7x in depth, 16x in FFN size, and routing-vs-none. When almost
the entire network can change without moving the effect, collecting more networks is the wrong
instrument. These two interventions change the MEASUREMENT instead, on a fixed model.

INTERVENTION 1 -- CONDITIONING RADIUS. Every number in this project comes from
p(x_i | x_{i-r..i-1}) at **r = 2**. That is a two-token prompt: the model is asked to continue
from almost nothing, which is exactly the regime where a language model falls back on
high-frequency filler. If the attractor is a property of that impoverished context rather than of
the model, it should weaken as r grows.

  * If top-1 share falls below the attractor threshold as r increases, the frozen phase is a
    property of the CONSTRUCTION, not the model. That extends the boundary F35 already draws for
    the damping length, and it means F58's critical point is the melting of a two-token-context
    degeneracy rather than a fact about language-model dynamics. The universality program would
    need restating in those terms.
  * If it survives to r=16, the attractor is a real property of the model's conditional and the
    programme's framing stands.

Both outcomes are reportable and the second is the one that would be convenient, which is why the
threshold is the one already fixed in the screen rather than a fresh one.

INTERVENTION 2 -- TOKEN ABLATION. `ARRule` already forbids special tokens by driving their logits
to -1e9. The same mechanism can forbid the *dominant* token. Ablating it one at a time separates
two very different situations:

  * the attractor DIES  -> it is one pathological token, and a tokenizer or data fix could remove
    it. The transition would be an artifact of a specific vocabulary entry.
  * the attractor RELOCATES to the next filler token at a similar share -> it is a structural pull
    toward low-information tokens, and no vocabulary fix helps. The phenomenon is about what the
    model does with almost no context, not about which token wins.

The second is the more interesting outcome and also the harder one to argue away, so it is stated
in advance as the prediction to beat.

MODELS. `pythia-410m` carries the strongest attractor measured (74.4% at T=0.02) and is the model
the entire universality program runs on. `gpt2-medium` has none (14.7%) and serves as the control:
if an intervention moves the control too, the intervention is doing something other than what it
claims.

Cheap for the same reason the screen was: both questions are about the SETTLED STATE, so each cell
is one settle run -- no twin, no common random numbers, no long sweep window.

Writes results/attractor_interventions.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/attractor_interventions.py
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

MODELS = [("EleutherAI/pythia-410m", "step143000", "attractor"),
          ("gpt2-medium",            None,         "control")]
RADII = [2, 4, 8, 16]
TEMPS = [0.02, 0.436]                  # strongest attractor; and F58's T_c
ABLATE_UP_TO = 5                       # forbid the top-1, then top-2, ... one at a time
N, B, SETTLE = 96, 8, 12
TOP1_HIGH, DISTINCT_LOW = 0.40, 0.30   # the SAME threshold the screen fixed, not a fresh one
OUT = str(_ROOT / "results" / "attractor_interventions.json")


def composition(rule, T, r, seed=5):
    """What the lattice relaxes to, under whatever `rule.forbidden` currently holds."""
    from ar_ca import run
    s = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme="none",
            init="random", seed=seed, order="per_replica")["final"]
    distinct, top1, toks = [], [], collections.Counter()
    for row in s:
        c = collections.Counter(row.tolist())
        distinct.append(len(c) / N)
        top1.append(c.most_common(1)[0][1] / N)
        toks.update(c)
    tid, _ = toks.most_common(1)[0]
    return dict(distinct_frac=round(float(np.mean(distinct)), 4),
                top1_share=round(float(np.mean(top1)), 4),
                dominant_id=int(tid), dominant_token=rule.tok.decode([tid]),
                has_attractor=bool(np.mean(top1) >= TOP1_HIGH
                                   and np.mean(distinct) <= DISTINCT_LOW))


def forbid(rule, ids):
    """Add token ids to the rule's forbidden set -- the mechanism it already uses for specials."""
    rule.forbidden = np.array(sorted(set(rule.forbidden.tolist()) | set(int(i) for i in ids)),
                              dtype=np.int64)
    rule._forbid_t = torch.tensor(rule.forbidden, device=rule.device, dtype=torch.long)
    rule.init_pool = np.array([i for i in range(rule.V) if i not in set(rule.forbidden.tolist())],
                              dtype=np.int64)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        models=[dict(name=m, role=role) for m, _, role in MODELS],
        radii=RADII, temps=TEMPS, ablate_up_to=ABLATE_UP_TO, N=N, B=B, settle=SETTLE,
        threshold=f"attractor iff top-1 >= {TOP1_HIGH} AND distinct/N <= {DISTINCT_LOW} "
                  f"-- the SAME threshold the nine-model screen fixed",
        radius_test="if top-1 falls below threshold as r grows, the frozen phase belongs to the "
                    "CONSTRUCTION (two-token context) rather than the model, extending F35's "
                    "boundary and restating F58's critical point as a context degeneracy",
        ablation_test="attractor DIES -> one pathological token, fixable in the vocabulary; "
                      "attractor RELOCATES at similar share -> structural pull toward filler, "
                      "no vocabulary fix helps. The second is stated in advance as the harder, "
                      "more interesting outcome",
        control="gpt2-medium has no attractor; if an intervention moves the control, the "
                "intervention is not doing what it claims",
        resumable="keyed by (model, kind, r|k, T)")
    runs = res["runs"]
    from ar_ca import ARRule

    for name, rev, role in MODELS:
        need = [k for k in
                [f"{name}|r{r}|T{T}" for r in RADII for T in TEMPS] +
                [f"{name}|ablate{k}|T{TEMPS[0]}" for k in range(ABLATE_UP_TO + 1)]
                if k not in runs]
        if not need:
            print(f"  {name}: already complete", flush=True); continue
        t0 = time.time()
        rule = ARRule(name, revision=rev) if rev else ARRule(name)
        print(f"\n  {name} ({role}) loaded in {time.time()-t0:.0f}s", flush=True)

        print(f"  -- radius sweep (the construction test) --", flush=True)
        for r in RADII:
            for T in TEMPS:
                key = f"{name}|r{r}|T{T}"
                if key in runs: continue
                c = composition(rule, T, r)
                runs[key] = dict(model=name, role=role, kind="radius", r=r, T=T, **c)
                print(f"     r={r:<3} T={T:<6} distinct={c['distinct_frac']*100:>5.1f}%  "
                      f"top1={c['top1_share']*100:>5.1f}%  dominant={c['dominant_token']!r}  "
                      f"attractor={c['has_attractor']}", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)

        print(f"  -- token ablation at r=2, T={TEMPS[0]} (die or relocate?) --", flush=True)
        banned = []
        for k in range(ABLATE_UP_TO + 1):
            key = f"{name}|ablate{k}|T{TEMPS[0]}"
            if key not in runs:
                c = composition(rule, TEMPS[0], 2)
                runs[key] = dict(model=name, role=role, kind="ablate", n_banned=k,
                                 banned=list(banned), T=TEMPS[0], **c)
                print(f"     banned={k}  distinct={c['distinct_frac']*100:>5.1f}%  "
                      f"top1={c['top1_share']*100:>5.1f}%  dominant={c['dominant_token']!r}  "
                      f"attractor={c['has_attractor']}", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
            tid = runs[key]["dominant_id"]
            if k < ABLATE_UP_TO:
                banned.append(tid); forbid(rule, [tid])

        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    out = {}
    print(f"\n=== INTERVENTION 1: does the attractor survive a longer context? ===")
    for name, _, role in MODELS:
        for T in TEMPS:
            pts = [(r, runs.get(f"{name}|r{r}|T{T}")) for r in RADII]
            pts = [(r, v) for r, v in pts if v]
            if not pts: continue
            line = "  ".join(f"r={r}:{v['top1_share']*100:.0f}%" for r, v in pts)
            surv = pts[-1][1]["has_attractor"] if pts else None
            print(f"  {name:>24} T={T:<6} {line}   attractor at r={pts[-1][0]}: {surv}")
            out[f"{name}|T{T}|radius"] = dict(
                points=[[r, v["top1_share"], v["has_attractor"]] for r, v in pts],
                survives_to_max_r=bool(surv))

    print(f"\n=== INTERVENTION 2: ablate the dominant token -- die or relocate? ===")
    for name, _, role in MODELS:
        pts = [(k, runs.get(f"{name}|ablate{k}|T{TEMPS[0]}")) for k in range(ABLATE_UP_TO + 1)]
        pts = [(k, v) for k, v in pts if v]
        if not pts: continue
        line = "  ".join(f"{k}:{v['top1_share']*100:.0f}%({v['dominant_token']!r})" for k, v in pts)
        print(f"  {name:>24} {line}")
        out[f"{name}|ablate"] = dict(
            points=[[k, v["top1_share"], v["dominant_token"], v["has_attractor"]] for k, v in pts])

    a = "EleutherAI/pythia-410m"
    rad = out.get(f"{a}|T{TEMPS[0]}|radius")
    abl = out.get(f"{a}|ablate")
    parts = []
    if rad:
        if not rad["survives_to_max_r"]:
            first = next((r for r, t, h in rad["points"] if not h), None)
            parts.append(f"RADIUS KILLS IT: pythia-410m's attractor is gone by r={first} "
                         f"(top-1 {dict((r, t) for r, t, _ in rad['points'])[first]*100:.1f}%). The "
                         f"frozen phase is a property of the two-token construction, not the model. "
                         f"F58's critical point is then the melting of a context degeneracy, and "
                         f"the universality program has to be restated in those terms.")
        else:
            parts.append(f"RADIUS DOES NOT KILL IT: the attractor survives to r={RADII[-1]} "
                         f"(top-1 {rad['points'][-1][1]*100:.1f}%), so it is a property of the "
                         f"model's conditional rather than of the impoverished context. The "
                         f"program's framing stands on this axis.")
    if abl:
        last = abl["points"][-1]
        if last[3]:
            toks = [p[2] for p in abl["points"]]
            parts.append(f"ABLATION RELOCATES IT: banning {ABLATE_UP_TO} dominant tokens in turn "
                         f"leaves an attractor at {last[1]*100:.1f}%, with the dominant token "
                         f"walking {toks}. A structural pull toward filler, not one bad vocabulary "
                         f"entry -- no tokenizer fix removes it.")
        else:
            n = next((k for k, t, tok, h in abl["points"] if not h), None)
            parts.append(f"ABLATION KILLS IT: the attractor is gone after banning {n} token(s). "
                         f"It was specific to those vocabulary entries, so the transition is an "
                         f"artifact of a handful of tokens rather than a general property.")
    ctrl = out.get(f"gpt2-medium|T{TEMPS[0]}|radius")
    if ctrl:
        moved = any(h for _, _, h in ctrl["points"])
        msg = ("ACQUIRED an attractor under intervention, which invalidates the reading above"
               if moved else
               "stays without an attractor throughout, so the interventions are not "
               "manufacturing the effect")
        parts.append(f"CONTROL: gpt2-medium {msg}.")
    verdict = " ".join(parts) if parts else "insufficient data"
    print(f"\n  -> {verdict}")

    res["analysis"] = out
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Two interventions on the low-temperature attractor, run because the observational axes "
        "were exhausted: the nine-model screen found the effect bimodal (68-78% vs 6-16%) and "
        "eliminated corpus (F63) and architecture (granite MoE vs dense agree within 2 points "
        "while differing 2x in width, 1.7x in depth, 16x in FFN and routing-vs-none). Radius asks "
        "whether the attractor belongs to the two-token conditioning window rather than the model "
        "-- if so, F58's critical point is a construction degeneracy and F35's boundary extends. "
        "Ablation asks whether it is one pathological vocabulary entry or a structural pull toward "
        "filler, by forbidding the dominant token in turn via the mechanism ARRule already uses "
        "for special tokens. gpt2-medium is the control: an intervention that moves it is not "
        "doing what it claims. Threshold is the screen's, not a fresh one.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
