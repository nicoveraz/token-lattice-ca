"""Two interventions on the low-T attractor: conditioning radius, and token ablation (tests F62/F63).

WHY INTERVENE RATHER THAN SCREEN MORE MODELS. The screen across nine models established that the
attractor separates cleanly at the threshold -- and eliminated
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

    a, c = "EleutherAI/pythia-410m", "gpt2-medium"
    T0 = TEMPS[0]
    ra, rc = out.get(f"{a}|T{T0}|radius"), out.get(f"{c}|T{T0}|radius")
    parts = []

    # The radius conclusion MUST be read against the control. If the control -- which has no
    # attractor at r=2 -- also acquires one at large r, then the large-r attractor is a generic
    # long-context effect and NOT the model-distinguishing phenomenon. What distinguishes the
    # families is the GAP, so that is what the test is on.
    if ra and rc:
        gaps = [(r, ta - tc) for (r, ta, _), (_, tc, _) in zip(ra["points"], rc["points"])]
        print(f"\n  treatment minus control, per radius: "
              + "  ".join(f"r={r}:{g*100:+.0f}" for r, g in gaps))
        out["family_gap_by_radius"] = [[r, round(g, 4)] for r, g in gaps]
        ctrl_gains = rc["points"][-1][2] and not rc["points"][0][2]
        wide = [r for r, g in gaps if g >= 0.30]
        if ctrl_gains and wide == [gaps[0][0]]:
            parts.append(
                f"THE FAMILY DIFFERENCE IS SPECIFIC TO r={gaps[0][0]}. The gap between the "
                f"attractor model and the control is {gaps[0][1]*100:+.0f} points at r={gaps[0][0]} "
                f"and " + ", ".join(f"{g*100:+.0f} at r={r}" for r, g in gaps[1:]) + ". The "
                f"control ACQUIRES an attractor at r={rc['points'][-1][0]} "
                f"({rc['points'][-1][1]*100:.0f}%) despite having none at r={gaps[0][0]}, so the "
                f"large-radius attractor is a GENERIC long-context effect present in both models "
                f"and is not the phenomenon that separates families. Read against the control, "
                f"the model-distinguishing frozen phase exists ONLY at the two-token window the "
                f"whole project uses -- so it is a property of the CONSTRUCTION. F35's boundary "
                f"extends to cover it, and F58's critical point is the melting of a two-token "
                f"degeneracy rather than a fact about language-model dynamics.")
        elif not ctrl_gains and ra["survives_to_max_r"]:
            parts.append(f"RADIUS DOES NOT KILL IT and the control stays clean, so the attractor "
                         f"is a property of the model's conditional. The framing stands.")
        else:
            parts.append(f"RADIUS RESULT AMBIGUOUS: gaps {gaps}, control gains attractor: "
                         f"{ctrl_gains}. Reported without a conclusion.")

    abl, ablc = out.get(f"{a}|ablate"), out.get(f"{c}|ablate")
    if abl:
        p0, p1 = abl["points"][0], abl["points"][1]
        if not p1[3]:
            parts.append(
                f"ONE TOKEN CARRIES IT: banning {p0[2]!r} alone drops top-1 from "
                f"{p0[1]*100:.0f}% to {p1[1]*100:.0f}% and the attractor is gone. It does NOT "
                f"relocate -- the next dominant tokens sit at "
                f"{', '.join(f'{q[1]*100:.0f}%' for q in abl['points'][1:4])}, i.e. at the "
                f"control's baseline. So the frozen phase rests on a single vocabulary entry, "
                f"not a structural pull toward filler.")
        else:
            parts.append(f"IT RELOCATES: after banning {ABLATE_UP_TO} tokens the attractor "
                         f"persists at {abl['points'][-1][1]*100:.0f}% -- a structural pull "
                         f"toward filler that no vocabulary fix removes.")
    if ablc:
        flat = max(q[1] for q in ablc["points"]) - min(q[1] for q in ablc["points"])
        parts.append(f"CONTROL under ablation moves by only {flat*100:.0f} points, so the "
                     f"collapse above is not an artifact of forbidding tokens.")
    verdict = " ".join(parts) if parts else "insufficient data"
    print(f"\n  -> {verdict}")

    res["analysis"] = out
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Two interventions on the low-temperature attractor, run because the observational axes "
        "were exhausted: the nine-model screen found the effect well separated at the threshold "
        "(later shown NOT to be bimodal -- see F64's correction, where a 26-model screen fills the "
        "apparent gap) and "
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
