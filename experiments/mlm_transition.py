"""M1/M2 (#89): does the CLEAN construction have a transition, and does it survive the checks?

WHY. F66 showed the AR construction -- p(x_i | x_{i-2}, x_{i-1}) -- measures an out-of-distribution
prompt artifact. The frozen phase exists only at r=2, is carried by a single token, a one-token BOS
prefix removes 50 of its 74 points, and the masked-LM construction shows no single-token
concentration at any temperature or radius tested. But that is the ABSENCE of a pathology, not the
presence of a result: nothing about MLM dynamics has been measured with the post-F57 machinery, and
none of `experiments/mlm_*.py` looks for a critical point.

THE GATE IS IN THE CODE, NOT THE README. M2 runs only if M1 finds a bracket. The AR line spent
roughly sixty hours measuring an artifact precisely because "check the object is real before
measuring its exponents" was an intention rather than a control-flow statement.

  M1  coarse temperature scan: does damage ever stop spreading?
  M2  only if M1 brackets a transition: re-run the interventions that killed the AR line --
      radius sweep with a control, and dominant-token ablation -- at the bracketed temperature.
  M3  exponents. NOT in this script. Do not start before M2 passes.

A NULL AT M1 IS A GOOD OUTCOME, AND IS PRE-REGISTERED AS SUCH. The AR frozen phase existed because
the lattice had an absorbing state: every site resampled to `'\\n'` regardless of context. F66 found
no such state in the MLM construction, so there may be no absorbing state to have an absorbing-state
transition into. If damage spreads at every temperature down to T=0.02, that is not a failed
experiment -- it says the transition was only ever the artifact, which strengthens the main claim
rather than weakening it.

WHAT IS DIFFERENT FROM THE AR RUNS, AND WHY IT MATTERS
  * The window is SYMMETRIC with the centre masked, `arange(i-r, i+r+1) % N`, so r=2 conditions on
    four neighbours rather than two left ones. Infilling a masked centre is BERT's native training
    objective, which is the whole point: the AR rule asked a model to continue from two tokens,
    which it had never seen in training.
  * `order="per_replica"` (F57) was plumbed through `mlm_ca.run` on 1 Aug and had never been used.
    The exact-zero CRN null was re-verified on this path under the new mode before this run: 0
    differing sites.
  * `MLMRule` already forbids all special tokens AND `[unused*]` placeholders, so its emission
    hygiene is stricter than the AR path's.

Writes results/mlm_transition.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/mlm_transition.py
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

BASE = "bert-base-uncased"
CONTROL = "prajjwal1/bert-medium"       # second MLM, so an M1 null is not one model's quirk
TEMPS = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
SEEDS = [91, 92]
N, B, R = 96, 64, 2
SETTLE, SWEEPS = 8, 40
SCHEME = "cls_sep"
# "damage dies" vs "damage spreads", stated before the run
LO, HI = 0.05, 0.20
M2_RADII = [2, 4, 8]
M2_ABLATE = 4
TOP1_HIGH, DISTINCT_LOW = 0.40, 0.30    # the nine-model screen's threshold, unchanged
OUT = str(_ROOT / "results" / "mlm_transition.json")


def trajectory(rule, T, seed, r=R):
    """Per-replica damaged-site counts, (sweeps, B). Same protocol as dp_class_n192, MLM rule."""
    from mlm_ca import run
    base = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme=SCHEME,
               init="random", seed=seed, order="per_replica")["final"]
    pool = np.array([i for i in range(rule.V) if i not in set(rule.forbidden.tolist())],
                    dtype=np.int64)
    flipped = base.copy()
    flipped[:, N // 2] = np.random.default_rng(seed).choice(pool, size=B)
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
    perm = np.argsort(np.random.default_rng(seed + 3).random((SWEEPS, B, N)), axis=2)
    snaps = run(rule, B=2 * B, N=N, r=r, T=T, sweeps=SWEEPS, scheme=SCHEME,
                init_state=np.concatenate([base, flipped], axis=0), seed=seed + 2, u_stream=u2,
                order="per_replica", order_stream=np.concatenate([perm, perm], axis=1))["snaps"]
    return (snaps[1:, :B] != snaps[1:, B:]).sum(axis=2)


def composition(rule, T, r):
    """Settled-lattice composition -- the F65 measurement, for M2."""
    from mlm_ca import run
    s = run(rule, B=8, N=N, r=r, T=T, sweeps=12, scheme=SCHEME,
            init="random", seed=5, order="per_replica")["final"]
    distinct, top1, toks = [], [], collections.Counter()
    for row in s:
        c = collections.Counter(row.tolist())
        distinct.append(len(c) / N); top1.append(c.most_common(1)[0][1] / N); toks.update(c)
    tid, _ = toks.most_common(1)[0]
    return dict(distinct_frac=round(float(np.mean(distinct)), 4),
                top1_share=round(float(np.mean(top1)), 4),
                dominant_token=rule.tok.decode([tid]),
                has_attractor=bool(np.mean(top1) >= TOP1_HIGH
                                   and np.mean(distinct) <= DISTINCT_LOW))


def forbid(rule, ids):
    rule.forbidden = np.array(sorted(set(rule.forbidden.tolist()) | set(int(i) for i in ids)),
                              dtype=np.int64)
    rule._forbid_t = torch.tensor(rule.forbidden, device=rule.device, dtype=torch.long)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=BASE, second_model=CONTROL, temps=TEMPS, seeds=SEEDS, N=N, B=B, r=R,
        settle=SETTLE, sweeps=SWEEPS, scheme=SCHEME,
        m1="coarse temperature scan: does damage ever stop spreading?",
        m2="ONLY IF M1 brackets a transition: radius sweep + dominant-token ablation at the "
           "bracketed temperature -- the interventions that killed the AR line",
        m3="exponents. NOT in this script, and not before M2 passes",
        null_is_good="if damage spreads at every temperature down to 0.02, the transition was only "
                     "ever the AR artifact; that strengthens the main claim and is reportable",
        window="SYMMETRIC, centre masked -- BERT's native objective, unlike the AR two-token prompt",
        visit_order="per_replica (F57); CRN null re-verified as exactly 0 on this path first",
        resumable="keyed by (model, kind, T|r|k, seed)")
    runs = res["runs"]
    from mlm_ca import MLMRule

    # ---------------- M1 ----------------
    print(f"M1: does the MLM CA have a transition? {len(TEMPS)} temps x {len(SEEDS)} seeds "
          f"x {len([BASE, CONTROL])} models", flush=True)
    for model in (BASE, CONTROL):
        if all(f"{model}|m1|T{T}|s{s}" in runs for T in TEMPS for s in SEEDS):
            print(f"  {model}: M1 already complete", flush=True); continue
        rule = MLMRule(model)
        print(f"  {model} loaded", flush=True)
        for T in TEMPS:
            for s in SEEDS:
                key = f"{model}|m1|T{T}|s{s}"
                if key in runs: continue
                t0 = time.time()
                traj = trajectory(rule, T, s)
                runs[key] = dict(model=model, kind="m1", T=T, seed=s,
                                 counts=traj.astype(int).tolist(),
                                 secs=round(time.time() - t0, 1))
                print(f"    T={T:<5} s={s}: P(end)={(traj[-1] > 0).mean():.3f} "
                      f"sites={traj.mean(axis=1)[-1]:.2f} ({runs[key]['secs']}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
                del traj; gc.collect()
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    bracket = m1_bracket(res)
    print(f"\n  M1 bracket: {bracket}", flush=True)

    # ---------------- M2, gated ----------------
    if bracket is None:
        print("  M2 SKIPPED: M1 found no bracket, so there is no transition to check. This is a "
              "pre-registered outcome, not a failure.", flush=True)
    else:
        Tb = round(sum(bracket) / 2, 4)
        print(f"\nM2: interventions at the bracket midpoint T={Tb}", flush=True)
        rule = MLMRule(BASE)
        for r in M2_RADII:
            key = f"{BASE}|m2r|r{r}|T{Tb}"
            if key in runs: continue
            c = composition(rule, Tb, r)
            runs[key] = dict(model=BASE, kind="m2_radius", r=r, T=Tb, **c)
            print(f"    r={r:<3} distinct={c['distinct_frac']*100:>5.1f}% "
                  f"top1={c['top1_share']*100:>5.1f}% dominant={c['dominant_token']!r} "
                  f"attractor={c['has_attractor']}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
        banned = []
        for k in range(M2_ABLATE + 1):
            key = f"{BASE}|m2a|k{k}|T{Tb}"
            if key not in runs:
                c = composition(rule, Tb, R)
                runs[key] = dict(model=BASE, kind="m2_ablate", n_banned=k, banned=list(banned),
                                 T=Tb, **c)
                print(f"    banned={k} distinct={c['distinct_frac']*100:>5.1f}% "
                      f"top1={c['top1_share']*100:>5.1f}% dominant={c['dominant_token']!r}",
                      flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
            if k < M2_ABLATE:
                tid = rule.tok.encode(runs[key]["dominant_token"], add_special_tokens=False)
                if tid:
                    banned.append(tid[0]); forbid(rule, [tid[0]])
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res, bracket)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def m1_bracket(res):
    """First adjacent temperature pair where surviving damage crosses from near-zero to substantial."""
    runs = [v for v in res["runs"].values() if v.get("kind") == "m1" and "counts" in v]
    if not runs:
        return None
    for model in (BASE,):
        pe = {}
        for T in TEMPS:
            cs = [np.array(v["counts"]) for v in runs if v["model"] == model and v["T"] == T]
            if cs:
                c = np.concatenate(cs, axis=1)
                pe[T] = float((c[-1] > 0).mean())
        ts = sorted(pe)
        for a, b in zip(ts, ts[1:]):
            if pe[a] < LO <= pe[b] or pe[a] < HI <= pe[b]:
                return (a, b)
    return None


def analyse(res, bracket):
    runs = res["runs"]
    print(f"\n=== M1: surviving damage vs temperature (MLM, symmetric masked-centre) ===")
    out = {}
    for model in (BASE, CONTROL):
        print(f"  {model}")
        print(f"  {'T':>7} {'reps':>6} {'P(end)':>9} {'sites(end)':>11} {'max sites':>10}")
        for T in TEMPS:
            cs = [np.array(v["counts"]) for v in runs.values()
                  if v.get("kind") == "m1" and v.get("model") == model and v.get("T") == T
                  and "counts" in v]
            if not cs: continue
            c = np.concatenate(cs, axis=1)
            out[f"{model}|T{T}"] = dict(replicas=int(c.shape[1]),
                                        P_end=round(float((c[-1] > 0).mean()), 4),
                                        sites_end=round(float(c[-1].mean()), 3),
                                        max_sites=int(c.max()))
            print(f"  {T:>7} {c.shape[1]:>6} {(c[-1] > 0).mean():>9.4f} "
                  f"{c[-1].mean():>11.3f} {int(c.max()):>10}")
        print()

    m2r = {k: v for k, v in runs.items() if v.get("kind") == "m2_radius"}
    m2a = {k: v for k, v in runs.items() if v.get("kind") == "m2_ablate"}
    if bracket is None:
        pe = [out[f"{BASE}|T{T}"]["P_end"] for T in TEMPS if f"{BASE}|T{T}" in out]
        verdict = (f"NO TRANSITION IN THE CLEAN CONSTRUCTION: surviving damage never drops below "
                   f"{min(pe):.3f} across T in {TEMPS}, down to T={min(TEMPS)} where sampling is "
                   f"essentially deterministic. There is no frozen phase and therefore no "
                   f"absorbing-state transition. This is the pre-registered good null: the AR "
                   f"frozen phase existed because the lattice had an absorbing state (every site "
                   f"resampling to a single token), F66 found no such state here, and this "
                   f"confirms the dynamical consequence. The transition the universality programme "
                   f"measured was only ever the out-of-distribution artifact. M2 and M3 are moot; "
                   f"claim A in paper/plan_paper2.md strengthens.")
    else:
        rad_ok = m2r and not any(v["has_attractor"] for v in m2r.values())
        abl_ok = m2a and not any(v["has_attractor"] for v in m2a.values())
        rad_str = ", ".join(f"r={v['r']}:{v['top1_share'] * 100:.0f}%" for v in m2r.values())
        if rad_ok and abl_ok:
            m2_msg = (f"PASSES: no single-token attractor at any radius ({rad_str}) or under "
                      f"ablation, so the object survives the checks that killed the AR line. "
                      f"M3 (exponents, under the F56 gate) is now licensed.")
        else:
            m2_msg = ("FAILS: the degeneracy the AR construction had is present here too. Do not "
                      "proceed to exponents; the clean construction is not clean.")
        verdict = f"TRANSITION BRACKETED at T in {list(bracket)}, and M2 {m2_msg}"
    print(f"  -> {verdict}")

    res["m1"] = out
    res["m1_bracket"] = list(bracket) if bracket else None
    res["m2_radius"] = {k: v for k, v in m2r.items()}
    res["m2_ablate"] = {k: v for k, v in m2a.items()}
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "M1/M2 of #89. F66 showed the AR construction measures an out-of-distribution prompt "
        "artifact and that the masked-LM construction shows no single-token concentration; this "
        "asks whether the clean construction has a damage-spreading TRANSITION at all, and gates "
        "the interventions on finding one. The gate is control flow, not intention: the AR line "
        "spent ~60 h on an artifact because that ordering was never enforced. A null is "
        "pre-registered as a good outcome -- the AR frozen phase required an absorbing state, F66 "
        "found none here, and no transition would confirm the dynamical consequence. Symmetric "
        "masked-centre window (BERT's native objective) rather than the AR two-token prompt; "
        "per-replica visit orders (F57), with the exact-zero CRN null re-verified on this path.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
