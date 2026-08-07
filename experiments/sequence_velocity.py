"""#107: a sequence alphabet gives F16/F21's velocity its first analytic reference.

THE GAP. F16/F21 measured the damage front velocity as PROPORTIONAL TO r and model-invariant, and
it has NO analytic reference anywhere in the validation ladder. The ECA rung validates class
separation, DK validates the damage field bit-for-bit, the logistic rung is a smooth-limit
arithmetic unit test, the Markov rung validates transition recovery. None of them predicts a
velocity.

THE CONSTRUCTION, WHICH NEEDS NO SPECIALIST MODEL. Restrict the lattice to digits, whose successor
rule an ordinary LM already implements: given "7 8" the next token is "9". That is exactly the
two-token window the CA uses, so the lattice approximates a KNOWN automaton -- a shift/successor
map on a ring -- whose damage cone is analytic: front velocity exactly r, and no healing, because a
shift map cannot heal. Unlike two random English tokens, a digit run is IN DISTRIBUTION at r=2,
which is the condition F66/F69 identified as the source of the degeneracy that ended the
universality programme.

PRE-REGISTERED:
  DETERMINISM GATE, FIRST. The rung only exists if the conditional actually implements successor.
  Measure P(next = prev+1 mod 10) on the sub-alphabet BEFORE any velocity is read. Below the
  threshold there is no known answer and the rung does not exist -- reported as such, not worked
  around.
  PRIMARY   does the measured front velocity equal r, for r in {2, 3, 4}? A known answer with no
            fitted constants.
  SECOND    does F16/F21's velocity-proportional-to-r scaling reproduce here?
  KILL      the conditional is not deterministic enough -> no rung. This is a real outcome and the
            gate is what makes it honest rather than a failed experiment.
  BOUNDARY  a near-deterministic conditional is low-entropy, so T*, melting and the attractor
            census probably do not exist here. This is a RUNG, NOT A RESULT about language models.

Writes results/sequence_velocity.json.  Resumable per (radius, seed).
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from mlm_lib import cone_front_velocity
from subalphabet import pick_tokens, damage_on_sub, lambda_of, DIGITS
from gatecheck import dynamic_range, carries_verdict

OUT = str(_ROOT / "results" / "sequence_velocity.json")
MODEL, REV = "EleutherAI/pythia-410m", "step4000"
T, B, N, SETTLE, SWEEPS = 0.7, 16, 64, 12, 22
RADII = [2, 3, 4]
SEEDS = [21, 22, 23, 24]
DETERMINISM_MIN = 0.50          # P(successor) below this and the rung does not exist


def determinism(rule, ids):
    """P(next token = successor) over all consecutive digit pairs, at this temperature.

    Measured on the model's OWN conditional restricted to the sub-alphabet -- the same projection
    the lattice samples through, so this is the rule the lattice actually runs, not an idealisation.
    """
    import torch as _t
    k = len(ids)
    wins, want = [], []
    for a in range(k):
        for b in range(k):
            wins.append([int(ids[a]), int(ids[b])])
            want.append((b + 1) % k)
    with _t.no_grad():
        x = _t.tensor(wins, device=rule.device)
        lg = rule.model(input_ids=x).logits[:, -1].float().cpu().double().numpy()
    p = np.exp((lg[:, ids] - lg[:, ids].max(axis=1, keepdims=True)) / T)
    p = p / p.sum(axis=1, keepdims=True)
    hit = float(np.mean([p[i, want[i]] for i in range(len(wins))]))
    top = float(np.mean([int(np.argmax(p[i])) == want[i] for i in range(len(wins))]))
    return dict(p_successor=round(hit, 4), argmax_successor=round(top, 4), n_windows=len(wins))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    from ar_ca import ARRule
    rule = ARRule(MODEL, revision=REV)
    ids, kept, dropped = pick_tokens(rule.tok, DIGITS)
    print(f"  digits: {len(ids)} single tokens" + (f"  DROPPED {dropped}" if dropped else ""), flush=True)
    det = res.get("determinism") or determinism(rule, ids)
    res["determinism"] = det
    res["_preregistration"] = dict(
        model=MODEL, revision=REV, T=T, radii=RADII, seeds=SEEDS, B=B, N=N, alphabet=DIGITS,
        determinism_min=DETERMINISM_MIN,
        gate="P(successor) on the projected conditional must clear the threshold BEFORE any "
             "velocity is read; below it there is no known answer and the rung does not exist",
        primary="does the measured damage front velocity equal r, for r in {2,3,4}?",
        second="does F16/F21's velocity-proportional-to-r scaling reproduce here?",
        boundary="a RUNG, not a result about language models; low entropy means T*/melting "
                 "probably do not exist on this alphabet")
    print(f"  DETERMINISM GATE: P(successor)={det['p_successor']:.4f}, "
          f"argmax-successor={det['argmax_successor']:.4f} over {det['n_windows']} windows "
          f"(threshold {DETERMINISM_MIN})", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    if det["p_successor"] < DETERMINISM_MIN:
        print("  -> below threshold: the successor rung does not exist. Recording and stopping.",
              flush=True)
    else:
        for r in RADII:
            for sd in SEEDS:
                k = f"r{r}|s{sd}"
                if k in res["cells"]:
                    continue
                t0 = time.time()
                _, rolled = damage_on_sub(rule, ids, None, T=T, r=r, B=B, N=N,
                                          settle=SETTLE, sweeps=SWEEPS, seed=sd)
                cone = rolled.mean(axis=1)
                row = lambda_of(rolled, N)
                row.update(r=r, seed=sd, velocity=float(cone_front_velocity(cone)),
                           secs=round(time.time() - t0, 1))
                res["cells"][k] = row
                print(f"  {k:12s} v={row['velocity']:.3f} (predicted {r}) lam={row['lambda_ca']:+.4f} "
                      f"ign={row['ignition']:.2f} ({row['secs']:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    del rule
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    parts, det = [], res.get("determinism", {})
    passed = det.get("p_successor", 0) >= DETERMINISM_MIN
    parts.append(
        f"DETERMINISM GATE, read before any velocity: P(successor) = {det.get('p_successor')} and "
        f"argmax-successor = {det.get('argmax_successor')} over {det.get('n_windows')} windows, "
        f"against a threshold of {DETERMINISM_MIN}. "
        + ("The projected conditional implements successor well enough for the shift map to be the "
           "known answer." if passed else
           "IT DOES NOT. The model's digit conditional is not a successor map at this temperature, "
           "so there is no analytic reference and THE RUNG DOES NOT EXIST. That is the registered "
           "kill, and it is reported rather than worked around -- lowering the threshold after "
           "seeing this number would be the defect this project has caught six times."))
    rows = {}
    for r in RADII:
        cs = [c for c in res["cells"].values() if c.get("r") == r and np.isfinite(c.get("velocity", np.nan))]
        if cs:
            rows[r] = dict(n=len(cs), v=round(float(np.mean([c["velocity"] for c in cs])), 4),
                           sd=round(float(np.std([c["velocity"] for c in cs])), 4),
                           lam=round(float(np.mean([c["lambda_ca"] for c in cs])), 5))
    if rows:
        print(f"\n  {'r':>3} {'n':>3} {'velocity':>9} {'sd':>8} {'predicted':>10} {'lambda':>9}")
        for r, v in rows.items():
            print(f"  {r:>3} {v['n']:>3} {v['v']:>9.3f} {v['sd']:>8.3f} {r:>10} {v['lam']:>+9.4f}")
        errs = {r: abs(v["v"] - r) / r for r, v in rows.items()}
        worst = max(errs.values())
        parts.append(
            f"PRIMARY: measured front velocity against the analytic prediction v = r -- "
            + ", ".join(f"r={r}: {v['v']:.2f} vs {r} ({errs[r]:.0%})" for r, v in rows.items())
            + f". Worst relative error {worst:.0%}. "
            + ("The shift map's velocity is recovered, so F16/F21's velocity result now has a "
               "known-answer rung." if worst <= 0.25 else
               "The prediction is NOT recovered, so either the lattice is not the shift map the "
               "determinism gate suggested, or the front estimator does not measure what the "
               "analytic cone predicts. Either way F16/F21 does not gain a rung here."))
        if len(rows) >= 3:
            rs = np.array(list(rows)); vs = np.array([rows[r]["v"] for r in rows])
            slope = float(np.polyfit(rs, vs, 1)[0])
            parts.append(f"SECOND LEG: velocity-vs-r slope is {slope:.3f} against F16/F21's "
                         f"proportionality (slope 1 through the origin).")
    parts.append(
        "BOUNDARY: this is a RUNG, not a result about language models. A near-deterministic "
        "conditional is low-entropy, so T*, melting and the attractor census are not expected to "
        "exist on this alphabet, and no number here should be quoted as a finding about LMs.")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows={str(k): v for k, v in rows.items()},
                           determinism=det, gate_passed=bool(passed))
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("#107. F16/F21's velocity result has no analytic reference in the ladder. A "
                    "digit alphabet whose successor rule an ordinary LM implements makes the "
                    "lattice a shift map on a ring, whose front velocity is exactly r with no "
                    "fitted constants. A rung, not a result.")


if __name__ == "__main__":
    main()
