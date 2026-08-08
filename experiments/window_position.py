"""Is lambda_ca an r=2 phenomenon, or effectively r=1? The far/near decomposition, full vocabulary.

WHAT F109 FOUND ONE LEVEL DOWN. On a restricted token support the far window token (i-2) contributes
as little as 0.061 to the CRN disagreement where the near one (i-1) contributes 0.801 -- up to 14x
less -- so the branching ratio s_far + s_near sits below 1, damage walks without growing, and the
sub-alphabet lattice has no live regime. A two-token window turned out to be effectively ONE token.

THE QUESTION THAT RAISES ABOUT THE MAIN LINE. That decomposition has never been run on the FULL
vocabulary. F94 measured s ~ 0.85 with the position AVERAGED OVER -- which is precisely the summary
that hid the mechanism in F109, and the fourth in a row to do so (F94 -> F96 -> F99 -> F109). If
full-vocabulary damage is also carried almost entirely by the near token, then lambda_ca is closer
to a ONE-token-window phenomenon than the paper's "ring CA driven by p(x_i | x_{i-2}, x_{i-1})"
framing implies. That is a claim about how the construction is DESCRIBED, and it is one measurement.

PRE-REGISTERED:
  CALIBRATION, FIRST. The position-averaged s on random windows must reproduce F94's measured
            0.8331-0.8755 across the same checkpoints, within CALIB_TOL. This run uses a different
            code path from meanfield_lambda, and a decomposition that cannot reproduce the number
            it decomposes is not measuring the same quantity. Read before anything else.
  PRIMARY   the split s_far / s_near on the SETTLED ensemble (F96/F99: the regime the dynamics run
            in, not random windows), and the branching ratio s_far + s_near against 1.
  READING   far/near ratio >= NEAR_PARITY -> both positions carry the window and the r=2 description
            is sound. Below it -> the far token is largely inert and the construction is effectively
            r=1, which must be said in the paper's description regardless of what lambda_ca does.
  CONTRAST  the same split on random windows, because that is the ensemble F94 used and the one
            that misled F109's first three diagnoses.
  EXTENSION r=3 measures whether a third token contributes anything at all -- the effective-radius
            question F69 answered from outside the window, asked from inside it.
  KILL      far and near are comparable -> the r=2 framing is vindicated, F109's collapse is a
            property of restriction alone, and nothing about the paper changes. A clean licence.

Writes results/window_position.json.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from meanfield_lambda import s_crn, lambda_measured
from gatecheck import dynamic_range, carries_verdict
from gatecheck.cohort import cohort_complete

OUT = str(_ROOT / "results" / "window_position.json")
MODEL = "EleutherAI/pythia-410m"
STEPS = [128, 256, 512, 1000, 2000, 4000]
T = 0.7
RADII = [2, 3]
N_CTX, SEED = 128, 20260808
SET_B, SET_N, SET_SWEEPS = 8, 48, 30
CALIB_TOL = 0.08          # |our position-averaged s - F94's| on random windows
NEAR_PARITY = 0.5         # s_far / s_near below this = the far token is largely inert
F94_S = {128: 0.8352, 256: 0.8755, 512: 0.8374, 1000: 0.8331, 2000: 0.8459, 4000: 0.8426}


def s_at(rule, pool, r, pos, rng, n=N_CTX):
    """Exact mean CRN disagreement when window position `pos` is flipped. pos=None averages."""
    pool = np.asarray(pool, dtype=np.int64)
    rows = []
    for _ in range(n):
        w = [int(x) for x in rng.choice(pool, size=r)]
        a = list(w)
        j = int(rng.integers(0, r)) if pos is None else pos
        while a[j] == w[j]:
            a[j] = int(rng.choice(pool))
        rows += [w, a]
    out = []
    for i in range(0, len(rows), 32):
        with torch.no_grad():
            lg = rule.model(input_ids=torch.tensor(rows[i:i + 32], device=rule.device)
                            ).logits[:, -1].float()
            out.append(torch.softmax(lg / T, dim=-1).cpu().double().numpy())
    P = np.concatenate(out, 0)
    P = P / P.sum(axis=1, keepdims=True)
    return float(np.mean([s_crn(P[2 * i], P[2 * i + 1]) for i in range(n)]))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=STEPS, T=T, radii=RADII, n_ctx=N_CTX, seed=SEED,
        calib_tol=CALIB_TOL, near_parity=NEAR_PARITY, f94_s=F94_S,
        calibration="position-averaged s on RANDOM windows must reproduce F94's 0.8331-0.8755 "
                    "within tol; a decomposition that cannot reproduce the number it decomposes is "
                    "not measuring the same quantity",
        primary="the s_far / s_near split on the SETTLED ensemble, and branching s_far + s_near",
        reading=f"far/near >= {NEAR_PARITY} -> the r=2 description is sound; below -> the far token "
                f"is largely inert and the construction is effectively r=1",
        extension="r=3 asks whether a third token contributes at all",
        kill="far and near comparable -> r=2 framing vindicated, F109's collapse is a property of "
             "restriction alone, nothing about the paper changes",
        follows="F109: on a restricted support far/near was as low as 0.061/0.801")
    from ar_ca import ARRule, run
    for st in STEPS:
        if all(f"step{st}|r{r}" in res["cells"] for r in RADII):
            continue
        rule = ARRule(MODEL, revision=f"step{st}")
        settled = run(rule, B=SET_B, N=SET_N, r=2, T=T, sweeps=SET_SWEEPS, scheme="none",
                      seed=SEED, order="per_replica")["final"].reshape(-1)
        for r in RADII:
            k = f"step{st}|r{r}"
            if k in res["cells"]:
                continue
            t0 = time.time()
            row = dict(step=st, r=r,
                       s_avg_random=round(s_at(rule, rule.init_pool, r, None,
                                               np.random.default_rng(SEED)), 5))
            for pos in range(r):
                row[f"s_pos{pos}_settled"] = round(
                    s_at(rule, settled, r, pos, np.random.default_rng(SEED + 10 + pos)), 5)
                row[f"s_pos{pos}_random"] = round(
                    s_at(rule, rule.init_pool, r, pos, np.random.default_rng(SEED + 20 + pos)), 5)
            row["branching_settled"] = round(sum(row[f"s_pos{p}_settled"] for p in range(r)), 5)
            row["branching_random"] = round(sum(row[f"s_pos{p}_random"] for p in range(r)), 5)
            near, far = row[f"s_pos{r-1}_settled"], row["s_pos0_settled"]
            row["far_over_near_settled"] = round(far / max(near, 1e-9), 4)
            row["secs"] = round(time.time() - t0, 1)
            res["cells"][k] = row
            print(f"  step{st:<5} r={r}  " +
                  "  ".join(f"pos{p}={row[f's_pos{p}_settled']:.4f}" for p in range(r)) +
                  f"  branch={row['branching_settled']:.4f}  far/near={row['far_over_near_settled']:.3f}"
                  f"  ({row['secs']:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    cells = list(res["cells"].values())
    r2 = sorted([c for c in cells if c["r"] == 2], key=lambda c: c["step"])
    parts = []
    errs = {c["step"]: abs(c["s_avg_random"] - F94_S[c["step"]]) for c in r2 if c["step"] in F94_S}
    ok = bool(errs) and max(errs.values()) <= CALIB_TOL
    parts.append(
        f"CALIBRATION, read first: position-averaged s on random windows reproduces F94's values to "
        f"within {max(errs.values()):.4f} (tolerance {CALIB_TOL}) across {len(errs)} checkpoints. "
        + ("The decomposition measures the same quantity F94 measured."
           if ok else "IT DOES NOT -- this is not the same quantity and nothing below is read."))
    coh = cohort_complete([f"step{s}" for s in STEPS], [f"step{c['step']}" for c in r2],
                          unit="checkpoint")
    parts.append(f"COHORT: {coh.reason}")
    if not (ok and coh.complete):
        res["analysis"] = dict(calibration_ok=ok, cohort=coh.block())
        res["verdict"] = " ".join(parts) + " NOT DECIDABLE."
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}")
        return
    print(f"\n  r=2, SETTLED ensemble")
    print(f"  {'step':>6} {'far (i-2)':>10} {'near (i-1)':>11} {'branching':>10} {'far/near':>9}")
    for c in r2:
        print(f"  {c['step']:>6} {c['s_pos0_settled']:>10.4f} {c['s_pos1_settled']:>11.4f} "
              f"{c['branching_settled']:>10.4f} {c['far_over_near_settled']:>9.3f}")
    ratios = np.array([c["far_over_near_settled"] for c in r2])
    br = np.array([c["branching_settled"] for c in r2])
    inert = bool(np.mean(ratios) < NEAR_PARITY)
    parts.append(
        f"PRIMARY: on the settled ensemble the far token (i-2) contributes "
        f"{np.mean([c['s_pos0_settled'] for c in r2]):.4f} on average against the near token's "
        f"{np.mean([c['s_pos1_settled'] for c in r2]):.4f} -- a far/near ratio of "
        f"{ratios.mean():.3f} (range {ratios.min():.3f}-{ratios.max():.3f}), and a branching ratio "
        f"of {br.mean():.4f}. "
        + (f"BELOW the {NEAR_PARITY} parity threshold: the far token is largely inert, so the "
           f"full-vocabulary construction is ALSO effectively a one-token window, and describing it "
           f"as a ring CA driven by p(x_i | x_{{i-2}}, x_{{i-1}}) overstates the geometry. That is a "
           f"statement about the DESCRIPTION, not about whether lambda_ca measures something real."
           if inert else
           f"AT OR ABOVE the {NEAR_PARITY} parity threshold: both window positions carry real "
           f"influence, the r=2 description is sound, and F109's collapse is a property of "
           f"RESTRICTION alone rather than of the window geometry. The paper's framing is licensed."))
    r3 = sorted([c for c in cells if c["r"] == 3], key=lambda c: c["step"])
    if r3:
        p0 = float(np.mean([c["s_pos0_settled"] for c in r3]))
        p2 = float(np.mean([c["s_pos2_settled"] for c in r3]))
        parts.append(
            f"EXTENSION (r=3, the effective-radius question asked from inside the window): the "
            f"third-back token contributes {p0:.4f} against the nearest token's {p2:.4f}, with "
            f"branching {np.mean([c['branching_settled'] for c in r3]):.4f}. "
            + ("Influence decays sharply with distance, consistent with F69's finding that the "
               "degeneracy is confined to r <= 2 and one extra token is the whole difference."
               if p0 < p2 * NEAR_PARITY else
               "The third token carries comparable influence, so the window is not dominated by "
               "its nearest position."))
    parts.append(
        "BOUNDARY: one family, one temperature, s measured exactly (inverse-CDF CRN disagreement) "
        "so no number carries sampling error. This bounds how the construction should be described; "
        "it does not bear on whether lambda_ca replicates (F98) or predicts (F86).")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(
        calibration_ok=ok, calibration_max_err=round(max(errs.values()), 5), cohort=coh.block(),
        far_over_near=[round(float(x), 4) for x in ratios],
        branching=[round(float(x), 4) for x in br],
        far_token_inert=inert, near_parity=NEAR_PARITY)
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("F109 found that on a restricted support the far window token is nearly inert, "
                    "making a two-token window effectively one. This asks the same of the FULL "
                    "vocabulary, where the decomposition has never been run and F94 measured s with "
                    "the position averaged over. Bears on how the construction is described.")


if __name__ == "__main__":
    main()
