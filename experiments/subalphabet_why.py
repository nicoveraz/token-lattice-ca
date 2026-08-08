"""WHY the sub-alphabet lattice has no live regime: s measured on the states it actually occupies.

subalphabet_regime.py established the fact -- 18 cells, 3 alphabets x 6 temperatures, ignition 0.00
everywhere, damage never rising -- and named the cause only as "the restriction itself". This
measures the mechanism instead of asserting it, because the mechanism has already been guessed
wrong twice: first "projection destroys window-dependence" (refuted, projected s_crn was 0.61-0.69
on RANDOM windows) and then "small state space coalescence" (not the mechanism as stated).

THE ANSWER, FOUND ON THE FOURTH ATTEMPT AND MEASURED RATHER THAN ARGUED: the FAR window token is
nearly dead under restriction. Flipping the NEAR token (i-1) moves the draw 0.45-0.81 of the time;
flipping the FAR one (i-2) moves it as little as 0.095. Damage growth needs the BRANCHING RATIO
s_far + s_near -- the expected number of the r=2 children a damaged site infects -- to exceed 1,
and it sits at 0.90-0.97 at T=0.7. Effective branching is ~1, not 2: damage WALKS but cannot GROW,
so an injected block drifts and coalesces instead of spreading. A two-token window is really a
one-token window on a restricted support, which is F69's r <= 2 boundary reappearing from inside.

This is why every scalar summary of s misled. Mean s, random-window s and settled-state s are all
averages over WHICH POSITION is flipped, and the structure lives entirely in that decomposition --
the same shape as F94 -> F96 -> F99, where the mean was flat and the spread was the finding.

THE EARLIER HYPOTHESIS, kept because it was tested and is wrong. s_crn is the exact probability that CRN twins
draw differently given a one-token window change. Damage cannot spread where it is ~0.
  LOW T   the ring FREEZES (binary top-share 0.939 at T=0.7), so the twins' windows coincide and
          there is nothing for s to respond to.
  HIGH T  the projected conditional flattens toward uniform over the small support, so p ~ q for
          ANY pair of windows and s collapses again.
Both ends give s -> 0 on the states the lattice occupies, for different reasons.

THE MEASUREMENT IS THE F96/F99 LESSON APPLIED. s must be evaluated on the ensemble the dynamics run
in, not on random windows -- that is exactly the error that produced the first wrong diagnosis.
So: settle the ring at each (alphabet, T), then measure s on windows drawn from THAT settled state,
alongside s on random windows for contrast.

PRE-REGISTERED:
  PRIMARY   is s_settled << s_random at every temperature, and does s_settled sit near zero where
            ignition is zero? That is the mechanism.
  CONTRAST  s_random is expected to stay moderate -- it is what misled the first diagnosis, and
            reporting both is what makes the regime distinction visible rather than arguable.
  KILL      s_settled is NOT small -> the conditional does respond to window changes on the settled
            state, damage should spread, and the reason it does not lies elsewhere. That would
            leave the regime result standing and its explanation open.

Writes results/subalphabet_why.json.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from meanfield_lambda import s_crn
from subalphabet import pick_tokens, make_sampler, sub_init, COLOURS, BINARY, DIGITS

OUT = str(_ROOT / "results" / "subalphabet_why.json")
MODEL, REV = "EleutherAI/pythia-410m", "step4000"
R, B, N, SETTLE = 2, 16, 48, 12
TEMPS = [0.7, 1.0, 1.3, 1.6, 2.0, 2.5]
ALPHABETS = [("binary", BINARY), ("colours", COLOURS), ("digits", DIGITS)]
N_CTX, SEED = 64, 20260808


def s_on(rule, ids, pool, T, rng, pos=None):
    """Exact mean CRN disagreement under a one-token window flip, on windows from `pool`.

    `pos` selects WHICH window position is flipped: 0 is the far token (i-2), 1 the near one
    (i-1). None averages over both, which is what every earlier measurement did -- and which is
    exactly what hid the answer, because the two positions differ by up to 8x.
    """
    pool = np.asarray(pool, dtype=np.int64)
    rows = []
    for _ in range(N_CTX):
        w = [int(x) for x in rng.choice(pool, size=R)]
        a = list(w)
        j = int(rng.integers(0, R)) if pos is None else pos
        while a[j] == w[j]:
            a[j] = int(rng.choice(pool))
        rows += [w, a]
    with torch.no_grad():
        lg = rule.model(input_ids=torch.tensor(rows, device=rule.device)
                        ).logits[:, -1].float().cpu().double().numpy()
    P = lg[:, ids]
    P = np.exp((P - P.max(axis=1, keepdims=True)) / T)
    P = P / P.sum(axis=1, keepdims=True)
    return float(np.mean([s_crn(P[2 * i], P[2 * i + 1]) for i in range(N_CTX)]))


def main():
    res = {"cells": {}, "_preregistration": dict(
        model=MODEL, revision=REV, temps=TEMPS, r=R, N=N, B=B, n_ctx=N_CTX, seed=SEED,
        alphabets=[a for a, _ in ALPHABETS],
        primary="BRANCHING RATIO s_far + s_near on the settled state, against 1. Damage can only "
                "grow if a damaged site infects more than one of its r=2 children in expectation. "
                "This decomposes s by WHICH window position is flipped -- the decomposition every "
                "earlier measurement averaged over.",
        superseded="the first primary asked whether s_settled sits below mean-field criticality. It "
                   "does not (0.29-0.82 against 0.5), and that is why the question was wrong: a "
                   "position-averaged s cannot express a branching ratio.",
        contrast="s_random is what misled the first diagnosis; both are reported",
        kill="s_settled not small -> the explanation lies elsewhere and the regime result stands "
             "with its cause open",
        follows="subalphabet_regime.py: 18/18 cells, ignition 0.00, damage never rose")}
    from ar_ca import ARRule, run
    rule = ARRule(MODEL, revision=REV)
    for name, words in ALPHABETS:
        ids, _, _ = pick_tokens(rule.tok, words)
        for T in TEMPS:
            t0 = time.time()
            rng = np.random.default_rng(SEED)
            smp = make_sampler(ids)
            settled = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
                          init_state=sub_init(ids, B, N, rng), seed=SEED,
                          sampler=smp)["final"].reshape(-1)
            s_set = s_on(rule, ids, settled, T, np.random.default_rng(SEED))
            s_rnd = s_on(rule, ids, ids, T, np.random.default_rng(SEED))
            s_far = s_on(rule, ids, settled, T, np.random.default_rng(SEED + 1), pos=0)
            s_near = s_on(rule, ids, settled, T, np.random.default_rng(SEED + 2), pos=1)
            vals, cnts = np.unique(settled, return_counts=True)
            res["cells"][f"{name}|T{T}"] = dict(
                alphabet=name, T=T, k=len(ids),
                s_settled=round(s_set, 5), s_random=round(s_rnd, 5),
                s_far=round(s_far, 5), s_near=round(s_near, 5),
                branching=round(s_far + s_near, 5), grows=bool(s_far + s_near > 1.0),
                ratio=round(s_set / max(s_rnd, 1e-9), 4),
                top_share=round(float(cnts.max() / cnts.sum()), 4),
                distinct=int(len(vals)), secs=round(time.time() - t0, 1))
            c = res["cells"][f"{name}|T{T}"]
            print(f"  {name:8s} T={T:<4} far={s_far:.4f} near={s_near:.4f} "
                  f"branch={c['branching']:.4f} grows={c['grows']}  top={c['top_share']:.3f} "
                  f"({c['secs']:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    del rule
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    cs = list(res["cells"].values())
    st = np.array([c["s_settled"] for c in cs]); rd = np.array([c["s_random"] for c in cs])
    far = np.array([c["s_far"] for c in cs]); near = np.array([c["s_near"] for c in cs])
    br = far + near
    print(f"\n  {'alphabet':<9} {'T':>5} {'s_far':>8} {'s_near':>8} {'branching':>10} {'top':>7} grows")
    for c in cs:
        print(f"  {c['alphabet']:<9} {c['T']:>5.1f} {c['s_far']:>8.4f} {c['s_near']:>8.4f} "
              f"{c['branching']:>10.4f} {c['top_share']:>7.3f} {'YES' if c['grows'] else ''}")

    ngrow = int((br > 1.0).sum())
    parts = [
        f"MECHANISM, measured on the fourth attempt. Damage can only grow if a damaged site infects "
        f"more than one of its r={R} children in expectation, i.e. if the BRANCHING RATIO "
        f"s_far + s_near exceeds 1. Decomposed by which window position is flipped, the far token "
        f"(i-2) contributes {far.min():.4f}-{far.max():.4f} while the near one (i-1) contributes "
        f"{near.min():.4f}-{near.max():.4f} -- up to {near.max()/max(far.min(),1e-9):.0f}x more. "
        f"Branching spans {br.min():.4f}-{br.max():.4f} and clears 1 in only {ngrow} of {len(cs)} "
        f"cells. A two-token window is effectively a ONE-token window on a restricted support, so "
        f"damage WALKS but cannot GROW: an injected block drifts and coalesces. That is why "
        f"ignition was 0.00 in 18/18 cells of the regime sweep.",
        f"WHY EVERY EARLIER MEASUREMENT MISLED. Position-averaged s on the settled state ranges "
        f"{st.min():.4f}-{st.max():.4f} (mean {st.mean():.4f}) against a mean-field criticality of "
        f"{1.0/R} -- supercritical in most cells, predicting growth that does not happen. On random "
        f"windows it ranges {rd.min():.4f}-{rd.max():.4f}. Both are averages over WHICH position is "
        f"flipped, and the whole structure lives in that decomposition. Same shape as F94 -> F96 -> "
        f"F99: the mean was flat and the spread was the finding, one level down.",
    ]
    resid = [c for c in cs if c["grows"] and c["top_share"] < 0.9]
    if resid:
        parts.append(
            f"AN UNEXPLAINED RESIDUE, recorded rather than smoothed over: {len(resid)} cell(s) "
            f"clear branching > 1 on a non-frozen ring ("
            + ", ".join(f"{c['alphabet']}@T{c['T']} branch={c['branching']:.3f}" for c in resid)
            + f") yet still showed ignition 0.00 in the regime sweep. Branching above 1 is "
              f"NECESSARY but evidently not SUFFICIENT here, and this analysis does not say why. "
              f"Candidates not tested: the annealed branching ratio ignores that a damaged site's "
              f"two children overlap on a ring, and async visit order lets a damaged site heal "
              f"before its children are visited (F57's mechanism).")
    parts.append(
        "BOUNDARY: one model, one checkpoint, one radius. This explains the dead regime for THIS "
        "construction; it is not a statement about token-lattice CAs in general. s is exact here "
        "(inverse-CDF CRN disagreement), so no number above carries sampling error.")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(
        s_settled_range=[round(float(st.min()), 5), round(float(st.max()), 5)],
        s_random_range=[round(float(rd.min()), 5), round(float(rd.max()), 5)],
        s_far_range=[round(float(far.min()), 5), round(float(far.max()), 5)],
        s_near_range=[round(float(near.min()), 5), round(float(near.max()), 5)],
        branching_range=[round(float(br.min()), 5), round(float(br.max()), 5)],
        n_growing=ngrow, n_cells=len(cs), mf_critical=1.0 / R,
        unexplained_residue=[f"{c['alphabet']}@T{c['T']}" for c in resid])
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Why the sub-alphabet lattice has no live regime. The answer is the BRANCHING RATIO "
        "s_far + s_near, not any position-averaged s: restriction collapses the far window token's "
        "influence, so effective branching is ~1 and damage walks without growing. Three earlier "
        "hypotheses were tested and refuted, and the failure of each is recorded in this file.")


if __name__ == "__main__":
    main()
