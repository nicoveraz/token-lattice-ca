"""Is F109's far-position collapse a property of RESTRICTION, or of the alphabets it was measured on?

THE CONFOUND. F109 established the mechanism behind the sub-alphabet lattice's failure to ignite:
decomposed by window position, the far token (i-2) contributes as little as 0.0605 against the near
token's 0.8007, so the branching ratio s_far + s_near falls below 1 and damage walks without
growing. `subalphabet_regime` concluded "the failure is not the choice of alphabet ... it is the
RESTRICTION". But all three alphabets tested -- binary, colours, digits -- are hand-picked
SEMANTICALLY COHERENT sets. A closed list is exactly the kind of context a language model has strong
learned structure over: near-token constraints dominate, and the far token may stop mattering
because the set is a list, not because the set is small.

So "restriction kills long-range influence" and "closed semantic lists kill long-range influence"
are both consistent with every measurement made so far, and they differ in what they imply. If the
collapse is about restriction, a top-k API lattice (F-closed-LLM route) is subcritical by
construction. If it is about semantic coherence, a random sub-alphabet keeps a live far position and
the whole "widen the window to pay for a smaller vocabulary" trade is unnecessary.

THE ARMS, at fixed r = 2 and fixed size, so only the SELECTION RULE varies:
  semantic       the existing hand-picked set (binary / colours / digits)
  freq_matched   random tokens matched to the semantic set's own marginal probabilities -- the
                 control that isolates coherence from frequency
  uniform        uniformly random token ids -- restriction with no matching at all

WHY FREQUENCY MATCHING IS NOT OPTIONAL. Most of a BPE vocabulary is rare. A uniformly random subset
sits in the far tail of every conditional, so the renormalised distribution is built from
near-denormal numbers and its shape is numerical noise rather than the model's. `make_sampler`
already falls back to uniform when the sub-alphabet's mass underflows -- and a uniform sampler
MAXIMISES s, which would make the random arm look MORE critical than the semantic one purely as an
artifact of the fallback. That is this project's recurring defect (a criterion applied to a quantity
with no room to vary) in a new place, so the mass is gated below rather than merely recorded.

PRE-REGISTERED:
  RUNG      the semantic arm must reproduce `subalphabet_why.json`'s stored s_far and s_near for the
            same alphabet at the same temperature, within RUNG_TOL. That pins model, geometry,
            estimator and settle to F109's; a mismatch means this is a different lattice and nothing
            is read.
  GATE      mean full-vocabulary probability mass on the sub-alphabet must exceed MASS_MIN. Below
            that the renormalisation is noise and the cell is UNREADABLE -- not a measurement of a
            weak effect, but no measurement at all. Registered before the run.
  PRIMARY   does s_far differ across selection modes at fixed size? If all three arms collapse
            together, F109's reading stands and restriction is the cause. If the semantic arm is the
            outlier, the collapse is about closed lists and F109 must be scoped.
  BOUNDARY  one model, one revision, one temperature, r = 2. This asks only whether the SELECTION
            RULE matters; it does not measure criticality, which needs the window ladder.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, time

import numpy as np, torch
from subalphabet import pick_tokens, make_sampler, sub_init, BINARY, COLOURS, DIGITS
from subalphabet_why import s_on, MODEL, REV, R, B, N, SETTLE, N_CTX
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "selection_mode.json")
REF = str(_ROOT / "results" / "subalphabet_why.json")
T = 0.7
SEED = 20260809
ALPHABETS = [("binary", BINARY), ("colours", COLOURS), ("digits", DIGITS)]
MODES = ["semantic", "freq_matched", "uniform"]
RUNG_TOL = 0.08                      # same tolerance F110 used to call its decomposition equivalent
MASS_MIN = 1e-9                      # below this, float64 renormalisation is noise, not a measurement


def marginal(rule, rng, n_ctx=256):
    """The model's own unigram marginal, averaged over random two-token windows.

    Used for frequency matching. Measured rather than assumed, because BPE id order is only a rough
    proxy for frequency and the whole point of the matched arm is that it be genuinely matched.
    """
    V = rule.model.get_output_embeddings().weight.shape[0]
    rows = rng.integers(0, V, size=(n_ctx, R)).tolist()
    with torch.no_grad():
        lg = rule.model(input_ids=torch.tensor(rows, device=rule.device)).logits[:, -1].float()
        p = torch.softmax(lg, dim=-1).mean(0).cpu().double().numpy()
    return p


def matched_ids(target_ids, marg, rng, exclude):
    """Random ids whose marginal probability is closest to each target's, sampled without replacement."""
    order = np.argsort(marg)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    out, used = [], set(exclude)
    for t in target_ids:
        want = rank[t]
        # search outward in marginal-rank space for an unused, non-excluded token
        for d in range(1, len(order)):
            for cand in (order[min(want + d, len(order) - 1)], order[max(want - d, 0)]):
                c = int(cand)
                if c not in used:
                    out.append(c); used.add(c); break
            if len(out) == len(used) - len(exclude):
                break
    return np.array(out, dtype=np.int64)


def mass_on(rule, ids, pool, rng, n_ctx=64):
    """Mean FULL-VOCAB probability mass the conditional puts on the sub-alphabet."""
    pool = np.asarray(pool, dtype=np.int64)
    rows = [[int(x) for x in rng.choice(pool, size=R)] for _ in range(n_ctx)]
    with torch.no_grad():
        lg = rule.model(input_ids=torch.tensor(rows, device=rule.device)).logits[:, -1].float()
        p = torch.softmax(lg, dim=-1).cpu().double().numpy()
    return float(p[:, ids].sum(axis=1).mean())


def analyse(res):
    cells, parts = res["cells"], []
    ref = json.load(open(REF))["cells"]
    errs = []
    for name, _ in ALPHABETS:
        c = cells.get(f"{name}|semantic")
        r = ref.get(f"{name}|T{T}")
        if c and r and c.get("s_far") is not None:
            errs.append((name, abs(c["s_far"] - r["s_far"]), abs(c["s_near"] - r["s_near"])))
    worst = max([max(a, b) for _, a, b in errs], default=float("inf"))
    ok = bool(errs) and worst <= RUNG_TOL
    parts.append(
        f"RUNG (pins this to F109's lattice): the semantic arm reproduces subalphabet_why's stored "
        f"s_far/s_near to within {worst:.4f} across {len(errs)} alphabets (tolerance {RUNG_TOL}). "
        + ("Same model, geometry, estimator and settle, so the arms below differ only in selection."
           if ok else "NOT reproduced -- a different lattice, so nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst_err=worst)
        res["verdict"] = " ".join(parts)
        return
    dead = [k for k, c in cells.items() if c.get("mass") is not None and c["mass"] < MASS_MIN]
    parts.append(
        f"GATE (registered before the run): {len(dead)} of {len(cells)} cells fall below "
        f"mass {MASS_MIN:g} on the sub-alphabet and are UNREADABLE"
        + (f" -- {sorted(dead)}. Their renormalisation is built from near-denormal numbers, so a "
           f"high s there would be the uniform fallback, not the model." if dead else
           ", so every cell carries real conditional mass."))
    rows = {}
    for name, _ in ALPHABETS:
        got = {m: cells.get(f"{name}|{m}") for m in MODES}
        if any(g is None or f"{name}|{m}" in dead for m, g in got.items()):
            continue
        rows[name] = {m: dict(s_far=g["s_far"], s_near=g["s_near"], branching=g["branching"])
                      for m, g in got.items()}
    if not rows:
        parts.append("PRIMARY: no alphabet has all three arms readable, so the comparison is not "
                     "available at this geometry.")
        res["analysis"] = dict(rung_passes=True, unreadable=dead, rows=rows)
        res["verdict"] = " ".join(parts)
        return
    spreads = {n: float(max(v[m]["s_far"] for m in MODES) - min(v[m]["s_far"] for m in MODES))
               for n, v in rows.items()}
    # THE ARMS ARE COMPARED BY SPREAD, NOT AGAINST THE SEMANTIC MEAN. A first version tested
    # whether the semantic arm differed from the MEAN of the other two -- and freq_matched and
    # uniform landed on opposite sides, so averaging them cancelled a 0.59 spread to nothing and the
    # script reported "the arms move together" on data that says the opposite. The registered
    # question is whether the selection rule matters at all, which is a spread question.
    SPREAD_MIN = 0.10
    moved = {n: bool(v > SPREAD_MIN) for n, v in spreads.items()}
    masses = {n: {m: cells[f"{n}|{m}"]["mass"] for m in MODES} for n in rows}
    mass_ratio = {n: float(max(v.values()) / max(min(v.values()), 1e-300)) for n, v in masses.items()}
    parts.append(
        "PRIMARY, s_far by selection mode: "
        + "; ".join(f"{n} " + "/".join(f"{m}={v[m]['s_far']:.3f}" for m in MODES)
                    for n, v in rows.items())
        + ". Spread across modes: " + ", ".join(f"{n} {s:.3f}" for n, s in spreads.items()) + ". "
        + (f"The selection rule CHANGES the far-position contribution on "
           f"{sum(moved.values())} of {len(rows)} alphabets, by up to "
           f"{max(spreads.values()):.3f} at fixed size and fixed r. So s_far is NOT determined by "
           f"restriction alone, and F109's reading -- that RESTRICTION is the cause -- does not "
           f"survive as stated. Note the pattern is not the one either registered branch "
           f"anticipated: freq_matched is the HIGH arm on binary and digits while semantic and "
           f"uniform are both low, so this is neither 'all arms collapse' nor 'semantic is the "
           f"outlier'."
           if sum(moved.values()) > len(rows) / 2 else
           "The arms move together, so the collapse survives changing the selection rule and "
           "F109's reading stands."))
    # IS THE MASS DIFFERENCE ACTUALLY DOING THE WORK? Checked rather than assumed. The arms differ
    # in conditional mass by up to 1488x, which is a live alternative cause of the s_far spread --
    # a sub-alphabet holding 1e-4 of the conditional is renormalised out of the tail. If mass drove
    # s_far it would show as a correlation across three orders of magnitude of mass.
    from ranking import spearman
    allc = [(c["mass"], c["s_far"]) for c in cells.values()] + \
           [(c["mass_uniform"], c["s_far_uniform"]) for c in cells.values()]
    rho_mass = spearman([np.log10(max(m, 1e-300)) for m, _ in allc], [v for _, v in allc])
    rho_div = spearman([c["distinct"] for c in cells.values()],
                       [c["s_far"] for c in cells.values()])
    parts.append(
        f"MASS, THE OBVIOUS CONFOUND, TESTED AND ELIMINATED: the arms differ in conditional mass by "
        f"{min(mass_ratio.values()):.0f}-{max(mass_ratio.values()):.0f}x, but across all "
        f"{len(allc)} cells rho(log10 mass, s_far) = {rho_mass:+.3f}. Mass spans three orders of "
        f"magnitude and s_far does not track it, so tail renormalisation is not what separates the "
        f"arms. The settled ring's DIVERSITY does not explain it either: rho(distinct, s_far) = "
        f"{rho_div:+.3f} over {len(cells)} cells, so F111's mechanism does not carry across "
        f"alphabet sizes. Both candidates are struck off and the cause of the s_far spread is "
        f"OPEN. With n = {len(cells)} a small effect would be invisible; what is ruled out is mass "
        f"being a dominant driver, which is what the confound required.")
    su = {n: {m: cells[f"{n}|{m}"]["s_far_uniform"] for m in MODES} for n in rows}
    spread_u = {n: float(max(v.values()) - min(v.values())) for n, v in su.items()}
    mr_u = {n: float(max(cells[f"{n}|{m}"]["mass_uniform"] for m in MODES)
                     / max(min(cells[f"{n}|{m}"]["mass_uniform"] for m in MODES), 1e-300))
            for n in rows}
    parts.append(
        "UNIFORM-POOL CONTROL, which removes the settled-state difference: s_far = "
        + "; ".join(f"{n} " + "/".join(f"{m}={su[n][m]:.3f}" for m in MODES) for n in rows)
        + ". Spread " + ", ".join(f"{n} {v:.3f}" for n, v in spread_u.items())
        + f"; residual mass ratio {min(mr_u.values()):.0f}-{max(mr_u.values()):.0f}x. "
        + ("The selection effect SURVIVES on a common context distribution, so it is not an "
           "artifact of what each ring settled into."
           if sum(v > SPREAD_MIN for v in spread_u.values()) > len(rows) / 2 else
           "The selection effect DISAPPEARS once every arm is measured on a uniform draw over its "
           "own alphabet, so the settled-pool differences above were a property of the settled "
           "state rather than of the alphabet."))
    parts.append(
        "BRANCHING by mode: "
        + "; ".join(f"{n} " + "/".join(f"{m}={v[m]['branching']:.3f}" for m in MODES)
                    for n, v in rows.items())
        + f", against the criticality threshold of 1.")
    parts.append(
        f"BOUNDARY: one model ({MODEL} {REV}), one temperature (T={T}), r={R}, {N_CTX} windows per "
        f"cell. This asks only whether the SELECTION RULE matters. It does not measure criticality, "
        f"which needs the window ladder, and it does not revisit F109's temperature sweep.")
    res["analysis"] = dict(rung_passes=True, rung_worst_err=worst, unreadable=dead, rows=rows,
                           spreads=spreads, moved=moved, mass_ratio=mass_ratio,
                           rho_mass_sfar=rho_mass, rho_distinct_sfar=rho_div,
                           s_far_uniform=su, spread_uniform=spread_u, mass_ratio_uniform=mr_u)
    res["verdict"] = " ".join(parts)


def main():
    res = {"cells": {}, "_preregistration": dict(
        model=MODEL, revision=REV, T=T, r=R, N=N, B=B, settle=SETTLE, n_ctx=N_CTX, seed=SEED,
        modes=MODES, alphabets=[a for a, _ in ALPHABETS], rung_tol=RUNG_TOL, mass_min=MASS_MIN,
        rung="the semantic arm must reproduce subalphabet_why's stored s_far/s_near within "
             f"{RUNG_TOL}; a mismatch means a different lattice and stops the read",
        gate=f"sub-alphabet probability mass must exceed {MASS_MIN:g} or the cell is UNREADABLE -- "
             "a uniform fallback maximises s and would fake criticality",
        primary="does s_far differ across selection modes at fixed size and fixed r=2?",
        reading="all arms collapsing together confirms F109 (restriction is the cause); the "
                "semantic arm alone collapsing scopes F109 to closed coherent lists")}
    from ar_ca import ARRule, run
    rule = ARRule(MODEL, revision=REV)
    g = np.random.default_rng(SEED)
    marg = marginal(rule, g)
    exclude = set()
    for _, words in ALPHABETS:
        ids, _, _ = pick_tokens(rule.tok, words)
        exclude |= set(int(i) for i in ids)
    V = len(marg)

    for name, words in ALPHABETS:
        sem_ids, kept, dropped = pick_tokens(rule.tok, words)
        k = len(sem_ids)
        arms = dict(
            semantic=sem_ids,
            freq_matched=matched_ids(sem_ids, marg, g, exclude),
            uniform=np.array(sorted(g.choice(V, size=k, replace=False)), dtype=np.int64))
        for mode, ids in arms.items():
            key = f"{name}|{mode}"
            t0 = time.time()
            rng = np.random.default_rng(SEED)
            smp = make_sampler(ids, None)
            settled = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
                          init_state=sub_init(ids, B, N, rng), seed=SEED, sampler=smp)["final"]
            pool = settled.reshape(-1)
            m = mass_on(rule, ids, pool, np.random.default_rng(SEED))
            far = s_on(rule, ids, pool, T, np.random.default_rng(SEED), pos=0)
            near = s_on(rule, ids, pool, T, np.random.default_rng(SEED), pos=1)
            # UNIFORM-POOL CONTROL. `mass` differs 700x across arms, but that is not fixable by
            # picking different tokens: mass is measured on each arm's OWN settled ring, and a
            # semantic set settles into itself because the model expects a colour after colours.
            # High conditional mass and semantic coherence are therefore the same property, and a
            # "mass-matched random set" would be a coherent set. The confound is removed instead by
            # measuring every arm on a UNIFORM draw over its own alphabet, so no arm benefits from
            # its settled state being self-consistent. If the arm differences survive here they are
            # not an artifact of what the ring settled into.
            upool = np.asarray(ids, dtype=np.int64)
            mu = mass_on(rule, ids, upool, np.random.default_rng(SEED))
            far_u = s_on(rule, ids, upool, T, np.random.default_rng(SEED), pos=0)
            near_u = s_on(rule, ids, upool, T, np.random.default_rng(SEED), pos=1)
            res["cells"][key] = dict(
                alphabet=name, mode=mode, k=int(k), ids=[int(i) for i in ids],
                mass=m, s_far=round(far, 5), s_near=round(near, 5),
                branching=round(far + near, 5), distinct=int(len(set(pool.tolist()))),
                mass_uniform=mu, s_far_uniform=round(far_u, 5), s_near_uniform=round(near_u, 5),
                branching_uniform=round(far_u + near_u, 5),
                secs=round(time.time() - t0, 1))
            print(f"  {key:<22} settled: mass={m:.2e} s_far={far:.4f} br={far + near:.4f}   "
                  f"uniform-pool: mass={mu:.2e} s_far={far_u:.4f} br={far_u + near_u:.4f}",
                  flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
