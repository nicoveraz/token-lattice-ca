"""Is F80's non-additivity self-repair? Identify the compensators. (#103)

WHAT F80 LEFT. Ablating all 24 attention layers singly moved lambda_ca by less than the seed
spread in every case (largest |dlam| = 0.0577 at L16 against sd 0.0611), while ablating the early
block moved it +0.345 and the 24 singles SUMMED to -0.224 -- the wrong sign. F80 concluded the
effect "is not localised and not diffuse", which describes the arithmetic without naming a
mechanism.

THE MECHANISM THAT PREDICTS THAT SHAPE. Self-repair (McGrath et al., arXiv:2307.15771): remove one
attention layer and downstream layers compensate, so the single-layer effect collapses toward zero
or overshoots; remove enough at once and the compensation capacity is exhausted, so the joint
effect appears superadditively. The repo cites the Hydra effect only as a NAMING collision
(findings.md:294, for why we avoid "repair" for xi_repair) and has never connected it to F80.

WHY THIS IS AN IDENTIFICATION AND NOT A RE-READING OF A NULL. A dose-response curve over ablation
depth would not discriminate: static redundancy also predicts convex saturation -- ablate 1 of 24
redundant layers and the rest already cover it, ablate 8 and coverage breaks -- with no
compensation anywhere in it. Worse, depth is confounded with position here: at the SAME k=8,
dlam is +0.345 (early), +0.009 (mid), -0.039 (late). Compensation is identifiable in a way
redundancy is not, so the primary measures WHICH layers take over, not how many are removed.

THE PRIMARY. For each downstream layer L (8..23), compare its contribution in two contexts:

    contrib_intact(L) = lambda(none)       - lambda(attn_L{L})
    contrib_early(L)  = lambda(attn_early) - lambda(attn_early + attn_L{L})
    delta(L)          = contrib_early(L) - contrib_intact(L)

Self-repair predicts delta(L) > 0 for specific L: with the early block gone, that layer is doing
MORE, so removing it costs more. Static redundancy predicts no directional increase -- coverage is
passive and nothing is recruited when a peer is removed.

ONLY THE COMPOUND ARMS ARE NEW. lambda(none), lambda(attn_early) and lambda(attn_L{L}) are already
measured (F79 in ablate_lambda.json, F80 in ablate_layers.json). Re-using them across runs assumes
the two runs are comparable, which is an assumption and not a fact, so `none` and `attn_early` are
RE-MEASURED here as a calibration rung and must reproduce their recorded values within the seed
spread. If the harness alone has moved, nothing built on the recorded singles is interpretable and
the verdict is NOT DECIDABLE -- the same gate F79 put on `none`.

WHAT A POSITIVE WOULD AND WOULD NOT BUY. It converts F80 from "the effect is not localised" into a
positive measurement of a named, independently documented mechanism, and it INTERVENES rather than
correlating -- which is what routes 1 (F78 co-timing) and 2 (induction heads) lacked. It would NOT
show that lambda_ca measures self-repair in general: one family, one architecture, greedy-scoped,
carrying the same generality debt as everything else here.

Usage:
    .venv/bin/python experiments/ablate_compensators.py --smoke   # plumbing only, 2 cells
    .venv/bin/python experiments/ablate_compensators.py           # the run (resumable)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"), str(_ROOT / "gatecheck" / "src")]

import argparse
import contextlib
import gc
import json
import os
import time

import numpy as np
import torch
from scipy import stats

from provenance import stamp, rel
from dev_transition_phase3 import measure, BASE, SEEDS, T
from lyapunov import lambda_of, run_ignited
from gatecheck import (NOT_DECIDABLE, carries_verdict, directional, dynamic_range,
                       noise_gate)
from meanfield_lambda import s_crn, lambda_mf
import ar_ca
import ablate_lambda as al

STEP, R, N, B = al.STEP, al.R, al.N, al.B
N_LAYERS = al.N_LAYERS
EARLY = "attn_early"                      # layers 0..7, the block carrying the effect (F79/F80)
DOWNSTREAM = list(range(8, N_LAYERS))     # mid+late: the candidate compensators
OUT = str(_ROOT / "results" / "ablate_compensators.json")

# Recorded levels the calibration rung must reproduce, from the runs that measured them.
REF = {"none": 0.3566, EARLY: 0.0115}
REF_TOL = 0.0611                          # F80's between-seed spread at this geometry
# The compensation this experiment is powered to find. Registered at the scale of F80's own
# largest single-layer effect (|dlam| = 0.0577 at L16): a layer that takes over for the early
# block should move lambda at least that much. A null is only interpretable if the seed floor is
# small enough that an effect this size would have cleared the gate.
MIN_DETECTABLE = 0.05
# Comparability (see the gate in `analyse`). lambda is defined over ignited replicas only, so
# two arms igniting at very different rates are not two conditions, they are two selections.
IGN_TOL = 0.25            # max |ignition_rate(arm) - ignition_rate(reference)| to be compared
MIN_COMPARABLE = 8        # half the downstream layers; below this the comparison is a subset
N_CTX_S = 128             # contexts per arm for `s` (F94/F96 geometry); no seeds, exact

# SEED EXTENSION, AND THE STOPPING RULE THAT MAKES IT LEGITIMATE (F101).
# The registered 8 seeds returned NOT DECIDABLE on power: the measured within-arm spread is 0.0974,
# giving a floor of 0.0974/sqrt(8) = 0.0344, against which MIN_DETECTABLE = 0.05 is only 1.45x --
# under the 2x gate. Since floor scales as 1/sqrt(n), n = 16 reaches 2.05x and n = 20 reaches 2.30x.
# 20 is chosen for margin: at 16 a modest upward drift in the realised spread would land back under
# the gate after another five hours of compute.
#
# ADDING SEEDS AFTER SEEING DATA IS OPTIONAL STOPPING UNLESS THE TARGET IS FIXED FIRST, so it is
# fixed here: the run goes to exactly TOTAL_SEEDS and is decided once, at that n. It is not
# extended again if the answer is unwelcome, and no interim verdict is read as a result. Only the
# PRECISION changes -- no threshold, no statistic and no branch moves -- which is what separates
# this from tuning a criterion to an outcome.
TOTAL_SEEDS = 20


def all_seeds():
    """The registered 8, extended to TOTAL_SEEDS by continuing the same integer sequence."""
    base = list(SEEDS)
    return base + list(range(max(base) + 1, max(base) + 1 + max(0, TOTAL_SEEDS - len(base))))


def specs_for(arm):
    """`arm` -> the list of single specs whose ablations compose it."""
    if arm == "none":
        return []
    return arm.split("+")


@contextlib.contextmanager
def ablating_many(arm):
    """`ablate_lambda.ablating`, generalised to a compound arm.

    NOT hoisted into ablate_lambda.py, and the reason is measured rather than stylistic: editing
    that file marks BOTH ablate_lambda.json and ablate_layers.json stale, and those are the
    multi-hour runs whose numbers this experiment is built on. Re-running them to accommodate a
    refactor would put the premise at risk to tidy the code that tests it. `apply_ablation`
    registers forward hooks, and hooks accumulate, so composing is exactly repeated application.
    """
    orig = ar_ca.ARRule.__init__
    subs = specs_for(arm)

    def patched(self, *a, **kw):
        orig(self, *a, **kw)
        for s in subs:
            al.apply_ablation(self.model, s)

    ar_ca.ARRule.__init__ = patched
    try:
        yield
    finally:
        ar_ca.ARRule.__init__ = orig


@torch.no_grad()
def held_out_loss_many(arm):
    """`ablate_lambda.held_out_loss` for a compound arm, on the identical fixed slice.

    Mirrors that function rather than calling it, for the same reason `ablating_many` does. The
    slice, window and corpus are taken from `ablate_lambda` so the two cannot drift apart on the
    parts that matter -- only the single-spec-vs-list difference lives here.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=STEP).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(dev, torch.float16 if dev != "cpu" else torch.float32)
    for s in specs_for(arm):
        al.apply_ablation(model, s)
    ids = np.load(_ROOT / "data_ar" / "ref_ids.npy")[:al.LOSS_TOKENS]
    tot, n, win = 0.0, 0, 512
    for i in range(0, len(ids) - win, win):
        chunk = torch.tensor(ids[i:i + win][None, :], dtype=torch.long, device=dev)
        tot += float(model(input_ids=chunk, labels=chunk).loss)
        n += 1
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    return tot / max(n, 1)


def settled_pool():
    """The UNABLATED ring's settled state at this checkpoint, used as the fixed context ensemble.

    F96 showed `s` is a property of the ensemble as much as of the model -- on uniformly random
    windows it sits flat at 0.833-0.876, on the states the ring actually occupies it spans 0.331.
    F99 then broke the circularity by holding the ensemble fixed and varying the model, which is
    exactly the shape needed here: every arm is measured on the SAME pool, drawn from the model
    before any ablation. Using each arm's own settled state would reintroduce the circularity F96
    named, and worse, an ablated ring barely settles at all.
    """
    from ar_ca import run
    rule = ar_ca.ARRule(BASE, revision=STEP)
    fin = run(rule, B=8, N=N, r=R, T=T, sweeps=30, scheme="none", seed=SEEDS[0],
              order="per_replica")["final"]
    del rule
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    return fin.reshape(-1).astype(np.int64)


@torch.no_grad()
def s_for_arm(arm, pool, rng, n_ctx=N_CTX_S, batch=32):
    """Exact mean CRN disagreement `s` for this arm's conditional, on windows from `pool`.

    RADIUS-GENERAL, unlike `transplant_s.measure`, which hardcodes a two-token window because the
    developmental grid runs at r=2. The ablation grid runs at r=3, and mean-field criticality is
    s = 1/r -- 0.333 here against 0.5 there -- so measuring at the wrong radius would compare two
    different systems. `s_crn` itself is imported, not reimplemented: the exact part stays single.

    No seeds and no sampling: `s_crn` is a deterministic functional of the conditional pair, so
    this carries none of the ignited-vs-unignited selection problem that the lambda arms do.
    """
    rule = ar_ca.ARRule(BASE, revision=STEP)
    for spec in specs_for(arm):
        al.apply_ablation(rule.model, spec)
    dev, mdl = rule.device, rule.model

    starts = rng.integers(0, len(pool) - R, size=n_ctx)
    base = np.stack([pool[s:s + R] for s in starts])
    rows, keys = [], []
    for w in base:
        pos = int(rng.integers(0, R))
        alt = [int(x) for x in w]
        while alt[pos] == int(w[pos]):
            alt[pos] = int(rng.choice(pool))       # replacement from the SAME ensemble (F56/F70)
        rows += [[int(x) for x in w], alt]
        keys.append(tuple(int(x) for x in w))
    rows = np.array(rows, np.int64)

    probs = []
    for i in range(0, len(rows), batch):
        x = torch.tensor(rows[i:i + batch], device=dev)
        lg = mdl(input_ids=x).logits[:, -1].float()
        probs.append(torch.softmax(lg / T, dim=-1).cpu().double().numpy())
    probs = np.concatenate(probs, 0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    vals = [s_crn(probs[2 * k], probs[2 * k + 1]) for k in range(len(base))]

    del rule, mdl
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    return dict(s=round(float(np.mean(vals)), 5), s_sd=round(float(np.std(vals)), 5),
                lambda_mf=round(lambda_mf(R, float(np.mean(vals))), 5),
                ctx_distinct=int(len(set(keys))), n_ctx=len(vals))


def ignition_rate(res, arm):
    """Mean ignition_prob over an arm's runs -- the FRACTION OF REPLICAS that ignited.

    Distinct from `run_ignited`, which asks whether a whole run produced any damage at all. A run
    can pass that test on a single ignited replica out of B, which is exactly the regime the early
    block puts the lattice in, so the rate is what the comparability gate needs.
    """
    v = [r.get("ignition_prob") for r in res["runs"].values()
         if r["arm"] == arm and r.get("ignition_prob") is not None]
    return float(np.mean(v)) if v else 0.0


def recorded_singles():
    """lambda(attn_L{L}) per downstream layer, from F80's sweep. {} if absent."""
    p = _ROOT / "results" / "ablate_layers.json"
    if not p.exists():
        return {}
    runs = (json.load(open(p)) or {}).get("runs") or {}
    by = {}
    for rec in runs.values():
        by.setdefault(rec.get("ablation"), []).append(rec)
    out = {}
    for L in DOWNSTREAM:
        rs = by.get(f"attn_L{L:02d}") or by.get(f"attn_L{L}")
        if rs:
            ign = [r for r in rs if run_ignited(r)]
            if ign:
                out[L] = float(np.median(lambda_of(ign)))
    return out


def arms(smoke=False):
    compound = [f"{EARLY}+attn_L{L:02d}" for L in DOWNSTREAM]
    if smoke:
        return ["none", EARLY, compound[0]]
    return ["none", EARLY] + compound


def analyse(res, singles):
    """Rung first, then the primary. Every gate is registered before the run."""
    def lam(arm):
        rs = [r for r in res["runs"].values() if r["arm"] == arm]
        ign = [r for r in rs if run_ignited(r)]
        return (float(np.median(lambda_of(ign))) if ign else None), len(ign), len(rs)

    parts, gates = [], []

    # --- the calibration rung: this harness must reproduce the runs it borrows from ------------
    rung = {}
    for arm, want in REF.items():
        got, nign, ntot = lam(arm)
        ok = got is not None and abs(got - want) <= REF_TOL
        rung[arm] = dict(recorded=want, remeasured=got, ignited=f"{nign}/{ntot}", reproduces=ok)
    parts.append("RUNG: " + "; ".join(
        f"{a}: recorded {v['recorded']:+.4f}, re-measured "
        f"{'none' if v['remeasured'] is None else format(v['remeasured'], '+.4f')} "
        f"({v['ignited']} ignited) -> {'reproduces' if v['reproduces'] else 'DOES NOT REPRODUCE'}"
        for a, v in rung.items()))
    if not all(v["reproduces"] for v in rung.values()):
        res["analysis"] = dict(rung=rung, decided=False)
        res["verdict"] = (" ".join(parts) + " NOT DECIDABLE: the harness does not reproduce the "
                          "runs whose recorded singles this comparison borrows, so nothing built "
                          "on them is interpretable.")
        return res["verdict"]

    l_none, l_early = rung["none"]["remeasured"], rung[EARLY]["remeasured"]
    ign_ref = ignition_rate(res, EARLY)
    # THE FLOOR IS THE STANDARD ERROR OF A PER-ARM CENTRE, AND IT MUST USE THE ARM'S OWN n.
    # The first version divided the pooled spread by sqrt(len(SEEDS)) -- the REGISTERED 8 -- which
    # stayed 8 after the seed extension while 20 seeds were actually being averaged, overstating
    # the noise by sqrt(2.5) = 1.58x and refusing a run that was in fact powered. Per-arm n also
    # matters in its own right: F42 drops unignited runs, so arms here carry between 13 and 20
    # ignited seeds, and a single sqrt(n) would overstate precision for the sparse ones. Computing
    # the standard error per arm and averaging those handles both, and is slightly MORE
    # conservative than pooling then dividing by sqrt(20) (0.02376 against 0.02316).
    ses = []
    for a in {r["arm"] for r in res["runs"].values()}:
        rs = [r for r in res["runs"].values() if r["arm"] == a and run_ignited(r)]
        if len(rs) > 1:
            sd = float(np.std(lambda_of(rs)))
            if np.isfinite(sd) and sd > 0:
                ses.append(sd / np.sqrt(len(rs)))
    floor = float(np.mean(ses)) if ses else None

    # --- the primary: does any downstream layer do MORE once the early block is gone? ----------
    rows = []
    for L in DOWNSTREAM:
        if L not in singles:
            continue
        l_comp, nign, ntot = lam(f"{EARLY}+attn_L{L:02d}")
        if l_comp is None:
            continue
        contrib_intact = l_none - singles[L]
        contrib_early = l_early - l_comp
        ign_comp = ignition_rate(res, f"{EARLY}+attn_L{L:02d}")
        rows.append(dict(layer=L, contrib_intact=round(contrib_intact, 5),
                         contrib_early=round(contrib_early, 5),
                         delta=round(contrib_early - contrib_intact, 5),
                         ignited=f"{nign}/{ntot}", ignition_rate=round(ign_comp, 4),
                         comparable=bool(abs(ign_comp - ign_ref) <= IGN_TOL)))

    if not rows or floor is None:
        res["analysis"] = dict(rung=rung, rows=rows, decided=False)
        res["verdict"] = (" ".join(parts) + " NOT DECIDABLE: no downstream arm produced an "
                          "ignited estimate, so lambda is undefined for the comparison (F42).")
        return res["verdict"]

    # THE COMPARABILITY GATE. lambda is defined only over IGNITED replicas (F42), so a difference
    # between two arms that ignite at very different rates is a difference between two differently
    # SELECTED subsets, not between two conditions. This is not hypothetical here: `attn_early`
    # ignites at ~0.156 against `none`'s ~1.000, and the smoke run found a compound arm igniting
    # at 0.375 where its own reference ignited at 0.000 -- adding an ablation made the lattice MORE
    # alive. Rows whose ignition rate is far from the reference's are dropped rather than compared,
    # and the drop is reported, because a silently shrinking denominator is how a selection effect
    # gets read as an identification.
    dropped = [r for r in rows if not r["comparable"]]
    rows = [r for r in rows if r["comparable"]]

    # THE DROPPED ARMS ARE AN OBSERVABLE, NOT ONLY AN EXCLUSION. Filtering them and moving on is
    # how a finding gets dismissed as an artifact: "ablating an ADDITIONAL layer revives a lattice
    # the early block had frozen" is a claim about the system, not a defect in the measurement.
    # The smoke run showed it at one seed (`attn_early` ignited 0.000, `attn_early+attn_L08`
    # ignited 0.375). Registered here with its own direction and floor so it can become a result
    # rather than a footnote to one -- and reported whether or not the primary decides.
    revivals = sorted((r for r in (dropped + rows) if r["ignition_rate"] > ign_ref + IGN_TOL),
                      key=lambda r: -r["ignition_rate"])
    suppress = sorted((r for r in (dropped + rows) if r["ignition_rate"] < ign_ref - IGN_TOL),
                      key=lambda r: r["ignition_rate"])
    if revivals or suppress:
        parts.append(
            f"REVIVAL: against a reference igniting at {ign_ref:.3f}, "
            f"{len(revivals)} downstream ablations RAISE ignition"
            + (" (" + ", ".join(f"L{r['layer']}:{r['ignition_rate']:.3f}"
                                for r in revivals[:5]) + ")" if revivals else "")
            + f" and {len(suppress)} lower it"
            + (" (" + ", ".join(f"L{r['layer']}:{r['ignition_rate']:.3f}"
                                for r in suppress[:5]) + ")" if suppress else "")
            + ". Removing MORE of the network making damage spread FURTHER is not what any "
              "monotone account of ablation predicts, and it is measured here rather than "
              "filtered away.")
    parts.append(
        f"COMPARABILITY: reference `{EARLY}` ignites at {ign_ref:.3f}; "
        f"{len(rows)} of {len(rows) + len(dropped)} downstream arms are within {IGN_TOL} of it."
        + ("" if not dropped else
           " DROPPED as not comparable: "
           + ", ".join(f"L{r['layer']}({r['ignition_rate']:.3f})" for r in dropped) + "."))
    if len(rows) < MIN_COMPARABLE:
        res["analysis"] = dict(rung=rung, rows=rows, dropped=dropped,
                               reference_ignition=round(ign_ref, 4), decided=False)
        res["verdict"] = (" ".join(parts) + f" NOT DECIDABLE: only {len(rows)} arms ignite "
                          f"comparably to the reference, under the {MIN_COMPARABLE} required. "
                          f"Ablating the early block leaves the lattice barely alive, so the "
                          f"comparison is between differently-selected replica subsets rather "
                          f"than between conditions. The fix is a reference context that keeps "
                          f"the lattice ignited, not more seeds.")
        return res["verdict"]

    deltas = [r["delta"] for r in rows]
    best = max(rows, key=lambda r: r["delta"])

    # THE BLOCKING GATE IS POWER, NOT RANGE, and getting this wrong would have inverted the null.
    # A range gate over delta(L) asks whether the series varies -- but a FLAT series at zero is
    # exactly the KILL this experiment is registered to detect, so gating on range would report
    # "no compensation anywhere", the predicted null, as NOT DECIDABLE. What actually has to hold
    # for a null to mean anything is that a compensation of the size we care about would have been
    # visible: the seed floor must be small enough that MIN_DETECTABLE clears a 2x gate. That is
    # `noise_gate` applied to the effect size rather than to the observation.
    gates.append(noise_gate(MIN_DETECTABLE, floor))
    verdict = carries_verdict(gates, value=best)
    # Selects the branch once power is established; not blocking, for the same reason `directional`
    # is not -- "the largest delta is inside the noise" is the null, not an inability to decide.
    big_enough = noise_gate(best["delta"], floor)
    # MULTIPLE COMPARISONS. `best` is the MAXIMUM of len(rows) deltas, not a pre-specified layer,
    # so its per-comparison p understates the chance of seeing that much by selection alone. The
    # correction is family-wise over the layers actually compared, and it is added with the data
    # in view -- admissible only because it can move the verdict in ONE direction, away from a
    # positive, exactly like the noise-gate tightening in F100.
    z_best = best["delta"] / floor if floor else 0.0
    p_one = float(stats.norm.sf(z_best))
    p_fw = float(1.0 - (1.0 - p_one) ** len(rows))
    # Directional, and NOT a blocking gate: a negative delta is evidence AGAINST self-repair, not
    # an inability to decide -- the reading gatecheck.directional exists to forbid.
    dirn = directional(best["delta"], expect="increase", floor=floor)

    parts.append(
        f"PRIMARY: largest delta is L{best['layer']} at {best['delta']:+.5f} "
        f"(contribution {best['contrib_intact']:+.5f} intact -> {best['contrib_early']:+.5f} with "
        f"the early block ablated), against a seed floor of {floor:.5f}. "
        f"{sum(1 for d in deltas if d > floor)} of {len(deltas)} layers exceed the floor.")

    if verdict.status == NOT_DECIDABLE:
        parts.append(
            f"NOT DECIDABLE: the run is underpowered for its own question -- {verdict.reason}. A "
            f"compensation of {MIN_DETECTABLE} would not have cleared this floor, so neither the "
            f"positive nor the kill can be read. More seeds, not more layers.")
        decided = False
    elif not (dirn.usable and big_enough.usable and p_fw < 0.05):
        parts.append(
            f"KILL: no downstream layer increases its contribution when the early block is "
            f"removed. Largest delta L{best['layer']} = {best['delta']:+.5f} is {z_best:.2f}x the "
            f"seed floor, one-sided p = {p_one:.4f}, but FAMILY-WISE p = {p_fw:.4f} over "
            f"{len(rows)} layers -- it was selected as the maximum, not predicted. "
            f"{dirn.reason if not dirn.usable else big_enough.reason} Static redundancy "
            f"accounts for F80's non-additivity without compensation, self-repair is not the "
            f"mechanism, and explanandum route 5 closes. A NULL IS A GOOD RESULT.")
        decided = True
    else:
        parts.append(
            f"COMPENSATION: L{best['layer']} does measurably more once the early block is gone, "
            f"which is the identifying signature no static-redundancy account predicts "
            f"(family-wise p = {p_fw:.4f} over {len(rows)} layers). "
            f"Gates: {verdict.reason}")
        decided = True

    # --- the competing account, registered because it can DISSOLVE the primary ----------------
    # The rule IS p(x_i | x_{i-r..i-1}). Damage spreads only when a flipped neighbour changes the
    # conditional enough that two CRN twins draw different tokens -- which is `s` exactly. Early
    # attention is where the window enters the computation, so zeroing it drives the rule toward a
    # constant map, twins draw identically, and damage heals by the same exact-zero property the
    # null test relies on. If s(early) sits below the mean-field critical 1/r while s(none) sits
    # above it, the ignition collapse is predicted with NO compensation anywhere in the account,
    # and a positive delta would need to survive that explanation rather than ignore it.
    sblock = res.get("s") or {}
    if sblock:
        crit = 1.0 / R
        s_none = (sblock.get("none") or {}).get("s")
        s_ref = (sblock.get(EARLY) or {}).get("s")
        revived = [r["layer"] for r in rows
                   if (sblock.get(f"{EARLY}+attn_L{r['layer']:02d}") or {}).get("s", 0) > (s_ref or 0)]
        crosses = (s_none is not None and s_ref is not None
                   and s_none > crit >= s_ref)
        parts.append(
            f"SENSITIVITY: s(none)={s_none}, s({EARLY})={s_ref}, mean-field critical 1/r={crit:.4f}"
            + (f". The early block carries the rule ACROSS the critical point, so the ignition "
               f"collapse follows from annealed mean field with no compensation in the account; "
               f"{len(revived)} downstream arms raise s back above the reference, which is the "
               f"same mechanism running in reverse."
               if crosses else
               ". The early block does NOT cross the critical point, so the sensitivity account "
               "does not by itself explain the ignition collapse and the compensation reading "
               "keeps its standing."))
        res.setdefault("analysis", {})
        sens = dict(s_none=s_none, s_reference=s_ref, critical=round(crit, 5),
                    crosses_critical=bool(crosses), arms_raising_s=revived)
    else:
        sens = None

    parts.append(
        "BOUNDARY: this attributes F80's non-additivity to a named mechanism in ONE model of ONE "
        "family at one checkpoint, under greedy decoding. It does not show lambda_ca measures "
        "self-repair in general, and the generality debt is unchanged.")

    res["analysis"] = dict(rung=rung, rows=rows, dropped=dropped,
                           reference_ignition=round(ign_ref, 4),
                           revivals=[r["layer"] for r in revivals],
                           suppressions=[r["layer"] for r in suppress],
                           seed_floor=round(floor, 5),
                           min_detectable=MIN_DETECTABLE, ignition_tolerance=IGN_TOL,
        comparability=f"lambda is defined over ignited replicas only (F42), so arms igniting more "
                      f"than {IGN_TOL} away from the reference's rate are DROPPED and reported; "
                      f"fewer than {MIN_COMPARABLE} survivors is NOT DECIDABLE",
                           gates=[g.block() for g in gates], directional=dirn.block(),
                           largest_delta_vs_floor=big_enough.block(),
                           best_z=round(z_best, 4), p_one_sided=round(p_one, 5),
                           p_family_wise=round(p_fw, 5), n_compared=len(rows),
                           mean_delta=round(float(np.mean(deltas)), 5),
                           n_positive=int(sum(1 for d in deltas if d > 0)), sensitivity=sens,
                           decided=decided)
    res["verdict"] = " ".join(parts)
    return res["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="3 arms x 1 seed: checks the plumbing, decides nothing")
    a = ap.parse_args()

    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}, "loss": {}}
    seeds = list(SEEDS)[:1] if a.smoke else all_seeds()
    todo_arms = arms(a.smoke)

    res["_preregistration"] = dict(
        issue=103, base=BASE, step=STEP, r=R, N=N, B=B, T=T, seeds=seeds,
        seed_extension=f"the registered 8 seeds returned NOT DECIDABLE on power (floor 0.0344, "
                       f"MIN_DETECTABLE/floor = 1.45x under a 2x gate). Extended to "
                       f"{TOTAL_SEEDS} for a projected 2.30x. The target is FIXED IN ADVANCE and "
                       f"decided once at that n -- not extended again if the answer is unwelcome. "
                       f"Only precision changes; no threshold, statistic or branch moves",
        early_block=EARLY, downstream=DOWNSTREAM, arms=todo_arms,
        primary="does any downstream layer L increase its contribution when the early block is "
                "ablated -- delta(L) = [lambda(early) - lambda(early+L)] - [lambda(none) - "
                "lambda(L)] > 0 above the seed floor?",
        rung=f"`none` and `{EARLY}` must re-measure within {REF_TOL} of their recorded "
             f"{REF} or the borrowed singles are not comparable and nothing is decidable",
        kill="no downstream layer increases its contribution -> static redundancy explains F80's "
             "non-additivity without compensation, self-repair is not the mechanism, and "
             "explanandum route 5 closes. A NULL IS A GOOD RESULT",
        min_detectable=MIN_DETECTABLE,
        not_decidable=f"the seed floor is too large for a compensation of {MIN_DETECTABLE} to "
                      f"clear a 2x gate (power), or arms fail to ignite (F42). NOTE: a flat "
                      f"delta series is NOT undecidable -- it is the kill",
        multiple_comparisons="the largest delta is the MAX over the compared layers, so its "
                             "significance is family-wise, not per-comparison",
        directional="reported, NOT blocking: a negative delta is evidence against self-repair "
                    "rather than an inability to decide",
        reuse="lambda(attn_L{L}) read from results/ablate_layers.json (F80), gated on the rung",
        secondary=f"IGNITION REVIVAL, registered in its own right: downstream ablations whose "
                  f"ignition rate exceeds the reference's by more than {IGN_TOL}. Removing more "
                  f"of the network making damage spread further contradicts any monotone account "
                  f"of ablation. Reported whether or not the primary decides, so a comparability "
                  f"exclusion cannot quietly dispose of it",
        competing_account=f"single-token sensitivity s, measured exactly (s_crn, no seeds) on a "
                          f"FIXED ensemble from the unablated ring so the model varies and the "
                          f"ensemble does not (F96/F99). If s(none) > 1/r >= s({EARLY}), annealed "
                          f"mean field predicts the ignition collapse with no compensation in the "
                          f"account, and a positive delta must survive that rather than ignore it",
        boundary="attributes non-additivity to a mechanism in one model, one family, greedy",
        resumable="keyed by (arm, seed)")

    for arm in todo_arms:
        if arm not in res["loss"]:
            t0 = time.time()
            res["loss"][arm] = round(held_out_loss_many(arm), 4)
            print(f"  loss[{arm:22s}] = {res['loss'][arm]:8.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

    res.setdefault("s", {})
    if any(a_ not in res["s"] for a_ in todo_arms):
        pool = settled_pool()
        rng = np.random.default_rng(SEEDS[0])
        for arm in todo_arms:
            if arm in res["s"]:
                continue
            t0 = time.time()
            res["s"][arm] = s_for_arm(arm, pool, rng)
            r_ = res["s"][arm]
            print(f"  s[{arm:22s}] = {r_['s']:.5f} +/- {r_['s_sd']:.5f}  "
                  f"lambda_MF={r_['lambda_mf']:+.4f}  ({r_['ctx_distinct']} distinct ctx, "
                  f"{time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

    todo = [(a_, s) for a_ in todo_arms for s in seeds if f"{a_}|s{s}" not in res["runs"]]
    print(f"\n{len(res['runs'])} cached, {len(todo)} lambda cells to run "
          f"(~{len(todo)*140/3600:.1f} h)\n", flush=True)

    for arm, seed in todo:
        t0 = time.time()
        with ablating_many(arm):
            lam, dn, md, ig = measure(STEP, N, B, seed, r=R)
        res["runs"][f"{arm}|s{seed}"] = dict(
            arm=arm, seed=seed, step=STEP, r=R, N=N, B=B, T=T,
            lambda_ca=lam, D_norm=dn, mean_damage=md, ignition_prob=ig,
            secs=round(time.time() - t0, 1))
        print(f"  {arm:22s} s={seed}  lambda={lam:+.4f}  D_norm={dn:.4f}  ign={ig:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    if a.smoke:
        print("\nSMOKE: plumbing only, no verdict. Re-run without --smoke for the experiment.")
        json.dump(res, open(OUT, "w"), indent=1)
        return 0

    print("\n  -> " + analyse(res, recorded_singles()))
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (f"#103. Compound arms are new; lambda(none), lambda({EARLY}) are re-measured "
                    f"as a rung and lambda(attn_L*) is read from F80's sweep, gated on the rung.")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
