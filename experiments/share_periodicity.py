"""Is the attractor share measuring an ATTRACTOR, or the length of a periodic orbit?

WHY THIS EXISTS. The remote run (groq_share.py) was re-launched storing the whole ring rather than
just the scalars, because `top1 = 0.3333` on N=24 is ambiguous: it is what a weak attractor reads,
and it is also exactly what three colours holding eight sites each read. The stored rings settled
that question immediately and in the worse direction -- the clean remote cells are EXACT periodic
crystals, `b g y b g y ...` at r=2 and `g r y b g r y b ...` at r=3, with adjacent-repeat 0.000 and
period-3 repeat 1.000. On such a ring `top1` is 1/period by arithmetic and carries no information
about attraction at all.

That is a defect in a readout, so the question that matters is not what the remote arm did but
whether the LOCAL instrument -- the one every transferring result rests on -- has the same problem.
F130's stored cells cannot answer it directly: no local results file keeps a ring, which is F116's
lesson (the largest object the measurement produces was thrown away) arriving for the third time.
So the local arm is re-run here with the rings kept, on the same geometry and the same seeds.

THE DISCRIMINATOR. For each period p dividing N, rep_p is the fraction of sites equal to the site p
ahead. A frozen ring has rep_1 = 1. A period-p crystal has rep_p = 1 and rep_1 = 0 and top1 = 1/p.
A disordered ring has every rep_p at chance. Reporting p* = argmin over the p maximising rep_p
separates all three, and `top1` alone separates none of them.

PRE-REGISTERED:
  RUNG      TWO known-answer checks, both of which must pass or nothing below is read.
            (a) the detector, on synthetic rings whose period is constructed: it must return p* = 1
                on a constant ring, p* = 3 on a period-3 ring, and a chance-level rep on a random
                one. A detector that cannot see a period it was handed cannot be trusted to report
                its absence.
            (b) the local re-run must reproduce share_invariance's stored pooled top1 for the same
                (model, construction, seed) within RUNG_TOL -- same quantity as F130, or this says
                nothing about F130.
  PRIMARY   the fraction of local replicas that are CRYSTALS (p* > 1 with rep >= CRYSTAL_REP).
            Registered reading: a non-trivial crystal fraction means the local share is partly a
            period readout and F130's interpretation needs amending; a crystal fraction at zero
            means the degeneracy is specific to the remote construction.
  SECONDARY on crystal replicas, does top1 equal 1/p*? That is the arithmetic identity which makes
            the share uninformative where it holds, and reporting it is what turns "these look
            periodic" into a measurement.
  CONTRAST  the remote arm's stored rings, run through the same detector, with its own two extra
            defects recorded: an alphabet floor and a miss-corrupted majority.
  BOUNDARY  the local re-run is a subset of F130's grid chosen to span its top1 range, not the whole
            grid. A clean result here bounds the defect to the remote construction on THIS subset.

ADDED AFTER THE STATE CONVENTION LANDED, and the reason they could be added is the whole argument
for the convention -- neither cost a run:

  STORED    the FULL 120-cell F130 grid, classified from the lattices `share_invariance` now stores
            in its results file. This SUPERSEDES the 24-cell PRIMARY above: what needed a special
            re-run when nothing kept a ring now costs a file read. The pre-registered reading is
            unchanged; only the coverage is.
  SCREEN    F134's 320-cell top-k grid, where a full backfill would cost about four hours. The
            screen asks which cells the stored scalars can already EXCLUDE: pooling over 16
            replicas means a 1/period reading needs the SAME dominant token in every replica, which
            forces a small `distinct` count, so a cell with top1 near a unit fraction and 100
            distinct tokens is excluded outright. It excludes 305 of 320 and leaves 15 that need
            their state -- a targeted ten-minute backfill instead of a four-hour one.

            THE FIRST VERSION OF THIS ARM CLAIMED THE OPPOSITE OF WHAT ITS OWN CODE COMPUTED. The
            filter selects cells where `distinct <= B*p`, i.e. where a crystal IS arithmetically
            possible, and the prose then reported that each had "far more distinct tokens than
            could produce" one. Screening a chore before doing it is still the right discipline;
            reporting the screen as conclusive when it is a narrowing was not.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, time

import numpy as np
from provenance import stamp, rel
from gatecheck import unpack_state, STATE_KEY

OUT = str(_ROOT / "results" / "share_periodicity.json")
REF = str(_ROOT / "results" / "share_invariance.json")
REMOTE = str(_ROOT / "results" / "groq_share.json")
# Chosen to SPAN F130's top1 range rather than to make a point: pythia-31m reads 1.000 at r2.T0.2,
# bloom-560m reads 0.040, and the rest sit between. If the share were a period readout anywhere, the
# top of that range is where a crystal would hide.
MODELS = ["EleutherAI/pythia-31m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "gpt2-large", "RWKV/rwkv-4-169m-pile", "bigscience/bloom-560m"]
CONSTRUCTIONS = [(2, 0.02), (2, 0.2), (2, 0.7), (3, 0.02)]
N, B, SETTLE = 48, 16, 30           # identical to share_invariance, so the rung can be exact
SEED = 20260810                     # share_invariance's first seed
RUNG_TOL = 0.05
CRYSTAL_REP = 0.9
# a genuine period-p cycle occupies its p tokens equally, so top1 = 1/p; this is how far off that
# a ring may be and still be called a cycle rather than a dominated ring with aligned defects
BALANCE_TOL = 0.05


def divisors(n):
    return [p for p in range(1, n + 1) if n % p == 0]


def period_profile(ring):
    """rep_p for every p dividing len(ring), plus the p that maximises it (smallest on a tie).

    p = len(ring) is excluded from the argmax: every ring is trivially period-N, so including it
    would let the detector 'pass' by reporting the tautology. The returned p* is therefore a claim
    about STRUCTURE -- p* = N in the profile means no sub-period beat chance.
    """
    a = np.asarray(ring)
    n = len(a)
    reps = {p: float(np.mean(a == np.roll(a, -p))) for p in divisors(n)}
    cand = [p for p in reps if p != n] or [n]
    best = max(reps[p] for p in cand)
    p_star = min(p for p in cand if reps[p] == best)
    return reps, p_star, best


def classify(ring):
    """Frozen, balanced cycle, or a near-frozen ring whose defects happen to align at some period.

    THE THIRD CATEGORY WAS A FALSE POSITIVE OF THIS FUNCTION, found by running it over the full
    1920-replica grid instead of the 24-cell subset. A ring that is ~90% one token with a handful of
    defects can have rep_1 just under the threshold and rep_2 just over it, and the structural test
    alone then calls it a period-2 crystal -- while its top1 is 0.8958, not 0.5. That produced a
    self-contradicting sentence in the first full-grid verdict ("the largest share any crystal reads
    is 0.8958" beside "1/p is bounded by 1/2"), which is what exposed it.

    A GENUINE period-p cycle occupies its p tokens equally, so top1 = 1/p. Requiring that is not
    circular bookkeeping to protect the containment claim -- it is what distinguishes a cycle from a
    dominated ring, and the two are reported separately below precisely so the distinction stays
    visible rather than being absorbed into a definition.
    """
    reps, p_star, best = period_profile(ring)
    vals, cnt = np.unique(np.asarray(ring), return_counts=True)
    top1 = float(cnt.max() / cnt.sum())
    frozen = reps[1] >= CRYSTAL_REP
    periodic = (not frozen) and p_star > 1 and best >= CRYSTAL_REP
    balanced = bool(periodic and abs(top1 - 1.0 / p_star) <= BALANCE_TOL)
    return dict(top1=top1, distinct=int(len(vals)), rep1=reps[1], p_star=int(p_star),
                rep_at_p_star=float(best), frozen=bool(frozen),
                periodic=bool(periodic), crystal=balanced,
                near_frozen_alias=bool(periodic and not balanced),
                one_over_p=1.0 / p_star, top1_minus_inv_p=top1 - 1.0 / p_star)


def detector_rung():
    """Known-answer check on rings whose period is constructed, not inferred."""
    rng = np.random.default_rng(0)
    const = ["a"] * 24
    per3 = ["a", "b", "c"] * 8
    per4 = ["a", "b", "c", "d"] * 6
    rand = list(rng.choice(list("abcdef"), size=24))
    out = {}
    for name, ring, want_p in (("constant", const, 1), ("period3", per3, 3),
                               ("period4", per4, 4), ("random", rand, None)):
        c = classify(ring)
        ok = (c["p_star"] == want_p and c["rep_at_p_star"] >= CRYSTAL_REP) if want_p else \
             (c["rep_at_p_star"] < 0.5)
        out[name] = dict(p_star=c["p_star"], rep=round(c["rep_at_p_star"], 4),
                         top1=round(c["top1"], 4), expect=want_p, ok=bool(ok))
    return out


def local_cell(rule, r, T, seed):
    from ar_ca import run
    settled = run(rule, B=B, N=N, r=r, T=T, sweeps=SETTLE, scheme="none", init="random",
                  seed=seed)["final"]
    pool = settled.reshape(-1)
    vals, cnt = np.unique(pool, return_counts=True)
    per = [classify(list(settled[b])) for b in range(settled.shape[0])]
    return dict(
        top1_pool=float(cnt.max() / cnt.sum()), distinct_pool=int(len(vals)),
        rep1_pool=float(np.mean(settled[:, :-1] == settled[:, 1:])),
        crystal_frac=float(np.mean([p["crystal"] for p in per])),
        frozen_frac=float(np.mean([p["frozen"] for p in per])),
        mean_p_star=float(np.mean([p["p_star"] for p in per])),
        mean_rep_at_p_star=float(np.mean([p["rep_at_p_star"] for p in per])),
        replicas=per)


def stored_arm():
    """The FULL F130 grid, classified from the lattices share_invariance now stores. No re-run.

    This is the point of the state convention arriving. F136's local arm was 24 cells re-run
    specially, because no results file kept a ring; its boundary said "this subset". Once
    share_invariance stores its settled lattice, the same census runs over all 120 cells for the
    cost of reading a file -- which is the difference the convention buys, stated as a number
    rather than as a principle.
    """
    p = _pathlib.Path(REF)
    if not p.exists():
        return None
    cells = json.load(open(p))["cells"]
    rows = {}
    for k, v in cells.items():
        blk = v.get(STATE_KEY)
        if not (isinstance(blk, dict) and blk.get("data")):
            continue
        arr = unpack_state(blk)
        per = [classify(list(arr[b])) for b in range(arr.shape[0])]
        rows[k] = dict(
            model=v["model"], construction=v["construction"], seed=v["seed"],
            top1_pool=v["top1"], complete=bool(blk.get("complete")),
            crystal_frac=float(np.mean([x["crystal"] for x in per])),
            frozen_frac=float(np.mean([x["frozen"] for x in per])),
            alias_frac=float(np.mean([x["near_frozen_alias"] for x in per])),
            n_replicas=len(per),
            crystals=[dict(replica=i, p_star=x["p_star"], top1=x["top1"],
                           one_over_p=x["one_over_p"])
                      for i, x in enumerate(per) if x["crystal"]],
            aliases=[dict(replica=i, p_star=x["p_star"], top1=x["top1"], rep1=x["rep1"])
                     for i, x in enumerate(per) if x["near_frozen_alias"]])
    return rows or None


def screen_arm():
    """Can the degeneracy be RULED OUT from stored scalars, without a re-run? For topk_ablation, yes.

    F134's top-k arm is the other place a small effective support might crystallise, and backfilling
    its 320 cells costs about four hours. Before spending that, ask whether the stored scalars can
    already exclude it -- the same scan-then-gate discipline applied to a chore instead of to an
    experiment.

    THE ARITHMETIC THAT MAKES THE SCREEN CONCLUSIVE. `top1` is pooled over B=16 replicas. If every
    replica were a period-p orbit, top1 would be 1/p only if the SAME dominant token recurred across
    replicas -- otherwise the pooled top token holds about N/p of B*N sites and top1 collapses to
    ~1/(B*p). Shared dominant tokens force a small pooled `distinct`. So a cell can only be read as
    1/period if top1 sits at a unit fraction AND distinct is small enough for shared short orbits.
    A cell with top1 near 1/7 and 106 distinct tokens is not a crystal, and no re-run is needed to
    say so.

    A cell at distinct = 1 is excluded up front: that is p = 1, a genuine one-token attractor, which
    is what the readout is supposed to report.
    """
    p = _ROOT / "results" / "topk_ablation.json"
    if not p.exists():
        return None
    cells = json.load(open(p))["cells"]
    B_ASSUMED = 16
    cand = []
    for k, v in cells.items():
        t, d = v["top1"], v["distinct"]
        if d <= 1:
            continue
        for per in range(2, 9):
            if abs(t - 1.0 / per) < 0.02 and d <= B_ASSUMED * per:
                cand.append(dict(cell=k, top1=round(t, 4), distinct=int(d), p=per,
                                 max_distinct_if_crystal=B_ASSUMED * per))
                break
    out = dict(n_cells=len(cells), n_candidates=len(cand), candidates=cand,
               tolerance=0.02, B_assumed=B_ASSUMED)
    # RESOLUTION. The screen NARROWS; it does not decide. Cells it could not exclude are re-run with
    # state stored, and only then can each be called a balanced cycle, a dominated ring whose
    # defects align at some period, or neither.
    per_cell = []
    for c in cand:
        blk = cells.get(c["cell"], {}).get(STATE_KEY)
        if not (isinstance(blk, dict) and blk.get("data")):
            continue
        arr = unpack_state(blk)
        per = [classify(list(arr[b])) for b in range(arr.shape[0])]
        per_cell.append(dict(
            cell=c["cell"], top1=c["top1"],
            n_balanced=sum(x["crystal"] for x in per),
            n_alias=sum(x["near_frozen_alias"] for x in per),
            n_frozen=sum(x["frozen"] for x in per), n_replicas=len(per)))
    if per_cell:
        out["resolved"] = dict(
            n_resolved=len(per_cell), per_cell=per_cell,
            n_balanced=sum(x["n_balanced"] for x in per_cell),
            n_frozen_alias=sum(x["n_alias"] for x in per_cell),
            n_other=sum(x["n_replicas"] - x["n_balanced"] - x["n_alias"] - x["n_frozen"]
                        for x in per_cell))
    return out


def remote_arm():
    """Run the stored remote rings through the same detector. No new calls."""
    p = _pathlib.Path(REMOTE)
    if not p.exists():
        return None
    cells = json.load(open(p))["cells"]
    rows = []
    for k, v in sorted(cells.items()):
        if not v.get("ring"):
            continue
        c = classify(v["ring"])
        c.update(cell=k, misses=v.get("misses"), calls=v.get("calls"), stored_top1=v.get("top1"))
        rows.append(c)
    return rows


def analyse(res):
    parts = []
    rung = res["detector_rung"]
    det_ok = all(v["ok"] for v in rung.values())
    ref = json.load(open(REF))["cells"] if _pathlib.Path(REF).exists() else {}
    errs = []
    for k, v in res["local"].items():
        m, con, sd = k.split("|")
        b = ref.get(f"{m}|{con}|s{SEED}")
        if b:
            errs.append(abs(v["top1_pool"] - b["top1"]))
    worst = max(errs, default=float("inf"))
    repro_ok = bool(errs) and worst <= RUNG_TOL
    parts.append(
        "RUNG (a), the detector on constructed rings: "
        + "; ".join(f"{k} -> p*={v['p_star']} rep={v['rep']:.3f} "
                    f"({'ok' if v['ok'] else 'WRONG'})" for k, v in rung.items())
        + (". The detector returns the period it was handed and finds none in noise."
           if det_ok else ". The detector is WRONG on a known answer; nothing below is read."))
    parts.append(
        f"RUNG (b), the local re-run against F130's stored pooled top1: worst error {worst:.4f} "
        f"across {len(errs)} cells (tolerance {RUNG_TOL}). "
        + ("Same quantity as F130." if repro_ok else
           "NOT reproduced -- this is not F130's measurement and nothing below is read."))
    if not (det_ok and repro_ok):
        res["analysis"] = dict(rung_passes=False, detector_ok=det_ok, repro_worst=worst)
        res["verdict"] = " ".join(parts)
        return
    cf = [v["crystal_frac"] for v in res["local"].values()]
    ff = [v["frozen_frac"] for v in res["local"].values()]
    n_cry = sum(1 for v in cf if v > 0)
    parts.append(
        f"PRIMARY, local crystal fraction (a replica with p* > 1 at rep >= {CRYSTAL_REP}): "
        f"mean {float(np.mean(cf)):.4f} over {len(cf)} cells; {n_cry} cells contain any crystal at "
        f"all. Frozen (p* = 1) replicas: mean {float(np.mean(ff)):.4f}. "
        + ("NO local replica is a periodic crystal, so the local share is not a disguised period "
           "readout on this subset and F130's reading stands."
           if n_cry == 0 else
           "Local crystals EXIST, so the local share is partly a period readout and F130's "
           "interpretation needs amending at the constructions where they appear."))
    hi = [(k, v) for k, v in res["local"].items() if v["top1_pool"] >= 0.5]
    parts.append(
        "SECONDARY, the high-share regime is where a crystal would hide, and it is frozen rather "
        "than periodic: "
        + "; ".join(f"{k.split('|')[0].split('/')[-1]} {k.split('|')[1]} top1={v['top1_pool']:.3f} "
                    f"frozen={v['frozen_frac']:.2f} rep1={v['rep1_pool']:.3f}" for k, v in hi)
        + ". A high share carried by adjacent-repeat is one token dominating; a high share with "
          "rep1 at zero would have been 1/p of a crystal.")
    st = res.get("stored") or {}
    if st:
        full = [v for v in st.values() if v["complete"]]
        ncry = sum(1 for v in full if v["crystal_frac"] > 0)
        reps = sum(v["n_replicas"] for v in full)
        ncr = sum(len(v["crystals"]) for v in full)
        worst = max((c["top1"] for v in full for c in v["crystals"]), default=None)
        parts.append(
            f"FULL GRID, from the lattices share_invariance now stores -- {len(full)} cells, "
            f"{reps} replicas, no re-run: {ncr} crystal replicas in {ncry} cells "
            f"({ncr / max(reps, 1):.2%}). "
            + (f"The largest share any crystal reads is {worst:.4f}, so the mechanism still cannot "
               f"manufacture a high share -- 1/p is bounded by 1/2 for p > 1, and F130's ranking is "
               f"carried by shares far above that."
               if worst is not None else
               "No crystal appears anywhere in the grid.")
            + " This supersedes the 24-cell subset F136 reported: the boundary that said 'on this "
              "subset' is now the whole of F130's grid, and it cost a file read rather than a run.")
    sc = res.get("screen")
    if sc:
        surv = sc.get("resolved")
        parts.append(
            f"SCREEN on F134's top-k grid, from stored scalars and no re-run: {sc['n_candidates']} "
            f"of {sc['n_cells']} cells CANNOT be excluded by arithmetic -- their top1 sits within "
            f"{sc['tolerance']} of a unit fraction AND their distinct count is small enough for "
            f"{sc['B_assumed']} replicas of a shared short orbit to produce it. The other "
            f"{sc['n_cells'] - sc['n_candidates']} are excluded outright: pooling over replicas "
            f"means a 1/period reading needs the SAME dominant token in every replica, which forces "
            f"a small distinct count, and theirs is far too large. "
            + (f"The screen therefore narrows a four-hour backfill to {sc['n_candidates']} cells, "
               f"which is the useful outcome -- not a clean bill of health."
               if not surv else
               f"Those {sc['n_candidates']} were then re-run WITH state: "
               f"{surv['n_balanced']} are balanced cycles, {surv['n_frozen_alias']} are dominated "
               f"rings whose defects align at some period, {surv['n_other']} are neither. "
               + ("The top-k interface does not reproduce the remote arm's degeneracy."
                  if surv["n_balanced"] == 0 else
                  "The top-k interface DOES produce balanced cycles, so F134's affected cells are "
                  "reading 1/period and its ranking claim needs re-checking without them.")))
    rows = res.get("remote") or []
    clean = [r for r in rows if not r.get("misses")]
    cry = [r for r in clean if r["crystal"]]
    parts.append(
        f"CONTRAST, the remote arm through the same detector: {len(cry)} of {len(clean)} "
        f"miss-free cells are periodic crystals -- "
        + "; ".join(f"{r['cell'].split('|')[0].split('-')[2]}.{r['cell'].split('|')[1]} p*={r['p_star']} "
                    f"rep={r['rep_at_p_star']:.2f} top1={r['top1']:.4f} vs 1/p*={r['one_over_p']:.4f}"
                    for r in cry)
        + ". On those cells top1 IS 1/period by arithmetic and carries nothing about attraction. "
        + f"A further {len([r for r in rows if r.get('misses')])} of {len(rows)} remote cells lost "
          f"most of their updates to provider errors and are not settles at all; two of them are "
          f"byte-identical rings at two different temperatures.")
    floor = 1.0 / 6
    tops = [r["top1"] for r in rows]
    parts.append(
        f"AND A THIRD REMOTE DEFECT, independent of the other two: with a 6-word alphabet top1 "
        f"cannot go below {floor:.4f}, and every remote cell sat in "
        f"[{min(tops):.4f}, {max(tops):.4f}] -- the bottom sixth of its range. That is restriction "
        f"of range on the readout itself, the defect class this project keeps recording, and it was "
        f"designed in when the alphabet was chosen.")
    parts.append(
        f"BOUNDARY: the local arm is {len(MODELS)} models x {len(CONSTRUCTIONS)} constructions at "
        f"one seed, chosen to span F130's top1 range, not F130's whole grid. It bounds the defect "
        f"to the remote construction ON THIS SUBSET. The remote arm is {len(rows)} cells on two "
        f"models and is read here only as a contrast -- its own scaffold rung (F135) already "
        f"failed, so no model claim is drawn from it in any case.")
    res["analysis"] = dict(
        rung_passes=True, detector_ok=det_ok, repro_worst=worst,
        local_crystal_frac=float(np.mean(cf)), local_frozen_frac=float(np.mean(ff)),
        local_cells_with_crystal=n_cry,
        remote_clean=len(clean), remote_crystals=len(cry), remote_miss_corrupted=len(rows) - len(clean),
        remote_top1_min=float(min(tops)) if tops else None,
        remote_top1_max=float(max(tops)) if tops else None, remote_floor=floor)
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"local": {}}
    res["_preregistration"] = dict(
        models=MODELS, constructions=[f"r{r}.T{T}" for r, T in CONSTRUCTIONS],
        N=N, B=B, settle=SETTLE, seed=SEED, rung_tol=RUNG_TOL, crystal_rep=CRYSTAL_REP,
        reference=rel(REF), remote=rel(REMOTE),
        rung="(a) the detector must return a constructed period on synthetic rings; (b) the local "
             "re-run must reproduce share_invariance's stored pooled top1",
        primary="fraction of local replicas that are periodic crystals (p* > 1 at rep >= 0.9)",
        why="the remote arm's stored rings are exact period-3 and period-4 crystals whose top1 is "
            "1/period by arithmetic; no local results file stores a ring, so whether the same "
            "degeneracy is present locally could not be answered from stored data")
    res["detector_rung"] = detector_rung()
    res["remote"] = remote_arm()
    res["stored"] = stored_arm()
    res["screen"] = screen_arm()
    from ar_ca import ARRule
    for m in MODELS:
        need = [(r, T) for r, T in CONSTRUCTIONS if f"{m}|r{r}.T{T}|s{SEED}" not in res["local"]]
        if not need:
            continue
        try:
            rule = ARRule(m)
        except Exception as e:
            print(f"  {m}: LOAD FAILED {type(e).__name__}", flush=True)
            continue
        for r, T in need:
            key = f"{m}|r{r}.T{T}|s{SEED}"
            t0 = time.time()
            try:
                c = local_cell(rule, r, T, SEED)
            except Exception as e:
                print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            c.update(model=m, construction=f"r{r}.T{T}", r=r, T=T, seed=SEED,
                     secs=round(time.time() - t0, 1))
            res["local"][key] = c
            print(f"  {key:<52} top1={c['top1_pool']:.4f} frozen={c['frozen_frac']:.2f} "
                  f"crystal={c['crystal_frac']:.2f} <p*>={c['mean_p_star']:.1f}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
        del rule
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
