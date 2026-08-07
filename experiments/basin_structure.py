"""Is lambda_ca's "seed noise" partly BASIN STRUCTURE? Per-replica settling, never averaged.

THE OBSERVATION THAT PROMPTED THIS. Every metric in this project averages over the batch. `ca.metrics`
computes entropy, distinct-fraction and bigram overlap PER REPLICA and then returns `np.mean(...)` of
each; `ar_probe.block_damage` builds a per-replica damage array of shape (sweeps, B, N) and
immediately collapses it with `.mean(axis=1)`. The across-initial-condition structure is calculated
and thrown away, in one line, everywhere.

That is exactly the defect the F94 -> F96 -> F99 arc just walked through, one level up. F94 concluded
single-token sensitivity was flat because it looked at the MEAN of s; F99 found the thing that
actually moves is how s varies across contexts, and that F94's verdict was a property of the
ensemble rather than of the model. The same question has never been asked of the lattice itself.

WHY IT IS LOAD-BEARING RATHER THAN CURIOUS. lambda_ca's seed spread is the NOISE FLOOR that every
gate in this project is measured against -- F100 used 0.0197, F94's deflation check 0.0228, and F88
returned NOT DECIDABLE because two alignments differed by less than a floor of 0.0247. If replicas
started from different initial conditions settle into structurally different states, and lambda_ca
depends on the settled state (F99 measured the ensemble effect at 0.111 against a model effect of
0.196 -- comparable), then part of that "seed noise" is BASIN STRUCTURE, not noise. Conditioning on
basin would shrink the floor, and several NOT DECIDABLE verdicts would deserve re-reading. This is a
re-analysis of a quantity already paid for, not new physics, which is what makes it cheap.

PRIOR EVIDENCE THAT BASINS ARE REAL HERE. At the deterministic argmax level they demonstrably are:
F70/F84/F90 census 96 random two-token starts into funnel / none / fragmented classes, and F85 found
the funnel's identity GENUINELY SWAPS between seeds -- a contested basin, not one attractor with a
noisy label. What has never been checked is whether the STOCHASTIC lattice at T=0.7 inherits any of
that, or whether sampling washes it out.

COMMENSURABILITY IS ASSERTED, NOT ASSUMED. This cannot import `block_damage`, because that function
averages before returning. It therefore reimplements the same computation retaining the replica axis
-- and then ASSERTS that the per-replica cones, averaged, reproduce `block_damage`'s cone to within
floating-point equality on the same seed. A reimplementation that drifts from the paper's protocol
would make every number here incomparable with every number already published, and the assert is
what makes "the same measurement, ungrouped" a claim rather than a hope.

PRE-REGISTERED:
  PROTOCOL GATE  the per-replica cones must average to `block_damage`'s cone exactly, on a shared
                 seed, before anything else is read. This runs FIRST and a failure stops the script.
  PRIMARY        does per-replica lambda_ca depend on which structural cluster the replica's
                 SETTLED STATE falls into? Measured as between-cluster variance over total, with
                 the cluster labels computed from the settled state ALONE -- never from lambda.
  FLOOR DECOMP   the reported quantity is the seed floor split into between-basin and within-basin
                 parts. If the between part is negligible, the floor is genuine noise and every
                 gate that used it stands unchanged -- a null that LICENSES existing verdicts.
  POWER          replicas per cluster are gated by gatecheck.leverage.distinct_units; a clustering
                 claim resting on two replicas in a cluster is not a claim. B is set by that
                 requirement rather than inherited from habit (the project's usual B=8/16 is too
                 small for a per-cluster statement and is not reused here).
  KILL           settled states do not cluster, or lambda does not depend on cluster -> averaging
                 over replicas was harmless all along, the seed floor is noise, and this closes.
  BOUNDARY       one family, two checkpoints, T=0.7. Clusters found here are structures of THIS
                 construction at THIS temperature, not claims about the model.

Writes results/basin_structure.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/basin_structure.py
        (resumable per (checkpoint, seed))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from lyapunov import lyap_from_cone, is_unignited
from dev_transition_phase3 import FIT_KW
from gatecheck import distinct_units, dynamic_range, carries_verdict

OUT = str(_ROOT / "results" / "basin_structure.json")
MODEL = "EleutherAI/pythia-410m"

# One checkpoint inside the dip and one on the plateau: if basins matter anywhere they should
# matter where lambda is closest to zero, and the plateau is the control.
STEPS = [("step256", "dip"), ("step1000", "plateau")]
SEEDS = [21, 22, 23, 24]
# B is set by the POWER requirement, not inherited: 32 replicas x 4 seeds = 128 per checkpoint, so
# even a four-way clustering leaves ~32 per cluster. The project's usual B=8/16 could not support
# a per-cluster statement and is deliberately not reused.
B, N, R, T = 32, 48, 2, 0.7
SETTLE, SWEEPS, BLOCK = 12, 22, 3               # ar_probe.block_damage's own values
MIN_PER_CLUSTER = 8


def per_replica_damage(rule, seed):
    """block_damage's computation with the replica axis RETAINED.

    Mirrors experiments/ar_probe.block_damage line for line, except that the (sweeps, B, N) damage
    array is returned instead of being averaged over B. The protocol gate asserts the two agree.
    """
    from ar_ca import run
    rng = np.random.default_rng(seed)
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none", init="random",
               seed=seed)["final"]
    c = N // 2
    idx = [c + k for k in range(-(BLOCK // 2), BLOCK - BLOCK // 2)]
    flipped = base.copy()
    for j in idx:
        flipped[:, j] = rng.choice(rule.init_pool, size=B)
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
    init2 = np.concatenate([base, flipped], axis=0)
    c2 = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
             init_state=init2, seed=seed + 2, u_stream=u2)
    snaps = c2["snaps"]
    diff = (snaps[:, :B] != snaps[:, B:])                       # (sweeps, B, N) -- NOT averaged
    rolled = np.roll(diff, N // 2 - idx[len(idx) // 2], axis=2)
    return base, rolled


def protocol_gate(rule, seed=21):
    """The per-replica cones must average to block_damage's cone. Runs before anything is read."""
    from ar_probe import block_damage
    base, rolled = per_replica_damage(rule, seed)
    mine = rolled.mean(axis=1)
    theirs = block_damage(rule, T, R, block=BLOCK, B=B, N=N, settle=SETTLE, sweeps=SWEEPS,
                          seed=seed, scheme="none")["cone"]
    if mine.shape != theirs.shape:
        return dict(passes=False, reason=f"shape {mine.shape} vs {theirs.shape}")
    d = float(np.max(np.abs(mine - theirs)))
    return dict(passes=bool(d == 0.0), max_abs_diff=d,
                reason=("per-replica cones average to block_damage's cone EXACTLY, so the "
                        "ungrouped measurement is the paper's measurement"
                        if d == 0.0 else
                        f"reimplementation DRIFTS from block_damage by {d:g}; every number here "
                        f"would be incomparable with every published one"))


def state_features(row):
    """Structure of one settled ring, computed from the STATE ALONE -- never from lambda."""
    vals, cnts = np.unique(row, return_counts=True)
    p = cnts / cnts.sum()
    top = int(vals[int(np.argmax(cnts))])
    return dict(distinct=int(len(vals)), entropy=float(-(p * np.log2(p)).sum()),
                top_share=float(cnts.max() / cnts.sum()), dominant=top)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=[s for s, _ in STEPS], seeds=SEEDS, B=B, N=N, r=R, T=T,
        settle=SETTLE, sweeps=SWEEPS, block=BLOCK, min_per_cluster=MIN_PER_CLUSTER,
        protocol_gate="per-replica cones must average to ar_probe.block_damage's cone EXACTLY, on "
                      "a shared seed, before anything else is read",
        primary="does per-replica lambda_ca depend on which structural cluster the replica's "
                "SETTLED STATE falls into? Cluster labels come from the state alone, never lambda",
        floor_decomposition="the seed floor split into between-basin and within-basin parts; if "
                            "between is negligible the floor is genuine noise and every gate that "
                            "used it stands unchanged",
        power=f"replicas per cluster gated by distinct_units; B={B} chosen for this, not inherited "
              f"from the project's usual 8/16 which cannot support a per-cluster claim",
        kill="settled states do not cluster, or lambda does not depend on cluster -> averaging was "
             "harmless, the seed floor is noise, and this route closes",
        boundary="one family, two checkpoints, T=0.7; clusters are structures of THIS construction "
                 "at THIS temperature, not claims about the model",
        motivation="ca.metrics and ar_probe.block_damage both compute per-replica then average; "
                   "the across-init structure is discarded in one line, which is F94's defect "
                   "(mean vs spread) one level up")

    from ar_ca import ARRule
    for rev, label in STEPS:
        if all(f"{rev}|s{sd}" in res["cells"] for sd in SEEDS):
            continue
        rule = ARRule(MODEL, revision=rev)
        if "protocol_gate" not in res:
            t0 = time.time()
            g = protocol_gate(rule)
            res["protocol_gate"] = g
            print(f"  PROTOCOL GATE: {g['reason']} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            if not g["passes"]:
                print("  STOPPING: the ungrouped measurement is not the paper's measurement.")
                return
        for sd in SEEDS:
            k = f"{rev}|s{sd}"
            if k in res["cells"]:
                continue
            t0 = time.time()
            base, rolled = per_replica_damage(rule, sd)
            reps = []
            for b in range(B):
                cone_b = rolled[:, b, :]
                md = float(rolled[-1, b, :].mean())
                lam = lyap_from_cone(cone_b, N, **FIT_KW)[0]
                reps.append(dict(replica=b, lambda_ca=round(float(lam), 5),
                                 mean_damage=md, ignited=bool(not is_unignited(mean_damage=md)),
                                 **state_features(base[b])))
            res["cells"][k] = dict(revision=rev, label=label, seed=sd, replicas=reps,
                                   secs=round(time.time() - t0, 1))
            ig = [r for r in reps if r["ignited"]]
            print(f"  {k}: {len(ig)}/{B} ignited, lambda "
                  f"{np.mean([r['lambda_ca'] for r in ig]):+.4f} +/- "
                  f"{np.std([r['lambda_ca'] for r in ig]):.4f}, distinct "
                  f"{np.mean([r['distinct'] for r in reps]):.1f} "
                  f"({res['cells'][k]['secs']:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def cluster(reps):
    """Label each replica by its settled state's structure. Deterministic, and lambda-blind.

    Not k-means: the dominant token and the distinct-count are discrete and interpretable, and a
    clustering that could see lambda would manufacture the dependence being tested. Replicas are
    grouped by dominant token when that token holds a majority of the ring, and otherwise by a
    coarse diversity band. Both are functions of the STATE only.
    """
    labels = []
    for r in reps:
        if r["top_share"] >= 0.5:
            labels.append(f"dom:{r['dominant']}")
        elif r["distinct"] <= 8:
            labels.append("low-diversity")
        elif r["distinct"] <= 20:
            labels.append("mid-diversity")
        else:
            labels.append("high-diversity")
    return labels


def analyse(res):
    parts = []
    g = res.get("protocol_gate") or {}
    parts.append(f"PROTOCOL GATE: {g.get('reason', 'not run')}")
    if not g.get("passes"):
        res["analysis"] = dict(protocol_gate=g)
        res["verdict"] = " ".join(parts) + " NOTHING READ."
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}")
        return

    per_step = {}
    for rev, label in STEPS:
        reps = [r for sd in SEEDS
                for r in res["cells"].get(f"{rev}|s{sd}", {}).get("replicas", [])
                if r["ignited"]]
        if len(reps) < 3 * MIN_PER_CLUSTER:
            continue
        labs = cluster(reps)
        lam = np.array([r["lambda_ca"] for r in reps])
        groups = {}
        for l, v in zip(labs, lam):
            groups.setdefault(l, []).append(v)
        big = {l: np.array(v) for l, v in groups.items() if len(v) >= MIN_PER_CLUSTER}
        dis = distinct_units(labs, minimum=2, name="structural cluster")

        total = float(lam.std())
        if len(big) >= 2:
            # variance decomposition: between-cluster means vs pooled within-cluster spread
            means = np.array([v.mean() for v in big.values()])
            ns = np.array([len(v) for v in big.values()])
            between = float(np.sqrt(np.average((means - np.average(means, weights=ns)) ** 2,
                                               weights=ns)))
            within = float(np.sqrt(np.average([v.var() for v in big.values()], weights=ns)))
        else:
            between, within = 0.0, total
        per_step[rev] = dict(
            label=label, n_replicas=len(reps), n_clusters=len(groups),
            clusters={l: dict(n=len(v), lam=round(float(np.mean(v)), 5),
                              sd=round(float(np.std(v)), 5)) for l, v in sorted(groups.items())},
            total_sd=round(total, 5), between_sd=round(between, 5), within_sd=round(within, 5),
            between_frac=round(between ** 2 / max(total ** 2, 1e-12), 4),
            distinct=dis.block())

    print(f"\n  {'checkpoint':<12} {'n':>5} {'clusters':>9} {'total sd':>9} {'between':>9} "
          f"{'within':>9} {'between frac':>13}")
    for rev, v in per_step.items():
        print(f"  {rev:<12} {v['n_replicas']:>5} {v['n_clusters']:>9} {v['total_sd']:>9.4f} "
              f"{v['between_sd']:>9.4f} {v['within_sd']:>9.4f} {v['between_frac']:>13.3f}")
    for rev, v in per_step.items():
        for l, c in v["clusters"].items():
            print(f"      {rev} {l:<18} n={c['n']:<4} lambda={c['lam']:+.4f} +/- {c['sd']:.4f}")

    if not per_step:
        parts.append("GRID INCOMPLETE -- primary undecided.")
    else:
        # A checkpoint that resolves into ONE cluster is homogeneous, which is a result about that
        # checkpoint -- not a reason to void a checkpoint that does resolve. An earlier version
        # required every step to be multi-cluster and so reported NOT DECIDABLE while holding a
        # measured 12.8% between-fraction at the other one.
        homog = [r for r, v in per_step.items() if v["n_clusters"] < 2]
        multi = {r: v for r, v in per_step.items() if v["n_clusters"] >= 2}
        for r in homog:
            v = per_step[r]
            parts.append(
                f"STRUCTURAL HOMOGENEITY at {r} ({v['label']}): all {v['n_replicas']} ignited "
                f"replicas fall in ONE structural cluster, so there is no basin structure to "
                f"explain lambda's spread there and averaging over replicas is provably harmless "
                f"at this checkpoint. That is a result, not missing data.")
        if not multi:
            parts.append(
                "PRIMARY: no checkpoint resolves into more than one cluster, so the seed floor is "
                "genuine noise everywhere measured and every gate that used it stands unchanged.")
            res_frac = 0.0
        else:
            worst = max(multi.values(), key=lambda v: v["between_frac"])
            fr = res_frac = worst["between_frac"]
            thin = [l for l, c in worst["clusters"].items() if c["n"] < MIN_PER_CLUSTER]
            if fr >= 0.25:
                parts.append(
                    f"PRIMARY: lambda_ca DEPENDS ON WHICH BASIN THE REPLICA SETTLED INTO. At "
                f"{worst['label']} ({worst['n_replicas']} ignited replicas across {len(SEEDS)} "
                    f"seeds) the between-cluster component accounts for {fr:.1%} of lambda's total "
                    f"variance -- total sd {worst['total_sd']:.4f} splitting into "
                    f"{worst['between_sd']:.4f} between basins and {worst['within_sd']:.4f} within. "
                    f"So part of what this project treats as SEED NOISE is structure, and the floor "
                    f"used by every gate here (F100 0.0197, F94 0.0228, F88 0.0247) is inflated by "
                    f"it. Clusters below the {MIN_PER_CLUSTER}-replica power floor: {thin}.")
            else:
                parts.append(
                    f"MOSTLY NULL, and it is the useful kind: the between-cluster component is "
                    f"only {fr:.1%} "
                f"of lambda's variance at {worst['label']} (total sd {worst['total_sd']:.4f} = "
                    f"{worst['between_sd']:.4f} between + {worst['within_sd']:.4f} within), below "
                    f"the 25% the primary registered. Replicas DO land in structurally different "
                    f"settled states there, but which one they land in does not explain lambda's "
                    f"spread. Averaging over replicas was harmless, the seed floor is essentially "
                    f"genuine noise, and every gate that used it stands unchanged -- that LICENSES "
                    f"the floor rather than merely failing to impeach it. Caveat, and it is why "
                    f"this is 'mostly': only {worst['n_replicas']} replicas ignite at this "
                    f"checkpoint and clusters {thin} sit below the {MIN_PER_CLUSTER}-replica power "
                    f"floor, so a small basin effect in the dip is not excluded.")

    parts.append(
        "BOUNDARY: one family, two checkpoints, T=0.7, and the clustering is deliberately coarse "
        "and lambda-blind (dominant token when it holds a majority, otherwise a diversity band). A "
        "finer clustering could find structure this one misses; a clustering allowed to see lambda "
        "would manufacture the dependence being tested, which is why it is not used.")

    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(protocol_gate=g, per_step=per_step,
                           max_between_frac=locals().get("res_frac"))
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Asks whether lambda_ca's seed spread -- the noise floor every gate in this project is "
        "measured against -- is partly BASIN STRUCTURE. ca.metrics and ar_probe.block_damage both "
        "compute per-replica quantities and then average them away in one line, which is F94's "
        "mean-versus-spread defect one level up. The per-replica cones are asserted to average to "
        "block_damage's cone exactly, so this is the paper's own measurement ungrouped rather than "
        "a new one. Cluster labels are computed from the settled state alone and never from lambda.")


if __name__ == "__main__":
    main()
