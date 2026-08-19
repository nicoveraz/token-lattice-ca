"""Is the rank-1 shortfall STRUCTURE? Partition arms by selected token. Zero forward passes.

F165 decomposed the vectors: u_prefix is WHICH token a prefix pulls trajectories toward, v_model is
whether THAT token self-continues. If so, dphi is not rank-1 in general -- a '\\n'-selecting prefix
and a '0'-selecting prefix load on DIFFERENT model properties, and pooling them forces one v to serve
both. The pooled fit landed at 0.790 with leave-one-column-out stability 1.000: stable, and short of
its 0.80 bar. This asks whether the shortfall is structure rather than noise.

THE CONFOUND THIS RUN EXISTS TO CONTROL. Smaller matrices fit better, mechanically, for reasons with
nothing to do with any mechanism. A rank-1 fit to four columns will beat one to twenty-nine. So the
token partition is scored against a PERMUTATION CONTROL frozen before any fit: the same partition
SIZES with arms assigned at random, refit. If random partitions of identical shape do as well, H1 is
dead however good the absolute numbers look.

Registered in experiments/prereg_token_partition.json as TIER 2 -- the pooled 0.790 was already seen,
so this prediction is informed by it; no partitioned fit has been seen by anyone.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import collections, json

import numpy as np

from provenance import stamp, rel
from bilinear_rank1 import als_rank1, frac_explained, load, SPECS, BASE_RAW, CENSUS_SEEDS

OUT = str(_ROOT / "results" / "token_partition_rank.json")
PREREG = "experiments/prereg_token_partition.json"
MIN_CELLS, MIN_MODELS, N_PERM = 12, 3, 200
POOLED_RANK1 = 0.7904        # from results/bilinear_rank1.json, already seen -- the comparison point


def arm_histograms():
    """(model, arm) -> Counter of endpoint token ids, summed over census seeds."""
    files = {s[0]: load(s[0]) for s in SPECS}
    out = {}
    for fname, finding, tmpl, arms, _rawt in SPECS:
        runs = files[fname]
        for a in arms:
            col = f"{finding}:{a}"
            for k, v in runs.items():
                p = k.split("|")
                if len(p) != 3 or p[2] != a or not isinstance(v, dict):
                    continue
                if "endpoint_histogram" not in v:
                    continue
                c = out.setdefault((p[0], col), collections.Counter())
                for row in v["endpoint_histogram"]:
                    c[(int(row[0]), row[1])] += int(row[2])
    return out


def main():
    res = {"_preregistration_file": PREREG,
           "_tier": "TIER 2 -- the pooled 0.790 was already seen; no partitioned fit has been",
           "pooled_rank1_comparison_point": POOLED_RANK1}

    bl = json.load(open(_ROOT / "results" / "bilinear_rank1.json"))
    cells = {tuple(k.split("||")): v for k, v in bl["cells"].items()}
    models = sorted({m for m, _c in cells})
    cols = sorted({c for _m, c in cells})
    hists = arm_histograms()

    # ARM-LEVEL partition: the token modal ACROSS models for that arm. Cell-level would let the
    # outcome pick the grouping, which is circular.
    arm_token, mixed = {}, []
    for c in cols:
        tot = collections.Counter()
        carriers = 0
        for m in models:
            h = hists.get((m, c))
            if not h:
                continue
            carriers += 1
            tid, dec = h.most_common(1)[0][0]
            tot[(tid, dec)] += 1
        if not carriers:
            continue
        (tok, share) = tot.most_common(1)[0]
        if share < carriers / 2:
            arm_token[c] = ("MIXED", "MIXED")
            mixed.append(c)
        else:
            arm_token[c] = tok
    res["arm_token"] = {c: {"token_id": t[0], "decoded": t[1]} for c, t in arm_token.items()}
    res["mixed_arms"] = mixed

    groups = collections.defaultdict(list)
    for c, t in arm_token.items():
        groups[t[1]].append(c)
    res["partitions"] = {k: sorted(v) for k, v in groups.items()}

    def build(colset):
        X = np.zeros((len(models), len(colset)))
        M = np.zeros_like(X, dtype=bool)
        for i, m in enumerate(models):
            for j, c in enumerate(colset):
                v = cells.get((m, c))
                if v is None:
                    continue
                X[i, j] = v["dphi"]
                M[i, j] = not v["masked"]
        return X, M

    def fit(colset):
        X, M = build(colset)
        rows = [i for i in range(X.shape[0]) if M[i].sum() >= 2]
        keep = [j for j in range(X.shape[1]) if M[:, j].sum() >= 2]
        if len(rows) < MIN_MODELS or len(keep) < 2:
            return None
        Xs, Ms = X[np.ix_(rows, keep)], M[np.ix_(rows, keep)]
        if Ms.sum() < MIN_CELLS:
            return None
        u, v = als_rank1(Xs, Ms)
        return dict(frac=float(frac_explained(Xs, Ms, u, v)), n_cells=int(Ms.sum()),
                    n_models=len(rows), n_cols=len(keep),
                    u={models[rows[i]].split("/")[-1]: round(float(u[i]), 4)
                       for i in range(len(rows))})

    readable, skipped = {}, {}
    for tok, cs in groups.items():
        f = fit(cs)
        (readable if f else skipped)[tok] = f or f"below floors ({len(cs)} arms)"
    res["partition_fits"] = readable
    res["partition_skipped"] = skipped

    parts = [f"PARTITION: {len(cols)} arms assigned a token modal across the models carrying them; "
             f"{len(groups)} groups, {len(mixed)} arms MIXED. Groups: "
             + "; ".join(f"{repr(k)} x{len(v)}" for k, v in sorted(groups.items(),
                                                                  key=lambda kv: -len(kv[1]))) + ". "]

    if len(readable) < 2:
        parts.append(
            f"NOT DECIDABLE FOR INSUFFICIENCY (K2): {len(readable)} partition(s) clear the floors of "
            f"{MIN_CELLS} cells and {MIN_MODELS} models. Nothing to compare.")
        res["verdict"] = " ".join(parts)
    else:
        beat = [t for t, f in readable.items() if f["frac"] > POOLED_RANK1]
        parts.append(
            "WITHIN-PARTITION rank-1: "
            + "; ".join(f"{repr(t)} {f['frac']:.3f} ({f['n_cells']} cells, {f['n_models']}x"
                        f"{f['n_cols']})" for t, f in sorted(readable.items(),
                                                             key=lambda kv: -kv[1]["frac"]))
            + f". Beats the pooled {POOLED_RANK1:.3f} on {len(beat)} of {len(readable)}. ")

        # PERMUTATION CONTROL, frozen before any fit: same shapes, random arms.
        rng = np.random.default_rng(20260819)
        sizes = [len(groups[t]) for t in readable]
        null = []
        for _ in range(N_PERM):
            perm = list(rng.permutation(cols))
            fr, i = [], 0
            for sz in sizes:
                f = fit(perm[i:i + sz])
                i += sz
                if f:
                    fr.append(f["frac"])
            if fr:
                null.append(float(np.mean(fr)))
        obs = float(np.mean([f["frac"] for f in readable.values()]))
        pct = float(np.mean([n >= obs for n in null])) if null else float("nan")
        res["permutation_control"] = dict(
            observed_mean=round(obs, 4), n_perm=len(null),
            null_mean=round(float(np.mean(null)), 4) if null else None,
            null_p95=round(float(np.percentile(null, 95)), 4) if null else None,
            frac_null_at_or_above_observed=round(pct, 4) if null == null else None)
        parts.append(
            f"PERMUTATION CONTROL ({len(null)} random partitions of the SAME shapes): observed mean "
            f"{obs:.3f}, null mean {np.mean(null):.3f}, null 95th pct "
            f"{np.percentile(null, 95):.3f}; {pct:.0%} of random partitions do at least as well. "
            + ("The token partition is NOT distinguishable from a random partition of the same "
               "shape, so H1 is DEAD: the within-partition improvement is the small-matrix effect "
               "the control was written to catch, not the mechanism."
               if pct > 0.05 else
               "Random partitions of the same shape do NOT reach it, so the improvement tracks the "
               "SELECTED TOKEN rather than matrix size -- which is what H1 predicted."))
    parts.append(
        f"NO THRESHOLD GAMING (K4): F164's 0.80 bar was set for the POOLED rank-1 fit and is not "
        f"re-applied here. Nothing above is reported as clearing it, and the pooled verdict remains "
        f"NOT DECIDABLE at {POOLED_RANK1:.3f}.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
