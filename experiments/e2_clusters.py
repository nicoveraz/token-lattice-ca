"""E2 at cluster level: does F171's reversal survive aggregation to independent units?

Registered in experiments/prereg_e2_clusters.json, frozen and hashed BEFORE any cluster statistic
existed (`5137678c…` pre-amendment, `4b4bbb7e…` post, both dated before this ran). Zero forward
passes -- every input is a stored number in results/inflow_funnel.json.

THE STANDING DECISION IS H0. F171 observed that endpoint tokens sit BELOW their frequency-matched
peers (per-model median 32.0 against 50 by construction) and declined to claim it. PLAN.md: that
"decision stands until a registered cluster analysis says otherwise". The burden is on H1, and this
script is the only thing licensed to move it.

WHY THE CLUSTERING RULE IS THE WHOLE BALLGAME, restated here because a reader of the code should not
have to find it in the prereg. PLAN.md §5.2 assumed models sharing an endpoint token share corpus
statistics exactly. They do not: the three models whose endpoint decodes to '0' split, with two at
corpus count 59956 and OLMo-2 at 8622. OLMo-2 is also the only model above 50. Group by glyph and it
is outvoted inside a cluster whose median is 12.0; separate it on corpus statistics and it stands
alone at 52.0. So the rule decides the verdict, which is precisely why it was registered first.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import collections, json

import numpy as np

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "e2_clusters.json")
PREREG = "experiments/prereg_e2_clusters.json"
SRC = _ROOT / "results" / "inflow_funnel.json"
MIN_CLUSTERS = 4          # K3, verbatim from PLAN.md §6
NULL = 50.0               # the matched control's expected value, by construction


def rows():
    d = json.load(open(SRC))
    return [r for r in d["per_model"]
            if "error" not in r and r.get("endpoint_corpus_count", 0) >= 100]


def cluster(rs, key):
    g = collections.defaultdict(list)
    for r in rs:
        g[key(r)].append(r)
    out = []
    for k, members in g.items():
        vals = [m["freq_matched_pctl"] for m in members]
        out.append(dict(key=str(k), n=len(members),
                        models=sorted(m["model"].split("/")[-1] for m in members),
                        values=sorted(vals),
                        median=round(float(np.median(vals)), 2),
                        below_null=bool(np.median(vals) < NULL)))
    return sorted(out, key=lambda c: c["median"])


def main():
    res = {"_preregistration_file": PREREG,
           "_prereg_sha256": open(_ROOT / "experiments" / "prereg_e2_clusters.sha256").read().strip(),
           "_forward_passes": 0,
           "_standing_decision": "H0 -- E2 reported descriptively, as F171 reports it. The burden is "
                                 "on H1 and only this registered analysis can move it."}
    rs = rows()
    res["n_models"] = len(rs)

    # PRIMARY: identical corpus statistics. Implemented on the stored triple per the logged
    # amendment -- a strictly finer partition than corpus count alone, so it can only split further.
    prim = cluster(rs, lambda r: (r["endpoint_corpus_count"],
                                  r["inflow_pctl_all_vocab"],
                                  r["endpoint_inflow_rank"]))
    # SENSITIVITY: the decoded glyph, i.e. the rule PLAN.md §5.2 assumed.
    sens = cluster(rs, lambda r: r["endpoint"])

    res["primary_clusters"] = prim
    res["sensitivity_clusters"] = sens

    # K7 -- verify the premise rather than assume it. Assuming it is what put the false one in
    # the plan, so the check runs on BOTH rules and reports where the glyph rule fails it.
    def premise_ok(clusters):
        bad = []
        for c in clusters:
            members = [r for r in rs if r["model"].split("/")[-1] in c["models"]]
            counts = {m["endpoint_corpus_count"] for m in members}
            ranks = {m["endpoint_inflow_rank"] for m in members}
            if len(counts) > 1 or len(ranks) > 1:
                # Name WHICH field violated. Printing only the counts made a rank-only violation
                # read as though a single corpus count were itself the defect.
                bad.append(dict(key=c["key"], models=c["models"],
                                violates=[f for f, v in (("corpus_count", counts),
                                                         ("inflow_rank", ranks)) if len(v) > 1],
                                corpus_counts=sorted(counts), inflow_ranks=sorted(ranks)))
        return bad
    res["K7_primary_premise_violations"] = premise_ok(prim)
    res["K7_sensitivity_premise_violations"] = premise_ok(sens)

    def summarise(clusters):
        singles = sum(1 for c in clusters if c["n"] == 1)
        return dict(n_clusters=len(clusters), n_singletons=singles,
                    all_below_null=all(c["below_null"] for c in clusters),
                    medians=[c["median"] for c in clusters],
                    at_or_above_null=[c["key"] for c in clusters if not c["below_null"]])
    sp, ss = summarise(prim), summarise(sens)
    res["primary_summary"], res["sensitivity_summary"] = sp, ss

    parts = [f"ZERO FORWARD PASSES; {len(rs)} readable models from results/inflow_funnel.json, "
             f"aggregated to clusters under two registered rules. "]

    parts.append(
        f"PRIMARY (identical corpus statistics): {sp['n_clusters']} clusters, {sp['n_singletons']} "
        f"of them singletons; cluster medians {sp['medians']}. "
        f"SENSITIVITY (decoded glyph, the rule the plan assumed): {ss['n_clusters']} clusters, "
        f"{ss['n_singletons']} singletons; medians {ss['medians']}. ")

    if res["K7_sensitivity_premise_violations"]:
        v = res["K7_sensitivity_premise_violations"]
        parts.append(
            f"K7: the SENSITIVITY rule violates the non-independence premise on {len(v)} cluster(s) "
            f"-- {[(x['key'], x['violates']) for x in v]} -- confirming, from the data rather "
            f"than by argument, that grouping on the decoded glyph merges models that were scored "
            f"against different corpus statistics. The plan's premise is false as written. ")
    if res["K7_primary_premise_violations"]:
        parts.append(f"K7 FAILURE on the PRIMARY rule: {res['K7_primary_premise_violations']}. The "
                     f"registered rule does not do what it claims and the analysis stops here.")
        res["verdict"] = " ".join(parts)
        res["_analysis_provenance"] = stamp(__file__)
        json.dump(res, open(OUT, "w"), indent=1)
        print(res["verdict"]); return
    parts.append("K7 passes on the PRIMARY rule: every cluster's members share corpus count and "
                 "inflow rank exactly, so each cluster is one test rather than several. ")

    # K3 -- the plan's floor
    if sp["n_clusters"] < MIN_CLUSTERS:
        parts.append(f"K3 FIRES: {sp['n_clusters']} clusters is below the floor of {MIN_CLUSTERS}. "
                     f"E2 is DESCRIPTIVE ONLY -- no rate, no comparison to {NULL:.0f}.")
        res["verdict"] = " ".join(parts)
    else:
        # K6 -- do the two rules agree on the verdict?
        agree = sp["all_below_null"] == ss["all_below_null"]
        res["K6_rules_agree"] = bool(agree)
        if not agree:
            parts.append(
                f"K6 FIRES: the two rules DISAGREE on the criterion -- primary all-below-{NULL:.0f} "
                f"= {sp['all_below_null']}, sensitivity = {ss['all_below_null']}. Neither is "
                f"reported as the answer. The disagreement IS the result: E2's direction is not "
                f"robust to a defensible change in aggregation, and it turns on whether one model "
                f"scored against a different corpus statistic is counted as its own unit. ")
            claim = False
        else:
            claim = sp["all_below_null"]
            parts.append(
                f"K6: the two rules AGREE (all clusters below {NULL:.0f} = {claim} under both). ")

        if claim:
            parts.append(
                f"H1 SUPPORTED: every cluster median falls below {NULL:.0f} under both registered "
                f"rules. The reversal survives aggregation to independent units.")
        else:
            offenders = sorted(set(sp["at_or_above_null"] + ss["at_or_above_null"]))
            parts.append(
                f"H1 NOT SUPPORTED, and H0 STANDS: unanimity fails -- cluster(s) at or above "
                f"{NULL:.0f}: {offenders}. E2 is reported exactly as F171 reports it: the reversal "
                f"is an observation, not a claim. ")

        # K5 -- was the clustering doing any work?
        if sp["n_singletons"] > sp["n_clusters"] / 2:
            parts.append(
                f"K5 FIRES: {sp['n_singletons']} of {sp['n_clusters']} primary clusters are "
                f"singletons, so clustering reduced {len(rs)} models to {sp['n_clusters']} units "
                f"and most of those units are one model. This analysis must NOT be presented as "
                f"having controlled for non-independence -- it is close to the per-model analysis "
                f"under another name, and the cluster count is reported with the singleton count "
                f"beside it for that reason. ")
        res["claim_supported"] = bool(claim)
        res["verdict"] = " ".join(parts)

    parts.append("REFUSALS, recorded before the numbers: no p-value (cluster count is below F149's "
                 "ten-cluster floor); no confidence interval on a median of fewer than ten "
                 "clusters; no rank correlation between cluster size and median; no re-measurement.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
