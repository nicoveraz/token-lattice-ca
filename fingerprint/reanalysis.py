"""Fingerprint reanalysis: does the existing screen data already carry model-identity signal?

EXPLORATORY, ZERO NEW COMPUTE. Reads three committed results files
(attractor_corpus_screen.json, degeneration_vs_tstar.json, evidence_falloff.json) and asks the
question critical_analysis.md 3 raised: the corpus-sensitivity result (gpt-neo vs gpt2, F64)
is used only as a control in a negative-result argument — is there enough signal in what is
already measured to justify developing black-box fingerprinting as a capability?

Nothing here is a finding. It is a Gate-0 feasibility read whose numbers feed
fingerprint/PROGRAM.md, where the actual pre-registered program lives. The independent-unit
hazard (six Pythia sizes are one family — the F68 lesson) is accounted, not repeated.

Run from the repo root:  python fingerprint/reanalysis.py [repo_root]
"""
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _HERE.parents[1]
for cand in (ROOT / "gatecheck" / "src", _HERE.parents[1] / "gatecheck" / "src"):
    if cand.exists():
        sys.path.insert(0, str(cand))
        break
from gatecheck import independence_report, save_results  # noqa: E402

RNG = np.random.default_rng(20260801)
N_PERM = 10000
TEMPS = ["0.02", "0.2", "0.436", "0.7"]

FAMILY = {  # model series = the independent unit for identity claims
    "EleutherAI/pythia-14m": "pythia", "EleutherAI/pythia-31m": "pythia",
    "EleutherAI/pythia-70m": "pythia", "EleutherAI/pythia-160m": "pythia",
    "EleutherAI/pythia-410m": "pythia", "EleutherAI/pythia-1b": "pythia",
    "gpt2": "gpt2", "gpt2-medium": "gpt2", "gpt2-large": "gpt2", "gpt2-xl": "gpt2",
    "ibm-granite/granite-3.0-1b-a400m-base": "granite",
    "ibm-granite/granite-3.0-2b-base": "granite",
    "state-spaces/mamba-130m-hf": "mamba", "state-spaces/mamba-370m-hf": "mamba",
    "EleutherAI/gpt-neo-125M": "gpt-neo", "RWKV/rwkv-4-169m-pile": "rwkv",
    "bigscience/bloom-560m": "bloom", "Qwen/Qwen2.5-0.5B": "qwen",
    "allenai/OLMo-1B-hf": "olmo", "Salesforce/codegen-350M-mono": "codegen",
    "bigcode/tiny_starcoder_py": "starcoder", "facebook/opt-350m": "opt",
    "microsoft/phi-1_5": "phi", "HuggingFaceTB/SmolLM2-360M": "smollm",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": "tinyllama",
    "stabilityai/stablelm-2-1_6b": "stablelm",
}
ATTENTION_FREE = {"RWKV/rwkv-4-169m-pile", "state-spaces/mamba-130m-hf",
                  "state-spaces/mamba-370m-hf"}


def load():
    R = ROOT / "results"
    screen = json.load(open(R / "attractor_corpus_screen.json"))
    tstar = json.load(open(R / "degeneration_vs_tstar.json"))
    falloff = json.load(open(R / "evidence_falloff.json"))
    rows = {}
    for m, r in screen["at_lowest_T"].items():
        prof = {}
        for T in TEMPS:
            rec = screen["runs"].get(f"{m}@{T}")
            if rec:
                prof[T] = rec["top1_share"]
        rows[m] = dict(
            family=FAMILY[m], corpus=r["corpus"], attention=m not in ATTENTION_FREE,
            top1_low=r["top1_share"], dominant=r["dominant_token"],
            profile=[prof.get(T) for T in TEMPS],
        )
    # T*: three-state encoding -- none (no attractor) < finite < censored_above (never melts)
    for src, state in ((tstar["melting"], "finite"), (tstar["censored_above"], "censored"),
                      (tstar["no_finite_tstar"], "none")):
        for m, r in src.items():
            if m in rows:
                rows[m]["tstar_state"] = state
                rows[m]["tstar"] = r["t_star"] if state == "finite" else None
                rows[m]["rep_4"] = r["rep_4"]
    for m, r in falloff["analysis"].items():
        if m in rows:
            rows[m]["marginal_top1"] = r["marginal_top1"]
    return rows


def pairwise_ratio(values, labels):
    """mean |diff| within-family over mean |diff| between-family; None if no within pairs."""
    v, lab = np.asarray(values, float), np.asarray(labels)
    win, btw = [], []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            (win if lab[i] == lab[j] else btw).append(abs(v[i] - v[j]))
    if not win or not btw:
        return None
    return float(np.mean(win) / np.mean(btw))


def perm_p(values, labels, observed):
    """Fraction of label permutations with ratio <= observed (small = families cohere)."""
    lab = np.asarray(labels)
    hits = 0
    for _ in range(N_PERM):
        r = pairwise_ratio(values, RNG.permutation(lab))
        if r is not None and r <= observed:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def loo_family_attribution(rows):
    """Leave-one-out nearest-family-centroid on the standardized 4-T top1 profile.

    Scored only on models whose family has >=2 members (a singleton's held-out family has no
    remaining centroid). The candidate set is ALL families each time -- 16-way, not 4-way.
    """
    names = [m for m in rows if all(p is not None for p in rows[m]["profile"])]
    X = np.array([rows[m]["profile"] for m in names])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    fam = np.array([rows[m]["family"] for m in names])
    multi = [i for i, m in enumerate(names) if (fam == fam[i]).sum() >= 2]
    correct, details = 0, []
    for i in multi:
        mask = np.ones(len(names), bool)
        mask[i] = False
        cents = {f: X[mask & (fam == f)].mean(0) for f in set(fam[mask])}
        pred = min(cents, key=lambda f: float(np.linalg.norm(X[i] - cents[f])))
        correct += pred == fam[i]
        details.append({"model": names[i], "true": fam[i], "pred": pred, "ok": bool(pred == fam[i])})
    n_fam = len(set(fam))
    return {"n_scored": len(multi), "n_correct": int(correct), "n_candidate_families": n_fam,
            "chance_expected": round(len(multi) / n_fam, 2), "per_model": details}


def main():
    rows = load()
    out = {"n_models": len(rows), "n_families": len(set(r["family"] for r in rows.values()))}

    # -- 1. does the signature cohere within families? (permutation, exploratory) ---------
    fams = [r["family"] for r in rows.values()]
    coher = {}
    for feat in ("top1_low", "rep_4"):
        vals = [r[feat] for r in rows.values()]
        obs = pairwise_ratio(vals, fams)
        coher[feat] = {"within_over_between": round(obs, 3), "perm_p": perm_p(vals, fams, obs)}
    fin = [(r["tstar"], r["family"]) for r in rows.values() if r.get("tstar") is not None]
    obs_t = pairwise_ratio([v for v, _ in fin], [f for _, f in fin])
    coher["tstar_finite_only"] = {
        "within_over_between": round(obs_t, 3) if obs_t else None,
        "perm_p": perm_p([v for v, _ in fin], [f for _, f in fin], obs_t) if obs_t else None,
        "n": len(fin),
        "caveat": "only pythia(6)+granite(2) contribute within-pairs; thin",
    }
    out["family_coherence"] = coher

    # -- 2. leave-one-out family attribution ----------------------------------------------
    out["family_attribution_loo"] = loo_family_attribution(rows)

    # -- 3. the controlled pair, quantified against within-family spread ------------------
    neo, g2 = rows["EleutherAI/gpt-neo-125M"], rows["gpt2"]
    within_ranges = {}
    for f in ("pythia", "gpt2", "granite", "mamba"):
        vs = [r["top1_low"] for r in rows.values() if r["family"] == f]
        within_ranges[f] = round(max(vs) - min(vs), 3)
    gap = neo["top1_low"] - g2["top1_low"]
    out["controlled_pair"] = {
        "gap_top1_low": round(gap, 3),
        "within_family_ranges": within_ranges,
        "gap_over_worst_within_range": round(gap / max(within_ranges.values()), 2),
        "tstar_states": {"gpt-neo": neo["tstar_state"], "gpt2": g2["tstar_state"]},
        "rep4_gap_for_contrast": round(abs(neo["rep_4"] - g2["rep_4"]), 3),
        "note": ("same tokenizer, different corpus: the CA signature separates by "
                 "gap/worst-within-range x, T* lands at OPPOSITE censoring ends "
                 "(never-melts vs never-exists), while greedy degeneration barely moves"),
    }

    # -- 4. corpus inference: one direction only, family as the unit ----------------------
    att = {m: r for m, r in rows.items() if r["attention"]}
    pile_att = [r["top1_low"] for r in att.values() if r["corpus"] == "Pile"]
    nonpile_att = [r["top1_low"] for r in att.values() if r["corpus"] != "Pile"]
    out["corpus_direction"] = {
        "pile_attention_min_top1": round(min(pile_att), 3),
        "pile_attention_models": len(pile_att),
        "pile_attention_families": len({r["family"] for r in att.values()
                                        if r["corpus"] == "Pile"}),
        "nonpile_attention_range": [round(min(nonpile_att), 3), round(max(nonpile_att), 3)],
        "reading": ("every Pile+attention model has the attractor; non-Pile spans the full "
                    "range -> Pile-like corpus is INFERABLE one-way (attractor absent => "
                    "not-Pile-like, given attention), never two-way"),
    }
    icc = independence_report(
        [r["top1_low"] for r in att.values() if r["corpus"] == "Pile"],
        [r["family"] for r in att.values() if r["corpus"] == "Pile"],
        unit_name="family")
    out["pile_claim_units"] = {
        "operative_n": "2 independent families (pythia, gpt-neo) — the F68 hazard applies "
                       "to this direction exactly as it did to T*; ~21 families would settle it",
        "feature_level_icc": icc.message(),
    }

    # -- 5. deflationary check available today: the one-token marginal --------------------
    both = [(r["marginal_top1"], r["top1_low"]) for r in rows.values()
            if r.get("marginal_top1") is not None]
    a = np.array(both)
    ranks = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(ranks(a[:, 0]), ranks(a[:, 1]))[0, 1])
    out["marginal_deflation"] = {
        "n": len(both), "spearman_marginal_vs_ca_top1": round(rho, 3),
        "reading": ("weak: the CA signature is not a re-read of the unconditional prior "
                    "(consistent with evidence_falloff's own verdict). The REAL deflationary "
                    "baseline -- direct two-token-conditional statistics -- is untested and "
                    "is Gate 1 of PROGRAM.md"),
    }

    out["gate0_verdict"] = (
        "COHERENCE YES, IDENTIFICATION NOT YET: the CA-derived signature coheres within "
        "families (top1 within/between 0.218, perm p~3e-4; T* 0.335, p~0.015 on a thin "
        "subset) while a generic degeneration metric does not (rep_4 0.745, p~0.13), and "
        "the controlled corpus pair separates at 2.4x the worst within-family range with "
        "T* at opposite censoring ends. But 16-way family attribution from the 4-number "
        "T-profile alone is weak (4/14 vs ~0.9 chance): the profile is effectively "
        "low-dimensional. Reading: enough signal to justify Gate 1; the feature battery "
        "must widen (dominant token, radius/BOS sensitivity, conditional stats) before "
        "any attribution claim. Exploratory throughout; nothing here is a finding."
    )
    doc = save_results(_HERE.parent / "reanalysis.json", out, script=__file__, root=ROOT,
                       independent_unit="family", forbid_paths=True)
    print(json.dumps({k: v for k, v in doc.items() if not k.startswith("_")}, indent=1))


if __name__ == "__main__":
    main()
