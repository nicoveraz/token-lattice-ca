"""Read the self-continuation sets against experiments/prereg_selfcont.json. Zero forward passes.

Everything here was registered before any model was loaded: the estimands, the tau ladder with 1.0
named primary, the |I| floor of 500, K2(b)'s constant-bit fraction of 0.90, the mandatory
anti-vacuity report, and both chance levels for the identification test. This script evaluates them
and writes the verdict; it chooses nothing.

THE ONE THING THAT IS NOT REGISTERED, AND IS REPORTED AS SUCH: the full pairwise distance matrix.
Only four comparisons were registered (the decisive pair and three should-be-far controls), and the
matrix is printed so a reader can see the deduped pair against the whole cohort rather than only
against the three it was matched with. It is descriptive and carries no verdict.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import itertools, json, os

import numpy as np

from provenance import stamp, rel
from gatecheck import balance_report

OUT = _ROOT / "results" / "selfcont_verdict.json"
PREREG = "experiments/prereg_selfcont.json"
PR = json.load(open(_ROOT / PREREG))

DECISIVE = ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped")
FAR = [("EleutherAI/pythia-410m", "state-spaces/mamba-370m-hf"),
       ("EleutherAI/pythia-410m", "RWKV/rwkv-4-430m-pile"),
       ("EleutherAI/pythia-410m", "EleutherAI/gpt-neo-125m")]
CONTROL = ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m@bf16")
TAUS = PR["thresholds"]["tau_ladder"]
TAU_P = PR["thresholds"]["tau_primary"]
MIN_I = PR["thresholds"]["min_intersection"]
MAX_CONST = 0.90


def load_cells():
    cells = {}
    for p in sorted((_ROOT / "results").glob("selfcont_set_*.json")):
        if p.name == "selfcont_set_failures.json":
            continue
        d = json.load(open(p))
        cells[d["cell"]] = d
    return cells


def hamming(a, b, idx):
    """Raw count of probe positions where the two bit vectors differ. The registered PRIMARY."""
    return int(np.sum(a["bits"][idx] != b["bits"][idx]))


def robust(a, b, idx, tau):
    """Disagreements that are not near-ties: |margin| >= tau on BOTH sides."""
    d = a["bits"][idx] != b["bits"][idx]
    m = np.minimum(np.abs(a["marg"][idx]), np.abs(b["marg"][idx])) >= tau
    return int(np.sum(d & m))


def main():
    cells = load_cells()
    fail_path = _ROOT / "results" / "selfcont_set_failures.json"
    failed = json.load(open(fail_path))["failed"] if fail_path.exists() else []
    expected = [m if dt == "fp32" else f"{m}@{dt}" for m, _fam, dt in _cohort()]
    missing = [c for c in expected if c not in cells]

    models = [c for c in cells if "@" not in c]          # the 12 distinct models
    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_selfcont.sha256").read().split()[0],
               _probe_strings_sha256=cells[models[0]]["_probe_strings_sha256"],
               cells_measured=sorted(cells), models=sorted(models),
               failed=failed, missing=missing)

    # ---- K4: cohort shape. Every gate below reads it, so it is resolved first. ----
    res["K4_cohort_shrank"] = bool(failed or missing)
    if failed or missing:
        res["K4_note"] = (f"cells unusable and NAMED, not dropped: "
                          f"{[f['cell'] for f in failed] + missing}. K2(b), K3's variable subset and "
                          f"the identification chance levels are computed on the survivors below.")

    # ---- Task 2: the intersection, over the 12 MODELS (the bf16 control shares a tokenizer) ----
    n_strings = len(cells[models[0]]["probe_token_ids"])
    pid = {m: np.array(cells[m]["probe_token_ids"]) for m in cells}
    ok = np.ones(n_strings, bool)
    for m in models:
        ok &= pid[m] >= 0
    idx = np.flatnonzero(ok)
    res["intersection_size"] = int(len(idx))
    res["probe_candidates"] = n_strings
    ctrl_same = bool(np.array_equal(pid[CONTROL[0]], pid[CONTROL[1]])) if CONTROL[1] in cells else None
    res["control_shares_tokenizer"] = ctrl_same
    if CONTROL[1] in cells and not ctrl_same:
        # the control is the same repo at a second dtype, so this cannot happen without something
        # having gone wrong -- and if it did, the control's margins would carry NaN at probe
        # positions the model resolves and the floor would be read off a shorter vector.
        raise AssertionError(
            "the precision control resolves the probe strings differently from the cell it "
            "controls, though both load the same repo id. The floor K1 reads would be computed "
            "over a different set of positions than the decisive distance.")

    # bit and margin vectors, indexed by PROBE STRING so models with different vocabularies align
    for m, d in cells.items():
        marg = np.array(d["margins"], np.float64)
        tid = pid[m]
        v = np.full(n_strings, np.nan)
        good = tid >= 0
        v[good] = marg[tid[good]]
        d["marg"] = v
        d["bits"] = v > 0

    parts = []
    parts.append(
        f"SELF-CONTINUATION SET, registered in {PREREG} (sha256 "
        f"{res['_prereg_sha256'][:12]}..., frozen before any model was loaded) over probe strings "
        f"frozen one commit earlier. {len(models)} models, 4 families, all Pile-trained so CORPUS IS "
        f"HELD FIXED across the whole cohort. Deterministic estimator: no census seeds, no random "
        f"starts, and every cell asserted bit-for-bit reproducible before it was written. ")
    if failed or missing:
        parts.append(f"K4: {res.get('K4_note')} ")
    parts.append(f"INTERSECTION: {len(idx)} of {n_strings} frozen strings encode to exactly one "
                 f"token under all {len(models)} tokenizers. ")

    # ---- K2(a) ----
    if len(idx) < MIN_I:
        res["K2_fires"] = True
        res["verdict"] = " ".join(parts + [
            f"K2 FIRES on the coverage arm: the intersection is {len(idx)} < the registered floor of "
            f"{MIN_I}. NOT DECIDABLE for insufficient signal; the run stops here as registered."])
        _write(res)
        return

    B = np.array([cells[m]["bits"][idx] for m in sorted(models)])
    const = B.all(axis=0) | (~B).all(axis=0)
    var = ~const
    n_var = int(var.sum())
    res["constant_bits"] = int(const.sum())
    res["variable_bits"] = n_var
    res["constant_fraction"] = round(float(const.mean()), 4)
    res["self_cont_in_all"] = int(B.all(axis=0).sum())
    res["self_cont_in_none"] = int((~B).all(axis=0).sum())
    res["per_model_probe_self_cont"] = {m: int(cells[m]["bits"][idx].sum()) for m in sorted(models)}

    # ---- K3, mandatory and stated BEFORE any distance is read ----
    parts.append(
        f"K3 (anti-vacuity, registered as mandatory): of those {len(idx)} probe tokens, "
        f"{res['self_cont_in_all']} self-continue in EVERY model and {res['self_cont_in_none']} in "
        f"NONE, leaving {n_var} VARIABLE. Fractions below are normalised by the variable subset "
        f"({n_var}), never by the intersection; the PRIMARY figure is the raw Hamming count, which "
        f"constant tokens cannot inflate because they contribute exactly zero to it. ")

    # ---- K2(b) ----
    if res["constant_fraction"] > MAX_CONST:
        res["K2_fires"] = True
        res["verdict"] = " ".join(parts + [
            f"K2 FIRES on the signal arm: {res['constant_fraction']:.1%} of probe bits are identical "
            f"across all {len(models)} models, above the registered 0.90. NOT DECIDABLE for "
            f"insufficient signal; the run stops here as registered."])
        _write(res)
        return
    res["K2_fires"] = False

    # ---- the registered comparisons ----
    def pair_row(a, b):
        if a not in cells or b not in cells:
            return None
        h = hamming(cells[a], cells[b], idx)
        return dict(a=a, b=b, hamming=h,
                    fraction_of_variable=round(h / n_var, 4) if n_var else None,
                    robust={str(t): robust(cells[a], cells[b], idx, t) for t in TAUS})

    res["decisive"] = pair_row(*DECISIVE)
    res["should_be_far"] = [r for r in (pair_row(*p) for p in FAR) if r]
    res["control"] = pair_row(*CONTROL)

    D = res["decisive"]["hamming"] if res["decisive"] else None
    C = res["control"]["hamming"] if res["control"] else None

    if D is None:
        # K4 again: without both halves of the decisive pair there is no H1 to decide, and saying so
        # is the whole point of naming a failed cell instead of dropping it.
        parts.append("H1 IS NOT DECIDABLE: one half of the decisive pair is missing from the "
                     "measured cells, so there is no distance to read. ")
    else:
        parts.append(
            f"DECISIVE PAIR (H1) pythia-410m vs pythia-410m-deduped -- the pair phi cannot separate "
            f"(0.458 vs 0.427, both funnel, same modal endpoint): HAMMING = {D} over {len(idx)} "
            f"probe tokens, {res['decisive']['fraction_of_variable']:.1%} of the variable subset; "
            f"robust disagreements at the PRIMARY tau={TAU_P} = "
            f"{res['decisive']['robust'][str(TAU_P)]}. ")
    if res["should_be_far"]:
        parts.append("SHOULD BE FAR: "
                     + "; ".join(f"vs {r['b'].split('/')[-1]} {r['hamming']}"
                                 f" (robust@{TAU_P} {r['robust'][str(TAU_P)]})"
                                 for r in res["should_be_far"]) + ". ")

    # ---- K1 ----
    if C is None or D is None:
        res["K1_fires"] = None
        parts.append("K1 IS NOT EVALUABLE: "
                     + ("the precision control cell is missing, so there is no floor to read the "
                        "decisive distance against. " if C is None else
                        "the decisive pair is incomplete. ")
                     + "H1 is NOT DECIDABLE. ")
    else:
        res["K1_fires"] = bool(D <= C)
        res["resolution_floor"] = C
        res["resolution_ratio"] = (None if C == 0 else round(D / C, 2))
        parts.append(
            f"THE FLOOR: the same weights at two numeric precisions (float32 vs bfloat16) differ on "
            f"{C} probe bits, with {res['control']['robust'][str(TAU_P)]} of those robust at "
            f"tau={TAU_P}. ")
        if res["K1_fires"]:
            parts.append(
                f"K1 FIRES -- H1 IS DEAD. The decisive distance {D} does not exceed the "
                f"precision floor {C}, so the set does not resolve below family and the finding is "
                f"the RESOLUTION FLOOR, not a distance. Registered before the run and reported as "
                f"registered. ")
        elif C == 0:
            parts.append(
                f"K1 does not fire, and the registered corollary applies: the floor is EXACTLY ZERO, "
                f"which makes K1 a WEAK test -- any nonzero distance clears it. What the zero floor "
                f"does establish is that bfloat16 rounding moves no bit at all, so the {D} bits that "
                f"separate the deduped pair are not numeric noise. It says nothing about a real "
                f"quantized variant, which is a far larger perturbation and is registered OWED. ")
        else:
            parts.append(
                f"K1 does not fire: {D} against a floor of {C}, a ratio of "
                f"{res['resolution_ratio']}x. ")

    # ---- descriptive, unregistered ----
    order = sorted(models)
    M = np.zeros((len(order), len(order)), int)
    for i, j in itertools.combinations(range(len(order)), 2):
        M[i, j] = M[j, i] = hamming(cells[order[i]], cells[order[j]], idx)
    res["pairwise_hamming"] = dict(order=order, matrix=M.tolist(),
                                   _status="DESCRIPTIVE, not registered: only the decisive pair and "
                                           "three far controls were registered. Carries no verdict.")

    # ---- Task 4: leave-one-out rank-1 FAMILY attribution ----
    fam = {m: cells[m]["family"] for m in order}
    labels = [fam[m] for m in order]
    bal = balance_report(labels, name="family label")
    n = len(order)
    hits, nn = 0, {}
    for i in range(n):
        d = M[i].astype(float).copy()
        d[i] = np.inf
        j = int(np.argmin(d))
        ties = [k for k in range(n) if k != i and M[i, k] == M[i, j]]
        nn[order[i]] = dict(nearest=order[j], distance=int(M[i, j]),
                            same_family=bool(fam[order[j]] == fam[order[i]]),
                            n_tied_at_that_distance=len(ties))
        hits += int(fam[order[j]] == fam[order[i]])
    chance_family = float(np.mean([(labels.count(fam[m]) - 1) / (n - 1) for m in order]))
    res["identification"] = dict(
        n=n, rank1_same_family=hits, rank1_accuracy=round(hits / n, 4),
        chance_instance_level=round(1 / (n - 1), 4),
        chance_family_level=round(chance_family, 4),
        majority_class_rate=round(bal.majority_rate, 4),
        balance=bal.reason, family_counts=bal.counts, nearest_neighbour=nn,
        _what_it_is_not="FAMILY ATTRIBUTION, not instance identification. Registered before the run: "
                        "a proper instance test needs repeated INDEPENDENT measurements of the same "
                        "weights, determinism makes those bit-identical, and a test that cannot fail "
                        "informatively is refused rather than run and over-read.",
        _confounds="family is confounded with tokenizer (Pythia/RWKV/Mamba use GPT-NeoX "
                   "vocabularies, GPT-Neo uses GPT-2's) and, for the two-member families, with "
                   "nearest-in-size. A rank-1 success cannot separate these.")
    parts.append(
        f"IDENTIFICATION (Task 4): leave-one-out rank-1 nearest neighbour by Hamming over the "
        f"intersection puts {hits} of {n} models next to a model of their OWN family, "
        f"{hits/n:.0%} against a family-level chance of {chance_family:.1%} and a majority-class "
        f"rate of {bal.majority_rate:.0%}. The instance-level 1/(n-1) = {1/(n-1):.1%} is the wrong "
        f"baseline here and is reported only so it cannot be quoted as the right one. This is FAMILY "
        f"ATTRIBUTION; instance identification was refused before the run, because determinism makes "
        f"repeated measurement of one checkpoint bit-identical and the test cannot fail. ")

    # ---- the within-family signature, outside the intersection ----
    outside = {}
    for m in order:
        d = cells[m]
        tid = pid[m]
        inter = set(int(x) for x in tid[idx])
        own = np.array(d["margins"], np.float64) > 0
        mask = np.ones(len(own), bool)
        mask[list(inter)] = False
        outside[m] = dict(vocab_measured=d["vocab_measured"],
                          n_outside=int(mask.sum()),
                          self_continuing_outside=int((own & mask).sum()),
                          self_continuing_total=d["n_self_continuing"])
    res["outside_intersection"] = dict(
        per_model=outside,
        _what_it_is="each model's self-continuation bits over the tokens of its OWN vocabulary that "
                    "are not in the shared probe set. Not comparable across tokenizers; comparable "
                    "within a tokenizer group, which is where the within-family signature lives.")

    # within-tokenizer full-vocabulary distances, where the vocabularies actually match
    groups = {}
    for m in order:
        groups.setdefault((cells[m]["vocab_measured"], cells[m]["vocab_logits"]), []).append(m)
    full = []
    for key, ms in sorted(groups.items()):
        if len(ms) < 2:
            continue
        for a, b in itertools.combinations(sorted(ms), 2):
            va = np.array(cells[a]["margins"], np.float64) > 0
            vb = np.array(cells[b]["margins"], np.float64) > 0
            full.append(dict(a=a, b=b, vocab=key[0], hamming_full_vocab=int((va != vb).sum()),
                             same_family=bool(fam[a] == fam[b])))
    res["full_vocab_within_tokenizer_group"] = dict(
        pairs=full,
        _caveat="grouped by (measured vocab size, logit rows), which is a NECESSARY not sufficient "
                "test of tokenizer identity. Two tokenizers of equal size could still disagree on a "
                "token id; the intersection arm above does not have this weakness and is the one "
                "every verdict uses.")

    res["verdict"] = " ".join(parts + [
        "REFUSALS, registered before the numbers: no p-value (12 checkpoints in 4 families is not a "
        "sample); no instance-identification claim; no adjustment of tau, of the intersection floor, "
        "or of K2(b) after the fact; no claim that the set fingerprints models whose tokenizers do "
        "not share the probe strings; no claim about WHY a bit differs. QUANTIZATION ROBUSTNESS IS "
        "OWED AND NOT RUN: no quantized variant of any cohort member is in the local cache, "
        "downloading one was not authorised, and bfloat16 rounding is a far smaller perturbation "
        "than 4- or 8-bit quantization. THE PRIOR-ART RE-CHECK IS OWED: F95 cleared a battery of "
        "SCALARS, this is a different feature, and no write-up may proceed until that gate runs."])
    _write(res)


def _cohort():
    import selfcont_set
    return selfcont_set.COHORT


def _write(res):
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
