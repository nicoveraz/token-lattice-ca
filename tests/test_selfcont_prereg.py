"""The self-continuation run's freeze integrity and its base rates.

THREE THINGS THIS CATCHES THAT READING WOULD NOT.

1. A FREEZE THAT DRIFTED. The probe strings and the prereg are hashed, and every measured cell
   records both hashes. If a string is added or a threshold edited after a cell was written, the
   cell's stored hash no longer matches the file and the whole design silently becomes
   post-hoc. Nothing about the JSON looks wrong when that happens.

2. THE BASE-RATE ERROR THIS PROJECT KEEPS MAKING. F163 joined a 1-vs-6 predictor to a two-class
   outcome and read 71% as a pass, when a rule ignoring the predictor scores 86%. Here the
   equivalent slip is quoting 1/(n-1) as the chance level for a FAMILY attribution test, which
   flatters the result by 2.5x on this cohort. The chance levels are therefore recomputed from the
   family counts and compared to what the verdict stored.

3. AN ESTIMATOR THAT STOPPED BEING DETERMINISTIC. The entire design rests on there being no seed:
   no census, no random starts, nothing to average over. Each cell asserts bit-for-bit
   reproducibility at write time; this asserts the flag survived into the file, and separately that
   the stored set and the stored margins still agree on which tokens self-continue.
"""
import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
RESULTS = ROOT / "results"
CELLS = sorted(RESULTS.glob("selfcont_set_*.json"))
CELLS = [p for p in CELLS if p.name != "selfcont_set_failures.json"]

pytestmark = pytest.mark.skipif(not CELLS, reason="no selfcont cells measured")


def _load(p):
    return json.load(open(p))


def test_the_frozen_files_still_hash_to_what_their_sha256_files_claim():
    pr = EXP / "prereg_selfcont.json"
    claimed = (EXP / "prereg_selfcont.sha256").read_text().split()[0]
    assert hashlib.sha256(pr.read_bytes()).hexdigest() == claimed, (
        "prereg_selfcont.json no longer hashes to prereg_selfcont.sha256. Either the prereg was "
        "edited after freezing -- which makes every kill condition in it post-hoc -- or the hash "
        "file was. Neither is repairable by re-hashing.")

    probes = _load(EXP / "probe_strings_selfcont.json")
    claimed_p = (EXP / "probe_strings_selfcont.sha256").read_text().split()[0]
    payload = json.dumps([e["s"] for e in probes["strings"]], ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == claimed_p == probes["strings_sha256"], (
        "the probe string list no longer hashes to its frozen value. The intersection every "
        "cross-model distance is computed over is defined by that list.")


def test_the_prereg_names_the_probe_list_it_was_frozen_against():
    pr = _load(EXP / "prereg_selfcont.json")
    probes = _load(EXP / "probe_strings_selfcont.json")
    assert pr["probe_strings_sha256"] == probes["strings_sha256"], (
        "prereg_selfcont.json records a different probe-string hash than the probe file holds. The "
        "registered comparisons would then be defined over a set nobody froze.")


def test_every_measured_cell_was_run_against_those_same_frozen_files():
    pr_sha = (EXP / "prereg_selfcont.sha256").read_text().split()[0]
    probe_sha = _load(EXP / "probe_strings_selfcont.json")["strings_sha256"]
    bad = []
    for p in CELLS:
        d = _load(p)
        if d.get("_prereg_sha256") != pr_sha or d.get("_probe_strings_sha256") != probe_sha:
            bad.append(p.name)
    assert not bad, (
        f"cells measured against a different freeze: {bad}. A cohort assembled from cells that "
        f"answer different designs is not a cohort, and the Hamming distances across it are not "
        f"comparable.")


def test_the_determinism_flag_and_the_stored_set_both_survived_into_every_cell():
    bad_flag, bad_set = [], []
    for p in CELLS:
        d = _load(p)
        if not (d.get("_deterministic") and d["_determinism_check"]["bit_for_bit_identical"]):
            bad_flag.append(p.name)
        derived = {i for i, m in enumerate(d["margins_e4"]) if m > 0}
        if derived != set(d["self_continuing_ids"]) or len(derived) != d["n_self_continuing"]:
            bad_set.append(p.name)
    assert not bad_flag, (
        f"cells not certified deterministic: {bad_flag}. The design has no seeds and no averaging; "
        f"if the estimator is not reproducible there is nothing to fall back on.")
    assert not bad_set, (
        f"the stored self-continuation set disagrees with the stored margins in {bad_set}. "
        f"bit(t) == (margin(t) > 0) is the definition, not a derived convenience -- if they part "
        f"company, the set and the tau ladder are measuring different things. Margins are stored as "
        f"integers scaled by 1e4 with a rule that never sends a nonzero margin to zero, precisely "
        f"so that this identity survives the rounding.")


@pytest.mark.skipif(not (RESULTS / "selfcont_verdict.json").exists(),
                    reason="selfcont_verdict.json not present")
def test_the_identification_chance_levels_are_the_ones_the_cohort_actually_implies():
    """F163's defect, on this run's shape: the wrong baseline makes a null look like a capability."""
    v = _load(RESULTS / "selfcont_verdict.json")
    ident = v["identification"]
    counts = ident["family_counts"]
    n = ident["n"]
    assert sum(counts.values()) == n, (
        f"the family counts {counts} do not sum to the {n} models scored. If a model failed to load "
        f"and the chance level was not recomputed, K4 was violated.")
    fam_chance = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    assert abs(ident["chance_family_level"] - fam_chance) < 5e-4, (
        f"stored family-level chance {ident['chance_family_level']} against {fam_chance:.4f} "
        f"recomputed from {counts}. This is the number the rank-1 result must beat.")
    assert abs(ident["chance_instance_level"] - 1 / (n - 1)) < 5e-4
    assert ident["chance_family_level"] > ident["chance_instance_level"], (
        "on any cohort with a family of more than one member, family-level chance EXCEEDS "
        "instance-level chance. If this ever inverts, the two have been swapped, and quoting "
        "1/(n-1) for a family test is exactly the base-rate slip gatecheck.balance exists for.")
