"""The decision layer for #84's cross-family loss collapse, separated from the measurement.

WHY THIS IS A SEPARATE MODULE. `loss_collapse_families.py` measures and decides in one pass, so
re-deciding cost a full re-measurement -- three families of checkpoint downloads, hours, and the
downloads dominate. The measurement is the expensive, non-reproducible half (weights get gated,
revisions get renamed); the decision is arithmetic over a JSON. Fusing them meant every change to
the verdict logic put the numbers at risk, which is the same coupling that made the import-closure
guard so expensive that `provenance.py` had to be exempted from it.

WHAT THIS BUYS BEYOND CONVENIENCE. Two runs can now be decided by *identical* logic. Routing the
verdict through `gatecheck` changes where the gates bind -- `noise_gate` refuses below 2x the seed
floor where the hand-rolled check refused below 1x -- so a run decided before the change and one
decided after are not comparable. Re-deciding the earlier run's cached cells removes that split at
no measurement cost.

WHY CHANGING THE CRITERION AFTER SEEING DATA IS ADMISSIBLE HERE, WHICH IT USUALLY IS NOT. The F80
meta-defect is a verdict layer edited while its statistic is in view, and that hazard is real. It
does not apply to a threshold that (a) predates this run, (b) is the shared default already binding
four other scripts, and (c) is strictly TIGHTER than what it replaces. A change that can only move
a verdict toward NOT_DECIDABLE cannot manufacture a result, which is the property that makes it
safe to apply to data already collected. A bespoke threshold chosen after seeing the answer would
have none of those three.

PROVENANCE IS SPLIT, DELIBERATELY. `_analysis_provenance` continues to name the script that
produced the measurements. This module writes `_decision_provenance` instead. Stamping the decider
over the measurer's slot would leave the file claiming a source that never computed its numbers --
the F45/F46 trap arriving through the decision half rather than the measurement half.

Usage:
    .venv/bin/python experiments/loss_collapse_decide.py results/loss_collapse_families.json
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src"),
                str(_ROOT / "gatecheck" / "src")]

from provenance import stamp                                            # noqa: E402
from gatecheck import (NOT_DECIDABLE, carries_verdict, directional,     # noqa: E402
                       dynamic_range, noise_gate)
from gatecheck.cohort import cohort_complete                            # noqa: E402
from loss_collapse_families import FAMILIES, _spread_at_matched   # noqa: E402


                                                    # uniform over bytes; nothing sane exceeds it
BPB_CEILING = 8.0
BPB_IDENTITY_TOL = 1e-4


def _unit_gate(cells):
    """Cells whose bpb is not a bits-per-byte number. Returns the offenders.

    THE REGISTERED INTERVAL (0.4, 2.5) WAS OURS TO GET WRONG, AND IT WAS WRONG. It was written to
    catch a units error -- nats-per-token recorded where bits-per-byte belongs -- but it was
    expressed as a range, and a range cannot tell "wrong units" from "high loss". A random-init
    checkpoint reads 3.92-3.95 bpb, which is a correct bits-per-byte number for a model that has
    learned nothing, and the grid *deliberately* includes random init as its chaotic-init control.
    So the gate rejected its own controls by construction, and `pythia-410m|step128` at 2.63 with
    it -- a real dip-region checkpoint. That was knowable from the design before any measurement,
    which is why fixing it is a specification repair rather than a criterion tuned to an outcome.

    The replacement checks the identity that DEFINES the quantity, which is what the gate was
    reaching for:

        bpb = nats_per_token * n_tokens / (ln 2 * n_bytes)

    Every field on the right is recorded per cell, so this is exact rather than a guess about
    plausible values. It is also strictly stronger against the error it was built for: nats and
    bits differ by a factor of ln 2, a 44% discrepancy, against a tolerance of 1e-4 -- while the
    observed worst deviation across the measured cells is 4e-6, which is the rounding in the
    stored value. The only range check kept is a ceiling at 8 bits/byte, the entropy of a uniform
    byte, above which the number cannot be a bits-per-byte loss whatever produced it.
    """
    bad = []
    for k, c in sorted(cells.items()) if isinstance(cells, dict) else enumerate(cells):
        name = k if isinstance(k, str) else f"{c['family']}|{c['revision']}"
        bpb, nb = c.get("bpb"), c.get("n_bytes")
        if bpb is None or not math.isfinite(bpb) or bpb <= 0 or bpb > BPB_CEILING:
            bad.append(f"{name}={bpb} (outside (0, {BPB_CEILING}])")
            continue
        if nb and c.get("n_tokens") and c.get("nats_per_token") is not None:
            pred = c["nats_per_token"] * c["n_tokens"] / (math.log(2) * nb)
            if abs(pred - bpb) / bpb > BPB_IDENTITY_TOL:
                bad.append(f"{name}: bpb={bpb:.5f} but nats/token, tokens and bytes imply "
                           f"{pred:.5f} -- these are not the same quantity")
    return bad


def _overlap_window(curves, key):
    """The interval on `key` where every usable family has coverage, or None.

    Same rule `_spread_at_matched` interpolates over -- max of the per-family minima, min of the
    maxima -- exposed separately so a gate can be applied to the series that is actually compared
    rather than to every cell in the file. Gating on all cells would be decorative here: the
    untrained checkpoints sit near 0.35 against a trained plateau near 0.18, so any range gate over
    the full set passes on the strength of points that never enter the comparison.
    """
    usable = {f: [(c[key], c["lambda_ca"]) for c in cs if c.get("lambda_ca") is not None]
              for f, cs in curves.items()}
    usable = {f: v for f, v in usable.items() if len(v) >= 3}
    if len(usable) < 2:
        return None
    lo = max(min(x for x, _ in v) for v in usable.values())
    hi = min(max(x for x, _ in v) for v in usable.values())
    return (lo, hi) if hi > lo else None


def decide(res):
    """Re-decide `res` in place from its cached cells. Returns the verdict string."""
    cells = [c for c in res["cells"].values() if "bpb" in c]
    parts, gates = [], []

    declared = [f"{f}|{r}" for f, _, cks, _ in FAMILIES for r, _ in cks]
    coh = cohort_complete(declared, [f"{c['family']}|{c['revision']}" for c in cells],
                          unit="checkpoint")
    parts.append(f"COHORT: {coh.reason}")

    bad = _unit_gate(cells)
    parts.append(
        f"UNITS: all {len(cells)} cells satisfy bpb = nats/token * tokens / (ln2 * bytes) to "
        f"{BPB_IDENTITY_TOL:g} and sit under the {BPB_CEILING} bits/byte ceiling."
        if not bad else f"UNITS: NOT BITS-PER-BYTE: {bad}.")

    curves = {}
    for c in sorted(cells, key=lambda c: (c["family"], c["tokens_B"])):
        curves.setdefault(c["family"], []).append(c)

    floors = [c["lambda_sd"] for c in cells if c.get("lambda_sd")]
    floor = float(np.mean(floors)) / np.sqrt(8) if floors else None
    s_bpb, n_bpb = _spread_at_matched(curves, "bpb")
    s_tok, n_tok = _spread_at_matched(curves, "tokens_B")

    if bad or not coh.complete or s_bpb is None or floor is None:
        why = ("the bpb unit gate rejected cells" if bad else
               "the declared cohort is incomplete" if not coh.complete else
               "no overlapping bpb range exists across families" if s_bpb is None else
               "no seed spread was recorded, so there is no floor to gate against")
        res["verdict"] = " ".join(parts) + f" NOT DECIDABLE: {why}."
        res["analysis"] = dict(cohort=coh.block(), unit_gate_failures=bad, decided=False)
        res["_decision_provenance"] = stamp(__file__)
        return res["verdict"]

    # The range gate, on the series actually compared rather than on every cell in the file.
    win = _overlap_window(curves, "bpb")
    inwin = [c["lambda_ca"] for c in cells
             if c.get("lambda_ca") is not None and win and win[0] <= c["bpb"] <= win[1]]
    gates.append(dynamic_range(inwin or [0.0], floor=floor,
                               name="lambda_ca across the matched-bpb window"))

    # The noise gate, before the two spreads are compared -- 2x the floor, not 1x.
    gates.append(noise_gate(abs(s_bpb - s_tok) if s_tok is not None else s_bpb, floor))

    # The hypothesis is directional -- bpb organises better than tokens, i.e. s_tok - s_bpb > 0 --
    # but this is deliberately NOT appended to `gates`. A sign opposite to the prediction is
    # evidence AGAINST the hypothesis, not an inability to decide, which is precisely the reading
    # gatecheck.directional exists to forbid. Blocking on it would report a contradicted
    # hypothesis as NOT_DECIDABLE and discard the finding, and the registration names "no collapse"
    # as a decidable outcome with its own kill text. It selects the branch instead, once the
    # blocking gates have established the comparison is readable at all.
    dirn = (directional(s_tok - s_bpb, expect="increase", floor=floor)
            if s_tok is not None else None)

    parts.append(
        f"PRIMARY: across-family spread of lambda_ca at matched bits-per-byte is {s_bpb:.4f} over "
        f"{n_bpb} families"
        + (f", against {s_tok:.4f} at matched token count over {n_tok}." if s_tok is not None
           else ", with no token-matched comparison available.")
        + f" Seed floor {floor:.4f}.")

    verdict = carries_verdict(gates, value=dict(s_bpb=s_bpb, s_tok=s_tok, floor=floor))
    if verdict.status == NOT_DECIDABLE:
        parts.append(
            f"NOT DECIDABLE, and for the reason F88 was: {verdict.reason}. Not a null about loss "
            f"-- the test is underpowered, and the fix is finer checkpoint spacing, which for the "
            f"non-Pythia families does not exist to be had.")
        decided = False
    elif dirn is not None and not dirn.usable:
        parts.append(
            f"NO COLLAPSE, AND THE HYPOTHESIS IS CONTRADICTED RATHER THAN UNSUPPORTED: matching on "
            f"loss aligns the families WORSE than matching on tokens ({s_bpb:.4f} against "
            f"{s_tok:.4f}), which is the opposite of what this issue proposed. {dirn.reason} That "
            f"is the registered kill: lambda_ca is not a function of model quality in this unit, "
            f"and cross-family timing stays unreachable by this route. Gates: {verdict.reason}")
        decided = True
    elif s_bpb <= floor:
        parts.append(
            f"COLLAPSE: at matched model quality the families agree to within the seed floor, so "
            f"lambda_ca is a function of HOW GOOD the model is rather than how long it trained. "
            f"Gates: {verdict.reason}")
        decided = True
    else:
        parts.append(
            f"NO COLLAPSE: the families do not agree at matched bits-per-byte ({s_bpb:.4f} against "
            f"a {floor:.4f} floor), so lambda_ca is not a function of model quality in this unit "
            f"either. Gates: {verdict.reason}")
        decided = True

    parts.append(
        "BOUNDARY: bits-per-byte removes the tokenizer confound, not the corpus one -- all three "
        "families are scored on Pile text, which is training distribution for Pythia and OLMo but "
        "not identically weighted for either. Architecture, data order and optimiser still differ "
        "across families simultaneously (F98's attribution note applies unchanged).")

    # Keep what is being replaced. Overwriting the measuring run's verdict in place would leave
    # the file asserting the corrected reading with no trace that a different one was published
    # first -- the correction living in prose while the artifact carries only the survivor. Only
    # the ORIGINAL is kept: re-deciding twice must not chain a stack of intermediate readings.
    if "verdict" in res and "_superseded_verdict" not in res:
        res["_superseded_verdict"] = dict(
            verdict=res["verdict"], analysis=res.get("analysis"),
            by=(res.get("_analysis_provenance") or {}).get("script"),
            why="superseded by loss_collapse_decide.py; see _decision_amendments")

    res["verdict"] = " ".join(parts)
    res["_decision_amendments"] = [dict(
        supersedes="unit_gate: bpb must be finite and within (0.4, 2.5) for every cell",
        replaced_by=("bpb = nats_per_token * n_tokens / (ln2 * n_bytes) to "
                     f"{BPB_IDENTITY_TOL:g} relative, plus a {BPB_CEILING} bits/byte ceiling"),
        why=("The registered interval could not distinguish wrong units from high loss. It "
             "rejected the grid's own random-init controls (3.92-3.95 bpb, correct values for a "
             "model that has learned nothing) and pythia-410m|step128 at 2.63, a real "
             "dip-region checkpoint. That follows from the design, not from the measurements, "
             "so this is a specification repair rather than a criterion tuned to an outcome."),
        strictness=("stronger against the error the gate was written for: nats and bits differ "
                    "by ln2, a 44% discrepancy, against a 1e-4 tolerance"))]
    res["analysis"] = dict(
        spread_at_matched_bpb=round(s_bpb, 5),
        spread_at_matched_tokens=None if s_tok is None else round(s_tok, 5),
        lambda_seed_floor=round(floor, 5), n_families_bpb=n_bpb, n_families_tokens=n_tok,
        matched_bpb_window=None if win is None else [round(w, 5) for w in win],
        n_in_window=len(inwin), decided=decided,
        gates=[g.block() for g in gates],
        directional=None if dirn is None else dirn.block(), cohort=coh.block())
    res["_decision_provenance"] = stamp(__file__)
    return res["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=pathlib.Path)
    ap.add_argument("--write", action="store_true",
                    help="write the re-decided verdict back; without it the run is a dry read")
    a = ap.parse_args()
    res = json.loads(a.results.read_text())
    print(decide(res))
    if a.write:
        a.results.write_text(json.dumps(res, indent=2) + "\n")
        print(f"\nwrote {a.results}")
    else:
        print("\n(dry run -- pass --write to update the file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
