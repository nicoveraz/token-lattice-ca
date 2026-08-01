# gatecheck — design document

**Date:** 1 August 2026. **Status:** v0.1.0, extracted from textca as a spin-off.
**Companion:** `../critical_analysis.md` §6.7, which recommended this extraction ("ship the
discipline as a product of its own"), and textca tracker issue #60 ("the calibration ladder is
a reusable methodology, not just this paper's step 1").

## 1. Purpose

textca's scientific programme produced mostly negative knowledge about its own probe; its
lasting asset is the machinery that made those negatives *discoverable*: six confident wrong
verdicts each caught by its own check before publication. That machinery was welded to one
repo — one file layout, one set of JSON keys, one macOS machine, tests that freeze one paper's
numbers. gatecheck is the extraction: the same patterns, made project-agnostic, installable,
and documented by the incidents that motivated them.

The intended user is anyone running computational experiments whose conclusions depend on
estimators, fits, controls, and results files — interpretability work very much included: SAE
evaluations, probing studies, and activation-patching pipelines fail in exactly the ways this
machinery detects (probe artifacts, pseudoreplication, estimator bias at the wrong geometry,
silent drift between results and manuscript).

## 2. Design rules

The package follows five rules, all learned from the source project. First, every guard is
motivated by a named incident, and the module docstring tells that story — a check whose cost
is not justified by a remembered failure gets deleted during some cleanup. Second, the honest
failure mode is loud and specific: `NOT_DECIDABLE`, `BrokenCouplingError`, `VacuousNullError`,
`EdgeRejection`, a `StampReport` that names the stale files — never a softened number. Third,
blindness is enforced by API shape where possible: `Gate.check` cannot see target data because
its signature has nowhere to put it. Fourth, no framework lock-in: plain functions and
dataclasses, numpy as the only dependency, helpers callable from any test runner. Fifth, the
package mechanizes discipline, not science: it will not choose your control or your reference
system, only make it hard to skip them silently.

## 3. Extraction map

| gatecheck module | textca origin | What changed in extraction |
|---|---|---|
| `gate.py` | `experiments/dp_calibration.py` (the F56 repair; also `assembly_calib.py`'s gate) | Reference system and estimator become caller-supplied callables; "geometry" becomes an opaque value; disputed-truth kindest-candidate rule and the dev+spread pass rule kept verbatim in spirit; ladder generalized with a pluggable cost model; `NOT_DECIDABLE` promoted to a first-class `Verdict` |
| `provenance.py` | `experiments/provenance.py` (#38, F45/F46, #52) | `root` made explicit; environment fingerprint added (closing the hole the original's docstring names: "third-party package versions are NOT covered"); external-script stamps marked unverifiable instead of failing; verification returns a structured report |
| `prereg.py` | the `_preregistration` blocks in `dp_fss_z.py`, `attractor_interventions.py`, `evidence_falloff.py`, `assembly_baselines.py`; the `_UNREGISTERED`/`_INFLATED` convention (F39) | Self-hash for tamper detection (new); kill conditions become named predicates that must ALL be evaluated (an unevaluated kill condition raises); quarantine gets a fence (no silent overwrite); `assert_no_smuggling` is new |
| `units.py` | the F57 diagnosis and the W1/A1 retraction; textca's rule "state what the independent unit is, and test it" | textca stated the rule in prose and fixed instances by hand; gatecheck implements the general check (ICC(1) with unbalanced-design correction, design effect, effective n) plus `unit_level` collapse |
| `fits.py` | `dp_calibration.slope` (shared estimator); the F59 scan-edge rule ("reject a minimum that lands on the scan edge") | `slope_loglog` near-verbatim; edge rejection generalized to any scanned cost with a margin parameter |
| `nulltest.py` | the exact-zero CRN null (`test_null_all_backends.py`, asserted in production in `real_generation_damage.py`); the anti-vacuity converse tests; F65's treatment-minus-control verdict | Exact equality kept tolerance-free by design; the null/effect pair packaged as a certificate; control adjustment packaged as a verdict block |
| `manifest.py` | `experiments/build_paper_manifest.py` + `tests/test_paper_numbers.py` (#48, #64) | The three kinds (measured/published/arithmetic) kept; `published` now *requires* a ref at construction; the recompute check — documented but never shipped in textca ("the promised recompute test does not exist", critical_analysis §5) — implemented via dot-paths + round/format specs |
| `results.py` | the results-file conventions scattered across textca scripts; `provenance.rel` and the twelve-logs leak (#52) | One save/load pair; the leak check runs at save time and refuses, rather than auditing after publication |
| `testing.py` | `tests/test_results_self_consistency.py` (staleness sweep), `test_paper_numbers.py` (retraction pinning, manifest), the single-implementation greps (#63) | Reusable assertions instead of project-specific tests; the grep guard ships with its known limitation stated (see §6) |

## 4. What was deliberately generalized beyond the original

Three gaps the critical analysis identified in textca's machinery are closed here: provenance
now covers the environment (package versions), not only project bytes; the manifest can
*recompute* literals from source files instead of only checking substring presence, so results
cannot drift under a stale manifest and manuscript while everything stays green; and stamps of
scripts outside the project root are marked external instead of producing unverifiable
entries. A fourth generalization is structural: preregistration blocks are self-hashed, so a
post-hoc edit of the contract is detectable, which the JSON-blob original could not promise.

## 5. What is deliberately absent (roadmap for 0.2+)

Golden-file tooling (textca's bit-identity regression with the do-not-relax rule) is absent
because its portability story — goldens are machine- and precision-locked — needs a design of
its own (per-device golden sets or tolerance tiers) before it generalizes responsibly. The
citation audit (`audit_refs.py` + offline `test_refs_match_arxiv.py`) is absent because it
needs a network fetch layer and caching policy; it extracts cleanly and should come next. The
anonymization mirror builder (`build_mirror.py`, with its derive-don't-list identifier rule)
is submission-specific but contains a reusable core. A pytest plugin proper (auto-collecting
staleness tests from a `results/` glob via entry point) would remove boilerplate; v0.1 keeps
plain helpers so nothing magic happens at collection time. Multiplicity bookkeeping (a
project-level test registry that forces a correction decision, textca's W8 lesson) is the
hardest and most valuable candidate.

## 6. Known limitations, stated up front

The single-implementation guard is a grep, and textca's own history documents the evasion: an
import-and-wrap alias (`from lyapunov import is_unignited as _unig`) slips past a definition
pattern while proliferating the adapter shape around it. The guard buys drift *detection*, not
proof; pair it with review. The ICC-based effective-n is a model (one-way random effects), not
an oracle: it catches the two incident classes that motivated it (shared-seed grids, shared
batch draws) but exotic dependence structures need their own accounting. The manifest's
presence check is substring-based and weak for short literals — that weakness is inherited
from textca and is why the recompute path exists; prefer entries that carry one. And the gate
is only as honest as the caller's discipline in sharing the estimator between reference and
target: the API cannot verify that the same function object runs on both (textca's
`test_assembly_calib.py` asserts shared function objects; do that in your project tests).

## 7. Packaging notes for the author

The name `gatecheck` has not been checked against PyPI — verify before publishing, and rename
cheaply now rather than expensively later if taken. No license is set in `pyproject.toml`;
textca itself has no LICENSE file, so the spin-off inherits an undefined status until you
choose one (MIT/BSD-3/Apache-2.0 are the conventional choices for tooling meant to spread —
note Apache-2.0's patent grant is the usual argument for it). The package is pure Python with
a numpy floor of 1.24; CI should run the suite on 3.10–3.13. The test suite runs in under a
second with no network and no model downloads, by construction — keep it that way; it is the
property that lets adopters run it in every CI job.

## 8. Relationship back to textca

textca can adopt gatecheck incrementally without invalidating existing results: `provenance`
is a drop-in (same block shape plus new fields), `dp_calibration` can become a thin
project-specific wrapper around `Gate` (the DK reference and exponent set stay in textca —
gatecheck does not know what Domany–Kinzel is, by design), the self-consistency tests can
delegate staleness sweeps to `testing.assert_fresh`, and the paper-number system can migrate
entry-by-entry, gaining recompute coverage as `path` specs are added. The one thing that must
not migrate is the frozen-paper conclusion pins — those belong to the submission tag, and
keeping them out of the reusable package is half the reason the package exists.
