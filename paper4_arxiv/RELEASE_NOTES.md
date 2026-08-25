# Prepared release notes — `v1.3.0`

**Not cut.** Held pending paper 3's arXiv identifier, so one release can carry both papers'
metadata. Everything below is computed against `v1.2.0` and is final except the two lines marked
`<<<`.

Cut with:

```bash
gh release create v1.3.0 \
  --title "v1.3.0 — paper 4 (draft complete): the evidence it cites, archived" \
  --notes-file paper4_arxiv/RELEASE_NOTES.md
```

---

Archival release for **paper 4**: the evidence it cites is now in the archived snapshot.

Paper 4 is **draft complete and reviewed, not submitted**, so it has no arXiv identifier yet. This
release exists for the reason `v1.1.0` and `v1.2.0` did: when the paper is submitted, the archived
DOI in its comments line must resolve to a snapshot that actually contains the results it names.

## Why this release was needed

The Zenodo concept DOI [10.5281/zenodo.21880472](https://doi.org/10.5281/zenodo.21880472) resolves
to the *latest* archived version, and paper 4 points readers at it for "code, per-run results,
pre-registrations and the findings ledger". Against `v1.2.0` (24 Aug) that link was wrong for paper
4's purposes — **every results file paper 4 names by source comment was absent from it**:

| results file paper 4 names | v1.2.0 | v1.3.0 |
|---|---|---|
| `results/selfcont_verdict.json` | absent | **present** |
| `results/escape_destinations.json` | absent | **present** |
| `results/escape_widening.json` | absent | **present** |
| `results/floor_survey.json` | absent | **present** |
| `paper4_arxiv/` | absent | **present** |
| new `results/` files | — | **47** |
| `findings.md` entries | 166 | **173** |

`tests/test_paper4_numbers.py` enforces the link: it fails if a source comment names a file that
does not exist, and separately if any decimal literal in the manuscript is absent from every results
file.

## Paper 4 — what it reports

*Where a Model Sends Its Own Repeated Token*, 8 pages, cs.CL.

Feed a model the most unnatural input available — one of its own tokens, twice — and read
`argmax p(· | t, t)` in a single forward pass. Do that for every token in the vocabulary and the
result is a map defined on the whole vocabulary, with two halves.

- **The first half fails, and is reported at the same length as the second.** Which tokens are fixed
  points of that map is partially anticipated by prior work, and it fails as a signature anyway: the
  natural distance on it is $83\%$ explained by set cardinality alone, it separates a corpus
  manipulation by two bits in $3471$ against a precision floor of zero, and attributes families at
  $0.5833$. It was registered first.
- **The second half is the finding.** Where the map sends tokens that are *not* fixed points is
  unrecorded in prior work — the one paper holding those tokens logged them as a zero. Pairing the
  comparison on the source token removes the cardinality confound by construction ($r$ from $0.9128$
  to $-0.0932$) and attributes model **families** at $0.8333$ — twelve models scored against a pool
  of nineteen — with a chance rate of $0.1389$. Family predicts agreement better than tokenizer
  ($0.2031$ against $0.1205$), and recurrent architectures cluster at balanced accuracy $0.90$ once
  each model's dominant destination is excluded.
- **The robustness envelope is measured, not assumed.** 8-bit weight rounding moves the map *less
  than deduplicating the training corpus does* ($0.9004$ against $0.6353$, on one support); 4-bit
  destroys it, and still does at the granularity deployment actually uses, so the failure is not an
  artefact of coarse quantization. The fingerprint is claimed for full-precision models only, and it
  identifies **families, not individual copies**.

## Also in this release

- **Eight frozen pre-registrations** with `.sha256` sidecars, each committed before the run it
  governs. The commit graph is the ordering evidence: `matched_entropy_pairing` was committed in a
  tree containing no Δφ at all, and its consumer verifies that hash before loading a model.
- **A prior-art gate that bit** (F186): 101 agents, full-text verification, and the finding that the
  binding constraint was this project's *own* paper 1. The delta paragraphs it mandated are §3 of
  the manuscript, and `tests/test_paper4_numbers.py` fails if either prohibited pitch reappears.
- **Three defects found in this work rather than in the world.** A registered primary estimand that
  was $83\%$ cardinality; an abstract asserting four things its own body withdraws, found by looking
  for it deliberately; and an anti-vacuity gate that tested the wrong quantity, so a censored
  comparison first read as a positive result.
- **Four manuscript-gating test files**, including `tests/test_abstract_matches_body.py`, which runs
  against **all four** papers and not only this one.

## Papers

| | |
|---|---|
| Paper 1 — the instrument | [arXiv:2608.10986](https://arxiv.org/abs/2608.10986) |
| Paper 2 — the domain | [arXiv:2608.21315](https://arxiv.org/abs/2608.21315) |
| Paper 3 — the cohort | `<<< arXiv:NNNN.NNNNN` |
| Paper 4 — provenance | draft complete, not submitted |

`CITATION.cff` carries every announced paper under `identifiers`, each with its bare arXiv ID and its
DOI; `preferred-citation` stays pointed at paper 1, which is the citation for the *software*.

`<<< before cutting: replace paper 3's row above, and confirm CITATION.cff and the README carry its identifier.`
