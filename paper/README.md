# Paper

Draft of the token-lattice cellular-automaton instrument paper.
(Repository and author names are deliberately omitted here: this directory is
bundled as anonymised supplementary material for a double-blind venue.)

- `paper.tex` — the paper. Builds with `tectonic -X compile paper.tex`.
- `NOTES.md` — the claim inventory and cut ledger; decide content there, assemble prose from it.
- `refs.bib` — BibTeX. **Several arXiv IDs are as supplied by the project brief and are
  flagged for verification before submission** (the novelty check owes direct-reads of
  arXiv:2607.09803 and QUIVER).

Figures are pulled from `../fig/` via `\graphicspath`. Referenced:
`phaseA_radius.png` (radius law / certification), `phaseA_velocity.png` (v∝r finite-size),
`repair_scale.png` (repair length + capacity), `calib_discrimination.png` (calibration).

## Build

No LaTeX toolchain is installed on the dev machine. Build on any TeX distribution
(TeX Live / MacTeX) or paste into Overleaf:

```bash
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

## Pre-submission checklist (from the novelty check)

1. Direct-read arXiv:2607.09803 and QUIVER; confirm no overlap with the instrument.
2. Novelty harness + reference audit DONE. Verdicts (`results/deep_research_novelty.md`):
   velocity∝r **NOT novel** (classic CA light cone — Bagnoli–Rechtman–Ruffo 1992,
   Lieb–Robinson, arXiv:2101.01313; reframed as import); damping length + edge-of-chaos
   PARTIALLY ANTICIPATED (cite arXiv:2505.19458); AR replication NOVEL. All refs.bib IDs
   web-verified (real titles/authors). Full-paper reads DONE: 2605.16378 (Glauber-on-MLM —
   maximal coupling / Hamming contraction for mixing, ≠ our CRN spatial propagation) and
   2605.23956 (QUIVER — zero method overlap, pipeline graphs).
3. Keep the **instrument** as the claimed novel core; frame edge-of-chaos as
   *measurement* of a decades-old idea, never a discovery.
4. Term is **damping length / error-damping length**, NOT "repair" (collides with
   self-repair / Hydra effect, arXiv:2307.15771 / 2402.15390) and NOT "self-correction"
   (SPARC arXiv:2607.09803). Disambiguate both in related work.
5. Verify every `refs.bib` entry's real title/authors/ID (2607.09803 = SPARC, not QUIVER;
   QUIVER = 2605.23956). Direct-reads owed near submission (2026 preprints may change).

## The anonymised mirror vs the tagged tree (#54)

The paper's Reproducibility appendix links an anonymised mirror on OSF. Two things about it are
worth stating explicitly, because both are the kind of difference that looks like a discrepancy
if you discover it rather than read it here.

**The mirror is not a byte-copy of the tag.** `build_mirror.py` rewrites absolute checkout paths
in twelve machine-written logs — lines of the form `wrote /Users/<user>/…/results/x.json` become
`wrote ./results/x.json` — and replaces this file's sibling `README.md` H1, which was the
repository name. Nothing else is altered. The repository's own logs are never touched; the
rewriting happens only in the derived mirror, so the evidence record keeps exactly what the
analyses wrote.

**The mirror contains the URL that points at it.** Because it is built from the *final* tag, and
that tag is the one carrying the appendix's mirror sentence, `paper/paper.tex` inside the mirror
already contains the mirror's own address. That is self-referential but correct, and it is the
reason the mirror is rebuilt and re-uploaded after the URL insertion rather than before. A mirror
built from the pre-insertion tag would not match the tag it claims to mirror.

So: mirror = tagged tree, minus 18 absolute paths and one heading. Verified by
`build_mirror.py`, which audits its own output for identifying strings derived at run time from
the git remote, `user.name`/`user.email`, and the checkout path.

At camera-ready: flip the real repository public, swap the URL, and leave the tag name and sha
unchanged. The tag pins the paper; the URL is the only thing that moves.
