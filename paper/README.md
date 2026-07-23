# Paper

arXiv-ready draft of the token-lattice-CA instrument paper.

- `paper.md` — the readable Markdown source (edit here first; keep in sync with `.tex`).
- `paper.tex` — arXiv LaTeX article (single-column `article` class, `natbib`).
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
2. Novelty harness DONE — see `results/deep_research_novelty.md`. Verdicts: velocity∝r
   NOVEL; damping length + edge-of-chaos PARTIALLY ANTICIPATED (cite arXiv:2505.19458);
   AR replication NOVEL. Foreground the measurement layer vs the shared Glauber-on-MLM
   substrate (arXiv:2605.16378, which uses maximal coupling ≠ our CRN).
3. Keep the **instrument** as the claimed novel core; frame edge-of-chaos as
   *measurement* of a decades-old idea, never a discovery.
4. Term is **damping length / error-damping length**, NOT "repair" (collides with
   self-repair / Hydra effect, arXiv:2307.15771 / 2402.15390) and NOT "self-correction"
   (SPARC arXiv:2607.09803). Disambiguate both in related work.
5. Verify every `refs.bib` entry's real title/authors/ID (2607.09803 = SPARC, not QUIVER;
   QUIVER = 2605.23956). Direct-reads owed near submission (2026 preprints may change).
