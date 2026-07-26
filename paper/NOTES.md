# Paper working notes

The place to think about the paper. **Add and modify findings here; write prose later.**

Rationale: iteratively micro-editing `paper.tex` to hit a page count mixes two different
jobs — deciding *what is true and what gets claimed*, and *fitting it on 5 pages*. This file
holds the first. The `.tex` gets assembled from it in one pass when the content is settled.

Status: `paper.pdf` builds (`tectonic -X compile paper.tex`), **10 pages: body 1–5, references 6,
Reproducibility appendix 7, checklist 8–10**. **The body fits the 5-page limit.** 0 undefined refs,
0 undefined citations, 0 overfull boxes. Compliance done: responsible-use section written,
double-blind clean, all 4 checklist TODOs resolved, all 5 unverified citations verified (F43).

---

## 1. Claim inventory — every number, with its source file

Nothing goes in the paper that isn't in this table with a `results/` path.

| # | Claim | Number | Source |
|---|---|---|---|
| C1 | DK damage field **is** the automaton on `p2=0` | **0** mismatching cells, 4096×1500×3 seeds; control **16** | `dk_calib.json:part_a_exact_identity` |
| C2 | Rule 90 closed form at p1=1 | density **0.03125** = 2^popcount(1500)/4096 | same |
| C3 | DK critical points | site DP **0.7065** vs 0.705489(4); W18 **0.8092**/damage **0.8089** | `dk_calib.json:calibration` |
| C4 | — and they cannot discriminate the disputed W18 values | ours ±1%, the two published are 0.96% apart | `dk.py:ANCHORS` |
| C5 | ECA coarse split on ignition probability | ordered **0.046** [0, 0.102] vs edge 0.668 / chaotic 0.682; **p=0.0, d=3.03** | `eca_ordered_vs_rest.json` |
| C6 | 3-class ordering fails | edge vs chaotic **p=0.470** | same |
| C7 | Ordered-group λ is a floor sentinel | 5/7 rules at exactly −0.4·ln10, zero-width CI | `eca_calib_hardened.json`, F40 |
| C8 | CML vs exact Benettin | max diff **0.0011**, non-monotone in ε | `cml_benettin.json` |
| C9 | Census recovers known matrices | self-TV **0.22** vs cross **0.95**, baseline 0.91 | `calib_census.json` |
| C10 | **Developmental transition** | λ_ca +0.0247→+0.1683 (N=48, **d=1.59**, n_pre=16), +0.0320→+0.1686 (N=96, **d=1.71**, n_pre=**15** after F42) — pre set is the pre-registered {256,512} | `dev_transition_shape.json:headline` |
| C10b | **Sign agreement** (pre-set-free restatement, preferred) | pre 6/16 and **7/15** negative; **0/48 plateau runs negative**, min +0.1074. Ordinal, so **unchanged by F42** | `...:sign_agreement` |
| C11 | All 4 pre-registered members survive BH-FDR | p_BH ≤ 2e−05 | `dev_transition_phase3.json`, `logs/phase3_dev.log` |
| C12 | Shape is non-monotone | overshoot **+1.4% to +22.4%**; separable in **1/4** cells after BH, and that cell is D_norm — λ_ca shows +1.4% (p_BH 0.78) at N=96 | `...:peak_vs_plateau` |
| C20 ✅ **in the paper** | **Transition timing vs model size** | replicates in 4/4 sizes (p_BH ≤ 0.015); crossing 70m <128, 160m 128→256, 410m & 1b 256→512 (saturates); plateau level non-monotone 0.162/0.164/0.174/0.166 → no capacity axis | `dev_transition_scale.json` |
| C13b | **λ_ca intensive, D_norm 1/N over 4×** | λ_ca 0.168/0.169/0.160 (N^−0.04); D_norm 0.569/0.306/0.139 (N^−1.02) | `dev_transition_n192.json` |
| C13 | λ_ca is size-robust | **95%** retention (was 104% before F42 dropped the unignited run); plateau levels differ by −0.0003, **95% CI [−0.0229, +0.0223]** = agree within **±14%** | `...:size_scaling_W9` |
| C14 | D_norm is size-dependent | 53% retention; level 0.569 vs 0.306, **p=1.3e−08** | same |
| C15 | Variance collapse | sd(λ) 3.7× / 3.1×, Levene p≤1.7e−04. **Not** "seeds agree to a few percent" — plateau CV is 21.9%/25.4% | `...:variance` |
| C16 | **Real generation absorbs nothing** | P_persist **1.000**, P_reconverge 0.000, 3 models × 32 trials, null exactly 0 | `real_generation_damage.json` |
| C17 | — distributionally too | TV_norm **≈0.967** | `real_generation_reconvergence.json` |
| C18 | Coupling is monotone, not maximal | excess disagreement **1.3–5.4%**; 1.16–1.38× near agreement | `coupling_gap.json`, F41 |
| C19 | Cross-level negative | Pythia r=+0.71 vs GPT-2 r=−0.43; pooled p=0.025 is Simpson | `crosslevel.json` |

**Retracted — must never reappear:** 3-class ECA ordering; ordered-group mean λ = −0.32;
capacity→sensitivity axis; "damping length *of generation*"; "damage numbers are a lower
bound over admissible couplings" (true on DK only); "the literature disagreement resolves
toward HWD"; step256-vs-step1000 as the headline contrast (d=5.04); **step256-alone as the
pre baseline** (d=2.74/2.65, inflated ~1.7×/1.5× over the pre-registered {256,512});
**"λ_ca crosses zero between steps 512 and 1000"** — the pre-group mean is positive and the
cell-mean crossing is 256→512.

---

## 2. What the paper argues (settled)

**Spine:** validated instrument → developmental transition (C10–C15) → the boundary that
bounds it (C16–C18).

1. **Validation by reproduction**, with C1 as the load-bearing rung because agreement is
   *exact*, not fitted. Rungs 1–2 (logistic, CML) are explicitly non-weight-bearing.
2. **λ_ca carries the developmental claim, D_norm corroborates.** Decided by C13/C14 — this
   changed after Phase 3. D_norm has two independent scale problems (N-dependence C14,
   non-extremal coupling C18), so it is never quoted as a lattice-free property.
3. **F35 is a result, not a caveat** (C16). Mechanism: free generation never resamples a
   token; the ring CA revisits every site.
4. **The construction-held-fixed argument** is what licenses C10 despite C16: same ring,
   radius, temperature, coupling, seeds across checkpoints, so a change across checkpoints
   is attributable to the model. This must stay explicit.
5. **Report the shape honestly** (C12): non-monotone, overshoot stated, but step1000 is *not*
   claimed as a distinct phase.

---

## 3. Open items

| item | state |
|---|---|
| ~~Responsible-use statement~~ | **DONE** — written from F35, merged with the conclusion as §9 |
| ~~Double-blind~~ | **DONE** — `paper.tex`, `neurips_checklist.tex` and `paper/README.md` all anonymised |
| ~~Checklist TODOs~~ | **DONE** — all 4 resolved, incl. an honest LLM-assistance disclosure |
| ~~Page fit~~ | **DONE** — body is **5 pages**, zero body lines on the References page. Guard de-xfailed and hardened twice (gutter filter + no tolerance). |
| Citations | **DONE, then reopened, now automated** (F43 → F50, #71). The hand audit missed a **fourth** wrong title (`ar_tempcrit`). Replaced by `experiments/audit_refs.py` + `tests/test_refs_match_arxiv.py`: 22/22 arXiv entries verified, 0 mismatches, offline test locks it. |
| Style file | `neurips_2025.sty` — 2026 not published yet (404). Swap when it appears; geometry sets the page count |
| Fig sources | `fig_validation_ladder.py`, `fig_developmental.py`, `fig_crosslevel.py` all regenerate from `results/` |

---

## 4. Cut ledger — CLOSED

Body went **13 pages → 5**. The fit is verified by
`tests/test_paper_numbers.py::test_body_fits_the_page_limit`, which locates the References
heading and asserts the body ends before it, and by `test_the_style_file_in_use_is_recorded`,
which pins the `.sty` the guarantee refers to. It is no longer a hand-checked property.

**What was cut**, in the order it went:

| cut | from → to |
|---|---|
| figures | 9 → 2 |
| Related work | 615 → ~180 words, every "we do not claim" disclaimer kept |
| Conclusion | 645 → merged into Responsible use as its closing sentence |
| Reproducibility | body → appendix |
| AR port, crossover | full subsections → one clause |
| "Other readings" | own section → two clauses folded into §2 |
| cross-level | full argument → one paragraph |
| a **duplicated paragraph** | the radius law appeared verbatim twice — never compiled, so never seen |

**What survived deliberately, and must not be cut to buy space:** the DK Part A sentence
(bit-exact, zero mismatching cells, nonzero off-line control); the coupling correction with its
replica-independence justification; the construction-held-fixed argument that licenses the
developmental claim despite F35; the responsible-use section; and the F42 clause stating that
λ statistics exclude unignited runs with *n* stated.

**Space already spent:** C20 went in (issue #44), funded by compressing the cross-level section
from 12 lines to 9 — a *negative* whose full argument lives in `findings.md`. The body still ends
on page 5, asserted by test. There is no further slack identified; the next cut would have to come
out of something on the do-not-cut list, which is a decision, not a trim.

**Live risk:** the fit is a property of `neurips_2025.sty`. The 2026 style is unpublished and its
geometry sets the page count — issue #55, hard stop Aug 26.

## 5. Decisions log

- **2026-07-26** — **The trim landed: body 6 pages → 5 (#62 closed).** Body prose 3555 → 3125
  words. Nothing on the do-not-cut list was touched: DK Part A, the coupling correction,
  construction-held-fixed, responsible use, the F42 *n*-stated clause, the binary-alphabet
  thesis, the methodology clause, the C16 scope sentence, the Nakaishi convergence and the loss
  paragraph all survive.

  **What was actually removed**, as opposed to reworded:
  - the intro's duplicated citation list and its "not a formality" sentence (the latter on the
    author's instruction); `bagnoli1992damage` and `lieb1972finite` were rehomed onto the
    phenomena they name rather than orphaned
  - the two-size equivalence bound (plateau_diff ± CI) — **superseded** by the third lattice
    size, which gives a scaling exponent instead of an interval around zero. Three manifest
    entries retired with it.
  - the repetition-robust "structure" confound clause — the paper's **only** mention of a metric
    no claim uses
  - §3 detail throughout, on the author's instruction to "mention what we got"

  **Figures were the real cost centre, not prose.** Each figure block was ~14 lines; pages 3–4
  held 38–39 lines against 52–53 elsewhere. See [[F54]].

- **2026-07-26** — **F54: both paper figures were defective, and one had shipped unreadable.**
  - `fig_developmental.py` panel A used a doubled backslash-n in a **non-raw f-string**, so
    matplotlib received a literal `\n`. The title rendered as one long line that **overprinted
    panel B's title**. The headline figure was illegible across the middle in a built PDF.
    It was only ever checked as a full-resolution PNG, where it looks fine.
  - Both figures were authored far wider than displayed (13.2in and 14.5in, included at 4.4in),
    so every label reproduced at ~30% of nominal. Figures are now authored at their display
    size: a 7pt label is 7pt on the page.
  - Panel A also hardcoded `±13.6%` and the CI that the trim removed from `paper.tex` — the
    figure would have been the last surviving site of a retired number.
  - The ladder's ECA panel encoded class by **colour alone** ("green ordered / amber edge / red
    chaotic"), unreadable in greyscale, photocopy, or with red-green CVD.
  - Its six panels were numbered (1)–(6) while the caption said "only (3)–(5) are
    weight-bearing" — which points at panels 3,4,5 and **excludes the census**, that §3 calls
    weight-bearing.

  Now: `experiments/figstyle.py`, classic-R monochrome (full box, ticks out, no grid), series
  separated by **marker and dash, never hue** — verified programmatically, max RGB channel
  spread **0** on both. Two panels each rather than three, so each is large enough to read.

- **2026-07-26** — **The page-fit guard was wrong twice, in opposite directions.** It counted
  `pdftotext` output without `-layout`, where the submission style's line-number gutter extracts
  as 43 standalone numerals; the original `> 2` threshold was compensating for that noise by
  tolerance rather than by filtering — and so **also tolerated two real lines of body text**.
  The paper sat at exactly two during the trim, so the guard would have certified a five-page
  fit that did not exist. Now: filter the gutter, then allow **zero** spill. Mutation-tested by
  restoring a cut sentence — it fires with "5 body line(s) spill onto it".


- **2026-07-26** — **F53: the perplexity-proxy objection is answered in the paper, not just in
  the repo (#72).** The transition sits where everything in training changes at once, so
  "λ_ca is an expensive perplexity proxy with a good test suite" is the strongest attack
  available. Held-out loss recomputed at the same 26 (model, checkpoint) pairs:

  | | result |
  |---|---|
  | loss monotone decreasing | **4/4 sizes** |
  | λ_ca overshoot | **3/4** (not 70m, which never crosses) |
  | loss steepest-descent bracket | **(512, 1000) for every size** |
  | λ_ca crossing bracket | none / 128–256 / 256–512 / 256–512 — **moves with size** |
  | Spearman ρ | −0.77 … −0.71, significant in **1 of 4** at n=6 |

  Two arguments, and the second was not anticipated in the pre-registration. **Shape:** a
  non-monotone function of a monotone variable is not a monotone transform of it.
  **Location:** the loss elbow is size-invariant while the λ_ca crossing is not, so λ_ca
  resolves an ordering the loss curve does not express, at a point where the loss curve has no
  feature. The correlation half is explicitly disclaimed in the paper — at n=6 with 1/4
  significant, leaning on ρ would be the weakest available argument.

  Scope shipped with it: shape only, not level, since WikiText-103 is not Pythia's training
  distribution. The Pile version is [[#84]].

  Guarding: the counts (4/4, 3/4) are deliberately **not** manifest literals — a single digit
  matches trivially anywhere in a manuscript, and an entry for "4" would be traceability
  theatre. They are asserted against the results file in
  `test_paper_loss_baseline_claims_match`, which also asserts the steepest-loss bracket is
  size-invariant and that the crossings differ from it. Mutation-tested by perturbing 70m's
  bracket — the guard fires.

  **Cost: +10 body lines.** The trim (#62) is now **37 lines (~480 words)**, deferred by
  explicit instruction ("add to paper, trim later").


- **2026-07-26** — **F52: the temperature scope is a WINDOW, not a point (#73).** The first pass
  ran T ∈ {0.3, 1.1} only, and three points is thin for the paper's most attackable limit. Adding
  T = 0.5 and 0.9 (32 runs, BH-FDR recomputed over the **full** four-temperature family, not
  appended to the old one — a correction that grows by accretion is not a correction):

  | T | ignition pre → plateau | λ_ca pre → plateau | p_BH |
  |---|---|---|---|
  | 0.3 | 0.195 → 0.211 | −0.037 → −0.001 | 0.59 — floor |
  | **0.5** | **0.227 → 0.805** | **−0.050 → +0.183** | **6×10⁻⁴ — survives** |
  | 0.7 | — | −0.019 → +0.179 | (Phase 3 family) |
  | 0.9 | 0.648 → 0.984 | **+0.187** → +0.221 | 0.72 — ceiling |
  | 1.1 | 0.984 → 0.992 | +0.300 → +0.265 | 0.59 — ceiling |

  Two things changed. **(a)** The effect now spans **two adjacent temperatures** (0.5 and 0.7),
  which is a different claim from "at one temperature". **(b)** The ceiling starts at **0.9, not
  1.1** — the paper had been reading 1.1's 0.98→0.99 as the ceiling, but at 0.9 the lattice is
  *already* super-critical before the training being measured (λ = +0.187 at the pre
  checkpoint). The old paragraph was right about the mechanism and wrong about where it begins.

  Ceiling onset between 0.7 and 0.9 sits just below `ar_tempcrit`'s T_c ≈ 1 — stated as
  consistent, not as the same measurement (different sampler; see [[F51]]).

  Pre-registration honoured: #73 said an outcome surviving only at 0.7 would force "at T=0.7"
  rather than "at intermediate temperature". That is not what happened, so the range language
  stands — but it now rests on two surviving temperatures and a mechanism that says where it
  stops, instead of one temperature and an assertion.


- **2026-07-26** — **F50: the hand citation audit was incomplete.** F43 (#37) checked five
  entries by hand and fixed three titles. A fourth wrong title survived it — `ar_tempcrit`
  claimed *"Critical Phase Transition in Large Language Models"*; arXiv says *"Phase transition
  in large language models and the criticality of natural languages"* — and was found only
  because the entry was opened for an unrelated reason. Same shape as #57: fixed as an instance,
  not as a class. Now: `experiments/audit_refs.py` fetches every entry's record (the export API
  was unreachable from this machine — timeouts then 429/503 — so it reads the abstract pages'
  Highwire `citation_*` meta tags), writes `paper/refs_verified.json`, and
  `tests/test_refs_match_arxiv.py` checks refs.bib against it **offline**. Mutation-tested: the
  guard reproduces the exact defect that shipped. **22/22 verified, 0 mismatches.**
  - A suspected malformed `eprint={2101.0}` on `lieb1972finite` was a **false alarm** — an
    artifact of a throwaway dump regex bleeding into the neighbouring entry. It has no eprint,
    correctly, being a 1972 CMP paper.
  - The year check was initially too strict: it flagged `edgeofchaos2024` (ICLR **2025** on a
    2024 preprint) and `simplicitybias` (ICML **2023** on a 2022 preprint). Both are correct as
    written. Only a bib year *preceding* the preprint is impossible, and that is what is now
    asserted — a check that cries wolf trains you to ignore it.

- **2026-07-26** — **F51: read `ar_tempcrit` instead of merely citing it, and it helps twice.**
  Nakaishi, Nishikawa & Hukushima (arXiv:2406.05335) was sitting in the bib as defensive padding
  in a list of prior temperature work. Verified against the source:
  - main-text analysis is **Pythia-160m** (410M–2.8B only in Appendix A); checkpoints
    k = 0, 16, 64, 128, 512, 143000; method is POS-tag correlation + power spectra — **no
    overlap with damage spreading**
  - *"the model begins to acquire nontrivial structures of the natural language around
    k_c ≈ 10²"* → **an orthogonal instrument places the onset where ours does.** Our 160m
    bracket is `[step128, step256]` (`dev_transition_scale.json`). Stated as agreement to
    within a factor of ~2, **not** as a reproduction: both checkpoint grids are coarse near the
    onset (theirs jumps 64 → 128 → 512).
  - *"a phase transition occurs at T_c ≈ 1"* → the temperature paragraph stops apologising. Our
    T=1.1 ceiling (ignition 0.98→0.99) is supercritical and our T=0.3 floor (0.20→0.21) deeply
    sub-critical **as that predicts**. Caveat shipped in the same sentence: their temperature
    parameterises *autoregressive generation*, ours the *in-place lattice update* — same softmax
    knob, different sampler.
  - New guard: `test_the_crossing_brackets_in_prose_match_the_scale_results`. The brackets were
    prose-only until now; the convergence claim makes the 160m one load-bearing, so a re-run
    that moved it would silently turn agreement into contradiction.

- **2026-07-26** — **Originality reframed (#74).** "CA framework in a new field" is the weakest
  claim available and invites a reviewer to go find prior CA-and-LM work — there is plenty, and
  temperature transitions in LLMs are ≥2 years old (`ar_tempcrit`, `critical_temp`). The
  defensible lead is the **calibration move**: damage-spreading implementations are
  conventionally validated against *fitted* critical exponents, which a wrong implementation can
  match for the wrong reasons; a **bit-exact identity cannot be**. One clause added to the intro.

- **2026-07-26** — **C16 scope stated out loud (#76).** The construction-held-fixed argument
  licenses *comparative* claims only. Added: we do not claim any model **is** critical, since
  that would need the ring, radius and D₀ to be principled rather than merely fixed — and they
  are choices. Volunteering the boundary beats having it extracted.

- **2026-07-26** — **The proxy objection is now being tested, not argued (#72).** λ_ca's
  transition sits where everything else in training changes, so "expensive perplexity proxy" is
  the strongest attack on the paper. `experiments/loss_baseline.py` measures held-out loss on
  the *same* 26 (model, checkpoint) pairs — no new lattice runs. Pre-registered: the
  discriminating test is **shape, not correlation** (a non-monotone function of a monotone
  variable is not a monotone transform of it), and **if λ_ca shows no overshoot the objection
  stands and the paper says so**.

- **2026-07-26** — λ_ca replaces D_norm as the headline carrier (C13/C14).
- **2026-07-26** — Abstract leads the DK rung with the bit-exact identity, not the 0.06%
  critical point: at our own ~1% accuracy that number cannot discriminate the two published
  values, so it was advertising the weak half.
- **2026-07-26** — Effect sizes are quoted against the *plateau*, never the step-1000 peak.
- **2026-07-26** — Switched to this file; paper prose deferred until content is settled.
- **2026-07-26** — F42: λ statistics exclude unignited runs; D_norm keeps them (asymmetric by
  design). Applied retroactively; moved N=96's λ numbers, left the ordinal headline untouched.
- **2026-07-26** — Headline restated ordinally (0/48 plateau runs negative) rather than as an
  effect size, because effect sizes depend on where the pre/post line is drawn and two separate
  defects had been found in exactly that choice.
- **2026-07-26** — Cut complete: body fits 5 pages. Conclusion merged into Responsible use.
- **2026-07-26** — N=192 landed (F45). λ_ca intensive across 4×, D_norm 1/N. My pre-registered
  D_norm band missed by 2% because I built it from N=96's observed ratio instead of N=48 and a
  theoretical factor of 2 — the band was mis-constructed, the hypothesis was right. Reported as
  a miss, not retuned.
- **2026-07-26** — Two process defects worth remembering: (a) editing an analysis script while
  its job runs does NOT change that job's analysis — Python had already imported the module, so
  the job wrote a pre-F42 analysis that "downgraded" the size claim as a pure artifact;
  (b) two scripts implemented the F42 asymmetry differently, inflating D_norm's plateau 14%.
