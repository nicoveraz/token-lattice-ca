# Tracker dump -- 2026-07-26T15:49:41Z

| # | state | milestone | labels | title |
|---|---|---|---|---|
| 62 | OPEN | neurips26-submission | blocking,paper | Body is 6 pages, not 5 — and the page-fit guard reported 5 for four commits |
| 61 | OPEN | post-submission | post-paper | A second model family — the prerequisite that makes the emergence hypothesis falsifiable |
| 60 | OPEN | post-submission | post-paper | The calibration ladder is a reusable methodology, not just this paper's step 1 |
| 59 | OPEN | post-submission | post-paper | Diffusion LMs resample in place, so F35's boundary may not apply to them |
| 58 | OPEN | post-submission | post-paper | Variance collapse as an externally computable phase-change signature — test it against known capability emergences |
| 57 | CLOSED | neurips26-submission | evidence | logs/scale.log carried the same stale-verdict defect as #46 — the class was not guarded |
| 56 | CLOSED | neurips26-submission | paper | Re-check the Limitations lattice-size/model-family sentence after the C20 decision |
| 55 | OPEN | neurips26-submission | blocking,paper | Watch for neurips_2026.sty; swap and re-verify the page fit |
| 54 | OPEN | neurips26-submission | blocking,packaging | Insert the anonymised mirror URL into paper.tex and re-tag |
| 53 | OPEN | neurips26-submission | blocking,packaging | Verify the appendix accompaniment promise holds of the mirror |
| 52 | OPEN | neurips26-submission | blocking,packaging | Build the anonymised mirror and scrub identifying strings |
| 51 | OPEN | neurips26-submission | blocking,packaging | Build the submission PDF from a clean worktree of the tag |
| 50 | OPEN | neurips26-submission | blocking,packaging | Cut the submission tag once content is frozen |
| 49 | CLOSED | neurips26-submission | evidence | Regenerate all figures from results/ at the frozen commit |
| 48 | CLOSED | neurips26-submission | blocking,packaging | Assert every number in paper.tex against its results/ file |
| 47 | CLOSED | neurips26-submission | paper | paper/NOTES.md cut ledger is stale |
| 46 | CLOSED | neurips26-submission | blocking,evidence | logs/n192.log carries the stale pre-F42 verdict and contradicts C13 |
| 45 | CLOSED | neurips26-submission | blocking,evidence | Commit or revert the temperature robustness results |
| 44 | CLOSED | neurips26-submission | blocking,paper | C20: decide whether the 4-size scale replication enters the body |
| 43 | CLOSED | - |  | Three analysis defects in dev_transition_scale.py: lexicographic step sort, impossible Spearman p, conflated verdict |
| 42 | CLOSED | - |  | A regex edit destroyed 16 refs.bib entries and a PDF shipped with [???] in the introduction |
| 41 | CLOSED | - |  | Page fit depends on neurips_2025.sty; the 2026 style is unpublished and can undo the cut |
| 40 | OPEN | post-submission | post-paper | pythia-1b has a non-monotone early lambda dip (step256 = -0.546, below its own step128) — unexplained |
| 39 | CLOSED | - |  | N and B are confounded by design (N·B held at 768) — one significant result was a batch-size artifact |
| 38 | CLOSED | - |  | Editing an analysis script mid-run does not change that job's analysis — it wrote two wrong results files |
| 37 | CLOSED | - |  | Three citations were carrying invented titles; bibliography printed "Title/authors to verify" |
| 36 | CLOSED | - |  | lambda_ca is emitted for runs where damage never ignited, and it is uninterpretable there |
| 35 | CLOSED | - |  | Developmental headline computed on step256 alone against a pre-registration of {256,512} |
| 34 | CLOSED | - |  | D_norm alternative floors are unrun on the LM backends |
| 33 | CLOSED | - |  | Double-blind violations in paper, checklist and paper README |
| 32 | CLOSED | - |  | Missing responsible-use statement (automatic desk reject) |
| 31 | CLOSED | - |  | paper.tex had never been compiled and was not in the workshop format |
| 30 | CLOSED | - |  | F35 not absorbed; "persistence length of generation" falsified by our own data |
| 29 | CLOSED | - |  | Ordered-group lambda is an estimator floor sentinel, not a measurement |
| 28 | CLOSED | - |  | Wrong p-value quoted for the ignition-probability claim (0.07 for a P_ignite sentence) |
| 27 | CLOSED | - |  | Retracted ECA three-class ordering asserted paper-wide, and still plotted in two figures |
| 26 | CLOSED | - |  | CRN is the monotone coupling, not the maximal one; "lower bound over couplings" retracted for |V|>2 |
| 25 | OPEN | post-submission | reproducibility,post-paper | Phase 1.5 (deferred): hoist duplication; ca.py mutable globals are a real hazard |
| 24 | CLOSED | Phase 2: validation ladder | rigor | Rebuild the ECA ordered-vs-rest test on ignition probability (F34) |
| 23 | CLOSED | Phase 2: validation ladder | rigor | Phase 2.3: Benettin/QR reference for the CML rung |
| 22 | CLOSED | Phase 2: validation ladder | rigor,methods | Phase 2.2: Domany-Kinzel rung (literature check FIRST) |
| 21 | CLOSED | Phase 4: paper + workshop submission | significance,writing | F35 follow-up: finish the 3-model run and reframe the paper around the delimiting result |
| 20 | OPEN | post-submission | enhancement,significance,methods,post-paper | Assembly theory as a compositional-complexity read-out of LM/CA generation (branch later; extends #13) |
| 19 | CLOSED | Future work (post-submission) | enhancement,significance,methods,scope-closed | New front: activation-lattice information-propagation cone (white-box CA) |
| 18 | CLOSED | - | neurips,writing | Reframe & tighten: drop the capacity claim, own the negative, wire multiplicity (#9) |
| 17 | CLOSED | neurips26-submission | methods,blocking,evidence | Greedy (T->0) deterministic rule vs sampled rule: is criticality a sampling phenomenon? |
| 16 | OPEN | post-submission | validity,methods,post-paper | Full-context vs windowed rule: does cross-level agreement improve with rule richness? |
| 15 | CLOSED | - | significance,rigor | De-confound the within-model cross-level test: correlate along the radius axis at fixed T |
| 14 | CLOSED | Phase 2: validation ladder | rigor,validity,methods | Ground-truth calibration of the criticality instrument via classical CA rules (known Lyapunov) |
| 13 | OPEN | post-submission | enhancement,methods,post-paper | Effective-rule complexity: an algorithmic-complexity axis via CA rule reconstruction (research thread) |
| 12 | CLOSED | Phase 4: paper + workshop submission | neurips,reproducibility | P2: NeurIPS mechanics — class, checklist TODOs, author line |
| 11 | CLOSED | - | neurips,writing | P2: Honest reframe of abstract/intro — demote failed headline, elevate defensible protocol |
| 10 | CLOSED | Phase 3: statistical rigor | scale | P1: Headline is at N=48 only; effect magnitude falls steeply with N |
| 9 | CLOSED | Phase 3: statistical rigor | rigor | P1: No multiple-comparisons correction across the test battery |
| 8 | CLOSED | - | rigor,writing | P1: Crossover "rescue" is single-seed and false at T=0.3 — retract "strengthens at every T" |
| 7 | CLOSED | - | rigor,writing | P1: v∝r is presented as more law-like than the data support (finite-size lift is N/4 clipping) |
| 6 | OPEN | post-submission | rigor,validity,post-paper | P1: Real-model census recovery is near-floor against the wrong corpus |
| 5 | CLOSED | - | rigor,validity | P1: AR "consistent joint" argument is overstated; AR seeds pool a bimodal T distribution |
| 4 | CLOSED | - | significance,neurips | P1 (flagship): Cross-level validation — black-box damping length vs white-box activation criticality |
| 3 | CLOSED | - | rigor | P0: λ "model-invariant" / kinematics⊥stability is cherry-picked at one cell — retract or rebuild |
| 2 | CLOSED | Phase 3: statistical rigor | rigor,validity | P0: D_norm normalization can manufacture the monotone rise and the ">1" signature — ablate it |
| 1 | CLOSED | Phase 3: statistical rigor | significance,rigor | P0: Capacity→sensitivity claim is pseudoreplicated (n=2 seeds) — rebuild the headline statistic |
