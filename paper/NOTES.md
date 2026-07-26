# Paper working notes

The place to think about the paper. **Add and modify findings here; write prose later.**

Rationale: iteratively micro-editing `paper.tex` to hit a page count mixes two different
jobs — deciding *what is true and what gets claimed*, and *fitting it on 5 pages*. This file
holds the first. The `.tex` gets assembled from it in one pass when the content is settled.

Status: `paper.pdf` builds (`tectonic -X compile paper.tex`), **11 pages: body 1–6, refs 6–7,
appendix 8, checklist 9–11**. Body is ~1 page over. 0 undefined refs, 0 overfull boxes.

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
| **Responsible-use statement** | **absent — automatic desk reject.** ~110 words. F35 gives the real sentence: the instrument characterises a resampling construction, so it must not be cited as evidence about deployed-model robustness |
| Double-blind | `paper.tex` author block is gone, but check the tree: `paper/README.md:3` names the repo; checklist `:26`, `:31`, `:66` |
| Checklist TODOs | 4 real ones at `neurips_checklist.tex` lines 31, 56, 66, 85 |
| Style file | `neurips_2025.sty` — 2026 not published yet (404). Swap when it appears; geometry sets the page count |
| Page fit | body ~1 page over. See §4 |
| Fig sources | `fig_validation_ladder.py`, `fig_developmental.py`, `fig_crosslevel.py` all regenerate from `results/` |

---

## 4. Cut ledger

Body went 13 pages → ~6. Done: dropped 7 of 9 figures; Related work 615→200 words;
Conclusion 645→130; Reproducibility → appendix; AR port and crossover sections cut to a
clause; **removed a verbatim duplicated paragraph** (the radius law appeared twice, lines
253–270 of the old file).

Remaining ~1 page to find, in preference order:

1. §7 cross-level → 3 sentences. It is a *negative* and does not need its full argument;
   the retraction detail lives in `findings.md`.
2. §6 "Other readings" → delete entirely, fold the radius law into §2 as one clause. The
   light cone is already in the ladder figure's regime discussion.
3. Merge §2 (instrument) into §1's last paragraph.
4. Figures to `0.9\linewidth`.

**Do not cut:** C1's sentence, the C18 coupling correction (compress, never disappear — it
corrects a claim that was public in the repo), the construction-held-fixed argument,
the responsible-use statement once written.

---

## 5. Decisions log

- **2026-07-26** — λ_ca replaces D_norm as the headline carrier (C13/C14).
- **2026-07-26** — Abstract leads the DK rung with the bit-exact identity, not the 0.06%
  critical point: at our own ~1% accuracy that number cannot discriminate the two published
  values, so it was advertising the weak half.
- **2026-07-26** — Effect sizes are quoted against the *plateau*, never the step-1000 peak.
- **2026-07-26** — Switched to this file; paper prose deferred until content is settled.
