> **WITHDRAWN — historical record.** The NeurIPS 2026 I4D submission this document plans for was withdrawn by author decision on 7 August 2026. It is kept because the reasoning and the constraints it records are real; nothing in it describes a live plan. The live manuscript is `paper_arxiv/main.tex`, targeting TMLR.

# Plan to submission — NeurIPS 2026 Interp4Discovery

**Deadline: Aug 29 2026, 11:59 PM AOE** (= Aug 30, 11:59 UTC). Today is Jul 31. **29 days.**
Non-archival, double-blind, 5 pages main text (references and appendices excluded), 6 at
camera-ready. Reviewers are explicitly instructed to weigh reproducibility and the
availability of code and data.

**Verified against the CFP on Jul 26** (`interpretability4discovery.github.io/cfp/`). Every
assumption above holds verbatim, including *"Failure cases and negative results are welcome."*
Notification Sept 29; workshop Dec 12 or 13, Atlanta; no camera-ready date published yet.
Portal: `openreview.net/group?id=NeurIPS.cc/2026/Workshop/Interp4Discovery`. Four things the
CFP says that this plan did not previously account for are folded in below: the 2026 style
file **now exists** (Gate D collapses), the societal-impact statement is the one stated
desk-reject trigger, the anonymization instruction names GitHub and Hugging Face usernames
specifically, and concurrent submission to an archival venue is explicitly permitted.

---

## 0. Post-submission state (added Jul 31) — and a decision the deadline still allows

The paper is submitted and pinned at `submission/neurips26-i4d`. **The deadline has not passed**:
29 days remain, so the submission can still be revised on OpenReview if that is wanted.

Work since the tag produced one result the paper does not contain and that would strengthen it:

- **F58 — the transition is a genuine critical point, not a finite-size crossover.** The survival
  exponent δ and the active-count exponent θ reach their directed-percolation values at a *common*
  temperature, T_c ∈ [0.4343, 0.4391]. Robust to the fit window at both ends and to boundary
  saturation, on an estimator gated against Domany–Kinzel *before* the LM numbers were read.
  This speaks directly to **F12**, which the paper currently reports as "the temperature
  transition is a finite-size crossover" — F58 does not contradict F12 (different quantity: F12 is
  the static susceptibility, F58 the damage-spreading transition) but a reader will ask, and the
  paper should answer before a reviewer does.
- **F59 — z is estimated near 1.35, below DP's 1.5807, but cannot be separated from it.** Not
  paper material yet; the exclusion did not survive adding a fourth lattice.

**UPDATE (Aug 1): the F58 question above is closed, and not by judgement.** F62–F66 showed the
transition is an out-of-distribution prompt artifact — only at r=2, carried by one token, absent
from the masked-LM construction. There is no F58 to fold in. The F12/F58 tension this section
worried a reviewer would probe does not exist either: F12 reports a finite-size crossover in the
static susceptibility, and what F58 found was a property of the probe. Leave the submission alone.

**DECIDED (Jul 31): the submission is left alone.** The work below is closed out; the next
paper is planned in [`plan_paper2.md`](plan_paper2.md).

**Recommendation as written at the time: do not reopen the paper for F59, and treat F58 as
a judgement call.** F58 is
solid and genuinely interesting, but folding it in means new prose, probably a figure, and a fresh
pass over the 5-page limit that took real effort to hit — against a non-archival venue where the
work can also just appear in the extended version. The safe default is to leave the submission
alone and put F56–F59 in the post-submission track. If it *is* reopened, the F12/F58 distinction
above is the one thing that must be stated explicitly.

Three retractions happened in this window (F56, F57, F59's first pass), all of the instrument
rather than the physics. None touch the submitted paper: `dev_transition_phase3.measure` seeds a
3-site block, which is far less exposed to F57's healing mechanism, and it already declares *"run
is the unit of analysis, not the lattice."*

---

## 1. How far — the honest read

The paper is **done as a document**. What remains is one content decision, three consistency
defects, packaging, and one external dependency outside your control.

Verified state as of `fdfa748`:

| | |
|---|---|
| `paper.pdf` | builds, **10 pages: body 1–5, refs 6, appendix 7, checklist 8–10** |
| Body | **fits the 5-page limit** |
| Build health | 0 undefined refs, 0 undefined citations, 0 overfull boxes |
| Compliance | responsible-use §9 written, double-blind clean, all 4 checklist TODOs resolved |
| Citations | all 5 unverified entries verified (F43); 3 titles were wrong and were fixed |
| Claim inventory | C1–C20, every number sourced to a `results/` path |
| Tags | **none** |

Remaining effort is roughly **3 focused working days**, spread over 34 calendar days
because of the style-file wait. The schedule risk is not running out of time. It is
**scope creep** — the temptation to run one more experiment — and the 2026 style file
landing late and breaking the page fit.

---

## 2. The one content decision: C20

**C20 is in the claim inventory and not in the paper.** `results/dev_transition_scale.json`
exists (49 KB, F46 complete at 192/192) and NOTES.md records: *replicates in 4/4 sizes
(p_BH ≤ 0.015); crossing 70m <128, 160m 128→256, 410m & 1b 256→512 (saturates); plateau level
non-monotone 0.162/0.164/0.174/0.166 → no capacity axis.*

`paper.tex` never states it. The only match for a model size in the body is the appendix's
`Pythia-14M--1B` in the model list. §4 still opens *"Applied across Pythia-410m checkpoints"*
and Limitations still says the result *"rests on one model family."*

**Recommendation: put it in.** It answers the first question a reviewer will ask — *does this
depend on the one model you happened to pick?* — and the answer is a 4/4 replication with a
timing shift that saturates. It also converts the "no capacity axis" negative into a stated
result rather than an unpublished null, which fits the paper's own thesis about reporting what
does not work. Cost is about six lines in §4 plus one clause in Limitations.

**Where the space comes from:** §6 (cross-level) is 12 lines for a negative whose full argument
lives in `findings.md`. NOTES.md's cut ledger already names this as cut candidate #1 — note
that ledger's section numbers are stale (it says "§7 cross-level"), which is its own issue.

**If you decide against it,** that is defensible on space grounds, but then Limitations must
say the scale replication exists and is held back, not stay silent — an unreported 4/4
replication sitting in the repo a reviewer can read is worse than either alternative.

---

## 3. Three defects found in the repo

**D1 — `logs/n192.log` contradicts the paper.** Its tail reads
`lambda_ca = 0.0153 -> lambda_ca OUTSIDE the predicted interval -- size-robustness DOWNGRADED
to 48->96`. C13/C13b claim λ_ca = 0.160 at N=192 and 95% retention. Your own decisions log
explains the discrepancy — the job's Python had already imported the analysis module, so it
wrote a pre-F42 analysis — and `results/dev_transition_n192.json` is correct. But the stale
verdict is committed, and a reviewer told to check data availability will grep the logs. Fix by
re-running the analysis step over the existing JSON and letting it rewrite the log, or by
appending a machine-written superseding block. Do not hand-edit the verdict line.

**D2 — the temperature robustness results are uncommitted.** `results/dev_transition_temp.json`
is untracked and `logs/temp.log` is modified, while the code landed at `fdfa748` ("add the
temperature robustness check (#17)"). Either commit them or revert them; an uncommitted
results file at tag time is the exact thing the tag is supposed to prevent.

**D3 — NOTES.md §4 is stale.** It says *"Remaining ~1 page to find"* and uses the pre-cut
section numbering, while both the file header and §3 say the page fit is done. It is the one
place in the repo that still describes the paper as over-length.

---

## 4. Phased plan with hard gates

### Gate 0 — style swap · **do this first, before Gate A**

New as of the CFP check. The 2026 template exists and is linked from the CFP:
`https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`.
It must move ahead of the content freeze rather than behind it, because **geometry sets the
page budget and the page budget is an input to the C20 in/out decision at Gate A.** Deciding
C20 against the 2025 budget and then discovering the 2026 budget is different means deciding
twice.

- Download the official ZIP and diff `neurips_2026.sty` against `neurips_2025.sty`.
- Switch to `\usepackage[dblblindworkshop]{neurips_2026}`. The 2026 style adds workshop
  options that 2025 did not have (`dblblindworkshop`, `sglblindworkshop`); this venue is
  double-blind, so the first is correct. Bare `\usepackage{neurips_2026}` still means
  anonymous-submission mode but defaults to the **main track**, whose footer names the
  conference and not the workshop.
- Set `\workshoptitle{Interpretability for Discovery}` in the preamble. A third-party mirror
  of the 2026 style shows the workshop options reading a `\workshoptitle` macro that defaults
  to empty and emits a package warning — not an error — when unset, which is the kind of
  defect that ships silently. **Verify this against the official ZIP**; the mirror advertises
  itself as a modified convenience version, so treat it as a lead, not a source.
- Re-verify the 5-page fit immediately. The same mirror shows `textwidth=5.5in`,
  `textheight=9in`, letterpaper — unchanged from 2025 — so the fit is likely to survive, but
  confirm rather than assume, and confirm from the official file.

### Gate A — content freeze · target **Fri Aug 7**

- Decide C20 in or out; if in, write it and adjust Limitations. Decide against the **2026**
  page budget, not the 2025 one.
- Commit or revert the temperature results (D2).
- Any experiment that could change a number in the paper: decide now or never.
- Confirm §9 satisfies the CFP's desk-reject criterion. Verbatim: *"Every submission must
  include a short statement covering potential societal impacts and suggested mitigations.
  A missing statement is grounds for desk rejection."* This is the **only** stated
  desk-rejection trigger in the CFP. §9 is written, but it must cover **both** halves —
  impacts *and* mitigations. A statement naming risks with no mitigations is half a statement.

**Gate condition: after Aug 7, nothing that writes to `results/` runs.** That is the whole
point of the date. The scale sweep is complete, N=192 has landed, and nothing else is queued —
so this costs you nothing today, but it will feel expensive on Aug 20 when a new idea arrives.

### Gate B — consistency · **Aug 8–14**

- Fix D1 (stale n192 log) and D3 (stale cut ledger).
- Extend `tests/test_results_self_consistency.py` so every number printed in `paper.tex` is
  asserted against its `results/` file. The appendix already *promises* this
  ("every number in the paper is traceable to a result file") — make it a test rather than a
  claim.
- Regenerate all figures from `results/` at the frozen commit; confirm they change nothing.
- Full suite green.

### Gate C — pin and package · **Aug 15–21**

- Tag the submission commit. Suggested: `submission/neurips26-i4d`.
- Build the PDF from a clean worktree of the tag —
  `git worktree add /tmp/paper-build submission/neurips26-i4d` — never from the working tree.
- `git archive` the tag into the anonymized mirror. `.gitignore` already excludes `.venv/`
  and `hf_cache/`, and `git archive` carries no commit history or author emails, so most of
  the anonymization is free.
- **Decide the anonymous-hosting mechanism.** The CFP recommends `anonymous.4open.science`
  for GitHub repositories, an anonymous Hugging Face account for large files, and an
  anonymous site or Streamlit app for demos. That is a *different mechanism* from the
  `git archive` static mirror this plan assumes — Anonymous GitHub is a proxy over a live
  GitHub repo, not a ZIP upload, and reaching a private repo means granting it a token with
  `repo` scope. Neither choice is wrong; they produce different URLs, so pick before the URL
  goes into `paper.tex`. The static mirror keeps you in control of exactly what ships, which
  suits a repo whose logs contain absolute home-directory paths.
- **Scrub before upload.** This is not hypothetical: `logs/n192.log` contains
  `/Users/nicoveraz/Documents/GitHub/textca/results/...`. Grep the archive for your name,
  your email, `/Users/`, and the org. `logs/` and machine-written JSON are the leak paths.
- **Grep the manuscript too, not just the archive.** The CFP is specific: *"Before
  submission, search the manuscript for the names, GitHub usernames, and Hugging Face
  usernames of all core contributors."* The archive scrub above does not cover `paper.tex`,
  and the mirror URL inserted at this gate is itself a candidate leak if it embeds a
  username. `nicoveraz` covers the GitHub handle and the `/Users/` path with one string; the
  Hugging Face handle is a separate string and needs its own pass.
- Verify the appendix's promise is true of the mirror: all code, all per-run result files,
  all figure scripts actually present. Pack size is 43.89 MiB, so this is shippable.
- Insert the mirror URL into `paper.tex`, then re-tag. The mirror and the tagged tree will
  differ by exactly that one line, which is normal and worth a sentence in the README.

### Gate D — ~~style-file watch~~ **RESOLVED, folded into Gate 0**

Superseded by the CFP check. The earlier 404 was a stale URL, not an unpublished file: the
2026 template is live and linked directly from the CFP, and `neurips_2026.sty` is dated
2026-01-29. There is nothing to watch weekly and no Aug 26 hard stop.

**This removes the plan's only external dependency.** The stated reason the schedule finished
a week early no longer exists. Keep the margin anyway — Gate E stays Aug 27 — but understand
that the margin is now buying insurance against *your* slippage, not against NeurIPS's, which
is a weaker reason to hold three weeks of slack. If the style swap lands cleanly in Gate 0 and
Gates A–C run on time, the paper is submittable well before Aug 27, and the honest question
becomes what to do with the recovered time rather than whether the schedule holds.

Residual risk, small: the organizers could post a workshop-specific template later. Unlikely —
the CFP links the standard NeurIPS 2026 ZIP and the style file already carries a
`dblblindworkshop` option, which is the mechanism by which NeurIPS 2026 handles workshops.
Glance at the CFP page once in mid-August; do not build a gate around it.

### Gate E — submit · **Thu Aug 27**, not Aug 29

Two days of margin. AOE means the true cutoff is Aug 30 11:59 UTC, but a submission portal
under load on the final evening is a solved problem you should not re-solve.

---

## 5. What not to do

**Do not start a second model family.** Limitations names it as the obvious next requirement,
and that is the correct place for it. Pythia is the family with public intermediate
checkpoints; a second one means OLMo or LLM360, which on an M1 16 GB is not a 34-day job. A
half-finished second family is strictly worse than a clearly stated limitation.

**Do not resume any experiment after Gate A**, including anything that looks like a two-hour
check. The failure mode is not the experiment, it is that its output lands in `results/` after
the numbers were quoted.

**Do not re-tune a band, threshold or window to make a number nicer.** The N=192 D_norm band
missed by 2% and NOTES.md records it as a miss rather than a retune. That decision is worth
more to this paper than the 2%.

**Do not touch the retraction list.** The C18 coupling correction in particular corrects a
claim that was public in the repo; NOTES.md already flags it as compress-never-delete.

---

## 6. Post-submission

Everything below moves to the `post-submission` milestone and waits for the tag. `main` keeps
moving after the tag — the tag is what guarantees the paper's state, so a long-lived parallel
branch would only add a second place to apply every fix. If the rebuttal or camera-ready needs
paper-only edits, cut that branch **then**, from the tag.

Already tracked and correctly deferred: the real-corpus census (#6), the compositional-
complexity axis (#13, #20), the activation-lattice cone (#19 — object already taken by
arXiv:2605.25225; cite, do not claim), Phase 1.5 duplication hoisting and the `ca.DATA_DIR`
mutable global.

The camera-ready page (5 → 6) is the natural home for whichever of C20 or §6 loses the space
argument in §2 above. CFP wording: *"Accepted papers may use one additional main-text page in
the camera-ready version, allowing up to six main-text pages to integrate reviewer feedback."*
**No camera-ready date is published** — only notification (Sept 29) and the workshop (Dec 12
or 13). Assume a deadline somewhere in Oct–Nov and do not schedule against it yet.

**Concurrent archival submission is explicitly permitted.** The CFP welcomes submissions
*"undergoing peer review at another venue... at the paper submission deadline"* and work
*"previously submitted to or accepted at a non-archival venue."* This changes the strategic
read: submitting here does not spend the archival option. The earlier verdict — strong fit for
a non-archival workshop, not yet archival-ready — described two sequential states, and the
venue rules make them concurrent. The workshop can be the venue that generates the reviews
that make the archival version ready, rather than a consolation for not being ready.

Accepted papers are presented as posters, *"with a subset selected for oral or spotlight
talks."*
