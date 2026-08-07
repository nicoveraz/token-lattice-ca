# TMLR submission — venue decision, policy findings, and the checklist

Created 7 Aug 2026. Records what was verified against the venues' own pages, so the submission is
not built on recollection. Every policy quote below was fetched, not remembered.

---

## 1. The decision

**Target: TMLR (Transactions on Machine Learning Research), as ONE paper subsuming the I4D
workshop submission rather than two papers side by side.**

The subsuming is not a convenience. Claim A + B (see `plan_paper2.md`) cannot stand without the
instrument validation that currently lives in the workshop paper — the Domany–Kinzel bit-exact
calibration, the ECA separation, the Markov recovery. A separate paper 2 would spend its opening
borrowing credibility it cannot display. TMLR has no page limit, so the constraint that forced the
split does not exist there.

**Why TMLR and not a main track.** F95's prior-art gate found real collisions: generic black-box
model identification is taken (Model Equality Testing, ICLR 2025; IRIS), quantization detection is
taken, and the tokenizer-merge mechanism is substantially anticipated. At a novelty-gated venue
those are the reviewer's opening argument. TMLR's second criterion explicitly forbids that move.

---

## 2. TMLR policy, verbatim

Fetched from `jmlr.org/tmlr/` and `jmlr.org/tmlr/editorial-policies.html`, 7 Aug 2026.

**Fees.** *"TMLR imposes no fees or payments to authors, reviewers, action editors, or
editors-in-chief."* Diamond open access: free to publish, free to read.

**Acceptance criteria**, both of which must be met:
1. *"Are the claims made in the submission supported by accurate, convincing and clear evidence?"*
2. *"Would some individuals in TMLR's audience be interested in the findings of this paper?"* — and
   this **must not** be used to reject work for lacking novelty or state-of-the-art results.

**Reuse rule.** *"There should not be any reuse of written text, figures or results between the
submitted paper and any paper which has been published, accepted for publication, or submitted in
parallel at another archival, peer-reviewed venue."* The page also prohibits submissions that are
*"expanded versions of conference papers."*

**The exemption that applies here.** Overlap IS permitted where the prior venue is *"publicly
declared, in writing, to be non-archival, such as workshops, or on preprint servers such as arXiv
and bioRxiv."*

**Review.** Open-reviewing, double-blind; author and reviewer identities withheld. Rolling
submission, shortened review period.

---

## 3. The I4D question, and why the exemption applies

**The workshop paper is SUBMITTED, NOT ACCEPTED.** Deadline 29 Aug 2026; notification 29 Sept 2026;
workshop 12–13 Dec, Atlanta. The tag `camera-ready/neurips26-i4d` is a revised pre-deadline
submission, not a camera-ready — the CFP publishes no camera-ready date. This matters because
TMLR's reuse rule covers work "submitted in parallel", not only published work.

**Two independent sources put I4D on the non-archival side:**

1. NeurIPS 2026 workshop guidance (`neurips.cc/Conferences/2026/WorkshopsGuidance`): all NeurIPS
   workshop papers are non-archival and do not appear in proceedings.
2. This project's own CFP verification, recorded in `plan_to_submission.md` on 26 Jul: I4D
   *"explicitly permits concurrent submission to an archival venue."*

**Consequence:** the TMLR exemption applies, text/figures/results may be reused, and the merged
paper is not an "expanded version of a conference paper" in the prohibited sense.

### 3.1 The written declaration, captured

Fetched from `interpretability4discovery.github.io/cfp.html` on 7 Aug 2026. These are the sentences
TMLR's exemption turns on, quoted so the submission does not depend on the page still existing after
notification:

> **"Non-archival"** (stated as the submission format)
>
> **"Prior non-archival work and work under review elsewhere are welcome. Workshop acceptance does
> not prevent later archival publication."**
>
> **"Submissions undergoing peer review at another venue, including ICLR or NeurIPS, at the paper
> submission deadline are welcome."**

That is an explicit, written, public non-archival declaration plus explicit permission for later
archival publication and for concurrent review. TMLR's exemption applies without needing to fall
back on the NeurIPS-wide guidance, though that guidance agrees.

**Archive the page anyway** (screenshot or Wayback), because it will likely change after the
29 Sept notification and the TMLR prior-work declaration should point at something durable.

### 3.2 Attendance is not required, so the workshop costs nothing to keep

> **"Authors are encouraged, but not required, to attend the workshop in person."**

The workshop is in Atlanta on 12-13 Dec and travel is not possible. That does NOT force a
withdrawal: presentation is by poster with a subset selected for spotlights, and attendance is
explicitly optional. So the submission stays, the 29 Sept notification is free information, and the
only thing forgone is the in-person feedback -- not the acceptance, and not the timestamp.

---

## 4. Checklist

| | Item | State |
|---|---|---|
| 1 | Paper written | **NOT STARTED** — `plan_paper2.md` is a plan, not a draft |
| 2 | `ignition_level` result | **RUNNING** — may amend F104, which is row 4 of the discriminator table |
| 3 | `gatecheck` shippable | Six §9.5 guards all exist as primitives; `pyproject.toml` and `README.md` present; **LICENSE missing** |
| 4 | Anonymised version | Machinery exists (`build_mirror.py`), built for I4D's rules — re-check against TMLR's |
| 5 | Code/data availability | Strong: repo, OSF mirror, per-run result files, provenance stamps |
| 6 | Prior-work declaration | Declare the I4D submission and its non-archival status, with the source |
| 7 | Scope statement | Single-family developmental transition, stated as a boundary rather than implied away |

**Item 3 needs an author decision, not engineering.** A LICENSE file is a rights choice and is not
mine to pick. Without one the package is not usable by a reader, which undercuts Claim B shipping
as a package rather than a narrative.

**Item 2 is the only scientific blocker.** F104 is load-bearing for row 4 of `plan_paper2` §7.1, and
the slope test may reframe it from "revival" to "regression to a common ignition level". Row 4
survives either way — both readings are the instrument responding to a change in the model with the
construction held fixed — but the section cannot be written twice.

---

## 5. What the audit says that is now stale

`critical_analysis.md` §9.5/§9.6 lists the six guard classes as one-offs needing extraction, and
says `gatecheck` is imported by `fingerprint/` and nothing else. Both have moved: all six exist as
primitives in `gatecheck.leverage`, and six experiment scripts now import the package. §9.6 should
be re-read before it is cited as an open item.
