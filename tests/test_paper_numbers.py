"""The paper must not contradict the result files it is derived from.

Rule 2 says never report a number not traceable to `results/`. Rule 8 says assert code against
its declared design. This is the same idea applied to the manuscript: every load-bearing number
in `paper/paper.tex` is checked here against the JSON it came from.

Why this exists. Three separate defects in this project were *numbers in prose drifting from
numbers in files* --- the retracted ECA ordering (F33/F36), the step256-only headline (F39), and
the effect sizes that moved again when F42's ignition rule was applied retroactively. Each time
the fix was a manual grep across five to seven files, and each time the risk was that one site
was missed. A test is cheaper than a grep and cannot forget.

If one of these fails, the paper is wrong OR the analysis was re-run and the paper has not caught
up. Either way the answer is to reconcile them, never to relax the tolerance.
"""
import json, pathlib, re
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "paper.tex"
RESULTS = ROOT / "results"


def _tex():
    if not TEX.exists():
        pytest.skip("paper.tex not present")
    return TEX.read_text()


def _load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    return json.load(open(p))


def _has(tex, *fragments):
    """All fragments must appear somewhere in the manuscript."""
    missing = [f for f in fragments if f not in tex]
    return missing


# ------------------------------------------------------------------ the DK rung (F38)
def test_paper_dk_numbers_match_dk_calib():
    tex, dk = _tex(), _load("dk_calib.json")
    pa = dk["part_a_exact_identity"]
    assert pa["max_mismatch"] == 0, "the exact identity no longer holds; the paper's claim is void"
    assert pa["control_offline_mismatch"] > 0, "the off-line control is zero -- test is vacuous"
    # the paper must state both the ring/steps and the control, or the claim is unfalsifiable
    missing = _has(tex, str(pa["N"]), str(pa["steps"]), "zero mismatching cells")
    assert not missing, f"paper omits DK Part A specifics: {missing}"
    assert str(pa["control_offline_mismatch"]) in tex, (
        f"paper does not state the off-line control count ({pa['control_offline_mismatch']}); "
        f"without it the bit-exact claim cannot be judged")


def test_paper_dk_critical_points_match():
    tex, dk = _tex(), _load("dk_calib.json")
    cal = dk["calibration"]
    for name, key in (("site DP", "site DP"), ("W18 activity", "W18 activity"),
                      ("W18 damage", "W18 damage")):
        est = cal[key]["estimate"]
        assert f"{est:.4f}" in tex, f"paper is missing the {name} estimate {est:.4f}"


# ------------------------------------------------------------- the developmental claim (F39/F42)
def test_paper_headline_is_stated_ordinally_and_matches():
    """The headline is sign agreement over all runs, not an effect size."""
    tex, sh = _tex(), _load("dev_transition_shape.json")
    pooled = sh["sign_agreement"]["pooled"]
    assert pooled["negative"] == 0, "a plateau run went negative; the headline claim is void"
    assert str(pooled["n"]) in tex, f"paper omits the plateau run count ({pooled['n']})"
    for N in (48, 96):
        rec = sh["sign_agreement"][f"N{N}"]
        frac = f"{rec['pre_negative']}/{rec['n_pre']}"
        assert frac in tex, (
            f"paper omits or misstates the N={N} pre-transition sign fraction {frac}")


def test_paper_effect_sizes_match_the_pre_registered_basis():
    tex, sh = _tex(), _load("dev_transition_shape.json")
    for N in (48, 96):
        h = sh["headline"][f"N{N}_lambda_ca"]
        assert f"{h['cohens_d']:.2f}" in tex, (
            f"paper's N={N} lambda effect size does not match {h['cohens_d']:.2f}")
        # and the UNREGISTERED variants must NOT appear as if they were the result
        bad = h["cohens_d_from_step256_only_UNREGISTERED"]
        assert f"$d{{=}}{bad:.2f}$" not in tex, (
            f"paper quotes the step256-only effect size {bad:.2f} -- this is F39's defect")


def test_paper_states_the_ignition_exclusion():
    """F42: if lambda stats exclude runs, the paper must say so and give the reduced n."""
    tex, sh = _tex(), _load("dev_transition_shape.json")
    total_dead = sum(v["n_unignited"] for v in sh["ignition"].values())
    if total_dead == 0:
        pytest.skip("no unignited runs to disclose")
    n_pre_96 = sh["headline"]["N96_lambda_ca"]["n_pre"]
    assert str(n_pre_96) in tex, (
        f"lambda statistics exclude {total_dead} unignited run(s) but the paper never states the "
        f"reduced n ({n_pre_96}); an unexplained n is how F39 happened")
    assert "ignit" in tex.lower(), "paper never mentions ignition despite excluding on it"


def test_paper_size_agreement_is_a_bound_not_a_null_p_value():
    tex, sh = _tex(), _load("dev_transition_shape.json")
    w9 = sh["size_scaling_W9"]["lambda_ca"]
    assert f"{w9['plateau_agree_within_pct']:.0f}" in tex, (
        "paper omits the +-% equivalence bound on the cross-size plateau agreement")
    for form in ("$p{=}0.91$", "p=0.91"):
        assert form not in tex, (
            "paper leans on p=0.91 -- absence of evidence is not evidence of equality; quote "
            "the CI instead. (Matched precisely: 0.91 also appears as the census baseline.)")


# ------------------------------------------------------------------ the ECA rung (F36)
def test_paper_eca_numbers_match_and_the_retraction_holds():
    tex, ign = _tex(), _load("eca_ordered_vs_rest.json")
    t = ign["tests"]
    d = t["cohens_d_ordered_vs_rest"]
    assert (f"{d}" in tex) or (f"{d:.2f}" in tex), (
        f"paper's ECA effect size matches neither {d} nor its 2dp rounding")
    assert f"{t['edge_lt_chaotic_p']:.3f}" in tex, (
        "paper omits the edge-vs-chaotic p-value that kills the three-class ordering")
    for bad in ("0.0665", "$p{=}0.07$"):
        assert bad not in tex, f"paper quotes the superseded ECA p-value {bad}"
    assert "ordered$<$edge" not in tex and "ordered<edge" not in tex, (
        "the retracted three-class ordering is asserted again")


# ------------------------------------------------------------------ F35 and the coupling (F41)
def test_paper_states_the_delimiting_result_and_never_calls_it_generation():
    tex, rg = _tex(), _load("real_generation_damage.json")
    persist = {k: v["P_persist"] for k, v in rg.items() if isinstance(v, dict)}
    assert set(persist.values()) == {1.0}, f"P_persist is no longer 1.000 everywhere: {persist}"
    assert "1.000" in tex, "paper omits P_persist=1.000"
    assert "of generation" not in tex, (
        "the falsified attribution 'damping length of generation' is back (F35)")


def test_paper_keeps_the_coupling_correction():
    """It corrects a claim that was public in the repo; it may be compressed, never removed."""
    tex = _tex()
    missing = _has(tex, "monotone", "replica-independence")
    assert not missing, f"the F41 coupling correction has been cut: missing {missing}"


def test_paper_keeps_the_construction_held_fixed_argument():
    """This is what licenses the developmental claim despite F35.

    Matched on whitespace-normalised text. The earlier version compared raw substrings and so
    failed the moment a reflow put a line break inside the phrase -- reporting the argument as
    "gone" when only its typography had moved. A guard that fires on rewrapping trains you to
    ignore it, which is the failure mode the citation-year check already had.
    """
    flat = " ".join(_tex().split())
    assert "held \\emph{fixed} across" in flat or "held fixed across" in flat, (
        "the construction-held-fixed argument is gone; without it F35 undercuts the headline")


# ------------------------------------------------------------------ submission hygiene
def test_every_cited_key_resolves():
    """The guard that would have caught a destroyed refs.bib.

    A regex edit to one entry once matched far past its end and deleted 16 others; the build
    still "succeeded" and the PDF shipped with [???] in the introduction. tectonic does not
    fail on unresolved citations, and the log check I was using did not catch it either. This
    compares the cite keys in the manuscript against the entry keys in the database directly.
    """
    import re as _re
    tex = _tex()
    bib = ROOT / "paper" / "refs.bib"
    if not bib.exists():
        pytest.skip("refs.bib not present")
    cited = set()
    for m in _re.finditer(r"\\cite[tp]?\{([^}]*)\}", tex):
        cited.update(k.strip() for k in m.group(1).split(","))
    defined = set(_re.findall(r"@[a-zA-Z]+\{([A-Za-z_0-9]+),", bib.read_text()))
    missing = sorted(cited - defined)
    assert not missing, (
        f"{len(missing)} cited key(s) have no bibliography entry and will render as [?]: "
        f"{missing}")


def test_pdf_has_no_unresolved_citation_marks():
    """Belt and braces: read the built PDF and look for the marks themselves."""
    import subprocess
    pdf = ROOT / "paper" / "paper.pdf"
    if not pdf.exists():
        pytest.skip("paper.pdf not built")
    try:
        txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                             text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("pdftotext unavailable")
    for mark in ("[???]", "[?]"):
        assert mark not in txt, f"the built PDF contains {mark} -- an unresolved citation"


def test_no_unverified_citations_reach_the_bibliography():
    """`plainnat` prints note= fields; 'to verify' must never be printable (F43)."""
    bib = ROOT / "paper" / "refs.bib"
    if not bib.exists():
        pytest.skip("refs.bib not present")
    s = bib.read_text()
    for marker in ("to verify", "TODO", "FIXME", "XXX", "verify author"):
        assert marker not in s, f"refs.bib still contains an unverified entry marker: {marker!r}"
    # plainnat PRINTS note= fields, so any working annotation ends up in the reference list.
    # F43 found five such entries; two more were working notes about the cited work itself.
    assert "note={" not in s, (
        "refs.bib has a note= field. plainnat prints it into the bibliography, so working "
        "annotations become visible to reviewers -- move the content elsewhere or delete it.")


def _identifying_strings():
    """The identifiers to forbid, DERIVED at runtime rather than written down here.

    The previous version hard-coded the author's GitHub handle and repo name as a literal list,
    with a docstring claiming "a reviewer reading a test that forbids an identifier learns
    nothing identifying from it". That argument is simply wrong -- the list *is* the identifier,
    and this file ships inside the anonymised mirror, so the guard against de-anonymisation was
    itself a de-anonymisation. Deriving them from the git remote and the checkout path keeps the
    guard exact where it matters (pre-tag, on a real clone) and leaves nothing to read in the
    mirror, where there is no .git and the test simply skips.
    """
    import subprocess
    out = set()
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"],
                             cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        url = ""
    m = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        out.update({m.group(1), m.group(2)})
    # the checkout path carries the OS username on a personal machine
    parts = ROOT.resolve().parts
    if "Users" in parts:
        out.add(parts[parts.index("Users") + 1])
    elif "home" in parts:
        out.add(parts[parts.index("home") + 1])
    return {s for s in out if s and len(s) > 3}


def test_no_self_identifying_strings_in_the_submission():
    """paper.tex must not contain the author's handle, repo name, or a hosting URL.

    Identifiers come from `_identifying_strings()`, which derives them rather than listing them
    -- see that docstring. The generic hosting domains stay literal: they identify no one.
    """
    tex = _tex()
    derived = _identifying_strings()
    if not derived:
        pytest.skip("no git remote and no user-bearing checkout path; nothing to derive")
    for s in derived | {"github.com", "huggingface.co"}:
        assert s not in tex, f"double-blind violation in paper.tex: {s!r}"


def test_responsible_use_section_exists():
    tex = _tex()
    assert re.search(r"\\section\{[^}]*Responsible use", tex), (
        "no responsible-use section -- automatic desk reject at this venue")


# ------------------------------------------------------------------ issue #41: the page limit
# Camera-ready limit, held at the SUBMISSION number by choice. Interp4Discovery's camera-ready
# allows 6 body pages ("one additional main-text page ... to integrate reviewer feedback",
# verified against the live CFP 3 Aug 2026); the restructure was brought back to 5 so that the
# venue's extra page stays in reserve for integrating actual reviewer feedback rather than being
# spent before reviews exist. If feedback needs the space, raising this to 6 is sanctioned by
# the CFP -- raising it beyond 6 never is.
BODY_PAGE_LIMIT = 5


def _pdf_pages_text():
    import subprocess
    pdf = ROOT / "paper" / "paper.pdf"
    if not pdf.exists():
        pytest.skip("paper.pdf not built")
    try:
        txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                             text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("pdftotext unavailable")
    return txt.split("\f")


def test_body_fits_the_page_limit():
    """The body must end within the venue's page limit.

    This is the submission's hardest constraint and nothing asserted it: the fit was verified by
    hand after each edit, and it is a property of the style file's geometry. That is now the
    official neurips_2026.sty, whose \newgeometry block is byte-identical to the 2025 one, so
    the swap did not move the budget -- and test_the_style_file_in_use_is_recorded pins those six
    values, so a future style that DOES move them fails loudly instead of silently undoing a cut
    that took the body from 13 pages to 5. References, appendix and checklist are excluded from
    the limit, so the test locates where the bibliography starts.
    """
    import re as _re
    pages = _pdf_pages_text()
    # The submission style prefixes every line with a line number, so "References" is not the
    # first token on its page. Match it as a standalone heading line instead.
    heading = _re.compile(r"^\s*\d*\s*References\s*$", _re.M)
    refs_page = next((i + 1 for i, p in enumerate(pages) if heading.search(p)), None)
    assert refs_page is not None, "could not locate the References heading in the built PDF"
    # The body does NOT necessarily end where References begins. If References starts partway
    # down its page, body text occupies that page too, and counting refs_page-1 silently
    # under-reports by one -- which is exactly how this test passed while the body had spilled.
    refs_text = pages[refs_page - 1]
    before = refs_text[:heading.search(refs_text).start()]
    # Any line of real body text on the References page means the body reached that page.
    #
    # Two things had to be fixed here. The submission style numbers every line, and without
    # -layout pdftotext emits that gutter as standalone numerals -- 43 of them on this page --
    # so a naive count sees "43 spilling lines" on a body that ends cleanly. The original guard
    # compensated with a `> 2` threshold rather than by filtering, which meant it ALSO tolerated
    # two real lines of body text; during this trim the paper sat at exactly two, so that guard
    # would have certified a five-page fit that did not exist. Filter the gutter, then allow no
    # spill at all: a tolerance wide enough to hide the defect it looks for is not a tolerance.
    spill = [l for l in before.splitlines()
             if l.strip() and not _re.fullmatch(r"\d+", l.strip())]
    body_pages = refs_page if spill else refs_page - 1
    assert body_pages <= BODY_PAGE_LIMIT, (
        f"the body occupies {body_pages} pages against a {BODY_PAGE_LIMIT}-page limit "
        f"(References starts on page {refs_page}; {len(spill)} body line(s) spill onto it). "
        f"See paper/NOTES.md §4 for the cut order and the do-not-cut list.")


def test_the_style_file_in_use_is_recorded():
    """If the style changes, the page-count result above changes with it -- say which is in use.

    Strengthened when the real NeurIPS 2026 style was finally located (#55). The old version
    matched only `\\usepackage{neurips_XXXX}` with no optional argument, so adding the required
    `[dblblindworkshop]` option would have broken it -- a guard that fails on the correct change
    is no better than one that passes on the wrong one. It now pins the four things the
    submission actually depends on.
    """
    tex = _tex()
    import re as _re
    m = _re.search(r"\\usepackage(?:\[([^\]]*)\])?\{(neurips_\d{4})\}", tex)
    assert m, "no neurips style package found; page-count guarantees are meaningless without one"
    opts = [o.strip() for o in (m.group(1) or "").split(",") if o.strip()]
    sty = ROOT / "paper" / f"{m.group(2)}.sty"
    assert sty.exists(), f"{m.group(2)}.sty is referenced but not present in paper/"

    # 1. anonymity. `final` and `preprint` both de-anonymise; so does sglblindworkshop, which
    #    sets \@anonymousfalse and differs from the correct option by three characters.
    for bad in ("final", "preprint", "sglblindworkshop", "nonanonymous"):
        assert bad not in opts, (
            f"style option '{bad}' de-anonymises the submission, and this venue is double-blind")

    # 2. the venue is a workshop, so the camera-ready footer must name it rather than the main
    #    track. \@trackname only renders under \if@neuripsfinal, so this is dormant at submission
    #    and load-bearing at camera-ready -- exactly the kind of thing that is forgotten later.
    if any(o.endswith("workshop") for o in opts):
        assert "\\workshoptitle{" in tex, (
            "a workshop option is set but \\workshoptitle is not; the camera-ready footer would "
            "read 'Workshop: .' with an empty name (the style warns, it does not error)")

    # 3. THE PAGE BUDGET. test_body_fits_the_page_limit is a claim about geometry, not about
    #    LaTeX in general. A style swap that changed any of these would move the page count
    #    silently, which is the entire risk #55 was opened to track.
    geom = _re.search(r"\\newgeometry\{(.*?)\}", sty.read_text(), _re.S)
    assert geom, f"no \\newgeometry block in {sty.name}; the page budget is unpinned"
    got = dict(_re.findall(r"(\w+)\s*=\s*([\w.]+)", geom.group(1)))
    expected = {"textheight": "9in", "textwidth": "5.5in", "top": "1in",
                "headheight": "12pt", "headsep": "25pt", "footskip": "30pt"}
    assert got == expected, (
        f"{sty.name} geometry is {got}, not {expected}. The page-fit result was measured against "
        f"the latter; re-measure before trusting it.")


# --------------------------------------------------- issue #48: every number traced to results/
MANIFEST = ROOT / "tests" / "paper_number_manifest.json"


def _manifest():
    if not MANIFEST.exists():
        pytest.skip("paper_number_manifest.json not present")
    return json.load(open(MANIFEST))


def _body():
    """paper.tex text a manifest literal may legitimately live in: body AND appendix.

    EXTENDED for the camera-ready restructure, not weakened. The old version cut at
    \\bibliographystyle, which precedes \\appendix, so it silently required every manifest
    literal to sit in the body proper. The Reproducibility promise is "every number in the
    paper is traceable to a result file" -- the appendix is part of the paper, and the
    camera-ready moves robustness detail there. Comments are still stripped, and the
    bibliography commands themselves are excluded so citation keys cannot satisfy a literal.
    """
    tex = _tex()
    return "\n".join(l for l in tex.splitlines()
                     if not l.lstrip().startswith("%")
                     and not l.strip().startswith(("\\bibliographystyle", "\\bibliography{")))


def _literal_appears(literal, body):
    """Boundary-aware match: a numeric literal must not be satisfied by a SUBSTRING of a longer
    number. Plain `in` let the manifest's 0.80 (the T=0.5 post-training ignition mean, stored
    0.8047) pass against the DK critical point 0.801(2) while the prose actually said 0.81 --
    a wrong number in the submitted paper that the traceability test existed to catch and
    missed. A literal that starts or ends with a digit now refuses digit (or digit-continuing)
    context on that side; non-numeric edges (braces, backslashes) keep exact matching.
    """
    pat = re.escape(literal)
    if literal[0].isdigit():
        pat = r"(?<![\d.])" + pat
    if literal[-1].isdigit():
        pat = pat + r"(?!\.?\d)"
    return re.search(pat, body) is not None


def test_every_manifest_number_appears_in_the_paper():
    """The Reproducibility appendix promises traceability; this is the test behind the promise.

    Each entry records a literal, the results file it came from, and the expression that
    derives it. If a literal is missing, either the paper changed a number without the results
    changing, or the manifest derivation is wrong. Both times this failed during development it
    was the manifest -- the newer artifact -- so check that first. Matching is boundary-aware:
    see _literal_appears for the erratum that plain substring matching hid.
    """
    body = _body()
    missing = [(m["literal"], m["source"], m["derivation"])
               for m in _manifest() if not _literal_appears(m["literal"], body)]
    assert not missing, (
        f"{len(missing)} manifest number(s) are not in paper.tex:\n" +
        "\n".join(f"  {lit} <- {src} :: {der}" for lit, src, der in missing))


def test_every_manifest_number_is_backed_by_an_existing_source():
    """Sources are of three kinds and are checked differently.

    A results file must exist on disk. A `published` value cites the module that records the
    literature anchor. An `arithmetic` value follows from the design and has no file -- but it
    must say so, so that a number with no provenance cannot hide behind a vague source string.
    """
    for m in _manifest():
        kind, src = m.get("kind", "measured"), m["source"]
        if kind == "arithmetic":
            assert src == "design", f"{m['literal']}: arithmetic entries must cite 'design'"
            continue
        if kind == "published":
            # A published value is either recorded in a repo module (e.g. src/dk.py ANCHORS)
            # or cited to an external source. The latter must carry a resolvable identifier --
            # an arXiv id or DOI -- so "published" cannot become a way to launder a number with
            # no provenance at all.
            in_repo = (ROOT / src.split()[0]).exists()
            cited = bool(re.search(r"arXiv:\d{4}\.\d{4,5}|doi\.org/|10\.\d{4,}/", src))
            assert in_repo or cited, (
                f"{m['literal']}: published source {src!r} names neither a repo file nor a "
                f"resolvable citation identifier")
            continue
        # measured: may name one file, or a brace set of files that all must exist
        names = ([src] if "{" not in src else
                 [src.replace(src[src.index("{"):src.index("}") + 1], part)
                  for part in src[src.index("{") + 1:src.index("}")].split(",")])
        for n in names:
            assert (RESULTS / n).exists(), (
                f"{m['literal']} claims to come from {n}, which is absent")


def test_manifest_covers_the_load_bearing_claims():
    """Guard against the manifest silently shrinking to whatever currently passes."""
    srcs = {m["source"] for m in _manifest()}
    required = {
        "dk_calib.json",                # the exact rung
        "eca_ordered_vs_rest.json",     # the ECA split
        "dev_transition_shape.json",    # the headline
        "dev_transition_n192.json",     # the third lattice size
        "dev_transition_scale.json",    # C20, the 4-size replication
        "dev_transition_temp.json",     # the temperature scope
        "real_generation_damage.json",  # F35, the delimiting result
        "coupling_gap.json",            # F41, the coupling correction
        "calib_census.json",            # the census rung
        "loss_baseline.json",           # #72, lambda_ca is not a perplexity proxy
    }
    assert required <= srcs, f"manifest no longer covers: {sorted(required - srcs)}"
    assert len(_manifest()) >= 60, f"manifest shrank to {len(_manifest())} entries"
    kinds = {m.get("kind", "measured") for m in _manifest()}
    assert {"measured", "published", "arithmetic"} <= kinds, (
        f"the manifest no longer distinguishes number kinds: {kinds}. A published literature "
        f"value and a measurement of ours must not be traced the same way.")


def test_the_crossing_brackets_in_prose_match_the_scale_results():
    """C20's per-size crossing brackets are prose claims; assert them against the file.

    These were never checked. That mattered little while they were only a timing anecdote, but
    the paper now hangs an external convergence claim on one of them -- Nakaishi et al.
    (arXiv:2406.05335) place the emergence of critical structure in Pythia-160m at k_c ~ 10^2
    steps, which is adjacent to our 160m bracket *only if that bracket really is 128--256*. A
    re-run that moved the bracket would silently turn a convergence into a contradiction.
    """
    d = _load("dev_transition_scale.json")
    pm = d.get("per_model")
    if not pm:
        pytest.skip("dev_transition_scale.json has no per_model block")
    tex = _tex()

    # 160m: the bracket the external convergence claim depends on
    got = pm["160m"]["crossing_interval"]
    assert got == ["step128", "step256"], (
        f"160m crossing bracket is {got}, but paper.tex claims 128--256 and uses it to assert "
        f"agreement with Nakaishi et al.'s k_c ~ 10^2. Reconcile before shipping.")
    assert re.search(r"160m crosses between steps \$128\$ and \$256\$", tex), (
        "paper.tex no longer states the 160m crossing bracket in the form the convergence "
        "claim depends on")

    # 410m and 1b share a bracket -- this is the tie that carries the learning-rate disclosure
    for k in ("410m", "1000m"):
        assert pm[k]["crossing_interval"] == ["step256", "step512"], (
            f"{k} crossing bracket moved to {pm[k]['crossing_interval']}; the paper claims "
            f"410m and 1b share the 256--512 bracket, which is what makes the LR confound "
            f"argument work (the two sizes sharing an LR are the two sharing a bracket)")

    # 70m has no crossing on this grid -- "already super-critical at the earliest checkpoint"
    assert pm["70m"]["crossing_interval"] is None, (
        f"70m now has a crossing interval ({pm['70m']['crossing_interval']}); the paper says it "
        f"is already super-critical at the earliest checkpoint probed")


def test_paper_loss_baseline_claims_match():
    """#72: the paper claims lambda_ca is not a monotone transform of held-out loss.

    Asserted against the file rather than through manifest literal-matching, because the load
    bearing quantities here are COUNTS -- "monotone at all four sizes", "overshoots at three" --
    and a single digit matches trivially anywhere in a manuscript. A manifest entry for "4"
    would be traceability theatre.
    """
    tex, lb = _tex(), _load("loss_baseline.json")
    a = lb["analysis"]

    # premise of the shape argument: loss must fall monotonically everywhere
    non_mono = [k for k, v in a.items() if not v["loss_monotone_decreasing"]]
    assert not non_mono, (
        f"held-out loss is no longer monotone at {non_mono}; the paper's "
        f"'non-monotone function of a monotone variable' argument collapses without it")

    # and lambda_ca must overshoot somewhere, or there is nothing to contrast
    over = [k for k, v in a.items() if v["lambda_overshoot"]]
    assert len(over) >= 3, (
        f"lambda_ca now overshoots at only {len(over)} of {len(a)} sizes ({over}); the paper "
        f"says three, and below that the proxy objection is no longer answered by shape")

    # the contrast with C20 is that the loss elbow does NOT move with size while the crossing does
    brackets = {tuple(v["loss_steepest_bracket"]) for v in a.values()}
    assert len(brackets) == 1, (
        f"the steepest-loss bracket is no longer identical across sizes ({brackets}); the "
        f"paper's 'for every size' claim is void")
    lo, hi = sorted(brackets.pop())
    assert f"${lo}$--${hi}$" in tex, (
        f"paper does not state the steepest-loss bracket as ${lo}$--${hi}$")

    # and the crossings must actually differ from it, or there is no disagreement to report
    crossings = {tuple(v["lambda_crossing_bracket"]) for v in a.values()
                 if v["lambda_crossing_bracket"]}
    assert crossings != {(lo, hi)}, (
        "every lambda_ca crossing now coincides with the steepest-loss bracket; the paper claims "
        "they disagree about location")

    # the correlation half is explicitly NOT leaned on -- check the paper still says so
    rho = [v["spearman_rho"] for v in a.values()]
    assert f"{abs(max(rho)):.2f}" in tex and f"{abs(min(rho)):.2f}" in tex, (
        f"paper omits the Spearman range {min(rho):.2f} to {max(rho):.2f}")
    assert "correlation does not" in tex or "not the correlation" in tex, (
        "the paper no longer says the correlation does not carry this claim; at n=6 checkpoints "
        "with 1 of 4 significant, leaning on rho would be the weakest available argument")


def test_section4_opening_bracket_matches_the_primary_results_file():
    """The headline section's OPENING sentence must state the bracket its own file supports.

    This is the gap that let a retracted claim survive. `test_the_crossing_brackets_in_prose_
    match_the_scale_results` checks the four-size paragraph and nothing else, so §4's first
    sentence drifted unguarded: it read "between steps 512 and 2000" while
    dev_transition_phase3.json puts the cell-mean sign change at 256->512 at BOTH lattice sizes.
    NOTES.md already lists "crosses zero between steps 512 and 1000" as retracted -- the opening
    sentence was that retraction with its right edge moved and its left edge untouched.

    Derived from the file rather than hard-coded, so a re-run that moved the crossing fails the
    paper instead of silently disagreeing with it. F42 applies: unignited runs are excluded from
    the cell means, which is what makes N=96's step256 mean -0.0116 rather than -0.0307.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "experiments"))
    from lyapunov import run_ignited
    import numpy as _np

    d = _load("dev_transition_phase3.json")["runs"]
    rows = [v for v in d.values() if isinstance(v, dict) and "lambda_ca" in v]
    if not rows:
        pytest.skip("no runs in dev_transition_phase3.json")

    def ignited(v):
        return run_ignited(v)

    steps = sorted({v["step"] for v in rows})
    brackets = set()
    for N in sorted({v["N"] for v in rows}):
        means = []
        for s in steps:
            vals = [v["lambda_ca"] for v in rows if v["N"] == N and v["step"] == s and ignited(v)]
            means.append(_np.mean(vals) if vals else _np.nan)
        cross = next(((steps[i], steps[i + 1]) for i in range(len(steps) - 1)
                      if means[i] < 0 <= means[i + 1]), None)
        assert cross is not None, f"no cell-mean sign change at N={N}; §4's opening claim is void"
        brackets.add(cross)

    assert len(brackets) == 1, (
        f"the lattice sizes no longer agree on the crossing bracket ({brackets}); §4 says "
        f"'at both lattice sizes'")
    lo, hi = brackets.pop()
    tex = " ".join(_tex().split())
    assert f"between steps ${lo}$ and ${hi}$ at both lattice sizes" in tex, (
        f"§4's opening sentence does not state the crossing bracket its own results file gives "
        f"({lo}->{hi}). NOTES.md lists 'crosses zero between steps 512 and 1000' as RETRACTED; "
        f"do not let it back in.")
    for bad in ("between steps $512$ and $2000$", "between steps $512$ and $1000$"):
        assert bad not in tex, f"the retracted crossing framing is back: {bad}"


def test_notes_do_not_name_a_style_file_the_paper_no_longer_uses():
    """NOTES.md must not describe a style the preamble abandoned.

    Third instance of one defect class: #47 ("the cut ledger is stale"), the submission plan's
    D3 ("NOTES.md §4 is stale"), and then -- after both were fixed by hand -- two rows still
    claiming `neurips_2025.sty` and "2026 not published yet (404)" a commit after the swap to
    the official 2026 style. A defect that recurs after a manual fix wants a test, not a fourth
    sweep.

    Deliberately narrow: it pins the style-file claim only, which is the row that is both
    load-bearing (the page budget depends on the geometry) and repeatedly wrong. It does not try
    to police prose currency in general, which no test can do.
    """
    import re as _re
    notes = ROOT / "paper" / "NOTES.md"
    if not notes.exists():
        pytest.skip("paper/NOTES.md not present")
    m = _re.search(r"\\usepackage(?:\[[^\]]*\])?\{(neurips_\d{4})\}", _tex())
    assert m, "no neurips style package in paper.tex"
    in_use = m.group(1)
    named = set(_re.findall(r"neurips_(\d{4})", notes.read_text()))
    stale = named - {in_use.split("_")[1]}
    assert not stale, (
        f"paper/NOTES.md still names neurips_{sorted(stale)} while paper.tex uses {in_use}. "
        f"The style file sets the page budget, so a stale row here is a stale claim about the "
        f"submission's hardest constraint.")
