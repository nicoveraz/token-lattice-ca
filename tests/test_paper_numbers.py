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
    """This is what licenses the developmental claim despite F35."""
    tex = _tex()
    assert "held \\emph{fixed} across" in tex or "held fixed across" in tex, (
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


def test_no_self_identifying_strings_in_the_submission():
    """NOTE for the anonymisation pass (#52): the identifiers below appear in THIS FILE as
    strings the paper is forbidden to contain. Scrubbing them out of the mirror would disable
    the guard, so this file is deliberately left as-is -- a reviewer reading a test that forbids
    an identifier learns nothing identifying from it."""
    tex = _tex()
    for s in ("token-lattice-ca", "nicoveraz", "github.com"):
        assert s not in tex, f"double-blind violation in paper.tex: {s!r}"


def test_responsible_use_section_exists():
    tex = _tex()
    assert re.search(r"\\section\{[^}]*Responsible use", tex), (
        "no responsible-use section -- automatic desk reject at this venue")


# ------------------------------------------------------------------ issue #41: the page limit
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


@pytest.mark.xfail(strict=True, reason=(
    "Body is 6 pages, not 5 -- see issue #62. Known and tracked; the trim is a Gate B "
    "words-only task, deliberately sequenced after project polish. strict=True so that when "
    "the trim lands this XPASSes and FAILS, forcing the marker to be removed rather than "
    "quietly outliving the defect."))
def test_body_fits_the_page_limit():
    """The body must end within the venue's page limit.

    This is the submission's hardest constraint and nothing asserted it: the fit was verified by
    hand after each edit, and it is currently a property of neurips_2025.sty. The 2026 style is
    unpublished (404 as of 2026-07-26) and its geometry sets the page count, so swapping it can
    silently undo a cut that took the body from 13 pages to 5. References, appendix and checklist
    are excluded from the limit, so the test locates where the bibliography starts.
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
    body_spills_onto_refs_page = len([l for l in before.splitlines() if l.strip()]) > 2
    body_pages = refs_page if body_spills_onto_refs_page else refs_page - 1
    assert body_pages <= BODY_PAGE_LIMIT, (
        f"the body occupies {body_pages} pages against a {BODY_PAGE_LIMIT}-page limit "
        f"(References starts on page {refs_page}). See paper/NOTES.md §4 for the cut order and "
        f"the do-not-cut list.")


def test_the_style_file_in_use_is_recorded():
    """If the style changes, the page-count result above changes with it -- say which is in use."""
    tex = _tex()
    import re as _re
    m = _re.search(r"\\usepackage\{(neurips_\d{4})\}", tex)
    assert m, "no neurips style package found; page-count guarantees are meaningless without one"
    sty = ROOT / "paper" / f"{m.group(1)}.sty"
    assert sty.exists(), f"{m.group(1)}.sty is referenced but not present in paper/"


# --------------------------------------------------- issue #48: every number traced to results/
MANIFEST = ROOT / "tests" / "paper_number_manifest.json"


def _manifest():
    if not MANIFEST.exists():
        pytest.skip("paper_number_manifest.json not present")
    return json.load(open(MANIFEST))


def _body():
    """paper.tex body, comments stripped, bibliography onward removed."""
    tex = _tex().split("\\bibliographystyle")[0]
    return "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))


def test_every_manifest_number_appears_in_the_paper():
    """The Reproducibility appendix promises traceability; this is the test behind the promise.

    Each entry records a literal, the results file it came from, and the expression that
    derives it. If a literal is missing, either the paper changed a number without the results
    changing, or the manifest derivation is wrong. Both times this failed during development it
    was the manifest -- the newer artifact -- so check that first.
    """
    body = _body()
    missing = [(m["literal"], m["source"], m["derivation"])
               for m in _manifest() if m["literal"] not in body]
    assert not missing, (
        f"{len(missing)} manifest number(s) are not in paper.tex:\n" +
        "\n".join(f"  {lit} <- {src} :: {der}" for lit, src, der in missing))


def test_every_manifest_number_is_backed_by_an_existing_results_file():
    for m in _manifest():
        p = RESULTS / m["source"]
        assert p.exists(), f"{m['literal']} claims to come from {m['source']}, which is absent"


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
    }
    assert required <= srcs, f"manifest no longer covers: {sorted(required - srcs)}"
    assert len(_manifest()) >= 45, f"manifest shrank to {len(_manifest())} entries"
