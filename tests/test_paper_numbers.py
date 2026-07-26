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
    tex = _tex()
    for s in ("token-lattice-ca", "nicoveraz", "github.com"):
        assert s not in tex, f"double-blind violation in paper.tex: {s!r}"


def test_responsible_use_section_exists():
    tex = _tex()
    assert re.search(r"\\section\{[^}]*Responsible use", tex), (
        "no responsible-use section -- automatic desk reject at this venue")
