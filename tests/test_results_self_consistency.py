"""Rule 8: a results file must not contradict its own declared design.

This test exists because of two defects that a prose grep could not have found -- neither
document in the repo contains the string `sel(N, {256}, m)`:

  * F39: `dev_transition_shape.json` declared `"pre": [256, 512]` and computed the headline
    over `{256}` alone. `n_pre` was 8 where the design gives 16, while the `variance` block
    in the SAME file used all 16. One assertion would have failed the run.
  * F42: `lambda_ca` was averaged over runs where damage never ignited, for which it is
    undefined, and the emitted value spans an order of magnitude for the same physical
    outcome (-0.1649 vs -1.7130).

The checks are about the DESIGN, not the values: they compare what a file says it did
against what it must have done had its stated design been followed. A future subset can only
pass by also changing the declared design, which is visible in review.
"""
import json, pathlib, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
RESULTS = ROOT / "results"


def _load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    return json.load(open(p))


def _unignited(v):
    from lyapunov import run_ignited
    return not run_ignited(v)


# ------------------------------------------------------- F39: declared design vs computed n
def test_shape_groups_match_declared_definitions():
    """The exact check that would have failed the first version of the shape script."""
    d = _load("dev_transition_shape.json")
    defs = d["_definitions"]
    exp_pre = len(defs["pre"]) * defs["n_seeds"]
    exp_plateau = len(defs["plateau"]) * defs["n_seeds"]
    assert defs["expected_n_pre"] == exp_pre
    assert defs["expected_n_plateau"] == exp_plateau
    for N in defs["sizes"]:
        # F42 makes the expected n metric-dependent: lambda drops unignited runs (no cone,
        # value undefined), D_norm keeps them (zero damage is a true zero). The design check
        # must account for that EXACTLY -- not be relaxed to an inequality, which would let
        # F39's silent-subset defect back in.
        def dead_in(step_set):
            return sum(v["n_unignited"] for k, v in d["ignition"].items()
                       if k.startswith(f"N{N}_step") and int(k.split("step")[1]) in step_set)
        for m in ("lambda_ca", "D_norm"):
            h = d["headline"][f"N{N}_{m}"]
            drop_pre = dead_in(defs["pre"]) if m == "lambda_ca" else 0
            drop_pl = dead_in(defs["plateau"]) if m == "lambda_ca" else 0
            assert h["n_pre"] == exp_pre - drop_pre, (
                f"N{N}_{m}: headline used n_pre={h['n_pre']} but the declared pre set "
                f"{defs['pre']} x {defs['n_seeds']} seeds minus {drop_pre} unignited = "
                f"{exp_pre - drop_pre}. This is F39's defect (or F42's filter misapplied).")
            assert h["n_plateau"] == exp_plateau - drop_pl


def test_shape_retains_the_unregistered_variants_for_audit():
    """The inflated numbers stay visible, under names that say what they are."""
    d = _load("dev_transition_shape.json")
    for k, v in d["headline"].items():
        assert "cohens_d_from_step256_only_UNREGISTERED" in v, k
        assert "cohens_d_vs_peak_INFLATED" in v, k
        assert v["cohens_d"] != v["cohens_d_from_step256_only_UNREGISTERED"], (
            f"{k}: pre-registered and step256-only effect sizes are identical -- the "
            f"correction was reverted or the selection is wrong again")


def test_phase3_runs_have_the_designed_count():
    d = _load("dev_transition_phase3.json")
    runs = [v for v in d["runs"].values() if "lambda_ca" in v]
    steps = {v["step"] for v in runs}; sizes = {v["N"] for v in runs}
    seeds = {v["seed"] for v in runs}
    assert len(runs) == len(steps) * len(sizes) * len(seeds)


# ------------------------------------------------------------------ F42: the ignition rule
def test_unignited_predicate_catches_what_the_floor_predicate_misses():
    from lyapunov import is_unignited, is_dead_damage_floor
    for lam in (-0.1649, -1.7130):                 # both observed with zero final damage
        assert not is_dead_damage_floor(lam), (
            f"lambda={lam} must NOT match the floor sentinel; if it does, F42's premise "
            f"is wrong and is_unignited is unnecessary")
    assert is_unignited(mean_damage=0.0)
    assert is_unignited(D_norm=0.0)
    # a negative-but-IGNITED run is a real measurement and must be kept
    assert not is_unignited(mean_damage=0.0250)
    assert not is_unignited(D_norm=0.0250)
    with pytest.raises(ValueError):
        is_unignited()


def test_estimator_is_N_independent_for_a_fixed_cone():
    """Guards the corrected mechanism statement in `is_unignited`'s docstring.

    It is tempting to explain the -0.1649 vs -1.7130 spread as the estimator scaling with
    lattice size. It does not: N appears only in the second return value. If this test ever
    fails, the docstring's mechanism paragraph must be rewritten, not the test relaxed.
    """
    from lyapunov import lyap_from_cone
    lams = []
    for N in (48, 96, 192):
        cone = np.zeros((23, N))
        cone[0, N // 2 - 1:N // 2 + 2] = 1.0       # 3-site seed that dies immediately
        lams.append(lyap_from_cone(cone, N)[0])
    assert len({round(x, 9) for x in lams}) == 1, (
        f"lambda varies with N for an identical cone: {lams}. The estimator IS "
        f"size-dependent and is_unignited's docstring is wrong.")


# ----------------------------------------------------------------- F42 metric asymmetry
def test_lambda_and_D_norm_use_different_ignition_filters():
    """The filter is asymmetric ON PURPOSE and a refactor must not collapse the two.

    lambda_ca: zero damage means NO CONE -> the value is UNDEFINED -> drop the run.
    D_norm   : zero damage means the ratio is GENUINELY ZERO -> a true measurement -> keep.

    Dropping unignited runs from D_norm too would raise its pre level and shrink its gap,
    biasing the metric that is not broken (N=96 pre would go 0.1030 -> 0.1099, retention
    53% -> 51%). The emitted `_definitions` must therefore record two different bases.
    """
    d = _load("dev_transition_shape.json")
    defs = d["_definitions"]
    assert "ignited" in defs["lambda_basis"], defs["lambda_basis"]
    assert "all runs" in defs["D_norm_basis"], defs["D_norm_basis"]
    exp_pre = len(defs["pre"]) * defs["n_seeds"]
    for N in defs["sizes"]:
        dead_pre = sum(v["n_unignited"] for k, v in d["ignition"].items()
                       if k.startswith(f"N{N}_step")
                       and int(k.split("step")[1]) in defs["pre"])
        assert d["headline"][f"N{N}_D_norm"]["n_pre"] == exp_pre, (
            f"N{N}: D_norm dropped runs it should have kept -- the asymmetry collapsed")
        assert d["headline"][f"N{N}_lambda_ca"]["n_pre"] == exp_pre - dead_pre, (
            f"N{N}: lambda kept {dead_pre} unignited run(s) it should have dropped")


def test_D_norm_zero_fallback_margin_is_asserted_not_its_current_value():
    """`is_unignited(D_norm=0)` needs a MARGIN, and the margin has a size at which it dies.

    Measured, not assumed: the smallest nonzero mean_damage is 1/(tail*N*B). It is equal at
    N=48/96/192 in this project ONLY because the design holds N*B = 768 fixed (B is halved
    as N doubles for the 16GB budget). It is neither N-independent nor 1/N -- it is
    1/(N*B). A design with fixed B would halve it at every doubling, and the fallback dies
    once round(quantum/D0, 5) == 0.

    Asserting today's 2.7e-4 would pass right up until the configuration where it stops
    being true and then fail silently. So assert the formula and the headroom instead.
    """
    tail = 8
    designs = ((48, 16), (96, 8), (192, 4))
    quanta = {(N, B): 1.0 / (tail * N * B) for N, B in designs}
    assert len(set(round(q, 12) for q in quanta.values())) == 1, (
        f"the design no longer holds N*B fixed: {quanta}. The fallback's margin now varies "
        f"across sizes and each must be checked separately.")
    for (N, B), q in quanta.items():
        for D0 in (0.1, 0.3, 0.6, 1.0):
            assert round(q / max(D0, 1e-3), 5) > 0.0, (
                f"at N={N} B={B} D0={D0} one damaged site rounds D_norm to 0.00000 -- the "
                f"fallback is unsound here; record mean_damage instead")
    # headroom: how much larger could N*B get before the fallback dies at the worst D0?
    worst_D0 = 1.0
    limit_NB = 1.0 / (tail * 0.5e-5 * worst_D0)
    assert designs[0][0] * designs[0][1] < limit_NB / 10, (
        f"N*B={designs[0][0]*designs[0][1]} is within 10x of the limit {limit_NB:.0f} where "
        f"a single damaged site rounds to zero -- stop using the D_norm fallback")


def test_phase3_unignited_count_is_what_F42_documents():
    """Phase 3 contains exactly one unignited run. If this changes, update F42, not the test."""
    d = _load("dev_transition_phase3.json")
    runs = [v for v in d["runs"].values() if "lambda_ca" in v]
    dead = [v for v in runs if _unignited(v)]
    assert len(dead) == 1, f"expected 1 unignited run, found {len(dead)}: {dead}"
    only = dead[0]
    assert (only["N"], only["step"], only["seed"]) == (96, 256, 22)


def test_n192_uses_the_same_ignition_asymmetry_as_the_shape_script():
    """Two scripts, one rule, must not have two behaviours (F45).

    dev_transition_n192.py originally applied the ignition filter to BOTH metrics while
    dev_transition_shape.py applied it only to lambda. That inflated D_norm's plateau by 14%
    -- on the quantity whose size scaling was the entire point of the run.
    """
    d = _load("dev_transition_n192.json")
    if "analysis" not in d:
        pytest.skip("n192 analysis not present")
    a = d["analysis"]
    assert "ignited runs only" in a["lambda_ca"]["basis"], a["lambda_ca"]["basis"]
    assert "ALL runs" in a["D_norm"]["basis"], a["D_norm"]["basis"]
    # and the n's must differ exactly by the unignited count wherever there is one
    dead_plateau = a["ignition"]["step143000"]["n_unignited"]
    assert a["D_norm"]["n_plateau"] - a["lambda_ca"]["n_plateau"] == dead_plateau, (
        "the two metrics' plateau group sizes do not differ by exactly the unignited count")


# ------------------------------------------------------- issue #38: stale-analysis detection
_STALENESS_PAIRS = [
    ("dev_transition_shape.json", "dev_transition_shape.py"),
    ("dev_transition_n192.json", "dev_transition_n192.py"),
    ("dev_transition_scale.json", "dev_transition_scale.py"),
    ("ignition_vs_size.json", "ignition_vs_size.py"),
    ("ignition_nb.json", "ignition_nb.py"),
    ("floor_decorrelation.json", "floor_decorrelation.py"),
    ("dev_transition_temp.json", "dev_transition_temp.py"),
    ("loss_baseline.json", "loss_baseline.py"),
    ("dev_transition_width.json", "dev_transition_width.py"),
    ("lyap_perturbation_size.json", "lyap_perturbation_size.py"),
    ("loss_collapse_pile.json", "loss_collapse_pile.py"),
    ("dev_transition_width_early.json", "dev_transition_width_early.py"),
    ("dev_transition_410m_early.json", "dev_transition_410m_early.py"),
    ("dp_survival_scan.json", "dp_survival_scan.py"),
    ("dp_pipeline_validation.json", "dp_pipeline_validation.py"),
    ("dp_narrow_bracket.json", "dp_narrow_bracket.py"),
    ("dp_class_n192.json", "dp_class_n192.py"),
    ("dp_fss_z.json", "dp_fss_z.py"),
]


@pytest.mark.parametrize("results_name,script_name", _STALENESS_PAIRS,
                         ids=[p[0] for p in _STALENESS_PAIRS])
def test_analysis_matches_the_source_that_claims_to_have_written_it(results_name, script_name):
    """A results file must not have been produced by a different version of its analysis.

    Python imports a module once, so editing an analysis script while its job runs leaves the
    job writing its END-OF-RUN analysis with the code imported at launch. That happened twice
    (F45, F46) and both times produced a finished-looking results file with a wrong conclusion
    -- in F45's case an inverted one ("size-robustness DOWNGRADED" where the correct answer was
    "invariant across a 4x range").

    Each analysis stamps the sha256 of its source; this recomputes it. A mismatch means the
    numbers in the file predate the code on disk: re-run the analysis, do not read the file.
    """
    import hashlib
    d = _load(results_name)
    prov = d.get("_analysis_provenance") or d.get("analysis", {}).get("_analysis_provenance")
    assert prov is not None, (
        f"{results_name} has no _analysis_provenance stamp -- it cannot be checked against the "
        f"code that wrote it (issue #38)")
    src = ROOT / "experiments" / script_name
    if not src.exists():
        pytest.skip(f"{script_name} not present")
    actual = hashlib.sha256(src.read_bytes()).hexdigest()
    assert prov["sha256"] == actual, (
        f"{results_name} was written by a different version of {script_name} "
        f"(stamped {prov['sha256'][:12]}, on disk {actual[:12]}). Re-run the analysis before "
        f"reading its numbers -- this is the F45/F46 stale-analysis trap.")


# --------------------------- a committed log must not contradict its current results file
LOG_RESULT_PAIRS = [
    ("n192.log",  "dev_transition_n192.json",  lambda d: list(d["verdict"].values())),
    ("scale.log", "dev_transition_scale.json", lambda d: [d["primary_verdict"]]),
    ("temp.log",  "dev_transition_temp.json",  lambda d: [d["verdict"]]),
]


@pytest.mark.parametrize("log_name,results_name,extract", LOG_RESULT_PAIRS,
                         ids=[p[0] for p in LOG_RESULT_PAIRS])
def test_log_does_not_contradict_its_results_file(log_name, results_name, extract):
    """The log a reviewer greps must agree with the results file the paper cites.

    This is the CLASS behind issue #46, which was first fixed only as an instance. A job writes
    its own end-of-run analysis using the code imported at launch, so any mid-run edit leaves a
    stale verdict in the log while the regenerated JSON is correct. It happened twice:

      * logs/n192.log said "size-robustness DOWNGRADED" where the truth is "INVARIANT across a
        4x range" -- an inverted conclusion;
      * logs/scale.log carried the pre-fix verdict string, less precise though not wrong.

    Fixing the first and not checking the second is exactly the failure this test exists to
    stop. The remedy is never to hand-edit the log: append a machine-written superseding block
    by re-running the analysis, which leaves the current verdict as the last one present.
    """
    log = ROOT / "logs" / log_name
    if not log.exists():
        pytest.skip(f"{log_name} not present")
    d = _load(results_name)
    text = log.read_text()
    for verdict in extract(d):
        assert verdict in text, (
            f"logs/{log_name} does not contain the current verdict from {results_name}:\n"
            f"  {verdict!r}\n"
            f"Append a machine-written superseding block by re-running the analysis; do not "
            f"hand-edit the log.")


def test_every_stamped_results_file_is_covered_by_the_staleness_check():
    """A file that records provenance but is not checked against it gains nothing.

    The stamp is only useful if something recomputes it. This asserts the parametrize list above
    keeps pace with the scripts that stamp, so adding provenance to a new analysis without adding
    it to the guard fails here rather than passing silently.
    """
    import glob, os
    stamped = set()
    for f in glob.glob(str(RESULTS / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        pr = d.get("_analysis_provenance")
        if pr is None and isinstance(d.get("analysis"), dict):
            pr = d["analysis"].get("_analysis_provenance")
        if pr:
            stamped.add(os.path.basename(f))
    covered = {results_name for results_name, _script in _STALENESS_PAIRS}
    assert stamped <= covered, (
        f"stamped but unchecked: {sorted(stamped - covered)}. Add them to _STALENESS_PAIRS.")


# ------------------------------------------- issue #52: our own prints must not leak abs paths
def test_no_log_carries_an_absolute_path_we_printed_ourselves():
    """A machine-written log must not contain a checkout path that we put there.

    Twelve logs shipped lines like `wrote /Users/<user>/Documents/GitHub/textca/results/x.json`,
    because twelve experiment scripts printed an absolute `OUT`. That is a de-anonymisation leak
    in an artifact the submission mirror publishes, and the absolute form carries nothing the
    relative one does not -- every such path is inside this repository. `provenance.rel()` is
    the fix; this test is what stops the thirteenth script from reintroducing it.

    SCOPE, stated so the pass is not mistaken for more than it is. This checks only what WE
    print. Python emits absolute paths of its own that no change here can reach:

      * tracebacks   -- `File "/Users/<user>/.../experiments/ar_probe.py", line 120`
      * stdlib warnings -- multiprocessing's leaked-semaphore notice, from the interpreter's own
        install directory

    Four archival logs under `results/logs_*/` carry those and are left alone; re-running the
    jobs that produced them means real model generation, and the mirror scrub already rewrites
    them. The one allowed exception below is ours but deliberately kept.
    """
    import re as _re
    allowed = {
        # The only record of the 32 temperature-widening runs (#73): its 32 per-run progress
        # lines exist nowhere else, and dev_transition_temp.py now resumes from cached runs, so
        # re-running regenerates the ANALYSIS log (temp.log) and not this one. Deleting it to
        # make a grep pass would destroy evidence to satisfy a test, which is backwards.
        "logs/temp_widen.log",
    }
    pat = _re.compile(r"wrote\s+/(?:Users|home)/")
    offenders = []
    for p in sorted(ROOT.glob("logs/*.log")) + sorted(ROOT.glob("results/logs_*/*.log")):
        relp = str(p.relative_to(ROOT))
        if relp in allowed:
            continue
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        for line in txt.splitlines():
            if pat.search(line):
                offenders.append(f"{relp}: {line.strip()[:90]}")
    assert not offenders, (
        "logs contain absolute paths printed by our own code; use provenance.rel(OUT) and "
        "re-run so the log is machine-written:\n  " + "\n  ".join(offenders))
def test_nobody_reimplements_the_f42_run_level_predicate():
    """The F42 adapter must exist once, in lyapunov.run_ignited (#63).

    `is_unignited` takes a VALUE; every caller needs a RUN-record adapter choosing which field to
    pass, because older records predate `mean_damage` and need the `D_norm` fallback. That adapter
    was hand-written thirteen times across experiments/ and tests/, and written WRONGLY twice --
    once applying the filter to both metrics, inflating D_norm's N=192 plateau 0.1393 -> 0.1592, a
    14% error on the quantity whose size scaling was that run's entire point.

    A rule enforced by prose gets re-derived by whoever writes the next script, so this asserts
    that no file outside lyapunov.py contains the field-choosing branch. It matches the SHAPE of
    the adapter -- an is_unignited call keyed on "mean_damage" -- not a name, because renaming a
    local helper is exactly how the fourteenth copy would slip past.
    """
    import re as _re
    pat = _re.compile(r'is_unignited\s*\(\s*mean_damage\s*=.*?"mean_damage"\s+in', _re.S)
    offenders = []
    for d in ("experiments", "tests"):
        for p in sorted((ROOT / d).rglob("*.py")):
            if p.name == "lyapunov.py":
                continue
            txt = p.read_text(errors="replace")
            for m in pat.finditer(txt):
                if len(txt[:m.start()].split("\n")[-1]) < 400:   # ignore prose in docstrings
                    offenders.append(f"{p.relative_to(ROOT)}:{txt[:m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        "the F42 run-level predicate is re-implemented outside lyapunov.py at "
        + ", ".join(offenders) + " -- import `run_ignited` instead. It has been written by hand "
        "thirteen times and gotten wrong twice.")


# ------------------------------------- issue #82: a calibration only licenses its own geometry
def _const(script, *names):
    """Module-level scalar constants, read without importing (these scripts pull in torch)."""
    import ast
    tree = ast.parse((ROOT / "experiments" / script).read_text())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgts = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if tgts and tgts[0] in names:
            out[tgts[0]] = ast.literal_eval(node.value)
        elif isinstance(node.targets[0], ast.Tuple):          # e.g. `SETTLE, SWEEPS = 8, 40`
            keys = [e.id for e in node.targets[0].elts if isinstance(e, ast.Name)]
            vals = ast.literal_eval(node.value)
            out.update({k: v for k, v in zip(keys, vals) if k in names})
    return out


# Every DP script that measures the LM, and the results file each writes. A script here either
# carries its own inline calibration or is covered by dp_pipeline_validation's transfer check.
_DP_LM_SCRIPTS = [
    ("dp_survival_scan.py", "dp_survival_scan.json"),
    ("dp_narrow_bracket.py", "dp_narrow_bracket.json"),
    ("dp_class_n192.py", "dp_class_n192.json"),
]


@pytest.mark.parametrize("script,results", _DP_LM_SCRIPTS, ids=[s for s, _ in _DP_LM_SCRIPTS])
def test_every_dp_run_is_calibrated_at_its_own_geometry(script, results):
    """A pipeline calibrated at one lattice geometry says nothing about another.

    dp_pipeline_validation originally varied only REPLICAS -- the cheap axis -- at a fixed
    N=512 over 200 sweeps, then concluded that the LM slopes were readable as measurements. The
    LM ran N=96 over 40 sweeps, where the identical estimator misses delta by ~20% on
    Domany-Kinzel data that provably IS directed percolation. The first phase-2 verdict therefore
    announced "not in the DP class" using a tolerance that rejects DK itself (F56).

    So the calibration geometry is a design constant with one source of truth. Each LM run must
    be covered at ITS OWN (N, sweeps) -- by an inline calibration, or by the validation script's
    transfer check.
    """
    geom = _const(script, "N", "SWEEPS")
    d = _load(results)
    here = (d.get("calibration_at_run_geometry") or {}).get("at_run_geometry")
    if here is not None:
        assert (here["N"], here["sweeps"]) == (geom["N"], geom["SWEEPS"]), (
            f"{script} runs N={geom['N']}/{geom['SWEEPS']} but its inline calibration is at "
            f"N={here['N']}/{here['sweeps']} -- it does not cover the run it licenses")
        return
    at = _load("dp_pipeline_validation.json").get("at_lm_geometry")
    assert at, f"{script} has no inline calibration and no transfer check covers it"
    assert (at["N"], at["sweeps"]) == (geom["N"], geom["SWEEPS"]), (
        f"{script} runs N={geom['N']}/{geom['SWEEPS']} but the only calibration covering it is "
        f"at N={at['N']}/{at['sweeps']}")


def test_the_dp_gate_has_exactly_one_implementation():
    """The gate must not be pasted into a second script, because a drifted gate is F56 again.

    F56 was a calibration applied at the wrong geometry. A copy of the gate that silently
    diverges -- a different tolerance, a different seed count, a bare threshold instead of one
    that counts its own spread -- is indistinguishable from that defect at the point where it
    matters. So the implementation lives in dp_calibration and the scripts import it.
    """
    owned = {"bias_at", "decides", "dk_exponents", "calibrate", "print_ladder"}
    src = (ROOT / "experiments" / "dp_calibration.py").read_text()
    for fn in owned:
        assert f"def {fn}(" in src, f"dp_calibration must define {fn}"
    for script, _ in _DP_LM_SCRIPTS:
        text = (ROOT / "experiments" / script).read_text()
        for fn in owned:
            assert f"def {fn}(" not in text and f"def _{fn}(" not in text, (
                f"{script} defines its own {fn} -- the DP gate must have one implementation, "
                f"imported from dp_calibration")


def test_no_dp_verdict_claims_a_class_the_calibration_cannot_support():
    """A DP verdict is only licensed when the estimator is shown to work at that geometry.

    Guards the specific retraction: phase 2 may not assert the transition is or is not in the DP
    class while its own calibration gate is failing.
    """
    nb = json.loads((ROOT / "results" / "dp_narrow_bracket.json").read_text())
    decides = nb["calibration_at_run_geometry"]["geometry_decides"]
    verdict = nb["verdict"].lower()
    if not decides:
        assert "not decidable" in verdict, (
            "the calibration gate is failing, so the verdict must be NOT DECIDABLE")
        assert "is not in the dp class" not in verdict and "dp-consistent" not in verdict, (
            f"verdict claims a DP class the calibration does not support: {nb['verdict']}")
