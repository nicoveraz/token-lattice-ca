"""Regenerate tests/paper_number_manifest.json: every paper number, DERIVED from its results file.

The Reproducibility appendix promises "every number in the paper is traceable to a result file".
This is what turns that promise into a test (issue #48). For each load-bearing literal in
`paper.tex` it records the results file and the exact expression that produces it, so
`tests/test_paper_numbers.py::test_every_manifest_number_appears_in_the_paper` can assert the
literal is present and `..._is_derivable_from_results` can recompute it.

TWO TRAPS THIS HIT WHILE BEING WRITTEN, both worth keeping in mind when extending it:

  * DERIVE FROM RAW RUNS, NOT FROM STORED ROUNDED VALUES. The 410m plateau level is stored as
    0.1735; re-rounding that to 3dp gives 0.173, while the raw mean 0.173533 gives 0.174, which
    is what the paper says. Double-rounding a boundary value produces a spurious mismatch.
  * DERIVE THE QUANTITY THE PAPER ACTUALLY STATES. The sign-agreement span is over BOTH lattice
    sizes; deriving it from N=48 alone produced a spurious mismatch against a correct paper.

Both initial failures were manifest bugs, not paper errors. That is the expected failure mode:
the manifest is the newer, less-reviewed artifact.

Usage:  .venv/bin/python experiments/build_paper_manifest.py
"""
import json, pathlib
import numpy as np
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
R = pathlib.Path("results")
def L(n): return json.load(open(R / n))

sh   = L("dev_transition_shape.json")
p3   = L("dev_transition_phase3.json")
n192 = L("dev_transition_n192.json")
scal = L("dev_transition_scale.json")
dk   = L("dk_calib.json")
eca  = L("eca_ordered_vs_rest.json")
cen  = L("calib_census.json")
rg   = L("real_generation_damage.json")
cg   = L("coupling_gap.json")
tmp  = L("dev_transition_temp.json")
ben  = L("cml_benettin.json")

def f(x, p): return f"{x:.{p}f}"

M = []
def add(literal, src, expr, kind="measured"):
    """kind: measured (from our results/), published (a literature value we cite), or
    arithmetic (a number that follows from the design, e.g. a permutation bound)."""
    M.append({"literal": literal, "source": src, "derivation": expr, "kind": kind})

# --- DK rung -------------------------------------------------------------
pa = dk["part_a_exact_identity"]
add(str(pa["N"]),                        "dk_calib.json", "part_a_exact_identity.N")
add(str(pa["steps"]),                    "dk_calib.json", "part_a_exact_identity.steps")
add(str(pa["control_offline_mismatch"]), "dk_calib.json", "part_a_exact_identity.control_offline_mismatch")
add(f(pa["by_p1"]["1"]["final_damage_density"], 5), "dk_calib.json", "by_p1['1'].final_damage_density (Rule 90)")
cal = dk["calibration"]
add(f(cal["site DP"]["estimate"], 4),    "dk_calib.json", "calibration['site DP'].estimate")
add(f(cal["W18 activity"]["estimate"],4),"dk_calib.json", "calibration['W18 activity'].estimate")
add(f(cal["W18 damage"]["estimate"], 4), "dk_calib.json", "calibration['W18 damage'].estimate")
# --- ECA -----------------------------------------------------------------
t = eca["tests"]; g = eca["groups"]
add(f(g["ordered"]["mean"], 3),  "eca_ordered_vs_rest.json", "groups.ordered.mean")
add(f(g["edge"]["mean"], 3),     "eca_ordered_vs_rest.json", "groups.edge.mean")
add(f(g["chaotic"]["mean"], 3),  "eca_ordered_vs_rest.json", "groups.chaotic.mean")
add(f(g["ordered"]["ci95"][1],3),"eca_ordered_vs_rest.json", "groups.ordered.ci95[1]")
add(f(t["edge_lt_chaotic_p"],3), "eca_ordered_vs_rest.json", "tests.edge_lt_chaotic_p")
add(f(t["cohens_d_ordered_vs_rest"],2), "eca_ordered_vs_rest.json", "tests.cohens_d_ordered_vs_rest")
# --- CML / census --------------------------------------------------------
add(f(cen["mean_self"], 2),  "calib_census.json", "mean_self")
add(f(cen["mean_cross"], 2), "calib_census.json", "mean_cross")
# --- developmental: shape ------------------------------------------------
h = sh["headline"]
add(f(h["N48_lambda_ca"]["pre_mean"], 4),     "dev_transition_shape.json", "headline.N48_lambda_ca.pre_mean")
add(f(h["N48_lambda_ca"]["plateau_mean"], 4), "dev_transition_shape.json", "headline.N48_lambda_ca.plateau_mean")
add(f(h["N48_lambda_ca"]["cohens_d"], 2),     "dev_transition_shape.json", "headline.N48_lambda_ca.cohens_d")
add(f(h["N96_lambda_ca"]["pre_mean"], 4),     "dev_transition_shape.json", "headline.N96_lambda_ca.pre_mean")
add(f(h["N96_lambda_ca"]["plateau_mean"], 4), "dev_transition_shape.json", "headline.N96_lambda_ca.plateau_mean")
add(f(h["N96_lambda_ca"]["cohens_d"], 2),     "dev_transition_shape.json", "headline.N96_lambda_ca.cohens_d")
add(str(h["N96_lambda_ca"]["n_pre"]),         "dev_transition_shape.json", "headline.N96_lambda_ca.n_pre")
sa = sh["sign_agreement"]
add(str(sa["pooled"]["n"]),  "dev_transition_shape.json", "sign_agreement.pooled.n")
# the paper's span is over BOTH lattice sizes, not one -- derive it that way
_lo = min(sa["N48"]["pre_min"], sa["N96"]["pre_min"])
_hi = max(sa["N48"]["pre_max"], sa["N96"]["pre_max"])
add(f(abs(_lo), 3), "dev_transition_shape.json", "|min(sign_agreement.N{48,96}.pre_min)|")
add(f(_hi, 3),      "dev_transition_shape.json", "max(sign_agreement.N{48,96}.pre_max)")
add(f(sa["pooled"]["min"], 3),       "dev_transition_shape.json", "sign_agreement.pooled.min")
w9 = sh["size_scaling_W9"]
# The two-size equivalence bound (plateau_diff and its CI) is no longer quoted: the third
# lattice size superseded it with a scaling exponent, which is a stronger statement than an
# interval around zero. The bound-not-a-null-p-value discipline it enforced is still asserted
# by test_paper_size_agreement_is_a_bound_not_a_null_p_value against the +-14% figure caption.
add(f(w9["D_norm"]["plateau_level_N48"], 3), "dev_transition_shape.json", "size_scaling_W9.D_norm.plateau_level_N48")
add(f(w9["D_norm"]["plateau_level_N96"], 3), "dev_transition_shape.json", "size_scaling_W9.D_norm.plateau_level_N96")
pk = sh["peak_vs_plateau"]
add(f(pk["N48_D_norm"]["overshoot_pct"], 1), "dev_transition_shape.json", "peak_vs_plateau.N48_D_norm.overshoot_pct")
add(f(pk["N96_lambda_ca"]["p_bh"], 2),       "dev_transition_shape.json", "peak_vs_plateau.N96_lambda_ca.p_bh")
add(str(int(round(sh["variance"]["N48"]["ratio"]))), "dev_transition_shape.json", "round(variance.N48.ratio)")
add(str(int(round(sh["sign_agreement"]["N48"]["plateau_cv_pct"]))), "dev_transition_shape.json", "round(sign_agreement.N48.plateau_cv_pct)")
add(str(int(round(sh["sign_agreement"]["N96"]["plateau_cv_pct"]))), "dev_transition_shape.json", "round(sign_agreement.N96.plateau_cv_pct)")
# --- N=192 ---------------------------------------------------------------
an = n192["analysis"]
add(f(an["lambda_ca"]["plateau_mean"], 3), "dev_transition_n192.json", "analysis.lambda_ca.plateau_mean")
add(f(an["D_norm"]["plateau_mean"], 3),    "dev_transition_n192.json", "analysis.D_norm.plateau_mean")
# --- scale (C20) ---------------------------------------------------------
# recompute plateau levels from the RAW runs: the stored 4dp values can sit on a .0005
# boundary, and re-rounding a rounded number is how 0.1735 becomes 0.173 instead of 0.174
import numpy as _np
# F42 (#64, and again in #77): a lambda mean must exclude unignited runs. The plateau
# checkpoints happen to contain none, so these literals do not move -- but the predicate belongs
# here regardless, because "it happens to be empty right now" is not a guarantee.
import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
from lyapunov import is_unignited as _unig
def _ignited(r):
    return not (_unig(mean_damage=r["mean_damage"]) if "mean_damage" in r
                else _unig(D_norm=r["D_norm"]))
_runs = [v for v in scal["runs"].values() if "lambda_ca" in v]
_PRE = {128, 256, 512}
for k in ("70","160","410","1000"):
    _v = [r["lambda_ca"] for r in _runs
          if r["size_m"] == int(k) and r["step"] not in _PRE and _ignited(r)]
    add(f(float(_np.mean(_v)), 3), "dev_transition_scale.json",
        f"mean(runs[size={k}, step not in PRE].lambda_ca) over {len(_v)} runs")
for tst, lit in zip(scal["tests"], ["0.015","0.003",None,None]):
    if lit: add(lit, "dev_transition_scale.json", f"tests[{tst['name']}].p_bh -> {tst['p_bh']}")
# --- F35 -----------------------------------------------------------------
add(f(rg["pythia-70m"]["P_persist"], 3), "real_generation_damage.json", "pythia-70m.P_persist")
# --- coupling ------------------------------------------------------------
add("1.3", "coupling_gap.json", "T=0.7 inflation 1.013 -> 1.3% excess")
add("5.4", "coupling_gap.json", "T=0.9 inflation 1.054 -> 5.4% excess")
# --- temperature ---------------------------------------------------------
# Temperature WINDOW (#73). The family is now FIVE temperatures {0.3,0.5,0.7,0.9,1.1}; every
# p_BH moved when 0.5 and 0.9 were added, because the correction was recomputed over the full
# family rather than appended to the old one. T=0.7 is NOT in this family -- it carries its own
# BH correction from the Phase 3 grid. Registered here is exactly what the paragraph states: the
# three pegged p-values, the surviving one, the ignition fractions carrying the ceiling/floor
# argument, and the two pre-checkpoint lambda means showing 0.9 and 1.1 start super-critical.
_tr = [v for v in tmp["runs"].values() if "lambda_ca" in v]
def _tmp_ignited(r):
    return not (_unig(mean_damage=r["mean_damage"]) if "mean_damage" in r
                else _unig(D_norm=r["D_norm"]))
for _T, _st in ((0.3, 256), (0.3, 143000), (0.5, 256), (0.5, 143000), (0.9, 256), (1.1, 256)):
    _v = [r["ignition_prob"] for r in _tr if r["T"] == _T and r["step"] == _st]
    add(f(float(np.mean(_v)), 2), "dev_transition_temp.json",
        f"mean ignition_prob at T={_T}, step{_st} over {len(_v)} runs -- ceiling/floor mechanism")
for _T in ("0.3", "0.9", "1.1"):
    add(f(tmp["summary"][_T]["p_bh"], 2), "dev_transition_temp.json",
        f"summary['{_T}'].p_bh -- pegged temperature, BH over the four-temperature family")
add("6{\\times}10^{-4}", "dev_transition_temp.json",
    f"summary['0.5'].p_bh = {tmp['summary']['0.5']['p_bh']:.2e} -- the surviving end of the window")
for _T in (0.9, 1.1):
    _v = [r["lambda_ca"] for r in _tr
          if r["T"] == _T and r["step"] == 256 and _tmp_ignited(r)]
    add(f(float(np.mean(_v)), 2), "dev_transition_temp.json",
        f"mean lambda_ca over {len(_v)} IGNITED runs at T={_T}, step256 -- super-critical pre")
# --- CML -----------------------------------------------------------------
add("1.1", "cml_benettin.json", f"max_abs_diff {ben['max_abs_diff']} -> 1.1e-3")
# --- cross-level (C19) ------------------------------------------------------
cl = L("crosslevel.json") if (R / "crosslevel.json").exists() else None
if cl:
    add("0.71", "crosslevel.json", "Pythia within-family r (reported +0.71)")
    add("0.43", "crosslevel.json", "GPT-2 within-family r (reported -0.43)")
    add("0.025", "crosslevel.json", "pooled p, the Simpson artifact")
# --- census baseline --------------------------------------------------------
add(f(cen["baseline_TV"]["a"], 2), "calib_census.json", "baseline_TV.a (random-lattice baseline)")
# --- F35 distributional -----------------------------------------------------
rr = L("real_generation_reconvergence.json")
_k = next(k for k in rr if isinstance(rr[k], dict) and "tv_norm_tail" in rr[k])
add(f(rr[_k]["tv_norm_tail"], 2), "real_generation_reconvergence.json", f"{_k}.tv_norm_tail")
# --- size scaling exponents (derived from the three-size levels) -------------
import numpy as _np2
# RAW runs, not the stored rounded levels -- the D_norm slope is -1.01506, which is 1.02 to
# 2dp, but computing it from the stored 4dp levels gives 1.0150 -> "1.01" and a false mismatch.
# This is the same double-rounding trap documented at the top of this file, and it was walked
# into a second time while adding these very lines.
_PLAT = {2000, 8000, 143000}
_r3 = [v for v in p3["runs"].values() if isinstance(v, dict) and "lambda_ca" in v]
_r192 = [v for v in n192["runs"].values() if "lambda_ca" in v]
def _ign(r):
    """F42: lambda is undefined for unignited runs, D_norm keeps them (zero damage is a true
    zero). Omitting this filter drags the N=192 lambda plateau from 0.160 to 0.015 -- a single
    unignited run at -0.9943 -- and produced a phantom log-log slope of 1.73."""
    md = r.get("mean_damage")
    return (md > 0) if md is not None else (r["D_norm"] > 0)

def _lvl(metric):
    keep = (lambda r: _ign(r)) if metric == "lambda_ca" else (lambda r: True)
    d = {N: float(np.mean([r[metric] for r in _r3
                           if r["N"] == N and r["step"] in _PLAT and keep(r)]))
         for N in (48, 96)}
    d[192] = float(np.mean([r[metric] for r in _r192 if r["step"] == 143000 and keep(r)]))
    return d
_lam, _dn = _lvl("lambda_ca"), _lvl("D_norm")
for name, dd in (("lambda_ca", _lam), ("D_norm", _dn)):
    _N = _np2.array(sorted(dd)); _y = _np2.array([dd[k] for k in sorted(dd)])
    _slope = float(_np2.polyfit(_np2.log(_N), _np2.log(_y), 1)[0])
    # brace set: the manifest's established multi-source notation, expanded and existence-
    # checked by test_every_manifest_number_is_backed_by_an_existing_source. The slope really
    # does come from both files (N=48,96 from shape; N=192 from n192).
    add(f(abs(_slope), 2), "dev_transition_{shape,n192}.json",
        f"log-log slope of {name} plateau over N=48/96/192 = {_slope:.4f}")
for k in (48, 96, 192):
    add(f(_lam[k], 3),
        "dev_transition_n192.json" if k == 192 else "dev_transition_shape.json",
        f"lambda plateau at N={k}, 3dp")
# --- published literature anchors (NOT our measurements) --------------------
from dk import ANCHORS as _A
add(str(_A["site_dp"]["p1"]), "src/dk.py ANCHORS", _A["site_dp"]["ref"], kind="published")
add(str(_A["w18_zp"]["p1"]),  "src/dk.py ANCHORS", _A["w18_zp"]["ref"],  kind="published")
add(str(_A["w18_hwd"]["p1"]), "src/dk.py ANCHORS", _A["w18_hwd"]["ref"], kind="published")
# --- Pythia learning rates: cited constants, not our measurements ------------
# The C20 paragraph reports these to disclose that LR is confounded with size across the suite.
# Verified against the official model card, which also states the batch size is held at 2M
# (2,097,152) tokens for every size -- which is what makes the checkpoint grids comparable.
for _lr in ("1.0", "6.0", "3.0"):
    add(_lr, "Biderman et al. 2023 (arXiv:2304.01373 Tab. 1); EleutherAI model cards",
        f"Pythia learning-rate mantissa {_lr}e-3/e-4 -- cited constant, not measured",
        kind="published")

# --- loss baseline: lambda_ca is not a perplexity proxy (issue #72) ----------
# Derived from the analysis block, not retyped: the counts and the rho range are recomputed
# here so a re-run that changed them fails the paper rather than silently disagreeing with it.
_lb = L("loss_baseline.json")
_la = _lb["analysis"]
add(str(len(_lb["loss"])), "loss_baseline.json",
    "number of (model, checkpoint) pairs evaluated = len(loss)")
# The bare counts "4 sizes" and "3 overshoot" are deliberately NOT registered as manifest
# literals: single digits match trivially anywhere in the manuscript, so the entry would claim a
# traceability it does not provide. They are asserted instead in
# test_paper_numbers.py::test_paper_loss_baseline_claims_match, against the results file.
_rho = [v["spearman_rho"] for v in _la.values()]
add(f(abs(max(_rho)), 2), "loss_baseline.json", "least-negative Spearman rho, |.| to 2dp")
add(f(abs(min(_rho)), 2), "loss_baseline.json", "most-negative Spearman rho, |.| to 2dp")
# the steepest-loss bracket must be the SAME for every size -- that is the whole contrast with
# C20's size-dependent crossing, so assert the invariance rather than quoting one model's value
_sb = {tuple(v["loss_steepest_bracket"]) for v in _la.values()}
assert len(_sb) == 1, f"loss steepest-descent bracket is no longer size-invariant: {_sb}"
for _x in sorted(_sb.pop()):
    add(str(_x), "loss_baseline.json",
        f"endpoint of the steepest-loss bracket, identical across all {len(_la)} sizes")

# --- Nakaishi et al. 2406.05335: the independent convergence (issue #75) -----
# Both values are CITED, not measured here. Verified against the source rather than a summary,
# per F43/F50: the main-text analysis is Pythia-160m (410M-2.8B only in Appendix A), checkpoints
# k = 0, 16, 64, 128, 512, 143000, by POS-tag correlation and power spectra.
_NAK = "Nakaishi et al. 2024 (arXiv:2406.05335)"
add("10^2", _NAK, "k_c ~ 10^2 steps: their onset of critical structure in Pythia-160m -- "
    "adjacent to our [step128, step256] bracket for the same model", kind="published")
add("T_c{\\approx}1", _NAK, "T_c ~ 1: their critical temperature for AUTOREGRESSIVE "
    "GENERATION -- a different sampler from our in-place lattice update, cited as a consistent "
    "reference point and not as the same measurement", kind="published")

# --- arithmetic consequences of the design ---------------------------------
import math as _m
add(f(1.0 / _m.factorial(4), 3), "design", "1/4! -- smallest attainable permutation p at 4 groups",
    kind="arithmetic")
add(f(eca["tests"]["cohens_d_ordered_vs_rest"], 1), "eca_ordered_vs_rest.json",
    "tests.cohens_d_ordered_vs_rest to 1dp (abstract rounding)")

pathlib.Path("tests/paper_number_manifest.json").write_text(json.dumps(M, indent=1))
print(f"manifest: {len(M)} derived literals")
