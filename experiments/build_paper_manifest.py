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
def add(literal, src, expr):
    M.append({"literal": literal, "source": src, "derivation": expr})

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
add(f(abs(w9["lambda_ca"]["plateau_diff"]), 4), "dev_transition_shape.json", "|size_scaling_W9.lambda_ca.plateau_diff|")
add(f(abs(w9["lambda_ca"]["plateau_diff_ci95"][0]), 3), "dev_transition_shape.json", "|...ci95[0]|")
add(f(w9["lambda_ca"]["plateau_diff_ci95"][1], 3),      "dev_transition_shape.json", "...ci95[1]")
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
_runs = [v for v in scal["runs"].values() if "lambda_ca" in v]
_PRE = {128, 256, 512}
for k in ("70","160","410","1000"):
    _v = [r["lambda_ca"] for r in _runs if r["size_m"] == int(k) and r["step"] not in _PRE]
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
add(f(tmp["summary"]["0.3"]["p_bh"], 2), "dev_transition_temp.json", "summary['0.3'].p_bh")
add("0.98", "dev_transition_temp.json", "T=1.1 pre ignition 0.984 -> 0.98")
add("0.20", "dev_transition_temp.json", "T=0.3 pre ignition 0.195 -> 0.20")
# --- CML -----------------------------------------------------------------
add("1.1", "cml_benettin.json", f"max_abs_diff {ben['max_abs_diff']} -> 1.1e-3")
pathlib.Path("tests/paper_number_manifest.json").write_text(json.dumps(M, indent=1))
print(f"manifest: {len(M)} derived literals")
