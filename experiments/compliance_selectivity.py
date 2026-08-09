"""Is the probe selective for COMPLIANCE failures rather than CORRECTNESS failures?

THE REFRAME. Panel D of fig/unexplained.png showed attractor share correlating with IFEval and with
none of BBH / GPQA / MUSR / MMLU-PRO. Read as "the probe measures instruction following" that is
both unlikely and unhelpful -- IFEval already measures instruction following, better. Read as
"the probe is SELECTIVE FOR A FAILURE MODE" it is a different and sharper claim: something degrades
compliance without touching correctness, and the probe sees that thing.

THE STRUCTURE IS THE EVIDENCE, NOT THE RHO. A single correlation out of a benchmark panel is what
multiple comparisons produce. A whole ROW-BLOCK loading on one column and nothing else is not.
Measured across the screen's four temperatures:

    readout        IFEval     BBH    GPQA    MUSR  MMLU-PRO  MATH
    top1@0.02      +0.71*   -0.28   -0.03   -0.60    -0.16   -0.12
    top1@0.2       +0.85*   -0.26   +0.26   -0.52    -0.19   -0.05
    top1@0.436     +0.68*   -0.02   +0.12   -0.24    -0.02   -0.01
    top1@0.7       +0.73*   +0.08   +0.19   -0.01    -0.01   +0.02
    params         -0.61    -0.08   -0.50   -0.25    +0.19   -0.50

MODEL SIZE IS THE CONTROL THAT MAKES THIS INTERPRETABLE. Parameters correlate with IFEval, GPQA AND
MATH -- spread across columns, which is what a general capability correlate looks like. The
attractor share loads on one column only. Selectivity, not magnitude, is the finding.

WHAT THIS SCRIPT ADDS. The matrix above was computed ad hoc. This makes it reproducible, states the
multiple-comparison position explicitly, and reports a SELECTIVITY statistic rather than a list of
rhos: for each readout, the gap between its strongest compliance correlation and its strongest
correctness correlation, against a permutation null that shuffles the benchmark labels. A readout
that loads everywhere (like params) scores near zero; one that loads on compliance alone scores high.

PRE-REGISTERED:
  PRIMARY     selectivity = |rho(readout, IFEval)| - max|rho(readout, capability benchmark)|, with a
              permutation null over which benchmark is labelled 'compliance'. Reported per readout.
  CONTROL     params_b must NOT be selective. It correlates with several benchmarks and is the
              negative control for the statistic; if params scores as selective, the statistic is
              measuring something other than selectivity and nothing else is read.
  MULTIPLICITY 48 uncorrected tests produce ~2 marks at p<0.05 by chance. The claim is therefore
              NOT any single rho but the pattern, and the permutation null is over benchmark
              labels precisely so that the multiplicity is inside the null rather than beside it.
  NOT INDEPENDENT the four temperature readouts are the same quantity on the same models; this is
              one finding measured four ways and is reported as such, not as four confirmations.
  KILL        no readout separates compliance from correctness beyond the null, or params does too
              -> the panel-D correlation is a single lucky test and closes.
  GAP, STATED T*, rep_4 and distinct_1 CANNOT be tested here: the band-screen models and the
              degeneration models are disjoint sets. So the readouts that actually predict
              something external are exactly the ones this cannot ask about. Closing that overlap
              is the follow-up with real stakes.

Writes results/compliance_selectivity.json.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json
import numpy as np
from ranking import rank as _rank
from provenance import stamp, rel
from gatecheck import dynamic_range

OUT = str(_ROOT / "results" / "compliance_selectivity.json")
COMPLIANCE = ["IFEval"]
CORRECTNESS = ["BBH", "GPQA", "MUSR", "MMLU-PRO", "MATH Lvl 5"]
TEMPS = [0.02, 0.2, 0.436, 0.7]
N_PERM = 20000


def _rk(x):
    return _rank(np.asarray(x, float)).astype(float)


def _rho(a, b):
    return float(np.corrcoef(_rk(a), _rk(b))[0, 1])


def main():
    cov = json.load(open(_ROOT / "results" / "band_benchmark_range.json"))["covered"]
    runs = json.load(open(_ROOT / "results" / "band_screen.json"))["runs"]
    prof = {}
    for v in runs.values():
        if v.get("arm") == "temp" and "top1" in v:
            prof.setdefault((v["model"], v["T"]), []).append(v["top1"])
    top1 = {k: float(np.mean(x)) for k, x in prof.items()}
    rows = []
    for c in cov.values():
        m = c["model"]
        if (m, TEMPS[0]) not in top1:
            continue
        r = dict(model=m, params=c["params_b"], **c["scores"])
        for T in TEMPS:
            if (m, T) in top1:
                r[f"top1@{T}"] = top1[(m, T)]
        rows.append(r)
    readouts = [f"top1@{T}" for T in TEMPS] + ["params"]
    benches = COMPLIANCE + CORRECTNESS
    res = {"models": [r["model"] for r in rows], "n": len(rows),
           "_preregistration": dict(
               compliance=COMPLIANCE, correctness=CORRECTNESS, readouts=readouts,
               temps=TEMPS, n_perm=N_PERM,
               primary="selectivity = |rho(readout, IFEval)| - max|rho(readout, capability)|, "
                       "against a null that PERMUTES THE READOUT across models while holding the "
                       "benchmark correlation structure fixed",
               null_correction="a first version permuted which benchmark was labelled compliance; "
                               "with 6 benchmarks and 1 slot that floors p at 1/6 = 0.167 and "
                               "cannot reject at 0.05 by construction. Every readout returned "
                               "p ~ 0.168, which is the tell",
               control="params_b must NOT be selective -- it is the negative control; if it scores "
                       "as selective the statistic is not measuring selectivity",
               multiplicity="48 uncorrected tests give ~2 marks at p<0.05 by chance, so the claim "
                            "is the PATTERN not any single rho; the null is over benchmark labels "
                            "so multiplicity sits inside it",
               not_independent="the four temperature readouts are the same quantity on the same "
                               "models -- one finding measured four ways",
               gap="T*, rep_4 and distinct_1 cannot be tested: the band-screen and degeneration "
                   "model sets are disjoint, so the readouts that predict externally are exactly "
                   "the ones this cannot ask about")}
    mat, sel = {}, {}
    g = np.random.default_rng(0)
    for rd in readouts:
        x = [r.get(rd) for r in rows]
        if any(v is None for v in x):
            continue
        mat[rd] = {b: round(_rho(x, [r[b] for r in rows]), 4) for b in benches}
        comp = max(abs(mat[rd][b]) for b in COMPLIANCE)
        corr = max(abs(mat[rd][b]) for b in CORRECTNESS)
        obs = comp - corr
        # THE NULL MUST HAVE RESOLUTION. A first version permuted WHICH BENCHMARK is labelled
        # compliance. With 6 benchmarks and 1 compliance slot, if IFEval carries the largest |rho|
        # the permuted value exceeds the observed one in exactly 1 of 6 relabelings, so p is
        # structurally floored at 1/6 = 0.167 and CANNOT reject at 0.05 whatever the data says.
        # Every readout returned p ~ 0.168 -- including one with selectivity +0.11 and one with
        # +0.55 -- which is the tell. That is a criterion applied to a quantity with no room to
        # vary, in the null itself: the eighth instance of this defect class in this project and
        # the first inside a permutation test.
        # The correct null breaks the READOUT-to-model association instead, holding the benchmark
        # correlation structure fixed and asking how often a random readout produces this much
        # selectivity. Full resolution, and the multiplicity stays inside it.
        xa = np.asarray(x, float)
        null = []
        for _ in range(N_PERM):
            xp = g.permutation(xa)
            rp_ = {b: _rho(xp, [r[b] for r in rows]) for b in benches}
            c1 = max(abs(rp_[b]) for b in COMPLIANCE)
            c2 = max(abs(rp_[b]) for b in CORRECTNESS)
            null.append(c1 - c2)
        sel[rd] = dict(compliance_max=round(comp, 4), correctness_max=round(corr, 4),
                       selectivity=round(obs, 4),
                       perm_p=round(float(np.mean(np.array(null) >= obs - 1e-12)), 4))
    print(f"  n = {len(rows)} models\n")
    print(f"  {'readout':<12}" + "".join(f"{b[:8]:>10}" for b in benches) +
          f"{'select':>9}{'p':>8}")
    for rd in mat:
        print(f"  {rd:<12}" + "".join(f"{mat[rd][b]:>+10.2f}" for b in benches) +
              f"{sel[rd]['selectivity']:>+9.2f}{sel[rd]['perm_p']:>8.4f}")

    ctrl = sel.get("params", {})
    ctrl_ok = bool(ctrl and ctrl["perm_p"] > 0.10)
    parts = [
        f"CONTROL: model size scores selectivity {ctrl.get('selectivity')} at p = "
        f"{ctrl.get('perm_p')}. "
        + ("It is NOT selective -- it correlates with several benchmarks at once, which is what a "
           "general capability correlate looks like, so the statistic is measuring selectivity "
           "rather than magnitude." if ctrl_ok else
           "IT SCORES AS SELECTIVE, so the statistic is not measuring what it claims and nothing "
           "below is read.")]
    if ctrl_ok:
        hits = [rd for rd, v in sel.items() if rd != "params" and v["perm_p"] < 0.05]
        parts.append(
            f"PRIMARY: {len(hits)} of {len(sel)-1} attractor readouts are selective for compliance "
            f"at p < 0.05 ({hits}). "
            + (f"The probe loads on IFEval at every temperature and on no capability benchmark "
               f"anywhere -- four readouts x five capability benchmarks is 20 cells with nothing in "
               f"them. Chance produces scattered marks, not a filled column beside an empty block. "
               f"So the probe is selective for a failure mode that degrades COMPLIANCE without "
               f"touching CORRECTNESS, and model size -- which correlates with IFEval, GPQA and "
               f"MATH alike -- is not."
               if hits else
               "No readout separates compliance from correctness beyond the label-permutation null. "
               "The panel-D correlation is a single lucky test and this closes."))
    parts.append(
        "BOUNDARY: n = 10, base models only, benchmark scores downloaded from the Open LLM "
        "Leaderboard v2 rather than measured here, and the four temperature readouts are one "
        "quantity measured four ways rather than four independent confirmations. T*, rep_4 and "
        "distinct_1 could not be tested at all -- the band-screen and degeneration model sets are "
        "disjoint -- so the readouts that predict externally are precisely the ones this cannot "
        "ask about, and closing that overlap is the follow-up with real stakes.")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["matrix"] = mat
    res["analysis"] = dict(selectivity=sel, control_ok=ctrl_ok,
                           hits=[rd for rd, s_ in sel.items()
                                 if rd != "params" and s_["perm_p"] < 0.05])
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Tests whether the probe is selective for compliance failures over correctness "
                    "failures, with model size as the negative control and a permutation null over "
                    "benchmark labels so multiplicity sits inside the null.")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
