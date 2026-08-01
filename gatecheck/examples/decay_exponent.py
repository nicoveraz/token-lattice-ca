"""gatecheck worked example: measuring a decay exponent without fooling yourself.

The toy science: a process whose observable decays as y(t) ~ t^(-delta). We want delta for an
"unknown" system, from short noisy series. Every trap in this script is one textca paid for:
a gate calibrated at the wrong geometry (F56), a null test that could pass vacuously, a grid
pseudoreplicated from two seeds (W1), an unregistered inflated variant (F39), and a results
file whose analysis code drifted after the run (#38).

Run:  PYTHONPATH=src python3 examples/decay_exponent.py
"""
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gatecheck import (
    Gate, gated, Preregistration, Manifest,
    certify_null, independence_report, save_results, slope_loglog,
)
from gatecheck.prereg import quarantine, evaluate_kills
from gatecheck import testing

TRUE_DELTA = 0.16          # the reference system's known exponent (think: DP's delta)
FIT_FROM = 3


def simulate(delta, n_steps, n_replicas, seed):
    """One experiment: replica-averaged noisy power-law decay."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_steps + 1, dtype=float)
    y = t[None, :] ** (-delta) * np.exp(rng.normal(0, 0.35, size=(n_replicas, n_steps)))
    return t, y.mean(axis=0)


def estimate_delta(t, y):
    """THE estimator — one implementation, shared by the gate and the measurement."""
    m = t >= FIT_FROM
    slope, _ = slope_loglog(t[m], y[m])
    return None if slope is None else -slope


def main():
    root = pathlib.Path(tempfile.mkdtemp(prefix="gatecheck_demo_"))
    # a realistic layout: the analysis script lives INSIDE the project root it stamps
    analysis_script = root / "analyze.py"
    analysis_script.write_text(pathlib.Path(__file__).read_text())
    print(f"demo project root: {root}\n")

    # -- 0. preregister, before any target data exist -------------------------------------
    prereg = Preregistration(
        name="decay-exponent-demo",
        hypotheses={"delta_hat": "point estimate of the unknown system's decay exponent"},
        frozen={"fit_from": FIT_FROM, "seeds": list(range(8)), "tolerance_pct": 10.0},
        kill_conditions={"K1": "estimator returns None (undefined fit) on the target data"},
        independent_unit="seed",
    )
    block = prereg.block()
    print(f"[prereg] frozen + self-hashed: {block['sha256'][:12]}...")

    # -- 1. the exact-null certification (CRN twins through the same pipeline) ------------
    t, y_a = simulate(0.3, 60, 32, seed=123)
    _, y_b = simulate(0.3, 60, 32, seed=123)        # same seed = same noise stream
    _, y_p = simulate(0.3, 60, 32, seed=999)        # a perturbed arm that must differ
    cert = certify_null(y_a - y_b, y_a - y_p)
    print(f"[null]   twins identical, effect arm nonzero -> certified non-vacuous "
          f"(|effect|={cert.effect_magnitude:.3f})")

    # -- 2. the calibration gate, at the measurement's own geometry -----------------------
    def on_reference(geometry, seed):
        n_steps, n_replicas = geometry
        return estimate_delta(*simulate(TRUE_DELTA, n_steps, n_replicas, seed))

    gate = Gate(on_reference, TRUE_DELTA, tolerance_pct=10.0, name="decay-gate")
    seeds = range(20)

    starved = gate.check(geometry=(12, 4), seeds=seeds)      # the geometry we can afford
    generous = gate.check(geometry=(120, 64), seeds=seeds)   # the geometry that decides
    for c in (starved, generous):
        q = c.quantities["value"]
        print(f"[gate]   geometry {c.geometry}: dev {q['dev_pct']:.1f}% "
              f"+ spread {q['spread_pct']:.1f}% vs tol {c.tolerance_pct}% "
              f"-> {'PASS' if c.passes else 'FAIL'}")

    # -- 3. the gated measurement: NOT_DECIDABLE is an answer -----------------------------
    UNKNOWN_DELTA = 0.21                                     # nature's secret
    for check in (starved, generous):
        n_steps, n_replicas = check.geometry
        verdict = gated(check, measure=lambda g=check.geometry: estimate_delta(
            *simulate(UNKNOWN_DELTA, g[0], g[1], seed=7)))
        shown = f"delta_hat = {verdict.value:.3f}" if verdict.decided else verdict.status
        print(f"[verdict] at geometry {check.geometry}: {shown}")
    final = gated(generous, measure=lambda: estimate_delta(
        *simulate(UNKNOWN_DELTA, 120, 64, seed=7)))

    kills = evaluate_kills(block, {"K1": lambda: final.value is None})
    print(f"[prereg] kill conditions fired: {kills['fired'] or 'none'}")

    # -- 4. the pseudoreplication trap, caught by accounting ------------------------------
    values, units = [], []
    for seed in (21, 22):                                    # W1: a 15-cell grid from 2 seeds
        base = estimate_delta(*simulate(UNKNOWN_DELTA, 120, 64, seed=seed))
        for cell in range(15):
            values.append(base + 1e-4 * cell)                # cells share the seed's noise
            units.append(seed)
    rep = independence_report(values, units, unit_name="seed")
    print(f"[units]  {rep.message()}")

    # -- 5. write a results file that carries its own audit trail -------------------------
    results = {"delta_hat": round(final.value, 3), "fit_from": FIT_FROM,
               "seeds": list(range(8)), "tolerance_pct": 10.0,
               "gate": final.gate.block(), "null_certificate": cert.block()}
    quarantine(results, "delta_hat_peak_window",
               round(final.value * 1.15, 3),
               "unregistered variant (kindest fit window); kept for audit, not a finding")
    out = root / "results" / "decay.json"
    save_results(out, results, script=analysis_script, root=root,
                 prereg=block, independent_unit="seed")
    testing.assert_fresh(out, root)
    print(f"[prov]   results stamped and verified fresh: {out.name}")
    analysis_script.write_text("# edited after the run\n")
    try:
        testing.assert_fresh(out, root)
    except AssertionError:
        print("[prov]   after editing analyze.py the same check goes red (as it must)")
    analysis_script.write_text(pathlib.Path(__file__).read_text())   # restore

    # -- 6. the manuscript cannot drift from the results file -----------------------------
    (root / "paper.md").write_text(
        f"We measure delta = {results['delta_hat']:.3f} behind a passing calibration gate.")
    m = Manifest()
    m.add(f"{results['delta_hat']:.3f}", "results/decay.json", path="delta_hat", fmt=".3f")
    m.save(root / "manifest.json")
    testing.assert_manifest(root / "manifest.json", root / "paper.md", root)
    print("[manifest] paper <-> manifest <-> results agree")

    print("\nEverything above is the point: the starved geometry returned NOT_DECIDABLE "
          "instead of a number,\nthe 30-observation grid was accounted as ~2 independent "
          "observations, and the inflated variant\nis quarantined where it can be audited "
          "but not quoted.")


if __name__ == "__main__":
    main()
