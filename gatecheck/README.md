# gatecheck

A measure-and-falsification toolbox for empirical research code. Extracted and generalized
from the **textca** project, whose measurement discipline caught six confident wrong verdicts
before any of them reached a paper — each one by a check that lives, generalized, in this
package.

The one-line philosophy: **an estimator must reproduce a known answer, at the geometry it will
actually be used at, before its verdict on an unknown counts** — and when it cannot, the honest
return value is `NOT_DECIDABLE`, not a number.

## Install

Not yet on PyPI (check the name for collisions before publishing; see DESIGN.md §7). From this
directory:

```bash
pip install -e ".[test]"
python -m pytest            # 42 tests, no network, < 1 s
python examples/decay_exponent.py
```

Dependencies: numpy only. Python ≥ 3.10.

## What is in the box

| Module | Pattern | The incident it descends from |
|---|---|---|
| `gatecheck.gate` | Calibration gates: run the *same* estimator on a known-answer reference at the *same* geometry, blind to the target; pass only when deviation **plus** seed spread clears tolerance; return `NOT_DECIDABLE` otherwise | textca F56: a tolerance calibrated at N=512 rejected the reference class itself at N=96 |
| `gatecheck.provenance` | Results files that hash the analysis source, its project-local import closure, and the environment; a test recomputes the hashes | textca #38/F45/F46: a mid-run edit silently inverted a written conclusion, twice |
| `gatecheck.prereg` | Self-hashed preregistration blocks, mechanically evaluated kill conditions, quarantine keys for unregistered variants, a smuggling guard | textca F39/F76: the null that was "a good result" was good *because* the kill condition predated the data |
| `gatecheck.units` | Declared independent unit + ICC/design-effect accounting: how many observations do you *actually* have | textca W1/F57: 15 grid cells from 2 seeds ("p<10⁻⁴") and 512 replicas sharing one visit order (error bars 8× too small) |
| `gatecheck.fits` | Scan-edge rejection for fitted optima; the shared log-log slope estimator | textca F59: a fitted z "excluding" the reference class was the scan-grid edge |
| `gatecheck.nulltest` | Exact-null assertions (no tolerance parameter, deliberately), anti-vacuity certification, control-adjusted verdicts | textca's CRN discipline; F65: the control acquired the effect, reclassifying it as generic |
| `gatecheck.leverage` | Does this quantity have room to carry the claim being made about it? Dynamic range, noise gates, directionality, distinct units — composed into `NOT_DECIDABLE` with the binding reason attached | textca F80/F88/F93/F94/F96: one defect class — a statistically shaped criterion applied to a quantity with no room to vary — caught by hand six times, three of them *after* preregistration |
| `gatecheck.ranking` | Tie-aware ranks and Spearman; a zero-variance input returns `nan` rather than a plausible float | textca F119: `argsort(argsort(x))` ranks a constant vector by input position, and reported ρ = +0.829 for a quantity measured at exactly 0.000 across 24 checkpoints |
| `gatecheck.state` | Keep the object a measurement was reduced *from*, JSON-safe, with an explicit downsample rule that refuses to stride the structure-carrying axis | textca F116 + the remote share campaign + F136: three instances of the same mechanism — the largest object the run produces is discarded, so its degeneracy is undetectable and every re-question costs a re-run |
| `gatecheck.discriminator` | The two-axis test: vary the construction with the model fixed, vary the model with the construction fixed, and decide whether a readout is `MODEL_DETERMINED`, `CONSTRUCTION_DETERMINED` or `NOT_DECIDABLE`. Loopness as an explicit parameter. See [PROTOCOL.md](PROTOCOL.md) | textca F128/F129/F130: two readouts on the same lattice, models and seeds — one has a 30× construction-to-model range ratio and a ranking stable at 0.030, the other ranks models identically across all six constructions |
| `gatecheck.manifest` | Every load-bearing literal in a manuscript traced to a results file — presence *and* recompute | textca #48/#64: a measurement silently dropped in a page-fit trim; caught by manifest disagreement |
| `gatecheck.results` | One save/load pair wiring stamps, prereg, unit declaration, and the absolute-path leak guard together | textca #52: twelve machine-written logs shipped the checkout path into a published artifact |
| `gatecheck.testing` | Test-suite helpers: staleness sweeps, retraction guards, manifest assertions, single-implementation greps | textca #27: a retracted ordering was still asserted paper-wide and plotted in two figures |

## Sixty-second tour

```python
import numpy as np
from gatecheck import Gate, gated, certify_null, independence_report, save_results
from gatecheck import testing

# 1. A calibration gate around YOUR estimator, on a reference with a KNOWN answer.
def on_reference(geometry, seed):
    data = make_reference_data(known_answer=0.16, geometry=geometry, seed=seed)
    return my_estimator(data)                     # the SAME estimator you will use for real

gate = Gate(on_reference, truth=0.16, tolerance_pct=10.0)
check = gate.check(geometry=my_geometry, seeds=range(20))   # blind: never sees target data

# 2. The measurement runs only behind a passing gate.
verdict = gated(check, measure=lambda: my_estimator(target_data))
if verdict.decided:
    print(verdict.value)
else:
    print(verdict.reason)                         # NOT_DECIDABLE, with the failing margin

# 3. Null tests must be exactly null AND demonstrably non-vacuous.
certify_null(null_arm_diff, effect_arm_diff)      # raises on either failure, in production

# 4. How many observations do you actually have?
rep = independence_report(values, seed_labels, unit_name="seed")
print(rep.message())                              # ICC, design effect, effective n

# 5. Results files that carry their own audit trail; tests that keep them honest.
save_results("results/run.json", payload, script=__file__, root=PROJECT_ROOT,
             prereg=my_prereg.block(), independent_unit="seed")
testing.assert_fresh("results/run.json", PROJECT_ROOT)      # red if the code drifted
```

`examples/decay_exponent.py` runs the full discipline end to end on a toy problem, including
the starved geometry that correctly returns `NOT_DECIDABLE`, the 30-observation grid that is
accounted as ~2 independent observations, and the results file whose check goes red the moment
its analysis script is edited.

## Wiring into pytest

```python
# tests/test_discipline.py
import glob, pathlib, pytest
from gatecheck import testing

ROOT = pathlib.Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("f", sorted(glob.glob(str(ROOT / "results" / "*.json"))))
def test_results_are_fresh(f):
    testing.assert_fresh(f, ROOT)

def test_paper_numbers():
    testing.assert_manifest(ROOT / "tests" / "manifest.json",
                            ROOT / "paper" / "paper.tex", ROOT)

def test_retractions_stay_retracted():
    testing.assert_retracted_stays_retracted(
        (ROOT / "paper" / "paper.tex").read_text(),
        {"capacity axis": ["capacity scales with sensitivity"]})
```

## What this package is not

It is not a statistics library (bring scipy), not an experiment tracker (it stamps results, it
does not schedule runs), and not a substitute for thinking about *which* control to run — it
mechanizes the discipline around controls, gates, and provenance, not their design. The
single-implementation guard is a grep: an import-and-wrap alias slips past it, a limitation
inherited knowingly from the project of origin (see DESIGN.md §6).

## Provenance of the patterns

Every module docstring names the textca finding or incident it descends from (F56, F57, F59,
F65, F39, F45/F46, W1, #38, #52, #27, #48, #63). DESIGN.md contains the full extraction map,
what was generalized beyond the original, and the roadmap of patterns deliberately left for a
later version.
