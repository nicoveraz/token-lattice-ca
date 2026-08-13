"""Pytest bootstrap: put src/ and experiments/ on sys.path so tests can import
the library (model, ca) and pipeline modules the same way the scripts do.
Tests are run from the repo root so the hardcoded data/ ckpt/ results/ paths resolve.

It also disarms a stale-bytecode trap (#65). CPython invalidates `__pycache__` on
`(mtime, size)`. Flipping a constant to another of the SAME BYTE LENGTH and re-running within
the SAME SECOND changes neither, so the stale `.pyc` is reused and the test reports an assertion
against a literal that is provably not in the source. That happened here: `BODY_PAGE_LIMIT` was
flipped 5 -> 4 -> 5 while proving the page guard fires, and pytest kept failing with
`assert 5 <= 4`.

The shape of that edit -- same-length constant, immediate re-run -- is exactly what a
"prove the guard fires" mutation check looks like, so the trap springs precisely when you are
trying to establish that a test is not vacuous. A wrong answer there is worse than elsewhere,
because it is the answer you are using to decide whether to trust everything else.

Two lines fix it and both are needed. `dont_write_bytecode` stops new `.pyc` files appearing
during a session; purging removes any already on disk, since Python will happily READ a stale
one regardless of that flag. Recompiling a handful of modules costs milliseconds against a
45-second suite.
"""
import pathlib
import shutil
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]

# --- #65: never read bytecode that (mtime, size) failed to invalidate --------------------
sys.dont_write_bytecode = True
for _d in ("tests", "experiments", "src"):
    for _pc in (_ROOT / _d).rglob("__pycache__"):
        shutil.rmtree(_pc, ignore_errors=True)


# --- skip accounting -------------------------------------------------------------------------
#
# CI runs offline (HF_HUB_OFFLINE=1), so the real-backend tests legitimately skip there and pass
# here. That is a useful fix and a dangerous shape: a suite whose green depends on skips can go
# green for the wrong reason, and two of those tests are the GOLDEN numerical-reproducibility
# checks. Two guards, deliberately narrow.
#
#   AVAILABILITY SKIPS MUST BE ZERO WHEN THE BACKEND IS AVAILABLE. Offline is a declared mode
#   (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE). Outside it, a "backend unavailable" skip means the
#   local cache is broken, and that must be loud rather than silent -- it is exactly the case where
#   a skip would hide a real failure.
#
#   COUNT -- opt-in via EXPECTED_SKIPS, set per job in the workflow. Measured, not derived.
#
# A FIRST VERSION OF THIS BLOCK ALLOWLISTED SKIP *REASONS*, and the offline run immediately showed
# why that was the wrong shape: this suite has ~15 legitimate skip sites with heterogeneous
# messages, so the allowlist would have to be maintained in lockstep with every one of them and
# would go red on each new legitimate skip. A gate whose natural formulation fires on everything is
# not yet a gate. The two guards below fire only on the case with real risk.
import os as _os

_AVAILABILITY_PREFIX = "backend unavailable"
_SKIPS = []


def _offline_declared():
    return any(_os.environ.get(v, "").strip() not in ("", "0", "false", "False")
               for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def pytest_configure(config):
    _SKIPS.clear()
    config._skip_reports = _SKIPS


def pytest_runtest_logreport(report):
    if report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        _SKIPS.append(reason.replace("Skipped: ", "", 1))


def pytest_sessionfinish(session, exitstatus):
    problems = []
    avail = [r for r in _SKIPS if r.startswith(_AVAILABILITY_PREFIX)]
    if avail and not _offline_declared():
        problems.append(
            f"{len(avail)} real-backend test(s) skipped for unavailable weights while OFFLINE mode "
            f"was NOT declared -- the local cache is broken and these tests are not running: "
            f"{avail[:3]}")
    want = _os.environ.get("EXPECTED_SKIPS")
    if want is not None and str(len(_SKIPS)) != want.strip():
        problems.append(f"{len(_SKIPS)} skips, expected {want.strip()} (EXPECTED_SKIPS)")
    if problems:
        session.config._skip_problem = "; ".join(problems)
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    problem = getattr(terminalreporter.config, "_skip_problem", None)
    if problem:
        terminalreporter.write_sep("=", "SKIP ACCOUNTING FAILED", red=True)
        terminalreporter.write_line(problem)
