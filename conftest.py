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
