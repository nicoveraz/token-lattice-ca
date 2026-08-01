"""Pytest bootstrap for gatecheck's own tests.

gatecheck is laid out as a separable package (`src/gatecheck/`, its own README, DESIGN and
.gitignore) and is not installed into the repo venv, so `import gatecheck` from inside its test
files resolves to the *directory* `gatecheck/` -- a namespace package with no modules in it -- and
every import fails at collection.

This is scoped to the subtree deliberately. The root conftest puts `src/` and `experiments/` on
sys.path for the main suite; adding gatecheck there too would make it importable everywhere and
quietly undo the separation the layout is expressing. Scripts that want it (fingerprint/*.py) add
`gatecheck/src` explicitly, and that stays true.
"""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
