"""#25: the lattice context must not leak between experiments sharing a process.

`src/ca.py` kept DATA_DIR/VOCAB/INIT_LO as plain module globals that experiments assigned to and
never put back. The failure is silent by construction: a wrong `init_lo` still produces a
well-formed lattice, just one drawn from the wrong support, so nothing raises and nothing looks
wrong in the output. These pin the scoping that replaces the assignment.
"""
import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

ca = pytest.importorskip("ca", reason="ca.py needs jax; skipped where the backend is absent")

_FIELDS = ("DATA_DIR", "VOCAB", "INIT_LO")


def _state():
    return tuple(getattr(ca, f) for f in _FIELDS)


def test_using_restores_every_field_on_exit():
    before = _state()
    with ca.using(data_dir="data_bpe", vocab=4096, init_lo=1):
        assert _state() == ("data_bpe", 4096, 1)
    assert _state() == before


def test_using_restores_on_exception_which_is_the_whole_point():
    """An experiment that raises must not leave the module pointing at its corpus."""
    before = _state()
    with pytest.raises(RuntimeError):
        with ca.using(data_dir="data_bpe", vocab=4096, init_lo=1):
            raise RuntimeError("boom")
    assert _state() == before


def test_nesting_restores_the_enclosing_context_not_the_defaults():
    before = _state()
    with ca.using(data_dir="data_bpe", vocab=4096, init_lo=1):
        outer = _state()
        with ca.using(vocab=64):
            assert ca.VOCAB == 64 and ca.DATA_DIR == "data_bpe"
        assert _state() == outer, "inner frame reset to defaults instead of its caller's context"
    assert _state() == before


def test_omitted_fields_are_left_alone():
    with ca.using(data_dir="data_bpe", vocab=4096, init_lo=1):
        with ca.using(init_lo=2):
            assert ca.DATA_DIR == "data_bpe" and ca.VOCAB == 4096 and ca.INIT_LO == 2


def test_the_leak_that_motivated_the_issue_cannot_happen():
    """Two experiments, one process: the second must not inherit the first's support.

    This is the bug verbatim -- a BPE run followed by a word-level run, where the second silently
    drew random init from [1, 4096) against a 64-token vocabulary.
    """
    before = _state()
    with ca.using(data_dir="data_bpe", vocab=4096, init_lo=1):
        pass
    assert _state() == before
    with ca.using(data_dir="data", vocab=64, init_lo=2):
        assert ca.INIT_LO == 2 and ca.VOCAB == 64
    assert _state() == before


def _assignments_to_ca_globals_inside_functions(path):
    """(function, field) pairs where a module global is assigned from inside a def.

    Assignment at `__main__` scope is permitted: a script entry point configures its own process
    once and nothing else runs there. Assignment inside a FUNCTION is the dangerous form, because
    the function returns and the setting outlives it.
    """
    tree = ast.parse(path.read_text())
    out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            for tgt in (node.targets if isinstance(node, ast.Assign) else []):
                for t in (tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]):
                    if (isinstance(t, ast.Attribute) and t.attr in _FIELDS
                            and isinstance(t.value, ast.Name) and t.value.id == "ca"):
                        out.append((fn.name, t.attr))
    return out


def test_no_experiment_sets_the_context_from_inside_a_function():
    """The guard that keeps #25 fixed rather than fixed-once.

    Reintroducing `ca.DATA_DIR = ...` inside a helper restores the exact hazard, and it would not
    fail any existing test -- the corrupted run still completes and still writes plausible numbers.
    """
    offenders = {}
    for p in sorted((ROOT / "experiments").glob("*.py")):
        found = _assignments_to_ca_globals_inside_functions(p)
        if found:
            offenders[p.name] = found
    assert not offenders, (
        f"module-global assignment to the lattice context inside a function: {offenders}. "
        f"Use `with ca.using(...)` so the setting is restored when the function returns.")
