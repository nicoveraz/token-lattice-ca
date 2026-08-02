"""The commit guard must block a live-write, and must NOT block anything else.

Why this exists. A `git add -A` committed `fingerprint/gate2.json` while `fingerprint/gate2.py`
was four hours into writing it -- the file grew by one run between the commit and the check that
caught it, so what landed was a partial results file that looks finished. Same class as the
F45/F46 stale-analysis trap, reached through git instead of through Python's import cache, and
recoverable only because it had not been pushed. The instruction not to use `git add -A` already
existed in the history, which is the point: a rule that depends on remembering is not a guard.

A guard that only ever says yes is worse than none, so the no-false-positive direction is tested
as hard as the blocking one. Committing a results file immediately after re-running its analysis
is the NORMAL workflow here and must stay unobstructed -- which is why the rule keys on a live
producing process and deliberately not on mtime.
"""
import os
import pathlib
import stat
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]

from precommit_guard import live_writers, GUARDED          # noqa: E402

HOOK = ROOT / ".githooks" / "pre-commit"

RUNNING = [(38183, ".venv/bin/python -u fingerprint/gate2.py"),
           (40296, "caffeinate -dimsu .venv/bin/python -u fingerprint/gate2.py"),
           (99999, "/usr/bin/vim notes.txt")]
IDLE = [(99999, "/usr/bin/vim notes.txt"), (12345, "-zsh")]


def test_blocks_a_file_whose_producing_script_is_running():
    """The exact incident, replayed."""
    hits = live_writers(["fingerprint/gate2.json"], RUNNING)
    assert len(hits) == 1
    path, script, pids = hits[0]
    assert path == "fingerprint/gate2.json"
    assert script == "gate2.py"
    assert set(pids) == {38183, 40296}


def test_does_not_block_when_the_job_has_finished():
    """The normal workflow: re-run an analysis, commit its result. Must not be obstructed."""
    assert live_writers(["fingerprint/gate2.json"], IDLE) == []
    assert live_writers(["results/dev_transition_radius.json"], IDLE) == []


def test_does_not_block_source_or_prose():
    """Only job-written data is guarded. Editing a script while its job runs is a different
    hazard and `provenance.stamp`'s import closure covers it."""
    staged = ["experiments/gate2.py", "paper/paper.tex", "findings.md", "src/lattice.py"]
    assert live_writers(staged, RUNNING) == []


def test_stem_matching_does_not_overreach():
    """`gate2.py` must not be matched by a process running `mygate2.py`, and a one-letter stem
    must not match every command that happens to contain that letter."""
    decoys = [(1, ".venv/bin/python -u fingerprint/mygate2.py"),
              (2, "python x.py")]
    assert live_writers(["fingerprint/gate2.json"], decoys) == []
    # substring-in-command matching would fire here; component matching must not
    assert live_writers(["results/x.json"], [(3, "python prefix_x.py")]) == []
    # ...but the real thing still matches
    assert len(live_writers(["results/x.json"], [(4, "python experiments/x.py")])) == 1


def test_guarded_directories_are_the_ones_jobs_write():
    for p in ("results/a.json", "fingerprint/b.json", "logs/c.log"):
        assert GUARDED.match(p), p
    for p in ("experiments/a.py", "src/b.py", "paper/c.tex", "findings.md"):
        assert not GUARDED.match(p), p


def test_the_hook_is_present_executable_and_routes_to_the_tested_module():
    """A hook that is not installed guards nothing, and one that duplicates the logic drifts."""
    assert HOOK.exists(), (
        ".githooks/pre-commit is missing. It is tracked on purpose -- a hook in .git/hooks is "
        "neither reviewable nor survives a clone.")
    assert HOOK.stat().st_mode & stat.S_IXUSR, f"{HOOK} is not executable"
    body = HOOK.read_text()
    assert "precommit_guard.py" in body, (
        "the hook must route to experiments/precommit_guard.py rather than reimplement the rule "
        "-- two copies of a guard drift, which is F56's anti-drift rule and what F73 caught")


def test_hooks_path_is_configured_or_the_guard_is_inert():
    """Tracked-but-unconfigured is the failure mode that looks installed and does nothing.

    Skipped rather than failed on a fresh clone, where nobody has run the config line yet -- but
    it names the command, so the gap is one copy-paste from closed.
    """
    import subprocess
    got = subprocess.run(["git", "config", "core.hooksPath"], cwd=ROOT,
                         capture_output=True, text=True, check=False).stdout.strip()
    if got != ".githooks":
        pytest.skip("core.hooksPath is not set to .githooks -- the tracked hook is INERT here. "
                    "Enable with:  git config core.hooksPath .githooks")
