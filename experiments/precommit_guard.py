"""Refuse to commit a data file that a live experiment is still writing.

WHY THIS EXISTS. A `git add -A` swept `fingerprint/gate2.json` into an unrelated commit while
`fingerprint/gate2.py` was four hours into writing it. The file grew by a run between the commit
and the check that caught it, so what landed was a PARTIAL results file that looks finished --
the same defect class as the F45/F46 stale-analysis trap, arriving through git rather than through
Python's import cache. It was recoverable only because it had not been pushed.

The instruction not to use `git add -A` already existed in the history. A rule that depends on
remembering is not a guard; this is the guard.

WHAT IT CHECKS. For every staged file under a guarded directory, is a process running whose command
line names the script that produces it? `results/foo.json` is produced by `foo.py`, and the repo's
naming is consistent enough that the stem match is precise. This is deliberately NOT an mtime
check: re-running an analysis and committing the result immediately is the normal workflow and
must not be blocked. Only a LIVE writer blocks.

Stdlib only, so the hook works without the venv activated.

Bypass with `git commit --no-verify` when you genuinely mean it -- e.g. committing a checkpoint of
a long run on purpose. Say so in the commit message if you do.
"""
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Directories whose contents are written BY long-running jobs. Source trees are not guarded:
# editing a script while its job runs is a different hazard, and `provenance.stamp` covers it.
GUARDED = re.compile(r"^(results|fingerprint|logs)/")


def staged_files():
    """Paths staged for commit (added/copied/modified), repo-relative."""
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def running_processes():
    """Command lines of every process owned by this user."""
    out = subprocess.run(["ps", "-o", "pid=,command=", "-u", str(os.getuid())],
                         capture_output=True, text=True, check=False)
    procs = []
    for ln in out.stdout.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        pid, _, cmd = ln.partition(" ")
        if pid.isdigit():
            procs.append((int(pid), cmd.strip()))
    return procs


def live_writers(staged, procs):
    """[(path, script, [pids])] for staged data files whose producing script is still running.

    Pure: `staged` and `procs` are injected so the rule is testable without spawning anything.
    """
    problems = []
    for path in staged:
        if not GUARDED.match(path):
            continue
        stem = pathlib.Path(path).stem
        if not stem:
            continue
        script = f"{stem}.py"
        # Match the script name as a path component, so `gate2.py` does not match `mygate2.py`
        # and a stem of `x` does not match every command containing the letter x.
        pat = re.compile(r"(^|[/\s])" + re.escape(script) + r"(\s|$)")
        pids = [pid for pid, cmd in procs if pat.search(cmd)]
        if pids:
            problems.append((path, script, pids))
    return problems


def main():
    problems = live_writers(staged_files(), running_processes())
    if not problems:
        return 0
    print("\nBLOCKED: staged file(s) are still being written by a live job.\n", file=sys.stderr)
    for path, script, pids in problems:
        print(f"  {path}", file=sys.stderr)
        print(f"      {script} is RUNNING (pid {', '.join(str(p) for p in pids)})",
              file=sys.stderr)
    print("\nCommitting these now stores a PARTIAL file that looks finished -- the defect this",
          file=sys.stderr)
    print("guard exists for. Options:\n", file=sys.stderr)
    print("  * unstage them and commit the rest:", file=sys.stderr)
    print(f"        git restore --staged {' '.join(p for p, _, _ in problems)}", file=sys.stderr)
    print("  * wait for the job to finish, then commit", file=sys.stderr)
    print("  * if you MEAN to commit a mid-run checkpoint: git commit --no-verify,", file=sys.stderr)
    print("    and say so in the commit message\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
