"""Stamp a results file with the analysis source that produced it.

Why this exists (issue #38). Python imports a module once. If a long job is running and its
analysis code is edited mid-flight, the job's OWN end-of-run analysis still uses the version
imported at launch. That happened twice in one session and both times wrote a wrong results
file that looked finished and machine-written:

  * F45 (N=192): the F42 ignition filter was added after launch, so the job averaged lambda
    over unignited runs and concluded "size-robustness DOWNGRADED". Regenerating inverted the
    conclusion to "INVARIANT across a 4x range".
  * F46 (cross-scale): the verdict logic was fixed mid-run and the job again wrote the pre-edit
    verdict -- this one was predicted in the fixing commit and still happened.

A stale-analysis write is worse than a crash, because a crash is visible. So: every analysis
records the sha256 of the source file that computed it, and a test recomputes that hash and
fails if the file on disk has moved on. That turns a silent wrong answer into a red test.

Usage, one line before dumping:

    out["_analysis_provenance"] = stamp(__file__)
"""
import hashlib
import pathlib


def source_sha256(path):
    """sha256 of a source file, or None if it cannot be read."""
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def stamp(path):
    """Provenance block naming the script that produced the analysis and its exact content.

    `path` should be the analysis script's `__file__`. The hash is of the file as it exists at
    the moment the analysis runs, which is precisely the thing a mid-run edit invalidates --
    note that a job which imported the module earlier will stamp the CURRENT bytes while having
    executed the OLD ones, so the test compares the stamp against disk and catches the drift
    only after the file is edited. That is the case that has actually bitten; the reverse (file
    reverted after a correct run) is caught too.
    """
    p = pathlib.Path(path)
    return {
        "script": p.name,
        "sha256": source_sha256(p),
        "note": ("sha256 of the analysis source at write time. If this does not match the file "
                 "on disk, the analysis was produced by a different version of the code -- "
                 "re-run it before reading the numbers (issue #38)."),
    }
