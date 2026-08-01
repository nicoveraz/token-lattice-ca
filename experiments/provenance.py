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
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def source_sha256(path):
    """sha256 of a source file, or None if it cannot be read."""
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def import_closure():
    """Every repo-local module currently imported, as {repo-relative path: sha256}.

    THE HOLE THIS CLOSES. Stamping only the top-level script leaves the guard blind to the code
    that script IMPORTS -- and the project's most load-bearing analysis code lives in imports, not
    in the scripts. `lyapunov.run_ignited` is the single definition of the F42 ignition rule and is
    consumed by at least four analyses; `dp_calibration` gates every DP verdict; `assembly_calib`
    holds the estimator that `assembly_temperature` and the pilot both import. Editing any of them
    invalidates every downstream results file, and before this nothing tripped: the consumers'
    own bytes were unchanged, so their stamps still matched.

    That is the same defect class as F45/F46, one level down the import graph, and it is arguably
    worse -- a shared predicate is exactly the code most likely to be edited and most likely to
    change many results at once. F45 measured the cost when one such rule was applied two ways:
    14% on a headline quantity.

    Reads `sys.modules` at write time, so it records what was ACTUALLY imported by this run rather
    than what a static scan guesses. `.venv` and stdlib are excluded: pinning third-party versions
    is a separate job (and one this repo does not yet do -- see the note in `stamp`).
    """
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = pathlib.Path(f)
        try:
            p = p.resolve()
            relp = p.relative_to(ROOT)
        except (ValueError, OSError):
            continue
        if ".venv" in relp.parts or p.suffix != ".py" or not p.exists():
            continue
        out[str(relp)] = source_sha256(p)
    return dict(sorted(out.items()))


def stamp(path):
    """Provenance block naming the script that produced the analysis and its exact content.

    `path` should be the analysis script's `__file__`. The hash is of the file as it exists at
    the moment the analysis runs, which is precisely the thing a mid-run edit invalidates --
    note that a job which imported the module earlier will stamp the CURRENT bytes while having
    executed the OLD ones, so the test compares the stamp against disk and catches the drift
    only after the file is edited. That is the case that has actually bitten; the reverse (file
    reverted after a correct run) is caught too.

    `imports` extends the same guarantee to the repo-local import closure -- see `import_closure`
    for why the script alone was not enough. Results files written before this field existed
    simply lack it, and the test checks it only when present; re-running any analysis adds it.

    STILL NOT COVERED: third-party package versions. A numpy or torch upgrade can move a number
    without touching a byte of this repo, and nothing here would notice.
    """
    p = pathlib.Path(path)
    return {
        "script": p.name,
        "sha256": source_sha256(p),
        "imports": import_closure(),
        "note": ("sha256 of the analysis source at write time, plus the repo-local import "
                 "closure. If any of these does not match the file on disk, the analysis was "
                 "produced by a different version of the code -- re-run it before reading the "
                 "numbers (issue #38; the imports half closes the same hole one level down the "
                 "import graph). Third-party package versions are NOT covered."),
    }

def rel(path):
    """Repo-relative form of a path, for printing (#52).

    An absolute path in a log is a de-anonymisation leak -- twelve machine-written logs shipped
    lines like `wrote /Users/<user>/Documents/GitHub/textca/results/x.json`, and the anonymised
    submission mirror had to rewrite them. The absolute form carries no information the relative
    one lacks: every one of those paths is inside this repository.

    This fixes only the paths WE print. Python emits absolute paths of its own in tracebacks and
    in stdlib warnings (e.g. multiprocessing's leaked-semaphore notice from the interpreter's
    own install directory), and no change here touches those; the mirror scrub remains the
    backstop for them.
    """
    p = pathlib.Path(path).resolve()
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)
