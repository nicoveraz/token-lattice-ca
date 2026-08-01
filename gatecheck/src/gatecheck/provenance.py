"""Provenance stamps: results files that can prove which code (and environment) wrote them.

Origin: textca issue #38 / F45 / F46. Python imports a module once; edit an analysis script
while its job is running and the job's end-of-run analysis still executes the version imported
at launch — twice in textca this wrote a finished-looking results file whose conclusion was
inverted. The fix: every analysis embeds the sha256 of its own source (and of its project-local
import closure) into the results file, and a test recomputes the hashes. A stale analysis then
becomes a red test instead of a silent wrong number.

Generalizations over the textca original:
  * `root` is explicit (no assumption about repo layout).
  * `environment_fingerprint` records Python and package versions — closing the hole textca's
    own docstring names ("STILL NOT COVERED: third-party package versions. A numpy or torch
    upgrade can move a number without touching a byte of this repo.").
  * `verify_stamp` returns a structured report rather than living inside a project test file.
"""
from __future__ import annotations

import hashlib
import pathlib
import platform
import sys
from dataclasses import dataclass, field


def source_sha256(path) -> str | None:
    """sha256 of a source file's bytes, or None if unreadable."""
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def import_closure(root) -> dict[str, str]:
    """Every project-local module currently imported, as {root-relative path: sha256}.

    Reads `sys.modules` at call time, so it records what was ACTUALLY imported by this run
    rather than what a static scan guesses. The most load-bearing analysis code usually lives
    in imports, not in the top-level script — a shared predicate is exactly the code most
    likely to be edited and most likely to change many results at once (textca measured 14% on
    a headline quantity when one shared rule was applied two ways). Virtualenvs and anything
    outside `root` are excluded.
    """
    root = pathlib.Path(root).resolve()
    out: dict[str, str] = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = pathlib.Path(f)
        try:
            p = p.resolve()
            relp = p.relative_to(root)
        except (ValueError, OSError):
            continue
        parts = set(relp.parts)
        if parts & {".venv", "venv", "site-packages", "node_modules"}:
            continue
        if p.suffix != ".py" or not p.exists():
            continue
        out[relp.as_posix()] = source_sha256(p)
    return dict(sorted(out.items()))


def environment_fingerprint(packages: list[str] | None = None) -> dict:
    """Python version, platform, and installed versions of the packages that can move numbers.

    If `packages` is None, a default scientific set is probed; absent packages are simply
    omitted. Record this in every results file: an upgrade can change a conclusion without
    touching a byte of your repository, and nothing else will notice.
    """
    from importlib import metadata

    probe = packages or [
        "numpy", "scipy", "torch", "jax", "transformers", "pandas", "scikit-learn",
        "matplotlib", "statsmodels",
    ]
    versions = {}
    for name in probe:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def stamp(script_path, root, *, include_imports: bool = True,
          include_environment: bool = True, packages: list[str] | None = None) -> dict:
    """Provenance block for a results file. Call with `__file__` just before dumping.

    Note the failure mode this is aimed at: a job that imported its analysis module earlier
    stamps the CURRENT bytes while having executed the OLD ones — so verification compares the
    stamp against disk and catches the drift as soon as the file is edited, which is the case
    that actually bites.
    """
    p = pathlib.Path(script_path).resolve()
    root = pathlib.Path(root).resolve()
    try:
        script = p.relative_to(root).as_posix()
        external = False
    except ValueError:
        script = p.name
        external = True     # outside root: hash recorded, but not verifiable against root
    block = {
        "script": script,
        "script_external": external,
        "sha256": source_sha256(p),
        "note": ("sha256 of the analysis source at write time, plus the project-local import "
                 "closure and environment fingerprint. If any hash disagrees with the file on "
                 "disk, this analysis was produced by a different version of the code: re-run "
                 "it before believing the numbers."),
    }
    if include_imports:
        block["imports"] = import_closure(root)
    if include_environment:
        block["environment"] = environment_fingerprint(packages)
    return block


@dataclass
class StampReport:
    ok: bool
    stale: dict[str, dict] = field(default_factory=dict)   # path -> {recorded, current}
    missing: list[str] = field(default_factory=list)       # recorded files no longer on disk

    def message(self) -> str:
        if self.ok:
            return "provenance verified: all recorded sources match disk"
        lines = [f"STALE ANALYSIS: {len(self.stale)} changed, {len(self.missing)} missing"]
        lines += [f"  changed: {k}" for k in sorted(self.stale)]
        lines += [f"  missing: {k}" for k in self.missing]
        lines.append("  re-run the analysis before reading its numbers")
        return "\n".join(lines)


def verify_stamp(block: dict, root) -> StampReport:
    """Recompute every hash a stamp records and report drift. Wire into your test suite."""
    root = pathlib.Path(root).resolve()
    report = StampReport(ok=True)
    targets = dict(block.get("imports", {}))
    if block.get("script") and block.get("sha256") and not block.get("script_external"):
        targets[block["script"]] = block["sha256"]
    for relp, recorded in targets.items():
        current = source_sha256(root / relp)
        if current is None:
            report.missing.append(relp)
        elif current != recorded:
            report.stale[relp] = {"recorded": recorded, "current": current}
    report.ok = not report.stale and not report.missing
    return report


def rel(path, root) -> str:
    """Root-relative form of a path, for printing and for machine-written logs.

    Origin: textca #52 — twelve machine-written logs carried the absolute checkout path, a
    de-anonymization leak in an artifact the submission mirror published. The absolute form
    carries no information the relative one lacks when the path is inside the project.
    """
    p = pathlib.Path(path).resolve()
    root = pathlib.Path(root).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)
