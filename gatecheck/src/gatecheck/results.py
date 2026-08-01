"""Results-file conventions: stamped, preregistered, unit-declared, leak-checked JSON.

This module is glue: it wires provenance stamps (issue-#38 class), preregistration blocks,
the declared independent unit (F57's cheap half), and the absolute-path leak guard (#52) into
one save/load pair, so a project gets the whole discipline by calling two functions.

Reserved keys written by `save_results`:
    _provenance        — gatecheck.provenance.stamp(...)
    _preregistration   — Preregistration.block(), if given
    _quarantine        — written only by gatecheck.prereg.quarantine
    independent_unit   — the declared unit of independent replication
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping

from .provenance import stamp, verify_stamp, StampReport


def check_no_absolute_paths(obj: Any, forbidden_prefixes: list[str]) -> list[str]:
    """Find strings carrying any forbidden path prefix (home dirs, checkout paths).

    Origin: textca #52 — machine-written logs shipped the absolute checkout path into a
    published artifact. Run this over payloads and over anything you print to a log file.
    """
    hits: list[str] = []

    def walk(x):
        if isinstance(x, str):
            if any(p and p in x for p in forbidden_prefixes):
                hits.append(x)
        elif isinstance(x, Mapping):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)
    return hits


def save_results(
    path,
    payload: dict,
    *,
    script,
    root,
    prereg: Mapping | None = None,
    independent_unit: str = "",
    packages: list[str] | None = None,
    forbid_paths: bool = True,
) -> dict:
    """Write a results JSON that carries its own audit trail.

    `script` should be the caller's `__file__`; `root` the project root. Refuses payloads that
    embed the project's absolute path (pass forbid_paths=False to opt out, and reconsider).
    Returns the full document as written.
    """
    root = pathlib.Path(root).resolve()
    doc = dict(payload)
    if forbid_paths:
        leaks = check_no_absolute_paths(doc, [str(root), str(pathlib.Path.home())])
        if leaks:
            raise ValueError(
                f"absolute-path leak in results payload ({len(leaks)} string(s), e.g. "
                f"{leaks[0]!r}): store root-relative paths (gatecheck.provenance.rel)"
            )
    doc["_provenance"] = stamp(script, root, packages=packages)
    if prereg is not None:
        doc["_preregistration"] = dict(prereg)
    if independent_unit:
        doc["independent_unit"] = independent_unit
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
    return doc


def load_results(path, *, root=None, verify: bool = False) -> tuple[dict, StampReport | None]:
    """Load a results JSON; optionally verify its provenance stamp against disk.

    With verify=True (requires `root`), returns the stamp report alongside the payload; a
    stale report means the numbers were produced by code that has since changed — re-run the
    analysis before reading them. Callers in test suites should assert `report.ok`.
    """
    doc = json.loads(pathlib.Path(path).read_text())
    report = None
    if verify:
        if root is None:
            raise ValueError("verify=True requires root")
        block = doc.get("_provenance")
        if block is None:
            report = StampReport(ok=False, missing=["<no _provenance block in file>"])
        else:
            report = verify_stamp(block, root)
    return doc, report
