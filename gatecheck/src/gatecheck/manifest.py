"""Number manifests: every load-bearing literal in a manuscript traces to a results file.

Origin: textca's `build_paper_manifest.py` + `tests/test_paper_numbers.py`, which caught a real
loss on their first run (a measurement silently dropped during a page-fit trim) and two
manifest-builder defects by paper/manifest disagreement. Generalized here with the piece the
textca version documented but never shipped: a RECOMPUTE check. An entry may carry a dot-path
into its source JSON plus a rounding spec, and `check` re-derives the literal from the source
file — so results can no longer drift under a stale manifest and a stale manuscript while
everything stays green.

Three entry kinds, because provenance laundering hides in the difference:
  measured   — comes from a results file this project produced (source path required;
               recomputable when `path` is given)
  published  — someone else's number (a citation is required in `ref`)
  arithmetic — derived by hand from other entries (spell out the derivation in `note`)
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any

KINDS = ("measured", "published", "arithmetic")


@dataclass
class Entry:
    literal: str                 # the exact string as it appears in the document
    source: str                  # results file (measured) / citation key or note (other kinds)
    kind: str = "measured"
    note: str = ""
    ref: str = ""                # required for kind="published" (arXiv/DOI/citation key)
    path: str = ""               # optional dot-path into the source JSON, e.g. "plateau.lambda.mean"
    round: int | None = None     # decimals to round the recomputed value to
    fmt: str = ""                # optional format spec applied after rounding, e.g. "+.3f"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == "published" and not self.ref:
            raise ValueError(f"published entry {self.literal!r} needs a `ref` (arXiv/DOI/key)")
        if not str(self.literal).strip():
            raise ValueError("empty literal")


def strip_tex_comments(text: str) -> str:
    """Drop TeX comments (unescaped % to end of line) so commented-out numbers cannot satisfy
    a presence check."""
    return re.sub(r"(?<!\\)%.*", "", text)


def _walk(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {part!r}")
    return cur


@dataclass
class Report:
    ok: bool
    missing_in_document: list[str] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)
    recompute_failures: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def message(self) -> str:
        if self.ok:
            return f"manifest verified ({self.counts})"
        lines = ["MANIFEST FAILURES:"]
        lines += [f"  literal not in document: {x!r}" for x in self.missing_in_document]
        lines += [f"  source file missing: {x}" for x in self.missing_sources]
        lines += [f"  recompute mismatch: {x}" for x in self.recompute_failures]
        return "\n".join(lines)


class Manifest:
    def __init__(self, entries: list[Entry] | None = None):
        self.entries: list[Entry] = list(entries or [])

    def add(self, literal, source, kind="measured", **kw) -> Entry:
        e = Entry(literal=str(literal), source=source, kind=kind, **kw)
        self.entries.append(e)
        return e

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        pathlib.Path(path).write_text(
            json.dumps([asdict(e) for e in self.entries], indent=1, sort_keys=True)
        )

    @classmethod
    def load(cls, path) -> "Manifest":
        raw = json.loads(pathlib.Path(path).read_text())
        return cls([Entry(**e) for e in raw])

    # -- checks --------------------------------------------------------------
    def check(self, document_text: str, root, *, strip=strip_tex_comments,
              min_entries: int = 1, require_all_kinds: bool = False) -> Report:
        """The three-way agreement check: document ⟷ manifest ⟷ results files.

        Presence is substring-based (textca's convention) and therefore weak for short
        literals — a bare "16" matches anywhere. The `path` recompute is the strong check;
        prefer entries that carry one, and treat presence-only entries as scaffolding.
        """
        root = pathlib.Path(root)
        doc = strip(document_text) if strip else document_text
        rep = Report(ok=True)

        if len(self.entries) < min_entries:
            rep.ok = False
            rep.recompute_failures.append(
                {"error": f"manifest has {len(self.entries)} entries < required {min_entries}"}
            )
        kinds = {k: sum(1 for e in self.entries if e.kind == k) for k in KINDS}
        rep.counts = {"entries": len(self.entries), **kinds}
        if require_all_kinds and not all(kinds.values()):
            rep.ok = False
            rep.recompute_failures.append(
                {"error": f"manifest must contain all kinds {KINDS}, has {kinds}"}
            )

        for e in self.entries:
            if e.literal not in doc:
                rep.missing_in_document.append(e.literal)
            if e.kind == "measured":
                src = root / e.source
                if not src.exists():
                    rep.missing_sources.append(e.source)
                elif e.path:
                    fail = self._recompute(e, src)
                    if fail:
                        rep.recompute_failures.append(fail)
        rep.ok = rep.ok and not (
            rep.missing_in_document or rep.missing_sources or rep.recompute_failures
        )
        return rep

    @staticmethod
    def _recompute(e: Entry, src: pathlib.Path) -> dict | None:
        try:
            value = _walk(json.loads(src.read_text()), e.path)
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            return {"literal": e.literal, "source": e.source, "path": e.path,
                    "error": f"{type(exc).__name__}: {exc}"}
        if e.round is not None:
            value = round(float(value), e.round)
        derived = format(value, e.fmt) if e.fmt else (
            # avoid float repr surprises for round-tripped values
            f"{value:.{e.round}f}" if (e.round is not None and isinstance(value, float))
            else str(value)
        )
        if derived != e.literal:
            return {"literal": e.literal, "source": e.source, "path": e.path,
                    "derived": derived,
                    "error": "derived value does not match the document literal"}
        return None
