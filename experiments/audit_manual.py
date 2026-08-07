"""Verify citations that CANNOT be machine-fetched -- the OpenReview-shaped hole in #71's audit.

WHY THIS EXISTS. `audit_refs.py` replaced the eyeball with a fetch for every arXiv citation in
`paper_arxiv/refs.bib`, because F43 found three invented titles and a hand audit then missed a fourth.
Two things fell outside it:

  1. OPENREVIEW CANNOT BE FETCHED. It returns HTTP 403 `ChallengeRequiredError` on the forum page,
     the v1 and v2 note APIs, and the PDF endpoint. That challenge is a deliberate anti-automation
     measure and is NOT routed around here.
  2. `assembly_theory.md` HAS ITS OWN BIBLIOGRAPHY AND NO MACHINERY. Its `[unverified]` markers are
     hand-typed prose. A citation could sit behind one indefinitely with nothing recording whether
     anyone had ever read the work -- which is F43's failure shape exactly ("a hand audit that
     misses one entry looks exactly like a hand audit that misses none"), reintroduced in a file
     the arXiv audit never covered.

THE FIX IS NOT AUTOMATION, IT IS EVIDENCE. Some sources genuinely cannot be fetched. What can be
made machine-checkable is whether a human check *happened* and *what it found*: a record in
`paper_arxiv/refs_manual.json` carrying the transcribed metadata, the date, what was read, and the
sha256 of the artifact. The hash makes the check reproducible and tamper-evident even though the
fetch is not automatable -- re-verification is one `shasum` rather than a re-read.

WHAT THIS DOES NOT DO. It does not decide whether a claim ABOUT a cited work is correct. It
asserts that the work was obtained, that its title/authors/venue are transcribed rather than
invented, and that the document's own `[unverified]` markers agree with the record. Reading the
paper is still reading the paper.

Usage:
  .venv/bin/python experiments/audit_manual.py                       # audit; exit 1 on a problem
  .venv/bin/python experiments/audit_manual.py --register KEY PATH   # pin a newly obtained PDF
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "experiments")]
import hashlib
import json
import re

from provenance import rel

RECORDS = _ROOT / "paper_arxiv" / "refs_manual.json"

# Markdown bibliographies that carry citations but no .bib machinery. Add to this list rather
# than writing a second auditor; one implementation, per F56/F73.
SOURCES = [_ROOT / "assembly_theory.md"]

OPENREVIEW = re.compile(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)")
UNVERIFIED_MARK = re.compile(r"\[unverified[^\]]*\]", re.I)


def normalise(s):
    """Compare titles on content, not on punctuation or markup accidents.

    Imported shape is deliberately the same as audit_refs.normalise; that one is defined over
    LaTeX, this one over markdown, and collapsing them would make each wrong for the other.
    """
    s = s.lower()
    s = re.sub(r"[*_`]", "", s)                 # markdown emphasis / code spans
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    return json.loads(RECORDS.read_text())


def cited_forum_ids():
    """{forum_id: [source basenames]} over every registered markdown bibliography."""
    out = {}
    for src in SOURCES:
        if not src.exists():
            continue
        for fid in set(OPENREVIEW.findall(src.read_text())):
            out.setdefault(fid, []).append(src.name)
    return out


def register(key, path):
    """Pin a newly obtained artifact: hash it and stamp the record verified."""
    p = _pathlib.Path(path).expanduser()
    if not p.exists():
        print(f"no such file: {path}")
        return 1
    doc = load()
    rec = doc["records"].get(key)
    if rec is None:
        print(f"no record named {key!r}. Add it to {rel(RECORDS)} first, with at least "
              f"`title`, `source` and `why_it_is_cited`.")
        return 1
    rec["pdf_basename"] = p.name
    rec["pdf_sha256"] = sha256(p)
    rec["status"] = "verified"
    rec.setdefault("checked_against", "local PDF obtained through a browser")
    print(f"  {key}: sha256 {rec['pdf_sha256'][:16]}...  ({p.name})")
    print(f"  NOW SET `checked_on` AND `checked_against` BY HAND -- this tool pins the artifact, "
          f"it does not certify that anyone read it.")
    RECORDS.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {rel(RECORDS)}")
    return 0


def audit():
    doc = load()
    recs = doc["records"]
    cited = cited_forum_ids()
    problems = []

    print(f"{len(recs)} record(s); {len(cited)} OpenReview citation(s) across "
          f"{len([s for s in SOURCES if s.exists()])} source document(s)\n")

    by_source = {}
    for key, r in recs.items():
        m = re.fullmatch(r"openreview:([A-Za-z0-9_-]+)", r.get("source", ""))
        if m:
            by_source[m.group(1)] = key

    for fid, where in sorted(cited.items()):
        if fid not in by_source:
            problems.append(f"cited but unrecorded: openreview:{fid} (in {', '.join(where)})")
            print(f"  {fid:16s} UNRECORDED  <- cited in {', '.join(where)}")

    for fid, key in sorted(by_source.items()):
        if fid not in cited:
            problems.append(f"recorded but uncited: {key} (openreview:{fid})")
            print(f"  {fid:16s} ORPHAN      <- record {key} cites nothing")

    print(f"\n{'key':26s} {'status':10s} {'artifact':10s} title")
    for key, r in sorted(recs.items()):
        st = r.get("status", "?")
        if st == "verified":
            for need in ("checked_on", "checked_against"):
                if not r.get(need):
                    problems.append(f"{key}: status=verified without `{need}`")
        elif st == "unverified":
            for need in ("reason", "action"):
                if not r.get(need):
                    problems.append(f"{key}: status=unverified without `{need}`")
        else:
            problems.append(f"{key}: unknown status {st!r}")

        art = "-"
        want = r.get("pdf_sha256")
        if want:
            found = _find(r.get("pdf_basename", ""))
            if found is None:
                art = "not here"
            elif sha256(found) == want:
                art = "hash ok"
            else:
                art = "HASH BAD"
                problems.append(f"{key}: {r['pdf_basename']} no longer matches its recorded "
                                f"sha256 -- the artifact behind the check changed")
        print(f"{key:26s} {st:10s} {art:10s} {r.get('title', '')[:44]}")

    # The document's own prose must agree with the record.
    for src in SOURCES:
        if not src.exists():
            continue
        text = src.read_text()
        for fid, key in by_source.items():
            if fid not in text:
                continue
            claimed_unverified = _marked_unverified_near(text, fid)
            actually_verified = recs[key].get("status") == "verified"
            if actually_verified and claimed_unverified:
                problems.append(f"{src.name} still marks openreview:{fid} '[unverified]', but "
                                f"{key} is verified -- update the prose")
            if not actually_verified and not claimed_unverified:
                problems.append(f"{src.name} cites openreview:{fid} with no '[unverified]' "
                                f"marker, but {key} is NOT verified -- the prose overclaims")

    print(f"\n{len(problems)} problem(s)")
    for p in problems:
        print(f"  {p}")
    return 1 if problems else 0


_ITEM = re.compile(r"^\s*[-*+]\s")


def _marked_unverified_near(text, fid):
    """Is there an `[unverified]` marker in the SAME bibliography entry as this citation?

    Scoped to the enclosing markdown list item, not to a character window. A window was the first
    implementation and it was VACUOUS: in a dense bibliography the +/-700 characters around one
    entry contain the next entry's markers too, so deleting an entry's own `[unverified]` still
    found a neighbour's and the guard passed. Caught by deliberately removing the marker and
    watching the test stay green -- the non-vacuity check that `test_null_all_backends.py`
    established as standard here.

    Entries wrap across indented continuation lines, so the block runs from the item's bullet to
    the next bullet, blank line, or heading.
    """
    lines = text.splitlines()
    hit = next((n for n, ln in enumerate(lines) if fid in ln), None)
    if hit is None:
        return False
    start = hit
    while start > 0 and not _ITEM.match(lines[start]):
        if not lines[start].strip() or lines[start].startswith("#"):
            break
        start -= 1
    end = hit + 1
    while end < len(lines):
        ln = lines[end]
        if not ln.strip() or ln.startswith("#") or _ITEM.match(ln):
            break
        end += 1
    return bool(UNVERIFIED_MARK.search("\n".join(lines[start:end])))


def _find(basename):
    """Look for a registered artifact in the usual places. Absent is not an error."""
    if not basename:
        return None
    for d in (_pathlib.Path.home() / "Downloads", _ROOT, _ROOT / "paper"):
        p = d / basename
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    if len(_sys.argv) >= 2 and _sys.argv[1] == "--register":
        if len(_sys.argv) != 4:
            print("usage: audit_manual.py --register KEY PATH")
            _sys.exit(2)
        _sys.exit(register(_sys.argv[2], _sys.argv[3]))
    _sys.exit(audit())
