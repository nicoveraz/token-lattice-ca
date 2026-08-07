"""Verify every arXiv citation in refs.bib against arXiv itself (issue #71; F43 class fix).

WHY THIS EXISTS. F43 found three citations with invented titles. `plainnat` prints `note=`
fields, so the bibliography literally read "Title/authors to verify" in a built PDF. That pass
was done by hand, and by hand it was incomplete: a fourth wrong title (`ar_tempcrit`) survived
it and was caught only when the entry was read for an unrelated reason. A hand audit that misses
one entry looks exactly like a hand audit that misses none.

So this replaces the eyeball with a fetch. For every entry carrying an `eprint`, ask arXiv what
that identifier actually is, and compare title / first author / year against what refs.bib
claims. Writes paper_arxiv/refs_verified.json -- the metadata AS ARXIV REPORTS IT -- which the test
suite then checks refs.bib against offline. Network here, no network in the test.

A citation with a wrong title is not a cosmetic defect. It is an unverifiable claim about the
literature sitting in a paper whose whole argument is that its claims are checked.

Usage:  .venv/bin/python experiments/audit_refs.py
"""
from provenance import rel
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
import html as html_mod

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIB = ROOT / "paper_arxiv" / "refs.bib"
OUT = ROOT / "paper_arxiv" / "refs_verified.json"
ABS = "https://arxiv.org/abs/{}"


def normalise(s):
    """Compare titles on content, not on LaTeX/whitespace/punctuation accidents."""
    s = s.lower()
    s = re.sub(r"\$[^$]*\$", " ", s)          # math
    s = re.sub(r"\\[a-z]+", " ", s)           # latex macros
    s = re.sub(r"[{}\\]", "", s)              # brace protection
    s = re.sub(r"[^a-z0-9]+", " ", s)         # punctuation, hyphens, newlines
    return " ".join(s.split())


def parse_bib(text):
    """(key, fields) for every entry. Field values may be brace- or quote-delimited."""
    out = []
    for m in re.finditer(r"@(\w+)\s*(\{)\s*([^,\s]+)\s*,", text):
        # brace-match from the entry's OPENING brace, not from the comma after the key
        i, depth = m.start(2), 0
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[m.end():i]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            j = fm.end()
            while j < len(body) and body[j] in " \t\n":
                j += 1
            if j >= len(body):
                continue
            if body[j] in "{\"":
                op, cl = ("{", "}") if body[j] == "{" else ("\"", "\"")
                k, d = j, 0
                while k < len(body):
                    if body[k] == op and (op != cl or k == j):
                        d += 1
                    elif body[k] == cl:
                        d -= 1
                        if d == 0:
                            break
                    k += 1
                fields[fm.group(1).lower()] = body[j + 1:k]
            else:
                k = body.find(",", j)
                fields[fm.group(1).lower()] = body[j:(k if k > 0 else len(body))].strip()
        out.append((m.group(3), fields))
    return out


def _meta(html, name):
    """All `content` values of <meta name="...">, HTML-unescaped, in document order."""
    out = []
    for m in re.finditer(r'<meta[^>]*\bname="' + name + r'"[^>]*>', html, re.I):
        c = re.search(r'\bcontent="([^"]*)"', m.group(0))
        if c:
            out.append(html_mod.unescape(c.group(1)).strip())
    return out


def _abs_page(aid, tries=5):
    """Fetch one arXiv abstract page.

    The export API (export.arxiv.org/api/query) was unreachable from this machine -- read
    timeouts, then sustained 429/503 -- so metadata is read from the abstract page's Highwire
    `citation_*` meta tags instead. Those are the same authoritative record the API serves,
    emitted for Google Scholar, and they are exact strings rather than rendered prose.
    """
    req = urllib.request.Request(
        ABS.format(aid),
        headers={"User-Agent": "Mozilla/5.0 (textca refs audit; citation verification)"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if k == tries - 1:
                raise
            wait = 10 * (k + 1)
            print(f"    ({type(e).__name__} on {aid}; retrying in {wait}s)", flush=True)
            time.sleep(wait)


def fetch(ids):
    got = {}
    for i, aid in enumerate(ids, 1):
        print(f"  [{i}/{len(ids)}] {aid}", flush=True)
        html = _abs_page(aid)
        title = _meta(html, "citation_title")
        if not title:
            print(f"      no citation_title -- treating as unresolvable")
            continue
        date = (_meta(html, "citation_date") or [""])[0]
        got[aid] = dict(
            arxiv_id=aid,
            title=" ".join(title[0].split()),
            authors=_meta(html, "citation_author"),
            published=date.replace("/", "-"),
            updated=date.replace("/", "-"))
        if i < len(ids):
            time.sleep(3.0)                    # be a polite client
    return got


def main():
    entries = parse_bib(BIB.read_text())
    with_id, without_id = [], []
    for key, f in entries:
        eid = (f.get("eprint") or "").strip()
        (with_id if re.fullmatch(r"\d{4}\.\d{4,5}", eid) else without_id).append((key, f, eid))

    print(f"{len(entries)} entries: {len(with_id)} with an arXiv id, {len(without_id)} without\n")
    for key, f, eid in without_id:
        if eid:
            print(f"  MALFORMED eprint  {key:18s} eprint={eid!r} -- not an arXiv identifier")

    meta = fetch([e for _, _, e in with_id])
    missing = [e for _, _, e in with_id if e not in meta]
    if missing:
        print(f"\n  arXiv returned nothing for: {missing}")

    verified, problems = {}, []
    print(f"\n{'key':18s} {'arxiv':>11s}  status")
    for key, f, eid in with_id:
        m = meta.get(eid)
        if not m:
            problems.append((key, "no-such-id", eid, ""))
            print(f"{key:18s} {eid:>11s}  NO SUCH ID")
            continue
        claim_t, real_t = f.get("title", ""), m["title"]
        bad = []
        if normalise(claim_t) != normalise(real_t):
            bad.append("title")
        # first author surname must appear in the real author list
        first = re.split(r"\s+and\s+", f.get("author", ""))[0]
        surname = first.split(",")[0].strip() if "," in first else first.split()[-1:] and \
            first.split()[-1]
        if surname and not any(normalise(surname) in normalise(a) for a in m["authors"]):
            bad.append("author")
        # A bib year LATER than the arXiv year is normal and correct: it is the peer-reviewed
        # venue's year (edgeofchaos2024 is ICLR 2025 on a 2024 preprint; simplicitybias is
        # ICML 2023 on a 2022 preprint). Only a year EARLIER than the preprint is impossible.
        yr, pub_yr = (f.get("year") or "").strip(), m["published"].split("-")[0]
        venue_year = None
        if yr and yr != pub_yr:
            if yr.isdigit() and pub_yr.isdigit() and int(yr) < int(pub_yr):
                bad.append(f"year(bib {yr} PRECEDES arXiv {pub_yr})")
            else:
                venue_year = yr

        verified[key] = dict(arxiv_id=eid, title=real_t, first_author=m["authors"][0],
                             n_authors=len(m["authors"]), published=m["published"],
                             arxiv_year=pub_yr, bib_year=yr or None,
                             venue_year_later_than_preprint=venue_year)
        if venue_year:
            print(f"{key:18s} {eid:>11s}  note: bib year {venue_year} > arXiv {pub_yr} "
                  f"(venue year, not an error)")
        if bad:
            problems.append((key, ",".join(bad), claim_t, real_t))
            print(f"{key:18s} {eid:>11s}  MISMATCH: {','.join(bad)}")
            if "title" in bad:
                print(f"{'':18s} {'':>11s}    bib:   {claim_t}")
                print(f"{'':18s} {'':>11s}    arXiv: {real_t}")
        else:
            print(f"{key:18s} {eid:>11s}  ok")

    OUT.write_text(json.dumps(dict(
        _note=("Citation metadata AS RETURNED BY THE ARXIV API, written by "
               "experiments/audit_refs.py. refs.bib is checked against this offline by "
               "tests/test_refs_match_arxiv.py. Regenerate after adding or editing any "
               "arXiv citation. Entries with no arXiv id are listed under `unverifiable` and "
               "must be checked by hand against the published venue."),
        verified=verified,
        unverifiable=[k for k, _, _ in without_id]), indent=1) + "\n")

    print(f"\n{len(problems)} mismatched of {len(with_id)} checked; wrote {rel(OUT)}")
    for key, what, claim, real in problems:
        print(f"  {key}: {what}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
