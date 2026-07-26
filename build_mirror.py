"""Build the anonymised submission mirror from a tag, scrub it, and prove it is clean (#52).

WHY A SCRIPT. The mirror has to be rebuilt at least twice -- once from the submission tag, and
again after the mirror URL is inserted and the tree is re-tagged (#54) -- and a hand-run sed
pipeline that is correct the first time is not evidence that the second run was. This also makes
the scrub auditable: the patterns are here, in one place, next to the assertion that they worked.

WHAT IT DOES NOT DO. It never writes inside the repository. The evidence record keeps its logs
exactly as the analyses wrote them; the mirror is a derived artifact and the rewriting happens
only there. It also does not upload anything -- choosing and using a host is a human decision.

WHAT GETS SCRUBBED, AND WHY THAT IS NOT FALSIFICATION. Twelve machine-written logs end with a
line like

    wrote /Users/<user>/Documents/GitHub/textca/results/dev_transition_n192.json

Twelve experiment scripts print an absolute `OUT` path. Rewriting that prefix to `./` preserves
every fact the line carries -- which analysis wrote which file -- and removes only the accident
of where the checkout happened to live. The class fix (scripts printing repo-relative paths) is
deferred post-submission because it changes 12 source hashes and invalidates 7
`_analysis_provenance` stamps, which is the wrong trade before a content freeze.

IDENTIFIERS ARE DERIVED, NOT LISTED. Writing the author's handle into this file would reproduce
the defect found in `tests/test_paper_numbers.py`, whose anonymity guard hard-coded the very
identifier it existed to forbid -- in a file that ships inside the mirror. Everything here comes
from the git remote and the checkout path at run time.

Usage:
    .venv/bin/python build_mirror.py <tag> [--out DIR]
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent


def identifiers():
    """Strings the mirror must not contain, derived from the environment."""
    out = set()
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        url = ""
    m = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        out.update({m.group(1), m.group(2)})
    for key in ("user.name", "user.email"):
        try:
            v = subprocess.run(["git", "config", "--get", key], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            v = ""
        if v:
            out.add(v)
            if "@" in v:
                out.add(v.split("@")[1])            # the mail domain is identifying too
    parts = ROOT.parts
    for anchor in ("Users", "home"):
        if anchor in parts:
            out.add(parts[parts.index(anchor) + 1])
    return {s for s in out if s and len(s) > 3}


def build(tag, out_dir):
    prefix = "submission-mirror/"
    tar_path = out_dir / "archive.tar"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(tar_path, "wb") as fh:
        subprocess.run(["git", "archive", "--format=tar", f"--prefix={prefix}", tag],
                       cwd=ROOT, stdout=fh, check=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir)
    tar_path.unlink()
    return out_dir / prefix.rstrip("/")


def scrub(mirror):
    """Rewrite absolute paths. Returns (files_touched, edits).

    TWO passes, because one is not enough and the audit proved it. The first rewrites the
    CHECKOUT prefix to `./`, which covers the twelve "wrote <abs>/results/x.json" lines. The
    second catches home-directory paths that never pointed inside the repo at all -- three logs
    carry a Python warning traceback through
    `/Users/<user>/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py`, which the
    checkout-prefix pattern cannot match. A scrub written only against the leak you already
    know about is how a scrub ships incomplete.
    """
    home_user = None
    parts = ROOT.parts
    for anchor in ("Users", "home"):
        if anchor in parts:
            home_user = (anchor, parts[parts.index(anchor) + 1])
    subs = [(re.compile(re.escape(str(ROOT)) + r"/?"), "./")]
    if home_user:
        anchor, user = home_user
        subs.append((re.compile(rf"/{anchor}/{re.escape(user)}(?=/|\b)"), f"/{anchor}/anon"))
    touched, edits = [], 0
    for p in sorted(mirror.rglob("*")):
        if not p.is_file():
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue                                 # binary: checked separately below
        new, total = text, 0
        for pat, rep in subs:
            new, n = pat.subn(rep, new)
            total += n
        if total:
            p.write_text(new)
            touched.append(p.relative_to(mirror))
            edits += total
    return touched, edits


def neutralise_readme(mirror):
    """The README's H1 was the repository name; a repo name is a search key."""
    r = mirror / "README.md"
    if not r.exists():
        return False
    lines = r.read_text().splitlines()
    if lines and lines[0].startswith("#"):
        lines[0] = "# Token-lattice cellular automaton --- anonymised submission mirror"
        r.write_text("\n".join(lines) + "\n")
        return True
    return False


# A home path is identifying only while it names a REAL user. `/Users/anon/anaconda3/...` is
# the scrubbed form and must not be reported -- an audit that flags its own output is an audit
# nobody finishes reading.
_HOMEPATH = re.compile(rb"/(?:Users|home)/(?!anon\b)[A-Za-z0-9._-]+")


def audit(mirror, ids):
    """Every identifier must be absent from every file, text or binary."""
    hits = {}
    for p in sorted(mirror.rglob("*")):
        if not p.is_file():
            continue
        try:
            blob = p.read_bytes()
        except OSError:
            continue
        found = [s for s in sorted(ids) if s.encode() in blob]
        m = _HOMEPATH.search(blob)
        if m:
            found.append(m.group(0).decode("utf-8", "replace"))
        if found:
            hits[str(p.relative_to(mirror))] = found
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out).resolve() if a.out else ROOT.parent / "mirror-build"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    ids = identifiers()
    print(f"identifiers derived (not listed in this file): {len(ids)} string(s)")
    mirror = build(a.tag, out_dir)
    n_files = sum(1 for p in mirror.rglob("*") if p.is_file())
    print(f"archive of {a.tag}: {n_files} files")

    touched, edits = scrub(mirror)
    print(f"scrubbed {edits} absolute path(s) across {len(touched)} file(s):")
    for t in touched:
        print(f"    {t}")
    print(f"README H1 neutralised: {neutralise_readme(mirror)}")

    hits = audit(mirror, ids)
    if hits:
        print("\nAUDIT FAILED -- identifiers still present:")
        for f, ss in hits.items():
            print(f"    {f}: {len(ss)} pattern(s)")
        return 1
    print(f"\nAUDIT CLEAN: 0 identifier hits across {n_files} files")
    print(f"mirror at {mirror}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
