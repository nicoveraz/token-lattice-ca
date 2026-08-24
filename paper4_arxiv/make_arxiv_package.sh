#!/usr/bin/env bash
# Build and VERIFY paper 4's arXiv submission tarball.
#
# Adapted from paper2_arxiv/make_arxiv_package.sh, for the reason that file gives for not sharing one
# script across papers: the papers differ in ways a single script would branch on, and a packaging
# script that silently does the wrong thing for one of three is worse than three that each do one.
#
# The traps this closes, each of which silently ships a broken or embarrassing submission:
#
#   1. THE DRAFTING NOTES BLOCK. main.tex opens with the lines addressed to whoever is writing it --
#      F186's three binding prohibitions, the note that E1 is a negative result and stays one,
#      and the instruction never to promote the tau=0.5 rung. arXiv distributes source. That block is
#      stripped from the STAGED copy, the repo copy keeps it, and the verification greps the tarball
#      to prove it is gone, because "I remembered to delete it" is not a check.
#   2. arXiv does not reliably run BibTeX, so main.bbl must ship -- and shipping it ALONE is not
#      enough. Without refs.bib the engine skips the rerun that resolves \cite keys and the paper
#      builds "successfully" with literal [?] markers in the text. Both files go in. Observed on
#      paper 1, not theorised.
#   3. \graphicspath{{../fig/}} escapes the submission root; arXiv unpacks into one directory and
#      cannot follow `..`. Paper 3 ships no figures TODAY and has no \graphicspath line at all, so
#      the rewrite below is a no-op right now. It is kept because the day someone adds a figure they
#      will add that line too, and the bug would then be live and invisible.
#
# ONE CHECK THIS ADDS BEYOND PAPER 2'S. The strip is the only transformation applied to the source,
# so a strip that ate too much is the failure mode with no symptom -- the tarball would still build,
# just as a different paper. The shipped file is therefore required to begin with \documentclass and
# to still carry both self-citations. That verifies the transformation rather than trusting awk.
#
# The verification is the point: the tarball is unpacked into a clean directory and built from its
# OWN contents, then the resulting PDF is inspected. A package that compiles is not the same as a
# package that compiles correctly.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
STAGE="$(mktemp -d)"; VERIFY="$(mktemp -d)"
OUT="$HERE/arxiv-submission.tar.gz"
trap 'rm -rf "$STAGE" "$VERIFY"' EXIT

command -v tectonic >/dev/null || { echo "tectonic not found"; exit 1; }

mkdir -p "$STAGE/fig"
cp "$HERE/main.tex" "$HERE/refs.bib" "$STAGE/"

# Strip the leading comment block: every line from the top up to the first non-comment line. Inline
# `% F188, results/...` source comments live below \documentclass and are NOT touched -- they are
# this paper's evidence convention, tests/test_paper3_numbers.py asserts the files they name exist,
# and a reader of the source should see them.
awk 'seen || !/^%/ { seen = 1; print }' "$STAGE/main.tex" > "$STAGE/main.stripped" \
  && mv "$STAGE/main.stripped" "$STAGE/main.tex"

# Only the figures main.tex actually includes -- none today. The `|| true` is why this script runs at
# all: `grep -o` exits 1 when it matches nothing, which under `set -euo pipefail` killed paper 2's
# copy before it printed a line. Paper 1's never hit it because paper 1 has figures.
{ grep -o 'includegraphics\[[^]]*\]{[^}]*}' "$HERE/main.tex" || true; } \
  | sed 's/.*{//;s/}//' | sort -u | while read -r f; do
      [ -n "$f" ] && cp "$ROOT/fig/$f" "$STAGE/fig/$f"
  done
sed -i.bak 's|\\graphicspath{{\.\./fig/}}|\\graphicspath{{fig/}}|' "$STAGE/main.tex"
rm -f "$STAGE/main.tex.bak"

( cd "$STAGE" && tectonic -X compile main.tex --outdir . --keep-intermediates >/dev/null 2>&1 )
[ -f "$STAGE/main.bbl" ] || { echo "FAIL: main.bbl was not produced"; exit 1; }
rm -f "$STAGE"/main.{aux,log,out,blg,pdf}

if [ -n "$(ls -A "$STAGE/fig" 2>/dev/null)" ]; then
  ( cd "$STAGE" && tar czf "$OUT" main.tex main.bbl refs.bib fig/ )
else
  rmdir "$STAGE/fig"
  ( cd "$STAGE" && tar czf "$OUT" main.tex main.bbl refs.bib )
fi

# VERIFY: build from the tarball alone, then check the OUTPUT, not just the exit status
tar xzf "$OUT" -C "$VERIFY"
( cd "$VERIFY" && tectonic -X compile main.tex --outdir . --keep-logs >/dev/null 2>&1 )
bad=$(grep -icE "undefined (citation|reference)" "$VERIFY/main.log" || true)
notes=$(grep -c "DRAFTING NOTES" "$VERIFY/main.tex" || true)
# USES only: a use is `\citepend{...}`, while the definition reads `\newcommand{\citepend}[1]`, so
# this pattern cannot match the tripwire itself. Anything above zero is an unresolved citation about
# to be distributed in red.
pend=$(grep -c 'citepend{' "$VERIFY/main.tex" || true)
# The strip is the only transformation, so verify it rather than trust it.
head1=$(head -1 "$VERIFY/main.tex")
selfcites=$(grep -c 'veraz2026probes\|veraz2026domain' "$VERIFY/main.tex" || true)
marks=0
if command -v pdftotext >/dev/null; then
  marks=$(pdftotext "$VERIFY/main.pdf" - 2>/dev/null | grep -c '\[?\]' || true)
fi
pages=$(command -v pdfinfo >/dev/null && pdfinfo "$VERIFY/main.pdf" | awk '/^Pages/{print $2}' || echo "?")

echo "  package : $OUT"
echo "  pages   : $pages"
echo "  undefined citations/references : $bad  (must be 0)"
echo "  literal [?] markers in the PDF : $marks  (must be 0)"
echo "  DRAFTING NOTES in shipped .tex : $notes  (must be 0)"
echo "  \\citepend USES in shipped .tex : $pend  (must be 0; the definition is not a use)"
echo "  shipped .tex begins with       : ${head1:0:30}  (must be \\documentclass)"
echo "  self-citation lines kept       : $selfcites  (must be >0; the strip must not eat the body)"
fail=0
[ "$bad" -ne 0 ] && fail=1
[ "$marks" -ne 0 ] && fail=1
[ "$notes" -ne 0 ] && fail=1
[ "$pend" -ne 0 ] && fail=1
case "$head1" in \\documentclass*) ;; *) fail=1 ;; esac
[ "$selfcites" -eq 0 ] && fail=1
if [ "$fail" -ne 0 ]; then
  echo "  FAIL -- do not upload."
  exit 1
fi
echo "  OK -- verified from the tarball's own contents."
