#!/usr/bin/env bash
# Build and VERIFY paper 2's arXiv submission tarball.
#
# Adapted from paper_arxiv/make_arxiv_package.sh rather than shared with it. The two papers differ in
# ways a single script would have to branch on anyway (paper 2 ships no figures and carries a drafting
# header paper 1 does not), and a packaging script that silently does the wrong thing for one of two
# papers is worse than two scripts that each do one thing.
#
# The traps this closes, each of which silently ships a broken or embarrassing submission:
#
#   1. THE DRAFTING NOTES BLOCK. main.tex opens with a comment block addressed to whoever is writing
#      it -- what was reframed after the prior-art gate, which headline claims were withdrawn, an
#      instruction not to reinstate an old framing from a named git commit. arXiv distributes source.
#      That block is stripped from the STAGED copy and the repo copy keeps it; the verification below
#      greps the tarball to prove it is gone, because "I remembered to delete it" is not a check.
#   2. \graphicspath{{../fig/}} escapes the submission root. arXiv unpacks into one directory and
#      cannot follow `..`. Paper 2 includes no figures today, so this is latent rather than live --
#      the rewrite and the figure staging are kept anyway, so adding one figure later does not
#      quietly reintroduce the bug that would then be live.
#   3. arXiv does not reliably run BibTeX, so main.bbl must ship. Shipping the .bbl ALONE is not
#      enough: without refs.bib the engine skips the rerun that resolves \cite keys, and the paper
#      builds "successfully" with literal [?] markers in the text. Both files go in. Observed on
#      paper 1, not theorised.
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

# Strip the leading comment block: every line from the top up to the first non-comment line. That
# block is the drafting header; inline `% F165, results/...` source comments live below
# \documentclass and are NOT touched -- they are part of the paper's evidence convention and a reader
# of the source should see them.
awk 'seen || !/^%/ { seen = 1; print }' "$STAGE/main.tex" > "$STAGE/main.stripped" \
  && mv "$STAGE/main.stripped" "$STAGE/main.tex"

# Only the figures main.tex actually includes -- none today, and the `|| true` is why this script
# runs at all. `grep -o` exits 1 when it matches nothing, which under `set -euo pipefail` killed the
# whole script before it printed a single line. Paper 1's copy never hit it because paper 1 has
# figures. A packaging script that dies silently on the no-figures case is exactly the failure this
# file's header claims to prevent, so it is fixed rather than worked around.
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
# USES only: a use is `\citepend{...}`, while the definition reads `\newcommand{\citepend}[1]`,
# so this pattern cannot match the tripwire itself. Anything above zero is an unresolved citation
# about to be distributed in red.
pend=$(grep -c 'citepend{' "$VERIFY/main.tex" || true)
marks=0
if command -v pdftotext >/dev/null; then
  marks=$(pdftotext "$VERIFY/main.pdf" - 2>/dev/null | grep -c '\[?\]' || true)
fi
pages=$(command -v pdfinfo >/dev/null && pdfinfo "$VERIFY/main.pdf" | awk '/^Pages/{print $2}' || echo "?")

echo "  package : $OUT"
echo "  pages   : $pages"
echo "  undefined citations/references : $bad"
echo "  literal [?] markers in the PDF : $marks"
echo "  DRAFTING NOTES in shipped .tex : $notes  (must be 0)"
echo "  \\citepend USES in shipped .tex : $pend  (must be 0; the definition is not a use)"
if [ "$bad" -ne 0 ] || [ "$marks" -ne 0 ] || [ "$notes" -ne 0 ] || [ "$pend" -ne 0 ]; then
  echo "  FAIL -- do not upload."
  exit 1
fi
echo "  OK -- verified from the tarball's own contents."
