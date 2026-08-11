#!/usr/bin/env bash
# Build and VERIFY the arXiv submission tarball.
#
# Three traps this exists to close, each of which silently ships a broken paper:
#
#   1. \graphicspath{{../fig/}} escapes the submission root. arXiv unpacks the tarball into one
#      directory and cannot follow `..`, so every figure would be missing. The staged copy rewrites
#      it to fig/ and the figures are copied in.
#   2. arXiv does not reliably run BibTeX, so main.bbl must be shipped. But shipping the .bbl ALONE
#      is not enough: without refs.bib the engine skips the rerun that resolves \cite keys against
#      it, and the paper builds "successfully" with 7 literal [?] markers in the text. Both files go
#      in. This was observed, not theorised.
#   3. neurips_2026.sty sits in paper_arxiv/ but is NOT loaded by main.tex -- a leftover from the
#      withdrawn submission. It is deliberately not staged; shipping unused style files invites
#      arXiv to pick a different compilation path.
#
# The verification is the point: the tarball is unpacked into a clean directory and built from its
# own contents, then the resulting PDF is checked for unresolved citations. A package that compiles
# is not the same as a package that compiles CORRECTLY.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
STAGE="$(mktemp -d)"; VERIFY="$(mktemp -d)"
OUT="$HERE/arxiv-submission.tar.gz"
trap 'rm -rf "$STAGE" "$VERIFY"' EXIT

command -v tectonic >/dev/null || { echo "tectonic not found"; exit 1; }

mkdir -p "$STAGE/fig"
cp "$HERE/main.tex" "$HERE/refs.bib" "$STAGE/"
# only the figures main.tex actually includes
grep -o 'includegraphics\[[^]]*\]{[^}]*}' "$HERE/main.tex" \
  | sed 's/.*{//;s/}//' | sort -u | while read -r f; do
      cp "$ROOT/fig/$f" "$STAGE/fig/$f"
  done
sed -i.bak 's|\\graphicspath{{\.\./fig/}}|\\graphicspath{{fig/}}|' "$STAGE/main.tex"
rm -f "$STAGE/main.tex.bak"

( cd "$STAGE" && tectonic -X compile main.tex --outdir . --keep-intermediates >/dev/null 2>&1 )
[ -f "$STAGE/main.bbl" ] || { echo "FAIL: main.bbl was not produced"; exit 1; }
rm -f "$STAGE"/main.{aux,log,out,blg,pdf}

( cd "$STAGE" && tar czf "$OUT" main.tex main.bbl refs.bib fig/ )

# VERIFY: build from the tarball alone, then check the OUTPUT, not just the exit status
tar xzf "$OUT" -C "$VERIFY"
( cd "$VERIFY" && tectonic -X compile main.tex --outdir . --keep-logs >/dev/null 2>&1 )
bad=$(grep -icE "undefined (citation|reference)" "$VERIFY/main.log" || true)
marks=0
if command -v pdftotext >/dev/null; then
  marks=$(pdftotext "$VERIFY/main.pdf" - 2>/dev/null | grep -c '\[?\]' || true)
fi
pages=$(command -v pdfinfo >/dev/null && pdfinfo "$VERIFY/main.pdf" | awk '/^Pages/{print $2}' || echo "?")

echo "  package : $OUT"
echo "  pages   : $pages"
echo "  undefined citations/references : $bad"
echo "  literal [?] markers in the PDF : $marks"
if [ "$bad" -ne 0 ] || [ "$marks" -ne 0 ]; then
  echo "  FAIL -- the tarball builds but the paper is wrong. Do not upload."
  exit 1
fi
echo "  OK -- verified from the tarball's own contents."
