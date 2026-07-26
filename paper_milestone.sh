#!/usr/bin/env bash
set -euo pipefail
REPO="nicoveraz/token-lattice-ca"
MS_SUB="neurips26-submission"
MS_POST="post-submission"
DUE="2026-08-29T23:59:00Z"
DUMP="tracker_dump.md"
DRY=1
[[ "${1:-}" == "--go" ]] && DRY=0
[[ $DRY -eq 1 ]] && echo "=== DRY RUN — nothing will be created. Re-run with --go. ===" || true
command -v gh >/dev/null || { echo "gh not found"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not authenticated"; exit 1; }
mklabel() {
  local name="$1" color="$2" desc="$3"
  if [[ $DRY -eq 1 ]]; then echo "DRY  label: $name"; return 0; fi
  gh label create "$name" --repo "$REPO" --color "$color" --description "$desc" --force >/dev/null
  echo "  label: $name"
}
echo "--- labels ---"
mklabel blocking     B60205 "Must close before the submission tag is cut"
mklabel evidence     1D76DB "Touches results/ or logs/ — must land before content freeze (Gate A)"
mklabel paper        0E8A16 "Touches paper.tex or the built PDF (Gate B)"
mklabel packaging    5319E7 "Tag, mirror, anonymisation, checklist (Gate C)"
mklabel post-paper   FBCA04 "Deferred past submission; do not start before the tag"
mklabel scope-closed BFBFBF "Out of scope for this paper — closed with a pointer, not left open"
mkmilestone() {
  local title="$1" due="$2" desc="$3"
  if gh api "repos/$REPO/milestones?state=all" --jq '.[].title' 2>/dev/null | grep -qxF "$title"; then
    echo "  SKIP (exists): $title"; return 0
  fi
  if [[ $DRY -eq 1 ]]; then echo "DRY  milestone: $title (due ${due:-none})"; return 0; fi
  if [[ -n "$due" ]]; then
    gh api "repos/$REPO/milestones" -f title="$title" -f due_on="$due" -f description="$desc" >/dev/null
  else
    gh api "repos/$REPO/milestones" -f title="$title" -f description="$desc" >/dev/null
  fi
  echo "  milestone: $title"
}
echo "--- milestones ---"
mkmilestone "$MS_SUB"  "$DUE" "Interp4Discovery @ NeurIPS 2026. Deadline Aug 29 11:59pm AOE (= Aug 30 11:59 UTC). Gate A content freeze Aug 7; Gate C tag Aug 15-21; submit Aug 27."
mkmilestone "$MS_POST" ""     "Work that resumes only after the submission tag is cut. Nothing here may write to results/ before then."

# --- issues (bodies live in /tmp/i4d_bodies, written by the companion python step) ---
BODIES="/tmp/i4d_bodies"
MANIFEST="$BODIES/manifest.json"
[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST — run the body-writer first"; exit 1; }

existing=$(gh issue list --repo "$REPO" --state all --limit 300 --json title --jq '.[].title')

n=$(python3 -c "import json;print(len(json.load(open('$MANIFEST'))))")
for i in $(seq 0 $((n-1))); do
  title=$(python3 -c "import json;print(json.load(open('$MANIFEST'))[$i]['title'])")
  labels=$(python3 -c "import json;print(json.load(open('$MANIFEST'))[$i]['labels'])")
  bodyf=$(python3 -c "import json;print(json.load(open('$MANIFEST'))[$i]['body_file'])")
  if grep -qxF "$title" <<< "$existing"; then
    echo "  SKIP (exists): $title"; continue
  fi
  if [[ $DRY -eq 1 ]]; then
    echo "DRY  issue [$labels] ($MS_SUB): $title"; continue
  fi
  gh issue create --repo "$REPO" --title "$title" --label "$labels" \
     --milestone "$MS_SUB" --body-file "$bodyf" | sed 's/^/  /'
done

echo "--- dumping current tracker to $DUMP ---"
{
  echo "# Tracker dump -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "| # | state | milestone | labels | title |"
  echo "|---|---|---|---|---|"
  gh issue list --repo "$REPO" --state all --limit 300 \
     --json number,state,title,labels,milestone \
     --jq '.[] | "| \(.number) | \(.state) | \(.milestone.title // "-") | \([.labels[].name] | join(",")) | \(.title) |"'
} > "$DUMP"
echo "  wrote $DUMP ($(wc -l < "$DUMP") lines)"
