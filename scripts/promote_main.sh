#!/bin/sh
# Bring main up to date with dev, without CLAUDE.md and MEMORY.md (they live on dev only).
#   sh scripts/promote_main.sh            # merge dev into main locally (does not push)
#   sh scripts/promote_main.sh --push     # ... and push main
# Runs in a temporary separate checkout (git worktree), so this folder stays on dev and the two
# files never disappear from it. Stops on any merge conflict other than those two files.
set -eu
cd "$(git rev-parse --show-toplevel)"
PUSH=${1:-}
[ "$(git branch --show-current)" = dev ] || { echo "run this from the dev branch"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "commit or stash your changes on dev first"; exit 1; }
WT=$(mktemp -d "${TMPDIR:-/tmp}/etch-main.XXXXXX")
cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; rm -rf "$WT"; }
trap cleanup EXIT
git worktree add -q "$WT" main
cd "$WT"
git merge --no-ff --no-commit dev >/dev/null 2>&1 || true
git rm -q --cached --ignore-unmatch CLAUDE.md MEMORY.md
rm -f CLAUDE.md MEMORY.md
if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
  echo "merge conflicts to resolve by hand:"; git diff --name-only --diff-filter=U
  git merge --abort; exit 1
fi
if git diff --cached --quiet; then echo "main already matches dev"; git merge --abort 2>/dev/null || true; exit 0; fi
git commit -q -m "Merge dev into main (without CLAUDE.md, MEMORY.md)"
git log --oneline -1
[ "$PUSH" = --push ] && git push origin main
