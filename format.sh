#!/usr/bin/env bash
# Usage:
#    bash ./format.sh              # Check/format files changed vs origin/main
#    bash ./format.sh --all        # Format the entire codebase
#
# This script formats all Python files that differ from origin/main.
# You are encouraged to run this locally before pushing changes for review.

set -eo pipefail

# Install yapf/ruff if not available
if ! (yapf --version &>/dev/null && ruff --version &>/dev/null); then
    pip install --upgrade pip
    pip install 'yapf==0.40.2' 'ruff>=0.6.0'
fi

YAPF_VERSION=$(yapf --version | awk '{print $2}')
RUFF_VERSION=$(ruff --version | awk '{print $2}')

echo "yapf: ${YAPF_VERSION}"
echo "ruff: ${RUFF_VERSION}"

# ----------------------------------------------------------------------
# YAPF — Python formatter
# ----------------------------------------------------------------------
echo 'yapf: Check Start'

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" .
}

format_changed() {
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             yapf --in-place "${YAPF_FLAGS[@]}"
    fi
}

if [[ "$1" == '--all' ]]; then
    format_all
else
    format_changed
fi
echo 'yapf: Done'

# ----------------------------------------------------------------------
# Ruff — Python linter
# ----------------------------------------------------------------------
echo 'ruff: Check Start'

lint_all() {
    ruff check .
}

lint_changed() {
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             ruff check
    fi
}

if [[ "$1" == '--all' ]]; then
    lint_all
else
    lint_changed
fi
echo 'ruff: Done'

# ----------------------------------------------------------------------
# Check for uncommitted changes
# ----------------------------------------------------------------------
if ! git diff --quiet &>/dev/null; then
    echo ''
    echo 'Reformatted files. Please review and stage the changes.'
    echo ''
    echo 'Changed files:'
    git --no-pager diff --name-only

    echo ''
    echo 'You can also review the full diff below:'
    echo ''
    git --no-pager diff

    exit 1
fi

echo ''
echo 'All checks passed.'
