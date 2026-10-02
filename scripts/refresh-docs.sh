#!/usr/bin/env bash
#
# ONE COMMAND THAT REPUBLISHES EVERY GENERATED DOCUMENT, in dependency order.
#
# WHY THIS EXISTS. There are five generators and they feed each other. Running
# them in the wrong order, or forgetting one, publishes a page that disagrees with
# the repo, and on 2026-09-29 six figures on the hardening page were measurably
# wrong at once: B2 shown as a Fail with 64 open long after it passed with zero,
# the assessment path claiming no open defect beside a measured 13, the suite
# composition stale by 125 files and 2,047 test definitions, and a criterion table
# listing five of seven criteria. Every one of those was a generator that had not
# been run, not a number somebody typed wrong.
#
# Remembering a five step order is not a process. This is.
#
#   ./scripts/refresh-docs.sh           republish everything
#   ./scripts/refresh-docs.sh --check   exit 1 if anything published is stale
#
# TWO PASSES, ON PURPOSE. The generators are mutually dependent: the register
# publishes counts the manifest reads, and the readiness board reads the gate,
# which reads the ledger the manifest refreshes. One pass leaves the second reader
# holding the first reader's previous answer. The --check run after them is what
# proves the pair converged, so a cycle that does NOT settle is a loud failure
# rather than a page that is quietly one run behind.
set -uo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
CHECK=""
LIST=""
[ "${1:-}" = "--check" ] && CHECK="--check"
# --list prints PATHS AND NOTHING ELSE, because its whole job is to be the
# argument of a `git commit --`. A listing that also prints progress lines is
# one a caller has to parse, and a caller that parses goes back to `git status`.
[ "${1:-}" = "--list" ] && { CHECK="--check"; LIST="1"; }

fail=0
run() {  # run <label> <cmd...>
    local label="$1"; shift
    if ! out=$("$@" 2>&1); then
        echo "FAILED: $label"
        echo "$out" | sed 's/^/    /'
        fail=1
    elif [ -n "$CHECK" ] && [ -z "$LIST" ]; then
        echo "  ok  $label"
    fi
}

# CAN THIS ENVIRONMENT CHECK AT ALL. The published figures were measured with
# optional extras installed (mcp, torch and others), and without them the surface
# walk measures a smaller package: 1,563 units instead of 1,580 on 2026-10-02.
# Every check below would then print STALE and blame the documents. That is a
# could-not-check, so it is said once, by name, and nothing is reported as
# either current or stale. It still exits 1: an unchecked tree is never a pass.
# It runs BEFORE a republish too, so a partial install cannot overwrite the
# published ledger with its smaller count either.
if ! envout=$("$PY" scripts/capability_status.py --env-check 2>&1); then
    echo "$envout"
    exit 1
fi

if [ -z "$CHECK" ]; then
    # The two beta-gate checks run first: the quality page renders their result
    # file, so it has to exist before the page is stamped. It EXECUTES both checks
    # (about half a minute), because a gate figure copied from an older run is the
    # stale-figure defect this script exists to remove.
    echo "running the two beta-gate checks"
    "$PY" scripts/emit_gate_checks.py    >/dev/null 2>&1 || { echo "FAILED: beta-gate checks"; exit 1; }
    echo "pass 1 of 2"
    "$PY" scripts/capability_status.py   >/dev/null 2>&1
    "$PY" scripts/bgl6_register.py       >/dev/null 2>&1
    bash  scripts/refresh-manifest.sh    >/dev/null 2>&1
    "$PY" scripts/beta_go_live_docs.py   >/dev/null 2>&1
    "$PY" scripts/stamp_status_badges.py >/dev/null 2>&1
    "$PY" scripts/stamp_proof_status.py   >/dev/null 2>&1
    "$PY" scripts/render_readiness.py     >/dev/null 2>&1
    "$PY" scripts/build_search_index.py   >/dev/null 2>&1
    echo "pass 2 of 2"
    "$PY" scripts/capability_status.py   >/dev/null 2>&1
    "$PY" scripts/bgl6_register.py       >/dev/null 2>&1
    bash  scripts/refresh-manifest.sh    >/dev/null 2>&1
    "$PY" scripts/beta_go_live_docs.py   >/dev/null 2>&1
    "$PY" scripts/stamp_status_badges.py >/dev/null 2>&1
    "$PY" scripts/stamp_proof_status.py   >/dev/null 2>&1
    "$PY" scripts/render_readiness.py     >/dev/null 2>&1
    "$PY" scripts/build_search_index.py   >/dev/null 2>&1
    echo "verifying it converged"
fi

# The check pass runs in BOTH modes. After a republish it proves the two passes
# settled; on its own it is the CI gate.
run "capability ledger"          "$PY" scripts/capability_status.py --check
run "second-round register"      "$PY" scripts/bgl6_register.py --check
run "manifest and library stats" "$PY" scripts/library_kpis.py --check
run "beta and hardening markdown" "$PY" scripts/beta_go_live_docs.py --check
run "status badges and grid"     "$PY" scripts/stamp_status_badges.py --check
run "readiness summary"          "$PY" scripts/render_readiness.py --check
# THE DOCSTRING STAMPS ARE DOCUMENTATION TOO. Each public unit carries its proof
# status in its own docstring, dated to the newest evidence. A new grading wave
# moves that date, and on 2026-09-29 it left 212 stamps across 92 files reading
# two days stale while every other document was current. It was a separate
# command somebody had to remember, which is the same failure this script exists
# to remove.
run "docstring proof stamps"     "$PY" scripts/stamp_proof_status.py --check
# THE SEARCH INDEX IS READ FROM THE PAGES THIS SCRIPT REWRITES, so it is the
# last generator in each pass. It was not in this script at all: every republish
# rewrote the quality page (its "As of" line among others) and left
# docs/site/data/search-index.json describing the previous text, so the suite's
# test_search_index_is_current failed on CI on 2026-10-02, and by then
# the index still said 0.1.0 was unreleased a day after it shipped.
run "search index"               "$PY" scripts/build_search_index.py --check

if [ "$fail" -ne 0 ]; then
    echo
    echo "Something published disagrees with the repo. Run ./scripts/refresh-docs.sh"
    exit 1
fi
[ -z "$LIST" ] && echo "every generated document matches the repo"

# THE DOCUMENTS THIS SCRIPT OWNS, named here so a commit does not have to ask
# `git status` which files changed. That question is a query of SHARED state: this
# working tree has several sessions in it, and using its answer as a pathspec once
# swept twelve of another session's files into a commit. `--list` prints the set,
# and a caller can pass it to `git commit -- $(...)` knowing every path came from
# this script's own knowledge of what it writes.
if [ "${1:-}" = "--list" ]; then
    cat <<'PATHS'
docs/API_REFERENCE.md
docs/BETA.md
docs/CAPABILITY_STATUS.md
docs/QUALITY_AND_HARDENING.md
docs/RELEASE_PLAN.md
docs/bgl6-audit-register.json
docs/capability-status.json
docs/gate-checks.json
docs/readiness-history.json
docs/site/api-reference/index.html
docs/site/css/styles.css
docs/site/data/capability-status.json
docs/site/data/library-stats.json
docs/site/data/search-index.json
docs/site/quality-and-hardening/index.html
docs/site/status/index.html
src/vfairness/_capability_status.json
vfairness-manifest.json
PATHS
    # The 92 source files whose docstrings carry a proof stamp, named by the
    # stamper itself rather than guessed at here.
    "$PY" scripts/stamp_proof_status.py --list 2>/dev/null
fi
