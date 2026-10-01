"""
vfairness Manifest Generator

Produces a JSON manifest of all registered capabilities and SVG templates.
The platform consumes this to derive coverage status automatically.

Usage:
    python -m vfairness._manifest > vfairness-manifest.json
    python -m vfairness._manifest                           # prints to stdout
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _count_lines_of_code() -> dict[str, int]:
    """Count lines of code in the shipped ``vfairness`` package.

    Walks the package this module lives in (``src/vfairness``), skipping
    ``__pycache__``, and returns a defensible breakdown:

    - ``total``: physical lines across all ``*.py`` files.
    - ``code``: non-blank, non-comment lines (the strict "lines of code";
      note a full-line ``#`` comment counts as a comment, inline comments do
      not, and multi-line-string bodies count as code, a standard, tool-free
      approximation).
    - ``files``: number of ``*.py`` files counted.

    Computed at manifest-generation time so the figure can never go stale.
    """
    package_dir = Path(__file__).resolve().parent
    total = code = files = 0
    for py in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        try:
            lines = py.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        files += 1
        total += len(lines)
        code += sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    return {"total": total, "code": code, "files": files}


def _proof_counts(proof: dict[str, Any]) -> dict[str, int]:
    """Capabilities per Beta Go-Live batch, plus the open-defect totals.

    Counted here rather than written down anywhere, so the published table can
    never drift from the ledger it claims to summarise.
    """
    counts: dict[str, int] = {}
    open_defects = 0
    fix_minutes = 0
    for key, rec in proof.items():
        counts[rec["batch"]] = counts.get(rec["batch"], 0) + 1
        # Two dispatch keys can resolve to ONE object and therefore carry the same
        # finding. Summing every record would report that finding twice and
        # inflate both the defect count and the hours. Count it once, against the
        # first key of its group.
        group = sorted([key, *rec.get("shared_with", [])])
        if group[0] != key:
            continue
        open_defects += rec["open_defects"]
        fix_minutes += rec["fix_minutes"]
    counts["open_defects"] = open_defects
    counts["fix_minutes"] = fix_minutes
    return counts


def generate_manifest() -> dict[str, Any]:
    """Collect registry + template discovery into a JSON-serializable manifest."""
    from ._proof_status import (
        CAPABILITY_PROOF,
        COUNTS_AS_OF,
        MEASURED_AT_COMMIT,
        MEASURED_ON,
        PROOF_BATCHES,
    )
    from ._registry import CAPABILITY_REGISTRY, DEFERRED_FROM_CAPABILITY_REGISTRY

    try:
        from .rendering.engine import list_templates

        # Count CHART templates only. list_templates() also returns internal partials
        # (e.g. `_shared_defs`, a shared <defs> block every chart <use>s); those are
        # not renderable charts, so an underscore-prefixed name is excluded from the
        # user-facing template count and list, matching what the SVG gallery documents.
        svg_templates = [t for t in list_templates() if not t.startswith("_")]
    except ImportError:
        svg_templates = []

    class_names = sorted(
        {entry["name"] for entry in CAPABILITY_REGISTRY.values() if entry["kind"] == "class"}
    )
    function_names = sorted(
        {entry["name"] for entry in CAPABILITY_REGISTRY.values() if entry["kind"] == "function"}
    )

    # NB: no ``generated_at`` timestamp. The manifest is a pure function of the
    # code (registry + templates + LOC), so a regeneration with no code change
    # produces byte-identical output. A wall-clock timestamp used to defeat the
    # "only rewrite if changed" guard in refresh-manifest.sh and surface the file
    # as spuriously dirty on every run (and across parallel instances). Nothing
    # consumes the timestamp; git history records when the manifest last changed.
    return {
        "version": _get_version(),
        "capabilities": CAPABILITY_REGISTRY,
        # LF-54, 2026-09-10. The deferral ledger is EMITTED, and until now it was
        # not. `DEFERRED_FROM_CAPABILITY_REGISTRY` existed so that a capability
        # staying out of the registry was a recorded decision rather than an
        # omission, and the record lived only in the source: the manifest carried
        # `capabilities` alone, so on the platform's coverage matrix a
        # deliberately excluded capability was indistinguishable from a gap.
        #
        # That is the defect this ledger was built to prevent, one layer up. The
        # decision was made, written down, and never reached the surface that
        # reads it. A reader of the matrix could not tell "we looked at this and
        # it is covered through another row" from "nobody has done this yet",
        # which is the same two-states-where-there-are-three shape as everything
        # else in the READINESS-6 register.
        #
        # Each reason states which it is: NOT A GAP or GATED, and what unblocks
        # the gated ones.
        "deferred": DEFERRED_FROM_CAPABILITY_REGISTRY,
        # BETA GO-LIVE, 2026-09-11. The platform's coverage matrix reads this
        # manifest, and until now a capability that had never been checked for the
        # fabricated-measurement defect was indistinguishable there from one that
        # had been checked and passed. That is the same two-states-where-there-are-
        # three shape the whole campaign exists to remove, one layer up: the
        # measurement was made, written down in _proof_status.py, and did not reach
        # the surface that reads it.
        #
        # Every registered capability carries a batch. BGL-C (UNPROVEN) is a
        # POSITIVE statement that nothing is known, never an absence, so a reader of
        # the matrix cannot mistake silence for a clean bill of health.
        "proof_batches": PROOF_BATCHES,
        "proof_status": CAPABILITY_PROOF,
        "proof_measured_on": MEASURED_ON,
        # The date the COUNTS are true for, which is NOT proof_measured_on.
        # A consumer given only the census date attaches it to whatever number
        # is nearest, and then reports that the census found what REMAINS.
        # Both halves true, sentence false; found on this page and on the
        # platform's coverage card on the same day.
        "proof_counts_as_of": COUNTS_AS_OF,
        "proof_measured_at_commit": MEASURED_AT_COMMIT,
        "svg_templates": svg_templates,
        "class_names": class_names,
        "function_names": function_names,
        "stats": {
            "total_capabilities": len(CAPABILITY_REGISTRY),
            "deferred_capabilities": len(DEFERRED_FROM_CAPABILITY_REGISTRY),
            "proof": _proof_counts(CAPABILITY_PROOF),
            "classes": len(class_names),
            "functions": len(function_names),
            "svg_templates": len(svg_templates),
            "lines_of_code": _count_lines_of_code(),
        },
    }


def _get_version() -> str:
    try:
        from . import __version__

        return __version__
    except ImportError:
        return "unknown"


def main() -> None:
    manifest = generate_manifest()
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
