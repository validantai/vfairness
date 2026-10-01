"""Canonical module taxonomy guard (VB-PKG-3).

The docs historically stated 6, 8, and 11 modules in different places. This test
pins the ONE canonical set of top-level public sub-packages and enforces that
every one of them declares its public surface with ``__all__``. It is the source
of truth: changing the package layout must update this list in the same commit
(and the docs), so the "how many modules" question can never drift again.

Taxonomy (15 top-level public sub-packages):

  Core fairness pipeline (6): preprocessing, in_processing, post_processing,
      evaluation, operations, rendering.
  Specialized analysis surfaces (8): llm, agents, multi_agent, xai, vision,
      legal, mcp, validity.
  Cross-cutting infrastructure (1): net.

Private/internal helpers are underscore-prefixed modules (e.g. ``_bands``,
``_manifest``, ``_registry``) and are intentionally NOT part of this set.
"""

import importlib
import os

import vfairness

PIPELINE = [
    "preprocessing",
    "in_processing",
    "post_processing",
    "evaluation",
    "operations",
    "rendering",
]
SPECIALIZED = ["llm", "agents", "multi_agent", "xai", "vision", "legal", "mcp", "validity"]
# Cross-cutting plumbing every surface depends on, not a fairness-analysis surface:
# net holds the SSRF egress guard that all outbound engine calls go through.
INFRASTRUCTURE = ["net"]
CANONICAL_SUBPACKAGES = sorted(PIPELINE + SPECIALIZED + INFRASTRUCTURE)


def _actual_public_subpackages():
    root = os.path.dirname(os.path.abspath(vfairness.__file__))
    out = []
    for name in os.listdir(root):
        if name.startswith("_") or name.startswith("."):
            continue
        if os.path.isfile(os.path.join(root, name, "__init__.py")):
            out.append(name)
    return sorted(out)


def test_canonical_subpackage_set_matches_disk():
    actual = _actual_public_subpackages()
    assert actual == CANONICAL_SUBPACKAGES, (
        "Top-level public sub-package set changed. Update CANONICAL_SUBPACKAGES "
        "here AND the module taxonomy in the docs/README in the same commit.\n"
        f"Added: {sorted(set(actual) - set(CANONICAL_SUBPACKAGES))}\n"
        f"Removed: {sorted(set(CANONICAL_SUBPACKAGES) - set(actual))}"
    )


def test_pipeline_and_specialized_are_disjoint_and_complete():
    assert set(PIPELINE).isdisjoint(SPECIALIZED)
    assert set(PIPELINE).isdisjoint(INFRASTRUCTURE)
    assert set(SPECIALIZED).isdisjoint(INFRASTRUCTURE)
    assert sorted(PIPELINE + SPECIALIZED + INFRASTRUCTURE) == CANONICAL_SUBPACKAGES
    assert len(PIPELINE) == 6 and len(SPECIALIZED) == 8 and len(INFRASTRUCTURE) == 1


def test_every_public_subpackage_declares_all():
    missing = []
    for pkg in CANONICAL_SUBPACKAGES:
        mod = importlib.import_module(f"vfairness.{pkg}")
        if not hasattr(mod, "__all__"):
            missing.append(pkg)
    assert not missing, (
        f"Public sub-packages without __all__ (public surface undeclared): {missing}"
    )
