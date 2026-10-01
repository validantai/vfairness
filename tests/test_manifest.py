"""
Tests for the vfairness capability manifest.

Validates that:
1. Every registered name resolves as an attribute of its module_path and is
   the declared kind (class / function). That is the registry contract.
   Membership in vfairness.__all__ is NOT part of it (F20, 2026-09-09).
2. Every SVG template referenced in registry exists
3. The committed vfairness-manifest.json matches generate_manifest() output
4. No duplicate dispatch keys exist
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

from vfairness._manifest import generate_manifest
from vfairness._registry import CAPABILITY_REGISTRY

# ---------------------------------------------------------------------------
# 1. Every registered class/function is importable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dispatch_key,entry",
    list(CAPABILITY_REGISTRY.items()),
    ids=list(CAPABILITY_REGISTRY.keys()),
)
def test_capability_is_importable(dispatch_key: str, entry: dict):
    """The registry contract, asserted where it is enforced (F20, 2026-09-09).

    ``ManifestEntry.name`` is the attribute under which the capability is
    reachable on ``vfairness.<module_path>``, and ``kind`` says whether that
    attribute is a class or a function. That is ALL the registry promises.
    Membership in ``vfairness.__all__`` is NOT part of the contract: measured
    2026-09-09, 58 of 194 entries were not top-level exports
    (FairnessAwareBCELoss, ConditionalIndependenceRegularizer, ...) while the
    comment on ManifestEntry claimed "must match __all__ export" and nothing
    checked it. Widening ``__all__`` is a public-surface decision taken
    separately; this test must not force it.
    """
    module_path = f"vfairness.{entry['module_path']}"
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"Cannot import module '{module_path}' for {dispatch_key}: {exc}")

    name = entry["name"]
    assert hasattr(mod, name), (
        f"Module '{module_path}' does not export '{name}' (dispatch_key={dispatch_key})"
    )
    obj = getattr(mod, name)
    kind = entry["kind"]
    if kind == "class":
        assert inspect.isclass(obj), (
            f"{dispatch_key}: '{name}' is registered as a class but "
            f"{module_path}.{name} is a {type(obj).__name__}"
        )
    elif kind == "function":
        assert callable(obj) and not inspect.isclass(obj), (
            f"{dispatch_key}: '{name}' is registered as a function but "
            f"{module_path}.{name} is a {type(obj).__name__}"
        )
    else:
        pytest.fail(f"{dispatch_key}: unknown kind {kind!r}; expected 'class' or 'function'")


def test_registry_contract_is_importability_from_module_path_not_all_membership():
    """The stated contract and the enforced contract must be the same one.

    Before 2026-09-09 the ManifestEntry comment said the name "must match
    __all__ export" while the only check was importability from module_path;
    58 of 194 entries failed the claim and nothing noticed. This pins the
    wording to the check: the schema comment may not reinstate the __all__
    claim, and every name that is NOT a top-level export must still satisfy
    the contract that IS enforced. The count itself is deliberately not
    asserted; it is a public-surface decision, not a correctness property.
    """
    import vfairness
    import vfairness._registry as registry_module

    source = Path(registry_module.__file__).read_text()
    field_line = re.search(r"^\s*name: str\s*#(?P<comment>.*)$", source, re.MULTILINE)
    assert field_line is not None, "ManifestEntry.name lost its contract comment"
    comment = field_line.group("comment")
    assert "must match __all__" not in comment and "module_path" in comment, (
        "ManifestEntry.name claims __all__ membership again; either enforce it with a "
        f"deliberate new test or keep the contract as importability from module_path: {comment!r}"
    )

    not_exported = sorted(
        {entry["name"] for entry in CAPABILITY_REGISTRY.values()} - set(vfairness.__all__)
    )
    unresolvable = []
    for entry in CAPABILITY_REGISTRY.values():
        if entry["name"] not in not_exported:
            continue
        mod = importlib.import_module(f"vfairness.{entry['module_path']}")
        if not hasattr(mod, entry["name"]):
            unresolvable.append(f"{entry['name']} from vfairness.{entry['module_path']}")
    assert not unresolvable, (
        "Registry names that are neither top-level exports NOR importable from "
        f"their module_path (the enforced contract): {unresolvable}"
    )


# ---------------------------------------------------------------------------
# 2. Every referenced SVG template exists
# ---------------------------------------------------------------------------


def test_svg_templates_exist():
    """All SVG templates referenced in registry entries must exist on disk."""
    try:
        from vfairness.rendering.engine import list_templates

        available = set(list_templates())
    except ImportError:
        pytest.skip("Rendering engine not available (jinja2 not installed)")

    missing = []
    for dispatch_key, entry in CAPABILITY_REGISTRY.items():
        for tpl in entry.get("svg_templates", []):
            if tpl not in available:
                missing.append(f"{dispatch_key} references '{tpl}'")

    assert not missing, "SVG templates not found:\n" + "\n".join(f"  - {m}" for m in missing)


# ---------------------------------------------------------------------------
# 3. Committed manifest matches generated output
# ---------------------------------------------------------------------------


def test_manifest_is_fresh():
    """The committed vfairness-manifest.json must match the current registry."""
    manifest_path = Path(__file__).parent.parent / "vfairness-manifest.json"
    if not manifest_path.exists():
        pytest.skip("vfairness-manifest.json not found (run: python -m vfairness._manifest)")

    committed = json.loads(manifest_path.read_text())
    current = generate_manifest()

    # Compare capabilities and templates. The manifest is deterministic (no
    # timestamp); LOC stats lag between refreshes, so they are not asserted here.
    assert committed["capabilities"] == current["capabilities"], (
        "vfairness-manifest.json is stale. Regenerate with:\n"
        "  python -m vfairness._manifest > vfairness-manifest.json"
    )
    assert committed["svg_templates"] == current["svg_templates"], (
        "SVG template list in manifest is stale."
    )


# ---------------------------------------------------------------------------
# 4. No duplicate dispatch keys (structural validation)
# ---------------------------------------------------------------------------


def test_no_duplicate_dispatch_keys_in_source():
    """No duplicate literal keys in the CAPABILITY_REGISTRY dict SOURCE.

    A runtime check on dict.keys() is tautological (Python silently keeps
    only the LAST duplicate at parse time, so the dict can never show one).
    Parse the module's AST instead: a duplicated literal key in the source
    means one registry entry silently shadows another.
    """
    import ast
    from collections import Counter

    import vfairness._registry as registry_module

    source = Path(registry_module.__file__).read_text()
    tree = ast.parse(source)

    dict_node = None
    for node in ast.walk(tree):
        # The registry is annotated: CAPABILITY_REGISTRY: dict[...] = {...}
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "CAPABILITY_REGISTRY"
            and isinstance(node.value, ast.Dict)
        ):
            dict_node = node.value
            break

    assert dict_node is not None, (
        "Could not find the CAPABILITY_REGISTRY dict literal in "
        "vfairness/_registry.py; if the registry is no longer a single "
        "dict literal, update this test to match the new construction."
    )

    literal_keys = [
        k.value for k in dict_node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
    ]
    # Sanity: the literal must account for the whole runtime registry,
    # otherwise this check is silently scanning the wrong structure.
    assert len(literal_keys) >= len(CAPABILITY_REGISTRY)

    dupes = {k: n for k, n in Counter(literal_keys).items() if n > 1}
    assert not dupes, (
        f"Duplicate dispatch keys in _registry.py source (later entries "
        f"silently shadow earlier ones): {dupes}"
    )


def test_dispatch_key_matches_dict_key():
    """The dispatch_key field must match the dict key for every entry."""
    mismatches = []
    for key, entry in CAPABILITY_REGISTRY.items():
        if entry["dispatch_key"] != key:
            mismatches.append(f"{key} has dispatch_key='{entry['dispatch_key']}'")
    assert not mismatches, "Dict key must match dispatch_key field:\n" + "\n".join(
        f"  - {m}" for m in mismatches
    )


# ---------------------------------------------------------------------------
# 5. Valid pipeline stages
# ---------------------------------------------------------------------------

VALID_STAGES = {"preprocessing", "in_processing", "post_processing", "evaluation", "operations"}


def test_valid_pipeline_stages():
    """All entries must have a valid pipeline_stage."""
    invalid = [
        f"{k}: '{v['pipeline_stage']}'"
        for k, v in CAPABILITY_REGISTRY.items()
        if v["pipeline_stage"] not in VALID_STAGES
    ]
    assert not invalid, "Invalid pipeline stages:\n" + "\n".join(f"  - {i}" for i in invalid)
