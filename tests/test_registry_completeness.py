"""
Registry completeness: nothing implemented may be missing from the ledger.

test_manifest.py validates the FORWARD direction (every registered entry is
importable, the manifest is fresh). This file validates the INVERSE: every
implemented capability must appear in CAPABILITY_REGISTRY or, deliberately
and with a reason, in DEFERRED_FROM_CAPABILITY_REGISTRY.

Why this exists: the vfairness.xai package (TreeSHAP / LinearSHAP /
KernelSHAP / LIME / DiCE / Anchors / IG) shipped through the consumer task
lane `vfairness_xai_explain` without a single registry entry. The registry is
what generates the manifest, the manifest is what the platform's coverage
matrix reads, so the platform reported shipping explainers as "Not Covered"
for months (found 2026-08-23). The registry stays hand-curated on purpose
(gated work like the validity axis must be able to stay out), but staying out
must be a recorded decision, never an omission.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from vfairness._registry import (
    CAPABILITY_REGISTRY,
    DEFERRAL_COVERS,
    DEFERRED_FROM_CAPABILITY_REGISTRY,
)
from vfairness.xai.explainers.registry import _NOT_YET_IMPLEMENTED, _REGISTRY

REGISTERED_NAMES = {entry["name"] for entry in CAPABILITY_REGISTRY.values()}
DEFERRED_NAMES = {name for names in DEFERRAL_COVERS.values() for name in names}

#: Packages whose whole public surface must be accounted for.
#:
#: LF-54, 2026-09-10. This file checked the XAI package and NOTHING ELSE, and the
#: `agents` package had ZERO registry entries: six shipped, dispatched
#: capabilities that the platform's coverage matrix derived as "Not Covered".
#: That is precisely the incident described in this module's own docstring,
#: repeated in a different package while a gate written to prevent it sat green.
#:
#: A checker's scope is not the whole. The gate was aimed at the package where
#: the defect was FOUND rather than at the property it was meant to hold, so it
#: could only ever catch a second xai adapter.
#: Packages whose whole public surface must be accounted for.
#:
#: LF-54, 2026-09-10. This file checked the XAI package and NOTHING ELSE, and the
#: `agents` package had ZERO registry entries: six shipped, dispatched
#: capabilities that the platform's coverage matrix derived as "Not Covered".
#: That is the incident in this module's docstring, repeated in a different
#: package while a gate written to prevent it sat green.
#:
#: A checker's scope is not the whole. The gate was aimed at the package where
#: the defect was FOUND rather than at the property it was meant to hold.
ACCOUNTED_PACKAGES = ("llm", "agents", "multi_agent")


#: Every package that exists, DISCOVERED from disk rather than listed here.
#:
#: 2026-09-10, later the same day, and this is the second lesson on top of the
#: first. `multi_agent` had 14 public symbols and all 14 unaccounted: a SECOND
#: whole package invisible to the coverage matrix, found by the other lane. The
#: gate missed it because ACCOUNTED_PACKAGES was hardcoded. Worse, the table
#: below used to be hand-written and named ELEVEN of the fifteen packages on
#: disk, so the gate's own declaration of its limits had the same hole as the
#: gate, and the number it published (246) was wrong by 21.
#:
#: A hand-enumerated set cannot report what its author did not think of, and it
#: fails in the direction of looking complete. So the package list is discovered
#: and the DECLARATION below must match what is discovered: a new package is a
#: red test, not a silent addition to a bucket nobody reads.
def discovered_packages() -> tuple:
    """Every importable subpackage of vfairness, from the filesystem."""
    root = Path(__import__("vfairness").__file__).parent
    return tuple(
        sorted(
            entry.name
            for entry in root.iterdir()
            if entry.is_dir() and (entry / "__init__.py").is_file()
        )
    )


#: The packages this gate KNOWS ABOUT but does not yet account for, with the
#: measured count of public symbols in neither ledger. Measured 2026-09-10 by
#: discovery, not by hand:
#:
#:     evaluation 52, operations 50, rendering 49, post_processing 30,
#:     in_processing 27, preprocessing 20, xai 9, validity 8, net 5,
#:     legal 2, vision 1, mcp 0
#:
#: Many are result types and enums the registry's convention already excludes,
#: but that exclusion is not ENCODED for these packages, so the total is an upper
#: bound on the work and not a count of gaps.
#:
#: Widening now would ship a gate red on its first run, which this repo has
#: already decided against once (the scope note in
#: .github/workflows/quality.yml): clean first, then widen, in that order.
KNOWN_UNACCOUNTED_PACKAGES = (
    "evaluation",
    "in_processing",
    "legal",
    "mcp",
    "net",
    "operations",
    "post_processing",
    "preprocessing",
    "rendering",
    "validity",
    "vision",
    "xai",
)

UNACCOUNTED_OUTSIDE_THE_GATE = 253


def test_every_implemented_xai_adapter_is_registered_or_deferred():
    """Each adapter the xai router can actually instantiate must be in the
    capability ledger. A new adapter added to xai.explainers.registry._REGISTRY
    without a CAPABILITY_REGISTRY entry (or an explicit deferral with a
    reason) fails here, which is the point."""
    missing = []
    for method, adapter_cls in _REGISTRY.items():
        name = adapter_cls.__name__
        # route_explainer is registered for the router itself; adapters by class name.
        if name in REGISTERED_NAMES:
            continue
        if method in DEFERRED_FROM_CAPABILITY_REGISTRY:
            continue
        missing.append(f"{method} ({name})")
    assert not missing, (
        "Implemented XAI adapters absent from BOTH CAPABILITY_REGISTRY and "
        f"DEFERRED_FROM_CAPABILITY_REGISTRY: {missing}. Register them, or add a "
        "deferral entry with the gating reason."
    )


def test_every_unimplemented_xai_method_is_in_the_deferral_ledger():
    """The xai package's own not-yet-implemented list must be mirrored in the
    deferral ledger, so the backlog is visible where the platform looks."""
    missing = [m for m in _NOT_YET_IMPLEMENTED if m not in DEFERRED_FROM_CAPABILITY_REGISTRY]
    assert not missing, (
        f"xai.explainers.registry._NOT_YET_IMPLEMENTED entries missing from "
        f"DEFERRED_FROM_CAPABILITY_REGISTRY: {missing}"
    )


def test_deferrals_carry_a_reason():
    empty = [k for k, v in DEFERRED_FROM_CAPABILITY_REGISTRY.items() if not v.strip()]
    assert not empty, f"Deferral entries without a reason: {empty}"


def test_nothing_is_both_registered_and_deferred():
    """An entry may live in exactly one ledger; both at once is a contradiction."""
    deferred_names = set(DEFERRED_FROM_CAPABILITY_REGISTRY)
    both = REGISTERED_NAMES & deferred_names
    both |= set(CAPABILITY_REGISTRY) & deferred_names
    assert not both, f"Present in both CAPABILITY_REGISTRY and the deferral ledger: {both}"


def _public_symbols(package: str):
    """Every class and function reachable as ``vfairness.<package>.<name>``."""
    module = importlib.import_module(f"vfairness.{package}")
    names = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]
    out = {}
    for name in names:
        obj = getattr(module, name, None)
        if obj is None or inspect.ismodule(obj):
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            out[name] = obj
    return out


@pytest.mark.parametrize("package", ACCOUNTED_PACKAGES)
def test_every_public_capability_is_registered_or_deferred(package):
    """The INVERSE direction for llm/ and agents/, which nothing checked.

    Every class or function a user can reach as ``vfairness.<package>.<name>``
    must be in CAPABILITY_REGISTRY or in the deferral ledger. Absent from both
    is the failure this file exists for: the platform reads the manifest to
    derive coverage, so an unregistered capability is reported as Not Covered
    however well it works.
    """
    missing = [
        name
        for name in _public_symbols(package)
        if name not in REGISTERED_NAMES
        and name not in DEFERRED_NAMES
        and name not in DEFERRED_FROM_CAPABILITY_REGISTRY
    ]
    assert not missing, (
        f"vfairness.{package} symbols absent from BOTH CAPABILITY_REGISTRY and the "
        f"deferral ledger: {missing}. Register them, or add a deferral entry whose "
        f"reason says NOT A GAP or GATED, plus the symbol in DEFERRAL_COVERS."
    )


def test_the_accounted_packages_are_not_empty():
    """NON-VACUITY. An import that silently yielded nothing would make the test
    above pass for every package forever, which is how a gate stops being one."""
    for package in ACCOUNTED_PACKAGES:
        symbols = _public_symbols(package)
        assert len(symbols) >= 10, f"vfairness.{package} exposed only {len(symbols)} symbols"


@pytest.mark.parametrize("package", ACCOUNTED_PACKAGES)
def test_each_accounted_package_has_at_least_one_registered_capability(package):
    """The `agents` failure was not a missing entry, it was a missing PACKAGE:
    zero of six. A package that is entirely deferred is a claim worth making
    explicitly rather than arriving at by attrition."""
    registered_here = [
        key
        for key, entry in CAPABILITY_REGISTRY.items()
        if entry["module_path"].split(".")[0] == package
    ]
    assert registered_here, (
        f"vfairness.{package} has no registered capability at all, so the platform's "
        f"coverage matrix reports the entire package as Not Covered."
    )


def test_every_deferral_cover_maps_to_a_real_deferral():
    """DEFERRAL_COVERS says which symbols a grouped deferral accounts for. A key
    with no matching ledger entry would silently excuse those symbols from the
    check above while nothing recorded the decision."""
    orphans = [k for k in DEFERRAL_COVERS if k not in DEFERRED_FROM_CAPABILITY_REGISTRY]
    assert not orphans, f"DEFERRAL_COVERS keys with no deferral entry: {orphans}"


def test_every_deferral_reason_says_whether_it_is_a_gap():
    """The reason is read on the platform's coverage matrix, where a deferral
    without an actionable reason is indistinguishable from a gap. Each must say
    which it is: NOT A GAP (covered elsewhere, or not an assessment) or GATED
    (a real gap, with the condition that closes it).

    Only the LF-54 entries are held to the wording; the older ones carry their
    reasoning in prose and are checked for substance instead.
    """
    # 2026-09-10. This checked the PREFIX only for entries in DEFERRAL_COVERS,
    # i.e. the ones added the same hour, and merely a length for the older six.
    # I then told the other lane "every reason begins NOT A GAP or GATED", which
    # was true of eight of fourteen. The gate's scope was narrower than the claim
    # I made from it, which is the same error as the xai-only completeness check
    # this file exists to widen, committed in the widening itself.
    #
    # Three of the six undeclared ones said plainly that an adapter is NOT
    # IMPLEMENTED. Those are GAPS sitting in a ledger whose name says they are
    # deliberate, so a matrix rendering the ledger as "decisions" would have
    # presented three unbuilt capabilities as choices somebody made. They are
    # declared GATED now, with what unblocks each.
    #
    # Every entry, no exemptions.
    vague = [
        key
        for key, reason in DEFERRED_FROM_CAPABILITY_REGISTRY.items()
        if not reason.startswith(("NOT A GAP", "GATED"))
    ]
    assert not vague, (
        f"Deferral reasons a platform reader cannot act on: {vague}. Every reason must "
        f"BEGIN with NOT A GAP (and say where coverage actually is) or GATED (and say "
        f"what unblocks it). The prefix is load-bearing: the platform matches on it, "
        f"and inferring intent from a phrase mid-sentence turns a reader's guess into "
        f"the platform's verdict."
    )


def test_a_gated_reason_says_what_unblocks_it():
    """GATED means a real gap. A gap with no stated condition is a gap nobody
    can close, and on the coverage matrix it is indistinguishable from one that
    has simply been forgotten."""
    missing = [
        key
        for key, reason in DEFERRED_FROM_CAPABILITY_REGISTRY.items()
        if reason.startswith("GATED") and "nblocked by" not in reason
    ]
    assert not missing, (
        f"GATED deferrals with no unblocking condition: {missing}. Say what closes it."
    )


def test_decompose_mediation_is_registered_not_deferred():
    """It sat in the deferral ledger reading "implemented and used by the causal
    task lane; registration decision pending". Implemented and used is the
    condition for REGISTERING; "we have not decided" is neither gated nor
    deliberately out, and while it sat there the coverage matrix reported a
    shipped, dispatched capability as absent.

    Pinned by name because `operations` is outside ACCOUNTED_PACKAGES, so the
    package gate above would not notice it going back.
    """
    assert "decompose_mediation" in CAPABILITY_REGISTRY
    assert "operations.causal.mediate.decompose_mediation" not in DEFERRED_FROM_CAPABILITY_REGISTRY
    entry = CAPABILITY_REGISTRY["decompose_mediation"]
    assert entry["module_path"] == "operations.causal"
    assert entry["kind"] == "function"


def test_the_gates_own_scope_is_measured_and_not_growing_silently():
    """The gate covers two packages. This asserts the size of what it does NOT
    cover, so the day somebody widens it the number moves deliberately, and the
    day somebody adds a package's worth of unaccounted capability the number
    moves loudly.

    Not a threshold anyone should tune: it is a tripwire on a comment that would
    otherwise rot, the same reason the numpy-variance assumption is pinned rather
    than described.
    """
    import importlib
    import inspect

    registered = {entry["name"] for entry in CAPABILITY_REGISTRY.values()}
    covered = {name for names in DEFERRAL_COVERS.values() for name in names}
    total = 0
    for package in KNOWN_UNACCOUNTED_PACKAGES:
        module = importlib.import_module(f"vfairness.{package}")
        names = getattr(module, "__all__", None) or [
            n for n in dir(module) if not n.startswith("_")
        ]
        for name in names:
            obj = getattr(module, name, None)
            if obj is None or inspect.ismodule(obj):
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            if name in registered or name in covered or name in DEFERRED_FROM_CAPABILITY_REGISTRY:
                continue
            total += 1

    assert total <= UNACCOUNTED_OUTSIDE_THE_GATE, (
        f"{total} symbols outside the gated packages are in neither ledger, up from "
        f"the {UNACCOUNTED_OUTSIDE_THE_GATE} measured on 2026-09-10. Something added "
        f"capability without accounting for it. Register it, defer it, or raise this "
        f"number deliberately and say why."
    )


def test_every_package_on_disk_is_either_gated_or_declared_unaccounted():
    """THE FIX FOR THE HOLE THAT PRODUCED THIS FILE'S SECOND LESSON.

    `multi_agent` existed on disk with 14 unaccounted capabilities and appeared
    in NEITHER the gated list nor the hand-written table of what the gate misses.
    Both lists were written by hand, so both had the same blind spot, and the
    limit number published from them was wrong by 21.

    Packages are DISCOVERED now, and every discovered package must be explicitly
    either accounted or declared-unaccounted. Adding a new package to the library
    turns this red until somebody decides which it is, which is the only way a
    list of exclusions can be trusted.
    """
    discovered = set(discovered_packages())
    declared = set(ACCOUNTED_PACKAGES) | set(KNOWN_UNACCOUNTED_PACKAGES)

    undeclared = sorted(discovered - declared)
    assert not undeclared, (
        f"packages on disk that are neither gated nor declared unaccounted: "
        f"{undeclared}. Add each to ACCOUNTED_PACKAGES (and account for its "
        f"symbols) or to KNOWN_UNACCOUNTED_PACKAGES (and raise "
        f"UNACCOUNTED_OUTSIDE_THE_GATE deliberately). This is the check that "
        f"`multi_agent` needed and did not have."
    )

    phantom = sorted(declared - discovered)
    assert not phantom, (
        f"declared packages that do not exist on disk: {phantom}. A stale name "
        f"here makes the declaration look more complete than it is."
    )


def test_discovery_actually_discovers_something():
    """NON-VACUITY, and the specific one this design needs.

    A discovery-based gate that discovers nothing is green and silent, which is
    the exact failure mode it replaces. If `discovered_packages()` ever returns
    an empty or tiny tuple, every assertion built on it passes for free.
    """
    discovered = discovered_packages()
    assert len(discovered) >= 12, f"discovery returned only {len(discovered)}: {discovered}"
    for expected in ("llm", "agents", "multi_agent", "evaluation", "operations"):
        assert expected in discovered, f"{expected} not discovered: {discovered}"
