"""A grade's evidence must be openable: the test file it names has to exist.

WHY. A grade of PROVEN is a claim with one piece of supporting evidence, the test
that pins the behaviour, and that evidence is published as a path. When the path
is wrong the grade still reads as PROVEN, the published totals still count it, and
a reader who goes looking finds nothing. Nothing in the suite noticed.

Measured on 2026-09-27: 42 of the 342 rows the BGL3 wave produced cited
``tests/test_bgl3_evaluation-1.py``, ``tests/test_bgl3_preprocessing-1.py`` or
``tests/test_bgl3_xai-1.py``. Those three files had been RENAMED to underscores
hours earlier, because a hyphen is not legal in a Python module name and ruff
refused them. The tests themselves were fine and were running the whole time. The
citations pointed at nothing, and 42 grades rested on a reference a reader could
not follow.

This is the stale-cross-reference shape, and it is worth a mechanical gate rather
than an editorial one for the same reason the rest of this suite exists: the
citation is written by whoever produced the grade, in prose, at the end of a long
batch, and it is the one field no other check reads.

Scope, stated rather than assumed: this pins that the FILE exists. It does not
pin that the named test reaches the unit, which is coverage work and is what the
independent audit pass does per row.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import unit_resolution  # noqa: E402

WAVES = sorted((ROOT / "docs").glob("surface-grading-*.json"))

# A path-shaped reference inside the free-text evidence field. Underscores and
# hyphens are both matched ON PURPOSE: a hyphenated name is exactly the broken
# case, so the pattern has to be able to see it in order to report it.
_REF = re.compile(r"tests/[A-Za-z0-9_\-]+\.py")


def _references():
    for wave in WAVES:
        items = json.loads(wave.read_text(encoding="utf-8")).get("items") or {}
        for unit, rec in items.items():
            for ref in _REF.findall(rec.get("test_file") or ""):
                yield wave.name, unit, ref


def test_the_waves_are_discovered_at_all():
    """Guards the guard. An empty sweep would pass every assertion below."""
    assert WAVES, "no surface-grading wave files found; the sweep below proves nothing"
    refs = list(_references())
    assert len(refs) > 500, f"only {len(refs)} test-file references found across {len(WAVES)} waves"


@pytest.mark.parametrize("wave", WAVES, ids=lambda p: p.name)
def test_every_test_file_a_grade_cites_exists(wave):
    missing = sorted(
        {(unit, ref) for name, unit, ref in _references() if name == wave.name}
        - {(unit, ref) for name, unit, ref in _references() if (ROOT / ref).exists()}
    )
    assert not missing, (
        f"{len(missing)} grade(s) in {wave.name} cite a test file that does not exist:\n"
        + "\n".join(f"  {unit}\n    -> {ref}" for unit, ref in missing[:12])
    )


def test_a_grade_that_claims_proven_cites_something():
    """PROVEN without a pin is not a grade, whatever the row says."""
    naked = []
    for wave in WAVES:
        items = json.loads(wave.read_text(encoding="utf-8")).get("items") or {}
        for unit, rec in items.items():
            if rec.get("grade") == "PROVEN" and not (rec.get("test_file") or "").strip():
                naked.append(f"{wave.name}: {unit}")
    assert not naked, "PROVEN with no test named:\n" + "\n".join(naked[:12])


# ─────────────────────────────────────────────────────────────────────────────
# The unit a grade is ABOUT must exist too, not only the test that pins it.
# ─────────────────────────────────────────────────────────────────────────────
#
# Measured 2026-09-27, by running this sweep for the first time: one of the 873
# graded units did not exist. ``FairnessMonitor.get_metric_summary`` was graded
# NOT A MEASUREMENT, overturned to PROVEN on audit, and counted in the published
# PROVEN total for nine days. The auditor's own note said the callable "does not
# exist" and recorded the grade regardless. A second row was addressed as
# ``vfairness.operations.pulse.build_art10`` when the function lives in the
# ``regulatory`` submodule, so the citation resolved to nothing.
#
# THIS CHECK HAS THREE OUTCOMES, not two, and the third is the reason it is
# trustworthy. A name this method cannot walk is reported as unresolved-by-this-
# method and is NOT counted as absent. Written with two outcomes, a first draft
# accused 14 correct rows: it never tried importing the top-level package, so
# every ``vfairness.plot_*`` re-export read as missing. A gate that over-accuses
# gets switched off, and then it protects nothing.


# MOVED 2026-09-30 to scripts/unit_resolution.py, unchanged, so that the applier
# that WRITES grading rows can ask the same question this test asks about them.
# It used to live here only, so the fix-wave applier could not consult it and
# guarded the SHAPE of a unit key instead; two rows about nothing were published
# as a result. That applier is internal tooling and the export does not publish
# it, which is why it is described here rather than named by path: a pointer into
# a file the reader cannot open tells them less than no pointer at all.
# The comment block above still records why the resolver has three
# outcomes, because that is the part a reader needs before trusting it.
_resolve = unit_resolution.resolve


def test_every_graded_unit_exists():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    absent, unresolved, seen = [], [], set()
    for wave in WAVES:
        items = json.loads(wave.read_text(encoding="utf-8")).get("items") or {}
        for unit in items:
            if unit in seen:
                continue
            seen.add(unit)
            ok, how = _resolve(unit)
            if ok:
                continue
            (absent if how.startswith("absent") else unresolved).append(f"{unit}\n    {how}")

    assert seen, "no graded units found; this sweep proves nothing"
    # The could-not-check state is reported, never folded into either verdict.
    if unresolved:
        print(f"\n{len(unresolved)} unit(s) this method could not resolve either way:")
        for u in unresolved[:10]:
            print("  " + u)
    assert not absent, (
        f"{len(absent)} graded unit(s) do not exist, so the grade is about nothing "
        f"and is still counted in the published totals:\n" + "\n  ".join(absent[:12])
    )


# ─────────────────────────────────────────────────────────────────────────────
# The resolver's own three outcomes, asked directly.
# ─────────────────────────────────────────────────────────────────────────────
#
# The sweep above passes today whichever way two of these branches are written,
# because no graded unit currently exercises them, and a branch nobody reaches is
# indistinguishable from one that works. Sabotaging the declared-field branch left
# the sweep GREEN, which is how that was found. So each outcome is asked for here
# against a fixture instead of being inferred from the corpus.


def test_the_resolver_finds_an_ordinary_attribute():
    ok, how = _resolve("vfairness.operations.experimentation.power.SamplingPlan.to_dict")
    assert (ok, how) == (True, "resolved")


def test_the_resolver_separates_an_unimportable_module_from_an_absent_one(tmp_path, monkeypatch):
    """The fourth outcome, asked directly and in any environment.

    This branch exists because the sweep reported vfairness.mcp.server.main as a
    graded unit that does not exist. It exists: the module raises ImportError on
    purpose when the optional mcp extra is absent, and an over-accusation would
    have had someone delete a valid grade.

    The FIRST version of this test used that module as its fixture, and that was
    the same mistake one level up: mcp is installed in .venv and absent from the
    other interpreter on this machine, so the test passed under one python and
    failed under the other. A test for "a module that cannot be imported" must
    BUILD one, not hope the environment is missing something.
    """
    pkg = tmp_path / "vfx_resolver_probe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "boom.py").write_text(
        "raise ImportError('needs an optional extra that is not installed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    ok, how = _resolve("vfx_resolver_probe.boom.something")
    assert ok is False
    assert not how.startswith("absent"), (
        f"a module that exists on disk is being reported as absent: {how}"
    )
    assert how.startswith("unimportable:"), how

    # BOTH DIRECTIONS IN ONE TEST, because a branch that answers
    # "could-not-check" to everything would pass the assertion above and destroy
    # the sweep's ability to find a grade about nothing.
    ok2, how2 = _resolve("vfx_resolver_probe.no_such_module.something")
    assert ok2 is False
    assert how2.startswith("absent"), (
        f"a module that is genuinely not there must be absent, not excused: {how2}"
    )


def test_the_resolver_reports_a_genuinely_absent_attribute_as_absent():
    ok, how = _resolve("vfairness.operations.experimentation.power.SamplingPlan.no_such_method")
    assert not ok
    assert how.startswith("absent:"), how
    assert "no_such_method" in how


def test_the_resolver_reports_an_unimportable_root_as_unresolved_not_absent():
    """The could-not-check state. Reporting this as absent is how a gate starts
    accusing correct rows, which is what a first draft of this file did to 14 of
    them."""
    ok, how = _resolve("not_a_real_package_at_all.thing")
    assert not ok
    assert how.startswith("unresolved:"), how


def test_a_dataclass_field_with_no_default_still_resolves():
    """The branch the sweep cannot reach today.

    ``hasattr`` on the CLASS is False for a dataclass field declared without a
    default, because no class attribute is created. Every graded field in the
    corpus happens to carry a default, so this branch is unexercised by the sweep
    and was proven dead-looking by sabotage. It is not dead: it is the difference
    between resolving such a field and accusing it of not existing.
    """
    import dataclasses
    import sys
    import types

    mod = types.ModuleType("_resolver_fixture_pkg")

    @dataclasses.dataclass
    class Plan:
        with_default: int = 0
        without_default: int = dataclasses.field(default_factory=int)
        bare: "int" = None  # noqa: RUF013

    @dataclasses.dataclass
    class Bare:
        needed: int

    mod.Plan, mod.Bare = Plan, Bare
    sys.modules["_resolver_fixture_pkg"] = mod
    try:
        assert not hasattr(Bare, "needed"), "fixture is wrong: hasattr already finds it"
        assert "needed" in Bare.__dataclass_fields__
        ok, how = _resolve("_resolver_fixture_pkg.Bare.needed")
        assert (ok, how) == (True, "resolved"), how
        # And the control: a name that is NOT a declared field is still absent.
        ok, how = _resolve("_resolver_fixture_pkg.Bare.not_declared")
        assert not ok and how.startswith("absent:"), how
    finally:
        del sys.modules["_resolver_fixture_pkg"]
