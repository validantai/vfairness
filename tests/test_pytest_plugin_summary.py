"""The fairness terminal summary must never clear a test that did not run.

WHY THIS FILE EXISTS. `_vfairness_pytest_plugin.pytest_terminal_summary` counted a
fairness item only if its nodeid turned up in terminalreporter.stats under
"passed", "failed" or "skipped". pytest files a setup/teardown ERROR under
stats["error"] and expected failures under stats["xfailed"] / stats["xpassed"], so
a fairness test that NEVER RAN was in none of the counted buckets: it dropped out
of a total computed as passed + failed + skipped, and the `elif passed > 0` branch
printed the unqualified "All fairness requirements met." One passing fairness test
plus one whose fixture raised produced

    1 fairness test(s): 1 passed, 0 failed, 0 skipped
    All fairness requirements met.

while pytest itself reported "1 passed, 1 error". When the only fairness test
errored, the plugin reported there had been none at all.

That module is the pytest11 entry point, which pytest AUTO-LOADS in every session
of any project that merely has vfairness installed, so the false all-clear went
into downstream users' CI logs. Nothing tested it: the only reference to the
plugin anywhere in tests/ was a hasattr() re-export check in
test_packaging_hygiene.py.

Every test here drives a REAL inner pytest session through pytester's
`runpytest_subprocess`, because the defect lives in what a session actually prints
and cannot be observed by calling the hook with a hand-built reporter.
"""

import os
import re
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

SRC = Path(__file__).resolve().parents[1] / "src"
PLUGIN = "_vfairness_pytest_plugin"
PLUGIN_SOURCE = SRC / f"{PLUGIN}.py"

# The exact strings a downstream CI log is greped for. Changing either of these is
# a breaking change for users, so they are pinned as literals, not rebuilt.
ALL_CLEAR = "All fairness requirements met."
NO_VERDICT = "COULD NOT RUN - no fairness verdict"
FAIL_WARNING = "WARNING: Fairness requirements not met"

_COUNT_LINE = re.compile(r"^\s*(\d+) fairness test\(s\): (.+)$", re.MULTILINE)


@pytest.fixture(autouse=True)
def _plugin_importable(monkeypatch):
    """Point the inner subprocess at THIS repo's plugin source.

    The module docstring of the plugin records an editable-install wrinkle: because
    the module is mapped into the wheel with `force-include` rather than living in
    the package directory, some installs materialise a COPY in site-packages. A
    test that silently ran against that copy would be testing the old code, so src
    goes FIRST on the path and `test_inner_session_loads_the_repo_source` proves
    which file was actually loaded.
    """
    prior = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(SRC), prior]) if prior else str(SRC))


def _run(pytester, *extra_args, **sources):
    """Run one real inner pytest session with the plugin under test loaded.

    `-p no:vfairness` blocks the auto-loaded entry point so the hook cannot be
    registered twice (which would print the section twice and mask a miscount),
    and `-p _vfairness_pytest_plugin` loads the module from src explicitly.
    """
    pytester.makepyfile(**sources)
    return pytester.runpytest_subprocess(
        "-p", "no:vfairness", "-p", PLUGIN, "-q", "--tb=no", *extra_args
    )


def _count_line(result):
    """Return (total, {label: count}) parsed from the summary's count line."""
    match = _COUNT_LINE.search("\n".join(result.outlines))
    assert match is not None, "the fairness summary printed no count line at all:\n" + "\n".join(
        result.outlines
    )
    total = int(match.group(1))
    parts = {}
    for chunk in match.group(2).split(", "):
        number, label = chunk.split(" ", 1)
        parts[label] = int(number)
    return total, parts


# ---------------------------------------------------------------------------
# The healthy case: this must not change. It is the over-correction control.
# ---------------------------------------------------------------------------


def test_all_pass_prints_the_all_clear(pytester):
    """Fully-measured, all-passing input keeps the exact pre-fix output."""
    result = _run(
        pytester,
        test_healthy="""
        import pytest

        @pytest.mark.fairness
        def test_a(): assert True

        @pytest.mark.fairness_gate
        def test_b(): assert True

        @pytest.mark.fairness
        def test_c(): assert True
        """,
    )
    result.assert_outcomes(passed=3)
    # Byte-for-byte the line the plugin printed before the fix, so a downstream
    # grep of a healthy run sees no change whatsoever.
    result.stdout.fnmatch_lines(
        [
            "  3 fairness test(s): 3 passed, 0 failed, 0 skipped",
            f"  {ALL_CLEAR}",
        ]
    )


def test_no_fairness_tests_prints_no_section_at_all(pytester):
    """The plugin is auto-loaded everywhere; an unrelated project must see nothing."""
    result = _run(pytester, test_unrelated="def test_x(): assert True")
    result.assert_outcomes(passed=1)
    assert "Fairness Test Summary" not in "\n".join(result.outlines)


def test_failure_still_prints_the_original_warning(pytester):
    """A real negative verdict keeps its original wording and withholds the all-clear."""
    result = _run(
        pytester,
        test_failing="""
        import pytest

        @pytest.mark.fairness
        def test_ok(): assert True

        @pytest.mark.fairness
        def test_biased(): assert False
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        [
            "  2 fairness test(s): 1 passed, 1 failed, 0 skipped",
            f"  {FAIL_WARNING}*",
        ]
    )
    assert ALL_CLEAR not in "\n".join(result.outlines)


# ---------------------------------------------------------------------------
# The defect: tests that never produced a verdict must never be cleared.
# ---------------------------------------------------------------------------


def test_errored_fairness_test_withholds_the_all_clear(pytester):
    """THE REPORTED REPRO. One pass + one erroring fixture printed a false all-clear."""
    result = _run(
        pytester,
        test_errors="""
        import pytest

        @pytest.fixture
        def broken():
            raise RuntimeError("fixture blew up")

        @pytest.mark.fairness
        def test_ok(): assert True

        @pytest.mark.fairness
        def test_never_ran(broken): assert True
        """,
    )
    result.assert_outcomes(passed=1, errors=1)

    out = "\n".join(result.outlines)
    assert ALL_CLEAR not in out, (
        "the plugin cleared a run in which a fairness test never executed; pytest "
        "itself reported an error for it"
    )
    assert NO_VERDICT in out

    total, parts = _count_line(result)
    assert total == 2, f"the errored item was dropped from the total: {total} != 2"
    assert parts["passed"] == 1
    assert parts["errored"] == 1


def test_sole_fairness_test_erroring_is_not_reported_as_zero_tests(pytester):
    """When the ONLY fairness test errored, the plugin claimed there were none."""
    result = _run(
        pytester,
        test_only="""
        import pytest

        @pytest.fixture
        def broken():
            raise RuntimeError("boom")

        @pytest.mark.fairness
        def test_only(broken): assert True
        """,
    )
    result.assert_outcomes(errors=1)

    total, parts = _count_line(result)
    assert total == 1, f"a collected fairness test vanished from the total: {total} != 1"
    assert parts["errored"] == 1
    out = "\n".join(result.outlines)
    assert ALL_CLEAR not in out
    assert NO_VERDICT in out


def test_teardown_error_after_a_passing_call_is_not_a_clean_pass(pytester):
    """One item can be in stats["passed"] AND stats["error"] at the same time.

    pytest files the call-phase pass under "passed" and the teardown blow-up under
    "error", so a bucket order that reads "passed" first launders a test that did
    not complete into a clean pass. This is the case that makes the bucket order in
    `_VERDICT_BUCKETS` load-bearing rather than cosmetic.
    """
    result = _run(
        pytester,
        test_teardown="""
        import pytest

        @pytest.fixture
        def td():
            yield
            raise RuntimeError("teardown boom")

        @pytest.mark.fairness
        def test_passes_then_teardown_errors(td): assert True
        """,
    )
    result.assert_outcomes(passed=1, errors=1)

    total, parts = _count_line(result)
    assert total == 1
    assert parts["passed"] == 0, "a test whose teardown errored was counted as a pass"
    assert parts["errored"] == 1
    assert ALL_CLEAR not in "\n".join(result.outlines)


def test_xfailed_and_xpassed_are_counted_and_withhold_the_all_clear(pytester):
    """xfail/xpass live in their own stats buckets and were dropped entirely."""
    result = _run(
        pytester,
        test_xfail="""
        import pytest

        @pytest.mark.fairness
        @pytest.mark.xfail(reason="known bias, tracked")
        def test_known_bias(): assert False

        @pytest.mark.fairness
        @pytest.mark.xfail(reason="known bias, tracked")
        def test_unexpectedly_fixed(): assert True
        """,
    )
    result.assert_outcomes(xfailed=1, xpassed=1)

    total, parts = _count_line(result)
    assert total == 2, f"both xfail items were dropped from the total: {total} != 2"
    assert parts["xfailed"] == 1
    assert parts["xpassed"] == 1
    out = "\n".join(result.outlines)
    assert ALL_CLEAR not in out
    assert NO_VERDICT in out


def test_skipped_fairness_test_withholds_the_all_clear(pytester):
    """A skipped fairness test did not run, so there is no verdict to clear.

    Deliberate behaviour change: the old `elif passed > 0` printed the all-clear
    for a run of one pass plus one skip. A skip is could-not-check, not pass.
    """
    result = _run(
        pytester,
        test_skip="""
        import pytest

        @pytest.mark.fairness
        def test_ok(): assert True

        @pytest.mark.fairness
        @pytest.mark.skip(reason="dataset unavailable")
        def test_never_ran(): assert False
        """,
    )
    result.assert_outcomes(passed=1, skipped=1)

    total, parts = _count_line(result)
    assert total == 2
    assert parts["skipped"] == 1
    out = "\n".join(result.outlines)
    assert ALL_CLEAR not in out
    assert NO_VERDICT in out


def test_collection_error_withholds_the_all_clear(pytester):
    """A module that never collected may have held fairness tests we cannot see.

    pytest aborts the session on a collection error by default, but
    --continue-on-collection-errors keeps going, and the fairness tests that DID
    collect must not clear a module that never loaded.
    """
    result = _run(
        pytester,
        "--continue-on-collection-errors",
        test_good="""
        import pytest

        @pytest.mark.fairness
        def test_ok(): assert True
        """,
        test_broken="raise ImportError('this module never loaded')",
    )
    out = "\n".join(result.outlines)
    assert ALL_CLEAR not in out, (
        "the plugin cleared the run although a module failed to collect; any "
        "fairness test inside it never even reached collection"
    )
    assert "failed to collect" in out


# ---------------------------------------------------------------------------
# The invariant: the printed parts always account for every collected item.
# ---------------------------------------------------------------------------


def test_printed_parts_sum_to_the_collected_total(pytester):
    """A mixed run must account for every collected fairness item, in one line.

    This is the arithmetic form of the whole blocker: the total is taken from
    len(fairness_items), so an item that lands in an unhandled stats bucket has to
    surface as a printed count rather than silently leaving the sum.
    """
    result = _run(
        pytester,
        test_mixed="""
        import pytest

        @pytest.fixture
        def broken():
            raise RuntimeError("boom")

        @pytest.mark.fairness
        def test_pass_1(): assert True

        @pytest.mark.fairness
        def test_pass_2(): assert True

        @pytest.mark.fairness
        def test_fail(): assert False

        @pytest.mark.fairness
        def test_error(broken): assert True

        @pytest.mark.fairness
        @pytest.mark.skip(reason="no data")
        def test_skip(): assert True

        @pytest.mark.fairness
        @pytest.mark.xfail(reason="known")
        def test_xfail(): assert False

        @pytest.mark.fairness_gate
        @pytest.mark.xfail(reason="known")
        def test_xpass(): assert True
        """,
    )
    total, parts = _count_line(result)
    assert total == 7, f"7 fairness items were collected, the line says {total}"
    assert sum(parts.values()) == total, (
        f"the printed parts {parts} sum to {sum(parts.values())}, not the collected {total}; "
        "an item was dropped from the summary"
    )
    assert parts == {
        "passed": 2,
        "failed": 1,
        "skipped": 1,
        "errored": 1,
        "xfailed": 1,
        "xpassed": 1,
    }
    out = "\n".join(result.outlines)
    assert ALL_CLEAR not in out
    assert FAIL_WARNING in out
    assert NO_VERDICT in out


def test_inner_session_loads_the_repo_source(pytester):
    """Prove the assertions above ran against the file in this repo.

    Without this, an editable install that materialised a stale COPY of the
    top-level module in site-packages would let every test in this file pass while
    the shipped code still held the defect.
    """
    # Written to a file rather than printed: the report header is suppressed under
    # the -q every run here uses, and pytest's own capture swallows a bare print
    # from conftest when no test fails.
    probe = pytester.path / "plugin_file.txt"
    pytester.makeconftest(
        f"""
        import pathlib
        import {PLUGIN}

        pathlib.Path({str(probe)!r}).write_text({PLUGIN}.__file__)
        """
    )
    result = _run(pytester, test_x="def test_x(): assert True")
    result.assert_outcomes(passed=1)

    assert probe.is_file(), "the inner session never imported the plugin module"
    loaded = Path(probe.read_text()).resolve()
    assert loaded == PLUGIN_SOURCE.resolve(), (
        f"the inner pytest session loaded {loaded}, not the repo source "
        f"{PLUGIN_SOURCE}; every assertion in this file would have been made "
        "against a stale copy of the plugin"
    )
