"""Auto-loaded pytest plugin for vfairness (top-level, deliberately outside the package).

CRITICAL, DO NOT MOVE THIS MODULE INTO THE ``vfairness`` PACKAGE.
pytest loads every ``pytest11`` entry point unconditionally at startup. While the
entry point pointed at ``vfairness.operations.cicd.pytest_plugin``, Python had to
execute ``vfairness/__init__.py`` (and two more package ``__init__`` modules) just
to reach it, which pulled numpy, pandas, scipy, scikit-learn and ~1500 further
modules into EVERY pytest session in ANY project that merely has vfairness in its
dependency tree. Measured on a one-assert throwaway project on 2026-08-27:

    pytest -q                  ->  3.24 / 3.21 / 3.33 s  (plugin auto-loaded)
    pytest -q -p no:vfairness  ->  0.36 / 0.34 / 0.36 s

This module sits at the top level of the wheel, so importing it touches no
vfairness package ``__init__`` at all. Keep its module scope free of vfairness
imports: the hooks below need nothing but pytest, and the two fixtures import
vfairness inside their bodies, which runs only when a test actually requests them.

``vfairness.operations.cicd.pytest_plugin`` re-exports these names, so the old
import path and ``-p vfairness.operations.cicd.pytest_plugin`` keep working.

One editable-install wrinkle: because this module is mapped into the wheel with
``force-include`` rather than being part of the package directory, an editable
install materialises a COPY of it in site-packages instead of pointing at this
file. After editing this module, reinstall (``pip install -e .``) before expecting
the change to take effect in that environment.

Markers:
    @pytest.mark.fairness - tag a test as a fairness test
    @pytest.mark.fairness_gate - tag a test as a gate-level check

Fixtures:
    fairness_gate - provides a pre-configured ModelFairnessGate
    assert_fairness_fn - provides the assert_fairness helper

Usage::

    @pytest.mark.fairness
    def test_demographic_parity(assert_fairness_fn):
        assert_fairness_fn(
            y_true, y_pred, gender,
            metrics=['demographic_parity_difference'],
            thresholds={'demographic_parity_difference': 0.1},
        )
"""

import pytest

# Plugin hooks


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "fairness: mark a test as a fairness test (deselect with '-m \"not fairness\"')"
    )
    config.addinivalue_line("markers", "fairness_gate: mark a test as a fairness gate check")


def pytest_collection_modifyitems(config, items):
    """Collect fairness test metadata for summary reporting."""
    fairness_items = []
    for item in items:
        if item.get_closest_marker("fairness") or item.get_closest_marker("fairness_gate"):
            fairness_items.append(item)

    # Store on config for terminal summary
    config._fairness_items = fairness_items


# The buckets pytest files a report under, in the order a fairness verdict must be
# decided. Two orderings here are load-bearing, DO NOT REORDER CASUALLY:
#   * "failed" first, because a test that failed its call has a real negative
#     verdict and a later teardown error must not downgrade it to "could not run".
#   * "error" ahead of "passed", because ONE ITEM CAN BE IN BOTH. pytest files a
#     teardown error under stats["error"] while the call-phase pass stays in
#     stats["passed"], so looking at "passed" first (as this hook did until
#     2026-08-28) counts a test that blew up in teardown as a clean pass.
_VERDICT_BUCKETS = ("failed", "error", "xpassed", "xfailed", "skipped", "passed")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a fairness test summary at the end of the test run.

    Three states, never two: a fairness test either passed, produced a negative
    verdict, or produced NO VERDICT AT ALL. The all-clear is emitted only when
    every collected fairness test actually passed.

    CRITICAL, NEVER COUNT ONLY passed/failed/skipped, AND NEVER REBUILD THE TOTAL
    BY ADDING THE PARTS. Until 2026-08-28 this hook looked each fairness item up
    in exactly those three buckets and silently dropped it when it was in none of
    them. pytest files a setup/teardown ERROR under stats["error"] and expected
    failures under stats["xfailed"] / stats["xpassed"], so a fairness test that
    NEVER RAN was in no counted bucket: it vanished from a total that was computed
    as passed + failed + skipped, and the `elif passed > 0` branch then printed the
    unqualified "All fairness requirements met." Reproduced with one passing
    fairness test plus one whose fixture raised: pytest itself reported
    "1 passed, 1 error" while this hook reported "1 fairness test(s): 1 passed,
    0 failed, 0 skipped / All fairness requirements met."; when the only fairness
    test errored, the hook reported there had been none at all.

    That is not merely an internal bug. This module is the pytest11 entry point,
    which pytest AUTO-LOADS in every session of any project that merely has
    vfairness installed, so the false all-clear was emitted into downstream users'
    CI logs. The total below is therefore taken from len(fairness_items) and the
    unexplained remainder is derived by SUBTRACTION, which makes a dropped item
    arithmetically impossible rather than merely unlikely.
    """
    fairness_items = getattr(config, "_fairness_items", [])
    if not fairness_items:
        return

    terminalreporter.section("Fairness Test Summary")

    stats = terminalreporter.stats
    reported = {
        bucket: {getattr(r, "nodeid", None) for r in stats.get(bucket, [])}
        for bucket in _VERDICT_BUCKETS
    }

    counts = dict.fromkeys(_VERDICT_BUCKETS, 0)
    for item in fairness_items:
        for bucket in _VERDICT_BUCKETS:
            if item.nodeid in reported[bucket]:
                counts[bucket] += 1
                break

    # The total is what was COLLECTED, not what was found in the stats buckets.
    total = len(fairness_items)
    passed = counts["passed"]
    failed = counts["failed"]
    skipped = counts["skipped"]
    # Whatever no bucket explained is an item that produced no verdict. Derived by
    # subtracting from the collected total so the printed parts always sum to it.
    unreported = total - sum(counts.values())

    # A module that fails to COLLECT never reaches pytest_collection_modifyitems,
    # so a fairness test inside it is invisible here: it can only be reported as
    # unknown, never cleared. By default pytest aborts the session on a collection
    # error (nothing is collected, so this hook returns above), but
    # --continue-on-collection-errors keeps the run going, and the fairness tests
    # that did collect must not clear a module that never loaded.
    collection_errors = sum(
        1 for r in stats.get("error", []) if getattr(r, "when", None) == "collect"
    )

    # Everything that is neither a pass nor a fail: it ran to no verdict, or not at
    # all. Equivalent to counts[error/xpassed/xfailed/skipped] + unreported, but
    # taken from the total so an unhandled bucket can never leak out of the sum.
    no_verdict = total - passed - failed

    parts = [f"{passed} passed", f"{failed} failed", f"{skipped} skipped"]
    for bucket, label in (("error", "errored"), ("xfailed", "xfailed"), ("xpassed", "xpassed")):
        if counts[bucket]:
            parts.append(f"{counts[bucket]} {label}")
    if unreported:
        parts.append(f"{unreported} unreported")
    terminalreporter.write_line(f"  {total} fairness test(s): " + ", ".join(parts))

    if failed > 0:
        terminalreporter.write_line(
            "  WARNING: Fairness requirements not met - review failed tests above."
        )
    if no_verdict > 0:
        terminalreporter.write_line(
            f"  {no_verdict} fairness test(s) COULD NOT RUN - no fairness verdict."
        )
    if collection_errors > 0:
        terminalreporter.write_line(
            f"  {collection_errors} module(s) failed to collect - no fairness verdict "
            "for any fairness test they contain."
        )
    # The all-clear requires every collected fairness test to have passed. Each
    # clause is stated separately on purpose: any one of them going false must be
    # enough to withhold it.
    if failed == 0 and no_verdict == 0 and collection_errors == 0 and passed == total:
        terminalreporter.write_line("  All fairness requirements met.")


# Fixtures


@pytest.fixture
def fairness_gate():
    """Provide a pre-configured ModelFairnessGate for testing.

    Returns a factory function so tests can customize the gate::

        def test_gate(fairness_gate):
            gate = fairness_gate(
                metrics=['demographic_parity_difference'],
                thresholds={'demographic_parity_difference': 0.1},
            )
            decision = gate.evaluate(y_true, y_pred, protected_attr)
            assert decision.approved
    """
    # Imported here, not at module scope: see the module docstring. A test that
    # never asks for this fixture must not pay for the vfairness import.
    from vfairness.operations.cicd.gate import ModelFairnessGate

    def _factory(**kwargs):
        return ModelFairnessGate(**kwargs)

    return _factory


@pytest.fixture
def assert_fairness_fn():
    """Provide the vfairness assert_fairness function as a fixture.

    Example::

        def test_model_fairness(assert_fairness_fn):
            assert_fairness_fn(
                y_true, y_pred, gender,
                metrics=['demographic_parity_difference'],
                thresholds={'demographic_parity_difference': 0.1},
            )
    """
    # Imported here, not at module scope: see the module docstring.
    from vfairness.evaluation.vfairness_metrics.integrations import assert_fairness

    return assert_fairness
