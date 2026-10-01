"""READINESS-6: the last three files on the register's open list.

Three unrelated modules, one shape. A number, a verdict and an exit code, each
produced over inputs that were never measured, each indistinguishable from the
same thing produced over inputs that were.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import pathlib
import warnings

import numpy as np
import pytest

from vfairness.agents.pipeline_tracker import PipelineTracker
from vfairness.operations.cicd.precommit import main as precommit_main
from vfairness.validity.aggregate import aggregate_validity
from vfairness.validity.groundedness import GroundednessResult


def _result(available, value):
    required = {
        name: None
        for name, f in GroundednessResult.__dataclass_fields__.items()
        if f.default is inspect._empty
        and getattr(f, "default_factory", inspect._empty) is inspect._empty
    }
    required["available"] = available
    required["value"] = value
    return GroundednessResult(**required)


# --------------------------------------------------------------- validity coverage


def test_a_grade_over_one_percent_of_a_batch_says_so():
    """`n` and `n_measured` were both in the envelope, so a reader COULD divide
    them. Nothing said it, nothing warned, and `available` is the field a gate
    reads. Measured before the fix: one measured result out of a hundred
    returned available=True with VG-001 mean 0.95 against a 0.8 threshold, a
    clean pass computed from 1% of the batch."""
    results = [_result(True, 0.95)] + [_result(False, None) for _ in range(99)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = aggregate_validity(results)

    assert out["n"] == 100 and out["n_measured"] == 1
    assert out["coverage"] == pytest.approx(0.01)
    assert out["note"] and "Partial coverage" in out["note"]
    assert any("coverage" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_full_coverage_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL. A complete batch must not grow a caveat, or the
    caveat stops being read on the batches that need it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = aggregate_validity([_result(True, 0.95) for _ in range(100)])

    assert out["coverage"] == 1.0
    assert out["note"] is None
    assert not [w for w in caught if "coverage" in str(w.message)]


def test_nothing_measured_still_fails_closed():
    """OVER-CORRECTION CONTROL in the other direction: the existing refusal must
    survive."""
    out = aggregate_validity([_result(False, None) for _ in range(10)])
    assert out["available"] is False
    assert out["n_measured"] == 0


# --------------------------------------------------------------- the pipeline tracker


def _three_stage_pipeline():
    tracker = PipelineTracker(["screen", "interview", "offer"])
    # 40 observations each way, a real and significant disparity.
    tracker.record_stage("screen", np.array([1.0, 0.0] * 20), np.array([1.0, 0.0, 0.0, 0.0] * 10))
    # ONE observation each way. No test can run.
    tracker.record_stage("interview", np.array([1.0]), np.array([0.0]))
    # Identical arms: a REAL reading of no difference.
    tracker.record_stage("offer", np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    return tracker


def _metrics_by_stage(tracker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return {r.stage_name: r.bias_metrics for r in tracker.compute_cumulative()}


def test_an_untestable_stage_does_not_report_p_equals_one():
    """1.0 is the most reassuring p-value there is: it reads as measured, and no
    difference at all. No test ran."""
    metrics = _metrics_by_stage(_three_stage_pipeline())
    assert math.isnan(metrics["interview"]["p_value"]), (
        f"one observation per arm reported p={metrics['interview']['p_value']!r}"
    )


def test_identical_arms_keep_their_measured_p_of_one():
    """OVER-CORRECTION CONTROL. Two arms with the same outcomes genuinely show
    no difference; refusing there would discard a confirmation."""
    metrics = _metrics_by_stage(_three_stage_pipeline())
    assert metrics["offer"]["p_value"] == 1.0


def test_a_real_test_still_runs():
    """OVER-CORRECTION CONTROL."""
    metrics = _metrics_by_stage(_three_stage_pipeline())
    assert metrics["screen"]["p_value"] < 0.05


def test_the_sample_sizes_are_reported_beside_the_disparity():
    """A disparity of 1.000 over one observation per arm and one over forty are
    different claims, and without the denominator they render identically."""
    metrics = _metrics_by_stage(_three_stage_pipeline())
    assert metrics["interview"]["n_a"] == 1 and metrics["interview"]["n_b"] == 1
    assert metrics["screen"]["n_a"] == 40 and metrics["screen"]["n_b"] == 40
    assert metrics["interview"]["abs_disparity"] == 1.0, "the disparity is still reported"


def test_an_untestable_stage_is_not_named_the_primary_bias_source():
    """The sentence an operator acts on. Measured before the fix: the
    one-observation stage was returned as the primary bias source, ahead of a
    stage with forty observations and a real p of 0.0221."""
    tracker = _three_stage_pipeline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source = tracker.identify_bias_source()

    assert source == "screen", f"named {source!r}, which could not be tested"
    assert any("excluded from the ranking" in str(w.message) for w in caught)


def test_a_pipeline_with_nothing_testable_refuses_to_name_a_source():
    """Three states. Not a stage name chosen from noise, and not silence."""
    tracker = PipelineTracker(["a", "b"])
    tracker.record_stage("a", np.array([1.0]), np.array([0.0]))
    tracker.record_stage("b", np.array([0.0]), np.array([1.0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(RuntimeError, match="No stage could be tested"):
            tracker.identify_bias_source()


def test_the_bias_adding_stage_is_still_the_answer():
    """OVER-CORRECTION CONTROL, and a re-pin of an earlier wave's finding: the
    source is the stage that ADDS the most bias, never the one that reduces it
    most."""
    tracker = PipelineTracker(["reduces", "adds"])
    tracker.record_stage("reduces", np.array([1.0, 0.0] * 20), np.array([1.0, 0.0] * 20))
    tracker.record_stage("adds", np.array([1.0] * 30 + [0.0] * 10), np.array([0.0] * 40))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert tracker.identify_bias_source() == "adds"


# --------------------------------------------------------------- the pre-commit hook


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = precommit_main(argv)
    return code, out.getvalue() + err.getvalue()


def test_a_misconfigured_hook_fails_closed():
    """pre-commit judges a hook purely by its exit code, so returning 0 with no
    subcommand made a misconfigured hook pass every commit while checking
    nothing."""
    code, output = _run([])
    assert code != 0, "a hook that checked nothing reported a pass"
    assert "nothing was checked" in output
    assert "misconfigured" in output


@pytest.mark.parametrize("command", ["check-config", "check-model-card"])
def test_a_correctly_configured_hook_with_no_matching_files_still_passes(command):
    """OVER-CORRECTION CONTROL, and the reason this fix is narrow.

    pre-commit passes only the staged files matching the hook's `files:`
    pattern, so an empty list genuinely means there was nothing of that kind in
    this commit. Failing there would make every unrelated commit red, and a hook
    that is always red gets removed.
    """
    code, _ = _run([command])
    assert code == 0


def test_identical_arms_report_a_measured_p_of_one_on_any_scipy():
    """The shortcut is LOAD-BEARING, and this is the test that found out.

    Written on 2026-09-10 as a tripwire under a comment claiming the
    `np.array_equal` shortcut in `compute_cumulative` was redundant, on the
    evidence that scipy 1.17 returns exactly 1.0 for identical arms. It asserted
    scipy's behaviour so the day the ground moved would be a red test rather
    than a comment that had quietly become false.

    It moved within hours. CI runs scipy 1.18.1, where `mannwhitneyu` returns
    NaN for identical arms, and the tripwire failed with exactly the sentence it
    was written to produce: "scipy now returns nan for identical arms. The
    array_equal shortcut is LOAD-BEARING now rather than redundant."

    So the assumption is resolved and the test changes shape. It no longer
    asserts what SCIPY does, because that is not ours and it now differs by
    version. It asserts what WE do, which is ours and must not differ: two
    identical constant arms report a measured p of 1.0, on any scipy, because
    the shortcut runs before the call. At temperature 0 that is the commonest
    genuinely fair shape, and without the shortcut every fair comparison would
    read as unmeasurable on a modern scipy.
    """
    tracker = PipelineTracker(["stage"])
    tracker.record_stage("stage", np.full(20, 0.5), np.full(20, 0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metrics = tracker.compute_cumulative()[0].bias_metrics

    assert metrics["p_value"] == 1.0, (
        f"identical arms reported p={metrics['p_value']!r}. On scipy >= 1.18 "
        f"mannwhitneyu answers NaN here, so this value can only come from the "
        f"array_equal shortcut; if it is gone, every genuinely fair comparison at "
        f"temperature 0 now reads as unmeasurable."
    )
    assert metrics["abs_disparity"] == 0.0


def test_the_shortcut_that_produces_it_is_still_in_the_source():
    """Belt and braces on the test above, and the reason is specific.

    The assertion above passes on scipy 1.17 even with the shortcut removed,
    because 1.17 happens to return 1.0 itself. On a machine with an older scipy
    a reader could delete the shortcut, watch the suite stay green, and ship a
    regression that only appears on CI. This checks the shortcut is present
    rather than inferring it from a value two scipy versions disagree about.
    """
    source = pathlib.Path("src/vfairness/agents/pipeline_tracker.py").read_text(encoding="utf-8")
    code = [ln for ln in source.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert any("np.array_equal" in ln for ln in code), (
        "the identical-arms shortcut is gone from compute_cumulative. On scipy "
        ">= 1.18 that turns every identical-arm comparison into a could-not-check."
    )
