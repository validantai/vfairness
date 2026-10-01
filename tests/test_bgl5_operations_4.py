"""BGL5 FIX WAVE, batch A-operations-4: the corrected behaviour, pinned.

The BGL4 audit overturned 15 grades in this batch over seven root causes. This
file pins what each fix now does, and for every one of them the OVER-CORRECTION
CONTROL with its real measured value, because a fix that refuses everything
passes every refusal test and destroys the library.

Every pin here was SABOTAGED before being trusted: the defect was put back, the
named test was confirmed red, the source was restored and ``diff`` confirmed it
byte-identical. The sabotage and its output are recorded in
/tmp/claude-501/bgl/fix/fix-A-operations-4.json.

Root causes, in the order they appear below:

1. ``cicd.precommit.check_fairness_config`` validated a threshold's TYPE and
   never its RANGE, so a demographic_parity_difference bound of 1e9 exited 0 and
   the gate built from that config approved a 0.90 disparity.
2. ``cicd.task_handlers.main`` printed ``"success": false`` and exited 0.
3. ``monitoring.sequential`` (cusum_drift / page_hinkley / the wrapper) tested
   the SERIES and never the decision parameters, so threshold=nan, slack=1e9,
   lambda_=nan or delta=1e9 turned an 80-point step into a measured "no drift".
4. ``monitoring.drift.run_sprt`` tested ``np.ptp(stream) == 0.0`` exactly, so one
   bit of float jitter restored the single-observation all-clear its own comment
   says it removed.
5. ``reporting.interactive.simulate_threshold_change`` (module function and the
   dashboard wrapper) never checked the proposed threshold, so nan projected
   "both alerts removed, a 100 percent reduction".
6. ``cicd.quality_report.build_quality_report`` had no outcome slot in either
   direction, so a clean outcome check and one that never ran were the same
   document.
7. ``reporting.compliance``: presence-not-content evidence in the ISO 42001 map,
   a partial scope limitation the assurance verdict could not see, an adverse
   action notice whose ``complete`` could not see a short feature_names, and a
   signed test log that sealed the wrong timestamp and left a returned field
   outside its own hash.
8. ``causal.graph`` cleared a graph with no edges, and ``causal.task_handlers``
   decoded dataset bytes with ``errors="replace"``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


def _caught(fn, *args, **kwargs):
    """Run ``fn`` and return ``(result, [warning texts])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# ===========================================================================
# 1. cicd.precommit: a bound outside the metric's range is not a bound
# ===========================================================================


def _config(tmp_path, payload, name="fairness.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


class TestPrecommitRefusesABoundNoValueCanBreach:
    """Measured on HEAD before the fix:

        {"metrics": ["demographic_parity_difference"],
         "thresholds": {"demographic_parity_difference": 1e9}}   -> exit 0

    and ModelFairnessGate built from that same config returned
    GateDecision(approved=True) for a measured demographic_parity_difference of
    0.90, which is verbatim the harm the checker's own comment exists to prevent.
    """

    @pytest.mark.parametrize(
        "payload,expected_in_output",
        [
            pytest.param(
                {
                    "metrics": ["demographic_parity_difference"],
                    "thresholds": {"demographic_parity_difference": 1e9},
                },
                "no value can exceed this bound",
                id="difference-bound-1e9",
            ),
            pytest.param(
                {
                    "metrics": ["demographic_parity_difference"],
                    "thresholds": {"demographic_parity_difference": 1.0},
                },
                "lies in [0, 1]",
                id="difference-bound-1.0-is-the-boundary",
            ),
            pytest.param(
                {
                    "metrics": ["equalized_odds_difference"],
                    "thresholds": {"equalized_odds_difference": -0.1},
                },
                "no magnitude can meet a negative maximum",
                id="negative-maximum-grades-nothing",
            ),
            pytest.param(
                {
                    "metrics": ["disparate_impact_ratio"],
                    "thresholds": {"disparate_impact_ratio": 0.0},
                },
                "every possible value meets it",
                id="ratio-floor-0.0",
            ),
            pytest.param(
                {
                    "metrics": ["disparate_impact_ratio"],
                    "thresholds": {"disparate_impact_ratio": 5.0},
                },
                "no value can reach this floor",
                id="ratio-floor-above-1",
            ),
        ],
    )
    def test_a_bound_outside_the_metrics_range_is_refused(
        self, tmp_path, capsys, payload, expected_in_output
    ):
        from vfairness.operations.cicd.precommit import check_fairness_config

        assert check_fairness_config([_config(tmp_path, payload)]) == 1
        printed = capsys.readouterr().out
        assert "FAIL" in printed and expected_in_output in printed, printed

    # ---- OVER-CORRECTION CONTROLS: the real numbers a working config keeps ----

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(
                {
                    "metrics": ["demographic_parity_difference"],
                    "thresholds": {"demographic_parity_difference": 0.1},
                },
                id="a-real-tolerance",
            ),
            pytest.param(
                {
                    "metrics": ["demographic_parity_difference"],
                    "thresholds": {"demographic_parity_difference": 0.0},
                },
                id="zero-tolerance-is-a-real-policy",
            ),
            pytest.param(
                {
                    "metrics": ["disparate_impact_ratio"],
                    "thresholds": {"disparate_impact_ratio": 0.8},
                },
                id="the-four-fifths-floor",
            ),
            pytest.param(
                {"metrics": ["dp"], "thresholds": {"dp": 0.1}},
                id="a-metric-whose-range-is-unknown-is-left-alone",
            ),
            pytest.param(
                {"metrics": ["salary_gap_chf"], "thresholds": {"salary_gap_chf": 8500.0}},
                id="a-continuous-outcome-gap-may-exceed-1",
            ),
        ],
    )
    def test_a_config_that_can_actually_gate_still_exits_zero(self, tmp_path, payload):
        from vfairness.operations.cicd.precommit import check_fairness_config

        assert check_fairness_config([_config(tmp_path, payload)]) == 0

    def test_the_orphan_threshold_refusal_is_pinned_on_its_own(self, tmp_path, capsys):
        """The auditor's second finding on this unit: the orphan-threshold branch
        was refused NOWHERE in the repository. Disabling it left all 86 tests in
        the three relevant files green, because the one fixture that claimed to
        cover it also carried an unbounded metric and reddened through the other
        branch. This fixture has NO other defect in it.
        """
        from vfairness.operations.cicd.precommit import check_fairness_config

        payload = {
            "metrics": ["demographic_parity_difference"],
            "thresholds": {
                "demographic_parity_difference": 0.1,
                "equalized_odds_difference": 0.1,  # names no gated metric
            },
        }
        assert check_fairness_config([_config(tmp_path, payload)]) == 1
        assert "name no metric in 'metrics'" in capsys.readouterr().out


# ===========================================================================
# 2. cicd.task_handlers.main: the exit code a CI step reads
# ===========================================================================


def _run_cli(payload_text: str, argv=("prog",)):
    import io
    import sys

    from vfairness.operations.cicd import task_handlers

    saved = (sys.argv, sys.stdin, sys.stdout)
    sys.argv, sys.stdin, sys.stdout = list(argv), io.StringIO(payload_text), io.StringIO()
    try:
        code = task_handlers.main()
        return code, sys.stdout.getvalue()
    finally:
        sys.argv, sys.stdin, sys.stdout = saved


class TestTheDataValidationCliExitCodeMatchesItsOwnEnvelope:
    """Measured on HEAD, as a shell sees it: a payload whose protected
    attributes are all absent printed ``"success": false`` and exited 0, while
    bad JSON and an unknown task type exited 1, so an infrastructure failure and
    a check that could not run were told apart and a REFUSAL was not.
    """

    def test_a_refusal_exits_one(self):
        frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        code, out = _run_cli(
            json.dumps(
                {"csv_data": frame.to_csv(index=False), "protected_attributes": ["not_here"]}
            )
        )
        payload = json.loads(out)
        assert payload["success"] is False, "the fixture stopped producing a refusal"
        assert "cannot be validated" in payload["error"]
        assert code == 1, f"a refusal exited {code}, which a CI step reads as success"

    def test_an_empty_payload_is_a_refusal_too(self):
        code, out = _run_cli("")
        assert json.loads(out)["success"] is False
        assert code == 1

    # ---- OVER-CORRECTION CONTROL ----

    def test_control_a_successful_validation_still_exits_zero(self):
        """The real measurement must still be a pass: 200 rows, gender present,
        tone 'pass', exit 0, and stdout unchanged in shape.
        """
        rng = np.random.default_rng(4)
        n = 200
        frame = pd.DataFrame(
            {
                "gender": np.where(rng.random(n) < 0.5, "M", "F"),
                "score": rng.normal(0, 1, n),
                "approved": rng.binomial(1, 0.6, n),
            }
        )
        code, out = _run_cli(
            json.dumps(
                {
                    "csv_data": frame.to_csv(index=False),
                    "protected_attributes": ["gender"],
                    "outcome_column": "approved",
                }
            )
        )
        payload = json.loads(out)
        assert payload["success"] is True
        assert payload["data"]["tone"] == "pass", payload["data"]["checks"]
        assert code == 0


# ===========================================================================
# 3. monitoring.sequential: a decision parameter that specifies no test
# ===========================================================================


def _stepped_series():
    """50 windows at 0.10 then 50 at 0.90. The defaults call this abrupt_drift."""
    rng = np.random.default_rng(0)
    return list(rng.normal(0.10, 0.01, 50)) + list(rng.normal(0.90, 0.01, 50))


class TestSequentialDetectorsRefuseAVacuousDecisionParameter:
    """Measured on HEAD, on a series carrying a genuine 80-point step:

        cusum_drift(step, threshold=nan) -> has_drift False, max_cusum 24.99
        cusum_drift(step, slack=1e9)     -> has_drift False, max_cusum 0.0
        page_hinkley(step, lambda_=nan)  -> has_drift False, drift_index None
        page_hinkley(step, delta=1e9)    -> has_drift False, drift_index None

    all with zero warnings. run_sprt was given exactly this guard for its alpha
    and beta in the previous wave; its siblings here had none.
    """

    @pytest.mark.parametrize(
        "kwargs,phrase",
        [
            ({"threshold": float("nan")}, "no accumulated statistic can reach"),
            ({"threshold": float("inf")}, "no accumulated statistic can reach"),
            ({"threshold": 0.0}, "the first window crosses it"),
            ({"threshold": -5.0}, "the first window crosses it"),
            ({"slack": float("nan")}, "both arms stay pinned at 0.0"),
            ({"slack": 1e9}, "no window could add anything"),
            # Not a number at all: is_measured refuses None, a bool and a string,
            # and np.isfinite would have raised TypeError on each of them.
            ({"slack": None}, "not a finite number"),
            ({"threshold": "5"}, "no accumulated statistic can reach"),
            ({"threshold": True}, "no accumulated statistic can reach"),
        ],
    )
    def test_cusum_refuses_and_names_the_parameter(self, kwargs, phrase):
        from vfairness.operations.monitoring.sequential import cusum_drift

        result, messages = _caught(cusum_drift, _stepped_series(), **kwargs)
        assert result["has_drift"] is None, (
            f"{kwargs} defines no decision boundary, so no CUSUM test ran; "
            f"has_drift was {result['has_drift']!r} with max_cusum {result['max_cusum']!r}"
        )
        assert np.isnan(result["max_cusum"]), result["max_cusum"]
        assert any(phrase in m for m in messages), messages

    @pytest.mark.parametrize(
        "kwargs,phrase",
        [
            ({"lambda_": float("nan")}, "which the statistic cannot exceed"),
            ({"lambda_": float("inf")}, "which the statistic cannot exceed"),
            ({"lambda_": 0.0}, "non-negative by construction"),
            ({"delta": float("nan")}, "the cumulative sum is undefined"),
            ({"delta": 1e9}, "more than the data could supply"),
            ({"alpha": 0.0}, "outside the (0, 1] range"),
            ({"alpha": 1.5}, "outside the (0, 1] range"),
            ({"alpha": None}, "outside the (0, 1] range"),
            ({"lambda_": "20"}, "which the statistic cannot exceed"),
        ],
    )
    def test_page_hinkley_refuses_and_names_the_parameter(self, kwargs, phrase):
        from vfairness.operations.monitoring.sequential import page_hinkley

        result, messages = _caught(page_hinkley, _stepped_series(), **kwargs)
        assert result["has_drift"] is None, (
            f"{kwargs} puts the PH threshold out of reach of any data, so nothing "
            f"was tested; has_drift was {result['has_drift']!r}"
        )
        assert np.isnan(result["min_ph"]), result["min_ph"]
        assert any(phrase in m for m in messages), messages

    def test_the_wrapper_classifies_it_not_assessed(self):
        from vfairness.operations.monitoring.sequential import sequential_fairness_drift

        result, messages = _caught(
            sequential_fairness_drift,
            _stepped_series(),
            {"threshold": float("nan")},
            {"lambda_": float("nan")},
        )
        assert result["classification"] == "not_assessed", result["classification"]
        assert result["cusum"]["has_drift"] is None
        assert result["page_hinkley"]["has_drift"] is None
        assert messages, "a refusal with no disclosure"

    # ---- OVER-CORRECTION CONTROLS: the real numbers, to the last digit ----

    def test_control_the_defaults_still_find_the_step(self):
        from vfairness.operations.monitoring.sequential import (
            cusum_drift,
            page_hinkley,
            sequential_fairness_drift,
        )

        series = _stepped_series()
        cusum, cusum_msgs = _caught(cusum_drift, series)
        assert (cusum["has_drift"], cusum["drift_index"]) == (True, 9)
        assert cusum["max_cusum"] == pytest.approx(24.98554351979067)
        assert cusum_msgs == []

        ph, ph_msgs = _caught(page_hinkley, series)
        assert (ph["has_drift"], ph["drift_index"]) == (True, 61)
        assert ph["min_ph"] == pytest.approx(-2.2727160989587887)
        assert ph_msgs == []

        wrapper, wrapper_msgs = _caught(sequential_fairness_drift, series)
        assert wrapper["classification"] == "abrupt_drift"
        assert wrapper_msgs == []

    def test_control_a_flat_measured_series_is_still_a_measured_false(self):
        """0.0 here is a verdict: the windows were read and they do not move."""
        from vfairness.operations.monitoring.sequential import (
            cusum_drift,
            page_hinkley,
            sequential_fairness_drift,
        )

        cusum, cusum_msgs = _caught(cusum_drift, [0.9] * 50)
        assert cusum["has_drift"] is False and cusum["max_cusum"] == 0.0
        assert cusum_msgs == []

        ph, ph_msgs = _caught(page_hinkley, [0.25] * 50)
        assert ph["has_drift"] is False and ph["ph_statistic"] == [0.0] * 50
        assert ph_msgs == []

        wrapper, wrapper_msgs = _caught(sequential_fairness_drift, [0.2] * 50)
        assert wrapper["classification"] == "stable"
        assert wrapper_msgs == []

    def test_control_a_monotone_series_keeps_its_measured_ph_false(self):
        """The PH statistic genuinely never rises on a monotone DECREASE, and
        that False must survive: the new delta guard compares the allowance
        against the largest standardized deviation the series contains, which is
        far above the default 0.05 here.
        """
        from vfairness.operations.monitoring.sequential import page_hinkley

        result, messages = _caught(page_hinkley, list(np.linspace(0.9, 0.1, 60)))
        assert result["has_drift"] is False
        assert result["min_ph"] == pytest.approx(-53.99510626089794)
        assert messages == []

    def test_control_a_stricter_but_legitimate_parameter_pair_still_runs(self):
        from vfairness.operations.monitoring.sequential import cusum_drift, page_hinkley

        cusum, cusum_msgs = _caught(cusum_drift, _stepped_series(), slack=0.25, threshold=3.0)
        assert cusum["has_drift"] is True and cusum_msgs == []
        ph, ph_msgs = _caught(page_hinkley, _stepped_series(), delta=0.1, lambda_=10.0)
        assert ph["has_drift"] is True and ph_msgs == []


# ===========================================================================
# 3b. monitoring.sequential: the SIBLING decision parameter, BGL6
# ===========================================================================


def _short_step():
    """25 windows at 0.10 then 25 at 0.40. The audit's series, exactly."""
    return [0.10] * 25 + [0.40] * 25


class TestTheSiblingDecisionParameterIsJudgedByTheSamePrecondition:
    """BGL6 AUDIT, 2026-09-29. The guards pinned in the class above were built
    for ONE of each detector's two decision parameters, and the sibling walked
    straight through. Measured on ``[0.10] * 25 + [0.40] * 25``, a real abrupt
    step, before this fix:

        cusum_drift(step)                 -> has_drift True, index 9,
                                             max_cusum 12.500000000000002
        cusum_drift(step, slack=1e9)      -> has_drift None, 1 warning  (OK)
        cusum_drift(step, threshold=1e9)  -> has_drift False, drift_index None,
                                             max_cusum 12.500000000000002,
                                             ZERO warnings
        cusum_drift(step, threshold=1e300) -> identical

        page_hinkley(step)                -> has_drift True, index 38, the PH
                                             statistic peaking at 32.87119981717022
        page_hinkley(step, delta=1e9)     -> has_drift None, 1 warning  (OK)
        page_hinkley(step, lambda_=1e9)   -> has_drift False, drift_index None,
                                             min_ph -1.248501149367767, ZERO warnings
        page_hinkley(step, lambda_=1e300) -> identical

        sequential_fairness_drift(step, {"threshold": 1e300}, {"lambda_": 1e300})
                                          -> classification "stable", both
                                             has_drift False, ZERO warnings
                                          (the defaults: "abrupt_drift")

    Either CUSUM arm is bounded by n_measured * max|z|, about 50 for that
    series, so 1e9 was out of reach BY CONSTRUCTION and "no drift" was a
    statement about the threshold, not about the data. WHEN TWO PARAMETERS SHARE
    A PRECONDITION, CHECKING ONE OF THEM IS NOT CHECKING THE PRECONDITION.
    """

    @pytest.mark.parametrize("threshold", [1e9, 1e300, 1e6])
    def test_cusum_refuses_a_finite_but_unreachable_threshold(self, threshold):
        from vfairness.operations.monitoring.sequential import cusum_drift

        result, messages = _caught(cusum_drift, _short_step(), threshold=threshold)
        assert result["has_drift"] is None, (
            f"threshold={threshold:g} is above anything either arm could reach, so no "
            f"CUSUM test ran; has_drift was {result['has_drift']!r} with max_cusum "
            f"{result['max_cusum']!r}"
        )
        assert np.isnan(result["max_cusum"]), result["max_cusum"]
        assert any("could not have fired for any arrangement" in m for m in messages), messages

    @pytest.mark.parametrize("lambda_", [1e9, 1e300, 1e6])
    def test_page_hinkley_refuses_a_finite_but_unreachable_lambda(self, lambda_):
        from vfairness.operations.monitoring.sequential import page_hinkley

        result, messages = _caught(page_hinkley, _short_step(), lambda_=lambda_)
        assert result["has_drift"] is None, (
            f"lambda_={lambda_:g} is above anything the PH statistic could reach, so "
            f"nothing was tested; has_drift was {result['has_drift']!r} with min_ph "
            f"{result['min_ph']!r}"
        )
        assert np.isnan(result["min_ph"]), result["min_ph"]
        assert any("could not have fired for any arrangement" in m for m in messages), messages

    @pytest.mark.parametrize("bound", [1e9, 1e300])
    def test_the_wrapper_inherits_the_refusal_instead_of_stable(self, bound):
        from vfairness.operations.monitoring.sequential import sequential_fairness_drift

        result, messages = _caught(
            sequential_fairness_drift,
            _short_step(),
            {"threshold": bound},
            {"lambda_": bound},
        )
        assert result["classification"] == "not_assessed", (
            f"a series carrying a real step, with both decision bounds at {bound:g}, "
            f"classified {result['classification']!r}"
        )
        assert result["cusum"]["has_drift"] is None
        assert result["page_hinkley"]["has_drift"] is None
        assert len(messages) == 2, messages

    def test_the_ceiling_is_named_so_a_reader_can_check_it(self):
        """The disclosure has to carry the bound, not just the word NOT ASSESSED:
        the reader is being told the parameter was unreachable and must be able
        to see what it was unreachable against.
        """
        from vfairness.operations.monitoring.sequential import cusum_drift, page_hinkley

        _, cusum_msgs = _caught(cusum_drift, _short_step(), threshold=1e9)
        assert "50 window(s)" in cusum_msgs[0] and " is 50, so" in cusum_msgs[0], cusum_msgs
        _, ph_msgs = _caught(page_hinkley, _short_step(), lambda_=1e9)
        assert "50 window(s)" in ph_msgs[0] and " is 197.3, so" in ph_msgs[0], ph_msgs

    # ---- OVER-CORRECTION CONTROLS: the real numbers, on the audit's series ----

    def test_control_the_real_step_is_still_found_at_its_real_index(self):
        from vfairness.operations.monitoring.sequential import (
            cusum_drift,
            page_hinkley,
            sequential_fairness_drift,
        )

        step = _short_step()
        cusum, cusum_msgs = _caught(cusum_drift, step)
        assert (cusum["has_drift"], cusum["drift_index"]) == (True, 9)
        assert cusum["max_cusum"] == pytest.approx(12.500000000000002)
        assert cusum_msgs == []

        ph, ph_msgs = _caught(page_hinkley, step)
        assert (ph["has_drift"], ph["drift_index"]) == (True, 38)
        assert ph["min_ph"] == pytest.approx(-1.248501149367767)
        assert max(ph["ph_statistic"]) == pytest.approx(32.87119981717022)
        assert ph_msgs == []

        wrapper, wrapper_msgs = _caught(sequential_fairness_drift, step)
        assert wrapper["classification"] == "abrupt_drift"
        assert wrapper_msgs == []

    def test_control_a_genuinely_stable_series_still_reads_stable(self):
        """Noise only, 50 windows, no step. Both detectors must keep their
        MEASURED False and the wrapper its "stable", because a ceiling that
        refused everything would pass every refusal pin above and publish
        nothing.
        """
        from vfairness.operations.monitoring.sequential import (
            cusum_drift,
            page_hinkley,
            sequential_fairness_drift,
        )

        series = list(np.random.default_rng(3).normal(0.25, 0.02, 50))

        cusum, cusum_msgs = _caught(cusum_drift, series)
        assert cusum["has_drift"] is False
        assert cusum["max_cusum"] == pytest.approx(2.520161290069448)
        assert cusum_msgs == []

        ph, ph_msgs = _caught(page_hinkley, series)
        assert ph["has_drift"] is False
        assert ph["min_ph"] == pytest.approx(-4.264825250948108)
        assert ph_msgs == []

        wrapper, wrapper_msgs = _caught(sequential_fairness_drift, series)
        assert wrapper["classification"] == "stable"
        assert wrapper_msgs == []

    def test_control_the_ceiling_can_only_refuse_a_provably_unreachable_bound(self):
        """The bound is an over-estimate of what the statistic could reach, so
        the guard cannot refuse a threshold that COULD have fired. Pinned at the
        two ends: the textbook defaults on a 50-window series are far below the
        ceiling and still measure, and a bound just under the ceiling measures
        too even though the data never approaches it.
        """
        from vfairness.operations.monitoring.sequential import cusum_drift, page_hinkley

        step = _short_step()
        cusum, cusum_msgs = _caught(cusum_drift, step, threshold=49.0)
        assert cusum["has_drift"] is False and cusum["max_cusum"] == pytest.approx(
            12.500000000000002
        )
        assert cusum_msgs == [], "49 is under the ceiling of 50, so this is a measurement"

        ph, ph_msgs = _caught(page_hinkley, step, lambda_=190.0)
        assert ph["has_drift"] is False and ph_msgs == []

    def test_control_a_series_too_short_for_the_default_bound_now_says_so(self):
        """A deliberate consequence of the ceiling, recorded rather than left to
        be discovered. Three windows cannot carry a CUSUM arm to h = 5 sigma and
        five cannot carry a PH statistic to lambda_ = 20 sigma, whatever the
        data does, so the old measured ``False`` there was the same neutral value
        published as a measurement. The defaults were calibrated on 50 to 200
        window series (see the page_hinkley docstring).
        """
        from vfairness.operations.monitoring.sequential import cusum_drift, page_hinkley

        rng = np.random.default_rng(11)
        cusum, cusum_msgs = _caught(cusum_drift, list(rng.normal(0.25, 0.02, 3)))
        assert cusum["has_drift"] is None and cusum_msgs
        ph, ph_msgs = _caught(page_hinkley, list(rng.normal(0.25, 0.02, 5)))
        assert ph["has_drift"] is None and ph_msgs
        # ... and ten windows are enough for both to measure again.
        long_enough = list(rng.normal(0.25, 0.02, 10))
        assert _caught(cusum_drift, long_enough)[0]["has_drift"] is False
        assert _caught(page_hinkley, long_enough)[0]["has_drift"] is False


# ===========================================================================
# 4. monitoring.drift.run_sprt: constancy is not exact equality
# ===========================================================================


class TestSprtRefusesAStreamWithNoMeaningfulVariation:
    """Measured on HEAD:

        run_sprt([0.9]*40,               0.9, 0.2) -> ('could_not_check', 40, nan)
        run_sprt([0.9]*39 + [0.9+1e-15], 0.9, 0.2) -> ('stable', 1, -9.8e+30)
        run_sprt([0.9]*39 + [0.9+1e-15], 0.1, 0.2) -> ('drift',  1, +3.0e+30)

    One float-rounding artefact defeated ``np.ptp(...) == 0.0`` and produced the
    fabrication the method's own comment says it removed: "an all-clear on a
    monitored metric, declared from a single observation, with a likelihood ratio
    of twenty billion behind it".
    """

    @pytest.mark.parametrize("null,alt", [(0.9, 0.2), (0.1, 0.2)])
    def test_one_bit_of_jitter_is_still_no_variation(self, null, alt):
        from vfairness.operations.monitoring.drift import FairnessDriftDetector

        stream = [0.9] * 39 + [0.9 + 1e-15]
        (decision, n, llr), messages = _caught(FairnessDriftDetector().run_sprt, stream, null, alt)
        assert decision == "could_not_check", (
            f"a stream whose range is {float(np.ptp(np.asarray(stream))):.3g} carries no "
            f"variation to measure evidence against; got {decision!r} at observation {n} "
            f"with log_lambda {llr!r}"
        )
        assert np.isnan(llr)
        assert any("float rounding noise" in m for m in messages), messages

    def test_the_bit_exact_constant_stream_keeps_its_own_wording(self):
        from vfairness.operations.monitoring.drift import FairnessDriftDetector

        (decision, _n, llr), messages = _caught(
            FairnessDriftDetector().run_sprt, [0.9] * 40, 0.9, 0.2
        )
        assert decision == "could_not_check" and np.isnan(llr)
        assert any("identical value 0.9" in m for m in messages), messages

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_noisy_stream_is_still_decided_with_its_real_ratio(self):
        from vfairness.operations.monitoring.drift import FairnessDriftDetector

        rng = np.random.default_rng(7)
        stable = list(rng.normal(0.9, 0.02, 40))
        (decision, n, llr), messages = _caught(FairnessDriftDetector().run_sprt, stable, 0.9, 0.2)
        assert (decision, n) == ("stable", 1)
        assert llr == pytest.approx(-932.6305984962497)
        assert messages == []

        drifted = list(rng.normal(0.2, 0.02, 40))
        (decision, n, llr), messages = _caught(FairnessDriftDetector().run_sprt, drifted, 0.9, 0.2)
        assert (decision, n) == ("drift", 1)
        assert llr == pytest.approx(748.0670725861512)
        assert messages == []

    def test_control_a_real_but_tiny_variation_is_still_measured(self):
        """Variation of 1e-5 around 0.9 is 1e-5 RELATIVE, seven orders of
        magnitude above the 1e-12 rounding floor, so it is data and is measured.
        """
        from vfairness.operations.monitoring.drift import FairnessDriftDetector

        (decision, n, llr), messages = _caught(
            FairnessDriftDetector().run_sprt, list(np.linspace(0.9, 0.9 + 1e-5, 40)), 0.9, 0.2
        )
        assert (decision, n) == ("stable", 1)
        assert np.isfinite(llr) and llr < 0
        assert messages == []


# ===========================================================================
# 5. reporting.interactive: a proposed threshold that is not a number
# ===========================================================================


def _metric_store(rows):
    from vfairness.operations.reporting.store import MetricsStore, StoredMetricRecord

    store = MetricsStore()
    now = datetime.now()
    for value, alert, group in rows:
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(days=1),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=float(value),
                group=group,
                alert=alert,
            )
        )
    return store


_ALERTING_ROWS = [(0.02, False, "a"), (0.95, True, "b"), (0.9, True, "b")]


class _Node:
    """Stand-in for a dash html/dcc component (same double as
    tests/test_readiness5_consumers5.py, kept local so the two files do not
    import each other): records children and renders their text."""

    def __init__(self, children=None, **kwargs):
        self.children = children
        self.kwargs = kwargs

    def text(self):
        parts = []
        child = self.children
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, _Node):
            parts.append(child.text())
        elif isinstance(child, (list, tuple)):
            for item in child:
                parts.append(item.text() if isinstance(item, _Node) else str(item))
        return " ".join(p for p in parts if p)


class _Namespace:
    def __getattr__(self, name):
        return _Node


class _RecordingApp:
    def __init__(self, *args, **kwargs):
        self.layout = None
        self.recorded = {}

    def callback(self, *args, **kwargs):
        def decorate(fn):
            self.recorded[fn.__name__] = fn
            return fn

        return decorate


class _FakeDash:
    Dash = _RecordingApp


def _whatif_panel_text(monkeypatch, store, threshold):
    """Run the SHIPPED update_whatif callback body with dash's components stubbed.

    dash is an optional dependency and is not installed in this venv, so the
    layout objects are doubles; the callback body executed here is the shipped
    one, which is the surface an operator reads.
    """
    from vfairness.operations.reporting import interactive as interactive_mod
    from vfairness.operations.reporting.interactive import InteractiveDashboard

    monkeypatch.setattr(interactive_mod, "dash", _FakeDash(), raising=False)
    monkeypatch.setattr(interactive_mod, "html", _Namespace(), raising=False)
    monkeypatch.setattr(interactive_mod, "dcc", _Namespace(), raising=False)
    monkeypatch.setattr(interactive_mod, "Input", lambda *a, **k: ("in", a), raising=False)
    monkeypatch.setattr(interactive_mod, "Output", lambda *a, **k: ("out", a), raising=False)
    monkeypatch.setattr(interactive_mod, "_DASH_AVAILABLE", True, raising=False)
    callbacks = InteractiveDashboard(store).create_dash_app().recorded
    return callbacks["update_whatif"](threshold, "demographic_parity").text()


class TestTheWhatIfSimulationRefusesABoundThatIsNotANumber:
    """Measured on HEAD, on a window holding 2 real alerts:

        simulate_threshold_change(store, "demographic_parity", nan)
          -> current 2, projected 0, groups_impacted [], change_abs -2,
             change_pct -100.0, records_simulated 3, not_simulated_reason ''

    Every ``value > nan`` is False, so no record was compared to anything, and an
    operator choosing a threshold was told the change removes both alerts.
    """

    @pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), None, True])
    def test_both_surfaces_refuse_it(self, threshold):
        from vfairness.operations.reporting.interactive import (
            InteractiveDashboard,
            simulate_threshold_change,
        )

        result = simulate_threshold_change(
            _metric_store(_ALERTING_ROWS), "demographic_parity", threshold
        )
        assert result["projected_alerts"] is None and result["not_simulated_reason"], (
            f"a proposed threshold of {threshold!r} is not a comparison; got "
            f"projected_alerts={result['projected_alerts']!r}, "
            f"change_pct={result['change_pct']!r}, reason={result['not_simulated_reason']!r}"
        )
        for key in ("current_alerts", "groups_impacted", "change_abs", "change_pct"):
            assert result[key] is None, f"{key} was {result[key]!r}"
        assert "COULD NOT SIMULATE" in result["not_simulated_reason"]

        wrapper = InteractiveDashboard(_metric_store(_ALERTING_ROWS)).simulate_threshold_change(
            "demographic_parity", threshold
        )
        assert wrapper == result, "the dashboard wrapper diverged from the module function"

    def test_the_dash_panel_shows_the_producers_own_reason(self, monkeypatch):
        """The disclosure has to reach the surface a person reads. The callback
        rendered one hardcoded sentence for every could-not-check, which would
        have described this refusal as "no record in this window was compared to
        a threshold": a statement about the data, and false of a nan bound.
        """
        text = _whatif_panel_text(monkeypatch, _metric_store(_ALERTING_ROWS), float("nan"))
        assert "Could not check" in text
        assert "which no value can be compared against" in text
        assert "no record in this window was compared" not in text

    def test_control_the_dash_panel_still_renders_a_real_simulation(self, monkeypatch):
        text = _whatif_panel_text(monkeypatch, _metric_store(_ALERTING_ROWS), 0.5)
        assert "Current alerts: 2" in text
        assert "Projected alerts: 2" in text
        assert "Groups impacted: b" in text
        assert "Could not check" not in text

    # ---- OVER-CORRECTION CONTROL ----

    def test_control_a_real_threshold_still_projects_its_real_counts(self):
        from vfairness.operations.reporting.interactive import simulate_threshold_change

        result = simulate_threshold_change(_metric_store(_ALERTING_ROWS), "demographic_parity", 0.5)
        assert (result["current_alerts"], result["projected_alerts"]) == (2, 2)
        assert result["groups_impacted"] == ["b"]
        assert (result["change_abs"], result["change_pct"]) == (0, 0.0)
        assert result["records_simulated"] == 3
        assert result["not_simulated_reason"] == ""


# ===========================================================================
# 6. cicd.quality_report: the outcome slot, in all three states
# ===========================================================================


def _clean_frame():
    rng = np.random.default_rng(11)
    n = 400
    g = np.where(rng.random(n) < 0.5, "M", "F")
    return pd.DataFrame(
        {
            "applicant_id": [f"a{i}" for i in range(n)],
            "gender": g,
            "score": rng.normal(0, 1, n),
            "approved": np.random.default_rng(3).binomial(1, 0.6, n),
        }
    )


def _gap_frame():
    rng = np.random.default_rng(11)
    n = 400
    g = np.where(rng.random(n) < 0.5, "M", "F")
    y = np.where(g == "M", rng.binomial(1, 0.8, n), rng.binomial(1, 0.3, n))
    return pd.DataFrame(
        {
            "applicant_id": [f"a{i}" for i in range(n)],
            "gender": g,
            "score": rng.normal(0, 1, n),
            "approved": y,
        }
    )


class TestTheQualityReportSaysWhetherTheOutcomeChecksRan:
    """Measured on HEAD, same 400 clean rows, outcome deliberately clean:

        outcome_column='approved' -> tone pass, keys [vf_representation,
                                     vf_missing, vf_hygiene]
        outcome_column=None       -> BYTE-IDENTICAL

    so a reader could not tell a clean outcome check from one that never ran.
    """

    def test_a_check_that_ran_and_one_that_never_ran_are_different_documents(self):
        from vfairness.operations.cicd.quality_report import build_quality_report

        df = _clean_frame()
        checked = build_quality_report(df, ["gender"], outcome_column="approved")
        never = build_quality_report(df, ["gender"], outcome_column=None)

        assert [c["key"] for c in checked["checks"]] != [c["key"] for c in never["checks"]]

        ran = [c for c in checked["checks"] if c["label"] == "Outcome checks"]
        assert len(ran) == 1 and ran[0]["status"] == "pass"
        assert "found no problem" in ran[0]["detail"]

        not_run = [c for c in never["checks"] if c["label"] == "Outcome checks"]
        assert len(not_run) == 1 and not_run[0]["status"] == "info"
        assert "were NOT performed" in not_run[0]["detail"]

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_the_pulse_case_is_still_a_clean_pass(self):
        """A caller with no outcome column (Pulse) asked nothing about outcomes,
        so the row is tone-neutral 'info' and the headline stays green.
        """
        from vfairness.operations.cicd.quality_report import build_quality_report

        report = build_quality_report(_clean_frame(), ["gender"], outcome_column=None)
        assert report["tone"] == "pass"
        assert report["headline"] == "The data is fit for a reliable fairness read."

    def test_control_a_real_gap_and_a_typo_are_unchanged(self):
        from vfairness.operations.cicd.quality_report import build_quality_report

        gap = build_quality_report(_gap_frame(), ["gender"], outcome_column="approved")
        assert gap["tone"] == "critical"
        assert any(c["label"] == "Raw outcome gap" for c in gap["checks"])
        assert not any(c["label"] == "Outcome checks" for c in gap["checks"]), (
            "the pass tile must not be rendered beside a measured gap"
        )

        typo = build_quality_report(_gap_frame(), ["gender"], outcome_column="aproved")
        assert typo["tone"] == "critical"
        assert any(c["label"] == "Outcome column" for c in typo["checks"])
        assert not any(c["label"] == "Outcome checks" for c in typo["checks"])

    def test_control_the_three_original_pass_tiles_are_still_there(self):
        from vfairness.operations.cicd.quality_report import build_quality_report

        report = build_quality_report(_clean_frame(), ["gender"], outcome_column="approved")
        by_key = {c["key"]: c["status"] for c in report["checks"]}
        assert by_key["vf_representation"] == "pass"
        assert by_key["vf_missing"] == "pass"
        assert by_key["vf_hygiene"] == "pass"
        assert by_key["vf_outcome"] == "pass"
        assert report["tone"] == "pass"


# ===========================================================================
# 7a. reporting.compliance: presence is not evidence (ISO 42001)
# ===========================================================================


_FULL_WIZARD = {
    "system_profile": {"purpose": "credit scoring"},
    "bias_results": [{"finding": "disparity"}],
    "metric_results": {"demographic_parity": 0.12},
    "risk_register": [
        {"risk": "proxy discrimination", "treatment": "reweighting", "residual_risk": 4}
    ],
    "monitoring_config": {"enabled": True},
    "intervention_results": [{"intervention": "threshold shift"}],
    "dpia": {"completed": True},
    "annex_iv": {"completed": True},
    "model_card": {"completed": True},
}

_PLACEHOLDER_WIZARD = {
    "system_profile": {"a": None},
    "bias_results": [0],
    "metric_results": [0],
    "risk_register": [{}],
    "monitoring_config": {"x": 0},
    "intervention_results": [{}],
    "dpia": {"x": 0},
    "annex_iv": {"x": 0},
    "model_card": {"x": 0},
}


class TestIso42001CreditsContentNotPresence:
    """Measured on HEAD: nine evidence keys holding one contentless placeholder
    each earned coverage_percent 92.3 with 11 controls 'covered' and 0 gaps, the
    same figure a FULL wizard earns, and ``{"risk_register": [{}]}`` alone earned
    26.9% with A.4.3 asserting "Treatment plans generated for each identified
    risk".
    """

    def test_contentless_placeholders_earn_almost_nothing(self):
        from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

        result = generate_iso42001_evidence_map(_PLACEHOLDER_WIZARD)
        assert result["coverage_percent"] < 25.0, (
            f"a wizard whose every section is an empty placeholder earned "
            f"{result['coverage_percent']}% of ISO 42001 with {result['covered']} "
            f"controls 'covered' and {result['gaps']} gaps"
        )
        assert result["gaps"] >= 10

    def test_an_empty_risk_entry_evidences_no_treatment_plan(self):
        from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

        result = generate_iso42001_evidence_map({"risk_register": [{}]})
        by_id = {c["control_id"]: c for c in result["controls"]}
        assert by_id["A.4.2"]["coverage_status"] == "gap"
        assert by_id["A.4.3"]["coverage_status"] == "gap"
        assert by_id["A.4.4"]["coverage_status"] == "gap"
        assert result["coverage_percent"] == 0.0

    def test_a_register_without_treatment_fields_is_partial_not_covered(self):
        from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

        result = generate_iso42001_evidence_map(
            {"risk_register": [{"risk": "proxy discrimination"}]}
        )
        by_id = {c["control_id"]: c for c in result["controls"]}
        assert by_id["A.4.2"]["coverage_status"] == "covered"
        assert by_id["A.4.3"]["coverage_status"] == "partial"
        assert by_id["A.4.4"]["coverage_status"] == "partial"
        assert "only 0 carry a treatment plan" in by_id["A.4.3"]["evidence_description"]

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_full_wizard_with_real_entries_still_earns_its_coverage(self):
        from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

        result = generate_iso42001_evidence_map(_FULL_WIZARD)
        assert result["coverage_percent"] == 92.3
        assert (result["covered"], result["gaps"], result["manual"]) == (11, 0, 2)

    def test_control_a_measured_zero_is_evidence(self):
        """A parity of 0.0 or a monitoring flag of False is a MEASUREMENT, and
        refusing it would be the mirror defect: a real value discarded as if it
        were a placeholder.
        """
        from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

        wizard = {
            **_FULL_WIZARD,
            "metric_results": {"demographic_parity": 0.0},
            "monitoring_config": {"enabled": False},
        }
        result = generate_iso42001_evidence_map(wizard)
        assert result["coverage_percent"] == 92.3
        assert result["gaps"] == 0

    def test_control_an_empty_wizard_is_still_zero(self):
        from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

        result = generate_iso42001_evidence_map({})
        assert result["coverage_percent"] == 0.0
        assert (result["covered"], result["partial"], result["manual"]) == (0, 0, 2)


# ===========================================================================
# 7b. reporting.compliance.build_assurance_verdict: a PARTIAL scope limitation
# ===========================================================================


_ONE_CLEAN = {"attribute": "gender", "assessable": True, "gap": 0.01, "significant": False}
_FIVE_REFUSED = [
    {"attribute": a, "assessable": False, "reason": "groups too small"}
    for a in ("race", "age", "disability", "religion", "nationality")
]


class TestTheAssuranceVerdictNamesEveryUnassessableAttribute:
    """Measured on HEAD: 5 of 6 protected variables refused ("groups too
    small"), the sixth clean -> overall 'Unqualified', blocksDeployment False,
    unassessed [], "no material fairness defect found on the assessed
    attributes", byte-identical to a run where 'gender' was the only attribute
    requested. The scope-limitation machinery was fed only from the bias list.
    """

    def test_five_of_six_refused_is_a_scope_limitation(self):
        from vfairness.operations.reporting.compliance import build_assurance_verdict

        partial = build_assurance_verdict(per_variable=[_ONE_CLEAN] + _FIVE_REFUSED, has_truth=True)
        whole = build_assurance_verdict(per_variable=[_ONE_CLEAN], has_truth=True)

        assert (partial["overall"], partial["oneLineVerdict"]) != (
            whole["overall"],
            whole["oneLineVerdict"],
        )
        assert partial["overall"] == "Qualified"
        assert partial["blocksDeployment"] is False, "a scope limitation is not a defect"
        assert "scope limitation" in partial["oneLineVerdict"].lower()
        named = {u["attribute"] for u in partial["unassessed"]}
        assert named == {"race", "age", "disability", "religion", "nationality"}
        for attribute in named:
            assert attribute in partial["oneLineVerdict"]

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_wholly_assessable_clean_run_is_still_unqualified(self):
        from vfairness.operations.reporting.compliance import build_assurance_verdict

        verdict = build_assurance_verdict(per_variable=[_ONE_CLEAN], has_truth=True)
        assert verdict["overall"] == "Unqualified"
        assert verdict["unassessed"] == []
        assert verdict["findings"] == []
        assert verdict["oneLineVerdict"] == (
            "Unqualified opinion: no material fairness defect found on the assessed "
            "attributes. Keep monitoring."
        )

    def test_control_a_real_gap_is_still_adverse_and_blocks(self):
        from vfairness.operations.reporting.compliance import build_assurance_verdict

        verdict = build_assurance_verdict(
            per_variable=[
                {
                    "attribute": "race",
                    "assessable": True,
                    "gap": 0.23,
                    "significant": True,
                    "worstGroup": "Black",
                    "referenceGroup": "White",
                    "ciLow": 0.1,
                    "ciHigh": 0.3,
                }
            ],
            has_truth=True,
            domain="hiring",
            jurisdiction="US",
        )
        assert verdict["overall"] == "Adverse"
        assert verdict["blocksDeployment"] is True

    def test_control_nothing_assessable_is_still_a_disclaimer(self):
        from vfairness.operations.reporting.compliance import build_assurance_verdict

        verdict = build_assurance_verdict(per_variable=_FIVE_REFUSED[:1], has_truth=True)
        assert verdict["overall"] == "Disclaimer"
        assert "Insufficient assessable data" in verdict["oneLineVerdict"]

    def test_control_the_no_input_disclaimer_sentence_is_byte_identical(self):
        from vfairness.operations.reporting.compliance import build_assurance_verdict

        verdict = build_assurance_verdict()
        assert verdict["oneLineVerdict"] == (
            "Insufficient assessable data to issue a fairness opinion. Provide "
            "grouped protected attributes and, ideally, outcome labels."
        )


# ===========================================================================
# 7c. reporting.compliance.compute_adverse_action_reasons: the other direction
# ===========================================================================


_SHAP = {
    "debt_ratio": -0.40,
    "income": -0.30,
    "zip_code": -0.25,
    "credit_score": -0.20,
    "age_proxy": -0.15,
    "employment": -0.10,
}


class TestTheAdverseActionNoticeSeesWhatItWasNotGiven:
    """Measured on HEAD: debt_ratio is the strongest adverse factor at -0.40 and
    is attributed in shap_values, but absent from feature_names. Result: RC01
    income, RC02 credit_score, RC03 employment, RC04 zip_code (the proxy),
    complete True, unattributed [], zero warnings, although ``complete`` asserts
    "the ranking considered the whole model".
    """

    def test_a_notice_that_never_saw_the_strongest_driver_is_incomplete(self):
        from vfairness.operations.reporting.compliance import compute_adverse_action_reasons

        names = ["income", "zip_code", "credit_score", "age_proxy", "employment"]
        notice, messages = _caught(
            compute_adverse_action_reasons,
            _SHAP,
            names,
            ["zip_code", "age_proxy"],
        )
        assert notice.complete is False, (
            "the strongest attributed adverse factor was never in the ranking, yet "
            f"the notice reports complete=True with top reason {notice[0]['feature']!r}"
        )
        assert notice.adverse_attributions_not_ranked == ["debt_ratio"]
        assert any("never saw them" in m for m in messages), messages
        assert "INCOMPLETE" in repr(notice)

    def test_the_incompleteness_survives_a_json_boundary(self):
        """``json.dumps`` of a list subclass writes the ARRAY, so ``complete``
        did not cross the only serialisation the class documents: the
        nothing-examined result crossed as ``[]``, identical to a notice that
        found no adverse factor.
        """
        from vfairness.operations.reporting.compliance import compute_adverse_action_reasons

        notice, _ = _caught(compute_adverse_action_reasons, _SHAP, [], [])
        assert json.dumps(notice) == "[]", "the array shape is part of the contract"
        payload = json.loads(json.dumps(notice.to_dict()))
        assert payload["complete"] is False
        assert payload["features_examined"] == 0
        assert payload["reasons"] == []

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_whole_model_notice_is_still_complete(self):
        from vfairness.operations.reporting.compliance import compute_adverse_action_reasons

        notice, messages = _caught(
            compute_adverse_action_reasons,
            _SHAP,
            list(_SHAP),
            ["zip_code", "age_proxy"],
        )
        assert [e["feature"] for e in notice] == [
            "debt_ratio",
            "income",
            "credit_score",
            "employment",
        ]
        assert notice.complete is True
        assert notice.features_examined == 6
        assert notice.adverse_attributions_not_ranked == []
        assert messages == []
        assert notice.to_dict()["complete"] is True

    def test_control_a_non_adverse_attribution_left_out_is_not_a_defect(self):
        """A POSITIVE attribution could never have earned a reason code, so its
        absence from feature_names does not make the notice incomplete.
        """
        from vfairness.operations.reporting.compliance import compute_adverse_action_reasons

        notice, messages = _caught(
            compute_adverse_action_reasons,
            {**_SHAP, "tenure": 0.05},
            list(_SHAP),
            ["zip_code", "age_proxy"],
        )
        assert notice.complete is True
        assert messages == []

    def test_control_all_positive_shap_is_still_a_measured_empty_notice(self):
        from vfairness.operations.reporting.compliance import compute_adverse_action_reasons

        positive = {k: -v for k, v in _SHAP.items()}
        notice, messages = _caught(compute_adverse_action_reasons, positive, list(positive), [])
        assert list(notice) == []
        assert notice.complete is True, "a measured absence of adverse factors"
        assert messages == []


# ===========================================================================
# 7d. reporting.compliance.compute_signed_test_log: what the seal covers
# ===========================================================================


_SEALED_PAYLOAD = {
    "metrics": [{"name": "dp", "value": 0.02, "threshold": 0.1, "passed": True}],
    "overall_pass": True,
}


def _recompute_hash(log, interventions, reproducibility_info=None):
    """The documented recomputation. The reproducibility block joined the sealed
    payload on 2026-09-28 (BGL6 F06: replacing its data_hash left the seal
    verifying), so it is part of the recipe now and defaults to the block the
    document carries."""
    return hashlib.sha256(
        json.dumps(
            {
                "test_timestamp": log["test_timestamp"],
                "test_timestamp_source": log["test_timestamp_source"],
                "log_built_at": log["log_built_at"],
                "data_hash": log["data_hash"],
                "lib_version": log["library_version"],
                "metrics_snapshot": log["metrics_snapshot"],
                "summary": log["test_results_summary"],
                "intervention_history": interventions,
                "reproducibility_info": (
                    log["reproducibility_info"]
                    if reproducibility_info is None
                    else reproducibility_info
                ),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


class TestTheSignedTestLogSealsWhatItReports:
    """Measured on HEAD:

        declared test timestamp : 2024-01-15T09:00:00+00:00
        sealed  test_timestamp  : 2026-09-27T18:17:53.266974+00:00

    and the recomputed content_hash of the same document with
    intervention_history ERASED MATCHED the sealed hash, although the docstring
    calls the result "a tamper-evident record of the test execution".
    """

    def test_the_declared_test_time_is_the_sealed_one(self):
        from vfairness.operations.reporting.compliance import compute_signed_test_log

        log = compute_signed_test_log(
            {**_SEALED_PAYLOAD, "timestamp": "2024-01-15T09:00:00+00:00"},
            data_hash="abc123",
            lib_version="0.1.0",
        )
        assert log["test_timestamp"] == "2024-01-15T09:00:00+00:00", (
            f"declared 2024-01-15T09:00:00+00:00, sealed {log['test_timestamp']!r}"
        )
        assert log["test_timestamp_source"] == "declared_by_caller"
        assert log["log_built_at"] != log["test_timestamp"]

    def test_a_caller_that_declared_no_time_is_told_so(self):
        """Three states: the build time is still reported, and it no longer
        claims to be a measurement of when the test ran.
        """
        from vfairness.operations.reporting.compliance import compute_signed_test_log

        log = compute_signed_test_log(_SEALED_PAYLOAD, data_hash="abc123", lib_version="0.1.0")
        assert log["test_timestamp_source"] == "log_build_time"
        assert log["test_timestamp"] == log["log_built_at"]

    def test_erasing_the_intervention_history_breaks_the_seal(self):
        from vfairness.operations.reporting.compliance import compute_signed_test_log

        log = compute_signed_test_log(
            _SEALED_PAYLOAD,
            data_hash="abc123",
            lib_version="0.1.0",
            intervention_history=[{"intervention": "reweighting"}],
        )
        assert _recompute_hash(log, log["intervention_history"]) == log["content_hash"], (
            "the documented recomputation must reproduce the sealed hash"
        )
        assert _recompute_hash(log, []) != log["content_hash"], (
            "the hash of the document with intervention_history erased equals the "
            "sealed hash, so the seal does not cover it"
        )

    def test_altering_the_declared_timestamp_breaks_the_seal(self):
        from vfairness.operations.reporting.compliance import compute_signed_test_log

        log = compute_signed_test_log(
            {**_SEALED_PAYLOAD, "timestamp": "2024-01-15T09:00:00+00:00"},
            data_hash="abc123",
            lib_version="0.1.0",
            intervention_history=[],
        )
        tampered = {**log, "test_timestamp": "2026-01-01T00:00:00+00:00"}
        assert _recompute_hash(tampered, []) != log["content_hash"]

    # ---- OVER-CORRECTION CONTROL ----

    def test_control_the_three_state_metric_counts_are_unchanged(self):
        from vfairness.operations.reporting.compliance import compute_signed_test_log

        log = compute_signed_test_log(
            {"metrics": [{"name": "dp", "value": 0.4, "threshold": 0.1}]},
            "d" * 8,
            "0.1.0",
        )
        assert log["test_results_summary"] == {
            "total_metrics": 1,
            "passed": 0,
            "failed": 0,
            "not_assessed": 1,
            "derived": 0,
            "overall_pass": None,
        }
        assert len(log["content_hash"]) == 64


# ===========================================================================
# 8a. causal.graph: a graph nobody authored was not cleared
# ===========================================================================


class TestTheCausalGraphRefusesWhatWasNeverAuthored:
    """Measured on HEAD: a graph declaring gender=protected and hiring=outcome
    and NO edges returned has_direct_discrimination() False, severity 'info',
    assessable True, not_assessable [], with no warning. The five shapes the
    previous fix pinned all have a MISSING ROLE.
    """

    def _graph(self, edges):
        from vfairness.operations.causal.graph import CausalFairnessGraph

        g = CausalFairnessGraph()
        g.add_variable("gender", protected=True)
        g.add_variable("hiring", outcome=True)
        for a, b in edges:
            g.add_edge(a, b)
        return g

    @pytest.mark.parametrize(
        "edges,phrase",
        [
            ([], "no edges"),
            ([("gender", "hirng")], "no edge points INTO any declared outcome"),
            ([("hiring", "gender")], "no edge points INTO any declared outcome"),
        ],
        ids=["edgeless", "misspelt-edge-target", "edge-points-the-wrong-way"],
    )
    def test_the_verdict_is_none_and_the_reason_is_printed(self, edges, phrase):
        graph = self._graph(edges)
        verdict, messages = _caught(graph.has_direct_discrimination)
        assert verdict is None, (
            "no causal structure was authored, so the graph was not traced; "
            f"summary severity was {graph.summary()['severity']!r}"
        )
        summary, _ = _caught(graph.summary)
        assert summary["severity"] == "unknown"
        assert summary["assessable"] is False
        assert any(phrase in r for r in summary["not_assessable"]), summary["not_assessable"]
        assert messages, "a refusal with no warning on any channel"

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_traced_graph_with_no_pathway_keeps_its_clean_false(self):
        """False here is a real finding: the structure was authored, the outcome
        has a parent, and no pathway from the protected attribute exists.
        """
        from vfairness.operations.causal.graph import CausalFairnessGraph

        g = CausalFairnessGraph()
        g.add_variable("gender", protected=True)
        g.add_variable("hiring", outcome=True)
        g.add_variable("tenure")
        g.add_edge("tenure", "hiring")
        verdict, messages = _caught(g.has_direct_discrimination)
        assert verdict is False
        assert g.summary()["severity"] == "info"
        assert g.not_assessable() == []
        assert messages == []

    def test_control_a_real_structure_is_still_classified(self):
        from vfairness.operations.causal.graph import CausalFairnessGraph

        g = CausalFairnessGraph()
        g.add_variable("gender", protected=True)
        g.add_variable("education", mediator=True)
        g.add_variable("zip", proxy=True)
        g.add_variable("hiring", outcome=True)
        for a, b in (
            ("gender", "education"),
            ("education", "hiring"),
            ("gender", "hiring"),
            ("gender", "zip"),
            ("zip", "hiring"),
        ):
            g.add_edge(a, b)
        verdict, messages = _caught(g.has_direct_discrimination)
        assert verdict is True
        summary = g.summary()
        assert summary["counts"] == {"direct": 1, "indirect": 1, "proxy": 1}
        assert summary["severity"] == "high"
        assert messages == []


# ===========================================================================
# 8b. causal.task_handlers.load_dataset_from_payload: errors="replace"
# ===========================================================================


_LATIN1_CSV = "grp,y\n\xc4,1\n\xc5,0\n\xc4,1\n\xc5,0\n".encode("latin-1")


class TestTheDatasetLoaderWillNotCorruptTheGroupColumn:
    """Measured on HEAD: a latin-1 CSV (what Excel exports) whose group column
    holds the two distinct values 0xC4 and 0xC5 came back as ONE group, because
    both bytes decode to U+FFFD under ``errors="replace"``, with no warning. A
    gzip blob came back as a 1x1 DataFrame of mojibake, so the handlers'
    documented ``data is None`` refusal never fired.
    """

    def test_undecodable_bytes_are_refused_not_merged(self):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(load_dataset_from_payload, {"dataset_bytes": _LATIN1_CSV})
        assert df is None, f"two distinct group labels became {sorted(set(df['grp']))!r}"
        assert any("not valid utf-8" in m for m in messages), messages

    def test_a_blob_that_is_not_text_is_not_a_dataset(self):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(
            load_dataset_from_payload, {"dataset_bytes": gzip.compress(b"grp,y\na,1\n")}
        )
        assert df is None, f"returned a {getattr(df, 'shape', None)} frame"
        assert messages

    def test_two_datasets_in_one_payload_are_disclosed(self):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(
            load_dataset_from_payload,
            {
                "dataset_bytes": b"grp,y\na,1\nb,0\n",
                "dataset_ref": {"inline_csv": "grp,y\nc,1\n"},
            },
        )
        assert sorted(set(df["grp"])) == ["a", "b"], "precedence itself is unchanged"
        assert any("IGNORED" in m for m in messages), messages

    def test_a_storage_key_with_nothing_attached_says_which_step_did_not_run(self):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(
            load_dataset_from_payload,
            {"dataset_ref": {"storage_key": "k/1.csv", "assessment_id": "a"}},
        )
        assert df is None
        assert any("could-not-load" in m for m in messages), messages

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_declared_encoding_still_loads_the_two_groups(self):
        """The escape hatch: a caller that KNOWS the export is latin-1 gets the
        real data, with both groups intact.
        """
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(
            load_dataset_from_payload,
            {"dataset_bytes": _LATIN1_CSV, "dataset_encoding": "latin-1"},
        )
        assert df is not None and df.shape == (4, 2)
        assert sorted(set(df["grp"])) == ["\xc4", "\xc5"]
        assert messages == []

    def test_control_utf8_csv_json_and_text_all_still_load(self):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        cases = [
            {"dataset_bytes": "grp,y\nA,1\nB,0\n".encode("utf-8")},
            {"dataset_bytes": "grp,y\n\xc4,1\n\xc5,0\n".encode("utf-8")},
            {"dataset_bytes": "grp,y\nA,1\nB,0\n"},
            {"dataset_ref": {"inline_csv": "grp,y\nA,1\nB,0\n"}},
            {"dataset_ref": {"inline_json": [{"grp": "A", "y": 1}, {"grp": "B", "y": 0}]}},
        ]
        for payload in cases:
            df, messages = _caught(load_dataset_from_payload, payload)
            assert df is not None, payload
            assert df.shape[1] == 2 and len(df) == 2, (payload, getattr(df, "shape", None))
            assert messages == [], (payload, messages)


# ===========================================================================
# 8c. causal.task_handlers: a bytearray is not `bytes`, BGL6
# ===========================================================================


class TestEveryBufferTypeTakesTheSameStrictDecode:
    """BGL6 AUDIT, 2026-09-29. ``_decode_dataset_bytes`` opened with
    ``if not isinstance(raw, bytes): return str(raw)``, and a ``bytearray`` and a
    ``memoryview`` are NOT ``bytes``, so both skipped the entire strict-decode
    guard pinned in the class above and were turned into their own repr. The
    consumer decrypts the stored file and attaches the result, which is exactly
    where a bytearray or a memoryview comes from. Measured before this fix, on
    the same latin-1 CSV:

        dataset_bytes=_LATIN1_CSV              -> None + 1 warning     (correct)
        dataset_bytes=bytearray(_LATIN1_CSV)   -> a (0, 4) frame whose columns are
            ["bytearray(b'grp", 'y\\\\n\\\\xc4', '1\\\\n\\\\xc5', "0\\\\n')"], ZERO warnings
        dataset_bytes=memoryview(_LATIN1_CSV)  -> a (0, 1) frame whose single column
            is '<memory at 0x11ec0c340>', ZERO warnings
        dataset_bytes=gzip.compress(...)       -> None + 1 warning     (correct)
        dataset_bytes=bytearray(gzip.compress(...)) -> a (0, 1) frame named with the
            repr of the compressed bytes, ZERO warnings
        dataset_bytes=bytearray(b"grp,y\\na,1\\nb,0\\n")  -> a (0, 4) frame of repr
            fragments, so a PERFECTLY VALID utf-8 CSV lost every row in silence

    ``_parse_csv_text`` could not catch any of it: shape[1] != 0 and the repr's
    \\x escapes are printable text, not control characters. So the handlers'
    documented ``data is None`` refusal never fired and a causal op ran on an
    empty frame of repr fragments.
    """

    @pytest.mark.parametrize("wrap", [bytearray, memoryview], ids=["bytearray", "memoryview"])
    def test_undecodable_bytes_are_refused_through_every_buffer_type(self, wrap):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(load_dataset_from_payload, {"dataset_bytes": wrap(_LATIN1_CSV)})
        assert df is None, (
            f"a {wrap.__name__} of the same latin-1 bytes returned a "
            f"{getattr(df, 'shape', None)} frame with columns {getattr(df, 'columns', None)}"
        )
        assert any("not valid utf-8" in m for m in messages), messages

    @pytest.mark.parametrize("wrap", [bytearray, memoryview], ids=["bytearray", "memoryview"])
    def test_a_compressed_blob_is_refused_through_every_buffer_type(self, wrap):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(
            load_dataset_from_payload,
            {"dataset_bytes": wrap(gzip.compress(b"grp,y\na,1\n"))},
        )
        assert df is None, f"returned a {getattr(df, 'shape', None)} frame"
        assert messages

    def test_a_header_with_no_rows_is_not_a_dataset(self):
        """The backstop, independent of the decode fix: a frame with columns and
        no observations is the shape every accidental repr arrived in, and it
        cannot support a causal estimate.
        """
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(
            load_dataset_from_payload, {"dataset_ref": {"inline_csv": "grp,y\n"}}
        )
        assert df is None, f"returned a {getattr(df, 'shape', None)} frame"
        assert any("NO ROWS" in m for m in messages), messages

    # ---- OVER-CORRECTION CONTROLS: the real row count ----

    @pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview], ids=["bytes", "ba", "mv"])
    def test_control_a_valid_csv_keeps_every_row_through_every_buffer_type(self, wrap):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(
            load_dataset_from_payload, {"dataset_bytes": wrap(b"grp,y\na,1\nb,0\n")}
        )
        assert df is not None and df.shape == (2, 2), getattr(df, "shape", None)
        assert list(df.columns) == ["grp", "y"]
        assert sorted(df["grp"]) == ["a", "b"]
        assert messages == [], messages

    @pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview], ids=["bytes", "ba", "mv"])
    def test_control_the_declared_encoding_escape_hatch_works_for_all_of_them(self, wrap):
        from vfairness.operations.causal.task_handlers import load_dataset_from_payload

        df, messages = _caught(
            load_dataset_from_payload,
            {"dataset_bytes": wrap(_LATIN1_CSV), "dataset_encoding": "latin-1"},
        )
        assert df is not None and df.shape == (4, 2), getattr(df, "shape", None)
        assert sorted(set(df["grp"])) == ["\xc4", "\xc5"]
        assert messages == []
