"""The public parameter boundary refuses what the library cannot honour.

Three findings from the 2026-08-28 pre-release audit meet here, and they are one
defect wearing three hats: a parameter was ACCEPTED, and then the thing it asked
for did not happen.

* ``backend='fairlearn'`` computed the NATIVE numbers and printed
  ``backend='fairlearn'`` in the repr that lands in a notebook or an audit log.
  Any string at all was accepted. No delegation code exists anywhere in the
  package: ``self._backend`` was read by exactly one place, ``__repr__``.
* ``confidence_level`` was never range-checked at any public entry point. 1.0
  turned into 4,000,000 permutations (a silent, unbounded hang); 0.0 and -0.2
  returned a collapsed interval AND ``is_significant=True``, which turns every
  result into a definite call; 95 surfaced as a raw numpy percentile error.
* ``task_type='ranking'`` is documented on the published API reference and
  refused by the runtime. Ranking fairness IS implemented, as standalone
  functions, so the refusal now says where to go.

Each test below is paired with a control that the healthy input is untouched.
"""

import inspect
import time
import typing

import numpy as np
import pytest

from vfairness import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics import _statistics as stats_mod
from vfairness.evaluation.vfairness_metrics._statistics import (
    MAX_PERMUTATIONS,
    _permutation_count,
    validate_confidence_level,
)
from vfairness.evaluation.vfairness_metrics._validation import VALID_TASK_TYPES
from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference_with_ci,
)
from vfairness.exceptions import ConfigurationError

N = 240


def _data():
    rng = np.random.default_rng(11)
    groups = np.array(["a", "b", "c"] * (N // 3))
    y_true = rng.integers(0, 2, N)
    y_pred = (rng.random(N) < np.where(groups == "a", 0.62, 0.34)).astype(int)
    return y_true, y_pred, groups


# The frozen `backend` parameter


class TestBackendNamesWhatActuallyRan:
    def test_the_two_implemented_backends_are_accepted(self):
        y_true, y_pred, groups = _data()

        for backend in ("auto", "native"):
            analyzer = FairnessAnalyzer(y_true, y_pred, groups, backend=backend)
            assert analyzer._backend == "native"
            assert "backend='native'" in repr(analyzer)

    def test_the_repr_reports_the_backend_that_ran_not_the_one_requested(self):
        """``backend='auto'`` and ``backend='native'`` are indistinguishable in
        the output, because they are indistinguishable in the computation."""
        y_true, y_pred, groups = _data()

        auto = repr(FairnessAnalyzer(y_true, y_pred, groups, backend="auto"))
        native = repr(FairnessAnalyzer(y_true, y_pred, groups, backend="native"))

        assert auto == native

    @pytest.mark.parametrize("backend", ["fairlearn", "aequitas"])
    def test_a_named_but_unimplemented_backend_is_refused(self, backend):
        y_true, y_pred, groups = _data()

        with pytest.raises(ConfigurationError) as excinfo:
            FairnessAnalyzer(y_true, y_pred, groups, backend=backend)

        message = str(excinfo.value)
        assert backend in message
        assert "not implemented" in message

    @pytest.mark.parametrize("backend", ["totally-made-up", "FAIRLEARN", "", 42, None])
    def test_an_unrecognised_backend_is_refused_rather_than_echoed(self, backend):
        """It used to be accepted verbatim, including 42 and None, and then
        printed in the repr as if it had computed something."""
        y_true, y_pred, groups = _data()

        with pytest.raises(ConfigurationError):
            FairnessAnalyzer(y_true, y_pred, groups, backend=backend)

    def test_the_advertised_backend_list_holds_only_backends_that_exist(self):
        assert FairnessAnalyzer.BACKENDS == ["native"]

    def test_the_type_annotation_advertises_exactly_what_the_runtime_accepts(self):
        """A Literal listing a value that raises is the same lie in static form."""
        annotation = inspect.signature(FairnessAnalyzer.__init__).parameters["backend"].annotation

        assert set(typing.get_args(annotation)) == {"auto", "native"}

    def test_the_class_docstring_no_longer_promises_delegation(self):
        doc = FairnessAnalyzer.__doc__ or ""

        assert "delegate to Fairlearn" not in doc
        assert "Aequitas" not in doc

    def test_no_computation_reads_the_backend(self):
        """The control for the refusal above: 'native' and the default produce
        identical metrics, so nothing about the numbers changed."""
        y_true, y_pred, groups = _data()

        default = FairnessAnalyzer(y_true, y_pred, groups).compute_all_metrics()
        explicit = FairnessAnalyzer(y_true, y_pred, groups, backend="native").compute_all_metrics()

        assert default == explicit


# confidence_level


BAD_LEVELS = [1.0, 0.0, -0.2, 95, 1.5, float("nan"), float("inf"), "0.95", None]


class TestConfidenceLevelIsRangeChecked:
    @pytest.mark.parametrize("level", BAD_LEVELS)
    def test_the_metric_entry_point_refuses_it(self, level):
        y_true, y_pred, groups = _data()

        with pytest.raises(ConfigurationError):
            demographic_parity_difference_with_ci(
                y_true, y_pred, groups, n_bootstrap=40, confidence_level=level, random_state=1
            )

    @pytest.mark.parametrize("level", BAD_LEVELS)
    def test_the_analyzer_report_refuses_it(self, level):
        y_true, y_pred, groups = _data()
        analyzer = FairnessAnalyzer(y_true, y_pred, groups)

        with pytest.raises(ConfigurationError):
            analyzer.get_report(include_ci=True, n_bootstrap=40, confidence_level=level)

    @pytest.mark.parametrize("level", BAD_LEVELS)
    def test_it_is_refused_even_when_no_interval_is_requested(self, level):
        """include_ci=False used to make the value dead, so a typo travelled
        undetected into the cache key and into the next call that DID use it."""
        y_true, y_pred, groups = _data()
        analyzer = FairnessAnalyzer(y_true, y_pred, groups, cache=True)

        with pytest.raises(ConfigurationError):
            analyzer.compute_all_metrics(include_ci=False, confidence_level=level)

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                lambda level: stats_mod.bootstrap_ci(
                    np.arange(50, dtype=float), np.mean, n_bootstrap=20, confidence_level=level
                ),
                id="bootstrap_ci",
            ),
            pytest.param(
                lambda level: stats_mod.stratified_bootstrap_ci(
                    np.arange(50, dtype=float),
                    np.array(["a", "b"] * 25),
                    lambda d, g: float(np.mean(d)),
                    n_bootstrap=20,
                    confidence_level=level,
                ),
                id="stratified_bootstrap_ci",
            ),
            pytest.param(
                lambda level: stats_mod.bayesian_proportion_ci(5, 20, confidence_level=level),
                id="bayesian_proportion_ci",
            ),
            pytest.param(
                lambda level: stats_mod.bayesian_difference_ci(
                    5, 20, 7, 20, confidence_level=level, n_samples=100
                ),
                id="bayesian_difference_ci",
            ),
            pytest.param(
                lambda level: stats_mod.wilson_score_interval(5, 20, confidence_level=level),
                id="wilson_score_interval",
            ),
            pytest.param(
                lambda level: stats_mod.bayesian_mean_ci(
                    np.arange(30, dtype=float), confidence_level=level, n_samples=100
                ),
                id="bayesian_mean_ci",
            ),
            pytest.param(
                lambda level: stats_mod.empirical_likelihood_ci(5, 20, confidence_level=level),
                id="empirical_likelihood_ci",
            ),
            pytest.param(
                lambda level: stats_mod.simultaneous_disparity_bounds(
                    {"a": (5, 20), "b": (9, 20)}, confidence_level=level
                ),
                id="simultaneous_disparity_bounds",
            ),
        ],
    )
    def test_every_interval_entry_point_refuses_an_out_of_range_level(self, call):
        with pytest.raises(ConfigurationError):
            call(1.0)
        with pytest.raises(ConfigurationError):
            call(95)

    def test_the_error_names_the_common_95_versus_0_95_slip(self):
        with pytest.raises(ConfigurationError) as excinfo:
            validate_confidence_level(95)

        assert "0.95" in str(excinfo.value)

    def test_one_point_zero_is_refused_immediately_instead_of_hanging(self):
        """It used to reach ``n_perm = 4/max(alpha, 1e-6)`` = 4,000,000 full
        metric recomputations, with no warning and no progress. The audit killed
        it after 100 seconds on 600 rows."""
        y_true, y_pred, groups = _data()

        started = time.monotonic()
        with pytest.raises(ConfigurationError):
            demographic_parity_difference_with_ci(
                y_true, y_pred, groups, n_bootstrap=40, confidence_level=1.0, random_state=1
            )
        assert time.monotonic() - started < 5.0

    def test_a_collapsed_interval_can_no_longer_be_produced(self):
        """0.0 and -0.2 returned a zero-width interval with is_significant=True.
        The verdict is read from the whole interval, so a point is a verdict
        nobody computed."""
        y_true, y_pred, groups = _data()

        for level in (0.0, -0.2):
            with pytest.raises(ConfigurationError):
                demographic_parity_difference_with_ci(
                    y_true, y_pred, groups, n_bootstrap=40, confidence_level=level, random_state=1
                )


class TestThePermutationBudgetIsBounded:
    def test_an_ordinary_level_is_unchanged(self):
        """The control. 0.95 asked for 199 permutations before the cap and asks
        for 199 now; every level down to 2e-4 alpha is likewise untouched."""
        assert _permutation_count(0.05) == (199, False)
        assert _permutation_count(0.01) == (400, False)
        assert _permutation_count(0.001) == (4000, False)
        assert _permutation_count(2e-4) == (MAX_PERMUTATIONS, False)

    def test_an_extreme_but_legal_level_is_capped(self):
        assert _permutation_count(1e-4) == (MAX_PERMUTATIONS, True)
        assert _permutation_count(1e-5) == (MAX_PERMUTATIONS, True)
        assert _permutation_count(0.0) == (MAX_PERMUTATIONS, True)

    def test_a_capped_run_says_so_in_the_result_metadata(self):
        """Could-not-check, not silently-absorbed: the gate ran at lower
        resolution than the design asks for, and the record carries the floor so
        a reader can compare it to alpha/2 themselves."""
        y_true, y_pred, groups = _data()

        result = demographic_parity_difference_with_ci(
            y_true, y_pred, groups, n_bootstrap=60, confidence_level=0.99999, random_state=5
        )

        assert result.metadata.get("permutation_resolution_limited") is True
        assert result.metadata["n_permutation"] <= MAX_PERMUTATIONS
        assert result.metadata["min_resolvable_p_value"] > 0

    def test_an_ordinary_run_carries_no_such_flag(self):
        y_true, y_pred, groups = _data()

        result = demographic_parity_difference_with_ci(
            y_true, y_pred, groups, n_bootstrap=60, confidence_level=0.95, random_state=5
        )

        assert "permutation_resolution_limited" not in result.metadata
        assert result.metadata["n_permutation"] == 199


class TestHealthyConfidenceLevelsAreUnchanged:
    @pytest.mark.parametrize("level", [0.5, 0.9, 0.95, 0.99, 0.999])
    def test_a_legal_level_still_produces_a_real_interval(self, level):
        y_true, y_pred, groups = _data()

        result = demographic_parity_difference_with_ci(
            y_true, y_pred, groups, n_bootstrap=60, confidence_level=level, random_state=7
        )

        assert result.confidence_level == level
        assert result.lower_bound <= result.point_estimate <= result.upper_bound
        assert result.upper_bound > result.lower_bound

    def test_a_wider_level_gives_a_wider_interval(self):
        y_true, y_pred, groups = _data()

        def width(level):
            r = demographic_parity_difference_with_ci(
                y_true, y_pred, groups, n_bootstrap=200, confidence_level=level, random_state=7
            )
            return r.upper_bound - r.lower_bound

        assert width(0.99) >= width(0.90)

    def test_validate_returns_the_value_as_a_float(self):
        assert validate_confidence_level(0.95) == 0.95
        assert isinstance(validate_confidence_level(np.float64(0.9)), float)


# task_type='ranking'


class TestRankingIsNotAnAnalyzerTaskType:
    def test_it_is_refused_and_points_at_the_functions_that_do_it(self):
        y_true, y_pred, groups = _data()

        with pytest.raises(ConfigurationError) as excinfo:
            FairnessAnalyzer(y_true, y_pred, groups, task_type="ranking")

        message = str(excinfo.value)
        assert "exposure_parity_difference" in message
        assert "rankings, groups" in message

    def test_the_hint_does_not_widen_the_accepted_set(self):
        assert VALID_TASK_TYPES == ("classification", "regression")

    @pytest.mark.parametrize("task_type", ["Ranking", "RANKING", "classifcation", ["ranking"]])
    def test_every_other_unrecognised_task_type_still_refuses_plainly(self, task_type):
        """Including an UNHASHABLE one: the hint lookup must not turn a refusal
        into a TypeError."""
        y_true, y_pred, groups = _data()

        with pytest.raises(ConfigurationError):
            FairnessAnalyzer(y_true, y_pred, groups, task_type=task_type)

    @pytest.mark.parametrize("task_type", ["classification", "regression"])
    def test_the_supported_task_types_are_unaffected(self, task_type):
        y_true, y_pred, groups = _data()

        analyzer = FairnessAnalyzer(y_true, y_pred, groups, task_type=task_type)

        assert analyzer.task_type == task_type
