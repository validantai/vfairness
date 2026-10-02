"""Readiness wave 6: five published claims the code did not keep.

Same shape as the inert options in `test_readiness2_inert.py`, one step further
out: here the caller was told something in a SIGNATURE or a DOCSTRING, and the
code did something else without ever saying so. A documented parameter is a
promise; an accepted-and-ignored one is a promise reported back as kept.

    R6-1  GroupCalibrator.transform "Raises ValueError"   REMOVED + WARNS
    R6-2  GroundednessScorer.score(gold=...)              REFUSED (warning)
    R6-3  proxy_score(per_feature, total_disparity)       REMOVED
    R6-4  plot_pareto_frontier(point_color=...)           IMPLEMENTED
    R6-5  plot_correlation_heatmap(threshold_lines=...)   IMPLEMENTED

Every one was reproduced by execution first, and every pin here carries an
OVER-CORRECTION CONTROL, because a refusal that also refuses the legitimate call,
or a "now it works" that changed the picture for everyone, is worse than the
inert claim it replaced.

On the two rendering pins: a matplotlib figure that is not closed leaks state
into the next render, so two genuinely different outputs can hash the same for
the wrong reason. `_png` builds a FRESH figure and closes it, and each rendering
class carries a vacuity control showing that a parameter which IS wired changes
the bytes on this very harness, and that the same call twice does not.
"""

from __future__ import annotations

import hashlib
import io
import warnings

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# matplotlib is the optional [viz] extra, so the module must still import
# without it (the lowest-versions CI job installs no extras). Only the tests
# that draw are marked needs_matplotlib and skip; the rest still run.
try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
    plt = None
else:
    matplotlib.use("Agg")  # no display in CI; must precede pyplot
    import matplotlib.pyplot as plt
needs_matplotlib = pytest.mark.skipif(
    matplotlib is None, reason="needs the optional [viz] extra (matplotlib)"
)

from vfairness.post_processing.calibration.group_calibrator import (  # noqa: E402
    GroupCalibrator,
    IntersectionalCalibrator,
    UnknownGroupWarning,
)
from vfairness.post_processing.calibration.visualization import (  # noqa: E402
    plot_pareto_frontier,
)
from vfairness.preprocessing.feature_engineering import visualization as fe_viz  # noqa: E402
from vfairness.preprocessing.feature_engineering.correlation import (  # noqa: E402
    CORRELATION_THRESHOLD_MEDIUM,
    FeatureCorrelationMatrix,
)
from vfairness.validity.groundedness import (  # noqa: E402
    GOLD_REFUSAL_NOTE,
    GoldUnsupportedWarning,
    GroundednessScorer,
)
from vfairness.xai.decomposition import (  # noqa: E402
    lundberg_fairness_decomposition,
    proxy_score,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if plt is not None:
        plt.close("all")


def _png(plot_fn, **kwargs) -> str:
    """sha256 of one rendering, on a FRESH figure that is closed again."""
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        plot_fn(ax=ax, **kwargs)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=60)
    finally:
        plt.close(fig)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


# R6-1. The unknown group that was documented to raise, and did not


def _fitted_calibrator() -> tuple[GroupCalibrator, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 400
    groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y_prob = rng.uniform(0.05, 0.95, n)
    y_true = (rng.uniform(size=n) < y_prob).astype(int)
    calibrator = GroupCalibrator(method="isotonic", min_group_size=50)
    calibrator.fit(y_true, y_prob, groups)
    return calibrator, y_prob, groups


class TestUnknownGroupsAreAnnounced:
    """Measured before the fix: fit on A and B, transform with a 'Z' present,
    and the Z rows came back GLOBALLY calibrated with no exception and no
    warning, byte-identical to `global_calibrator_.transform` on the same
    inputs. The docstring promised `ValueError: If unknown groups encountered`.
    """

    def test_an_unknown_group_warns_and_names_itself(self):
        calibrator, _, _ = _fitted_calibrator()
        with pytest.warns(UnknownGroupWarning) as record:
            calibrator.transform(np.array([0.1, 0.5, 0.9]), np.array(["A", "B", "Z"]))
        message = str(record[0].message)
        assert "'Z'" in message, message
        assert "NOT group calibrated" in message, message

    def test_the_unknown_rows_really_are_only_globally_calibrated(self):
        """The warning must describe what happened, not just that something did."""
        calibrator, _, _ = _fitted_calibrator()
        probabilities = np.array([0.5, 0.9])
        with pytest.warns(UnknownGroupWarning):
            out = calibrator.transform(probabilities, np.array(["Z", "Z"]))
        assert calibrator.global_calibrator_ is not None
        assert np.allclose(out, calibrator.global_calibrator_.transform(probabilities))

    def test_a_caller_who_wants_the_documented_exception_can_have_one(self):
        calibrator, _, _ = _fitted_calibrator()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnknownGroupWarning)
            with pytest.raises(UnknownGroupWarning):
                calibrator.transform(np.array([0.5]), np.array(["Z"]))

    def test_the_intersectional_calibrator_says_it_too(self):
        """Same silent fallback lived in the sibling class in the same file, where
        an unseen intersection is the ORDINARY case rather than the exotic one."""
        rng = np.random.default_rng(1)
        n = 400
        frame = pd.DataFrame({"race": rng.choice(["a", "b"], n), "sex": rng.choice(["m", "f"], n)})
        y_prob = rng.uniform(0.05, 0.95, n)
        y_true = (rng.uniform(size=n) < y_prob).astype(int)
        calibrator = IntersectionalCalibrator(method="isotonic", min_group_size=20)
        calibrator.fit(y_true, y_prob, frame)
        with pytest.warns(UnknownGroupWarning, match="z_q"):
            calibrator.transform(
                np.array([0.4, 0.6]), pd.DataFrame({"race": ["a", "z"], "sex": ["m", "q"]})
            )


class TestControlTheKnownGroupsAreUntouched:
    """An over-correction here would be a warning on the ordinary call, or a
    change to the numbers. Neither is allowed."""

    def test_a_transform_of_known_groups_is_silent(self):
        calibrator, y_prob, groups = _fitted_calibrator()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # ANY warning fails this test
            calibrator.transform(y_prob, groups)

    def test_the_numbers_are_the_ones_measured_before_the_fix(self):
        calibrator, _, _ = _fitted_calibrator()
        with pytest.warns(UnknownGroupWarning):
            out = calibrator.transform(
                np.array([0.1, 0.5, 0.5, 0.9, 0.3]), np.array(["A", "B", "Z", "Z", "A"])
            )
        assert np.allclose(out, [0.04545455, 0.6, 0.57894737, 0.92592593, 0.24324324], atol=1e-7)

    def test_the_fallback_strategy_none_pass_through_is_not_an_unknown_group(self):
        """`fallback_strategy='none'` deliberately leaves a small group
        uncalibrated. That was the caller's choice at fit time and must stay
        silent, or the warning becomes noise nobody reads."""
        rng = np.random.default_rng(2)
        groups = np.array(["A"] * 200 + ["B"] * 10)
        y_prob = rng.uniform(0.05, 0.95, 210)
        y_true = (rng.uniform(size=210) < y_prob).astype(int)
        calibrator = GroupCalibrator(method="isotonic", min_group_size=50, fallback_strategy="none")
        calibrator.fit(y_true, y_prob, groups)
        assert calibrator.calibrators_["B"] is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            calibrator.transform(y_prob, groups)


# R6-2. The reference answer nothing read


class _StubJudge:
    """Returns a fixed verdict, so any change in the result comes from the code
    under test rather than from a model."""

    def judge_groundedness(self, *, answer, contexts, question, language):
        return {"groundedness": 0.5, "faithfulness": 0.5, "note": "stub"}


class TestGoldIsRefusedNotIgnored:
    """Measured before the fix: `gold=None`, `gold='Paris.'`, `gold='COMPLETELY
    WRONG GOLD ANSWER'` and `gold=12345` all returned value=0.5, faith=0.5,
    note='stub'. The name appeared exactly once in the module, in the signature.
    """

    def test_supplying_a_reference_answer_warns(self):
        scorer = GroundednessScorer(judge=_StubJudge())
        with pytest.warns(GoldUnsupportedWarning, match="VG-007"):
            scorer.score("an answer", ["a context"], gold="the reference answer")

    def test_the_refusal_is_recorded_on_the_result_not_only_in_a_warning(self):
        """A warning does not reach `handle_validity_run`'s JSON. The note does,
        and an unreadable refusal is the same defect one layer up."""
        scorer = GroundednessScorer(judge=_StubJudge())
        with pytest.warns(GoldUnsupportedWarning):
            result = scorer.score("an answer", ["a context"], gold="Paris")
        assert GOLD_REFUSAL_NOTE in result.note
        assert "stub" in result.note, "the rung's own note must survive"

    def test_every_return_path_carries_it(self):
        """Including the two that never reach a rung, which is where a partial
        fix would leave a result that looks like it used the reference."""
        scorer = GroundednessScorer()  # no rung wired -> the refusal rung
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            unwired = scorer.score("an answer", ["a context"], gold="Paris")
            empty = scorer.score("   ", ["a context"], gold="Paris")
        assert GOLD_REFUSAL_NOTE in unwired.note
        assert GOLD_REFUSAL_NOTE in empty.note

    def test_a_caller_who_wants_a_hard_stop_can_have_one(self):
        scorer = GroundednessScorer(judge=_StubJudge())
        with warnings.catch_warnings():
            warnings.simplefilter("error", GoldUnsupportedWarning)
            with pytest.raises(GoldUnsupportedWarning):
                scorer.score("an answer", ["a context"], gold="Paris")


class TestControlTheScorerIsOtherwiseUnchanged:
    def test_a_call_without_gold_is_silent_and_scores_as_before(self):
        scorer = GroundednessScorer(judge=_StubJudge())
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # ANY warning fails this test
            result = scorer.score("an answer", ["a context"])
        assert (result.value, result.faithfulness, result.note) == (0.5, 0.5, "stub")

    def test_the_groundedness_score_is_not_changed_by_the_reference_answer(self):
        """The refusal must not become a second, quieter way of altering a score."""
        scorer = GroundednessScorer(judge=_StubJudge())
        without = scorer.score("an answer", ["a context"])
        with pytest.warns(GoldUnsupportedWarning):
            with_gold = scorer.score("an answer", ["a context"], gold="anything at all")
        assert with_gold.value == without.value
        assert with_gold.faithfulness == without.faithfulness
        assert with_gold.available == without.available

    def test_the_batch_still_runs_when_a_record_carries_a_reference_answer(self):
        """THE reason this is a warning and not a raise. The documented
        `vfairness_validity_run` envelope accepts an optional `reference_answer`
        and forwards it to `score(gold=...)`. Measured with a raise in that
        position: the handler died on the first such record and took the VG-001
        measurement of every other record in the batch with it, uncaught."""
        from vfairness.operations.validity.task_handlers import handle_validity_run

        payload = {
            "records": [
                {
                    "answer": "Paris is the capital.",
                    "retrieved_context": ["The capital of France is Paris."],
                    "reference_answer": "Paris",
                },
                {
                    "answer": "Bern is the Swiss capital.",
                    "retrieved_context": ["Bern is the capital of Switzerland."],
                },
            ]
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            out = handle_validity_run(payload)
        assert out["success"] is True
        records = out["data"]["records"]
        assert out["data"]["record_count"] == 2
        assert GOLD_REFUSAL_NOTE in records[0]["note"]
        assert GOLD_REFUSAL_NOTE not in records[1]["note"], "only the record that sent one"


# R6-3. The argument that could not have been used


class TestTotalDisparityIsGone:
    """Measured before the removal: `0.55`, `-0.55`, `0.0`, `nan` and `1e9` as
    the second argument all returned the identical dict."""

    def test_the_keyword_is_refused_loudly(self):
        with pytest.raises(TypeError, match="total_disparity"):
            proxy_score({"a": 0.04, "b": 0.06}, total_disparity=0.10)

    def test_a_second_positional_argument_is_refused_loudly(self):
        """The removal must not be SILENT for the caller who passed it
        positionally, which was the only way the old signature allowed."""
        with pytest.raises(TypeError, match="positional"):
            proxy_score({"a": 0.04, "b": 0.06}, 0.10)

    def test_the_signature_holds_the_one_argument_the_formula_needs(self):
        import inspect

        # List what it HAS rather than asking whether one name is absent: a
        # getattr miss on a wrong name looks exactly like a real absence.
        assert list(inspect.signature(proxy_score).parameters) == ["per_feature"]


class TestControlTheDecompositionStillDecomposes:
    def test_the_documented_formula_is_unchanged(self):
        scores = proxy_score({"a": 0.04, "b": -0.06, "c": 0.0})
        assert pytest.approx(sum(scores.values()), abs=1e-9) == 1.0
        assert scores["b"] > scores["a"]
        assert scores["c"] == 0.0

    def test_the_zero_denominator_is_still_handled(self):
        assert proxy_score({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}

    def test_the_spine_still_runs_and_still_asserts_its_identity(self):
        """The only in-library call site. If the removal had been done wrong,
        this raises TypeError before the 1e-6 identity is ever checked."""
        rng = np.random.default_rng(3)
        shap_values = rng.normal(size=(200, 3))
        shap_values[:100, 0] += 0.4
        decomposition = lundberg_fairness_decomposition(
            shap_values=shap_values,
            group_labels=np.array(["m"] * 100 + ["f"] * 100),
            feature_names=["x", "y", "z"],
            metric="demographic_parity",
            protected_attribute="gender",
            subject_id="s",
            audit_artifact_id="a",
        )
        decomposition.assert_identity()
        assert pytest.approx(sum(decomposition.proxy_scores.values()), abs=1e-9) == 1.0
        assert "x" in decomposition.flagged_proxies


# R6-4. The colour that coloured nothing

_PARETO_CAL = [0.05, 0.08, 0.10, 0.12, 0.15, 0.09, 0.13]
_PARETO_FAIR = [0.20, 0.12, 0.08, 0.06, 0.05, 0.18, 0.19]


def _pareto(**kwargs) -> str:
    return _png(
        plot_pareto_frontier,
        calibration_errors=_PARETO_CAL,
        fairness_violations=_PARETO_FAIR,
        **kwargs,
    )


class TestPointColorReachesThePoints:
    """Measured before the fix: '#1f77b4', '#00ff00' and 'magenta' produced
    byte-identical PNGs, because the non-frontier scatter was hardcoded to
    'gray'."""

    @needs_matplotlib
    def test_two_colours_render_differently(self):
        assert _pareto(point_color="#00ff00") != _pareto(point_color="magenta")

    @needs_matplotlib
    def test_the_default_is_the_colour_the_signature_documents(self):
        assert _pareto() == _pareto(point_color="#1f77b4")

    @needs_matplotlib
    def test_the_old_hardcoded_colour_is_still_reachable(self):
        """The default picture changed with this fix (gray -> #1f77b4), so the
        old look must remain available by name rather than be lost."""
        assert _pareto(point_color="gray") != _pareto()


class TestControlTheParetoHarnessIsNotVacuous:
    @needs_matplotlib
    def test_the_same_call_twice_hashes_the_same(self):
        assert _pareto() == _pareto()

    @needs_matplotlib
    def test_a_parameter_that_was_already_wired_changes_the_bytes(self):
        """If this fails, the harness cannot see ANY colour change and the
        pins above prove nothing."""
        assert _pareto(frontier_color="#00ff00") != _pareto(frontier_color="#d62728")

    @needs_matplotlib
    def test_the_frontier_itself_is_untouched(self):
        """point_color must colour the dominated points ONLY."""
        assert _pareto(point_color="magenta") != _pareto(
            point_color="magenta", frontier_color="#00ff00"
        )


# R6-5. The threshold markers that were never drawn

_FEATURES = ["income", "zipcode", "age", "tenure"]
_ATTRIBUTES = ["race", "gender"]


@pytest.fixture
def correlations() -> FeatureCorrelationMatrix:
    values = pd.DataFrame(
        [[0.62, 0.05], [0.41, 0.12], [0.09, 0.31], [0.28, np.nan]],
        index=_FEATURES,
        columns=_ATTRIBUTES,
    )
    return FeatureCorrelationMatrix(
        correlations=values,
        pvalues=pd.DataFrame(np.full((4, 2), 0.01), index=_FEATURES, columns=_ATTRIBUTES),
        feature_names=_FEATURES,
        protected_attributes=_ATTRIBUTES,
        method="pearson",
    )


def _heatmap(matrix, **kwargs) -> str:
    return _png(fe_viz.plot_correlation_heatmap, correlation_matrix=matrix, **kwargs)


def _marker_pixels(matrix, **kwargs) -> int:
    """Pixels painted in exactly THRESHOLD_MARKER_COLOR by one render.

    A Rectangle on `ax.patches` is not yet a mark on the picture. This reads the
    raster, so a marker that is added and then not drawn (wrong zorder, zero
    linewidth, clipped away) still fails.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        fe_viz.plot_correlation_heatmap(matrix, ax=ax, **kwargs)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
    finally:
        plt.close(fig)
    target = np.array(
        [int(fe_viz.THRESHOLD_MARKER_COLOR[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8
    )
    return int(np.all(rgba[:, :, :3] == target, axis=-1).sum())


def _outlines(matrix, **kwargs):
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        fe_viz.plot_correlation_heatmap(matrix, ax=ax, **kwargs)
        return sorted(
            (round(patch.get_x(), 3), round(patch.get_y(), 3))
            for patch in ax.patches
            if isinstance(patch, matplotlib.patches.Rectangle) and not patch.get_fill()
        )
    finally:
        plt.close(fig)


class TestThresholdMarkersAreDrawn:
    """Measured before the fix: True and False produced byte-identical PNGs.

    The byte-hash test below is NOT on its own proof that the markers are drawn:
    `threshold_lines` also names the threshold on the colour bar, and a sabotage
    that removed the outlines while keeping that label left the hash test GREEN
    (measured 2026-09-10, while the two geometry pins went red). That is why the
    outlines are pinned by coordinate and by PAINTED PIXELS as well.
    """

    @needs_matplotlib
    def test_turning_them_off_changes_the_picture(self, correlations):
        assert _heatmap(correlations, threshold_lines=True) != _heatmap(
            correlations, threshold_lines=False
        )

    @needs_matplotlib
    def test_the_outlines_are_painted_not_merely_attached(self, correlations):
        """~1400 pixels of the marker colour with the markers on, ~14 without.
        Not zero without: text antialiasing lands on that colour a few times,
        which is why this is an order-of-magnitude pin and not an equality."""
        on = _marker_pixels(correlations, threshold_lines=True)
        off = _marker_pixels(correlations, threshold_lines=False)
        assert on > 20 * max(off, 1), f"markers on={on} off={off}"

    @needs_matplotlib
    def test_the_default_is_on_as_documented(self, correlations):
        assert _heatmap(correlations) == _heatmap(correlations, threshold_lines=True)

    @needs_matplotlib
    def test_exactly_the_pairs_get_high_correlations_reports_are_marked(self, correlations):
        """The picture and the number must not be able to disagree: same
        threshold, same set. (income, race), (zipcode, race), (age, gender)
        cross 0.3; (tenure, race) at 0.28 and the NaN cell do not."""
        expected = correlations.get_high_correlations(CORRELATION_THRESHOLD_MEDIUM)
        marked = _outlines(correlations, threshold_lines=True)
        assert len(marked) == len(expected) == 3
        # seaborn/pcolormesh: cell (row i, col j) has its corner at (j, i).
        assert marked == [(0.0, 0.0), (0.0, 1.0), (1.0, 2.0)]

    @needs_matplotlib
    def test_nothing_is_marked_when_they_are_off(self, correlations):
        assert _outlines(correlations, threshold_lines=False) == []

    @needs_matplotlib
    def test_the_matplotlib_only_branch_marks_the_same_cells(self, correlations, monkeypatch):
        """The two branches use different cell geometry (imshow centres cells on
        integer coordinates, pcolormesh does not), so an offset that is right in
        one is half a cell wrong in the other."""
        monkeypatch.setattr(fe_viz, "HAS_SEABORN", False)
        assert _heatmap(correlations, threshold_lines=True) != _heatmap(
            correlations, threshold_lines=False
        )
        assert _outlines(correlations, threshold_lines=True) == [
            (-0.5, -0.5),
            (-0.5, 0.5),
            (0.5, 1.5),
        ]


class TestControlTheHeatmapHarnessIsNotVacuous:
    @needs_matplotlib
    def test_the_same_call_twice_hashes_the_same(self, correlations):
        assert _heatmap(correlations) == _heatmap(correlations)

    @needs_matplotlib
    def test_a_parameter_that_was_already_wired_changes_the_bytes(self, correlations):
        assert _heatmap(correlations, annotate=True) != _heatmap(correlations, annotate=False)

    @needs_matplotlib
    def test_the_markers_do_not_move_the_axes(self, correlations):
        """Adding patches must not rescale the heatmap under the reader."""
        limits = []
        for flag in (True, False):
            fig, ax = plt.subplots(figsize=(6, 4))
            fe_viz.plot_correlation_heatmap(correlations, ax=ax, threshold_lines=flag)
            limits.append((ax.get_xlim(), ax.get_ylim()))
            plt.close(fig)
        assert limits[0] == limits[1]
