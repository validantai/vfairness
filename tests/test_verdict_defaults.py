"""Wave 3: no verdict surface may DEFAULT a metric's direction, and nothing that
was never measured may be reassured about.

Three waves have now shipped a fix for one bug class: the better-direction of a
metric being GUESSED instead of resolved, so a maximal violation reads as a pass.
Wave 1 fixed the deployment gate, wave 2 fixed the shared helper we had just
written with the same flaw, and acceptance found it a third time as a latent
DEFAULT: ``analyzer._fairness_verdict`` accepted ``metric_name=None`` and graded
that call as lower-is-better. Every in-repo caller happened to name its metric,
so nothing was mis-graded in production, but the next caller to pass a ratio
without naming it would have had its verdict inverted, silently, exactly as
before.

The reason the class keeps coming back is that the repo's reintroduction guard
only ever looked at ONE file (``tests/test_metric_direction.py`` scans
``_metric_direction.py`` and nothing else), while each recurrence appeared in a
DIFFERENT file. This module widens that guard to the whole ``src/vfairness``
package, reusing the one shared executable-source scanner rather than copying
it, and adds a signature guard for the default itself, which no source scan for
substrings could ever see.

It also pins the two remaining "reassured about what was never measured"
surfaces found in the same pass:

* the explainer attached the recommendation written for a PASSING metric
  ("Continue monitoring this metric ... Document your fairness practices") to a
  metric whose own evaluation line said "Unable to compute due to insufficient
  data";
* the detailed-report SVG adapter marked a could-not-check row ``passed=False``,
  which is the representation of a MEASURED failure, and signalled the third
  state only by a prose prefix a reader could reword away.

And it pins the methodology document against the version every report stamps, so
a report can never again cite a methodology version the authored document does
not describe.

WAVE 4 (2026-08-27) added three things here.

* ``_fairness_verdict`` contradicted ITSELF on sign. Its point-estimate branch
  read a lower-is-better metric as a violation MAGNITUDE (``abs(value) <=
  threshold``) while its interval branch compared the RAW bounds, so a CI of
  ``[-0.50, -0.40]`` against a 0.10 bound returned 'fair' and its mirror
  ``[+0.40, +0.50]`` returned 'unfair'. Both branches now use the symmetric
  band, and both signs are pinned.
* The same unbreachable-bound hole that ``check_threshold`` had was open here: a
  required minimum of 0.0 on a ratio placed every interval in the fair band.
* The signature pin was written for ONE helper. It now covers every helper that
  resolves a direction from a metric name, plus an AST sweep of the package that
  finds the helper somebody adds tomorrow. A default is invisible to any source
  scan, which is why that recurrence survived two waves of guard widening.

The copied marker list that used to live in this module is gone: it was a
second, weaker twin of the detector in ``tests/test_audit_final_release.py``,
and the two disagreed about what "the scanner" catches. There is one detector,
and this module calls it.
"""

import pathlib
import re
import warnings

import numpy as np
import pytest

from vfairness._methodology import METHODOLOGY_VERSION
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    check_threshold,
    improvement_amount,
    is_ratio_metric,
    metric_direction,
    relax_threshold,
)
from vfairness.evaluation.vfairness_metrics.analyzer import (
    FairnessAnalyzer,
    _fairness_verdict,
)
from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner
from vfairness.rendering.adapters_fairness import _metric_state

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "vfairness"

_PASSING_METRIC_RECOMMENDATION = "Continue monitoring this metric"


# ── 1. the direction may not be defaulted ──────────────────────────────────


class TestMetricNameIsRequiredForAVerdict:
    """``_fairness_verdict`` must be told which metric it is grading."""

    def test_omitting_the_metric_name_is_an_error(self):
        """The four-argument form silently assumed lower-is-better. It is gone.

        This is the whole point of the change: an unnamed call used to be
        answered, in a direction nobody had checked. Now it cannot be made.
        """
        with pytest.raises(TypeError):
            _fairness_verdict(0.02, 0.01, 0.04, 0.10)

    def test_the_default_cannot_come_back_through_the_signature(self):
        """Signature guard: no source scan for substrings can see a DEFAULT.

        The substring scanner below reads executable text; a parameter default
        is invisible to it. This is the pin that actually watches the third
        recurrence, so it is asserted directly on the signature object.
        """
        import inspect

        param = inspect.signature(_fairness_verdict).parameters["metric_name"]
        assert param.default is inspect.Parameter.empty, (
            "metric_name has a default again. A default is a guessed direction: "
            "the previous one graded every unnamed call as lower-is-better, "
            "which inverts the verdict for the whole ratio family."
        )

    def test_a_maximal_ratio_violation_can_no_longer_be_graded_unnamed(self):
        """The exact call that used to return 'fair' for a 0.05 four-fifths ratio."""
        with pytest.raises(TypeError):
            _fairness_verdict(0.05, 0.01, 0.10, 0.80)
        # Named, it is what it always should have been.
        assert (
            _fairness_verdict(0.05, 0.01, 0.10, 0.80, metric_name="disparate_impact_ratio")
            == "unfair"
        )

    def test_an_explicit_none_still_fails_closed(self):
        """A caller that passes None on purpose gets could-not-check, never 'fair'."""
        assert (
            _fairness_verdict(0.02, 0.01, 0.04, 0.10, metric_name=None) == "insufficient_evidence"
        )

    def test_an_unknown_named_metric_fails_closed(self):
        """Unchanged, and re-pinned here because it is the fallback that matters."""
        assert (
            _fairness_verdict(0.01, 0.00, 0.02, 0.10, metric_name="some_bespoke_client_metric")
            == "insufficient_evidence"
        )

    @pytest.mark.parametrize(
        "helper",
        [
            _fairness_verdict,
            metric_direction,
            is_ratio_metric,
            check_threshold,
            relax_threshold,
            improvement_amount,
            _metric_state,
        ],
    )
    def test_no_direction_helper_defaults_the_metric_it_is_grading(self, helper):
        """Wave 4: the signature pin, extended past the one helper that had the bug.

        A parameter default is invisible to any source scan, which is why the
        third recurrence of this class survived two waves of guard widening.
        The pin was written for ``_fairness_verdict`` alone; every helper that
        resolves a direction from a metric name has the same hole available to
        it, so each one is asserted directly on its signature object.
        """
        import inspect

        params = inspect.signature(helper).parameters
        offenders = {
            n: p.default
            for n, p in params.items()
            if _names_a_metric(n) and p.default is not inspect.Parameter.empty
        }
        assert not offenders, (
            f"{helper.__name__} defaults the metric name it grades: {offenders}. "
            "A default is a guessed direction, and a guessed direction inverts "
            "the verdict for the whole ratio family."
        )


def _names_a_metric(identifier):
    """True for a parameter that carries the METRIC NAME a direction is read from.

    Deliberately narrow. ``title``, ``x_label`` and ``y_label`` are chart text
    and legitimately default; ``metric_name``, ``metric_key``, ``name`` and
    ``key`` on a direction-deciding helper are the thing whose direction is
    about to be resolved, and defaulting one of those is the defect.
    """
    parts = [
        p
        for p in re.split(r"[^a-z0-9]+", re.sub(r"(?<!^)(?=[A-Z])", "_", str(identifier)).lower())
        if p
    ]
    return "metric" in parts or str(identifier).strip("_").lower() in {"name", "key", "mname"}


# Names whose presence in a function body means that function DECIDES A
# DIRECTION, and so must never be handed a defaulted metric name.
_DIRECTION_MARKERS = (
    "metric_direction",
    "MetricDirection",
    "check_threshold",
    "is_ratio_metric",
    "_fairness_verdict",
    "relax_threshold",
    "improvement_amount",
    "higher_is_better",
    "lower_is_better",
)


def _defaulted_metric_name_params(path):
    """``(function, parameter, default)`` for every direction helper in ``path``.

    The sweep that catches the NEXT helper. The parametrized signature test
    above names the helpers we know; this one finds the one somebody adds
    tomorrow, and it is an AST scan rather than a text scan because a default
    is a tree node with no reliable spelling.
    """
    import ast

    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover - a source file that will not parse
        return []
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dumped = ast.dump(fn)
        if not any(marker in dumped for marker in _DIRECTION_MARKERS):
            continue
        args = fn.args
        positional = args.posonlyargs + args.args
        pairs = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
        for arg, default in pairs:
            if _names_a_metric(arg.arg):
                found.append((fn.name, arg.arg, ast.unparse(default)))
    return found


class TestNoDirectionHelperAnywhereDefaultsItsMetricName:
    """The sweep, so the pin is not limited to the helpers we happened to list."""

    def test_the_package_has_no_defaulted_metric_name_on_a_direction_helper(self):
        offenders = {}
        for path in sorted(_SRC.rglob("*.py")):
            hits = _defaulted_metric_name_params(path)
            if hits:
                offenders[str(path.relative_to(_SRC))] = hits
        assert not offenders, (
            "a direction-deciding helper defaults the metric name it grades: "
            f"{offenders}. That is a guessed direction hiding where no source "
            "scan for substrings can see it, which is exactly how the third "
            "recurrence survived two waves."
        )

    def test_the_sweep_sees_a_planted_default(self, tmp_path):
        """Positive control: the sweep must not be able to go silently green."""
        planted = tmp_path / "planted.py"
        planted.write_text(
            "def grade(value, threshold, metric_name=None):\n"
            "    return metric_direction(metric_name)\n"
        )
        assert _defaulted_metric_name_params(planted) == [("grade", "metric_name", "None")]

    def test_the_sweep_leaves_chart_text_alone(self, tmp_path):
        """Negative control: a defaulted axis label is not a guessed direction."""
        innocent = tmp_path / "innocent.py"
        innocent.write_text(
            "def plot(metric_name, title=None, x_label='Calibration Error'):\n"
            "    return metric_direction(metric_name), title, x_label\n"
        )
        assert _defaulted_metric_name_params(innocent) == []


class TestTheTwoBranchesAgreeOnSign:
    """Wave 4: ``_fairness_verdict`` contradicted itself between its branches.

    The point-estimate fallback compared ``abs(value)`` against the bound, which
    is the right reading for a violation MAGNITUDE and the same reading as
    ``check_threshold`` and ``adapters_reporting._is_breached``. The interval
    branch compared the RAW ``hi`` and ``lo``. So the same violation was graded
    two ways depending only on whether a confidence interval happened to be
    available. Executed before the fix, all on
    ``demographic_parity_difference`` against a 0.10 bound:

        CI [-0.50, -0.40]  -> 'fair'      (a 0.45 gap, certified fair)
        CI [+0.40, +0.50]  -> 'unfair'
        point estimate -0.45 -> 'unfair'

    The sign of a max-minus-min gap is an artefact of which group the
    implementation happened to subtract from which. Grading on it is the same
    defect wave 3 fixed in ``adapters_reporting._is_breached``, where an
    unsigned test reported a -0.45 demographic parity difference as no breach.
    Latent here only because the four in-repo callers feed non-negative gaps.
    """

    LOWER = "demographic_parity_difference"
    HIGHER = "disparate_impact_ratio"

    def test_a_negative_interval_breach_is_unfair_just_like_its_mirror(self):
        assert _fairness_verdict(-0.45, -0.50, -0.40, 0.10, metric_name=self.LOWER) == "unfair"
        assert _fairness_verdict(0.45, 0.40, 0.50, 0.10, metric_name=self.LOWER) == "unfair"

    def test_the_interval_branch_agrees_with_the_point_branch_on_both_signs(self):
        nan = float("nan")
        for value in (-0.45, 0.45, -0.02, 0.02):
            interval = _fairness_verdict(
                value, value - 0.01, value + 0.01, 0.10, metric_name=self.LOWER
            )
            point = _fairness_verdict(value, nan, nan, 0.10, metric_name=self.LOWER)
            assert interval == point, f"value={value}: CI said {interval}, point said {point}"

    @pytest.mark.parametrize(
        "lo,hi,expected",
        [
            # Wholly inside the fair band [-0.10, +0.10], either side of zero.
            (-0.04, -0.01, "fair"),
            (0.01, 0.04, "fair"),
            (-0.04, 0.04, "fair"),
            # Wholly outside it, either side of zero.
            (0.40, 0.50, "unfair"),
            (-0.50, -0.40, "unfair"),
            # Straddling a band edge: neither shown fair nor shown unfair.
            (0.05, 0.20, "insufficient_evidence"),
            (-0.20, -0.05, "insufficient_evidence"),
            # Straddling the WHOLE band: a wide interval proves nothing.
            (-0.50, 0.50, "insufficient_evidence"),
        ],
    )
    def test_the_fair_band_is_symmetric_about_zero_for_a_magnitude(self, lo, hi, expected):
        assert _fairness_verdict((lo + hi) / 2, lo, hi, 0.10, metric_name=self.LOWER) == expected

    @pytest.mark.parametrize("lo,hi", [(-0.04, -0.01), (0.40, 0.50), (0.05, 0.20), (-0.50, 0.50)])
    def test_mirroring_the_interval_mirrors_nothing_about_the_verdict(self, lo, hi):
        """Formal statement of the property: sign is not evidence.

        For a violation magnitude, ``verdict(v, lo, hi)`` must equal
        ``verdict(-v, -hi, -lo)``. This is what the raw-bound interval branch
        violated.
        """
        forward = _fairness_verdict((lo + hi) / 2, lo, hi, 0.10, metric_name=self.LOWER)
        mirrored = _fairness_verdict(-(lo + hi) / 2, -hi, -lo, 0.10, metric_name=self.LOWER)
        assert forward == mirrored, f"[{lo}, {hi}] -> {forward} but its mirror -> {mirrored}"

    @pytest.mark.parametrize(
        "lo,hi,expected",
        [
            (0.90, 0.99, "fair"),
            (0.00, 0.05, "unfair"),
            (0.70, 0.95, "insufficient_evidence"),
        ],
    )
    def test_the_ratio_family_is_unchanged_and_stays_one_sided(self, lo, hi, expected):
        """Negative control: a required MINIMUM is not a symmetric band.

        Making the magnitude branch symmetric must not make the four-fifths
        rule symmetric too; there the fair band is ``[threshold, ...)`` and
        nothing below the floor may read fair.
        """
        assert _fairness_verdict((lo + hi) / 2, lo, hi, 0.80, metric_name=self.HIGHER) == expected


class TestTheVerdictAgreesWithTheGateOnADegenerateBound:
    """The same unbreachable bound must not be 'fair' here and could-not-check there.

    ``check_threshold`` now refuses a required minimum of 0.0 on a
    higher-is-better metric, because every possible value satisfies it. The
    verdict helper placed the interval against that same bound and answered
    'fair', so one surface would have said the metric could not be graded while
    the other certified it.
    """

    def test_a_zero_floor_on_a_ratio_is_not_a_fair_verdict(self):
        assert (
            _fairness_verdict(0.00, 0.00, 0.00, 0.0, metric_name="disparate_impact_ratio")
            == "insufficient_evidence"
        )

    def test_the_point_estimate_branch_refuses_it_too(self):
        nan = float("nan")
        assert (
            _fairness_verdict(0.00, nan, nan, 0.0, metric_name="disparate_impact_ratio")
            == "insufficient_evidence"
        )

    def test_a_negative_maximum_on_a_magnitude_is_not_a_verdict_either(self):
        assert (
            _fairness_verdict(0.02, 0.01, 0.03, -0.10, metric_name="demographic_parity_difference")
            == "insufficient_evidence"
        )

    def test_zero_tolerance_on_a_magnitude_still_grades(self):
        """Negative control: the mirror bound is real and must keep grading."""
        name = "demographic_parity_difference"
        assert _fairness_verdict(0.00, 0.00, 0.00, 0.0, metric_name=name) == "fair"
        assert _fairness_verdict(0.30, 0.25, 0.35, 0.0, metric_name=name) == "unfair"

    def test_a_real_floor_on_a_ratio_still_grades(self):
        """Negative control: the four-fifths rule must not be disarmed."""
        name = "disparate_impact_ratio"
        assert _fairness_verdict(0.05, 0.01, 0.10, 0.80, metric_name=name) == "unfair"
        assert _fairness_verdict(0.95, 0.90, 0.99, 0.80, metric_name=name) == "fair"


# ── 2. the guard, widened from one file to the whole package ───────────────


def _detector():
    """The ONE direction-test detector, shared with the other direction pins.

    Imported, never copied. Wave 3 wrote a SECOND marker list here, next to the
    one in ``tests/test_audit_final_release.py``, and the two then disagreed
    about what "the scanner" catches: this copy knew ten tokens in six operand
    names and could not see ``.lower()``, ``.find()``, an attribute operand, a
    renamed operand or a token held in a constant. Wave 4 rebuilt the detector
    on the parsed tree, so this module now calls that one rather than keeping a
    weaker twin alive beside it.
    """
    try:
        from tests.test_audit_final_release import _direction_hit_records, _rank_direction_hits
    except ImportError:  # pragma: no cover - depends on how pytest set sys.path
        import importlib.util

        path = pathlib.Path(__file__).with_name("test_audit_final_release.py")
        spec = importlib.util.spec_from_file_location("_wave3_pin_scanner", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _direction_hit_records = mod._direction_hit_records
        _rank_direction_hits = mod._rank_direction_hits
    return _direction_hit_records, _rank_direction_hits


class TestNoSubstringDirectionTestAnywhereInThePackage:
    """Same pattern, widened from ``_metric_direction.py`` to every module.

    Each recurrence of this bug lived in a file the previous guard did not
    read: the gate, then the metrics helper, then a rendering adapter. Scoping
    the guard to the file that was fixed last time is what let the next copy
    ship.
    """

    def test_no_module_classifies_a_direction_by_substring(self):
        direction_hit_records, rank_direction_hits = _detector()
        records = []
        for path in sorted(_SRC.rglob("*.py")):
            records.extend(direction_hit_records(path))
        assert not records, (
            "a substring direction test is back in the package. 'ratio' is a "
            "substring of 'cali[bratio]n_difference' and 'disparate_impact' is a "
            "prefix of 'disparate_impact_difference'. Resolve the direction "
            "through evaluation.vfairness_metrics._metric_direction instead. "
            "Ranked by whether the hit reaches a verdict:\n" + rank_direction_hits(records)
        )

    def test_the_widened_guard_still_fires_on_a_real_code_hit(self, tmp_path):
        """Positive control, so the widened scan cannot go silently green.

        Prose must stay legal: a guard whose only green states are "the bug is
        absent" or "the explanation is deleted" pressures the next author to
        delete the explanation, which is how this understanding was lost the
        first time.
        """
        direction_hit_records, _rank = _detector()

        offender = tmp_path / "offender.py"
        offender.write_text(
            '"""Prose only: \'"disparate_impact" in name\' is the bug."""\n'
            '# comment mentioning "disparate_impact" in name\n'
            "def f(name):\n"
            '    return "disparate_impact" in name\n'
        )
        assert direction_hit_records(offender), (
            "the widened guard no longer sees the bug in executable code"
        )

        innocent = tmp_path / "innocent.py"
        innocent.write_text(
            '"""Explains why a naive \'"disparate_impact" in name\' test is wrong."""\n'
            "def f(name):\n"
            '    return name == "disparate_impact" or name.endswith("_ratio")\n'
        )
        assert not direction_hit_records(innocent), (
            "the widened guard fires on prose; documenting the incident would fail CI"
        )

    @pytest.mark.parametrize(
        "needle,haystack,must_fire,why",
        [
            ('"ratio"', "name", True, "the token that started the bug class"),
            ('"disparate_impact"', "metric_name", True, "the snake_case spelling"),
            (
                '"disparate impact"',
                "label.lower()",
                True,
                "the SPACED display-label form, which the first attempt at the "
                "narrowing below would have blinded the guard to",
            ),
            ('"Disparate Impact"', "label", True, "mixed case, still a metric name"),
            ('"calibration"', "metric", True, "another family token"),
            (
                '"Fix all error-level issues"',
                "str(r)",
                False,
                "the FALSE POSITIVE: an English sentence containing the direction "
                "token 'error', over a stringified recommendation. The guard reported "
                "a recommendation dedup in operations/cicd/validator.py as a "
                "substring direction test on 2026-09-27, and the only way to make it "
                "green was to delete a dedup the product needs.",
            ),
            (
                '"proceed to model training"',
                "str(r)",
                False,
                "prose with no metric word at all",
            ),
        ],
    )
    def test_the_needle_must_be_a_metric_name_and_not_a_sentence(
        self, tmp_path, needle, haystack, must_fire, why
    ):
        """Both directions of the narrowing, in one table.

        A guard that over-accuses gets switched off, and then it protects nothing.
        A guard that under-accuses protects nothing either. The vocabulary is
        DERIVED from the metric name tables and the direction tokens, so a metric
        added there cannot fall out of it, and these rows say what that buys.
        """
        direction_hit_records, _rank = _detector()
        module = tmp_path / "case.py"
        module.write_text(
            f"def f(name, metric, metric_name, label, r):\n    return {needle} in {haystack}\n"
        )
        fired = bool(direction_hit_records(module))
        assert fired is must_fire, (
            f"{needle} in {haystack}: the detector "
            f"{'missed' if must_fire else 'over-accused'} it. {why}"
        )


# ── 3. nothing unmeasured may be reassured about ───────────────────────────


def _not_assessable_analyzer():
    """One group above ``min_group_size``, so every disparity metric is NaN."""
    rng = np.random.default_rng(0)
    n = 120
    sens = np.array(["A"] * 110 + ["B"] * 10)
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y_true, y_pred, sens, min_group_size=30)


def _assessable_fair_analyzer():
    """Two ample groups with near-identical selection rates: a real PASS."""
    rng = np.random.default_rng(11)
    n = 800
    sens = np.array(["A"] * 400 + ["B"] * 400)
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y_true, y_pred, sens, min_group_size=30)


class TestCouldNotCheckIsNotGivenThePassingAdvice:
    def test_unmeasured_metrics_do_not_get_the_passing_recommendation(self):
        """The recommendation must not read as "you are fine, keep monitoring".

        Before this pin, every metric on a not-assessable run carried
        evaluation "Unable to compute due to insufficient data." beside
        recommendation "Continue monitoring this metric as part of regular
        fairness audits. Document your fairness practices for compliance
        purposes.", which is the advice for a metric that PASSED.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanations = _not_assessable_analyzer().get_explanations()

        metrics = explanations.get("metrics", {})
        assert metrics, "the fixture produced no metric explanations to check"
        for name, entry in metrics.items():
            # Wording changed from "Unable to compute due to insufficient data."
            # to a COULD NOT CHECK sentence when the severity beside it stopped
            # being "info" (the severity a graded, benign metric gets). What this
            # test pins is the RECOMMENDATION, and that is unchanged below.
            assert "could not check" in entry["evaluation"].lower(), name
            assert entry.get("severity") == "could_not_check", name
            rec = entry["recommendation"]
            assert _PASSING_METRIC_RECOMMENDATION not in rec, (
                f"{name} was never measured, yet it carries the recommendation "
                f"written for a passing metric: {rec!r}"
            )
            assert "not measured" in rec.lower(), (
                f"{name}: the recommendation must say the metric was not measured, got {rec!r}"
            )
            assert "min_group_size" in rec, (
                f"{name}: the recommendation must tell the reader what to do "
                f"(min_group_size / more data / re-run), got {rec!r}"
            )

    def test_the_excluded_group_is_named(self):
        """ "Collect more data" is useless without saying for which group."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanations = _not_assessable_analyzer().get_explanations()
        recs = [e["recommendation"] for e in explanations["metrics"].values()]
        assert recs
        assert all("'B'" in r for r in recs), (
            f"group B was the excluded stratum and is not named: {recs[:1]}"
        )

    def test_a_measured_passing_metric_keeps_the_monitoring_advice(self):
        """Negative control: the passing branch must NOT have been broadened."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanations = _assessable_fair_analyzer().get_explanations()
        recs = explanations["metrics"]
        assert recs
        assert any(_PASSING_METRIC_RECOMMENDATION in e["recommendation"] for e in recs.values()), (
            "a genuinely passing metric lost its monitoring recommendation"
        )

    def test_an_ungradeable_direction_gets_its_own_advice_too(self):
        """A metric that was measured but never COMPARED is also could-not-check."""
        explainer = FairExplAIner()
        # Instance-level copy: the class attribute is a shared module dict.
        explainer.metrics_definitions = dict(explainer.metrics_definitions)
        explainer.metrics_definitions["some_bespoke_client_metric"] = {
            "definition": "A bespoke client metric with no declared direction.",
            "thresholds": {"acceptable": 0.10},
        }
        explanation = explainer.explain_metric("some_bespoke_client_metric", 0.42)
        assert "could not check" in explanation.evaluation.lower()
        assert _PASSING_METRIC_RECOMMENDATION not in explanation.recommendation
        assert "direction" in explanation.recommendation.lower()


# ── 4. the NOT ASSESSABLE prose, and no em-dashes in it ────────────────────


class TestNotAssessableProse:
    def test_the_summary_carries_no_em_dash(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = _not_assessable_analyzer().get_report()
        summary = report["assessment"]["summary"]
        assert summary.startswith("NOT ASSESSABLE")
        assert "—" not in summary and "–" not in summary, (
            f"an em/en-dash is rasterised into the Assessment Summary panel: {summary!r}"
        )


# ── 5. the adapter uses the third badge, not a measured FAIL ───────────────


class TestDetailedReportAdapterUsesTheThirdState:
    def _capture(self, monkeypatch, report):
        from vfairness.rendering import adapters_post_processing as ad

        captured = {}

        def fake_render(_template, data):
            captured.update(data)
            return "<svg/>"

        monkeypatch.setattr(ad, "render_svg", fake_render)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ad.fairness_detailed_report_to_svg(report)
        return captured

    def test_a_could_not_check_row_is_not_marked_as_a_measured_failure(self, monkeypatch):
        report = {
            "assessment": {"fairness_score": 0.5},
            "metrics": {
                "some_bespoke_client_metric": {"value": 0.42, "threshold": 0.10},
                "demographic_parity_difference": {"value": 0.02, "threshold": 0.10},
            },
            "group_statistics": {},
        }
        data = self._capture(monkeypatch, report)
        rows = {m["name"]: m for m in data["metrics"]}
        unchecked = rows["Some Bespoke Client Metric"]
        assert unchecked.get("state") == "could_not_check", (
            "the adapter must declare the third state outright; the template "
            "supports it, and a prose prefix is not a state"
        )
        assert unchecked["passed"] is not False, (
            "passed=False is the representation of a MEASURED failure; a metric "
            "whose threshold was never applied did not fail"
        )
        assert "COULD NOT CHECK" in unchecked["interpretation"]

        graded = rows["Demographic Parity Difference"]
        assert graded.get("state") != "could_not_check"
        assert graded["passed"] is True

    def test_a_measured_breach_is_still_a_failure(self, monkeypatch):
        """Negative control: the third state must not swallow real failures."""
        report = {
            "assessment": {"fairness_score": 0.5},
            "metrics": {
                "demographic_parity_difference": {"value": 0.42, "threshold": 0.10},
            },
            "group_statistics": {},
        }
        data = self._capture(monkeypatch, report)
        row = data["metrics"][0]
        assert row["passed"] is False
        assert row.get("state") != "could_not_check"


# ── 6. the stamped methodology version must be an authored one ─────────────


class TestMethodologyDocumentMatchesTheStampedVersion:
    """Every report stamps ``METHODOLOGY_VERSION``; the document must describe it.

    ``_methodology.py`` moved to M1.1 while ``docs/METHODOLOGY.md`` still said
    "RATIFIED (M1.0)" and its changelog ended at the M1.0 row, so every report
    cited a methodology the authored document did not contain.
    """

    def _doc(self):
        path = _DOCS / "METHODOLOGY.md"
        if not path.exists():  # pragma: no cover - source checkout only
            pytest.skip("docs/METHODOLOGY.md not present in this install")
        return path.read_text(encoding="utf-8")

    def test_the_status_line_names_the_stamped_version(self):
        text = self._doc()
        head = "\n".join(text.splitlines()[:12])
        assert METHODOLOGY_VERSION in head, (
            f"the document's status block does not name {METHODOLOGY_VERSION}, "
            "which is the version every report stamps"
        )

    def test_the_changelog_has_a_row_for_the_stamped_version(self):
        text = self._doc()
        pattern = rf"^\|\s*{re.escape(METHODOLOGY_VERSION)}\s*\|"
        assert re.search(pattern, text, flags=re.MULTILINE), (
            f"the methodology changelog has no {METHODOLOGY_VERSION} row"
        )

    def test_the_changelog_is_append_only(self):
        """The superseded rows stay; a bump adds a row, it does not rewrite one."""
        text = self._doc()
        for older in ("M1.0", "M0.1-draft"):
            assert re.search(rf"^\|\s*{re.escape(older)}\s*\|", text, flags=re.MULTILINE), (
                f"the {older} changelog row was removed; the changelog is append-only"
            )
