"""Readiness 3, annotations lane: a type annotation is a claim, so it may not
advertise a value the code refuses, and a breaking removal must reach the
changelog.

Both findings were reproduced by execution on 2026-09-10 before either fix.

FINDING 1, sklearn_wrappers.py. ``FairRegressor.__init__`` was annotated
``method: Literal["offset", "reweighting", "constrained"]`` and
``fairness_constraint: Literal["mean_parity", "error_parity", "bounded_loss"]``,
and ``make_fair_regressor`` carried the same ``fairness_constraint``. Measured
against a real ``fit()``, four of those six advertised values raise:
``'reweighting'`` ValueError (removed), ``'constrained'`` NotImplementedError,
``'error_parity'`` and ``'bounded_loss'`` NotImplementedError. An editor
autocompletes an annotation, a type checker blesses it and a reader trusts it,
so this is the same claims-versus-code defect as an inert parameter, one layer
up. The annotations now name only ``'offset'`` and ``'mean_parity'``.

FINDING 2, CHANGELOG.md. The removal of ``FairRegressor(method='reweighting')``
was not recorded, and neither were the keyword-only post-processing ``fit()``
signatures, the five removed parameters, or the option values that now raise.
A changelog that records some breaking changes and not others reads as
complete, which is worse than one that records none.

The pins are written so that they cannot pass vacuously:

* The annotation pin DERIVES the advertised values from the annotation itself
  and puts every one of them through a real ``fit()``, so it fails whether the
  annotation grows a refused value or the code starts refusing an advertised
  one. It also asserts the honoured value is still named, so deleting the
  ``Literal`` does not make it green.
* The refusal pin asserts the removed and unimplemented values STILL raise from
  an untyped caller, which is the only place they can now arrive from.
* The over-correction controls assert MEASURED numbers on the healthy paths:
  the mean-parity offsets really close the gap to the tolerance, and the
  ``FairClassifier`` Literals, which the code does honour in full, were not
  narrowed along with the dishonest ones.
* The changelog pin checks each removal BY EXECUTION first (the keyword really
  is a TypeError, the signature really is keyword-only) and only then requires
  the changelog to name it, so it cannot drift into a spelling test. Its own
  detector is proved by a harness control that feeds it a changelog with one
  entry missing and requires a report.
"""

from __future__ import annotations

import inspect
import re
import typing
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from vfairness.in_processing.wrappers.sklearn_wrappers import (
    FairClassifier,
    FairRegressor,
    make_fair_regressor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"


# --------------------------------------------------------------------------
# Fixtures. Label arrays are dtype=object on purpose: a numpy "<U5" array
# truncates a longer label silently, which quietly undoes the fixture.
# --------------------------------------------------------------------------


def _regression_data(n_per: int = 120):
    rng = np.random.default_rng(20260910)
    groups = np.array(["group_alpha"] * n_per + ["group_beta"] * n_per, dtype=object)
    assert groups[-1] == "group_beta", "label array truncated; use dtype=object"
    is_beta = (groups == "group_beta").astype(float)
    X = np.column_stack([rng.normal(0.0, 1.0, size=2 * n_per), is_beta])
    y = X[:, 0] + 4.0 * is_beta + rng.normal(0.0, 0.05, size=2 * n_per)
    return X, y, groups


def _classification_data(n_per: int = 90):
    rng = np.random.default_rng(20260910)
    groups = np.array(["group_alpha"] * n_per + ["group_beta"] * n_per, dtype=object)
    assert groups[-1] == "group_beta", "label array truncated; use dtype=object"
    is_beta = (groups == "group_beta").astype(float)
    X = np.column_stack([rng.normal(0.0, 1.0, size=2 * n_per), is_beta])
    y = ((X[:, 0] + 1.4 * is_beta) > 0).astype(int)
    return X, y, groups


def _literal_values(func, parameter: str) -> tuple:
    """The values a parameter's ``Literal`` annotation advertises."""
    hints = typing.get_type_hints(func)
    assert parameter in hints, (
        f"{func.__qualname__} has no annotation for {parameter!r}; it has {sorted(hints)}"
    )
    annotation = hints[parameter]
    assert typing.get_origin(annotation) is typing.Literal, (
        f"{func.__qualname__}.{parameter} is annotated {annotation!r}, not a Literal; "
        f"this pin only means something while the annotation enumerates values"
    )
    values = typing.get_args(annotation)
    assert values, f"{func.__qualname__}.{parameter} advertises an empty Literal"
    return values


# --------------------------------------------------------------------------
# FINDING 1, the pin: every value the annotation advertises must survive fit().
# --------------------------------------------------------------------------


def test_every_fairregressor_option_the_annotation_advertises_is_accepted_by_fit() -> None:
    """The claim the annotation makes, executed.

    Before the fix this reported four refusals out of six advertised values:
      method='reweighting'            ValueError
      method='constrained'            NotImplementedError
      fairness_constraint='error_parity'   NotImplementedError
      fairness_constraint='bounded_loss'   NotImplementedError
    """
    X, y, groups = _regression_data()
    advertised = {
        "fairness_constraint": _literal_values(FairRegressor.__init__, "fairness_constraint"),
        "method": _literal_values(FairRegressor.__init__, "method"),
    }

    refused = []
    for parameter, values in advertised.items():
        for value in values:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    FairRegressor(base_estimator=Ridge(), **{parameter: value}).fit(
                        X, y, sensitive_attr=groups
                    )
            except Exception as exc:  # noqa: BLE001 - the point is what it raised
                refused.append(f"{parameter}={value!r} -> {type(exc).__name__}: {exc}")

    assert not refused, (
        "the FairRegressor annotations advertise values fit() refuses:\n  " + "\n  ".join(refused)
    )
    # The pin must not go green by the annotation being emptied or replaced.
    assert advertised["fairness_constraint"] == ("mean_parity",), advertised
    assert advertised["method"] == ("offset",), advertised


def test_make_fair_regressor_advertises_only_what_it_can_build() -> None:
    """The convenience function carried the same over-wide Literal."""
    X, y, groups = _regression_data()
    values = _literal_values(make_fair_regressor, "fairness_constraint")
    assert values == ("mean_parity",), values

    refused = []
    for value in values:
        try:
            make_fair_regressor(base_estimator=Ridge(), fairness_constraint=value).fit(
                X, y, sensitive_attr=groups
            )
        except Exception as exc:  # noqa: BLE001
            refused.append(f"fairness_constraint={value!r} -> {type(exc).__name__}: {exc}")
    assert not refused, (
        "make_fair_regressor advertises a constraint it cannot fit:\n  " + "\n  ".join(refused)
    )


# --------------------------------------------------------------------------
# FINDING 1, the refusal pin: narrowing the annotation must not soften the
# runtime refusal, which is now the only thing an untyped caller meets.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parameter", "value", "expected", "needle"),
    [
        ("method", "reweighting", ValueError, "was removed"),
        ("method", "constrained", NotImplementedError, "not implemented"),
        ("fairness_constraint", "error_parity", NotImplementedError, "not implemented"),
        ("fairness_constraint", "bounded_loss", NotImplementedError, "not implemented"),
    ],
)
def test_a_removed_or_unimplemented_option_still_refuses_loudly(
    parameter: str, value: str, expected: type, needle: str
) -> None:
    X, y, groups = _regression_data()
    # The `type: ignore` comments in this file are not noise, they are the
    # evidence: mypy refusing these calls is exactly the property the narrowed
    # annotations deliver, and the runtime refusal below is what an untyped
    # caller meets. CI runs `mypy src`, so tests are not type-checked there.
    with pytest.raises(expected) as caught:
        FairRegressor(base_estimator=Ridge(), **{parameter: value}).fit(  # type: ignore[arg-type]
            X, y, sensitive_attr=groups
        )
    message = str(caught.value)
    assert value in message, message
    assert needle in message, message
    # And the annotation no longer offers it.
    assert value not in _literal_values(FairRegressor.__init__, parameter)


# --------------------------------------------------------------------------
# FINDING 1, over-correction control 1: the honoured path still measures what
# it claims. Narrowing an annotation must not touch behaviour.
# --------------------------------------------------------------------------


def test_the_honoured_mean_parity_offset_path_still_closes_the_gap_it_measures() -> None:
    X, y, groups = _regression_data()
    tolerance = 0.5
    reg = FairRegressor(
        base_estimator=Ridge(),
        fairness_constraint="mean_parity",
        tolerance=tolerance,
        method="offset",
    ).fit(X, y, sensitive_attr=groups)

    alpha = groups == "group_alpha"
    unadjusted = np.asarray(reg._inner_model.predict(X), dtype=float)
    adjusted = reg.predict_with_sensitive_attr(X, groups)

    gap_before = abs(unadjusted[alpha].mean() - unadjusted[~alpha].mean())
    gap_after = abs(adjusted[alpha].mean() - adjusted[~alpha].mean())

    # Measured on this fixture, 2026-09-10.
    assert gap_before == pytest.approx(3.877029, abs=1e-5), gap_before
    assert gap_after == pytest.approx(0.5, abs=1e-9), gap_after
    assert reg.group_offsets_["group_alpha"] == pytest.approx(1.688514, abs=1e-5)
    assert reg.group_offsets_["group_beta"] == pytest.approx(-1.688514, abs=1e-5)

    # predict() without the sensitive attribute is still the unadjusted answer,
    # and still says so.
    with pytest.warns(UserWarning, match="UNADJUSTED"):
        plain = reg.predict(X)
    assert np.allclose(plain, unadjusted)


# --------------------------------------------------------------------------
# FINDING 1, over-correction control 2: the Literals the code DOES honour in
# full were not narrowed with the dishonest ones.
# --------------------------------------------------------------------------


def test_the_fairclassifier_literals_are_honest_and_were_left_alone() -> None:
    X, y, groups = _classification_data()

    methods = _literal_values(FairClassifier.__init__, "method")
    metrics = _literal_values(FairClassifier.score, "metric")
    assert methods == ("reductions", "threshold", "grid_search"), methods
    assert metrics == ("accuracy", "fairness", "combined"), metrics

    # Measured on this fixture, 2026-09-10: every advertised method fits, and
    # every advertised metric returns its own distinct number.
    expected = {
        "reductions": {"accuracy": 0.833333, "fairness": -0.077778, "combined": 0.755556},
        "threshold": {"accuracy": 0.811111, "fairness": -0.033333, "combined": 0.777778},
        "grid_search": {"accuracy": 0.761111, "fairness": -0.066667, "combined": 0.694444},
    }
    for method in methods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = FairClassifier(
                base_estimator=LogisticRegression(), method=method, random_state=0
            ).fit(X, y, sensitive_attr=groups)
            scored = {
                metric: float(clf.score(X, y, sensitive_attr=groups, metric=metric))
                for metric in metrics
            }
        assert scored == pytest.approx(expected[method], abs=1e-5), (method, scored)
        assert scored["combined"] == pytest.approx(
            scored["accuracy"] + scored["fairness"], abs=1e-9
        ), scored


# --------------------------------------------------------------------------
# FINDING 2, the changelog. Each row is verified by EXECUTION first, then
# required to be named in the changelog.
# --------------------------------------------------------------------------


def _changelog_text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


def _unreleased_section(text: str) -> str:
    """The body of the top version heading, where unreleased work accumulates."""
    headings = [m.start() for m in re.finditer(r"^## \[", text, re.M)]
    assert len(headings) >= 2, "changelog has fewer than two version headings"
    return text[headings[0] : headings[1]]


REMOVED_KEYWORDS = [
    # (label, callable that must raise TypeError, the keyword's name)
    ("PredictionReweighter(preserve_ranking=)", "preserve_ranking"),
    ("CalibratedEqualizer(preserve_calibration=)", "preserve_calibration"),
    ("TemperatureScaling(init_temperature=)", "init_temperature"),
    ("compute_feature_correlations(include_pvalues=)", "include_pvalues"),
    ("generate_recommendation(tradeoff_analysis=)", "tradeoff_analysis"),
]

KEYWORD_ONLY_FIT_CLASSES = (
    "PredictionReweighter",
    "RejectionOptionClassifier",
    "CalibratedEqualizer",
    "DistributionMatcher",
    "BaseReweighter",
    "ThresholdOptimizer",
    "GroupThresholdOptimizer",
    "MultiObjectiveThresholdOptimizer",
    "BaseThresholdOptimizer",
)


def _call_with_removed_keyword(keyword: str):
    """Call the real surface with the removed keyword. Returns the exception."""
    import pandas as pd

    from vfairness.in_processing.analyzer import FairnessTrainingAnalyzer
    from vfairness.post_processing.calibration.methods import TemperatureScaling
    from vfairness.post_processing.reweighting.reweighter import (
        CalibratedEqualizer,
        PredictionReweighter,
    )
    from vfairness.preprocessing.feature_engineering.correlation import (
        compute_feature_correlations,
    )

    def _analyzer_call():
        rng = np.random.default_rng(0)
        analyzer = FairnessTrainingAnalyzer(
            rng.normal(size=(40, 2)),
            rng.integers(0, 2, 40),
            np.array(["a"] * 20 + ["b"] * 20, dtype=object),
        )
        analyzer.generate_recommendation([], tradeoff_analysis=None)

    calls = {
        "preserve_ranking": lambda: PredictionReweighter(preserve_ranking=True),  # type: ignore[call-arg]
        "preserve_calibration": lambda: CalibratedEqualizer(preserve_calibration=True),  # type: ignore[call-arg]
        "init_temperature": lambda: TemperatureScaling(init_temperature=1.0),  # type: ignore[call-arg]
        "include_pvalues": lambda: compute_feature_correlations(  # type: ignore[call-arg]
            pd.DataFrame({"race": ["a", "b"] * 10, "feature": np.arange(20.0)}),
            protected_attributes=["race"],
            include_pvalues=False,
        ),
        "tradeoff_analysis": _analyzer_call,
    }
    try:
        calls[keyword]()
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


@pytest.mark.parametrize(
    ("label", "keyword"), REMOVED_KEYWORDS, ids=[r[1] for r in REMOVED_KEYWORDS]
)
def test_a_removed_parameter_really_is_removed_and_the_changelog_says_so(
    label: str, keyword: str
) -> None:
    exc = _call_with_removed_keyword(keyword)
    assert isinstance(exc, TypeError), f"{label} did not raise TypeError; got {exc!r}"
    assert keyword in str(exc), str(exc)

    section = _unreleased_section(_changelog_text())
    assert keyword in section, (
        f"{label} raises TypeError but the unreleased changelog section never "
        f"names {keyword!r}; a breaking removal that is not recorded reads as "
        f"still supported"
    )


def test_the_keyword_only_fit_signatures_are_real_and_recorded() -> None:
    from vfairness.post_processing.reweighting.reweighter import (
        BaseReweighter,
        CalibratedEqualizer,
        DistributionMatcher,
        PredictionReweighter,
        RejectionOptionClassifier,
    )
    from vfairness.post_processing.threshold_optimization.optimizer import (
        BaseThresholdOptimizer,
        GroupThresholdOptimizer,
        MultiObjectiveThresholdOptimizer,
        ThresholdOptimizer,
    )

    classes = [
        PredictionReweighter,
        RejectionOptionClassifier,
        CalibratedEqualizer,
        DistributionMatcher,
        BaseReweighter,
        ThresholdOptimizer,
        GroupThresholdOptimizer,
        MultiObjectiveThresholdOptimizer,
        BaseThresholdOptimizer,
    ]
    assert [c.__name__ for c in classes] == list(KEYWORD_ONLY_FIT_CLASSES)

    positional = []
    for cls in classes:
        for name, parameter in inspect.signature(cls.fit).parameters.items():  # type: ignore[attr-defined]
            if name in ("y_true", "y_prob", "sensitive_attr"):
                if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
                    positional.append(f"{cls.__name__}.fit({name}) is {parameter.kind.name}")
    assert not positional, "these fit() arguments are still positional:\n  " + "\n  ".join(
        positional
    )

    # And a positional call really is a TypeError, not merely an annotation.
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_prob = rng.random(100)
    groups = np.array(["a"] * 50 + ["b"] * 50, dtype=object)
    with pytest.raises(TypeError, match="positional"):
        ThresholdOptimizer().fit(y_true, y_prob, groups)  # type: ignore[misc]

    section = _unreleased_section(_changelog_text())
    missing = [name for name in KEYWORD_ONLY_FIT_CLASSES if name not in section]
    assert not missing, (
        "fit() is keyword-only on these classes and the unreleased changelog "
        f"section does not name them: {missing}"
    )
    assert "keyword" in section.lower(), "the changelog never says the arguments became keywords"


def _unreleased_bullets(section: str) -> list:
    """The top-level ``- **`` bullets of the unreleased section.

    Scoping to a single bullet matters: a bare substring search over the whole
    section passes on an unrelated sentence that happens to use the same word,
    which is how a record can read as present while the entry that carried it
    is gone.
    """
    parts = re.split(r"\n(?=- \*\*)", section)
    return [part for part in parts if part.lstrip().startswith("- **")]


def test_the_fairregressor_reweighting_removal_is_recorded() -> None:
    """The removal the docs lane found missing, checked in both directions."""
    X, y, groups = _regression_data()
    with pytest.raises(ValueError, match="reweighting") as caught:
        FairRegressor(base_estimator=Ridge(), method="reweighting").fit(  # type: ignore[arg-type]
            X, y, sensitive_attr=groups
        )
    assert "removed" in str(caught.value), str(caught.value)

    bullets = _unreleased_bullets(_unreleased_section(_changelog_text()))
    assert bullets, "the unreleased changelog section has no bullets to read"
    matching = [b for b in bullets if "FairRegressor" in b and "reweighting" in b]
    assert matching, (
        "FairRegressor(method='reweighting') raises ValueError and no bullet in "
        "the unreleased changelog section records it; an unrecorded breaking "
        "removal reads as still supported"
    )
    bullet = matching[0]
    assert "method='reweighting'" in bullet, bullet[:400]
    assert "ValueError" in bullet, bullet[:400]
    assert "'offset'" in bullet, "the entry does not say what to use instead"


def test_the_option_values_that_now_raise_are_recorded() -> None:
    from vfairness.post_processing.calibration.group_calibrator import (
        IMPLEMENTED_FALLBACK_STRATEGIES,
        GroupCalibrator,
    )
    from vfairness.post_processing.reweighting.reweighter import DistributionMatcher
    from vfairness.post_processing.threshold_optimization.optimizer import (
        GroupThresholdOptimizer,
    )

    assert IMPLEMENTED_FALLBACK_STRATEGIES == ("global", "none")
    with pytest.raises(NotImplementedError, match="borrow"):
        GroupCalibrator(fallback_strategy="borrow")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="grid"):
        GroupThresholdOptimizer(grid_search=False)
    with pytest.raises(ValueError, match="quantile"):
        DistributionMatcher(method="moment")

    section = _unreleased_section(_changelog_text())
    for token in ("fallback_strategy", "borrow", "grid_search", "DistributionMatcher"):
        assert token in section, f"the changelog never names {token!r}"


def test_the_changelog_detector_reports_a_missing_entry() -> None:
    """Harness control: a checker that cannot go red proves nothing.

    Feed ``_unreleased_section`` a changelog whose unreleased section has been
    stripped of one required name and require the same assertion to fail.
    """
    text = _changelog_text()
    section = _unreleased_section(text)
    assert "preserve_ranking" in section

    stripped = text.replace("preserve_ranking", "REDACTED_FOR_THE_CONTROL")
    assert "preserve_ranking" not in _unreleased_section(stripped)

    # The section extractor must still find a real section, not an empty string,
    # so the control proves the NAME is missing rather than the parser broken.
    assert _unreleased_section(stripped).startswith("## [")
    assert len(_unreleased_section(stripped)) > 1000
