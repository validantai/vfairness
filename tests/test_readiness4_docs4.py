"""Readiness lane docs4: promises the shipped docs make that the code refuses.

Every pin comes in two halves, because half a pin catches nothing:

  * a REFUSAL PIN, red the moment the false promise is put back into a
    published file or the refusal is removed from the code, and
  * an OVER-CORRECTION CONTROL, red if the fix was applied by deleting a
    capability that IS real or by making the library refuse everything. Each
    control asserts a MEASURED number, never membership of a broad set.

Findings, all reproduced by EXECUTION on 2026-09-10 before any edit:

  1. ``docs/site/api-reference/index.html`` documented
     ``regression_fairness_report(missing_strategy=...)`` as "'exclude' or
     'impute'". Executed:
         ConfigurationError: Unknown missing_strategy: 'impute'. Accepted
         values are 'exclude', 'as_group', 'error'.
     'impute' was never a value; the library does not invent labels.

  2. The same page documented
     ``classification_fairness_report(multiple_testing_correction=...)`` as
     "(e.g., 'bonferroni', 'fdr_bh')". Executed:
         ConfigurationError: Unknown correction method: fdr_bh
     'fdr_bh' IS a real value, but of the NEIGHBOURING parameter
     ``analyze_statistical_disparities(correction_method=...)``, documented
     100 sections earlier on the same page, where it is both the default and
     accepted (executed: 1 measured result, p=1.18e-32, effect 0.595). That
     is how the wrong spelling survived, and it is why the pin below asserts
     BOTH that the report refuses it and that the neighbour still takes it.

  3. THE BIG ONE. A Bayesian small-sample interval was promised for the
     DISPARITY metrics in eight places, and no such estimator exists.
     Executed on two groups of 20, with the metric asked for three ways:

         method='auto'      -> result.method='stratified_bootstrap_fold_debiased'
           warning: Small sample (min group: 20 < 30): running the stratified
           bootstrap with 400 resamples. A Bayesian interval is not implemented.
         method='bayesian'  -> result.method='stratified_bootstrap_fold_debiased'
           warning: Bayesian method requested but using bootstrap with enhanced
           sampling: a Bayesian estimator is not implemented (min group: 20).
         method='bootstrap' -> result.method='stratified_bootstrap_fold_debiased'

     interval_type was CONFIDENCE in all three, never CREDIBLE. The seven
     sealed ``*_with_ci`` gates were checked the same way and every one of
     them returned 'stratified_bootstrap_percentile' or
     'stratified_bootstrap_fold_debiased': neither BCa nor a credible
     interval is reachable on that path at all.

     But SOME Bayesian estimators are real, and deleting those claims would
     be its own defect. Executed: ``bayesian_proportion_ci(45, 100)`` returns
     method='bayesian_beta_binomial', IntervalType.CREDIBLE, posterior
     Beta(46, 56); ``bayesian_difference_ci`` returns
     'bayesian_difference_monte_carlo', CREDIBLE, and is the estimator the
     intersectional findings path actually calls; ``bayesian_mean_ci``
     returns 'bayesian_normal'; and
     ``get_group_metrics_with_ci(method='bayesian')`` returns a real Beta
     credible interval per group. The line that has to be drawn is between a
     group's RATE (Bayesian available) and a group GAP (bootstrap only), and
     the controls below assert both sides of it.

Note on scope: ``compute_metric_with_ci`` and ``select_method`` in
``_statistics.py`` already carried accurate docstrings (F18, 2026-09-09) and
were not touched. The remaining false site in that file was the MODULE
docstring, which advertised "Bayesian credible intervals for small sample
groups" next to "Automatic method selection based on sample size".
"""

from __future__ import annotations

import ast
import html as _html
import io
import pathlib
import re
import typing
import warnings
from contextlib import redirect_stdout
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from vfairness import (
    bayesian_difference_ci,
    bayesian_mean_ci,
    bayesian_proportion_ci,
    demographic_parity_difference,
    get_group_metrics_with_ci,
)
from vfairness.evaluation.vfairness_metrics._statistics import (
    IntervalType,
    compute_metric_with_ci,
    select_method,
)
from vfairness.evaluation.vfairness_metrics.report import (
    classification_fairness_report,
    regression_fairness_report,
)
from vfairness.exceptions import ConfigurationError, InvalidDataError
from vfairness.preprocessing.bias_detection.statistical import analyze_statistical_disparities

_LIB_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DOCS = _LIB_ROOT / "docs"

_API_HTML = _DOCS / "site" / "api-reference" / "index.html"
_METHODOLOGY_MD = _DOCS / "METHODOLOGY.md"
_OVERVIEW_MD = _DOCS / "LIBRARY_OVERVIEW.md"
_API_MD = _DOCS / "API_REFERENCE.md"
_CONCEPTS_HTML = _DOCS / "site" / "concepts" / "index.html"
_BUSINESS_HTML = _DOCS / "site" / "business-guide" / "index.html"
_STATISTICS_PY = (
    _LIB_ROOT / "src" / "vfairness" / "evaluation" / "vfairness_metrics" / "_statistics.py"
)

_ALL_SITES = (
    _API_HTML,
    _METHODOLOGY_MD,
    _OVERVIEW_MD,
    _API_MD,
    _CONCEPTS_HTML,
    _BUSINESS_HTML,
    _STATISTICS_PY,
)


def _text(path: pathlib.Path) -> str:
    if not path.exists():  # pragma: no cover - the docs tree ships with the repo
        pytest.skip(f"{path} not present in this checkout")
    return path.read_text(encoding="utf-8")


def _normalised(path: pathlib.Path) -> str:
    """File text with runs of whitespace collapsed, so a claim that a hand
    edit happened to wrap across two lines is still matched."""
    return re.sub(r"\s+", " ", _text(path))


def _reader_surface(path: pathlib.Path) -> str:
    """What a READER of this file is told, normalised.

    For a .py file that is the MODULE DOCSTRING, not the whole source. Checking
    the source body instead is trivially satisfied by the ``def`` line itself:
    a first attempt at the estimator-mention control stayed green after the
    docstring stopped naming ``bayesian_mean_ci``, because the function it
    names is defined 900 lines below.
    """
    if path.suffix != ".py":
        return _normalised(path)
    doc = ast.get_docstring(ast.parse(_text(path))) or ""
    assert doc, f"{path} has no module docstring"
    return re.sub(r"\s+", " ", doc)


# ---------------------------------------------------------------------------
# Reading the published page the way a reader does.
# ---------------------------------------------------------------------------

_FUNCTION_ANCHOR = re.compile(r'<span class="api-function-name">([^<]+)</span>')
_PARAM_RE = re.compile(
    r'<div class="api-param">\s*'
    r'<div class="api-param-header">\s*'
    r'<span class="api-param-name">([^<]+)</span>.*?'
    r'<div class="api-param-desc">(.*?)</div>',
    re.DOTALL,
)
_LITERAL_RE = re.compile(r"<code>'([^']+)'</code>")

# Cues that a description ADMITS a value is refused or not implemented, rather
# than offering it. Checked case-insensitively.
_REFUSAL_CUES = (
    "not accepted",
    "not implemented",
    "raises",
    "refused",
    "configurationerror",
    "no bayesian estimator",
)


def _function_section(page: str, name: str) -> str:
    """The published entry for one function, ending where the next one starts."""
    anchors = [(m.start(), m.group(1)) for m in _FUNCTION_ANCHOR.finditer(page)]
    starts = [i for i, (_, label) in enumerate(anchors) if label == name]
    assert len(starts) == 1, f"{len(starts)} entries on the page are named {name!r}"
    i = starts[0]
    begin = anchors[i][0]
    end = anchors[i + 1][0] if i + 1 < len(anchors) else len(page)
    return page[begin:end]


def _documented_params(section: str) -> Dict[str, str]:
    params = {name.strip(): desc for name, desc in _PARAM_RE.findall(section)}
    assert params, "no api-param blocks parsed out of this entry"
    return params


def _quoted_literals(desc: str) -> Set[str]:
    return set(_LITERAL_RE.findall(desc))


def _has_refusal_cue(desc: str) -> bool:
    low = _html.unescape(re.sub(r"<[^>]+>", "", desc)).lower()
    return any(cue in low for cue in _REFUSAL_CUES)


_ACCEPTED_RE = re.compile(r"Accepted values are ([^.]+)\.")


def _accepted_from_refusal_message(exc: BaseException) -> Set[str]:
    """The accepted set as the LIBRARY states it when it refuses something.

    Read out of the message a caller actually hits, so this pin cannot drift
    away from the code by copying a list into the test.
    """
    match = _ACCEPTED_RE.search(str(exc))
    assert match, f"refusal message carries no accepted-value list: {exc}"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def _literal_options(func: Any, param: str) -> Set[str]:
    """The option set declared by the parameter's Literal annotation."""
    hints = typing.get_type_hints(func)
    args = typing.get_args(hints[param])
    assert args, f"{func.__name__}.{param} carries no Literal options"
    return set(args)


# ---------------------------------------------------------------------------
# Fixtures: hand-built so every asserted number is exact, not a seeded draw.
#
# The label array is dtype=object on purpose. A numpy "<U..." array silently
# TRUNCATES longer strings, which would void the '__missing__' group the
# as_group control depends on.
# ---------------------------------------------------------------------------


def _regression_fixture(with_missing: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two groups whose MAE gap is exactly 0.75, plus an optional third group
    (sensitive attribute missing) whose MAE is 2.0, taking the gap to 1.75."""
    base = [float(i % 7) for i in range(40)]
    y_true: List[float] = base + base
    y_pred: List[float] = [v + 1.0 for v in base] + [v + 0.25 for v in base]
    attr: List[Any] = ["A"] * 40 + ["B"] * 40
    if with_missing:
        extra = [float(i % 7) for i in range(35)]
        y_true += extra
        y_pred += [v + 2.0 for v in extra]
        attr += [None] * 35
    return (
        np.array(y_true, dtype=float),
        np.array(y_pred, dtype=float),
        np.array(attr, dtype=object),
    )


def _classification_fixture() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group A: TPR 0.80 / FPR 0.10. Group B: TPR 0.50 / FPR 0.30. No RNG."""
    y_true = ([1] * 100 + [0] * 100) * 2
    y_pred = ([1] * 80 + [0] * 20 + [1] * 10 + [0] * 90) + (
        [1] * 50 + [0] * 50 + [1] * 30 + [0] * 70
    )
    attr = ["A"] * 200 + ["B"] * 200
    return np.array(y_true), np.array(y_pred), np.array(attr, dtype=object)


def _small_group_fixture() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two groups of 20, i.e. both under SMALL_SAMPLE_THRESHOLD (30), which is
    the only regime in which a Bayesian interval was ever promised."""
    y_true = ([1] * 10 + [0] * 10) * 2
    y_pred = ([1] * 8 + [0] * 12) + ([1] * 3 + [0] * 17)
    attr = ["A"] * 20 + ["B"] * 20
    return np.array(y_true), np.array(y_pred), np.array(attr, dtype=object)


# ===========================================================================
# Finding 1: missing_strategy='impute'
# ===========================================================================


class TestImputeWasNeverAValue:
    def test_refusal_pin_the_code_refuses_impute_and_names_the_real_set(self):
        """REFUSAL PIN. The documented value raises, and says what is accepted."""
        y_true, y_pred, attr = _regression_fixture(with_missing=False)
        with pytest.raises(ConfigurationError) as excinfo:
            regression_fairness_report(y_true, y_pred, attr, missing_strategy="impute")
        assert "impute" in str(excinfo.value)
        assert _accepted_from_refusal_message(excinfo.value) == {
            "exclude",
            "as_group",
            "error",
        }

    def test_refusal_pin_the_page_no_longer_offers_impute(self):
        """REFUSAL PIN. Every value the page quotes for this parameter is one
        the code accepts, or sits in a description that says it is refused."""
        page = _text(_API_HTML)
        section = _function_section(page, "regression_fairness_report")
        desc = _documented_params(section)["missing_strategy"]

        y_true, y_pred, attr = _regression_fixture(with_missing=False)
        with pytest.raises(ConfigurationError) as excinfo:
            regression_fairness_report(y_true, y_pred, attr, missing_strategy="impute")
        accepted = _accepted_from_refusal_message(excinfo.value)

        offered = _quoted_literals(desc)
        refused = offered - accepted
        assert refused <= {"impute"}, f"page quotes unknown values {sorted(refused)}"
        if refused:
            assert _has_refusal_cue(desc), (
                "the page quotes a value the code refuses without saying so: " + desc
            )

    def test_control_all_three_real_strategies_are_documented_and_measure(self):
        """OVER-CORRECTION CONTROL. Deleting the parameter, or narrowing it to
        'exclude', would be an easy way to make the pin above green. All three
        accepted values must still be offered on the page AND still produce
        their own distinct MEASURED number."""
        page = _text(_API_HTML)
        section = _function_section(page, "regression_fairness_report")
        offered = _quoted_literals(_documented_params(section)["missing_strategy"])
        assert {"exclude", "as_group", "error"} <= offered

        clean = _regression_fixture(with_missing=False)
        dirty = _regression_fixture(with_missing=True)

        # No missing rows: all three strategies agree, and the gap is exactly
        # 1.00 (group A) minus 0.25 (group B).
        for strategy in ("exclude", "as_group", "error"):
            report = regression_fairness_report(
                *clean, min_group_size=30, missing_strategy=strategy
            )
            assert report["metrics"]["mae_parity_difference"] == pytest.approx(0.75, abs=1e-12)
            assert report["data_info"]["valid_groups"] == ["A", "B"]
            assert report["data_info"]["final_size"] == 80

        # 35 rows with no sensitive attribute: the strategies now DIVERGE, and
        # the numbers are what tells them apart.
        excluded = regression_fairness_report(*dirty, min_group_size=30, missing_strategy="exclude")
        assert excluded["metrics"]["mae_parity_difference"] == pytest.approx(0.75, abs=1e-12)
        assert excluded["data_info"]["n_excluded"] == 35
        assert excluded["data_info"]["valid_groups"] == ["A", "B"]

        grouped = regression_fairness_report(*dirty, min_group_size=30, missing_strategy="as_group")
        assert grouped["metrics"]["mae_parity_difference"] == pytest.approx(1.75, abs=1e-12)
        assert grouped["data_info"]["n_excluded"] == 0
        assert grouped["data_info"]["final_size"] == 115
        assert grouped["data_info"]["valid_groups"] == ["A", "B", "__missing__"]

        with pytest.raises(InvalidDataError) as excinfo:
            regression_fairness_report(*dirty, min_group_size=30, missing_strategy="error")
        assert "35 rows with missing values" in str(excinfo.value)


# ===========================================================================
# Finding 2: multiple_testing_correction='fdr_bh'
# ===========================================================================


class TestFdrBhBelongsToTheOtherFunction:
    def test_refusal_pin_the_report_refuses_fdr_bh(self):
        """REFUSAL PIN."""
        y_true, y_pred, attr = _classification_fixture()
        with pytest.raises(ConfigurationError) as excinfo:
            classification_fairness_report(
                y_true,
                y_pred,
                attr,
                include_ci=True,
                n_bootstrap=200,
                multiple_testing_correction="fdr_bh",
                random_state=7,
            )
        assert "Unknown correction method: fdr_bh" in str(excinfo.value)

    def test_refusal_pin_the_page_offers_only_what_the_signature_declares(self):
        """REFUSAL PIN. The values quoted for this parameter are the Literal
        options, and any extra quoted value sits in a description that says it
        is refused."""
        page = _text(_API_HTML)
        section = _function_section(page, "classification_fairness_report")
        desc = _documented_params(section)["multiple_testing_correction"]

        declared = _literal_options(classification_fairness_report, "multiple_testing_correction")
        assert declared == {"bonferroni", "fdr", "none"}

        offered = _quoted_literals(desc)
        assert declared <= offered, f"page omits {sorted(declared - offered)}"
        extra = offered - declared
        assert extra <= {"fdr_bh"}, f"page quotes unknown values {sorted(extra)}"
        if extra:
            assert _has_refusal_cue(desc), (
                "the page quotes 'fdr_bh' for this parameter without saying it is refused: " + desc
            )

    def test_control_the_two_real_corrections_produce_measured_p_values(self):
        """OVER-CORRECTION CONTROL. Both corrections still run, and they give
        DIFFERENT adjusted p-values on the same three tests: 'bonferroni'
        multiplies the smallest raw p (8.6952e-06) by 3, Benjamini-Hochberg by
        3/2. A fix that disabled the correction, or collapsed the two methods
        into one, is red here.

        UPDATED 2026-09-10 (READINESS-6). This fixture used to pin a family of
        FIVE with the p-vector
        ``[0.3118042508, 0.3118042508, 8.6952e-06, 5.154e-05, 8.6952e-06]``, and
        those repeats are the defect, not a property of the data:
        ``demographic_parity_difference`` and ``demographic_parity_ratio`` are
        the gap and the ratio of the identical pair of group selection rates,
        and ``equalized_odds_difference`` reuses the TPR p-value whenever the
        TPR gap is its larger one. Correcting for one hypothesis three times
        suppresses real findings (measured on another fixture the same day: a
        demographic-parity finding at p=0.016225 that m=5 buries and the honest
        m=3 reports). The family is now one member per DISTINCT hypothesis and
        the folded metrics are disclosed in
        ``metrics_sharing_a_tested_hypothesis``. ``n_rejected`` moves 3 -> 2 for
        the same reason: the two REAL rejections (the TPR gap and the PPV gap)
        are unchanged, and the third was the TPR gap counted twice. The
        control's own point, that the two methods still differ, is asserted
        below on the numbers that survive. Pinned in
        tests/test_readiness6_metrics.py."""
        y_true, y_pred, attr = _classification_fixture()

        bonf = classification_fairness_report(
            y_true,
            y_pred,
            attr,
            include_ci=True,
            n_bootstrap=200,
            multiple_testing_correction="bonferroni",
            random_state=7,
        )["multiple_testing_correction"]
        fdr = classification_fairness_report(
            y_true,
            y_pred,
            attr,
            include_ci=True,
            n_bootstrap=200,
            multiple_testing_correction="fdr",
            random_state=7,
        )["multiple_testing_correction"]

        assert bonf["method"] == "bonferroni"
        assert fdr["method"] == "benjamini_hochberg"
        assert bonf["tested_metrics"] == fdr["tested_metrics"]
        assert bonf["tested_metrics"] == [
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "predictive_parity_difference",
        ]
        # The two folded metrics are named, not dropped in silence.
        assert bonf["metrics_sharing_a_tested_hypothesis"] == {
            "demographic_parity_difference": ["demographic_parity_ratio"],
            "equal_opportunity_difference": ["equalized_odds_difference"],
        }

        raw = [float(p) for p in bonf["original_p_values"]]
        assert raw == pytest.approx([0.3118042508, 8.6952e-06, 5.154e-05], rel=1e-4)
        assert [float(p) for p in bonf["adjusted_p_values"]] == pytest.approx(
            [0.9354127523, 2.60857e-05, 0.0001546200], rel=1e-4
        )
        assert [float(p) for p in fdr["adjusted_p_values"]] == pytest.approx(
            [0.3118042508, 2.60857e-05, 7.731e-05], rel=1e-4
        )
        assert bonf["n_rejected"] == 2 and fdr["n_rejected"] == 2
        # The whole point of the control: the two methods still DIFFER.
        assert bonf["adjusted_p_values"][2] != fdr["adjusted_p_values"][2]

    def test_control_fdr_bh_still_works_where_it_really_belongs(self):
        """OVER-CORRECTION CONTROL. 'fdr_bh' IS a real value of the
        neighbouring parameter. Sweeping it out of the page, or out of the
        code, would break a working call. Asserted by executing it and by
        checking the page still documents it there."""
        frame = pd.DataFrame(
            {
                "gender": ["M"] * 200 + ["F"] * 200,
                "approved": [1] * 160 + [0] * 40 + [1] * 40 + [0] * 160,
            }
        )
        results = analyze_statistical_disparities(
            frame,
            ["gender"],
            outcome_columns=["approved"],
            include_quality_analysis=False,
            include_intersectional=False,
            correction_method="fdr_bh",
        )
        assert len(results) == 1
        found = results[0]
        assert found.feature == "approved"
        assert found.protected_attribute == "gender"
        assert found.pvalue == pytest.approx(1.1834715432612276e-32, rel=1e-6)
        assert float(found.effect_size) == pytest.approx(0.595, abs=1e-6)

        page = _text(_API_HTML)
        section = _function_section(page, "analyze_statistical_disparities")
        desc = _documented_params(section)["correction_method"]
        assert "fdr_bh" in _quoted_literals(desc), (
            "the page stopped documenting 'fdr_bh' for the parameter that really does accept it"
        )


# ===========================================================================
# Finding 3: the Bayesian disparity interval that does not exist
# ===========================================================================

# The exact sentences that were shipped, one per site. A pin that only checked
# behaviour would stay green while the page went back to promising the feature.
_FALSE_CLAIMS: Tuple[Tuple[pathlib.Path, str], ...] = (
    (_API_HTML, "<code>'exclude'</code> or <code>'impute'</code>"),
    (_API_HTML, "(e.g., <code>'bonferroni'</code>, <code>'fdr_bh'</code>)"),
    (_API_HTML, "wrap the base metric with a bootstrap or Bayesian interval"),
    (_API_HTML, "Bootstrap / Bayesian credible-interval visualization"),
    (
        _API_HTML,
        "CI method: <code>'auto'</code>, <code>'bootstrap'</code>, or <code>'bayesian'</code>.",
    ),
    (_METHODOLOGY_MD, "switches to Bayesian below the small-sample"),
    (_OVERVIEW_MD, "Bootstrap CI (n ≥ 30) or Bayesian CI (n < 30)"),
    (_API_MD, "BCa or Bayesian small-n interval"),
    (
        _CONCEPTS_HTML,
        "Use Bayesian confidence intervals (<code>method='bayesian'</code>) for small samples.",
    ),
    (
        _BUSINESS_HTML,
        "Use Bayesian methods (<code>method='bayesian'</code>) for small groups.",
    ),
    (_STATISTICS_PY, "- Bayesian credible intervals for small sample groups"),
)

# The functions the library really does provide. A sweeping "delete every
# mention of Bayesian" fix is red on this list.
_REAL_BAYESIAN_FUNCTIONS = (
    "bayesian_proportion_ci",
    "bayesian_difference_ci",
    "bayesian_mean_ci",
)

# Which of them each site names, COUNTED on 2026-09-10 after the corrections.
# An "at least one of them survives" check was too weak: it stayed green when
# one name was deleted from METHODOLOGY.md, because two others were still
# there. Naming the expected set per file makes any single deletion red.
_EXPECTED_ESTIMATOR_MENTIONS: Dict[pathlib.Path, Set[str]] = {
    _API_HTML: {"bayesian_proportion_ci", "bayesian_difference_ci", "bayesian_mean_ci"},
    _METHODOLOGY_MD: {
        "bayesian_proportion_ci",
        "bayesian_difference_ci",
        "bayesian_mean_ci",
        "get_group_metrics_with_ci",
    },
    _OVERVIEW_MD: {
        "bayesian_proportion_ci",
        "bayesian_difference_ci",
        "bayesian_mean_ci",
        "get_group_metrics_with_ci",
    },
    _API_MD: {"bayesian_proportion_ci", "bayesian_difference_ci"},
    _CONCEPTS_HTML: {
        "bayesian_proportion_ci",
        "bayesian_difference_ci",
        "bayesian_mean_ci",
        "get_group_metrics_with_ci",
    },
    _BUSINESS_HTML: {
        "bayesian_proportion_ci",
        "bayesian_difference_ci",
        "get_group_metrics_with_ci",
    },
    _STATISTICS_PY: {
        "bayesian_proportion_ci",
        "bayesian_difference_ci",
        "bayesian_mean_ci",
        "get_group_metrics_with_ci",
    },
}


class TestBayesianDisparityIntervalDoesNotExist:
    @pytest.mark.parametrize(
        "path,claim",
        _FALSE_CLAIMS,
        ids=[f"{p.name}:{c[:36]}" for p, c in _FALSE_CLAIMS],
    )
    def test_refusal_pin_the_false_sentence_is_not_shipped(self, path, claim):
        """REFUSAL PIN, one per published site."""
        assert claim not in _normalised(path), (
            f"{path.relative_to(_LIB_ROOT)} ships a claim the code refuses: {claim!r}"
        )

    def test_refusal_pin_asking_for_bayesian_returns_a_bootstrap_and_says_so(self):
        """REFUSAL PIN. Three states, not two: the request is neither honoured
        silently nor rejected. It is answered with a bootstrap, a warning that
        names the gap, and a result whose provenance field cannot be mistaken
        for a credible interval."""
        y_true, y_pred, attr = _small_group_fixture()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_metric_with_ci(
                demographic_parity_difference,
                y_true,
                y_pred,
                attr,
                n_bootstrap=200,
                method="bayesian",
                random_state=1,
                min_group_size=5,
            )
        messages = " ".join(str(w.message) for w in caught)
        assert "not implemented" in messages, messages
        assert "Bayesian" in messages, messages

        assert "bootstrap" in result.method
        assert "bayesian" not in result.method
        assert result.interval_type is IntervalType.CONFIDENCE
        assert np.isfinite(result.point_estimate)

        # 'auto' on the same small sample must not quietly claim otherwise.
        with warnings.catch_warnings(record=True) as caught_auto:
            warnings.simplefilter("always")
            auto = compute_metric_with_ci(
                demographic_parity_difference,
                y_true,
                y_pred,
                attr,
                n_bootstrap=200,
                method="auto",
                random_state=1,
                min_group_size=5,
            )
        auto_messages = " ".join(str(w.message) for w in caught_auto)
        assert "Bayesian interval is not implemented" in auto_messages, auto_messages
        assert auto.interval_type is IntervalType.CONFIDENCE
        assert "bayesian" not in auto.method

    def test_refusal_pin_prefer_bayesian_is_refused_not_ignored(self):
        """REFUSAL PIN. The size-based recommender refuses to be steered
        towards a method it has no estimator for."""
        with pytest.raises(ConfigurationError) as excinfo:
            select_method({"A": 20, "B": 20}, prefer_bayesian=True)
        assert "no Bayesian estimator for a disparity metric" in str(excinfo.value)

    def test_refusal_pin_the_report_ci_method_docs_admit_bayesian_is_unbuilt(self):
        """REFUSAL PIN. Both report entries quote 'bayesian' as an accepted
        argument, which it is. Each must also say it is not implemented."""
        page = _text(_API_HTML)
        for func, name in (
            (classification_fairness_report, "classification_fairness_report"),
            (regression_fairness_report, "regression_fairness_report"),
        ):
            declared = _literal_options(func, "ci_method")
            assert declared == {"auto", "bootstrap", "bayesian"}
            desc = _documented_params(_function_section(page, name))["ci_method"]
            assert "bayesian" in _quoted_literals(desc), name
            assert _has_refusal_cue(desc), f"{name} ci_method offers 'bayesian' bare: {desc}"

    def test_control_the_real_bayesian_estimators_still_measure(self):
        """OVER-CORRECTION CONTROL. Beta-Binomial and Normal-Normal posteriors
        are closed form, so these are exact numbers, not tolerances of
        convenience. Removing any of these functions, or turning them into
        bootstraps, is red."""
        prop = bayesian_proportion_ci(successes=45, trials=100, confidence_level=0.95)
        assert prop.method == "bayesian_beta_binomial"
        assert prop.interval_type is IntervalType.CREDIBLE
        # Posterior mean of Beta(1 + 45, 1 + 55) is exactly 46/102.
        assert prop.point_estimate == pytest.approx(46.0 / 102.0, abs=1e-12)
        assert prop.lower_bound == pytest.approx(0.35600159811478516, abs=1e-9)
        assert prop.upper_bound == pytest.approx(0.5477807813438309, abs=1e-9)
        assert prop.metadata["posterior"] == "Beta(46.0, 56.0)"

        diff = bayesian_difference_ci(
            successes1=45,
            trials1=100,
            successes2=30,
            trials2=100,
            confidence_level=0.95,
            random_state=42,
        )
        assert diff.method == "bayesian_difference_monte_carlo"
        assert diff.interval_type is IntervalType.CREDIBLE
        # Analytic posterior-mean gap: 46/102 - 31/102 = 15/102.
        assert diff.point_estimate == pytest.approx(15.0 / 102.0, abs=0.01)
        assert diff.lower_bound > 0.0, "the 95% credible interval must exclude 0 here"
        assert diff.upper_bound == pytest.approx(0.275, abs=0.01)
        assert float(diff.metadata["p_greater"]) > 0.98
        repeat = bayesian_difference_ci(
            successes1=45,
            trials1=100,
            successes2=30,
            trials2=100,
            confidence_level=0.95,
            random_state=42,
        )
        assert repeat.point_estimate == diff.point_estimate

        mean = bayesian_mean_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), random_state=0)
        assert mean.method == "bayesian_normal"
        assert mean.interval_type is IntervalType.CREDIBLE
        assert mean.point_estimate == pytest.approx(2.9850746268656723, abs=1e-9)

    def test_control_per_group_rates_really_do_get_a_credible_interval(self):
        """OVER-CORRECTION CONTROL. This is the half of the promise that was
        TRUE, and the docs still send readers here. The group rate, not the
        gap, is where method='bayesian' is honoured."""
        y_true, y_pred, attr = _classification_fixture()
        groups = get_group_metrics_with_ci(
            y_true, y_pred, attr, method="bayesian", min_group_size=30
        )
        assert set(groups) == {"A", "B"}

        rate_a = groups["A"]["positive_rate"]
        assert rate_a.method == "bayesian_beta_binomial"
        assert rate_a.interval_type is IntervalType.CREDIBLE
        assert rate_a.point_estimate == pytest.approx(0.45, abs=1e-12)
        assert rate_a.lower_bound == pytest.approx(0.38258936705080276, abs=1e-9)
        assert rate_a.upper_bound == pytest.approx(0.5193294595268572, abs=1e-9)
        assert rate_a.metadata["posterior"] == "Beta(91.0, 111.0)"

        rate_b = groups["B"]["positive_rate"]
        assert rate_b.method == "bayesian_beta_binomial"
        assert rate_b.point_estimate == pytest.approx(0.40, abs=1e-12)
        assert rate_b.metadata["posterior"] == "Beta(81.0, 121.0)"

    def test_control_select_method_still_recommends_bayesian_for_small_groups(self):
        """OVER-CORRECTION CONTROL. select_method returns a RECOMMENDATION.
        Silencing it, so that nothing anywhere says 'bayesian', would destroy
        a real signal about sample size."""
        assert select_method({"A": 20, "B": 20}) == (
            "bayesian",
            {"A": "bayesian", "B": "bayesian"},
        )
        assert select_method({"A": 20, "B": 200}) == (
            "mixed",
            {"A": "bayesian", "B": "bootstrap"},
        )

    def test_control_every_site_still_names_the_estimators_that_exist(self):
        """OVER-CORRECTION CONTROL. The corrected pages must send the reader
        somewhere real, not merely delete the promise. The expected set is
        named per file, so deleting ONE estimator from a page that lists
        several is red; an "at least one survives" check was not."""
        for path, expected in _EXPECTED_ESTIMATOR_MENTIONS.items():
            body = _reader_surface(path)
            missing = sorted(name for name in expected if name not in body)
            assert not missing, (
                f"{path.relative_to(_LIB_ROOT)} no longer names {missing}: the "
                "correction deleted a Bayesian estimator that really exists"
            )
            assert any(fn in body for fn in _REAL_BAYESIAN_FUNCTIONS), (
                f"{path.relative_to(_LIB_ROOT)} names no Bayesian estimator at all"
            )

    def test_control_the_documented_bayesian_example_runs_from_the_page(self):
        """OVER-CORRECTION CONTROL. The example is located IN docs/API_REFERENCE.md
        and executed, so this cannot pass against a copy no reader will see."""
        raw = _text(_API_MD)
        blocks = re.findall(r"^```[^\n]*\n(.*?)^```", raw, re.M | re.S)
        hits = [b for b in blocks if "bayesian_proportion_ci(successes=45" in b]
        assert len(hits) == 1, f"{len(hits)} published blocks carry the example"

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(compile(hits[0], "<published-docs>", "exec"), {})  # noqa: S102
        printed = buffer.getvalue()
        assert "Posterior rate: 0.451 [0.356, 0.548]" in printed, printed
        assert "Rate difference: 0.146 [0.015, 0.275]" in printed, printed

    def test_control_the_sealed_gates_still_return_a_measured_interval(self):
        """OVER-CORRECTION CONTROL. The correction to the docs was to say these
        are bootstraps. They must still BE intervals, with finite measured
        bounds, and must still name the estimator that ran."""
        from vfairness import (
            disparate_impact_ratio_with_ci,
            equalized_odds_difference_with_ci,
        )

        y_true, y_pred, attr = _classification_fixture()
        for func in (disparate_impact_ratio_with_ci, equalized_odds_difference_with_ci):
            result = func(y_true, y_pred, attr, n_bootstrap=200, random_state=3)
            assert result.method.startswith("stratified_bootstrap"), result.method
            assert result.interval_type is IntervalType.CONFIDENCE
            assert np.isfinite(result.point_estimate)
            assert np.isfinite(result.lower_bound) and np.isfinite(result.upper_bound)
            assert result.lower_bound <= result.point_estimate <= result.upper_bound

        eod = equalized_odds_difference_with_ci(
            y_true, y_pred, attr, n_bootstrap=200, random_state=3
        )
        # TPR gap 0.30, FPR gap 0.20; the max-min spread statistic is 0.30.
        assert eod.point_estimate == pytest.approx(0.30, abs=1e-12)
