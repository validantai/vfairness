"""Audit 6, wave 3, finding S-03: total exclusion reported as perfect parity.

``_norm_int_group`` built the Pulse intersectional payload with::

    "relativeToOverall": pick("relative_to_overall", ...) or 1.0,
    "relativeToBest":    pick("relative_to_best", ...)    or 1.0,

``pick`` already answers ``None`` when the key is absent or holds ``None``, so
the ``or 1.0`` bought nothing for the missing case and instead fired
ADDITIONALLY on a MEASURED ``0.0``. On this scale 0.0 and 1.0 are opposite
ends: ``relative_to_best == 0.0`` is a group that received ZERO positive
outcomes (total exclusion, the strongest disparate-impact reading there is,
which is why ``intersectional.py`` keeps ``zero_selection_alerts``), and 1.0 is
exact parity with the best group. The single worst measurement was published as
the single best.

Three states, never two:
  * measured 0.0  -> 0.0 (and never 1.0)
  * measured x    -> x, byte-for-byte unchanged
  * not reported  -> None, plus a warning, so the consumer can render
    could-not-check instead of a parity claim.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.intersectional import (
    intersectional_disparity_analysis,
)
from vfairness.operations.pulse.orchestrator import _norm_int_group, _normalise_intersectional

# --------------------------------------------------------------------------
# 1. REFUSAL PINS: a measured 0.0 must survive as 0.0
# --------------------------------------------------------------------------


def test_measured_zero_relative_to_best_is_not_reported_as_parity():
    """The regression pin. 0.0 in, 0.0 out, and specifically NOT 1.0."""
    out = _norm_int_group(
        {
            "group": "female_over50",
            "positive_rate": 0.0,
            "size": 240,
            "relative_to_overall": 0.0,
            "relative_to_best": 0.0,
            "severity": "critical",
        }
    )
    assert out is not None
    assert out["relativeToBest"] == 0.0
    assert out["relativeToOverall"] == 0.0
    # Explicit: the defect published the opposite end of the scale.
    assert out["relativeToBest"] != 1.0
    assert out["relativeToOverall"] != 1.0


def test_measured_zero_survives_the_camelcase_spelling_too():
    out = _norm_int_group(
        {"group": "cell", "relativeToOverall": 0.0, "relativeToBest": 0.0, "size": 55}
    )
    assert out is not None
    assert out["relativeToOverall"] == 0.0
    assert out["relativeToBest"] == 0.0


def test_measured_zero_survives_a_dataclass_group_advantage():
    """Pulse calls the engine in-process, so it gets GroupAdvantage instances,
    not dicts. The zero must survive that path as well."""
    from vfairness.evaluation.vfairness_metrics.intersectional import GroupAdvantage

    ga = GroupAdvantage(
        group="female_over50",
        positive_rate=0.0,
        size=240,
        relative_to_overall=0.0,
        relative_to_best=0.0,
        disparity_contribution=0.42,
        severity="critical",
    )
    out = _norm_int_group(ga)
    assert out is not None
    assert out["relativeToBest"] == 0.0
    assert out["relativeToOverall"] == 0.0


def test_zero_is_carried_through_the_full_intersectional_payload():
    """Not the helper in isolation: the whole ``intersectional`` block that
    Pulse hands to the platform."""
    payload = _normalise_intersectional(
        {
            "intersectional_analysis": {
                "privileged_group": {
                    "group": "male_under30",
                    "positive_rate": 0.5,
                    "size": 400,
                    "relative_to_overall": 1.4,
                    "relative_to_best": 1.0,
                },
                "disadvantaged_group": {
                    "group": "female_over50",
                    "positive_rate": 0.0,
                    "size": 240,
                    "relative_to_overall": 0.0,
                    "relative_to_best": 0.0,
                },
                "all_groups": [
                    {
                        "group": "female_over50",
                        "positive_rate": 0.0,
                        "size": 240,
                        "relative_to_overall": 0.0,
                        "relative_to_best": 0.0,
                    }
                ],
                "overall_rate": 0.35,
                "max_disparity": 0.5,
            }
        },
        label_free=False,
    )
    assert payload["disadvantagedGroup"]["relativeToBest"] == 0.0
    assert payload["disadvantagedGroup"]["relativeToOverall"] == 0.0
    assert payload["allGroups"][0]["relativeToBest"] == 0.0
    # Over-correction control inside the same payload: the privileged group's
    # real ratios are untouched.
    assert payload["privilegedGroup"]["relativeToOverall"] == 1.4
    assert payload["privilegedGroup"]["relativeToBest"] == 1.0


def test_end_to_end_a_totally_excluded_cell_reaches_the_payload_as_zero():
    """Real engine, real data, no hand-made ratio: one cell gets zero positive
    predictions, so ``relative_to_best`` is legitimately 0.0."""
    rng = np.random.default_rng(20260909)
    n_per = 120
    cells = ["male_under30", "male_over50", "female_under30", "female_over50"]
    sensitive = np.repeat(cells, n_per)
    y_pred = np.concatenate(
        [
            (rng.random(n_per) < 0.60).astype(int),  # male_under30
            (rng.random(n_per) < 0.45).astype(int),  # male_over50
            (rng.random(n_per) < 0.30).astype(int),  # female_under30
            np.zeros(n_per, dtype=int),  # female_over50: TOTAL EXCLUSION
        ]
    )
    y_true = (rng.random(n_per * 4) < 0.45).astype(int)

    raw = intersectional_disparity_analysis(y_true, y_pred, sensitive, min_group_size=30)
    engine_groups = {g.group: g for g in raw["intersectional_analysis"]["all_groups"]}
    assert engine_groups["female_over50"].positive_rate == 0.0
    assert engine_groups["female_over50"].relative_to_best == 0.0

    payload = _normalise_intersectional(raw, label_free=False)
    excluded = [g for g in payload["allGroups"] if g["group"] == "female_over50"]
    assert len(excluded) == 1
    assert excluded[0]["relativeToBest"] == 0.0, (
        "the engine measured total exclusion; the Pulse payload must not publish it as parity"
    )
    assert excluded[0]["relativeToOverall"] == 0.0
    assert excluded[0]["relativeToBest"] != 1.0

    # Over-correction control on the SAME end-to-end run: the other three
    # cells keep the exact ratios the engine computed for them.
    for name in ("male_under30", "male_over50", "female_under30"):
        got = [g for g in payload["allGroups"] if g["group"] == name][0]
        assert got["relativeToBest"] == pytest.approx(
            engine_groups[name].relative_to_best, abs=1e-12
        )
        assert got["relativeToOverall"] == pytest.approx(
            engine_groups[name].relative_to_overall, abs=1e-12
        )
        assert got["relativeToBest"] > 0.0


# --------------------------------------------------------------------------
# 2. THE MISSING CASE: None, not 1.0
# --------------------------------------------------------------------------


def test_missing_ratios_are_none_not_a_parity_claim():
    """Decision: an unreported ratio is could-not-check, so it leaves as None.
    1.0 for "we do not know" is the same defect in a quieter form, and this
    library already made that call for the same field in
    rendering/adapters_discovery.py ("1.0 was the worst default on this
    canvas")."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _norm_int_group({"group": "unknown_cell", "size": 12})
    assert out is not None
    assert out["relativeToOverall"] is None
    assert out["relativeToBest"] is None


def test_explicit_null_ratio_is_also_none():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _norm_int_group({"group": "c", "relative_to_overall": None, "relative_to_best": None})
    assert out is not None
    assert out["relativeToOverall"] is None
    assert out["relativeToBest"] is None


def test_the_missing_case_is_named_in_a_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _norm_int_group({"group": "unknown_cell", "size": 12})
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(msgs) == 1, msgs
    assert "unknown_cell" in msgs[0]
    assert "relativeToOverall" in msgs[0]
    assert "relativeToBest" in msgs[0]


def test_a_measured_zero_raises_no_missing_warning():
    """The warning is about absence. A measured 0.0 is a measurement and must
    not be announced as unreported."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _norm_int_group(
            {"group": "female_over50", "relative_to_overall": 0.0, "relative_to_best": 0.0}
        )
    assert out is not None
    assert out["relativeToBest"] == 0.0
    assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []


def test_one_missing_one_measured_keeps_them_apart():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _norm_int_group({"group": "half", "relative_to_best": 0.0})
    assert out is not None
    assert out["relativeToBest"] == 0.0
    assert out["relativeToOverall"] is None
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(msgs) == 1, msgs
    assert "relativeToOverall" in msgs[0]
    assert "relativeToBest" not in msgs[0]


# --------------------------------------------------------------------------
# 3. OVER-CORRECTION CONTROLS: ordinary measured ratios pass through unchanged
# --------------------------------------------------------------------------


def test_ordinary_measured_ratios_pass_through_as_the_exact_numbers():
    out = _norm_int_group(
        {
            "group": "male_under30",
            "positive_rate": 0.42,
            "size": 900,
            "relative_to_overall": 1.23,
            "relative_to_best": 0.87,
            "disparity_contribution": 0.07,
            "severity": "medium",
        }
    )
    assert out is not None
    assert out["relativeToOverall"] == 1.23
    assert out["relativeToBest"] == 0.87
    # The neighbours the fix touched nothing of.
    assert out["group"] == "male_under30"
    assert out["positiveRate"] == 0.42
    assert out["size"] == 900
    assert out["disparityContribution"] == 0.07
    assert out["severity"] == "medium"
    assert out["lowNWarning"] is False


@pytest.mark.parametrize(
    "ratio",
    [0.0, 0.004, 0.5, 0.7999, 0.8, 1.0, 1.0000001, 2.5, 17.0],
)
def test_every_ratio_including_exact_parity_is_returned_verbatim(ratio):
    """A refusal that swallowed the ratio, or rounded it, or clamped it, would
    fail here. Exact parity (1.0) must still read 1.0."""
    out = _norm_int_group({"group": "g", "relative_to_overall": ratio, "relative_to_best": ratio})
    assert out is not None
    assert out["relativeToOverall"] == ratio
    assert out["relativeToBest"] == ratio
    assert isinstance(out["relativeToBest"], float)


def test_numpy_floats_survive_including_a_numpy_zero():
    out = _norm_int_group(
        {
            "group": "np_cell",
            "relative_to_overall": np.float64(0.0),
            "relative_to_best": np.float64(0.6),
        }
    )
    assert out is not None
    assert float(out["relativeToOverall"]) == 0.0
    assert float(out["relativeToOverall"]) != 1.0
    assert float(out["relativeToBest"]) == pytest.approx(0.6)
