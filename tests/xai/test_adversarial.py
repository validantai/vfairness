"""
Adversarial-probe productionisation tests (backlog #P2-08).

Acceptance: on a scaffolded biased model (Slack-style OOD construction)
the flag returns True with confidence > 0.9; on a clean model the flag
returns False.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.xai.diagnostics import (
    AdversarialProbeResult,
    multi_seed_adversarial_probe,
)


def _background_on_manifold(n: int = 400) -> np.ndarray:
    """Points living on the line x1 == x0 (a 1-D manifold in 2-D)."""
    rng = np.random.RandomState(7)
    a = rng.normal(0, 1, n)
    return np.column_stack([a, a])


def _scaffolded_predict(arr: np.ndarray) -> np.ndarray:
    """Slack-style scaffold: 'innocent' (1.0) on the data manifold,
    'true' biased behaviour (0.0) off it. A perturbation that breaks the
    x1==x0 correlation lands off-manifold and is detected."""
    arr = np.asarray(arr)
    on_manifold = np.abs(arr[:, 1] - arr[:, 0]) < 0.5
    return on_manifold.astype(float)


def _clean_predict(arr: np.ndarray) -> np.ndarray:
    """Honest model: depends only on x0, no OOD sensitivity."""
    arr = np.asarray(arr)
    return (arr[:, 0] > 0).astype(float)


def test_probe_fires_with_high_confidence_on_scaffolded_model() -> None:
    bg = _background_on_manifold()
    x = bg[0]
    result = multi_seed_adversarial_probe(
        predict_fn=_scaffolded_predict,
        x=x,
        background=bg,
        n_perturbations=200,
        threshold=0.3,
        n_seeds=10,
    )
    assert isinstance(result, AdversarialProbeResult)
    assert result.flag is True
    assert result.confidence > 0.9
    assert result.reason  # non-empty


def test_probe_clean_on_honest_model() -> None:
    bg = _background_on_manifold()
    x = bg[0]
    result = multi_seed_adversarial_probe(
        predict_fn=_clean_predict,
        x=x,
        background=bg,
        n_perturbations=200,
        threshold=0.3,
        n_seeds=10,
    )
    assert result.flag is False
    assert result.reason == ""


def test_single_shot_probe_still_works() -> None:
    bg = _background_on_manifold()
    flag, reason = __import__(
        "vfairness.xai.diagnostics", fromlist=["slack_adversarial_probe"]
    ).slack_adversarial_probe(
        predict_fn=_scaffolded_predict,
        x=bg[0],
        background=bg,
        threshold=0.3,
    )
    assert flag is True and reason


# ---------------------------------------------------------------------------
# A THRESHOLD THE COMPARISON CANNOT DISAGREE WITH IS NOT A BOUND.
#
# B4 tier-1 audit, 2026-09-30. ``_probe_gap`` returns ``abs(...)``, so every gap
# is >= 0 by construction and ``gap > threshold`` for a NEGATIVE threshold cannot
# be False for any data in the statistic's range. Both entry points tested only
# ``np.isfinite(threshold)``, which leaves the FINITE vacuous one wide open.
#
# Measured on this tree by loading HEAD's adversarial.py beside the fixed one in
# one process, on the fixture below, threshold -0.5:
#
#   HEAD,  the CLEAN flat model -> multi flag True, confidence 1.0,
#          fired_fraction 1.0, mean_gap 0.0, ZERO warnings; slack (True, 'Slack
#          probe: perturbation-mean prediction differs from background by 0.000
#          (threshold -0.5). Likely OOD-scaffolding detection. Treat SHAP/LIME
#          outputs as suspect ...')
#   FIXED, the same model       -> multi flag None, confidence/mean_gap/
#          fired_fraction NaN, 2 warnings; slack (None, 'COULD NOT CHECK: ...')
#
# A named scaffolding finding with remediation advice, at maximum confidence,
# off a gap of exactly 0.000, against a model that has nothing to hide.
#
# The fixture and the refusal wording are shared with the twin,
# vfairness.evaluation.vfairness_metrics.explanation_diagnostics, whose pins live
# in tests/test_bgl_w4_eval_rest.py. Two surfaces answering one question must not
# answer it twice, and this file's own source comment used to say the twin carried
# the defect while the twin's said the same about this one.
# ---------------------------------------------------------------------------


def _probe_world():
    """A real 400-row background, a scaffolded model and a clean one.

    The same world the twin's pins use. Scaffolding is spelled as exact
    membership of the background, so the perturbation cloud is off-manifold by
    construction and the gap is a clean 0.8, while the flat model's gap is
    exactly 0.0 rather than rounding noise.
    """
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(400, 3))

    def scaffolded(arr):
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        on_manifold = np.any(np.all(np.isclose(arr[:, None, :], bg[None, :, :]), axis=2), axis=1)
        p = np.where(on_manifold, 0.1, 0.9)
        return np.column_stack([1 - p, p])

    def clean(arr):
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        p = np.full(len(arr), 0.5)
        return np.column_stack([1 - p, p])

    return bg, bg[0], scaffolded, clean


def _both_probes(model, bg, x, threshold):
    """Run both public entry points on one input, warnings captured."""
    import warnings

    from vfairness.xai.diagnostics.adversarial import slack_adversarial_probe

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        multi = multi_seed_adversarial_probe(
            predict_fn=model,
            x=x,
            background=bg,
            n_perturbations=60,
            n_seeds=4,
            threshold=threshold,
        )
        single = slack_adversarial_probe(
            predict_fn=model,
            x=x,
            background=bg,
            n_perturbations=60,
            threshold=threshold,
            seed=7,
        )
    return multi, single, [str(c.message) for c in caught]


@pytest.mark.parametrize("model_name", ["clean", "scaffolded"])
def test_a_negative_threshold_is_refused_by_both_entry_points(model_name) -> None:
    """REFUSAL PIN. ``gap > -0.5`` cannot be False, so neither probe could have
    disagreed with itself, and the CLEAN model was flagged at confidence 1.0."""
    bg, x, scaffolded, clean = _probe_world()
    model = clean if model_name == "clean" else scaffolded

    multi, single, messages = _both_probes(model, bg, x, -0.5)

    assert multi.flag is None, multi
    assert np.isnan(multi.confidence)
    assert np.isnan(multi.mean_gap)
    assert np.isnan(multi.fired_fraction)
    assert "never below zero" in multi.reason
    assert "COULD NOT CHECK" in multi.reason

    assert single[0] is None, single
    assert "never below zero" in single[1]
    assert "COULD NOT CHECK" in single[1]

    # A refusal nobody can see is not a disclosure: one warning per probe.
    assert len(messages) == 2, messages


def test_the_non_finite_half_of_the_same_bound_still_refuses() -> None:
    """The guard it was extended from, asserted so the extension did not
    displace it. A NaN threshold is refused for the opposite reason: no gap
    however large can exceed it."""
    bg, x, scaffolded, _clean = _probe_world()
    multi, single, messages = _both_probes(scaffolded, bg, x, float("nan"))

    assert multi.flag is None and single[0] is None
    assert "not a finite number" in multi.reason
    assert "not a finite number" in single[1]
    assert "never below zero" not in multi.reason, "the wrong reason is worse than none"
    assert len(messages) == 2, messages


def test_control_a_usable_threshold_still_produces_both_real_verdicts() -> None:
    """OVER-CORRECTION CONTROL, with the REAL numbers on both sides.

    A guard that refuses everything passes every refusal test above. These are
    the values measured on this fixture, identical before and after the change.
    """
    bg, x, scaffolded, clean = _probe_world()

    hot, hot_single, hot_messages = _both_probes(scaffolded, bg, x, 0.3)
    assert hot.flag is True
    assert hot.confidence == 1.0
    assert hot.mean_gap == pytest.approx(0.8)
    assert hot.fired_fraction == 1.0
    assert hot_single[0] is True
    assert "0.800" in hot_single[1]
    assert hot_messages == [], hot_messages

    cold, cold_single, cold_messages = _both_probes(clean, bg, x, 0.3)
    assert cold.flag is False
    assert cold.confidence == 1.0
    assert cold.mean_gap == pytest.approx(0.0)
    assert cold.fired_fraction == 0.0
    assert cold_single[0] is False
    assert cold_single[1] == ""
    assert cold_messages == [], cold_messages


def test_control_zero_is_a_usable_bound_and_is_not_refused() -> None:
    """Zero is deliberately NOT swept up, and this is why.

    ``gap > 0.0`` is False for a gap of exactly 0.0, so the comparison is still
    falsifiable, and on this fixture it comes out BOTH ways: the flat model's gap
    is exactly 0.0 and reads clean, the scaffolded model's is 0.8 and is flagged.
    Refusing zero would have removed a real, strict bound, and the twin draws the
    line in the same place.
    """
    bg, x, scaffolded, clean = _probe_world()

    hot, hot_single, hot_messages = _both_probes(scaffolded, bg, x, 0.0)
    assert hot.flag is True
    assert hot.mean_gap == pytest.approx(0.8)
    assert hot_single[0] is True
    assert hot_messages == []

    cold, cold_single, cold_messages = _both_probes(clean, bg, x, 0.0)
    assert cold.flag is False
    assert cold.mean_gap == pytest.approx(0.0)
    assert cold_single[0] is False
    assert cold_messages == []
