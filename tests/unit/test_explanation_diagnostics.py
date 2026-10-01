"""Explanation trustworthiness diagnostics: faithfulness, stability, adversarial."""

import json

import numpy as np

from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
    attribution_stability,
    diagnose_local_attribution,
    multi_seed_adversarial_probe,
    removal_curve_auc,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _predict(X):
    X = np.asarray(X, dtype=float)
    return _sigmoid(3.0 * X[:, 0] + 0.2 * X[:, 1] + 0.0 * X[:, 2])


def _bg(seed=0, n=120):
    return np.random.default_rng(seed).normal(0, 1, size=(n, 3))


def test_faithful_attribution_beats_misleading_one():
    # BGL6 F04-1, 2026-09-29. The misleading vector was [0.0, 0.0, 3.0], and the
    # two tied zeros left the masking order of features 0 and 1 to the column
    # arrangement: measured, that vector scores 0.2661001427525318 in this column
    # order and 0.09174518427926061 with the tied pair swapped, on identical data.
    # removal_curve_auc now returns NaN for that rather than one of the two arms,
    # so the comparison had nothing to compare. The SUBJECT is unchanged (a
    # faithful attribution outscores one pointing at the irrelevant feature); only
    # the fixture is corrected to name an order, feature 2 over 1 over 0, which is
    # still exactly backwards from the model's real weights.
    bg = _bg()
    x = np.array([2.0, 0.5, 0.5])
    true_attr = np.array([3.0, 0.2, 0.0])  # tracks the real weights
    wrong_attr = np.array([0.0, 0.1, 3.0])  # ranks the irrelevant feature first
    bg_mean = bg.mean(axis=0)
    good = removal_curve_auc(_predict, x, true_attr, bg_mean)
    bad = removal_curve_auc(_predict, x, wrong_attr, bg_mean)
    assert np.isfinite(good) and np.isfinite(bad), (good, bad)
    assert good > bad, f"faithful attribution should score higher ({good} vs {bad})"


def test_stability_zero_for_identical_reruns():
    v = np.array([0.4, -0.1, 0.0])
    import math
    import warnings

    assert attribution_stability([v, v.copy(), v.copy()]) < 1e-9  # identical -> ~0
    # Was `assert attribution_stability([v]) == 0.0  # single run -> 0 by
    # definition`. It is not 0 by definition: stability is variation ACROSS
    # runs, and one run has none to measure. 0.0 sigma is the BEST score on
    # this scale, so a caller who supplied nothing to compare was handed
    # "perfectly stable".
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert math.isnan(attribution_stability([v]))
    assert caught


def test_clean_model_not_flagged_adversarial():
    bg = _bg(seed=1)
    x = np.array([1.0, 0.0, 0.0])
    probe = multi_seed_adversarial_probe(_predict, x, bg, n_seeds=6)
    # A plain smooth model behaves the same off-manifold, so no scaffold flag.
    assert probe.flag is False
    assert probe.confidence >= 0.5


def test_diagnose_local_attribution_shape_and_json():
    bg = _bg(seed=2)
    x = np.array([2.0, 0.5, 0.5])
    attr = np.array([3.0, 0.2, 0.0])
    out = diagnose_local_attribution(_predict, x, attr, bg, reruns=[attr, attr * 1.01], n_seeds=4)
    assert set(out) >= {
        "faithfulness",
        "stability",
        "adversarial_flag",
        "adversarial_reason",
        "notes",
    }
    assert out["faithfulness"] is not None and out["faithfulness"] >= 0
    assert out["stability"] is not None
    # A REAL probe run gives a real bool. The three-state None belongs to the
    # degenerate case below, and asserting bool here is what keeps the two apart:
    # if this ever starts returning None on good input, the probe stopped running
    # and the fix over-corrected.
    assert isinstance(out["adversarial_flag"], bool)
    assert out["adversarial_reason"] is None, out["adversarial_reason"]
    json.dumps(out)  # must be serialisable for the task envelope


def test_diagnose_never_raises_on_bad_input():
    """Empty background / degenerate inputs degrade to None, never raise.

    The assertion below used to read ``is False``, contradicting this test's own
    first line. False is a VERDICT: it says the probe ran and found no
    adversarial instability. On an empty background the probe does not run at
    all, so False was a finding nobody made, and a caller could not tell it from
    a clean result. Corrected 2026-09-10 along with the code, which now returns
    None and says why in ``adversarial_reason``.
    """
    out = diagnose_local_attribution(_predict, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], np.empty((0, 3)))
    assert out["adversarial_flag"] is None
    assert "COULD NOT CHECK" in (out["adversarial_reason"] or ""), out["adversarial_reason"]
    json.dumps(out)
