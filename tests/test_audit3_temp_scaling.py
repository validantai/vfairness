"""Audit 3 (HIGH, calibration): TemperatureScaling.fit inverted-NLL-gradient fix.

The defect: TemperatureScaling.fit computed d(NLL)/dT with the wrong sign and
stepped a fixed-step gradient descent on the raw summed gradient. The descent
therefore climbed NLL instead of descending it and drove T to the wrong side of
1.0, so fitting made calibration *worse* than the uncalibrated input on exactly
the over-confident predictions temperature scaling exists to fix (Guo et al.,
2017: minimize NLL of sigmoid(logit / T) over T > 0; T > 1 softens
over-confident logits, T < 1 sharpens under-confident ones).

These tests pin the corrected behaviour by execution: the defect scenario now
gives the right answer, and an already-calibrated input is not overcorrected.
"""

import numpy as np

from vfairness.post_processing.calibration.methods import TemperatureScaling


def _ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return e


def _nll(y, p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def _bruteforce_optimal_temp(y, p):
    logits = np.log(np.clip(p, 1e-10, 1 - 1e-10) / (1 - np.clip(p, 1e-10, 1 - 1e-10)))
    grid = np.linspace(0.05, 20.0, 2000)
    return min(grid, key=lambda T: _nll(y, 1 / (1 + np.exp(-logits / T))))


def _make_overconfident(n=8000, sharpen=2.5, seed=0):
    """True prob from a moderate logit; model over-sharpens it (optimal T > 1)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.2, n)
    p_true = 1 / (1 + np.exp(-z))
    y = (rng.uniform(size=n) < p_true).astype(float)
    p_model = 1 / (1 + np.exp(-sharpen * z))
    return y, p_model


def _make_underconfident(n=8000, soften=0.5, seed=0):
    """Model under-sharpens the true logit (optimal T < 1)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.2, n)
    p_true = 1 / (1 + np.exp(-z))
    y = (rng.uniform(size=n) < p_true).astype(float)
    p_model = 1 / (1 + np.exp(-soften * z))
    return y, p_model


# --- Defect case FIRST: the exact scenario that used to give the wrong answer -


def test_overconfident_fit_lowers_ece_and_moves_temp_above_one():
    """Over-confident input: T must land > 1 and ECE must DROP.

    Old (buggy) behaviour: T pinned to the 0.01 floor and ECE rose
    (~0.13 -> ~0.29). Fixed: T ~ 2.4 and ECE falls toward zero.
    """
    y, p_model = _make_overconfident()
    ece_before = _ece(y, p_model)

    ts = TemperatureScaling().fit(y, p_model)
    p_cal = ts.transform(p_model)
    ece_after = _ece(y, p_cal)

    assert ts.temperature_ > 1.0, f"T should soften over-confident logits, got {ts.temperature_}"
    assert ece_after < ece_before, f"ECE should drop: {ece_before} -> {ece_after}"
    # It used to make things much worse; assert a real improvement.
    assert ece_after < 0.5 * ece_before


def test_overconfident_fit_never_worse_than_uncalibrated_nll():
    """After fit, NLL must not exceed the uncalibrated NLL (T = 1 is in-bounds).

    Old behaviour drove NLL from ~0.68 to ~6.4 (an order of magnitude worse).
    """
    y, p_model = _make_overconfident()
    nll_before = _nll(y, p_model)

    ts = TemperatureScaling().fit(y, p_model)
    nll_after = _nll(y, ts.transform(p_model))

    assert nll_after <= nll_before + 1e-9, f"NLL must not worsen: {nll_before} -> {nll_after}"
    assert nll_after < nll_before, "NLL should strictly improve on miscalibrated input"


def test_extreme_overconfident_not_worse_than_doing_nothing():
    """The audit's egregious repro: p in {0.99, 0.01}, true rates 0.7/0.3.

    Old behaviour: T -> 0.01 floor, NLL 1.395 -> 8.329 (worse than doing
    nothing). Fixed: NLL must not exceed the uncalibrated NLL.
    """
    rng = np.random.default_rng(3)
    n = 20000
    high = rng.uniform(size=n) < 0.5
    p_model = np.where(high, 0.99, 0.01)
    true_rate = np.where(high, 0.7, 0.3)
    y = (rng.uniform(size=n) < true_rate).astype(float)

    nll_before = _nll(y, p_model)
    ts = TemperatureScaling().fit(y, p_model)
    nll_after = _nll(y, ts.transform(p_model))

    assert ts.temperature_ > 1.0
    assert nll_after <= nll_before + 1e-9, (
        f"must not be worse than doing nothing: {nll_before} -> {nll_after}"
    )


def test_underconfident_fit_moves_temp_below_one():
    """Under-confident input: T must land < 1 (sharpen).

    Old behaviour drove T up without bound (a consistent sign inversion in
    the other direction). Fixed: T ~ 0.48.
    """
    y, p_model = _make_underconfident()
    nll_before = _nll(y, p_model)

    ts = TemperatureScaling().fit(y, p_model)
    nll_after = _nll(y, ts.transform(p_model))

    assert ts.temperature_ < 1.0, f"T should sharpen under-confident logits, got {ts.temperature_}"
    assert nll_after <= nll_before + 1e-9


def test_fit_converges_to_bruteforce_optimal_temp():
    """The fit must minimize NLL, not merely move T the right way.

    Pins that the optimizer actually converges to the NLL minimum (catches
    both the inverted sign and the fixed-step overshoot that left T ~ 257).
    """
    for maker in (_make_overconfident, _make_underconfident):
        y, p_model = maker()
        opt = _bruteforce_optimal_temp(y, p_model)
        ts = TemperatureScaling().fit(y, p_model)
        assert abs(ts.temperature_ - opt) / opt < 0.05, (
            f"fitted T {ts.temperature_} far from optimal {opt}"
        )


# --- Does-not-overcorrect case ---------------------------------------------


def test_well_calibrated_input_is_not_overcorrected():
    """Already-calibrated input: T stays near 1 and calibration is not worsened."""
    rng = np.random.default_rng(7)
    n = 8000
    p_model = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p_model).astype(float)

    ece_before, nll_before = _ece(y, p_model), _nll(y, p_model)
    ts = TemperatureScaling().fit(y, p_model)
    p_cal = ts.transform(p_model)
    ece_after, nll_after = _ece(y, p_cal), _nll(y, p_cal)

    assert 0.8 < ts.temperature_ < 1.25, (
        f"T should stay near 1 on calibrated data, got {ts.temperature_}"
    )
    assert nll_after <= nll_before + 1e-9
    assert ece_after <= ece_before + 0.01


def test_sample_weights_are_honoured():
    """Weighted fit runs and still lands a sensible (>1) temperature.

    Weight the over-confident half heavily; the fit must still soften.
    """
    y, p_model = _make_overconfident()
    w = np.ones_like(y)
    w[p_model > 0.5] = 3.0
    ts = TemperatureScaling().fit(y, p_model, sample_weight=w)
    assert ts.temperature_ > 1.0
    out = ts.transform(p_model)
    assert np.all((out >= 0) & (out <= 1))


def test_fit_result_records_temperature_and_nll():
    y, p_model = _make_overconfident()
    ts = TemperatureScaling().fit(y, p_model)
    assert ts.fit_result is not None
    assert ts.fit_result.parameters["temperature"] == ts.temperature_
    assert ts.fit_result.fit_metrics["nll"] > 0
