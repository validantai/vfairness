"""The MetricsStore's ε-DP claim must survive more than one query.

Register #10. The module docstring states that groups between k and
``noisy_threshold`` "receive calibrated Laplace noise (ε-DP)" and cites Dwork
et al. (2006). The noise was redrawn from the unseeded global RNG on EVERY
query, so any consumer that can poll averages it away: executed on one record
with true value 0.42, three successive reads gave [-0.1168, -3.0016, -0.414]
and the mean of 3000 reads was 0.4370. The guarantee held for exactly one
query, and 1813 of those 3000 draws fell outside the [0, 1] the module
documents.

Fixed by drawing the noise ONCE per record, deterministically from a secret
per-store seed, so repeated reads return the identical value and are therefore
post-processing of a single release.

**The trap this file is built around.** "Every query returns the same value" is
trivially satisfiable by adding NO noise at all, which would be perfect
stability and zero privacy. Stability alone is therefore not evidence of
anything. ``test_the_noise_is_a_real_laplace_draw`` and
``test_the_value_is_actually_perturbed`` are what stop this file from passing on
a mechanism that quietly does nothing.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from vfairness.operations.reporting.store import (
    MetricsStore,
    MetricsStoreConfig,
    PrivacyLevel,
    StoredMetricRecord,
    _record_noise,
)

TRUE_VALUE = 0.42


def _store(*, group_size: int = 20, dp_seed: int | None = None, enable: bool = True):
    """One record in the k..noisy_threshold tier (k=10, noisy_threshold=50)."""
    store = MetricsStore(MetricsStoreConfig(dp_seed=dp_seed, enable_privacy=enable))
    store._records.append(
        StoredMetricRecord(
            timestamp=datetime(2026, 8, 27, 12, 0, 0),
            source="monitor",
            metric_name="demographic_parity_difference",
            value=TRUE_VALUE,
            group="g",
            group_size=group_size,
            batch_id="b1",
        )
    )
    return store


def _value(store) -> float:
    return float(store.get_metrics()["value"].iloc[0])


# --- the defect: repeated reads must not average the noise away --------------


def test_repeated_queries_return_the_identical_value():
    store = _store()
    values = {_value(store) for _ in range(500)}
    assert len(values) == 1, (
        "fresh noise per query is what let 3000 reads recover the true value; "
        f"got {len(values)} distinct answers"
    )


def test_polling_does_not_converge_on_the_true_value():
    """The attack, run directly."""
    store = _store()
    mean = float(np.mean([_value(store) for _ in range(3000)]))
    assert mean == pytest.approx(_value(store)), "the mean is just the one released value"
    # And that released value is not the secret.
    assert mean != pytest.approx(TRUE_VALUE, abs=1e-9)


# --- the trap: stability is worthless if the mechanism adds nothing ----------


def test_the_value_is_actually_perturbed():
    """A no-op mechanism would pass every stability test above."""
    perturbed = [_value(_store(dp_seed=seed)) for seed in range(40)]
    assert any(v != pytest.approx(TRUE_VALUE) for v in perturbed), (
        "no seed perturbed the value at all; the mechanism is a no-op"
    )
    assert len(set(perturbed)) > 1, "the released value must depend on the secret seed"


def test_the_noise_is_a_real_laplace_draw():
    """Distributional check: not a constant, not uniform, the right scale.

    Without this, `_record_noise` could return 0.0 forever and every other test
    in this file would still pass.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    scale = 1.0
    draws = np.array(
        [
            _record_noise(
                dp_seed=99,
                timestamp=f"t{i}",
                source="s",
                metric_name="m",
                group="g",
                group_size=20,
                value=TRUE_VALUE,
                scale=scale,
            )
            for i in range(20000)
        ]
    )
    assert draws.std() == pytest.approx(scale * np.sqrt(2), rel=0.05), "wrong Laplace scale"
    assert abs(draws.mean()) < 0.05, "Laplace noise must be centred on zero"
    ks = scipy_stats.kstest(draws, scipy_stats.laplace(0, scale).cdf)
    assert ks.pvalue > 0.01, f"draws are not Laplace(0, {scale}): {ks}"


def test_noise_depends_on_the_record_not_just_the_seed():
    """Two different records under one seed must not share a draw."""
    a = _record_noise(
        dp_seed=7,
        timestamp="t1",
        source="s",
        metric_name="m",
        group="g",
        group_size=20,
        value=0.42,
        scale=1.0,
    )
    b = _record_noise(
        dp_seed=7,
        timestamp="t2",
        source="s",
        metric_name="m",
        group="g",
        group_size=20,
        value=0.42,
        scale=1.0,
    )
    assert a != b


def test_the_seed_is_secret_so_two_stores_disagree():
    """Unseeded stores draw their own secret, so the mapping is not public."""
    values = {_value(_store()) for _ in range(30)}
    assert len(values) > 1, (
        "every unseeded store produced the same answer, so the record-to-noise "
        "mapping is public and the noise can be subtracted"
    )


# --- the range the module documents -----------------------------------------


def test_the_noisy_value_stays_inside_the_documented_range():
    for seed in range(200):
        v = _value(_store(dp_seed=seed))
        assert 0.0 <= v <= 1.0, f"seed {seed} produced {v}, outside the documented [0, 1]"


# --- a persisted seed carries the guarantee across processes ----------------


def test_a_persisted_seed_reproduces_the_same_release():
    assert _value(_store(dp_seed=12345)) == _value(_store(dp_seed=12345))


def test_different_persisted_seeds_are_different_releases():
    assert _value(_store(dp_seed=1)) != _value(_store(dp_seed=2))


# --- controls: the other two tiers must be untouched ------------------------


def test_small_groups_are_still_suppressed():
    store = _store(group_size=5)  # below k=10
    row = store.get_metrics().iloc[0]
    assert np.isnan(row["value"])
    assert row["privacy_level"] == PrivacyLevel.SUPPRESSED.value


def test_large_groups_are_still_exact():
    store = _store(group_size=500)  # above noisy_threshold=50
    row = store.get_metrics().iloc[0]
    assert row["value"] == pytest.approx(TRUE_VALUE)
    assert row["privacy_level"] == PrivacyLevel.EXACT.value


def test_privacy_can_still_be_disabled():
    store = _store(enable=False)
    assert _value(store) == pytest.approx(TRUE_VALUE)


def test_the_noisy_tier_is_still_labelled_noisy():
    assert _store().get_metrics()["privacy_level"].iloc[0] == PrivacyLevel.NOISY.value
