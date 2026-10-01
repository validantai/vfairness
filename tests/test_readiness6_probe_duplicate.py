"""READINESS-6: a model that returned nothing was reported as an explainer that
is certainly sound, by TWO independent copies of the same function.

`multi_seed_adversarial_probe` exists twice, in `xai/diagnostics/adversarial.py`
and in `evaluation/vfairness_metrics/explanation_diagnostics.py`, with different
source (1948 vs 1396 bytes) and different signatures. Both carried the identical
defect, so a fix to one would not have reached the other.

`gaps > threshold` is False for a NaN gap, so a seed whose probe produced
NOTHING counted as a seed that fired nothing. Measured 2026-09-10, a flat clean
model against a model returning all-NaN predictions::

    clean  -> flag=False confidence=1.0 mean_gap=0.0 reason='' 0 warnings
    outage -> flag=False confidence=1.0 mean_gap=nan reason='' 0 warnings

`confidence = 1.0 - fired_fraction` asserts MAXIMUM certainty that the explainer
is clean, and the only difference between the two rows is a mean_gap no consumer
is obliged to read.

These pins run BOTH copies through the same cases and assert they agree, so the
next person to fix one is told about the other.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
    multi_seed_adversarial_probe as probe_evaluation,
)
from vfairness.xai.diagnostics.adversarial import (
    multi_seed_adversarial_probe as probe_xai,
)


def _background():
    rng = np.random.default_rng(0)
    return rng.normal(size=(60, 4)), rng.normal(size=4)


def _clean(X):
    """A model with no OOD scaffolding: every prediction identical."""
    return np.full(len(np.atleast_2d(X)), 0.5)


def _outage(X):
    """A model that returned nothing at all."""
    return np.full(len(np.atleast_2d(X)), np.nan)


def _run(which, predict_fn):
    background, x = _background()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if which == "xai":
            result = probe_xai(
                predict_fn=predict_fn,
                x=x,
                background=background,
                n_seeds=4,
                n_perturbations=40,
            )
        else:
            result = probe_evaluation(predict_fn, x, background, n_perturbations=40, n_seeds=4)
    return result, [str(w.message) for w in caught]


BOTH = pytest.mark.parametrize("which", ["xai", "evaluation"])


@BOTH
def test_a_model_that_returned_nothing_is_not_a_clean_explainer(which):
    result, texts = _run(which, _outage)

    assert result.flag is None, (
        f"{which}: a model returning no predictions reported flag={result.flag!r}, "
        "which reads as 'the explainer is sound'"
    )
    assert math.isnan(result.confidence), (
        f"{which}: confidence {result.confidence!r} over a probe that never ran"
    )
    assert "COULD NOT CHECK" in result.reason
    assert "not a finding" in result.reason
    assert texts, f"{which}: the refusal was silent"


@BOTH
def test_a_genuinely_clean_model_is_untouched(which):
    """OVER-CORRECTION CONTROL. The whole value of the probe is that it can
    clear an explainer, so a real clean run must keep its exact verdict."""
    result, texts = _run(which, _clean)

    assert result.flag is False
    assert result.confidence == 1.0
    assert result.mean_gap == 0.0
    assert result.reason == ""
    assert not texts, f"{which}: a clean run warned about something"


def test_the_two_copies_agree():
    """The structural pin. These are separate implementations; a fix to one
    does not reach the other, and that is how both came to carry this."""
    for predict_fn in (_clean, _outage):
        a, _ = _run("xai", predict_fn)
        b, _ = _run("evaluation", predict_fn)
        assert a.flag is b.flag, f"{predict_fn.__name__}: {a.flag!r} vs {b.flag!r}"
        assert (math.isnan(a.confidence) and math.isnan(b.confidence)) or (
            a.confidence == b.confidence
        ), f"{predict_fn.__name__}: confidence {a.confidence!r} vs {b.confidence!r}"
        assert ("COULD NOT CHECK" in a.reason) == ("COULD NOT CHECK" in b.reason)


@BOTH
def test_a_partly_unmeasurable_run_says_what_it_dropped(which):
    """Some seeds usable, some not: measure the ones that are, and disclose."""
    calls = {"n": 0}

    def flaky(X):
        calls["n"] += 1
        # The first probe's draws are unusable, the rest are fine.
        if calls["n"] <= 2:
            return np.full(len(np.atleast_2d(X)), np.nan)
        return np.full(len(np.atleast_2d(X)), 0.5)

    result, texts = _run(which, flaky)
    if result.flag is None:
        pytest.skip("this seeding made every draw unusable; covered above")
    assert result.flag is False
    assert math.isfinite(result.confidence)
    assert any("seed" in t for t in texts), (
        f"{which}: seeds were dropped from the agreement fraction in silence"
    )
