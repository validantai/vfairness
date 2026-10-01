"""Two guards for one question must answer it the same way.

WHY. A background with no spread collapses the perturbation cloud onto the
instance itself, so the adversarial probe's gap is exactly 0.0 for ANY model,
including a deliberately scaffolded one, and 0.0 is FINITE so every
finiteness filter counts it as measured. Two implementations of that guard exist,
one in ``evaluation/vfairness_metrics/explanation_diagnostics.py`` and one in
``xai/diagnostics/adversarial.py``.

Two fix agents closed the same defect in the same hour on 2026-09-27 and CHOSE
DIFFERENT TESTS. The xai one took a floor relative to each column's own
magnitude; this one kept ``spread > 0.0``. Measured on the same three backgrounds,
50 rows of 3 columns:

    frozen, 50 copies of one row     both refuse
    frozen plus a 1e-12 jitter       xai refuses, the evaluation twin did NOT
    a real sample                    neither refuses

A background one part in 1e12 away from constant is not exactly zero. The
per-column spread measured 4.0e-12 to 5.3e-12 against column magnitudes of 1.3 to
2.5, so the exact test said "this background has spread" while
``sigma = background.std(axis=0) + 1e-9`` still collapsed the cloud and cleared
the scaffold.

TWO SIBLING GUARDS DISAGREEING ABOUT ONE INPUT IS THE SHAPE THAT LET THE ORIGINAL
HOLE SURVIVE, so the thing pinned here is the AGREEMENT, not either answer. The
floor is imported rather than re-declared for the same reason: one definition
cannot drift from itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
    _background_cannot_be_perturbed as _evaluation_guard,
)
from vfairness.xai.diagnostics.adversarial import _MIN_RELATIVE_SPREAD
from vfairness.xai.diagnostics.adversarial import (
    _unperturbable_background as _xai_guard,
)


def _backgrounds():
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 1.0, size=(50, 3))
    frozen = np.repeat(base[:1], 50, axis=0)
    return {
        "frozen": (frozen, True),
        "jitter_1e_12": (frozen + rng.normal(0, 1e-12, size=frozen.shape), True),
        "jitter_1e_6": (frozen + rng.normal(0, 1e-6, size=frozen.shape), False),
        "real_sample": (base, False),
        "one_row": (base[:1], True),
        "single_column_frozen": (frozen[:, :1], True),
        "single_column_real": (base[:, :1], False),
    }


@pytest.mark.parametrize("name", sorted(_backgrounds()))
def test_both_guards_reach_the_same_verdict(name):
    background, _expected = _backgrounds()[name]
    ev = bool(_evaluation_guard(background))
    xai = bool(_xai_guard(background))
    assert ev == xai, (
        f"on the {name!r} background the evaluation guard says refuse={ev} and the "
        f"xai guard says refuse={xai}. One question, two answers, is how the "
        "original hole survived a fix."
    )


@pytest.mark.parametrize("name", sorted(_backgrounds()))
def test_each_guard_reaches_the_right_verdict(name):
    """Agreement alone is not enough: both agreeing on the WRONG answer would pass.

    So the expected verdict is asserted too, and the 1e-12 row is the one that
    discriminates: it is the case the exact test got wrong.
    """
    background, expected = _backgrounds()[name]
    assert bool(_evaluation_guard(background)) is expected, name
    assert bool(_xai_guard(background)) is expected, name


def test_the_floor_is_one_definition_not_two():
    """Imported, not re-declared. Two copies is how they drifted."""
    import vfairness.evaluation.vfairness_metrics.explanation_diagnostics as ed

    assert ed._MIN_RELATIVE_SPREAD is _MIN_RELATIVE_SPREAD, (
        "the evaluation module has its own copy of the floor, so the two guards can "
        "drift apart again the next time one is tuned"
    )


def test_the_refusals_say_what_they_measured():
    """A refusal that does not name the number it compared is not evidence."""
    frozen, _ = _backgrounds()["frozen"]
    for guard in (_evaluation_guard, _xai_guard):
        reason = str(guard(frozen))
        assert "spread" in reason, reason
        assert "floor" in reason, reason
        assert "0.0 for every" in reason, reason
