"""Methodology versioning tests (VB-DOC-1).

Every analyzer report must record the methodology version that produced it, so a
rating is reproducible against a pinned methodology independent of the code
version.
"""

import numpy as np

from vfairness import FairnessAnalyzer
from vfairness._methodology import METHODOLOGY_VERSION


def _report():
    rng = np.random.RandomState(0)
    n = 120
    s = np.array(["A"] * 60 + ["B"] * 60)
    y_true = rng.binomial(1, 0.5, n)
    y_pred = rng.binomial(1, 0.5, n)
    return FairnessAnalyzer(y_true, y_pred, s, task_type="classification").get_report()


def test_report_stamps_methodology_version():
    report = _report()
    assert report["methodology_version"] == METHODOLOGY_VERSION


def test_methodology_version_is_a_nonempty_string():
    assert isinstance(METHODOLOGY_VERSION, str) and METHODOLOGY_VERSION
