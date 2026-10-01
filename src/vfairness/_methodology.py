"""Methodology version (VB-DOC-1).

A ratings agency must be able to reproduce a published rating years later, even
after the code has moved on. The methodology is therefore versioned separately
from the code (the code version is `vfairness.__version__`; the methodology
version is here), and every report stamps the methodology version that produced
it (see `FairnessAnalyzer.get_report`).

Bump `METHODOLOGY_VERSION` only when the assessment methodology itself changes
(a metric definition, a threshold, a verdict rule), following the policy in
`docs/METHODOLOGY.md`. A pure code refactor that does not change results does
NOT bump it.
"""

from __future__ import annotations

# Ratified 2026-08-22: docs/METHODOLOGY.md is authored and reviewed against the
# implemented code for the 0.1.0 beta (all six sections). The "-draft" suffix is
# dropped now that the methodology is frozen; bump only under the policy in that
# document (a metric definition, a threshold, a verdict rule, or the metric set).
#
# M1.1, 2026-08-27: a VERDICT RULE changed, which the policy in
# docs/METHODOLOGY.md lists as a bump trigger, and the change moves published
# PASS/FAIL counts.
#   1. Both report surfaces and the analyzer's confidence-interval verdict now
#      grade every metric in ITS OWN direction, resolved once by
#      evaluation/vfairness_metrics/_metric_direction.py. Previously the report
#      special-cased a single name for the higher-is-better branch and the
#      analyzer assumed the fair band was [0, threshold] for EVERY metric, so
#      the whole ratio family (the four-fifths / disparate-impact rule) was
#      graded backwards: a ratio of 0.00 passed and perfect parity failed.
#   2. A metric whose better-direction cannot be resolved is now COULD-NOT-CHECK
#      (excluded from the verdict and from the fairness score), where it was
#      previously graded as if lower were better, i.e. usually passed.
# Minor rather than major: the metric set, the metric definitions and every
# threshold are unchanged; only how a value is compared to its threshold is.
METHODOLOGY_VERSION = "M1.1"
