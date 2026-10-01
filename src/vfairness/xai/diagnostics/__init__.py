"""
vfairness.xai.diagnostics
=========================

Explanation-quality diagnostics that must run alongside every Pulse:

* :mod:`faithfulness` -- removal-curve AUC, infidelity.
* :mod:`stability` -- variance of attributions over repeated runs.
* :mod:`adversarial` -- Slack et al. (2020) scaffolding probe.

These are surfaced in the Engineer-persona panel and stored on the
``XaiAssessment.diagnostics`` field.
"""

from .adversarial import (
    AdversarialProbeResult,
    multi_seed_adversarial_probe,
    slack_adversarial_probe,
)
from .faithfulness import local_r_squared, removal_curve_auc
from .stability import attribution_stability

__all__ = [
    "removal_curve_auc",
    "local_r_squared",
    "attribution_stability",
    "slack_adversarial_probe",
    "multi_seed_adversarial_probe",
    "AdversarialProbeResult",
]
