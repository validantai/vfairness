"""Network egress control for the vfairness engine.

Cross-cutting infrastructure rather than a fairness-analysis surface: every
outbound call the engine makes (LLM proxy, judges, sidecar) goes through the
SSRF guard here, so a user-supplied endpoint URL cannot be pointed at loopback,
RFC1918, link-local or cloud-metadata addresses.
"""

from vfairness.net.egress import (
    GuardedSession,
    PinnedIPAdapter,
    SSRFError,
    guarded_post,
    validate_endpoint,
)

__all__ = [
    "GuardedSession",
    "PinnedIPAdapter",
    "SSRFError",
    "guarded_post",
    "validate_endpoint",
]
