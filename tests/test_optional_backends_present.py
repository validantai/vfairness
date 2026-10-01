"""Turns a silently-skipped optional backend into a loud CI failure.

Why this file exists. Every test that needs torch, shap, captum, lime,
dice-ml, anchor-exp, sage, fairlearn or aif360 gates itself with
`pytest.importorskip`. That is correct for a contributor running a core-only
install, but it is invisible in aggregate: a run with 41 unreachable tests
prints the same green "N passed" line as a run with none, and the skip
reasons are only shown under `-rs`. Measured 2026-08-27, the CI job installed
`.[dev,rendering,viz,monitoring]` while claiming in its own comment that the
full suite runs, so the whole explainability subsystem, the in-processing
torch losses and the fairlearn/aif360 parity battery had never executed in
any environment. That silence hid the regression pins for two fixes the audit
had rated CRITICAL (the AdversarialDebiasingLoss sign inversion, and SHAP
class-consistency on the modern >= 0.45 array convention): nobody could say
whether either still held.

The contract. Set VFAIRNESS_REQUIRE_BACKENDS=1 in any environment that claims
to run the full suite (CI does, see .github/workflows/fairness-checks.yml).
A missing backend then FAILS here and names the extra that installs it,
instead of quietly subtracting tests from the run. Unset, this module skips
and a core-only install stays green, as it should.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

# import name -> the pyproject extra that provides it.
REQUIRED_BACKENDS = {
    "torch": "training (also pulled by xai)",
    "shap": "xai",
    "captum": "xai",
    "lime": "xai",
    "dice_ml": "xai",
    "anchor": "xai (anchor-exp)",
    "sage": "xai (sage-importance)",
    "fairlearn": "parity",
    "aif360": "parity",
}

_ENFORCED = os.environ.get("VFAIRNESS_REQUIRE_BACKENDS") == "1"

pytestmark = pytest.mark.skipif(
    not _ENFORCED,
    reason="VFAIRNESS_REQUIRE_BACKENDS != 1 (core-only install; optional backends not required)",
)


@pytest.mark.parametrize("module,extra", sorted(REQUIRED_BACKENDS.items()))
def test_optional_backend_is_installed(module: str, extra: str) -> None:
    """Fail loudly, naming the extra, rather than letting importorskip hide it."""
    assert importlib.util.find_spec(module) is not None, (
        f"Optional backend '{module}' is not installed, so every test gated on it "
        f"silently skipped instead of running. Install it with the '{extra}' extra: "
        f'pip install -e ".[dev,rendering,viz,monitoring,xai,training,parity]"'
    )
