"""BGL6 PIN, batch A-operations-3-b (2026-09-29): pulse/orchestrator.py.

``pulse.regulatory.build_recheck`` is three-state and the states are pinned in
tests/test_bgl5_operations_3.py, and the MAIN production call site could not
reach the third one: orchestrator.py flattened the screen with
``bool((regulatory_exports.get("ll144") or {}).get("applicable"))`` over a block
that falls back to ``empty_exports(...)``, whose ``applicable=False`` means "not
evaluated". So the payload published "No annual audit statute matched this run",
a finding about the law that no code on that pathway made.

The pin runs the REAL publishing path (``run_pulse`` with the export stage forced
to collapse) and reads the recheck block out of the payload, not the unit in
isolation, because the unit was already correct. The two controls assert the
determined cases end to end, with their exact windows.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from vfairness.operations.pulse import orchestrator as _orchestrator
from vfairness.operations.pulse import regulatory as _regulatory


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


# ===========================================================================
# ROOT CAUSE 4. build_recheck's third state was unreachable from the MAIN
# production call site, which published a legal finding nobody made.
# ===========================================================================


def _hiring_df(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    half = n // 2
    gender = np.array(["male"] * half + ["female"] * half, dtype=object)
    pred = np.concatenate([rng.binomial(1, 0.7, half), rng.binomial(1, 0.2, half)])
    return pd.DataFrame({"gender": gender, "prediction": pred, "feature": rng.normal(size=n)})


def _run(df: pd.DataFrame, inputs: dict) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _orchestrator.run_pulse(df, inputs)
    assert result["success"] is True
    return result["data"]


def test_the_orchestrator_no_longer_publishes_a_statute_finding_it_never_screened(monkeypatch):
    """REFUSAL PIN on the PUBLISHED payload, not on the unit. orchestrator.py
    read ``bool((regulatory_exports.get("ll144") or {}).get("applicable"))``, and
    ``regulatory_exports`` falls back to ``empty_exports(...)`` when the export
    stage collapses. That block carries ``applicable=False`` meaning "not
    evaluated", so the expression answered a plain False and build_recheck's
    third state was unreachable from the pathway that publishes the recheck.

    Measured before, through run_pulse with build_regulatory_exports forced to
    raise (the real publishing path, not the expression in isolation)::

        regulatoryExports.screenRan: False
        PUBLISHED recheck.basis: 'No annual audit statute matched this run, so
            the 6-month window is Pulse's default hygiene interval: data and
            behaviour drift make an older read stale evidence.'

    That sentence is a finding about the law, and no LL144 screen ran at all."""

    def _collapse(*args, **kwargs):
        raise RuntimeError("forced export-stage collapse")

    monkeypatch.setattr(_regulatory, "build_regulatory_exports", _collapse)
    data = _run(
        _hiring_df(), {"domain": "hiring", "jurisdiction": "US", "protected_attributes": ["gender"]}
    )

    assert data["regulatoryExports"].get("screenRan") is False, "the fixture must collapse"
    basis = data["recheck"]["basis"]
    assert "could not be determined on this run" in basis, basis
    assert "NOT a finding that none applies" in basis, basis
    assert "No annual audit statute matched this run" not in basis, basis
    # The window is 6 months either way, so nothing a caller schedules changes:
    # what changes is that the payload stops asserting the statute was tested.
    assessed = datetime.fromisoformat(data["recheck"]["assessedAt"].replace("Z", "+00:00"))
    until = datetime.fromisoformat(data["recheck"]["validUntil"].replace("Z", "+00:00"))
    assert until == _regulatory._add_months(assessed, 6)


def test_control_a_completed_screen_still_states_the_statute_it_matched():
    """OVER-CORRECTION CONTROL on the same published payload. A hiring tool in
    the US IS covered, the screen runs and decides, and the recheck must still
    say so with the binding 12-month interval. A fix that turned every run into
    'could not be determined' would pass the pin above and destroy the product."""
    data = _run(
        _hiring_df(), {"domain": "hiring", "jurisdiction": "US", "protected_attributes": ["gender"]}
    )
    ll144 = data["regulatoryExports"].get("ll144") or {}
    assert ll144.get("applicable") is True and ll144.get("applicabilityDetermined") is True
    basis = data["recheck"]["basis"]
    assert "NYC Local Law 144 requires a bias audit" in basis, basis
    assert "could not be determined" not in basis, basis
    assessed = datetime.fromisoformat(data["recheck"]["assessedAt"].replace("Z", "+00:00"))
    until = datetime.fromisoformat(data["recheck"]["validUntil"].replace("Z", "+00:00"))
    assert until == _regulatory._add_months(assessed, 12)


def test_control_a_determined_no_match_still_reads_as_a_finding():
    """OVER-CORRECTION CONTROL, the case the refusal is most likely to eat: a
    screen that RAN and found nothing must keep saying 'No annual audit statute
    matched this run', with the 6-month hygiene window. Credit scoring in
    Germany gives applicable False with applicabilityDetermined True, which is a
    measurement and not the third state."""
    rng = np.random.default_rng(23)
    n = 160
    half = n // 2
    group = np.array(["g1"] * half + ["g2"] * half, dtype=object)
    label = np.concatenate([rng.binomial(1, 0.7, half), rng.binomial(1, 0.4, half)])
    pred = np.where(rng.random(n) < 0.85, label, 1 - label)
    df = pd.DataFrame(
        {"group": group, "label": label, "prediction": pred, "feature": rng.normal(size=n)}
    )
    data = _run(
        df,
        {
            "domain": "credit scoring",
            "jurisdiction": "Germany",
            "protected_attributes": ["group"],
        },
    )
    ll144 = data["regulatoryExports"].get("ll144") or {}
    assert ll144.get("applicable") is False and ll144.get("applicabilityDetermined") is True
    basis = data["recheck"]["basis"]
    assert "No annual audit statute matched this run" in basis, basis
    assert "could not be determined" not in basis, basis
    assessed = datetime.fromisoformat(data["recheck"]["assessedAt"].replace("Z", "+00:00"))
    until = datetime.fromisoformat(data["recheck"]["validUntil"].replace("Z", "+00:00"))
    assert until == _regulatory._add_months(assessed, 6)
