"""
Compliance Reporting: Fairness Navigator Wizard Support
========================================================

Regulatory compliance report generators for the Fairness Navigator wizard.
Produces structured outputs for EU AI Act (Annex IV), GDPR (Art. 35 DPIA),
ECOA Reg B adverse action, ISO 42001, and general model card generation.

These functions are designed to be called from the wizard pipeline, where
upstream modules (bias detection, metric computation, SHAP explanation,
intervention) have already produced their respective result dicts.

Functions
---------
- ``generate_risk_register_from_audit``: Cross-reference bias + metric results into risk entries
- ``generate_dpia_sections``: GDPR Art. 35(7) five-section DPIA
- ``compute_adverse_action_reasons``: ECOA Reg B reason codes with proxy exclusion
- ``compute_signed_test_log``: Annex IV Section 6 signed test log
- ``generate_annex_iv_data``: Full Annex IV 9-section aggregator
- ``generate_model_card``: Fairness-enhanced Markdown model card
- ``generate_iso42001_evidence_map``: ISO 42001 Annex A control mapping

References
----------
EU AI Act Annex IV; GDPR Art. 35; ECOA Regulation B; ISO/IEC 42001:2023.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ...branding import branding_enabled
from ...evaluation.vfairness_metrics._metric_direction import (
    ThresholdOutcome,
    check_threshold,
)

# Constants & Mappings

# Maps bias detection categories to EU AI Act Art. 9 risk categories.
_BIAS_TO_ART9_CATEGORY: Dict[str, str] = {
    "demographic_parity": "Discriminatory output: demographic disparity",
    "equalized_odds": "Discriminatory output: unequal error rates",
    "equal_opportunity": "Discriminatory output: unequal true positive rates",
    "predictive_parity": "Discriminatory output: unequal predictive value",
    "calibration": "Accuracy and robustness: calibration failure",
    "representation": "Training data: representation bias",
    "measurement": "Training data: measurement bias",
    "historical": "Training data: historical bias",
    "aggregation": "Training data: aggregation bias",
    "proxy": "Feature selection: proxy discrimination",
    "label": "Training data: label bias",
    "selection": "Training data: selection bias",
    "sample": "Training data: sampling bias",
}

# Default ECOA Reg B reason code descriptions.
_DEFAULT_REASON_CODES: Dict[str, str] = {
    "RC01": "Insufficient credit history length",
    "RC02": "Delinquent or derogatory accounts",
    "RC03": "High credit utilization ratio",
    "RC04": "Insufficient income relative to obligations",
    "RC05": "Limited account diversity",
    "RC06": "Recent credit inquiries",
    "RC07": "Insufficient collateral value",
    "RC08": "Employment stability below threshold",
    "RC09": "High debt-to-income ratio",
    "RC10": "Insufficient deposit account history",
}

# Annex IV section definitions.
_ANNEX_IV_SECTIONS = [
    (1, "General Description", "Art. 11"),
    (2, "Detailed Description of Elements and Process", "Art. 11"),
    (3, "Monitoring, Functioning, and Control", "Art. 12-14"),
    (4, "Risk Management System", "Art. 9"),
    (5, "Data and Data Governance", "Art. 10"),
    (6, "Testing and Validation", "Art. 15"),
    (7, "Accuracy and Robustness", "Art. 15"),
    (8, "Transparency and User Information", "Art. 13"),
    (9, "Fundamental Rights Impact Assessment", "Art. 29a"),
]

# ISO 42001 Annex A controls relevant to fairness.
_ISO42001_CONTROLS = [
    # AI Management System
    ("A.2.2", "AI Policy", "AI Management System"),
    ("A.2.3", "Roles and Responsibilities", "AI Management System"),
    ("A.2.4", "Resources", "AI Management System"),
    # Risk Management
    ("A.4.2", "AI Risk Assessment", "Risk Management"),
    ("A.4.3", "AI Risk Treatment", "Risk Management"),
    ("A.4.4", "AI Risk Residual Assessment", "Risk Management"),
    # Data Management
    ("A.5.2", "Data Quality for AI", "Data Management"),
    ("A.5.3", "Data Provenance", "Data Management"),
    ("A.5.4", "Data Preparation", "Data Management"),
    # Monitoring
    ("A.6.2", "Performance Monitoring", "Monitoring"),
    ("A.6.3", "AI System Changes", "Monitoring"),
    ("A.6.4", "Third-Party AI Components", "Monitoring"),
    # Fairness / Ethics
    ("A.8.2", "Bias and Fairness", "Monitoring"),
    ("A.8.3", "Transparency", "Monitoring"),
    ("A.8.4", "Accountability", "Monitoring"),
]


# Helpers


def _now_iso() -> str:
    """UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, lo: float = 1.0, hi: float = 5.0) -> float:
    """Clamp a numeric value to [lo, hi]."""
    return max(lo, min(hi, value))


def _compute_coverage(auto_fields: List[str], manual_fields: List[str]) -> float:
    """Percentage of fields that are auto-populated (0-100)."""
    total = len(auto_fields) + len(manual_fields)
    if total == 0:
        # DEAD for every call site, and that is the disposition rather than a
        # deferral: all nine callers pass literal field-name lists written in this
        # module, so `total` is a constant of the code and never 0. If a section is
        # ever built from data, this 0.0 becomes the same lie the ISO coverage
        # fallback was (see generate_iso42001_evidence_map) and must become None.
        return 0.0
    return round(len(auto_fields) / total * 100, 1)


# Three-state metric verdict
#
# A metric row that never recorded a pass/fail must never be GRADED. Until
# 2026-08-28 every reader in this module was ``m.get("passed", True)`` (and, in
# the signed log's snapshot, ``m.get("passed", False)``), which manufactured two
# opposite fabrications out of the same missing measurement:
#
#   * a row with no ``passed`` key was counted and printed as PASS, even with a
#     value four times its own threshold sitting in the same row;
#   * ``.get(key, default)`` does NOT fire when the key is PRESENT holding None,
#     and None is falsy, so the IDENTICAL unmeasured state printed FAIL.
#
# One missing measurement therefore produced a fabricated all-clear and a
# fabricated breach in one table, decided by nothing but whether the key was
# absent or None. Worse, ``compute_signed_test_log`` wrote BOTH readings into ONE
# document (snapshot defaulting False, summary defaulting True) and computed a
# content_hash over the contradiction, sealing a fabricated verdict into a
# provenance record.
#
# Three states, never two: assessed-pass / assessed-fail / could-not-check.
# Could-not-check is carried in its own bucket and is never folded into either of
# the others, in any count, table cell, sentence or hashed summary.

_MISSING = object()

#: How a row's verdict was arrived at. Rendered and sealed alongside the verdict,
#: because "the pipeline reported a pass" and "this report computed a pass from
#: the row" are different claims and a compliance artifact may not blur them.
_VERDICT_REPORTED = "reported"
_VERDICT_DERIVED = "derived"
_VERDICT_UNASSESSED = "unassessed"


def _coerce_passed(raw: Any) -> Optional[bool]:
    """Read a reported pass flag, or None when it does not carry a verdict.

    Only real booleans (including numpy's) and the ints 0/1 are accepted. A
    string, a NaN or any other object is NOT truthiness-tested: bool("False")
    and bool(float("nan")) are both True, so guessing here is how an unchecked
    input renders as a green PASS.

    Deliberately the same contract, and the same rejections, as
    ``rendering.adapters_validation._coerce_passed``; duplicated rather than
    imported so that ``operations`` does not depend on ``rendering``.
    ``tests/test_compliance_three_state.py`` pins the two against each other so
    they cannot drift apart.
    """
    if raw is _MISSING or raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    item = getattr(raw, "item", None)  # numpy.bool_ and friends
    if callable(item):
        try:
            unwrapped = item()
        except Exception:  # noqa: BLE001 - an object that cannot unwrap has no verdict
            return None
        if isinstance(unwrapped, bool):
            return unwrapped
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    return None


def _as_float(raw: Any) -> Optional[float]:
    """Read a recorded number, or None when the field is not one.

    ``bool`` is rejected even though it is an ``int`` subclass (True is not a
    measurement of 1.0), and so is a numeric-looking string: parsing "0.41" here
    would mean guessing that a caller's free-text field is a measurement. NaN is
    let through on purpose, because :func:`check_threshold` owns that case and
    answers COULD_NOT_CHECK for it, and as of READINESS-6 that function owns
    infinity the same way.

    ``np.bool_`` is named beside ``bool`` because it is NOT a Python bool, so a
    boolean read out of a DataFrame walked past the first clause and became 1.0.
    Measured before this line changed, ``_as_severity(np.bool_(True))`` returned
    severity 1 on the 1-5 risk register, from a flag rather than a measurement.
    """
    if raw is None or isinstance(raw, (bool, np.bool_, str, bytes)):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_measured_number(raw: Any) -> Optional[float]:
    """Read a number that will be GRADED, or None when nothing was measured.

    Stricter than :func:`_as_float` in exactly one way: a non-finite number is
    refused here. That function hands NaN on to :func:`check_threshold`, which
    owns the case and answers COULD_NOT_CHECK for it; the risk register has no
    such owner, and a NaN travelling into ``round(_clamp(...))`` becomes a real
    integer on a 1-5 scale, which is the fabrication this reader exists to
    prevent.

    READINESS-6, 2026-09-10. The test was ``num != num``, which is NaN only, and
    infinity walked straight past the sentence above into exactly the arithmetic
    it describes. Measured before this line changed: ``_as_severity(inf)``
    returned **5**, the maximum severity on the 1-5 register, and
    ``_as_severity(-inf)`` returned **1**, the minimum, both graded from a value
    nobody measured. The clamp is what converts them: ``_clamp`` pins inf to the
    top of the scale and -inf to the bottom, so the register reads as either a
    top-priority risk or an all-clear, and neither is a reading of anything.
    """
    num = _as_float(raw)
    if num is None or not math.isfinite(num):
        return None
    return num


def _as_severity(raw: Any) -> Optional[int]:
    """Read a metric row's reported severity on the 1-5 scale, or None.

    None means the row recorded no severity. It is NOT 3: see the note on
    :func:`generate_risk_register_from_audit`.
    """
    num = _as_measured_number(raw)
    if num is None:
        return None
    return int(round(_clamp(num)))


def _risk_score(entry: Any) -> Optional[float]:
    """Read one register entry's risk score, or None when it was never scored.

    Every consumer of a register goes through this. ``entry.get("risk_score", 0)``
    does NOT survive an unscored entry: the key is PRESENT holding None, so the
    default never fires and ``None >= 15`` raises TypeError. Worse, the shape it
    was protecting against (an entry with no score at all) silently counted as 0,
    i.e. below every escalation line, which is a fabricated all-clear.
    """
    if not isinstance(entry, dict):
        return None
    score = entry.get("risk_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    if score != score:  # NaN
        return None
    return float(score)


def _is_scored_at_or_above(entry: Any, floor: float) -> bool:
    """True when this entry WAS scored and the score is at or above *floor*.

    An unscored entry answers False here and is counted by
    :func:`_count_unscored` instead, so it appears in exactly one bucket and in
    neither as a silent zero.
    """
    score = _risk_score(entry)
    return score is not None and score >= floor


def _count_unscored(risk_register: Any) -> int:
    """How many entries carry no risk score at all. Always reported next to any
    "N risks above the line" count, because "0 high-severity risks" out of a
    register nobody could score is not an all-clear, it is an absence."""
    return sum(1 for r in risk_register or [] if _risk_score(r) is None)


def _high_risk_paragraph(high_risks: List[dict], unscored_risks: List[dict]) -> str:
    """DPIA Art. 35(7)(c) paragraph naming the high-severity risks.

    "No high-severity risks identified." is an unqualified all-clear over the
    WHOLE register, so it may only be written when the whole register was
    scored. With an unscored entry present it would silently cover a risk this
    DPIA has no severity for at all.
    """
    if high_risks:
        return "\nHigh-severity risks:\n" + "\n".join(
            f"  - {r['risk_id']}: {r['description']} "
            f"(score: {r['risk_score']}, category: {r['category']})"
            for r in high_risks
        )
    if unscored_risks:
        return (
            "\nNo high-severity risk was identified among the scored risks, but "
            f"{len(unscored_risks)} risk(s) carry no score, so this section makes no "
            "severity claim about them."
        )
    return "\nNo high-severity risks identified."


def _unscored_risk_paragraph(unscored_risks: List[dict]) -> str:
    """The unscored entries, named. An entry that is only a count is not visible
    to the DPO who has to score it."""
    if not unscored_risks:
        return ""
    return "\n\nNot scored:\n" + "\n".join(
        f"  - {r['risk_id']}: {r['description']} "
        f"(no {' or '.join(r.get('not_assessed_fields') or ['score'])}, "
        f"category: {r['category']})"
        for r in unscored_risks
    )


def _metric_verdict(metric: Any) -> tuple:
    """Read one metric row as three states: ``(passed, basis, reason)``.

    ``passed`` is True (assessed pass), False (assessed fail) or None
    (could-not-check). ``basis`` is one of ``reported`` / ``derived`` /
    ``unassessed``. ``reason`` is empty for a pass and states why otherwise.

    **Should a row carrying a value AND a threshold but no verdict be COMPARED,
    or called unknown?** It is COMPARED, and only through the shared
    :func:`vfairness.evaluation.vfairness_metrics._metric_direction.check_threshold`.

    For comparing: the value was measured, the bound was declared, both are in
    the row, and this library already owns the rule that turns that pair into a
    verdict. Refusing to apply our own rule to evidence in hand would report "not
    assessed" for a 4x breach this library can grade, which is a different
    dishonesty and the one an auditor is least served by. It is the same move
    ``rendering.adapters_validation._verdict`` already makes when it derives FAIL
    from listed blocking issues instead of shrugging.

    Against comparing: the direction is not knowable from a bare name. On
    ``demographic_parity`` 0.41 against 0.10 the value fails by being ABOVE,
    while on ``disparate_impact_ratio`` 0.41 against 0.80 it fails by being
    BELOW, so a hand-rolled ``value > threshold`` here would invert the verdict
    for the entire ratio family (the cali-BRATIO-n incident, recorded in
    ``_metric_direction``). That objection is answered by DELEGATING, not by
    refusing: ``check_threshold`` resolves the direction from the one shared
    table and returns COULD_NOT_CHECK, never PASS, for an unknown direction, a
    NaN value or threshold, or a bound that no value could breach. So the only
    comparisons performed here are ones this library can defend, and every other
    row stays in the third bucket.

    A derived verdict is labelled ``derived`` wherever it is rendered or hashed.
    """
    if not isinstance(metric, dict):
        return (None, _VERDICT_UNASSESSED, "the metric row is not a mapping")

    raw = metric.get("passed", _MISSING)
    reported = _coerce_passed(raw)
    if reported is not None:
        return (reported, _VERDICT_REPORTED, "" if reported else "reported as failed")

    value = _as_float(metric.get("value"))
    threshold = _as_float(metric.get("threshold"))
    if value is not None and threshold is not None:
        name = str(metric.get("name") or "")
        try:
            outcome, message = check_threshold(name, value, threshold)
        except Exception:  # noqa: BLE001 - report assembly never raises
            logging.getLogger(__name__).debug("threshold check failed", exc_info=True)
            return (
                None,
                _VERDICT_UNASSESSED,
                "the recorded value and threshold could not be compared",
            )
        if outcome is ThresholdOutcome.PASS:
            return (True, _VERDICT_DERIVED, "")
        if outcome is ThresholdOutcome.FAIL:
            return (False, _VERDICT_DERIVED, message)
        return (None, _VERDICT_UNASSESSED, message)

    # No verdict was reported, and there is nothing here to compare either.
    if raw is _MISSING:
        why = "the metric carries no pass/fail flag"
    elif raw is None:
        why = "the metric's pass/fail flag is empty"
    else:
        why = f"the pass/fail flag is not a boolean but a {type(raw).__name__}"
    absent = [k for k in ("value", "threshold") if _as_float(metric.get(k)) is None]
    if absent:
        why += (
            f", and its {' and '.join(absent)} "
            f"{'is' if len(absent) == 1 else 'are'} not a number to compare"
        )
    return (None, _VERDICT_UNASSESSED, why)


def _counts_from_verdicts(verdicts: Any) -> Dict[str, int]:
    """Bucket already-read verdicts into passed / failed / not_assessed.

    ``not_assessed`` is a first-class bucket: it is never added to ``passed`` or
    to ``failed``, and ``passed + failed + not_assessed == total`` always holds,
    so a reader can see how much of the total was actually graded. ``derived``
    is a breakdown of how many of the graded rows were graded here rather than
    reported by the pipeline; it overlaps passed/failed and is not part of the
    total.

    Callers that also RENDER the rows pass the same verdict list they render
    from, so a summary line cannot report "1 passed" above a row printed as NOT
    ASSESSED. Two independent readings agreeing today is not the same guarantee.
    """
    counts = {"total": 0, "passed": 0, "failed": 0, "not_assessed": 0, "derived": 0}
    for passed, basis, _reason in verdicts or []:
        counts["total"] += 1
        if basis == _VERDICT_DERIVED:
            counts["derived"] += 1
        if passed is True:
            counts["passed"] += 1
        elif passed is False:
            counts["failed"] += 1
        else:
            counts["not_assessed"] += 1
    return counts


def _metric_verdict_counts(metrics: Any) -> Dict[str, int]:
    """Read metric rows and bucket them; see :func:`_counts_from_verdicts`."""
    return _counts_from_verdicts([_metric_verdict(m) for m in metrics or []])


# 1. Risk Register


def generate_risk_register_from_audit(
    bias_results: dict,
    metric_results: dict,
    domain: str,
) -> list[dict]:
    """Cross-reference bias detection and metric results into a risk register.

    Each bias finding is matched against metric failures to determine severity,
    and mapped to the corresponding EU AI Act Art. 9 risk category.

    Parameters
    ----------
    bias_results : dict
        Output from the bias detection module.  Expected keys:
        ``findings`` (list of dicts with ``id``, ``category``, ``description``,
        ``confidence``, ``affected_groups``).  ``confidence`` is load-bearing:
        it is the ONLY input to ``likelihood``, so a finding that reported no
        confidence (key absent, ``None``, or anything that is not a number)
        gets ``likelihood: None`` rather than a mid-scale default.
    metric_results : dict
        Output from the metric computation module.  Expected keys:
        ``metrics`` (list of dicts with ``name``, ``value``, ``threshold``,
        ``passed``, ``severity``).  ``passed`` is load-bearing and is read as
        THREE states (see :func:`_metric_verdict`): a boolean verdict, or a
        verdict derived from ``value`` against ``threshold`` when the direction
        of the named metric is known, or could-not-check.  A metric that cannot
        be graded never counts as failed and never raises a severity; it is
        listed in the entry's ``unassessed_metric_names`` instead.
    domain : str
        Application domain (e.g. ``"credit"``, ``"hiring"``, ``"healthcare"``).

    Returns
    -------
    list[dict]
        Risk register entries, each containing:
        ``risk_id``, ``description``, ``category``, ``likelihood``,
        ``severity``, ``risk_score``, ``treatment``, ``status``,
        ``residual_risk``, ``assessment_state``, ``not_assessed_fields``,
        ``art9_reference``, ``related_bias_ids``, ``related_metric_names``,
        ``unassessed_metric_names``.

        THREE STATES, NEVER TWO, for the score as well as for the metric rows.
        ``likelihood`` is derived from the finding's reported ``confidence`` and
        ``severity`` from the severities reported by the related FAILED metrics.
        Either one that was never reported stays ``None``; ``risk_score`` and
        ``residual_risk`` are then ``None`` too, ``assessment_state`` reads
        ``"not_assessed"``, ``not_assessed_fields`` names which factor is
        missing, and ``status`` is ``"not_assessed"`` (never ``"accepted"``).

        Until 2026-08-28 both factors were invented instead: an absent
        ``confidence`` became 0.5 (so ``likelihood`` 2 on a 1-5 scale) and
        ``severity`` was a hardcoded 3, giving every uncorroborated finding a
        ``risk_score`` of 6 on an ISO-31000-shaped register, comfortably below
        the ``>= 15`` escalation line this module uses in three places. Nothing
        in the emitted entry said any of the three numbers was invented. Use
        :func:`_risk_score` to read the field: ``entry.get("risk_score", 0)`` is
        wrong in both directions here, because the key is present holding None.

    Examples
    --------
    >>> bias = {"findings": [{"id": "B001", "category": "demographic_parity",
    ...     "description": "Gender disparity", "confidence": 0.85,
    ...     "affected_groups": ["female"]}]}
    >>> metrics = {"metrics": [{"name": "demographic_parity_difference",
    ...     "value": 0.15, "threshold": 0.1, "passed": False, "severity": 4}]}
    >>> register = generate_risk_register_from_audit(bias, metrics, "credit")
    >>> register[0]["risk_score"]
    16
    >>> register[0]["assessment_state"]
    'assessed'

    A finding that measured no confidence is listed, and is not scored:

    >>> unmeasured = {"findings": [{"id": "B002", "category": "proxy"}]}
    >>> entry = generate_risk_register_from_audit(unmeasured, metrics, "credit")[0]
    >>> entry["likelihood"], entry["severity"], entry["risk_score"]
    (None, None, None)
    >>> entry["status"]
    'not_assessed'

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: generate_risk_register. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    findings = bias_results.get("findings", [])
    metrics = metric_results.get("metrics", [])

    # Index failed metrics by name for quick lookup.
    #
    # A metric is FAILED here only when a verdict says so. The old
    # ``not m.get("passed", True)`` read an unassessed row as a pass when the key
    # was absent (so it vanished from the register) and as a failure when the key
    # was present holding None (so it raised a finding's severity on a
    # measurement that never happened). Neither is allowed: an unassessed row is
    # indexed separately and reported on the entry, so a finding that could not
    # be corroborated says so rather than looking either corroborated or clean.
    failed_metrics: Dict[str, dict] = {}
    unassessed_metrics: Dict[str, dict] = {}
    for m in metrics:
        passed, _basis, _reason = _metric_verdict(m)
        if passed is False:
            failed_metrics[m["name"]] = m
        elif passed is None:
            unassessed_metrics[m["name"]] = m

    register: list[dict] = []
    for idx, finding in enumerate(findings):
        category = finding.get("category", "unknown")
        confidence = _as_measured_number(finding.get("confidence"))
        finding_id = finding.get("id", f"B{idx + 1:03d}")

        # Likelihood from confidence: scale [0,1] -> [1,5]. None when the
        # detector reported no confidence, and NEVER the old 0.5 default: that
        # default put likelihood 2 on the register for a finding nobody scored
        # (round(0.5 * 5) = 2 under banker's rounding), and the number was then
        # multiplied into a risk_score and banded against the >= 15 escalation
        # line, so a measurement that never happened produced a quiet, graded
        # all-clear. The present-but-None spelling of the same missing input
        # used to raise TypeError here instead; both spellings now take one path.
        likelihood = None if confidence is None else round(_clamp(confidence * 5))

        # Severity: the max severity REPORTED by related failed metrics. None
        # when no related failed metric reported one, and never the old
        # hardcoded 3: that 3 was the midpoint of a 1-5 scale handed to every
        # finding this register could not corroborate, and 3 x the fabricated
        # likelihood 2 is a risk_score of 6, sitting below every escalation
        # threshold in this module. The metric FAILING is a measurement; how bad
        # that failure is, is a judgement the failing row either recorded or did
        # not, and this library cannot derive it from a name.
        related_metric_names: list[str] = []
        max_severity: Optional[int] = None
        # Simple heuristic: relate metrics whose name shares the category keyword.
        cat_keywords = category.replace("_", " ").split()
        for m_name, m_data in failed_metrics.items():
            if any(kw in m_name.lower() for kw in cat_keywords):
                related_metric_names.append(m_name)
                reported = _as_severity(m_data.get("severity"))
                if reported is not None:
                    max_severity = reported if max_severity is None else max(max_severity, reported)

        # Related rows that carry no verdict. They do NOT move the severity (that
        # would be grading an unmeasured metric), but they are named on the entry
        # so the gap in the evidence is visible to whoever reads the register.
        unassessed_metric_names = [
            m_name
            for m_name in unassessed_metrics
            if any(kw in m_name.lower() for kw in cat_keywords)
        ]

        severity = max_severity
        # Three states for the score itself: an entry is scored only when BOTH
        # factors were measured. A product with an invented factor in it is not
        # a partial measurement, it is a whole fabricated number.
        not_assessed_fields = [
            name
            for name, value in (("likelihood", likelihood), ("severity", severity))
            if value is None
        ]
        # The SAME condition as `not_assessed_fields`, restated so mypy can see
        # it. The list above is empty exactly when neither factor is None, but
        # that narrowing happens inside a comprehension, which the type checker
        # cannot follow, so it read line 581 as `None * int` and failed Quality
        # with three errors. Behaviour is identical; this is not a new guard.
        #
        # Worth stating why that mattered out of proportion to three type errors:
        # The deployment workflow ships the library to the host that runs it
        # only when BOTH the test suite and Quality are green for a commit.
        # Quality was red on every commit because of this line, so the library
        # could not reach production at all.
        #
        # The workflow and the host are deliberately unnamed: this file SHIPS in
        # the public export, and the leak gate refuses an internal name in it.
        # It refused this very comment on 2026-09-07.
        if likelihood is None or severity is None:
            risk_score = None
        else:
            risk_score = likelihood * severity

        art9_ref = _BIAS_TO_ART9_CATEGORY.get(category, "Unclassified risk: manual review required")

        # Treatment suggestion based on domain and category.
        if risk_score is None:
            # The third state. NOT "accepted": accepted is a decision taken
            # about a scored risk, and it is the reading the old fabricated 6
            # produced for every uncorroborated finding. The finding itself is
            # real and stays on the register; what is withheld is the grade.
            missing = " and ".join(not_assessed_fields)
            because = {
                "likelihood": "the detection reported no confidence",
                "severity": (
                    "no related failed metric reported a severity"
                    if related_metric_names
                    else "no related metric failed, so nothing recorded a severity"
                ),
            }
            treatment = (
                f"NOT SCORED: this risk has no {missing} on the 1-5 scale because "
                + "; ".join(because[f] for f in not_assessed_fields)
                + ". It is neither accepted nor escalated here and carries no risk "
                "score; assess it before this register is relied upon."
            )
            status = "not_assessed"
        elif risk_score >= 15:
            treatment = (
                f"Immediate mitigation required for {domain} domain. "
                "Apply fairness-aware retraining or post-processing intervention."
            )
            status = "open"
        elif risk_score >= 8:
            treatment = (
                "Schedule review within next sprint cycle. "
                "Consider threshold adjustment or data augmentation."
            )
            status = "under_review"
        else:
            treatment = "Monitor and reassess at next audit cycle."
            status = "accepted"

        if unassessed_metric_names:
            # Said in the treatment text, not only in a field, because the
            # treatment is what gets copied into DPIA section 4.
            treatment += (
                f" Note: {len(unassessed_metric_names)} related metric(s) "
                f"({', '.join(unassessed_metric_names)}) carry no pass/fail "
                "verdict and could not corroborate or clear this finding; "
                "assess them before closing this risk."
            )

        register.append(
            {
                "risk_id": f"R-{idx + 1:03d}",
                "description": finding.get("description", f"Risk from {category} bias"),
                "category": category,
                "likelihood": likelihood,
                "severity": severity,
                "risk_score": risk_score,
                "treatment": treatment,
                "status": status,
                # An unscored risk has no residual risk either: residual is what
                # is left of a score after treatment, and there was no score.
                "residual_risk": (
                    None
                    if risk_score is None
                    else (max(1, risk_score - 4) if risk_score >= 8 else risk_score)
                ),
                # Machine-readable third state, so a consumer does not have to
                # infer "was this graded?" from a None it might coerce.
                "assessment_state": "not_assessed" if not_assessed_fields else "assessed",
                "not_assessed_fields": not_assessed_fields,
                "art9_reference": art9_ref,
                "related_bias_ids": [finding_id],
                "related_metric_names": related_metric_names,
                "unassessed_metric_names": unassessed_metric_names,
            }
        )

    return register


# 2. DPIA Generator


def generate_dpia_sections(
    system_profile: dict,
    data_ref: dict,
    risk_register: list,
    metric_results: dict,
) -> dict:
    """Generate a GDPR Art. 35(7) Data Protection Impact Assessment.

    Produces the five mandatory DPIA sections.  Sections that can be
    fully auto-generated are marked ``status='auto'``; those needing
    manual DPO review are ``status='partial'``.

    Parameters
    ----------
    system_profile : dict
        System metadata: ``name``, ``purpose``, ``description``,
        ``domain``, ``data_subjects``, ``legal_basis``.
    data_ref : dict
        Data reference: ``dataset_name``, ``n_records``, ``features``,
        ``protected_attributes``, ``retention_period``.
    risk_register : list
        Output from :func:`generate_risk_register_from_audit`.
    metric_results : dict
        Output from the metric computation module.  Each ``metrics`` row's
        ``passed`` flag is read as three states (see :func:`_metric_verdict`);
        rows that cannot be graded are reported as "not assessed" in section 2
        rather than counted as passed or failed, so the Art. 35(7)(b) text never
        claims an assessment that did not happen.

    Returns
    -------
    dict
        Keys: ``dpia_id``, ``generated_at``, ``system_name``, ``sections``
        (list of section dicts with ``section_number``, ``title``,
        ``article``, ``status``, ``content``).

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: generate_dpia_sections. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    sys_name = system_profile.get("name", "Unnamed AI System")
    purpose = system_profile.get("purpose", "Not specified")
    description = system_profile.get("description", "")
    domain = system_profile.get("domain", "general")
    data_subjects = system_profile.get("data_subjects", "Not specified")
    legal_basis = system_profile.get("legal_basis", "Not specified")

    ds_name = data_ref.get("dataset_name", "Unknown")
    n_records = data_ref.get("n_records", "Unknown")
    features = data_ref.get("features", [])
    protected = data_ref.get("protected_attributes", [])
    retention = data_ref.get("retention_period", "Not specified")

    metrics = metric_results.get("metrics", [])
    # Art. 35(7)(b) is a statement about what WAS assessed. A row that carries no
    # verdict and cannot be graded from its value and threshold is neither a pass
    # nor a failure, so it is counted in its own bucket and named as such below;
    # folding it into either count would make this section claim an assessment
    # that did not happen.
    counts = _metric_verdict_counts(metrics)
    n_failed = counts["failed"]
    n_passed = counts["passed"]
    n_unassessed = counts["not_assessed"]

    open_risks = [r for r in risk_register if r.get("status") == "open"]
    # ``r.get("risk_score", 0) >= 15`` was wrong twice over: an unscored entry
    # carries the key holding None, so the default never fired and the
    # comparison raised TypeError, and the shape the default WAS written for
    # counted as 0, i.e. silently below the escalation line. Read through
    # :func:`_risk_score`, and report the unscored entries in their own bucket
    # so "0 high-severity risks" cannot be read as "none of them are severe".
    high_risks = [r for r in risk_register if _is_scored_at_or_above(r, 15)]
    unscored_risks = [r for r in risk_register if _risk_score(r) is None]

    sections = [
        {
            "section_number": 1,
            "title": "Systematic Description of Processing",
            "article": "Art. 35(7)(a)",
            "status": "auto",
            "content": (
                f"System: {sys_name}\n"
                f"Purpose: {purpose}\n"
                f"Description: {description}\n"
                f"Domain: {domain}\n"
                f"Data subjects: {data_subjects}\n"
                f"Dataset: {ds_name} ({n_records} records, "
                f"{len(features)} features, "
                f"{len(protected)} protected attributes)\n"
                f"Retention: {retention}\n"
                f"Legal basis: {legal_basis}"
            ),
        },
        {
            "section_number": 2,
            "title": "Necessity and Proportionality",
            "article": "Art. 35(7)(b)",
            "status": "partial",
            "content": (
                f"The system processes {len(protected)} protected attribute(s) "
                f"({', '.join(protected) if protected else 'none identified'}) "
                f"across {n_records} data records.\n"
                f"Fairness metrics: {n_passed} passed, {n_failed} failed, "
                f"{n_unassessed} not assessed, out of {len(metrics)} recorded.\n"
                + (
                    f"{n_unassessed} recorded metric(s) carry no pass/fail verdict "
                    "and could not be graded from their value and threshold. They "
                    "are neither passes nor failures: this assessment makes no "
                    "compliance claim about them, and they must be assessed before "
                    "this DPIA is relied upon.\n"
                    if n_unassessed
                    else ""
                )
                + "Data minimisation and purpose limitation assessment requires "
                "manual DPO review."
            ),
        },
        {
            "section_number": 3,
            "title": "Risk Assessment",
            "article": "Art. 35(7)(c)",
            # "auto" says this section needs no manual DPO work. A register
            # holding a risk nobody could score does need exactly that, so an
            # unscored entry keeps the section "partial", the same way an
            # ungraded metric row keeps Annex IV section 6 partial.
            "status": ("auto" if risk_register and not unscored_risks else "partial"),
            "content": (
                f"Total risks identified: {len(risk_register)}\n"
                f"High-severity risks (score >= 15): {len(high_risks)}\n"
                f"Open risks requiring treatment: {len(open_risks)}\n"
                f"Risks that could not be scored: {len(unscored_risks)}\n"
                + _high_risk_paragraph(high_risks, unscored_risks)
                + _unscored_risk_paragraph(unscored_risks)
            ),
        },
        {
            "section_number": 4,
            "title": "Mitigation Measures",
            "article": "Art. 35(7)(d)",
            "status": "auto" if risk_register else "partial",
            "content": (
                "Planned treatments:\n"
                + (
                    "\n".join(
                        f"  - {r['risk_id']}: {r['treatment']} "
                        f"(residual risk: {r.get('residual_risk', 'TBD')})"
                        for r in risk_register
                    )
                    if risk_register
                    else "  No risk entries to generate treatments from."
                )
                + "\n\nTechnical safeguards: bias detection, metric monitoring, "
                "drift detection, intervention pipeline."
            ),
        },
        {
            "section_number": 5,
            "title": "Monitoring and Review",
            "article": "Art. 35(11)",
            "status": "partial",
            "content": (
                "Automated monitoring is configured via the vfairness pipeline:\n"
                "  - Real-time fairness metric tracking\n"
                "  - Drift detection with adaptive thresholds\n"
                "  - Alert prioritisation and escalation\n"
                "  - Periodic re-assessment schedule: [requires manual input]\n"
                "  - DPO review frequency: [requires manual input]"
            ),
        },
    ]

    return {
        "dpia_id": f"DPIA-{hashlib.sha256(sys_name.encode()).hexdigest()[:8].upper()}",
        "generated_at": _now_iso(),
        "system_name": sys_name,
        "sections": sections,
    }


# 3. Adverse Action Reasons (ECOA Reg B)


class AdverseActionReasons(list):
    """The reason-code list of an adverse action notice, plus what it omits.

    Behaves exactly as the plain ``list`` this function always returned
    (indexing, ``len``, iteration, JSON serialisation as an array). The two
    extra attributes exist because a notice that silently dropped a feature it
    could not rank was being presented as complete. R-6, 2026-09-09.

    Attributes:
        unattributed_features: Features in ``feature_names`` with no usable
            attribution in ``shap_values`` (absent, ``None`` or NaN). They
            could not be ranked, so they appear in no slot; the notice is
            INCOMPLETE while this is non-empty and must be presented as such.
        features_examined: How many features the ranking was given to consider.
            Zero means NOTHING was examined, which is not the same claim as
            "no feature pushed this decision toward denial".
        adverse_attributions_not_ranked: Features that HAVE a usable adverse
            attribution in ``shap_values`` and are absent from
            ``feature_names``, so the ranking never saw them. BGL5, 2026-09-27:
            this is the mirror image of ``unattributed_features`` and it was
            invisible. Measured with debt_ratio at -0.40 attributed and left out
            of ``feature_names``: RC01 became income, zip_code (a proxy) entered
            at RC04 and ``complete`` was True, which is the harm R-6 recorded
            for the other direction, reached from the other side.
        complete: ``True`` only when at least one feature was examined, every
            examined feature had an attribution, AND no adverse attribution was
            left out of the ranking, so the ranking considered the whole model.

    Note:
        This is a ``list`` subclass, so ``json.dumps(notice)`` writes the reason
        ARRAY and drops every attribute above, ``complete`` included. Use
        :meth:`to_dict` to serialise the notice with its disclosure; a consumer
        that ships the bare array ships no completeness statement at all.
    """

    def __init__(
        self,
        entries: Iterable[dict] = (),
        unattributed_features: Iterable[str] = (),
        features_examined: Optional[int] = None,
        adverse_attributions_not_ranked: Iterable[str] = (),
    ) -> None:
        super().__init__(entries)
        self.unattributed_features: list[str] = list(unattributed_features)
        self.adverse_attributions_not_ranked: list[str] = list(adverse_attributions_not_ranked)
        # BGL-3. Defaults to what the object itself can see (its own entries
        # plus the features named as unattributed) so an instance built by hand
        # keeps its old reading; the producer passes the real count, which also
        # covers the features that were attributed but not adverse.
        self.features_examined: int = (
            len(self) + len(self.unattributed_features)
            if features_examined is None
            else int(features_examined)
        )

    @property
    def complete(self) -> bool:
        # BGL-3, measured 2026-09-27. `not self.unattributed_features` alone is
        # True over an EMPTY examination: compute_adverse_action_reasons(shap,
        # [], []) returned `[]` with complete True, unattributed [] and no
        # warning, so "the ranking considered the whole model" was asserted
        # about zero features, and an empty notice built on nothing was
        # byte-identical to one built on five fully attributed features none of
        # which was adverse.
        #
        # BGL5, measured 2026-09-27: the third clause. debt_ratio attributed at
        # -0.40 and absent from feature_names gave codes [income, credit_score,
        # employment, zip_code] with complete True and zero warnings, i.e. the
        # strongest denial driver was dropped from a notice asserting that "the
        # ranking considered the whole model".
        return (
            self.features_examined > 0
            and not self.unattributed_features
            and not self.adverse_attributions_not_ranked
        )

    def __repr__(self) -> str:
        base = super().__repr__()
        if self.complete:
            return base
        if self.features_examined == 0:
            return f"{base} <INCOMPLETE: no feature was examined>"
        if self.adverse_attributions_not_ranked:
            return (
                f"{base} <INCOMPLETE: adverse attribution(s) never ranked "
                f"{self.adverse_attributions_not_ranked}>"
            )
        return f"{base} <INCOMPLETE: unattributed {self.unattributed_features}>"

    def to_dict(self) -> dict:
        """The notice AND its disclosure, in one JSON-serialisable object.

        BGL5, 2026-09-27. ``json.dumps`` on this object writes the bare reason
        array, because that is what a ``list`` subclass serialises to, so
        ``complete`` (the field a consumer reads to decide whether the notice may
        be issued) did not survive the only serialisation the class documents:
        the nothing-examined result crossed the boundary as ``[]``, identical to
        a notice that found no adverse factor. Use this method at any JSON
        boundary.
        """
        return {
            "reasons": list(self),
            "complete": self.complete,
            "features_examined": self.features_examined,
            "unattributed_features": list(self.unattributed_features),
            "adverse_attributions_not_ranked": list(self.adverse_attributions_not_ranked),
        }


def compute_adverse_action_reasons(
    shap_values: dict,
    feature_names: list,
    proxy_features: list,
    reason_code_mapping: Optional[dict] = None,
) -> AdverseActionReasons:
    """Compute ECOA Reg B adverse action reason codes with proxy exclusion.

    Regulation B limits adverse action notices to the top 4 reason codes.
    Features identified as proxies for protected attributes are excluded
    from the reason list and replaced by the next-most-influential
    non-proxy feature.

    Parameters
    ----------
    shap_values : dict
        Mapping of feature name to SHAP contribution score (negative values
        indicate features that pushed the decision toward denial).
    feature_names : list
        Ordered list of all feature names in the model.
    proxy_features : list
        Feature names identified as proxies for protected attributes.
    reason_code_mapping : dict, optional
        Custom mapping of feature names to ``(code, description)`` tuples.
        Falls back to generic codes if not provided.

    Returns
    -------
    AdverseActionReasons
        A ``list`` of up to 4 reason code entries, each with: ``code``,
        ``feature``, ``description``, ``contribution_score``,
        ``is_proxy_excluded``; plus ``unattributed_features`` (features that
        could not be ranked because ``shap_values`` holds no attribution for
        them), ``features_examined``, ``adverse_attributions_not_ranked`` and
        ``complete``. A feature with no attribution is never ranked, never
        scored as 0.0, and a warning names it; while ``unattributed_features``
        is non-empty the notice is incomplete. An EMPTY ``feature_names``
        examined nothing: the result is empty, ``complete`` is False and a
        warning says so, because "no adverse reason was found" and "no
        attribution was ranked" are different claims. A feature with an ADVERSE
        attribution that ``feature_names`` omits is equally never ranked: it is
        named in ``adverse_attributions_not_ranked``, warned about, and makes
        ``complete`` False. Use :meth:`AdverseActionReasons.to_dict` at a JSON
        boundary; ``json.dumps`` of the object itself writes only the array.

    Examples
    --------
    >>> shap = {"income": -0.3, "zip_code": -0.25, "credit_score": -0.2,
    ...         "age_proxy": -0.15, "employment": -0.1}
    >>> reasons = compute_adverse_action_reasons(
    ...     shap, list(shap.keys()), proxy_features=["zip_code", "age_proxy"])
    >>> len(reasons)
    4
    >>> reasons[0]["feature"]
    'income'
    """
    code_map = reason_code_mapping or {}

    # Build scored list: only features with negative SHAP (adverse).
    #
    # R-6, 2026-09-09. This was `shap_values.get(feat, 0.0)`: a feature in
    # feature_names but missing from shap_values scored 0.0, was never < 0, and
    # so could never appear on the notice, while a PROXY feature was pulled
    # into the slot it vacated. Measured: with debt_ratio (the strongest
    # adverse factor at -0.4) unmeasured, the top reason became income and
    # zip_code, a proxy, entered at RC04. A feature with no attribution
    # cannot be ranked: it is excluded from the ranking, named in a warning,
    # and reported on the result so the notice is not presented as complete.
    scored: list[tuple[str, float]] = []
    unattributed: list[str] = []
    # BGL-3, measured 2026-09-27. No feature names means no ranking happened at
    # all, and the old return for it was `[]` with `complete` True and no
    # warning: an ECOA notice that examined nothing, presented as one that found
    # no adverse reason. Named here in the same words the rest of this module
    # uses for a could-not-check, and `complete` is False for it below.
    if not list(feature_names):
        warnings.warn(
            "Adverse action reasons: no feature names were supplied, so no "
            "attribution was ranked and NOTHING was examined. The empty result is "
            "a could-not-check, not a finding that no feature pushed this decision "
            "toward denial; `complete` is False and the notice must not be issued "
            "from it.",
            UserWarning,
            stacklevel=2,
        )
    for feat in feature_names:
        val = shap_values.get(feat)
        if val is None or not math.isfinite(float(val)):
            unattributed.append(feat)
            continue
        if val < 0:
            scored.append((feat, float(val)))
    if unattributed:
        warnings.warn(
            f"Adverse action reasons: {len(unattributed)} feature(s) have no SHAP "
            f"attribution and could not be ranked: {unattributed}. The notice is "
            "computed from the attributed features only and is INCOMPLETE; it must "
            "not be presented as the full set of adverse reasons.",
            UserWarning,
            stacklevel=2,
        )

    # BGL5, 2026-09-27. The MIRROR IMAGE of the R-6 block above, and it was
    # unguarded: a feature that HAS an adverse attribution and is absent from
    # feature_names is never ranked either, and nothing said so. Measured on
    # shap {debt_ratio -0.40, income -0.30, zip_code -0.25, credit_score -0.20,
    # age_proxy -0.15, employment -0.10} with feature_names omitting debt_ratio:
    #   before -> RC01 income, RC02 credit_score, RC03 employment, RC04 zip_code
    #             (a PROXY), complete True, unattributed [], 0 warnings
    #   after  -> the same four codes (the ranking cannot invent a feature it was
    #             not given) but complete False, adverse_attributions_not_ranked
    #             ['debt_ratio'] and this warning, so the notice cannot be issued
    #             from it.
    # R-6 recorded this exact harm in its own comment ("the top reason became
    # income and zip_code, a proxy, entered at RC04") and fixed only the
    # direction it arrived from.
    #
    # BGL6 AUDIT, 2026-09-29. The guard was ``isinstance(val, (int, float)) and
    # not isinstance(val, bool)``, which np.float32 FAILS: np.float64 subclasses
    # float and passed, np.float32 does not, and np.float32 is what the shap
    # library returns for tree models and the dtype of any float32 array. Measured
    # with the SAME numbers as the block above cast to np.float32:
    #   complete True, adverse_attributions_not_ranked [], 0 warnings, and RC04 =
    #   zip_code, a PROXY, in the slot the omitted strongest driver vacated,
    # byte for byte the BEFORE state this fix records. A canonical "is this
    # measured?" predicate written with isinstance discards real evidence while
    # reading as caution, which is this defect class running backwards.
    #
    # Read through _as_measured_number, this module's own reader: it accepts any
    # numeric type float() accepts, including np.float32 and np.int64, and refuses
    # None, bool, np.bool_, a numeric-looking string, NaN and infinity, which is
    # everything the old chain was reaching for.
    examined = set(feature_names)
    adverse_not_ranked = []
    for feat, val in shap_values.items():
        if feat in examined:
            continue
        num = _as_measured_number(val)
        if num is not None and num < 0:
            adverse_not_ranked.append(str(feat))
    if adverse_not_ranked:
        warnings.warn(
            f"Adverse action reasons: {len(adverse_not_ranked)} feature(s) carry an "
            f"ADVERSE attribution and are absent from feature_names, so the ranking "
            f"never saw them: {adverse_not_ranked}. The notice below is a ranking of "
            f"the features it was given, not of the model; `complete` is False and it "
            f"must not be presented as the full set of adverse reasons.",
            UserWarning,
            stacklevel=2,
        )

    # Sort by magnitude (most negative first).
    scored.sort(key=lambda x: x[1])

    proxy_set = set(proxy_features)
    reasons: list[dict] = []
    proxy_excluded: list[dict] = []

    for feat, score in scored:
        is_proxy = feat in proxy_set
        entry = _build_reason_entry(feat, score, is_proxy, code_map, len(reasons) + 1)

        if is_proxy:
            entry["is_proxy_excluded"] = True
            proxy_excluded.append(entry)
        else:
            entry["is_proxy_excluded"] = False
            reasons.append(entry)

        if len(reasons) >= 4:
            break

    # ECOA Reg B caps the adverse action notice at 4 reason codes total.
    # Non-proxy reasons fill the slots first; proxy-excluded entries are
    # only attached for transparency while slots remain, and codes are
    # deduplicated so the notice never carries the same code twice.
    result: list[dict] = []
    seen_codes: set = set()
    for entry in reasons[:4] + proxy_excluded:
        if len(result) >= 4:
            break
        if entry["feature"] in code_map:
            if entry["code"] in seen_codes:
                # A custom-mapped code cannot be renumbered without
                # mislabelling the feature, so a duplicate is dropped.
                continue
        elif entry["code"] in seen_codes:
            # Generic codes are positional; re-issue the next free RC code
            # (proxy entries were numbered independently at build time and
            # can collide with the non-proxy sequence).
            pos = len(result) + 1
            code = f"RC{pos:02d}"
            while code in seen_codes:
                pos += 1
                code = f"RC{pos:02d}"
            entry["code"] = code
            entry["description"] = _DEFAULT_REASON_CODES.get(
                code, f"Adverse factor: {entry['feature'].replace('_', ' ')}"
            )
        seen_codes.add(entry["code"])
        result.append(entry)

    return AdverseActionReasons(
        result,
        unattributed_features=unattributed,
        features_examined=len(list(feature_names)),
        adverse_attributions_not_ranked=adverse_not_ranked,
    )


def _build_reason_entry(
    feature: str,
    score: float,
    is_proxy: bool,
    code_map: dict,
    position: int,
) -> dict:
    """Build a single reason code entry."""
    if feature in code_map:
        code, description = code_map[feature]
    else:
        # Generic code based on position.
        code = f"RC{position:02d}"
        description = _DEFAULT_REASON_CODES.get(
            code, f"Adverse factor: {feature.replace('_', ' ')}"
        )

    return {
        "code": code,
        "feature": feature,
        "description": description,
        "contribution_score": round(abs(score), 6),
        "is_proxy_excluded": is_proxy,
    }


# 4. Signed Test Log


def _reads_as_iso_time(candidate: object) -> bool:
    """Say whether ``candidate`` is a string this module can READ as a time.

    ONE reader for every branch of the timestamp dispatch in
    :func:`compute_signed_test_log`.

    BGL7 A-operations-4-c, 2026-09-29. This predicate exists because the
    previous fix validated only ONE of the two branches that can produce a
    candidate string. The string branch parsed with ``datetime.fromisoformat``;
    the object branch believed whatever ``.isoformat()`` returned. ``pd.NaT`` IS
    an instance of :class:`datetime.datetime` and its ``isoformat()`` returns the
    literal string ``'NaT'``, so the value ``DataFrame.to_dict('records')``
    produces for a missing timestamp cell was sealed as
    ``test_timestamp='NaT'`` with ``test_timestamp_source='declared_by_caller'``
    and ZERO warnings. Any object with an ``isoformat()`` that returned
    ``'banana'`` did the same. The candidate is now read in ONE place, ABOVE the
    branch that produced it, so a second producer cannot reopen the hole.
    """
    if not isinstance(candidate, str):
        return False
    stripped = candidate.strip()
    if not stripped:
        return False
    try:
        datetime.fromisoformat(stripped)
    except (ValueError, TypeError):
        return False
    return True


def compute_signed_test_log(
    test_results: dict,
    data_hash: str,
    lib_version: str,
    intervention_history: Optional[list] = None,
) -> dict:
    """Create an Annex IV Section 6 signed test log.

    Produces a tamper-evident record of the test execution, including a
    SHA-256 content hash that covers the test results, data hash, and
    library version.

    Parameters
    ----------
    test_results : dict
        Metric test results: ``metrics`` (list of metric dicts),
        ``overall_pass`` (bool), ``timestamp`` (str, optional).
        Each metric's ``passed`` flag is load-bearing and is read as three
        states by :func:`_metric_verdict`: assessed-pass, assessed-fail, or
        could-not-check.  ``passed`` in ``metrics_snapshot`` is therefore
        ``True``/``False``/``None`` and never a default, a row whose verdict was
        not reported carries ``verdict_basis`` and ``verdict_note``, and
        ``test_results_summary`` carries a ``not_assessed`` count so the
        ``content_hash`` seals how much was actually graded.  ``overall_pass``
        is likewise ``None`` when the caller reported none.
    data_hash : str
        SHA-256 hash of the evaluation dataset.
    lib_version : str
        vfairness library version used for the test run.
    intervention_history : list, optional
        List of intervention records applied before testing.

    Returns
    -------
    dict
        Signed test log with: ``test_timestamp``, ``test_timestamp_source``,
        ``log_built_at``, ``data_hash``, ``library_version``,
        ``test_results_summary``, ``intervention_history``,
        ``metrics_snapshot``, ``reproducibility_info``, ``content_hash``.

        ``test_timestamp`` is the time the CALLER declared in
        ``test_results["timestamp"]`` when it declared one, and
        ``test_timestamp_source`` says which of the THREE states it is
        (``declared_by_caller``, ``log_build_time`` when nothing was declared, or
        ``declared_but_unreadable`` when something was declared that this
        function cannot read as a time, which is neither of the other two),
        because a provenance record must not present the moment the log was
        assembled as the moment the test ran, and must not report a declaration
        it could not read as no declaration at all.

        ``content_hash`` covers EVERY field above except itself: the timestamps
        and their source, the data hash, the library version, the metrics
        snapshot, the summary (including its ``not_assessed`` count), the
        ``intervention_history`` and the ``reproducibility_info`` block. To verify,
        rebuild that payload from the returned document with ``sort_keys=True``
        and ``default=str`` and compare.
    """
    now = _now_iso()

    # BGL5 AUDIT, 2026-09-27. ``timestamp`` is documented in this function's own
    # Parameters section above and was never read, so the sealed test_timestamp
    # was the moment the log was BUILT, presented inside an Annex IV provenance
    # record as the time of the test. Measured:
    #   declared test timestamp : 2024-01-15T09:00:00+00:00
    #   sealed  test_timestamp  : 2026-09-27T18:17:53.266974+00:00
    # After this change the declared value is sealed as test_timestamp, the build
    # time travels separately as log_built_at, and test_timestamp_source names
    # which of the two the reader is looking at. THREE STATES rather than a
    # silent substitution: declared_by_caller, or log_build_time when no
    # timestamp was declared (in which case test_timestamp still carries the
    # build time, as before, but no longer claims to be a measurement of when
    # the test ran).
    #
    # BGL6 AUDIT, 2026-09-29. The guard was ``isinstance(declared, str)``, so a
    # declaration that was not a string fell to the ELSE branch and the document
    # reported ``test_timestamp_source`` = 'log_build_time', whose documented
    # meaning is "no timestamp was declared". Measured before this change with
    # ``test_results["timestamp"] = datetime(2024, 1, 15, 9, 0, 0)``: sealed
    # test_timestamp = the build time, source 'log_build_time'. The caller HAD
    # declared one, so a provenance record stated a falsehood about its own
    # provenance, which is the two-state collapse this block was written to end
    # arriving through the type of the field instead of its absence.
    #
    # THREE STATES, and now they are three and not two-dressed-as-three:
    #   declared_by_caller       the caller declared a time this function could
    #                            read (an ISO string, or a datetime/date, which
    #                            is normalised through .isoformat() and then READ
    #                            like any other candidate string, because
    #                            .isoformat() returning something is not the same
    #                            as it returning a time: see pd.NaT below).
    #   log_build_time           nothing was declared. test_timestamp carries the
    #                            build time and does not claim to be a
    #                            measurement of when the test ran.
    #   declared_but_unreadable  something WAS declared and could not be read as
    #                            a time. NOT folded into either of the others. An
    #                            epoch number is deliberately here rather than
    #                            converted: seconds and milliseconds cannot be
    #                            told apart from the value, and guessing the unit
    #                            would put a fabricated date in a sealed record.
    #                            A warning names the value that was refused, so
    #                            the caller sees what this document could not use.
    #
    # BGL7 A-operations-4-c, 2026-09-29. The block above fixed the STRING branch
    # and left its SIBLING, the object branch, believing whatever .isoformat()
    # returned. Measured before this change:
    #   compute_signed_test_log({"timestamp": pd.NaT, "metrics": []}, ...)
    #     -> test_timestamp 'NaT', source 'declared_by_caller', ZERO warnings
    #   an object whose isoformat() returns "banana"
    #     -> test_timestamp 'banana', source 'declared_by_caller', ZERO warnings
    # pd.NaT IS an instance of datetime.datetime, and it is what
    # DataFrame.to_dict('records') puts in a missing timestamp cell, so the exact
    # provenance falsehood the block above says it ended was reachable through
    # the ordinary way these records are built. BOTH branches now hand their
    # candidate to :func:`_reads_as_iso_time`, the ONE reader, so the unreadable
    # object falls to 'declared_but_unreadable' below exactly where the
    # unreadable string and the epoch int already went.
    declared = test_results.get("timestamp")
    _declared_iso = None
    if not isinstance(declared, str) and hasattr(declared, "isoformat"):
        try:
            _iso = declared.isoformat()
            # READ before believed, the same test the string branch applies.
            if _reads_as_iso_time(_iso):
                _declared_iso = _iso.strip()
        except Exception:  # noqa: BLE001
            _declared_iso = None
    # BGL5 A-operations-4-b, 2026-09-29. The string branch was
    # `isinstance(declared, str) and declared.strip()` and PARSED NOTHING, so the
    # one type this function did not validate was the one type it trusted.
    # Measured before, compute_signed_test_log({"timestamp": X, "metrics": []},
    # "abc123", "0.1.0"):
    #   X = "banana"     -> test_timestamp 'banana', source 'declared_by_caller',
    #                       ZERO warnings
    #   X = "2024-13-45" -> test_timestamp '2024-13-45', source
    #                       'declared_by_caller', ZERO warnings
    # while an epoch int on the same run was correctly refused as
    # 'declared_but_unreadable' with a loud warning. 'declared_by_caller' is
    # documented three comment blocks up as "a time this function could read", so
    # the string was folded into a provenance state nothing established, and then
    # sealed into content_payload and the 64-char content_hash below.
    #
    # The string is now READ before it is believed. It still travels verbatim when
    # it parses, so the seal covers the caller's own spelling rather than a
    # re-formatted copy of it (tests/test_bgl4_operations_4.py asserts that
    # equality); a string that does not parse falls to the
    # 'declared_but_unreadable' branch below, exactly where the epoch int already
    # went. A blank string is NOT touched by this: it stays "nothing was
    # declared" in the branch after next.
    _declared_str = declared.strip() if isinstance(declared, str) else ""
    # The SAME reader the object branch above uses, so the two cannot drift apart
    # again (BGL7 A-operations-4-c).
    _declared_str_reads_as_time = _reads_as_iso_time(_declared_str)
    if _declared_str and _declared_str_reads_as_time:
        test_timestamp = _declared_str
        timestamp_source = "declared_by_caller"
    elif _declared_iso is not None:
        test_timestamp = _declared_iso
        timestamp_source = "declared_by_caller"
    elif declared is None or (isinstance(declared, str) and not declared.strip()):
        test_timestamp = now
        timestamp_source = "log_build_time"
    else:
        test_timestamp = now
        timestamp_source = "declared_but_unreadable"
        warnings.warn(
            f"compute_signed_test_log: test_results['timestamp'] was declared as "
            f"{declared!r} ({type(declared).__name__}), which this function cannot "
            f"read as a time, so test_timestamp carries the LOG BUILD TIME and "
            f"test_timestamp_source is 'declared_but_unreadable'. This is not "
            f"'no timestamp was declared' and it is not a measurement of when the "
            f"test ran. Pass an ISO 8601 string or a datetime. An epoch number is "
            f"refused on purpose: seconds and milliseconds cannot be distinguished "
            f"from the value. A STRING is refused when datetime.fromisoformat "
            f"cannot read it, and so is an OBJECT whose .isoformat() returns "
            f"something that does not read as a time (pd.NaT returns the string "
            f"'NaT'), because a sealed record must not report a provenance state "
            f"this function never established.",
            UserWarning,
            stacklevel=2,
        )

    metrics = test_results.get("metrics", [])
    # Same three states for the run-level flag: a caller that reported no
    # overall verdict gets None (could-not-check), not the fabricated FAIL that
    # ``.get("overall_pass", False)`` used to seal into the hash.
    overall_pass = _coerce_passed(test_results.get("overall_pass", _MISSING))

    # ONE reader for the snapshot AND the summary.
    #
    # These two blocks used to disagree by construction: the snapshot defaulted
    # ``passed`` to False and the summary defaulted it to True, so a single
    # metric with no verdict was recorded as failed in one half of this document
    # and counted as passed in the other. The content_hash below was then
    # computed OVER that contradiction, sealing a fabricated verdict into a
    # provenance record. Both halves now read every row through
    # :func:`_metric_verdict`, so the document cannot contradict itself, and the
    # unassessed count travels inside the hashed summary so the seal covers the
    # true state rather than a rounded-off version of it.
    verdicts = [_metric_verdict(m) for m in metrics]

    metrics_snapshot = []
    for m, (passed, basis, reason) in zip(metrics, verdicts):
        entry = {
            "name": m.get("name", "unknown"),
            "value": m.get("value"),
            "threshold": m.get("threshold"),
            "passed": passed,  # True / False / None, never a default
        }
        if basis != _VERDICT_REPORTED:
            # Present exactly when this row's verdict was NOT reported by the
            # pipeline, so its presence in a sealed log is itself the signal.
            entry["verdict_basis"] = basis
            entry["verdict_note"] = reason
        metrics_snapshot.append(entry)

    counts = _counts_from_verdicts(verdicts)  # the SAME reading the snapshot used
    summary = {
        "total_metrics": len(metrics),
        "passed": counts["passed"],
        "failed": counts["failed"],
        # Third bucket. Always present, including as 0: an omitted-when-empty
        # field would leave "nothing was left ungraded" to be inferred from an
        # absence, which is the same silence this fix exists to remove.
        "not_assessed": counts["not_assessed"],
        "derived": counts["derived"],
        "overall_pass": overall_pass,
    }

    interventions = intervention_history or []

    reproducibility_info = {
        "library": "vfairness",
        "library_version": lib_version,
        "data_hash_algorithm": "SHA-256",
        "data_hash": data_hash,
        "timestamp": now,
        "python_reproducibility_note": (
            "Results are reproducible given the same data hash, library version, "
            "and random seed (if applicable)."
        ),
    }

    # Build content hash for tamper evidence.
    #
    # BGL5 AUDIT, 2026-09-27. ``intervention_history`` is RETURNED in this
    # document and was excluded from the payload, so the seal did not cover it.
    # Measured: a log built with intervention_history=[{"intervention":
    # "reweighting"}], then rehashed over the same five fields with the history
    # ERASED, produced content_hash c682d5894c939e7c... which MATCHED the sealed
    # hash, although the docstring calls this "a tamper-evident record of the
    # test execution" and lists the interventions as "applied before testing".
    # Deleting every record of a mitigation left the seal intact. It is inside
    # the payload now, with the timestamp fields, so a recomputation over the
    # returned document reproduces the hash only while all of it is unchanged.
    # AND THE REPRODUCIBILITY BLOCK, which was the same hole in a second returned
    # field (BGL6 F06, 2026-09-28). It was excluded from the payload, and the
    # docstring said so, but a docstring is not a disclosure a reader of the JSON
    # can see, and this document is called "a tamper-evident record of the test
    # execution". Measured before: erasing intervention_history broke the seal (the
    # 2026-09-27 fix), while replacing reproducibility_info['data_hash'] with a
    # different 64-character hash, or ['library_version'] with '9.9.9', or the note
    # with "Any result may be substituted freely.", all left the recomputed hash
    # EQUAL to the sealed one. The block duplicates two fields that ARE sealed, so a
    # reader taking the dataset identity from it was reading an unsealed, forgeable
    # copy inside a sealed document, and the copy is the one a person is most likely
    # to read: it is the block whose name says reproducibility.
    #
    # This CHANGES THE HASH for a given run, deliberately. A log written by an
    # earlier version verifies against the rule that version used; there is no
    # silent re-interpretation, because the seal is recomputed from the document in
    # front of the verifier and the payload composition is the rule. Nothing about a
    # previously issued log becomes false: it was sealed over less, and this is what
    # stops the next one being.
    content_payload = json.dumps(
        {
            "test_timestamp": test_timestamp,
            "test_timestamp_source": timestamp_source,
            "log_built_at": now,
            "data_hash": data_hash,
            "lib_version": lib_version,
            "metrics_snapshot": metrics_snapshot,
            "summary": summary,
            "intervention_history": interventions,
            "reproducibility_info": reproducibility_info,
        },
        sort_keys=True,
        default=str,
    )
    content_hash = hashlib.sha256(content_payload.encode("utf-8")).hexdigest()

    return {
        "test_timestamp": test_timestamp,
        "test_timestamp_source": timestamp_source,
        "log_built_at": now,
        "data_hash": data_hash,
        "library_version": lib_version,
        "test_results_summary": summary,
        "intervention_history": interventions,
        "metrics_snapshot": metrics_snapshot,
        "reproducibility_info": reproducibility_info,
        "content_hash": content_hash,
    }


# 5. Annex IV Aggregator


def generate_annex_iv_data(
    system_profile: dict,
    metric_results: dict,
    risk_register: list,
    monitoring_config: Optional[dict] = None,
    intervention_results: Optional[dict] = None,
) -> dict:
    """Aggregate wizard data into the EU AI Act Annex IV 9-section structure.

    Each section reports which fields were auto-populated from the wizard
    pipeline and which require manual completion.  Coverage percentage
    reflects the proportion of auto-populated fields.

    Parameters
    ----------
    system_profile : dict
        System metadata (name, purpose, description, domain, version, etc.).
    metric_results : dict
        Output from the metric computation module.  Each ``metrics`` row's
        ``passed`` flag is read as three states (see :func:`_metric_verdict`).
        Sections 6 and 9 report ``not_assessed`` separately from ``passed`` and
        ``failed``, and are ``partial`` rather than ``auto`` when no metric could
        be graded at all.
    risk_register : list
        Output from :func:`generate_risk_register_from_audit`.
    monitoring_config : dict, optional
        Monitoring pipeline configuration (alert thresholds, drift settings).
    intervention_results : dict, optional
        Results of fairness interventions (pre/post metrics, method used).

    Returns
    -------
    dict
        Keys: ``annex_iv_id``, ``generated_at``, ``system_name``,
        ``overall_coverage``, ``sections`` (list of section dicts).

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: generate_annex_iv. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    sys_name = system_profile.get("name", "Unnamed AI System")
    monitoring = monitoring_config or {}
    interventions = intervention_results or {}

    metrics = metric_results.get("metrics", [])
    has_metrics = len(metrics) > 0
    # Recording metrics is not the same as having graded any of them. A run whose
    # every row carries no verdict must not report sections 6 and 9 as
    # auto-covered: "Testing and Validation: auto" over nothing that was
    # actually assessed is the section-level version of the PASS default.
    metric_counts = _metric_verdict_counts(metrics)
    has_assessed_metrics = (metric_counts["passed"] + metric_counts["failed"]) > 0
    has_risks = len(risk_register) > 0
    # The register-level counterpart of ``has_assessed_metrics``: holding risk
    # entries is not the same as having SCORED any of them. Sections 4 and 9 read
    # this rather than ``has_risks`` for their status, so "Risk Management System:
    # auto" is not printed over a register whose every entry is unscored.
    has_scored_risks = has_risks and _count_unscored(risk_register) < len(risk_register)
    has_monitoring = bool(monitoring)
    has_interventions = bool(interventions)

    sections: List[Dict[str, Any]] = []

    # Section 1: General Description
    s1_auto = ["name", "purpose", "domain"]
    s1_manual = ["intended_use_conditions", "geographic_scope", "ce_marking_info"]
    sections.append(
        {
            "section_number": 1,
            "title": "General Description",
            "article": "Art. 11",
            "status": "partial",
            "coverage_percent": _compute_coverage(s1_auto, s1_manual),
            "auto_fields": s1_auto,
            "manual_fields": s1_manual,
            "content": {
                "name": sys_name,
                "purpose": system_profile.get("purpose", ""),
                "domain": system_profile.get("domain", ""),
                "version": system_profile.get("version", ""),
                "description": system_profile.get("description", ""),
            },
        }
    )

    # Section 2: Detailed Description
    s2_auto = ["architecture", "input_features", "output_type"]
    s2_manual = ["training_methodology_detail", "hardware_requirements"]
    if system_profile.get("architecture"):
        s2_auto.append("architecture_detail")
    sections.append(
        {
            "section_number": 2,
            "title": "Detailed Description of Elements and Process",
            "article": "Art. 11",
            "status": "partial",
            "coverage_percent": _compute_coverage(s2_auto, s2_manual),
            "auto_fields": s2_auto,
            "manual_fields": s2_manual,
            "content": {
                "architecture": system_profile.get("architecture", ""),
                "input_features": system_profile.get("features", []),
                "output_type": system_profile.get("output_type", ""),
            },
        }
    )

    # Section 3: Monitoring, Functioning, and Control
    s3_auto = ["monitoring_metrics", "alert_config"] if has_monitoring else []
    s3_manual = ["human_oversight_procedures", "shutdown_mechanism"]
    if not has_monitoring:
        s3_manual.extend(["monitoring_metrics", "alert_config"])
    sections.append(
        {
            "section_number": 3,
            "title": "Monitoring, Functioning, and Control",
            "article": "Art. 12-14",
            "status": "auto" if has_monitoring else "partial",
            "coverage_percent": _compute_coverage(s3_auto, s3_manual),
            "auto_fields": s3_auto,
            "manual_fields": s3_manual,
            "content": {
                "monitoring_config": monitoring,
                "metrics_tracked": [m.get("name", "") for m in metrics],
            },
        }
    )

    # Section 4: Risk Management System
    s4_auto = ["risk_register", "risk_categories", "treatments"] if has_risks else []
    s4_manual = (
        ["risk_acceptance_criteria"]
        if has_risks
        else ["risk_register", "risk_categories", "treatments", "risk_acceptance_criteria"]
    )
    sections.append(
        {
            "section_number": 4,
            "title": "Risk Management System",
            "article": "Art. 9",
            "status": "auto" if has_scored_risks else "partial",
            "coverage_percent": _compute_coverage(s4_auto, s4_manual),
            "auto_fields": s4_auto,
            "manual_fields": s4_manual,
            "content": {
                "risk_register": risk_register,
                "total_risks": len(risk_register),
                # Read through _risk_score, never `.get("risk_score", 0)`: the
                # key is PRESENT holding None on an unscored entry, so the
                # default cannot fire and the comparison raises. The unscored
                # count travels beside the high count so a zero there is not
                # readable as "none of them are high".
                "high_risks": sum(1 for r in risk_register if _is_scored_at_or_above(r, 15)),
                "not_scored": _count_unscored(risk_register),
            },
        }
    )

    # Section 5: Data and Data Governance
    s5_auto = ["dataset_description", "protected_attributes"]
    s5_manual = ["data_collection_process", "annotation_methodology", "data_gaps"]
    sections.append(
        {
            "section_number": 5,
            "title": "Data and Data Governance",
            "article": "Art. 10",
            "status": "partial",
            "coverage_percent": _compute_coverage(s5_auto, s5_manual),
            "auto_fields": s5_auto,
            "manual_fields": s5_manual,
            "content": {
                "features": system_profile.get("features", []),
                "protected_attributes": system_profile.get("protected_attributes", []),
            },
        }
    )

    # Section 6: Testing and Validation
    s6_auto = ["metric_results", "pass_fail_summary", "test_log"] if has_metrics else []
    s6_manual = (
        ["test_environment_description"]
        if has_metrics
        else [
            "metric_results",
            "pass_fail_summary",
            "test_log",
            "test_environment_description",
        ]
    )
    sections.append(
        {
            "section_number": 6,
            "title": "Testing and Validation",
            "article": "Art. 15",
            "status": "auto" if has_assessed_metrics else "partial",
            "coverage_percent": _compute_coverage(s6_auto, s6_manual),
            "auto_fields": s6_auto,
            "manual_fields": s6_manual,
            "content": {
                "metrics": metrics,
                # ``total_recorded`` is how many rows arrived; ``total_evaluated``
                # is how many of them a verdict could be established for. They
                # were the same number only because unassessed rows used to be
                # counted as passes.
                "total_recorded": len(metrics),
                "total_evaluated": metric_counts["passed"] + metric_counts["failed"],
                "passed": metric_counts["passed"],
                "failed": metric_counts["failed"],
                "not_assessed": metric_counts["not_assessed"],
            },
        }
    )

    # Section 7: Accuracy and Robustness
    s7_auto = ["fairness_metrics"] if has_metrics else []
    s7_manual = ["accuracy_metrics", "robustness_testing", "adversarial_testing"]
    if has_interventions:
        s7_auto.append("intervention_results")
    else:
        s7_manual.append("intervention_results")
    sections.append(
        {
            "section_number": 7,
            "title": "Accuracy and Robustness",
            "article": "Art. 15",
            "status": "partial",
            "coverage_percent": _compute_coverage(s7_auto, s7_manual),
            "auto_fields": s7_auto,
            "manual_fields": s7_manual,
            "content": {
                "fairness_metrics": metrics,
                "interventions_applied": interventions,
            },
        }
    )

    # Section 8: Transparency and User Information
    s8_auto = ["model_card_available"]
    s8_manual = ["user_instructions", "limitations_disclosure", "contact_info"]
    sections.append(
        {
            "section_number": 8,
            "title": "Transparency and User Information",
            "article": "Art. 13",
            "status": "partial",
            "coverage_percent": _compute_coverage(s8_auto, s8_manual),
            "auto_fields": s8_auto,
            "manual_fields": s8_manual,
            "content": {
                "model_card_available": True,
                "system_description": system_profile.get("description", ""),
            },
        }
    )

    # Section 9: Fundamental Rights Impact Assessment
    s9_auto = ["risk_register_summary", "metric_fairness_summary"] if has_risks else []
    s9_manual = ["stakeholder_consultation", "fundamental_rights_analysis"]
    if not has_risks:
        s9_manual.extend(["risk_register_summary", "metric_fairness_summary"])
    sections.append(
        {
            "section_number": 9,
            "title": "Fundamental Rights Impact Assessment",
            "article": "Art. 29a",
            "status": "auto" if has_scored_risks and has_assessed_metrics else "partial",
            "coverage_percent": _compute_coverage(s9_auto, s9_manual),
            "auto_fields": s9_auto,
            "manual_fields": s9_manual,
            "content": {
                "risk_register_summary": {
                    "total": len(risk_register),
                    # Same reading as section 4: scored-and-high, with the
                    # unscored entries in their own bucket rather than folded
                    # into "not high".
                    "high_severity": sum(1 for r in risk_register if _is_scored_at_or_above(r, 15)),
                    "not_scored": _count_unscored(risk_register),
                },
                "fairness_metrics_summary": {
                    "total": len(metrics),
                    "failed": metric_counts["failed"],
                    "passed": metric_counts["passed"],
                    "not_assessed": metric_counts["not_assessed"],
                },
            },
        }
    )

    # Overall coverage.
    all_auto = sum(len(s["auto_fields"]) for s in sections)
    all_total = all_auto + sum(len(s["manual_fields"]) for s in sections)
    overall_coverage = round(all_auto / all_total * 100, 1) if all_total else 0.0

    return {
        "annex_iv_id": f"AIV-{hashlib.sha256(sys_name.encode()).hexdigest()[:8].upper()}",
        "generated_at": _now_iso(),
        "system_name": sys_name,
        "overall_coverage": overall_coverage,
        "sections": sections,
    }


# 6. Model Card


def generate_model_card(
    system_profile: dict,
    metric_results: dict,
    interventions: Optional[list] = None,
) -> str:
    """Generate a fairness-enhanced model card in Markdown format.

    Follows the structure from Mitchell et al. (2019) "Model Cards for
    Model Reporting", extended with fairness-specific sections.

    Parameters
    ----------
    system_profile : dict
        System metadata (name, purpose, domain, version, description,
        architecture, limitations, etc.).
    metric_results : dict
        Output from the metric computation module: ``metrics`` is a list of
        dicts with ``name``, ``value``, ``threshold`` and ``passed``.

        ``passed`` is load-bearing.  It is read as THREE states, never two (see
        :func:`_metric_verdict`): a real boolean is the reported verdict; a row
        with no boolean verdict is graded from ``value`` against ``threshold``
        when the named metric's better-direction is known, and marked
        ``(derived)``; anything else is rendered ``NOT ASSESSED``, counted in its
        own bucket, and never counted as passed or failed.  A card whose rows
        carry no verdict therefore reports that fact instead of an all-clear.
    interventions : list, optional
        List of intervention dicts with ``method``, ``description``,
        ``pre_metric``, ``post_metric``.

    Returns
    -------
    str
        Markdown-formatted model card.

    Warns
    -----
    UserWarning
        When the card is a COULD-NOT-CHECK: no metric row was recorded at all,
        or some rows could not be graded. The card says so in its own text, but
        a programmatic consumer gets back one Markdown string and would have to
        string-match to find out, so the state is also raised. Added BGL-S2b
        (2026-09-17); until then the only channel was the prose.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: generate_model_card. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    name = system_profile.get("name", "Unnamed AI System")
    version = system_profile.get("version", "N/A")
    purpose = system_profile.get("purpose", "Not specified")
    domain = system_profile.get("domain", "general")
    description = system_profile.get("description", "")
    architecture = system_profile.get("architecture", "Not specified")
    limitations = system_profile.get("limitations", "Not documented")
    protected = system_profile.get("protected_attributes", [])

    metrics = metric_results.get("metrics", [])
    interventions = interventions or []

    # Read every row ONCE, three-state, and reuse that reading for the summary
    # counts, the table and the ethical-considerations verdict, so the card
    # cannot say "1 passed" above a row it prints as NOT ASSESSED.
    verdicts = [_metric_verdict(m) for m in metrics]
    counts = _counts_from_verdicts(verdicts)
    n_unassessed = counts["not_assessed"]

    lines = [
        f"# Model Card: {name}",
        "",
        f"**Version:** {version}  ",
        f"**Generated:** {_now_iso()}  ",
        "**Generator:** vfairness compliance reporting",
        "",
        "---",
        "",
        "## Model Details",
        "",
        f"- **Name:** {name}",
        f"- **Version:** {version}",
        f"- **Architecture:** {architecture}",
        f"- **Domain:** {domain}",
        f"- **Description:** {description}",
        "",
        "## Intended Use",
        "",
        f"- **Primary purpose:** {purpose}",
        f"- **Target domain:** {domain}",
        # "Any use OUTSIDE the stated purpose has not been evaluated" carries
        # the implicature that the use INSIDE it has been. On a zero-metric
        # card that implicature is false, so the sentence says which it is.
        (
            "- **Out-of-scope uses:** Any use outside the stated purpose or domain "
            "has not been evaluated for fairness."
            if metrics
            else (
                "- **Out-of-scope uses:** Any use outside the stated purpose or domain "
                "has not been evaluated for fairness. Neither has any use inside it: "
                "no fairness metric was recorded for this model at all."
            )
        ),
        "",
        "## Fairness Considerations",
        "",
        # SIBLING OF THE ETHICS FIX (BGL-S2b, 2026-09-17). "evaluated" was
        # unconditional and printed one line above "Number of fairness metrics
        # recorded: 0". Nothing was evaluated on a zero-metric card; the
        # attributes were DECLARED by the submitter. The ethical-considerations
        # sentence below was already relabelled for exactly this case and this
        # line, which a reader meets first, was not.
        (
            f"- **Protected attributes evaluated:** "
            f"{', '.join(protected) if protected else 'None specified'}"
            if metrics
            else (
                f"- **Protected attributes declared:** "
                f"{', '.join(protected) if protected else 'None specified'} "
                f"(declared by the submitter; NOT evaluated, see below)"
            )
        ),
        # "recorded", not "evaluated": with a third bucket in play, the number of
        # rows that arrived is no longer the number of rows that were graded, and
        # the old label claimed the larger of the two.
        f"- **Number of fairness metrics recorded:** {len(metrics)}",
        f"- **Metrics passed:** {counts['passed']}",
        f"- **Metrics failed:** {counts['failed']}",
        # The third bucket is stated always, including at 0. A count that only
        # appears when it is non-zero teaches a reader to read its absence as
        # reassurance, which is the same silence the PASS default provided.
        f"- **Metrics not assessed:** {n_unassessed}",
        "",
    ]

    if interventions:
        lines.append("### Interventions Applied")
        lines.append("")
        for i, intv in enumerate(interventions, 1):
            lines.append(f"**{i}. {intv.get('method', 'Unknown method')}**")
            lines.append(f"- Description: {intv.get('description', 'N/A')}")
            if "pre_metric" in intv and "post_metric" in intv:
                lines.append(
                    f"- Pre-intervention: {intv['pre_metric']:.4f} "
                    f"-> Post-intervention: {intv['post_metric']:.4f}"
                )
            lines.append("")

    lines.extend(
        [
            "## Metrics",
            "",
            "| Metric | Value | Threshold | Status |",
            "|--------|-------|-----------|--------|",
        ]
    )
    for m, (passed, basis, _reason) in zip(metrics, verdicts):
        # NOT ASSESSED is its own token in this column. It is not a PASS with a
        # caveat and not a soft FAIL: the previous "PASS if passed else FAIL"
        # printed a green verdict on a row whose value breached its own threshold
        # four times over, purely because the row carried no flag.
        if passed is None:
            status = "NOT ASSESSED"
        elif basis == _VERDICT_DERIVED:
            status = "PASS (derived)" if passed else "FAIL (derived)"
        else:
            status = "PASS" if passed else "FAIL"
        value = m.get("value", "N/A")
        threshold = m.get("threshold", "N/A")
        if isinstance(value, float):
            value = f"{value:.4f}"
        if isinstance(threshold, float):
            threshold = f"{threshold:.4f}"
        lines.append(f"| {m.get('name', 'unknown')} | {value} | {threshold} | {status} |")

    # The legend appears only when the table actually contains one of these
    # states, so a fully-reported card is unchanged.
    if any(basis != _VERDICT_REPORTED for _passed, basis, _reason in verdicts):
        lines.append("")
        if n_unassessed:
            lines.append(
                "*NOT ASSESSED*: the row carries no pass/fail verdict and could not "
                "be graded from its recorded value and threshold (unknown metric "
                "direction, a value or threshold that is not a number, or a bound "
                "no value could breach). It is neither a pass nor a failure, and "
                "this card makes no claim that the metric was checked."
            )
        if counts["derived"]:
            lines.append(
                "*(derived)*: no verdict was reported for the row, so it was graded "
                "here by comparing the recorded value against the recorded "
                "threshold in that metric's known direction."
            )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            f"{limitations}",
            "",
            "## Ethical Considerations",
            "",
            # THREE STATES on the headline sentence too. "has been evaluated"
            # was unconditional, so a card built from zero metric rows asserted
            # that an evaluation happened. It did not: the same call renders an
            # empty Metrics table and "Number of fairness metrics recorded: 0"
            # a few lines above.
            (
                "- This model has been evaluated using the vfairness fairness assessment pipeline."
                if metrics
                else "- This model was submitted to the vfairness pipeline, but 0 "
                "fairness metric(s) were recorded for it."
            ),
            f"- {len(protected)} protected attribute(s) were considered during evaluation."
            if metrics
            else f"- {len(protected)} protected attribute(s) are declared for this model.",
        ]
    )

    n_failed = counts["failed"]
    if n_failed > 0:
        lines.append(
            f"- **Warning:** {n_failed} fairness metric(s) did not meet their "
            "threshold. Review the risk register and consider mitigation before "
            "deployment."
        )
    elif not metrics:
        # ZERO recorded metrics satisfies `n_unassessed == 0` vacuously, so the
        # guard below (which exists precisely to stop a fabricated all-clear)
        # was passed by the one case with no evidence at all: an empty metric
        # list printed "All evaluated fairness metrics met their thresholds",
        # byte-identical to the card for a model that genuinely passed. An
        # empty set of measurements is a could-not-check, never a pass.
        lines.append(
            "- **Not assessed:** no fairness metric was recorded for this model. "
            "This card makes no claim that any fairness property was checked, and "
            "the absence of a reported failure below is not a pass."
        )
    elif n_unassessed == 0:
        # The all-clear is only available when every recorded metric was actually
        # graded. "All evaluated metrics met their thresholds" printed above a
        # table of rows nobody evaluated is a fabricated all-clear, and it was
        # the sentence a reader was most likely to act on.
        lines.append(
            "- All evaluated fairness metrics met their thresholds at the time of assessment."
        )
    if n_unassessed > 0:
        lines.append(
            f"- **Not assessed:** {n_unassessed} recorded fairness metric(s) carry "
            "no pass/fail verdict and could not be graded from their value and "
            "threshold. They are neither passes nor failures; this card does not "
            "claim they were checked, and they must be assessed before the "
            "remaining results are relied upon."
        )

    if interventions:
        lines.append(
            f"- {len(interventions)} fairness intervention(s) were applied "
            "to improve model behaviour."
        )

    if branding_enabled():
        lines.extend(
            [
                "",
                "---",
                "",
                "*Generated by vfairness compliance reporting | validant.ai*",
            ]
        )

    # The third state, on a channel a program can branch on. generate_model_card
    # returns one Markdown string, so before this the ONLY way to tell a card
    # built from zero measurements from a card that passed was to match its
    # prose. A caller that renders this into a portal, or counts cards, saw no
    # difference at all.
    if not metrics:
        warnings.warn(
            f"generate_model_card({name!r}): 0 fairness metric(s) were recorded, so "
            f"this card reports NOTHING that was checked. The "
            f"{len(protected)} protected attribute(s) it names are declared by the "
            f"submitter, not evaluated. This is could not check, NOT a pass.",
            UserWarning,
            stacklevel=2,
        )
    elif n_unassessed:
        warnings.warn(
            f"generate_model_card({name!r}): {n_unassessed} of {len(metrics)} recorded "
            f"fairness metric(s) carry no pass/fail verdict and could not be graded "
            f"from their value and threshold. They are rendered NOT ASSESSED and are "
            f"counted as neither passes nor failures; the card withholds its "
            f"all-clear sentence while any remain.",
            UserWarning,
            stacklevel=2,
        )

    return "\n".join(lines)


# 7. ISO 42001 Evidence Map


def _iso_value_has_content(value: Any, _depth: int = 0) -> bool:
    """Whether a value nested anywhere inside a wizard section is EVIDENCE.

    BGL6 AUDIT, 2026-09-29. :func:`_iso_section_has_content` tested the leaves of
    a mapping one level deep only, so a value that was itself a mapping counted
    as content whenever it was non-empty, and ``{"purpose": {"note": None}}``
    read as a filled-in purpose. Measured before this helper existed: the nine
    sections of the wizard each holding one singly nested ``None`` earned
    coverage_percent 92.3 with 11 covered, 2 partial and 0 gaps, a result
    identical in every actionable field to the FULL wizard.

    A scalar that is not None is content at any depth, so a nested measured zero
    (``{"metric": {"value": 0.0}}``) and a nested ``False`` both count; refusing
    them would be the mirror defect. A container is content only when something
    inside it is, recursively.

    ``_depth`` caps the walk so a self-referential wizard cannot take the report
    generator down with a RecursionError; past the cap the answer is False, which
    understates coverage rather than claiming evidence that was never read.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        if _depth >= 12:
            return False
        return any(_iso_value_has_content(v, _depth + 1) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        if _depth >= 12:
            return False
        return any(_iso_value_has_content(v, _depth + 1) for v in value)
    return True


def _iso_section_has_content(value: Any) -> bool:
    """Whether a wizard section carries EVIDENCE, not merely a key.

    BGL5 AUDIT, 2026-09-27. Every control test in
    :func:`generate_iso42001_evidence_map` was ``bool(wizard_data.get(key))``,
    which is True for a section holding one contentless placeholder. Measured on
    nine sections each holding one of ``{"a": None}``, ``[0]``, ``[{}]`` or
    ``{"x": 0}``:

        coverage_percent 92.3, covered 11, partial 2, gaps 0

    the same figure the previous fix recorded for a FULL wizard, with
    "Treatment plans generated for each identified risk" and "Residual risk
    scores computed post-treatment" asserted from a list holding one empty dict.
    With this test the same input reports 19.2% with 10 gaps, while the full
    wizard still reports 84.6% with 0 gaps.

    The rule is structural, never a truthiness test on the leaf: a mapping has
    content when any value is not None, not a blank string and not an empty
    container, so a real ``{"demographic_parity": 0.0}`` or
    ``{"enabled": False}`` still counts (a measured zero is evidence, and
    refusing it would be the mirror defect). A list section is a record set in
    this pipeline, so an element counts only when it is a mapping WITH content or
    a non-blank string: ``[0]`` is not a bias result.
    """
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        # BGL6, 2026-09-29: the walk RECURSES now. The previous loop skipped a
        # value only when it was None, blank or an EMPTY container, so one level
        # of nesting around a None restored the whole defect this function was
        # written to end. See _iso_value_has_content.
        return any(_iso_value_has_content(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(
            (isinstance(item, Mapping) and _iso_section_has_content(item))
            or (isinstance(item, str) and bool(item.strip()))
            for item in value
        )
    # A bare scalar is not a wizard section: every section this map reads is a
    # document (mapping) or a record set (list).
    return False


def _iso_risk_entries_with(field_names: Tuple[str, ...], register: Any) -> int:
    """How many risk entries carry a non-empty value under any of ``field_names``.

    BGL5, 2026-09-27. A.4.3 claimed "Treatment plans generated for each
    identified risk" and A.4.4 "Residual risk scores computed post-treatment"
    from the PRESENCE of a risk register, so both were asserted over
    ``[{}]`` and over ``[{"risk": "proxy discrimination"}]``, which carries
    neither field. :func:`generate_risk_register_from_audit` writes ``treatment``
    and ``residual_risk`` on every entry, so the fields are checkable: when they
    are absent the control is PARTIAL with the reason stated, never covered.
    """
    if not isinstance(register, (list, tuple)):
        return 0
    count = 0
    for entry in register:
        if not isinstance(entry, Mapping):
            continue
        for name in field_names:
            # BGL6, 2026-09-29: recursive, for the same reason as
            # _iso_section_has_content. A `treatment` holding {"plan": None} was
            # counted as a recorded treatment plan, so A.4.3 read "Treatment
            # plans recorded for 1 of 1 risk entries" over a placeholder.
            if _iso_value_has_content(entry.get(name)):
                count += 1
                break
    return count


def generate_iso42001_evidence_map(wizard_data: dict) -> dict:
    """Map wizard outputs to ISO 42001 Annex A controls.

    Inspects the wizard data dict for evidence of coverage against each
    ISO 42001 control and classifies each as ``covered``, ``partial``,
    ``gap``, or ``manual`` (a control this map cannot determine from wizard data;
    excluded from both sides of ``coverage_percent`` rather than credited).

    A section counts as evidence only when it carries CONTENT, not merely a key:
    see :func:`_iso_section_has_content` for the measurement, and
    :func:`_iso_risk_entries_with` for the two controls whose claim names an
    artefact inside the risk register.

    Parameters
    ----------
    wizard_data : dict
        Aggregated wizard output.  Expected top-level keys (all optional):
        ``system_profile``, ``bias_results``, ``metric_results``,
        ``risk_register``, ``monitoring_config``, ``intervention_results``,
        ``dpia``, ``annex_iv``, ``model_card``.

    Returns
    -------
    dict
        Keys: ``map_id``, ``generated_at``, ``standard``,
        ``total_controls``, ``covered``, ``partial``, ``gaps``,
        ``coverage_percent``, ``controls`` (list of control dicts).
    """
    # CONTENT, not presence. See _iso_section_has_content for the measurement:
    # nine sections each holding one contentless placeholder earned 92.3% of ISO
    # 42001 with zero gaps under `bool(wizard_data.get(key))`.
    has_profile = _iso_section_has_content(wizard_data.get("system_profile"))
    has_bias = _iso_section_has_content(wizard_data.get("bias_results"))
    has_metrics = _iso_section_has_content(wizard_data.get("metric_results"))
    has_risks = _iso_section_has_content(wizard_data.get("risk_register"))
    has_monitoring = _iso_section_has_content(wizard_data.get("monitoring_config"))
    has_interventions = _iso_section_has_content(wizard_data.get("intervention_results"))
    has_dpia = _iso_section_has_content(wizard_data.get("dpia"))
    has_annex = _iso_section_has_content(wizard_data.get("annex_iv"))
    has_card = _iso_section_has_content(wizard_data.get("model_card"))

    # The two claims that name an artefact INSIDE the register rather than the
    # register itself. Counted, not assumed; see _iso_risk_entries_with.
    register = wizard_data.get("risk_register")
    n_risk_entries = (
        sum(1 for e in register if isinstance(e, Mapping) and _iso_section_has_content(e))
        if isinstance(register, (list, tuple))
        else 0
    )
    n_treated = _iso_risk_entries_with(("treatment", "treatment_plan", "mitigation"), register)
    n_residual = _iso_risk_entries_with(
        ("residual_risk", "residual", "residual_risk_score"), register
    )

    # Mapping from control ID to (evidence_source, coverage_status, description).
    control_evidence: Dict[str, tuple] = {
        # AI Management System
        "A.2.2": (
            "system_profile",
            "covered" if has_profile else "gap",
            "AI policy derived from system profile and purpose statement."
            if has_profile
            else "No system profile provided.",
        ),
        "A.2.3": (
            "system_profile",
            "partial" if has_profile else "gap",
            "Roles inferred from system profile; manual assignment needed."
            if has_profile
            else "No system profile provided.",
        ),
        "A.2.4": (
            "system_profile",
            # "manual", not "partial": nothing here inspects any evidence, so there
            # is no finding to award half credit to. See the coverage_percent note.
            "manual",
            "Resource allocation requires manual documentation. This map cannot "
            "determine it from wizard data and does not score it.",
        ),
        # Risk Management
        "A.4.2": (
            "risk_register",
            "covered" if has_risks else "gap",
            f"Risk register with {n_risk_entries} entries carrying content."
            if has_risks
            else "No risk assessment performed.",
        ),
        "A.4.3": (
            "risk_register",
            # COUNTED, not assumed. "Treatment plans generated for each
            # identified risk" was asserted from the register's presence alone,
            # so it held over [{}] and over an entry with no treatment field.
            "covered"
            if has_risks and n_treated >= n_risk_entries and n_treated > 0
            else ("partial" if has_risks else "gap"),
            f"Treatment plans recorded for {n_treated} of {n_risk_entries} risk entries."
            if has_risks and n_treated >= n_risk_entries and n_treated > 0
            else (
                f"Risk register present with {n_risk_entries} entry(ies), but only "
                f"{n_treated} carry a treatment plan, so treatment coverage could not be "
                f"evidenced for the rest."
                if has_risks
                else "No risk treatment plans available."
            ),
        ),
        "A.4.4": (
            "risk_register",
            "covered"
            if has_risks and n_residual >= n_risk_entries and n_residual > 0
            else ("partial" if has_risks else "gap"),
            f"Residual risk scored for {n_residual} of {n_risk_entries} risk entries."
            if has_risks and n_residual >= n_risk_entries and n_residual > 0
            else (
                f"Risk register present with {n_risk_entries} entry(ies), but only "
                f"{n_residual} carry a residual-risk value, so post-treatment residual "
                f"risk could not be evidenced for the rest."
                if has_risks
                else "No residual risk assessment available."
            ),
        ),
        # Data Management
        "A.5.2": (
            "metric_results",
            "covered" if has_metrics else "gap",
            "Data quality assessed via fairness metric evaluation."
            if has_metrics
            else "No data quality assessment performed.",
        ),
        "A.5.3": (
            "system_profile",
            "partial" if has_profile else "gap",
            "Data provenance partially documented in system profile."
            if has_profile
            else "No data provenance information.",
        ),
        "A.5.4": (
            "bias_results",
            "covered" if has_bias else "gap",
            "Data preparation bias analysis completed."
            if has_bias
            else "No bias analysis of data preparation.",
        ),
        # Monitoring
        "A.6.2": (
            "monitoring_config",
            "covered" if has_monitoring else "gap",
            "Performance monitoring configured with alert thresholds."
            if has_monitoring
            else "No monitoring configuration provided.",
        ),
        "A.6.3": (
            "intervention_results",
            # The else-branch was "partial", which paid out half credit precisely
            # when there was no intervention evidence at all. Absence of evidence
            # is a gap.
            "covered" if has_interventions else "gap",
            "System changes tracked through intervention history."
            if has_interventions
            else "No intervention tracking configured, so system-change control "
            "could not be evidenced.",
        ),
        "A.6.4": (
            "system_profile",
            # "manual" for the same reason as A.2.4: no evidence test exists.
            "manual",
            "Third-party component assessment requires manual review. This map "
            "cannot determine it from wizard data and does not score it.",
        ),
        # Fairness / Ethics
        "A.8.2": (
            "bias_results + metric_results",
            "covered"
            if (has_bias and has_metrics)
            else ("partial" if (has_bias or has_metrics) else "gap"),
            "Bias detection and fairness metrics provide comprehensive coverage."
            if (has_bias and has_metrics)
            else "Partial fairness assessment available.",
        ),
        "A.8.3": (
            "model_card + annex_iv",
            "covered"
            if (has_card and has_annex)
            else ("partial" if (has_card or has_annex) else "gap"),
            "Transparency documented via model card and Annex IV technical docs."
            if (has_card and has_annex)
            else "Partial transparency documentation.",
        ),
        "A.8.4": (
            "dpia + risk_register",
            "covered"
            if (has_dpia and has_risks)
            else ("partial" if (has_dpia or has_risks) else "gap"),
            "Accountability established through DPIA and risk register."
            if (has_dpia and has_risks)
            else "Partial accountability documentation.",
        ),
    }

    controls = []
    for control_id, control_name, category in _ISO42001_CONTROLS:
        evidence = control_evidence.get(
            control_id, ("unknown", "gap", "No evidence mapped for this control.")
        )
        controls.append(
            {
                "control_id": control_id,
                "control_name": control_name,
                "category": category,
                "evidence_source": evidence[0],
                "coverage_status": evidence[1],
                "evidence_description": evidence[2],
            }
        )

    n_covered = sum(1 for c in controls if c["coverage_status"] == "covered")
    n_partial = sum(1 for c in controls if c["coverage_status"] == "partial")
    n_gap = sum(1 for c in controls if c["coverage_status"] == "gap")
    n_manual = sum(1 for c in controls if c["coverage_status"] == "manual")
    total = len(controls)

    # A CONTROL THIS MAP CANNOT ASSESS EARNS NO CREDIT. It used to earn half.
    #
    # A.2.4 and A.6.4 were hardcoded "partial" with no evidence test of any kind,
    # and A.6.3 returned "partial" in the branch where there was no intervention
    # evidence at all. Because coverage_percent credited every partial at 0.5, an
    # EMPTY wizard reported 10.0% ISO 42001 coverage: (0 + 3*0.5) / 15. Three
    # controls were paying out for evidence nobody had supplied and the function
    # had never looked for.
    #
    # Those controls are now "manual", a fourth state meaning "outside what this
    # map can determine", and they are excluded from the percentage on both sides
    # of the fraction rather than scored as a half-success. That keeps the reader
    # from reading a could-not-assess as partial compliance, and it keeps the
    # denominator honest: the percentage is of what was actually assessable.
    assessable = total - n_manual
    # AND AN EMPTY DENOMINATOR IS NOT 0% (BGL5, 2026-09-28). The fallback here was
    # 0.0, so a map in which NOTHING is assessable reported coverage_percent 0.0:
    # "assessed, and none of it is covered", the strongest adverse compliance
    # statement this function can make, from an assessment that never ran. There
    # is no percentage of zero assessable controls, so it is None and the basis
    # below says which of the two a reader is looking at. Reachable the moment the
    # manual set covers every control, which is a one-line edit away.
    coverage = round((n_covered + n_partial * 0.5) / assessable * 100, 1) if assessable else None

    return {
        "map_id": f"ISO42001-{hashlib.sha256(_now_iso().encode()).hexdigest()[:8].upper()}",
        "generated_at": _now_iso(),
        "standard": "ISO/IEC 42001:2023",
        "total_controls": total,
        "assessable_controls": assessable,
        "covered": n_covered,
        "partial": n_partial,
        "gaps": n_gap,
        "manual": n_manual,
        "coverage_percent": coverage,
        "coverage_basis": (
            (
                f"{n_covered} covered + {n_partial} partial at half credit, over the "
                f"{assessable} control(s) this map can assess from wizard data. "
                f"{n_manual} control(s) require manual documentation and are excluded "
                f"rather than credited."
            )
            if assessable
            else (
                f"NOT ASSESSED: none of the {total} control(s) can be determined from "
                f"wizard data ({n_manual} require manual documentation), so there is "
                "no assessable denominator and coverage_percent is null. This is NOT "
                "0% coverage."
            )
        ),
        "controls": controls,
    }


# ASSURANCE-AUDIT VERDICT ENGINE
#
# Turns the already-computed analysis sections into ONE structured,
# audit-grade verdict object: an assurance opinion
# (Unqualified / Qualified / Adverse / Disclaimer, Lam et al. FAccT 2024),
# a cited one-liner, structured findings (stable IDs + panel refs), routed
# recommendations, deferred-metric rationale, and an audit trail. This is
# the durable artefact that powers the Pulse banner AND pre-fills the
# Navigator. Shared by Pulse and the Navigator (one source of truth). It is
# pure assembly + rules and NEVER raises.
#
# The jurisdiction-aware legal overlay lives HERE, not in the metrics: the
# US EEOC four-fifths 0.80 line is applied only for US employment; the EU /
# UK / CH framing is proportionality + objective justification with no
# numeric bar.

_SEV_RANK = {
    "severe": 4,
    "critical": 4,
    "high": 3,
    "warn": 2,
    "warning": 2,
    "medium": 2,
    "caution": 1,
    "low": 1,
    # RepresentationSeverity.OVERREPRESENTED: a measured, minor skew.
    "overrepresented": 1,
    "info": 0,
    "negligible": 0,
    "pass": 0,
    # RepresentationSeverity.ADEQUATE and HistoricalRiskLevel.NONE: measured
    # clean verdicts. They used to fall to the default rank 1 (minor).
    "adequate": 0,
    "none": 0,
}

# Severity words that mean "no verdict was measured". R-10, 2026-09-09.
# ``RepresentationSeverity.INSUFFICIENT_DATA`` ("Too few rows for any
# verdict") was not in ``_SEV_RANK``, so ``_sev`` handed it the default rank 1,
# a measured minor finding. It then fell under the ``raw >= 2`` materiality cut
# and vanished from the findings of an Unqualified opinion: could-not-check
# collapsed onto minor and the reader was told "no material fairness defect
# found". These words, and any word this table does not know, grade as None.
_UNASSESSED_SEVERITIES = frozenset(
    {"insufficient_data", "unassessed", "not_assessed", "unmeasured", "unscored", "unknown"}
)


def _sev(v: Any) -> Optional[int]:
    """Rank a severity word, or ``None`` when no verdict was measured.

    ``None`` is the third state: the finding exists but carries no measured
    severity (``insufficient_data`` and its synonyms, or a word outside the
    vocabulary, which this grader cannot honestly rank either). Callers must
    exclude ``None`` from every max, cut and comparison and report the
    finding as unassessed; they must never read it as 0 or 1. Before
    2026-09-09 the default was 1.
    """
    word = str(v).lower().split(".")[-1]
    if word in _UNASSESSED_SEVERITIES:
        return None
    return _SEV_RANK.get(word)


def _bias_finding_confirmed(b: dict) -> bool:
    """True when a bias-taxonomy finding is statistically confirmed and may
    therefore escalate the OVERALL assurance opinion to high/critical.

    Confirmation is an explicit ``confirmed``/``significant`` flag, or a
    real ``statisticalTest`` object whose p-value clears alpha (a test
    object that carries no p-value is trusted as its own confirmation,
    e.g. an omnibus-gated chi-square). A finding with ``statisticalTest``
    None and no flag is magnitude-only and is NOT confirmed: it is still
    surfaced, but capped at ``warn`` for the verdict roll-up so sampling
    noise cannot, by itself, produce an Adverse / blocks-deployment
    opinion. Mirrors the intersectional significance gate.
    """
    if b.get("confirmed") is True or b.get("significant") is True:
        return True
    st = b.get("statisticalTest") or b.get("statistical_test")
    if isinstance(st, dict) and st:
        p = st.get("pValue", st.get("p_value"))
        if p is None:
            return True
        try:
            return float(p) < 0.05
        except (TypeError, ValueError):
            return True
    return False


def _is_us_employment(domain: str, jurisdiction: str) -> bool:
    j = (jurisdiction or "").lower()
    d = (domain or "").lower()
    us = (
        any(t in j for t in ("us", "u.s", "usa", "united states", "eeoc", "title vii"))
        and "eu" not in j
    )
    return us and any(
        t in d for t in ("hir", "employ", "recruit", "job", "work", "labor", "labour")
    )


def build_assurance_verdict(
    *,
    schema: Optional[dict] = None,
    per_variable: Optional[List[dict]] = None,
    metrics: Optional[List[dict]] = None,
    bias: Optional[List[dict]] = None,
    proxies: Optional[dict] = None,
    statistical: Optional[dict] = None,
    intersectional: Optional[dict] = None,
    disparity_matrix: Optional[dict] = None,
    recommended: Optional[dict] = None,
    domain: str = "",
    jurisdiction: str = "",
    has_truth: bool = False,
    artifact_hash: Optional[str] = None,
    pulse_version: str = "1.0",
) -> Dict[str, Any]:
    """Assemble the structured assurance verdict. Never raises.

    ``recommended`` is the ``recommend_fairness_definition`` result for
    this run. Only its ``primary`` key is consumed, and only to NAME the
    definition in the disparate-impact recommendation, which otherwise
    says "the recommended fairness definition" without saying which one.
    Nothing else in the verdict is routed off it. When it is absent, empty
    or carries no usable ``primary`` (the agent / LLM / vision probes pass
    ``{}`` because they never run the definition recommender), the generic
    wording is kept: the sentence never names a definition that was not
    handed to it.
    """
    schema = schema or {}
    per_variable = per_variable or []
    metrics = metrics or []
    bias = bias or []
    proxies = proxies or {}
    statistical = statistical or {}
    intersectional = intersectional or {}
    disparity_matrix = disparity_matrix or {}
    recommended = recommended or {}
    primary_definition = ""
    if isinstance(recommended, dict):
        _prim = recommended.get("primary")
        if isinstance(_prim, str) and _prim.strip():
            primary_definition = _prim.strip()
    us_emp = _is_us_employment(domain, jurisdiction)
    findings: List[Dict[str, Any]] = []

    def add(prefix, i, ftype, severity, evidence, panel, **extra):
        f = {
            "id": f"{prefix}-{i:03d}",
            "type": ftype,
            "severity": str(severity).lower().split(".")[-1],
            "evidence": evidence,
            "panelRef": panel,
        }
        f.update(extra)
        findings.append(f)

    # SCHEMA: PII / oracle leakage + declared-vs-inferred mismatches.
    try:
        pii = schema.get("pii_leakage") or []
        if pii:
            add(
                "SCH",
                1,
                "identity_pii_leakage",
                "high",
                f"{len(pii)} identity/PII column(s) present that must never "
                f"feed a model: {', '.join(pii[:6])}" + ("..." if len(pii) > 6 else ""),
                "schema",
                affected_features=pii,
            )
        for j, m in enumerate(schema.get("mismatches") or [], 1):
            add(
                "SCH",
                1 + j,
                "schema_mismatch",
                "warn",
                m.get("detail", ""),
                "schema",
                affected_features=[m.get("column")],
            )
        if schema.get("refuse"):
            add("SCH", 99, "unassessable", "critical", schema.get("refuse_reason", ""), "schema")
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    # DISPARATE IMPACT: per-variable significant gaps (jurisdiction overlay).
    try:
        # BGL6 AUDIT, 2026-09-29. THREE STATES on the gap, AT THE RANKING.
        #
        # ``key=lambda v: v.get("gap", 0.0)`` compares None against a float and
        # raises TypeError, which the ``except Exception`` closing this block
        # swallowed, so ONE assessable row whose gap was never computed deleted
        # EVERY disparate-impact finding in the run. Measured before this change:
        # ``per_variable=[{race, gap 0.23, significant True}]`` gave overall
        # "Adverse", blocksDeployment True, 1 finding; adding one row
        # ``{age, assessable True, gap None}`` gave "Unqualified",
        # blocksDeployment False, findings [], unassessed [], "no material
        # fairness defect found on the assessed attributes". This is the failure
        # the long note below describes for ciLow/ciHigh, still live through the
        # sort key.
        #
        # ``gap = float(v.get("gap") or 0.0)`` was the second half of the same
        # defect: an unreadable magnitude became 0.0 and the row was then skipped
        # by ``gap < 0.05`` as though it had been measured clean, so the
        # attribute appeared nowhere in the payload.
        #
        # Read through this file's own reader, which accepts np.float32 (a
        # `isinstance(raw, (int, float))` guard would not) and refuses NaN, inf,
        # bool and a numeric-looking string. A measured magnitude is ranked; an
        # unmeasurable one is named as a scope limitation in the block below,
        # never ranked at 0.0 and never dropped.
        _graded: List[Tuple[float, dict]] = []
        for v in per_variable:
            if not v.get("assessable"):
                continue
            _g = _as_measured_number(v.get("gap"))
            if _g is not None:
                _graded.append((_g, v))
        _graded.sort(key=lambda pair: pair[0], reverse=True)
        for i, (gap, v) in enumerate(_graded, 1):
            ff = v.get("fourFifthsRatio")
            sig = v.get("significant")
            if gap < 0.05 and not (us_emp and ff is not None and ff < 0.8):
                continue
            if us_emp and ff is not None and ff < 0.8:
                sev = "critical" if sig else "high"
                legal = (
                    f"Fails the US EEOC four-fifths rule (ratio {ff:.2f} < 0.80, US employment)."
                )
            else:
                sev = (
                    "critical" if gap >= 0.20 and sig else "high" if gap >= 0.10 and sig else "warn"
                )
                legal = (
                    "Assessed under EU/UK/CH proportionality + "
                    "objective-justification (no fixed numeric bar): "
                    "a disparity of this size requires documented "
                    "objective justification."
                )
            # CRITICAL, three states inside a regulatory sentence. `or 0` fires
            # on the ciLow/ciHigh the producers set to None ON PURPOSE when the
            # interval was not computable (orchestrator.py:1230,
            # ``"ciLow": None if lo != lo else float(lo)``), so an interval that
            # was never computed was printed as a measured one. Executed on this
            # repo before the fix, one row {gap 0.23, significant True, ciLow
            # None, ciHigh None} produced:
            #
            #   "Black" is selected 23 pts less than "White" on race
            #   (significant; 95% CI 0-0 pts). Assessed under EU/UK/CH ...
            #
            # "95% CI 0-0 pts" is both invented and self-contradictory: an
            # infinitely tight interval around zero, printed beside a 23-point
            # point estimate, in the one sentence a compliance reader relies on.
            # The NaN pair was worse again: ``round(nan * 100)`` raised
            # ValueError, the ``except Exception`` below swallowed it, and EVERY
            # disparate-impact finding in the run, the 23-point gap included,
            # left the verdict silently.
            #
            # Guarded like the two other consumers of these same two fields
            # (orchestrator.py:3288 and :5847-5852, the second carrying the
            # comment that says small groups yield None here), read through this
            # file's own reader so a bool and a NaN are refused as well. An
            # interval nobody computed is NAMED as not computed; it is not
            # rounded to zero and it is not quietly dropped either, because a
            # sentence that simply omits the clause reads as a gap reported
            # without any precision claim rather than as one whose precision was
            # never established.
            ci_lo = _as_measured_number(v.get("ciLow"))
            ci_hi = _as_measured_number(v.get("ciHigh"))
            ci_txt = (
                f"95% CI {round(ci_lo * 100)}-{round(ci_hi * 100)} pts"
                if ci_lo is not None and ci_hi is not None
                else "no 95% confidence interval was computed, so the precision of this "
                "estimate could not be checked"
            )
            add(
                "DSP",
                i,
                "disparate_impact",
                sev,
                f'"{v.get("worstGroup")}" is selected '
                f'{round(gap * 100)} pts less than "{v.get("referenceGroup")}" '
                f"on {v.get('attribute')} "
                f"({'significant' if sig else 'not yet significant'}; "
                f"{ci_txt}). {legal}",
                "disparityMatrix",
                attribute=v.get("attribute"),
                affected_groups=[v.get("worstGroup")],
                regulatory=_BIAS_TO_ART9_CATEGORY.get("demographic_parity"),
            )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    # PER-VARIABLE PARTIAL FAILURES: assessable, but the gap was never measured.
    #
    # BGL6 AUDIT, 2026-09-29. Its OWN try block, deliberately: the ranking above
    # and this disclosure must not be able to swallow each other, which is the
    # exact mechanism by which one None gap used to erase a 23-point finding.
    #
    # The refusal machinery below covers only ``assessable=False``, so a PARTIAL
    # failure (the attribute was in scope, the magnitude could not be read) was
    # invisible: measured before this block existed,
    # ``per_variable=[{"attribute": "age", "assessable": True, "gap": None}]``
    # returned overall "Unqualified", unassessed [], findings [], "no material
    # fairness defect found on the assessed attributes. Keep monitoring.", and
    # the string "age" appeared nowhere in the payload. ``assessed=False`` and
    # the "unassessed" severity route it into the scope limitation, so it is
    # neither a clean result nor a material finding.
    try:
        for i, v in enumerate(
            [
                v
                for v in per_variable
                if v.get("assessable") and _as_measured_number(v.get("gap")) is None
            ],
            1,
        ):
            attr = v.get("attribute")
            where = f" for '{attr}'" if attr else ""
            raw_gap = v.get("gap")
            add(
                "DSU",
                i,
                "disparate_impact_bias_unassessed",
                "unassessed",
                f"the selection-rate gap{where} was NOT measured "
                f"(recorded value {raw_gap!r}), so this attribute carries no "
                "disparate-impact verdict. It is outside the scope of this "
                "opinion: neither a clean result nor a measured disparity.",
                "disparityMatrix",
                attribute=attr,
                assessed=False,
            )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    # PER-VARIABLE REFUSALS: an attribute that could not be assessed at all.
    #
    # BGL5 AUDIT, 2026-09-27. The BGL-3 fix above covers only the ALL-or-nothing
    # case (`if not assessable`), so a PARTIAL scope limitation was invisible.
    # Measured with 5 of 6 protected variables refused ("groups too small") and
    # the sixth clean:
    #
    #   overall "Unqualified", blocksDeployment False, unassessed [], findings 0,
    #   "Unqualified opinion: no material fairness defect found on the assessed
    #    attributes. Keep monitoring."
    #
    # byte-identical to a run where 'gender' was the ONLY attribute requested, so
    # nothing anywhere in the payload named race, age, disability, religion or
    # nationality. This function's own scope-limitation machinery below ("an
    # opinion built on unassessed evidence must say so where the reader looks ...
    # It is never dropped") was fed only from the `bias` list, and a per_variable
    # entry with assessable=False was used NOWHERE except the two
    # `if v.get("assessable")` filters.
    #
    # After: overall "Qualified opinion (scope limitation)", blocksDeployment
    # False (a scope limitation is an "except for" matter, not a defect), and the
    # five attributes named in oneLineVerdict and in the `unassessed` list. The
    # severity word is "unassessed" so _sev returns None and these enter no max:
    # a refusal must not be promotable to a material finding either.
    try:
        refused = [v for v in per_variable if not v.get("assessable")]
        for i, v in enumerate(refused, 1):
            attr = v.get("attribute")
            reason = str(v.get("reason") or "").strip()
            where = f" for '{attr}'" if attr else ""
            add(
                "PVU",
                i,
                "group_fairness_bias_unassessed",
                "unassessed",
                f"group fairness could not be assessed{where}"
                + (f": {reason}" if reason else ": no reason was recorded")
                + ". This attribute is outside the scope of this opinion; it is "
                "neither a clean result nor a minor one.",
                "disparityMatrix",
                attribute=attr,
                assessed=False,
            )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    # PROXY / systemic leakage.
    try:
        for i, leak_rec in enumerate(proxies.get("leakage") or [], 1):
            if leak_rec.get("systemic_leakage"):
                add(
                    "PRX",
                    i,
                    "systemic_proxy_leakage",
                    "critical",
                    leak_rec.get("interpretation", ""),
                    "proxies",
                    attribute=leak_rec.get("protected_attribute"),
                )
        hi_px = [
            p
            for p in (proxies.get("proxies") or [])
            if str(p.get("severity", "")).lower() in ("high", "critical", "severe")
        ]
        if hi_px:
            add(
                "PRX",
                90,
                "proxy_features",
                "high",
                f"{len(hi_px)} high-risk proxy feature(s): "
                + ", ".join(sorted({str(p.get("feature")) for p in hi_px})[:6]),
                "proxies",
            )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    # INTERSECTIONAL.
    try:
        if not intersectional.get("skipped"):
            dg = intersectional.get("disadvantagedGroup") or {}
            md = intersectional.get("maxDisparity")
            # Significance gate (audit fix): the magnitude alone must NOT
            # escalate. A 20-pt gap between two 30-person cells is noise;
            # escalating it to "critical" tripped the whole verdict to
            # Adverse on sampling error. Only confirmed (BH-corrected)
            # intersectional disparity may reach critical/high; an
            # unconfirmed gap is capped at "warn" so it is surfaced but
            # cannot, by itself, block deployment.
            n_sig = int(
                ((intersectional.get("statisticalSummary") or {}).get("significantFindings"))
                or (intersectional.get("summary") or {}).get("significantFindings")
                or 0
            )
            dg_sig = dg.get("significant")
            confirmed = bool(n_sig > 0) and dg_sig is not False
            if dg and md and md >= 0.10:
                if confirmed:
                    sev = "critical" if md >= 0.20 else "high"
                    conf_txt = ""
                else:
                    sev = "warn"
                    conf_txt = (
                        " (not statistically confirmed after "
                        "multiple-testing correction, surfaced for "
                        "review, not used to block deployment)"
                    )
                add(
                    "INT",
                    1,
                    "intersectional_disparity",
                    sev,
                    f"The most disadvantaged combined subgroup "
                    f"({dg.get('group')}) has a {round(md * 100)}-point gap "
                    f"versus everyone else, a problem that stays hidden when "
                    f"each attribute is checked on its own{conf_txt}.",
                    "intersectional",
                    affected_groups=[dg.get("group")],
                )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    # BIAS taxonomy roll-up (representation / measurement / etc.).
    #
    # Significance gate (audit fix 2026-07-11): the SAME discipline the
    # intersectional roll-up already applies must guard every bias-taxonomy
    # finding. Several pulse stages (calibration ECE gap, cohort error
    # lift, label base-rate skew, memory-descriptor share) emit a
    # magnitude-only "critical" with statisticalTest=None; at the n>=20
    # group floor these are dominated by small-sample noise, and a single
    # unconfirmed "critical" flips the whole verdict to Adverse +
    # blocksDeployment. A magnitude-only finding is still SURFACED (the
    # panels render it at its computed severity), but for the OVERALL
    # opinion it is capped at "warn": deploy-with-monitoring, never
    # Adverse-blocking, unless it is statistically confirmed.
    try:
        worst_by_type: Dict[str, dict] = {}
        # R-10. An entry whose severity is could-not-check (insufficient_data
        # or any word the grader does not know) is not a candidate for "worst
        # measured finding of its type": it competes with nothing, it is
        # dropped by nothing, and it is reported below as its own state.
        unassessed_bias: List[dict] = []
        for b in bias:
            t = str(b.get("type", "bias"))
            b_rank = _sev(b.get("severity"))
            if b_rank is None:
                unassessed_bias.append(b)
                continue
            cur = worst_by_type.get(t)
            cur_rank = _sev(cur.get("severity")) if cur is not None else None
            if cur is None or cur_rank is None or b_rank > cur_rank:
                worst_by_type[t] = b
        for i, (t, b) in enumerate(worst_by_type.items(), 1):
            raw = _sev(b.get("severity"))
            if raw is not None and raw >= 2:
                sev = b.get("severity", "warn")
                note = ""
                if raw >= 3 and not _bias_finding_confirmed(b):
                    sev = "warn"
                    note = (
                        " (magnitude only, not statistically confirmed: "
                        "surfaced for review, not used to block "
                        "deployment)"
                    )
                add("BIA", i, f"{t}_bias", sev, str(b.get("plain", ""))[:240] + note, "bias")
        # R-10. Reported, never silently dropped, and never readable as a
        # clean or minor measurement: the severity word is kept verbatim
        # (it IS the state), the type says "unassessed", and ``assessed``
        # is False so no reader needs to know the vocabulary.
        for j, b in enumerate(unassessed_bias, len(worst_by_type) + 1):
            t = str(b.get("type", "bias"))
            attr = b.get("attribute")
            plain = str(b.get("plain", "")).strip()[:240]
            where = f" for '{attr}'" if attr else ""
            evidence = (plain + " " if plain else "") + (
                f"{t.replace('_', ' ')} bias could not be assessed{where}: no verdict "
                "was measured, so this is neither a clean result nor a minor one."
            )
            add(
                "BIA",
                j,
                f"{t}_bias_unassessed",
                str(b.get("severity") or "unassessed"),
                evidence,
                "bias",
                attribute=attr,
                assessed=False,
            )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    # Overall assurance opinion
    if schema.get("refuse"):
        overall, one_line, blocks = (
            "Disclaimer",
            "We cannot form a fairness opinion on this artifact as "
            f"provided: {schema.get('refuse_reason', '')}",
            True,
        )
    else:
        # R-10. An unassessed finding has no rank and enters no max.
        max_sev = max(
            [s for s in (_sev(f["severity"]) for f in findings) if s is not None],
            default=0,
        )
        assessable = [v for v in per_variable if v.get("assessable")]
        if not assessable and max_sev < 2:
            # Disclaimer only when there is ALSO no material finding.
            # Material findings (e.g. generative-output disparities, PII
            # leakage) must still drive Qualified/Adverse even without a
            # tabular per-variable analysis.
            #
            # BGL-3, measured 2026-09-27. This condition carried `and not
            # has_truth`, so the presence of an OUTCOME COLUMN was enough to
            # turn "nothing could be assessed" into an all-clear. Measured:
            #
            #   build_assurance_verdict(per_variable=[], has_truth=True)
            #   build_assurance_verdict(per_variable=[{"attribute": "race",
            #       "assessable": False, "reason": "groups too small"}],
            #       has_truth=True)
            #     -> both: overall "Unqualified", blocksDeployment False,
            #        "Unqualified opinion: no material fairness defect found on
            #        the assessed attributes. Keep monitoring.", findings [],
            #        unassessed [].
            #
            # The set of assessed attributes is EMPTY there, so the sentence is
            # vacuously true and reads as a clean audit opinion. Ground truth
            # says nothing about whether any protected attribute was
            # assessable, and nothing in this function grades the label-
            # dependent metrics into an opinion either. The orchestrator's own
            # headline for the identical state already says "Could not compute
            # a reliable fairness verdict on this data. No protected variable
            # had enough grouped data to assess." (orchestrator.py:6516), so
            # the two halves of one report disagreed, and the audit-grade half
            # was the one fabricating.
            overall, blocks = "Disclaimer", False
            one_line = "Insufficient assessable data to issue a fairness opinion. " + (
                "No protected variable had enough grouped data to assess, "
                "so this opinion covers nothing; outcome labels alone are "
                "not enough. Provide grouped protected attributes with "
                "enough rows per group and re-run."
                if has_truth
                else "Provide grouped protected attributes and, ideally, outcome labels."
            )
        elif max_sev >= 4:
            overall, blocks = "Adverse", True
            tops = [f for f in findings if (_sev(f["severity"]) or 0) >= 4][:3]
            # First-sentence extraction must NOT split on periods inside
            # decimals (e.g. "AUC 0.95"). Splitting on "." alone produced
            # "AUC 0" in the one-line verdict, which looked like a formatting
            # bug downstream. Use a sentence-boundary regex: a period
            # followed by whitespace or end-of-string, never one wedged
            # between two digits.
            import re as _re_local

            def _first_sentence(s: str) -> str:
                parts = _re_local.split(r"(?<!\d)\.(?:\s|$)", s, maxsplit=1)
                return parts[0].rstrip(" .") if parts else s

            one_line = (
                "Adverse opinion: material fairness defects make this "
                "system unfit to deploy as-is. "
                + "; ".join(_first_sentence(t["evidence"]) for t in tops)
                + "."
            )
        elif max_sev >= 2:
            overall, blocks = "Qualified", False
            one_line = (
                "Qualified opinion: deployable only with the documented "
                "remediations and active monitoring below; material but "
                "bounded fairness issues were found."
            )
        else:
            overall, blocks = "Unqualified", False
            one_line = (
                "Unqualified opinion: no material fairness defect "
                "found on the assessed attributes. Keep monitoring."
            )

    # R-10. An opinion built on unassessed evidence must say so where the
    # reader looks: in the one-line verdict, and in a structured list. In the
    # assurance vocabulary this module borrows (Lam et al. FAccT 2024, after
    # financial audit) a scope limitation is an "except for" matter, so an
    # opinion that would otherwise be Unqualified becomes Qualified without
    # blocking deployment; a Qualified, Adverse or Disclaimer opinion keeps
    # its category and gains the limitation. It is never dropped.
    unassessed_findings = [f for f in findings if f.get("assessed") is False]
    unassessed_summary = [
        {
            "id": f["id"],
            "type": f["type"],
            "attribute": f.get("attribute"),
            "severity": f["severity"],
            "evidence": f["evidence"],
        }
        for f in unassessed_findings
    ]
    if unassessed_findings and not schema.get("refuse"):
        limitation = "; ".join(
            f"{f['type'].replace('_bias_unassessed', '').replace('_', ' ')} bias could not "
            f"be assessed for '{f.get('attribute')}' ({f['severity']})"
            if f.get("attribute")
            else f"{f['type'].replace('_bias_unassessed', '').replace('_', ' ')} bias could "
            f"not be assessed ({f['severity']})"
            for f in unassessed_findings
        )
        if overall == "Unqualified":
            overall = "Qualified"
            one_line = (
                "Qualified opinion (scope limitation): no material fairness "
                "defect found on the assessed attributes, but "
                f"{limitation}. This opinion does not cover what was not "
                "assessed; collect the missing data and re-run."
            )
        else:
            one_line = f"{one_line} Scope limitation: {limitation}."

    # Routed recommendations
    recs: List[Dict[str, Any]] = []
    rid = 1

    def rec(action, addresses, route, fn, priority):
        nonlocal rid
        recs.append(
            {
                "id": f"REC-{rid:03d}",
                "priority": priority,
                "action": action,
                "addressesFindings": addresses,
                "routeTo": route,
                "vfairnessFunction": fn,
            }
        )
        rid += 1

    pii_f = [f["id"] for f in findings if f["type"] == "identity_pii_leakage"]
    if pii_f:
        rec(
            "Remove or pseudonymise the identity/PII columns before any "
            "modelling; never feed them as features.",
            pii_f,
            "navigator_setup_step",
            "classify_column_roles",
            1,
        )
    prx_f = [f["id"] for f in findings if "proxy" in f["type"]]
    if prx_f:
        rec(
            "Transform or drop the proxy features (single-column removal is "
            "insufficient under systemic leakage); re-run to confirm.",
            prx_f,
            "navigator_preprocessing_wizard",
            "multivariate_proxy_leakage",
            1 if any(f["severity"] == "critical" for f in findings if f["id"] in prx_f) else 2,
        )
    dsp_f = [f["id"] for f in findings if f["type"] == "disparate_impact"]
    if dsp_f:
        rec(
            (
                f"Apply the recommended fairness definition "
                f"({primary_definition}) and a mitigation "
                f"(reweighing / threshold optimisation), then re-measure."
                if primary_definition
                else "Apply the recommended fairness definition and a "
                "mitigation (reweighing / threshold optimisation), then "
                "re-measure."
            ),
            dsp_f,
            "navigator_full_assessment",
            "recommend_interventions",
            1 if overall == "Adverse" else 2,
        )
    int_f = [f["id"] for f in findings if f["type"] == "intersectional_disparity"]
    if int_f:
        rec(
            "Investigate the worst intersection explicitly; single-axis "
            "fixes will not close an intersectional gap.",
            int_f,
            "navigator_full_assessment",
            "intersectional_disparity_analysis",
            2,
        )
    una_f = [f["id"] for f in unassessed_findings]
    if una_f:
        # R-10. The route out of a scope limitation is more data, not a fix.
        una_attrs = sorted(
            {str(f.get("attribute")) for f in unassessed_findings if f.get("attribute")}
        )
        rec(
            "Bias could not be assessed"
            + (f" for {', '.join(una_attrs)}" if una_attrs else "")
            + ": collect more rows for the named attribute(s) (or supply a "
            "population benchmark) and re-run. This opinion does not cover them.",
            una_f,
            "navigator_full_assessment",
            "detect_representation_bias"
            if all("representation" in f["type"] for f in unassessed_findings)
            else "run_pulse",
            2,
        )
    if not recs:
        rec(
            "No blocking fairness issue. Convert to a tracked assessment and keep monitoring.",
            [],
            "navigator_full_assessment",
            "run_pulse",
            3,
        )

    # Deferred metrics
    deferred = sorted(
        {
            name
            for m in metrics
            if m.get("labelDependent") and not m.get("computable")
            for name in (m.get("metric") or m.get("label"),)
            if name is not None
        }
    )
    metrics_deferred = (
        {
            "metrics": deferred,
            "rationale": (
                "These compare decisions to the true outcome and need "
                "ground-truth labels, which were not present. Re-run "
                "with an outcome column to compute them."
            )
            if deferred
            else "",
        }
        if deferred
        else {"metrics": [], "rationale": ""}
    )

    # Dual-label every finding against BOTH regulator-facing frameworks:
    # Suresh & Guttag (2021) ML lifecycle stage AND NIST SP 1270 (systemic /
    # statistical-computational / human-cognitive). Pure mapping.
    def _dual_label(ftype: str):
        t = (ftype or "").lower()
        if "pii" in t or "schema" in t or "unassessable" in t:
            return ("Data governance / pre-processing", "systemic")
        if "proxy" in t:
            return ("Feature / measurement (proxy)", "statistical-computational")
        if "temporal" in t:
            return ("Deployment / feedback loop", "systemic")
        if "historical" in t:
            return ("Historical (pre-existing inequity)", "systemic")
        if "representation" in t or "sampling" in t:
            return ("Representation / sampling", "statistical-computational")
        if "measurement" in t or "aggregation" in t or "label" in t:
            return ("Measurement / aggregation", "statistical-computational")
        if "intersectional" in t:
            return ("Evaluation (subgroup)", "statistical-computational")
        if "disparate" in t:
            return ("Evaluation / outcome", "statistical-computational")
        return ("Evaluation", "statistical-computational")

    for _f in findings:
        lc, nist = _dual_label(_f.get("type", ""))
        _f["lifecycle"] = lc  # Suresh & Guttag 2021
        _f["nistCategory"] = nist  # NIST SP 1270

    # Attach a structured historical-discrimination envelope to every
    # assurance finding whose (attribute, domain) matches a documented
    # precedent (race+lending => redlining, age+hiring => ADEA, gender+
    # hiring => pay gap, ...). The Pulse UI keys its "Historical pattern"
    # bias-spectrum channel off this flag instead of regex-matching the
    # finding's free-text evidence, so race/age/gender/national-origin
    # findings in regulated domains land in the channel even when their
    # evidence reads "selection-rate gap" rather than the word "historical".
    # Native B-prefix bias findings carrying their own pattern_type
    # (Geographic Redlining etc.) keep their citations. Never raises.
    try:
        from vfairness.preprocessing.bias_detection.historical import attribute_historical_pattern

        for _f in findings:
            if _f.get("historicalPattern"):
                continue
            # Findings expose `attribute` directly; fall back to the first
            # affected feature so e.g. Geographic Redlining (which keys on
            # `affected_features=[zip_code]`) still resolves.
            attr = _f.get("attribute")
            if not attr:
                affected = _f.get("affected_features") or []
                if affected:
                    attr = affected[0]
            pat = attribute_historical_pattern(attr, domain, jurisdiction)
            if pat:
                _f["historicalPattern"] = pat
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)

    # Domain-level documented precedent + regulatory pointers (real,
    # canonical sources only; None when the domain is unknown).
    hist = None
    try:
        from vfairness.preprocessing.bias_detection import domain_historical_context

        hist = domain_historical_context(domain, jurisdiction)
    except Exception:  # noqa: BLE001
        hist = None

    return {
        "overall": overall,  # Unqualified|Qualified|Adverse|Disclaimer
        "blocksDeployment": blocks,
        "oneLineVerdict": one_line,
        # R-10. Findings with no measured severity, listed where a reader
        # looks for them, so an opinion never presents unassessed evidence
        # as assessed. Empty when everything was graded.
        "unassessed": unassessed_summary,
        "historicalContext": hist,
        "regulatory": (hist.get("regulatory") if hist else []),
        "findings": findings,
        "recommendations": sorted(recs, key=lambda r: r["priority"]),
        "metricsDeferred": metrics_deferred,
        "jurisdictionBasis": (
            "US EEOC four-fifths (employment)"
            if us_emp
            else "EU/UK/CH proportionality + objective justification (no fixed numeric threshold)"
        ),
        "auditTrail": {
            "pulseVersion": pulse_version,
            "generatedAt": _now_iso(),
            "domain": domain,
            "jurisdiction": jurisdiction,
            "hasGroundTruth": bool(has_truth),
            "artifactHash": artifact_hash,
            "engine": "vfairness",
        },
    }
