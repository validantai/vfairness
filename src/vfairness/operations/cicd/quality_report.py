"""Canonical plain-language data-quality report.

Single source of truth for translating a ``DataBiasValidator`` result into
the non-expert, audit-grade quality summary that the product surfaces
(Pulse's "Pre-flight · Data check" and the Navigator's "Run Quality Check").

Both consumers MUST call :func:`build_quality_report` rather than
re-implementing the translation, so the breadth and wording of the checks
stay identical everywhere. The validator itself (representation, outcome
disparity, missing patterns, label quality, and the Turing-Module-4 data
hygiene dimension) remains the only place that makes the judgments; this
module only renders them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Binning / identifier handling is the single canonical module -- never
# re-implement it here (divergent local binning caused incorrect output).
from vfairness.preprocessing.protected_binning import prepare_protected_attributes


def _affected_column(issue: Any) -> Optional[str]:
    """Pull the column an issue refers to (validator messages quote it)."""
    msg = getattr(issue, "message", "") or ""
    m = re.search(r"'([^']+)'", msg)
    if m:
        return m.group(1)
    det = getattr(issue, "details", None)
    if isinstance(det, dict):
        for k in ("column", "attribute", "feature"):
            if det.get(k):
                return str(det[k])
    return None


def _affected_group_rows(issue: Any) -> List[Dict[str, Any]]:
    """Structured affected groups so the UI charts them instead of dumping
    a long "Affected: a (1.5%); b (2.0%); ..." sentence the user can't read.

    Returns rows of {group, fraction?, count?}; empty when nothing structured
    is available (the text fallback in _affected_groups still applies).
    """
    det = getattr(issue, "details", None) or {}
    rows: List[Dict[str, Any]] = []
    if isinstance(det, dict) and isinstance(det.get("group_fractions"), dict):
        for g, fr in det["group_fractions"].items():
            try:
                rows.append({"group": str(g), "fraction": float(fr)})
            except (TypeError, ValueError):
                continue
    elif isinstance(det, dict) and isinstance(det.get("group_counts"), dict):
        for g, n in det["group_counts"].items():
            try:
                rows.append({"group": str(g), "count": int(n)})
            except (TypeError, ValueError):
                continue
    # Smallest share / count first -- that is the worst-served group.
    rows.sort(key=lambda r: r.get("fraction", r.get("count", 0)))
    return rows


def _affected_groups(issue: Any, max_named: int = 5) -> str:
    """Readable list of the SPECIFIC groups that triggered the issue.

    The user must see *which* groups, not just "some groups". Uses the
    validator's structured details (group_counts -> "g (n=4)";
    group_fractions -> "g (1.2%)") and falls back to affected_groups labels.
    """
    det = getattr(issue, "details", None) or {}
    pairs: List[str] = []
    if isinstance(det, dict) and isinstance(det.get("group_counts"), dict):
        for g, n in list(det["group_counts"].items()):
            pairs.append(f"{g} (n={int(n)})")
    elif isinstance(det, dict) and isinstance(det.get("group_fractions"), dict):
        for g, fr in list(det["group_fractions"].items()):
            pairs.append(f"{g} ({float(fr) * 100:.1f}%)")
    elif isinstance(det, dict) and isinstance(det.get("zero_rate_groups"), list):
        pairs = [str(g) for g in det["zero_rate_groups"]]
    if not pairs:
        ag = getattr(issue, "affected_groups", None) or []
        pairs = [str(g) for g in ag]
    if not pairs:
        return ""
    shown = pairs[:max_named]
    extra = len(pairs) - len(shown)
    text = ", ".join(shown)
    if extra > 0:
        text += f" and {extra} more"
    return text


_SEV_TO_STATUS = {"critical": "critical", "error": "critical", "warning": "warn"}

_LABELS = {
    "insufficient_group_samples": "Group sizes",
    "underrepresented_groups": "Group representation",
    "biased_missing_pattern": "Missing-data fairness",
    "high_missing_values": "Missing values",
    "missing_labels": "Outcome completeness",
    "extreme_class_imbalance": "Outcome balance",
    "outcome_disparity": "Raw outcome gap",
    "insufficient_total_samples": "Sample size",
    "duplicate_rows": "Duplicate rows",
    "constant_columns": "Informative columns",
    "missing_protected_attributes": "Attribute present",
    # Could-not-check rows. They carry the SAME label as the check they replace,
    # so a reader sees one row per check and never a pass tile beside a refusal.
    "representation_not_measurable": "Group representation",
    "missing_outcome_column": "Outcome column",
}

_PLAIN = {
    "missing_protected_attributes": "The attribute being assessed is not in the data.",
    "insufficient_group_samples": "Some groups are too small for a reliable read (under 30 people).",
    "underrepresented_groups": "Some groups are a very small share of the data; their numbers are noisy.",
    "high_missing_values": "A column has too many blank values to rely on.",
    "biased_missing_pattern": "Values are missing more often for some groups than others, which itself biases the read.",
    "missing_labels": "The true-outcome column has too many blanks.",
    "extreme_class_imbalance": "The outcome is almost always the same value, so fairness is hard to read.",
    "outcome_disparity": "Outcome rates differ sharply between groups in the raw data.",
    "insufficient_total_samples": "Not many rows overall, so the confidence ranges will be wide.",
    "duplicate_rows": "Repeated rows quietly reweight groups and bias every number.",
    "constant_columns": "Some columns never change, so they add no information.",
    "representation_not_measurable": (
        "No row carries a value for this attribute, so there is no group to size and "
        "group representation could NOT be checked. This is not a finding that the "
        "groups are large enough."
    ),
    "missing_outcome_column": (
        "The outcome column named for this check is not in the data, so the raw "
        "outcome-gap and outcome-completeness checks could NOT run. Nothing here says "
        "the outcomes are even."
    ),
}


def build_quality_report(
    df: pd.DataFrame,
    protected_attributes: List[str],
    outcome_column: Optional[str] = None,
    info_notes: Optional[List[Tuple[str, str]]] = None,
    unavailable_attributes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the canonical validator and render a plain-language report.

    Args:
        df: the dataset to validate.
        protected_attributes: columns to assess (the validator checks
            representation / group size against these).
        outcome_column: the true-outcome column when one is present
            (enables label-quality + outcome-disparity checks). Pass None
            when only model predictions are available (Pulse). Either way the
            report carries an "Outcome checks" row saying which of the three
            states applies: the checks ran and found nothing (``pass``), they
            were never requested (``info``, tone-neutral), or the named column
            is absent so they could not run (``critical``, from the validator's
            own ``missing_outcome_column`` issue).
        info_notes: optional ``(label, detail)`` pairs appended as
            ``status="info"`` rows (e.g. an auto-binning disclosure). These
            never affect the overall tone.
        unavailable_attributes: attributes the CALLER asked to assess that are
            not columns of ``df`` at all, so they were never passed to the
            validator. Each becomes a ``status="warn"`` row naming it. A
            caller that drops such a name silently gets a report whose green
            tone claims a breadth it never had; see
            ``operations.cicd.task_handlers.handle_data_validation``, where
            that was measured.

    Returns:
        ``{tone, headline, rows, columns, checks}`` -- the exact shape the
        front-end DataQuality block renders.
    """
    checks: List[Dict[str, Any]] = []

    def add(
        key: str,
        label: str,
        status: str,
        detail: str,
        groups: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        check: Dict[str, Any] = {
            "key": key,
            "label": label,
            "status": status,
            "detail": detail,
        }
        if groups:
            # Structured affected groups -> the UI renders a bar chart and
            # does NOT show the long "Affected:" text.
            check["groups"] = groups
        checks.append(check)

    n_rows = int(len(df))
    n_cols = int(df.shape[1])

    # Canonical, single-source preparation: derive age from DOB, bin
    # continuous attrs, prefer a binned sibling, and EXCLUDE identifier-grade
    # columns (never group by a date / name / id). Everything is disclosed.
    prep = prepare_protected_attributes(df, protected_attributes)
    prepared_df, used_attrs = prep.frame, prep.usable
    bin_notes = prep.notes
    primary = used_attrs[0] if used_attrs else "the attribute"

    # Excluded attributes are surfaced as warnings so the user sees WHY a
    # variable they picked is not in the fairness read (e.g. raw DOB).
    for j, (xcol, reason) in enumerate(prep.excluded):
        add(f"excluded_{j}", "Attribute excluded", "warn", reason)

    # Same reason, one step earlier: an attribute that is not in the data at all
    # never reaches `prepare_protected_attributes`, so nothing above can
    # disclose it. "warn" rather than "info" is the point: it lifts the tone off
    # "pass", so the headline can no longer read "fit for a reliable fairness
    # read" for a request that was only half carried out.
    for j, name in enumerate(unavailable_attributes or []):
        add(
            f"unavailable_{j}",
            "Attribute not in the data",
            "warn",
            f'"{name}" was requested for this check but is not a column of the data, '
            f"so it was never assessed. Nothing in this report covers it.",
        )

    if not used_attrs:
        add(
            "no_usable_attr",
            "No assessable attribute",
            "critical",
            "None of the chosen attributes can be assessed for fairness "
            "(all were identifiers or could not be grouped). Pick a "
            "categorical or age/income-style attribute.",
        )

    try:
        from vfairness.operations.cicd.validator import DataBiasValidator

        # CRITICAL (fail closed): when no attribute survives preparation, the
        # validator cannot run, and this used to substitute an empty stub
        # object -- `type("E", (), {"issues": []})()` -- whose zero issues then
        # flowed into the three "nothing was wrong" branches below and
        # MANUFACTURED three green PASS tiles. Measured 2026-09-10 on a frame
        # with 6 rows, 3 duplicate rows (50 percent) and a constant column,
        # whose only chosen attribute was an identifier: the report rendered
        #   [pass] Group representation: Every group of "the attribute" is
        #          large enough for a reliable read.
        #   [pass] Missing values: No column has a problematic level of blank
        #          values.
        #   [pass] Data hygiene: Enough rows, no duplicate rows, no dead
        #          columns.
        # while the real validator on the same frame reported
        # insufficient_total_samples, duplicate_rows (50.0%) and
        # constant_columns. An absence of findings from a validator that never
        # ran is not a finding of no problems.
        #
        # `validated` is the flag the PASS branches below are gated on; when it
        # is False they emit could-not-check ("warn") tiles instead, saying
        # which check did not run and why.
        validated = bool(used_attrs)
        result = (
            DataBiasValidator(protected_attributes=used_attrs).validate(
                prepared_df, outcome_column=outcome_column
            )
            if validated
            else None
        )
        issues = result.issues if result is not None else []

        # Aggregate by issue_type: ONE row per kind of problem, listing every
        # affected column -- instead of N identical repeated tiles.
        by_type: Dict[str, Dict[str, Any]] = {}
        for issue in issues:
            sev = getattr(issue.severity, "value", str(issue.severity)).lower()
            status = _SEV_TO_STATUS.get(sev, "warn")
            col = _affected_column(issue)
            slot = by_type.setdefault(
                issue.issue_type,
                {
                    "status": status,
                    # per-column -> the specific groups that triggered it
                    "by_col": {},
                    # per-column -> structured [{group, fraction|count}, ...]
                    "by_col_rows": {},
                    "plain": _PLAIN.get(issue.issue_type) or issue.message,
                    "rec": getattr(issue, "recommendation", None),
                },
            )
            # escalate to the worst severity seen for this type
            if status == "critical":
                slot["status"] = "critical"
            key = col or ""  # "" => dataset-wide (no per-column prefix)
            groups = _affected_groups(issue)
            # keep the most specific group list if the same column repeats
            if key not in slot["by_col"] or (groups and not slot["by_col"][key]):
                slot["by_col"][key] = groups
            rows = _affected_group_rows(issue)
            if rows and (key not in slot["by_col_rows"] or not slot["by_col_rows"][key]):
                slot["by_col_rows"][key] = rows

        for itype, slot in by_type.items():
            label = _LABELS.get(itype, itype.replace("_", " ").capitalize())
            # Flatten structured per-column rows: [{column, group, fraction|count}].
            group_rows: List[Dict[str, Any]] = []
            for c, rows in slot["by_col_rows"].items():
                for r in rows:
                    group_rows.append({**r, "column": c or None})
            if group_rows:
                # Chart the groups; keep only the plain explanation + the
                # recommendation as text (no unreadable "Affected:" dump).
                detail = slot["plain"]
                if slot["rec"]:
                    detail = f"{detail} {slot['rec']}"
                add(f"vf_{itype}", label, slot["status"], detail, groups=group_rows)
            else:
                frags: List[str] = []
                for c, groups in slot["by_col"].items():
                    if not c:
                        continue  # dataset-wide issue: no per-column prefix
                    frags.append(f"{c}: {groups}" if groups else c)
                where = f"Affected: {'; '.join(frags)}. " if frags else ""
                detail = f"{where}{slot['plain']}"
                if slot["rec"]:
                    detail = f"{detail} {slot['rec']}"
                add(f"vf_{itype}", label, slot["status"], detail)

        # A PASS tile is a positive claim: "this check ran and found nothing".
        # It is only ever emitted when the validator actually ran. When it did
        # not, the same slot says so instead, at "warn": could-not-check is not
        # a pass, and it is not a measured problem either.
        _NOT_RUN = (
            "The vfairness validator could not run on this data, because none of the "
            "chosen attributes can be grouped for a fairness read, so this check "
            "was never performed. Pick a categorical or age/income-style attribute "
            "to have it checked."
        )
        types = set(by_type)
        # `representation_not_measurable` belongs in this set for the same
        # reason the other two do: it OCCUPIES the representation slot. Without
        # it the validator's could-not-check row and a green "Every group is
        # large enough" tile were rendered SIDE BY SIDE off the same result.
        # Measured 2026-09-27 on 300 rows whose only attribute was blank in
        # every row (the validator fix landed first): the pass tile was still
        # emitted. BGL3 operations-4.
        if not types & {
            "insufficient_group_samples",
            "underrepresented_groups",
            "representation_not_measurable",
        }:
            if validated:
                add(
                    "vf_representation",
                    "Group representation",
                    "pass",
                    f'Every group of "{primary}" is large enough for a reliable read.',
                )
            else:
                add("vf_representation", "Group representation", "warn", _NOT_RUN)
        # This slot is about BLANK VALUES, and the test for it is a substring
        # match. ``missing_outcome_column`` says a named column is absent, which
        # is a different fact, and letting it match suppressed this row
        # altogether: the missing-values check had run and found nothing, and
        # the report then said neither "pass" nor "could not check" about it.
        # Measured 2026-09-27, right after the outcome column stopped being
        # dropped silently. BGL3 operations-4.
        if not any("missing" in t and t != "missing_outcome_column" for t in types):
            if validated:
                add(
                    "vf_missing",
                    "Missing values",
                    "pass",
                    "No column has a problematic level of blank values.",
                )
            else:
                add("vf_missing", "Missing values", "warn", _NOT_RUN)
        # THE OUTCOME SLOT. BGL5 AUDIT, 2026-09-27: it did not exist, in either
        # direction. Measured on the same 400 clean rows:
        #   outcome_column="approved" (checks RAN, found nothing)
        #     -> tone pass, keys [vf_representation, vf_missing, vf_hygiene]
        #   outcome_column=None       (checks NEVER RAN)
        #     -> BYTE-IDENTICAL
        # So a reader could not tell a clean outcome check from one that never
        # happened, while the representation slot and a typo'd outcome name had
        # each been given a third state in this campaign.
        #
        # Three states, distinct keys so the two documents are distinguishable at
        # a glance, and the same label on all of them (the convention the
        # could-not-check rows above already follow). The not-requested row is
        # "info", NOT "warn", deliberately: Pulse passes outcome_column=None
        # because only predictions exist, it asked nothing about outcomes and
        # nothing is owed about them, so the tone must stay "pass" (pinned by
        # tests/test_bgl3_operations_4.py::test_control_no_outcome_requested_is_
        # still_a_clean_pass). What was missing was the SENTENCE, not a warning.
        _OUTCOME_TYPES = {
            "outcome_disparity",
            "missing_labels",
            "extreme_class_imbalance",
            "missing_outcome_column",
        }
        if not types & _OUTCOME_TYPES:
            if not outcome_column:
                add(
                    "vf_outcome_not_requested",
                    "Outcome checks",
                    "info",
                    "No outcome column was supplied, so the raw outcome-gap and "
                    "outcome-completeness checks were NOT performed. Nothing in this "
                    "report says the outcomes are even; it covers the data, not the "
                    "decisions.",
                )
            elif validated:
                add(
                    "vf_outcome",
                    "Outcome checks",
                    "pass",
                    f"The raw outcome-gap and outcome-completeness checks ran on "
                    f'"{outcome_column}" and found no problem.',
                )
            else:
                add("vf_outcome", "Outcome checks", "warn", _NOT_RUN)
        if not types & {"insufficient_total_samples", "duplicate_rows", "constant_columns"}:
            if validated:
                add(
                    "vf_hygiene",
                    "Data hygiene",
                    "pass",
                    "Enough rows, no duplicate rows, no dead columns.",
                )
            else:
                add("vf_hygiene", "Data hygiene", "warn", _NOT_RUN)
    except Exception:  # noqa: BLE001 -- never let validation crash a consumer
        add(
            "vf_unavailable",
            "Validation engine",
            "critical",
            "The canonical vfairness validator could not run, so data quality "
            "could not be confirmed. Treat the result as indicative only.",
        )

    # Binning disclosures first (so the user understands the group columns),
    # then any caller-supplied notes.
    for i, (label, detail) in enumerate(bin_notes):
        add(f"bin_{i}", label, "info", detail)
    for i, (label, detail) in enumerate(info_notes or []):
        add(f"info_{i}", label, "info", detail)

    rank = {"pass": 0, "info": 0, "warn": 1, "critical": 2}
    tone = "pass"
    for c in checks:
        if rank[c["status"]] > rank[tone]:
            tone = c["status"]
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    n_crit = sum(1 for c in checks if c["status"] == "critical")
    if tone == "pass":
        headline = "The data is fit for a reliable fairness read."
    elif tone == "warn":
        headline = (
            f"The data is usable, but {n_warn} thing(s) widen the uncertainty. "
            "Read the result with that in mind."
        )
    else:
        headline = (
            f"{n_crit} data problem(s) make the result indicative only. "
            "Fix these for an audit-grade result."
        )

    return {
        "tone": tone,
        "headline": headline,
        "rows": n_rows,
        "columns": n_cols,
        "checks": checks,
    }
