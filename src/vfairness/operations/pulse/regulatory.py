"""Regulatory packaging for Pulse (close-plan Phase 4).

One module owns everything that turns a finished Pulse run into
regulator-facing artefacts, so the orchestrator only does thin wiring:

* G-13  legal admissibility: column-level (use_case x jurisdiction)
        classification, ported from the quarantined legacy pipeline
        (vfairness.legal rule packs; honest ``uncovered`` fallback).
* G-18  regulatory exports: NYC Local Law 144 impact-ratio table and an
        EU AI Act Art. 10 section draft, both assembled ONLY from values
        the run actually computed (never fabricated numbers).
* G-24  ISO/IEC TR 24027:2021 metric crosswalk for the metrics Pulse
        actually computes on this pathway, with honest clause labels
        (no invented subclause numbers).
* G-31  historical-context catalog: documented discrimination patterns
        per domain with REAL canonical citations only; an unknown domain
        reports ``available: False`` instead of inventing history.
* G-32  reference-group override disclosure (the per-variable override
        itself is threaded through the orchestrator) and the
        four-fifths target block.
* G-34  recheck window: assessedAt / validUntil (+12 months when LL144
        applies, else +6) plus a re-runnable suite echo with every
        secret stripped (auth tokens, API keys, model blobs, org keys).

Every builder here is total-by-design where practical and is ALSO run
inside the orchestrator's ``_safe`` stages so a failure degrades and is
recorded, never crashes a Pulse.
"""

from __future__ import annotations

import calendar
import math
import numbers
import re
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────────────────────────
# Input parsing (camelCase twins accepted; total, never raises)
# ───────────────────────────────────────────────────────────────────────────


def parse_pulse_options(inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise the Phase-4 optional inputs.

    Returns {harm_direction, scores_exposed, reference_group,
    target_ratio}. ``harm_direction`` is 'assistive' | 'punitive' | None;
    ``reference_group`` is a str->str dict (attribute -> requested
    reference group), {} when absent or malformed; ``target_ratio`` is
    passed through RAW (clamping happens in build_targets so the clamp
    can be disclosed).
    """
    try:
        src = inputs or {}

        def pick(*names):
            for n in names:
                v = src.get(n)
                if v is not None:
                    return v
            return None

        hd: Optional[str] = str(pick("harm_direction", "harmDirection") or "").strip().lower()
        if hd not in ("assistive", "punitive"):
            hd = None
        se = bool(pick("scores_exposed", "scoresExposed") or False)
        rg = pick("reference_group", "referenceGroup")
        ref: Dict[str, str] = {}
        if isinstance(rg, dict):
            for k, v in rg.items():
                if v is None:
                    continue
                ref[str(k)] = str(v)
        return {
            "harm_direction": hd,
            "scores_exposed": se,
            "reference_group": ref,
            "target_ratio": pick("target_ratio", "targetRatio"),
        }
    except Exception:  # option parsing is never fatal
        return {
            "harm_direction": None,
            "scores_exposed": False,
            "reference_group": {},
            "target_ratio": None,
        }


# ───────────────────────────────────────────────────────────────────────────
# Jurisdiction / domain heuristics (substring, mirroring the orchestrator's
# framework resolver; deliberately conservative)
# ───────────────────────────────────────────────────────────────────────────

_EMPLOYMENT_KEYS = (
    "employ",
    "hir",
    "recruit",
    "promot",
    "talent",
    "workforce",
    "staffing",
    "applicant",
)
_US_PATTERN = re.compile(
    r"(?:^|[^a-z])(us|usa|u\.s\.?|united states|america|nyc|new york)"
    r"(?:[^a-z]|$)"
)
_EU_PATTERN = re.compile(r"(?:^|[^a-z])(eu|eea)(?:[^a-z]|$)")

# EU AI Act obligations bind in every member state, so the EU screen must
# fire for a member-state jurisdiction ("Germany", "de", "France"), not
# only for the literal strings "eu"/"eea". Names match as substrings
# (all are long enough to be unambiguous); ISO 3166-1 alpha-2 codes match
# only as standalone tokens, same boundary convention as _US_PATTERN
# (jurisdiction is a short declared field, not free prose). Greece is
# listed under both its ISO code "gr" and its EU code "el".
_EU_MEMBER_NAMES = (
    "austria",
    "belgium",
    "bulgaria",
    "croatia",
    "cyprus",
    "czech",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "deutschland",
    "greece",
    "hungary",
    "ireland",
    "italy",
    "latvia",
    "lithuania",
    "luxembourg",
    "malta",
    "netherlands",
    "poland",
    "portugal",
    "romania",
    "slovakia",
    "slovenia",
    "spain",
    "sweden",
)
_EU_MEMBER_CODE_PATTERN = re.compile(
    r"(?:^|[^a-z])(at|be|bg|hr|cy|cz|dk|ee|fi|fr|de|el|gr|hu|ie|it|lv|lt|"
    r"lu|mt|nl|pl|pt|ro|sk|si|es|se)(?:[^a-z]|$)"
)


def _looks_employment(domain: str) -> bool:
    d = (domain or "").strip().lower()
    return any(k in d for k in _EMPLOYMENT_KEYS)


def _looks_us(jurisdiction: str) -> bool:
    j = (jurisdiction or "").strip().lower()
    return bool(_US_PATTERN.search(j))


def _looks_eu(jurisdiction: str) -> bool:
    j = (jurisdiction or "").strip().lower()
    if bool(_EU_PATTERN.search(j)) or "europ" in j:
        return True
    if any(name in j for name in _EU_MEMBER_NAMES):
        return True
    return bool(_EU_MEMBER_CODE_PATTERN.search(j))


def legal_context_as_of() -> str:
    """Server-side 'as of' date for the legal context (UTC, YYYY-MM-DD)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ───────────────────────────────────────────────────────────────────────────
# G-13: legal admissibility (port of the quarantined pipeline.py stage)
# ───────────────────────────────────────────────────────────────────────────


def uncovered_admissibility(jurisdiction: str, error: Optional[str] = None) -> Dict[str, Any]:
    """The honest fallback block when classification cannot run.

    Every counter is null, never 0. This block is returned when NO column
    was classified at all (the stage collapsed, or no rule pack matched),
    so "0 forbidden columns" would be a count nobody took: it reads as
    "we looked and found none". Null is the module's convention for an
    unmeasured count (see ``build_ll144``'s unknownDemographics and
    ``empty_exports``). The ``unknown`` key is present so the shape
    matches a real ``classify_columns`` summary, which carries five
    counters; a consumer keying on it must not silently get 0 from a
    missing key either.
    """
    out: Dict[str, Any] = {
        "coverage": "uncovered",
        "useCase": "generic",
        "jurisdiction": jurisdiction,
        "jurisdictionId": None,
        "sourceRevision": None,
        "findings": [],
        "summary": {
            "forbidden": None,
            "restricted": None,
            "monitoringOnly": None,
            "allowed": None,
            "unknown": None,
        },
        "summaryNote": (
            "No column was classified on this run, so these counters are "
            "null rather than 0: nothing was counted."
        ),
        "unmappedColumns": [],
    }
    if error:
        out["error"] = f"legal admissibility check unavailable: {error}"
    return out


def legal_admissibility(columns: List[str], domain: str, jurisdiction: str) -> Dict[str, Any]:
    """Column-level legal admissibility for (use_case x jurisdiction).

    Same rule-pack classifier the quarantined legacy pipeline used
    (vfairness.legal.classify_columns). When no pack matches, the block
    reports coverage='uncovered' so the UI stays honest about the gap
    instead of fabricating a verdict.
    """
    try:
        from vfairness.legal import classify_columns, map_domain_to_use_case

        return classify_columns(
            [str(c) for c in columns],
            map_domain_to_use_case(domain),
            jurisdiction,
        )
    except Exception as exc:  # degrade, never crash Pulse
        return uncovered_admissibility(jurisdiction, error=str(exc))


# ───────────────────────────────────────────────────────────────────────────
# G-18: NYC Local Law 144 impact-ratio export
# ───────────────────────────────────────────────────────────────────────────

_LL144_SELF_SERVE_NOTE = "a self-serve Pulse run is not an independent LL144 bias audit"


def build_ll144(
    work: pd.DataFrame,
    usable: List[str],
    y_pred: np.ndarray,
    df: pd.DataFrame,
    requested: List[str],
    domain: str,
    jurisdiction: str,
    score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """NYC Local Law 144 style impact-ratio table.

    Applicable only when the domain reads as employment/hiring AND the
    jurisdiction reads as US/NYC (substring heuristics). Selection rates
    and impact ratios come straight from the run's decisions; the
    above-median scoring rate is filled only when a continuous score
    exists (else null, never invented). Rows with missing values in a
    protected column are counted as unknownDemographics, mirroring the
    LL144 'unknown' reporting category; when no protected column is
    present at all that count is null, not 0, because no demographic
    column was read (0 would read as a measured absence of unknowns).
    """
    prot_cols = [c for c in (requested or []) if c in df.columns]
    if not prot_cols:
        prot_cols = [c for c in (usable or []) if c in df.columns]
    # None, not 0: with no protected column in the frame nothing about
    # demographics was measured, and the module's own convention for an
    # unmeasured count is null (see empty_exports).
    unknown: Optional[int] = int(df[prot_cols].isna().any(axis=1).sum()) if prot_cols else None

    applicable = _looks_employment(domain) and _looks_us(jurisdiction)
    if not applicable:
        if not _looks_employment(domain):
            reason = (
                "Local Law 144 covers automated employment decision "
                "tools; the declared domain "
                f"('{domain or 'unspecified'}') does not read as "
                "employment or hiring."
            )
        else:
            reason = (
                "Local Law 144 applies to tools used for employment "
                "decisions in New York City; the declared "
                "jurisdiction "
                f"('{jurisdiction or 'unspecified'}') does not read "
                "as US/NYC."
            )
        return {
            "applicable": False,
            "reason": reason,
            # True: the applicability question WAS decided on this run and the
            # answer is no. empty_exports writes False for the same key, so a
            # consumer can tell a determined "does not apply" from a collapse
            # without parsing `reason`. See _ll144_screen_state.
            "applicabilityDetermined": True,
            "impactRatios": [],
            # None, not False: the screen did not apply here, which is a
            # different fact from an applicable screen that could not run.
            "impactRatiosComputed": None,
            "unknownDemographics": unknown,
            "notes": [_LL144_SELF_SERVE_NOTE],
        }

    n_rows = min(len(work), len(y_pred))
    yp = np.asarray(y_pred[:n_rows], dtype=float)
    sc: Optional[np.ndarray] = None
    median_score: Optional[float] = None
    if score is not None and len(score) >= n_rows and n_rows > 0:
        sc = np.asarray(score[:n_rows], dtype=float)
        finite = sc[np.isfinite(sc)]
        median_score = float(np.median(finite)) if len(finite) else None

    impact: List[Dict[str, Any]] = []
    for attr in usable or []:
        if attr not in work.columns:
            continue
        col = work[attr].iloc[:n_rows]
        known = col.notna().to_numpy()
        labels = col.astype("string")
        per_group: List[Dict[str, Any]] = []
        best_rate = 0.0
        for lab in pd.unique(labels[known]):
            # fillna(False): the nullable-string comparison yields pd.NA
            # for missing rows, which poisons the numpy boolean mask.
            mask = (labels == lab).fillna(False).to_numpy(dtype=bool) & known
            n = int(mask.sum())
            if n == 0:
                continue
            rate = float(yp[mask].mean())
            best_rate = max(best_rate, rate)
            above: Optional[float] = None
            if sc is not None and median_score is not None:
                svals = sc[mask]
                svals = svals[np.isfinite(svals)]
                if len(svals):
                    above = float((svals > median_score).mean())
            per_group.append(
                {
                    "attribute": attr,
                    "group": str(lab),
                    "n": n,
                    "selectionRate": rate,
                    "impactRatio": None,
                    "aboveMedianRate": above,
                }
            )
        for row in per_group:
            row["impactRatio"] = float(row["selectionRate"] / best_rate) if best_rate > 0 else None
        per_group.sort(key=lambda r: (r["impactRatio"] is None, r["impactRatio"] or 0.0))
        impact.extend(per_group)

    notes = [
        "Impact ratio = each group's selection rate divided by the "
        "highest group's selection rate, following the NYC DCWP Local "
        "Law 144 methodology.",
        "Rows with a missing value in a protected column are counted "
        "under unknownDemographics and excluded from that column's "
        "impact-ratio rows, mirroring the LL144 'unknown' category.",
        _LL144_SELF_SERVE_NOTE,
    ]
    if unknown is None:
        notes.append(
            "No protected column was present in this run, so the "
            "unknown-demographics count is null rather than 0: it was "
            "not measured."
        )
    if sc is None:
        notes.insert(
            2,
            "No continuous score column exists in this run, "
            "so the above-median scoring rate is null rather "
            "than invented.",
        )
    # An EMPTY table under applicable=True reads as "the screen ran and
    # no group showed a disparity". It can also mean the screen never
    # ran: no assessable protected attribute reached this function, no
    # decision was scored, or every protected value was missing. Those
    # are different facts and the block must say which one happened, the
    # same way unknownDemographics already distinguishes null from 0.
    screened: Optional[bool] = bool(impact)
    if not impact:
        screen_note: str
        if n_rows == 0:
            screen_note = (
                "No impact ratio could be computed: this run carried no "
                "scored decision row, so no selection rate was measured. "
                "An empty table here is an unrun screen, not a clean one."
            )
        elif not [a for a in (usable or []) if a in work.columns]:
            screen_note = (
                "No impact ratio could be computed: no assessable protected "
                "attribute was available on this run. An empty table here is "
                "an unrun screen, not a clean one."
            )
        else:
            screen_note = (
                "No impact ratio could be computed: every row of the "
                "protected attributes carried a missing value, so no group "
                "selection rate exists. An empty table here is an unrun "
                "screen, not a clean one."
            )
        notes.append(screen_note)
    return {
        "applicable": True,
        "reason": "",
        # The applicability question was decided on this run; see
        # _ll144_screen_state and empty_exports for the collapsed counterpart.
        "applicabilityDetermined": True,
        "impactRatios": impact,
        # Three states at the surface: True = groups were screened and the
        # rows above are the result; False = nothing could be screened and
        # the note says why. Never silently empty.
        "impactRatiosComputed": screened,
        "unknownDemographics": unknown,
        "notes": notes,
    }


# ───────────────────────────────────────────────────────────────────────────
# G-18: EU AI Act Art. 10 section draft
# ───────────────────────────────────────────────────────────────────────────


def build_art10(
    domain: str,
    jurisdiction: str,
    artifact_label: Optional[str],
    artifact_hash: Optional[str],
    rows: int,
    columns: int,
    schema: Dict[str, Any],
    data_quality: Dict[str, Any],
    verdict: Dict[str, Any],
    bias_findings: List[Dict[str, Any]],
    interventions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Draft the four Art. 10 data-governance sections from the run's
    REAL data (artifact identity, schema findings, verdict, controls).
    Applicable only for an EU jurisdiction; every body sentence traces
    to a computed value, nothing is fabricated.

    ``domain`` selects the G-31 historical-discrimination catalog entry
    that the Art. 10(2)(f) examination section cites, so a hiring model
    and a credit model do not receive the same examination text. Nothing
    domain-specific is written here: the patterns, priority checks and
    citations are the catalog's own, verbatim. A domain with no catalog
    entry, and a run with no declared domain, are BOTH disclosed in the
    section body and in ``domainContext`` rather than papered over with
    generic prose, so a domain-independent draft is never mistaken for a
    domain-specific one.
    """
    if not _looks_eu(jurisdiction):
        return {
            "applicable": False,
            "reason": (
                "The EU AI Act Art. 10 data-governance draft is "
                "assembled only for EU jurisdictions; the "
                "declared jurisdiction "
                f"('{jurisdiction or 'unspecified'}') does not "
                "read as EU."
            ),
            "sections": [],
        }

    label = str(artifact_label or "").strip() or "the uploaded artifact"
    hash_txt = (
        f"content hash {artifact_hash}"
        if artifact_hash
        else "no content hash was supplied with this run"
    )
    src_body = (
        f"Assessed artifact: {label} ({hash_txt}). The dataset as "
        f"received contains {int(rows)} rows and {int(columns)} columns. "
        "This Pulse run analysed the artifact exactly as uploaded; no "
        "external data sources were joined."
    )

    pii = [str(c) for c in (schema.get("pii_leakage") or [])]
    oracle = [str(c) for c in (schema.get("oracle_columns") or [])]
    model_out = [str(c) for c in (schema.get("model_output") or [])]
    gov_parts: List[str] = []
    if pii:
        gov_parts.append(
            "identity/PII columns detected and excluded from "
            "feature-based checks: " + ", ".join(pii[:8])
        )
    if oracle:
        gov_parts.append(
            "oracle/answer-key columns detected and excluded: " + ", ".join(oracle[:8])
        )
    if model_out:
        gov_parts.append("model-output columns identified: " + ", ".join(model_out[:8]))
    dq_head = str((data_quality or {}).get("headline") or "").strip()
    if dq_head:
        gov_parts.append("data-quality reading: " + dq_head)
    # Did the column-role screen actually produce a result? Its collapse
    # fallback is an all-empty block (orchestrator ``schema_roles``), which
    # is byte-identical to a screen that ran and flagged nothing. Asserting
    # "screening ran" and "nothing was flagged" from that block writes an
    # unrun check into an EU AI Act Art. 10 draft. A screen that ran always
    # reports per-column roles, so any non-empty screen output is the
    # evidence it ran; nothing at all is a gap, and this section says so.
    _screen_keys = (
        "roles",
        "pii_leakage",
        "oracle_columns",
        "model_output",
        "protected",
        "proxy_candidates",
        "job_relevant",
        "unknown",
        "mismatches",
    )
    screen_ran = any((schema or {}).get(k) for k in _screen_keys)
    if screen_ran:
        gov_head = (
            "Column-role screening and data-quality checks ran before any metric was computed. "
        )
        gov_tail = (
            "; ".join(gov_parts) + "."
            if gov_parts
            else "No PII or oracle columns were flagged by the screen."
        )
    else:
        gov_head = "Data-quality checks ran before any metric was computed. "
        gov_tail = (
            "The column-role screen returned no result on this run, so this "
            "draft cannot state whether identity, oracle or model-output "
            "columns are present: that is an unchecked gap, not a clean "
            "screen."
        )
        if gov_parts:
            gov_tail += " Recorded separately: " + "; ".join(gov_parts) + "."
    gov_body = gov_head + gov_tail

    v = verdict or {}
    top = [
        str(f.get("plain") or "").strip()
        for f in (bias_findings or [])
        if str(f.get("plain") or "").strip()
    ][:3]
    exam_body = (
        str(v.get("headline") or "").strip() + " " + str(v.get("summary") or "").strip()
    ).strip() or "No verdict text was produced on this run."
    if top:
        exam_body += " Top findings: " + " | ".join(top)

    # G-18 x G-31. Art. 10(2)(f) asks for an examination "in view of possible
    # biases", which is a domain question: the same statistics mean different
    # things for a hiring screen and a credit scorecard. The draft therefore
    # selects this module's OWN curated historical-discrimination material for
    # the declared domain instead of reading identically for every domain.
    # Three states, never two: a catalog match, a domain that was declared but
    # has no catalog entry, and no declared domain at all. The last two get a
    # plain statement that the draft is domain independent, never invented
    # domain text.
    declared = str(domain or "").strip()
    hist: Dict[str, Any] = historical_context(declared) if declared else {"available": False}
    if hist.get("available"):
        domain_context: Dict[str, Any] = {
            "state": "catalog_match",
            "declaredDomain": declared,
            "catalogDomain": hist["domain"],
            "citations": list(hist["citations"]),
        }
        exam_body += (
            f" Domain-specific examination for '{declared}' "
            f"(historical-discrimination catalog entry: {hist['domain']}). "
            "Documented patterns: "
            + " ".join(hist["patterns"])
            + " Priority checks for this domain: "
            + "; ".join(hist["checkPriorities"])
            + ". Sources: "
            + "; ".join(hist["citations"])
            + "."
        )
    elif declared:
        domain_context = {
            "state": "declared_no_catalog_entry",
            "declaredDomain": declared,
            "catalogDomain": None,
            "citations": [],
        }
        exam_body += (
            " No historical-discrimination catalog entry matches the declared "
            f"domain '{declared}', so this draft carries no domain-specific "
            "examination material: the sections are domain independent and "
            "rest on this run's own measurements alone."
        )
    else:
        domain_context = {
            "state": "not_declared",
            "declaredDomain": None,
            "catalogDomain": None,
            "citations": [],
        }
        exam_body += (
            " No domain was declared for this run, so this draft carries no "
            "domain-specific examination material: the sections are domain "
            "independent and rest on this run's own measurements alone."
        )

    ctl = [
        f"{str(i.get('title') or '').strip()}: "
        f"{str(i.get('how') or i.get('control') or '').strip()}"
        for i in (interventions or [])
        if str(i.get("title") or "").strip()
    ][:5]
    mit_body = (
        "Recommended controls mapped from the detected bias mechanisms: " + " | ".join(ctl) + "."
        if ctl
        else "No intervention recommendations were produced on this "
        "run; convert to a full assessment to plan mitigation."
    )

    sections = [
        {"id": "data_sources", "title": "Data and data sources (Art. 10(2))", "body": src_body},
        {
            "id": "data_governance",
            "title": "Data governance and management practices (Art. 10(2))",
            "body": gov_body,
        },
        {
            "id": "bias_examination",
            "title": "Examination in view of possible biases (Art. 10(2)(f))",
            "body": exam_body,
        },
        {
            "id": "bias_mitigation",
            "title": "Measures to detect, prevent and mitigate biases (Art. 10(2)(g))",
            "body": mit_body,
        },
    ]
    return {
        "applicable": True,
        "reason": "",
        "domainContext": domain_context,
        # Machine-readable twin of the governance section's own statement,
        # so a reader does not have to parse prose to learn that the
        # column-role screen never reported on this run.
        "columnRoleScreen": "reported" if screen_ran else "unavailable",
        "sections": sections,
    }


# ───────────────────────────────────────────────────────────────────────────
# G-24: ISO/IEC TR 24027:2021 metric crosswalk
# ───────────────────────────────────────────────────────────────────────────


def iso24027_crosswalk(
    has_truth: bool = False, calibration_available: bool = False, output_type: str = "binary"
) -> List[Dict[str, str]]:
    """Static crosswalk from the metrics Pulse actually computed on this
    pathway to their ISO/IEC TR 24027:2021 names. Clause labels are kept
    honest: the TR's fairness-metrics catalogue is named without
    inventing precise subclause numbers.
    """
    clause = "ISO/IEC TR 24027:2021 fairness metrics"
    bias_clause = "ISO/IEC TR 24027:2021 sources of unwanted bias"
    rows: List[Dict[str, str]] = []
    if output_type in ("binary",):
        rows.append(
            {
                "ourMetric": "selection_rate_difference",
                "isoName": "Demographic parity difference (statistical parity)",
                "clause": clause + " (demographic parity)",
            }
        )
        rows.append(
            {
                "ourMetric": "four_fifths_ratio",
                "isoName": "Impact ratio (four-fifths / adverse-impact screen)",
                "clause": clause + " (impact ratio)",
            }
        )
        rows.append(
            {
                "ourMetric": "cramers_v_association",
                "isoName": "Association between protected attribute and features (proxy screening)",
                "clause": bias_clause + " (indirect / proxy variables)",
            }
        )
    if output_type == "binary" and has_truth:
        rows.append(
            {
                "ourMetric": "equalized_odds_difference",
                "isoName": "Equalized odds (error-rate balance)",
                "clause": clause + " (equalized odds)",
            }
        )
        rows.append(
            {
                "ourMetric": "equal_opportunity_difference",
                "isoName": "Equal opportunity (true-positive-rate parity)",
                "clause": clause + " (equal opportunity)",
            }
        )
    if calibration_available:
        rows.append(
            {
                "ourMetric": "group_expected_calibration_error",
                "isoName": "Calibration across groups",
                "clause": clause + " (calibration)",
            }
        )
    if output_type == "rank":
        rows.append(
            {
                "ourMetric": "ranking_exposure_ndkl",
                "isoName": "No direct ISO/IEC TR 24027 equivalent; nearest "
                "concept is demographic parity applied to "
                "ranking exposure",
                "clause": clause + " (nearest concept; NDKL itself is not enumerated in the TR)",
            }
        )
    if output_type == "continuous":
        rows.append(
            {
                "ourMetric": "mean_prediction_difference",
                "isoName": "Group mean-outcome difference (regression parity)",
                "clause": clause + " (parity of predicted outcomes)",
            }
        )
    return rows


def build_regulatory_exports(
    work: pd.DataFrame,
    usable: List[str],
    y_pred: np.ndarray,
    df: pd.DataFrame,
    requested: List[str],
    domain: str,
    jurisdiction: str,
    score: Optional[np.ndarray],
    artifact_label: Optional[str],
    artifact_hash: Optional[str],
    schema: Dict[str, Any],
    data_quality: Dict[str, Any],
    verdict: Dict[str, Any],
    bias_findings: List[Dict[str, Any]],
    interventions: List[Dict[str, Any]],
    has_truth: bool,
    calibration_available: bool,
    output_type: str = "binary",
) -> Dict[str, Any]:
    """Assemble the full regulatoryExports block (G-18 + G-24)."""
    return {
        "legalContextAsOf": legal_context_as_of(),
        "ll144": build_ll144(
            work, usable, y_pred, df, requested, domain, jurisdiction, score=score
        ),
        "art10": build_art10(
            domain,
            jurisdiction,
            artifact_label,
            artifact_hash,
            int(len(df)),
            int(df.shape[1]),
            schema,
            data_quality,
            verdict,
            bias_findings,
            interventions,
        ),
        "iso24027": iso24027_crosswalk(
            has_truth=has_truth,
            calibration_available=calibration_available,
            output_type=output_type,
        ),
    }


_NO_REASON_GIVEN = "the regulatory stage collapsed and no reason was recorded by the caller"


def empty_exports(reason: str) -> Dict[str, Any]:
    """Fallback exports block when the stage collapses.

    `applicable: False` here means "this could not be evaluated", NOT "no regulation
    applies", and the only thing on the block that distinguishes those two readings is
    `reason`. So a blank reason is substituted rather than passed through: empty_exports
    with reason="" produced `{"applicable": false, "reason": ""}`, which a reader can
    only take as a considered finding that the regulation does not apply.
    """
    if not (reason or "").strip():
        warnings.warn(
            "empty_exports was called with no reason. The block it returns says "
            "applicable=False for every framework, which reads as 'no regulation "
            "applies' unless the reason says otherwise.",
            stacklevel=2,
        )
        reason = _NO_REASON_GIVEN
    return {
        "legalContextAsOf": None,
        # BGL5 A-operations-3, 2026-09-27. MACHINE-READABLE, beside the prose
        # `reason`. `applicable: False` on this block means "not evaluated", and
        # until now the only thing separating it from a determined "does not
        # apply" was that sentence, which no consumer parses. A caller reading
        # `applicable` alone therefore could not tell the two apart, and
        # orchestrator.py's `bool((exports.get("ll144") or {}).get("applicable"))`
        # flattened the collapse into the determined False that publishes
        # "No annual audit statute matched this run". Use
        # ``_ll144_screen_state(exports)`` to read the three-state value.
        "screenRan": False,
        "ll144": {
            "applicable": False,
            "reason": reason,
            # False here, True on both returns of build_ll144: whether the
            # applicability question was actually decided on this run.
            "applicabilityDetermined": False,
            "impactRatios": [],
            "impactRatiosComputed": None,
            "unknownDemographics": None,
            "notes": [_LL144_SELF_SERVE_NOTE],
        },
        "art10": {"applicable": False, "reason": reason, "sections": []},
        "iso24027": [],
    }


# ───────────────────────────────────────────────────────────────────────────
# G-34: recheck window + secret-free suite echo
# ───────────────────────────────────────────────────────────────────────────

_SECRET_KEY_TOKENS = (
    "auth_token",
    "authtoken",
    "api_key",
    "apikey",
    "secret",
    "password",
    "credential",
    "model_base64",
    "modelbase64",
    "model_envelope",
    "modelenvelope",
    "org_key",
    "orgkey",
    "organization_key",
    "access_key",
    "accesskey",
    "private_key",
    "privatekey",
)


def _is_secret_key(key: Any) -> bool:
    k = str(key).strip().lower().replace("-", "_")
    if k in ("token", "auth", "authorization"):
        return True
    return any(tok in k for tok in _SECRET_KEY_TOKENS)


def sanitize_suite(obj: Any) -> Any:
    """Deep copy of the run inputs with every secret-bearing key dropped
    (auth tokens, API keys, passwords, model blobs, org keys), including
    nested dicts such as llm_config. Non-JSON leaves are stringified so
    the suite always serialises.
    """
    if isinstance(obj, dict):
        return {str(k): sanitize_suite(v) for k, v in obj.items() if not _is_secret_key(k)}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_suite(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _add_months(dt: datetime, months: int) -> datetime:
    """Calendar-safe month addition (day clamped to the target month)."""
    month_index = dt.month - 1 + int(months)
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _iso_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _ll144_screen_state(regulatory_exports: Any) -> Optional[bool]:
    """Three-state reading of an exports block's LL144 applicability screen.

    ``True`` / ``False`` when the screen ran and decided, ``None`` when it did
    not run, which is the state ``build_recheck`` needs and which
    ``bool(exports["ll144"]["applicable"])`` destroys: a collapsed block from
    ``empty_exports`` carries ``applicable=False`` meaning "not evaluated", and
    flattening it to ``False`` publishes the legal finding "No annual audit
    statute matched this run" that nobody made.

    Added BGL5 A-operations-3, 2026-09-27, so a caller holding the exports block
    can pass the third state through instead of a bool.
    """
    if not isinstance(regulatory_exports, dict):
        return None
    block = regulatory_exports.get("ll144")
    if not isinstance(block, dict):
        return None
    if regulatory_exports.get("screenRan") is False:
        return None
    determined = block.get("applicabilityDetermined")
    if determined is False:
        return None
    applicable = block.get("applicable")
    if isinstance(applicable, (bool, np.bool_)):
        # An older block with no `applicabilityDetermined` key at all: a missing
        # marker is not a determination, so only an explicit True marker or a
        # True applicability is taken at face value. A bare `applicable: False`
        # from an unmarked block stays undetermined, which is the safe side.
        if determined is None and applicable is False:
            return None
        return bool(applicable)
    return None


def build_recheck(inputs: Dict[str, Any], ll144_applicable: Optional[bool]) -> Dict[str, Any]:
    """G-34: when this Pulse read stops being current.

    +12 months when LL144 applies (annual bias-audit cadence), else +6
    months (Pulse's default hygiene interval for drift). The suite echo
    is the caller's inputs with every secret stripped, so the exact run
    can be repeated without re-capturing configuration.

    ``ll144_applicable`` is three-state. None means the LL144 screen did
    not run (the export stage collapsed), and the basis then SAYS the
    statute test did not happen instead of claiming "no annual audit
    statute matched this run", which is a finding about the law that
    nobody made. The window still defaults to 6 months, because a
    shorter recheck interval is the safe side of an unknown.

    Only a genuine boolean counts as a determination. Anything else (a NaN, the
    string "unknown", an int, any other object) is read as the third state and
    warned about: BGL5 A-operations-3 measured ``build_recheck({}, float("nan"))``
    and ``build_recheck({}, "unknown")`` both publishing "NYC Local Law 144
    requires a bias audit ..." with a 12-month window, because every non-None
    spelling of could-not-measure is truthy.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    # BGL5 A-operations-3, 2026-09-27. `months = 12 if ll144_applicable else 6`
    # and `elif ll144_applicable is None` made `is None` the ONLY route to the
    # third state, and truthiness the route to the strongest claim.
    # Measured before, build_recheck({}, <value>)["basis"]:
    #   float("nan") -> "NYC Local Law 144 requires a bias audit ..." (12 months)
    #   "unknown"    -> "NYC Local Law 144 requires a bias audit ..." (12 months)
    # Measured after: both -> "Whether an annual audit statute applies could not
    # be determined on this run ..." (6 months), plus a warning naming the type.
    # True, False and None are unchanged, which the over-correction control
    # asserts with the exact windows (+12, +6, +6).
    if ll144_applicable is not None and not isinstance(ll144_applicable, (bool, np.bool_)):
        warnings.warn(
            f"build_recheck received ll144_applicable={ll144_applicable!r} "
            f"({type(ll144_applicable).__name__}), which is not a statute screen "
            "verdict. Only True, False or None are verdicts, so this run is "
            "recorded as UNDETERMINED rather than as the statute applying.",
            stacklevel=2,
        )
        ll144_applicable = None
    elif isinstance(ll144_applicable, np.bool_):
        ll144_applicable = bool(ll144_applicable)
    months = 12 if ll144_applicable else 6
    if ll144_applicable:
        basis = (
            "NYC Local Law 144 requires a bias audit conducted no "
            "more than one year before use, so this read should be "
            "redone within 12 months."
        )
    elif ll144_applicable is None:
        basis = (
            "Whether an annual audit statute applies could not be "
            "determined on this run: the regulatory screen did not "
            "complete, so this is NOT a finding that none applies. The "
            "6-month window is Pulse's default hygiene interval; if NYC "
            "Local Law 144 covers this tool, the binding interval is 12 "
            "months from the audit date."
        )
    else:
        basis = (
            "No annual audit statute matched this run, so the "
            "6-month window is Pulse's default hygiene interval: "
            "data and behaviour drift make an older read stale "
            "evidence."
        )
    return {
        "assessedAt": _iso_utc(now),
        "validUntil": _iso_utc(_add_months(now, months)),
        "basis": basis,
        "suite": sanitize_suite(inputs or {}),
    }


# ───────────────────────────────────────────────────────────────────────────
# G-31: historical-context domain catalog (REAL citations only)
# ───────────────────────────────────────────────────────────────────────────

# Ordered list: first keyword match wins. Citations are canonical,
# verifiable references; per the house rule, none may be fabricated and
# an unknown domain must return {available: False} instead of inventing.
_DOMAIN_HISTORY: List[Dict[str, Any]] = [
    {
        "domain": "hiring",
        "keywords": (
            "hir",
            "recruit",
            "employ",
            "promot",
            "talent",
            "workforce",
            "staffing",
            "applicant",
        ),
        "patterns": [
            "Identical resumes get fewer callbacks when the name signals "
            "a minority group; name, address and club memberships act as "
            "stand-ins for race and class.",
            "Screening tools trained on past hiring learn the old "
            "workforce's shape: terms associated with women or minority "
            "institutions get penalised.",
            "Requirements that look neutral (credentials, tests, "
            "physical criteria) can screen out protected groups without "
            "predicting job performance, the classic disparate-impact "
            "pattern.",
            "Age screening: older applicants filtered out via graduation "
            "dates, 'digital native' phrasing or experience caps.",
        ],
        "defaultAttributes": ["gender", "race_ethnicity", "age", "disability", "national_origin"],
        "checkPriorities": [
            "Selection rates and impact ratios per protected group "
            "(the four-fifths screen) before anything else",
            "Proxy features: names, postcodes/ZIP, employment gaps, graduation years, affiliations",
            "Qualified-candidate parity (equal opportunity) once ground-truth outcomes exist",
        ],
        "citations": [
            "Bertrand and Mullainathan 2004 (AER): resume callback disparity by name",
            "Reuters 2018: Amazon scrapped a resume tool penalizing the word women's",
            "Griggs v. Duke Power Co., 401 U.S. 424 (1971): disparate "
            "impact doctrine for employment screens",
            "EEOC Uniform Guidelines on Employee Selection Procedures (1978): the four-fifths rule",
        ],
    },
    {
        "domain": "lending",
        "keywords": ("lend", "credit", "loan", "mortgage", "underwrit", "bnpl"),
        "patterns": [
            "Redlining: whole neighbourhoods historically denied credit "
            "by geography; postcode and geography features still carry "
            "that legacy.",
            "Minority borrowers historically received worse pricing and "
            "higher denial rates at the same creditworthiness.",
            "Credit-history features encode past exclusion: groups "
            "denied credit in the past have thinner files today, so "
            "'thin file' penalties recycle the original discrimination.",
        ],
        "defaultAttributes": ["race_ethnicity", "gender", "age", "marital_status", "zip_code"],
        "checkPriorities": [
            "Approval-rate and pricing disparity per group at matched risk levels",
            "Geographic proxies (postcode/ZIP, census tract) standing in for race",
            "Thin-file and credit-history features that penalise historically excluded groups",
        ],
        "citations": [
            "Munnell et al. 1996 (AER): Boston Fed study of race disparities in mortgage lending",
            "Bartlett, Morse, Stanton and Wallace 2022 (Journal of "
            "Financial Economics): consumer-lending discrimination in "
            "the FinTech era",
            "US Fair Housing Act (1968) and Equal Credit Opportunity "
            "Act (1974): statutory record of redlining and credit "
            "discrimination",
        ],
    },
    {
        "domain": "healthcare",
        "keywords": ("health", "clinic", "medic", "patient", "hospital", "diagnos", "triage"),
        "patterns": [
            "Cost used as a proxy for need: because less money was "
            "historically spent on Black patients, cost-trained models "
            "under-estimated their illness at the same severity.",
            "Race 'corrections' baked into clinical formulas (kidney "
            "function, lung function) shifted diagnosis and treatment "
            "thresholds by race without a sound physiological basis.",
            "Measurement devices and datasets skewed toward "
            "lighter-skinned or male patients degrade accuracy for "
            "everyone else (for example pulse oximetry).",
        ],
        "defaultAttributes": [
            "race_ethnicity",
            "gender",
            "age",
            "socioeconomic_status",
            "disability",
        ],
        "checkPriorities": [
            "Proxy targets first: is the label (cost, utilisation) a "
            "stand-in for the real clinical need?",
            "Error-rate parity (false negatives especially) per group",
            "Calibration: does the same risk score mean the same clinical reality across groups?",
        ],
        "citations": [
            "Obermeyer et al. 2019 (Science): healthcare cost-proxy bias",
            "Sjoding et al. 2020 (NEJM): racial bias in pulse oximetry measurement",
            "Vyas, Eisenstein and Jones 2020 (NEJM): hidden in plain "
            "sight, race correction in clinical algorithms",
        ],
    },
    {
        "domain": "criminal_justice",
        "keywords": ("recidiv", "justice", "bail", "parole", "polic", "sentenc", "criminal"),
        "patterns": [
            "Risk scores produced higher false-positive 'high risk' "
            "labels for Black defendants than white defendants at the "
            "same reoffending outcomes (the COMPAS debate).",
            "Arrest and enforcement data reflect where policing "
            "happened, not where crime happened; models trained on it "
            "send patrols back to the same neighbourhoods (feedback "
            "loops).",
            "Socioeconomic inputs (employment, housing stability, "
            "family criminality) function as class and race proxies in "
            "risk instruments.",
        ],
        "defaultAttributes": [
            "race_ethnicity",
            "age",
            "gender",
            "socioeconomic_status",
            "zip_code",
        ],
        "checkPriorities": [
            "False-positive-rate parity first: liberty-affecting harm "
            "concentrates in wrongful high-risk labels",
            "Calibration versus error-rate trade-off made explicit "
            "(they cannot all hold when base rates differ)",
            "Feedback loops: is the training data itself an enforcement artifact?",
        ],
        "citations": [
            "ProPublica 2016 COMPAS analysis (Angwin et al.): machine bias in recidivism scores",
            "Chouldechova 2017 (Big Data): fair prediction with "
            "disparate impact, the base-rate impossibility",
            "Dressel and Farid 2018 (Science Advances): COMPAS accuracy "
            "no better than untrained humans",
            "Lum and Isaac 2016 (Significance): feedback loops in predictive policing",
        ],
    },
    {
        "domain": "insurance",
        "keywords": ("insur", "actuar", "premium"),
        "patterns": [
            "Territorial pricing tracks residential segregation: "
            "postcode-based rates charged minority neighbourhoods more "
            "at comparable underlying risk.",
            "Gender pricing was standard in EU insurance until it was "
            "ruled discriminatory (Test-Achats); legacy gender features "
            "may persist in data.",
            "Credit-based insurance scores import credit history's "
            "documented racial and income skews into premiums.",
        ],
        "defaultAttributes": [
            "race_ethnicity",
            "gender",
            "age",
            "zip_code",
            "socioeconomic_status",
        ],
        "checkPriorities": [
            "Premium / acceptance disparity per group at matched actuarial risk",
            "Geographic and credit-score proxies standing in for protected traits",
            "Jurisdiction rules on rating factors (EU gender ban vs US state-by-state regimes)",
        ],
        "citations": [
            "CJEU Test-Achats ruling (C-236/09, 2011): gender rating in "
            "insurance premiums banned in the EU",
            "Consumer Reports and ProPublica 2017: car insurance "
            "premiums higher in minority neighbourhoods at similar risk",
            "US FTC 2007 report to Congress: credit-based insurance "
            "scores and their demographic impacts",
        ],
    },
    {
        "domain": "education",
        "keywords": ("educat", "admiss", "school", "univers", "college", "student", "grading"),
        "patterns": [
            "Algorithmic grading anchored to a school's historical "
            "results downgrades strong students from historically "
            "lower-attaining (often poorer) schools.",
            "Admissions criteria (standardised tests, legacy "
            "preferences, extracurricular signals) proxy family wealth "
            "and race.",
            "Tracking and ability-grouping decisions made early "
            "compound into unequal opportunity later.",
        ],
        "defaultAttributes": [
            "socioeconomic_status",
            "race_ethnicity",
            "gender",
            "disability",
            "school_type",
        ],
        "checkPriorities": [
            "Whether school-level or neighbourhood-level features drive "
            "individual student outcomes",
            "Selection/placement rates per group and per school type",
            "Outcome parity for equally qualified students once labels exist",
        ],
        "citations": [
            "UK Ofqual 2020 A-level grading algorithm: downgrades "
            "concentrated on students from historically lower-attaining "
            "schools; withdrawn after public outcry",
            "Students for Fair Admissions v. Harvard (US Supreme Court "
            "2023): litigation record on admissions criteria and race",
        ],
    },
]


def historical_context(domain: str) -> Dict[str, Any]:
    """G-31: documented historical discrimination patterns for the run's
    domain. Real canonical citations only; an unknown domain returns
    {available: False, reason} rather than fabricated history.
    """
    d = (domain or "").strip().lower()
    if d:
        for entry in _DOMAIN_HISTORY:
            if any(k in d for k in entry["keywords"]):
                return {
                    "available": True,
                    "domain": entry["domain"],
                    "patterns": list(entry["patterns"]),
                    "defaultAttributes": list(entry["defaultAttributes"]),
                    "checkPriorities": list(entry["checkPriorities"]),
                    "citations": list(entry["citations"]),
                }
    return {
        "available": False,
        "reason": (
            "No documented historical-discrimination catalog entry "
            f"matches the domain '{domain or 'unspecified'}'. "
            "Rather than fabricate precedent, Pulse reports none; "
            "the statistical findings above stand on their own."
        ),
    }


# ───────────────────────────────────────────────────────────────────────────
# G-32: four-fifths target block + reference-group disclosure
# ───────────────────────────────────────────────────────────────────────────


def _as_real(value: Any) -> Optional[float]:
    """Float for any real number, None for anything that is not one.

    ``isinstance(v, (int, float))`` is NOT the test: it rejects
    ``np.float32`` and ``np.int64`` (real measurements a producer may
    hand over, discarded as if unmeasurable) while accepting ``bool``,
    where ``float(True) == 1.0`` would manufacture a perfect ratio out of
    a flag. numbers.Real covers the numpy scalars and the stdlib types;
    bool is excluded explicitly, and a numeric string is still read
    because the pulse inputs arrive as JSON.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    try:  # Decimal, Fraction, anything else with a real __float__
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp_target_ratio(raw: Any) -> float:
    """Parse and clamp the caller's target_ratio to [0.5, 1.0]; 0.8 when
    absent or unparseable (including a bool, which is a flag and not a
    ratio: ``float(True)`` would silently set the strictest target)."""
    r = _as_real(raw)
    if r is None or r != r:  # unparseable, or NaN
        return 0.8
    return max(0.5, min(1.0, r))


def build_targets(per_variable: List[Dict[str, Any]], target_ratio_raw: Any) -> Dict[str, Any]:
    """Per-attribute pass/fail against the caller's four-fifths target.

    ``passes`` is fourFifthsRatio >= targetRatio, computed only for rows
    that actually carry a numeric four-fifths ratio (never invented for
    non-binary rows).
    """
    ratio = clamp_target_ratio(target_ratio_raw)
    rows: List[Dict[str, Any]] = []
    # Attributes that WERE assessed but carry no usable four-fifths ratio.
    # Dropping them silently leaves an empty perAttribute list that reads
    # exactly like "every attribute passed" / "nothing was assessable".
    unrated: List[str] = []
    for r in per_variable or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("attribute") or "")
        if not r.get("assessable"):
            # Not assessed at all (one group only, every group too small,
            # excluded by preparation). Skipping it silently left an empty
            # table that reads as "every attribute cleared the target".
            unrated.append(name)
            continue
        ff = _as_real(r.get("fourFifthsRatio"))
        if ff is None or not math.isfinite(ff):
            # None: four-fifths does not apply (continuous / rank output)
            # or the producer could not compute it. NaN or an infinity:
            # undefined. Either way there is no ratio to test, and no
            # pass/fail may be invented for it.
            unrated.append(name)
            continue
        rows.append(
            {
                "attribute": name,
                "fourFifthsRatio": ff,
                "passes": bool(ff >= ratio),
            }
        )
    note = (
        f"An attribute passes when its worst-to-best selection-rate "
        f"ratio (the four-fifths ratio) reaches at least {ratio:.2f}. "
        "0.80 mirrors the EEOC adverse-impact screen; a higher target "
        "is stricter than the legal floor."
    )
    # Disclose honestly WHICH correction happened: a numeric value outside
    # [0.5, 1.0] was clamped; a value that could not be parsed at all (or
    # NaN) fell back to the 0.80 default. Saying "clamped" for garbage
    # input would falsely imply the caller's number was read.
    clamped = False
    unparseable = False
    if target_ratio_raw is not None:
        raw_f = _as_real(target_ratio_raw)
        if raw_f is None or raw_f != raw_f:  # unparseable, or NaN
            unparseable = True
        else:
            clamped = abs(raw_f - ratio) > 1e-9
    if clamped:
        note += " The requested target was outside the accepted 0.50 to 1.00 band and was clamped."
    elif unparseable:
        note += (
            " The requested target could not be parsed as a number, so the default 0.80 was used."
        )
    if unrated:
        note += (
            " No four-fifths ratio exists for "
            + ", ".join(sorted(n for n in unrated if n))
            + ", so no pass or fail is reported for "
            + ("those attributes" if len(unrated) > 1 else "that attribute")
            + ": absence from the table above is an unrun test, not a pass. "
            "The per-attribute detail block records why for each."
        )
    return {
        "targetRatio": ratio,
        "perAttribute": rows,
        # Assessed attributes that carry no four-fifths ratio at all. An
        # empty perAttribute list otherwise reads as "nothing failed".
        "ratioUnavailable": sorted(n for n in unrated if n),
        "note": note,
    }


def build_reference_groups(
    requested: Dict[str, str], per_variable: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """G-32 disclosure: per attribute, which reference group was
    requested, which one the engine actually used, and whether the
    request was honored. Covers requested attributes that were not
    assessable, too, so a silently-ignored request can never hide.

    ``honored`` is three-state. True and False are the engine's own
    ``referenceOverrideHonored`` verdict. None means the engine recorded
    no verdict for this attribute (the key is absent, as on the
    non-binary per-variable rows): treating that absence as False made
    the block assert '"male" was requested but was not found among this
    attribute's groups' about a row whose reference group IS "male": a
    statement about the data produced by a check that never ran.
    """
    requested = requested or {}
    pv_by_attr: Dict[str, Dict[str, Any]] = {}
    for r in per_variable or []:
        if isinstance(r, dict) and r.get("attribute"):
            pv_by_attr[str(r["attribute"])] = r
    out: Dict[str, Any] = {}
    for attr in sorted(set(list(pv_by_attr.keys()) + [str(k) for k in requested.keys()])):
        row = pv_by_attr.get(attr)
        req = requested.get(attr)
        used = (
            str(row.get("referenceGroup"))
            if row and row.get("assessable") and row.get("referenceGroup") is not None
            else None
        )
        honored: Optional[bool] = None
        if row is not None and row.get("referenceOverrideHonored") is not None:
            honored = bool(row.get("referenceOverrideHonored"))
        if req is None:
            note = (
                "No reference group was requested; the engine used "
                "its default (the most-populous reliable group)."
            )
        elif row is None or not row.get("assessable"):
            note = (
                f'"{req}" was requested, but this attribute was not '
                "assessable on this run, so no reference was applied."
            )
        elif honored:
            note = (
                f'The requested reference group "{req}" exists in the '
                "data and was used as the comparison baseline for "
                "this attribute."
            )
        elif honored is None:
            note = (
                f'"{req}" was requested, but this run recorded no verdict on '
                "whether the override was applied to this attribute, so "
                "whether the request was honored is unknown. The reference "
                "group actually used is reported as "
                + (f'"{used}".' if used is not None else "unavailable.")
            )
        elif used is None:
            note = (
                f'"{req}" was requested but was not found among this '
                "attribute's groups, and no default reference was "
                "recorded for it either."
            )
        else:
            note = (
                f'"{req}" was requested but was not found among this '
                "attribute's groups; the engine fell back to its "
                f'default reference "{used}".'
            )
        out[attr] = {"requested": req, "used": used, "honored": honored, "note": note}
    return out


# ───────────────────────────────────────────────────────────────────────────
# G-19 support: does the data show differing base rates?
# ───────────────────────────────────────────────────────────────────────────


def _warn_base_rates(reason: str) -> None:
    """Say out loud that the base-rate question went unanswered.

    The return value (None) already carries it, but the caller chain ends
    in prose a person reads, so the reason must be available to a log as
    well as to a branch.
    """
    warnings.warn(
        "base_rates_differ: group base rates could not be compared because "
        f"{reason}. Reporting None (not measured), never False.",
        UserWarning,
        stacklevel=3,
    )


def base_rates_differ(
    work: pd.DataFrame,
    usable: List[str],
    y_true: Optional[np.ndarray],
    threshold: float = 0.05,
    min_group_labelled: int = 10,
) -> Optional[bool]:
    """True when any protected attribute's group base rates (ground-truth
    positive rates) spread by >= threshold.

    Three states, and False is the narrowest of them: it is returned ONLY
    when at least two groups of at least ``min_group_labelled`` labelled
    rows were actually compared and the spread came in under the
    threshold. None means the question could not be answered: no ground
    truth, no protected column in the frame, no group large enough, every
    label missing, or an internal error. Returning False in those cases
    published "no material base-rate difference was confirmed on this
    data" (recommend.py) for data whose real spread was 1.00, and
    suppressed the calibration / error-rate impossibility acknowledgment
    that a genuine divergence triggers.

    A group's base rate is taken over its LABELLED rows only; a single
    missing label used to poison the whole group's mean to NaN, and a NaN
    spread compares False against every threshold, so a real divergence
    read as "no difference".
    """
    try:
        if y_true is None:
            _warn_base_rates("no ground-truth column was supplied")
            return None
        yt = np.asarray(y_true, dtype=float)
        compared_any = False
        for attr in usable or []:
            if attr not in work.columns:
                continue
            g = work[attr].astype("string").fillna("missing").to_numpy()
            n = min(len(g), len(yt))
            if n == 0:
                continue
            ytn = yt[:n]
            gn = g[:n]
            rates: List[float] = []
            for lab in pd.unique(gn):
                vals = ytn[gn == lab]
                vals = vals[np.isfinite(vals)]
                if len(vals) < min_group_labelled:
                    continue
                rates.append(float(vals.mean()))
            if len(rates) < 2:
                continue
            compared_any = True
            if (max(rates) - min(rates)) >= threshold:
                return True
        if not compared_any:
            _warn_base_rates(
                "no protected attribute had two groups with at least "
                f"{int(min_group_labelled)} labelled rows each"
            )
            return None
        return False
    except Exception:  # advisory flag, never fatal
        _warn_base_rates("the base-rate comparison raised")
        return None


# ───────────────────────────────────────────────────────────────────────────
# Probe / non-binary attachment (thin, mutating, never raises)
# ───────────────────────────────────────────────────────────────────────────


def attach_lightweight_regulatory(
    result: Any,
    inputs: Dict[str, Any],
    domain: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> Any:
    """Attach the pathway-agnostic regulatory pieces to a probe or
    non-binary result: recheck + regulatoryExports.legalContextAsOf
    always; legal admissibility and historical context additionally when
    the caller supplies columns / domain (the non-binary tabular path).
    Mutates result['data'] in place. An attachment failure is NEVER fatal, and it is
    never silent either: it is recorded in
    ``result['data']['regulatoryExports']['attachmentIncomplete']``.

    WHY THE RECORD, when the swallow itself is deliberate and right. This block adds
    regulatory context (LL144 impact ratios, EU AI Act Article 10 sections, historical
    context) to a result that is already valid without it, so raising here would fail a
    whole Pulse run over an additive step. But `except Exception: pass` left no
    difference between two cases a reader has to tell apart:

      * the caller supplied no `columns`/`domain`, so admissibility and historical
        context were NOT REQUESTED and their absence means nothing, and
      * the attachment was requested and RAISED, so their absence means the
        regulatory view of this run is incomplete.

    On a compliance surface those are opposite readings, and the second one silently
    looked like the first.
    """
    try:
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, dict):
            return result
        exports = data.get("regulatoryExports")
        if not isinstance(exports, dict):
            exports = {}
            data["regulatoryExports"] = exports
        exports.setdefault("legalContextAsOf", legal_context_as_of())
        if "recheck" not in data:
            # BGL5 A-operations-3, 2026-09-27. This hardcoded `False`, and False
            # means "the statute screen ran and nothing matched". Nothing on this
            # pathway runs an LL144 screen at all: this is the probe / non-binary
            # attachment, which never builds an ll144 block. Measured before:
            # attach_lightweight_regulatory({"data": {}}, {"domain": "hiring"})
            # wrote basis "No annual audit statute matched this run, so the
            # 6-month window is Pulse's default hygiene interval", a finding about
            # the law that no code here made. After: "Whether an annual audit
            # statute applies could not be determined on this run ... this is NOT
            # a finding that none applies". The 6-month window is unchanged. This
            # is also the only production call site inside this module, so the
            # None branch is now reachable from one.
            screen = _ll144_screen_state(exports)
            data["recheck"] = build_recheck(inputs or {}, screen)
        if columns is not None and "legalAdmissibility" not in data:
            data["legalAdmissibility"] = legal_admissibility(
                list(columns), domain or "", jurisdiction or ""
            )
        if domain is not None and "historicalContext" not in data:
            data["historicalContext"] = historical_context(domain)
    except Exception as exc:  # attachment is additive, never fatal, never silent
        try:
            data = result.get("data") if isinstance(result, dict) else None
            if isinstance(data, dict):
                exports = data.get("regulatoryExports")
                if not isinstance(exports, dict):
                    exports = {}
                    data["regulatoryExports"] = exports
                exports["attachmentIncomplete"] = (
                    f"{type(exc).__name__}: the regulatory attachment did not complete, "
                    f"so any regulatory field missing below was NOT evaluated rather "
                    f"than found not to apply"
                )
        except Exception:
            # The recorder itself must not turn an additive step into a failure.
            pass
        warnings.warn(
            "attach_lightweight_regulatory: the regulatory attachment raised "
            f"{type(exc).__name__} and was not completed. The result is still valid, "
            "but its regulatory exports are INCOMPLETE, which is not the same as a "
            "run to which no regulation applies.",
            stacklevel=2,
        )
    return result
