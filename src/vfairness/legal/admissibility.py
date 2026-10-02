"""Legal-admissibility classifier.

Reads curated JSON rule packs from ``data/`` and classifies dataset columns
against a (use_case x jurisdiction) pair. Universal at runtime: always
returns a block, even when the pack is missing -- the block reports its
own coverage state (``covered`` / ``partial`` / ``uncovered``) so the
caller can render an honest finding.

Source of truth: the curated JSON packs. No LLM is consulted at runtime.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_DATA_DIR = Path(__file__).resolve().parent / "data"
_RULES_DIR = _DATA_DIR / "rules"

LEGAL_DATA_REVISION = "2026-05-20"

#: What an unnamed column is called in ``unmappedColumns``. A fixed label, not
#: ``str(value)``: the doors of absence print as "None", "nan" and "<NA>", and a
#: column named after the absence of a name has no place in a legal block.
_UNNAMED_COLUMN = "(unnamed column)"

# Order matters: most-restrictive verdict among matching packs wins.
#
# R-9, 2026-09-09. ``unknown`` (the column mapped to a sensitive attribute but
# no rule pack has an entry for it, so its legal status was never determined)
# was not in this table and fell to the default rank 0, BELOW ``allowed``. In
# a list sorted most-severe-first the columns nobody checked the law for sat
# beneath the one confirmed legally fine, and none of the summary counters
# mentioned them. Unresolved is more urgent than resolved-fine: ``unknown`` is
# a fourth state that sorts above ``allowed`` and has its own counter. A
# status word outside this vocabulary is likewise undetermined and ranks with
# ``unknown``; it is never ranked as a determined verdict.
_STATUS_PRIORITY = {
    "forbidden": 5,
    "restricted": 4,
    "monitoring_only": 3,
    "unknown": 2,
    "allowed": 1,
    "info": 0,
}
_KNOWN_STATUSES = frozenset(_STATUS_PRIORITY)


@lru_cache(maxsize=1)
def _load_taxonomy() -> Dict[str, Any]:
    with open(_DATA_DIR / "taxonomy.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _load_use_cases() -> Dict[str, Any]:
    with open(_DATA_DIR / "use_cases.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=64)
def load_rules(use_case: str, jurisdiction: str) -> Optional[Dict[str, Any]]:
    """Load a rule pack; resolves `inherits:` chains. Returns None if absent.

    Raises:
        ValueError: If the `inherits:` chain contains a cycle.
    """
    return _load_rules_chain(use_case, jurisdiction, ())


def _load_rules_chain(
    use_case: str,
    jurisdiction: str,
    visited: Tuple[str, ...],
) -> Optional[Dict[str, Any]]:
    """Resolve one pack plus its parents, tracking visited pack ids.

    The visited tuple exists to fail loudly on cyclic `inherits:` references
    in curated packs, which previously recursed until RecursionError.
    """
    pack_id = f"{use_case}.{jurisdiction}"
    if pack_id in visited:
        chain = " -> ".join((*visited, pack_id))
        raise ValueError(f"Cyclic 'inherits' chain in legal rule packs: {chain}")
    path = _RULES_DIR / f"{pack_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        pack = json.load(fh)
    parent_ref = pack.get("inherits")
    if parent_ref:
        try:
            p_use, p_jur = parent_ref.split(".", 1)
        except ValueError:
            p_use, p_jur = use_case, parent_ref
        parent = _load_rules_chain(p_use, p_jur, (*visited, pack_id)) or {}
        merged_rules = dict((parent.get("rules") or {}))
        merged_rules.update(pack.get("rules") or {})
        pack = dict(pack)
        pack["rules"] = merged_rules
    return pack


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _absent_free_text(value: Any) -> bool:
    """True when a free-text field carries NO value, through any of its doors.

    ``if not domain`` answers None and the empty string; the other four doors
    reach ``_normalise`` and die there. Measured 2026-09-30:
    ``map_domain_to_use_case(float("nan"))`` raised ``AttributeError: 'float'
    object has no attribute 'lower'``, ``map_jurisdiction(pd.NA)`` raised
    ``TypeError: boolean value of NA is ambiguous`` from the ``if not`` itself,
    and an int or a bool raised the same AttributeError. A missing cell in a CSV
    or a nullable pandas column is exactly how a domain and a jurisdiction
    arrive, so all four are ordinary input.

    THE ANSWER IS THE ONE ``None`` ALREADY GETS, not a new policy: an absent
    domain maps to the ``generic`` use case and an absent jurisdiction maps to
    ``None``, which ``classify_columns`` already renders as coverage
    "uncovered". Nothing that worked changes; four crashes become the existing
    honest answer. A non-string that is NOT absent (an int, a bool) is also
    treated as no value rather than stringified, because ``str(123)`` would be
    matched against the keyword lists as though someone had typed it.
    """
    if isinstance(value, str):
        return not value.strip()
    if value is None:
        return True
    try:
        import pandas as _pd

        if bool(_pd.isna(value)):
            return True
    except (TypeError, ValueError, ImportError):
        pass
    # Anything else is not free text at all.
    return True


def map_domain_to_use_case(domain: Optional[str]) -> str:
    """Map a Pulse `domain` string to a known use_case id, or `generic`.

    An absent domain, through any of its doors, is the `generic` use case. See
    :func:`_absent_free_text`.
    """
    if _absent_free_text(domain):
        return "generic"
    # _absent_free_text returns False only for a non-blank str, so domain is a
    # str here. The assert states that for the type checker; it cannot fire.
    assert isinstance(domain, str)
    norm = _normalise(domain)
    for entry in _load_use_cases()["use_cases"]:
        for kw in entry.get("domain_keywords", []):
            if kw and kw in norm:
                return entry["id"]
    return "generic"


def map_jurisdiction(jurisdiction: Optional[str]) -> Optional[str]:
    """Map a free-text jurisdiction to a known id (us-federal/eu/de/...).

    An absent jurisdiction, through any of its doors, is None, which
    :func:`classify_columns` renders as coverage "uncovered". See
    :func:`_absent_free_text`.
    """
    if _absent_free_text(jurisdiction):
        return None
    # _absent_free_text returns False only for a non-blank str, so jurisdiction
    # is a str here. The assert states that for the type checker; it cannot fire.
    assert isinstance(jurisdiction, str)
    norm = _normalise(jurisdiction)
    keywords: Dict[str, List[str]] = _load_use_cases().get("jurisdiction_keywords", {})
    # Prefer longer, more specific keywords first (e.g. "us-ca" before "us").
    flat: List[Tuple[str, str]] = sorted(
        ((jid, kw) for jid, kws in keywords.items() for kw in kws),
        key=lambda t: -len(t[1]),
    )
    for jid, kw in flat:
        if _normalise(kw) in norm:
            return jid
    return None


def _match_column_to_attribute(col_name: str) -> Optional[str]:
    """Return canonical attribute id for a column name, or None."""
    n = _normalise(col_name)
    attrs = _load_taxonomy()["attributes"]
    # Exact alias hit first
    for attr in attrs:
        for alias in attr.get("aliases", []):
            if n == _normalise(alias):
                return attr["id"]
    # Substring hit on alias (for compound names like "applicant_dob")
    for attr in attrs:
        for alias in attr.get("aliases", []):
            an = _normalise(alias)
            if len(an) >= 3 and (an in n or n in an):
                return attr["id"]
    return None


def classify_columns(
    columns: Sequence[str],
    use_case: Optional[str],
    jurisdiction: Optional[str],
) -> Dict[str, Any]:
    """Classify a column list against `(use_case, jurisdiction)`.

    Returns a JSON-serialisable block with shape::

        {
          coverage: "covered" | "partial" | "uncovered",
          useCase: str,
          jurisdiction: str | None,
          jurisdictionId: str | None,
          sourceRevision: str,
          findings: [
            {column, attributeId, attributeLabel, status, legalBasis, carveOut?, notes?, lifecycle?}
          ],
          summary: {forbidden, restricted, monitoringOnly, allowed, unknown},
          unmappedColumns: [str, ...]
        }

    ``summary.unknown`` counts findings whose legal status was NOT determined
    (no rule-pack entry for the attribute in the active jurisdiction, or a
    status word outside the pack vocabulary). It is a could-not-check count,
    never part of ``allowed``, and the findings it counts sort above
    ``allowed`` because an unresolved status is more urgent than a resolved
    clean one.
    """
    # A BARE STRING IS A Sequence[str], AND NEVER THE INTENT. Measured
    # 2026-09-30: ``classify_columns("race", "recruitment", "us")`` iterated the
    # CHARACTERS and reported four findings over the columns 'r', 'a', 'c' and
    # 'e', each counted into ``summary.unknown``, so a caller who passed one
    # column name instead of a list got a legal block about four columns that do
    # not exist. The annotation cannot catch it; this can.
    if isinstance(columns, str):
        raise TypeError(
            "classify_columns(columns=...) takes a SEQUENCE of column names and was given a "
            f"single string {columns!r}. Iterating it would classify its characters as "
            f"{len(columns)} separate columns. Pass [{columns!r}]."
        )
    use_case = use_case if isinstance(use_case, str) and use_case.strip() else "generic"
    jurisdiction_id = map_jurisdiction(jurisdiction)

    primary_pack: Optional[Dict[str, Any]] = None
    fallback_pack: Optional[Dict[str, Any]] = None
    coverage = "uncovered"
    if jurisdiction_id:
        primary_pack = load_rules(use_case, jurisdiction_id)
        fallback_pack = load_rules("generic", jurisdiction_id)
        if primary_pack:
            coverage = "covered"
        elif fallback_pack:
            coverage = "partial"

    rules_primary: Dict[str, Any] = (primary_pack or {}).get("rules", {}) if primary_pack else {}
    rules_fallback: Dict[str, Any] = (fallback_pack or {}).get("rules", {}) if fallback_pack else {}

    taxonomy_map = {a["id"]: a for a in _load_taxonomy()["attributes"]}

    findings: List[Dict[str, Any]] = []
    unmapped: List[str] = []
    counts = {"forbidden": 0, "restricted": 0, "monitoringOnly": 0, "allowed": 0, "unknown": 0}

    for col in columns:
        # A column name that IS the absence of a name cannot be matched against
        # the taxonomy, and `_normalise(None)` raised AttributeError from inside
        # the matcher, so one missing header took the whole legal block down.
        # It is reported as unmapped under a fixed placeholder rather than
        # stringified: `str(None)` would put a column called "None" in the block
        # a lawyer reads. Same six doors as everywhere else in this library.
        if _absent_free_text(col):
            unmapped.append(_UNNAMED_COLUMN)
            continue
        attr_id = _match_column_to_attribute(col)
        if attr_id is None:
            unmapped.append(col)
            continue
        rule = rules_primary.get(attr_id) or rules_fallback.get(attr_id)
        if not rule:
            # Mapped to an attribute but no rule pack opinion -- emit a soft
            # finding so the user still sees that a sensitive-looking column
            # exists.
            attr_label = taxonomy_map.get(attr_id, {}).get("label", attr_id)
            findings.append(
                {
                    "column": col,
                    "attributeId": attr_id,
                    "attributeLabel": attr_label,
                    "status": "unknown",
                    "legalBasis": None,
                    "notes": "No rule pack entry for this attribute in the active jurisdiction.",
                }
            )
            # R-9. Counted, never silently absent: this column's legal status
            # is undetermined, which is not the same as allowed.
            counts["unknown"] += 1
            continue
        attr_label = taxonomy_map.get(attr_id, {}).get("label", attr_id)
        status = rule.get("status", "unknown")
        finding = {
            "column": col,
            "attributeId": attr_id,
            "attributeLabel": attr_label,
            "status": status,
            "legalBasis": rule.get("legal_basis"),
        }
        if rule.get("carve_out"):
            finding["carveOut"] = rule["carve_out"]
        if rule.get("lifecycle"):
            finding["lifecycle"] = rule["lifecycle"]
        if rule.get("notes"):
            finding["notes"] = rule["notes"]
        findings.append(finding)
        if status == "forbidden":
            counts["forbidden"] += 1
        elif status == "restricted":
            counts["restricted"] += 1
        elif status == "monitoring_only":
            counts["monitoringOnly"] += 1
        elif status == "allowed":
            counts["allowed"] += 1
        elif status == "unknown" or status not in _KNOWN_STATUSES:
            # R-9. A pack entry without a recognised status determined nothing.
            counts["unknown"] += 1

    # Stable ordering: most-severe first, then column name. An undetermined
    # status ranks as ``unknown`` (above ``allowed``), never as rank 0. R-9.
    findings.sort(
        key=lambda f: (
            -_STATUS_PRIORITY.get(f.get("status", ""), _STATUS_PRIORITY["unknown"]),
            f["column"],
        )
    )

    return {
        "coverage": coverage,
        "useCase": use_case,
        "jurisdiction": jurisdiction,
        "jurisdictionId": jurisdiction_id,
        "sourceRevision": (primary_pack or fallback_pack or {}).get(
            "source_revision", LEGAL_DATA_REVISION
        ),
        "findings": findings,
        "summary": counts,
        "unmappedColumns": unmapped,
    }
