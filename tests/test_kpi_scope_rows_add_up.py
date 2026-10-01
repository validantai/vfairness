"""Every published scope row must sum to its own total.

FOUND BY A READER ADDING UP THE ROW (2026-09-28). The quality page publishes the
capability ledger split into Core and Preview, four numbers each. Core had a
``core_fix_pending`` key and Preview had no equivalent, so the published Preview row
read 182 total against 84 checked plus 90 not checked, which is 174. The eight missing
units are the preview units with a KNOWN OPEN DEFECT, and the cell where they belong
rendered a dash, which a reader takes as none.

Nothing checked it. The staleness gate compares the published block field for field
against what it can measure, so it was perfectly happy: the fields that existed were
all correct. A missing field is invisible to a field-by-field comparison, which is why
this file asks a different question, about ARITHMETIC rather than freshness.

Three questions, because each catches a different way of getting it wrong:
  1. each scope's parts sum to its own total;
  2. the scopes sum to the ledger totals, so no unit belongs to neither;
  3. every number the page renders for a scope has a filler behind it, so a hardcoded
     literal cannot sit where a measurement should be.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "docs" / "site" / "data" / "library-stats.json"
PAGE = ROOT / "docs" / "site" / "quality-and-hardening" / "index.html"
SCOPES = ("core", "preview")
PARTS = ("checked", "fix_pending", "not_checked")


@pytest.fixture(scope="module")
def ledger() -> dict:
    led = (json.loads(STATS.read_text(encoding="utf-8")).get("kpis") or {}).get("ledger")
    assert led, "the published statistics carry no ledger block"
    assert led.get("available") is True, led
    return led


@pytest.mark.parametrize("scope", SCOPES)
def test_every_part_of_a_scope_is_published(ledger, scope):
    """A missing key is the defect. It reads as nothing rather than as unknown."""
    missing = [p for p in PARTS if f"{scope}_{p}" not in ledger]
    assert not missing, (
        f"the {scope} row publishes no {missing}, so the page has nothing to render "
        "there and a reader sees a dash where a count belongs"
    )


@pytest.mark.parametrize("scope", SCOPES)
def test_a_scope_sums_to_its_own_total(ledger, scope):
    parts = {p: int(ledger[f"{scope}_{p}"]) for p in PARTS}
    total = int(ledger[f"{scope}_total"])
    assert sum(parts.values()) == total, (
        f"the published {scope} row does not add up: {parts} sums to "
        f"{sum(parts.values())} against a total of {total}. The difference is units "
        "that are in the total and in no column."
    )


def test_the_scopes_sum_to_the_ledger(ledger):
    """No unit may belong to neither scope, or the split hides it."""
    for part in PARTS:
        whole = int(ledger[part.replace("fix_pending", "fix_pending")])
        by_scope = sum(int(ledger[f"{s}_{part}"]) for s in SCOPES)
        assert by_scope == whole, (
            f"{part}: the scopes publish {by_scope} against {whole} for the whole "
            "ledger, so units are in the ledger and in neither scope"
        )
    assert sum(int(ledger[f"{s}_total"]) for s in SCOPES) == int(ledger["total"])


@pytest.mark.parametrize("scope", SCOPES)
def test_every_scope_cell_on_the_page_is_filled_from_the_ledger(scope):
    """A hardcoded literal in a measured cell is the shape of the original defect.

    The page carries an inline value as its no-JavaScript fallback and a ``fill``
    call that replaces it, so a cell with a span and no fill is a number that can go
    stale silently, and a cell with neither is a number nobody publishes.
    """
    page = PAGE.read_text(encoding="utf-8")
    for part in PARTS:
        # The page's class names are not the ledger's key names, so the mapping is
        # written down rather than derived: "not_checked" is rendered as "unchecked".
        suffix = {"checked": "checked", "fix_pending": "fix", "not_checked": "unchecked"}[part]
        css = f"kpi-{scope}-{suffix}"
        assert f'class="{css}"' in page, f"the page has no {css} cell"
        assert re.search(rf"fill\('\.{re.escape(css)}'", page), (
            f"{css} is rendered but never filled from the ledger, so it can only be "
            "as current as the day someone typed it"
        )


# ---------------------------------------------------------------------------
# The general form, asked of every published group rather than the one that broke
# ---------------------------------------------------------------------------
#
# The Preview row was found by a READER adding it up, and the specific checks above
# would not have caught the next one somewhere else. These are the relations that
# must hold across the whole published statistics file, expressed once.

#: ``group: (total_key, part_keys)``. A group is listed here BY HAND on purpose. A
#: sweep that guessed which keys are parts of which total would either miss a group
#: or invent a relation, and a gate that invents its own subject is worse than none.
#: Adding a published count means adding it here, and the completeness test below
#: refuses an integer key that belongs to no relation and is not excused.
_RELATIONS = {
    "ledger scopes vs whole": ("total", ("core_total", "preview_total")),
    "ledger core": ("core_total", ("core_checked", "core_fix_pending", "core_not_checked")),
    "ledger preview": (
        "preview_total",
        ("preview_checked", "preview_fix_pending", "preview_not_checked"),
    ),
    "ledger whole": ("total", ("checked", "fix_pending", "not_checked")),
    # The assessment path is a SUBSET of core, not a scope beside it, so it has no
    # relation to the whole. Its own three parts still have to account for its total,
    # which is the check that caught "none carries an open defect" sitting beside a
    # measured 13.
    "ledger assessment path": (
        "path_total",
        ("path_checked", "path_fix_pending", "path_not_checked"),
    ),
}

#: Integer keys in the ledger that are DERIVED or descriptive rather than a part of
#: a total, each with the reason. Anything not here and not in a relation above fails
#: the completeness test, so a new count cannot be published unchecked.
_NOT_A_PART = {
    "available": "a flag, not a count",
    "core_to_close": "a convenience sum of two parts already checked above",
    "preview_to_close": "the same, for preview",
}


@pytest.mark.parametrize("name", sorted(_RELATIONS))
def test_every_published_relation_holds(ledger, name):
    total_key, part_keys = _RELATIONS[name]
    missing = [k for k in (total_key, *part_keys) if k not in ledger]
    assert not missing, f"{name}: the published ledger has no {missing}"
    parts = {k: int(ledger[k]) for k in part_keys}
    total = int(ledger[total_key])
    assert sum(parts.values()) == total, (
        f"{name} does not add up: {parts} sums to {sum(parts.values())} against "
        f"{total_key}={total}. The difference is units in the total and in no part."
    )


def test_no_published_count_sits_outside_every_relation():
    """A published integer nobody checks is how the Preview row went wrong.

    The eight missing units were not a WRONG number, they were an ABSENT one, and a
    field-by-field staleness comparison cannot see an absent field. This test comes
    at it from the other side: every integer the ledger publishes must be part of a
    checked relation or be excused by name, so adding a count without a check fails.
    """
    led = (json.loads(STATS.read_text(encoding="utf-8")).get("kpis") or {}).get("ledger") or {}
    integers = {k for k, v in led.items() if isinstance(v, int) and not isinstance(v, bool)}
    integers |= {k for k, v in led.items() if isinstance(v, bool)}
    covered = {k for total, parts in _RELATIONS.values() for k in (total, *parts)}
    stray = sorted(integers - covered - set(_NOT_A_PART))
    assert not stray, (
        f"the ledger publishes {stray} and no relation above checks them. Add each to "
        "_RELATIONS if it is part of a total, or to _NOT_A_PART with the reason."
    )
