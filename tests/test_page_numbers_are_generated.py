"""Every figure on the hardening page is generated, or it is written down WHY not.

WHY THIS EXISTS. Four separate figures on that page were measured wrong on
2026-09-29, and none of them could have been caught by any check the page had:

  * ``139 of the 383 test files`` and ``1,679 of the 6,735 test-function
    definitions``, in three different paragraphs, while scripts/library_kpis.py
    was measuring 246 of 508 and 2,422 of 8,782 and publishing them into the very
    file the page fetches on load.
  * A beta-criterion table listing FIVE criteria when the gate had seven, giving
    B1 as "1564 of 1564", and reporting B2 as a FAIL with "64 open (37 measuring,
    27 unclassified)" long after B2 reached zero open and started passing.
  * ``Of those, 251 have been examined and are clean, none carries an open
    defect``, about the assessment path, while the ledger counted 13 with a known
    open defect.
  * A whole second scope table, eight cells of it, typed as plain digits one
    screen below a table that IS checked.

The kpi- contract is what makes a figure live, and tests/test_kpi_spans_are_live.py
already refuses a kpi- class that nothing fills. This test comes at it from the
other side, which is where all four escaped: a number with no kpi- class at all is
invisible to that contract. Here, every number of three digits or more that the
page shows a reader must either carry a kpi- class, sit inside a generated block,
or be named below with the reason it is typed.

The reasons are nearly all the same one and it is a good one: a dated historical
measurement is not a live figure and must NOT move. "27 of 120 dismissals were
overturned" is an account of something that happened; rewiring it to today's
numbers would destroy the record. The point of this list is that each of those is
a decision somebody made, rather than a figure nobody noticed had rotted.
"""

from __future__ import annotations

import collections
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "site" / "quality-and-hardening" / "index.html"

#: Blocks scripts/bgl6_register.py regenerates from the gate and the register. Their
#: figures are measured at build time and guarded by
#: tests/test_bgl6_register_is_honest.py, so they are not typed by anybody.
GENERATED = ("BGL6:register", "BGL6:readiness", "BGL6:plan", "STATUS:site", "READINESS")
#: STATUS:site is written by scripts/stamp_status_badges.py from the ledger and from
#: docs/gate-checks.json, and `stamp_status_badges.py --check` (run by
#: scripts/refresh-docs.sh) refuses the page when that block differs from what the
#: script would write now. Added 2026-10-01: the scope table and the beta-gate table
#: went into that block, and this test read their computed cells as typed figures.
#: READINESS is written by scripts/render_readiness.py from the same sources plus the
#: dated docs/readiness-history.json, and its --check (also in refresh-docs.sh)
#: refuses the page when the block differs.

#: number as it appears on the page -> why it is typed rather than generated.
#: A number not in here and not carrying a kpi- class fails this test.
TYPED_NUMBERS = {
    "2026": "the year, in dates",
    "2017": "a citation year, Kull et al. beta calibration",
    "100": "the total row's share, which is 100% by construction",
    "000": "the fractional half of ECE = 0.000 in a quoted defect report",
    "200": "a documented threshold in the metric description, n=200 for Fisher exact",
    "228": "mypy's source-file count on the dated CI run quoted beside it",
    "215": "the capability census, fixed at the 2026-09-11 beta gate",
    "242": "the deep audit's finding count, a closed historical set",
    "132": "coverage-untouched units as measured on 2026-09-25, stated with its date",
    "128": "a sort of the suite by file-name prefix on a stated date, not a measurement",
    "193": "the same sort under the wider rule, stated beside it",
    "124": "the before figure of a fall from 124 to 35, a historical pair",
    "1,546": "what the next census will count, a forward statement",
    "204": "the module count in that same forward statement",
    "129": "grades in a re-run batch, a closed historical batch",
    "120": "dismissals reviewed in a closed historical batch",
    "904": "remaining units in a dated cost estimate",
    "69,000": "tokens per unit in that cost estimate",
    "138,000": "tokens for the two-reviewer option in that estimate",
    "16,000": "tokens for the batched option in that estimate",
    "53,000": "the measured token cost the estimate is compared against",
    "167": "units the surface probe could not call, from its dated run",
    "300": "an explicitly labelled guess at the suspect count",
    "1,049": "units in the surface probe's dated run",
    "882": "units the probe reached in that run",
    "138": "units found fabricating in the waves so far, a closed set",
    "420": "units examined in those waves",
    "105": "original_size in a quoted defect report",
    "180": "sites found by a scan on a dated commit",
    "831": "the suite size before a stated change",
    "1,280": "the suite size after it",
    "500": "mypy errors before the burndown, a historical pair with zero",
    "4,354": "tests passing on a named commit",
    "1,869": "a quoted pytest line",
    "1,930": "the second quoted pytest line it is compared with",
}


def _reader_text() -> str:
    """What a reader sees, with everything generated or live taken out."""
    s = PAGE.read_text(encoding="utf-8")
    for name in GENERATED:
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        if start in s and end in s:
            s = s[: s.index(start)] + s[s.index(end) + len(end) :]
    s = re.sub(r"<script\b.*?</script\s*>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<style\b.*?</style\s*>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    # Any ELEMENT carrying a kpi- class, contents included: those are filled on load.
    s = re.sub(r'<(\w+)\b[^>]*class="[^"]*\bkpi-[\w-]+[^"]*"[^>]*>.*?</\1>', " ", s, flags=re.S)
    return re.sub(r"<[^>]+>", " ", s)


def _typed() -> collections.Counter:
    return collections.Counter(re.findall(r"\b\d[\d,]{2,}\b", _reader_text()))


def test_every_typed_figure_is_declared():
    """A number with no kpi- class is one nobody can catch going stale."""
    found = _typed()
    undeclared = sorted(n for n in found if n not in TYPED_NUMBERS)
    text = _reader_text()
    detail = []
    for n in undeclared:
        i = text.index(n)
        detail.append(f"  {n}: ...{re.sub(r'[ ]+', ' ', text[max(0, i - 70) : i + 40]).strip()}")
    assert not undeclared, (
        "these figures are typed into the page and nothing keeps them true:\n"
        + "\n".join(detail)
        + "\n\nEither give the figure a kpi- class and fill it in the page script, or "
        "add it to TYPED_NUMBERS with the reason it must NOT move."
    )


@pytest.mark.parametrize("number", sorted(TYPED_NUMBERS))
def test_every_declared_figure_carries_a_reason(number):
    reason = TYPED_NUMBERS[number]
    assert reason and len(reason) > 15, f"{number} is excused without a reason"


def test_the_declared_list_has_not_outlived_the_page():
    """An allowlist that only grows stops being a record of decisions.

    A number nobody types any more should leave this list, or the next reader
    cannot tell which entries are live decisions and which are debris.
    """
    found = _typed()
    stale = sorted(n for n in TYPED_NUMBERS if n not in found)
    assert not stale, (
        f"TYPED_NUMBERS excuses figures the page no longer shows: {stale}. "
        "Remove them, so the list keeps meaning what it says."
    )
