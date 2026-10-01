"""Everything that says "not released yet" must flip in the SAME change as the tag.

THE DEFECT THIS EXISTS FOR. ``README.md`` is the PyPI ``long_description``
(``pyproject.toml`` sets ``readme = "README.md"``). Today it carries a
"not yet published" status badge, a blockquote saying 0.1.0 "has **not** been
released", an Installation section opening "There is no install command that
works today", and a Beta section saying it "has not shipped". Every one of those
is TRUE right now and becomes FALSE the moment the artifact exists. **A PyPI
version's description is immutable**, so publishing as-is leaves the 0.1.0
project page telling every visitor, permanently, that 0.1.0 was never published.
``CITATION.cff`` has the same shape (an absent ``date-released``, deliberately),
and so does every published page that says nothing has been cut.

WHY A TEST AND NOT A CHECKLIST. The flip was already written down, in the private
runbook's step 0, with the stakes explained. It was still a manual step that
nothing executed: no test, no workflow step, and no check in the export script.
A step that is only remembered is a step that is eventually forgotten, and this
one cannot be corrected afterwards.

HOW IT WORKS. The release state is READ, not configured: it comes from the top
``## [X.Y.Z] - ...`` heading of ``CHANGELOG.md``, which is undated while the
version is unreleased and dated in the change that pushes the tag. Everything
else is then required to agree with it, in BOTH directions:

* dating the changelog alone turns this red until ``README.md``,
  ``CITATION.cff`` and the published pages follow;
* stripping the honest warnings BEFORE the tag turns it red too, so the library
  cannot start claiming to be published while it is not.

The second direction matters as much as the first. The failure this project has
actually shipped is a page claiming more than the evidence supports.

WHAT IT DOES NOT ESTABLISH. It does not check PyPI, and it does not read the
built wheel. It compares the repository's own files to the repository's own
declared state. A release cut with every file flipped and no tag pushed would
pass here and fail at the workflow's tag-versus-``__version__`` assert instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
README = LIB_ROOT / "README.md"
CHANGELOG = LIB_ROOT / "CHANGELOG.md"
CITATION = LIB_ROOT / "CITATION.cff"
PYPROJECT = LIB_ROOT / "pyproject.toml"
SITE = LIB_ROOT / "docs" / "site"

UNRELEASED = "unreleased"
RELEASED = "released"

# The top changelog heading, e.g. `## [0.1.0] - UNRELEASED` or `## [0.1.0] - 2026-09-04`.
_TOP_HEADING_RE = re.compile(r"^## \[([^\]]+)\]\s*-\s*(.+?)\s*$", re.MULTILINE)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# ---------------------------------------------------------------------------
# The claims that have to flip. These are LITERALS lifted from the files as they
# stand while 0.1.0 is unreleased, not paraphrases, so "present" and "absent" are
# both decidable and a reword has to come here too.
# ---------------------------------------------------------------------------

README_UNRELEASED_CLAIMS = (
    "status-0.1.0%20beta%2C%20not%20yet%20published",  # the shields.io badge
    "### Not yet published",
    "is prepared but has **not** been released",
    "**There is no install command that works today.**",
    "has not shipped",
    "nothing from this codebase has reached PyPI or TestPyPI",
)

CHANGELOG_UNRELEASED_CLAIMS = ("Nothing has been cut and nothing has been published.",)

# Pages a reader meets first. Each must carry the state while it is true, so the
# warning cannot be quietly dropped from the surfaces that matter most.
PAGES_THAT_MUST_STATE_THE_STATE = (
    "index.html",
    "getting-started/index.html",
    "quality-and-hardening/index.html",
    "api-reference/index.html",
)

# Sentences that are false the moment ANYTHING is published under this name.
UNCONDITIONAL_UNRELEASED_PHRASES = (
    "Nothing has been released",
    "nothing has been cut",
    "no artifact has reached PyPI",
    "the repository carries no tags",
    "This install command does not work yet",
    "This version is not published yet",
    "There is no install command that works today",
    "never run to a publish",
)


def _searchable(text: str) -> str:
    """The text, plus a percent-decoded copy of it.

    THE BADGE HOLE. The README's status badge carries its words inside a
    shields.io URL, percent-encoded:

        https://img.shields.io/badge/status-0.1.0%20beta%2C%20not%20yet%20published-orange.svg

    The scoped patterns below look for "not yet published" with real spaces, so
    they matched the badge's ALT TEXT and never the URL. Editing the alt text at
    release while leaving the URL would have left a badge reading "0.1.0 beta,
    not yet published" on the PyPI page, which cannot be edited after upload,
    with every gate green.

    Scanning both forms closes it without loosening any pattern: a separator
    class permissive enough to span %20 would also span punctuation and start
    matching sentences nobody wrote.
    """
    return text + "\n" + unquote(text)


def _version_scoped_patterns(version: str) -> Tuple[re.Pattern, ...]:
    """Phrases that are false once THIS version is published.

    Scoped to the version being released so that a page may still say a LATER
    version is unpublished, which is a true statement and not this test's
    business.
    """
    v = re.escape(version)
    return (
        re.compile(rf"{v}.{{0,60}}not yet published", re.IGNORECASE | re.DOTALL),
        re.compile(rf"{v}.{{0,60}}has not been released", re.IGNORECASE | re.DOTALL),
        re.compile(rf"{v}.{{0,60}}has not been cut", re.IGNORECASE | re.DOTALL),
        re.compile(rf"{v}.{{0,60}}is prepared and unreleased", re.IGNORECASE | re.DOTALL),
        re.compile(rf"no {v} artifact exists", re.IGNORECASE),
        re.compile(rf"There is no `?v?{v}`? tag", re.IGNORECASE),
        re.compile(rf"\[{v}\]\s*-\s*UNRELEASED", re.IGNORECASE),
    )


# ---------------------------------------------------------------------------
# Reading the declared state
# ---------------------------------------------------------------------------


class ReleaseState:
    """What the changelog says about the version at the top of it."""

    def __init__(self, version: str, state: str, date: Optional[str], heading: str) -> None:
        self.version = version
        self.state = state
        self.date = date
        self.heading = heading


def read_state(changelog: str) -> Tuple[Optional[ReleaseState], List[str]]:
    """Parse the top heading. Returns (state, problems); never guesses.

    A heading that is neither clearly UNRELEASED nor clearly dated is its own
    third state and is reported, rather than being read as one of the two. That
    is the whole rule this library is built on, applied to its own metadata.
    """
    match = _TOP_HEADING_RE.search(changelog)
    if match is None:
        return None, [
            "CHANGELOG.md has no `## [X.Y.Z] - ...` heading; the release state cannot be read"
        ]
    version, rest = match.group(1), match.group(2)
    heading = match.group(0)
    date = _ISO_DATE_RE.search(rest)
    says_unreleased = "unreleased" in rest.lower()
    if date and says_unreleased:
        return None, [
            f"the top changelog heading is both dated and marked UNRELEASED, so it states two "
            f"different things: {heading!r}"
        ]
    if date:
        return ReleaseState(version, RELEASED, date.group(1), heading), []
    if says_unreleased:
        return ReleaseState(version, UNRELEASED, None, heading), []
    return None, [
        f"the top changelog heading is neither dated nor marked UNRELEASED, so it does not say "
        f"whether {version} was released: {heading!r}"
    ]


def _citation_date(citation: str) -> Optional[str]:
    """The `date-released` value, ignoring the explanatory comment lines."""
    for line in citation.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("date-released:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


def _citation_version(citation: str) -> Optional[str]:
    for line in citation.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("version:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def tense_problems(
    *,
    changelog: str,
    readme: str,
    citation: str,
    pages: Dict[str, str],
) -> List[str]:
    """Every disagreement between the declared release state and the artifacts.

    Pure in its inputs so the controls below can hand it a released changelog
    without a release existing.
    """
    state, problems = read_state(changelog)
    if state is None:
        return problems

    cff_version = _citation_version(citation)
    if cff_version is not None and cff_version != state.version:
        problems.append(
            f"CITATION.cff says version {cff_version!r} and the changelog heading says "
            f"{state.version!r}; a citation would name a version whose notes are elsewhere"
        )

    cff_date = _citation_date(citation)

    if state.state == UNRELEASED:
        for claim in README_UNRELEASED_CLAIMS:
            if claim not in readme:
                problems.append(
                    f"the changelog says {state.version} is UNRELEASED, but README.md no longer "
                    f"carries {claim!r}. README.md is the PyPI long description: while nothing is "
                    "published it must say so, and it may only stop saying so in the change that "
                    "pushes the tag."
                )
        for claim in CHANGELOG_UNRELEASED_CLAIMS:
            if claim not in changelog:
                problems.append(
                    f"the changelog heading says UNRELEASED but the body no longer carries "
                    f"{claim!r}, so the two halves of the same file disagree"
                )
        if cff_date is not None:
            problems.append(
                f"CITATION.cff carries date-released: {cff_date!r} while the changelog says "
                f"{state.version} is UNRELEASED. Academic citations are generated from that "
                "field, so it may only be filled in when the release actually happens."
            )
        for name in PAGES_THAT_MUST_STATE_THE_STATE:
            page = pages.get(name)
            if page is None:
                problems.append(
                    f"docs/site/{name} is missing; its release-state claim cannot be read"
                )
                continue
            if not _states_unreleased(page, state.version):
                problems.append(
                    f"docs/site/{name} no longer says {state.version} is unpublished, while the "
                    "changelog still does. A reader of that page would take the library to be on "
                    "PyPI."
                )
        return problems

    # RELEASED: nothing anywhere may still say the version does not exist.
    scoped = _version_scoped_patterns(state.version)
    surfaces: Dict[str, str] = {"README.md": readme, "CHANGELOG.md": changelog}
    surfaces.update({f"docs/site/{name}": text for name, text in pages.items()})

    for where, text in sorted(surfaces.items()):
        haystack = _searchable(text)
        for phrase in UNCONDITIONAL_UNRELEASED_PHRASES:
            if phrase in haystack:
                problems.append(
                    f"{where} still says {phrase!r}, but the changelog dates {state.version} to "
                    f"{state.date}. That sentence is false the moment anything is published."
                )
        for pattern in scoped:
            found = pattern.search(haystack)
            if found:
                problems.append(
                    f"{where} still says {found.group(0)[:80]!r} about {state.version}, which the "
                    f"changelog dates to {state.date}."
                )

    if cff_date is None:
        problems.append(
            f"CITATION.cff has no date-released while the changelog dates {state.version} to "
            f"{state.date}; a generated citation would carry no publication date"
        )
    elif cff_date != state.date:
        problems.append(
            f"CITATION.cff says date-released: {cff_date!r} and the changelog says {state.date!r}"
        )
    return problems


def _states_unreleased(page: str, version: str) -> bool:
    page = _searchable(page)
    if any(phrase in page for phrase in UNCONDITIONAL_UNRELEASED_PHRASES):
        return True
    return any(pattern.search(page) for pattern in _version_scoped_patterns(version))


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------


def _site_pages() -> Dict[str, str]:
    pages = {
        str(path.relative_to(SITE)): path.read_text(encoding="utf-8")
        for path in sorted(SITE.rglob("*.html"))
    }
    assert pages, f"no HTML under {SITE}; this check would pass by finding nothing"
    return pages


def _real_inputs() -> Dict[str, object]:
    return {
        "changelog": CHANGELOG.read_text(encoding="utf-8"),
        "readme": README.read_text(encoding="utf-8"),
        "citation": CITATION.read_text(encoding="utf-8"),
        "pages": _site_pages(),
    }


def test_the_readme_really_is_the_pypi_long_description() -> None:
    """The premise, pinned first.

    If ``readme`` ever points somewhere else, every README assertion below stops
    being a statement about the PyPI page while still passing.
    """
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert pyproject["project"]["readme"] == "README.md", (
        "pyproject no longer builds the long description from README.md, so this file is "
        "checking the wrong document"
    )


def test_the_release_state_is_readable_and_is_one_of_the_two() -> None:
    """Guard against a vacuous run: an unparseable heading must fail loudly."""
    state, problems = read_state(CHANGELOG.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)
    assert state is not None
    assert state.state in {UNRELEASED, RELEASED}


def test_every_artifact_agrees_with_the_declared_release_state() -> None:
    """The pin. Red if the cut flips one file and forgets another, either way."""
    problems = tense_problems(**_real_inputs())  # type: ignore[arg-type]
    assert not problems, "the release tense does not agree with the changelog:\n\n" + "\n\n".join(
        problems
    )


def test_the_version_is_the_same_number_everywhere() -> None:
    """A tag matching ``__version__`` is asserted in CI; the prose has to match too."""
    import vfairness

    state, problems = read_state(CHANGELOG.read_text(encoding="utf-8"))
    assert not problems and state is not None, problems
    assert state.version == vfairness.__version__, (
        f"CHANGELOG heading is [{state.version}] and vfairness.__version__ is "
        f"{vfairness.__version__}"
    )
    assert _citation_version(CITATION.read_text(encoding="utf-8")) == state.version


# ---------------------------------------------------------------------------
# The rule, checked against both directions
# ---------------------------------------------------------------------------

_RELEASED_CHANGELOG = "# Changelog\n\n## [0.1.0] - 2026-09-04\n\nThings.\n"
_UNRELEASED_CHANGELOG = (
    "# Changelog\n\n## [0.1.0] - UNRELEASED\n\n"
    "Nothing has been cut and nothing has been published.\n"
)
_RELEASED_README = "# vfairness\n\n## Installation\n\n```bash\npip install vfairness\n```\n"
_RELEASED_CITATION = "version: 0.1.0\ndate-released: '2026-09-04'\n"
_UNRELEASED_CITATION = "version: 0.1.0\n# date-released is deliberately ABSENT until the tag.\n"


def _unreleased_readme() -> str:
    """A SYNTHETIC unreleased README, carrying every claim the gate looks for.

    This used to return ``README.read_text()``, the real file, while the class
    docstring below promised that "nothing here depends on the project actually
    being unreleased when it runs". That was false: dating the changelog at the
    cut turned two of these controls red for the wrong reason, because their
    fixture was the very file under flip. The controls exist to prove the RULE
    catches a half-done cut; they must not themselves depend on which half the
    repository is in.

    Built from README_UNRELEASED_CLAIMS so the fixture cannot drift from the
    literals the rule scans for: add a claim there and it appears here.
    """
    return "# vfairness\n\n" + "\n\n".join(README_UNRELEASED_CLAIMS) + "\n"


def _healthy_unreleased_pages() -> Dict[str, str]:
    return {name: "<p>0.1.0 is not yet published.</p>" for name in PAGES_THAT_MUST_STATE_THE_STATE}


class TestTheRuleRefusesEachHalfOfTheFlip:
    """Plant every partial cut and require it to be reported.

    Each of these is the real hazard in miniature: the operator changes one of
    the three files and stops. The controls run on synthetic strings, so no
    repository file is touched and nothing here depends on the project actually
    being unreleased when it runs.
    """

    def test_a_released_changelog_with_the_unreleased_readme_is_caught(self) -> None:
        problems = tense_problems(
            changelog=_RELEASED_CHANGELOG,
            readme=_unreleased_readme(),
            citation=_RELEASED_CITATION,
            pages={},
        )
        assert problems, (
            "dating the changelog while the README says the version does not exist passed"
        )
        assert any("README.md" in p for p in problems), problems

    def test_a_released_changelog_with_no_citation_date_is_caught(self) -> None:
        problems = tense_problems(
            changelog=_RELEASED_CHANGELOG,
            readme=_RELEASED_README,
            citation=_UNRELEASED_CITATION,
            pages={},
        )
        assert any("date-released" in p for p in problems), problems

    def test_a_citation_date_that_disagrees_with_the_changelog_is_caught(self) -> None:
        problems = tense_problems(
            changelog=_RELEASED_CHANGELOG,
            readme=_RELEASED_README,
            citation="version: 0.1.0\ndate-released: '2026-01-01'\n",
            pages={},
        )
        assert any("2026-01-01" in p for p in problems), problems

    def test_a_published_page_left_in_the_unreleased_tense_is_caught(self) -> None:
        problems = tense_problems(
            changelog=_RELEASED_CHANGELOG,
            readme=_RELEASED_README,
            citation=_RELEASED_CITATION,
            pages={"index.html": "<p>Nothing has been released. No tag.</p>"},
        )
        assert any("index.html" in p for p in problems), problems

    def test_the_footer_label_is_caught_too(self) -> None:
        # The version-scoped pattern, on the exact footer text 18 pages carry.
        problems = tense_problems(
            changelog=_RELEASED_CHANGELOG,
            readme=_RELEASED_README,
            citation=_RELEASED_CITATION,
            pages={"legal/index.html": '<a href="x">PyPI (0.1.0 not yet published)</a>'},
        )
        assert any("legal/index.html" in p for p in problems), problems

    def test_an_early_readme_flip_is_caught(self) -> None:
        """The other direction: claiming publication before the tag exists."""
        problems = tense_problems(
            changelog=_UNRELEASED_CHANGELOG,
            readme=_RELEASED_README,
            citation=_UNRELEASED_CITATION,
            pages=_healthy_unreleased_pages(),
        )
        assert problems, "stripping the README's unreleased warnings before the tag passed"
        assert any("UNRELEASED" in p and "README.md" in p for p in problems), problems

    def test_an_early_citation_date_is_caught(self) -> None:
        problems = tense_problems(
            changelog=_UNRELEASED_CHANGELOG,
            readme=_unreleased_readme(),
            citation=_RELEASED_CITATION,
            pages=_healthy_unreleased_pages(),
        )
        assert any("date-released" in p for p in problems), problems

    def test_a_page_that_stops_stating_the_state_is_caught(self) -> None:
        pages = _healthy_unreleased_pages()
        pages["getting-started/index.html"] = "<p>Install it with pip.</p>"
        problems = tense_problems(
            changelog=_UNRELEASED_CHANGELOG,
            readme=_unreleased_readme(),
            citation=_UNRELEASED_CITATION,
            pages=pages,
        )
        assert any("getting-started" in p for p in problems), problems

    def test_a_heading_that_is_neither_state_is_refused(self) -> None:
        state, problems = read_state("# Changelog\n\n## [0.1.0] - beta\n")
        assert state is None
        assert any("neither dated nor marked UNRELEASED" in p for p in problems), problems

    def test_a_heading_that_is_both_states_is_refused(self) -> None:
        state, problems = read_state("# Changelog\n\n## [0.1.0] - UNRELEASED 2026-09-04\n")
        assert state is None
        assert any("two different things" in p for p in problems), problems

    def test_a_fully_flipped_release_is_accepted(self) -> None:
        """Control for the controls: the rule must be satisfiable by doing the work."""
        problems = tense_problems(
            changelog=_RELEASED_CHANGELOG,
            readme=_RELEASED_README,
            citation=_RELEASED_CITATION,
            pages={"index.html": "<p>Install with pip install vfairness.</p>"},
        )
        assert problems == [], problems

    def test_a_fully_unreleased_state_is_accepted(self) -> None:
        problems = tense_problems(
            changelog=_UNRELEASED_CHANGELOG,
            readme=_unreleased_readme(),
            citation=_UNRELEASED_CITATION,
            pages=_healthy_unreleased_pages(),
        )
        assert problems == [], problems


def test_the_badge_is_caught_by_its_url_not_only_by_its_alt_text():
    """THE HOLE THIS GATE HAD, and why it mattered more than its size.

    The README's status badge carries its words inside a shields.io URL,
    percent-encoded. The scoped patterns look for "not yet published" with real
    spaces, so they matched the badge's ALT TEXT and never the URL.

    At release, editing the alt text and leaving the URL would have shipped a
    badge reading "0.1.0 beta, not yet published" onto the PyPI project page,
    which cannot be edited after upload, with this gate green. The badge is the
    first thing on that page.

    Both halves are asserted: the URL-only form must be caught, and the plain
    form must still be caught, so the fix is not one that stopped reading the
    text it always read.
    """
    url_only = (
        '<img src="https://img.shields.io/badge/'
        'status-0.1.0%20beta%2C%20not%20yet%20published-orange.svg">'
    )
    assert _states_unreleased(url_only, "0.1.0"), (
        "a badge whose only unreleased claim is inside a percent-encoded URL "
        "was not detected; alt text is not the only place words live"
    )

    alt_only = (
        '<img src="https://img.shields.io/badge/x-orange.svg" alt="0.1.0 beta, not yet published">'
    )
    assert _states_unreleased(alt_only, "0.1.0")


def test_the_decoder_does_not_make_the_gate_fire_on_anything():
    """OVER-CORRECTION CONTROL. Decoding must not turn every page into a match:
    a page that says nothing about being unreleased must still read as released."""
    assert not _states_unreleased(
        '<img src="https://img.shields.io/badge/status-0.1.0%20stable-green.svg">', "0.1.0"
    )
    assert not _states_unreleased("vfairness 0.1.0 is available on PyPI.", "0.1.0")
