"""The PyPI project page is written once and can never be edited.

THE DEFECT THIS EXISTS FOR. A packaging review reported the project page as
"18 links, 8 of them 404". Re-measured on 2026-09-10 by rendering README.md the
way PyPI renders it (readme_renderer 46.0, the same library warehouse uses), the
page carries **23** distinct external references and **9** of them 404. The five
the review never saw are exactly the five that are not markdown links: the four
shields badges and, above them, the hero image::

    <img src="https://raw.githubusercontent.com/validantai/vfairness/main/docs/img/vfairness-hero.png" ...>

A ``](...)`` regex cannot see an ``src="..."``. So the count was produced over a
sample that excluded the single most visible element on the page, and reported
as if it covered the page. That is an unmeasured value counted as a measurement,
and it is the whole defect class in one line.

It was not covered anywhere else either. ``tests/test_site_links_truth.py``
scans ``docs/site/**/*.html`` for the site and, for the shipped markdown, uses a
markdown-link extractor plus a ``github.com/validantai/vfairness/(blob|tree|raw)``
pattern. The hero lives on ``raw.githubusercontent.com``, a different host, in an
HTML attribute, in a markdown file. Nothing in this repository looked at it.
Rename ``docs/img/vfairness-hero.png`` and every test stays green while the top
of the immutable project page becomes a broken-image icon.

THREE STATES, NOT TWO. Nine of those 23 URLs 404 right now and that is CORRECT:
the public repository ``validantai/vfairness`` still holds only LICENSE and
README.md until the release export is pushed (measured 2026-09-10 through the
GitHub trees API: two blobs). Those nine are not broken links to be "fixed" into
something else, and they are not passes either. They are **pending**, and what
this file does is make "pending" a measured claim rather than an assurance:

* the path must exist in this checkout, and
* the path must be on the export admission manifest
  (``tests/fixtures/export_published_paths.txt``), so the export actually ships
  it, and
* the URL SHAPE must be proven live by a control URL under the same prefix
  (``.../main/LICENSE``, which is one of the two files already published), so a
  wrong owner, repo or branch cannot hide inside "the export has not run yet".

WHAT RUNS WHERE. Everything above the "opt in" banner is offline and
deterministic, and runs in CI. The two network sections are SKIPPED unless
``VFAIRNESS_CHECK_EXTERNAL_LINKS=1``, following
``tests/test_site_links_truth.py``. Said plainly rather than left to look like
coverage: **the network sections do not run in CI**, so nothing here re-verifies
the live state automatically. Run them before a release. They are the last check
before a page that cannot be taken back.
"""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
README = LIB_ROOT / "README.md"
PYPROJECT = LIB_ROOT / "pyproject.toml"
EXPORT_MANIFEST = LIB_ROOT / "tests" / "fixtures" / "export_published_paths.txt"

# The public publish repository and the branch the page's links name.
PUBLIC_OWNER_REPO = "validantai/vfairness"
PUBLIC_BRANCH = "main"

_BLOB_PREFIX = f"https://github.com/{PUBLIC_OWNER_REPO}/blob/{PUBLIC_BRANCH}/"
_RAW_PREFIX = f"https://raw.githubusercontent.com/{PUBLIC_OWNER_REPO}/{PUBLIC_BRANCH}/"

# Files already published in the public repository, used as SHAPE CONTROLS: they
# prove owner, repo and branch resolve, so a sibling 404 means "not pushed yet"
# and nothing else. Measured 2026-09-10, both 200:
#   curl -sI https://github.com/validantai/vfairness/blob/main/LICENSE
#   curl -sI https://raw.githubusercontent.com/validantai/vfairness/main/LICENSE
SHAPE_CONTROLS: Dict[str, str] = {
    _BLOB_PREFIX: _BLOB_PREFIX + "LICENSE",
    _RAW_PREFIX: _RAW_PREFIX + "LICENSE",
}

# The nine references that 404 TODAY and are expected to. Every one is under the
# public repository, every path is on the export admission manifest, and none of
# them may be rewritten to point somewhere else. Measured 2026-09-10 with
# `curl -s -o /dev/null -w '%{http_code}' -L <url>`.
PENDING_UNTIL_EXPORT: Dict[str, str] = {
    _RAW_PREFIX + "docs/img/vfairness-hero.png": (
        "the hero image at the top of the project page. This is the one the "
        "release ORDER exists for: scripts/cut_release.py's docstring says the "
        "export push must land BEFORE the PyPI publish, because PyPI caches the "
        "rendered description and would bake in a broken image permanently."
    ),
    _BLOB_PREFIX + "CHANGELOG.md": "linked from 'Beta and API stability'.",
    _BLOB_PREFIX + "CITATION.cff": "linked from 'Citation'.",
    _BLOB_PREFIX + "CODE_OF_CONDUCT.md": "linked from 'Contributing'.",
    _BLOB_PREFIX + "CONTRIBUTING.md": "linked from 'Contributing'.",
    _BLOB_PREFIX + "NOTICE": "linked from 'License'.",
    _BLOB_PREFIX + "SECURITY.md": "linked from 'Contributing'.",
    _BLOB_PREFIX + "docs/API_STABILITY.md": "linked from 'Beta and API stability'.",
    _BLOB_PREFIX + "docs/DIVERGENCES.md": "linked from 'Why vfairness'.",
}

# The public-repository references that already resolve, because the placeholder
# repository holds these two files today. Recorded separately rather than folded
# into the registry above: a reference that is live now and 404s later is a
# regression, and one that was never expected to work is not. Measured
# 2026-09-10: blob/main/LICENSE -> 200.
ALREADY_LIVE_IN_PUBLIC_REPO: Dict[str, str] = {
    _BLOB_PREFIX + "LICENSE": (
        "LICENSE and README.md are the only two blobs in the placeholder "
        "repository, which is why LICENSE is also the SHAPE CONTROL above."
    ),
}

# Every host the page is allowed to name. A host that is not here is either a
# typo or, worse, an internal one, and either way it must be a deliberate edit
# to this list rather than a silent addition to the page.
PUBLIC_HOSTS: Set[str] = {
    "github.com",
    "raw.githubusercontent.com",
    "img.shields.io",
    "semver.org",
    "validant.ai",
    "vfairness.validant.ai",
}

# The export's own fail-closed leak gate patterns, READ AT RUN TIME from
# scripts/export-vfairness-to-public.sh rather than copied here.
#
# Copying them was the first version of this file and it was wrong twice over.
# It would have published the entire denylist as literals INTO the public repo,
# so the export gate would have aborted on this very file the moment it was
# committed (measured: 8 of the 11 patterns fired against it). And a copied
# denylist drifts: a test that carries its own stale copy goes on checking a
# list the script stopped using, which is the same claims-versus-code defect
# this whole audit removed from the library.
#
# Matched case-INSENSITIVELY there, so here too. That is not incidental: the
# platform spells one of these tokens with a capital while the list spells it
# lower case, and a one-character variant used to walk straight through. The
# token itself is deliberately not written here, for the reason above.
_EXPORT_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export-vfairness-to-public.sh"


def _recon_patterns() -> Tuple[str, ...]:
    """The LEAK_PATTERNS_RECON array, from the script that actually enforces it."""
    src = _EXPORT_SCRIPT.read_text(encoding="utf-8")
    opened = re.search(r"^LEAK_PATTERNS_RECON=\(", src, re.M)
    assert opened, f"could not find LEAK_PATTERNS_RECON in {_EXPORT_SCRIPT}"
    body = src[opened.end() :].split("\n)", 1)[0]
    # ENTRIES ONLY. A naive findall over the whole body also swallows the quoted
    # examples inside the array's own comments: the first version of this reader
    # returned 17 "patterns", 6 of them the NEGATIVE controls the comments use to
    # explain the word boundary. A denylist reader that quietly picks up
    # non-patterns is the same defect class as everything else here, so an entry
    # must START its line.
    found = re.findall(r"""^\s*['"]([^'"]+)['"]""", body, re.M)
    assert found, f"LEAK_PATTERNS_RECON in {_EXPORT_SCRIPT} parsed to nothing"
    return tuple(found)


INTERNAL_RECON_PATTERNS: Tuple[str, ...] = _recon_patterns() if _EXPORT_SCRIPT.is_file() else ()

_MD_LINK_RE = re.compile(r"\]\(\s*(https?://[^)\s]+)")
_HTML_ATTR_RE = re.compile(r'(?:src|href)="(https?://[^"]+)"')
_ANY_URL_RE = re.compile(r"""https?://[^\s"'<>)\]]+""")


def _readme() -> str:
    text = README.read_text(encoding="utf-8")
    assert text.strip(), "README.md is empty; it IS the PyPI long description"
    return text


def _trim(url: str) -> str:
    return url.rstrip(".,;:")


def markdown_urls(text: str) -> Set[str]:
    """What a ``](...)`` scan sees. The 2026-09-10 review used only this."""
    return {_trim(m.group(1)) for m in _MD_LINK_RE.finditer(text)}


def html_attribute_urls(text: str) -> Set[str]:
    """What an ``src=`` / ``href=`` scan sees, which markdown scans do not."""
    return {_trim(m.group(1)) for m in _HTML_ATTR_RE.finditer(text)}


def page_urls() -> Set[str]:
    """Every absolute URL PyPI will render as a link or an image.

    Both extractors, because the page is markdown WITH inline HTML and either
    one alone reports a clean count over a sample that excludes the other.
    """
    text = _readme()
    return markdown_urls(text) | html_attribute_urls(text)


def _repo_paths_on_the_page() -> Dict[str, str]:
    """URL -> repository-relative path, for links into the public repository."""
    out: Dict[str, str] = {}
    for url in page_urls():
        for prefix in (_BLOB_PREFIX, _RAW_PREFIX):
            if url.startswith(prefix):
                out[url] = url[len(prefix) :].split("#")[0].split("?")[0]
    return out


def export_manifest() -> Set[str]:
    lines = EXPORT_MANIFEST.read_text(encoding="utf-8").splitlines()
    paths = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
    assert len(paths) > 100, (
        f"the export admission manifest at {EXPORT_MANIFEST} yielded only "
        f"{len(paths)} paths; every check that reads it would be checking nothing"
    )
    return paths


def _is_exported(path: str) -> bool:
    """The manifest deliberately omits src/ and tests/: they ARE the package."""
    return path in export_manifest() or path.startswith(("src/", "tests/"))


def pyproject() -> Dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def project_table() -> Dict[str, object]:
    table = pyproject()["project"]
    assert isinstance(table, dict)
    return table


# --------------------------------------------------------------------------
# 1. The inventory. The blind spot that produced the wrong count.
# --------------------------------------------------------------------------


def test_the_extractor_sees_the_urls_a_markdown_only_scan_misses() -> None:
    """REFUSAL PIN for the defect in the docstring.

    Reinstate the markdown-only scan in ``page_urls`` and this goes red, naming
    the hero image. It is written as a floor plus an explicit membership test
    rather than "the sets differ", because a set difference of one badge would
    satisfy a weaker assertion while the hero was still missing.
    """
    text = _readme()
    markdown_only = markdown_urls(text)
    whole_page = page_urls()

    hero = _RAW_PREFIX + "docs/img/vfairness-hero.png"
    assert hero not in markdown_only, (
        "the hero image is now a markdown link, so this test no longer "
        "reproduces the blind spot it exists for; rewrite it against whatever "
        "the page's non-markdown references are today"
    )
    assert hero in whole_page, (
        "the hero image is not in the extracted inventory. Every link check in "
        "this file is then blind to the largest element on the immutable page."
    )

    missed = whole_page - markdown_only
    assert len(missed) >= 5, (
        f"a markdown-only scan misses {len(missed)} of the page's references, "
        f"expected at least 5 (the hero image and four badges): {sorted(missed)}"
    )
    assert all("img.shields.io" in u or u == hero for u in missed), (
        f"the non-markdown references are no longer only badges and the hero: "
        f"{sorted(missed)}. Confirm each one is checked before widening this."
    )


def test_the_inventory_covers_every_url_in_the_file() -> None:
    """OVER-CORRECTION CONTROL, and a guard against a third extractor being needed.

    GFM turns a bare ``https://...`` in prose into a link. Neither extractor
    above would see one, so the page could grow a reference that no check in
    this file ever fetches. Compared against a raw scan for the scheme, which
    cannot miss anything a reader can click.
    """
    raw = {_trim(m.group(0)) for m in _ANY_URL_RE.finditer(_readme())}
    missing = raw - page_urls()
    assert not missing, (
        "README.md contains URLs that neither the markdown nor the HTML "
        f"extractor picks up, so nothing checks them: {sorted(missing)}"
    )
    assert len(raw) >= 20, (
        f"only {len(raw)} URLs found in README.md; the page carried 23 distinct "
        "references on 2026-09-10, so the extractor is broken or the page lost "
        "most of its links"
    )


def test_the_offline_extractor_matches_what_pypi_actually_renders() -> None:
    """OVER-CORRECTION CONTROL: the offline inventory IS the rendered page.

    Skipped, never passed, when readme_renderer is absent: a missing library is
    "could not check", not agreement. readme_renderer is what warehouse renders
    the description with, so this is the authoritative comparison.
    """
    renderer = pytest.importorskip(
        "readme_renderer.markdown",
        reason="could not check: readme_renderer[md] is not installed",
    )
    rendered: Optional[str] = renderer.render(_readme())
    assert rendered is not None, (
        "readme_renderer refused to render README.md. PyPI would show the raw "
        "text instead of the page, permanently."
    )
    from_rendered = {
        _trim(m.group(1)) for m in re.finditer(r'(?:href|src)="(https?://[^"]+)"', rendered)
    }
    assert from_rendered == page_urls(), (
        "the offline inventory and the rendered page disagree.\n"
        f"  only in the rendered page: {sorted(from_rendered - page_urls())}\n"
        f"  only in the inventory:     {sorted(page_urls() - from_rendered)}"
    )


# --------------------------------------------------------------------------
# 2. The nine pending references, made checkable without a network.
# --------------------------------------------------------------------------


def test_every_public_repo_link_names_a_path_this_checkout_ships() -> None:
    """A link to a path that does not exist 404s after the export too, forever."""
    missing = [
        f"{url} -> {path!r} does not exist in this checkout"
        for url, path in sorted(_repo_paths_on_the_page().items())
        if not (LIB_ROOT / path).exists()
    ]
    assert not missing, "\n".join(missing)


def test_every_public_repo_link_is_on_the_export_admission_manifest() -> None:
    """REFUSAL PIN. Existing here is not enough: the export must publish it.

    This is the failure mode that produced twelve permanently dead notebook
    links on the docs site. A well-formed URL to a file the publish boundary
    drops is indistinguishable from a good one until a reader clicks it.
    """
    unshipped = [
        f"{url} -> {path!r} is not on {EXPORT_MANIFEST.relative_to(LIB_ROOT)}, so the "
        f"export does not publish it and this link 404s permanently"
        for url, path in sorted(_repo_paths_on_the_page().items())
        if not _is_exported(path)
    ]
    assert not unshipped, "\n".join(unshipped)


def test_the_hero_image_is_the_file_the_export_ships() -> None:
    """OVER-CORRECTION CONTROL asserting measured values, not membership.

    The two tests above iterate a set. If that set were ever empty they would
    both pass having established nothing, which is how the original count came
    to exclude this file. So the hero is named, its bytes are read, and its
    export admission is asserted individually.
    """
    url = _RAW_PREFIX + "docs/img/vfairness-hero.png"
    assert url in page_urls(), "the hero image URL is not on the page"

    path = "docs/img/vfairness-hero.png"
    on_disk = LIB_ROOT / path
    assert on_disk.is_file(), f"{path} does not exist in this checkout"
    assert on_disk.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", (
        f"{path} is not a PNG; PyPI would render a broken image"
    )
    assert on_disk.stat().st_size > 10_000, (
        f"{path} is {on_disk.stat().st_size} bytes, too small to be the hero art"
    )
    assert path in export_manifest(), (
        f"{path} is not on the export admission manifest. The image at the top "
        "of the immutable project page would 404 forever."
    )


def test_the_pending_registry_matches_the_page() -> None:
    """The registry is a claim about the page; drift makes it fiction."""
    on_page = page_urls()
    stale = sorted((set(PENDING_UNTIL_EXPORT) | set(ALREADY_LIVE_IN_PUBLIC_REPO)) - on_page)
    assert not stale, (
        f"the registries record URLs the page no longer carries: {stale}. "
        "Remove them, or the live check below excuses a 404 nobody links to."
    )
    recorded = set(PENDING_UNTIL_EXPORT) | set(ALREADY_LIVE_IN_PUBLIC_REPO)
    unregistered = sorted(u for u in _repo_paths_on_the_page() if u not in recorded)
    assert not unregistered, (
        f"these public-repository references are recorded in neither "
        f"PENDING_UNTIL_EXPORT nor ALREADY_LIVE_IN_PUBLIC_REPO: {unregistered}. "
        "Fetch each one and file it under the state it is actually in, or the "
        "release operator has no list to verify after the export push."
    )
    overlap = sorted(set(PENDING_UNTIL_EXPORT) & set(ALREADY_LIVE_IN_PUBLIC_REPO))
    assert not overlap, f"a reference cannot be both pending and already live: {overlap}"
    assert len(PENDING_UNTIL_EXPORT) == 9, (
        f"the registry holds {len(PENDING_UNTIL_EXPORT)} entries; it held 9 when "
        "it was measured on 2026-09-10. Re-verify before changing the number."
    )


# --------------------------------------------------------------------------
# 3. Hosts. Nothing internal may reach a page a stranger reads.
# --------------------------------------------------------------------------


def test_the_page_names_only_public_hosts() -> None:
    from urllib.parse import urlparse

    hosts = {urlparse(u).hostname or "" for u in page_urls()}
    unexpected = sorted(hosts - PUBLIC_HOSTS)
    assert not unexpected, (
        f"the PyPI page names hosts that are not on the public allowlist: "
        f"{unexpected}. Confirm each is public and reachable from outside, then "
        "add it to PUBLIC_HOSTS deliberately."
    )
    assert "raw.githubusercontent.com" in hosts and "vfairness.validant.ai" in hosts, (
        f"the host scan found {sorted(hosts)}, which does not include the two "
        "hosts the page cannot work without; the extractor is broken"
    )


@pytest.mark.skipif(
    not INTERNAL_RECON_PATTERNS,
    reason="the export script is not in this checkout, so there is no denylist to read",
)
def test_no_internal_recon_token_reaches_the_published_files() -> None:
    """REFUSAL PIN, using the export gate's own patterns, read from the script.

    gitleaks does not help here and never will: recon is not credential-shaped.
    A denylist is the only check that will ever look for these.
    """
    hits: List[str] = []
    for path in (README, PYPROJECT):
        text = path.read_text(encoding="utf-8")
        for pattern in INTERNAL_RECON_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                line = text[: m.start()].count("\n") + 1
                hits.append(f"{path.name}:{line}: {pattern!r} matched {m.group(0)!r}")
    assert not hits, "internal-infrastructure token on a published file:\n" + "\n".join(hits)


# --------------------------------------------------------------------------
# 4. The metadata a visitor sees, and the build floor that decides it.
# --------------------------------------------------------------------------


def test_requires_python_is_the_supported_floor() -> None:
    """REFUSAL PIN for the exact mistake that has to be yanked.

    The 0.0.1 placeholder on PyPI declares ``Requires-Python: >=3.9`` (read from
    the PyPI JSON API, 2026-09-10). Nothing in this codebase runs on 3.9 or
    3.10: the numpy, pandas, scipy and matplotlib floors above are all keyed to
    the first cp311 release. A requires-python below 3.11 hands a 3.9 user a
    resolvable install that fails at the import line.
    """
    assert project_table()["requires-python"] == ">=3.11"


def test_the_python_classifiers_agree_with_requires_python() -> None:
    """OVER-CORRECTION CONTROL asserting the measured minors, not a subset.

    Membership of a broad set pins nothing here: "some 3.x classifier is
    present" is true of ``Python :: 3.9`` as well.
    """
    classifiers = project_table()["classifiers"]
    assert isinstance(classifiers, list)
    minors = sorted(
        tuple(int(part) for part in c.rsplit(" :: ", 1)[1].split("."))
        for c in classifiers
        if c.startswith("Programming Language :: Python :: ") and "." in c.rsplit(" :: ", 1)[1]
    )
    assert minors == [(3, 11), (3, 12), (3, 13)], (
        f"the page advertises Python {minors}, which does not match "
        f"requires-python {project_table()['requires-python']!r}"
    )
    assert "Programming Language :: Python :: 3" in classifiers


def test_the_licence_is_an_spdx_expression_with_its_files() -> None:
    """REFUSAL PIN tied to the build-system floor comment above it.

    Measured 2026-09-10: hatchling 1.26.3 turns this same ``license`` string
    into the legacy free-text ``License:`` header and emits NO ``License-File``
    header at all, so LICENSE and NOTICE vanish from the metadata with no error.
    The floor and the two fields it exists to enable are asserted together,
    because either one alone is satisfiable while the other is broken.
    """
    project = project_table()
    assert project.get("license") == "Apache-2.0", (
        f"project.license is {project.get('license')!r}; it must be a PEP 639 "
        "SPDX expression, not a table, a file path or absent"
    )
    assert "license-files" in project, (
        "project.license-files is not declared at all, so no License-File header "
        "is emitted and LICENSE and NOTICE are absent from the metadata"
    )
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    classifiers = project["classifiers"]
    assert isinstance(classifiers, list)
    assert not [c for c in classifiers if c.startswith("License ::")], (
        "a legacy License :: classifier alongside an SPDX expression is rejected on upload"
    )

    build_system = pyproject()["build-system"]
    assert isinstance(build_system, dict)
    requires = build_system["requires"]
    assert requires == ["hatchling>=1.27"], (
        f"build-system.requires is {requires!r}. 1.27 is the measured boundary "
        "at which License-Expression and License-File start being emitted; "
        "below it both are silently dropped."
    )


def test_the_readme_is_what_gets_rendered_as_markdown() -> None:
    """``Description-Content-Type`` is derived from this filename, not declared."""
    assert project_table()["readme"] == "README.md", (
        "the long description no longer comes from README.md, so the content "
        "type PyPI renders it with is no longer text/markdown by derivation"
    )
    assert README.suffix == ".md"


def test_the_summary_is_a_single_useful_line() -> None:
    summary = project_table()["description"]
    assert isinstance(summary, str)
    assert "\n" not in summary
    assert 40 <= len(summary) <= 200, f"summary is {len(summary)} chars: {summary!r}"
    assert "fairness" in summary.lower()


def test_the_page_discloses_the_placeholder_that_carries_this_name() -> None:
    """A visitor arriving from ``pip install vfairness`` must be told.

    True in both release tenses: before the cut the README names the unrelated
    ``0.0.1``; after ``scripts/cut_release.py`` runs, it says the same version
    has been yanked. Dropping the disclosure in either direction is caught here.
    """
    assert "0.0.1" in _readme(), (
        "README.md no longer mentions the 0.0.1 distribution that holds this "
        "name on PyPI. A 3.9 or 3.10 user still resolves to it."
    )


# --------------------------------------------------------------------------
# Opt in, and NOT part of CI. Stated so the offline sections above are not
# mistaken for a live check of anything.
# --------------------------------------------------------------------------

_OPT_IN = pytest.mark.skipif(
    os.environ.get("VFAIRNESS_CHECK_EXTERNAL_LINKS") != "1",
    reason="opt-in network check; set VFAIRNESS_CHECK_EXTERNAL_LINKS=1 to run it",
)


def _status(url: str, timeout: int = 30) -> Tuple[Optional[int], str]:
    """(status, detail). status is None when the answer could not be obtained."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # noqa: BLE001  - never silently pass
        return None, f"{type(exc).__name__}: {exc}"


@_OPT_IN
def test_every_url_on_the_page_answers_in_three_states(capsys: pytest.CaptureFixture) -> None:
    """live / pending / could-not-check, and pending has to earn it.

    A 404 is excused ONLY when all three of these hold: the URL is in
    PENDING_UNTIL_EXPORT, its path is on the export manifest (asserted offline
    above), and the SHAPE CONTROL under the same prefix answers live in THIS
    run. Without the control, a typo in the owner, the repo or the branch name
    would sit inside the excuse indefinitely: every URL under a misspelled
    prefix 404s exactly like one that has not been pushed yet.

    403 / 405 / 429 / a TLS chain this interpreter cannot build / a timeout are
    printed as could-not-check and do not fail. They are not evidence the page
    is gone, and a check that cries wolf gets deleted, which ends in the same
    place as a check that cannot fail.
    """
    controls: Dict[str, bool] = {}
    for prefix, control_url in SHAPE_CONTROLS.items():
        status, detail = _status(control_url)
        controls[prefix] = status is not None and status < 400
        with capsys.disabled():
            print(f"\nshape control {control_url} -> {status if status else detail}")

    dead: List[Tuple[str, str]] = []
    pending: List[str] = []
    unearned: List[str] = []
    unchecked: List[Tuple[str, str]] = []
    live = 0

    for url in sorted(page_urls()):
        status, detail = _status(url)
        if status is None:
            unchecked.append((url, detail))
            continue
        if status < 400:
            live += 1
            continue
        if status in {404, 410}:
            matched: Optional[str] = next((p for p in SHAPE_CONTROLS if url.startswith(p)), None)
            if url in PENDING_UNTIL_EXPORT and matched is not None and controls[matched]:
                pending.append(url)
            elif url in PENDING_UNTIL_EXPORT and matched is not None:
                unearned.append(f"{url} -> {status}, and {SHAPE_CONTROLS[matched]} is not live")
            else:
                dead.append((url, str(status)))
        else:
            unchecked.append((url, str(status)))

    with capsys.disabled():
        print(
            f"\nPyPI page references: {len(page_urls())} total, {live} live, "
            f"{len(pending)} pending export, {len(dead)} dead, "
            f"{len(unearned)} unearned pending, {len(unchecked)} could not check"
        )
        for url in sorted(pending):
            print(f"  pending export push: {url}")
        for url, why in unchecked:
            print(f"  could not check: {url} -> {why}")

    assert live >= 10, (
        f"only {live} references verified live; this run established almost "
        "nothing and must not be read as a pass"
    )
    # COULD-NOT-CHECK IS NOT A PASS FOR THIS CLAIM. A 403 among the badges is
    # noise and is tolerated above. A 404 under the public repository whose
    # shape control is not live is the opposite: the excuse "the export has not
    # run yet" is precisely what could not be established, so it must not be
    # granted by default.
    assert not unearned, (
        "these 404s are recorded as pending the export push, but that could not "
        "be established in this run because the control file under the same "
        "prefix did not answer live:\n" + "\n".join(f"  {u}" for u in unearned)
    )
    assert not dead, "dead references on an immutable page:\n" + "\n".join(
        f"  {u} -> {c}" for u, c in dead
    )


@_OPT_IN
def test_a_yank_claim_on_the_page_is_backed_by_pypi(capsys: pytest.CaptureFixture) -> None:
    """The released tense asserts the 0.0.1 placeholder "has been yanked".

    That sentence is the reason a 3.9 or 3.10 user does not silently resolve to
    a 4.5 KB package with no ``FairnessAnalyzer`` in it (measured 2026-09-10:
    the 0.0.1 wheel is 4544 bytes and its ``__init__.py`` holds a docstring and
    ``__version__``, nothing else). It is a claim about a live index that no
    other check in this repository looks at.

    Could-not-check FAILS here, unlike in the sweep above. You opted into the
    network for the one fact that decides whether an immutable page tells the
    truth; not getting an answer is not a pass.
    """
    import json
    import urllib.request

    if "has been yanked" not in _readme():
        pytest.skip(
            "README.md is still in the unreleased tense and makes no yank claim; "
            "nothing to verify (this is not a pass)"
        )

    try:
        with urllib.request.urlopen(  # noqa: S310
            "https://pypi.org/pypi/vfairness/json", timeout=30
        ) as response:
            payload = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"could not check whether 0.0.1 is yanked on PyPI: {exc}")

    files = payload.get("releases", {}).get("0.0.1")
    assert files, "PyPI reports no 0.0.1 release, but README.md says one was yanked"
    unyanked = [f["filename"] for f in files if not f.get("yanked")]
    with capsys.disabled():
        print(f"\n0.0.1 files on PyPI: {[f['filename'] for f in files]}, unyanked: {unyanked}")
    assert not unyanked, (
        "README.md tells every reader the 0.0.1 placeholder has been yanked, and "
        f"these files are not yanked: {unyanked}. A 3.9 or 3.10 resolver still "
        "selects it. Yank them, or the immutable page is false."
    )


@pytest.mark.skipif(
    not _EXPORT_SCRIPT.is_file(),
    reason="the export script is not in this checkout",
)
def test_the_denylist_reader_read_the_real_list() -> None:
    """ANTI-VACUITY, checked against an INDEPENDENT read of the same array.

    A reader that returns nothing, or the wrong things, makes every recon test
    above pass for the wrong reason. Two failure modes, both measured while
    writing this file: a COPIED denylist drifts from the one the script
    enforces, and a naive findall over the array body also swallows the quoted
    negative controls inside the array's own comments (it returned 17 entries
    instead of 11).

    So the parse is verified by asking bash to expand the array itself, which is
    the same thing the gate does at export time. No pattern is spelled out here:
    this module is published, and carrying the denylist is the defect it exists
    to prevent.
    """
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{_EXPORT_SCRIPT}" >/dev/null 2>&1 || true; '
            'printf "%s\\n" "${LEAK_PATTERNS_RECON[@]}"',
        ],
        capture_output=True,
        text=True,
    )
    from_bash = tuple(x for x in proc.stdout.splitlines() if x)
    if not from_bash:
        pytest.skip("the export script could not be sourced in this environment")
    assert INTERNAL_RECON_PATTERNS == from_bash, (
        "the Python reader and bash disagree about LEAK_PATTERNS_RECON:\n"
        f"  python: {len(INTERNAL_RECON_PATTERNS)} entries\n"
        f"  bash:   {len(from_bash)} entries"
    )


def test_this_file_carries_no_denylisted_literal_of_its_own() -> None:
    """The reason this file READS the list instead of carrying it.

    The first version hard-coded every pattern as a regex literal. This module
    lives under tests/ and is published, so committing it would have put the
    whole denylist into the public repository and the export gate would have
    aborted on this very file. Measured then: 8 of the 11 patterns fired. This
    test also caught the SECOND version, where the tokens survived inside my own
    explanatory comments and one assertion.
    """
    if not INTERNAL_RECON_PATTERNS:
        pytest.skip("no denylist available in this checkout")
    text = Path(__file__).read_text(encoding="utf-8")
    hits = [p for p in INTERNAL_RECON_PATTERNS if re.search(p, text, re.IGNORECASE)]
    assert not hits, (
        "this test module itself matches the export denylist, so committing it "
        f"would abort the export: {len(hits)} pattern(s)"
    )
