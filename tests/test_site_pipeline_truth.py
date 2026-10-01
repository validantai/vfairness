"""The published docs site may not describe a pipeline the workflow does not have.

Blocker B5 of the publish-readiness audit (2026-08-28). ``docs/RELEASE_PIPELINE.md``
was corrected when the TestPyPI leg was removed from ``release.yml``; the SITE was
not, and the site is the surface a stranger meets. Still live on
``docs/site/release-pipeline/index.html`` at the time of this file:

- "rehearsed on TestPyPI before they touch PyPI";
- an enumerated job "3. Publish to TestPyPI" that no longer exists (while the job
  that does exist, ``github-release``, was described nowhere);
- an OPERATOR INSTRUCTION to create "GitHub Environments ``testpypi`` and
  ``pypi``", one of which nothing uses;
- and the worst of them, "TestPyPI is the rehearsal: a failure there stops the
  pipeline before it touches PyPI" -- a claimed safety control in front of an
  irreversible upload, which was not there at all.

A prior commit (f59f206) claimed every surviving mention had been "read in
context" and was a statement of fact. It had edited three lines and checked two
phrases. Prose review is what failed, so the pin is executable and derived from
``.github/workflows/release.yml`` rather than from a list of forbidden phrases:

- the flow section must enumerate exactly the jobs the workflow declares;
- a page may name a package index the workflow does not publish to ONLY to say it
  is absent ("nothing has reached PyPI or TestPyPI" is fine and stays);
- an environment name presented as a GitHub Environment must be one the workflow
  actually declares;
- and the human-approval claim must carry the caveat that an environment which
  was never created is auto-created by GitHub with no reviewers, and therefore
  gates nothing. That is the three-states rule applied to documentation: an
  approval that cannot be verified from the workflow file must not read as an
  approval that certainly happens.

What these checks DO NOT establish: they read the shipped HTML, not the rendered
page, and the "absence" rule is satisfied by a negation cue anywhere in the same
sentence, so a sufficiently contorted false sentence could pass. They catch the
shape of the defect that actually shipped, and every one of them is executed
against that defect, reinstated verbatim, in
``TestTheseChecksAreThemselvesChecked``.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Set
from urllib.parse import urlparse

import pytest

yaml = pytest.importorskip("yaml")

LIB_ROOT = Path(__file__).resolve().parents[1]
SITE = LIB_ROOT / "docs" / "site"
RELEASE_WORKFLOW = LIB_ROOT / ".github" / "workflows" / "release.yml"
# MERGED 2026-09-28. The release pipeline was a page of its own and is now the
# "How a release happens" section of the hardening page, because a reader should not
# have to hold two pages in their head to judge one release. The rules below did not
# change and are still derived from .github/workflows/release.yml; only the file they
# read moved. If that section is ever split out again, point this at the new file and
# the 22 checks follow it.
RELEASE_PIPELINE_PAGE = SITE / "release-pipeline" / "index.html"  # its own page since 2026-10-01

PUBLISH_ACTION = "pypa/gh-action-pypi-publish"

# Package indexes this project could plausibly upload to, and the hosts that
# identify them. Matched on the parsed netloc, never as a substring: "pypi.org"
# is a substring of "test.pypi.org", so a substring test reads a TestPyPI URL as
# PyPI and would conclude the pipeline publishes where it does not.
INDEX_HOSTS = {
    "pypi.org": "pypi",
    "upload.pypi.org": "pypi",
    "test.pypi.org": "testpypi",
    "upload.test.pypi.org": "testpypi",
}
INDEX_NAMES = {"pypi": "PyPI", "testpypi": "TestPyPI"}

# A mention of an index the pipeline does not use is acceptable only as a
# statement of absence. Deliberately generous: the point is to catch a sentence
# that DESCRIBES a rehearsal upload, not to police English.
_NEGATION_CUE = re.compile(
    r"\b(no|not|never|nothing|none|neither|without|removed|remove|absent|cannot)\b",
    re.IGNORECASE,
)


# --- Reading the workflow ----------------------------------------------------


def _workflow() -> dict:
    assert RELEASE_WORKFLOW.exists(), (
        f"{RELEASE_WORKFLOW} is missing; the pipeline these pages describe cannot be read, "
        "and a docs check with nothing to compare against is not a check"
    )
    return yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))


def _jobs(workflow: dict) -> Dict[str, dict]:
    return dict(workflow["jobs"])


def _declared_environments(workflow: dict) -> Set[str]:
    """Environment names the workflow actually declares on a job."""
    found: Set[str] = set()
    for job in workflow["jobs"].values():
        env = job.get("environment")
        if env is None:
            continue
        if isinstance(env, dict):
            env = env.get("name")
        if env:
            found.add(str(env).strip())
    return found


def _publish_targets(workflow: dict) -> Set[str]:
    """Which package indexes the workflow uploads to, read from its publish steps.

    No ``repository-url`` means the action's own default, which is PyPI. An
    unrecognised host raises rather than being ignored: silently dropping a
    target would tell the docs check that an index is "not used" and license
    every sentence describing it.
    """
    targets: Set[str] = set()
    for job in workflow["jobs"].values():
        for step in job.get("steps") or []:
            if PUBLISH_ACTION not in str(step.get("uses", "")):
                continue
            url = str((step.get("with") or {}).get("repository-url", "")).strip()
            if not url:
                targets.add("pypi")
                continue
            host = urlparse(url).netloc.lower()
            if host not in INDEX_HOSTS:
                raise ValueError(
                    f"publish step uploads to an unrecognised index host {host!r}; teach "
                    "INDEX_HOSTS about it rather than letting this check ignore a real target"
                )
            targets.add(INDEX_HOSTS[host])
    return targets


# --- Reading a page ----------------------------------------------------------


class _PageReader(HTMLParser):
    """Visible text plus the identifier-shaped tokens that sit inside <code>."""

    _SKIP = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.code_tokens: Set[str] = set()
        self._skip = 0
        self._code = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "code":
            self._code += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "code" and self._code:
            self._code -= 1

    def handle_data(self, data):
        if self._skip:
            return
        self.parts.append(data)
        if self._code and _IDENTIFIER.match(data.strip()):
            self.code_tokens.add(data.strip())


# An environment name is lowercase and identifier-shaped. Anything with a space,
# a slash, a dot or a capital is a path, a command or an English word, and is
# left alone.
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]*$")


class Page:
    """One shipped HTML file, as text a rule can be written against."""

    def __init__(self, path: Path, html: str | None = None) -> None:
        self.path = path
        # `html` is how the sabotage tests hand a rule a page with the defect put
        # back, without writing the defect to disk.
        self.html = path.read_text(encoding="utf-8") if html is None else html
        reader = _PageReader()
        reader.feed(self.html)
        text = " ".join(reader.parts)
        text = re.sub(r"\s+", " ", text)
        # Tag boundaries leave a space in front of punctuation ("the pypi
        # environment ."), which would otherwise defeat sentence splitting.
        self.text = re.sub(r"\s+([,.;:!?)])", r"\1", text).strip()
        self.code_tokens = reader.code_tokens

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(SITE))

    def sentences(self) -> List[str]:
        return [s for s in re.split(r"(?<=[.!?])\s+", self.text) if s]

    def section(self, section_id: str) -> str:
        """The raw HTML of one <section id="...">, for section-scoped rules."""
        match = re.search(
            rf'<section id="{re.escape(section_id)}".*?</section>', self.html, re.DOTALL
        )
        assert match is not None, f'{self.rel}: no <section id="{section_id}"> to inspect'
        return match.group(0)


def _site_pages() -> List[Path]:
    pages = sorted(SITE.rglob("*.html"))
    assert pages, f"no HTML under {SITE}; this check would pass by finding nothing"
    return pages


SITE_PAGES = _site_pages()


def _text_of(fragment: str) -> str:
    reader = _PageReader()
    reader.feed(fragment)
    text = re.sub(r"\s+", " ", " ".join(reader.parts))
    return re.sub(r"\s+([,.;:!?)])", r"\1", text).strip()


# --- Rule 1: the flow section enumerates exactly the workflow's jobs ----------
#
# The cue for each job is spelled out here rather than derived from its `name:`,
# because "Publish to PyPI" and "Publish the GitHub Release" share every word a
# naive tokeniser would key on. The map is kept honest by
# test_the_cue_map_covers_exactly_the_workflow_jobs, which fails loudly the
# moment a job is added, removed or renamed, so this cannot quietly go stale the
# way the page it checks did.
JOB_FLOW_CUES = {
    # Added 2026-09-11. The gate that refuses to build from a commit whose suite
    # was red. It was added to release.yml and never described on the page, which
    # is exactly what this test exists to catch; it had been red at HEAD while CI
    # stayed green, because CI does not collect this module.
    "suite-green-on-this-commit": "suite green",
    "build": "build",
    "smoke": "smoke",
    "publish-pypi": "publish to pypi",
    "github-release": "github release",
}

_H3 = re.compile(r"<h3>(.*?)</h3>", re.DOTALL)


def _flow_headings(page: Page) -> List[str]:
    return [_text_of(h).strip() for h in _H3.findall(page.section("flow"))]


def _flow_problems(page: Page, workflow: dict) -> List[str]:
    jobs = _jobs(workflow)
    headings = _flow_headings(page)
    problems: List[str] = []
    if not headings:
        return [f"{page.rel}: the flow section enumerates no steps at all"]

    matched: Dict[str, List[str]] = {name: [] for name in jobs}
    for heading in headings:
        lowered = heading.lower()
        hits = [name for name in jobs if JOB_FLOW_CUES.get(name, name) in lowered]
        if not hits:
            problems.append(
                f"{page.rel}: the flow lists {heading!r}, which matches no job in "
                f"release.yml (jobs: {sorted(jobs)})"
            )
        for name in hits:
            matched[name].append(heading)

    for name, hit in matched.items():
        if not hit:
            problems.append(
                f"{page.rel}: release.yml declares the job {name!r}, which the flow section "
                "never describes"
            )
        elif len(hit) > 1:
            problems.append(f"{page.rel}: the job {name!r} is described by several steps: {hit}")
    return problems


# --- Rule 2: an unused index may be named only to say it is unused -----------


def _index_problems(page: Page, workflow: dict) -> List[str]:
    unused = set(INDEX_NAMES) - _publish_targets(workflow)
    problems: List[str] = []
    for key in sorted(unused):
        pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
        for sentence in page.sentences():
            if not pattern.search(sentence):
                continue
            if _NEGATION_CUE.search(sentence):
                continue
            problems.append(
                f"{page.rel}: describes {INDEX_NAMES[key]}, which release.yml does not publish "
                f"to, as something that happens: {sentence!r}"
            )
    return problems


# --- Rule 3: only environments the workflow declares may be presented as such -

# "the pypi environment" / "Environment, named pypi" / "environment: pypi".
#
# The trailing alternative uses a LOOKAHEAD, so it does not swallow the word
# "environment" itself. Consuming it made the two alternatives fight: in "GitHub
# Environments testpypi and pypi" the second alternative matched "ub
# Environments" first, and because re.finditer yields non-overlapping matches the
# name that followed was never examined at all. The check passed on the exact
# defect it exists for.
_ENV_ADJACENT = re.compile(
    r"[Ee]nvironments?\b[,:]?\s+(?:named\s+)?(?P<after>[a-z][a-z0-9_-]*)"
    r"|(?P<before>[a-z][a-z0-9_-]*)\s+(?=[Ee]nvironments?\b)"
)


def _claimed_environments(page: Page) -> Set[str]:
    """Names the page presents as GitHub Environments.

    A candidate must be BOTH adjacent to the word "environment" and marked up as
    code on that page, so ordinary prose ("a clean environment", "the current
    environment") is not mistaken for a name.
    """
    claimed: Set[str] = set()
    for match in _ENV_ADJACENT.finditer(page.text):
        for token in (match.group("after"), match.group("before")):
            if token and token in page.code_tokens:
                claimed.add(token)
    return claimed


def _environment_problems(page: Page, workflow: dict) -> List[str]:
    declared = _declared_environments(workflow)
    problems = []
    for name in sorted(_claimed_environments(page) - declared):
        problems.append(
            f"{page.rel}: presents {name!r} as a GitHub Environment, but release.yml declares "
            f"only {sorted(declared) or 'none'}. A reader told to create it would configure a "
            "credential and a protection rule that gate nothing."
        )
    return problems


# --- Rule 4: the approval claim must carry the caveat that makes it honest ----
#
# `environment: pypi` in a workflow file does NOT establish that an approval
# happens. If no environment of that name exists, GitHub creates it implicitly,
# with no reviewers and no protection rules, and the run walks straight into the
# upload; nothing in the workflow or the run log distinguishes that from a
# configured one. So a page claiming the approval must also say what it depends
# on: could-not-check may not be rendered as assessed-pass.

_APPROVAL_CLAIM = re.compile(r"required reviewer|human approval|pauses for approval", re.IGNORECASE)
_APPROVAL_CAVEAT = re.compile(
    r"created implicitly|creates the environment implicitly|no reviewers|gates nothing"
    r"|imposes no pause|never created",
    re.IGNORECASE,
)


def _caveat_problems(page: Page) -> List[str]:
    if not _APPROVAL_CLAIM.search(page.text):
        return []
    problems = []
    caveats = [
        s for s in page.sentences() if "environment" in s.lower() and _APPROVAL_CAVEAT.search(s)
    ]
    if not caveats:
        problems.append(
            f"{page.rel}: claims a human approval gate but never says that an environment "
            "which was never created is auto-created with no reviewers, and therefore gates "
            "nothing. That is the one sentence an operator has to read."
        )
    setup = _text_of(page.section("setup"))
    if not _APPROVAL_CAVEAT.search(setup):
        problems.append(
            f"{page.rel}: the One-Time Setup section tells an operator to create the "
            "environment without warning that a missing one is auto-created with no "
            "reviewers, which is where that warning has to be"
        )
    return problems


# --- The pins ----------------------------------------------------------------


def _page(path: Path) -> Page:
    return Page(path)


class TestTheFlowSectionMatchesTheWorkflow:
    def test_the_cue_map_covers_exactly_the_workflow_jobs(self):
        # Fail-closed drift detection for the hand-written cue map above. If a
        # job is added or renamed, this fails and someone decides how the page
        # should describe it, instead of the check quietly stopping at the jobs
        # it happens to know.
        assert set(JOB_FLOW_CUES) == set(_jobs(_workflow())), (
            "release.yml's jobs and JOB_FLOW_CUES have diverged; update the map, and the page "
            "it checks, together"
        )

    def test_every_job_is_described_and_nothing_else_is(self):
        problems = _flow_problems(_page(RELEASE_PIPELINE_PAGE), _workflow())
        assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", SITE_PAGES, ids=lambda p: str(p.relative_to(SITE)))
class TestNoPageDescribesAnUnusedIndex:
    def test_an_unused_index_is_only_ever_named_as_absent(self, path):
        problems = _index_problems(_page(path), _workflow())
        assert not problems, "\n".join(problems)

    def test_no_page_invents_an_environment(self, path):
        problems = _environment_problems(_page(path), _workflow())
        assert not problems, "\n".join(problems)


class TestTheApprovalClaimCarriesItsCaveat:
    def test_the_release_page_says_what_the_approval_depends_on(self):
        problems = _caveat_problems(_page(RELEASE_PIPELINE_PAGE))
        assert not problems, "\n".join(problems)


# The One-Time Setup section exactly as it shipped at f59f206, kept verbatim so
# the sabotage below reinstates the real defect rather than a paraphrase of it.
# Two things are wrong with it: it instructs an operator to create a `testpypi`
# environment nothing uses, and its "Until this is done" callout treats both
# prerequisites as failing closed, when a missing environment does not fail at
# all.
SHIPPED_SETUP_SECTION = """<section id="setup">
                    <h2>One-Time Setup (Prerequisite)</h2>
                    <p>
                        Before the first release, two things must exist. They are configured once by
                        a maintainer with PyPI and repository-admin access.
                    </p>
                    <ol>
                        <li><strong>A Trusted Publisher</strong> for project "vfairness" on \
<strong>PyPI</strong>, bound to the owner, this repository, and workflow <code>release.yml</code>\
.</li>
                        <li><strong>GitHub Environments</strong> <code>testpypi</code> and \
<code>pypi</code>, with a <strong>required reviewer</strong> on <code>pypi</code> so a human \
approves the production publish.</li>
                    </ol>
                    <div class="callout callout-warning">
                        <div class="callout-title">Until this is done</div>
                        <p>
                            The publish jobs fail closed; the build, verify, and smoke jobs still run,
                            so the pipeline is exercised end to end without publishing.
                        </p>
                    </div>
                </section>"""


class TestTheseChecksAreThemselvesChecked:
    """Reinstate the shipped defect, verbatim, and confirm each check goes red.

    Every string below is copied from the page as it stood at f59f206. A check
    that has only ever been run against a corrected page is not known to be a
    check, which is precisely how these sentences survived the previous audit.
    """

    def _mutated(self, old: str, new: str) -> Page:
        html = RELEASE_PIPELINE_PAGE.read_text(encoding="utf-8")
        assert old in html, (
            "the text this sabotage replaces is no longer on the page, so the mutation would "
            f"be a no-op and the test would pass without testing anything: {old[:70]!r}"
        )
        return Page(RELEASE_PIPELINE_PAGE, html=html.replace(old, new))

    def _with_shipped_setup_section(self) -> Page:
        """The One-Time Setup section exactly as it shipped at f59f206.

        Taken from the live page rather than hand-typed, and swapped in for
        whatever the section says today, so this sabotage keeps working if the
        corrected section is reworded later.
        """
        current = Page(RELEASE_PIPELINE_PAGE)
        return Page(
            RELEASE_PIPELINE_PAGE,
            html=current.html.replace(current.section("setup"), SHIPPED_SETUP_SECTION),
        )

    def test_the_testpypi_job_in_the_flow_list_is_caught(self):
        page = self._mutated(
            "<h3>4. Publish to PyPI</h3>",
            "<h3>3. Publish to TestPyPI</h3>",
        )
        problems = _flow_problems(page, _workflow())
        assert problems, "a flow list naming a job the workflow does not have passed the check"
        assert any("matches no job" in p for p in problems), problems
        assert any("publish-pypi" in p and "never describes" in p for p in problems), problems

    def test_dropping_a_real_job_from_the_flow_list_is_caught(self):
        page = self._mutated(
            "<h3>5. Publish the GitHub Release</h3>",
            "<h3>4. And then it is done</h3>",
        )
        problems = _flow_problems(page, _workflow())
        assert any("github-release" in p and "never describes" in p for p in problems), problems

    def test_the_dress_rehearsal_sentence_is_caught(self):
        page = self._mutated(
            "<h3>5. Publish the GitHub Release</h3>",
            "<h3>5. Publish the GitHub Release</h3>\n<p>Uploads to TestPyPI via Trusted "
            "Publishing as a dress rehearsal. A re-run is idempotent "
            "(<code>skip-existing</code>).</p>",
        )
        problems = _index_problems(page, _workflow())
        assert problems, "a described TestPyPI upload passed the index check"
        assert any("dress rehearsal" in p for p in problems), problems

    def test_the_testpypi_is_the_rehearsal_sentence_is_caught(self):
        # The worst of the surviving lines: a claimed safety control in front of
        # an irreversible upload.
        page = self._mutated(
            "A PyPI version cannot be overwritten.",
            "TestPyPI is the rehearsal: a failure there stops the pipeline before it touches "
            "PyPI. A PyPI version cannot be overwritten.",
        )
        problems = _index_problems(page, _workflow())
        assert problems, "the 'TestPyPI is the rehearsal' claim passed the index check"

    def test_the_testpypi_first_summary_is_caught(self):
        page = self._mutated(
            "<li><strong>Trusted Publishing (OIDC).</strong>",
            "<li><strong>TestPyPI first, then PyPI</strong>, with a required human approval "
            "before the production publish.</li>\n<li><strong>Trusted Publishing (OIDC).</strong>",
        )
        assert _index_problems(page, _workflow()), "the 'TestPyPI first' summary passed the check"

    def test_the_operator_instruction_to_create_two_environments_is_caught(self):
        page = self._with_shipped_setup_section()
        problems = _environment_problems(page, _workflow())
        assert problems, "an instruction to create an unused environment passed the check"
        assert any("testpypi" in p for p in problems), problems
        # And it is caught twice over, because the same sentence describes an
        # index the pipeline does not publish to.
        assert _index_problems(page, _workflow())

    def test_the_shipped_setup_section_loses_the_auto_created_caveat(self):
        # The section as it shipped told the operator to create the environments
        # and said nothing about a missing one being created implicitly, which is
        # the sentence that decides whether the approval gate exists at all.
        problems = _caveat_problems(self._with_shipped_setup_section())
        assert problems, "a setup section with no caveat about a missing environment passed"
        assert any("One-Time Setup" in p for p in problems), problems

    def test_a_page_with_no_caveat_anywhere_is_caught(self):
        # Independent of this page's wording: an approval claim, a setup section,
        # and nothing that says what the approval depends on.
        page = Page(
            RELEASE_PIPELINE_PAGE,
            html=(
                "<h1>Release Pipeline</h1>"
                "<p>A required human approval on the <code>pypi</code> environment stands "
                "between a tag and a production publish.</p>"
                '<section id="setup"><h2>One-Time Setup</h2><ol><li>A GitHub Environment '
                "named <code>pypi</code>, with a required reviewer on it.</li></ol></section>"
            ),
        )
        problems = _caveat_problems(page)
        assert len(problems) == 2, problems
        assert any("never created" in p for p in problems), problems

    def test_the_checks_are_derived_from_the_workflow_not_from_a_word_list(self):
        """Add a TestPyPI job back to the workflow: the demand must lift.

        If the rule were a blocklist of the word "TestPyPI" it would keep firing
        after the pipeline gained the leg again, and the next maintainer would
        delete the rule rather than the false claim.
        """
        page = self._mutated(
            "<h3>5. Publish the GitHub Release</h3>",
            "<h3>5. Publish the GitHub Release</h3>\n<p>Uploads to TestPyPI via Trusted "
            "Publishing as a dress rehearsal.</p>",
        )
        workflow = _workflow()
        assert _index_problems(page, workflow), "control: the sentence must fail as things stand"
        workflow["jobs"]["publish-testpypi"] = {
            "runs-on": "ubuntu-latest",
            "environment": "testpypi",
            "steps": [
                {
                    "uses": f"{PUBLISH_ACTION}@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
                    "with": {"repository-url": "https://test.pypi.org/legacy/"},
                }
            ],
        }
        assert _publish_targets(workflow) == {"pypi", "testpypi"}
        assert not _index_problems(page, workflow), (
            "the index rule kept refusing a TestPyPI sentence after the workflow regained a "
            "TestPyPI publish job, so it is a word list rather than a check"
        )
        assert not _environment_problems(page, workflow)

    def test_an_unrecognised_index_host_is_not_silently_ignored(self):
        # Fail closed: an unknown upload target must raise, not shrink the set of
        # "used" indexes and thereby license more prose.
        workflow = _workflow()
        for job in workflow["jobs"].values():
            for step in job.get("steps") or []:
                if PUBLISH_ACTION in str(step.get("uses", "")):
                    step["with"]["repository-url"] = "https://packages.example.com/legacy/"
        with pytest.raises(ValueError):
            _publish_targets(workflow)


class TestTheHealthyPageIsUnchangedByTheseRules:
    """Control: the corrected page passes every rule, and for the right reason.

    A rule set that refused everything would also refuse the defect, so the pins
    above are only worth something alongside this.
    """

    def test_the_workflow_publishes_to_pypi_only(self):
        assert _publish_targets(_workflow()) == {"pypi"}

    def test_the_workflow_declares_exactly_one_environment(self):
        assert _declared_environments(_workflow()) == {"pypi"}

    def test_the_page_still_names_the_pypi_environment(self):
        # The rules must not have been satisfied by deleting the subject.
        assert "pypi" in _claimed_environments(_page(RELEASE_PIPELINE_PAGE))

    def test_the_page_still_states_the_testpypi_leg_is_gone(self):
        page = _page(RELEASE_PIPELINE_PAGE)
        mentions = [s for s in page.sentences() if re.search(r"\btestpypi\b", s, re.IGNORECASE)]
        assert mentions, (
            "the page no longer mentions TestPyPI at all; a reader who has seen the old page, "
            "or the old README, is left to guess whether the rehearsal still exists"
        )
        assert all(_NEGATION_CUE.search(s) for s in mentions), mentions

    def test_every_flow_step_is_still_described(self):
        assert len(_flow_headings(_page(RELEASE_PIPELINE_PAGE))) == len(_jobs(_workflow()))


def test_the_page_reader_extracts_what_these_rules_assume():
    """Pin the instrument, before the rules that depend on it.

    If the reader returned nothing (a parser change, a page renamed), every rule
    above would pass by having no text to judge.
    """
    page = _page(RELEASE_PIPELINE_PAGE)
    assert len(page.text) > 2000, "the page reader produced almost no text"
    assert "Release Pipeline" in page.text
    assert "pypi" in page.code_tokens
    # Script bodies are skipped: the page carries inline JS, which is not prose.
    assert "addEventListener" not in page.text
    sentences = page.sentences()
    assert len(sentences) > 40, len(sentences)


def _readable(paths: Iterable[Path]) -> int:
    return sum(1 for p in paths if p.read_text(encoding="utf-8"))


def test_the_sweep_actually_reads_every_shipped_page():
    # The parametrised classes above would report success if SITE_PAGES were
    # empty or unreadable.
    assert _readable(SITE_PAGES) == len(SITE_PAGES) >= 10
