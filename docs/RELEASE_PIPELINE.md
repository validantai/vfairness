# Release Pipeline

How a new version of vfairness gets from a commit to PyPI, what a maintainer
does, and how quality is checked at each step. This is the operator runbook; the
overview a reader wants, with the readiness bar and what is still open, is on
[Quality and Hardening](QUALITY_AND_HARDENING.md).

Releases are automated and tag-triggered. Nothing is ever uploaded from a
laptop; the only upload path is the workflow at `.github/workflows/release.yml`,
and it publishes to PyPI only after a required human approval.

There is no TestPyPI rehearsal. It was removed on 2026-08-28, because a second
account is real overhead for a solo maintainer. That removal took a rehearsal
upload out of the pipeline, so the gates that remain are the only things
standing between a tag and an immutable upload. They are listed below, and none
of them may be weakened.

## Which repository publishes

**Releases are cut only from the public repository `validantai/vfairness`.**
Development happens in a private monorepo, where this library lives in a
`vfairness/` subtree and a per-release snapshot is exported to the public
repository; the tag that triggers a publish is the one on the public repository.

Trusted Publishing binds to exactly one repository, and the point of a fairness
assurance library is that its provenance points at source a consumer can read,
so the public repository is the only one bound to the PyPI publisher.

This is enforced, not merely documented. Every publishing job carries the
condition

```yaml
if: github.event_name == 'push'
    && startsWith(github.ref, 'refs/tags/v')
    && github.repository == 'validantai/vfairness'
```

so a publish requires all three at once: a pushed ref rather than a manual run,
a `v`-prefixed tag, and the public repository. The copy of this workflow that
sits in the private monorepo is therefore structurally incapable of publishing:
the repository clause is false there by construction, so a tag pushed in the
monorepo runs build, verify and smoke as a rehearsal and then stops.

The event clause is load-bearing and is not redundant with the tag clause: a
`workflow_dispatch` can be aimed at a tag ref, and `github.ref` is then
`refs/tags/v...`, so on the ref clause alone a manual run would reach the
upload. Do not remove any of the three to "make a release work" from another
repository or from a manual run. Before these guards existed, the only thing
preventing a publish from the private repository was that no Trusted Publisher
happened to be bound to it.

## At a glance

- **Tag-triggered, on the public repository.** Pushing a `vX.Y.Z` tag to
  `validantai/vfairness` runs the release workflow; normal pushes never publish,
  and a tag on any other repository never publishes.
- **A required human approval** on the `pypi` environment before the production
  publish, provided that environment carries a required reviewer (see
  [One-time setup](#one-time-setup-prerequisite): an environment that was never
  created gates nothing).
- **Trusted Publishing (OIDC).** No long-lived PyPI API token is ever stored; the
  upload uses short-lived, workflow-scoped credentials.
- **Every build is verified** (`twine check`, `check-wheel-contents`), gets a
  CycloneDX SBOM of the built wheel, and is installed into a clean environment
  and smoke-tested before any publish.
- **Build provenance attestations** are attached to the PyPI upload, requested
  explicitly on the publish step rather than left to the publish action's
  default.
- **The SBOM is published**, on the GitHub Release next to the wheel, so a
  consumer can actually fetch it.

## How a release flows

`release.yml` runs four jobs in order; each runs only if the previous ones
passed, so a failure anywhere stops the release before it reaches PyPI.

1. **Build and verify distribution.** First asserts that the pushed tag matches
   `__version__`, so a mistyped tag cannot produce an artifact labelled with a
   version nobody tagged. Then builds the sdist and wheel with
   `python -m build`, gates on `twine check` (metadata and README render)
   and `check-wheel-contents` (nothing missing or stray in the wheel), and
   generates a CycloneDX SBOM. The SBOM is generated from a throwaway virtual
   environment that contains **only the freshly built wheel and its resolved
   runtime closure**, so it describes the package rather than the CI runner, and
   a following step fails the build if the SBOM is missing any declared runtime
   dependency or contains any build or publish tooling. The artifacts are
   uploaded for the later jobs.
2. **Clean-room install smoke test (VB-REL-3).** Installs the freshly built wheel
   into a fresh environment and exercises the public API from outside the source
   tree, so a missing data file, a wrong entry point, an unshipped subpackage, or
   a missing runtime dependency blocks the release. This catches packaging defects
   the source-tree test suite cannot see.
3. **Publish to PyPI.** Runs only on a `vX.Y.Z` tag on the public repository, and
   only after a maintainer approves the `pypi` environment, then uploads to PyPI
   via Trusted Publishing with provenance attestations. A PyPI version number
   cannot be reused, so this job is one-way.
4. **Publish the GitHub Release.** Creates the release for the tag and attaches
   the SBOM alongside the sdist and wheel, so the SBOM is reachable by a
   consumer rather than expiring as a workflow artifact.

## Cutting a release

For a maintainer, a release is five small steps; the pipeline does the rest.
**Steps 0 to 3 happen wherever the code is developed. Step 4, the tag, must
happen on the public repository `validantai/vfairness`, because that is the only
repository whose publish jobs can run.**

This page is the runbook. The reader-facing picture of what the release rests on,
including where the readiness bar currently stands, is one page:
[Quality and Hardening](QUALITY_AND_HARDENING.md).

0. **Run the readiness gate and read it.**

   ```
   python scripts/release_gate.py
   ```

   ADDED 2026-09-28, because this runbook did not mention it: on that date the gate
   returned NOT BETA READY while these steps said nothing about it, so a maintainer
   following the runbook exactly could cut a release without ever reading it.

   **What it checks for a beta (eight conditions, all required):** every public
   code unit executed and its result recorded (B1) and published (B1b); no open
   defect in code that returns a fairness number (B2) or in the second-round audit
   (B2b); every open defect ranked (B3); the capability census clear (B5); the same
   answer as an established library wherever one computes the number (B6); and no
   invented number on broken data from any capability that measures (B7). On
   2026-10-01 it reports **BETA READY**, all eight met.

   **What it checks for 1.0, and a beta does not need:** a second, independent
   examiner confirming every grade (G2). That moved out of the beta bar on
   2026-10-01 by the maintainer's decision, because B6 and B7 already re-check
   every capability that returns a number by running it. It stays open, and it is
   stated on the quality page, until the remaining units are re-checked.

   The gate is NOT the test suite, and this step is not step 1. A failing
   criterion is not a failing test: the suite passes. The gate measures how much
   of the public surface has been adversarially audited and how much nobody has
   examined yet, so it fails on coverage of the audit rather than on broken
   behaviour. Read the two separately, and do not "fix" a red gate by lowering
   it.

   It is deliberately NOT wired into the workflow as a blocking job, so running it
   is part of this step and not optional. If a release is ever cut against a red
   beta gate, the release notes must name the failing criterion ids, because the
   page a user reads carries the same numbers and the two must not disagree.

1. **Confirm the quality gates are green** on `main` (see
   [Quality and Hardening](QUALITY_AND_HARDENING.md)).

   **This is enforced now, and no longer only remembered.** The release
   workflow's first job, `suite-green-on-this-commit`, reads the check runs for
   the exact commit the tag points at and refuses to build unless every
   `Full suite + coverage (...)` check concluded `success`. Nothing downstream
   runs if it does not.

   It fails closed. A check that is missing, queued, cancelled or skipped is not
   a pass, it is a could-not-check, and at a publish boundary that blocks. Zero
   matching checks is fatal rather than silent, because a gate that examines
   nothing passes forever.

   Why it was added: on 2026-09-10 the full suite was RED on `main` for eleven
   consecutive commits while local runs reported 8246 passed and 0 failed, and
   nothing would have stopped a tag being cut from any of them. A published
   version is immutable, so noticing afterwards is not a recovery.
2. **Bump the version.** There is a single source of truth in
   `src/vfairness/__init__.py` (`__version__`); the build reads it from there.
   The workflow refuses to build if the tag does not match this value.
3. **Flip the unreleased tense, in ONE commit, immediately before tagging.**
   Run `python scripts/cut_release.py --date <YYYY-MM-DD>`; it does the whole
   flip or none of it, and `--check` shows what it would do without writing.

   > **Not in the published repository.** `cut_release.py` is release tooling
   > for the development monorepo and is excluded from the public export, so
   > this step is here to document what the maintainers do, not as an
   > instruction a reader of the published repository can follow. The result of
   > the flip IS published: it is what turns the unreleased CHANGELOG heading
   > and the "not yet published" README badge into the released wording.

   Three files carry the tense in PROSE, and each becomes a false statement the
   moment the artifact exists. Two of the three are effectively permanent once
   wrong: a PyPI version description is IMMUTABLE, so publishing as-is leaves the
   version's own project page saying forever that it was never published.

   **It is not only those three.** Measured 2026-10-01 with `--check`: 20 exact
   edits plus 23 shared navigation replacements across 25 files (it was 24 files on
   2026-09-09). The anchors go stale whenever a page is reworded: on 2026-10-01
   three of them no longer matched and the script refused, correctly, until they
   were updated. Run `--check` well before release day, not on it. Twenty `docs/site` pages carry a shared navigation literal
   ("PyPI (0.1.0 not yet published)"), and further claims sit in a callout title,
   a table cell, a `version-date` span and several prose paragraphs. The gate
   below checks EVERY `docs/site` page on the released side, so editing only the
   three prose files leaves the suite red with twenty-odd places left to find, at
   the one moment in a release when attention is thinnest. That is what the
   script exists to prevent, and why it refuses to write anything unless every
   anchor it knows matches exactly once.

   | File | While unreleased | At the cut |
   | --- | --- | --- |
   | `CHANGELOG.md` | `## [X.Y.Z] - UNRELEASED`, deliberately undated, plus a paragraph saying no tag and no artifact exist | `## [X.Y.Z] - <release date>`. Ships in the sdist. |
   | `README.md` | status badge "not yet published", "There is no install command that works today", "has **not** shipped" | the real install command, released tense. This file is the PyPI long description. |
   | `CITATION.cff` | `date-released` absent, with the reason in a comment | add `date-released` with the real publication date. Academic citations are generated from this. |

   There is no separate `[Unreleased]` changelog section to move entries out of.
   While `X.Y.Z` is itself unreleased, everything accumulates under its own
   heading, so that no work sits under one heading while shipping inside another.

   **This step is enforced, not remembered.** `tests/test_release_tense.py`
   reads the release state out of the changelog heading and requires
   `README.md`, `CITATION.cff` and the published pages to agree with it, in both
   directions: dating the changelog alone turns the suite red until the other two
   follow, and deleting the honest warnings before the tag turns it red as well.
   `tests/test_docs_truth.py::test_changelog_top_section_is_the_unreleased_version`
   additionally refuses a date on the heading while the project is unreleased, so
   it goes red the moment you date it. That is intentional and it is the gate for
   this step: update that test in the SAME commit so it asserts the released
   form. Do not reach for `--no-verify`; a gate you can only satisfy by bypassing
   it is a broken gate, and here it is telling you the step is not finished.
4. **Tag and push, on `validantai/vfairness`.** Create an annotated tag and push
   it:

   ```bash
   # Run this in a clone of validantai/vfairness, not in the private monorepo.
   git tag -a v0.1.0 -m "vfairness 0.1.0"
   git push origin v0.1.0
   ```

   Build, verify and smoke run automatically. When the run reaches the `pypi`
   environment it pauses for approval, provided that environment carries a
   required reviewer; approve it to publish to PyPI. The GitHub Release with the
   SBOM is created after that.

   **Maintainers working in the private monorepo do not tag there.** The
   snapshot export is what puts the commit and the tag on the public repository,
   and it is the step that turns steps 1 to 3 into the tag in step 4. A tag
   pushed in the monorepo runs build, verify and smoke and then stops,
   publishing nothing.

   The export procedure is documented in the private monorepo's own
   `docs/PUBLISHING-RUNBOOK.md`. That file is internal and is deliberately not
   part of this repository, so if you are reading this in `validantai/vfairness`
   there is no such path here to follow; everything the published artifact
   depends on is on this page.

**Why a plain `0.1.0` and not a PEP 440 pre-release.** The beta ships as a
normal SemVer 0.x version with the `Development Status :: 4 - Beta` classifier,
so `pip install vfairness` finds it without the `--pre` flag (a pre-release
suffix would hide the package from every default resolve, which is the wrong
trade for a library seeking testers). SemVer already reserves API stability for
`1.0.0`. PEP 440 pre-release suffixes (such as `1.0.0b1`) remain available for
rehearsing a specific future release; those do require `pip install --pre`.

## One-time setup (prerequisite)

Before the first release, two things must exist. They are configured once by a
maintainer with PyPI and repository-admin access.

1. **A Trusted Publisher** for project "vfairness" on **PyPI**,
   bound to owner `validantai`, repository `vfairness`, workflow
   `release.yml` and environment `pypi`. The project already exists on PyPI (a
   0.0.1 name reservation uploaded by the maintainer in March 2026, which contains
   none of this code), so the publisher is added to that existing project, not
   created as a pending publisher. Bind it to the public repository only; the workflow's
   repository guard means binding any other repository would have no effect
   anyway.
2. **A GitHub Environment** named `pypi` **in `validantai/vfairness`**,
   with a **required reviewer** on it, so a human approves the production
   publish. The reviewer must be an account that will still exist on release day:
   an approval nobody can give blocks the release just as surely as a missing
   gate lets one through. Its deployment-branch rule must admit `v*` tags, since
   the release is started by a tag.

   This one has to be created deliberately, and then checked. The publish job
   declares `environment: pypi`; if no environment of that name exists, GitHub
   does not fail the job, it creates the environment implicitly on first use
   with no reviewers and no protection rules, and the run proceeds straight to
   the upload. The approval pause described throughout this page then does not
   happen, and nothing in the workflow file or the run log distinguishes that
   case from a correctly configured one. The repository's Environments settings
   are the only place the difference is visible, so confirm the required
   reviewer is present there before the first tag.

The two prerequisites fail differently, and the difference matters. Until the
Trusted Publisher exists the publish job fails closed: build, verify and smoke
still run, so everything up to the upload is exercised without publishing. A
missing environment does not fail closed at all.

## How quality is checked

Quality is enforced in two layers: continuous integration on changes, and
checks on the exact artifact at release time. Which checks run **in this
repository** is worth stating precisely, because it is not identical to the set
of gates the library is developed under.

**What runs here.** This repository carries the workflows in
`.github/workflows/`, and that directory is the authoritative list. Alongside
`release.yml` there are five CI workflows. All of them run on pushes to `main`
and on pull requests; `security.yml` and `secret-scan.yml` also run on a weekly
schedule.

- **`tests.yml` — full suite and coverage on Python 3.11, 3.12 and 3.13**, so
  the support window the package advertises is verified rather than claimed. The
  library is installed with its optional extras, coverage is branch-inclusive
  behind a `--cov-fail-under=55` floor, and `VFAIRNESS_REQUIRE_BACKENDS=1` turns
  a missing optional backend into a named failure instead of a silent skip.
  There is deliberately no `paths:` filter, so no change can skip its own gate.
- **`quality.yml` — lint, types and new-code coverage.** `ruff check` and
  `ruff format --check` over `src` and `tests`, and `mypy src` on a pinned mypy,
  both blocking. On pull requests, a `diff-cover` job additionally requires 80%
  coverage of the lines that pull request adds or changes.
- **`security.yml` — Bandit, pip-audit and CodeQL.** `bandit -r src -ll` and a
  `pip-audit --strict` scan of the resolved dependency set. The CodeQL job is
  guarded on
  `github.event.repository.private == false`, so unlike in the private
  development repository, where it skips because code scanning needs a public
  repository, **CodeQL genuinely runs here**. It is skipped for pull requests
  from forks, whose read-only token cannot upload results; a fork's code is
  still analysed once it lands on `main`.
- **`fairness-checks.yml` — the fairness gate and the dependency floor.** The
  first job installs every optional extra the suite exercises and runs the whole
  suite on Python 3.11 behind its own `--cov-fail-under=48` floor, then
  exercises the library's own `ModelFairnessGate` on synthetic data and posts
  the report card as a pull-request comment. That last step demonstrates the
  gate; it is not an assessment of the library. The second job resolves the
  lowest direct dependency versions declared in `pyproject.toml` and runs the
  suite against them, so a dependency floor that is quietly wrong fails here
  rather than on a user's machine.
- **`secret-scan.yml` — gitleaks over the full history and all refs**, from a
  pinned, checksum-verified binary. It carries a positive control: before
  trusting a clean result it plants a secret the ruleset is not entitled to
  allowlist and fails the job if gitleaks does not flag it, because a clean
  result from a broken scanner looks exactly like a clean result from a working
  one.

One limit worth stating: whether any of these is a *required* status check is a
repository setting, not something a workflow file can assert, so this page does
not claim they block a merge.

**What does not run here.** vfairness is developed in a private monorepo, which
carries a few workflows that are not part of the exported subtree and therefore
do not run in this repository: an OpenSSF Scorecard workflow, mutation testing
(informational, not blocking; see [Mutation testing](MUTATION_TESTING.md)), and
dedicated cross-library metric parity and Pulse output contract workflows that
publish evidence artifacts.

Be careful with those last two: what is missing here is the dedicated workflow
and its evidence artifact, not the checking. The parity and Pulse test modules
are part of the suite, and `tests.yml` runs the suite whole with the `parity`
extra installed, so they execute here on every push and pull request like any
other test.

The full inventory of what the library is developed under is on the
[Quality and Hardening](QUALITY_AND_HARDENING.md) page, which covers both
repositories. Recorded 2026-08-28; `.github/workflows/` is the live answer.

**At release time**, the pipeline runs its own checks on the exact artifact that
will ship: the tag-versus-`__version__` assert, `twine check`,
`check-wheel-contents`, the CycloneDX SBOM and its scope check, and the
clean-room install smoke test. `workflow_dispatch` is the dry run: it builds and
smokes and cannot publish, because both publishing jobs require a pushed tag and
a dispatch is not a push.

## Security of the pipeline

- **No stored PyPI token.** Publishing uses OIDC short-lived credentials via PyPI
  Trusted Publishing, so there is no long-lived secret to leak or rotate.
- **Provenance attestations** are attached to the PyPI upload, so consumers can
  verify the artifact was built by this workflow.
- **A CycloneDX SBOM** records the dependency set of the built wheel: the
  declared runtime dependencies and their resolved transitive closure, and
  nothing from the CI runner. A content check in the same job fails the build if
  a declared dependency is missing or any build or publish tooling appears, so
  the SBOM cannot quietly go back to describing the runner. It is attached to
  the GitHub Release, so a consumer can fetch it.
- **Publishing is bound to one repository, one event and one ref shape.** Every
  publish job is guarded on `github.repository`, so no other repository can
  publish this package even if a tag is pushed there, and on
  `github.event_name == 'push'` plus a `refs/tags/v` ref, so no manual run can
  publish either.
- **A required human approval** stands between a tag and a production publish.
  That is a property of the `pypi` environment rather than of this workflow: an
  environment with no required reviewer, including one GitHub created
  implicitly, imposes no pause. See [One-time setup](#one-time-setup-prerequisite).
- **Least-privilege tokens.** The workflow runs with `contents: read`;
  `id-token: write` is granted only on the PyPI publish job that needs it, and
  `contents: write` only on the job that creates the GitHub Release.

## Versioning and stability

- **Semantic Versioning**, with a single source of version truth in
  `src/vfairness/__init__.py`.
- **The beta line is plain SemVer 0.x** (`0.1.0`, `0.2.0`, ...) carrying the
  Beta trove classifier; PEP 440 pre-release suffixes stay available for
  rehearsals and install with `pip install --pre`.
- The **frozen public surface** and deprecation policy govern what a version bump
  may change (see [API_STABILITY.md](API_STABILITY.md)); every change is announced
  in the changelog.

## If something goes wrong

- **PyPI releases are immutable**: a version cannot be overwritten. To correct a
  bad release, publish a new patch version and, if needed, **yank** the bad one on
  PyPI (yank hides it from new resolves without breaking existing pins).
- **`workflow_dispatch` is the rehearsal**: it executes build and smoke and
  **never touches PyPI**, because both publishing jobs require
  `github.event_name == 'push'` and a dispatch is not a push. That holds even if
  the dispatch is aimed at a tag ref. Re-run it as often as you like. It does
  not exercise the upload itself, so it cannot tell you whether the Trusted
  Publisher binding or the environment approval are configured correctly.
