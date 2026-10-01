"""Release-mechanics pins (pre-release audit, 2026-08-27).

Two blockers that turn a green-looking release into a broken one:

- The `mcp` extra was uncapped (`mcp>=1.2.0`), so a fresh resolver picks mcp
  2.x. mcp 2.0.0 removed `mcp.server.fastmcp`, which is the only import
  `vfairness.mcp.server` makes, so the shipped `vfairness-mcp` console script
  is dead on essentially every fresh install. Worse, the import guard caught
  the resulting ModuleNotFoundError and told the reader to install the extra
  they had already installed, hiding the real cause (a version incompatibility).

- Both release.yml copies invoked `cyclonedx-py environment ... --outfile`,
  and cyclonedx-bom has no `--outfile` flag (it is `-o` / `--output-file`), so
  the SBOM step aborts the build job and nothing is ever published.

Two more, from the fourth-iteration audit (findings 1 and 4):

- The private monorepo carried a live, tag-triggered publish path that the
  runbook says must never publish. Nothing structural stopped it; it merely
  failed at the OIDC exchange because no Trusted Publisher happened to be bound
  to that repository, which is luck, not a guard. Every publishing job now
  carries a `github.repository` condition, and these tests refuse a workflow
  that lost it.

- The SBOM step ran `cyclonedx-py environment` with no path argument, so it
  snapshotted the CI runner (71 components, including twine, keyring and pip)
  instead of the package (15 components: the 5 declared runtime dependencies,
  their closure, and vfairness itself). It was also uploaded only as a workflow
  artifact, which expires and which no consumer can reach.

Three more, from the fifth-iteration audit:

- The publish step passed only `packages-dir: dist`, so the build-provenance
  attestations the header promises consumers were left to whatever the pinned
  action SHA defaults to. A security property has to be produced by a line, not
  by a default a version bump can flip.

- The guard was `startsWith(github.ref, 'refs/tags/v') && github.repository ==
  ...` under a comment claiming a workflow_dispatch "CANNOT publish, because the
  tag condition is false without a tag". workflow_dispatch takes a ref, and both
  the UI ref selector and the dispatches API accept a TAG, so that condition was
  TRUE for a dispatched run and the manual "dry run" reached an immutable PyPI
  upload. The comment described an enforcement that did not exist.

- `github-release` runs `gh release view/create/upload` in a job with no
  actions/checkout and no GH_REPO, so gh had no repository to resolve. It would
  fail AFTER publish-pypi's irreversible upload, leaving a PyPI release with no
  GitHub Release and no consumer-reachable SBOM.

Six more, from the sixth-iteration audit (2026-08-28), every one the same shape:
a gate that is claimed but never read, or a check that cannot run where it counts.

- This file required BOTH release.yml copies unconditionally. In the exported
  public repo the library subtree IS the repo root, so LIB_ROOT.parent is the
  runner's work directory and holds no .github at all, and all 39 `[monorepo]`
  parametrisations raised FileNotFoundError. Measured on a pristine export tree:
  "39 failed, 46 passed". The public repository's very first push to main would
  have gone red on a test file that is green here. The copies are now resolved by
  LAYOUT, and the layout is itself pinned so this cannot silently become a run
  that checks nothing.

- release.yml's own header names five gates and says none of them may be
  weakened. Three had no check at all: `environment: pypi` (which the header
  calls the LAST human checkpoint before an immutable upload), the `smoke`
  dependency, and the `twine check` / `check-wheel-contents` steps. Deleting all
  three at once left this file green.

- The `smoke` job claimed to catch "a missing data file, a wrong entry point".
  scripts/wheel_smoke.py reads neither: with the shipped templates and legal
  rules deleted from site-packages it still printed "SMOKE OK" and exited 0. The
  workflow now carries a step that loads both through the package's own loader
  and runs a console script, and this file pins that the step is still there.

- tests.yml installed no extra carrying PyYAML, and this file opens with
  `pytest.importorskip("yaml")`, so in the very workflow advertised as "the full
  suite on 3.11, 3.12 and 3.13" every test here collapsed into one skip line.

- scripts/export-vfairness-to-public.sh replaced the public repo's tree wholesale
  and pushed, so a merged community pull request was staged as a deletion and
  reverted with the staged diff never printed and no prompt.

- Its "tag already exists" guard read a `--depth 1` clone, which carries only the
  tags pointing at the fetched tip, so every OLDER release tag was invisible to
  it; and main was pushed BEFORE the tag, so the rejected tag push landed after
  main had already been overwritten.

These tests read the shipped configuration, and execute the workflow's own tag
guard, its own SBOM content check, its own publish conditions, and the export
script's own tag and deletion guards, rather than trusting that someone
remembered.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent

PYPROJECT = LIB_ROOT / "pyproject.toml"
UV_LOCK = LIB_ROOT / "uv.lock"
TESTS_WORKFLOW = LIB_ROOT / ".github" / "workflows" / "tests.yml"
SMOKE_SCRIPT = LIB_ROOT / "scripts" / "wheel_smoke.py"

# WHICH REPOSITORY IS THIS? Answered by the script that BUILDS the public repo.
# scripts/export-vfairness-to-public.sh sits at the MONOREPO root, outside the
# vfairness/ subtree, so `git archive HEAD:vfairness` structurally cannot carry
# it into an export tree. Its presence therefore means "a monorepo encloses this
# library"; its absence means "this checkout IS the repository". Nothing weaker
# works: on Actions the public checkout is /home/runner/work/vfairness/vfairness,
# so even the directory NAME matches the monorepo layout.
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export-vfairness-to-public.sh"
IN_MONOREPO = EXPORT_SCRIPT.is_file()

# The copies of the publish pipeline this checkout actually has: the monorepo's
# (paths prefixed with vfairness/) and the library's (root-relative). Both must
# be fixed in the monorepo, or the defect simply moves to whichever repo cuts the
# tag - but only the library's exists once the subtree has been exported.
#
# RESOLVED BY LAYOUT, NOT ASSUMED. Both paths used to be required unconditionally,
# and in the exported tree LIB_ROOT.parent is the runner's work directory, which
# has no .github: 39 of the file's 85 tests then raised FileNotFoundError on the public
# repo's first push. The library copy is required in EVERY layout, and
# test_the_workflow_copies_under_test_match_this_checkout pins the resolution, so
# this cannot decay into a parametrisation that inspects nothing.
RELEASE_WORKFLOWS = {"library": LIB_ROOT / ".github" / "workflows" / "release.yml"}
if IN_MONOREPO:
    RELEASE_WORKFLOWS["monorepo"] = REPO_ROOT / ".github" / "workflows" / "release.yml"

monorepo_only = pytest.mark.skipif(
    not IN_MONOREPO,
    reason=(
        "scripts/export-vfairness-to-public.sh exists only in the private monorepo; "
        "in the exported public repo there is no export script to check"
    ),
)


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _mcp_requirements() -> list[str]:
    return list(_pyproject()["project"]["optional-dependencies"]["mcp"])


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_steps(path: Path) -> list[dict]:
    return list(_workflow(path)["jobs"]["build"]["steps"])


def _run_scripts(path: Path) -> list[str]:
    return [str(step["run"]) for step in _build_steps(path) if "run" in step]


def _executable_lines(script: str) -> str:
    """The script with shell comment lines stripped.

    The fixed steps carry a comment naming the bad flag so the mistake is not
    made again; that comment must not itself read as the mistake.
    """
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


# --- Which copies are under test is derived, so pin the derivation -----------


def test_the_workflow_copies_under_test_match_this_checkout():
    """Three states, never two: monorepo, exported repo, and "no copy at all".

    In the monorepo BOTH release.yml copies must be present and parametrised; in
    the exported public repo only the library's exists, and its ABSENCE is a
    failure rather than a skip. The one outcome that must be impossible is an
    empty or library-less parametrisation, which would turn every parametrised
    test in this file into a vacuous pass while reading as green.
    """
    assert "library" in RELEASE_WORKFLOWS, (
        "the library's own release.yml dropped out of the parametrisation; every "
        "check below would then inspect only the monorepo copy, or nothing"
    )
    assert RELEASE_WORKFLOWS["library"].is_file(), (
        f"no release workflow at {RELEASE_WORKFLOWS['library']}; the public repo "
        "publishes through that file and nothing here can read it"
    )
    if IN_MONOREPO:
        assert sorted(RELEASE_WORKFLOWS) == ["library", "monorepo"], sorted(RELEASE_WORKFLOWS)
        assert RELEASE_WORKFLOWS["monorepo"].is_file(), (
            f"{EXPORT_SCRIPT} says this is the private monorepo, but its own "
            f"{RELEASE_WORKFLOWS['monorepo']} is gone, so the copy that must NEVER "
            "publish is no longer being checked"
        )
    else:
        # The exported tree. LIB_ROOT.parent is the runner's work directory; a
        # release.yml found up there would not be this project's.
        assert sorted(RELEASE_WORKFLOWS) == ["library"], sorted(RELEASE_WORKFLOWS)


# --- Structural checks, written against a parsed workflow dict ---------------
#
# These take the workflow as a dict rather than a path so that the sabotage
# tests below can hand them a deliberately broken workflow and prove the check
# goes red. A checker that has only ever been run against the good file is not
# known to be a checker at all.

PUBLIC_REPO = "validantai/vfairness"
PUBLISH_ACTION = "pypa/gh-action-pypi-publish"

# THE REPOSITORY THAT MUST NEVER PUBLISH, WRITTEN AS A CLASS RATHER THAN A NAME.
# This file is exported to the public repo, and until 2026-09-10 it spelled the
# private development monorepo's own slug out four times in order to assert that
# no workflow guard names it. The assertions were right and the spelling was the
# leak: the export tree carried the private repo's address on a public
# repository, which is exactly the recon class the export gate's denylist exists
# for. That slug is on the denylist now, so a literal here would abort the
# export on this file.
#
# Nothing weakened. The guard under test pins github.repository to ONE exact
# value, so any non-public slug exercises the same branch, and the checks below
# now assert on the class (every repository the guards name must be the public
# one) instead of on a single forbidden instance, which also catches a guard
# pointed at some third repository nobody thought to hard-code.
NOT_THE_PUBLIC_REPO = "example-org/private-monorepo"

#: Every `github.repository == '<slug>'` (or `!=`) comparison in a guard.
_REPOSITORY_COMPARISON = re.compile(r"github\.repository\s*[=!]=\s*'([^']*)'")


def _repositories_named(text: str) -> set[str]:
    """Every repository slug a piece of workflow text compares against."""
    return set(_REPOSITORY_COMPARISON.findall(text.replace('"', "'")))


# Matches the COMMAND, not the string "cyclonedx-py" wherever it appears. The
# SBOM content check names the component "cyclonedx-python-lib", which contains
# "cyclonedx-py" as a substring; a plain `in` test therefore mistook that check
# for a second SBOM generation step and reported it as missing its flags.
_CYCLONEDX_CALL = re.compile(r"(?:^|\s)cyclonedx-py\s")


def _invokes_cyclonedx(script: str) -> bool:
    return _CYCLONEDX_CALL.search(script) is not None


def _all_run_scripts(workflow: dict) -> list[str]:
    return [
        str(step["run"])
        for job in workflow["jobs"].values()
        for step in (job.get("steps") or [])
        if "run" in step
    ]


def _publishing_jobs(workflow: dict) -> dict[str, dict]:
    """Jobs that can push the package out, by any route.

    Detected by what a step actually does, not by the job's name: uploading a
    distribution to a package index, or creating/uploading a public GitHub
    Release. A renamed job must not escape the guard.
    """
    found: dict[str, dict] = {}
    for name, job in workflow["jobs"].items():
        steps = job.get("steps") or []
        uses = " ".join(str(step.get("uses", "")) for step in steps)
        runs = " ".join(str(step.get("run", "")) for step in steps)
        if PUBLISH_ACTION in uses or "gh release create" in runs or "gh release upload" in runs:
            found[name] = job
    return found


def _repository_guard_problems(workflow: dict) -> list[str]:
    """Every publishing job must pin the repository AND require a version tag.

    Checked per job, deliberately, rather than accepting a guard inherited
    through `needs:`. A `needs:` edge is not a permission boundary: reordering
    or dropping one would silently re-enable publishing from the wrong repo.
    """
    jobs = _publishing_jobs(workflow)
    problems: list[str] = []
    if not jobs:
        problems.append(
            "no publishing job detected at all; either publishing moved to a route this "
            "check does not know about, or the check is looking at the wrong file"
        )
    for name, job in sorted(jobs.items()):
        condition = str(job.get("if", "")).replace('"', "'")
        if not condition:
            problems.append(
                f"{name}: no job-level `if:` guard, so whichever repository holds this "
                "workflow can publish"
            )
            continue
        if f"github.repository == '{PUBLIC_REPO}'" not in condition:
            problems.append(
                f"{name}: guard does not pin github.repository to {PUBLIC_REPO}: {condition!r}"
            )
        if "refs/tags/v" not in condition:
            problems.append(
                f"{name}: guard does not require a version tag, so a workflow_dispatch "
                f"reaches the publish with no tag involved: {condition!r}"
            )
    return problems


def _sbom_scope_problems(workflow: dict) -> list[str]:
    """The SBOM must be built from the package, never from the job's own env."""
    scripts = [_executable_lines(s) for s in _all_run_scripts(workflow) if _invokes_cyclonedx(s)]
    problems: list[str] = []
    if not scripts:
        problems.append("no cyclonedx-py step at all; the release generates no SBOM")
    for script in scripts:
        match = re.search(r"cyclonedx-py\s+environment\b(?P<rest>[^\n\\]*)", script)
        if match is None:
            problems.append("cyclonedx-py is invoked, but not in `environment` mode")
            continue
        tokens = match.group("rest").split()
        if not tokens or tokens[0].startswith("-"):
            problems.append(
                "`cyclonedx-py environment` is given no interpreter path, so it snapshots "
                "the CI runner (build, twine, check-wheel-contents, cyclonedx-bom) instead "
                "of the package"
            )
        if re.search(r"pip\s+install\s+(-e\s+)?\.(\s|$)", script):
            problems.append(
                "the SBOM step installs the package into the job's own interpreter; that is "
                "the runner environment, which is exactly what must not be snapshotted"
            )
        if "venv" not in script:
            problems.append(
                "the SBOM step builds no dedicated environment, so whatever it snapshots is "
                "shared with the build tooling"
            )
        if ".whl" not in script:
            problems.append("the SBOM is not generated from the built wheel")
    return problems


def _sbom_delivery_problems(workflow: dict) -> list[str]:
    """A workflow artifact expires and no consumer can reach it."""
    scripts = _all_run_scripts(workflow)
    if not any("gh release" in s and ".cdx.json" in s for s in scripts):
        return [
            "the SBOM is never attached to a GitHub Release, so it is delivered to no "
            "consumer even in principle"
        ]
    return []


def _sbom_content_check_script(workflow: dict) -> str | None:
    """The build step that reads the produced SBOM and judges its contents."""
    for script in _all_run_scripts(workflow):
        if "vfairness-sbom.cdx.json" in script and not _invokes_cyclonedx(script):
            return script
    return None


# --- Executing a job condition, rather than spelling-checking it -------------
#
# Substring-matching an `if:` proves a clause is PRESENT. It does not prove the
# condition REFUSES anything, and the two came apart badly: the guard read
#   startsWith(github.ref, 'refs/tags/v') && github.repository == '...'
# under a comment claiming a workflow_dispatch "CANNOT publish, because the tag
# condition is false without a tag". A dispatch can be aimed at a TAG ref (the
# UI ref selector and POST .../dispatches both accept refs/tags/v0.1.0), so that
# condition was TRUE for a dispatched run and publish-pypi would have run.
# Nothing here caught it, because every check was a spelling check.

_STARTSWITH_TERM = re.compile(
    r"^startsWith\(\s*(?P<ctx>github\.[a-z_]+)\s*,\s*'(?P<value>[^']*)'\s*\)$"
)
_COMPARISON_TERM = re.compile(r"^(?P<ctx>github\.[a-z_]+)\s*(?P<op>==|!=)\s*'(?P<value>[^']*)'$")


def _lookup(reference: str, context: dict[str, str]) -> str:
    key = reference.split(".", 1)[1]
    if key not in context:
        raise ValueError(
            f"the guard reads {reference}, which this simulated event does not define; "
            "extend the context rather than letting the term evaluate to nothing"
        )
    return context[key]


def _evaluate_term(term: str, context: dict[str, str]) -> bool:
    term = term.strip()
    match = _STARTSWITH_TERM.match(term)
    if match is not None:
        return _lookup(match.group("ctx"), context).startswith(match.group("value"))
    match = _COMPARISON_TERM.match(term)
    if match is not None:
        actual = _lookup(match.group("ctx"), context)
        wanted = match.group("value")
        return actual == wanted if match.group("op") == "==" else actual != wanted
    raise ValueError(f"unrecognised term in a publish guard: {term!r}")


def _evaluate_condition(condition: str, context: dict[str, str]) -> bool:
    """Evaluate the subset of GitHub expression syntax these guards use.

    FAILS CLOSED ON SYNTAX IT DOES NOT UNDERSTAND: it raises, and never returns
    False. Returning False for an unparseable guard would make every "cannot
    publish" assertion below pass vacuously the moment someone rewrote a guard
    into a shape this cannot read, which is precisely the green-while-broken
    failure these tests exist to prevent. A raise is a loud "go extend me".
    """
    expr = condition.strip().replace('"', "'")
    if expr.startswith("${{") and expr.endswith("}}"):
        expr = expr[3:-2].strip()
    if not expr:
        raise ValueError("empty condition")
    if "||" in expr:
        raise ValueError(f"disjunction is not supported by this evaluator: {expr!r}")
    if "!" in expr.replace("!=", ""):
        raise ValueError(f"negation is not supported by this evaluator: {expr!r}")
    return all(_evaluate_term(term, context) for term in expr.split("&&"))


# The four events worth simulating. The public pushed tag is the CONTROL: a
# guard that denied it too would be "deny everything", which passes every
# refusal test and ships nothing.
PUSHED_TAG_IN_PUBLIC_REPO = {
    "event_name": "push",
    "ref": "refs/tags/v0.1.0",
    "ref_name": "v0.1.0",
    "repository": PUBLIC_REPO,
}
DISPATCH_AIMED_AT_A_TAG = {**PUSHED_TAG_IN_PUBLIC_REPO, "event_name": "workflow_dispatch"}
DISPATCH_ON_A_BRANCH = {
    **PUSHED_TAG_IN_PUBLIC_REPO,
    "event_name": "workflow_dispatch",
    "ref": "refs/heads/main",
    "ref_name": "main",
}
PUSHED_TAG_IN_PRIVATE_MONOREPO = {
    **PUSHED_TAG_IN_PUBLIC_REPO,
    "repository": NOT_THE_PUBLIC_REPO,
}


def _jobs_reached_by(workflow: dict, context: dict[str, str]) -> list[str]:
    """Publishing jobs whose own `if:` is true for a simulated event."""
    reached = []
    for name, job in sorted(_publishing_jobs(workflow).items()):
        condition = str(job.get("if", ""))
        # No condition at all means the job runs, for every event.
        if not condition.strip() or _evaluate_condition(condition, context):
            reached.append(name)
    return reached


def _dispatch_guard_problems(workflow: dict) -> list[str]:
    """workflow_dispatch is the dry run, so it must not reach a publish.

    Enforced by EXECUTING each guard against a dispatch aimed at a tag ref,
    which is the case the old comment got wrong.
    """
    problems: list[str] = []
    if not _publishing_jobs(workflow):
        problems.append(
            "no publishing job detected at all; either publishing moved to a route this "
            "check does not know about, or the check is looking at the wrong file"
        )
    for context, label in (
        (DISPATCH_AIMED_AT_A_TAG, "a workflow_dispatch aimed at refs/tags/v0.1.0"),
        (DISPATCH_ON_A_BRANCH, "a workflow_dispatch on refs/heads/main"),
    ):
        for name in _jobs_reached_by(workflow, context):
            problems.append(
                f"{name}: publishes on {label}. A dispatch is meant to be the dry run, and "
                "a PyPI upload cannot be undone."
            )
    return problems


def _attestation_problems(workflow: dict) -> list[str]:
    """Build provenance is a security claim, so it must be set, not defaulted."""
    steps = [
        step
        for job in workflow["jobs"].values()
        for step in (job.get("steps") or [])
        if PUBLISH_ACTION in str(step.get("uses", ""))
    ]
    problems: list[str] = []
    if not steps:
        problems.append(f"no {PUBLISH_ACTION} step at all; nothing publishes and nothing attests")
    for step in steps:
        inputs = step.get("with") or {}
        if "attestations" not in inputs:
            problems.append(
                "the publish step does not set `attestations:`, so whether build-provenance "
                "attestations are produced depends on the pinned action version's default, "
                "which a SHA bump can flip while the header keeps promising them"
            )
            continue
        if str(inputs["attestations"]).strip().lower() != "true":
            problems.append(
                f"the publish step sets attestations: {inputs['attestations']!r}, which turns "
                "off the provenance the header promises"
            )
    return problems


def _gh_release_repo_problems(workflow: dict) -> list[str]:
    """`gh` needs a repository, and this job has no checkout to infer one from."""
    problems: list[str] = []
    found = False
    for name, job in workflow["jobs"].items():
        steps = job.get("steps") or []
        has_checkout = any("actions/checkout" in str(s.get("uses", "")) for s in steps)
        for step in steps:
            script = str(step.get("run", ""))
            if "gh release" not in script:
                continue
            found = True
            env = {**(job.get("env") or {}), **(step.get("env") or {})}
            if "GH_REPO" in env or "GH_REPO=" in script or has_checkout:
                continue
            problems.append(
                f"{name}: runs `gh release` with neither GH_REPO nor an actions/checkout step, "
                "so gh cannot resolve the repository. The step fails AFTER the immutable PyPI "
                "upload, leaving a PyPI release with no GitHub Release and no SBOM a consumer "
                "can fetch."
            )
    if not found:
        problems.append("no `gh release` step at all; the Release and its attached SBOM are gone")
    return problems


# --- Defect 1: the mcp extra must exclude the incompatible 2.x line ----------


class TestMcpExtraIsCapped:
    def test_extra_declares_an_upper_bound(self):
        reqs = _mcp_requirements()
        assert reqs, "the mcp extra declares no requirement at all"
        mcp_req = next(r for r in reqs if r.split(";")[0].strip().startswith("mcp"))
        assert "<2" in mcp_req, (
            f"the mcp extra is uncapped ({mcp_req!r}); a fresh resolver picks mcp 2.x, "
            "which removed mcp.server.fastmcp and kills the vfairness-mcp entry point"
        )

    def test_extra_keeps_its_working_floor(self):
        mcp_req = next(r for r in _mcp_requirements() if r.split(";")[0].strip().startswith("mcp"))
        assert ">=1.2.0" in mcp_req

    def test_uv_lock_does_not_pin_the_incompatible_major(self):
        # uv.lock is what `uv sync --extra mcp` actually installs. A capped
        # pyproject with a stale lock still hands developers the broken 2.x.
        text = UV_LOCK.read_text(encoding="utf-8")
        match = re.search(r'^name = "mcp"\nversion = "([^"]+)"', text, re.M)
        assert match is not None, "uv.lock has no mcp entry to check"
        major = int(match.group(1).split(".")[0])
        assert major == 1, f"uv.lock pins mcp {match.group(1)}, which has no mcp.server.fastmcp"


class TestImportGuardNamesTheRealCause:
    """The guard must distinguish "not installed" from "installed but wrong".

    Negative case first: with an `mcp` distribution present that lacks
    `mcp.server.fastmcp` (exactly what mcp 2.x looks like), the old guard said
    "install vfairness[mcp]", which is advice the reader has already followed.
    """

    def _import_server(
        self,
        tmp_path: Path,
        stub: str | None,
        *,
        allow_ambient_mcp: bool = False,
        preload_mcp: bool = False,
    ) -> subprocess.CompletedProcess:
        # Run in a subprocess: importing vfairness.mcp.server mutates sys.modules
        # and the stub package must not leak into the rest of the session.
        stub_dir = tmp_path / "stub"
        if stub is not None:
            (stub_dir / "mcp" / "server").mkdir(parents=True)
            (stub_dir / "mcp" / "__init__.py").write_text(stub, encoding="utf-8")
            (stub_dir / "mcp" / "server" / "__init__.py").write_text("", encoding="utf-8")
        else:
            stub_dir.mkdir(parents=True)
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys
            sys.path.insert(0, sys.argv[1])
            sys.path.insert(0, sys.argv[2])
            if sys.argv[4] == "preload":
                import mcp.server.fastmcp
                print("MCP_PRELOADED")
            # An empty directory on sys.path does not hide site-packages or
            # PYTHONPATH. Full-suite CI deliberately installs the mcp extra.
            # Clear cached modules too: startup hooks can import them before
            # this subprocess reaches the guard being tested.
            for name in list(sys.modules):
                if name == "mcp" or name.startswith("mcp."):
                    del sys.modules[name]
            if sys.argv[3] == "missing":
                class MissingMcp(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "mcp" or fullname.startswith("mcp."):
                            raise ModuleNotFoundError(
                                f"No module named {fullname!r}", name=fullname
                            )
                        return None
                sys.meta_path.insert(0, MissingMcp())
            try:
                import vfairness.mcp.server  # noqa: F401
            except ImportError as exc:
                print("MESSAGE:" + str(exc).replace("\\n", " | "))
                print("CHAINED:" + type(exc.__cause__).__name__)
                print("CAUSE_NAME:" + str(getattr(exc.__cause__, "name", None)))
                sys.exit(0)
            print("MESSAGE:<no error raised>")
            sys.exit(0)
            """
        )
        return subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(stub_dir),
                str(LIB_ROOT / "src"),
                "missing" if stub is None and not allow_ambient_mcp else "present",
                "preload" if preload_mcp else "",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )

    def test_incompatible_mcp_is_reported_as_a_version_problem(self, tmp_path):
        proc = self._import_server(tmp_path, stub='__version__ = "2.0.0"\n')
        assert proc.returncode == 0, proc.stderr
        message = next(
            line[len("MESSAGE:") :]
            for line in proc.stdout.splitlines()
            if line.startswith("MESSAGE:")
        )
        assert "<no error raised>" not in message
        lowered = message.lower()
        # Must name the real culprit: the mcp package, the submodule that is
        # gone, and the constraint that would fix it.
        assert "mcp.server.fastmcp" in message, message
        assert "<2" in message, message
        assert "incompatible" in lowered or "too new" in lowered, message

    def test_missing_mcp_is_reported_as_a_missing_install(self, tmp_path):
        proc = self._import_server(tmp_path, stub=None)
        assert proc.returncode == 0, proc.stderr
        message = next(
            line[len("MESSAGE:") :]
            for line in proc.stdout.splitlines()
            if line.startswith("MESSAGE:")
        )
        assert "<no error raised>" not in message
        assert 'pip install "vfairness[mcp]"' in message, message
        assert "CHAINED:ModuleNotFoundError" in proc.stdout, proc.stdout
        assert "CAUSE_NAME:mcp\n" in proc.stdout, proc.stdout

    @pytest.mark.parametrize("preload_mcp", [False, True])
    def test_missing_mcp_is_isolated_from_an_ambient_usable_package(
        self, tmp_path, monkeypatch, preload_mcp
    ):
        # A small import-compatible fixture, not an MCP protocol implementation.
        # It implements precisely the constructor/decorator interface used at
        # server import time. The control imports the REAL vfairness server,
        # proving this ambient package would otherwise defeat the missing case.
        ambient = tmp_path / "ambient"
        package = ambient / "mcp" / "server"
        package.mkdir(parents=True)
        (ambient / "mcp" / "__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "fastmcp.py").write_text(
            textwrap.dedent(
                """
                class FastMCP:
                    def __init__(self, name):
                        self.name = name

                    def tool(self):
                        return lambda function: function

                    def resource(self, uri):
                        return lambda function: function

                    def prompt(self):
                        return lambda function: function
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv(
            "PYTHONPATH",
            os.pathsep.join(filter(None, [str(ambient), os.environ.get("PYTHONPATH")])),
        )
        control = self._import_server(
            tmp_path / "control", stub=None, allow_ambient_mcp=True, preload_mcp=preload_mcp
        )
        assert control.returncode == 0, control.stderr
        assert "MESSAGE:<no error raised>" in control.stdout, control.stdout + control.stderr

        missing = self._import_server(tmp_path / "missing", stub=None, preload_mcp=preload_mcp)
        assert missing.returncode == 0, missing.stderr
        if preload_mcp:
            assert "MCP_PRELOADED" in missing.stdout, missing.stdout
        assert "MESSAGE:<no error raised>" not in missing.stdout, missing.stdout
        assert 'pip install "vfairness[mcp]"' in missing.stdout, missing.stdout
        assert "CHAINED:ModuleNotFoundError" in missing.stdout, missing.stdout
        assert "CAUSE_NAME:mcp\n" in missing.stdout, missing.stdout

    def test_original_exception_is_chained(self, tmp_path):
        proc = self._import_server(tmp_path, stub='__version__ = "2.0.0"\n')
        assert "CHAINED:ModuleNotFoundError" in proc.stdout, proc.stdout


# --- Defect 2: the SBOM step, the pin, and the tag guard ---------------------


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestReleaseWorkflow:
    def test_sbom_step_does_not_use_the_nonexistent_outfile_flag(self, copy_name):
        scripts = _run_scripts(RELEASE_WORKFLOWS[copy_name])
        sbom = [_executable_lines(s) for s in scripts if _invokes_cyclonedx(s)]
        assert sbom, "the build job has no cyclonedx-py step"
        for script in sbom:
            assert "--outfile" not in script, (
                "cyclonedx-bom has no --outfile flag; the SBOM step aborts the build job"
            )
            assert "--output-file" in script or re.search(r"(^|\s)-o(\s|=)", script), script

    def test_cyclonedx_bom_is_version_pinned(self, copy_name):
        scripts = _run_scripts(RELEASE_WORKFLOWS[copy_name])
        install = [s for s in scripts if "cyclonedx-bom" in s and "pip install" in s]
        assert install, "cyclonedx-bom is never installed"
        assert any(re.search(r"cyclonedx-bom==\d+\.\d+", s) for s in install), (
            "cyclonedx-bom is unpinned, so its CLI can drift out from under the SBOM step again"
        )

    def test_build_job_asserts_the_tag_matches_the_package_version(self, copy_name):
        scripts = _run_scripts(RELEASE_WORKFLOWS[copy_name])
        guards = [s for s in scripts if "GITHUB_REF_NAME" in s and "__version__" in s]
        assert guards, (
            "no step compares the pushed tag to the package version, so a mistyped "
            "tag publishes a mislabelled artifact"
        )

    def test_tag_guard_accepts_the_matching_tag(self, copy_name):
        script = self._guard_script(copy_name)
        version = _package_version()
        proc = _run_guard(script, f"v{version}")
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_tag_guard_refuses_a_mismatched_tag(self, copy_name):
        # The negative case is the whole point: a guard that only says yes is
        # not a guard.
        script = self._guard_script(copy_name)
        proc = _run_guard(script, "v99.99.99")
        assert proc.returncode != 0, (
            "the tag guard accepted a tag that does not match the package version:\n"
            + proc.stdout
            + proc.stderr
        )

    @staticmethod
    def _guard_script(copy_name: str) -> str:
        scripts = _run_scripts(RELEASE_WORKFLOWS[copy_name])
        guards = [s for s in scripts if "GITHUB_REF_NAME" in s and "__version__" in s]
        assert guards, "no tag guard step to execute"
        return guards[0]


def _package_version() -> str:
    src = (LIB_ROOT / "src" / "vfairness" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"', src, re.M)
    assert match is not None
    return match.group(1)


def _run_guard(script: str, ref_name: str) -> subprocess.CompletedProcess:
    # Executes the workflow's own shell, verbatim, from the library root (which
    # is the working directory both copies resolve to on the runner).
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(LIB_ROOT),
        env={
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
            "GITHUB_REF_NAME": ref_name,
            "HOME": str(Path.home()),
        },
    )


def _run_step(
    script: str, cwd: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run a workflow step's shell verbatim in a prepared directory."""
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env={
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(Path.home()),
            **(extra_env or {}),
        },
    )


def _declared_runtime_dependencies() -> list[str]:
    return [
        re.split(r"[<>=!~\[; ]", req, maxsplit=1)[0].strip().lower().replace("_", "-")
        for req in _pyproject()["project"]["dependencies"]
    ]


def _fake_sbom(component_names: list[str]) -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "components": [
            {"type": "library", "name": name, "version": "1.0.0"} for name in component_names
        ],
    }


def _sbom_workdir(tmp_path: Path, component_names: list[str]) -> Path:
    """A directory shaped like the runner's working directory for the check.

    The real pyproject.toml is copied in, not a stub, because the check reads
    the declared dependency list from it rather than restating it.
    """
    work = tmp_path / "sbomcheck"
    (work / "sbom").mkdir(parents=True, exist_ok=True)
    (work / "sbom" / "vfairness-sbom.cdx.json").write_text(
        json.dumps(_fake_sbom(component_names)), encoding="utf-8"
    )
    (work / "pyproject.toml").write_text(PYPROJECT.read_text(encoding="utf-8"), encoding="utf-8")
    return work


# --- Finding 1: only the public repository may publish -----------------------


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestPublishingIsBoundToOneRepository:
    """The private monorepo must be structurally incapable of publishing.

    Before this guard existed, the only thing stopping a tag pushed in the
    private development monorepo from publishing was that no Trusted Publisher
    was bound to it. That is an accident of configuration living in someone
    else's account, not a property of the repository, and "make the release
    work" would have been fixed by binding one.
    """

    def test_every_publishing_job_carries_the_repository_guard(self, copy_name):
        problems = _repository_guard_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_no_guard_names_a_repository_other_than_the_public_one(self, copy_name):
        # Naming some OTHER repository in the guard would satisfy a naive
        # "there is a github.repository condition" check while re-enabling
        # exactly the defect. Checked as a set rather than against one
        # blacklisted slug: the private development monorepo is the repository
        # this was written for, but a guard pointed at any third repository is
        # the same hole, and naming the private one here would put its address
        # in the public export (see NOT_THE_PUBLIC_REPO).
        text = RELEASE_WORKFLOWS[copy_name].read_text(encoding="utf-8")
        for job_name, job in _publishing_jobs(_workflow(RELEASE_WORKFLOWS[copy_name])).items():
            condition = str(job.get("if", ""))
            # Asserted rather than assumed: with no guard at all this test would
            # otherwise pass vacuously, which is the "green while the defect is
            # live" shape it exists to prevent.
            assert condition, f"{job_name} carries no guard at all"
            named = _repositories_named(condition)
            assert named == {PUBLIC_REPO}, (
                f"{job_name} is guarded on {sorted(named)} rather than on {PUBLIC_REPO} "
                f"alone, so a repository that must never publish can: {condition!r}"
            )
        # And nowhere else in the file either, so the defect cannot hide in a
        # step-level `if:` or a commented-out guard someone re-enables.
        in_file = _repositories_named(text)
        assert in_file <= {PUBLIC_REPO}, (
            f"{RELEASE_WORKFLOWS[copy_name]} compares github.repository against "
            f"{sorted(in_file - {PUBLIC_REPO})}, which is not the publishing repository"
        )

    def test_both_copies_use_the_same_repository(self, copy_name):
        # The two files diverge on paths by design, but they must agree on who
        # is allowed to publish, or the defect just moves to the other copy.
        conditions = {
            str(job.get("if", ""))
            for job in _publishing_jobs(_workflow(RELEASE_WORKFLOWS[copy_name])).values()
        }
        assert conditions, "no publishing job found"
        assert len(conditions) == 1, f"publishing jobs carry differing guards: {conditions}"
        only = next(iter(conditions))
        assert f"github.repository == '{PUBLIC_REPO}'" in only.replace('"', "'"), (
            f"the shared guard does not pin the publishing repository: {only!r}"
        )


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestRepositoryGuardCheckerIsItselfChecked:
    """Sabotage: reinstate the exact defect and confirm the checker goes red.

    A guard that has only ever been run against a passing file is not known to
    be a guard.
    """

    def _sabotage(self, copy_name: str, mutate) -> list[str]:
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        mutate(workflow)
        return _repository_guard_problems(workflow)

    def test_removing_every_guard_is_caught(self, copy_name):
        def mutate(workflow):
            for job in _publishing_jobs(workflow).values():
                job.pop("if", None)

        problems = self._sabotage(copy_name, mutate)
        assert problems, "an unguarded publish workflow passed the repository check"
        assert any("no job-level" in p for p in problems), problems

    def test_pointing_the_guard_at_another_repository_is_caught(self, copy_name):
        def mutate(workflow):
            for job in _publishing_jobs(workflow).values():
                job["if"] = (
                    "startsWith(github.ref, 'refs/tags/v') && "
                    f"github.repository == '{NOT_THE_PUBLIC_REPO}'"
                )

        problems = self._sabotage(copy_name, mutate)
        assert problems, "a workflow guarded on a non-publishing repository passed the check"

    def test_guarding_only_an_ancestor_job_is_caught(self, copy_name):
        """Guarding the FIRST publishing job and letting `needs:` carry it.

        A `needs:` edge is not a permission boundary: a downstream job runs with
        its own `if`, and with none it runs unconditionally.

        This deliberately keeps the guard on whichever publishing job comes
        first and strips it from the rest, resolved DYNAMICALLY rather than by
        naming a job. It used to name `publish-testpypi`, and when TestPyPI was
        removed from the pipeline on 2026-08-28 that name matched nothing, so
        the mutation stripped every guard and this test silently became a
        duplicate of test_removing_every_guard_is_caught. It still passed, which
        is the failure mode: the scenario in the title had stopped being tested.
        """

        def mutate(workflow):
            jobs = _publishing_jobs(workflow)
            first = next(iter(jobs), None)
            assert first is not None, "no publishing job found to guard"
            for name, job in jobs.items():
                if name != first:
                    job.pop("if", None)

        problems = self._sabotage(copy_name, mutate)
        assert problems, "a workflow relying on `needs:` to carry the guard passed the check"

    def test_dropping_the_tag_half_of_the_guard_is_caught(self, copy_name):
        def mutate(workflow):
            for job in _publishing_jobs(workflow).values():
                job["if"] = "github.repository == 'validantai/vfairness'"

        problems = self._sabotage(copy_name, mutate)
        assert problems, "a workflow_dispatch-reachable publish passed the check"
        assert any("version tag" in p for p in problems), problems

    def test_a_renamed_publishing_job_is_still_inspected(self, copy_name):
        # Detection is by what a step does, not by the job's name.
        def mutate(workflow):
            jobs = workflow["jobs"]
            assert "publish-pypi" in jobs
            jobs["ship-it"] = jobs.pop("publish-pypi")
            jobs["ship-it"].pop("if", None)

        problems = self._sabotage(copy_name, mutate)
        assert any(p.startswith("ship-it:") for p in problems), problems


# --- Finding 4: the SBOM describes the package and reaches a consumer --------


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestSbomDescribesThePackage:
    def test_sbom_is_generated_from_the_package_not_the_runner(self, copy_name):
        problems = _sbom_scope_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_sbom_is_attached_to_a_github_release(self, copy_name):
        problems = _sbom_delivery_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_the_workflow_checks_the_sbom_contents(self, copy_name):
        assert _sbom_content_check_script(_workflow(RELEASE_WORKFLOWS[copy_name])) is not None, (
            "nothing inspects the produced SBOM; a runner-scoped SBOM is valid CycloneDX "
            "and exits 0, which is how the previous one shipped"
        )


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestSbomScopeCheckerIsItselfChecked:
    """Sabotage: put the old command back and confirm the checker goes red."""

    def _sabotage(self, copy_name: str, replacement: str) -> list[str]:
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in workflow["jobs"].values():
            for step in job.get("steps") or []:
                if "run" in step and _invokes_cyclonedx(str(step["run"])):
                    step["run"] = replacement
        return _sbom_scope_problems(workflow)

    def test_the_original_bare_environment_command_is_caught(self, copy_name):
        # This is the defect verbatim, flag fix included.
        problems = self._sabotage(
            copy_name,
            "mkdir -p sbom\n"
            "python -m pip install .\n"
            "cyclonedx-py environment --output-format JSON "
            "--output-file sbom/vfairness-sbom.cdx.json\n",
        )
        assert problems, "the runner-environment SBOM command passed the scope check"
        assert any("no interpreter path" in p for p in problems), problems
        assert any("job's own interpreter" in p for p in problems), problems

    def test_removing_the_sbom_step_entirely_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in workflow["jobs"].values():
            job["steps"] = [
                s for s in (job.get("steps") or []) if not _invokes_cyclonedx(str(s.get("run", "")))
            ]
        assert _sbom_scope_problems(workflow), "a workflow with no SBOM step passed the check"

    def test_snapshotting_the_runner_by_pointing_at_its_own_python_is_caught(self, copy_name):
        problems = self._sabotage(
            copy_name,
            'mkdir -p sbom\npython -m pip install .\ncyclonedx-py environment "$(which python)" '
            "--output-format JSON --output-file sbom/vfairness-sbom.cdx.json\n",
        )
        assert problems, "pointing the SBOM at the job's own interpreter passed the check"

    def test_dropping_the_release_upload_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in workflow["jobs"].values():
            job["steps"] = [
                s for s in (job.get("steps") or []) if "gh release" not in str(s.get("run", ""))
            ]
        assert _sbom_delivery_problems(workflow), (
            "an SBOM that reaches no consumer passed the delivery check"
        )


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestSbomContentCheckExecutes:
    """Run the workflow's own SBOM content check, both ways.

    The check is what keeps the SBOM honest after this test file is forgotten,
    so it is executed rather than read.
    """

    def _script(self, copy_name: str) -> str:
        script = _sbom_content_check_script(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert script is not None, "no SBOM content check step to execute"
        return script

    def test_it_accepts_a_package_scoped_sbom(self, copy_name, tmp_path):
        work = _sbom_workdir(tmp_path, [*_declared_runtime_dependencies(), "vfairness", "certifi"])
        proc = _run_step(self._script(copy_name), work)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_it_refuses_a_runner_scoped_sbom(self, copy_name, tmp_path):
        # The exact defect: everything the package needs, plus the build
        # toolchain that was only ever in the CI interpreter.
        work = _sbom_workdir(
            tmp_path,
            [
                *_declared_runtime_dependencies(),
                "vfairness",
                "build",
                "twine",
                "check-wheel-contents",
                "cyclonedx-bom",
                "keyring",
                "pip",
            ],
        )
        proc = _run_step(self._script(copy_name), work)
        assert proc.returncode != 0, (
            "the SBOM content check accepted an SBOM of the CI runner:\n"
            + proc.stdout
            + proc.stderr
        )
        assert "twine" in proc.stderr, proc.stderr

    def test_it_refuses_an_sbom_missing_a_declared_dependency(self, copy_name, tmp_path):
        declared = _declared_runtime_dependencies()
        work = _sbom_workdir(tmp_path, [*declared[1:], "vfairness"])
        proc = _run_step(self._script(copy_name), work)
        assert proc.returncode != 0, (
            "the SBOM content check accepted an SBOM missing a declared runtime dependency:\n"
            + proc.stdout
            + proc.stderr
        )
        assert declared[0] in proc.stderr, proc.stderr

    def test_it_refuses_an_sbom_without_the_package_itself(self, copy_name, tmp_path):
        work = _sbom_workdir(tmp_path, _declared_runtime_dependencies())
        proc = _run_step(self._script(copy_name), work)
        assert proc.returncode != 0, proc.stdout + proc.stderr


# --- Fifth-iteration audit, finding 2: a dispatch must not be able to publish -


def test_the_condition_evaluator_reads_the_expressions_it_claims_to():
    """The evaluator is the instrument for everything below, so pin it first.

    It must answer BOTH ways for terms it understands, and RAISE for anything
    it does not. If it silently returned False on unfamiliar syntax, every
    "cannot publish" test in this file would go green the moment a guard was
    rewritten, whether or not the guard still refused anything.
    """
    ctx = {"event_name": "push", "ref": "refs/tags/v1.2.3", "repository": PUBLIC_REPO}
    assert _evaluate_condition("github.event_name == 'push'", ctx) is True
    assert _evaluate_condition("github.event_name == 'workflow_dispatch'", ctx) is False
    assert _evaluate_condition("github.event_name != 'workflow_dispatch'", ctx) is True
    assert _evaluate_condition("startsWith(github.ref, 'refs/tags/v')", ctx) is True
    assert _evaluate_condition("startsWith(github.ref, 'refs/heads/')", ctx) is False
    assert (
        _evaluate_condition(
            "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v') "
            f"&& github.repository == '{PUBLIC_REPO}'",
            ctx,
        )
        is True
    )
    assert (
        _evaluate_condition(
            "github.event_name == 'push' && github.repository == 'someone/else'", ctx
        )
        is False
    )
    # Fail-closed, not silently False.
    for unreadable in (
        "contains(github.ref, 'v')",
        "github.event_name == 'push' || true",
        "!cancelled()",
        "success()",
        "",
    ):
        with pytest.raises(ValueError):
            _evaluate_condition(unreadable, ctx)
    # An expression that reads a context this simulation does not model must
    # also raise, rather than quietly scoring the term as absent.
    with pytest.raises(ValueError):
        _evaluate_condition("github.actor == 'someone'", ctx)


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestWorkflowDispatchCannotPublish:
    """workflow_dispatch is the dry run. The REF never made that true.

    The guard used to be `startsWith(github.ref, 'refs/tags/v') &&
    github.repository == '...'`, under a comment asserting that a dispatch
    "CANNOT publish, because the tag condition below is false without a tag".
    workflow_dispatch takes a ref, and both the UI ref selector and
    POST /repos/{o}/{r}/actions/workflows/{id}/dispatches accept a TAG. Aimed at
    refs/tags/v0.1.0 the tag clause is TRUE, the repository clause is TRUE in the
    public repo, and the manual run walks into an upload PyPI will not let anyone
    take back. The event clause is the thing that actually enforces the claim.
    """

    def test_a_dispatch_cannot_reach_a_publishing_job(self, copy_name):
        problems = _dispatch_guard_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_a_pushed_tag_in_the_public_repo_still_publishes(self, copy_name):
        # CONTROL, and the reason the tests above are worth anything: a guard of
        # `false` refuses every bad case and also never ships. Both publishing
        # jobs must still be reachable by the one event that is supposed to
        # reach them.
        reached = _jobs_reached_by(
            _workflow(RELEASE_WORKFLOWS[copy_name]), PUSHED_TAG_IN_PUBLIC_REPO
        )
        assert sorted(reached) == ["github-release", "publish-pypi"], (
            "a pushed v-tag in the public repo no longer reaches the publishing jobs; the "
            f"guard now refuses the release itself. Reached: {reached}"
        )

    def test_a_pushed_tag_in_the_private_monorepo_still_cannot_publish(self, copy_name):
        reached = _jobs_reached_by(
            _workflow(RELEASE_WORKFLOWS[copy_name]), PUSHED_TAG_IN_PRIVATE_MONOREPO
        )
        assert reached == [], f"the private monorepo reaches publishing jobs: {reached}"


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestDispatchGuardCheckerIsItselfChecked:
    """Sabotage: reinstate the pre-2026-08-28 guard and watch the check go red."""

    def _sabotage(self, copy_name: str, mutate) -> list[str]:
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        mutate(workflow)
        return _dispatch_guard_problems(workflow)

    def test_the_previous_two_clause_guard_is_caught(self, copy_name):
        # The exact condition this file carried until 2026-08-28, verbatim.
        def mutate(workflow):
            for job in _publishing_jobs(workflow).values():
                job["if"] = (
                    f"startsWith(github.ref, 'refs/tags/v') && github.repository == '{PUBLIC_REPO}'"
                )

        problems = self._sabotage(copy_name, mutate)
        assert problems, (
            "the pre-2026-08-28 guard passed the dispatch check, so this check would not "
            "have caught the defect it exists for"
        )
        assert any("refs/tags/v0.1.0" in p for p in problems), problems
        # And it must be caught for the RIGHT reason: that guard correctly
        # refuses a dispatch on a branch, and is only wrong about a tag ref.
        assert not any("refs/heads/main" in p for p in problems), problems

    def test_removing_every_guard_is_caught(self, copy_name):
        def mutate(workflow):
            for job in _publishing_jobs(workflow).values():
                job.pop("if", None)

        assert self._sabotage(copy_name, mutate), (
            "an unguarded publish workflow passed the dispatch check"
        )

    def test_guarding_only_an_ancestor_job_is_caught(self, copy_name):
        # `needs:` is not a permission boundary here either.
        def mutate(workflow):
            jobs = _publishing_jobs(workflow)
            first = next(iter(jobs), None)
            assert first is not None, "no publishing job found to guard"
            for name, job in jobs.items():
                if name != first:
                    job.pop("if", None)

        assert self._sabotage(copy_name, mutate), (
            "a workflow relying on `needs:` to carry the event guard passed the check"
        )

    def test_a_renamed_publishing_job_is_still_inspected(self, copy_name):
        def mutate(workflow):
            jobs = workflow["jobs"]
            assert "publish-pypi" in jobs
            jobs["ship-it"] = jobs.pop("publish-pypi")
            jobs["ship-it"].pop("if", None)

        problems = self._sabotage(copy_name, mutate)
        assert any(p.startswith("ship-it:") for p in problems), problems


# --- Fifth-iteration audit, finding 1: attestations are set, not defaulted ---


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestAttestationsAreEnabledExplicitly:
    """The header promises build provenance, so a line must produce it.

    The publish step passed only `packages-dir: dist`, so whether attestations
    happened was entirely the pinned action SHA's default. A default is not an
    enforcement: the next SHA bump could flip it while the header went on
    promising attested artifacts to consumers of a fairness-assurance library.
    """

    def test_the_publish_step_sets_attestations(self, copy_name):
        problems = _attestation_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_the_header_only_claims_attestations_because_a_line_produces_them(self, copy_name):
        # The claim and the enforcement must live or die together.
        text = RELEASE_WORKFLOWS[copy_name].read_text(encoding="utf-8")
        assert "attestations" in text.split("jobs:", 1)[0], (
            "the header no longer mentions attestations; if the claim was dropped, drop this "
            "test with it, but do not leave the claim and the input out of step"
        )
        assert "attestations: true" in text.split("jobs:", 1)[1]

    def test_omitting_the_input_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in workflow["jobs"].values():
            for step in job.get("steps") or []:
                if PUBLISH_ACTION in str(step.get("uses", "")):
                    step["with"].pop("attestations", None)
        problems = _attestation_problems(workflow)
        assert problems, "a publish step with no attestations input passed the check"
        assert any("default" in p for p in problems), problems

    def test_setting_the_input_to_false_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in workflow["jobs"].values():
            for step in job.get("steps") or []:
                if PUBLISH_ACTION in str(step.get("uses", "")):
                    step["with"]["attestations"] = False
        problems = _attestation_problems(workflow)
        assert problems, "attestations: false passed the check"
        assert any("turns\noff" in p or "turns off" in p for p in problems), problems


# --- Fifth-iteration audit, finding 4: the release job can resolve the repo ---


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestGithubReleaseCanResolveTheRepository:
    """`gh release` in a job with no checkout has no repository to talk to.

    The failure lands AFTER publish-pypi, and PyPI uploads are immutable, so the
    outcome is a released version with no GitHub Release and an SBOM that exists
    only as an expiring workflow artifact. GH_REPO from the github context costs
    nothing; a checkout would clone the tree to answer a question the context
    already answers.
    """

    def test_every_gh_release_step_can_resolve_a_repository(self, copy_name):
        problems = _gh_release_repo_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_the_release_job_really_has_no_checkout(self, copy_name):
        # Pins WHY GH_REPO is required. If a checkout is ever added, this test
        # fails and whoever added it decides deliberately which mechanism stays,
        # instead of leaving a second, unexplained one lying around.
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        job = workflow["jobs"]["github-release"]
        assert not any(
            "actions/checkout" in str(s.get("uses", "")) for s in (job.get("steps") or [])
        ), "github-release now checks out; re-decide whether GH_REPO or the checkout is the guard"

    def test_dropping_gh_repo_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in workflow["jobs"].values():
            for step in job.get("steps") or []:
                if "gh release" in str(step.get("run", "")):
                    (step.get("env") or {}).pop("GH_REPO", None)
        problems = _gh_release_repo_problems(workflow)
        assert problems, "a `gh release` step with no repository context passed the check"
        assert any("GH_REPO" in p for p in problems), problems

    def test_a_checkout_would_also_satisfy_it(self, copy_name):
        # The check enforces "gh can resolve a repo", not "GH_REPO is spelled
        # here". Proving the alternative passes keeps it from silently becoming
        # a spelling rule that would flag a legitimate future refactor.
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        job = workflow["jobs"]["github-release"]
        for step in job.get("steps") or []:
            if "gh release" in str(step.get("run", "")):
                (step.get("env") or {}).pop("GH_REPO", None)
        job["steps"].insert(
            0, {"uses": "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"}
        )
        assert not _gh_release_repo_problems(workflow)


# =============================================================================
# Sixth-iteration audit, 2026-08-28
# =============================================================================
#
# release.yml's header lists FIVE gates that replaced the removed TestPyPI
# rehearsal and says "none of them may be weakened". Three of them were read by
# nothing at all: `environment: pypi`, the `smoke` dependency, and the twine /
# check-wheel-contents verification. Deleting all three from a copy of the
# workflow left every test above green.

APPROVAL_ENVIRONMENT = "pypi"
SMOKE_JOB = "smoke"
BUILD_JOB = "build"


def _publish_action_jobs(workflow: dict) -> dict[str, dict]:
    """Jobs that run the PyPI upload action itself.

    Narrower than _publishing_jobs, deliberately: only the upload is gated by an
    environment, and only it is the irreversible half.
    """
    return {
        name: job
        for name, job in workflow["jobs"].items()
        if any(PUBLISH_ACTION in str(s.get("uses", "")) for s in (job.get("steps") or []))
    }


def _approval_environment_problems(workflow: dict) -> list[str]:
    """Gate 4: the `pypi` environment, the header's LAST human checkpoint.

    WHAT THIS CANNOT SEE, and must never be read as covering: whether that
    environment has a REQUIRED REVIEWER. Reviewers live in the repository's
    settings, GitHub creates a missing environment implicitly with no reviewers
    at all, and a run that never pauses is indistinguishable from an approved
    one. So this is two of three states, not three: the workflow half is checked
    here, and the reviewer half is COULD-NOT-CHECK from inside the repository.
    scripts/export-vfairness-to-public.sh says so to the operator in its closing
    message, which is the only place a human sees it.
    """
    jobs = _publish_action_jobs(workflow)
    problems: list[str] = []
    if not jobs:
        problems.append(
            f"no {PUBLISH_ACTION} step at all; either publishing moved to a route this "
            "check does not know about, or the check is looking at the wrong file"
        )
    for name, job in sorted(jobs.items()):
        environment = job.get("environment")
        if isinstance(environment, dict):
            environment = environment.get("name")
        if not environment:
            problems.append(
                f"{name}: uploads to PyPI under no `environment:` at all, so there is no "
                "point at which a human can be asked to approve an upload that cannot be "
                "taken back"
            )
        elif str(environment) != APPROVAL_ENVIRONMENT:
            problems.append(
                f"{name}: publishes through environment {str(environment)!r}, not "
                f"{APPROVAL_ENVIRONMENT!r}. Any required reviewer is configured on the "
                "latter, so this route goes around the approval"
            )
    return problems


def _needs_closure(workflow: dict, job_name: str) -> set[str]:
    """Every job `job_name` depends on, transitively.

    Transitive on purpose: publish-pypi may reach `smoke` directly today and
    through another job tomorrow, and either satisfies the gate. What must never
    happen is reaching PyPI without the clean-room install having run.
    """
    seen: set[str] = set()
    stack = [job_name]
    while stack:
        needs = workflow["jobs"].get(stack.pop(), {}).get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        for dependency in needs:
            if dependency not in seen:
                seen.add(dependency)
                stack.append(dependency)
    return seen


def _smoke_gate_problems(workflow: dict) -> list[str]:
    """Gate 3: nothing reaches PyPI without the clean-room wheel install."""
    problems: list[str] = []
    if SMOKE_JOB not in workflow["jobs"]:
        problems.append(
            f"there is no `{SMOKE_JOB}` job; the clean-room install that stands between a "
            "broken artifact and an immutable PyPI upload is gone entirely"
        )
    jobs = _publishing_jobs(workflow)
    if not jobs:
        problems.append("no publishing job detected at all")
    for name in sorted(jobs):
        if SMOKE_JOB not in _needs_closure(workflow, name):
            problems.append(
                f"{name}: does not depend, even transitively, on `{SMOKE_JOB}`, so a wheel "
                "that fails to install or import can still be published"
            )
    return problems


_VERIFICATION_COMMANDS = {
    "twine check": (
        "twine check reads the built metadata; without it a long_description PyPI "
        "cannot render ships anyway, and the version's description is immutable"
    ),
    "check-wheel-contents": (
        "check-wheel-contents inspects the wheel's layout; without it a wheel that "
        "ships the wrong tree (or nothing at all) passes the build job"
    ),
}


def _verification_lines(workflow: dict) -> str:
    """Every run line except installs, so an INSTALL cannot pass for a RUN.

    `pip install --upgrade pip build twine check-wheel-contents ...` contains
    both command names as substrings. Installing a checker is not running one,
    and a naive search over the whole script would report both gates present on
    a workflow that had deleted both steps.
    """
    lines = []
    for script in _all_run_scripts(workflow):
        for line in _executable_lines(script).splitlines():
            if "pip install" in line:
                continue
            lines.append(line)
    return "\n".join(lines)


def _artifact_verification_problems(workflow: dict) -> list[str]:
    """Gate 2 (the artifact half): twine check and check-wheel-contents run."""
    lines = _verification_lines(workflow)
    problems: list[str] = []
    for command, why in _VERIFICATION_COMMANDS.items():
        if not re.search(rf"(?:^|\s){re.escape(command)}(?:\s|$)", lines, re.M):
            problems.append(f"no step runs `{command}`: {why}")
    return problems


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestTheHeadersFiveGatesAreAllRead:
    """Each gate the header names must be checked by a line, not by the prose."""

    def test_the_publish_job_runs_under_the_approval_environment(self, copy_name):
        problems = _approval_environment_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_no_publishing_job_can_skip_the_clean_room_smoke(self, copy_name):
        problems = _smoke_gate_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_the_build_job_verifies_the_artifact_before_anything_can_publish(self, copy_name):
        problems = _artifact_verification_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_the_header_still_names_the_gates_these_checks_enforce(self, copy_name):
        # Claim and enforcement live or die together, the same treatment
        # attestations already gets. If the gate list is ever rewritten, this
        # fails and whoever rewrote it re-decides deliberately.
        text = RELEASE_WORKFLOWS[copy_name].read_text(encoding="utf-8")
        for phrase in ("required reviewer", "`smoke`", "may be weakened"):
            assert phrase in text, (
                f"release.yml no longer says {phrase!r}. If a gate was genuinely dropped, "
                "drop its check here in the same change; do not leave the claim and the "
                "enforcement out of step"
            )


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestTheThreeNewGateCheckersAreThemselvesChecked:
    """Sabotage: the exact weakening each checker exists for, one at a time.

    All three were performed at once on a copy of the export tree on 2026-08-28
    (`environment: pypi` deleted, `needs: [build, smoke]` rewritten to
    `needs: build`, both verification steps removed) and the file stayed green:
    "85 passed". These are the checks that were missing.
    """

    def test_deleting_the_approval_environment_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in _publish_action_jobs(workflow).values():
            job.pop("environment", None)
        problems = _approval_environment_problems(workflow)
        assert problems, "a publish job with no environment passed the approval check"
        assert any("no `environment:`" in p for p in problems), problems

    def test_pointing_the_publish_at_a_different_environment_is_caught(self, copy_name):
        # An environment with no reviewer configured on it is not the gate; the
        # name is what the reviewer is attached to.
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in _publish_action_jobs(workflow).values():
            job["environment"] = "staging"
        problems = _approval_environment_problems(workflow)
        assert problems, "publishing through an unreviewed environment passed the check"
        assert any("staging" in p for p in problems), problems

    def test_an_environment_written_in_the_mapping_form_still_passes(self, copy_name):
        # `environment: {name: pypi, url: ...}` is the same gate spelled longer.
        # Proving it passes keeps the check from becoming a spelling rule that
        # would refuse a legitimate refactor.
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in _publish_action_jobs(workflow).values():
            job["environment"] = {
                "name": APPROVAL_ENVIRONMENT,
                "url": "https://pypi.org/p/vfairness",
            }
        assert not _approval_environment_problems(workflow)

    def test_dropping_the_smoke_dependency_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for name in _publishing_jobs(workflow):
            workflow["jobs"][name]["needs"] = BUILD_JOB
        problems = _smoke_gate_problems(workflow)
        assert problems, "a publish that no longer waits for the smoke job passed the check"
        assert any(SMOKE_JOB in p for p in problems), problems

    def test_deleting_the_smoke_job_itself_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        workflow["jobs"].pop(SMOKE_JOB)
        problems = _smoke_gate_problems(workflow)
        assert problems, "a workflow with no smoke job at all passed the check"
        assert any("no `smoke` job" in p for p in problems), problems

    def test_reaching_smoke_through_another_job_still_passes(self, copy_name):
        # The gate is "the clean-room install ran", not "publish-pypi names it".
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        workflow["jobs"]["gate"] = {"needs": [BUILD_JOB, SMOKE_JOB], "runs-on": "ubuntu-latest"}
        for name in _publishing_jobs(workflow):
            workflow["jobs"][name]["needs"] = ["gate"]
        assert not _smoke_gate_problems(workflow)

    def test_removing_the_verification_steps_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in workflow["jobs"].values():
            job["steps"] = [
                step
                for step in (job.get("steps") or [])
                if not any(
                    command in str(step.get("run", "")) for command in _VERIFICATION_COMMANDS
                )
                or "pip install" in str(step.get("run", ""))
            ]
        problems = _artifact_verification_problems(workflow)
        assert len(problems) == 2, problems
        assert any("twine check" in p for p in problems), problems
        assert any("check-wheel-contents" in p for p in problems), problems

    def test_installing_the_checkers_without_running_them_is_not_enough(self, copy_name):
        # The defect this shape hides: `pip install ... twine check-wheel-contents`
        # contains both names, so a search over the raw scripts reports both
        # gates present on a workflow that runs neither.
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        for job in workflow["jobs"].values():
            job["steps"] = [
                {"run": "python -m pip install --upgrade pip build twine check-wheel-contents"}
                if any(command in str(step.get("run", "")) for command in _VERIFICATION_COMMANDS)
                else step
                for step in (job.get("steps") or [])
            ]
        problems = _artifact_verification_problems(workflow)
        assert len(problems) == 2, problems


# --- The smoke job's claim, and the step that earns it -----------------------
#
# release.yml said the smoke job "catches a missing data file, a wrong entry
# point, an unshipped subpackage, or a missing runtime dependency". Only the last
# two were true: scripts/wheel_smoke.py imports the package and computes a
# metric, and the wheel ships payload it never opens. Measured 2026-08-28 on a
# clean venv holding the built wheel, with vfairness/rendering/templates/ and
# vfairness/legal/data/ both moved aside: "SMOKE OK", rc=0.


def _smoke_steps(workflow: dict) -> list[dict]:
    return list(workflow["jobs"].get(SMOKE_JOB, {}).get("steps") or [])


def _payload_smoke_script(workflow: dict) -> str | None:
    """The step that reads shipped payload through the package's own loader."""
    for step in _smoke_steps(workflow):
        script = str(step.get("run", ""))
        if "importlib.resources" in script:
            return script
    return None


def _payload_smoke_problems(workflow: dict) -> list[str]:
    script = _payload_smoke_script(workflow)
    if script is None:
        return [
            "the smoke job opens no shipped data file through importlib.resources, so the "
            "'catches a missing data file' half of its own claim is produced by nothing"
        ]
    problems: list[str] = []
    for module, payload in (
        ("vfairness.rendering", "the SVG templates"),
        ("vfairness.legal", "the legal rule files"),
    ):
        if f'files("{module}")' not in script and f"files('{module}')" not in script:
            problems.append(f"the smoke job never loads {payload} out of {module}")
    if not any(
        entry_point in str(step.get("run", ""))
        for step in _smoke_steps(workflow)
        for entry_point in ("vfairness-precommit", "vfairness-mcp")
    ):
        problems.append(
            "the smoke job runs neither console script, so 'a wrong entry point' is "
            "claimed and not checked"
        )
    return problems


@pytest.mark.parametrize("copy_name", sorted(RELEASE_WORKFLOWS))
class TestTheSmokeJobChecksTheShippedPayload:
    def test_the_payload_and_entry_point_step_exists(self, copy_name):
        problems = _payload_smoke_problems(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert not problems, "\n".join(problems)

    def test_removing_the_step_is_caught(self, copy_name):
        workflow = _workflow(RELEASE_WORKFLOWS[copy_name])
        workflow["jobs"][SMOKE_JOB]["steps"] = [
            step
            for step in _smoke_steps(workflow)
            if "importlib.resources" not in str(step.get("run", ""))
        ]
        problems = _payload_smoke_problems(workflow)
        assert problems, "a smoke job that reads no shipped payload passed the check"
        assert any("importlib.resources" in p for p in problems), problems

    def test_the_step_passes_against_the_installed_package(self, copy_name, tmp_path):
        # EXECUTED, not read: the workflow's own shell, from a scratch directory,
        # against the vfairness on this interpreter's path.
        script = _payload_smoke_script(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert script is not None
        proc = _run_step(script, tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "payload OK" in proc.stdout, proc.stdout
        # Both console scripts, and vfairness-mcp reported as shipped-but-not-
        # runnable rather than skipped, whenever the mcp extra is absent here.
        assert "entry point OK: vfairness-precommit" in proc.stdout, proc.stdout
        assert "entry point OK: vfairness-mcp" in proc.stdout, proc.stdout

    def test_the_step_refuses_a_package_whose_console_scripts_were_not_shipped(
        self, copy_name, tmp_path
    ):
        # The entry-point negative case: everything importable, but neither
        # console script installed. A PATH holding only `python` is exactly what
        # a wheel that lost its [project.scripts] looks like.
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        (fake_bin / "python").symlink_to(sys.executable)
        script = _payload_smoke_script(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert script is not None
        proc = _run_step(script, tmp_path, extra_env={"PATH": f"{fake_bin}:/usr/bin:/bin"})
        assert proc.returncode != 0, (
            "the smoke accepted an install with no console scripts on PATH:\n"
            + proc.stdout
            + proc.stderr
        )
        assert "payload OK" in proc.stdout, (
            "it failed before reaching the entry-point half, so this is not the case "
            "the test names:\n" + proc.stdout + proc.stderr
        )
        assert "vfairness-precommit" in proc.stdout + proc.stderr

    @staticmethod
    def _payload_free_stub(tmp_path: Path, *, keep_directories: bool) -> Path:
        """A `vfairness` that imports but ships no payload.

        Two shapes, because a wheel can lose payload two ways: the directories
        survive and are EMPTY (data files not matched by the build config), or
        they are absent entirely (the whole tree not shipped).
        """
        stub = tmp_path / ("stub-empty" if keep_directories else "stub-absent")
        for package in ("vfairness", "vfairness/rendering", "vfairness/legal"):
            (stub / package).mkdir(parents=True)
            (stub / package / "__init__.py").write_text("", encoding="utf-8")
        if keep_directories:
            (stub / "vfairness" / "rendering" / "templates").mkdir()
            (stub / "vfairness" / "legal" / "data" / "rules").mkdir(parents=True)
        return stub

    @pytest.mark.parametrize("keep_directories", [True, False])
    def test_the_step_refuses_a_package_whose_payload_was_not_shipped(
        self, copy_name, tmp_path, keep_directories
    ):
        # The negative case is the whole point: the step must REFUSE a package
        # that carries no template and no legal rule file, and say which.
        stub = self._payload_free_stub(tmp_path, keep_directories=keep_directories)
        script = _payload_smoke_script(_workflow(RELEASE_WORKFLOWS[copy_name]))
        assert script is not None
        proc = _run_step(script, tmp_path, extra_env={"PYTHONPATH": str(stub)})
        assert proc.returncode != 0, (
            "the payload smoke accepted a package that ships no template and no legal "
            "rule file:\n" + proc.stdout + proc.stderr
        )
        # Named, not a bare traceback: an absent directory must read the same as
        # an empty one.
        assert "ships no SVG template" in proc.stdout + proc.stderr, proc.stdout + proc.stderr


# --- The workflow that claims to run this file, and the extra that lets it ---


def _distributions_providing(module_name: str) -> set[str]:
    """Which installed distributions provide `import <module_name>`.

    Derived from the environment rather than restated, so the PyYAML/yaml
    spelling difference cannot go stale here.
    """
    return {
        name.strip().lower().replace("_", "-")
        for name in packages_distributions().get(module_name, [])
    }


def _extras_declaring(distributions: set[str]) -> set[str]:
    extras = set()
    for extra, requirements in _pyproject()["project"]["optional-dependencies"].items():
        for requirement in requirements:
            name = re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip()
            if name.lower().replace("_", "-") in distributions:
                extras.add(extra)
    return extras


def _installed_extras(workflow: dict) -> set[str]:
    extras: set[str] = set()
    for script in _all_run_scripts(workflow):
        if "pip install" not in script:
            continue
        for match in re.finditer(r"\.\[([^\]]+)\]", script):
            extras.update(part.strip() for part in match.group(1).split(","))
    return extras


def _yaml_extra_problems(workflow: dict) -> list[str]:
    providers = _distributions_providing("yaml")
    if not providers:
        # Cannot happen while this module is collected (it importorskips yaml),
        # but say so rather than passing on an empty intersection.
        return ["cannot tell which distribution provides `yaml` in this environment"]
    wanted = _extras_declaring(providers)
    if not wanted:
        return [
            f"no optional-dependency extra declares {sorted(providers)}, so no CI install "
            "line can bring it in"
        ]
    installed = _installed_extras(workflow)
    if not installed:
        return ["the workflow installs no extras at all"]
    if not (wanted & installed):
        return [
            f"the install line carries {sorted(installed)}, none of which declares "
            f"{sorted(providers)}. tests/test_release_mechanics.py opens with "
            "pytest.importorskip('yaml'), so all of its tests become ONE skip line in the "
            f"workflow advertised as the full suite. Add one of {sorted(wanted)}."
        ]
    return []


def _tests_workflow() -> dict:
    return yaml.safe_load(TESTS_WORKFLOW.read_text(encoding="utf-8"))


class TestTheFullSuiteWorkflowCanActuallyRunThisFile:
    """A skip is invisible: "85 passed" and "1 skipped" are the same green.

    tests.yml is the workflow docs/RELEASE_PIPELINE.md advertises as "full suite
    and coverage on 3.11, 3.12 and 3.13". It installed no extra carrying PyYAML,
    and every test in this file - the ones that keep the PUBLISH pipeline honest -
    collapsed into a single skip line there.

    WHERE THIS PIN BITES, and where it cannot: it runs wherever `yaml` is
    importable, which is this file's own precondition. In an environment that
    already lost PyYAML the whole module skips and this check goes with it, so it
    catches the regression from the OTHER side (locally, and in the workflow that
    does install the extra) rather than from inside the broken environment. The
    check that would catch it from inside belongs in the REQUIRED_BACKENDS map of
    tests/test_optional_backends_present.py, which does not import yaml.
    """

    def test_the_install_line_carries_the_extra_that_provides_yaml(self):
        problems = _yaml_extra_problems(_tests_workflow())
        assert not problems, "\n".join(problems)

    def test_dropping_that_extra_is_caught(self):
        workflow = _tests_workflow()
        wanted = _extras_declaring(_distributions_providing("yaml"))
        for job in workflow["jobs"].values():
            for step in job.get("steps") or []:
                if "run" not in step:
                    continue
                script = str(step["run"])
                for extra in wanted:
                    script = script.replace(f",{extra}]", "]").replace(f"[{extra},", "[")
                step["run"] = script
        assert _installed_extras(workflow), "the sabotage removed the install line itself"
        problems = _yaml_extra_problems(workflow)
        assert problems, (
            "an install line with no PyYAML-bearing extra passed the check, so this pin "
            "would not have caught the defect it exists for"
        )
        assert any("importorskip" in p for p in problems), problems

    def test_the_module_level_skip_this_protects_is_still_there(self):
        # If the importorskip is ever removed, the reasoning above changes and
        # this pin should be re-decided rather than left standing on a premise
        # that stopped being true.
        source = Path(__file__).read_text(encoding="utf-8")
        assert 'pytest.importorskip("yaml")' in source


# --- The publish boundary: scripts/export-vfairness-to-public.sh -------------
#
# The script is one-way and writes to a repository nobody else can un-write, so
# its tag guard, its deletion guard and its push ordering are EXECUTED here
# rather than read. Each is extracted between named markers in the script itself,
# so a rename or a deletion fails loudly instead of silently testing nothing.

_SHELL_HELPERS = textwrap.dedent(
    """
    set -euo pipefail
    say() { printf '%s\\n' "$*"; }
    die() { printf 'ABORT: %s\\n' "$*" >&2; exit 1; }
    """
)


def _export_block(name: str) -> str:
    text = EXPORT_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^# --- BEGIN {re.escape(name)}\b[^\n]*\n(?P<body>.*?)^# --- END {re.escape(name)} ---",
        text,
        re.S | re.M,
    )
    assert match is not None, (
        f"{EXPORT_SCRIPT} carries no `{name}` block. Either the guard was removed, or the "
        "markers these tests extract it by were renamed. Both need a decision; neither is "
        "a pass."
    )
    return match.group("body")


def _shell_env(home: Path, **overrides: str) -> dict[str, str]:
    git = shutil.which("git")
    assert git is not None, "git is required to exercise the export script's guards"
    return {
        "PATH": f"{Path(git).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        # Hermetic: the operator's global git config must not decide whether a
        # test passes (commit.gpgsign alone would break every commit below).
        "GIT_CONFIG_GLOBAL": str(home / "no-such-gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        **overrides,
    }


def _run_export_block(name: str, home: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", _SHELL_HELPERS + _export_block(name)],
        capture_output=True,
        text=True,
        cwd=str(home),
        env=_shell_env(home, **env),
    )


def _git(cwd: Path, *args: str, home: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.email=export-test@example.invalid",
            "-c",
            "user.name=Export Test",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=_shell_env(home or cwd),
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed:\n{proc.stdout}{proc.stderr}"
    return proc


def _origin_with_two_releases(tmp_path: Path) -> Path:
    """A bare 'public repo' carrying v0.1.0 (older) and v0.1.1 (at the tip)."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "seed"
    work.mkdir()
    _git(tmp_path, "init", "--bare", str(origin))
    _git(work, "init")
    (work / "f.txt").write_text("one\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "Release v0.1.0")
    _git(work, "tag", "-a", "v0.1.0", "-m", "vfairness 0.1.0")
    (work / "f.txt").write_text("two\n", encoding="utf-8")
    _git(work, "commit", "-am", "Release v0.1.1")
    _git(work, "tag", "-a", "v0.1.1", "-m", "vfairness 0.1.1")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "origin", "main", "--tags")
    return origin


@monorepo_only
class TestTheExportAsksTheRemoteWhetherTheTagExists:
    """The guard read a `--depth 1` clone, which only fetches the tip's tags."""

    def test_a_shallow_clone_really_cannot_see_the_older_tag(self, tmp_path):
        # The premise, executed rather than asserted from memory. Without this,
        # the ls-remote guard below looks like a stylistic preference.
        origin = _origin_with_two_releases(tmp_path)
        clone = tmp_path / "shallow"
        _git(tmp_path, "clone", "--depth", "1", f"file://{origin}", str(clone))
        tags = _git(clone, "tag", "-l").stdout.split()
        assert tags == ["v0.1.1"], tags
        stale = subprocess.run(
            ["git", "rev-parse", "v0.1.0"],
            cwd=str(clone),
            capture_output=True,
            text=True,
            env=_shell_env(tmp_path),
        )
        assert stale.returncode != 0, (
            "the shallow clone can see v0.1.0 after all; the old guard was not blind and "
            "this whole section needs re-deriving"
        )

    def test_it_refuses_a_version_whose_tag_is_not_at_the_tip(self, tmp_path):
        origin = _origin_with_two_releases(tmp_path)
        proc = _run_export_block(
            "remote tag guard", tmp_path, PUBLIC_REPO_URL=f"file://{origin}", VERSION="0.1.0"
        )
        assert proc.returncode != 0, (
            "re-exporting 0.1.0 was allowed, which overwrites public main and then dies on "
            "the rejected tag push:\n" + proc.stdout + proc.stderr
        )
        assert "already exists" in proc.stderr, proc.stderr

    def test_it_refuses_the_version_whose_tag_is_at_the_tip(self, tmp_path):
        origin = _origin_with_two_releases(tmp_path)
        proc = _run_export_block(
            "remote tag guard", tmp_path, PUBLIC_REPO_URL=f"file://{origin}", VERSION="0.1.1"
        )
        assert proc.returncode != 0, proc.stdout + proc.stderr
        assert "already exists" in proc.stderr, proc.stderr

    def test_it_allows_a_version_the_remote_has_never_seen(self, tmp_path):
        # CONTROL. A guard that refuses everything blocks every release and
        # passes every refusal test above.
        origin = _origin_with_two_releases(tmp_path)
        proc = _run_export_block(
            "remote tag guard", tmp_path, PUBLIC_REPO_URL=f"file://{origin}", VERSION="0.2.0"
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_an_unreachable_remote_is_not_read_as_a_missing_tag(self, tmp_path):
        # Three states. `git ls-remote` on a remote it cannot reach prints
        # nothing on stdout, which is byte-identical to "that tag is not there".
        proc = _run_export_block(
            "remote tag guard",
            tmp_path,
            PUBLIC_REPO_URL=f"file://{tmp_path / 'does-not-exist.git'}",
            VERSION="0.2.0",
        )
        assert proc.returncode != 0, (
            "an unreachable remote passed the tag guard, so a network or auth failure "
            "reads as permission to publish:\n" + proc.stdout + proc.stderr
        )
        assert "could not ask" in proc.stderr, proc.stderr


@monorepo_only
class TestTheExportRefusesToOverwritePublicContent:
    """A merged community pull request was staged as a deletion and pushed.

    Two arms, because a contribution can arrive in two shapes: a NEW file, which
    the wholesale replace stages as `D`, and an EDIT to an existing file, which
    stages as a perfectly ordinary `M` that no deletion list can distinguish from
    the library's own change. The tip-commit subject catches the second.
    """

    def _staged_public_repo(
        self,
        tmp_path: Path,
        snapshot: dict[str, str],
        subject: str = "Merge PR #1: community bug fix",
    ) -> Path:
        public = tmp_path / "public"
        public.mkdir()
        _git(public, "init", home=tmp_path)
        (public / "f.txt").write_text("one\n", encoding="utf-8")
        (public / "COMMUNITY_FIX.py").write_text("# merged from PR #1\n", encoding="utf-8")
        _git(public, "add", "-A", home=tmp_path)
        _git(public, "commit", "-m", subject, home=tmp_path)
        # Exactly what the script does before the guard: wipe everything but
        # .git, lay the snapshot down, stage it all.
        for entry in public.iterdir():
            if entry.name == ".git":
                continue
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        for name, content in snapshot.items():
            (public / name).write_text(content, encoding="utf-8")
        _git(public, "add", "-A", home=tmp_path)
        return public

    def test_it_refuses_a_snapshot_that_removes_a_merged_contribution(self, tmp_path):
        public = self._staged_public_repo(tmp_path, {"f.txt": "two\n"})
        proc = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="0"
        )
        assert proc.returncode != 0, (
            "the export silently reverted a merged pull request:\n" + proc.stdout + proc.stderr
        )
        assert "COMMUNITY_FIX.py" in proc.stderr, proc.stderr
        assert "DELETE" in proc.stderr, proc.stderr

    def test_it_prints_the_staged_diff_the_operator_never_used_to_see(self, tmp_path):
        public = self._staged_public_repo(tmp_path, {"f.txt": "two\n"})
        proc = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="0"
        )
        assert "D\tCOMMUNITY_FIX.py" in proc.stdout, proc.stdout
        assert "M\tf.txt" in proc.stdout, proc.stdout

    def test_it_refuses_an_edit_that_stages_as_a_plain_modification(self, tmp_path):
        # The case the deletion list CANNOT see: the contributor changed a file
        # the library already ships, so the snapshot overwrites it with an `M`
        # that looks exactly like the library's own edit.
        public = self._staged_public_repo(
            tmp_path, {"f.txt": "two\n", "COMMUNITY_FIX.py": "# merged from PR #1\n"}
        )
        proc = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="0"
        )
        assert proc.returncode != 0, (
            "the export overwrote a merged contribution that deleted nothing:\n"
            + proc.stdout
            + proc.stderr
        )
        assert "Merge PR #1" in proc.stderr, proc.stderr

    def test_an_explicit_accept_deletions_lets_a_real_removal_through(self, tmp_path):
        public = self._staged_public_repo(tmp_path, {"f.txt": "two\n"})
        proc = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="1"
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_an_ordinary_release_over_the_previous_release_is_not_blocked(self, tmp_path):
        # CONTROL: the healthy path must behave exactly as before, or the guard
        # has simply stopped releases. Public main's tip is the previous export,
        # and the snapshot removes nothing.
        public = self._staged_public_repo(
            tmp_path,
            {"f.txt": "two\n", "COMMUNITY_FIX.py": "# merged from PR #1\n", "NEW.py": "x\n"},
            subject="Release v0.1.0",
        )
        proc = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="0"
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "A\tNEW.py" in proc.stdout, proc.stdout

    def test_the_first_ever_export_into_an_empty_repo_is_not_blocked(self, tmp_path):
        # CONTROL, and the reason the tip check treats "" as fine: a public repo
        # with no commits overwrites nothing, and `git log -1` there exits 128.
        public = tmp_path / "public"
        public.mkdir()
        _git(public, "init", home=tmp_path)
        (public / "f.txt").write_text("first\n", encoding="utf-8")
        _git(public, "add", "-A", home=tmp_path)
        proc = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="0"
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "A\tf.txt" in proc.stdout, proc.stdout


@monorepo_only
class TestTheTagIsPushedBeforeMain:
    """Both pushes can fail, so the ORDER decides what a half export leaves.

    Main-first left public main overwritten with the snapshot and no tag: no
    release.yml run, no PyPI upload, no GitHub Release. Tag-first fails before
    main moves.
    """

    def _prepared_clone(self, tmp_path: Path, version: str) -> tuple[Path, Path]:
        origin = _origin_with_two_releases(tmp_path)
        clone = tmp_path / "public"
        _git(tmp_path, "clone", f"file://{origin}", str(clone))
        (clone / "f.txt").write_text("snapshot\n", encoding="utf-8")
        _git(clone, "commit", "-am", f"Release v{version}", home=tmp_path)
        # -f because the re-export case is exactly "this tag is already on the
        # remote": the local tag has to point at the NEW snapshot commit for the
        # push to be the thing that gets rejected.
        _git(clone, "tag", "-f", "-a", f"v{version}", "-m", f"vfairness {version}", home=tmp_path)
        return origin, clone

    def _remote_main(self, origin: Path, tmp_path: Path) -> str:
        return _git(tmp_path, "--git-dir", str(origin), "rev-parse", "main").stdout.strip()

    def test_a_rejected_tag_push_leaves_public_main_untouched(self, tmp_path):
        # v0.1.0 already exists on the remote, so the tag push is rejected. With
        # main pushed first, main would already carry the snapshot by then.
        origin, clone = self._prepared_clone(tmp_path, "0.1.0")
        before = self._remote_main(origin, tmp_path)
        proc = _run_export_block(
            "push order",
            tmp_path,
            PUB_DIR=str(clone),
            VERSION="0.1.0",
            PUBLIC_REPO_URL=f"file://{origin}",
        )
        assert proc.returncode != 0, proc.stdout + proc.stderr
        assert self._remote_main(origin, tmp_path) == before, (
            "public main moved even though the tag push was rejected: the export left a "
            "rewritten main with no tag, no release and no PyPI upload"
        )

    def test_a_clean_release_pushes_both(self, tmp_path):
        # CONTROL: reordering must not break the release it exists to protect.
        origin, clone = self._prepared_clone(tmp_path, "0.2.0")
        before = self._remote_main(origin, tmp_path)
        proc = _run_export_block(
            "push order",
            tmp_path,
            PUB_DIR=str(clone),
            VERSION="0.2.0",
            PUBLIC_REPO_URL=f"file://{origin}",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert self._remote_main(origin, tmp_path) != before
        assert "v0.2.0" in _git(tmp_path, "--git-dir", str(origin), "tag", "-l").stdout
