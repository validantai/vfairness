"""The release must refuse to build unless the suite passed on THIS commit.

WHY THIS EXISTS. `docs/RELEASE_PIPELINE.md` step 1 is "Confirm the quality gates
are green on main". That was a step a person was supposed to remember, and on
2026-09-10 nobody did: the full suite had been RED on `main` for ELEVEN
consecutive commits while local runs reported 8246 passed / 0 failed, and
nothing anywhere would have stopped a tag being cut from any one of them.

A published version is immutable. PyPI does not allow a release to be replaced,
so "we will notice afterwards" is not a recovery plan.

VERIFIED BY EXECUTION, 2026-09-10, against the real check-runs API:

    commit    state                          gate said
    c638433   3 full-suite checks green      ALLOWED
    2168af3   3 full-suite checks failed     REFUSED, naming all three legs
    c638433   job name changed (0 matches)   REFUSED (non-vacuity)
    ceb2337   red checks                     REFUSED

The third row is the one that matters most. A gate that examines nothing passes
forever, which is the failure mode that let eleven red commits through in the
first place, so zero matching checks is fatal rather than silent.

These tests pin the STRUCTURE, and say so: the shell logic itself was verified
by the executions above rather than by re-running the API from a unit test.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML parses the workflow")

RELEASE = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
GATE_JOB = "suite-green-on-this-commit"


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert RELEASE.is_file(), f"{RELEASE} is missing; the release workflow is the publish path"
    return yaml.safe_load(RELEASE.read_text(encoding="utf-8"))


def test_the_gate_job_exists(workflow):
    assert GATE_JOB in workflow["jobs"], (
        f"the {GATE_JOB!r} job is gone. Without it a tag can be cut from a commit "
        f"whose tests never passed, and the result cannot be unpublished."
    )


def test_every_publishing_job_is_downstream_of_the_gate(workflow):
    """A gate nothing depends on is decoration."""
    jobs = workflow["jobs"]
    build_needs = jobs["build"].get("needs")
    needs = [build_needs] if isinstance(build_needs, str) else list(build_needs or [])
    assert GATE_JOB in needs, f"build does not depend on the gate; its needs are {needs}"

    # And the publish job must be transitively downstream, via build.
    publish_needs = jobs["publish-pypi"].get("needs")
    publish_needs = [publish_needs] if isinstance(publish_needs, str) else list(publish_needs or [])
    assert "build" in publish_needs, (
        f"publish-pypi no longer depends on build, so the gate is bypassed: {publish_needs}"
    )


def test_the_gate_can_read_checks(workflow):
    """Without `checks: read` the API call returns nothing, the gate finds zero
    matching checks, and the non-vacuity clause below turns that into a refusal.
    It fails closed either way, but for the wrong reason and with a confusing
    message, so the permission is pinned."""
    perms = workflow["jobs"][GATE_JOB].get("permissions") or {}
    assert perms.get("checks") == "read", f"gate permissions are {perms}"


def _gate_script(workflow: dict) -> str:
    steps = workflow["jobs"][GATE_JOB]["steps"]
    return "\n".join(step.get("run", "") for step in steps)


def test_the_gate_refuses_when_it_examined_nothing(workflow):
    """NON-VACUITY, the clause this whole gate turns on.

    If the test workflow's job name changes, the gate matches zero checks. A
    gate that examines nothing and passes is exactly what let eleven red commits
    through, so zero matches must be fatal.
    """
    script = _gate_script(workflow)
    assert "count" in script and "-eq 0" in script, (
        "the zero-matches branch is gone from the gate script"
    )
    assert "exit 1" in script, "the gate has no failing exit path"


def test_the_gate_requires_success_and_not_merely_completion(workflow):
    """`skipped`, `cancelled` and `neutral` are all "completed". None is a pass.
    The comparison must be against success specifically."""
    script = _gate_script(workflow)
    assert '"success"' in script or "'success'" in script, (
        "the gate no longer compares against success; a skipped check would pass it"
    )


def test_the_gate_runs_on_the_commit_being_released(workflow):
    """Not on main's head, and not on a branch: on the exact SHA the tag points
    at. Checking a different commit is checking a different program."""
    script = _gate_script(workflow)
    assert "github.sha" in script, "the gate is not keyed on the released commit"


def test_the_runbook_still_names_the_step_this_automates(workflow):
    """The human step stays in the runbook. This gate makes it enforceable; it
    does not make it unnecessary to understand."""
    # RELEASE is <root>/.github/workflows/release.yml, so the root is parents[2].
    runbook = RELEASE.parents[2] / "docs" / "RELEASE_PIPELINE.md"
    text = runbook.read_text(encoding="utf-8")
    assert "quality gates are green" in text.lower() or "gates are green" in text.lower(), (
        "the runbook no longer names the green-gates precondition this job enforces"
    )
