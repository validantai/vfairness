"""The route the beta docs give a tester has to actually exist.

`docs/BETA.md` asks for about five external testers, which is a stated beta exit
criterion (VB-GOV-3), and gives them exactly one way to report back:

    Open a "Beta feedback" issue on GitHub (there is a template).

Until 2026-09-07 there was no template. `.github/` held a pull-request template
and the workflows, and nothing else. So the single instruction given to the
people whose feedback gates the release pointed at something that did not exist,
and the sentence asserting otherwise had been shipping for weeks.

That is the same defect class the rest of this suite exists for, in its least
technical form: a document claiming something the repository does not contain.
It is worse than a broken link, because a broken link is visible and this reads
as a working process right up until someone follows it.

A test rather than a one-off fix, for the same reason as everywhere else here:
the template can be deleted, renamed, or moved out of the export, and none of
those would change the sentence in BETA.md.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML ships with the cicd extra")

LIB_ROOT = pathlib.Path(__file__).resolve().parents[1]
BETA = LIB_ROOT / "docs" / "BETA.md"
TEMPLATE_DIR = LIB_ROOT / ".github" / "ISSUE_TEMPLATE"


def test_beta_doc_promises_a_template_and_one_exists():
    text = BETA.read_text(encoding="utf-8")
    promises = "there is a template" in text
    templates = sorted(p for p in TEMPLATE_DIR.glob("*.y*ml") if p.name != "config.yml")
    if promises:
        assert templates, (
            "docs/BETA.md tells a beta tester to open a feedback issue and says "
            "'there is a template', but .github/ISSUE_TEMPLATE holds no issue form. "
            "Either add the template or stop promising it."
        )
    if templates and not promises:
        pytest.fail(
            "an issue template exists but docs/BETA.md no longer points testers at it; "
            "the feedback route is now undiscoverable from the docs"
        )


def test_the_feedback_template_is_a_valid_github_issue_form():
    """A malformed form is silently ignored by GitHub, which looks like no template."""
    forms = [p for p in TEMPLATE_DIR.glob("*.y*ml") if p.name != "config.yml"]
    assert forms, "no issue form found"
    for path in forms:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("name", "description", "body"):
            assert key in data, f"{path.name}: issue forms require a top-level {key!r}"
        assert data["body"], f"{path.name}: the form has no fields"
        required = [f for f in data["body"] if (f.get("validations") or {}).get("required")]
        assert required, (
            f"{path.name}: every field is optional, so the form can be submitted empty. "
            "At least the 'what happened' field should be required."
        )


def test_security_reports_are_routed_away_from_the_public_tracker():
    """A tester following the feedback route must not file a vulnerability in public."""
    cfg = TEMPLATE_DIR / "config.yml"
    assert cfg.exists(), "no ISSUE_TEMPLATE/config.yml, so nothing routes security reports"
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    links = data.get("contact_links") or []
    assert any("SECURITY" in (link.get("url") or "").upper() for link in links), (
        "config.yml does not point security reports at SECURITY.md"
    )
    assert (LIB_ROOT / "SECURITY.md").exists(), "SECURITY.md is missing but linked"


def test_the_beta_doc_still_names_the_tester_target():
    """NON-VACUITY for the criterion itself.

    If the "about five testers" sentence is ever dropped, the exit criterion
    stops being visible to the people it applies to, and the tests above would
    happily go on passing about a document that no longer asks for anything.
    """
    text = BETA.read_text(encoding="utf-8")
    assert "external testers" in text, "BETA.md no longer states the tester criterion"
    assert "feedback" in text.lower(), "BETA.md no longer explains how to give feedback"
