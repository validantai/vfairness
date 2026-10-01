"""What the security docs say about the egress guard must match the code.

SECURITY.md and docs/LIBRARY_OVERVIEW.md both named three call sites as issuing
"an unguarded request with a caller-supplied URL" and told readers to validate
URLs themselves before using them. All three had been wired to the guard on
2026-08-27. The docs were eleven days stale on 2026-09-07, in a file whose entire
purpose is telling people what is and is not protected.

Both directions of that are bad, and this file checks both:

- UNDERSTATING protection trains a reader to distrust a control that works, and
  to hand-roll their own, which is how a worse check replaces a better one.
- OVERSTATING it is the failure this project cares most about. Two calls really
  do bypass the guard. Neither takes a URL from the caller, so neither is an SSRF
  surface, but a doc that flatly claimed "universal" would be wrong, and the fix
  for a stale disclosure must not be a fresh overstatement.

So the rule is not "the docs must say the guard is universal". It is: the docs
must not name a guarded call site as unguarded, and must not claim universal
coverage while an unguarded call site exists.
"""

from __future__ import annotations

import pathlib
import re

LIB_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = LIB_ROOT / "src" / "vfairness"
SECURITY = LIB_ROOT / "SECURITY.md"
OVERVIEW = LIB_ROOT / "docs" / "LIBRARY_OVERVIEW.md"

# Modules that take an endpoint URL from the caller and attach a credential.
# These are the SSRF surfaces; each must route through vfairness.net.
CREDENTIAL_BEARING = (
    "llm/api_proxy.py",
    "llm/scorers.py",
    "validity/judge.py",
    "xai/sidecar_cli.py",
)

_GUARD_USE = re.compile(r"guarded_post|GuardedSession|validate_endpoint")


def _guarded(rel: str) -> bool:
    return bool(_GUARD_USE.search((SRC / rel).read_text(encoding="utf-8")))


def test_every_credential_bearing_call_site_uses_the_guard():
    """The security claim itself, checked against the code rather than the prose."""
    unguarded = [rel for rel in CREDENTIAL_BEARING if not _guarded(rel)]
    assert not unguarded, (
        "these modules take a caller-supplied URL and carry a credential, but do not "
        "reference the egress guard:\n  " + "\n  ".join(unguarded)
    )


def test_the_docs_do_not_call_a_guarded_call_site_unguarded():
    """The stale-disclosure direction, which is what actually happened."""
    problems = []
    for doc in (SECURITY, OVERVIEW):
        text = doc.read_text(encoding="utf-8")
        if "not yet universal" not in text:
            continue
        for rel in CREDENTIAL_BEARING:
            module = "vfairness." + rel.removesuffix(".py").replace("/", ".")
            if module in text and _guarded(rel):
                problems.append(
                    f"{doc.name}: says the guard is 'not yet universal' and names "
                    f"{module}, which HAS been guarded since 2026-08-27"
                )
    assert not problems, "\n".join(problems)


def test_the_docs_do_not_claim_universal_coverage_while_a_bypass_exists():
    """The overstatement direction, so the fix cannot swing too far.

    `operations/pulse/task_handlers.py` and
    `preprocessing/bias_detection/geographic_data.py` still call out directly.
    Neither takes a URL from the caller, so neither is an SSRF surface, but a
    blanket "all outbound calls are guarded" would be false while they exist.
    """
    bypasses = [
        rel
        for rel in (
            "operations/pulse/task_handlers.py",
            "preprocessing/bias_detection/geographic_data.py",
        )
        if not _guarded(rel)
    ]
    if not bypasses:
        return  # they were wired up; the docs may legitimately be simplified
    banned = re.compile(r"(every|all)\s+outbound\s+(call|request)s?\s+(is|are)\s+guarded", re.I)
    for doc in (SECURITY, OVERVIEW):
        text = doc.read_text(encoding="utf-8")
        assert not banned.search(text), (
            f"{doc.name} claims blanket coverage while these still bypass the guard:\n  "
            + "\n  ".join(bypasses)
        )


def test_the_docs_still_discuss_the_guard_at_all():
    """NON-VACUITY. Deleting the whole section would pass every test above."""
    for doc in (SECURITY, OVERVIEW):
        text = doc.read_text(encoding="utf-8")
        assert "egress guard" in text or "vfairness.net" in text, (
            f"{doc.name} no longer describes the egress guard; the checks above "
            "have nothing left to hold to the code"
        )
