"""The Beta Go-Live proof ledger must never be able to flatter the library.

These tests exist because the ledger is a claim ABOUT a defect class, and so it
can carry that defect class itself. The specific failure to prevent: a capability
nobody checked ending up indistinguishable from one that passed. Every assertion
below is aimed at that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vfairness._proof_status import (
    CAPABILITY_PROOF,
    MEASURED_AT_COMMIT,
    MEASURED_ON,
    PROOF_BATCHES,
    proof_status,
    proof_summary,
)
from vfairness._registry import CAPABILITY_REGISTRY

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "docs" / "beta-go-live-census-2026-09-11.json"


def test_every_registered_capability_has_a_proof_record() -> None:
    """No capability may be absent. Absence is how "unchecked" becomes invisible."""
    missing = sorted(set(CAPABILITY_REGISTRY) - set(CAPABILITY_PROOF))
    assert not missing, f"{len(missing)} capabilities carry no proof record: {missing[:10]}"


def test_the_ledger_invents_no_capabilities() -> None:
    extra = sorted(set(CAPABILITY_PROOF) - set(CAPABILITY_REGISTRY))
    assert not extra, f"proof records for unregistered keys: {extra}"


def test_every_batch_label_is_defined() -> None:
    for key, rec in CAPABILITY_PROOF.items():
        assert rec["batch"] in PROOF_BATCHES, f"{key} claims undefined batch {rec['batch']!r}"


def test_a_proven_capability_has_no_open_defect() -> None:
    """BGL-A and BGL-B both assert open_defects == 0. If that slips, the whole
    ladder inverts: the batch a reader trusts most would be the one hiding a
    reproduced fabrication."""
    for key, rec in CAPABILITY_PROOF.items():
        if rec["batch"] in ("BGL-A", "BGL-B"):
            assert rec["open_defects"] == 0, (
                f"{key} is {rec['batch']} while carrying {rec['open_defects']} open defect(s)"
            )


def test_a_defect_capability_actually_has_a_defect() -> None:
    for key, rec in CAPABILITY_PROOF.items():
        if rec["batch"] == "BGL-D":
            assert rec["open_defects"] >= 1, f"{key} is BGL-D with no open defect"
            assert rec["fix_minutes"] > 0, f"{key} is BGL-D with no effort estimate"


def test_every_pin_file_exists() -> None:
    """A pin naming a file that is not there is worse than no pin: it reads as
    protection. This is the assertion that makes BGL-A mean anything."""
    for key, rec in CAPABILITY_PROOF.items():
        for pin in rec["pins"]:
            assert (ROOT / pin).exists(), f"{key} claims a pin that does not exist: {pin}"


def test_proven_requires_a_pin_and_semi_proven_requires_none() -> None:
    """The single distinction between BGL-A and BGL-B. If it blurs, SEMI-PROVEN
    silently becomes PROVEN and the table stops carrying information."""
    for key, rec in CAPABILITY_PROOF.items():
        if rec["batch"] == "BGL-A":
            assert rec["pins"], f"{key} is BGL-A with no pin"
        if rec["batch"] == "BGL-B":
            assert not rec["pins"], f"{key} is BGL-B but has pins {rec['pins']}"


def test_unproven_claims_nothing() -> None:
    """BGL-C must never carry evidence. If it does, it was really B or D and the
    ladder has rounded a measurement down to "unknown" -- the reverse defect."""
    for key, rec in CAPABILITY_PROOF.items():
        if rec["batch"] == "BGL-C":
            assert rec["open_defects"] == 0, f"{key} is BGL-C with a known defect"
            assert not (rec["reached"] and rec["judged"]), (
                f"{key} is BGL-C but was both reached and judged; it is at least BGL-B"
            )


def test_the_summary_counts_every_capability_exactly_once() -> None:
    summary = proof_summary()
    assert sum(summary.values()) == len(CAPABILITY_REGISTRY)


def test_unknown_key_returns_none_not_a_clean_bill() -> None:
    """``None`` means "not a capability". It must never be confused with a record
    whose batch is BGL-C: one is "no such thing", the other is "unchecked"."""
    assert proof_status("definitely_not_a_capability") is None
    some = next(iter(CAPABILITY_PROOF))
    assert proof_status(some) is not None


def test_the_ledger_matches_its_evidence_file() -> None:
    """The ledger is derived, not authored. If someone edits it by hand to make a
    number look better, this goes red."""
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["measured_on"] == MEASURED_ON
    assert evidence["commit"] == MEASURED_AT_COMMIT
    caps = evidence["capabilities"]
    for key, rec in CAPABILITY_PROOF.items():
        src = caps.get(key)
        assert src is not None, f"{key} is in the ledger but not in the evidence"
        assert rec["open_defects"] == src["open_defects"], (
            f"{key}: ledger says {rec['open_defects']} open defects, evidence says "
            f"{src['open_defects']}"
        )
        assert rec["judged"] == src["judged"], f"{key}: judged count drifted from the evidence"


def test_the_ledger_is_not_vacuous() -> None:
    """A gate that examines nothing passes forever. These floors are deliberately
    set from the 2026-09-11 census: if a later change empties the ledger or turns
    every record into the same batch, this fails rather than going quietly green."""
    summary = proof_summary()
    assert len(CAPABILITY_PROOF) >= 200, "the ledger lost capabilities"
    assert sum(1 for r in CAPABILITY_PROOF.values() if r["pins"]) >= 50, (
        "pin detection collapsed; BGL-A would be empty and the table meaningless"
    )
    # This asserted ">= 3 occupied batches" until 2026-09-17. That expectation
    # was calibrated while BGL-C and BGL-D still had members, so CLOSING them
    # made it fail: the assertion punished the outcome the ledger exists to
    # reach. The SUBJECT it was guarding is still real and is kept here, namely
    # that the ladder must not collapse into a single flattering batch. What
    # changed is only the mechanism: at least two batches occupied, and no
    # single batch holding essentially everything.
    occupied = [b for b, n in summary.items() if n > 0]
    assert len(occupied) >= 2, (
        f"every capability landed in one batch ({occupied}); the ladder has stopped discriminating"
    )
    biggest = max(summary.values())
    assert biggest <= 0.95 * len(CAPABILITY_PROOF), (
        f"one batch holds {biggest} of {len(CAPABILITY_PROOF)} capabilities, so "
        "the grade carries almost no information"
    )


@pytest.mark.parametrize("batch", sorted(PROOF_BATCHES))
def test_each_batch_definition_states_its_limits(batch: str) -> None:
    """A batch that says what it proves but not what it does NOT prove is the
    two-state failure in prose form."""
    d = PROOF_BATCHES[batch]
    for field in ("label", "means", "requires", "may_not_conclude"):
        assert d[field].strip(), f"{batch} has an empty {field}"
    assert len(d["may_not_conclude"]) > 60, f"{batch} does not seriously state its limits"


# ---------------------------------------------------------------------------
# The mark on the function itself.
#
# The ledger is machine-readable; a docstring is what a person actually reads.
# A correct measurement no reader can see is the same defect one layer up, so
# these tests treat the docstring as the delivery surface, not as decoration.
# ---------------------------------------------------------------------------


def _capability_objects():
    import importlib

    for key, entry in sorted(CAPABILITY_REGISTRY.items()):
        mod = importlib.import_module(f"vfairness.{entry['module_path']}")
        yield key, entry, getattr(mod, entry["name"], None)


def test_every_capability_carries_its_proof_mark_at_runtime() -> None:
    """``help(thing)`` must say what is and is not known about it."""
    missing = [
        key
        for key, _entry, obj in _capability_objects()
        if "Beta Go-Live proof status" not in (getattr(obj, "__doc__", None) or "")
    ]
    assert not missing, (
        f"{len(missing)} capabilities are reachable with no proof mark in their "
        f"docstring: {missing[:10]}"
    )


def _mark_block(obj) -> str:
    """Just the stamped block, never the surrounding prose.

    Locating the subject by quoting it is how the first version of this test went
    blind: ``exposure_parity_difference`` appears in its own docstring examples,
    so "is the dispatch key mentioned anywhere" was true whether or not the mark
    named it.
    """
    doc = " ".join((getattr(obj, "__doc__", None) or "").split())
    start = doc.find("Beta Go-Live proof status")
    if start < 0:
        return ""
    end = doc.find("(end Beta Go-Live proof status)", start)
    if end < 0:
        return ""
    return doc[start : end + len("(end Beta Go-Live proof status)")]


def test_the_mark_names_the_ledger_row() -> None:
    """Without the dispatch key IN THE MARK a reader cannot get from the docstring
    back to the evidence, which is where the batch is justified."""
    orphan = []
    for key, _entry, obj in _capability_objects():
        block = _mark_block(obj)
        if not block:
            continue
        row = block.split("Ledger row")
        if len(row) < 2 or key not in row[1].split("See docs/")[0]:
            orphan.append(key)
    assert not orphan, f"proof marks that do not name their ledger row: {orphan[:10]}"


def test_a_shared_object_carries_the_worst_of_its_batches() -> None:
    """Two dispatch keys can land on one function, and a docstring holds one
    answer. If the better batch won, a proven alias would launder a capability
    that fabricates through its other key."""
    rank = {"BGL-D": 3, "BGL-C": 2, "BGL-B": 1, "BGL-A": 0}
    by_obj: dict[int, list[str]] = {}
    objs: dict[int, object] = {}
    for key, _entry, obj in _capability_objects():
        if obj is None:
            continue
        by_obj.setdefault(id(obj), []).append(key)
        objs[id(obj)] = obj
    for oid, keys in by_obj.items():
        if len(keys) < 2:
            continue
        worst = max(keys, key=lambda k: rank[CAPABILITY_PROOF[k]["batch"]])
        census = CAPABILITY_PROOF[worst]["batch"]
        obj = objs[oid]
        block = _mark_block(obj)
        if not block:
            continue
        found = [b for b in PROOF_BATCHES if b in block]
        assert len(found) == 1, (
            f"object shared by {keys} carries {found} batch marks; one answer or none"
        )
        # NO BETTER THAN THE CENSUS, rather than equal to it. The subject is
        # unchanged and it is the only thing that matters here: if the better batch
        # won, a proven alias would launder a capability that fabricates through
        # its other key. A mark that is WORSE cannot do that, and the stamper
        # writes one deliberately when today's ledger records an open defect that
        # the dated census predates. See _live_fix_pending_units.
        assert rank[found[0]] >= rank[census], (
            f"object shared by {keys} is marked {found[0]}, which is BETTER than "
            f"{worst}'s {census}, so the proven alias launders the other key"
        )
        if rank[found[0]] > rank[census]:
            assert _live_fix_pending_units(obj), (
                f"object shared by {keys} is marked {found[0]} against a census of "
                f"{census}, and no unit of it is FIX PENDING in the live ledger, so "
                "nothing accounts for the downgrade"
            )


def _live_fix_pending_units(obj) -> list[str]:
    """This object's units that the LIVE ledger records as FIX PENDING.

    The census in ``_proof_status.py`` is DATED (MEASURED_ON) and keeps its
    historical meaning on purpose. ``scripts/stamp_proof_status.py`` deliberately
    stamps what is established NOW, reading docs/capability-status.json, and a unit
    in FIX PENDING forces the block down to BGL-D whatever the census says. Its
    docstring calls that "THE GATE THAT DID NOT EXIST", and it is the gate that
    stopped 59 rows claiming PROVEN for sixteen days after a defect was found.

    So the two sources disagree BY DESIGN from the moment a defect is found after
    the census date, and the disagreement is always in the same direction: the
    docstring gets worse, never better. Reproduced 2026-09-28 after a stamping run:
    51 units are FIX PENDING and seventeen capability marks read BGL-D against a
    census of BGL-A or BGL-B.
    """
    module = getattr(obj, "__module__", "") or ""
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "")
    if not module or not qualname:
        return []
    qual = f"{module}.{qualname}"
    path = ROOT / "docs" / "capability-status.json"
    if not path.exists():
        return []
    units = (json.loads(path.read_text(encoding="utf-8")).get("units") or {}).items()
    return sorted(
        q
        for q, rec in units
        if (q == qual or q.startswith(qual + ".")) and rec.get("status") == "FIX PENDING"
    )


def test_the_mark_agrees_with_the_ledger() -> None:
    """The batch printed in the docstring must BE the batch in the ledger (or the
    worst among the keys sharing that object). The first version of this test only
    checked the limit text "if the batch is mentioned", so editing a BGL-D mark to
    read BGL-A made the test skip instead of fail: the escape hatch was the bug."""
    rank = {"BGL-D": 3, "BGL-C": 2, "BGL-B": 1, "BGL-A": 0}
    shared: dict[int, list[str]] = {}
    for key, _entry, obj in _capability_objects():
        if obj is not None:
            shared.setdefault(id(obj), []).append(key)

    wrong = []
    for key, _entry, obj in _capability_objects():
        block = _mark_block(obj)
        if not block:
            continue
        siblings = shared.get(id(obj), [key])
        expected = CAPABILITY_PROOF[
            max(siblings, key=lambda k: rank[CAPABILITY_PROOF[k]["batch"]])
        ]["batch"]
        found = [b for b in PROOF_BATCHES if b in block]
        if found == [expected]:
            continue
        # WORSE IS ALLOWED, BETTER IS NOT, and only with a live reason. See
        # _live_fix_pending_units: the stamper reads today's ledger, so a defect
        # found after the census date legitimately drags the mark down. Requiring
        # exact equality here demanded the stamper be run (the staleness check two
        # tests up) and then failed because it had been. An UPGRADE still fails,
        # which is the direction that would be a fabrication, and so does a
        # downgrade nothing in the live ledger accounts for. What this test does NOT
        # cover, deliberately: a live downgrade EDITED BACK to the census value
        # reads as equal here and is caught by test_no_docstring_stamp_is_stale,
        # which compares against the live evidence rather than the census.
        if len(found) == 1 and rank[found[0]] > rank[expected]:
            pending = _live_fix_pending_units(obj)
            if pending:
                continue
            wrong.append((key, expected, found, "downgraded with no FIX PENDING unit"))
            continue
        wrong.append((key, expected, found, "not worse-or-equal to the census"))
    assert not wrong, (
        f"docstring marks disagreeing with the ledger (key, expected, found, why): {wrong[:10]}"
    )


def test_the_mark_states_what_it_does_not_prove() -> None:
    """Every batch's mark has to carry its own limit, or the mark becomes the
    two-state failure in prose."""
    limits = {
        "BGL-A": "does",
        "BGL-B": "not protected",
        "BGL-C": "UNKNOWN",
        "BGL-D": "DEFECT OPEN",
    }
    for key, _entry, obj in _capability_objects():
        doc = getattr(obj, "__doc__", None) or ""
        if "Beta Go-Live proof status" not in doc:
            continue
        block = _mark_block(obj)
        found = [b for b in limits if b in block]
        assert len(found) == 1, f"{key}: mark names {found} batches, expected exactly one"
        assert limits[found[0]] in block, f"{key}: {found[0]} mark states no limit"


# ---------------------------------------------------------------------------
# Staleness. Every published surface is generated; CI has to notice when one of
# them stops matching its source, or the tables quietly become decoration.
# ---------------------------------------------------------------------------


def _run(script: str) -> tuple[int, str]:
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [_sys.executable, str(ROOT / "scripts" / script), "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_the_ledger_is_not_stale() -> None:
    code, out = _run("beta_go_live_ledger.py")
    assert code == 0, f"the ledger no longer matches its evidence file:\n{out}"


def test_no_docstring_stamp_is_stale() -> None:
    """A capability whose batch changed but whose docstring still claims the old
    one is worse than an unstamped capability: it is a wrong answer in the place
    a reader looks first."""
    code, out = _run("stamp_proof_status.py")
    assert code == 0, f"stale proof stamps; run scripts/stamp_proof_status.py:\n{out}"


def test_the_published_tables_match_the_ledger() -> None:
    code, out = _run("beta_go_live_docs.py")
    assert code == 0, f"docs tables drifted from the ledger:\n{out}"


def test_the_plan_document_exists_and_defines_every_batch() -> None:
    plan = ROOT / "docs" / "BETA_GO_LIVE_PLAN.md"
    assert plan.exists(), "docs/BETA_GO_LIVE_PLAN.md is referenced from 215 docstrings"
    text = plan.read_text(encoding="utf-8")
    for batch in PROOF_BATCHES:
        assert batch in text, f"{batch} is not defined in the plan the docstrings point at"


def test_keys_sharing_one_object_share_one_verdict() -> None:
    """Two dispatch keys can resolve to the SAME live object. Their evidence must
    be the same evidence.

    This test exists because the first build of the ledger mapped name -> key with
    a plain dict, which keeps only the LAST key of a duplicated name. Evidence
    landed on one key of each pair and its twin was recorded as never checked:
    `calibration_disparity` came out BGL-C UNPROVEN while `calibration_difference`,
    the same function, came out BGL-A PROVEN. That is the REVERSE defect, real
    evidence thrown away while reading as caution, committed by the very ledger
    built to catch it.
    """
    import importlib

    by_obj: dict[int, list[str]] = {}
    for key, entry in sorted(CAPABILITY_REGISTRY.items()):
        mod = importlib.import_module(f"vfairness.{entry['module_path']}")
        obj = getattr(mod, entry["name"], None)
        if obj is not None:
            by_obj.setdefault(id(obj), []).append(key)

    disagreeing = []
    for keys in by_obj.values():
        if len(keys) < 2:
            continue
        batches = {CAPABILITY_PROOF[k]["batch"] for k in keys}
        defects = {CAPABILITY_PROOF[k]["open_defects"] for k in keys}
        if len(batches) > 1 or len(defects) > 1:
            disagreeing.append((keys, sorted(batches), sorted(defects)))
    assert not disagreeing, (
        "dispatch keys resolving to one object disagree about it "
        f"(keys, batches, open_defects): {disagreeing}"
    )


def test_shared_with_names_only_real_siblings() -> None:
    """``shared_with`` must list keys that really do resolve to the same object,
    both ways. A stale sibling list would make the test above vacuous."""
    for key, rec in CAPABILITY_PROOF.items():
        for sib in rec["shared_with"]:
            assert sib in CAPABILITY_PROOF, f"{key} lists unknown sibling {sib}"
            assert key in CAPABILITY_PROOF[sib]["shared_with"], (
                f"{key} claims sibling {sib}, but {sib} does not claim {key}"
            )


def test_the_evidence_file_is_not_self_certified() -> None:
    """No capability may clear its OWN row in the evidence file.

    Found live on 2026-09-17: a fix agent, working on create_reweighter and
    integrated_gradients, set ``open_defects`` to 0 for those two rows in
    docs/beta-go-live-census-2026-09-11.json. Its code fix may well have been
    right; that is not the point. The evidence file is the INDEPENDENT record the
    ledger is generated from, and a worker writing its own verdict into it is the
    same move as the six hand-edited docstring stamps: a claim entered where a
    measurement belongs.

    A cleared row must carry the marker the audited pipeline writes, and that
    marker is only written by scripts run from the orchestrating session after an
    independent auditor held the fix. A row at zero with no marker means someone
    edited the file by hand.
    """
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    unmarked = []
    for key, rec in sorted(evidence["capabilities"].items()):
        if rec.get("open_defects"):
            continue
        # A row that never had a defect needs no marker; only a CLEARED one does.
        if rec.get("severities") or rec.get("fix_minutes"):
            unmarked.append(key)
            continue
        if rec.get("stage1_fixed") or rec.get("stage2_fixed"):
            continue
        # Untouched rows (never had a defect) are indistinguishable from cleared
        # ones only if they also carry no fix_minutes and no severities, which
        # the branch above already required. Nothing more to check here.
    assert not unmarked, (
        "evidence rows at open_defects=0 that still carry severities or "
        f"fix_minutes, i.e. cleared without the audited marker: {unmarked[:10]}"
    )


def test_a_cleared_row_names_the_stage_that_cleared_it() -> None:
    """Anti-vacuity for the test above. If no row carries a marker, that test
    passes over an empty set and proves nothing."""
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    marked = [
        k
        for k, r in evidence["capabilities"].items()
        if r.get("stage1_fixed") or r.get("stage2_fixed")
    ]
    assert len(marked) >= 50, (
        f"only {len(marked)} evidence rows carry an audited-clearance marker; "
        "the self-certification check has almost nothing to examine"
    )


# ---------------------------------------------------------------------------
# The ledger's own SCOPE. The counts above grade 215 capability rows; the
# exported surface is larger. Reporting one as the other is this campaign's
# defect class turned on its own report, so the published scope is measured
# here rather than asserted in prose.
# ---------------------------------------------------------------------------


def _ungraded_exported_functions() -> list[str]:
    import inspect

    import vfairness

    known = set(CAPABILITY_PROOF)
    for record in CAPABILITY_PROOF.values():
        known.update(record.get("shared_with") or ())
    out = []
    for name in getattr(vfairness, "__all__", ()):
        obj = getattr(vfairness, name, None)
        if not inspect.isfunction(obj) or name in known:
            continue
        if "Beta Go-Live proof status" in (inspect.getdoc(obj) or ""):
            continue
        out.append(name)
    return sorted(out)


@pytest.mark.parametrize("doc", ["QUALITY_AND_HARDENING.md", "BETA_GO_LIVE_PLAN.md"])
def test_the_published_scope_states_the_real_number_of_ungraded_functions(doc: str) -> None:
    """The count in the document must equal what the package actually exports.

    Measured on 2026-09-17: 215 graded rows, and 88 exported functions carrying
    no batch at all. Before this block the page said the census ran "across the
    whole surface" and printed a 100% total, which a reader could only read as
    covering the library.
    """
    text = (ROOT / "docs" / doc).read_text(encoding="utf-8")
    assert "BGL:scope:start" in text, f"{doc} publishes counts with no scope statement"
    n = len(_ungraded_exported_functions())
    assert f"| Plain functions with no batch | {n} |" in text, (
        f"{doc} does not state the real ungraded-function count ({n}); "
        "run scripts/beta_go_live_docs.py"
    )
    for name in _ungraded_exported_functions()[:5]:
        assert f"`{name}`" in text, f"{doc} omits the ungraded function {name}"


@pytest.mark.parametrize("doc", ["QUALITY_AND_HARDENING.md", "BETA_GO_LIVE_PLAN.md"])
def test_no_document_claims_the_census_covered_the_whole_surface(doc: str) -> None:
    """The exact sentence that was wrong, pinned so it cannot come back."""
    text = (ROOT / "docs" / doc).read_text(encoding="utf-8")
    assert "across the whole surface" not in text, (
        f"{doc} claims whole-surface coverage; the census graded "
        f"{len(CAPABILITY_PROOF)} rows and left "
        f"{len(_ungraded_exported_functions())} exported functions ungraded"
    )


def test_the_scope_check_is_not_vacuous() -> None:
    """If the ungraded set were empty the two tests above would assert nothing.

    This is the anti-vacuity anchor. When the next wave grades these functions
    this test is what forces the scope paragraph to be rewritten rather than
    quietly passing over an empty set.
    """
    ungraded = _ungraded_exported_functions()
    assert len(ungraded) >= 1, (
        "no exported function is ungraded any more: rewrite the scope block "
        "in scripts/beta_go_live_docs.py instead of leaving a paragraph that "
        "describes a set with nothing in it"
    )
    assert "identify_proxy_features" in ungraded or len(ungraded) < 88
