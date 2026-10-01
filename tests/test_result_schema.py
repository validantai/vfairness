"""Tests for the library-wide structured result envelope (vfairness.result).

Guards the contract that VI_A_GR_030 introduced:
    - TaskResult.ok / fail / from_envelope build the right shape.
    - to_dict() stays backward compatible with the historical
      ``{"success": ..., "data"|"error": ...}`` dict (only additive keys).
    - The two wired task handlers still emit success / data / error the same
      way, now carrying schema_version + task_type.
"""

from __future__ import annotations

from vfairness.result import SCHEMA_VERSION, TaskResult

# TaskResult unit behaviour


def test_ok_shape():
    r = TaskResult.ok("vfairness_data_validation", data={"rows": 10})
    d = r.to_dict()
    assert d["success"] is True
    assert d["data"] == {"rows": 10}
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["task_type"] == "vfairness_data_validation"
    # A success envelope must NOT carry an error key (backward compatible).
    assert "error" not in d


def test_fail_shape():
    r = TaskResult.fail("vfairness_pulse_run", "boom")
    d = r.to_dict()
    assert d["success"] is False
    assert d["error"] == "boom"
    assert d["task_type"] == "vfairness_pulse_run"
    # A failure envelope must NOT carry a data key.
    assert "data" not in d


def test_backward_compatible_keys():
    """Old key-based consumers keep working: success + data/error present,
    and none of the historical values changed."""
    ok = TaskResult.ok("t", data={"x": 1}).to_dict()
    fail = TaskResult.fail("t", "e").to_dict()
    # The historical subset is exactly reproduced.
    assert {k: ok[k] for k in ("success", "data")} == {"success": True, "data": {"x": 1}}
    assert {k: fail[k] for k in ("success", "error")} == {"success": False, "error": "e"}


def test_warnings_only_emitted_when_present():
    without = TaskResult.ok("t", data={}).to_dict()
    assert "warnings" not in without
    with_w = TaskResult.ok("t", data={}, warnings=["slow"]).to_dict()
    assert with_w["warnings"] == ["slow"]


def test_from_envelope_preserves_data_without_nesting():
    env = {"success": True, "data": {"verdict": "pass"}}
    d = TaskResult.from_envelope("vfairness_pulse_run", env).to_dict()
    assert d["success"] is True
    # data is preserved verbatim, never wrapped in another data layer.
    assert d["data"] == {"verdict": "pass"}
    assert "data" not in d["data"]
    assert d["schema_version"] == SCHEMA_VERSION


def test_from_envelope_preserves_error():
    env = {"success": False, "error": "no column"}
    d = TaskResult.from_envelope("vfairness_pulse_run", env).to_dict()
    assert d["success"] is False
    assert d["error"] == "no column"
    assert "data" not in d


def test_from_envelope_handles_non_dict():
    d = TaskResult.from_envelope("t", None).to_dict()
    assert d["success"] is False
    assert "non-dict" in d["error"]


# Handler integration


def test_cicd_handler_emits_versioned_envelope():
    from vfairness.operations.cicd.task_handlers import handle_data_validation

    # Missing dataset -> failure envelope in the new shape.
    d = handle_data_validation({})
    assert d["success"] is False
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["task_type"] == "vfairness_data_validation"
    assert "data" not in d

    # A valid tiny dataset -> success envelope, data preserved.
    csv = "gender,approved\nM,1\nF,0\nM,1\nF,1\n"
    ok = handle_data_validation(
        {"csv_data": csv, "protected_attributes": ["gender"], "outcome_column": "approved"}
    )
    assert ok["success"] is True
    assert ok["task_type"] == "vfairness_data_validation"
    assert isinstance(ok["data"], dict)
    assert "error" not in ok


def test_pulse_handler_missing_artifact_envelope():
    from vfairness.operations.pulse.task_handlers import handle_pulse_run

    d = handle_pulse_run({"inputs": {}})
    assert d["success"] is False
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["task_type"] == "vfairness_pulse_run"
    assert "artifact" in d["error"].lower()
