"""B6 and B7: the two executable checks are conditions of the one beta gate.

Added 2026-10-01 after the quality page said "Beta gate passed" from the two
checks while scripts/release_gate.py said NOT BETA READY. They are now criteria
of that gate, and each must fail closed: missing evidence, a disagreement with a
reference, or a single fabrication turns it red.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import release_gate as RG  # noqa: E402

GOOD = {
    "check1_correct_on_normal_data": {
        "state": "ran",
        "entries": 110,
        "with_reference": 34,
        "agree": 34,
        "differ": [],
        "no_reference": 76,
    },
    "check2_honest_on_broken_data": {
        "state": "ran",
        "entries": 110,
        "pass": 109,
        "fabricates": 0,
        "over_refuses": 0,
        "not_a_measurement": 1,
        "not_covered": 0,
    },
}


def _verdict(monkeypatch, data):
    monkeypatch.setattr(RG, "_gate_checks", lambda: data)
    return {c["id"]: c["passes"] for c in RG._check_criteria()}


def test_both_pass_on_good_evidence(monkeypatch):
    assert _verdict(monkeypatch, GOOD) == {"B6": True, "B7": True}


def test_missing_evidence_fails_both(monkeypatch):
    assert _verdict(monkeypatch, None) == {"B6": False, "B7": False}


@pytest.mark.parametrize(
    "path,value,failing",
    [
        (("check1_correct_on_normal_data", "differ"), ["auroc_parity"], "B6"),
        (("check1_correct_on_normal_data", "agree"), 33, "B6"),
        (("check1_correct_on_normal_data", "state"), "could_not_check", "B6"),
        (("check2_honest_on_broken_data", "pass"), 108, "B7"),
        (("check2_honest_on_broken_data", "state"), "could_not_check", "B7"),
    ],
)
def test_each_failure_turns_its_condition_red(monkeypatch, path, value, failing):
    data = copy.deepcopy(GOOD)
    data[path[0]][path[1]] = value
    v = _verdict(monkeypatch, data)
    assert v[failing] is False
    assert v[{"B6": "B7", "B7": "B6"}[failing]] is True


def test_both_are_in_the_beta_gate():
    assert {"B6", "B7"} <= {c["id"] for c in RG._beta_criteria()}
