"""G10 grading pins: vfairness.llm._base serialisers and vfairness.exceptions.

THE SERIALISER IS WHERE A RESULT STOPS BEING AN OBJECT AND BECOMES WHAT A READER
SEES. ``SerializableMixin`` is shared by roughly twenty result types across
``llm/``, ``multi_agent/`` and ``agents/``, and ``RunMetadata`` rides on all of
them, so a defect here is a defect in every one of their published envelopes.

Two defects pinned, both reproduced by execution first:

D3 ``to_json`` wrote a bare ``NaN`` / ``Infinity`` token, which is not JSON. Python's
   own ``json.loads`` accepts it as an extension and every strict parser rejects the
   WHOLE document, so the one result whose number could not be measured was the one
   a browser consumer could not read at all. This is exactly the defect G005-JSON
   fixed for ``CalibrationReport``; it was still live in the mixin the other twenty
   result types share.
D4 ``default=str`` MINTED CONTENT from absence: ``pd.NA`` serialised as the string
   ``"<NA>"``, ``pd.NaT`` as ``"NaT"``, ``np.ma.masked`` inside a dict as ``"--"``,
   and a nested ``np.float32(0.3)`` as the STRING ``"0.3"``, a number published as
   text. ``RunMetadata.parameters`` is precisely such a nested dict.

The exception classes are NOT measurements, and the check that matters for them is
that each carries its reason and does not lose it in ``str()``: a refusal nobody can
read is the same defect as a fabrication one layer up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

import vfairness
from vfairness.exceptions import (
    ConfigurationError,
    InsufficientDataError,
    InvalidDataError,
    ProtectedAttributeError,
    VfairnessError,
)
from vfairness.llm._base import RunMetadata, SerializableMixin

SPECIFIC = [InvalidDataError, InsufficientDataError, ProtectedAttributeError, ConfigurationError]
ALL_ERRORS = [VfairnessError] + SPECIFIC


@dataclass
class _Result(SerializableMixin):
    """A stand-in for the twenty real result dataclasses that share the mixin."""

    score: Any = None
    nested: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    arr: Any = None
    meta: Any = None


def _strict(payload: str) -> dict:
    """json.loads that REFUSES the NaN/Infinity extension, like every other parser."""

    def _reject(name):
        raise AssertionError(f"non-JSON constant {name!r} in the published envelope")

    return json.loads(payload, parse_constant=_reject)


# --------------------------------------------------------------------- D3: strict JSON

NON_FINITE = [
    float("nan"),
    float("inf"),
    -float("inf"),
    np.float64("nan"),
    np.float32("nan"),
    np.float32("inf"),
]


@pytest.mark.parametrize("value", NON_FINITE)
def test_to_json_is_strict_json_and_uses_null_for_a_non_finite_number(value):
    """D3. np.float32 needs its own branch: it does NOT subclass float, so an
    isinstance test against float alone lets a 32 bit NaN reach the encoder."""
    result = _Result(score=value, nested={"inner": value}, rows=[value], arr=np.array([1.0, value]))
    parsed = _strict(result.to_json())
    assert parsed["score"] is None
    assert parsed["nested"]["inner"] is None, "a nested container was never walked"
    assert parsed["rows"] == [None]
    assert parsed["arr"] == [1.0, None]
    assert "NaN" not in result.to_json() and "Infinity" not in result.to_json()


@pytest.mark.parametrize("value", NON_FINITE)
def test_to_dict_keeps_the_non_finite_float_for_in_process_consumers(value):
    """The split G005-JSON established: to_dict is the in-process surface and its
    consumers test it with np.isfinite, so it must NOT be flattened to None."""
    kept = _Result(score=value).to_dict()["score"]
    assert kept is not None
    assert not np.isfinite(kept)


# ------------------------------------------------------------------- D4: minted content

ABSENT = [pd.NA, pd.NaT, np.datetime64("NaT"), np.ma.masked]


@pytest.mark.parametrize("value", ABSENT)
def test_absence_serialises_to_null_and_never_to_a_minted_string(value):
    """D4. ``str(pd.NA)`` is ``'<NA>'``: a group named '<NA>' once carried a maximal
    finding of discrimination against a group that is not a group."""
    result = _Result(score=value, nested={"group": value}, rows=[value])
    payload = result.to_json()
    for minted in ('"<NA>"', '"NaT"', '"--"', '"nan"'):
        assert minted not in payload, f"{minted} was minted from an absent value"
    parsed = _strict(payload)
    assert parsed["score"] is None
    assert parsed["nested"]["group"] is None
    assert parsed["rows"] == [None]
    assert "score" in parsed, "an absent key is a different thing from a null one"


def test_np_ma_masked_is_null_and_the_route_it_takes_there_is_pinned():
    """``bool(pd.isna(np.ma.masked))`` is False, so the pandas test at the bottom of
    ``_json_safe`` cannot see it, which is why the identity line exists.

    MEASURED, AND THE MEASUREMENT IS THE POINT: removing that identity line left
    this whole file green, because masked IS an ndarray subclass whose ``tolist()``
    already answers None. So the line is defence in depth, not load-bearing, and
    the two premises it rests on are pinned here: the day either stops holding,
    this test goes red instead of the string ``'--'`` quietly appearing in an
    envelope.
    """
    assert bool(pd.isna(np.ma.masked)) is False, "then the pandas test would suffice"
    assert isinstance(np.ma.masked, np.ndarray), "then the array branch would not catch it"
    assert np.ma.masked.tolist() is None, "then the array branch would not answer None"
    assert str(np.ma.masked) == "--", "the content default=str used to mint"
    assert _strict(_Result(nested={"g": np.ma.masked}).to_json())["nested"]["g"] is None
    assert _strict(_Result(score=np.ma.masked).to_json())["score"] is None


NUMERIC = [
    (np.float32(0.5), 0.5),
    (np.float64(0.25), 0.25),
    (np.int64(3), 3),
    (np.int32(7), 7),
    (np.bool_(True), True),
    (np.bool_(False), False),
]


@pytest.mark.parametrize("value,expected", NUMERIC)
def test_a_nested_numpy_scalar_stays_a_number_and_is_not_published_as_text(value, expected):
    """D4's other half. ``"1" == 1`` is False, and a string where a number was
    expected is the door this campaign keeps finding."""
    result = _Result(nested={"threshold": value}, rows=[{"t": value}])
    parsed = _strict(result.to_json())
    assert parsed["nested"]["threshold"] == expected
    assert not isinstance(parsed["nested"]["threshold"], str)
    assert parsed["rows"][0]["t"] == expected
    assert not isinstance(parsed["rows"][0]["t"], str)
    plain = result.to_dict()["nested"]["threshold"]
    assert type(plain) in (float, int, bool), f"to_dict promised plain, gave {type(plain)}"


def test_a_numpy_boolean_is_not_turned_into_a_number():
    """A flag is never a measurement: np.bool_ must stay a JSON boolean, because on
    a 0-1 scale a 1.0 is a perfect score and on a 1-5 register it is a severity."""
    parsed = _strict(_Result(score=np.bool_(True), nested={"f": np.bool_(True)}).to_json())
    assert parsed["score"] is True and parsed["nested"]["f"] is True
    assert not isinstance(parsed["score"], (int, float)) or isinstance(parsed["score"], bool)


def test_control_every_measured_value_survives_including_a_real_zero():
    """CONTROL for D3 and D4. A serialiser that nulls everything passes every
    refusal pin above and destroys the unit."""
    result = _Result(
        score=0.0,
        nested={"zero": 0.0, "false": False, "empty": "", "neg": -1.5, "big": 1e12},
        rows=[0.0, 1.0, {"deep": 0.0}],
        arr=np.array([[0.0, 0.25]]),
    )
    parsed = _strict(result.to_json())
    assert parsed["score"] == 0.0 and parsed["score"] is not None
    assert parsed["nested"] == {"zero": 0.0, "false": False, "empty": "", "neg": -1.5, "big": 1e12}
    assert parsed["rows"] == [0.0, 1.0, {"deep": 0.0}]
    assert parsed["arr"] == [[0.0, 0.25]]


def test_control_a_string_that_looks_absent_is_left_alone():
    """DO NOT OVER-CORRECT. 'None', 'nan', 'NA', 'null', 'missing' and
    'None of the above' are all labels a caller may legitimately mean."""
    labels = ["None", "nan", "NaN", "<NA>", "NA", "null", "missing", "None of the above", "--"]
    parsed = _strict(_Result(rows=labels, nested={"g": "None"}).to_json())
    assert parsed["rows"] == labels
    assert parsed["nested"]["g"] == "None"


def test_to_dict_still_delegates_to_a_nested_result_and_to_items_in_a_list():
    inner = _Result(score=0.5)
    outer = _Result(meta=inner, rows=[inner, inner])
    payload = outer.to_dict()
    assert payload["meta"]["score"] == 0.5
    assert [row["score"] for row in payload["rows"]] == [0.5, 0.5]


def test_to_json_indent_is_honoured():
    assert "\n" not in _Result(score=1.0).to_json(indent=None)
    assert "\n" in _Result(score=1.0).to_json()


# ----------------------------------------------------------------------- RunMetadata


def test_run_metadata_carries_a_real_audit_trail():
    meta = RunMetadata(random_seed=7, system_type="llm", model_name="m")
    payload = meta.to_dict()
    assert payload["library_version"] == vfairness.__version__
    assert payload["random_seed"] == 7
    assert payload["system_type"] == "llm" and payload["model_name"] == "m"
    # The timestamp must be a real, parseable UTC instant, not a placeholder.
    from datetime import datetime

    stamp = datetime.fromisoformat(payload["timestamp"])
    assert stamp.tzinfo is not None
    assert set(payload) == {
        "timestamp",
        "library_version",
        "parameters",
        "random_seed",
        "system_type",
        "model_name",
    }, "a field-by-field rebuild here would drop every field added later"


def test_run_metadata_parameters_are_plain_and_json_safe():
    """D4 at the place it actually rides: `parameters` is caller-supplied and holds
    numpy scalars read off a DataFrame, on every result in the library."""
    meta = RunMetadata(
        parameters={
            "threshold": np.float32(0.25),
            "n": np.int64(7),
            "strict": np.bool_(False),
            "group": pd.NA,
            "unmeasured": float("nan"),
        }
    )
    plain = meta.to_dict()["parameters"]
    assert plain["threshold"] == pytest.approx(0.25, abs=1e-6)
    assert type(plain["threshold"]) is float
    assert plain["n"] == 7 and type(plain["n"]) is int
    assert plain["strict"] is False
    assert not np.isfinite(plain["unmeasured"]), "to_dict keeps NaN for np.isfinite consumers"

    parsed = _strict(_Result(meta=meta).to_json())["meta"]["parameters"]
    assert parsed["threshold"] == pytest.approx(0.25, abs=1e-6)
    assert not isinstance(parsed["threshold"], str), "a threshold published as text"
    assert parsed["n"] == 7 and not isinstance(parsed["n"], str)
    assert parsed["strict"] is False
    assert parsed["group"] is None, "pd.NA minted a group named '<NA>'"
    assert parsed["unmeasured"] is None


def test_run_metadata_default_parameters_are_per_instance():
    RunMetadata().parameters["leak"] = 1
    assert RunMetadata().parameters == {}


# ------------------------------------------------------------------------- exceptions


@pytest.mark.parametrize("exc", ALL_ERRORS)
def test_every_error_keeps_its_reason_through_str_and_args(exc):
    """A refusal nobody can read is the same defect as a fabrication one layer up."""
    reason = "protected attribute 'gender' has only one value, so no groups to compare"
    raised = exc(reason)
    assert str(raised) == reason
    assert raised.args == (reason,)
    assert reason in repr(raised)
    with pytest.raises(exc) as caught:
        raise exc(reason)
    assert str(caught.value) == reason


@pytest.mark.parametrize("exc", SPECIFIC)
def test_specific_errors_are_catchable_as_the_base_and_as_valueerror(exc):
    """Both inheritance legs are load-bearing: the base is the single catch the
    hierarchy exists for, ValueError keeps every pre-existing handler working."""
    assert issubclass(exc, VfairnessError) and issubclass(exc, ValueError)
    for catcher in (exc, VfairnessError, ValueError, Exception):
        with pytest.raises(catcher):
            raise exc("reason")


def test_the_base_is_not_a_valueerror_so_the_two_legs_stay_distinguishable():
    assert not issubclass(VfairnessError, ValueError)


@pytest.mark.parametrize("exc", ALL_ERRORS)
def test_errors_are_exported_and_mint_no_value(exc):
    assert exc.__name__ in vfairness.__all__
    assert getattr(vfairness, exc.__name__) is exc
    assert exc.__module__ == "vfairness.exceptions"
    # An exception is not a measurement and must not carry one.
    raised = exc("reason")
    assert not hasattr(raised, "value")
    assert not hasattr(raised, "to_dict")


def test_an_error_raised_with_no_reason_reads_as_empty_not_as_a_reason():
    """A bare raise must not fabricate a message; str() of it is empty, and that is
    a visible absence rather than an invented explanation."""
    for exc in ALL_ERRORS:
        assert str(exc()) == ""


def test_the_hierarchy_is_exactly_the_documented_set():
    from vfairness import exceptions

    assert set(exceptions.__all__) == {e.__name__ for e in ALL_ERRORS}
