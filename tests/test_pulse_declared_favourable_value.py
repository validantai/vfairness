"""A text decision column is coded from its DECLARED favourable value, never guessed.

Measured 2026-10-09 on the recruitment sample (tests/fixtures/
recruitment_fairness_dataset.csv, the platform's public sample, byte for byte):
its decision column ``invite_decision`` is "Yes"/"No". Recoded as
"Invited"/"Rejected", the same decisions read three different ways:

* ``run_pulse`` coded a fixed list of "truthy" words as 1. "invited" is not on it,
  so every applicant read as not invited and every selection rate was 0.
* ``DataBiasValidator`` took the alphabetically later value as 1, so "Rejected"
  became the favourable outcome: the gender min/max rate ratio read 0.955, a
  pass, where the true reading is 0.358, a fail.
* the quarantined legacy pipeline did the same as the validator.

Now a text decision is coded only from ``inputs["decision_labels"]
["favourable_values"][column]`` (the shape the platform's other analyses use) or
``DataBiasValidator.validate(..., outcome_favourable_value=...)``. Undeclared,
Pulse refuses by name and the validator reports ``outcome_not_coded``.

Each pin was run against origin/main and failed there, except the controls.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.validator import DataBiasValidator, _to_numeric_binary
from vfairness.operations.pulse import orchestrator as o

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "recruitment_fairness_dataset.csv"
TRUE_GENDER_RATIO = 0.3576  # 1 of 41 Non-binary vs 92 of 1,349 Male invited

ATTRS = ["age", "gender", "race_ethnicity", "marital_status"]


def _recruitment(coding: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURE).drop(columns=["true_qualification_score", "_bias_flags"])
    if coding == "invited":
        df["invite_decision"] = df["invite_decision"].map({"Yes": "Invited", "No": "Rejected"})
    elif coding == "numeric":
        df["invite_decision"] = (df["invite_decision"] == "Yes").astype(int)
    return df


def _inputs(**extra):
    d = {
        "source_kind": "predictive",
        "domain": "hiring",
        "jurisdiction": "EU",
        "protected_attributes": ATTRS,
    }
    d.update(extra)
    return d


def _declare(value):
    return {"decision_labels": {"favourable_values": {"invite_decision": value}}}


class _Captured(BaseException):
    """Raised once the per-variable rows exist, so the rest of the run is skipped."""


@pytest.fixture
def per_variable_rows(monkeypatch):
    """run_pulse's per-variable rows, captured right after the BH-FDR step.

    The rows are exactly the engine's; only the stages after them are skipped.
    A refused run returns its envelope before that point."""
    original = o._safe

    def spy(fn, default, *args, **kwargs):
        out = original(fn, default, *args, **kwargs)
        if kwargs.get("label") == "per_variable_fdr":
            for cell in fn.__closure__ or ():
                value = cell.cell_contents
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    raise _Captured(value)
        return out

    monkeypatch.setattr(o, "_safe", spy)

    def run(df, inputs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return {"envelope": o.run_pulse(df, inputs)}
            except _Captured as captured:
                rows = captured.args[0]
                return {r["attribute"]: r for r in rows if r.get("assessable")}

    return run


def _ratio(rates: pd.Series) -> float:
    return round(float(rates.min() / rates.max()), 4)


# --------------------------------------------------------------- run_pulse


def test_an_undeclared_text_decision_is_refused_by_name(per_variable_rows):
    """THE DEFECT. On main this ran on the "truthy" guess."""
    out = per_variable_rows(_recruitment("yes_no"), _inputs())
    env = out.get("envelope")
    assert env is not None, "an undeclared text decision ran on a guess"
    assert env["success"] is False and env["refused"] is True
    assert env["decisionColumn"] == "invite_decision"
    assert env["decisionValues"] == ["No", "Yes"]
    assert env["setting"] == "decision_labels.favourable_values"
    assert "does not guess" in env["error"] and "invite_decision" in env["error"]


def test_invited_rejected_is_read_as_declared_not_as_nobody_invited(per_variable_rows):
    """THE FLIP. Declared "Invited" reads exactly as the 0/1-coded column; on main
    the truthy list coded every applicant 0."""
    declared = per_variable_rows(_recruitment("invited"), _inputs(**_declare("Invited")))
    numeric = per_variable_rows(_recruitment("numeric"), _inputs())

    assert declared["gender"]["fourFifthsRatio"] == pytest.approx(TRUE_GENDER_RATIO, abs=1e-4)
    for attr in ATTRS:
        assert declared[attr]["gap"] == pytest.approx(numeric[attr]["gap"]), attr
        assert declared[attr]["fourFifthsRatio"] == pytest.approx(
            numeric[attr]["fourFifthsRatio"]
        ), attr
        assert declared[attr]["significant"] == numeric[attr]["significant"], attr


def test_yes_no_declared_yes_matches_the_numeric_column(per_variable_rows):
    declared = per_variable_rows(_recruitment("yes_no"), _inputs(**_declare("Yes")))
    numeric = per_variable_rows(_recruitment("numeric"), _inputs())
    for attr in ATTRS:
        assert declared[attr]["gap"] == pytest.approx(numeric[attr]["gap"]), attr


def test_a_declared_value_the_column_does_not_hold_is_refused(per_variable_rows):
    out = per_variable_rows(_recruitment("invited"), _inputs(**_declare("Hired")))
    env = out["envelope"]
    assert env["refused"] is True and "'Hired'" in env["error"]


def test_an_unfavourable_polarity_still_lands_the_favourable_value_on_one(per_variable_rows):
    """Under "1 = unfavourable" the declared favourable value is coded 0, so the
    engine's own polarity flip makes it the favourable outcome again: the reading
    is identical to the favourable-polarity run."""
    fav = per_variable_rows(_recruitment("invited"), _inputs(**_declare("Invited")))
    unfav = per_variable_rows(
        _recruitment("invited"),
        _inputs(outcome_polarity="positive_unfavorable", **_declare("Invited")),
    )
    assert unfav["gender"]["fourFifthsRatio"] == pytest.approx(fav["gender"]["fourFifthsRatio"])


def test_a_numeric_decision_needs_no_declaration(per_variable_rows):
    """CONTROL (passes on main too)."""
    rows = per_variable_rows(_recruitment("numeric"), _inputs())
    assert rows["gender"]["fourFifthsRatio"] == pytest.approx(TRUE_GENDER_RATIO, abs=1e-4)


# ------------------------------------------------- the coding step itself


def test_the_coding_record_states_what_was_applied():
    s = pd.Series(["Invited", "Rejected", None, "Rejected"], name="d")
    coded, record, refusal = o._code_decision_column(
        s, "d", _declare_on("d", "Invited"), "positive_favorable"
    )
    assert refusal is None
    assert coded.tolist()[:2] == [1.0, 0.0] and np.isnan(coded.iloc[2])
    assert record["declared"] is True and record["favourableValue"] == "Invited"
    assert record["coding"] == {"Invited": 1, "Rejected": 0}
    assert record["rows"] == {"Invited": 1, "Rejected": 2} and record["missing"] == 1


def _declare_on(column, value):
    return {"decision_labels": {"favourable_values": {column: value}}}


@pytest.mark.parametrize("values", [["true", "false", "TRUE"], ["False", "True"]])
def test_self_describing_text_needs_no_declaration(values):
    s = pd.Series(values, name="d")
    coded, record, refusal = o._code_decision_column(s, "d", {}, "positive_favorable")
    assert refusal is None
    assert coded.tolist() == [1.0 if v.lower() == "true" else 0.0 for v in values]
    assert record["declared"] is False


def test_text_that_parses_as_numbers_is_read_as_numbers():
    coded, record, refusal = o._code_decision_column(
        pd.Series(["0", "1", "1"], name="d"), "d", {}, "positive_favorable"
    )
    assert refusal is None and record is None
    assert coded.tolist() == [0, 1, 1]


def test_the_direct_coercer_refuses_text_instead_of_guessing():
    with pytest.raises(ValueError, match="never guessed"):
        o._coerce_binary(pd.Series(["Invited", "Rejected"], name="d"))
    assert o._coerce_binary(pd.Series(["true", "false"])).tolist() == [1, 0]


# ---------------------------------------------------------- the validator


def _validator_ratio(df, **kw):
    result = DataBiasValidator(protected_attributes=["gender"]).validate(
        df, outcome_column="invite_decision", **kw
    )
    return result, (result.metrics.get("outcome_disparity") or {})


def test_the_validator_no_longer_reads_rejected_as_favourable():
    """THE FLIP, in the validator: main read 0.955 (a pass) from Rejected = 1."""
    df = _recruitment("invited")
    rates = _to_numeric_binary(df["invite_decision"], "Invited").groupby(df["gender"]).mean()
    assert _ratio(rates) == pytest.approx(TRUE_GENDER_RATIO, abs=1e-4)
    _result, disparity = _validator_ratio(df, outcome_favourable_value="Invited")
    numeric_rates = _recruitment("numeric").groupby("gender")["invite_decision"].mean()
    for group, rate in disparity["gender"]["outcome_rates"].items():
        assert rate == pytest.approx(numeric_rates[group]), group


def test_the_validator_names_an_undeclared_text_outcome():
    df = _recruitment("invited")
    result, disparity = _validator_ratio(df)
    issues = {i.issue_type: i for i in result.issues}
    assert "outcome_not_coded" in issues
    assert "Invited" in issues["outcome_not_coded"].message
    assert disparity["gender"]["disparity_measured"] is False
    assert result.passed is False
    assert _to_numeric_binary(df["invite_decision"]).isna().all()


def test_the_validator_refuses_a_declared_value_the_column_lacks():
    with pytest.raises(ValueError, match="not one of"):
        _validator_ratio(_recruitment("invited"), outcome_favourable_value="Hired")


# ------------------------------------------- the quarantined legacy coercer


def test_the_legacy_pipeline_coercer_refuses_instead_of_alphabetising():
    from vfairness.operations.pulse.pipeline import _coerce_binary

    with pytest.raises(ValueError, match="never coded by guessing"):
        _coerce_binary(pd.Series(["Invited", "Rejected", "Invited"]))
    arr, _ = _coerce_binary(pd.Series(["True", "False"]))
    assert arr.tolist() == [1, 0]
