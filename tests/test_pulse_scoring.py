"""Tests for vfairness.operations.pulse.scoring (F-07).

The scoring analysis (feature resolution, matrix building with one-hot
categoricals, validation, scored-frame assembly) lives in the library;
the consumer only injects build_predict. These tests run locally,
deterministic, no network, no model deps: build_predict is a stub.
"""

import json

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse.scoring import score_model_over_frame

# ── stubs ───────────────────────────────────────────────────────────────────


class RowMeanPredict:
    """Stub build_predict returning per-row means; records what it saw."""

    def __init__(self):
        self.calls = []
        self.model_cols = None

    def __call__(self, scoring_payload, model_cols):
        self.model_cols = list(model_cols)

        def predict(X):
            arr = np.asarray(X, dtype=float)
            self.calls.append(arr)
            return arr.mean(axis=1)

        return predict, "stub"


class PositionWeightedPredict:
    """Stub whose output depends on WHICH column is hot (weights by column
    index), so a change in a one-hot indicator changes the prediction even
    when the row sum does not."""

    def __init__(self):
        self.model_cols = None

    def __call__(self, scoring_payload, model_cols):
        self.model_cols = list(model_cols)

        def predict(X):
            arr = np.asarray(X, dtype=float)
            weights = np.arange(1, arr.shape[1] + 1, dtype=float)
            return arr @ weights

        return predict, "stub"


# ── fixtures ────────────────────────────────────────────────────────────────


def _frame():
    """Numeric + categorical + free-text + protected columns, 8 rows."""
    return pd.DataFrame(
        {
            "age": [25, 40, 31, 52, 47, 29, 36, 44],
            "income": [30.0, 55.0, 41.0, 70.0, 62.0, 33.0, 48.0, 59.0],
            "city": ["paris", "london", "paris", "london", "paris", "london", "paris", "london"],
            "notes": [
                "applicant submitted a very long free text motivation letter one",
                "applicant submitted a very long free text motivation letter two",
                "applicant submitted a very long free text motivation letter three",
                "applicant submitted a very long free text motivation letter four",
                "applicant submitted a very long free text motivation letter five",
                "applicant submitted a very long free text motivation letter six",
                "applicant submitted a very long free text motivation letter seven",
                "applicant submitted a very long free text motivation letter eight",
            ],
            "gender": ["f", "m", "f", "m", "f", "m", "f", "m"],
        }
    )


def _inputs():
    return {"protected_attributes": ["gender"]}


def _payload():
    return {"model_base64": "ZmFrZQ=="}  # truthy resolved model bytes


# ── no-op and skip paths ───────────────────────────────────────────────────


def test_noop_without_model_or_endpoint():
    df = _frame()
    stub = RowMeanPredict()
    out, cols, notes, degr = score_model_over_frame(df, _inputs(), {}, stub)
    assert out is df
    assert cols is None
    assert notes == [] and degr == []
    assert stub.calls == []


def test_named_decision_skips_rescoring_but_returns_features():
    df = _frame()
    df["decision"] = [1, 0, 1, 0, 1, 0, 1, 0]
    stub = RowMeanPredict()
    out, cols, notes, degr = score_model_over_frame(df, _inputs(), _payload(), stub)
    assert "prediction" not in out.columns  # not rescored
    assert stub.calls == []  # predict never invoked
    assert cols is not None and "age" in cols and "income" in cols
    assert any("not rescored" in n for n in notes)


# ── categorical encoding (the F-07 upgrade) ─────────────────────────────────


def test_categorical_column_is_encoded_and_used():
    df = _frame()
    stub = PositionWeightedPredict()
    out, cols, notes, degr = score_model_over_frame(df, _inputs(), _payload(), stub)

    # The categorical column participates in scoring.
    assert "city" in cols
    assert any(c.startswith("city=") for c in stub.model_cols)
    assert any("One-hot encoded categorical feature 'city'" in n for n in notes)

    # Prediction DIFFERS when only the categorical value changes.
    df2 = df.copy()
    assert df2.loc[0, "city"] == "paris"
    df2.loc[0, "city"] = "london"
    stub2 = PositionWeightedPredict()
    out2, _cols2, _n2, _d2 = score_model_over_frame(df2, _inputs(), _payload(), stub2)
    assert out.loc[0, "prediction"] != out2.loc[0, "prediction"]
    # Rows whose categorical value did not change are unaffected.
    assert out.loc[1, "prediction"] == out2.loc[1, "prediction"]


def test_one_hot_columns_are_deterministic_sorted():
    df = _frame()
    stub = RowMeanPredict()
    score_model_over_frame(df, _inputs(), _payload(), stub)
    city_cols = [c for c in stub.model_cols if c.startswith("city=")]
    assert city_cols == ["city=london", "city=paris"]
    # Full order: dataset order with the categorical expanded in place.
    assert stub.model_cols == ["age", "income", "city=london", "city=paris"]


def test_free_text_column_dropped_with_degradation():
    df = _frame()
    stub = RowMeanPredict()
    out, cols, notes, degr = score_model_over_frame(df, _inputs(), _payload(), stub)
    assert "notes" not in cols
    assert not any(c.startswith("notes") for c in stub.model_cols)
    hits = [d for d in degr if d["stage"] == "model_scoring" and "'notes'" in d["detail"]]
    assert hits, "dropping the free-text column must record a degradation"


def test_protected_attribute_never_scored():
    df = _frame()
    stub = RowMeanPredict()
    _out, cols, _n, _d = score_model_over_frame(df, _inputs(), _payload(), stub)
    assert "gender" not in cols
    assert not any(c.startswith("gender") for c in stub.model_cols)


# ── feature_names_in_ constraint forbids one-hot ────────────────────────────


def test_expected_feature_names_forbid_one_hot_and_map_case():
    df = _frame()
    stub = RowMeanPredict()
    payload = dict(_payload())
    payload["expected_feature_names"] = ["Age", "Income", "City"]
    out, cols, notes, degr = score_model_over_frame(df, _inputs(), payload, stub)
    # Numeric expected names matched case-insensitively, model-side casing
    # handed to build_predict.
    assert stub.model_cols == ["Age", "Income"]
    assert cols == ["age", "income"]
    # The non-numeric expected column was dropped WITH a degradation.
    assert any("'City'" in d["detail"] or "'city'" in d["detail"] for d in degr)
    assert "prediction" in out.columns


# ── imputation, validation, output shape ────────────────────────────────────


def test_numeric_imputation_records_degradation():
    df = _frame()
    df.loc[2, "income"] = np.nan
    stub = RowMeanPredict()
    out, _cols, _notes, degr = score_model_over_frame(df, _inputs(), _payload(), stub)
    assert np.isfinite(out["prediction"]).all()
    assert any("Imputed missing feature values" in d["detail"] for d in degr)
    assert any("'income'" in d["detail"] for d in degr)


def test_scored_frame_has_prediction_of_right_length():
    df = _frame()
    stub = RowMeanPredict()
    out, _cols, _notes, _degr = score_model_over_frame(df, _inputs(), _payload(), stub)
    assert "prediction" in out.columns
    assert len(out["prediction"]) == len(df)
    assert out["prediction"].dtype == float
    # The input frame is not mutated.
    assert "prediction" not in df.columns


def test_row_count_mismatch_raises():
    df = _frame()

    def bad_build_predict(scoring_payload, model_cols):
        return (lambda X: np.zeros(3)), "stub"

    with pytest.raises(ValueError, match="scores for"):
        score_model_over_frame(df, _inputs(), _payload(), bad_build_predict)


def test_unusable_only_features_raise_clear_error():
    df = _frame()[["notes", "gender"]].copy()
    stub = RowMeanPredict()
    with pytest.raises(ValueError, match="no usable feature columns"):
        score_model_over_frame(df, _inputs(), _payload(), stub)


def test_notes_and_degradations_are_json_safe():
    df = _frame()
    df.loc[1, "age"] = np.nan
    stub = RowMeanPredict()
    _out, cols, notes, degr = score_model_over_frame(df, _inputs(), _payload(), stub)
    assert all(isinstance(n, str) for n in notes)
    assert all(
        isinstance(d, dict) and isinstance(d["stage"], str) and isinstance(d["detail"], str)
        for d in degr
    )
    json.dumps({"featureColumns": cols, "notes": notes, "degradations": degr})


# ── the factory's own string never reaches a log line or a note ─────────────


class SecretLabelPredict(RowMeanPredict):
    """A factory that returns a credential-bearing string as its 'source'."""

    SECRET = "https://user:tok-SECRET-123@scorer.example/predict?key=sk-SECRET-456"

    def __call__(self, scoring_payload, model_cols):
        predict, _ = super().__call__(scoring_payload, model_cols)
        return predict, self.SECRET


def test_the_factory_source_string_is_never_printed(caplog):
    """CodeQL py/clear-text-logging-sensitive-data: the injected factory also
    receives the endpoint and auth token, so its returned string must not be
    logged or written into a report note verbatim. Only fixed labels are."""
    caplog.set_level("DEBUG")
    stub = SecretLabelPredict()
    _out, _cols, notes, _degr = score_model_over_frame(
        _frame(), _inputs(), {"endpoint_url": "https://scorer.example/predict"}, stub
    )
    printed = " ".join(notes) + " " + caplog.text
    assert "SECRET" not in printed, printed
    assert "Scored injected model" in " ".join(notes), notes


@pytest.mark.parametrize("source, label", [("endpoint", "endpoint"), ("upload", "uploaded")])
def test_the_consumer_labels_still_read_plainly(source, label):
    class Labelled(RowMeanPredict):
        def __call__(self, scoring_payload, model_cols):
            predict, _ = super().__call__(scoring_payload, model_cols)
            return predict, source

    _out, _cols, notes, _degr = score_model_over_frame(_frame(), _inputs(), _payload(), Labelled())
    assert f"Scored {label} model" in " ".join(notes), notes
