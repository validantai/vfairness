"""What the explainers do when there is nothing to attribute, pinned.

WHY THIS FILE EXISTS. Measured 2026-09-25, 79 of the 81 code units on the
explainability surface carried no grade of any kind, and not because anybody
judged them risky: `scripts/surface_probe.py` builds its worlds from labels,
scores, groups and a frame, and an explainer needs a fitted model, a matrix and a
background sample. Its healthy call raised, so clause 1 of `grade_from_probe.py`
correctly refused to conclude anything. The gap was in our fixtures and it was
published as if it were a gap in the library.

WHAT THE JUDGEMENT WAS, because two adapters were flagged and NEITHER is a defect.
`scripts/xai_probe.py` reported `LimeExplainer.explain_global` and
`LinearShapExplainer.explain_global` answering on worlds where it treats an
attribution as undefined. Examined:

* an all-constant matrix whose BACKGROUND is that same constant matrix. A
  deviation-based attribution of an instance identical to its own background is
  exactly zero, and zero is the correct answer, not a fabricated one. Calling it a
  defect and "fixing" it would push the adapter into refusing a case it can
  answer, which is the same failure pointing the other way.
* a zero-row matrix. `explain_global` returns one Explanation per ROW, so an empty
  list is literally what no rows means. Nothing in the library aggregates that
  list (`xai.worker.runner`'s execute step is a documented stub), so no caller can
  turn it into "no feature mattered".

So nothing is fixed here and that is the finding. What was missing is evidence:
both behaviours were unpinned, and the empty-list case is one careless `mean()`
away from becoming a real fabricated all-clear. These tests are that evidence.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from vfairness.xai.explainers.shap_adapter import LinearShapExplainer  # noqa: E402

KW = dict(subject_id="probe-subject", model_hash="0" * 64, data_hash="1" * 64)


@pytest.fixture(scope="module")
def fitted():
    """A model that depends on feature 0 and on nothing else."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(20260925)
    X = rng.normal(0, 1, (160, 4))
    y = (X[:, 0] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(max_iter=300).fit(X, y), X, y


@pytest.fixture(scope="module")
def shap_explainer():
    pytest.importorskip("shap")
    return LinearShapExplainer()


def _attributions(explanation) -> np.ndarray:
    """The contributions, in the feature order the Explanation reports them.

    `Explanation.attributions` is a list of `Attribution(feature, contribution,
    base_value)`, not a vector of floats, and reading it as one is how the first
    version of this file failed with "float() argument must be ... not
    'Attribution'". Pulling `.contribution` is the whole conversion, and it must
    keep the reported ORDER: these tests assert WHICH feature was named.
    """
    return np.asarray([a.contribution for a in explanation.attributions], dtype=float)


def _named_feature(explanation) -> str:
    values = np.abs(_attributions(explanation))
    return explanation.attributions[int(np.argmax(values))].feature


def test_control_a_real_dependence_is_attributed_to_the_right_feature(shap_explainer, fitted):
    """The measurement half. Without this, every refusal below proves nothing:
    an explainer that refuses everything would pass all of them."""
    model, X, _y = fitted
    out = shap_explainer.explain_global(model, X[:20], X[:60], **KW)
    assert len(out) == 20
    mean_abs = np.mean([np.abs(_attributions(e)) for e in out], axis=0)
    assert int(np.argmax(mean_abs)) == 0, (
        f"the model depends on feature 0 alone and the attribution named feature "
        f"{int(np.argmax(mean_abs))}: {mean_abs}"
    )
    assert mean_abs[0] > 2 * np.median(mean_abs[1:]), f"the signal did not dominate: {mean_abs}"


def test_zero_rows_yields_no_explanations_and_never_a_zero_one(shap_explainer, fitted):
    """An empty list is what no rows means. A list holding a zero-attribution
    Explanation would be an explanation of nothing, and that is the shape a
    caller reads as 'no feature mattered'."""
    model, X, _y = fitted
    out = shap_explainer.explain_global(model, X[:0], X[:60], **KW)
    assert out == [] or len(out) == 0


def test_an_instance_identical_to_its_background_attributes_exactly_zero(shap_explainer, fitted):
    """Zero here is a MEASUREMENT, and it must stay one.

    If a later change makes this refuse, a real and correct answer has been
    thrown away; if it starts returning non-zero, the attribution is invented.
    """
    model, _X, _y = fitted
    constant = np.ones((5, 4))
    out = shap_explainer.explain_global(model, constant, constant, **KW)
    assert len(out) == 5
    for e in out:
        values = _attributions(e)
        assert np.allclose(values, 0.0, atol=1e-9), (
            f"an instance identical to its own background was attributed {values}"
        )


def test_the_arms_of_the_contract_hold_for_local_too(shap_explainer, fitted):
    """explain_local on one real row still names feature 0."""
    model, X, _y = fitted
    strong = X[np.argmax(X[:, 0])]
    e = shap_explainer.explain_local(model, strong, X[:60], instance_id="one", **KW)
    values = np.abs(_attributions(e))
    assert int(np.argmax(values)) == 0, (
        f"local attribution named {_named_feature(e)} rather than the only feature "
        f"the model uses: {values}"
    )
