"""
Explainer registry tests. The registry is the worker's dispatch seam:
routed method string -> concrete adapter. A specified-but-unbuilt method
must fail with a clear, backlog-referencing NotImplementedError, not a
bare KeyError.
"""

from __future__ import annotations

import pytest

from vfairness.xai.explainers import available_methods, get_explainer


def test_available_methods_includes_phase2_adapters() -> None:
    methods = available_methods()
    for m in ("dice", "ig", "lime", "anchors", "shap.TreeExplainer"):
        assert m in methods


@pytest.mark.parametrize("method", ["sp-shap", "shap.DeepExplainer"])
def test_unbuilt_methods_raise_not_implemented(method: str) -> None:
    with pytest.raises(NotImplementedError) as ei:
        get_explainer(method)
    # Message is actionable: names the method and points at the backlog.
    assert "not yet implemented" in str(ei.value)


def test_unknown_method_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_explainer("not-a-real-method")  # type: ignore[arg-type]
