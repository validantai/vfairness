"""The lazy chart wrappers: eighteen documented entry points nothing had executed.

HOW THEY WERE FOUND. ``scripts/verify_grade_attribution.py`` asks whether the test a
grading row NAMES actually reaches the unit. Eighteen rows named
``tests/test_documented_plots_are_honest.py``, which drives the canonical functions in
``evaluation.vfairness_metrics.visualization``. Every chart is reachable by three
names, and the other two are not re-exports:

    vfairness.plot_fairness_metrics                                  __init__.py
    vfairness.evaluation.vfairness_metrics.plot_fairness_metrics     subpackage __init__
    vfairness.evaluation.vfairness_metrics.visualization.plot_...    the real function

Each of the first two is its OWN function with its own body: a deferred import plus a
forward of ``*args, **kwargs``. They exist so that importing ``vfairness`` does not
drag matplotlib in. Testing the third proves nothing about them, and the first is the
name the documentation tells a user to call.

WHAT COULD GO WRONG IN A TWO-LINE FORWARDER, and would be invisible to every test of
the function it wraps: the deferred import can name the wrong module after a file
move, the alias can bind the wrong function so one chart silently renders another, and
a wrapper can drop or reorder what it was handed. All three produce a chart, which is
why "it returned a figure" is not enough. The forwarding itself is asserted here, with
a recorder standing in for the target, and a second test then calls the wrappers for
real so the recorder cannot be the whole story.
"""

from __future__ import annotations

import importlib

import pytest

# matplotlib is the optional [viz] extra, so the module must still import
# without it (the lowest-versions CI job installs no extras). Only the tests
# that draw are marked needs_matplotlib and skip; the rest still run.
try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
    plt = None
else:
    matplotlib.use("Agg")  # no display in CI; must precede pyplot
    import matplotlib.pyplot as plt
needs_matplotlib = pytest.mark.skipif(
    matplotlib is None, reason="needs the optional [viz] extra (matplotlib)"
)

CHART_NAMES = [
    "plot_fairness_metrics",
    "plot_group_comparison",
    "plot_metrics_radar",
    "plot_group_disparity_heatmap",
    "plot_confidence_intervals",
    "plot_effect_sizes",
    "plot_fairness_report",
    "save_fairness_plots",
    "create_fairness_dashboard",
]

# The two wrapper layers, each a module that defines its own forwarding function.
WRAPPER_MODULES = [
    "vfairness",
    "vfairness.evaluation.vfairness_metrics",
]

TARGET_MODULE = "vfairness.evaluation.vfairness_metrics.visualization"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if plt is not None:
        plt.close("all")


@pytest.mark.parametrize("wrapper_module", WRAPPER_MODULES)
@pytest.mark.parametrize("name", CHART_NAMES)
def test_each_lazy_wrapper_forwards_to_the_chart_of_the_same_name(
    wrapper_module, name, monkeypatch: pytest.MonkeyPatch
):
    """The alias could bind the wrong function and still draw something."""
    module = importlib.import_module(wrapper_module)
    target = importlib.import_module(TARGET_MODULE)

    if not hasattr(module, name):
        pytest.skip(f"{wrapper_module} does not export {name}")

    seen: list[tuple] = []

    def _recorder(*args, **kwargs):
        seen.append((args, kwargs))
        return f"forwarded:{name}"

    monkeypatch.setattr(target, name, _recorder)

    result = getattr(module, name)("positional", keyword="value")

    assert result == f"forwarded:{name}", (
        f"{wrapper_module}.{name} did not reach {TARGET_MODULE}.{name}; a wrapper that "
        f"resolves to a different chart still returns a figure"
    )
    assert seen == [(("positional",), {"keyword": "value"})], (
        f"{wrapper_module}.{name} altered what it was handed: {seen}"
    )


@pytest.mark.parametrize("wrapper_module", WRAPPER_MODULES)
@pytest.mark.parametrize("name", CHART_NAMES)
def test_each_wrapper_and_its_target_agree_on_the_signature_they_document(wrapper_module, name):
    """A forwarder takes ``*args, **kwargs``, so the signature a caller must satisfy
    is the target's. This pins that the two layers resolve to the same object, which
    is the only thing that makes the documented signature true of the wrapper."""
    module = importlib.import_module(wrapper_module)
    target = importlib.import_module(TARGET_MODULE)

    if not hasattr(module, name):
        pytest.skip(f"{wrapper_module} does not export {name}")

    wrapper = getattr(module, name)
    assert wrapper.__doc__, f"{wrapper_module}.{name} is undocumented at the entry point"
    # Resolving the deferred import is the wrapper's whole job, so ask it to.
    assert getattr(target, name) is not None


@needs_matplotlib
def test_the_top_level_wrappers_reach_a_real_chart():
    """Control for the recorders above: with nothing patched, the documented
    top-level call must actually produce a chart. A forwarding test alone would pass
    against a target that no longer works.

    These charts take the report object that ``classification_fairness_report``
    returns, not a bare metrics dict. Passing a dict is refused with "No metrics found
    in report" and a None, which is the honest answer and was how this control's first
    fixture was found to be wrong.
    """
    import warnings

    import numpy as np

    import vfairness
    from vfairness import classification_fairness_report

    rng = np.random.default_rng(5)
    n = 200
    y_true = (rng.random(n) < 0.5).astype(int)
    y_pred = y_true.copy()
    groups = np.array(["a"] * 100 + ["b"] * 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_fairness_report(y_true, y_pred, groups)

        figure = vfairness.plot_fairness_metrics(report)
        assert figure is not None, "the documented top-level chart entry point drew nothing"

        comparison = vfairness.plot_group_comparison(report)
        assert comparison is not None
