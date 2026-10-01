"""Beta Go-Live final pass, group d01: two capabilities still recorded DEFECT OPEN.

``create_reweighter`` and ``integrated_gradients``. Every value quoted below was
READ OFF the public entry in this checkout before the change, not recalled.

create_reweighter (REPRODUCED 2026-09-17, fixed here)
-----------------------------------------------------
The frame is 40 A + 40 B + 1 unscored C. A scores in [0.60, 0.98] and B in
[0.02, 0.40], so A's positive rate is exactly 1.00 against B's 0.00 and their
mean probabilities are 0.7795 against 0.1692: a CRITICAL, fully measured gap.
C carries one row whose score is NaN.

  BEFORE, through ``create_reweighter``:
      rejection_option        original {'disparity': nan}          reduction nan
      distribution_matching   original {'mean_disparity': nan}     reduction nan
      calibrated_equalization original {'mean_disparity': nan}     reduction nan
      multiplicative/additive group_adjustments {'A': nan, 'B': nan, 'C': nan}
  because ``_group_disparity`` returned NaN when ANY group value was non-finite
  and the default target rate was ``np.average`` over the raw column. One
  unscored row of one group discarded the whole measurement: could-not-check
  reported where a 1.00 gap had actually been measured, which is the refusal
  defect running backwards.

  AFTER: disparity 1.0 (rates) / 0.6103 (means) over ['A', 'B'], with
  ``groups_not_compared == ['C']`` on the result and a warning naming C.

  CalibratedEqualizer also published ``{'A': 'quantile_mapping', 'B':
  'quantile_mapping', 'C': 'quantile_mapping'}`` while every C row came back
  NaN from transform() and C sat in ``unfittable_groups_``.

  PredictionReweighter on a single cohort with target_rate=None published
  ``{'only': 1.0}`` (multiplicative) and ``{'only': 0.0}`` (additive) for every
  dataset tried: a structurally fixed value that reads as "this group needed no
  correction", warned about nowhere.

  ``RejectionOptionClassifier(unprivileged_group='Z').fit()`` on single-group
  data returned a fitted object instead of raising, because the assessability
  branch pre-empted the name check. Misspell the group and you got a silent
  no-op.

integrated_gradients (NOT reproduced on 2026-09-17: ALREADY CORRECT)
--------------------------------------------------------------------
Both open items were already closed in this tree by the ``COMPLETENESS_TOL``
and ``completeness_scale`` changes. Measured here at ``explain_local``:

  * 12 random ReLU nets at the adapter's own default n_steps=50 grade
    ``completeness_ok=True`` (0/12 false alarms; residual share 0.016% to
    1.322%), and so does a trained ReLU net at the default step count.
  * a model whose output moves 0.0005 with attributions summing to exactly 0.0
    grades ``completeness_ok=False`` and warns "100.0% of the movement ...
    unexplained", so the absolute floor of 1.0 no longer passes a 100 percent
    unexplained explanation for a small-output model.

Both are pinned here anyway: an unpinned correct behaviour is one refactor from
regressing, and neither of the pre-existing controls could see either case (they
use models where IG is exact or near-exact).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from vfairness import (
    CalibratedEqualizer,
    PredictionReweighter,
    RejectionOptionClassifier,
    create_reweighter,
)

FACTORY_METHODS = [
    "multiplicative",
    "additive",
    "rejection_option",
    "distribution_matching",
    "calibrated_equalization",
]


def _caught(fn):
    """Run fn() capturing every warning; return (value, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _disparity(metrics: Dict[str, Any]) -> float:
    """The disparity field, whichever of the two names this class uses."""
    for key in ("disparity", "mean_disparity"):
        if key in metrics:
            return float(metrics[key])
    raise AssertionError(f"no disparity field in {metrics!r}")


def _abc_frame(seed: int = 11) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two healthy groups carrying a real gap, plus ONE unmeasurable group.

    This is the shape neither existing control can see: both of them use two
    healthy groups, so a rule that discards the whole measurement when any one
    group is unmeasurable is invisible to them.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.60, 0.98, 40)
    b = rng.uniform(0.02, 0.40, 40)
    y_prob = np.concatenate([a, b, np.array([np.nan])])
    sens = np.array(["A"] * 40 + ["B"] * 40 + ["C"])
    y_true = np.concatenate([np.ones(40), np.zeros(41)]).astype(int)
    return y_true, y_prob, sens


def _healthy(n_per_group: int = 100, seed: int = 11) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CONTROL fixture: two groups, both well above min_group_size, real gap."""
    rng = np.random.default_rng(seed)
    sens = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = rng.integers(0, 2, 2 * n_per_group)
    y_prob = np.concatenate(
        [
            rng.uniform(0.55, 0.95, n_per_group),
            rng.uniform(0.05, 0.45, n_per_group),
        ]
    )
    return y_true, y_prob, sens


def test_the_abc_fixture_really_carries_a_critical_measured_gap():
    """The fixture is the subject of every test below, so it is proved first.

    A fixture that quietly stopped carrying a gap would make the whole class
    below pass by measuring nothing.
    """
    _, y_prob, sens = _abc_frame()
    a, b = y_prob[sens == "A"], y_prob[sens == "B"]
    assert float(np.mean(a >= 0.5)) == 1.0
    assert float(np.mean(b >= 0.5)) == 0.0
    assert float(np.mean(a) - np.mean(b)) == pytest.approx(0.6103, abs=1e-3)
    assert int(np.count_nonzero(~np.isfinite(y_prob))) == 1
    assert (sens == "C").sum() == 1


# ===========================================================================
# 1. A measured gap must survive an unmeasurable third group
# ===========================================================================


class TestPartialComparisonIsStillAMeasurement:
    @pytest.mark.parametrize("method", FACTORY_METHODS)
    def test_one_unmeasurable_group_does_not_discard_the_other_two(self, method):
        """BEFORE: every method published disparity nan and reduction nan."""
        y_true, y_prob, sens = _abc_frame()

        rw = _quiet(lambda: create_reweighter(method))
        _, messages = _caught(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
        published = rw.result_.to_dict()

        original = _disparity(published["original_metrics"])
        assert np.isfinite(original), (method, published["original_metrics"])
        assert original > 0.5, (method, original)

        # ... and the reader is told, at the published surface, that the number
        # covers A and B only. Without this the partial figure is
        # indistinguishable from a whole one.
        assert published["groups_not_compared"] == ["C"], published["groups_not_compared"]
        assert any(
            "max-min spread over ['A', 'B'] only" in msg and "['C']" in msg for msg in messages
        ), messages

    @pytest.mark.parametrize("method", FACTORY_METHODS)
    def test_the_disclosure_reaches_the_summary_a_person_reads(self, method):
        """to_dict() is machine-readable; summary() is what a human is shown."""
        y_true, y_prob, sens = _abc_frame()
        rw = _quiet(lambda: create_reweighter(method))
        _quiet(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        summary = rw.result_.summary()
        assert "Groups NOT compared:" in summary, summary
        assert "C" in summary.split("Groups NOT compared:")[1].splitlines()[1]

    @pytest.mark.parametrize("method", FACTORY_METHODS)
    def test_a_single_measurable_group_is_still_a_refusal(self, method):
        """OVER-CORRECTION GUARD on the fix itself.

        Reporting a spread over the measurable groups must NOT quietly become
        "report something whatever happens". With one group measurable there is
        no pair to subtract, and the answer stays NaN.
        """
        rng = np.random.default_rng(4)
        y_prob = np.concatenate([rng.uniform(0.05, 0.95, 40), np.full(40, np.nan)])
        sens = np.array(["A"] * 40 + ["B"] * 40)
        y_true = rng.integers(0, 2, 80)
        assert int(np.count_nonzero(np.isfinite(y_prob[sens == "B"]))) == 0

        rw = _quiet(lambda: create_reweighter(method))
        _quiet(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
        published = rw.result_.to_dict()

        assert math.isnan(_disparity(published["original_metrics"])), published
        assert math.isnan(published["fairness_improvement"]["disparity_reduction"])
        assert published["groups_not_compared"] == ["B"]

    def test_eighty_scored_rows_are_not_thrown_away_by_one_unscored_one(self):
        """BEFORE: the default target rate was np.average over the RAW column,
        so one NaN row made target_rate_ NaN and every group's factor NaN."""
        y_true, y_prob, sens = _abc_frame()

        rw = PredictionReweighter(method="multiplicative")
        _quiet(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert np.isfinite(rw.target_rate_), rw.target_rate_
        assert np.isfinite(rw.adjustments_["A"]) and np.isfinite(rw.adjustments_["B"])
        assert math.isnan(rw.adjustments_["C"]), rw.adjustments_
        assert rw.result_.unfittable_groups and "C" in rw.result_.unfittable_groups

        out = _quiet(lambda: rw.transform(y_prob, sens))
        assert np.isfinite(out[sens != "C"]).all()
        assert np.isnan(out[sens == "C"]).all()

    def test_a_partly_scored_group_keeps_the_rows_that_were_scored(self):
        """BEFORE: np.mean over a group holding ONE NaN returned NaN, so a group
        with 39 scored rows and one hole was recorded as unmeasurable."""
        rng = np.random.default_rng(6)
        a = rng.uniform(0.60, 0.98, 40)
        b = rng.uniform(0.02, 0.40, 40)
        b[0] = np.nan
        y_prob = np.concatenate([a, b])
        sens = np.array(["A"] * 40 + ["B"] * 40)
        y_true = np.concatenate([np.ones(40), np.zeros(40)]).astype(int)

        eq = CalibratedEqualizer()
        _, messages = _caught(lambda: eq.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert np.isfinite(_disparity(eq.result_.original_metrics))
        assert eq.result_.to_dict()["groups_not_compared"] == []
        assert any("excluded from its mean" in msg for msg in messages), messages

    # ---- controls -------------------------------------------------------

    @pytest.mark.parametrize("method", FACTORY_METHODS)
    def test_control_healthy_two_group_data_still_measures_and_closes_a_gap(self, method):
        """CONTROL. Nothing above may be bought with a blanket disclosure: on
        clean data the run is silent, the gap is measured, and it closes."""
        y_true, y_prob, sens = _healthy()

        rw = create_reweighter(method)
        _, messages = _caught(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
        published = rw.result_.to_dict()

        assert messages == [], (method, messages)
        assert published["groups_not_compared"] == []
        assert published["unfittable_groups"] == {}
        assert np.isfinite(_disparity(published["original_metrics"]))
        assert _disparity(published["original_metrics"]) > 0.3
        assert published["fairness_improvement"]["disparity_reduction"] > 0.0
        assert "Groups NOT compared:" not in rw.result_.summary()

    def test_control_a_genuinely_equal_pair_is_still_a_measured_zero(self):
        """CONTROL. Two groups drawn from ONE distribution really do have almost
        no gap, and that must stay a small NUMBER, never a refusal."""
        rng = np.random.default_rng(9)
        sens = np.array(["A", "B"] * 100)
        y_prob = rng.uniform(0.2, 0.8, 200)
        y_true = rng.integers(0, 2, 200)

        for method in FACTORY_METHODS:
            rw = create_reweighter(method)
            _, messages = _caught(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
            assert messages == [], (method, messages)
            original = _disparity(rw.result_.original_metrics)
            assert np.isfinite(original), (method, rw.result_.to_dict())
            assert original < 0.2, (method, original)


# ===========================================================================
# 2. CalibratedEqualizer must not claim a mapping it did not fit
# ===========================================================================


class TestCalibratedEqualizerAdjustmentsAreFitted:
    def test_an_unfittable_group_is_not_announced_as_quantile_mapped(self):
        """BEFORE: group_adjustments said 'quantile_mapping' for C while every C
        row came back NaN, because the dict was built from the groups PRESENT."""
        y_true, y_prob, sens = _abc_frame()

        eq = CalibratedEqualizer()
        _quiet(lambda: eq.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
        published = eq.result_.to_dict()

        assert published["group_adjustments"]["A"] == "quantile_mapping"
        assert published["group_adjustments"]["B"] == "quantile_mapping"
        assert published["group_adjustments"]["C"] != "quantile_mapping"
        assert "NOT FITTED" in published["group_adjustments"]["C"]

        # and the reason is published, not only held on the estimator
        assert "C" in published["unfittable_groups"]
        assert "C" in eq.unfittable_groups_
        assert "NOT FITTED" in eq.result_.summary()
        assert "Groups NOT fitted:" in eq.result_.summary()

        out = _quiet(lambda: eq.transform(y_prob, sens))
        assert np.isnan(out[sens == "C"]).all()
        assert np.isfinite(out[sens != "C"]).all()

    def test_control_every_fitted_group_is_still_announced_as_mapped(self):
        """CONTROL. The claim must still be made where it is TRUE."""
        y_true, y_prob, sens = _healthy()

        eq = CalibratedEqualizer()
        _, messages = _caught(lambda: eq.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
        published = eq.result_.to_dict()

        assert messages == []
        assert published["group_adjustments"] == {"A": "quantile_mapping", "B": "quantile_mapping"}
        assert published["unfittable_groups"] == {}
        assert "NOT FITTED" not in eq.result_.summary()


# ===========================================================================
# 3. A structurally fixed factor is not a fitted correction
# ===========================================================================


class TestSingleCohortAdjustmentIsNotAMeasurement:
    @pytest.mark.parametrize("method,neutral", [("multiplicative", 1.0), ("additive", 0.0)])
    def test_the_structural_factor_is_named_in_the_warning(self, method, neutral):
        """BEFORE: the warning named the disparity fields only, while
        group_adjustments published the neutral factor unremarked."""
        rng = np.random.default_rng(0)
        y_prob = rng.uniform(0.01, 0.99, 50)
        y_true = (y_prob > 0.5).astype(int)
        sens = np.array(["only"] * 50)

        rw = _quiet(lambda: create_reweighter(method))
        _, messages = _caught(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert rw.result_.group_adjustments["only"] == pytest.approx(neutral)
        assert any(
            "group_adjustments is not a measurement either" in msg
            and f"structurally exactly {neutral}" in msg
            for msg in messages
        ), messages

    def test_the_factor_really_is_structural_across_datasets(self):
        """The claim in that warning is checked, not asserted: six unrelated
        datasets, one published factor."""
        published = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            y_prob = rng.uniform(0.01, 0.99, 50)
            rw = PredictionReweighter(method="multiplicative")
            _quiet(
                lambda: rw.fit(
                    y_true=(y_prob > 0.5).astype(int),
                    y_prob=y_prob,
                    sensitive_attr=np.array(["only"] * 50),
                )
            )
            published.append(float(rw.result_.group_adjustments["only"]))
        assert np.unique(np.asarray(published)).tolist() == [1.0], published

    def test_control_an_explicit_target_rate_is_a_real_fitted_factor(self):
        """CONTROL. With a target the caller supplied, the factor IS measured
        from the data, so the extra clause must NOT fire."""
        rng = np.random.default_rng(0)
        y_prob = rng.uniform(0.01, 0.99, 50)
        y_true = (y_prob > 0.5).astype(int)
        sens = np.array(["only"] * 50)

        rw = _quiet(lambda: create_reweighter("multiplicative", target_rate=0.4))
        _, messages = _caught(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        factor = float(rw.result_.group_adjustments["only"])
        assert factor != pytest.approx(1.0)
        assert factor == pytest.approx(0.4 / float(np.mean(y_prob)))
        assert not any("group_adjustments is not a measurement" in msg for msg in messages)
        # the disparity refusal is still made: one group is still one group
        assert any("no between-group disparity exists" in msg for msg in messages), messages


# ===========================================================================
# 4. A named group that is not in the data is refused whatever the structure
# ===========================================================================


class TestNamedUnprivilegedGroupIsValidatedFirst:
    def test_a_missing_named_group_is_refused_on_single_group_data(self):
        """BEFORE: `if not self.assessable_` pre-empted the name check, so this
        returned a fitted object and corrected nothing, silently."""
        rng = np.random.default_rng(3)
        y_prob = rng.uniform(0.01, 0.99, 50)
        y_true = (y_prob > 0.5).astype(int)
        sens = np.array(["only"] * 50)

        with pytest.raises(ValueError, match="does not appear in sensitive_attr"):
            _quiet(
                lambda: RejectionOptionClassifier(unprivileged_group="Z").fit(
                    y_true=y_true, y_prob=y_prob, sensitive_attr=sens
                )
            )

    def test_a_named_group_nobody_scored_is_not_corrected_toward(self):
        """Present in the data, but every score in it is NaN: the gap this fit
        would correct toward was never measured, so nothing is moved."""
        rng = np.random.default_rng(5)
        y_prob = np.concatenate(
            [rng.uniform(0.55, 0.95, 40), rng.uniform(0.05, 0.45, 40), np.full(10, np.nan)]
        )
        sens = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 10)
        y_true = rng.integers(0, 2, 90)

        roc = RejectionOptionClassifier(unprivileged_group="C")
        out, messages = _caught(lambda: roc.fit_transform(y_true, y_prob, sens))

        assert roc.assessable_ is False
        assert roc.detected_unprivileged_ is None
        assert any("has no measurable positive rate" in msg for msg in messages), messages
        scored = np.isfinite(y_prob)
        assert np.array_equal(out[scored], y_prob[scored])

    def test_control_a_named_group_that_is_there_is_still_honoured(self):
        """CONTROL. The check must refuse only what is really absent."""
        y_true, y_prob, sens = _healthy()

        roc = RejectionOptionClassifier(unprivileged_group="B")
        out, messages = _caught(lambda: roc.fit_transform(y_true, y_prob, sens))

        assert messages == []
        assert roc.assessable_ is True
        assert roc.detected_unprivileged_ == "B"
        assert roc.result_.calibration_impact["n_modified"] > 0
        assert not np.array_equal(out, y_prob)


# ===========================================================================
# 5. integrated_gradients: the completeness check, both directions
# ===========================================================================

torch = pytest.importorskip("torch", reason="torch is an optional xai extra")
pytest.importorskip("captum", reason="captum is an optional xai extra")


def _explain(model, x, background=None, **kw):
    from vfairness.xai.explainers import IntegratedGradientsExplainer

    return _caught(
        lambda: IntegratedGradientsExplainer().explain_local(
            model,
            x,
            background,
            instance_id="i",
            subject_id="s",
            model_hash="h",
            data_hash="d",
            **kw,
        )
    )


def _relu_net(in_features: int, seed: int):
    import torch.nn as nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_features, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
    )


def _trained_relu_net():
    """A net that has actually LEARNED something, not a random one.

    The fixture asserts its own training loss, because "a trained ReLU net" is
    the whole subject of the control: an untrained one would make it pass for
    the wrong reason.
    """
    import torch.nn as nn

    torch.manual_seed(7)
    net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    rng = np.random.default_rng(7)
    x = rng.normal(size=(600, 6)).astype(np.float32)
    weights = rng.normal(size=6).astype(np.float32)
    y = (x @ weights + 0.3 * rng.normal(size=600)).astype(np.float32).reshape(-1, 1)
    xt, yt = torch.tensor(x), torch.tensor(y)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss = torch.tensor(float("nan"))
    for _ in range(300):
        opt.zero_grad()
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.2, f"the fixture is not a TRAINED net: MSE {float(loss)}"
    return net, x


class TestIntegratedGradientsCompleteness:
    def test_healthy_relu_nets_pass_at_the_adapters_own_defaults(self):
        """BEFORE (auditor, 1e-3 relative against a hardcoded 1.0 floor): 3 of
        12 random ReLU nets were graded completeness_ok=False with "the
        contributions are not a faithful decomposition of this prediction".

        IG approximates a path integral with a Riemann sum; on a piecewise
        linear network a residual of a fraction of a percent is quadrature
        error, not a decomposition failure. A false alarm here is the reverse
        defect: a real measurement refused.
        """
        graded_bad: List[int] = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            exp, messages = _explain(
                _relu_net(5, seed),
                rng.normal(size=5).astype(np.float32),
                rng.normal(size=(40, 5)).astype(np.float32),
            )
            if exp.params["completeness_ok"] is not True or messages:
                graded_bad.append(seed)
        assert graded_bad == [], graded_bad

    def test_a_trained_relu_net_passes_at_the_default_step_count(self):
        """CONTROL the pre-existing ones could not provide: they use a linear
        model (IG exact) or a smooth tanh, and a trained ReLU net at
        n_steps=400. This one is a trained ReLU net at the DEFAULT n_steps."""
        net, x = _trained_relu_net()
        exp, messages = _explain(net, x[0], x[:100])

        assert exp.params["n_steps"] == 50, "the point of this control is the DEFAULT"
        assert exp.params["completeness_ok"] is True, exp.params
        assert messages == [], messages

    def test_a_small_output_model_cannot_hide_behind_an_absolute_floor(self):
        """BEFORE (auditor): scale = max(|pred - base|, 1.0) turned the relative
        check into an absolute 1e-3 for any model moving less than 1.0, so a
        model moving 0.0005 with attributions summing to exactly 0.0 (100% of
        the movement unexplained) returned completeness_ok=True and no warning.
        """
        import torch.nn as nn

        class TinyStep(nn.Module):
            """Moves 0.0005 from the baseline; every input gradient is 0."""

            def forward(self, t):
                keep_graph = 0.0 * t.sum(dim=1, keepdim=True)
                step = torch.where(
                    t.sum(dim=1, keepdim=True) > 0.5,
                    torch.full((t.shape[0], 1), 0.0005),
                    torch.zeros(t.shape[0], 1),
                )
                return keep_graph + step

        exp, messages = _explain(TinyStep(), np.array([1.0, 1.0], dtype=np.float32))

        # the fixture really is the confident-value shape: finite zeros
        assert [a.contribution for a in exp.attributions] == [0.0, 0.0]
        assert exp.prediction - exp.base_value == pytest.approx(0.0005, abs=1e-6)
        assert exp.params["completeness_scale"] == pytest.approx(0.0005, abs=1e-6)
        assert exp.params["completeness_ok"] is False, exp.params
        assert any("100.0%" in msg and "unexplained" in msg for msg in messages), messages

    def test_the_denominator_is_published_so_the_verdict_can_be_recomputed(self):
        """A verdict a reader cannot recompute is a verdict they must take on
        trust. The scale was a hidden constant; it is a published field."""
        net, x = _trained_relu_net()
        exp, _ = _explain(net, x[0], x[:100])
        params = exp.params
        assert set(params) >= {"completeness_residual", "completeness_scale", "completeness_tol"}
        recomputed = bool(
            params["completeness_residual"]
            <= params["completeness_tol"] * params["completeness_scale"]
        )
        assert recomputed is params["completeness_ok"]

    def test_control_an_uncomputable_residual_is_still_none_not_a_pass(self):
        """OVER-CORRECTION GUARD. Loosening the tolerance must not have turned
        the third state into a pass: a NaN attribution still grades None."""
        import torch.nn as nn

        torch.manual_seed(1)
        linear = nn.Linear(3, 1)
        exp, messages = _explain(linear, np.array([1.0, np.nan, 1.0], dtype=np.float32))

        assert exp.params["completeness_ok"] is None
        assert math.isnan(exp.params["completeness_residual"])
        assert exp.params["unattributed_features"] == ["f1"]
        assert messages
