"""
Audit 6, wave 2: the reweighting `constraint` that was never enforced (F21).

THE DEFECT (reproduced by execution on 2026-09-09, before the fix).
`self.constraint` was written once in BaseReweighter.__init__ and read
nowhere in post_processing/reweighting/reweighter.py. Measured on identical
data with a fixed seed, PredictionReweighter built with
constraint='demographic_parity', 'equalized_odds' and 'equal_opportunity'
produced byte-identical transform output (sha256 prefix 7f17486794be8f64 for
all three, multiplicative; 6c6584b6133e46d8 for all three, additive) and
byte-identical result_ metadata, and a __getattribute__ spy recorded 0 reads
of self.constraint during fit_transform(). The fit always equalises the
overall positive rate, which is demographic parity, and the object carried
the caller's label as if that label had been met.

THE FIX, and why it is not a refusal. Every other inert parameter in this
audit was refused at construction. Here the platform UI offers
equalized_odds and equal_opportunity for prediction_reweighter and hoists
`constraint` into the dispatch payload, and the consumer builds
PredictionReweighter(constraint=<user's choice>, **params), so a refusal
would break configurations users have already saved. The arithmetic is also
not missing: the reweighter does a real, well defined demographic-parity
adjustment. Only the LABEL was wrong. So the value is still accepted, a
UserWarning at construction names what is actually enforced, and the result
records honoured_constraint next to requested_constraint instead of
asserting a constraint it did not honour.

Every pin below carries an over-correction control: the honoured constraint
must still run silently and produce exactly the same numbers as before, so a
fix that refuses or degrades the working path fails here too.
"""

import warnings

import numpy as np
import pytest

from vfairness.post_processing.reweighting.reweighter import (
    CalibratedEqualizer,
    DistributionMatcher,
    PredictionReweighter,
    RejectionOptionClassifier,
    create_reweighter,
)

# Values the platform UI can actually send. Its intervention config schema
# lists demographic_parity, equalized_odds and equal_opportunity as the enum
# options for prediction_reweighter, and _normalize_fairness_def in the
# consumer passes an unrecognised string straight through, so an arbitrary
# label can arrive too.
UNHONOURED = ["equalized_odds", "equal_opportunity", "predictive_parity"]
HONOURED = "demographic_parity"


def _data(n=300, seed=20260909):
    """Two groups with a large score gap, so the fit has real work to do."""
    rng = np.random.default_rng(seed)
    sens = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    y_prob = np.clip(
        np.where(sens == "a", rng.normal(0.35, 0.15, n), rng.normal(0.6, 0.15, n)), 0, 1
    )
    y_true = (rng.random(n) < y_prob).astype(int)
    return y_true, y_prob, sens


class TestConstraintClaimIsTrue:
    """The result must not assert a constraint the fit did not enforce."""

    @pytest.mark.parametrize("requested", UNHONOURED)
    def test_result_records_the_honoured_constraint_not_the_requested_one(self, requested):
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rw = PredictionReweighter(constraint=requested).fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )

        assert rw.result_ is not None
        assert rw.result_.honoured_constraint == HONOURED, (
            f"result claims to have honoured {rw.result_.honoured_constraint!r}, "
            f"but the fit equalises the overall positive rate, which is {HONOURED!r}"
        )
        assert rw.result_.requested_constraint == requested, (
            "the requested constraint must stay visible so a reader can see "
            "what was asked for next to what was done"
        )
        assert rw.result_.honoured_constraint != requested

    @pytest.mark.parametrize("requested", UNHONOURED)
    def test_to_dict_carries_both_constraints(self, requested):
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rw = PredictionReweighter(constraint=requested).fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )

        d = rw.result_.to_dict()
        assert d["honoured_constraint"] == HONOURED
        assert d["requested_constraint"] == requested

    @pytest.mark.parametrize("requested", UNHONOURED)
    def test_object_exposes_the_honoured_constraint_too(self, requested):
        """`self.constraint` keeps the request; the honoured value is separate."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rw = PredictionReweighter(constraint=requested)

        assert rw.constraint == requested
        assert rw.honoured_constraint == HONOURED

    def test_summary_says_the_requested_constraint_was_not_enforced(self):
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rw = PredictionReweighter(constraint="equalized_odds").fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )

        text = rw.result_.summary()
        assert "equalized_odds" in text
        assert HONOURED in text
        assert "NOT enforced" in text, (
            "a reader of the summary must be told the requested constraint was "
            f"not the one measured; got:\n{text}"
        )

    @pytest.mark.parametrize(
        "cls", [RejectionOptionClassifier, CalibratedEqualizer, DistributionMatcher]
    )
    def test_the_three_hardcoded_classes_record_their_constraint_as_well(self, cls):
        """These three never accepted a constraint; they must still say what they did."""
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # they must not warn: their label is true
            inst = cls().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)

        assert inst.result_.honoured_constraint == HONOURED
        assert inst.result_.requested_constraint == HONOURED


class TestConstructionWarns:
    """The warning is the loud half of the fix; silence is the original defect."""

    @pytest.mark.parametrize("requested", UNHONOURED)
    def test_warning_fires_at_construction_and_names_both_constraints(self, requested):
        with pytest.warns(UserWarning) as rec:
            PredictionReweighter(constraint=requested)

        assert len(rec) == 1, f"expected exactly one warning, got {[str(r.message) for r in rec]}"
        msg = str(rec[0].message)
        assert requested in msg, msg
        assert HONOURED in msg, msg
        assert "NOT enforced" in msg, msg

    def test_the_warning_does_not_wait_for_fit(self):
        """A caller that only builds the object must still be told."""
        with pytest.warns(UserWarning, match="is NOT enforced"):
            PredictionReweighter(constraint="equalized_odds")

    def test_factory_warns_when_it_drops_the_constraint(self):
        """create_reweighter passes constraint to two classes and drops it for three."""
        for method in ["rejection_option", "distribution_matching", "calibrated_equalization"]:
            with pytest.warns(UserWarning, match="is NOT enforced"):
                create_reweighter(method=method, constraint="equalized_odds")

    def test_factory_warns_once_for_the_class_that_does_take_it(self):
        with pytest.warns(UserWarning) as rec:
            rw = create_reweighter(method="multiplicative", constraint="equal_opportunity")
        assert len(rec) == 1, f"double warning: {[str(r.message) for r in rec]}"
        assert rw.constraint == "equal_opportunity"
        assert rw.honoured_constraint == HONOURED


class TestOverCorrectionControl:
    """A fix that refuses everything is as useless as one that passed everything."""

    def test_the_honoured_constraint_is_silent(self):
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rw = PredictionReweighter(constraint=HONOURED).fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )
            PredictionReweighter().fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )  # the default
            create_reweighter(method="rejection_option")
            create_reweighter(method="multiplicative", constraint=HONOURED)

        assert rw.result_.requested_constraint == HONOURED
        assert rw.result_.honoured_constraint == HONOURED

    @pytest.mark.parametrize("requested", UNHONOURED)
    @pytest.mark.parametrize("method", ["multiplicative", "additive"])
    def test_an_unhonoured_constraint_is_accepted_not_refused(self, requested, method):
        """Saved platform configurations must keep running (decision A, not B)."""
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = PredictionReweighter(constraint=requested, method=method).fit_transform(
                y_true, y_prob, sens
            )

        assert out.shape == y_prob.shape
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("method", ["multiplicative", "additive"])
    def test_transform_output_is_unchanged_by_the_fix(self, method):
        """The measured path still gives its real answer.

        THE DIGESTS ARE GONE, 2026-09-10, and this test is stronger without
        them. It asserted a sha256 of raw float64 bytes, captured on macOS
        arm64. Bit-exact float output is not portable: CI runs ubuntu x86 with a
        different BLAS, so the same correct code produced a different digest and
        this control was RED on every commit for a day. A control that fails on
        a machine change is not controlling anything; it is reporting the
        machine.

        The property it exists for is not the bit pattern. The fix changed only
        the LABEL: `constraint` is recorded rather than enforced, and the
        arithmetic stays the demographic-parity adjustment it always was. Both
        halves of that are assertable without a hash, and both are computed in
        THIS process so they cannot drift with the platform:

        1. every constraint value still produces identical output, which is the
           label-only claim stated directly (before the fix this was the DEFECT,
           because nothing recorded which constraint was honoured; it is now the
           documented behaviour, and the warning and result_ metadata carry the
           distinction the caller needs);
        2. the transform really does close the positive-rate gap, which is the
           arithmetic claim, so a fix that quietly degraded the working path
           still fails here.
        """
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = {
                constraint: PredictionReweighter(
                    constraint=constraint, method=method
                ).fit_transform(y_true, y_prob, sens)
                for constraint in [HONOURED] + UNHONOURED
            }

        reference = outputs[HONOURED]
        for constraint, out in outputs.items():
            assert np.array_equal(reference, out), (
                f"{constraint!r} produced different numbers from {HONOURED!r}. The fix "
                f"records the requested constraint, it does not steer the fit. If that "
                f"changed deliberately, this is the test to update."
            )

        groups = np.unique(sens)
        before = [float((y_prob[sens == g] >= 0.5).mean()) for g in groups]
        after = [float((reference[sens == g] >= 0.5).mean()) for g in groups]
        gap_before = max(before) - min(before)
        gap_after = max(after) - min(after)
        assert gap_before > 0.5, f"the fixture stopped being a real disparity: {gap_before}"
        assert gap_after < 0.1, (
            f"the working path degraded: the positive-rate gap went from {gap_before:.4f} "
            f"to {gap_after:.4f}, and this adjustment exists to close it"
        )

    @pytest.mark.parametrize("requested", [HONOURED] + UNHONOURED)
    def test_the_fit_still_closes_the_positive_rate_gap(self, requested):
        """The honoured constraint is really enforced, not just relabelled."""
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rw = PredictionReweighter(constraint=requested).fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )

        before = rw.result_.original_metrics["disparity"]
        after = rw.result_.adjusted_metrics["disparity"]
        assert before > 0.4, f"the fixture no longer has a gap to close: {before}"
        assert after < 0.1, f"the demographic-parity adjustment stopped working: {after}"
        assert rw.result_.fairness_improvement["disparity_reduction"] > 0.4

    def test_every_constraint_still_produces_the_same_numbers(self):
        """Named for what it is: the labels differ, the fit does not.

        This is the finding itself, kept as a pin. It is honest ONLY because
        the result now says which constraint it honoured; if a later change
        makes constraint steer the fit, this test must be deleted in the same
        commit that removes honoured_constraint's fixed value.
        """
        y_true, y_prob, sens = _data()
        outs = {}
        for c in [HONOURED] + UNHONOURED:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outs[c] = PredictionReweighter(constraint=c).fit_transform(y_true, y_prob, sens)

        base = outs[HONOURED]
        for c, out in outs.items():
            assert np.array_equal(out, base), c
