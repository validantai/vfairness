"""BGL5 batch A-xai-1: the eight grades an independent audit overturned, closed.

Every test here was written against a MEASURED reproduction of the auditor's
input, and every fix it pins was sabotage-checked (the sabotage and what it
printed are recorded in /tmp/claude-501/bgl/fix/fix-A-xai-1.json).

ONE MECHANISM CARRIED FOUR OF THE EIGHT: an exact comparison with zero on a
quantity that a degenerate input drives to exactly zero, so the degenerate case
passed every finiteness filter and was published as a measurement.

1. ``slack_adversarial_probe`` and ``multi_seed_adversarial_probe`` draw their
   perturbation cloud as ``rng.normal(loc=x, scale=background.std(axis=0) +
   1e-9)``. With a background that has no spread the cloud IS x, x's nearest
   neighbours are copies of x, and the gap is exactly 0.0 for any model at all.
   0.0 is FINITE, so the READINESS-6 filter counted every seed as measured.
   Measured on the same Slack-style scaffold (``|x1 - x0| < 0.5``), the same
   instance and the same manifold, changing only the background:

       400 varying rows          gap 0.640  single (True, 'Slack probe: ...')
                                            multi  flag True  confidence 1.0
                                                   mean_gap 0.7075
       400 copies of one row     gap 0.0    single (False, '')   <- BEFORE
                                            multi  flag False confidence 1.0
                                                   mean_gap 0.0, 0 warnings
       the same + a 1e-12 jitter gap 0.0    identical to the line above

   and a THRESHOLD of nan gave (False, '') / flag False confidence 1.0 over a
   MEASURED gap of 0.640, because the BGL3 fix guarded one operand of
   ``gap > threshold`` and not the other.

2. ``LimeExplainer.explain_local`` asked ``float(np.ptp(finite)) == 0.0``, so the
   same background plus a 1e-12 jitter was not "no spread" any more. Measured
   against a LogisticRegression whose own prediction moves 0.9999996 -> 6.3e-07
   when feature 'c' flips sign: every attribution came back at 1e-17, fidelity
   0.268, ZERO warnings and ``params['unperturbable_features'] == []``, the key a
   machine reader is told to branch on, positively asserting that every feature
   WAS measured. ``explain_global`` inherited it row by row.

3. ``WorkerLoop.run`` branched its summary on ``_polls_failed`` alone, so a run
   with BOTH counters at zero (SIGTERM during startup) logged "XAI worker stopped
   cleanly", the exact sentence that block exists to prevent. And ``main``
   returned None on all three outcomes, so a worker whose every pgmq read raised
   RuntimeError('relation "pgmq.q_xai_jobs" does not exist') printed "NOT ONE
   answered read ... this is NOT a clean stop" and left a process exit status of
   0, which is the only channel a supervisor reads.

4. ``TreeShapExplainer.explain_global`` and ``KernelShapExplainer.explain_global``
   were graded PROVEN on tests that do not reach them. Measured with coverage
   isolated to its own data file, with both named tests selected: TreeShap's
   global path executed 20 body lines but only under the HEALTHY test, which
   asserts row counts alone, and the name-count guard could be removed from that
   path with both named tests green; KernelShap's global path executed ZERO body
   lines and gutting it to ``return []`` left both green. Those two are SEMI to
   PROVEN by the pins below, not by a code change: the behaviour was already
   right.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from vfairness.xai.diagnostics.adversarial import (  # noqa: E402
    _probe_gap,
    multi_seed_adversarial_probe,
    slack_adversarial_probe,
)

# === 1. the two adversarial probes ==========================================


def _scaffolded(arr: np.ndarray) -> np.ndarray:
    """Slack-style OOD scaffold: innocent on the x1 == x0 manifold, biased off it.

    Byte-identical to the fixture tests/test_bgl3_xai_1.py uses to prove the probe
    CAN detect a scaffold, so the refusals below are measured against a scaffold
    the same code really does find.
    """
    arr = np.asarray(arr, dtype=float)
    return (np.abs(arr[:, 1] - arr[:, 0]) < 0.5).astype(float)


def _manifold_background(n: int = 400) -> np.ndarray:
    rng = np.random.RandomState(7)
    a = rng.normal(0, 1, n)
    return np.column_stack([a, a])


def _frozen_background() -> np.ndarray:
    return np.repeat(_manifold_background()[:1], 400, axis=0)


def _jittered_background() -> np.ndarray:
    """The frozen background, one part in 1e12 away from constant.

    This is the input the exact ``np.ptp(...) == 0.0`` test cannot see, and the
    reason both guards are relative.
    """
    frozen = _frozen_background()
    return frozen + np.random.default_rng(3).normal(scale=1e-12, size=frozen.shape)


def test_the_scaffold_is_still_found_on_a_background_that_varies() -> None:
    """OVER-CORRECTION CONTROL, with the real numbers.

    A probe that refused everything would satisfy every refusal below and destroy
    the diagnostic. Measured after the fix, unchanged from before it: gap 0.640,
    flag True, confidence 1.0, mean_gap 0.7075, and no warning.
    """
    bg = _manifold_background()
    gap = _probe_gap(predict_fn=_scaffolded, x=bg[0], background=bg, n_perturbations=200, seed=7)
    assert gap == pytest.approx(0.64, abs=5e-3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flag, reason = slack_adversarial_probe(predict_fn=_scaffolded, x=bg[0], background=bg)
        out = multi_seed_adversarial_probe(
            predict_fn=_scaffolded, x=bg[0], background=bg, n_seeds=4
        )
    assert flag is True and "differs from background by 0.640" in reason
    assert out.flag is True
    assert out.confidence == 1.0
    assert out.mean_gap == pytest.approx(0.7075, abs=5e-3)
    assert [str(w.message) for w in caught] == []


@pytest.mark.parametrize(
    ("label", "background"),
    [
        ("400 copies of one row", _frozen_background()),
        ("those copies plus a 1e-12 jitter", _jittered_background()),
    ],
)
def test_the_single_draw_probe_refuses_a_background_it_cannot_perturb(label, background) -> None:
    """OVERTURN 1, closed. BEFORE: (False, '') with 0 warnings for the scaffold
    the same call flags True on a background that varies, because sigma collapses
    to the 1e-9 floor and the gap is 0.0 by construction. AFTER: flag None, a
    reason opening COULD NOT CHECK, and one warning naming the counts.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flag, reason = slack_adversarial_probe(
            predict_fn=_scaffolded, x=background[0], background=background
        )
    assert flag is None, f"{label}: returned {(flag, reason)!r}"
    assert reason.startswith("COULD NOT CHECK")
    assert "could not have fired on any scaffold" in reason
    messages = [str(w.message) for w in caught]
    assert any("slack_adversarial_probe" in m and "could not check" in m for m in messages), (
        f"{label}: the refusal was silent, got {messages}"
    )
    # And the draw itself refuses, so no internal caller can be handed the 0.0.
    assert np.isnan(
        _probe_gap(
            predict_fn=_scaffolded,
            x=background[0],
            background=background,
            n_perturbations=200,
            seed=7,
        )
    ), f"{label}: _probe_gap still returned a number"


@pytest.mark.parametrize(
    ("label", "background"),
    [
        ("400 copies of one row", _frozen_background()),
        ("those copies plus a 1e-12 jitter", _jittered_background()),
    ],
)
def test_the_multi_seed_probe_is_not_certain_about_a_probe_that_could_not_fire(
    label, background
) -> None:
    """OVERTURN 2, closed. BEFORE: flag False, confidence 1.0, mean_gap 0.0,
    fired_fraction 0.0, reason '' and 0 warnings. flag False with confidence 1.0
    is this function's own words for maximum certainty that the explainer is
    clean. AFTER: flag None with every statistic NaN and one warning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = multi_seed_adversarial_probe(
            predict_fn=_scaffolded, x=background[0], background=background, n_seeds=4
        )
    assert out.flag is None, f"{label}: flag={out.flag} confidence={out.confidence}"
    assert not (out.flag is False and out.confidence == 1.0)
    assert np.isnan(out.confidence) and np.isnan(out.mean_gap) and np.isnan(out.fired_fraction)
    assert out.reason.startswith("COULD NOT CHECK")
    assert any("multi_seed_adversarial_probe" in str(w.message) for w in caught), f"{label}: silent"


def test_neither_probe_clears_an_explainer_against_a_threshold_it_cannot_compare() -> None:
    """OVERTURN 1 and 2, second half. ``gap > threshold`` is False for a NaN
    THRESHOLD exactly as it was for a NaN gap. Measured before, on the varying
    background where the gap is a MEASURED 0.640 and 0.7075:

        single -> (False, '')                                    0 warnings
        multi  -> flag False confidence 1.0 mean_gap 0.7075      0 warnings

    After: flag None in both, with the threshold named in the reason.
    """
    bg = _manifold_background()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flag, reason = slack_adversarial_probe(
            predict_fn=_scaffolded, x=bg[0], background=bg, threshold=float("nan")
        )
        out = multi_seed_adversarial_probe(
            predict_fn=_scaffolded, x=bg[0], background=bg, n_seeds=4, threshold=float("nan")
        )
    assert flag is None and "not a finite number" in reason
    assert out.flag is None and np.isnan(out.confidence)
    assert len([str(w.message) for w in caught]) >= 2, "a threshold nobody could use was accepted"


# === 2. LIME: an all-zero explanation that claimed to have measured everything


@pytest.fixture(scope="module")
def lime_world():
    """A LogisticRegression that really depends on 'c', the row with the largest
    c, and the 1e-12 jitter background that the exact spread test could not see.
    """
    pytest.importorskip("lime")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 4))
    y = (X[:, 2] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=800).fit(X, y)
    x = X[int(np.argmax(X[:, 2]))]
    jittered = np.tile(x, (50, 1)) + rng.normal(scale=1e-12, size=(50, 4))
    return model, X, x, jittered


_LIME_KW = dict(subject_id="s", model_hash="m", data_hash="d", feature_names=["a", "b", "c", "d"])


def _lime():
    from vfairness.xai.explainers.lime_adapter import LimeExplainer

    return LimeExplainer()


def test_the_model_really_does_depend_on_the_feature_lime_scored_at_zero(lime_world) -> None:
    """The premise of the two pins below, asserted rather than assumed: if the
    model did not move when 'c' flips, a zero attribution for 'c' would be the
    right answer and there would be no defect.
    """
    model, _X, x, _jittered = lime_world
    flipped = x.copy()
    flipped[2] = -flipped[2]
    assert model.predict_proba(x.reshape(1, -1))[0, 1] > 0.99
    assert model.predict_proba(flipped.reshape(1, -1))[0, 1] < 0.01


def test_lime_refuses_a_background_whose_spread_is_only_numerical_noise(lime_world) -> None:
    """OVERTURN 3, closed. BEFORE: attributions [('a', 1.4e-17), ('b', -5.5e-18),
    ('c', -7.4e-17), ('d', -8.7e-18)], fidelity 0.268, ZERO warnings and
    ``params['unperturbable_features'] == []``. AFTER: ValueError naming the
    relative floor.
    """
    model, _X, x, jittered = lime_world
    with pytest.raises(ValueError) as excinfo:
        _lime().explain_local(
            model, x, jittered, instance_id="i", stability_reruns=2, num_samples=400, **_LIME_KW
        )
    message = str(excinfo.value)
    assert "COULD NOT MEASURE" in message
    assert "no feature influenced this decision" in message


def test_lime_global_propagates_that_refusal_row_by_row(lime_world) -> None:
    """OVERTURN 4, closed. BEFORE: explain_global over the same jittered
    background returned 3 Explanations with every attribution at or below
    2.1e-05, ``unperturbable_features == []`` on all three, fidelity 0.085 and
    ZERO warnings, for a model that depends on 'c' alone.
    """
    model, X, _x, jittered = lime_world
    with pytest.raises(ValueError, match="COULD NOT MEASURE"):
        _lime().explain_global(
            model, X[:3], jittered, stability_reruns=2, num_samples=400, **_LIME_KW
        )


def test_lime_refuses_a_frozen_background_even_with_one_name_too_many(lime_world) -> None:
    """The auditor's secondary input. ``_unperturbable_features`` only examines
    min(n_columns, n_names) columns, so ``len(unperturbable) == len(names)`` was
    4 == 5 and the refusal was skipped for an EXACTLY constant background.
    Measured before: an Explanation came back with a warning only. The count now
    compares against the columns examined.
    """
    model, _X, x, _jittered = lime_world
    constant = np.tile(x, (50, 1))
    kwargs = dict(_LIME_KW, feature_names=["a", "b", "c", "d", "e"])
    with pytest.raises(ValueError, match="COULD NOT MEASURE"):
        _lime().explain_local(
            model, x, constant, instance_id="i", stability_reruns=1, num_samples=200, **kwargs
        )


def test_lime_still_measures_a_neighbourhood_it_can_sample(lime_world) -> None:
    """OVER-CORRECTION CONTROL, with the real numbers. A relative spread test
    that refused real data would be worse than the defect. Measured after the
    fix: c = 0.5667, the dominant attribution, fidelity 0.4106,
    unperturbable_features [] and no warning.
    """
    model, X, x, _jittered = lime_world
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = _lime().explain_local(
            model, x, X, instance_id="i", stability_reruns=2, num_samples=400, **_LIME_KW
        )
    contributions = {a.feature: a.contribution for a in explanation.attributions}
    assert contributions["c"] == pytest.approx(0.5667, abs=0.02)
    assert max(contributions, key=lambda k: abs(contributions[k])) == "c"
    assert explanation.fidelity == pytest.approx(0.4106, abs=0.05)
    assert explanation.params["unperturbable_features"] == []
    assert [str(w.message) for w in caught] == []

    out = _lime().explain_global(model, X[:3], X, stability_reruns=2, num_samples=200, **_LIME_KW)
    assert len(out) == 3


def test_a_feature_whose_own_scale_is_1e_12_is_still_perturbable(lime_world) -> None:
    """THE OTHER OVER-CORRECTION, which an ABSOLUTE tolerance would have caused.
    The whole dataset scaled by 1e-12 has a per-column spread of about 5e-12,
    numerically identical to the jitter above, and it is real variation: relative
    to its own magnitude it is 2.4, not 2.4e-12. Measured: unperturbable_features
    [] and no warning, so the guard is scale free.
    """
    model, X, x, _jittered = lime_world
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = _lime().explain_local(
            model,
            x * 1e-12,
            X * 1e-12,
            instance_id="i",
            stability_reruns=2,
            num_samples=200,
            **_LIME_KW,
        )
    assert explanation.params["unperturbable_features"] == []
    assert [str(w.message) for w in caught] == []


# === 3. the SHAP global paths no named test reached =========================


def _forest(n_features: int = 3):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, n_features))
    y = (X[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(int)
    return RandomForestClassifier(n_estimators=8, random_state=0).fit(X, y), X


_SHAP_KW = dict(subject_id="s", model_hash="m", data_hash="d")


def test_tree_shap_global_refuses_a_feature_name_count_mismatch() -> None:
    """OVERTURN 5, closed by a pin. The behaviour was already right; no named
    test reached it. Measured with coverage isolated to its own data file: the
    test the grade named (test_shap_adapters_refuse_a_feature_name_count_mismatch)
    executed 0 of this method's body lines, and replacing its guarded
    ``_build_attributions(row, names)`` with an unguarded per-row pairing left
    BOTH named tests green.
    """
    pytest.importorskip("shap")
    from vfairness.xai.explainers.shap_adapter import TreeShapExplainer

    model, X = _forest()
    with pytest.raises(ValueError, match="2 name.* for 3 attributed value"):
        TreeShapExplainer().explain_global(model, X[:2], feature_names=["a", "b"], **_SHAP_KW)


def test_tree_shap_global_labels_every_row_with_the_names_it_was_given() -> None:
    """OVERTURN 5, second half. Discarding the caller's feature_names on this path
    (``names = feature_names or [...]`` -> always synthesise f0..fn) also left both
    named tests green, because they assert row counts alone. Measured values:
    row-0 (a 0.45858, b 0.05048, c 0.07844) over base 0.41250, prediction 1.0,
    which is rf.predict_proba(x)[0, 1]; row-1 (a -0.37137, b -0.06116, c 0.02003),
    prediction 0.0.
    """
    pytest.importorskip("shap")
    from vfairness.xai.explainers.shap_adapter import TreeShapExplainer

    model, X = _forest()
    rows = TreeShapExplainer().explain_global(
        model, X[:2], feature_names=["a", "b", "c"], **_SHAP_KW
    )
    assert [r.instance_id for r in rows] == ["s#row-0", "s#row-1"]
    assert [[a.feature for a in r.attributions] for r in rows] == [["a", "b", "c"]] * 2
    contributions = [round(a.contribution, 5) for a in rows[0].attributions]
    assert contributions == [0.45858, 0.05048, 0.07844]
    assert rows[0].base_value == pytest.approx(0.4125)
    assert [r.prediction for r in rows] == pytest.approx(list(model.predict_proba(X[:2])[:, 1]))


def test_kernel_shap_global_explains_every_row_with_the_names_it_was_given() -> None:
    """OVERTURN 6, closed by a pin. NO test in the suite called this method:
    coverage reported bodyexec=0 for both named tests and
    ``grep -rn 'explain_global' tests/ | grep -i kernel`` returned nothing, so
    gutting it to ``return []`` left both named tests green. Measured values:
    row-0 (a 0.49271, b 0.02396, c 0.08333) over base_value 0.4, prediction 1.0.
    """
    pytest.importorskip("shap")
    from vfairness.xai.explainers.shap_adapter import KernelShapExplainer

    model, X = _forest()
    rows = KernelShapExplainer().explain_global(
        model.predict_proba, X[:2], X[:20], feature_names=["a", "b", "c"], **_SHAP_KW
    )
    assert [r.instance_id for r in rows] == ["s#row-0", "s#row-1"]
    assert [[a.feature for a in r.attributions] for r in rows] == [["a", "b", "c"]] * 2
    assert [round(a.contribution, 5) for a in rows[0].attributions] == [0.49271, 0.02396, 0.08333]
    assert rows[0].base_value == pytest.approx(0.4)
    assert [r.prediction for r in rows] == pytest.approx(list(model.predict_proba(X[:2])[:, 1]))
    # Zero rows in is zero out: a per-row mapping, not a verdict.
    assert (
        KernelShapExplainer().explain_global(
            model.predict_proba, np.zeros((0, 3)), X[:20], feature_names=["a", "b", "c"], **_SHAP_KW
        )
        == []
    )


def test_kernel_shap_global_refuses_a_feature_name_count_mismatch() -> None:
    """OVERTURN 6, second half: the refusal reaches the global path too."""
    pytest.importorskip("shap")
    from vfairness.xai.explainers.shap_adapter import KernelShapExplainer

    model, X = _forest()
    with pytest.raises(ValueError, match="2 name.* for 3 attributed value"):
        KernelShapExplainer().explain_global(
            model.predict_proba, X[:2], X[:20], feature_names=["a", "b"], **_SHAP_KW
        )


# === 4. the worker: a clean stop, and an exit status, it had not earned =====


class _Response:
    def __init__(self, data):
        self.data = data


class _Rpc:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def execute(self):
        if self.behaviour == "raise":
            raise RuntimeError('relation "pgmq.q_xai_jobs" does not exist')
        return _Response([])


class _Client:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def rpc(self, *_a, **_k):
        return _Rpc(self.behaviour)


class _Writer:
    def __init__(self, behaviour="empty"):
        self._client = _Client(behaviour)


def _loop(monkeypatch, behaviour: str, ticks_before_stop: int = 3):
    from vfairness.xai.worker import runner

    loop = runner.WorkerLoop(config=runner.WorkerConfig(poll_interval_s=0.0))
    monkeypatch.setattr(runner, "SupabaseWriter", lambda **_k: _Writer(behaviour))
    seen = {"n": 0}

    def _fake_sleep(_seconds):
        seen["n"] += 1
        if seen["n"] >= ticks_before_stop:
            loop.stop()

    monkeypatch.setattr(runner.time, "sleep", _fake_sleep)
    return loop


def test_the_worker_does_not_report_a_clean_stop_after_zero_polls(monkeypatch, caplog) -> None:
    """OVERTURN 7, closed. The summary branched on _polls_failed alone, so a run
    with BOTH counters at zero took the else. Measured before, stop() then run():

        INFO worker stop requested
        INFO XAI worker started; queue=xai_jobs, poll=0.0s
        INFO XAI worker stopped cleanly            <- polls_answered 0, failed 0
        run() returned None, process exit status 0

    After: one ERROR naming the zero polls, and run() returns 3.
    """
    from vfairness.xai.worker import runner

    loop = _loop(monkeypatch, "empty")
    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")
    loop.stop()  # SIGTERM arrived during startup
    status = loop.run()

    assert (loop._polls_answered, loop._polls_failed) == (0, 0)
    assert "stopped cleanly" not in caplog.text, (
        "a worker that never polled reported a clean stop: " + caplog.text
    )
    assert "never asked whether there was work" in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert status == runner.EXIT_NOTHING_MEASURED


def test_a_run_whose_every_poll_failed_returns_a_non_zero_status(monkeypatch, caplog) -> None:
    """OVERTURN 8, at the loop. The log already said so; the value a process
    boundary can carry did not exist. Measured: 0 answered, 3 failed, one ERROR,
    and run() now returns 3 where it returned None.
    """
    from vfairness.xai.worker import runner

    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")
    loop = _loop(monkeypatch, "raise")
    status = loop.run()
    assert (loop._polls_answered, loop._polls_failed) == (0, 3)
    assert "NOT ONE answered read" in caplog.text
    assert status == runner.EXIT_NOTHING_MEASURED


def test_an_idle_worker_still_exits_clean(monkeypatch, caplog) -> None:
    """OVER-CORRECTION CONTROL. An idle worker is the NORMAL case: three answered
    polls, zero failures, "XAI worker stopped cleanly", no record above INFO, and
    a status of 0. A run that always reported failure would restart-loop every
    healthy deployment.
    """
    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")
    loop = _loop(monkeypatch, "empty")
    status = loop.run()
    assert (loop._polls_answered, loop._polls_failed) == (3, 0)
    assert "XAI worker stopped cleanly" in caplog.text
    assert status == 0
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_the_process_exit_status_carries_the_could_not_check(monkeypatch) -> None:
    """OVERTURN 8, closed. ``main`` returned None on all three outcomes, so
    measured with a worker whose every pgmq read raised, the log said "NOT ONE
    answered read of queue xai_jobs ... this is NOT a clean stop" and the process
    exit status was 0. ``python -m vfairness.xai.worker`` is a documented
    invocation with its own __main__.py, so that 0 went to a deployment script.
    """
    from vfairness.xai.worker import runner

    class _Recorder:
        def __init__(self, status):
            self.status = status

        def run(self):
            return self.status

    monkeypatch.setattr(runner, "WorkerLoop", lambda *a, **k: _Recorder(3))
    with pytest.raises(SystemExit) as excinfo:
        runner.main()
    assert excinfo.value.code == runner.EXIT_NOTHING_MEASURED


@pytest.mark.parametrize("status", [0, None])
def test_a_measured_run_still_exits_zero(monkeypatch, status) -> None:
    """OVER-CORRECTION CONTROL for the entry point, both shapes. 0 is a real
    clean run; None is what the recorder the rest of the suite substitutes for
    WorkerLoop returns, and raising SystemExit(0) would have failed that test
    while looking like a hardening.
    """
    from vfairness.xai.worker import runner

    class _Recorder:
        def run(self):
            return status

    monkeypatch.setattr(runner, "WorkerLoop", lambda *a, **k: _Recorder())
    runner.main()  # must not raise
