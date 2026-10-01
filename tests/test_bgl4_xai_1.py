"""BGL4 audit of batch A-xai-1: the overturns, as executable evidence.

Written by the AUDITOR of the BGL3 xai-1 / xai-2 grades on 2026-09-27. Nothing
here is a fix. Each test asserts the behaviour the graded unit SHOULD have and
is marked ``xfail(strict=True)``, so the suite stays green while the defect is
open and turns RED the moment one is fixed and its marker is not removed. That
is this repository's own convention (``xfail_strict = true``, register N-22): a
lenient marker is inert in both directions and rots.

WHEN YOU FIX ONE OF THESE, DELETE ITS MARKER IN THE SAME CHANGE.

Each overturn is a SECOND unmeasurable input the graded pin never tried. The
pins themselves were re-broken and all of them do go red, so the fixes are
load-bearing for the input they cover; they simply do not cover these.

1. ``slack_adversarial_probe`` / ``multi_seed_adversarial_probe`` were graded
   PROVEN for an all-NaN model. Measured here on the SAME Slack-style
   scaffolded model, the same instance and the same manifold, changing only the
   background:

       varying background (the passing test's fixture)
           _probe_gap 0.64   single (True, 'Slack probe: ...')
                             multi  flag True  confidence 1.0  mean_gap 0.7075
       400 rows of ONE repeated row
           _probe_gap 0.0    single (False, '')
                             multi  flag False confidence 1.0  mean_gap 0.0
                             0 warnings from either

   ``sigma = background.std(axis=0) + 1e-9``, so with no spread the draw never
   leaves the manifold and the probe cannot fire on a scaffold however bad it
   is. ``False`` / ``confidence 1.0`` is the maximum-certainty clean verdict,
   and it was returned for a probe that could not have disagreed. The lime and
   anchors adapters were fixed in this same batch for exactly this input class.

2. ``LimeExplainer.explain_local`` was graded PROVEN for a background with no
   spread. ``_unperturbable_features`` tests ``float(np.ptp(finite)) == 0.0``,
   an EXACT comparison, so the identical background plus a 1e-12 jitter is not
   "no spread" any more. Measured, 50 copies of the instance plus that jitter,
   against a model whose own prediction moves 0.9999996 -> 6.3e-07 when feature
   'c' flips sign: every attribution came back exactly 0.0 / -0.0, fidelity
   0.268, ``params['unperturbable_features'] == []`` and ZERO warnings. The
   empty list is worse than silence: it is the key a machine reader is told to
   branch on, positively asserting that every feature WAS measured.

3. ``WorkerLoop.run`` was graded PROVEN for three failed polls. It reports on
   ``_polls_failed`` only, so a run that polled ZERO times (SIGTERM during
   startup, or ``stop()`` before ``run()``) takes the ``else`` branch and logs
   "XAI worker stopped cleanly" with both counters at 0: a clean stop over a
   worker that never learned whether there was work, which is the sentence the
   fix exists to prevent.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from vfairness.xai.diagnostics.adversarial import (  # noqa: E402
    multi_seed_adversarial_probe,
    slack_adversarial_probe,
)


def _scaffolded(arr: np.ndarray) -> np.ndarray:
    """Slack-style OOD scaffold: innocent on the x1 == x0 manifold, biased off it.

    Byte-identical to the fixture ``tests/test_bgl3_xai_1.py`` uses to prove the
    probe CAN detect a scaffold.
    """
    arr = np.asarray(arr, dtype=float)
    return (np.abs(arr[:, 1] - arr[:, 0]) < 0.5).astype(float)


def _manifold_background(n: int = 400) -> np.ndarray:
    rng = np.random.RandomState(7)
    a = rng.normal(0, 1, n)
    return np.column_stack([a, a])


def test_the_scaffold_is_still_detected_on_a_varying_background() -> None:
    """CONTROL, and it passes. Without it the two xfails below prove nothing:
    a probe that refused everything would satisfy them.
    """
    bg = _manifold_background()
    flag, reason = slack_adversarial_probe(predict_fn=_scaffolded, x=bg[0], background=bg)
    assert flag is True and "Slack probe" in reason
    out = multi_seed_adversarial_probe(predict_fn=_scaffolded, x=bg[0], background=bg, n_seeds=4)
    assert out.flag is True and out.confidence == 1.0


def test_slack_probe_must_not_clear_a_scaffold_it_could_not_probe() -> None:
    """OVERTURN of the BGL3 PROVEN grade on slack_adversarial_probe, CLOSED in BGL5.

    A background with no spread gives sigma 1e-9, so the draw never left the
    manifold and the probe could not fire; it returned (False, '') with no
    warning, which is the verdict "no OOD scaffolding detected" over a probe that
    could not disagree. The marker is gone because the defect is: the same input
    now returns flag None with a COULD NOT CHECK reason and one warning. The fix's
    own pins, its sabotage and the over-correction control live in
    tests/test_bgl5_xai_1.py.
    """
    bg = np.repeat(_manifold_background()[:1], 400, axis=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flag, reason = slack_adversarial_probe(predict_fn=_scaffolded, x=bg[0], background=bg)
    assert flag is not False, (
        f"returned the clean verdict {(flag, reason)!r} with {len(caught)} warning(s) for a "
        "scaffolded model the same probe flags True on a background that varies"
    )
    assert flag is None and reason.startswith("COULD NOT CHECK"), (flag, reason)
    assert caught, "the refusal was silent"


def test_multi_seed_probe_must_not_be_certain_about_a_probe_that_could_not_fire() -> None:
    """OVERTURN of the BGL3 PROVEN grade on multi_seed_adversarial_probe, CLOSED.

    The READINESS-6 fix guarded a NaN gap; a gap of exactly 0.0 from a background
    with no spread is finite, so every seed "measured" it and the result was
    flag=False with confidence 1.0, the maximum-certainty clean verdict. Now the
    background is judged before the draw and every statistic comes back NaN
    beside flag None.
    """
    bg = np.repeat(_manifold_background()[:1], 400, axis=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = multi_seed_adversarial_probe(
            predict_fn=_scaffolded, x=bg[0], background=bg, n_seeds=4
        )
    assert not (out.flag is False and out.confidence == 1.0), (
        f"flag={out.flag} confidence={out.confidence} mean_gap={out.mean_gap} with "
        f"{len(caught)} warning(s), over a scaffolded model"
    )
    assert out.flag is None and np.isnan(out.confidence) and np.isnan(out.mean_gap)
    assert out.reason.startswith("COULD NOT CHECK")
    assert caught, "the refusal was silent"


def test_lime_must_not_report_an_all_zero_explanation_as_fully_measured() -> None:
    """OVERTURN of the BGL3 PROVEN grade on LimeExplainer.explain_local, CLOSED.

    _unperturbable_features compared np.ptp with 0.0 EXACTLY, so a background with
    a 1e-12 spread was not flagged, the refusal did not fire, every attribution
    came back at 1e-17 and params['unperturbable_features'] was [], which asserts
    that every feature WAS measured. The test is kept as the auditor wrote it,
    including the two assertions that establish the model really does depend on
    'c'; only the refusal branch is now the expected one rather than the escape
    hatch.
    """
    pytest.importorskip("lime")
    from sklearn.linear_model import LogisticRegression

    from vfairness.xai.explainers.lime_adapter import LimeExplainer

    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 4))
    y = (X[:, 2] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=800).fit(X, y)
    x = X[int(np.argmax(X[:, 2]))]

    flipped = x.copy()
    flipped[2] = -flipped[2]
    assert model.predict_proba(x.reshape(1, -1))[0, 1] > 0.99
    assert model.predict_proba(flipped.reshape(1, -1))[0, 1] < 0.01, (
        "the model must really depend on 'c', or a zero attribution for it is correct"
    )

    background = np.tile(x, (50, 1)) + rng.normal(scale=1e-12, size=(50, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            explanation = LimeExplainer().explain_local(
                model,
                x,
                background,
                instance_id="i",
                subject_id="s",
                model_hash="m",
                data_hash="d",
                feature_names=["a", "b", "c", "d"],
                stability_reruns=2,
                num_samples=400,
            )
        except ValueError as exc:
            # The refusal, which is the corrected behaviour. Asserted rather than
            # returned from, so a refusal with the wrong reason still fails.
            assert "COULD NOT MEASURE" in str(exc), str(exc)
            return
    contributions = {a.feature: a.contribution for a in explanation.attributions}
    said_nothing = all(abs(v) < 1e-9 for v in contributions.values())
    claimed_complete = explanation.params["unperturbable_features"] == []
    raise AssertionError(
        "explain_local returned an Explanation for a background whose spread is "
        f"numerical noise: {contributions} with unperturbable_features="
        f"{explanation.params['unperturbable_features']}, fidelity={explanation.fidelity}, "
        f"{len(caught)} warning(s), said_nothing={said_nothing}, "
        f"claimed_complete={claimed_complete}"
    )


def test_the_worker_must_not_report_a_clean_stop_after_zero_polls(monkeypatch, caplog) -> None:
    """OVERTURN of the BGL3 PROVEN grade on WorkerLoop.run, CLOSED.

    The summary branched on _polls_failed alone, so zero polls of any kind fell
    into the else branch and logged "XAI worker stopped cleanly" with both
    counters at 0. The fourth state now has its own branch, and run() returns the
    exit status the run earned so main can hand it to a supervisor.
    """
    from vfairness.xai.worker import runner

    class _Writer:
        def __init__(self) -> None:
            self._client = None

    monkeypatch.setattr(runner, "SupabaseWriter", lambda **_k: _Writer())
    loop = runner.WorkerLoop(config=runner.WorkerConfig(poll_interval_s=0.0))
    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")
    loop.stop()  # SIGTERM arrived during startup
    status = loop.run()

    assert (loop._polls_answered, loop._polls_failed) == (0, 0)
    assert "stopped cleanly" not in caplog.text, (
        "a worker that never polled reported a clean stop: " + caplog.text
    )
    assert status == runner.EXIT_NOTHING_MEASURED, (
        "the could-not-check reached the log and not the exit status, which is the "
        "only channel a supervisor reads"
    )
