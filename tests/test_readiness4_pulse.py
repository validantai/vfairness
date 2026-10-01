"""Pulse agent probe: four multi-agent verdicts, three states each.

``vfairness.operations.pulse.agent_probe`` reuses two detectors that
already answer honestly. ``EmergentBiasDetector`` returns
``is_emergent=None`` / ``is_significant=None`` plus a UserWarning when it
had nothing to compare system bias against, and
``GroupthinkDetector.analyze_convergence`` returns
``has_groupthink=None`` / ``is_significant=None`` plus a UserWarning when
the observed convergence could not be measured.

The probe collapsed all four with ``bool()``, so a verdict nobody could
reach was published as a clean "no emergent bias" / "no groupthink",
gated on, counted into a section total that reads as a census of what was
checked, and graded ("info") off a nan amplification factor. The same
calls ran through ``_quiet()``, which suppressed the producer's warning,
so the structured answer and the human readable one went silent together.

Measured on the three-group fixture below, HEAD 6daf8377 reported:

    summary "1 of 3 group comparison(s) show significant emergent
    amplification", with A/C and B/C at isEmergent=False,
    isSignificant=False, severity='info', amplificationFactor=nan

Two of those three comparisons were never made. Every test here is either
a REFUSAL PIN (a could-not-check must not read as a measured no) or an
OVER CORRECTION CONTROL (a measured verdict, True or False, must still
report exactly as it did before).
"""

import warnings

import numpy as np
import pandas as pd

from vfairness.operations.pulse import run_pulse

SEED = 7


def _pulse(df):
    return run_pulse(df, {"source_kind": "agent", "protected_attributes": ["group"]})["data"]


def _pulse_recording(df):
    """Pulse plus every Python warning that escaped the probe."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = _pulse(df)
    return data, [str(w.message) for w in caught]


# ── fixtures ────────────────────────────────────────────────────────────────


def _amplified_df(n_per_group=40, poison=False):
    """Per-episode traces with a tiny per-component gap and a 0.6 SYSTEM
    gap: planted emergent amplification. ``poison`` overflows one episode
    of each component score to +inf, which is exactly the input on which
    EmergentBiasDetector reports "could not check"."""
    rng = np.random.default_rng(SEED)
    n = 2 * n_per_group
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group, dtype=object)
    is_a = (group == "A").astype(float)
    screener = 0.50 + 0.02 * is_a + rng.normal(0.0, 0.01, n)
    writer = 0.50 + 0.01 * is_a + rng.normal(0.0, 0.01, n)
    if poison:
        screener[0] = np.inf
        writer[0] = np.inf
    return pd.DataFrame(
        {
            "group": group,
            "tool": ["respond"] * n,
            "agent_screener_score": screener,
            "agent_writer_score": writer,
            "system_score": 0.20 + 0.60 * is_a + rng.normal(0.0, 0.02, n),
        }
    )


def _flat_system_df(n_per_group=40):
    """MEASURED absence: the system gap (0.20) is no larger than the
    component gaps (0.20), so there is no amplification and the detector
    can say so."""
    rng = np.random.default_rng(SEED)
    n = 2 * n_per_group
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group, dtype=object)
    is_a = (group == "A").astype(float)
    return pd.DataFrame(
        {
            "group": group,
            "tool": ["respond"] * n,
            "agent_screener_score": 0.50 + 0.20 * is_a + rng.normal(0.0, 0.01, n),
            "agent_writer_score": 0.50 + 0.20 * is_a + rng.normal(0.0, 0.01, n),
            "system_score": 0.20 + 0.20 * is_a + rng.normal(0.0, 0.02, n),
        }
    )


def _three_group_df(per=30):
    """Three adequately sampled groups, so three pairwise comparisons.
    One group C episode overflowed, which voids both comparisons that
    involve C and leaves A/B fully measurable."""
    rng = np.random.default_rng(SEED)
    group = np.array(["A"] * per + ["B"] * per + ["C"] * per, dtype=object)
    n = group.size
    is_a = (group == "A").astype(float)
    screener = 0.50 + 0.02 * is_a + rng.normal(0.0, 0.01, n)
    writer = 0.50 + 0.01 * is_a + rng.normal(0.0, 0.01, n)
    screener[2 * per] = np.inf
    writer[2 * per] = np.inf
    return pd.DataFrame(
        {
            "group": group,
            "tool": ["respond"] * n,
            "agent_screener_score": screener,
            "agent_writer_score": writer,
            "system_score": 0.20 + 0.60 * is_a + rng.normal(0.0, 0.02, n),
        }
    )


def _opinion_df(one, two):
    n = len(one)
    return pd.DataFrame(
        {
            "group": ["A" if j % 2 == 0 else "B" for j in range(n)],
            "tool": ["respond"] * n,
            "timestamp": list(range(n)),
            "agent_one_opinion": [float(v) for v in one],
            "agent_two_opinion": [float(v) for v in two],
        }
    )


def _converging_df(n_rounds=6, per_round=10):
    """Two agents drifting from independent opinions to identical ones.
    Each round carries its OWN mean zero base direction, so any round
    paired with any other scores near zero and the rise across rounds is
    the only structure, which is what the detector's permutation test
    asks about."""
    rng = np.random.default_rng(SEED)
    one, two = [], []
    for w in range(n_rounds):
        base = rng.normal(0.0, 1.0, per_round)
        spread = 1.2 * (1.0 - w / (n_rounds - 1)) + 0.05
        one.extend(base + spread * rng.normal(0.0, 1.0, per_round))
        two.extend(base + spread * rng.normal(0.0, 1.0, per_round))
    return _opinion_df(one, two)


def _unmeasurable_round_df(where, n=60):
    """One agent opinion overflowed to +inf. In the LAST ordered window
    that voids the detector's echo chamber score and its permutation
    test; in the FIRST it voids only the Kendall tau trend, so the two
    verdicts must part company."""
    rng = np.random.default_rng(SEED)
    one = rng.normal(0.5, 0.05, n)
    two = rng.normal(0.5, 0.05, n)
    one[-1 if where == "last" else 0] = np.inf
    return _opinion_df(one, two)


def _no_convergence_df(n=60):
    """MEASURED absence: independent opinions throughout, no trend."""
    rng = np.random.default_rng(11)
    return _opinion_df(rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n))


# ── amplification: refusal pins ─────────────────────────────────────────────


def test_amplification_could_not_check_is_never_no_emergent_bias():
    """REFUSAL PIN. EmergentBiasDetector reports is_emergent=None when no
    component had a finite bias to compare the system against. HEAD
    published that as isEmergent=False, isSignificant=False, severity
    'info' and amplificationFactor nan, which is a measured looking floor
    for a comparison nobody made."""
    d = _pulse(_amplified_df(poison=True))
    amp = d["agent"]["amplification"]
    comp = amp["perComparison"][0]
    assert comp["isEmergent"] is None
    assert comp["isSignificant"] is None
    # Nothing graded off a value nobody measured.
    assert comp["severity"] is None
    assert comp["amplificationFactor"] is None
    assert comp["maxComponentBias"] is None
    assert comp["pValue"] is None
    # The reason travels with the comparison, in words.
    assert any("could not be measured" in r for r in comp["couldNotCheck"])
    # Section level: not available, and explicitly not an absence.
    assert amp["available"] is False
    assert amp["notApplicable"] is False
    assert "could-not-check, not an absence" in amp["reason"]
    assert amp["comparisonsChecked"] == 0
    assert amp["comparisonsUnchecked"] == 1
    assert [f for f in d["bias"] if f["type"] == "emergent_amplification"] == []


def test_amplification_detector_warning_reaches_the_caller():
    """REFUSAL PIN for the second channel. _quiet() suppressed the
    detector's own could-not-check sentence, so the probe went silent in
    both channels at once. The suppression towards the caller stays (no
    Python warning escapes the probe), but the sentence has to arrive."""
    d, escaped = _pulse_recording(_amplified_df(poison=True))
    notes = d["agent"]["amplification"]["detectorNotes"]
    assert any("EmergentBiasDetector" in n for n in notes)
    assert any("is_emergent=None (could not check)" in n for n in notes)
    assert not [w for w in escaped if "EmergentBiasDetector" in w]


def test_amplification_total_counts_only_comparisons_that_were_checked():
    """REFUSAL PIN on the census. Three groups give three pairwise
    comparisons; two of them cannot be measured. HEAD summarised this as
    "1 of 3 group comparison(s)", a denominator a reader takes as the
    number of comparisons that were actually run."""
    d = _pulse(_three_group_df())
    amp = d["agent"]["amplification"]
    assert len(amp["perComparison"]) == 3
    assert amp["comparisonsChecked"] == 1
    assert amp["comparisonsUnchecked"] == 2
    assert "1 of 1 checked group comparison(s)" in amp["summary"]
    assert "A further 2 comparison(s) could not be checked" in amp["summary"]
    assert "of 3 group comparison(s)" not in amp["summary"]
    by_pair = {(c["groupA"], c["groupB"]): c for c in amp["perComparison"]}
    assert by_pair[("A", "B")]["isEmergent"] is True
    assert by_pair[("A", "C")]["isEmergent"] is None
    assert by_pair[("B", "C")]["isEmergent"] is None


# ── amplification: over correction controls ─────────────────────────────────


def test_amplification_measured_emergence_still_reports_as_before():
    """OVER CORRECTION CONTROL. A real emergent amplification must report
    exactly as it did on HEAD: True, True, critical, one finding."""
    d = _pulse(_amplified_df())
    amp = d["agent"]["amplification"]
    comp = amp["perComparison"][0]
    assert amp["available"] is True
    assert comp["isEmergent"] is True
    assert comp["isSignificant"] is True
    assert comp["severity"] == "critical"
    assert comp["amplificationFactor"] > 1.5
    assert comp["systemBias"] > comp["maxComponentBias"]
    assert "couldNotCheck" not in comp
    assert amp["comparisonsChecked"] == 1
    assert amp["comparisonsUnchecked"] == 0
    assert amp["summary"].startswith("1 of 1 checked group comparison(s)")
    findings = [f for f in d["bias"] if f["type"] == "emergent_amplification"]
    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"
    assert findings[0]["statisticalTest"]["detector"] == "EmergentBiasDetector"


def test_amplification_measured_absence_stays_false_and_is_still_graded():
    """OVER CORRECTION CONTROL. A comparison that WAS made and found no
    amplification must stay a measured False with its 'info' severity.
    Turning every False into None would hide real evidence of absence."""
    d = _pulse(_flat_system_df())
    amp = d["agent"]["amplification"]
    comp = amp["perComparison"][0]
    assert amp["available"] is True
    assert comp["isEmergent"] is False
    assert comp["isSignificant"] is False
    assert comp["severity"] == "info"
    assert isinstance(comp["amplificationFactor"], float)
    assert comp["amplificationFactor"] <= 1.5
    assert "couldNotCheck" not in comp
    assert amp["comparisonsChecked"] == 1
    assert amp["comparisonsUnchecked"] == 0
    assert [f for f in d["bias"] if f["type"] == "emergent_amplification"] == []


# ── groupthink: refusal pins ────────────────────────────────────────────────


def test_groupthink_could_not_check_is_never_no_groupthink():
    """REFUSAL PIN. With the final round's convergence unmeasurable the
    detector returns has_groupthink=None, is_significant=None and a nan
    echo chamber score, and warns. HEAD published False, False and a raw
    nan score."""
    d, escaped = _pulse_recording(_unmeasurable_round_df("last"))
    gt = d["agent"]["groupthink"]
    assert gt["hasGroupthink"] is None
    assert gt["isSignificant"] is None
    assert gt["echoChamberScore"] is None
    assert gt["pValue"] is None
    assert gt["available"] is False
    assert gt["notApplicable"] is False
    assert "could not be measured" in gt["reason"]
    assert any("no groupthink verdict was reached" in r for r in gt["couldNotCheck"])
    assert any("GroupthinkDetector" in n for n in gt["detectorNotes"])
    assert not [w for w in escaped if "GroupthinkDetector" in w]
    assert [f for f in d["bias"] if f["type"] == "groupthink_convergence"] == []


def test_groupthink_unmeasurable_early_round_voids_only_the_trend_verdict():
    """REFUSAL PIN, and the finer half of it. One unmeasurable round in
    the MIDDLE of the trajectory makes Kendall's tau nan, and `nan > 0.3`
    is a silent False: the detector raises no warning at all here, so the
    probe's own contract is the only place the gap can be caught. The
    permutation test on the final round still ran, so its verdict stays a
    measured False rather than being blanket nulled."""
    d = _pulse(_unmeasurable_round_df("first"))
    gt = d["agent"]["groupthink"]
    assert gt["hasGroupthink"] is None
    assert gt["isSignificant"] is False
    assert gt["convergenceTrajectory"][0] is None
    assert all(isinstance(v, float) for v in gt["convergenceTrajectory"][1:])
    assert isinstance(gt["echoChamberScore"], float)
    assert gt["available"] is False
    assert [f for f in d["bias"] if f["type"] == "groupthink_convergence"] == []


# ── groupthink: over correction controls ────────────────────────────────────


def test_groupthink_measured_convergence_still_fires_as_before():
    """OVER CORRECTION CONTROL. A real echo chamber must report exactly
    as it did on HEAD: True, True, a critical finding carrying the
    detector's own permutation p."""
    d = _pulse(_converging_df())
    gt = d["agent"]["groupthink"]
    assert gt["available"] is True
    assert gt["hasGroupthink"] is True
    assert gt["isSignificant"] is True
    assert gt["echoChamberScore"] > 0.9
    assert gt["pValue"] < 0.05
    assert gt["rounds"] == 6
    assert "couldNotCheck" not in gt
    assert "reason" not in gt
    findings = [f for f in d["bias"] if f["type"] == "groupthink_convergence"]
    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"
    assert findings[0]["statisticalTest"]["echoChamberScore"] == gt["echoChamberScore"]


def test_groupthink_measured_absence_stays_false():
    """OVER CORRECTION CONTROL. Independent opinions throughout: the
    trend WAS measured and there is no groupthink. That False is a
    result, and it must survive as False."""
    d = _pulse(_no_convergence_df())
    gt = d["agent"]["groupthink"]
    assert gt["available"] is True
    assert gt["hasGroupthink"] is False
    assert isinstance(gt["echoChamberScore"], float)
    assert all(isinstance(v, float) for v in gt["convergenceTrajectory"])
    assert "couldNotCheck" not in gt
    assert [f for f in d["bias"] if f["type"] == "groupthink_convergence"] == []


def test_partial_exclusion_note_reaches_the_caller_on_a_measured_verdict():
    """Both at once: the verdict IS measured (over correction control) and
    the detector's caveat still has to arrive (refusal pin on the second
    channel). One system output overflowed, so EmergentBiasDetector drops
    that episode and says the analysis rests on 79 of 80 samples. HEAD
    published the confident True and swallowed the sentence that said what
    it rests on."""
    n_per = 40
    n = 2 * n_per
    group = np.array(["A"] * n_per + ["B"] * n_per, dtype=object)
    is_a = (group == "A").astype(float)
    system = 0.20 + 0.60 * is_a
    system[0] = np.inf
    df = pd.DataFrame(
        {
            "group": group,
            "tool": ["respond"] * n,
            "agent_screener_score": [0.5] * n,
            "agent_writer_score": [0.5] * n,
            "system_score": system,
        }
    )
    d, escaped = _pulse_recording(df)
    amp = d["agent"]["amplification"]
    comp = amp["perComparison"][0]
    assert comp["isEmergent"] is True
    assert comp["isSignificant"] is True
    assert "couldNotCheck" not in comp
    assert any("not finite and were excluded" in note for note in amp["detectorNotes"])
    assert any("rests on the 79 samples" in note for note in amp["detectorNotes"])
    assert not [w for w in escaped if "EmergentBiasDetector" in w]
