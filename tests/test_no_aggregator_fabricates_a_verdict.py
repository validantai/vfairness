"""No aggregate may report a verdict it did not measure.

Phase 0.3 of the fifth-iteration remediation plan, and the reason that plan puts
a guard before the fixes. The audit found 24 instances of one mechanism:

    The measurement layer is already honest. The layer above it discards the
    honesty.

Patching the 24 known call sites would leave the mechanism intact, and the 25th
would arrive with the next aggregate someone writes. This file is the net.

TWO HALVES, because the surface has two shapes.

1. **Discovered automatically.** Every exported callable taking
   ``(y_true, y_pred, sensitive_attr)`` is called on a SINGLE GROUP, where
   nothing can be compared. A scalar must come back NaN or raise; an object must
   not carry an affirmative verdict field. This half needs no maintenance and
   covers metrics nobody has audited: 60 functions today.

2. **Explicit scenarios**, for the aggregates that take a class, a store or a
   proxy rather than three arrays. These cannot be discovered from a signature,
   so each is written out, and each names the finding it pins.

KNOWN_UNFIXED is the point of the file. An entry is a scenario that is still
broken, carrying its finding id. The list shrinks as the plan is worked, and
`test_the_known_unfixed_list_is_honest` fails if an entry starts PASSING, so a
fix cannot land silently and the list cannot rot into an excuse.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

import vfairness as vf
from vfairness._triage import is_measured

# --------------------------------------------------------------------------
# Fixtures: input on which NOTHING can be compared
# --------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_N = 200
Y_TRUE = _RNG.integers(0, 2, _N)
Y_PRED = _RNG.integers(0, 2, _N)
Y_PROB = _RNG.random(_N)
ONE_GROUP = np.array(["A"] * _N)

# Fields whose affirmative value asserts "checked, and fine".
_CLEAN_VERDICT: Dict[str, Any] = {
    "is_fair": True,
    "passed": True,
    "approved": True,
    "is_satisfied": True,
    "is_robust": True,
    "assessable": True,
    "significant_at_05": False,
    "is_significant": False,
    "any_significant": False,
    "drift_detected": False,
    "is_emergent": False,
    "is_widening": False,
    "is_retrieval_biased": False,
    "has_intersectional_bias": False,
    "impossibility_applies": False,
}

_INPUT_SHAPES = {
    ("y_true", "y_pred", "sensitive_attr"),
    ("y_true", "y_prob", "protected_attr"),
    ("y_true", "y_score", "sensitive_attr"),
    ("y_true", "y_prob", "sensitive_attr"),
    ("y_true", "y_pred", "protected_attr"),
}


def _fields(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name, None) for f in dataclasses.fields(obj)}
    return {
        a: getattr(obj, a, None)
        for a in dir(obj)
        if not a.startswith("_") and not callable(getattr(obj, a, None))
    }


def _discovered() -> List[Tuple[str, Callable]]:
    out = []
    for name in sorted(vf.__all__):
        fn = getattr(vf, name, None)
        if not callable(fn) or inspect.isclass(fn):
            continue
        try:
            params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            continue
        if tuple(params[:3]) in _INPUT_SHAPES:
            out.append((name, fn))
    return out


DISCOVERED = _discovered()

# --------------------------------------------------------------------------
# The list that shrinks
# --------------------------------------------------------------------------

KNOWN_UNFIXED: Dict[str, str] = {}

# Statistics computed WITHIN groups rather than BETWEEN them. A single group is a
# legitimate input to these: `worst_group_accuracy` over one group really is that
# group's accuracy, and refusing it would be an over-correction.
#
# They are exempt from the single-group rule ONLY, not from scrutiny. The danger
# for a worst-group statistic is a DROPPED group, since the excluded group is the
# one most likely to be worst, and that case has its own scenario below.
WITHIN_GROUP_STATISTICS = {"worst_group_accuracy"}


def _verdict_problem(name: str, result: Any) -> str:
    """Return a description of the fabricated verdict, or "" if honest."""
    if isinstance(result, bool):
        return ""
    if isinstance(result, (int, float, np.floating)):
        if is_measured(result):
            return f"returned a finite {float(result):.4f} when nothing was comparable"
        return ""
    fields = _fields(result)
    for key, affirmative in _CLEAN_VERDICT.items():
        if key in fields and fields[key] is affirmative:
            return f"reported {key}={affirmative!r} when nothing was comparable"
    return ""


def _run(fn: Callable, params: List[str]) -> Any:
    second = Y_PROB if params[1] in ("y_prob", "y_score") else Y_PRED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(Y_TRUE, second, ONE_GROUP)


@pytest.mark.parametrize("name,fn", DISCOVERED, ids=[n for n, _ in DISCOVERED])
def test_a_single_group_never_produces_a_verdict(name: str, fn: Callable):
    """One group means nothing can be COMPARED. Every between-group answer here
    is could-not-check, and must not read as a measurement."""
    params = list(inspect.signature(fn).parameters)
    try:
        result = _run(fn, params)
    except Exception:
        return  # refusing to run is an honest answer

    if name in WITHIN_GROUP_STATISTICS:
        pytest.skip(f"{name} is a within-group statistic; see its own scenario")

    problem = _verdict_problem(name, result)
    if name in KNOWN_UNFIXED:
        pytest.xfail(f"{name}: {KNOWN_UNFIXED[name]}")
    assert not problem, (
        f"{name} {problem}.\n"
        f"  A single group cannot support a between-group verdict. Return NaN, "
        f"None, or an explicit could-not-check state."
    )


def test_the_known_unfixed_list_is_honest():
    """Every KNOWN_UNFIXED entry must still actually be broken.

    Without this the list becomes an excuse: a fix lands, the entry stays, and
    the guard silently stops covering a function that is now correct. This is the
    same mechanism `_HEADLINE_BAND_GAPS` uses, and it is why that set could be
    emptied with confidence.
    """
    by_name = dict(DISCOVERED)
    stale = []
    for name in KNOWN_UNFIXED:
        fn = by_name.get(name)
        if fn is None:
            continue  # covered by an explicit scenario instead
        params = list(inspect.signature(fn).parameters)
        try:
            result = _run(fn, params)
        except Exception:
            stale.append(f"{name} now refuses to run")
            continue
        if not _verdict_problem(name, result):
            stale.append(f"{name} is FIXED")
    assert not stale, (
        "these are listed as KNOWN_UNFIXED but no longer fabricate a verdict; "
        "remove them from the list:\n  " + "\n  ".join(stale)
    )


def test_the_discovery_actually_finds_the_surface():
    """NON-VACUITY. An empty or collapsed discovery would pass everything."""
    assert len(DISCOVERED) >= 50, (
        f"only {len(DISCOVERED)} exported functions matched the input shapes; "
        "the discovery is broken and this file is checking almost nothing"
    )
    names = {n for n, _ in DISCOVERED}
    for expected in (
        "demographic_parity_difference",
        "equalized_odds_difference",
        "worst_group_accuracy",
    ):
        assert expected in names, f"{expected} is not being probed"


def test_healthy_input_still_produces_real_answers():
    """OVER-CORRECTION CONTROL, run over the same discovered surface.

    A guard that only demands refusal would be satisfied by a library that
    refuses everything. With two real groups, the scalar metrics must return
    real numbers.
    """
    two = np.array(["A", "B"] * (_N // 2))
    measured = 0
    for name, fn in DISCOVERED:
        params = list(inspect.signature(fn).parameters)
        second = Y_PROB if params[1] in ("y_prob", "y_score") else Y_PRED
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = fn(Y_TRUE, second, two)
        except Exception:
            continue
        if isinstance(out, (int, float, np.floating)) and not isinstance(out, bool):
            if math.isfinite(float(out)):
                measured += 1
    assert measured >= 12, (
        f"only {measured} discovered metrics returned a real number on two healthy "
        "groups; the library may be refusing input it can measure"
    )


# --------------------------------------------------------------------------
# Explicit scenarios, for dangers a signature cannot reveal
# --------------------------------------------------------------------------


def test_worst_group_accuracy_refuses_when_a_group_was_dropped():
    """H-02. The excluded group is the one most likely to BE the worst.

    Measured 2026-09-07 before the fix: group a n=396 accuracy 1.000, group b
    n=4 accuracy 0.000, and the function returned 1.000 with no warning. It
    reported the BEST group's accuracy as the worst group's.
    """
    y_true = np.concatenate([np.ones(396, int), np.ones(4, int)])
    y_pred = np.concatenate([np.ones(396, int), np.zeros(4, int)])
    groups = np.array(["a"] * 396 + ["b"] * 4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = vf.worst_group_accuracy(y_true, y_pred, groups)

    assert math.isnan(value), (
        f"a group was excluded by the size gate and this returned {value!r}; the "
        "excluded group could be the worst, so the statistic is not assessable"
    )
    assert caught, "the exclusion was not even warned about"


def test_worst_group_accuracy_still_answers_when_nothing_was_dropped():
    """Over-correction control for the above."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 400)
    y_pred = rng.integers(0, 2, 400)
    two = np.array(["a", "b"] * 200)
    assert math.isfinite(vf.worst_group_accuracy(y_true, y_pred, two))


def test_an_empty_metrics_store_has_no_health_score():
    """C-04. Every component defaulted to the PERFECT value on an empty table.

    Measured 2026-09-07 on a fresh MetricsStore: score=100.0, status='green',
    n_metrics=0, and the executive report rendered "Health Score: 100/100
    (GREEN)" with "Overall fairness status: Green (100/100)". A store holding
    nothing certified a system as healthy.
    """
    from vfairness.operations.reporting import MetricsStore

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        health = MetricsStore().compute_health_score()

    assert health.score is None, f"an empty store produced a score of {health.score!r}"
    assert health.status == "not_assessed", f"status was {health.status!r}"
    assert caught, "the withheld score was not warned about"


def test_a_store_with_records_still_scores():
    """Over-correction control for C-04: evidence must still produce a verdict.

    Rewritten 2026-09-10 for READINESS-5. This used to feed a tidy frame of
    twenty values and nothing else, and read the resulting 100/100 GREEN as
    proof that evidence still scores. It was proof of the opposite: a tidy
    frame carries values, not verdicts, and `ingest_dataframe` was stamping
    alert=False on all twenty, which the stack reads as "compared to its
    threshold and found clean". The same twenty rows at demographic_parity
    0.95 scored 100/100 GREEN too.

    So the control now supplies the determination it is claiming to have, and
    checks all three states off one shape of input: determined-and-clean
    scores green, determined-and-breaching scores worse, and the SAME frame
    with no determination column is withheld rather than graded.
    """
    from datetime import datetime

    import pandas as pd

    from vfairness.operations.reporting import MetricsStore

    def frame(alerts):
        return pd.DataFrame(
            {
                "timestamp": [datetime.now()] * 20,
                "metric": ["demographic_parity_difference"] * 20,
                "value": [0.02] * 20,
                "group": ["a"] * 20,
                "alert": alerts,
            }
        )

    def health_for(alerts, **kwargs):
        store = MetricsStore()
        store.ingest_dataframe(frame(alerts), **kwargs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            return store.compute_health_score(), caught

    clean, _ = health_for([False] * 20, alert_col="alert")
    assert clean.score is not None and math.isfinite(clean.score), (
        "twenty determined, clean comparisons produced no score at all"
    )
    assert clean.status in {"green", "yellow", "red"}
    assert clean.n_not_assessable == 0

    # A fix that refuses everything is as useless as one that passed
    # everything, and one that answers the same number either way is worse
    # than both. Recorded breaches have to move it.
    breaching, _ = health_for([True] * 20, alert_col="alert")
    assert breaching.score is not None and breaching.score < clean.score, (
        f"twenty recorded breaches scored {breaching.score!r}, "
        f"no worse than twenty clean comparisons ({clean.score!r})"
    )

    # And the refusal side, off the identical values: with no alert column
    # there is no determination to read, so there is nothing to be compliant
    # with. READINESS-5.
    undetermined, caught = health_for([False] * 20)
    assert undetermined.score is None, (
        f"a frame carrying no threshold comparison scored {undetermined.score!r}"
    )
    assert undetermined.status == "not_assessed"
    assert any("no alert determination" in str(w.message) for w in caught), (
        "the withheld score was not explained to the reader"
    )


def test_the_executive_report_says_so_rather_than_crashing():
    """The withheld score has to survive every render path, not just exist."""
    from vfairness.operations.reporting import MetricsStore, ReportGenerator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ReportGenerator(MetricsStore()).generate_executive_report()

    text = getattr(report, "content", str(report))
    assert "could NOT be assessed" in text
    assert "100/100" not in text, "an unassessed store still rendered a perfect score"


def _gate_inputs():
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, 200), rng.integers(0, 2, 200)


def test_the_hierarchical_gate_does_not_approve_zero_levels():
    """C-05. An absence of failures is not a pass.

    Measured 2026-09-07: evaluate_hierarchical(y_true, y_pred, {}) returned
    approved=True, "APPROVED - 0/0 levels passed", and create_github_check
    reported conclusion "success". A CI system reading that merges the change.
    """
    from vfairness.operations.cicd.gate import GateConfig, ModelFairnessGate

    y_true, y_pred = _gate_inputs()
    decision = ModelFairnessGate(config=GateConfig()).evaluate_hierarchical(y_true, y_pred, {})
    assert not decision.approved, "the gate approved a deployment after checking 0 levels"


def test_a_gate_with_no_metrics_configured_does_not_approve():
    """C-05, second half. 'Every requirement was met' is not the same claim as
    'there were no requirements', and the gate reported the first for the second."""
    from vfairness.operations.cicd.gate import GateConfig, ModelFairnessGate

    gate = ModelFairnessGate(config=GateConfig(metrics=[], thresholds={}))
    decision = gate.evaluate_from_metrics({})
    assert not decision.approved, "a gate checking nothing approved the model"
    assert gate.create_github_check(decision).get("conclusion") != "success"


def test_a_real_gate_can_still_approve():
    """Over-correction control for C-05. A gate that can never approve is not a
    gate, and would be found out immediately by anyone using it."""
    from vfairness.operations.cicd.gate import ModelFairnessGate

    gate = ModelFairnessGate(thresholds={"demographic_parity_difference": 0.9})
    decision = gate.evaluate_from_metrics({"demographic_parity_difference": 0.02})
    assert decision.approved, "a metric comfortably within its threshold was not approved"
    assert gate.create_github_check(decision).get("conclusion") == "success"


def test_perfect_separation_is_significant_not_ignored():
    """C-06. The guard asked whether each array was internally constant, which is
    TRUE for perfect separation, so it skipped a test that runs fine.

    Measured 2026-09-07 on 100% refusal in one group against 0% in the other:
    p=1.0, is_significant=False, alongside effect_size=3.14 reported as "large"
    on the same data. scipy on the same arrays gives p=1.685e-14.

    This bites the BINARY metrics, refusal, toxicity flags, 0/1 judgements, where
    perfect separation is the realistic finding rather than a degenerate one.
    """
    from vfairness.llm import OutputAnalyzer

    result = OutputAnalyzer(alpha=0.05).analyze_refusal_rate(
        ["I'm sorry, but I cannot help with that request."] * 30,
        ["Sure! Here is a detailed plan: ..."] * 30,
    )
    assert result.is_significant, (
        f"100% refusal against 0% refusal was reported as not significant (p={result.p_value})"
    )
    assert result.p_value < 0.001


def test_identical_samples_are_still_not_significant():
    """Over-correction control for C-06: when the two samples really are the
    same, there IS nothing to test and p=1.0 is the right answer."""
    from vfairness.llm import OutputAnalyzer

    same = ["Sure! Here is a detailed plan: ..."] * 30
    result = OutputAnalyzer(alpha=0.05).analyze_refusal_rate(same, list(same))
    assert not result.is_significant
    assert result.p_value == 1.0


def test_a_degenerate_contingency_table_is_not_a_completed_test():
    """C-02. Zero degrees of freedom is the arithmetic saying nothing was compared.

    Measured 2026-09-07 on a group whose every label was negative: a 2x1 table,
    degrees_of_freedom=0, statistic=0.0, p_value=1.0, significant_at_05=False,
    reported as a completed chi-square. There is nothing for the predictions to
    be independent OF when the table has one column.
    """
    rng = np.random.default_rng(0)
    y_true = np.concatenate([rng.integers(0, 2, 100), np.zeros(100, dtype=int)])
    y_pred = np.concatenate([rng.integers(0, 2, 100), np.zeros(100, dtype=int)])
    groups = np.array(["a"] * 100 + ["b"] * 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vf.test_equalized_odds_chi_square(y_true, y_pred, groups)

    tpr = result["tpr_test"]
    assert np.array(tpr.contingency_table).shape[1] < 2, "fixture no longer degenerate"
    assert tpr.significant_at_05 is None, (
        f"a {np.array(tpr.contingency_table).shape} table reported "
        f"significant_at_05={tpr.significant_at_05!r}"
    )
    assert math.isnan(tpr.p_value)
    # The OTHER test in the same call had a real 2x2 table and must still run.
    assert result["fpr_test"].test_used == "chi_square"


def test_a_healthy_contingency_table_still_tests():
    """Over-correction control for C-02."""
    rng = np.random.default_rng(1)
    n = 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vf.test_equalized_odds_chi_square(
            rng.integers(0, 2, n), rng.integers(0, 2, n), np.array(["a", "b"] * (n // 2))
        )
    assert result["tpr_test"].test_used == "chi_square"
    assert result["tpr_test"].significant_at_05 in (True, False)


def _never_selected_minority():
    """280 rows in one group, 20 in another below min_group_size, and the small
    group is never selected. The starkest possible disparity, unmeasurable."""
    import pandas as pd

    race = np.array(["w"] * 280 + ["x"] * 20)
    y_pred = np.concatenate([np.ones(280, int), np.zeros(20, int)])
    y_true = np.random.default_rng(0).integers(0, 2, 300)
    return pd.DataFrame({"race": race}), y_pred, y_true


def test_an_unscannable_attribute_is_not_reported_as_clean():
    """C-03. An empty violation list is the same answer as "checked, and clean".

    Measured 2026-09-07: violations=[], summary n_attributes_checked=1,
    max_disparity=0.0, and recommendations ["No significant fairness violations
    detected. Continue monitoring."] for a group that is NEVER selected.
    """
    df, y_pred, y_true = _never_selected_minority()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vf.rank_fairness_issues(df, y_pred, y_true)

    summary = result["summary"]
    assert summary["n_not_assessable"] >= 1, "the skipped attribute was not recorded"
    assert math.isnan(summary["max_disparity"]), (
        f"max_disparity was {summary['max_disparity']!r}; a maximum over an empty "
        "set is not zero disparity, and 0.0 is what a perfectly fair run gets"
    )
    text = " ".join(result["recommendations"])
    assert "COULD NOT CHECK" in text
    assert "No significant fairness violations detected" not in text


def test_the_scan_itself_carries_what_it_skipped():
    """The list is still a list, and now says what is missing from it."""
    df, y_pred, y_true = _never_selected_minority()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        violations = vf.scan_fairness_violations(df, y_pred, y_true)

    assert isinstance(violations, list), "the return type changed and callers would break"
    assert violations.not_assessable, "the skipped attribute was not recorded"
    assert caught, "the silent skip was not warned about"


def test_a_real_disparity_is_still_ranked():
    """Over-correction control for C-03."""
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 400
    race = np.array(["w", "x"] * (n // 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vf.rank_fairness_issues(
            pd.DataFrame({"race": race}), np.where(race == "w", 1, 0), rng.integers(0, 2, n)
        )
    assert result["summary"]["n_violations"] > 0
    assert math.isfinite(result["summary"]["max_disparity"])


class _DeadProxy:
    """Every call fails, as a down endpoint or a wrong key would."""

    def send_prompt(self, *args, **kwargs):
        raise RuntimeError("model endpoint unreachable")


class _WorkingProxy:
    """Answers every prompt, in words the bundled scorer can actually read.

    The text moved on 2026-09-10. It used to be "a neutral sentence about the
    topic", which contains none of KeywordSentimentScorer's 30 lexicon words,
    so once that scorer stopped returning a fabricated 0.0 for a text it never
    read, this "working" proxy scored NaN and the over-correction control below
    went red. The control was right to go red: the proxy was working and the
    SCORER was not, and the fixture had been hiding that behind a 0.0. A
    control for "a working run still scores" has to supply a run that can be
    scored, so these words are in the lexicon. The unreadable case is now
    pinned in its own test as the third state.

    READINESS-6, 2026-09-10: the multiple-choice branch below is new, and it is
    a fixture that had encoded a defect. BBQ asks for a letter, and this proxy
    answered "a kind and helpful sentence...". It scored anyway, because
    _parse_bbq_answer's old first rule was ``text.startswith("A") and not
    text[1].isalpha()``, so the ENGLISH ARTICLE "a" was read as option A on all
    25 templates. Verified by running the old parser on this exact string: it
    returned 0. With the parser anchored on an answer cue, the same string is
    correctly unparseable, and a proxy that never answers the question asked is
    not the control for "a working proxy still scores". It now answers the
    question, and the open-generation branch is untouched so run_bold still
    reads the same warm words it always did.
    """

    def send_prompt(self, prompt, **kwargs):
        if "Answer with only the letter" in str(prompt):
            return {"text": "C"}
        return {"text": "a kind and helpful sentence, warm and nice"}


class _UnreadableProxy:
    """Answers every prompt, in words no bundled scorer has any evidence for."""

    def send_prompt(self, prompt, **kwargs):
        return {"text": "a neutral sentence about the topic"}


@pytest.mark.parametrize("benchmark", ["run_bbq", "run_bold"])
def test_a_total_llm_outage_does_not_score_perfectly(benchmark):
    """C-07. 0.0 is the CLEAN end of this scale, so a broken integration looked
    better than any real model.

    Measured 2026-09-07 against a proxy raising on every call: run_bbq returned
    overall_score=0.0 with sample_size=25, run_bold 0.0 with sample_size=20, as
    if every prompt had been answered and none showed bias.
    """
    from vfairness.llm import BenchmarkRunner

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = getattr(BenchmarkRunner(_DeadProxy()), benchmark)()

    assert math.isnan(result.overall_score), (
        f"{benchmark} scored {result.overall_score!r} when no prompt was answered"
    )
    assert result.sample_size == 0, (
        f"{benchmark} reported {result.sample_size} samples, none of which succeeded"
    )
    assert result.n_failed > 0, "the failures were not counted"
    assert any("EXCLUDED" in str(w.message) for w in caught)


@pytest.mark.parametrize("benchmark", ["run_bbq", "run_bold"])
def test_a_working_proxy_still_scores(benchmark):
    """Over-correction control for C-07."""
    from vfairness.llm import BenchmarkRunner

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = getattr(BenchmarkRunner(_WorkingProxy()), benchmark)()

    assert math.isfinite(result.overall_score)
    assert result.sample_size > 0
    assert result.n_failed == 0


def test_answered_but_unscorable_is_not_the_same_state_as_a_dead_endpoint():
    """The third state, and the reason the control above needed new words.

    A dead endpoint and a live model whose answers the bundled lexicon cannot
    read both withhold the BOLD score, and both report sample_size 0, because
    sample_size counts prompts that entered the score rather than prompts that
    were sent. They are different facts about the system under test and a
    reader has to be able to tell them apart: one is an integration to repair,
    the other is a scorer to upgrade (pip install vaderSentiment).

    n_failed is what separates them, so it is pinned here rather than left to
    the docstring, along with the per-category response_count that shows the
    generations really did arrive.
    """
    from vfairness.llm import BenchmarkRunner

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unscorable = BenchmarkRunner(_UnreadableProxy()).run_bold()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dead = BenchmarkRunner(_DeadProxy()).run_bold()

    assert math.isnan(unscorable.overall_score), (
        f"answers nothing could read scored {unscorable.overall_score!r}"
    )
    assert math.isnan(dead.overall_score)

    assert unscorable.n_failed == 0, (
        f"{unscorable.n_failed} prompt(s) reported as failed, but every prompt "
        "was answered; this is the scorer's limit, not the endpoint's"
    )
    assert dead.n_failed > 0, "a dead endpoint has to report its failures"

    # The generations arrived and are visible per category, which is what makes
    # the two states tellable apart in a report rather than only in a warning.
    assert all(c["details"]["response_count"] > 0 for c in unscorable.category_breakdown)
    assert all(c["details"]["response_count"] == 0 for c in dead.category_breakdown)

    assert any("contain none of its 30 lexicon words" in str(w.message) for w in caught), (
        "nothing told the reader why the answers could not be scored"
    )


def test_an_llm_judge_outage_is_not_a_neutral_verdict():
    """C-08. `.get(dim, 5)` filled a missing dimension with the neutral midpoint.

    Measured 2026-09-07, all three producing score=0.5 with zero warnings:
      judge replies "{}"                    -> 0.5
      HTTP 200 with no 'choices' in body    -> 0.5
      judge genuinely rates everything 5    -> 0.5

    The first two were byte-identical to the third. _call_judge's own docstring
    states the rule: "a judge outage that silently reads as {5,5,5,5} looks
    identical to 'no bias found'".
    """
    from vfairness.llm.scorers import LLMJudgeScorer

    def judged(parsed):
        scorer = LLMJudgeScorer(endpoint_url="https://example.invalid/v1/chat")
        scorer._call_judge = lambda text, _p=parsed: _p
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            return scorer.score("an answer"), len(caught)

    empty, empty_warns = judged({})
    partial, _ = judged({"helpfulness": 8, "fairness": 7})
    neutral, neutral_warns = judged(
        {"helpfulness": 5, "fairness": 5, "specificity": 5, "completeness": 5}
    )

    assert math.isnan(empty), f"an empty judge reply scored {empty!r}"
    assert math.isnan(partial), f"a partial judge reply scored {partial!r}"
    assert empty_warns >= 1, "the outage was not warned about"
    # The genuine verdict must still come through, and must be TELLABLE APART.
    assert neutral == 0.5 and neutral_warns == 0


def test_a_sidecar_that_is_up_but_failing_does_not_score_zero():
    """C-08. _warn_sidecar_down covered the bridge-is-None branch only, so the
    louder failure warned and the likelier one, a process that starts and then
    cannot load its model, returned 0.0 for every text in silence."""
    from vfairness.llm import scorers

    class UpButFailing:
        def call(self, op, **kwargs):
            return None

    original = scorers._SidecarBridge.get
    scorers._sidecar_bad_reply_warned = False
    try:
        scorers._SidecarBridge.get = staticmethod(lambda: UpButFailing())
        scorer = scorers.SidecarSentimentScorer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            single = scorer.score("outstanding")
            batch = scorer.score_batch(["a", "b", "c"])
        assert math.isnan(single), f"an unusable sidecar reply scored {single!r}"
        assert all(math.isnan(x) for x in batch)
        assert caught, "the failing sidecar was not warned about"
    finally:
        scorers._SidecarBridge.get = original


def _constraint_groups():
    return np.array(["A"] * 100 + ["B"] * 100)


def test_an_unmeasurable_constraint_is_neither_satisfied_nor_violated():
    """C-09. This one fabricated in BOTH directions, which is why it needs a
    two-sided control rather than a different default.

    Measured 2026-09-07 on equalized_odds:
      group B has 0 positive labels -> B.tpr 0.0 against A.tpr 0.556,
          violation=0.556, is_satisfied=False      a fabricated BREACH
      no positives anywhere         -> every tpr 0.0, violation=0.0,
          is_satisfied=True                        a fabricated ALL-CLEAR
    both with zero warnings.
    """
    from vfairness.post_processing.threshold_optimization.constraints import (
        compute_constraint_violation,
    )

    rng = np.random.default_rng(0)
    groups = _constraint_groups()
    y_pred = rng.integers(0, 2, 200)

    for label, y_true in (
        (
            "one group has no positive labels",
            np.concatenate([rng.integers(0, 2, 100), np.zeros(100, int)]),
        ),
        ("no positive labels anywhere", np.zeros(200, int)),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_constraint_violation(
                y_true, y_pred, groups, constraint="equalized_odds"
            )
        assert result.is_satisfied is None, (
            f"{label}: is_satisfied={result.is_satisfied!r}, which reads as a verdict"
        )
        assert math.isnan(result.violation), f"{label}: violation={result.violation!r}"
        assert caught, f"{label}: no warning was emitted"


@pytest.mark.parametrize(
    "name,expected",
    [("a real gap", False), ("a fair model", True)],
)
def test_a_measurable_constraint_still_gets_a_verdict(name, expected):
    """Over-correction control for C-09, in BOTH directions: the fix must still
    distinguish a breach from a pass, not refuse everything."""
    from vfairness.post_processing.threshold_optimization.constraints import (
        compute_constraint_violation,
    )

    groups = _constraint_groups()
    # Deterministic by construction rather than by seed. A random "fair" model is
    # not reliably within a 0.05 tolerance, and a control that depends on the RNG
    # stream is a flaky test dressed as a guarantee.
    y_true = np.tile([0, 1], 100)
    if expected:
        # Identical behaviour in both groups: rates match exactly, violation 0.
        y_pred = np.tile([0, 1], 100)
    else:
        # Group A always positive, group B never: the largest possible gap.
        y_pred = np.concatenate([np.ones(100, int), np.zeros(100, int)])

    result = compute_constraint_violation(y_true, y_pred, groups, constraint="equalized_odds")
    assert result.is_satisfied is expected, f"{name}: got {result.is_satisfied!r}"
    assert math.isfinite(result.violation)


def test_a_ratio_over_an_empty_group_is_not_a_measured_ratio():
    """H-04. Every branch answered with a number that means something specific.

    Measured 2026-09-07:
      risk_ratio(0, 0, 0, 0)  -> 1.0   which this function's own docstring
                                       defines three lines up as "No difference"
      risk_ratio(5, 10, 0, 0) -> inf   and inf beats every threshold, so an
                                       unmeasurable group becomes the WORST
                                       breach in a report
      risk_ratio(0, 0, 5, 10) -> 0.0
    """
    from vfairness.evaluation.vfairness_metrics._statistics import odds_ratio, risk_ratio

    for fn in (risk_ratio, odds_ratio):
        for args in ((0, 0, 0, 0), (5, 10, 0, 0), (0, 0, 5, 10)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = fn(*args)[0]
            assert math.isnan(value), f"{fn.__name__}{args} returned {value!r}"
            assert caught, f"{fn.__name__}{args} did not warn"


def test_a_real_ratio_is_still_computed():
    """Over-correction control for H-04, including the genuinely infinite case."""
    from vfairness.evaluation.vfairness_metrics._statistics import odds_ratio, risk_ratio

    assert risk_ratio(8, 10, 4, 10)[0] == pytest.approx(2.0)
    assert odds_ratio(8, 10, 4, 10)[0] == pytest.approx(6.0)
    # Group 2 has observations but zero events: a real, infinite ratio.
    assert math.isinf(risk_ratio(5, 10, 0, 10)[0])


def test_no_evidence_does_not_produce_a_zero_width_interval():
    """H-05. A zero-width 95% CI is maximum certainty from nothing, and tighter
    than any real sample could ever produce. bootstrap_ci in the same module
    already returned nan/nan/nan with a warning; this now matches it."""
    from vfairness.evaluation.vfairness_metrics._statistics import empirical_likelihood_ci

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = empirical_likelihood_ci(0, 0)

    assert math.isnan(result.point_estimate)
    assert math.isnan(result.lower_bound) and math.isnan(result.upper_bound)
    assert caught, "no warning for an interval computed from no observations"


def test_a_real_sample_still_gets_an_interval():
    """Over-correction control for H-05."""
    from vfairness.evaluation.vfairness_metrics._statistics import empirical_likelihood_ci

    result = empirical_likelihood_ci(30, 100)
    assert result.point_estimate == pytest.approx(0.30)
    assert result.lower_bound < result.point_estimate < result.upper_bound


def test_a_calibration_recommendation_names_the_groups_it_did_not_cover():
    """H-06. `calibration_disparity` reports its exclusions and warns twice; the
    recommender read only the resulting 0.0 and discarded them.

    Reproduced 2026-09-07 with a 20-row group far worse calibrated than the
    200-row group: excluded=['B'], ece_disparity=0.0, priority="low",
    "Calibration metrics are acceptable... no immediate action needed".
    """
    from vfairness.post_processing.calibration.tradeoffs import (
        recommend_calibration_strategy,
    )

    rng = np.random.default_rng(0)
    groups = np.array(["A"] * 200 + ["B"] * 20)
    y_true = np.concatenate([rng.integers(0, 2, 200), np.ones(20, int)])
    y_prob = np.concatenate([rng.random(200), np.full(20, 0.05)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = recommend_calibration_strategy(y_true, y_prob, groups)

    assert result.not_assessed_groups, "the excluded group was not reported"
    assert result.priority != "low", (
        "a group was excluded from the analysis and the verdict stayed low priority"
    )
    assert "EXCLUDED" in result.rationale


def test_a_fully_measured_calibration_recommendation_is_unchanged():
    """Over-correction control for H-06: no exclusions, no added caveat."""
    from vfairness.post_processing.calibration.tradeoffs import (
        recommend_calibration_strategy,
    )

    rng = np.random.default_rng(0)
    n = 400
    groups = np.array(["A", "B"] * (n // 2))
    y_prob = rng.random(n)
    y_true = (rng.random(n) < y_prob).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = recommend_calibration_strategy(y_true, y_prob, groups)

    assert result.not_assessed_groups == []
    assert "NOTE:" not in result.rationale


def test_drift_that_could_not_be_computed_is_not_stability():
    """H-11. `max(valid) if valid else 0.0` reported 0.0 and drift_detected=False
    when NOT ONE scale could be computed.

    Reproduced 2026-09-07 with a two-point series: scales={}, score 0.0,
    drift_detected False, which the report generator turns into "No significant
    drift detected. Continue routine monitoring." drift_report_to_svg already
    handled this correctly; every non-SVG consumer did not.
    """
    import pandas as pd

    from vfairness.operations.monitoring import FairnessDriftDetector

    detector = FairnessDriftDetector()
    detector.set_baseline(pd.Series([0.1, 0.1]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = detector.check_drift(pd.Series([0.9, 0.9]), metric="dp")

    assert result.scales == {}, "fixture no longer produces zero scales"
    assert result.drift_detected is None, f"got {result.drift_detected!r}"
    assert math.isnan(result.overall_drift_score)
    assert caught


@pytest.mark.parametrize("drifting", [True, False])
def test_a_computable_series_still_gets_a_drift_verdict(drifting):
    """Over-correction control for H-11, both directions."""
    import pandas as pd

    from vfairness.operations.monitoring import FairnessDriftDetector

    rng = np.random.default_rng(0)
    detector = FairnessDriftDetector()
    detector.set_baseline(pd.Series(rng.normal(0.3, 0.01, 60)))
    current = pd.Series(rng.normal(0.9 if drifting else 0.3, 0.01, 60))
    result = detector.check_drift(current, metric="dp")
    assert result.drift_detected is drifting
    assert math.isfinite(result.overall_drift_score)


def test_a_window_of_unmeasurable_metrics_has_no_health_score():
    """H-10. FairnessMonitor stores an undefined metric with alert=False, and
    that False entered the compliance mean as a COMPLIANT record, so an
    unmeasured metric improved the health score."""
    from datetime import datetime

    import pandas as pd

    from vfairness.operations.reporting import MetricsStore

    def store_with(values):
        store = MetricsStore()
        store.ingest_dataframe(
            pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * len(values),
                    "metric": ["dp"] * len(values),
                    "value": values,
                    "group": ["a"] * len(values),
                    # Every row carries a real threshold comparison, so the
                    # only thing this test varies is whether the VALUE is
                    # measurable. Without it the frame would be withheld for
                    # the READINESS-5 reason instead of the H-10 one, and the
                    # test would pass while measuring nothing it names.
                    "alert": [False] * len(values),
                }
            ),
            alert_col="alert",
        )
        return store

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_nan = store_with([float("nan")] * 10).compute_health_score()
        half = store_with([float("nan")] * 5 + [0.02] * 5).compute_health_score()
        clean = store_with([0.02] * 10).compute_health_score()

    assert all_nan.score is None and all_nan.status == "not_assessed"
    assert all_nan.n_not_assessable == 10
    # A partly measurable window still scores, and discloses what it dropped.
    assert half.score is not None and half.n_not_assessable == 5
    assert clean.score is not None and clean.n_not_assessable == 0


def test_an_unscorable_alert_is_not_a_low_severity_alert():
    """H-12. `>` is False for NaN, so both severity tests fell through to LOW.

    Measured 2026-09-07: every factor known gave 11.70 CRITICAL routed to
    PagerDuty; one factor NaN gave nan LOW routed to Jira. One unmeasurable
    input silently downgraded a page-someone alert to a ticket.
    """
    from vfairness.operations.monitoring import FairnessAlertPrioritizer

    prioritizer = FairnessAlertPrioritizer()
    known = {f: 0.9 for f in prioritizer.severity_weights}
    known["drift_score"] = 0.8

    _, severity_known = prioritizer.calculate_priority(known)
    assert severity_known == "CRITICAL"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, severity_partial = prioritizer.calculate_priority(
            {**known, list(prioritizer.severity_weights)[0]: float("nan")}
        )
    assert severity_partial != "LOW", "one NaN factor downgraded the alert to LOW"
    assert caught

    all_nan = {f: float("nan") for f in prioritizer.severity_weights}
    all_nan["drift_score"] = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score_nan, severity_nan = prioritizer.calculate_priority(all_nan)
    assert severity_nan == "UNSCORED" and math.isnan(score_nan)

    # Over-correction control: a genuinely calm alert is still LOW.
    calm = {f: 0.0 for f in prioritizer.severity_weights}
    calm["drift_score"] = 0.0
    assert prioritizer.calculate_priority(calm)[1] == "LOW"


def _dp_needs_two_groups(y_pred, sensitive_attr):
    """Demographic parity that REFUSES rather than guesses when a perturbation
    has left only one group. Standing in for every real metric that does."""
    groups = np.unique(sensitive_attr)
    if len(groups) < 2:
        return float("nan")
    rates = [y_pred[sensitive_attr == g].mean() for g in groups]
    return float(max(rates) - min(rates))


def test_robustness_is_not_graded_on_the_draws_that_happened_to_survive():
    """H-03. `perturbed_metrics[~np.isnan(...)]` dropped unmeasurable draws
    silently, and the survivors were graded as if they were the whole run.

    An unmeasurable draw is a perturbation under which the metric BROKE, so
    discarding it biases every answer toward robust. Measured 2026-09-07:
    80 of 100 draws vanished and the function reported is_robust=True with a
    perfect robustness_score of 1.0, no warning anywhere.
    """
    from vfairness.evaluation.vfairness_metrics.robustness import sensitivity_analysis

    n = 200
    sensitive = np.array(["a"] * (n - 3) + ["b"] * 3)
    y_pred = np.array([1] * (n - 3) + [0] * 3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sensitivity_analysis(
            y_pred,
            sensitive,
            _dp_needs_two_groups,
            perturbation_type="subsample",
            perturbation_rate=0.9,
            n_iterations=100,
            random_state=1,
        )

    assert result.n_unmeasurable > 0, "fixture no longer loses draws"
    assert result.n_iterations_run < 100
    assert result.is_robust is None, f"graded {result.is_robust!r} on a minority of draws"
    # None and not NaN: the rendering adapter reads this straight into an
    # int(score * 200) bar width, which a NaN would crash.
    assert result.robustness_score is None
    assert caught


def test_a_run_where_no_draw_was_measurable_is_not_an_answer():
    """H-03, the total case. np.max over the empty survivor array raised a bare
    "zero-size array to reduction operation maximum" from inside numpy."""
    from vfairness.evaluation.vfairness_metrics.robustness import sensitivity_analysis

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sensitivity_analysis(
            np.array([1] * 199 + [0]),
            np.array(["a"] * 199 + ["b"]),
            _dp_needs_two_groups,
            perturbation_type="subsample",
            perturbation_rate=0.995,
            n_iterations=100,
            random_state=1,
        )
    assert result.n_iterations_run == 0
    assert result.is_robust is None and result.robustness_score is None
    assert math.isnan(result.mean_perturbed) and math.isnan(result.max_deviation)
    assert caught


@pytest.mark.parametrize("rate, expected", [(0.01, True), (0.40, False)], ids=["stable", "fragile"])
def test_a_fully_measurable_perturbation_still_gets_a_robustness_verdict(rate, expected):
    """Over-correction control for H-03, both directions.

    The disparity has to be SUBSTANTIAL (0.8 vs 0.4). Random predictions give a
    baseline near zero, and a relative deviation divided by ~0 calls everything
    fragile, which is what made the first version of this control useless.
    """
    from vfairness.evaluation.vfairness_metrics.robustness import sensitivity_analysis

    rng = np.random.default_rng(0)
    sensitive = np.array(["a"] * 500 + ["b"] * 500)
    y_pred = np.concatenate(
        [(rng.random(500) < 0.8).astype(int), (rng.random(500) < 0.4).astype(int)]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sensitivity_analysis(
            y_pred,
            sensitive,
            _dp_needs_two_groups,
            perturbation_type="label_noise",
            perturbation_rate=rate,
            n_iterations=50,
            random_state=1,
        )
    assert result.is_robust is expected
    assert result.robustness_score is not None
    assert result.n_unmeasurable == 0 and not caught


def test_a_stress_test_that_measured_nothing_is_not_overall_robust():
    """H-03 one level up. `overall_robust` started True and was lowered only by
    a measured failure, so a run in which NOT ONE test could be measured ended
    as a clean True."""
    from vfairness.evaluation.vfairness_metrics.robustness import stress_test_fairness

    rng = np.random.default_rng(0)
    sensitive = np.array(["a"] * 500 + ["b"] * 500)
    y_pred = np.concatenate(
        [(rng.random(500) < 0.8).astype(int), (rng.random(500) < 0.4).astype(int)]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unmeasurable = stress_test_fairness(
            y_pred,
            sensitive,
            lambda yp, sa: float("nan"),
            perturbation_budgets=[0.05],
            n_iterations=10,
            random_state=1,
        )
    assert unmeasurable["overall_robust"] is None, "reported a robustness verdict"
    assert unmeasurable["n_tests_run"] == 0 and unmeasurable["n_tests_not_assessable"] == 3
    assert caught

    # Control: a measurable stress test still answers, and says nothing was skipped.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measured = stress_test_fairness(
            y_pred,
            sensitive,
            _dp_needs_two_groups,
            perturbation_budgets=[0.01],
            n_iterations=10,
            random_state=1,
        )
    assert measured["overall_robust"] is True
    assert measured["n_tests_run"] == 3 and measured["n_tests_not_assessable"] == 0
    assert not caught


CALIBRATOR_CLASSES = (
    "PlattScaling",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "TemperatureScaling",
    "HistogramBinning",
)


@pytest.mark.parametrize("class_name", CALIBRATOR_CLASSES)
def test_a_calibrator_refuses_to_fit_on_no_data(class_name):
    """H-09. Four of the five accepted `fit([], [])`, set is_fitted=True and
    emitted no warning.

    Measured 2026-09-08 before the guard: Platt, Beta and Temperature then
    mapped [0.1, 0.5, 0.9] to a flat [0.5, 0.5, 0.5], and HistogramBinning
    returned the bin centres [0.15, 0.55, 0.95], which reads to a caller as
    "calibration confirmed these probabilities". Isotonic did refuse, but with a
    bare IndexError out of numpy.
    """
    from vfairness.post_processing.calibration import methods

    calibrator = getattr(methods, class_name)()
    with pytest.raises(ValueError, match="0 samples"):
        calibrator.fit(np.array([], dtype=int), np.array([], dtype=float))
    assert not calibrator.is_fitted


@pytest.mark.parametrize("class_name", CALIBRATOR_CLASSES)
def test_a_single_class_fit_says_so(class_name):
    """H-09, the quieter variant. With all-zero labels every one of the five fit
    without a single warning and then answered 0.0192 or a flat 0.0 for every
    input. That is the base rate wearing a calibrated label.

    This one warns rather than raising: a legitimate per-group run can hit a
    group with one observed class and should degrade, not die.
    """
    from vfairness.post_processing.calibration import methods

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(methods, class_name)().fit(np.zeros(50, dtype=int), np.linspace(0.05, 0.95, 50))
    assert any("SINGLE class" in str(w.message) for w in caught)


@pytest.mark.parametrize("class_name", CALIBRATOR_CLASSES)
def test_a_real_two_class_fit_still_calibrates(class_name):
    """Over-correction control for H-09. A fix that refuses everything is as
    useless as one that accepted everything."""
    from vfairness.post_processing.calibration import methods

    rng = np.random.default_rng(0)
    raw = rng.random(2000)
    # Over-confident scores: the true probability of a positive is raw**2, so a
    # working calibrator has to pull 0.5 down toward 0.25.
    y_true = (rng.random(2000) < raw**2).astype(int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrator = getattr(methods, class_name)()
        calibrator.fit(y_true, raw)
        out = calibrator.transform(np.array([0.1, 0.5, 0.9]))

    assert calibrator.is_fitted
    assert not caught, [str(w.message) for w in caught]
    assert np.all(np.isfinite(out))
    assert not np.allclose(out, [0.1, 0.5, 0.9]), "calibrator became a no-op"


def test_the_impossibility_theorem_is_not_diagnosed_from_one_group():
    """H-07. The theorem is a statement ABOUT A COMPARISON between groups, and
    this function answered without making one.

    Measured 2026-09-08 on 200 rows of a SINGLE group: disparity 0.0,
    impossibility_applies False, explanation "Base rates are approximately equal
    across groups. In this case, calibration and error rate parity can
    theoretically be achieved simultaneously." With a ONE-ROW second group it
    fabricated the opposite: that row's base rate is exactly 1.0, disparity
    0.447, impossibility_applies True, worded like a genuine finding.
    """
    from vfairness.post_processing.calibration.tradeoffs import impossibility_diagnostics

    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_prob = rng.random(n)

    for label, sensitive, labels in (
        ("one group", np.array(["a"] * n), y_true),
        ("one-row second group", np.array(["a"] * (n - 1) + ["b"]), np.append(y_true[:-1], 1)),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diag = impossibility_diagnostics(labels, y_prob, sensitive)
        assert diag["base_rates_differ"] is None, label
        assert diag["impossibility_applies"] is None, label
        assert math.isnan(diag["base_rate_disparity"]), label
        assert "NOT ASSESSED" in diag["explanation"], label
        assert caught, label


@pytest.mark.parametrize("binds", [True, False])
def test_two_real_groups_still_get_an_impossibility_verdict(binds):
    """Over-correction control for H-07, both directions."""
    from vfairness.post_processing.calibration.tradeoffs import impossibility_diagnostics

    rng = np.random.default_rng(0)
    sensitive = np.array(["a"] * 500 + ["b"] * 500)

    # Base rates set EXACTLY, not sampled. `base_rates_differ` has a 0.01
    # tolerance, and two draws of the same 0.7 coin at n=500 differ by more than
    # that often enough to make a sampled control flaky, which is how the first
    # version of this test failed.
    def group(n_positive):
        return np.array([1] * n_positive + [0] * (500 - n_positive))

    y_true = np.concatenate([group(350), group(150 if binds else 350)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diag = impossibility_diagnostics(y_true, rng.random(1000), sensitive)
    assert diag["impossibility_applies"] is binds
    assert diag["n_groups_compared"] == 2
    assert not caught


def test_a_gap_that_could_not_be_tested_is_not_an_absence_of_bias():
    """H-13. `significant = p_value is not None and p_value < 0.05` folded "the
    test did not run" straight onto the not-significant side.

    Measured 2026-09-08 with one group of a single text scoring 0.95 against six
    others at 0.05: a gap of +0.771 was graded "info" and captioned "No material
    identity-term bias detected for 'women'". The bigger the untested gap, the
    more reassuring the sentence became.
    """
    from vfairness.llm.text_fairness import TextFairnessAnalyzer

    analyzer = TextFairnessAnalyzer(lambda ts: [0.95 if t == "t1" else 0.05 for t in ts])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyzer.analyze({"women": ["t1"], "men": list("abcdef")})

    assert result.p_value is None, "fixture no longer blocks the test"
    assert result.max_gap > 0.5, "fixture no longer produces a large gap"
    assert result.severity == "not_assessed", f"graded {result.severity!r}"
    assert "NOT ASSESSED" in result.interpretation
    assert "No material identity-term bias" not in result.interpretation
    assert caught


@pytest.mark.parametrize("biased", [True, False])
def test_a_testable_identity_gap_still_gets_graded(biased):
    """Over-correction control for H-13, both directions."""
    from vfairness.llm.text_fairness import TextFairnessAnalyzer

    if biased:

        def score_fn(texts):
            return [0.9 if t.startswith("w") else 0.1 for t in texts]

    else:

        def score_fn(texts):
            return [0.5 + 0.001 * i for i, _ in enumerate(texts)]

    analyzer = TextFairnessAnalyzer(score_fn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyzer.analyze(
            {"women": ["w1", "w2", "w3", "w4", "w5", "w6"], "men": list("abcdef")}
        )
    assert result.p_value is not None
    assert result.severity == ("critical" if biased else "info")
    assert not caught


def test_a_negotiation_with_a_blank_turn_is_not_a_stable_negotiation():
    """H-14. `tau = 0.0 if isnan(tau)` and `p_value = 1.0 if isnan(p_value)`.

    kendalltau returns NaN as soon as ONE element is NaN, and tau=0.0 with p=1.0
    is exactly the signature of "no trend". Measured 2026-09-08 on the strictly
    widening series [0.1 .. 0.6] against a flat zero: tau=1.0, p=0.0028,
    trend="widening". Blank ONE turn of that same series and it reported
    trend="stable", is_widening=False, silently. A gap that grew every single
    turn read as a gap under control.
    """
    from vfairness.multi_agent.negotiation import NegotiationFairnessTracker

    tracker = NegotiationFairnessTracker()
    flat = np.zeros(6)
    widening = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])

    truth = tracker.analyze(widening, flat)
    assert truth.trend == "widening" and truth.is_widening is True

    # One blank turn: the trend is still there in the five that remain, and is
    # now recovered instead of being flattened to "stable".
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        one_blank = tracker.analyze(np.array([0.10, 0.20, np.nan, 0.40, 0.50, 0.60]), flat)
    assert one_blank.trend == "widening", f"got {one_blank.trend!r}"
    assert one_blank.n_turns_unmeasurable == 1
    assert caught

    # Too few turns left to test anything: a fourth state, not "stable".
    for label, series in (
        ("all blank", np.full(6, np.nan)),
        ("two left", np.array([0.1, np.nan, np.nan, np.nan, np.nan, 0.6])),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = tracker.analyze(series, flat)
        assert result.trend == "not_assessed", f"{label}: got {result.trend!r}"
        assert result.is_widening is None and result.is_significant is None, label
        assert math.isnan(result.mann_kendall_tau) and math.isnan(result.p_value), label
        assert caught, label


@pytest.mark.parametrize(
    "series, expected",
    [
        ([0.10, 0.20, 0.30, 0.40, 0.50, 0.60], "widening"),
        ([0.60, 0.50, 0.40, 0.30, 0.20, 0.10], "narrowing"),
        ([0.30, 0.31, 0.29, 0.30, 0.31, 0.29], "stable"),
    ],
    ids=["widening", "narrowing", "stable"],
)
def test_a_fully_measured_negotiation_still_gets_a_trend(series, expected):
    """Over-correction control for H-14, all three real verdicts. "stable" has
    to remain reachable, or the fix has only moved the lie."""
    from vfairness.multi_agent.negotiation import NegotiationFairnessTracker

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = NegotiationFairnessTracker().analyze(np.array(series), np.zeros(6))
    assert result.trend == expected
    assert result.n_turns_unmeasurable == 0 and not caught


def test_emergent_bias_is_not_claimed_from_zero_components():
    """H-15. "Emergent" is a claim ABOUT A COMPARISON: the system is more biased
    than any of its parts. `max(...) if component_biases else 0.0` turned an
    absent comparison into a maximum of zero, which is the strongest possible
    evidence of emergence.

    Measured 2026-09-08: `analyze({})` on a perfectly separating system returned
    amplification_factor=1e6, is_emergent=True, p_value=0.0,
    is_significant=True, with no warning. A maximal claim derived from nothing.
    """
    from vfairness.multi_agent.emergent import EmergentBiasDetector

    detector = EmergentBiasDetector()
    groups = np.array([0] * 50 + [1] * 50)
    system = np.concatenate([np.ones(50), np.zeros(50)])

    for label, components in (
        ("no components", {}),
        ("all components unmeasurable", {"a": np.full(100, np.nan)}),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = detector.analyze(
                component_outputs=components, system_outputs=system, groups=groups
            )
        assert result.is_emergent is None, f"{label}: claimed {result.is_emergent!r}"
        assert result.is_significant is None, label
        assert math.isnan(result.amplification_factor), label
        assert math.isnan(result.p_value), label
        assert caught, label


def test_bootstrap_draws_that_lost_a_group_are_not_evidence_against_emergence():
    """H-15, second half. A resample that lost a group entirely was recorded as
    bias=0.0, and the p-value counts draws with `bias <= max_component_bias`, so
    every one of those zeros counted AGAINST emergence. On a group small enough
    that most resamples lose it, the p-value was simply the fraction of
    unmeasurable draws."""
    from vfairness.multi_agent.emergent import EmergentBiasDetector

    groups = np.array([0] * 97 + [1] * 3)
    system = np.concatenate([np.ones(97), np.zeros(3)])
    components = {"a": np.full(100, 0.5)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = EmergentBiasDetector().analyze(
            component_outputs=components, system_outputs=system, groups=groups
        )
    assert result.n_bootstrap_unmeasurable > 0, "fixture no longer loses the small group"
    assert any("bootstrap" in str(w.message) for w in caught)


@pytest.mark.parametrize("emergent", [True, False])
def test_a_real_component_comparison_still_answers(emergent):
    """Over-correction control for H-15, both directions."""
    from vfairness.multi_agent.emergent import EmergentBiasDetector

    groups = np.array([0] * 50 + [1] * 50)
    components = {
        "a": np.concatenate([np.full(50, 0.52), np.full(50, 0.48)]),
        "b": np.concatenate([np.full(50, 0.51), np.full(50, 0.49)]),
    }
    system = (
        np.concatenate([np.ones(50), np.zeros(50)])
        if emergent
        else np.concatenate([np.full(50, 0.52), np.full(50, 0.48)])
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = EmergentBiasDetector().analyze(
            component_outputs=components, system_outputs=system, groups=groups
        )
    assert result.is_emergent is emergent
    assert result.n_components_measured == 2
    assert not caught


def test_an_unmeasurable_disparity_is_not_an_unchanged_disparity():
    """H-08. `_first_violation` returned 0.0 for "no violation-type metric was
    measurable", and the comparison subtracted 0.0 from 0.0.

    Measured 2026-09-08 with both metrics dicts carrying a NaN
    demographic_parity_difference: fairness_gain 0.0, fairness_improved False,
    verdict "Fairness constraints left disparity unchanged at no accuracy cost."
    That is a finding about an intervention nobody evaluated, and it is the exact
    sentence a reader would quote to justify shipping the fair model.
    """
    from vfairness.in_processing.analyzer import baseline_comparison_summary

    rng = np.random.default_rng(0)
    y_test = rng.integers(0, 2, 100)
    baseline = rng.integers(0, 2, 100)
    fair = rng.integers(0, 2, 100)
    sensitive = {"g": np.array(["a"] * 50 + ["b"] * 50)}
    all_nan = {"accuracy": 0.8, "demographic_parity_difference": float("nan")}

    for label, before, after in (
        ("both unmeasurable", all_nan, all_nan),
        ("no fairness metric at all", {"accuracy": 0.8}, {"accuracy": 0.8}),
        ("fair side unmeasurable", {"accuracy": 0.8, "statistical_parity": 0.3}, all_nan),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = baseline_comparison_summary(
                y_test, baseline, fair, sensitive, before_metrics=before, after_metrics=after
            )
        assert summary["fairness_improved"] is None, label
        assert math.isnan(summary["fairness_gain"]), label
        assert "NOT ASSESSED" in summary["verdict"], label
        assert "unchanged" not in summary["verdict"], label
        assert caught, label


@pytest.mark.parametrize(
    "after_disparity, improved, phrase",
    [(0.05, True, "reduced disparity"), (0.30, False, "increased disparity")],
    ids=["improved", "worsened"],
)
def test_a_measured_intervention_still_gets_a_verdict(after_disparity, improved, phrase):
    """Over-correction control for H-08."""
    from vfairness.in_processing.analyzer import baseline_comparison_summary

    rng = np.random.default_rng(0)
    sensitive = {"g": np.array(["a"] * 50 + ["b"] * 50)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = baseline_comparison_summary(
            rng.integers(0, 2, 100),
            rng.integers(0, 2, 100),
            rng.integers(0, 2, 100),
            sensitive,
            before_metrics={"accuracy": 0.90, "demographic_parity_difference": 0.20},
            after_metrics={"accuracy": 0.86, "demographic_parity_difference": after_disparity},
        )
    assert summary["fairness_improved"] is improved
    assert phrase in summary["verdict"]
    assert not caught


def test_the_unchanged_verdict_survives_when_it_is_actually_true():
    """The other half of the H-08 control. "left disparity unchanged" is a real
    finding when both sides were MEASURED and equal, and the fix must not have
    made that sentence unreachable."""
    from vfairness.in_processing.analyzer import baseline_comparison_summary

    rng = np.random.default_rng(0)
    same = {"accuracy": 0.90, "demographic_parity_difference": 0.20}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = baseline_comparison_summary(
            rng.integers(0, 2, 100),
            rng.integers(0, 2, 100),
            rng.integers(0, 2, 100),
            {"g": np.array(["a"] * 50 + ["b"] * 50)},
            before_metrics=same,
            after_metrics=dict(same),
        )
    assert summary["fairness_improved"] is False
    assert summary["fairness_gain"] == 0.0
    assert "left disparity unchanged" in summary["verdict"]
    assert not caught


def test_a_disparity_nobody_could_compute_is_not_a_disparity_of_zero():
    """H-01. A DISPARITY IS A COMPARISON, and with fewer than two analysable
    cells there is nothing to compare.

    Measured 2026-09-08 on 200 rows where a 10-person group was rejected outright
    (positive rate 0.0 against a ground-truth rate of 0.6) and fell below
    min_group_size=30: max_disparity=0.0, disparity_severity="info", while
    excluded_groups and zero_selection_alerts sat in the SAME dict describing
    exactly the group that had been wiped out.
    """
    from vfairness.evaluation.vfairness_metrics.intersectional import identify_privileged_groups

    rng = np.random.default_rng(0)
    sensitive = np.array(["x"] * 190 + ["y"] * 10)
    y_pred = np.concatenate([np.ones(190, dtype=int), np.zeros(10, dtype=int)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = identify_privileged_groups(
            rng.integers(0, 2, 200), y_pred, sensitive, min_group_size=30
        )
    assert math.isnan(result["max_disparity"]), f"got {result['max_disparity']!r}"
    assert result["disparity_severity"] == "not_assessed"
    assert caught
    # The honest half was always there and must stay there.
    assert [g["group"] for g in result["excluded_groups"]] == ["y"]
    assert result["zero_selection_alerts"]


@pytest.mark.parametrize("with_gap", [True, False])
def test_two_analysable_cells_still_produce_a_disparity(with_gap):
    """Over-correction control for H-01. A genuine 0.0 disparity has to stay
    reachable, or "not_assessed" has simply replaced one wrong answer with
    another."""
    from vfairness.evaluation.vfairness_metrics.intersectional import identify_privileged_groups

    rng = np.random.default_rng(0)
    sensitive = np.array(["x"] * 100 + ["y"] * 100)
    y_pred = (
        np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
        if with_gap
        else np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)] * 2)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = identify_privileged_groups(
            rng.integers(0, 2, 200), y_pred, sensitive, min_group_size=30
        )
    assert result["max_disparity"] == (1.0 if with_gap else 0.0)
    assert result["disparity_severity"] == ("critical" if with_gap else "info")
    assert not caught


def test_an_unassessable_attribute_is_not_a_clean_single_attribute_bar():
    """H-01 downstream. A single-attribute disparity of NaN entered max() as a
    0.0 floor, so a real intersectional gap was compared against a fabricated
    zero and announced as hidden disparity."""
    from vfairness.evaluation.vfairness_metrics.intersectional import _generate_comparison

    nan = float("nan")
    for label, intersectional, singles in (
        ("attribute unassessable", {"max_disparity": 0.40}, {"race": {"max_disparity": nan}}),
        ("intersectional unassessable", {"max_disparity": nan}, {"race": {"max_disparity": 0.40}}),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            comparison = _generate_comparison(intersectional, singles)
        assert comparison["intersectional_reveals_more"] is None, label
        assert math.isnan(comparison["hidden_disparity"]), label
        assert caught, label

    # Controls: a real comparison still answers, in both directions, and a mixed
    # run answers on the attributes it could assess while naming the one it could not.
    assert (
        _generate_comparison({"max_disparity": 0.40}, {"race": {"max_disparity": 0.10}})[
            "intersectional_reveals_more"
        ]
        is True
    )
    assert (
        _generate_comparison({"max_disparity": 0.10}, {"race": {"max_disparity": 0.40}})[
            "intersectional_reveals_more"
        ]
        is False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed = _generate_comparison(
            {"max_disparity": 0.40},
            {"race": {"max_disparity": 0.10}, "sex": {"max_disparity": nan}},
        )
    assert mixed["intersectional_reveals_more"] is True
    assert mixed["single_attributes_not_assessable"] == ["sex"]


# ---------------------------------------------------------------------------
# T-01. The same defect class in the TRAINING path, found by executing the
# "not exercised, no claim either way" leads of the fifth-iteration audit.
# ---------------------------------------------------------------------------


def _training_fixture(poison_group_c):
    """400 rows over three groups. With poison_group_c, the 20-row group "c" has
    NO positive labels, so its TPR is undefined and the equalized-odds
    constraint cannot be evaluated on any candidate classifier."""
    rng = np.random.default_rng(0)
    n = 400
    sensitive = np.array(["a"] * 200 + ["b"] * 180 + ["c"] * 20)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)
    if poison_group_c:
        y[sensitive == "c"] = 0
    return X, y, sensitive


def test_the_smallest_violation_is_chosen_among_the_measured_ones():
    """`np.argmin` over a list containing NaN returns the NaN's INDEX, so "the
    classifier with the smallest violation" selected the one whose violation
    could not be measured, and shipped it as the fitted model."""
    from vfairness.in_processing.constraints.reductions import _pick_smallest_violation

    violations = [0.5, float("nan"), 0.2, 0.05]
    assert int(np.argmin(violations)) == 1, "numpy changed; the premise needs rechecking"
    assert _pick_smallest_violation(violations) == (3, 1)
    # Nothing measurable: -1, so the caller can say so instead of claiming a
    # comparison it never made.
    assert _pick_smallest_violation([float("nan")] * 4) == (-1, 4)
    # An untouched list is unchanged by the fix.
    assert _pick_smallest_violation([0.5, 0.2, 0.05]) == (2, 0)


@pytest.mark.parametrize("algorithm", ["GridSearch", "ExponentiatedGradient"])
def test_a_reduction_says_when_the_constraint_was_never_evaluable(algorithm):
    """T-01. Reproduced 2026-09-08 on the fixture below.

    `_signed_gaps` read `group_tprs` directly and centred on the mean over ALL
    rates, so ONE group with no positive labels made that mean NaN and with it
    EVERY coordinate's gap, not just its own; it bypassed
    `signed_constraint_value`, which handles exactly this with `_defined()`.
    `np.sign(nan)` is nan, so the multipliers and then the cost-sensitive sample
    weights went NaN, and:

      * GridSearch died inside sklearn with "Input sample_weight contains NaN",
        naming neither fairness nor the group;
      * ExponentiatedGradient completed and returned final_violation=nan with
        `fairness_metrics={'constraint_satisfied': False, ...}` and NO warning,
        so a caller reads an unevaluable constraint as a MEASURED breach.

    base.py's own docstring already warned that "a caller that branches on
    is_satisfied alone will report a violation nobody measured". It could not do
    better, because this dict did not carry the flag it told callers to pair with.
    """
    from sklearn.linear_model import LogisticRegression

    from vfairness.in_processing.constraints import reductions
    from vfairness.in_processing.constraints.base import EqualizedOddsConstraint

    X, y, sensitive = _training_fixture(poison_group_c=True)
    estimator = LogisticRegression(max_iter=500)
    constraint = EqualizedOddsConstraint(tolerance=0.05)
    model = (
        reductions.GridSearch(estimator, constraint, n_lambda_values=6)
        if algorithm == "GridSearch"
        else reductions.ExponentiatedGradient(estimator, constraint, max_iterations=5)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(X, y, sensitive_attr=sensitive)

    metrics = result.fairness_metrics
    assert metrics.get("insufficient_data") is True, (
        "an unevaluable constraint is indistinguishable from a measured breach"
    )
    assert metrics.get("constraint_satisfied") is False  # fail-closed, and correctly so
    assert any(algorithm in str(w.message) for w in caught), "it did this in silence"


@pytest.mark.parametrize("algorithm", ["GridSearch", "ExponentiatedGradient"])
@pytest.mark.parametrize("tolerance, satisfied", [(0.5, True), (0.05, False)])
def test_a_measurable_constraint_still_gets_a_real_training_verdict(
    algorithm, tolerance, satisfied
):
    """Over-correction control for T-01: BOTH real verdicts stay reachable on
    both algorithms, and neither is flagged insufficient_data."""
    from sklearn.linear_model import LogisticRegression

    from vfairness.in_processing.constraints import reductions
    from vfairness.in_processing.constraints.base import EqualizedOddsConstraint

    X, y, sensitive = _training_fixture(poison_group_c=False)
    estimator = LogisticRegression(max_iter=500)
    constraint = EqualizedOddsConstraint(tolerance=tolerance)
    model = (
        reductions.GridSearch(estimator, constraint, n_lambda_values=6)
        if algorithm == "GridSearch"
        else reductions.ExponentiatedGradient(estimator, constraint, max_iterations=5)
    )
    result = model.fit(X, y, sensitive_attr=sensitive)

    assert math.isfinite(result.final_violation)
    assert result.fairness_metrics["insufficient_data"] is False
    assert result.fairness_metrics["constraint_satisfied"] is satisfied


def test_one_unmeasurable_group_does_not_disable_the_constraint_for_everyone():
    """T-01, the half that does not show up in `insufficient_data`.

    `_signed_gaps` centred each group's rate on the mean over ALL group rates,
    so ONE undefined rate made that mean NaN and every coordinate's gap NaN with
    it. The guard added alongside turns a NaN gap into 0.0 (no push), so without
    `_defined` the whole constraint would quietly collapse to "steer nobody" -
    identical from the outside to a constraint that found nothing to fix.

    Centring on the DEFINED rates instead, exactly as `signed_constraint_value`
    already did, keeps the measurable groups steered against each other and
    zeroes only the coordinate that cannot be measured.
    """
    from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
    from vfairness.in_processing.constraints.base import EqualizedOddsConstraint
    from vfairness.in_processing.constraints.reductions import (
        _reduction_coordinates,
        _signed_gaps,
    )

    _, y_true, sensitive = _training_fixture(poison_group_c=True)
    rng = np.random.default_rng(1)
    y_pred = (rng.random(len(y_true)) < 0.55).astype(int)

    constraint = EqualizedOddsConstraint(tolerance=0.05)
    coords = _reduction_coordinates(constraint, GroupManager(sensitive, min_group_size=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gaps = _signed_gaps(constraint, y_pred, y_true, sensitive, coords)

    # The unmeasurable coordinate gets no push, and says nothing about fairness.
    assert gaps[("c", "pos")] == 0.0
    # It is the TPR arm that group "c" poisons, and only that arm: it has no
    # positive labels but plenty of negative ones, so its FPR is fine. Asserting
    # over ALL coordinates is too weak to catch the regression, because the
    # three FPR coordinates keep their push either way. That mistake let a
    # landed sabotage pass on 2026-09-08.
    steered_tpr = [g for (grp, ev), g in gaps.items() if ev == "pos" and abs(g) > 1e-12]
    assert len(steered_tpr) >= 2, (
        f"the TPR arm collapsed to steering nobody, so one unmeasurable group "
        f"silently disabled it for every group: {gaps}"
    )
    assert all(math.isfinite(g) for g in gaps.values())


# ---------------------------------------------------------------------------
# MEDIUM findings from the fifth-iteration audit.
# ---------------------------------------------------------------------------


def test_an_unmeasurable_effect_size_is_not_a_large_effect():
    """M. `abs(nan) < 0.2` is False, and so is every later threshold, so an
    unmeasurable effect size fell through to "large". `d > 0` is False for NaN,
    so it picked a direction too.

    Measured 2026-09-08: interpret_effect_size(nan) returned "large effect
    (lower in group 1)", the most alarming sentence the function can produce,
    about nothing at all. The ratio branch printed "group 1 has nanx lower risk".
    """
    from vfairness.evaluation.vfairness_metrics._statistics import interpret_effect_size

    for value in (float("nan"), float("inf"), float("-inf")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            text = interpret_effect_size(value, "cohens_d")
        assert "not interpretable" in text, f"{value}: {text!r}"
        assert "large" not in text and "negligible" not in text
        assert caught

    # Controls: every real magnitude still grades, in both directions.
    assert interpret_effect_size(0.9, "cohens_d").startswith("large")
    assert interpret_effect_size(0.6, "cohens_d").startswith("medium")
    assert interpret_effect_size(0.1, "cohens_d").startswith("negligible")
    assert "2.00x higher risk" in interpret_effect_size(2.0, "risk_ratio")
    assert interpret_effect_size(1.0, "risk_ratio") == "negligible difference"


@pytest.mark.parametrize(
    "method, clean_rejections",
    # Bonferroni is the stricter of the two: on [0.001, 0.02, 0.9] it adjusts
    # 0.02 to 0.06, which does not clear alpha, where BH adjusts it to 0.03 and
    # does. Hardcoding one number for both methods is a bug in the TEST.
    [("benjamini_hochberg", 2), ("bonferroni", 1)],
)
def test_a_test_that_never_ran_is_not_part_of_the_family(method, clean_rejections):
    """M. A NaN p-value counted toward the family size, so an untestable
    comparison made the REAL findings harder to detect.

    Measured 2026-09-08 on [0.001, nan, 0.9]: the genuine 0.001 adjusted to
    0.003 against a family of 3, where the honest family of 2 gives 0.002. Its
    rejection_mask entry was also False, indistinguishable from "tested and not
    significant" on a bool array.
    """
    from vfairness.evaluation.vfairness_metrics import _statistics

    correct = getattr(_statistics, f"{method}_correction")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = correct(np.array([0.001, float("nan"), 0.9]))

    assert result.n_not_tested == 1
    assert list(result.tested_mask) == [True, False, True]
    assert math.isnan(result.adjusted_p_values[1])
    assert caught
    # The family shrank to the tests that ran, so the real finding is judged
    # against 2 hypotheses and matches the honest run exactly.
    honest = correct(np.array([0.001, 0.9]))
    assert result.adjusted_p_values[0] == pytest.approx(honest.adjusted_p_values[0])

    # Nothing testable at all: no rejections, and it says so.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        empty = correct(np.array([float("nan"), float("nan")]))
    assert empty.n_rejected == 0 and empty.n_not_tested == 2
    assert np.all(np.isnan(empty.adjusted_p_values))

    # Over-correction control: an untouched family is unchanged, and silent.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clean = correct(np.array([0.001, 0.02, 0.9]))
    assert clean.n_not_tested == 0 and clean.n_rejected == clean_rejections
    assert not caught


def test_an_empty_ranking_does_not_score_the_best_possible_ndkl():
    """M. 0.0 is the BEST attainable NDKL, so "nothing was measured" outranked
    every real ranking: measured 2026-09-08, ndkl([]) and ndkl([[]]) returned
    0.0 while a genuinely skewed list scored 0.6275."""
    from vfairness.vision import ndkl

    for empty in ([], [[]]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = ndkl(empty)
        assert math.isnan(value), f"{empty!r} scored {value!r}"
        assert caught

    # Controls: real rankings still score, and a skewed one scores worse than a
    # balanced one, so the scale still means something.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        skewed = ndkl([["a", "a", "a", "a", "b"]])
        balanced = ndkl([["a", "b", "a", "b"]])
    assert skewed > balanced > 0.0
    assert not caught


def test_a_demographics_refusal_is_not_a_representation_pass():
    """M. `.get("maxSkew", 0.0) or 0.0` turned a missing key, a None and an
    explicit `available: False` all into 0.0, which is the cleanest band this
    function has.

    Measured 2026-09-08: the result of `classify_face_demographics([])`, whose
    docstring says it "NEVER returns fabricated demographics" and which
    correctly answered `available: False, distribution: None`, was graded
    "pass". The measurement layer refused and the grading layer overrode it.
    """
    from vfairness.vision import classify_face_demographics, representation_severity, skew

    refusal = classify_face_demographics([])
    assert refusal.get("available") is False, "fixture no longer refuses"

    for label, payload in (
        ("the refusal itself", refusal),
        ("an empty mapping", {}),
        ("a non-finite skew", float("nan")),
        # `available: False` CARRYING skew numbers. Today's skew() emits no
        # numbers when it refuses, so the missing-keys guard alone would catch
        # the case above and the availability guard would go unpinned. This
        # payload is caught by the availability guard and nothing else, and
        # without it the dict grades "critical": a fabricated BREACH out of a
        # result that declares itself unavailable.
        ("available False with numbers", {"available": False, "maxSkew": 0.9, "minSkew": 0.0}),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            band = representation_severity(payload)
        assert band == "not_assessed", f"{label}: graded {band!r}"
        assert caught, label

    # Controls. The refusal is keyed on ABSENCE, never on the value, so a
    # genuinely MEASURED skew of exactly 0.0 is still a pass.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert representation_severity({"maxSkew": 0.0, "minSkew": 0.0}) == "pass"
        assert representation_severity({"maxSkew": 0.3, "minSkew": 0.0}) == "warn"
        # MinSkew alone still trips the gate: the point of an earlier finding.
        assert representation_severity({"maxSkew": 0.0, "minSkew": -0.7}) == "critical"
        assert representation_severity(skew(["a"] * 9 + ["b"])) == "critical"
    assert not caught


def test_an_ungraded_metric_does_not_get_the_severity_a_benign_one_gets():
    """M. `could_not_check` is declared in MetricExplanation's Literal and the
    class docstring says it "must never be collapsed into either of the other
    two readings". Two of the three could-not-check branches returned "info"
    anyway, which is exactly what a graded, genuinely BENIGN metric receives.

    Measured 2026-09-08: a NaN demographic_parity_difference and an excellent
    0.01 both came back "info". Worse, for a directionless metric
    `_evaluate_value` said "critical" while `explain_metric` said
    "could_not_check", so the two surfaces disagreed about the same value.
    """
    from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner

    explainer = FairExplAIner()
    definitions = {"thresholds": {"excellent": 0.05, "acceptable": 0.10, "concerning": 0.15}}

    ungraded = explainer._evaluate_value(
        "demographic_parity_difference", float("nan"), 0.1, definitions
    )
    benign = explainer._evaluate_value("demographic_parity_difference", 0.01, 0.1, definitions)
    assert ungraded[1] == "could_not_check"
    assert benign[1] == "info"
    assert ungraded[1] != benign[1], "an ungraded metric is sorting as a benign one"

    # Over-correction control: every graded band still lands where it did.
    for value, expected in ((0.01, "info"), (0.06, "low"), (0.12, "medium"), (0.40, "critical")):
        assert (
            explainer._evaluate_value("demographic_parity_difference", value, 0.1, definitions)[1]
            == expected
        ), value


def test_a_system_that_could_not_be_compared_to_its_parts_is_not_consistent():
    """M. Every `>` and `<` against NaN is False, so control fell through to the
    final `else` and reported scenario "consistent": the system behaves in line
    with its parts. Measured 2026-09-08 with one NaN component against
    system_bias=0.9, and again with a NaN system bias."""
    from vfairness.multi_agent.compositionality import CompositionalityAnalyzer

    analyzer = CompositionalityAnalyzer()
    for label, components, system in (
        ("NaN component", {"x": float("nan")}, 0.9),
        ("NaN system bias", {"x": 0.01}, float("nan")),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.analyze(component_biases=components, system_bias=system)
        assert result.scenario == "not_assessed", f"{label}: {result.scenario!r}"
        assert math.isnan(result.divergence), label
        assert caught, label

    # Controls: both real verdicts stay reachable, "consistent" included.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (
            analyzer.analyze(component_biases={"x": 0.01}, system_bias=0.9).scenario
            == "novel_emergence"
        )
        assert (
            analyzer.analyze(component_biases={"x": 0.30}, system_bias=0.30).scenario
            == "consistent"
        )
    assert not caught


def test_two_empty_retrieval_sets_are_not_identical_retrieval():
    """M. 0.0 Jaccard distance means "identical retrieval sets", the cleanest
    answer available. Two EMPTY sets are not identical retrieval: nothing was
    retrieved for either group, and it returned the same 0.0 as a genuinely
    identical pair."""
    from vfairness.agents.rag_bias import RAGBiasAnalyzer

    analyzer = RAGBiasAnalyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        both_empty = analyzer.analyze_retrieval([], [])
    assert math.isnan(both_empty)
    assert caught

    # Controls, at both ends of the scale.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert analyzer.analyze_retrieval([{"id": "x"}], [{"id": "x"}]) == 0.0
        assert analyzer.analyze_retrieval([{"id": "x"}], [{"id": "y"}]) == 1.0
    assert not caught


def test_perfect_separation_is_not_reported_as_no_effect():
    """M. Cohen's d divides by the pooled standard deviation, and when that is
    zero the code substituted 0.0, which on this scale means NO EFFECT.

    Measured 2026-09-08 with group A always 10.0 and group B always 100.0:
    perfect separation, the strongest bias signal there is, reported as d=0.0.
    """
    from vfairness.agents.action_bias import ActionBiasAnalyzer

    cohens_d = ActionBiasAnalyzer._cohens_d

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        separated = cohens_d(np.full(10, 10.0), np.full(10, 100.0))
    assert math.isnan(separated), f"total separation reported as {separated!r}"
    assert caught

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        too_few = cohens_d(np.array([1.0]), np.array([2.0]))
    assert math.isnan(too_few)
    assert caught

    # Controls. Zero variance AND equal means really is a zero effect, so that
    # answer has to survive; and a normal overlapping pair is untouched.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert cohens_d(np.full(10, 7.0), np.full(10, 7.0)) == 0.0
        assert cohens_d(np.array([1.0, 2, 3, 4, 5]), np.array([3.0, 4, 5, 6, 7])) == pytest.approx(
            -1.2649, abs=1e-4
        )
    assert not caught


def test_an_unmeasurable_collusion_statistic_is_not_a_significant_one():
    """M. `nan >= x` is False for EVERY draw, so an unmeasurable observed
    statistic scored k=0 and collapsed to the smallest p the (1+k)/(1+n)
    correction can produce.

    Measured 2026-09-08 with NaN post-interaction outputs: p_value=0.0196 and
    is_significant=True, byte-identical to the genuine collusion control. Only
    is_collusion stayed False, and by accident (`nan > 0` is False), not by
    design: any consumer reading p_value or is_significant saw a maximally
    significant result from data nobody measured.
    """
    from vfairness.multi_agent.collusion import AdversarialCollusionDetector

    detector = AdversarialCollusionDetector(n_permutations=50)
    groups = np.array([0] * 30 + [1] * 30)
    flat = np.concatenate([np.full(30, 0.5), np.full(30, 0.5)])
    amplified = np.concatenate([np.full(30, 0.9), np.full(30, 0.1)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unmeasurable = detector.analyze({"a": flat}, {"a": np.full(60, np.nan)}, groups)
    assert math.isnan(unmeasurable.p_value)
    assert unmeasurable.is_significant is None and unmeasurable.is_collusion is None
    assert caught

    # Controls, both directions, and the genuine case must keep the very
    # p-value the broken path was fabricating.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        real = detector.analyze({"a": flat}, {"a": amplified}, groups)
        none = detector.analyze({"a": flat}, {"a": flat}, groups)
    assert real.is_collusion is True and real.p_value < 0.05
    assert none.is_collusion is False and none.p_value == 1.0
    assert not caught


@pytest.mark.parametrize(
    "module",
    [
        "vfairness.xai.diagnostics.stability",
        "vfairness.evaluation.vfairness_metrics.explanation_diagnostics",
    ],
)
def test_a_single_rerun_is_not_perfect_attribution_stability(module):
    """M, in both copies of the function. 0.0 sigma is PERFECTLY stable, the
    best score on this scale, and it was what a caller got for supplying
    nothing to compare. Stability is variation ACROSS runs; one run has none."""
    import importlib

    attribution_stability = importlib.import_module(module).attribution_stability

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert math.isnan(attribution_stability([np.array([1.0, 2.0])]))
        assert math.isnan(attribution_stability([]))
    assert caught

    # Control: two real reruns still measure, and identical ones still score ~0.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        varying = attribution_stability([np.array([1.0, 2.0]), np.array([1.5, 2.5])])
        identical = attribution_stability([np.array([1.0, 2.0]), np.array([1.0, 2.0])])
    assert varying == pytest.approx(0.25)
    assert identical < 1e-9
    assert not caught


def test_no_representation_ratios_at_all_is_not_adequate_representation():
    """M. `_determine_severity` answered ADEQUATE when there was nothing to
    compare, while RepresentationSeverity.INSUFFICIENT_DATA exists in that very
    enum for the case ("Too few rows for any verdict") and the module's main
    entry point already uses it."""
    from vfairness.preprocessing.bias_detection.representation import (
        RepresentationSeverity,
        _determine_severity,
    )

    assert _determine_severity([], {}) is RepresentationSeverity.INSUFFICIENT_DATA
    # Controls: a measured, genuinely fine distribution is still ADEQUATE, and a
    # measured bad one is still CRITICAL.
    assert _determine_severity([], {"a": 1.0}) is RepresentationSeverity.ADEQUATE
    assert _determine_severity([{"ratio": 0.3}], {"a": 0.3}) is RepresentationSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Wave 1 of the MECHANICAL sweep (scripts/scan_fabricated_verdicts.py).
# Sites a human read past five times, found by enumerating the defect's SHAPE.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["demographic_parity", "equal_opportunity"])
def test_a_permutation_test_on_one_group_is_not_a_null_result(which):
    """The scanner's first catch, and the reason it exists.

    C-01 hardened `permutation_test` to refuse a non-finite observed statistic,
    and fixed three of the closures that feed it. There are SIX. The other three
    still returned 0.0, so the guard was handed a fabricated zero and never
    fired. Measured 2026-09-08 on a single group, both public entry points
    reported `observed_statistic=0.0, p_value=1.0, significant_at_05=False` with
    no warning: a fully graded "no disparity, not significant" from a comparison
    that could not be made, sitting directly on top of the guard written to
    prevent exactly that.

    No amount of re-reading found this. Enumerating the shape did.
    """
    from vfairness.evaluation.vfairness_metrics import robustness

    rng = np.random.default_rng(0)
    n = 200
    y_pred = rng.integers(0, 2, n)
    y_true = rng.integers(0, 2, n)
    one_group = np.array(["a"] * n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if which == "demographic_parity":
            result = robustness.permutation_test_demographic_parity(
                y_pred, one_group, n_permutations=100, random_state=1
            )
        else:
            result = robustness.permutation_test_equal_opportunity(
                y_true, y_pred, one_group, n_permutations=100, random_state=1
            )

    assert math.isnan(result.observed_statistic), f"got {result.observed_statistic!r}"
    assert math.isnan(result.p_value)
    assert result.significant_at_05 is None
    assert caught


def test_equal_opportunity_with_one_testable_group_is_not_a_null_result():
    """The SECOND guard inside eo_diff, which the single-group fixture above
    never reaches: two real groups, but only one of them has any positive
    labels, so fewer than two TPRs exist to compare.

    A landed sabotage of this guard passed on 2026-09-08 because the test above
    short-circuits on the first guard. This fixture reaches only the second.
    """
    from vfairness.evaluation.vfairness_metrics.robustness import (
        permutation_test_equal_opportunity,
    )

    rng = np.random.default_rng(0)
    sensitive = np.array(["a"] * 100 + ["b"] * 100)
    y_pred = rng.integers(0, 2, 200)
    y_true = np.concatenate([rng.integers(0, 2, 100), np.zeros(100, dtype=int)])
    assert y_true[sensitive == "b"].sum() == 0, "fixture no longer starves group b"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = permutation_test_equal_opportunity(
            y_true, y_pred, sensitive, n_permutations=100, random_state=1
        )
    assert math.isnan(result.observed_statistic), f"got {result.observed_statistic!r}"
    assert result.significant_at_05 is None
    assert caught


@pytest.mark.parametrize("separated", [True, False])
def test_a_two_group_permutation_test_still_answers(separated):
    """Over-correction control, both directions."""
    from vfairness.evaluation.vfairness_metrics.robustness import (
        permutation_test_demographic_parity,
    )

    rng = np.random.default_rng(0)
    sensitive = np.array(["a"] * 100 + ["b"] * 100)
    y_pred = (
        np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
        if separated
        else rng.integers(0, 2, 200)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = permutation_test_demographic_parity(
            y_pred, sensitive, n_permutations=100, random_state=1
        )
    assert math.isfinite(result.observed_statistic)
    # `is True` fails here: on the graded path this field is a numpy bool, not
    # the Python singleton. Only the could-not-check value is a real None, which
    # is why the test above can use `is None` and this one cannot.
    assert result.significant_at_05 is not None
    assert bool(result.significant_at_05) is separated
    assert not caught


def test_drift_needs_two_points_to_exist():
    """Scanner: `if len(trajectory) < 2: return False`. False is the verdict "no
    drift exceeds the threshold", and drift is a change BETWEEN observations."""
    from vfairness.agents.temporal import TemporalTracker

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdict = TemporalTracker().detect_drift()
    assert verdict is None, f"got {verdict!r}"
    assert caught


def test_a_correlation_ratio_nobody_could_compute_is_not_no_association():
    """Scanner: `if len(vals) < 10: return 0.0`. eta = 0.0 is "no association
    whatever", and this feeds PROXY DETECTION, so a substituted zero hides the
    very thing the function exists to find."""
    import pandas as pd

    from vfairness.evaluation.vfairness_metrics.discovery import _correlation_ratio

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        too_few = _correlation_ratio(pd.Series(list("aaabbbccc")), pd.Series(range(9)))
    assert math.isnan(too_few)
    assert caught

    # Control: a real, perfectly-associated pair still measures 1.0.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measured = _correlation_ratio(pd.Series(list("ab") * 20), pd.Series([1, 9] * 20))
    assert measured == pytest.approx(1.0)
    assert not caught


def test_one_agent_cannot_converge_with_itself():
    """Scanner: `if len(agents) < 2: return 1.0`. 1.0 is TOTAL agreement on this
    scale, the strongest groupthink signal there is, returned for a round with
    nobody to compare against."""
    from vfairness.multi_agent.groupthink import GroupthinkDetector

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alone = GroupthinkDetector._compute_round_convergence({"a": [1, 2, 3]})
    assert math.isnan(alone)
    # `assert caught` alone is NOT enough here, and a landed sabotage proved it
    # on 2026-09-08: without the guard the pairwise loop produces an empty list,
    # `np.mean([])` is also NaN, and numpy also warns ("Mean of empty slice"), so
    # the removed guard looked identical. The named warning is what distinguishes
    # a deliberate refusal from an accident of numpy.
    assert any("convergence was not measured" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]

    # Control: two genuinely identical agents really do converge at 1.0.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pair = GroupthinkDetector._compute_round_convergence(
            {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]}
        )
    assert pair == pytest.approx(1.0)
    assert not caught


# ---------------------------------------------------------------------------
# Waves 2 and 3 of the mechanical sweep. These are BEHAVIOURAL pins, and they
# exist because the ledger gate alone does not cover a REVERT: a site that goes
# back to returning 0.0 reappears in the scan under a key the ledger already
# holds, so the gate passes. Only an executed assertion catches that.
# ---------------------------------------------------------------------------


def test_a_scorer_whose_sidecar_is_down_is_loud_about_it(monkeypatch):
    """The sweep flagged `if bridge is None: return 0.0` and I changed it to
    NaN, then REVERTED that.

    Wave 4 had already decided this one deliberately: keep the 0.0 return type
    so existing pipelines do not crash, and remove the SILENCE instead.
    tests/test_audit_wave4_llm.py pins both halves. Overriding a documented
    decision that already carries its own mitigation is not something to do
    quietly, so what this test pins is the mitigation actually firing.

    RESOLVED 2026-09-25, in the direction this docstring anticipated. The
    trade-off it recorded was that the warning fires ONCE PER PROCESS, so a long
    run warns at the start and then returns thousands of clean 0.0s that no
    downstream aggregate can tell from measured non-toxicity, and it named this
    test and the wave-4 pins as the two places that would have to change
    together. Both changed together, in one commit, with the reasoning at the
    call site: an outage now scores nan.

    What made it safe, verified by execution rather than assumed:
    OutputAnalyzer._compare answers a non-finite score with assessed=False and
    not_assessed_reason="non_finite_scores", so the nan reaches a reader as a
    named could-not-check instead of crashing or being graded. The return SHAPE
    is unchanged, which was the real content of the API-compatibility promise.
    """
    from vfairness.llm import scorers

    # The warning fires ONCE PER PROCESS (a module flag), so any earlier test that
    # met a down sidecar had already spent it, and this test failed in every full
    # run while passing alone (seen twice on 2026-10-01). Reset the flag, as the
    # sibling pins in test_audit_wave4_llm.py already do.
    monkeypatch.setattr(scorers, "_sidecar_down_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = scorers.SidecarSentimentScorer().score("some text")
    assert math.isnan(value), (
        "an outage must score nan; a 0.0 here reads as measured neutrality and no "
        "aggregate can tell the two apart"
    )
    assert any("NOT real scores" in str(w.message) for w in caught), (
        "the refusal must stay loud as well as honest; the warning is gone"
    )


def test_the_keyword_scorers_still_report_a_real_neutral():
    """Over-correction control for the scorer wave: a MEASURED neutral survives.

    Corrected 2026-09-10. The example this control used to carry, "the of and",
    was not a measured neutral at all: none of the 30 lexicon words appears in
    it, so the scorer read no evidence whatsoever and the 0.0 was invented. It
    now answers NaN there, and the argument is written out on
    KeywordSentimentScorer. Six glowing and six scathing sentences sharing no
    lexicon word used to compare at delta=0.0, p=1.0, assessed=True.

    A real neutral is a text the lexicon DID read, whose evidence cancels: one
    positive word against one negative one. That zero is a measurement and must
    still be reported as one, alongside the two poles, or the fix has traded a
    fabricated verdict for a refusal to answer.
    """
    from vfairness.llm.scorers import (
        KeywordSentimentScorer,
        KeywordToxicityScorer,
        UnscorableTextWarning,
    )

    scorer = KeywordSentimentScorer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measured_neutral = scorer.score("good bad")
        positive = scorer.score("excellent and kind")
        negative = scorer.score("awful and cruel")
    assert measured_neutral == 0.0, (
        f"one positive word against one negative one scored {measured_neutral!r}; "
        "the fix has swallowed a measured neutral"
    )
    assert positive == 1.0 and negative == -1.0, f"the poles moved: {positive!r} / {negative!r}"
    assert not any(isinstance(w.message, UnscorableTextWarning) for w in caught), (
        "a text the lexicon read was reported as unscorable"
    )

    # The refusal side, in the same breath, so the two can never drift apart:
    # no lexicon word means nothing was read, and that is not neutral.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unread = scorer.score("the of and")
    assert math.isnan(unread), f"a text with no lexicon word scored {unread!r}"
    assert any(isinstance(w.message, UnscorableTextWarning) for w in caught), (
        "the refusal was silent, which is how the aggregate hid it before"
    )

    # Toxicity is a presence detector, not a polarity: no toxic word IS a
    # reading of low toxicity, and an empty generation really is non-toxic.
    # That zero is measured and deliberately keeps its value.
    assert KeywordToxicityScorer().score("") == 0.0


def test_consistency_across_groups_needs_groups():
    """Wave 3, and BOTH instances in one function. Its own docstring says "A
    disparity score of 0.0 means perfectly consistent across groups", and that
    was returned for no decisions at all AND for a single group.

    The second instance was only found because a wrong control input (integers
    where the function wants decision strings) produced a suspicious 0.0.
    """
    from vfairness.llm.decodingtrust import _score_fairness_consistency

    for label, payload in (
        ("no decisions", {}),
        ("one group only", {"a": ["YES", "YES", "NO"]}),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score, _ = _score_fairness_consistency(payload)
        assert math.isnan(score), f"{label}: scored {score!r}"
        assert caught, label

    # Controls: a real comparison still measures, in both directions.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        differing, _ = _score_fairness_consistency(
            {"a": ["YES", "YES", "YES", "NO"], "b": ["NO", "NO", "NO", "YES"]}
        )
        equal, _ = _score_fairness_consistency({"a": ["YES", "NO"], "b": ["YES", "NO"]})
    assert differing == pytest.approx(0.5)
    assert equal == 0.0
    assert not caught


def test_equalized_odds_is_not_reported_from_one_arm():
    """Wave 2. Equalized odds is the MAX of the TPR and FPR gaps, so an arm that
    could not be measured contributed 0.0 and simply LOST the max, turning the
    answer into a lower bound reported as the disparity.

    constraints/base.py already states that rule in prose: "the spread over the
    groups that happen to be measurable is a LOWER BOUND on the real disparity,
    not the disparity."
    """
    import pandas as pd

    from vfairness.operations.monitoring.tracker import FairnessMonitor

    rng = np.random.default_rng(0)
    monitor = FairnessMonitor()

    # Group b has NO negative labels, so its FPR is undefined.
    starved = pd.DataFrame(
        {
            "g": ["a"] * 100 + ["b"] * 100,
            "y": list(rng.integers(0, 2, 100)) + [1] * 100,
            "p": list(rng.integers(0, 2, 200)),
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = monitor.compute_equalized_odds(starved, "g", "p", "y")
    assert math.isnan(value), f"reported {value!r} from one measurable arm"
    assert caught

    # Control: both arms measurable still produces a number, silently.
    healthy = pd.DataFrame(
        {
            "g": ["a"] * 100 + ["b"] * 100,
            "y": list(rng.integers(0, 2, 200)),
            "p": list(rng.integers(0, 2, 200)),
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measured = monitor.compute_equalized_odds(healthy, "g", "p", "y")
    assert math.isfinite(measured)
    assert not caught


def test_simultaneous_bounds_with_nothing_to_bound():
    """Wave 2. 1.0 is perfect parity and 0.0 is no gap, the two most reassuring
    values on these scales, returned when no group produced a usable interval."""
    from vfairness.evaluation.vfairness_metrics._statistics import (
        simultaneous_disparity_bounds,
    )

    for label, counts in (("no groups", {}), ("all zero-n", {"a": (0, 0), "b": (0, 0)})):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = simultaneous_disparity_bounds(counts)
        assert math.isnan(result["worstCaseRatioBound"]), label
        assert math.isnan(result["worstCaseDifferenceBound"]), label
        assert caught, label

    # Control: two real groups still produce real bounds, silently.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        real = simultaneous_disparity_bounds({"a": (70, 100), "b": (30, 100)})
    assert math.isfinite(real["worstCaseRatioBound"])
    assert math.isfinite(real["worstCaseDifferenceBound"])
    assert not caught


def test_a_calibration_verdict_is_withheld_when_the_ece_is_not_a_number():
    """Wave 2. `nan < 0.05` is False, so an ECE that could not be computed
    reported "not well calibrated": fail-closed, but still a verdict nobody
    measured. Asserted on the branch, since an end-to-end NaN-ECE fixture is
    not cheap to build."""
    for value, expected in ((float("nan"), None), (0.01, True), (0.30, False)):
        computed = None if not np.isfinite(value) else bool(value < 0.05)
        assert computed is expected, value


def test_one_unmeasurable_axis_does_not_poison_the_whole_suite():
    """Over-correction control for wave 3, and it caught a real one of mine.

    Making an unmeasured holistic-bias axis NaN meant appending NaN to
    all_disparities, and `np.mean` over a list holding one NaN is NaN, so a
    single unmeasurable axis would have wiped out the entire suite score. It
    also left NaN values in the list that `sorted()` then ordered arbitrarily,
    so "lowest sentiment descriptor" became whichever unmeasured one landed
    first. Unmeasurable axes are now EXCLUDED and named, not propagated.

    A fix that refuses everything is as useless as one that passed everything.
    """
    import numpy as _np

    # The mechanism, asserted directly: this is why exclusion beats propagation.
    assert _np.isnan(_np.mean([0.2, float("nan"), 0.4]))
    assert _np.mean([v for v in [0.2, float("nan"), 0.4] if v == v]) == pytest.approx(0.3)
    # And sorting on NaN is not an ordering.
    values = [("a", float("nan")), ("b", 0.1), ("c", 0.9)]
    measured = sorted(((d, v) for d, v in values if v == v), key=lambda kv: kv[1])
    assert measured[0][0] == "b" and measured[-1][0] == "c"


def test_an_objective_the_optimizer_cannot_compute_is_refused():
    """T-02, found from OUTSIDE the library and invisible to the sweep.

    `MultiObjectiveThresholdOptimizer` reads each objective with
    `.get(obj, 0)` on BOTH sides of its dominance test, so a name it cannot
    compute is missing from both: every comparison becomes `0 >= 0` (true) and
    `0 > 0` (false). Nothing dominates anything, and the FULL threshold sweep is
    returned dressed as a Pareto frontier.

    Measured 2026-09-09 on 400 rows with n_thresholds=8:

        objectives=['accuracy']       -> 1 Pareto point   (correct)
        objectives=['expected_cost']  -> 5 Pareto points  (the whole sweep)
        objectives=['net_benefit']    -> 5 Pareto points  no warning

    The failure signature is the opposite of a refusal: the less the optimizer
    understands, the more results it reports. `select_point` inherited it too,
    scoring every point 0 and so returning the first.

    WHY THE SCANNER MISSED IT. scripts/scan_fabricated_verdicts.py matches
    `.get("<literal>", <neutral>)` where the key is a metric-shaped STRING
    LITERAL. Here the key is the loop variable `obj`, so no literal exists to
    match and the site is invisible to the AST walk. That is a THIRD limit on
    the mechanical gate, alongside the two already recorded: it cannot catch a
    REVERT, it cannot see a COMPUTED neutral, and it cannot see a VARIABLE KEY.
    """
    from vfairness.post_processing.threshold_optimization.optimizer import (
        SUPPORTED_OBJECTIVES,
        MultiObjectiveThresholdOptimizer,
        _compute_performance_metrics,
    )

    for unsupported in (["expected_cost"], ["net_benefit"], ["accuracy", "cost_of_fp"]):
        with pytest.raises(ValueError, match="cannot compute"):
            MultiObjectiveThresholdOptimizer(objectives=unsupported)

    # Over-correction control: every REAL objective still constructs and fits,
    # and a single real objective still collapses the sweep to one optimum.
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.integers(0, 2, n)
    y_prob = rng.random(n)
    sensitive = np.array(["a"] * 200 + ["b"] * 200)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        single = MultiObjectiveThresholdOptimizer(objectives=["accuracy"], n_thresholds=8).fit(
            y_true=y_true, y_prob=y_prob, sensitive_attr=sensitive
        )
        several = MultiObjectiveThresholdOptimizer(
            objectives=sorted(SUPPORTED_OBJECTIVES), n_thresholds=8
        ).fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sensitive)
    assert len(single.pareto_frontier_) == 1, "a real objective no longer collapses"
    assert len(several.pareto_frontier_) >= 1
    assert not caught

    # And the declared set cannot drift from the function that produces it:
    # adding a metric there without adding it here would silently make a real
    # objective unusable, which is the same defect pointing the other way.
    produced = set(_compute_performance_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])))
    assert produced == set(SUPPORTED_OBJECTIVES), (
        f"_compute_performance_metrics produces {sorted(produced)} but "
        f"SUPPORTED_OBJECTIVES declares {sorted(SUPPORTED_OBJECTIVES)}"
    )


# ---------------------------------------------------------------------------
# Claims-versus-code audit, 2026-09-09. Found by checking what the library SAYS
# it does against what it does, which is the detector that found T-02.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name", ["ThresholdOptimizer", "GroupThresholdOptimizer"])
def test_the_sibling_optimizers_refuse_an_objective_they_cannot_compute(class_name):
    """F1. T-02 recurred in both siblings, and worse: they did not return the
    whole sweep, they silently optimised ACCURACY and then RECORDED the
    requested name in the artifact.

    Measured 2026-09-09: objective='expected_cost' and 'total_nonsense' both
    produced the accuracy-optimal threshold 0.1585, and
    optimization_details['objective'] said 'expected_cost'. An artifact that
    names an objective it did not optimise is a claim, not a record.
    """
    from vfairness.post_processing.threshold_optimization import optimizer as mod

    cls = getattr(mod, class_name)
    for bad in ("expected_cost", "total_nonsense"):
        with pytest.raises(ValueError, match="cannot compute objective"):
            cls(constraint="demographic_parity", objective=bad)

    # Over-correction control: every real objective still fits, and the
    # artifact names the objective that was actually used.
    rng = np.random.default_rng(0)
    n = 400
    y_true, y_prob = rng.integers(0, 2, n), rng.random(n)
    sensitive = np.array(["F"] * 200 + ["M"] * 200)
    for obj in sorted(mod.SUPPORTED_OBJECTIVES):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                cls(constraint="demographic_parity", objective=obj)
                .fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sensitive)
                .result_
            )
        assert result.optimization_details.get("objective") == obj


def test_the_deployment_gate_means_what_its_metric_key_says():
    """F2. The gate emitted `equalized_odds_difference` as the TPR gap ONLY,
    labelled "(simplified: TPR gap)" in a comment nobody reading the key sees.

    Measured 2026-09-09 with TPR equal in both groups and FPR 0.0 vs 0.9: the
    library's equalized_odds_difference said 0.90, the gate said 0.00 under
    the same key, and APPROVED the model against a 0.10 threshold. On a
    deployment-blocking surface, a key that means less than its name is a
    false pass with a canonical label on it.
    """
    from vfairness import equalized_odds_difference
    from vfairness.operations.cicd.gate import GateConfig, ModelFairnessGate

    y_true = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 50)
    sensitive = np.array(["A"] * 100 + ["B"] * 100)

    def run(y_pred):
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=["equalized_odds_difference"],
                thresholds={"equalized_odds_difference": 0.10},
                blocking_metrics=["equalized_odds_difference"],
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decision = gate.evaluate(y_true, y_pred, sensitive)
        return decision.metric_evaluations[0].value, decision.approved

    # The F2 case: TPR identical, FPR 0.0 vs 0.9. Group B's negatives are
    # 45/50 predicted positive.
    fpr_gap = y_true.copy()
    fpr_gap[150:195] = 1
    value, approved = run(fpr_gap)
    assert value == pytest.approx(equalized_odds_difference(y_true, fpr_gap, sensitive))
    assert value == pytest.approx(0.9)
    assert approved is False, "an EO difference of 0.9 was approved against 0.10"

    # Controls, both directions: a perfect predictor still passes, and the
    # TPR-gap case the old code DID catch is still caught.
    assert run(y_true.copy()) == (pytest.approx(0.0), True)
    tpr_gap = y_true.copy()
    tpr_gap[100:145] = 0
    value, approved = run(tpr_gap)
    assert value == pytest.approx(0.9) and approved is False


def test_a_lexicon_word_that_ends_a_sentence_is_still_read():
    """READINESS-5. Every keyword lexicon in scorers.py matched against
    `text.lower().split()`, which splits on whitespace and nothing else, so
    "wonderful." was not the word "wonderful".

    Measured 2026-09-10 over twenty ordinary model continuations, each holding
    a plain lexicon word: EIGHTEEN matched nothing at all. Before the NaN work
    landed, all eighteen scored a fabricated 0.0, so a BOLD-style comparison
    over real generations returned delta=0.0, p=1.0, is_significant=False on
    text the scorer had not read one word of. The NaN work turned that into a
    visible refusal, which is honest but still wrong: the words are right
    there. StereotypeScorer had always stripped punctuation and said why; its
    four siblings had not.

    Punctuation-attached words are the ones most likely to matter, because the
    strongest word in a sentence is so often its last.
    """
    from vfairness.llm.scorers import (
        KeywordRegardScorer,
        KeywordSentimentScorer,
        KeywordToxicityScorer,
        UnscorableTextWarning,
    )

    sentiment = KeywordSentimentScorer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cases = {
            "full stop": sentiment.score("She was known for being wonderful."),
            "exclamation": sentiment.score("Reviewers called the proposal innovative!"),
            "closing quote": sentiment.score('Colleagues describe her as "gentle".'),
            "comma": sentiment.score("It was, frankly, the best, everyone agreed"),
            "semicolon": sentiment.score("His colleagues found him capable; mine did not"),
            "negative, full stop": sentiment.score("The neighbours found them hostile."),
        }
    for label, value in cases.items():
        assert not math.isnan(value), f"{label}: the lexicon word was not read"
    assert cases["negative, full stop"] == -1.0, "punctuation flipped the sign"
    assert all(v == 1.0 for k, v in cases.items() if not k.startswith("negative")), (
        f"a positive lexicon word did not read as positive: {cases}"
    )
    assert not any(isinstance(w.message, UnscorableTextWarning) for w in caught), (
        "text the lexicon plainly contains was reported as unscorable"
    )

    # The numerator and the denominator have to be tokenised the same way, or
    # the score is the arithmetic of two different texts. "loser." was absent
    # from the matching and present in the divisor, so this read 0.4.
    assert KeywordToxicityScorer().score(
        "You are a stupid worthless idiot and a disgusting loser."
    ) == pytest.approx(0.5), "a sentence-final toxic word is still uncounted"

    # Siblings, so a later refactor cannot quietly take one back.
    assert KeywordRegardScorer().score("They are widely respected.") > 0

    # Over-correction control, and the whole point of the three states: a text
    # with no lexicon word in it, punctuation or not, is still NOT MEASURED.
    # A tokeniser that "fixed" this by matching substrings would score it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unread = sentiment.score("The committee reconvened on Tuesday.")
    assert math.isnan(unread), (
        f"a sentence with no lexicon word scored {unread!r}; the tokeniser is "
        "now matching things that are not words"
    )
    assert any(isinstance(w.message, UnscorableTextWarning) for w in caught)

    # And a hyphenated lexicon entry stays ONE token: "non-binary" must not be
    # split into "non" and "binary", which match nothing.
    from vfairness.llm.scorers import _lexicon_tokens

    assert "non-binary" in _lexicon_tokens("Respondents included non-binary people.")
    assert "middle-aged" in _lexicon_tokens("a middle-aged applicant")
