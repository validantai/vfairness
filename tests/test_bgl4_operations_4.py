"""BGL4 AUDIT of batch A-operations-4: the overturns, as executable evidence.

Every test in this file asserts the CORRECT behaviour. They were written by the
auditor as ``xfail(strict=True)`` because the shipped code did something else,
which made the file green while the defects were live and LOUD the moment one was
fixed (strict xfail turns an unexpected pass into a failure), so a marker could
not quietly outlive its defect.

STATUS, 2026-09-27, BGL5 fix wave: all 16 OVERTURN markers have been removed
because the defects they describe are fixed in ``src/`` and every one of those
tests now PASSES as an ordinary assertion. The per-test docstrings are kept
verbatim: each still names what was measured on the defective code, which is the
evidence the fix was needed. The corrected behaviour, its sabotages and its
over-correction controls are in ``tests/test_bgl5_operations_4.py``.

TWO tests remain ``xfail(strict=True)``, and both are AUDIT FINDINGS rather than
overturns, on units whose grade the auditor CONFIRMED:
  * ``test_the_decomposition_refusal_survives_a_strict_json_boundary``
    (experimentation/analysis.py: the third state is a bare NaN token, and the
    producer substitutes 1e-10 for a zero total effect).
  * ``test_the_agent_probe_does_not_call_a_truncated_export_a_non_export``
    (pulse/agent_probe.py, a file the BGL5 A-operations-4 batch does not own).
Both are recorded as deferrals in /tmp/claude-501/bgl/fix/fix-A-operations-4.json
with the change each one needs; neither is closed here.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


# ===========================================================================
# 1. monitoring.sequential: a non-finite decision parameter is not a test
# ===========================================================================


def _stepped_series():
    """50 windows at 0.10 then 50 at 0.90. The defaults call this abrupt_drift."""
    rng = np.random.default_rng(0)
    return list(rng.normal(0.10, 0.01, 50)) + list(rng.normal(0.90, 0.01, 50))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"slack": float("nan")},
        {"slack": 1e9},
    ],
)
def test_cusum_refuses_a_decision_parameter_that_specifies_no_test(kwargs):
    """Measured on HEAD: cusum_drift(step, threshold=nan) -> has_drift False,
    max_cusum 24.99, warnings []. The statistic reached 25 and the verdict said
    no drift. run_sprt was given exactly this guard for alpha/beta in the same
    wave; cusum_drift's own two decision parameters are unchecked.
    """
    from vfairness.operations.monitoring.sequential import cusum_drift

    result = cusum_drift(_stepped_series(), **kwargs)
    assert result["has_drift"] is None, (
        f"{kwargs} defines no decision boundary, so no CUSUM test ran; "
        f"has_drift was {result['has_drift']!r} with max_cusum {result['max_cusum']!r}"
    )


@pytest.mark.parametrize(
    "kwargs", [{"lambda_": float("nan")}, {"lambda_": float("inf")}, {"delta": float("nan")}]
)
def test_page_hinkley_refuses_a_decision_parameter_that_specifies_no_test(kwargs):
    """Measured on HEAD: page_hinkley(step, lambda_=nan) -> has_drift False,
    drift_index None, no warning, on a series the defaults flag at index 61.
    """
    from vfairness.operations.monitoring.sequential import page_hinkley

    result = page_hinkley(_stepped_series(), **kwargs)
    assert result["has_drift"] is None, (
        f"{kwargs} puts the PH threshold out of reach of any data, so nothing "
        f"was tested; has_drift was {result['has_drift']!r}"
    )


def test_the_wrapper_does_not_call_an_untested_series_stable():
    """Measured on HEAD:
        sequential_fairness_drift(step, {"threshold": nan}, {"lambda_": nan})
          -> classification 'stable', 0 warnings
    on the same series the defaults classify 'abrupt_drift'. The wrapper's own
    docstring: "stable ... is a MEASURED verdict and it requires both detectors
    to have run."
    """
    from vfairness.operations.monitoring.sequential import sequential_fairness_drift

    result = sequential_fairness_drift(
        _stepped_series(), {"threshold": float("nan")}, {"lambda_": float("nan")}
    )
    assert result["classification"] != "stable", (
        "neither detector had a decision boundary, so 'neither detector flags "
        "drift' is not a statement anyone can make"
    )


# ===========================================================================
# 2. monitoring.drift.run_sprt: ptp == 0.0 is defeated by one bit of jitter
# ===========================================================================


def test_sprt_refuses_a_stream_with_no_meaningful_variation():
    """Measured on HEAD:
        run_sprt([0.9]*40,               0.9, 0.2) -> ('could_not_check', 40, nan)
        run_sprt([0.9]*39 + [0.9+1e-15], 0.9, 0.2) -> ('stable', 1, -9.8e+30)
    One float-rounding artefact, which is what a real averaged metric stream
    carries, turns the refusal into the all-clear the method's own comment says
    it removed: "declared from a single observation, with a likelihood ratio of
    twenty billion behind it".
    """
    from vfairness.operations.monitoring.drift import FairnessDriftDetector

    stream = [0.9] * 39 + [0.9 + 1e-15]
    decision, n, llr = FairnessDriftDetector().run_sprt(stream, 0.9, 0.2)
    assert decision == "could_not_check", (
        f"a stream whose range is {float(np.ptp(np.asarray(stream))):.3g} carries no "
        f"variation to measure evidence against; got {decision!r} at observation {n} "
        f"with log_lambda {llr!r}"
    )


# ===========================================================================
# 3. reporting.interactive: a proposed threshold that is not a number
# ===========================================================================


def _metric_store(rows):
    from vfairness.operations.reporting.store import MetricsStore, StoredMetricRecord

    store = MetricsStore()
    now = datetime.now()
    for value, alert, group in rows:
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(days=1),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=float(value),
                group=group,
                alert=alert,
            )
        )
    return store


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_the_simulation_refuses_a_threshold_that_is_not_a_number(threshold):
    """Measured on HEAD, on a window holding 2 real alerts:
        simulate_threshold_change(store, "demographic_parity", nan)
          -> current 2, projected 0, groups_impacted [], change_abs -2,
             change_pct -100.0, records_simulated 3, not_simulated_reason ''
    Every `value > nan` is False, so no record was compared to anything, and an
    operator choosing a threshold is told the change removes both alerts. The
    same function refuses an unknown metric DIRECTION by contract, and the
    docstring says an empty groups_impacted "would read as no group is
    impacted, which is a measurement".
    """
    from vfairness.operations.reporting.interactive import simulate_threshold_change

    store = _metric_store([(0.02, False, "a"), (0.95, True, "b"), (0.9, True, "b")])
    result = simulate_threshold_change(store, "demographic_parity", threshold)
    assert result["projected_alerts"] is None and result["not_simulated_reason"], (
        f"a proposed threshold of {threshold!r} is not a comparison; got "
        f"projected_alerts={result['projected_alerts']!r}, "
        f"change_pct={result['change_pct']!r}, reason={result['not_simulated_reason']!r}"
    )


# ===========================================================================
# 4. cicd.quality_report: the outcome slot has no could-not-check state
# ===========================================================================


def _clean_frame_with_a_real_gap():
    rng = np.random.default_rng(11)
    n = 400
    g = np.where(rng.random(n) < 0.5, "M", "F")
    y = np.where(g == "M", rng.binomial(1, 0.8, n), rng.binomial(1, 0.3, n))
    return pd.DataFrame(
        {
            "applicant_id": [f"a{i}" for i in range(n)],
            "gender": g,
            "score": rng.normal(0, 1, n),
            "approved": y,
        }
    )


def test_an_outcome_check_that_never_ran_is_not_a_clean_one():
    """Measured on HEAD, same 400 clean rows, outcome deliberately clean:
        outcome_column='approved' -> tone pass, keys [vf_representation,
                                     vf_missing, vf_hygiene]
        outcome_column=None       -> BYTE-IDENTICAL
    The representation slot was given a could-not-check row in this campaign
    and a typo'd outcome name was given one too; naming no outcome column at
    all still produces three green tiles and "The data is fit for a reliable
    fairness read", with nothing saying the outcome-disparity and label-quality
    checks were never performed.
    """
    from vfairness.operations.cicd.quality_report import build_quality_report

    df = _clean_frame_with_a_real_gap()
    df["approved"] = np.random.default_rng(3).binomial(1, 0.6, len(df))  # no gap

    checked = build_quality_report(df, ["gender"], outcome_column="approved")
    never = build_quality_report(df, ["gender"], outcome_column=None)
    assert [c["key"] for c in checked["checks"]] != [c["key"] for c in never["checks"]], (
        "a report over an outcome column that was graded and one over no "
        "outcome column at all are the same document: "
        f"{[c['key'] for c in never['checks']]}"
    )


# ===========================================================================
# 5. reporting.compliance.build_assurance_verdict: partial assessability
# ===========================================================================


def test_five_unassessable_attributes_of_six_are_a_scope_limitation():
    """Measured on HEAD: 5 of 6 protected variables refused ("groups too
    small"), the sixth clean ->
        overall 'Unqualified', blocksDeployment False, unassessed [],
        "no material fairness defect found on the assessed attributes"
    byte-identical to a run where 'gender' was the only attribute requested.
    The function's own scope-limitation machinery is fed only from the `bias`
    list, never from per_variable entries carrying assessable=False.
    """
    from vfairness.operations.reporting.compliance import build_assurance_verdict

    per_variable = [
        {"attribute": "gender", "assessable": True, "gap": 0.01, "significant": False}
    ] + [
        {"attribute": a, "assessable": False, "reason": "groups too small"}
        for a in ("race", "age", "disability", "religion", "nationality")
    ]
    partial = build_assurance_verdict(per_variable=per_variable, has_truth=True)
    whole = build_assurance_verdict(
        per_variable=[per_variable[0]],
        has_truth=True,
    )
    assert (partial["overall"], partial["oneLineVerdict"]) != (
        whole["overall"],
        whole["oneLineVerdict"],
    ) or partial["unassessed"], (
        "an opinion covering one of six protected variables reads exactly like "
        f"one covering the only variable requested: {partial['oneLineVerdict']!r}"
    )


# ===========================================================================
# 6. reporting.compliance.generate_iso42001_evidence_map: presence is not evidence
# ===========================================================================


def test_contentless_evidence_does_not_earn_full_iso42001_coverage():
    """Measured on HEAD: nine evidence keys holding one contentless placeholder
    each ({"a": None}, [0], [{}], {"x": 0}) earn coverage_percent 92.3 with 11
    controls 'covered' and 0 gaps, the same figure the fix's note records for a
    full wizard; {"risk_register": [{}]} alone earns 26.9% with A.4.3 asserting
    "Treatment plans generated for each identified risk". The empty wizard was
    fixed to 0.0%; the adjacent input class still pays out in full.
    """
    from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

    placeholders = {
        "system_profile": {"a": None},
        "bias_results": [0],
        "metric_results": [0],
        "risk_register": [{}],
        "monitoring_config": {"x": 0},
        "intervention_results": [{}],
        "dpia": {"x": 0},
        "annex_iv": {"x": 0},
        "model_card": {"x": 0},
    }
    result = generate_iso42001_evidence_map(placeholders)
    assert result["coverage_percent"] < 90.0, (
        f"a wizard whose every section is an empty placeholder earned "
        f"{result['coverage_percent']}% of ISO 42001 with {result['covered']} "
        f"controls 'covered' and {result['gaps']} gaps"
    )


# ===========================================================================
# 7. cicd.precommit: a bound no value can exceed is a gate that cannot fail
# ===========================================================================


def test_a_bound_no_value_can_exceed_is_refused(tmp_path):
    """Measured on HEAD, end to end:
        check_fairness_config([fairness.json]) == 0   for
          {"metrics": ["demographic_parity_difference"],
           "thresholds": {"demographic_parity_difference": 1e9}}
        ModelFairnessGate built from it -> GateDecision(approved) for a
          demographic_parity_difference of 0.90
    which is verbatim the harm this checker's own comment exists to prevent
    ("that config APPROVED a model whose equalized_odds_difference was 0.90").
    A rate difference lies in [0, 1], so a bound of 1e9 is a gate that cannot
    fail; the checker validates the bound's TYPE and never its range.
    """
    from vfairness.operations.cicd.precommit import check_fairness_config

    path = tmp_path / "fairness.json"
    path.write_text(
        json.dumps(
            {
                "metrics": ["demographic_parity_difference"],
                "thresholds": {"demographic_parity_difference": 1e9},
            }
        )
    )
    assert check_fairness_config([str(path)]) == 1, (
        "a demographic-parity-difference bound of 1e9 cannot be exceeded by any "
        "value, so the gate it configures can never fail"
    )


def test_the_gate_no_longer_approves_that_disparity_either():
    """THE SECOND HALF, NOW CLOSED TOO. Its SUBJECT is unchanged: the harm of a
    bound no value can exceed is what the GATE built from that config does with a
    real disparity, not what the checker prints. That is why this test exists
    beside the one above, so the overturn never rested on reading the gate's code.

    What changed is the answer, and only because the defect was fixed. Executed
    here on 2026-09-27 and again on 2026-09-30 BEFORE the runtime guard landed:

        ModelFairnessGate(thresholds={demographic_parity_difference: 1e9})
        .evaluate_from_metrics({demographic_parity_difference: 0.90})
        -> GateDecision(approved, approved=True)

    so the pre-commit hook was the ONLY layer refusing it, and a gate configured
    in code never went past the hook at all. The gate now applies the same rule at
    runtime (a bound is unusable when the comparison it controls cannot be False
    for any data in the metric's range), so both halves refuse it and this asserts
    the current one. Keeping `assert approved is True` here would pin the defect.
    """
    from vfairness.operations.cicd.gate import ModelFairnessGate

    gate = ModelFairnessGate(
        metrics=["demographic_parity_difference"],
        thresholds={"demographic_parity_difference": 1e9},
    )
    decision = gate.evaluate_from_metrics({"demographic_parity_difference": 0.90})
    assert decision.approved is False, (
        f"a bound of 1e9 on a rate difference grades nothing, so a 0.90 disparity "
        f"must not be approved against it, got {decision!r}"
    )
    reasons = decision.blocking_reasons
    assert any("COULD NOT CHECK" in r for r in reasons), reasons
    # The measurement itself is still reported: what is withheld is the verdict.
    assert decision.metric_evaluations[0].value == 0.90


# ===========================================================================
# 8. causal.graph: a graph with roles and no edges was never traced
# ===========================================================================


def test_a_graph_with_no_edges_was_not_cleared_of_direct_discrimination():
    """Measured on HEAD: a graph declaring gender=protected and hiring=outcome
    and NO edges returns has_direct_discrimination() False, severity 'info',
    assessable True, not_assessable [], with no warning. The five shapes the
    fix pinned all have a MISSING ROLE; the edge-analogue of its own
    number_of_nodes() == 0 guard is absent, and the reason it prints for the
    empty graph ("no protected -> outcome pathway can exist by construction")
    is true word for word of a graph with no edges.
    """
    from vfairness.operations.causal.graph import CausalFairnessGraph

    g = CausalFairnessGraph()
    g.add_variable("gender", protected=True)
    g.add_variable("hiring", outcome=True)
    assert g.has_direct_discrimination() is None, (
        "no causal structure was authored, so the graph was not traced; "
        f"summary severity was {g.summary()['severity']!r}"
    )


# ===========================================================================
# 9. causal.task_handlers.load_dataset_from_payload: errors="replace"
# ===========================================================================


def test_undecodable_bytes_do_not_silently_merge_two_groups():
    """Measured on HEAD: a latin-1 CSV (what Excel exports) whose group column
    holds two distinct values 0xC4 and 0xC5 comes back as ONE group, because
    both bytes decode to U+FFFD under errors="replace". No warning. The column
    a fairness analysis groups by is the one being corrupted.
    """
    from vfairness.operations.causal.task_handlers import load_dataset_from_payload

    raw = "grp,y\n\xc4,1\n\xc5,0\n\xc4,1\n\xc5,0\n".encode("latin-1")
    df = load_dataset_from_payload({"dataset_bytes": raw})
    assert df is None or len(set(df["grp"])) == 2, (
        f"two distinct group labels became {sorted(set(df['grp']))!r}"
    )


def test_a_blob_that_is_not_text_is_not_a_dataset():
    """Measured on HEAD: gzip bytes (a stand-in for a still-encrypted or parquet
    payload the consumer was supposed to decode) return a 1x1 DataFrame whose
    column name is mojibake, with no warning, so the handlers' `data is None`
    refusal never fires and the mediation runs on one garbage cell.
    """
    import gzip

    from vfairness.operations.causal.task_handlers import load_dataset_from_payload

    df = load_dataset_from_payload({"dataset_bytes": gzip.compress(b"grp,y\na,1\n")})
    assert df is None, f"returned a {df.shape} frame with columns {list(df.columns)!r}"


# ===========================================================================
# 10. reporting.compliance.compute_adverse_action_reasons: the other direction
# ===========================================================================


def test_a_notice_that_never_saw_the_strongest_driver_is_not_complete():
    """Measured on HEAD: debt_ratio is the strongest adverse factor at -0.40 and
    is attributed in shap_values, but absent from feature_names. Result:
    RC01 income, RC02 credit_score, RC03 employment, RC04 zip_code (the proxy),
    complete True, unattributed [], zero warnings. R-6 fixed the mirror image
    (a name in feature_names with no attribution) and recorded exactly this
    harm: "the top reason became income and zip_code, a proxy, entered at
    RC04". `complete` asserts "the ranking considered the whole model".
    """
    from vfairness.operations.reporting.compliance import compute_adverse_action_reasons

    shap = {
        "debt_ratio": -0.40,
        "income": -0.30,
        "zip_code": -0.25,
        "credit_score": -0.20,
        "age_proxy": -0.15,
        "employment": -0.10,
    }
    names = ["income", "zip_code", "credit_score", "age_proxy", "employment"]
    notice = compute_adverse_action_reasons(shap, names, proxy_features=["zip_code", "age_proxy"])
    assert notice.complete is False, (
        "the strongest attributed adverse factor was never in the ranking, yet "
        f"the notice reports complete=True with top reason {notice[0]['feature']!r}"
    )


# ===========================================================================
# 11. reporting.compliance.compute_signed_test_log: what the seal covers
# ===========================================================================


def test_the_signed_log_seals_the_timestamp_the_caller_declared():
    """Measured on HEAD: test_results["timestamp"], documented in this
    function's own Parameters section, is never read. The sealed
    test_timestamp is the moment the log was BUILT, so an Annex IV provenance
    record asserts a test time nobody measured.
    """
    from vfairness.operations.reporting.compliance import compute_signed_test_log

    log = compute_signed_test_log(
        {
            "metrics": [{"name": "dp", "value": 0.02, "threshold": 0.1, "passed": True}],
            "overall_pass": True,
            "timestamp": "2024-01-15T09:00:00+00:00",
        },
        data_hash="abc123",
        lib_version="0.1.0",
    )
    assert log["test_timestamp"] == "2024-01-15T09:00:00+00:00", (
        f"declared 2024-01-15T09:00:00+00:00, sealed {log['test_timestamp']!r}"
    )


def test_the_content_hash_covers_the_intervention_history_it_returns():
    """Measured on HEAD: erasing every record from intervention_history leaves
    the recomputed content_hash IDENTICAL, so the field is not sealed. The
    docstring calls the result "a tamper-evident record of the test execution"
    and lists the interventions as "applied before testing".
    """
    import hashlib

    from vfairness.operations.reporting.compliance import compute_signed_test_log

    payload = {
        "metrics": [{"name": "dp", "value": 0.02, "threshold": 0.1, "passed": True}],
        "overall_pass": True,
    }
    log = compute_signed_test_log(
        payload,
        data_hash="abc123",
        lib_version="0.1.0",
        intervention_history=[{"intervention": "reweighting"}],
    )
    recomputed = hashlib.sha256(
        json.dumps(
            {
                "test_timestamp": log["test_timestamp"],
                "data_hash": log["data_hash"],
                "lib_version": log["library_version"],
                "metrics_snapshot": log["metrics_snapshot"],
                "summary": log["test_results_summary"],
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    assert recomputed != log["content_hash"], (
        "the hash of the document with intervention_history erased equals the "
        "sealed hash, so the seal does not cover it"
    )


# ===========================================================================
# 12. cicd.task_handlers.main: a refusal exits 0
# ===========================================================================


def test_the_cli_does_not_exit_zero_on_a_refusal():
    """Measured on HEAD, as a shell sees it:
        python -m vfairness.operations.cicd.task_handlers vfairness_data_validation
        <<< a payload whose protected attributes are absent
          -> exit 0, stdout {"success": false, "error": "None of the protected
             attributes are present in the data, so group fairness cannot be
             validated."}
    The module header names CI jobs as a consumer, bad JSON and an unknown task
    type already exit 1, and the sibling precommit.main was fixed in this same
    campaign for exactly this: "pre-commit judges a hook purely by its exit
    code ... rather than reporting a pass it did not earn".
    """
    import io
    import sys

    from vfairness.operations.cicd import task_handlers

    frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    saved = (sys.argv, sys.stdin, sys.stdout)
    sys.argv = ["prog"]
    sys.stdin = io.StringIO(
        json.dumps({"csv_data": frame.to_csv(index=False), "protected_attributes": ["not_here"]})
    )
    sys.stdout = io.StringIO()
    try:
        code = task_handlers.main()
        out = sys.stdout.getvalue()
    finally:
        sys.argv, sys.stdin, sys.stdout = saved

    assert json.loads(out)["success"] is False, "the fixture stopped producing a refusal"
    assert code == 1, f"a refusal exited {code}, which a CI step reads as success"


# ===========================================================================
# 13. experimentation.analysis: the pinned NaN cannot cross a JSON boundary
# ===========================================================================


@pytest.mark.xfail(strict=True, reason="AUDIT FINDING: the third state is a bare NaN token")
def test_the_decomposition_refusal_survives_a_strict_json_boundary():
    """Measured on HEAD: CausalDecomposition.to_dict() carries the undefined
    proportion as float nan, json.dumps writes the bare literal NaN, and
    json.dumps(allow_nan=False) refuses the document outright. This repo's own
    standard, tests/test_bgl2_cicd_and_experimentation.py: "a bare NaN token is
    an unparseable answer and reads as an infrastructure failure rather than as
    the refusal it is". Separately, the producer never emits that nan at all:
    analysis.py substitutes 1e-10 for a zero total effect and clamps, so an
    undefined proportion is reported as 1.0 ("fully mediated").
    """
    from vfairness.operations.experimentation.analysis import CausalDecomposition

    payload = CausalDecomposition(
        total_effect=0.0,
        direct_effect=0.0,
        indirect_effect=0.0,
        mediator="credit_score",
        proportion_mediated=float("nan"),
    ).to_dict()
    json.dumps(payload, allow_nan=False)


# ===========================================================================
# 14. pulse.agent_probe: the second consumer cannot tell the two Nones apart
# ===========================================================================


@pytest.mark.xfail(strict=True, reason="AUDIT FINDING: _resolve_trace_frame passes no diagnostics")
def test_the_agent_probe_does_not_call_a_truncated_export_a_non_export():
    """maybe_flatten_spans is honest: it returns None, warns, and fills
    ``diagnostics`` so "the caller can now tell the two Nones apart" (its own
    docstring). agent_probe._resolve_trace_frame calls it WITHOUT diagnostics
    and then states the reassuring one. Measured on HEAD, on a truncated span
    export: notes = "The expected columns are missing and the frame is not a
    recognizable span/observation export; it was used as supplied."
    """
    from vfairness.operations.pulse.agent_probe import _resolve_trace_frame

    records = [
        {
            "trace_id": f"t{i}",
            "span_id": f"s{i}",
            "name": "tool.call",
            "attributes": {"tool.name": "search", "user.group": ["A", "B"][i % 2]},
            "start_time": 1000 + i,
        }
        for i in range(8)
    ]
    truncated = pd.DataFrame({"span": [json.dumps(r)[:60] for r in records]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _frame, _act, _grp, disclosure = _resolve_trace_frame(truncated, {}, "tool", "group")
    assert "is not a recognizable" not in disclosure["notes"], (
        f"a truncated trace export is disclosed as not being one: {disclosure['notes']!r}"
    )
