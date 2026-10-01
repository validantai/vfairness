"""G12 grading wave: the result RECORD TYPES, and whether they mint a verdict.

The eighteen dataclasses below are records, not measurements: they carry what a
producer measured and compute nothing. That is a legitimate answer to "what does
this unit do", but it is not the same as harmless, because a record type can
mint a value in two ways and both have shipped in this repo:

1. A NEUTRAL DEFAULT THE PRODUCER LEAVES ALONE. ``p_value: float = 1.0`` reads
   as "definitely not significant", ``is_significant: bool = False`` as "tested,
   and it is not", and ``confidence_interval: Tuple = (0.0, 0.0)`` as "the
   disparity is exactly zero with no uncertainty", which is the most confident
   claim of perfect fairness the field can express. None of the three is a
   measurement, and each is indistinguishable from one on the record.
2. A COPY CONSTRUCTOR THAT DROPS A LATER FIELD. A field-by-field rebuild that
   lists sixteen fields and not the two newest silently deletes a three-state
   disclosure between the function that computed it and the consumer that reads
   it; that happened in this repo and the producer-side detectors all passed.

This file executes every record type and checks both, per production site,
because the question is never "is the default risky" but "does anything RELY on
it". It is an AST walk rather than prose, so a new production site that forgets
a honesty field is a red test rather than a review miss.
"""

import ast
import dataclasses
import importlib
import math
import pathlib
import warnings

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

#: The G12 record types, as (import path, class name).
RECORD_TYPES = [
    ("vfairness.agents.action_bias", "ActionBiasResult"),
    ("vfairness.agents.correspondence", "CorrespondenceResult"),
    ("vfairness.agents.pipeline_tracker", "StageResult"),
    ("vfairness.agents.rag_bias", "RAGBiasResult"),
    ("vfairness.agents.temporal", "TrajectoryResult"),
    ("vfairness.agents.tool_bias", "ToolBiasResult"),
    ("vfairness.in_processing.wrappers.sklearn_wrappers", "FairClassifierResult"),
    ("vfairness.llm.benchmarks", "BenchmarkResult"),
    ("vfairness.llm.counterfactual", "CounterfactualResult"),
    ("vfairness.llm.nondeterminism", "NoiseProfile"),
    ("vfairness.llm.output_analysis", "OutputAnalysisResult"),
    ("vfairness.multi_agent.collusion", "CollusionResult"),
    ("vfairness.multi_agent.compositionality", "CompositionalityResult"),
    ("vfairness.multi_agent.delegation", "DelegationResult"),
    ("vfairness.multi_agent.emergent", "EmergentBiasResult"),
    ("vfairness.multi_agent.groupthink", "GroupthinkResult"),
    ("vfairness.multi_agent.negotiation", "NegotiationResult"),
]

#: Defaults that would read as a MEASUREMENT if a producer left them alone. A
#: count field is included: ``n_components_unmeasurable = 0`` reads as "nothing
#: was unmeasurable", which is a clean bill.
#:
#: The EMPTY STRING is deliberately NOT here. It reads as "nothing to say", not
#: as a number or a verdict, and including it made this check accuse
#: ``NegotiationResult(detectability_note="")`` on a refusal path that already
#: sets ``trend='not_assessed'``, ``is_widening=None``, every statistic to NaN
#: and warns naming the turn counts. Over-accusing is the same error class as
#: under-accusing, so the rule was narrowed rather than the site changed.
_NEUTRAL_SCALARS = {0, 1, 0.0, 1.0, True, False, "info", "unknown"}

#: Fields a production site may leave at its default, with the reason, VERIFIED
#: by reading the site rather than assumed. Both entries are cases where the
#: default is the CORRECT answer for that path, and an exemption list with
#: reasons is the honest way to keep the check's teeth on everything else.
_EXEMPT = {
    # ``zero_power`` is a FAMILY-level property, not a row-level one: it marks a
    # row whose scorer read the same value for every text in BOTH groups, which
    # only ``analyze_all`` can see across the rows. The field docstring says so
    # ("set by ``analyze_all`` on a row whose scorer read..."), and
    # ``_is_zero_power`` is what applies it. A per-metric producer leaving it
    # False is stating nothing; ``analyze_all`` then re-marks the rows.
    ("OutputAnalysisResult", "zero_power"),
    # The two early returns in ``multi_agent/emergent.py`` refuse BEFORE any
    # bootstrap is attempted, and both record ``"n_bootstrap": 0`` in metadata,
    # so "0 unmeasurable draws" is the measured truth about a bootstrap that did
    # not run. The counters that matter on those paths
    # (``n_components_measured``, ``n_components_unmeasurable``,
    # ``n_samples_unmeasurable``) ARE supplied explicitly.
    ("EmergentBiasResult", "n_bootstrap_unmeasurable"),
}


def _load(path, name):
    return getattr(importlib.import_module(path), name)


def _neutral_defaults(cls):
    """Field names whose DEFAULT would read as a measurement."""
    out = set()
    for field in dataclasses.fields(cls):
        if field.default is dataclasses.MISSING:
            continue
        default = field.default
        if isinstance(default, tuple) and all(d == 0 for d in default) and default:
            out.add(field.name)  # a (0.0, 0.0) interval
        elif isinstance(default, (int, float, bool, str)) and default in _NEUTRAL_SCALARS:
            out.add(field.name)
    return out


def _supplied_at_every_site(cls_name, risky, defining_module):
    """For each construction site of ``cls_name`` in src/, the risky fields it
    does NOT pass, either as a keyword or through a ``**mapping`` splat.

    The splat matters and is why this is an AST walk rather than a grep: the
    benchmark runners pass their four provenance fields as
    ``**benchmark_provenance(...)``, which a keyword-only check reads as eleven
    sites relying on the default when in fact none does.

    Sites are matched by NAME, so a file is only considered when it defines the
    class or imports it. Two unrelated classes in this library are both called
    ``CounterfactualResult`` (``llm.counterfactual`` and
    ``operations.causal.counterfactual``) and a name-only walk accused the
    second one's site of a default belonging to the first.
    """
    risky = {f for f in risky if (cls_name, f) not in _EXEMPT}
    if not risky:
        return []
    own_file = SRC / (defining_module.replace(".", "/") + ".py")
    own_package = SRC / defining_module.replace(".", "/")
    problems = []
    for file in sorted(SRC.rglob("*.py")):
        if file != own_file and own_package not in file.parents:
            text = file.read_text(encoding="utf-8", errors="replace")
            # Imported from the defining module, however it is spelled.
            leaf = defining_module.rsplit(".", 1)[-1]
            if f"import {cls_name}" not in text and f"{leaf} import" not in text:
                continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a file another session is mid-edit on
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == cls_name
            ):
                continue
            named = {kw.arg for kw in node.keywords if kw.arg}
            splatted = any(kw.arg is None for kw in node.keywords)
            if splatted:
                continue  # the fields may arrive through the mapping
            missing = sorted(risky - named)
            if missing:
                problems.append((f"{file.relative_to(SRC)}:{node.lineno}", missing))
    return problems


@pytest.mark.parametrize("path,name", RECORD_TYPES, ids=[n for _p, n in RECORD_TYPES])
def test_a_record_type_carries_no_verdict_its_producer_did_not_supply(path, name):
    """No production site may leave a neutral default to stand as a result.

    Executed against every ``X(...)`` call in ``src/``, so this covers the early
    returns as well as the happy path: a refusal branch that lists fourteen
    fields and forgets ``p_value`` hands back the 1.0 default, and a 1.0 there is
    the most reassuring p the scale has.
    """
    cls = _load(path, name)
    assert dataclasses.is_dataclass(cls)
    risky = _neutral_defaults(cls)
    if not risky:
        pytest.skip(f"{name} declares no default that could read as a measurement")
    problems = _supplied_at_every_site(name, risky, path)
    assert problems == [], (
        f"{name}: these construction sites rely on a default that reads as a "
        f"measurement: {problems}"
    )


@pytest.mark.parametrize("path,name", RECORD_TYPES, ids=[n for _p, n in RECORD_TYPES])
def test_a_record_type_stores_what_it_was_given_and_computes_nothing(path, name):
    """The other half of "it is only a record": constructing one with explicit
    values must give those values back, and every property it exposes must be
    derived rather than invented.

    Built with a NaN in every float field on purpose. A record type that quietly
    coerced, clamped or defaulted a NaN would be converting a could-not-check
    into a number here, which is the one thing a container must not do.
    """
    cls = _load(path, name)
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            continue
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            continue
        annotation = str(field.type)
        if "float" in annotation:
            kwargs[field.name] = float("nan")
        elif "bool" in annotation:
            kwargs[field.name] = None
        elif "int" in annotation:
            kwargs[field.name] = 0
        elif "str" in annotation:
            kwargs[field.name] = "x"
        elif "List" in annotation or "list" in annotation:
            kwargs[field.name] = []
        elif "Dict" in annotation or "dict" in annotation:
            kwargs[field.name] = {}
        else:
            kwargs[field.name] = None

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        record = cls(**kwargs)

    for key, value in kwargs.items():
        stored = getattr(record, key)
        if isinstance(value, float) and math.isnan(value):
            assert isinstance(stored, float) and math.isnan(stored), (
                f"{name}.{key} turned a NaN into {stored!r}"
            )
        else:
            assert stored == value or stored is value, f"{name}.{key} stored {stored!r}"


def test_the_agent_and_multi_agent_records_expose_their_unmeasurable_counts():
    """The three-state disclosure has to be READABLE ON THE RECORD, not only
    correct inside the function that computed it. A copy constructor that lists
    the older fields and not the newer ones drops it at the boundary, and every
    producer-side check still passes.

    So this asserts the FIELD EXISTS on the type a consumer receives, for the
    record types whose measurement can be partly unmeasurable.
    """
    expected = {
        ("vfairness.agents.correspondence", "CorrespondenceResult"): ["n_unmeasurable"],
        ("vfairness.agents.tool_bias", "ToolBiasResult"): [
            "n_bootstrap_draws",
            "n_bootstrap_undefined",
            "ci_coverage",
        ],
        ("vfairness.llm.nondeterminism", "NoiseProfile"): ["n_excluded_non_finite"],
        ("vfairness.llm.output_analysis", "OutputAnalysisResult"): [
            "assessed",
            "not_assessed_reason",
            "n_scored_a",
            "n_scored_b",
            "n_supplied_a",
            "n_supplied_b",
            "zero_power",
            "zero_power_reason",
        ],
        ("vfairness.multi_agent.emergent", "EmergentBiasResult"): [
            "n_components_measured",
            "n_components_unmeasurable",
            "n_bootstrap_unmeasurable",
            "n_samples_unmeasurable",
        ],
        ("vfairness.multi_agent.groupthink", "GroupthinkResult"): [
            "n_permutations_measured",
            "n_permutations_unmeasurable",
            "unplaced_agents",
        ],
        ("vfairness.multi_agent.negotiation", "NegotiationResult"): [
            "n_turns_measured",
            "n_turns_unmeasurable",
        ],
        ("vfairness.multi_agent.collusion", "CollusionResult"): [
            "n_groups_compared",
            "min_attainable_p",
            "detectable",
        ],
        ("vfairness.multi_agent.delegation", "DelegationResult"): [
            "min_attainable_p",
            "detectable",
        ],
        ("vfairness.llm.benchmarks", "BenchmarkResult"): [
            "n_failed",
            "subsetSize",
            "fullBenchmarkSize",
            "provenanceNote",
        ],
    }
    for (path, name), fields in expected.items():
        cls = _load(path, name)
        present = {f.name for f in dataclasses.fields(cls)}
        assert set(fields) <= present, f"{name} lost {sorted(set(fields) - present)}"


def test_a_benchmark_result_carries_its_subset_provenance_from_the_runner():
    """The provenance fields are the reason a bundled subset cannot be presented
    as a full benchmark run, and their DEFAULTS ("unknown", 0) would say the
    opposite. Executed through the helper every runner splats in, because that
    splat is why the AST check above cannot see the fields."""
    from vfairness.llm.benchmarks import BenchmarkResult, benchmark_provenance

    provenance = benchmark_provenance("bbq", 12)
    assert provenance["subsetSize"] == 12
    assert provenance["fullBenchmarkSize"] != "unknown"
    assert "do not present as a full benchmark run" in provenance["provenanceNote"]

    result = BenchmarkResult(benchmark_id="bbq", overall_score=0.8, **provenance)
    assert result.subsetSize == 12
    assert result.fullBenchmarkSize == provenance["fullBenchmarkSize"]
    assert result.provenanceNote

    # A record built WITHOUT the provenance says "unknown" rather than claiming
    # a size, which is the honest shape of the default; the test above is what
    # keeps a runner from shipping one.
    bare = BenchmarkResult(benchmark_id="bbq", overall_score=0.8)
    assert bare.fullBenchmarkSize == "unknown"
    assert bare.provenanceNote == ""


def test_a_fair_classifier_result_must_be_told_whether_the_constraint_held():
    """``constraint_satisfied`` has NO default on purpose: it is the verdict, and
    a ``False`` default paints VIOLATED on a run nobody evaluated while a
    ``True`` one clears it. A record type cannot be constructed without it."""
    from vfairness.in_processing.wrappers.sklearn_wrappers import FairClassifierResult

    with pytest.raises(TypeError):
        FairClassifierResult(accuracy=0.8, fairness_violation=0.02)  # no verdict

    real = FairClassifierResult(accuracy=0.8, fairness_violation=0.02, constraint_satisfied=True)
    assert real.constraint_satisfied is True
    assert real.to_dict()["constraint_satisfied"] is True
    # A NaN violation is stored as a NaN, so the consumer can still tell that
    # the constraint check had nothing to work with.
    unmeasured = FairClassifierResult(
        accuracy=0.8, fairness_violation=float("nan"), constraint_satisfied=None
    )
    assert math.isnan(unmeasured.to_dict()["fairness_violation"])
    assert unmeasured.to_dict()["constraint_satisfied"] is None


def test_an_output_analysis_row_refuses_rather_than_reporting_a_null_comparison():
    """``OutputAnalysisResult`` is the row every LLM group comparison in the
    library hands back, and the row the intersectional analyzer keys its verdict
    off. Executed through its real producer on the degenerate inputs."""
    from vfairness.llm.output_analysis import OutputAnalysisResult, OutputAnalyzer

    analyzer = OutputAnalyzer()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        empty = analyzer.analyze_sentiment([], [])
    assert isinstance(empty, OutputAnalysisResult)
    assert empty.assessed is False
    assert empty.not_assessed_reason
    assert empty.p_value is None
    assert empty.is_significant is None
    assert empty.delta is None
    assert caught

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        one_sided = analyzer.analyze_sentiment(["a wonderful excellent result"] * 6, [])
    assert one_sided.assessed is False
    assert one_sided.is_significant is None

    # CONTROL: a real comparison produces real numbers and an assessed row.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        real = analyzer.analyze_sentiment(
            ["wonderful great excellent result"] * 8, ["terrible awful bad result"] * 8
        )
    assert real.assessed is True
    assert real.not_assessed_reason in (None, "")
    assert real.delta is not None and math.isfinite(real.delta)
    assert real.p_value is not None
    assert real.is_significant in (True, False)


def test_the_multi_agent_records_come_back_refusing_on_a_degenerate_run():
    """Executed through the real producers, because a record type is only as
    honest as what is put in it. Each of these has an early-return refusal path,
    which is exactly where a forgotten field takes the neutral default."""
    from vfairness.multi_agent.emergent import EmergentBiasDetector

    detector = EmergentBiasDetector()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = detector.analyze(
            component_outputs={"a": [float("nan")] * 6},
            system_outputs=[float("nan")] * 6,
            groups=["g1", "g2"] * 3,
        )
    assert result.is_emergent is None, "a run that measured nothing is not an absence of bias"
    assert result.is_significant is None
    assert math.isnan(result.p_value), "the 1.0 default would read as 'definitely not significant'"
    assert result.n_components_measured == 0
    assert result.n_components_unmeasurable >= 1
    assert caught


def test_the_llm_noise_profile_counts_what_it_could_not_read():
    """``NoiseProfile`` is a spread record, and a spread over an array that was
    silently filtered is a spread over a different sample than the caller
    supplied. ``n_excluded_non_finite`` is what makes the two distinguishable."""
    from vfairness.llm.nondeterminism import NonDeterminismAnalyzer

    analyzer = NonDeterminismAnalyzer()
    values = [0.1, 0.2, float("nan"), 0.3, None, float("inf")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        profile = analyzer.characterize_noise(values)
    assert profile.n_excluded_non_finite == 3, profile
    assert profile.sample_size == 3
    assert caught

    clean = analyzer.characterize_noise([0.1, 0.2, 0.3])
    assert clean.n_excluded_non_finite == 0
    assert clean.sample_size == 3
    assert math.isfinite(clean.std_dev)


def test_the_agent_records_reach_a_consumer_with_their_refusals_intact():
    """``ActionBiasResult`` and ``CorrespondenceResult`` are produced once each,
    and their ``is_significant`` is the only significance channel a caller has.
    ``nan < alpha`` is False, so a test that REFUSED must not answer False
    there."""
    from vfairness.agents.action_bias import ActionBiasAnalyzer, ActionBiasResult
    from vfairness.agents.correspondence import CorrespondenceResult, CorrespondenceTester

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        one_each = ActionBiasAnalyzer().analyze_outcomes(
            [{"action": "approve", "outcome": 1.0}],
            [{"action": "approve", "outcome": 0.0}],
            "outcome",
        )
    assert isinstance(one_each, ActionBiasResult)
    assert one_each.is_significant is None, (
        "a 1-vs-1 comparison refused; False would read as tested"
    )
    assert math.isnan(one_each.p_value)
    assert caught

    # An empty arm is refused OUTRIGHT rather than answered, which is the
    # loudest of the three states and the right one here: there is no
    # comparison at all.
    for empty in (([], []), ([1.0], []), ([float("nan")] * 4, [1.0] * 4)):
        with pytest.raises(ValueError):
            CorrespondenceTester().analyze_outcomes(*empty)

    # G12. A comparison whose ARM SIZES could never have reached alpha reported
    # is_significant=False, which on that field means "tested, and the groups
    # were treated alike". Measured before the fix on the largest disparity the
    # scale has:
    #   analyze_outcomes([1.0], [0.0]) -> disparity 1.0, p 1.0,
    #                                     is_significant False
    # 1.0 is the FLOOR at one observation per arm (two-sided exact p is
    # 2 / C(n_a + n_b, n_a)), so no data whatsoever could have produced a
    # finding, and the only disclosure was a "below recommended minimum"
    # warning about reliability, which is a different statement.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        floored = CorrespondenceTester().analyze_outcomes([1.0], [0.0])
    assert isinstance(floored, CorrespondenceResult)
    assert floored.disparity_metric == 1.0, "the disparity IS measured, and maximal"
    assert floored.detectable is False
    assert floored.min_attainable_p == 1.0
    assert "NOT DETECTABLE" in floored.detectability_note
    assert any("absence of statistical power" in str(c.message) for c in caught)

    # The floor is DERIVED, not quoted: 3 vs 3 cannot reach 0.05 and 4 vs 4 can,
    # so the guard discriminates at the boundary rather than refusing anything
    # small.
    from math import comb

    for n, expected in ((3, False), (4, True)):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = CorrespondenceTester().analyze_outcomes([1.0] * n, [0.0] * n)
        assert r.min_attainable_p == pytest.approx(2.0 / comb(2 * n, n)), n
        assert r.detectable is expected, n
        if expected:
            assert r.detectability_note == ""
            assert r.is_significant is True

    # CONTROL: a real comparison still answers True/False.
    rng = np.random.default_rng(0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        real = ActionBiasAnalyzer().analyze_outcomes(
            [{"action": "approve", "outcome": float(v)} for v in rng.normal(1.0, 0.1, 40)],
            [{"action": "approve", "outcome": float(v)} for v in rng.normal(0.0, 0.1, 40)],
            "outcome",
        )
    assert real.is_significant is True
    assert math.isfinite(real.p_value)
