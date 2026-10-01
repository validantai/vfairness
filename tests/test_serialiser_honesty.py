"""Every serialiser on the public surface emits what its result declares.

WHY THIS IS A GATE AND NOT A ONE-OFF SCRIPT. A serialiser is the boundary where a
result stops being an object and becomes the thing a consumer reads: a dashboard, a
stored row, a report. Two defects live there and nothing else in the suite looks for
them:

  1. A FIELD-BY-FIELD REBUILD DROPS EVERY LATER FIELD. A ``to_dict`` that lists
     sixteen keys by hand does not fail when a seventeenth field is added; it just
     stops carrying it. When the dropped field is a could-not-check disclosure, the
     consumer sees the measurement and never the caveat that qualifies it, so the
     defect the field was added to fix is back at the only layer a reader sees.
  2. A NON-FINITE VALUE BECOMES A NUMBER ON THE WAY OUT. A NaN serialised as 0.0
     turns "this could not be measured" into "this measured perfectly".

Found by running it: ``CalibrationRecommendation.to_dict`` dropped
``not_assessed_groups``, the field added for H-06 so a reader could tell "no
disparity found" from "the disparity does not cover these groups". The object was
fixed and the serialiser was not, so anything working from ``to_dict`` saw a
recommendation with no trace of the groups it does not cover. No test caught it,
because every test asserted on the object.

WHAT THE TWO CHECKS CANNOT DO, stated so silence is not read as coverage. The
field/key check reads the AST of the serialiser, so a key computed at runtime
(``out[name] = ...`` with a variable name) is invisible to it; those serialisers
build their dict wholesale and are recognised as such rather than being asserted
about. The NaN check must CONSTRUCT an instance, and roughly a third of these
results cannot be built from type annotations alone; that is reported as a
could-not-check, is never counted as a pass, and is held at a floor below.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import pathlib
import sys
import textwrap
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


# A serialiser may legitimately leave a declared field out of its output, and each
# case below was adjudicated by reading it. The REASON is the point: every entry
# names what carries the disclosure instead, because "it is bulky" alone is not a
# reason to drop a field that is the only evidence a reader has.
#
# An entry here is not a hole. test_no_adjudication_is_stale fails when a named
# serialiser stops dropping its field, so a fix removes the exemption with it and a
# NEW drop can never hide behind an old one.
ADJUDICATED: dict[str, dict[str, str]] = {
    "vfairness.evaluation.vfairness_metrics.robustness.PermutationTestResult.to_dict": {
        "null_distribution": (
            "Raw array of every permuted statistic. Summarised into null_mean and "
            "null_std, and the emptied-null case emits NaN for both rather than a "
            "mean of nothing, so a refusal still reads as a refusal."
        )
    },
    "vfairness.evaluation.vfairness_metrics.robustness.SensitivityResult.to_dict": {
        "perturbed_metrics": (
            "Raw per-draw array. The disclosure is carried by n_iterations_run "
            "beside n_unmeasurable, which is what the array would be inspected "
            "for: how many draws produced no usable metric."
        )
    },
    "vfairness.post_processing.calibration.metrics.CalibrationMetricResult.to_dict": {
        "bin_details": (
            "Per-bin numpy arrays, which are not JSON. The disclosure is carried by "
            "n_groups_compared beside max_group_disparity and "
            "measured_subset_disparity, so a NaN disparity can be told apart from a "
            "measured parity of 0.0 without the bins."
        )
    },
    "vfairness.post_processing.calibration.tradeoffs.TradeoffAnalysisResult.to_dict": {
        "metrics_at_thresholds": (
            "Per-threshold detail. pareto_points carries the same curve in the "
            "emitted dict, and tradeoff_severity carries the three-state reading "
            "including 'not assessed'."
        )
    },
    "vfairness.in_processing.constraints.reductions.ReductionResult.to_dict": {
        "classifiers": (
            "Trained estimators, which are not JSON and whose absence the docstring "
            "states. n_classifiers is emitted, which is the part a consumer can act "
            "on: an ensemble of zero would otherwise be invisible."
        )
    },
    "vfairness.operations.monitoring.alerts.AlertPayload.to_dict": {
        "drift_event": (
            "The raw event that triggered the alert, i.e. this payload's INPUT. The "
            "reading derived from it is emitted in full: metric_name, drift_score, "
            "mean_shift, affected_groups, severity and priority_score."
        )
    },
    "vfairness.operations.pulse.causal_skeleton.CausalSkeleton.to_dict": {
        "gml": (
            "A text serialisation of the same graph that nodes, edges and paths "
            "carry structurally. identified beside unassessed is the three-state "
            "disclosure, and both are emitted."
        )
    },
    "vfairness.operations.reporting.reports.GeneratedReport.to_dict": {
        "content": (
            "The fully rendered document body, which is the artifact save() writes "
            "rather than metadata about it. sections carries the same content "
            "structurally."
        )
    },
    "vfairness.xai.schemas.FairnessDecomposition.to_db_row": {
        "id": "Primary key assigned by the database on insert, so a row that "
        "carried one would be asserting an identity it cannot know."
    },
}

# Anti-vacuity. The census walks importable modules, so a missing optional extra
# shrinks it silently and every assertion below would pass over a smaller surface
# than it claims. These floors are the observed counts less a little slack; they
# fail loudly rather than letting the gate quietly stop covering things.
MIN_SERIALISERS = 70
MIN_IN_SCOPE = 55
MIN_CONSTRUCTED = 35


def _serialisers():
    """Every public to_*/as_* whose owner is a dataclass, resolved from the census."""
    warnings.filterwarnings("ignore")
    import library_kpis as K

    K._walk_public_surface()
    found = []
    for qual, entry in K._OBJECTS.items():
        name = qual.split(".")[-1]
        if not (name.startswith("to_") or name.startswith("as_")):
            continue
        fn = entry[1]
        if not callable(fn):
            continue
        owner_entry = K._OBJECTS.get(".".join(qual.split(".")[:-1]))
        owner = owner_entry[1] if owner_entry else None
        if owner is None or not dataclasses.is_dataclass(owner):
            continue
        found.append((qual, owner, fn))
    return sorted(found)


def _emitted_keys(fn) -> tuple[set[str], bool]:
    """String keys the serialiser puts in its output, and whether it parsed.

    dedent, NOT inspect.cleandoc. cleandoc is for docstrings; on a method's source
    it leaves the class indentation in place, ast.parse raises, and an earlier
    version swallowed that exception and reported the unparsed serialiser as
    dropping every field it declares. Fifty-three of them, in the tool built to
    catch a could-not-check reported as a finding. Parse failure is its own state.
    """
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except Exception:
        return set(), False
    try:
        tree = ast.parse(src)
    except Exception:
        return set(), False
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys |= {
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
        # Keys inserted after the literal: `out["error"] = ...`, which is how a
        # serialiser omits a None instead of emitting a null. TaskResult does
        # exactly that, and a literal-only rule called it a dropped field.
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                keys.add(node.slice.value)
    return keys, True


def _builds_wholesale(fn) -> bool:
    """asdict / ** / vars build the mapping from the fields themselves, so there is
    no hand-written list to fall behind."""
    try:
        src = inspect.getsource(fn)
    except Exception:
        return False
    return ("asdict" in src) or ("**" in src) or ("vars(" in src)


def _returns_a_mapping_it_built(fn) -> bool:
    """Whether the field-to-key rule applies to this serialiser at all.

    IT APPLIES ONLY TO A SERIALISER THAT BUILDS A MAPPING IN ITS OWN BODY, and that
    boundary is drawn structurally rather than by a list of exceptions. Without it
    the rule reported nine findings that were never findings: to_json, to_svg and
    to_dataframe return a string, an SVG document or a frame, and they DELEGATE,
    usually to the to_dict this file already checks. Comparing a declared field
    against the keys of a mapping that does not exist flags every field at once,
    which is the signature of a rule applied outside its scope rather than of a
    defect. Nine permission slips would have worked and would have left the rule
    still wrong, so the next serialiser of that shape would need a tenth.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:
        return False
    built_here = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if isinstance(node.value, ast.Dict):
            return True
        if isinstance(node.value, ast.Name) and node.value.id in built_here:
            return True
    return False


def _drops(qual, owner, fn) -> tuple[list[str], bool, bool]:
    keys, parsed = _emitted_keys(fn)
    in_scope = parsed and _returns_a_mapping_it_built(fn)
    if not in_scope or _builds_wholesale(fn):
        return [], parsed, in_scope
    declared = [f.name for f in dataclasses.fields(owner)]
    return [d for d in declared if d not in keys], True, True


@pytest.fixture(scope="module")
def audit():
    rows = []
    for qual, owner, fn in _serialisers():
        missing, parsed, in_scope = _drops(qual, owner, fn)
        rows.append(
            {
                "qual": qual,
                "owner": owner,
                "fn": fn,
                "missing": missing,
                "parsed": parsed,
                "in_scope": in_scope,
            }
        )
    return rows


def test_the_field_rule_still_reaches_the_serialisers_it_is_meant_to(audit):
    """The scope narrowing above is load-bearing, so it needs a floor under it.

    A bug in _returns_a_mapping_it_built that put everything out of scope would
    make every assertion in this file pass while checking nothing at all, and the
    output would be indistinguishable from a clean run.
    """
    in_scope = [r["qual"] for r in audit if r["in_scope"]]
    assert len(in_scope) >= MIN_IN_SCOPE, (
        f"only {len(in_scope)} of {len(audit)} serialisers were judged to build a "
        f"mapping, below the floor of {MIN_IN_SCOPE}. The scope test has gone wrong "
        f"and the field-to-key rule is no longer looking at most of the surface."
    )


def test_the_census_still_reaches_the_whole_serialiser_surface(audit):
    assert len(audit) >= MIN_SERIALISERS, (
        f"only {len(audit)} serialisers were found, below the floor of "
        f"{MIN_SERIALISERS}. An optional extra is probably missing, in which case "
        f"every assertion in this file is passing over a smaller surface than it "
        f"claims to cover."
    )


def test_no_serialiser_could_not_be_read(audit):
    """A serialiser whose source will not parse is a could-not-check, and it is not
    allowed to sit quietly among the passes."""
    unreadable = [r["qual"] for r in audit if not r["parsed"]]
    assert not unreadable, (
        f"these serialisers could not be parsed, so nothing here checked them: {unreadable}"
    )


def test_every_serialiser_emits_every_field_its_result_declares(audit):
    findings = []
    for row in audit:
        if not row["in_scope"]:
            continue
        allowed = ADJUDICATED.get(row["qual"], {})
        unexplained = [f for f in row["missing"] if f not in allowed]
        if unexplained:
            findings.append(f"{row['qual']} drops {unexplained}")
    assert not findings, (
        "these serialisers declare a field and do not emit it, so a consumer "
        "reading the output cannot see it:\n  "
        + "\n  ".join(findings)
        + "\n\nIf the omission is correct, add it to ADJUDICATED with the reason and "
        "name what carries the disclosure instead. If it is not, the field was "
        "added after the key list was written and this is the defect."
    )


def test_no_adjudication_is_stale(audit):
    """An exemption that no longer applies is how a real drop gets dismissed.

    When a serialiser is fixed, or a field renamed or removed, its entry here must
    go with it. Otherwise the file accumulates permission slips and the next reader
    cannot tell which ones were ever examined.
    """
    by_qual = {r["qual"]: r for r in audit}
    stale = []
    for qual, allowed in ADJUDICATED.items():
        row = by_qual.get(qual)
        if row is None:
            stale.append(f"{qual} is no longer a serialiser on the surface")
            continue
        if not row["in_scope"]:
            stale.append(f"{qual} no longer builds a mapping, so the rule does not reach it")
            continue
        for field_name in allowed:
            if field_name not in row["missing"]:
                stale.append(f"{qual} now emits {field_name!r}, so the exemption is spent")
    assert not stale, "ADJUDICATED has entries that no longer apply:\n  " + "\n  ".join(stale)


def _nan_kwargs(owner):
    kwargs = {}
    for f in dataclasses.fields(owner):
        t = str(f.type)
        if "float" in t:
            kwargs[f.name] = float("nan")
        elif "str" in t:
            kwargs[f.name] = ""
        elif "List" in t or "list" in t:
            kwargs[f.name] = []
        elif "Dict" in t or "dict" in t:
            kwargs[f.name] = {}
        else:
            kwargs[f.name] = None
    return kwargs


@pytest.fixture(scope="module")
def nan_probe(audit):
    """Build each result with NaN in every float field and see what comes out.

    Roughly a third cannot be constructed from annotations alone. That is recorded
    as a could-not-check and never as a pass.
    """
    constructed, zeroed = [], []
    for row in audit:
        owner, qual = row["owner"], row["qual"]
        try:
            inst = owner(**_nan_kwargs(owner))
            payload = getattr(inst, qual.split(".")[-1])()
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        constructed.append(qual)
        for key, value in payload.items():
            if isinstance(value, float) and not math.isnan(value) and value == 0.0:
                zeroed.append(f"{qual} -> {key!r} came out 0.0 from a NaN")
    return constructed, zeroed


def test_a_non_finite_value_is_not_zeroed_on_the_way_out(nan_probe):
    constructed, zeroed = nan_probe
    assert not zeroed, (
        "a NaN went in and a 0.0 came out, which turns a value that could not be "
        "measured into one that measured perfectly:\n  " + "\n  ".join(zeroed)
    )


def test_the_nan_probe_actually_constructed_enough_to_mean_something(nan_probe):
    """Without this, every result failing to construct reads as a clean pass."""
    constructed, _ = nan_probe
    assert len(constructed) >= MIN_CONSTRUCTED, (
        f"the NaN probe only built {len(constructed)} results, below the floor of "
        f"{MIN_CONSTRUCTED}, so it is reporting silence as safety"
    )
