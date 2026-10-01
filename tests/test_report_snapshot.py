"""Structural snapshot of the FairnessAnalyzer report (VB-TEST-10).

The point of this test is a tripwire: if the *shape* of the report dict
changes (a top-level section appears or vanishes, a metric key is renamed, a
per-group stat is added or dropped), this test fails loudly so the change is a
deliberate, reviewed decision rather than a silent break of every downstream
consumer of the report.

It is a STRUCTURAL snapshot, not a value snapshot. We deliberately do not pin
metric values, because those depend on numerics and would make the test a
brittle golden-value test (that job belongs to test_golden_metrics.py). We
pin:

  * the sorted top-level keys,
  * the sorted keys of each nested section,
  * the per-group stat key set,
  * the metric names present.

Normalization applied before comparison:

  * dicts keyed by concrete group labels (``group_sizes``, ``group_stats``)
    are collapsed to a single canonical ``"<group>"`` key so the snapshot
    does not depend on group naming or ordering, only on the per-group shape,
  * any volatile field that a future report version might add
    (timestamps, versions, run ids) is stripped, so the snapshot stays about
    structure. Today's report contains none of these; the guard is here so the
    snapshot does not have to be regenerated the day one is introduced.

The fixture is fully seeded, so the report is reproducible run to run.
"""

import numpy as np
import pytest

from vfairness import FairnessAnalyzer

# Keys whose *value* is volatile across runs/builds and must never enter the
# structural snapshot even if a future report version adds them. Matched
# case-insensitively as whole keys.
_VOLATILE_KEYS = {
    "timestamp",
    "generated_at",
    "created_at",
    "date",
    "datetime",
    "version",
    "vfairness_version",
    "run_id",
    "report_id",
    "uuid",
}

# Sections whose keys are concrete group labels; collapse them to one canonical
# entry so the snapshot pins per-group shape without pinning group names.
_GROUP_KEYED_SECTIONS = {"group_sizes", "group_stats"}
_CANONICAL_GROUP_KEY = "<group>"


def _skeleton(obj, key=None):
    """Reduce a report to a structural skeleton.

    Dicts become ``{sorted key: skeleton}``; lists become ``["list", elem
    skeleton]`` using the first element (report lists are homogeneous);
    scalars become the literal string ``"scalar"``. Volatile keys are dropped
    and group-keyed sections are collapsed to a single canonical key.
    """
    if isinstance(obj, dict):
        out = {}
        collapse = key in _GROUP_KEYED_SECTIONS
        for k in sorted(obj, key=str):
            if str(k).lower() in _VOLATILE_KEYS:
                continue
            skel = _skeleton(obj[k], key=k)
            if collapse:
                # All group entries share a shape; keep exactly one.
                out[_CANONICAL_GROUP_KEY] = skel
            else:
                out[str(k)] = skel
        return out
    if isinstance(obj, (list, tuple)):
        return ["list", _skeleton(obj[0])] if obj else ["list"]
    return "scalar"


# Pinned structural snapshot for the classification report (include_ci=False).
# If the report shape legitimately changes, update this dict IN THE SAME change
# that alters the report, and note it in CHANGELOG.md.
EXPECTED_CLASSIFICATION_SKELETON = {
    "assessment": {
        "assessable": "scalar",
        "failed_metrics": ["list"],
        "fairness_score": "scalar",
        # Groups excluded from the verdict (n < min_group_size) are surfaced
        # here as explicit insufficient-evidence records; empty when every
        # group clears the size gate, as in this seeded fixture.
        "insufficient_evidence_groups": ["list"],
        # Metrics that could not be computed (NaN, e.g. undefined TPR) are
        # surfaced here and excluded from the score; empty in this seeded fixture.
        "not_assessable_metrics": ["list"],
        "passed_metrics": [
            "list",
            {
                "metric": "scalar",
                "status": "scalar",
                "threshold": "scalar",
                "value": "scalar",
            },
        ],
        "summary": "scalar",
    },
    "data_info": {
        # Added 2026-09-27 by the BGL3 campaign. validate_inputs now discloses what
        # SHARE of the input every number rests on, and which groups lost every row,
        # because dropping a whole group moved demographic_parity_ratio from 0.000 to
        # 0.895 on the same predictions and n_excluded alone did not say so. Purely
        # additive: invalid_groups, valid_groups and group_sizes are unchanged.
        "coverage": "scalar",
        "groups_dropped": ["list"],
        "n_groups_after": "scalar",
        "n_groups_before": "scalar",
        "final_size": "scalar",
        "group_sizes": {"<group>": "scalar"},
        "invalid_groups": ["list"],
        "is_intersectional": "scalar",
        "missing_strategy": "scalar",
        "n_excluded": "scalar",
        "n_groups": "scalar",
        "n_samples": "scalar",
        "original_size": "scalar",
        "valid_groups": ["list", "scalar"],
    },
    "group_stats": {
        "<group>": {
            "accuracy": "scalar",
            "base_rate": "scalar",
            "fpr": "scalar",
            "positive_rate": "scalar",
            "precision": "scalar",
            "size": "scalar",
            "tpr": "scalar",
        }
    },
    "metrics": {
        "demographic_parity_difference": "scalar",
        "demographic_parity_ratio": "scalar",
        "equal_opportunity_difference": "scalar",
        "equalized_odds_difference": "scalar",
        "predictive_parity_difference": "scalar",
    },
    "methodology_version": "scalar",
    "task_type": "scalar",
    "thresholds_used": {
        "auroc_parity": "scalar",
        "calibration_difference": "scalar",
        "demographic_parity_difference": "scalar",
        "demographic_parity_ratio": "scalar",
        "equal_opportunity_difference": "scalar",
        "equalized_odds_difference": "scalar",
        "integrated_calibration_index": "scalar",
        "multicalibration": "scalar",
        "predictive_parity_difference": "scalar",
    },
}


@pytest.fixture
def classification_analyzer():
    """Deterministic classification analyzer over a fixed, seeded fixture."""
    rng = np.random.default_rng(1234)
    n = 400
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    gender = rng.choice(["female", "male"], n)
    return FairnessAnalyzer(y_true, y_pred, gender)


def test_report_top_level_sections_are_stable(classification_analyzer):
    """The set of top-level report sections must not drift silently."""
    report = classification_analyzer.get_report(include_ci=False)
    assert sorted(report.keys()) == sorted(EXPECTED_CLASSIFICATION_SKELETON.keys())


def test_report_structure_matches_snapshot(classification_analyzer):
    """Full structural skeleton matches the pinned classification snapshot."""
    report = classification_analyzer.get_report(include_ci=False)
    skeleton = _skeleton(report)
    assert skeleton == EXPECTED_CLASSIFICATION_SKELETON, (
        "FairnessAnalyzer report structure changed. If this is intentional, "
        "update EXPECTED_CLASSIFICATION_SKELETON and note it in CHANGELOG.md."
    )


def test_snapshot_is_reproducible(classification_analyzer):
    """Two reports from the same seeded fixture share one structure."""
    first = _skeleton(classification_analyzer.get_report(include_ci=False))
    second = _skeleton(classification_analyzer.get_report(include_ci=False))
    assert first == second


def test_metric_names_present(classification_analyzer):
    """The core classification metric names are all present in the report."""
    report = classification_analyzer.get_report(include_ci=False)
    expected_metrics = set(EXPECTED_CLASSIFICATION_SKELETON["metrics"].keys())
    assert set(report["metrics"].keys()) == expected_metrics


def test_volatile_keys_are_stripped():
    """The skeleton reducer drops timestamp/version/run-id style keys."""
    noisy = {
        "task_type": "classification",
        "generated_at": "2026-08-10T00:00:00Z",
        "version": "0.0.9",
        "run_id": "abc-123",
        "metrics": {"demographic_parity_difference": 0.1},
    }
    skeleton = _skeleton(noisy)
    assert "generated_at" not in skeleton
    assert "version" not in skeleton
    assert "run_id" not in skeleton
    assert skeleton["task_type"] == "scalar"
    assert skeleton["metrics"] == {"demographic_parity_difference": "scalar"}
