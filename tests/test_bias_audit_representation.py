"""The bias-audit representation panel must report the benchmark it was given.

WHY THIS FILE EXISTS. ``RepresentationBiasResult`` carries two dicts that are
always read together: ``group_distributions``, keyed by the labels the data
actually uses ("Female", "Non-binary", or a number out of a numeric column), and
``representation_ratios``, which can only be keyed by the folded label the
benchmark join produced ("female"). With plain dicts on both sides that join
MISSES for any dataset whose labels are not already lower-case, which is nearly
all of them.

The panel used to hide the miss behind a ``ratios.get(group, 1.0)`` default, and
a ratio of 1.0 draws the population benchmark EQUAL to the dataset share: a
picture of perfect representation for a group nobody benchmarked. Removing that
default was right, and it turned the fabricated all-clear into a fabricated
absence: the panel printed "no population benchmark was reported for this group"
on all six rows of a report that supplied a benchmark for every one of them,
two inches under a HIGH "Severe underrepresentation in 'gender': ['female']"
computed from those same ratios. One page, two contradictory claims, and the
reasonable reading is the wrong one: an auditor concludes no benchmark data
existed and stops looking for under-representation that is really there.

So both directions are pinned here:

* a benchmark that WAS supplied must reach the canvas as the measured number,
* a benchmark that was NOT supplied must still read as could-not-check, and
  must never be invented from the dataset share.
"""

import warnings

import numpy as np
import pandas as pd

from vfairness.preprocessing.bias_detection import BiasDetector
from vfairness.preprocessing.bias_detection.representation import (
    detect_representation_bias,
)
from vfairness.rendering.adapters import _representation_bars, bias_audit_to_svg

NO_BENCHMARK_SENTENCE = "no population benchmark was reported for this group"


def _audit(df, attributes, benchmarks=None):
    """A real BiasDetector.full_audit, the way the render gallery calls it."""
    with warnings.catch_warnings():
        # The default-benchmark and small-sample warnings are other tests' subject.
        warnings.simplefilter("ignore")
        detector = BiasDetector(
            df,
            protected_attributes=attributes,
            outcome_column="outcome",
            benchmarks=benchmarks,
        )
        return detector.full_audit(
            include_historical=True,
            include_representation=True,
            include_disparities=True,
            include_proxies=True,
        )


def _gallery_frame(seed=42, n=2000):
    """The render gallery's own loan frame: mixed-case labels, as real data is."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "race": rng.choice(
                ["White", "Black", "Hispanic", "Asian"], n, p=[0.6, 0.15, 0.18, 0.07]
            ),
            "gender": rng.choice(["Male", "Female", "Non-binary"], n, p=[0.72, 0.22, 0.06]),
            "outcome": rng.integers(0, 2, n),
        }
    )


GALLERY_BENCHMARKS = {
    "gender": {"Male": 0.49, "Female": 0.51, "Non-binary": 0.03},
    "race": {"White": 0.58, "Black": 0.13, "Hispanic": 0.19, "Asian": 0.06},
}


# ── the join itself, at the source ──────────────────────────────────────────


def test_ratio_answers_to_the_label_group_distributions_uses():
    """The two dicts of one result must join on the label the data spells.

    This is the defect in its smallest form. Every label in
    ``group_distributions`` has a ratio, so asking for it by that label has to
    answer; if it does not, everything downstream reports an absence that the
    same object contradicts one field away.
    """
    df = pd.DataFrame({"gender": ["Male"] * 70 + ["Female"] * 30})
    (result,) = detect_representation_bias(
        df, ["gender"], benchmarks={"gender": {"Male": 0.5, "Female": 0.5}}
    )

    for label in result.group_distributions:
        assert result.representation_ratios.get(label) is not None, (
            f"no ratio answered for {label!r}, which is a label this very result "
            f"reports a distribution for"
        )
        assert label in result.representation_ratios

    assert result.representation_ratios.get("Female") == result.representation_ratios.get("female")
    assert result.representation_ratios.get("Female") == 0.6


def test_stored_keys_stay_normalised_so_serialisation_is_unchanged():
    """The lookup is folded, the STORAGE is not: iteration and to_dict are stable.

    Callers persist ``representation_ratios`` and compare it to fixtures. Making
    lookups case-insensitive must not silently re-key what is written out.
    """
    df = pd.DataFrame({"gender": ["Male"] * 70 + ["Female"] * 30})
    (result,) = detect_representation_bias(
        df, ["gender"], benchmarks={"gender": {"Male": 0.5, "Female": 0.5}}
    )
    assert sorted(result.representation_ratios) == ["female", "male"]
    assert result.to_dict()["representation_ratios"] == {"male": 1.4, "female": 0.6}

    # A bulk insert must fold too. dict.update and dict.__init__ do NOT call
    # __setitem__ on their own, so an unfolded key inserted that way would sit
    # in the mapping where no folded lookup could ever reach it.
    bulk = type(result.representation_ratios)({"Female": 0.6})
    bulk.update({"Non-Binary": 0.2})
    assert sorted(bulk) == ["female", "non-binary"]
    assert bulk.get("Female") == 0.6
    assert bulk.get("Non-Binary") == 0.2


def test_a_group_the_benchmark_never_mentioned_still_has_no_ratio():
    """The fix must not manufacture a ratio, only find the one that exists.

    A case-insensitive lookup that answered for an unbenchmarked group would be
    the fabricated all-clear coming back in through the front door.
    """
    df = pd.DataFrame({"gender": ["Male"] * 60 + ["Female"] * 30 + ["Non-binary"] * 10})
    (result,) = detect_representation_bias(
        df, ["gender"], benchmarks={"gender": {"Male": 0.5, "Female": 0.5}}
    )
    assert result.representation_ratios.get("Non-binary") is None
    assert "Non-binary" not in result.representation_ratios


# ── the panel, end to end, on a real full_audit ─────────────────────────────


def test_panel_reports_every_benchmark_the_report_supplied():
    """Healthy input: six rows, six measured benchmarks, and no absence claimed."""
    report = _audit(_gallery_frame(), ["race", "gender"], GALLERY_BENCHMARKS)
    bars, without = _representation_bars(report)

    assert len(bars) == 6
    assert without == [], f"benchmarks were supplied for every group, got {without}"
    for bar in bars:
        assert bar["benchmark_pct"] is not None, (
            f"{bar['group']!r} lost its benchmark; the report supplied one for it"
        )

    svg = bias_audit_to_svg(report)
    assert NO_BENCHMARK_SENTENCE not in svg
    assert svg.count("% benchmark") == 6

    # The panel must agree with the critical issue above it. Female sits near
    # 22 percent of the data against a benchmark near 50 percent, which is the
    # under-representation the issue table reports at HIGH.
    female = next(b for b in bars if b["group"] == "Female")
    assert female["benchmark_pct"] > 0.45
    assert female["dataset_pct"] < 0.30
    assert female["benchmark_pct"] > 2 * female["dataset_pct"]


def test_panel_still_says_so_when_no_benchmark_was_supplied():
    """The could-not-check state survives, for the case that really is one."""
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame(
        {
            "squad": rng.choice(["Squad Alpha", "Squad Beta", "Squad Gamma"], n),
            "outcome": rng.integers(0, 2, n),
        }
    )
    # "squad" matches no built-in default benchmark either, so nothing at all
    # was reported about this population.
    report = _audit(df, ["squad"], benchmarks=None)
    bars, without = _representation_bars(report)

    assert bars, "the rows themselves are still drawn; it is the benchmark that is absent"
    assert len(without) == len(bars)
    for bar in bars:
        assert bar["benchmark_pct"] is None
        assert bar["dataset_pct"] is not None

    svg = bias_audit_to_svg(report)
    assert svg.count(NO_BENCHMARK_SENTENCE) == len(bars)
    assert "% benchmark" not in svg


def test_an_absent_benchmark_is_never_drawn_equal_to_the_dataset_share():
    """The original fabrication, pinned by its signature.

    ``benchmark_pct == dataset_pct`` is what a defaulted ratio of 1.0 produces,
    and it renders as two bars of identical length under a "Dataset / Population
    benchmark" legend.
    """
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame(
        {
            "squad": rng.choice(["Squad Alpha", "Squad Beta"], n),
            "outcome": rng.integers(0, 2, n),
        }
    )
    bars, _ = _representation_bars(_audit(df, ["squad"], benchmarks=None))
    for bar in bars:
        assert bar["benchmark_pct"] != bar["dataset_pct"]


def test_mixed_rows_grade_only_the_groups_that_were_benchmarked():
    """One panel, both states, each row on its own merits."""
    rng = np.random.default_rng(7)
    n = 600
    df = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female", "Non-binary"], n, p=[0.6, 0.34, 0.06]),
            "outcome": rng.integers(0, 2, n),
        }
    )
    report = _audit(df, ["gender"], {"gender": {"Male": 0.49, "Female": 0.51}})
    bars, without = _representation_bars(report)

    by_group = {b["group"]: b for b in bars}
    assert by_group["Male"]["benchmark_pct"] is not None
    assert by_group["Female"]["benchmark_pct"] is not None
    assert by_group["Non-binary"]["benchmark_pct"] is None
    assert without == ["Non-binary"]

    svg = bias_audit_to_svg(report)
    assert svg.count(NO_BENCHMARK_SENTENCE) == 1
    assert "1 of 3 groups had no population benchmark" in svg


# ── the count on the canvas counts the canvas ───────────────────────────────


def test_ungraded_count_counts_the_rows_the_canvas_actually_draws():
    """A count printed on a canvas has to be a count OF that canvas.

    The bar list and the ungraded-group list used to be capped at six
    INDEPENDENTLY, so a report with more groups than fit could print "6 of 6
    groups had no population benchmark" across six rows that visibly carry one.
    """
    rng = np.random.default_rng(5)
    n = 900
    regions = ["North", "South", "East", "West", "Central", "Coastal"]
    df = pd.DataFrame(
        {
            "region": rng.choice(regions, n),
            "squad": rng.choice(["Squad Alpha", "Squad Beta", "Squad Gamma"], n),
            "outcome": rng.integers(0, 2, n),
        }
    )
    # region fills all six drawn rows and is fully benchmarked; squad is not
    # benchmarked at all and falls entirely outside the six.
    report = _audit(
        df,
        ["region", "squad"],
        {"region": {r: 1 / len(regions) for r in regions}},
    )
    bars, without = _representation_bars(report)

    assert len(bars) == 6
    assert all(b["attribute"] == "region" for b in bars), "squad is off the canvas"
    assert without == [], "every row DRAWN carries a benchmark, so the canvas must claim no absence"

    svg = bias_audit_to_svg(report)
    assert "had no population benchmark" not in svg
    assert NO_BENCHMARK_SENTENCE not in svg


def test_the_header_count_matches_the_rows_that_say_it():
    """The header sentence and the per-row sentences report the same number."""
    rng = np.random.default_rng(9)
    n = 800
    df = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female", "Non-binary"], n, p=[0.55, 0.35, 0.10]),
            "outcome": rng.integers(0, 2, n),
        }
    )
    report = _audit(df, ["gender"], {"gender": {"Male": 0.49, "Female": 0.51}})
    bars, without = _representation_bars(report)
    svg = bias_audit_to_svg(report)

    assert f"{len(without)} of {len(bars)} groups had no population benchmark" in svg
    assert svg.count(NO_BENCHMARK_SENTENCE) == len(without)


def test_render_never_crashes_and_never_returns_empty():
    """Both directions still produce a canvas."""
    rng = np.random.default_rng(13)
    n = 300
    df = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], n),
            "outcome": rng.integers(0, 2, n),
        }
    )
    for benchmarks in ({"gender": {"Male": 0.5, "Female": 0.5}}, None):
        svg = bias_audit_to_svg(_audit(df, ["gender"], benchmarks))
        assert svg.strip().startswith("<svg")
        assert len(svg) > 1000
