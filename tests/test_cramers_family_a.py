"""Cramer family A: the association statistics behind the exported proxy screen.

The defect: a statistic that is MATHEMATICALLY UNDEFINED for the table it was
handed was published as 0.0, and 0.0 is the strongest reassurance on this scale.
Measured on the tree before the repair (2026-09-17), 200 rows, a random m/f
``gender`` and a ``customer_ref`` holding one unique id per row, so
``customer_ref`` determines ``gender`` perfectly in sample::

    vfairness.identify_proxy_features(df, protected_attr="gender")  ->  []
        warnings: NONE
    association_strength(df.customer_ref, df.gender)                ->  0.0
        warnings: NONE
    discovery._cramers_v(df.customer_ref, df.gender)                ->  0.0

The last line of ``_cramers_v`` read
``return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0``. With one row
per level the Bergsma (2013) correction consumes the whole table, ``denom`` is
0, and a pair nobody could measure was published as a pair with no association
at all.

Four more fabricated values in the same family, each reproduced at the public
entry before it was changed, and each pinned below:

  * a DEGENERATE MARGIN in ``_cramers_v`` (one level on a side). 50 rows holding
    a single gender returned ``[]`` in silence: a clean bill of health for a
    frame in which the proxy question cannot be asked.
  * ONE GROUP in ``_correlation_ratio``. eta is 0.0 by arithmetic when there is
    nothing to compare against, and a single-group frame with a numeric salary
    likewise returned ``[]`` in silence.
  * EVERY LEVEL A SINGLETON in ``_correlation_ratio``, the same defect running
    BACKWARDS: eta is then 1.0 by arithmetic, and a unique-per-row
    ``customer_ref`` was published as a ``risk_level: high`` proxy for ``age``
    scoring 1.0000000000000002, outside the documented [0, 1] range, with no
    warning at all.
  * ``find_proxy_chains`` in bias_detection/proxy.py compared each link with
    ``corr >= correlation_threshold``. ``nan >= 0.3`` is False, so every
    unmeasurable link left the scan silently and the function answered ``[]``.

The control tests are as load bearing as the refusals: a screen that refuses
everything passes every degenerate test and finds nothing real. Every expected
value below is computed in the test from the formula, never copied from what
the code returned.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import vfairness
from vfairness.evaluation.vfairness_metrics.discovery import (
    ProxyScanIncompleteWarning,
    _correlation_ratio,
    _cramers_v,
    association_strength,
    identify_proxy_features,
)
from vfairness.preprocessing.bias_detection import proxy as proxy_mod

N = 200


def _caught(fn):
    """Run `fn` with every warning recorded. A refusal carried only in a
    warning is invisible to a test that suppresses warnings."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, rec


def _messages(rec) -> str:
    return " || ".join(str(r.message) for r in rec)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def unique_id_frame(rng):
    """The frame from the reproduction: customer_ref determines gender."""
    return pd.DataFrame(
        {
            "gender": rng.choice(["m", "f"], N),
            "customer_ref": [f"REF{i:05d}" for i in range(N)],
        }
    )


# --- SITE 1: _cramers_v, the three tables it cannot measure ------------------


def test_the_bergsma_correction_consuming_the_table_is_not_no_association(
    unique_id_frame,
):
    """denom <= 0. The headline defect, at the helper."""
    value, rec = _caught(
        lambda: _cramers_v(unique_id_frame["customer_ref"], unique_id_frame["gender"])
    )
    assert math.isnan(value), f"a table the correction consumes must be NaN, got {value!r}"
    assert "COULD NOT BE MEASURED" in _messages(rec)
    assert "200x2" in _messages(rec), "the warning must name the table it refused"


def test_a_single_level_side_is_not_no_association():
    """A degenerate margin: min(r, k) - 1 is 0, so V is 0/0."""
    feature = pd.Series(["A", "B"] * 25, name="salary_band")
    single_group = pd.Series(["f"] * 50, name="gender")
    value, rec = _caught(lambda: _cramers_v(feature, single_group))
    assert math.isnan(value)
    assert "NOT DEFINED" in _messages(rec)
    assert "2x1" in _messages(rec)


def test_an_empty_table_is_not_no_association():
    """Nothing was observed, so nothing was measured. Not reachable from
    `association_strength`, which refuses under ten aligned rows first, so it is
    pinned at the helper."""
    value, rec = _caught(
        lambda: _cramers_v(pd.Series([], dtype=object), pd.Series([], dtype=object))
    )
    assert math.isnan(value)
    assert "empty" in _messages(rec)


def test_a_measurable_table_is_still_measured_exactly(rng):
    """CONTROL for the three refusals above, and the one that decides whether
    they are worth anything. The expected value is computed here from the
    Bergsma (2013) definition against scipy's chi-square, not read back from
    the implementation."""
    race = pd.Series(rng.choice(["W", "B", "A"], 600), name="race")
    # An imperfect association: zip tracks race 85% of the time.
    draw = rng.random(600)
    zipc = pd.Series(
        np.where(
            draw < 0.85,
            race.map({"W": "Z1", "B": "Z2", "A": "Z3"}),
            race.map({"W": "Z2", "B": "Z3", "A": "Z1"}),
        ),
        name="zipc",
    )
    ct = pd.crosstab(zipc, race)
    chi2 = float(stats.chi2_contingency(ct, correction=False)[0])
    n = int(ct.values.sum())
    r, k = ct.shape
    phi2corr = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    expected = math.sqrt(phi2corr / (min(rcorr, kcorr) - 1))

    value, rec = _caught(lambda: _cramers_v(zipc, race))
    assert value == pytest.approx(expected, abs=1e-9)
    assert 0.3 < value < 1.0, "precondition: this is a real, findable association"
    assert not rec, f"a measurable table must not warn: {_messages(rec)}"


# --- SITE 2a: association_strength, the one source of truth ------------------


def test_association_strength_refuses_the_table_it_cannot_measure(unique_id_frame):
    value, rec = _caught(
        lambda: association_strength(unique_id_frame["customer_ref"], unique_id_frame["gender"])
    )
    assert math.isnan(value), "0.0 here is a clean bill of health for a perfect proxy"
    assert rec, "the refusal must be visible without reading the source"


def test_association_strength_refuses_a_single_group_numeric_pair():
    """The mixed path. eta is 0.0 by arithmetic when the grouping has one
    level, which is not a measurement of independence."""
    salary = pd.Series(np.linspace(30000, 90000, 50), name="salary")
    gender = pd.Series(["f"] * 50, name="gender")
    value, rec = _caught(lambda: association_strength(salary, gender))
    assert math.isnan(value)
    assert "says nothing about the data" in _messages(rec)


def test_association_strength_refuses_a_zero_variance_numeric_pair():
    """Pearson is 0/0 when a side never varies. `identify_proxy_features`
    already called this "Pearson undefined: zero variance" and refused it,
    while this function returned 0.0 for the same input."""
    constant = pd.Series([7.0] * 50, name="tenure_days")
    varying = pd.Series(np.arange(50.0), name="age")
    value, rec = _caught(lambda: association_strength(constant, varying))
    assert math.isnan(value)
    assert "zero variance" in _messages(rec)


def test_association_strength_still_measures_a_real_association_exactly(rng):
    """CONTROL. eta computed here from the sum-of-squares definition."""
    groups = pd.Series(["W", "B", "A"] * 100, name="race")
    values = pd.Series(
        groups.map({"W": 750.0, "B": 580.0, "A": 700.0}) + rng.normal(0, 25, 300),
        name="income",
    )
    grand = values.to_numpy().mean()
    ss_total = float(((values.to_numpy() - grand) ** 2).sum())
    ss_between = 0.0
    for level in groups.unique():
        grp = values.to_numpy()[groups.to_numpy() == level]
        ss_between += len(grp) * (grp.mean() - grand) ** 2
    expected = math.sqrt(ss_between / ss_total)

    value, rec = _caught(lambda: association_strength(values, groups))
    assert value == pytest.approx(expected, abs=1e-9)
    assert 0.5 < value < 1.0, "precondition: a real association, not a degenerate one"
    assert not rec, f"a measurable pair must not warn: {_messages(rec)}"


# --- SITE 2b: identify_proxy_features, the exported entry --------------------


def test_the_exported_scan_does_not_report_an_unmeasurable_column_as_clean(
    unique_id_frame,
):
    """THE PUBLIC ENTRY. `vfairness.identify_proxy_features` answered `[]` with
    no warning for a frame holding a perfect proxy."""
    result, rec = _caught(
        lambda: vfairness.identify_proxy_features(unique_id_frame, protected_attr="gender")
    )
    assert [p["column"] for p in result] == [], "precondition: it still cannot measure it"
    assert result.not_assessable, "the empty list must not be readable as 'no proxies exist'"
    assert "customer_ref" in " ".join(result.not_assessable)
    assert "could NOT be assessed" in _messages(rec)


def test_a_single_group_protected_attribute_is_a_could_not_check():
    """50 rows, one gender. Nothing about proxy risk can be established from a
    sample with no variation in the attribute being screened against."""
    df = pd.DataFrame({"gender": ["f"] * 50, "salary_band": ["A", "B"] * 25})
    result, rec = _caught(lambda: vfairness.identify_proxy_features(df, protected_attr="gender"))
    assert list(result) == []
    assert result.not_assessable, "a single-group frame must not read as screened and clean"
    assert "salary_band" in " ".join(result.not_assessable)
    assert rec


def test_a_unique_identifier_is_not_published_as_a_perfect_proxy(rng):
    """The defect running BACKWARDS, and the more dangerous direction for a
    reader: eta is 1.0 by arithmetic when every level holds one row, so a
    customer reference scored `risk_level: high` against age, at
    1.0000000000000002, outside the documented [0, 1] range."""
    df = pd.DataFrame(
        {
            "age": rng.normal(45, 10, 60),
            "customer_ref": [f"REF{i}" for i in range(60)],
        }
    )
    result, rec = _caught(lambda: vfairness.identify_proxy_features(df, protected_attr="age"))
    assert [p["column"] for p in result] == [], "an arithmetic 1.0 is not a proxy finding"
    assert "customer_ref" in " ".join(result.not_assessable)
    assert "1.0 by arithmetic" in _messages(rec)


def test_the_scan_still_finds_a_real_proxy_and_stays_silent(rng):
    """CONTROL at the public entry. The fix must not turn the screen into a
    machine that refuses everything: a genuine proxy is still found, graded,
    and measured to its exact value, with no warning raised."""
    race = pd.Series(rng.choice(["W", "B", "A"], 300), name="race")
    df = pd.DataFrame(
        {
            "race": race,
            "zipc": race.map({"W": "Z1", "B": "Z2", "A": "Z3"}),
            "noise": rng.normal(size=300),
        }
    )
    result, rec = _caught(lambda: vfairness.identify_proxy_features(df, protected_attr="race"))
    assert [p["column"] for p in result] == ["zipc"]
    assert result[0]["risk_level"] == "high"
    # A one-to-one mapping makes the bias-corrected V exactly 1: phi2corr and
    # the corrected denominator are the same number.
    assert result[0]["abs_correlation"] == pytest.approx(1.0, abs=1e-9)
    assert list(result.not_assessable) == []
    assert not rec, f"a fully assessable scan must stay silent: {_messages(rec)}"


def test_the_scan_reports_the_proxy_it_could_measure_beside_the_one_it_could_not(rng):
    """A refusal must never swallow the finding next to it."""
    race = pd.Series(rng.choice(["W", "B", "A"], 300), name="race")
    df = pd.DataFrame(
        {
            "race": race,
            "zipc": race.map({"W": "Z1", "B": "Z2", "A": "Z3"}),
            "customer_ref": [f"REF{i}" for i in range(300)],
        }
    )
    result, rec = _caught(lambda: vfairness.identify_proxy_features(df, protected_attr="race"))
    assert [p["column"] for p in result] == ["zipc"]
    assert "customer_ref" in " ".join(result.not_assessable)
    assert rec


# --- SITE 2c: find_proxy_chains in bias_detection/proxy.py -------------------


def test_the_chain_scan_records_the_links_it_could_not_measure(unique_id_frame):
    """`corr >= correlation_threshold` answers False for a NaN, so every
    unmeasurable link used to leave the scan without a trace."""
    result, rec = _caught(lambda: proxy_mod.find_proxy_chains(unique_id_frame, "gender"))
    assert list(result) == []
    assert result.pairs_not_computed, "an empty chain list must not mean 'none exists'"
    assert result.pairs_not_computed[0][0] == "customer_ref"
    assert "COULD NOT BE MEASURED" in _messages(rec)


def test_the_chain_scan_says_when_the_attribute_is_not_even_present(unique_id_frame):
    result, rec = _caught(lambda: proxy_mod.find_proxy_chains(unique_id_frame, "not_a_column"))
    assert list(result) == []
    assert result.protected_attribute_present is False
    assert "NO chain was examined" in _messages(rec)


def test_the_chain_scan_still_finds_a_real_chain(rng):
    """CONTROL. credit -> income -> race, where credit tracks the part of
    income that race does not explain, so it is a chain and not a direct
    proxy. Both link strengths are recomputed here from their definitions."""
    n = 200
    race = rng.choice(["W", "B", "A"], n)
    latent = rng.normal(0, 1, n)
    income = 0.8 * np.where(race == "W", 1.0, np.where(race == "B", -1.0, 0.0)) + latent
    credit = latent + rng.normal(0, 0.15, n)
    df = pd.DataFrame({"race": race, "income": income, "credit": credit})

    grand = income.mean()
    ss_total = float(((income - grand) ** 2).sum())
    ss_between = sum(
        len(income[race == lvl]) * (income[race == lvl].mean() - grand) ** 2
        for lvl in np.unique(race)
    )
    eta_income_race = math.sqrt(ss_between / ss_total)
    cov = float(((credit - credit.mean()) * (income - income.mean())).sum())
    pearson_credit_income = abs(
        cov
        / math.sqrt(
            float(((credit - credit.mean()) ** 2).sum())
            * float(((income - income.mean()) ** 2).sum())
        )
    )

    result, rec = _caught(lambda: proxy_mod.find_proxy_chains(df, "race"))
    assert [c["chain"] for c in result] == [["credit", "income", "race"]]
    assert result[0]["indirect_correlation"] == pytest.approx(
        pearson_credit_income * eta_income_race, abs=1e-9
    )
    assert result[0]["indirect_correlation"] > 0.3, "precondition: a real chain"
    assert list(result.pairs_not_computed) == []
    assert not rec, f"a fully measurable frame must stay silent: {_messages(rec)}"


# --- SITE 3: the refusal must survive the warning filter ---------------------


def _degenerate_pair():
    n = 60
    feature = pd.Series([f"Z{i:04d}" for i in range(n)], name="zipcode")
    attr = pd.Series([f"R{i % 60}" for i in range(n)], name="race")
    return feature, attr


def test_the_unmeasurable_statistic_survives_w_error_user_warning():
    """`warnings.warn` RAISES under `-W error::UserWarning`, and the enclosing
    `except Exception` swallowed it, so `cramers_v` and `chi2_pvalue` vanished
    from the dict entirely: measured, the keys came back as
    ['pearson', 'pvalue', 'spearman']. A disappearing key is not a third state
    a caller can read."""
    feature, attr = _degenerate_pair()
    fenc = proxy_mod._encode_for_correlation(feature)
    aenc = proxy_mod._encode_for_correlation(attr)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        results = proxy_mod._compute_all_correlations(fenc, aenc, feature, attr, False, False, True)
    assert "cramers_v" in results, "the statistic must not vanish with the warning"
    assert math.isnan(results["cramers_v"])
    assert "chi2_pvalue" in results


def test_the_same_refusal_is_warned_about_under_the_default_filter():
    feature, attr = _degenerate_pair()
    fenc = proxy_mod._encode_for_correlation(feature)
    aenc = proxy_mod._encode_for_correlation(attr)
    results, rec = _caught(
        lambda: proxy_mod._compute_all_correlations(fenc, aenc, feature, attr, False, False, True)
    )
    assert math.isnan(results["cramers_v"])
    assert "not defined" in _messages(rec)


def test_a_measurable_table_still_reports_a_real_cramers_v(rng):
    """CONTROL for the dict lane, with the value recomputed here."""
    n = 400
    attr = pd.Series(rng.choice(["W", "B", "A"], n), name="race")
    draw = rng.random(n)
    feature = pd.Series(
        np.where(draw < 0.8, attr.map({"W": "Z1", "B": "Z2", "A": "Z3"}), "Z9"),
        name="zipc",
    )
    fenc = proxy_mod._encode_for_correlation(feature)
    aenc = proxy_mod._encode_for_correlation(attr)
    results, rec = _caught(
        lambda: proxy_mod._compute_all_correlations(fenc, aenc, feature, attr, False, False, True)
    )
    ct = pd.crosstab(pd.Categorical(feature), pd.Categorical(attr))
    chi2 = float(stats.chi2_contingency(ct)[0])
    total = int(ct.values.sum())
    r, k = ct.shape
    phi2corr = max(0.0, chi2 / total - (r - 1) * (k - 1) / (total - 1))
    rcorr = r - (r - 1) ** 2 / (total - 1)
    kcorr = k - (k - 1) ** 2 / (total - 1)
    expected = math.sqrt(phi2corr / (min(rcorr, kcorr) - 1))
    assert results["cramers_v"] == pytest.approx(expected, abs=1e-9)
    assert results["cramers_v"] > 0.5, "precondition: a real, findable association"
    assert "not defined" not in _messages(rec)


# --- the boundary the whole family exists to defend -------------------------


def test_no_refusal_ever_leaves_the_unit_interval(rng):
    """Every value the screen publishes is either a measurement in [0, 1] or a
    NaN. The 1.0000000000000002 that reached `risk_level: high` was neither."""
    pairs = [
        # The exact pair that produced 1.0000000000000002 before the repair.
        # eta is a square root of a ratio that rounds just above 1 when every
        # level is a singleton, so this fixture is kept seed for seed.
        (
            pd.Series([f"REF{i}" for i in range(60)]),
            pd.Series(np.random.default_rng(1).normal(45, 10, 60)),
        ),
        (pd.Series([f"REF{i}" for i in range(60)]), pd.Series(rng.normal(size=60))),
        (pd.Series(["f"] * 60), pd.Series(rng.normal(size=60))),
        (pd.Series([7.0] * 60), pd.Series(rng.normal(size=60))),
        (pd.Series([f"Z{i}" for i in range(60)]), pd.Series(["m", "f"] * 30)),
        (pd.Series(rng.normal(size=60)), pd.Series(rng.normal(size=60))),
    ]
    for a, b in pairs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = association_strength(a, b)
        assert math.isnan(value) or 0.0 <= value <= 1.0, f"out of range: {value!r}"


def test_a_constant_numeric_side_is_still_a_measured_zero():
    """The reverse of this campaign, and it counts too. A CONSTANT continuous
    variable cannot be explained by any grouping, so eta 0 is the right answer
    and refusing it would throw away evidence. The sibling implementation in
    proxy.py was reviewed and its 0.0 recorded as legitimate on the same
    ground: `categories` is the grouping side by contract, so the degenerate
    side here is always the explained variable and never the protected
    attribute. Pinned here so a later sweep does not "fix" it into a refusal."""
    value, rec = _caught(
        lambda: _correlation_ratio(pd.Series(list("abc") * 20), pd.Series([5.0] * 60))
    )
    assert value == 0.0
    assert not rec, f"a constant numeric side is measured, not refused: {_messages(rec)}"


def test_the_package_export_and_the_module_function_are_the_same_refusal(unique_id_frame):
    """`vfairness.identify_proxy_features` (__init__.py) and the module-level
    function are one object, so the pin above covers the exported surface. The
    typed warning is the channel a caller can filter on."""
    assert vfairness.identify_proxy_features is identify_proxy_features
    with pytest.warns(ProxyScanIncompleteWarning, match="could NOT be assessed"):
        direct = identify_proxy_features(unique_id_frame, protected_attr="gender")
    assert list(direct) == []
    assert direct.not_assessable


def test_a_single_observation_table_is_not_no_association(monkeypatch):
    """The third refusal in `_cramers_v`, `n <= 1`, where Bergsma's correction
    divides by n - 1. NOT REACHABLE through `pd.crosstab`: a table with two
    rows and two columns needs at least two observations, so the degenerate
    margin above catches every real input first. The branch is defensive, and
    it is pinned here through a stubbed crosstab rather than left unpinned,
    because unpinned correct behaviour is one refactor away from regressing."""
    from vfairness.evaluation.vfairness_metrics import discovery as mod

    one_row = pd.DataFrame([[1, 0], [0, 0]], index=["a", "b"], columns=["x", "y"])
    monkeypatch.setattr(mod.pd, "crosstab", lambda *a, **k: one_row)
    value, rec = _caught(lambda: _cramers_v(pd.Series(["a"]), pd.Series(["x"])))
    assert math.isnan(value)
    assert "1 observation(s)" in _messages(rec)
