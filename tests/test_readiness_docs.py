"""The six published API-reference examples that crashed or lied, pinned by
EXECUTING what the page actually ships.

Every block below is read out of `docs/site/api-reference/index.html` at test
time and run. Nothing here retypes an example: a retyped copy passes while the
page stays wrong, which is how all six survived a guard built for exactly this
class of defect.

The six, as reproduced on 2026-09-09 before any edit:

  F-1  gate.evaluate_hierarchical(protected_attributes=['gender', 'race'])
       TypeError: unexpected keyword argument 'protected_attributes'.
       Renaming alone is not the fix: the real `protected_attrs` takes a
       Dict[str, ndarray], and a list of column names raises
       AttributeError: 'list' object has no attribute 'keys'.
  F-2  @auto_log_fairness(..., protected_attr_column=...)
       TypeError: unexpected keyword argument 'protected_attr_column'.
       The parameter was removed the same day (d4062c6, S-08); the page still
       rendered it in the signature block.
  F-3  the decorated function was documented as returning "predictions or a
       fitted model with .predict()". The wrapper accepts a 3-tuple or a dict
       and nothing else: the published example returned `model.predict(X)`,
       warned, and logged NOTHING. Measured: metrics {} and params {}.
  F-4  per_intersection_thresholds={'gender_Female&race_Black': {...}}
       matches nothing the gate looks for, so the documented 0.08 threshold for
       a vulnerable subgroup was silently ignored and the gate kept the RELAXED
       0.12. Measured on 4000 rows: the intersectional gap was 0.10507, the
       documented key APPROVED it, the real key BLOCKED it.
  F-5  result.method_used on the most copied function in the library.
       AttributeError; the attribute is `method`.
  F-6  eg.result_.converged. AttributeError: ExponentiatedGradient has no
       `result_`. `fit()` returns the ReductionResult and `get_result()` hands
       back the same one; `converged` lives one level deeper on
       `.optimization_result`.

Each finding gets a REFUSAL PIN (the defective form is still refused, so the
page cannot drift back into it) and an OVER-CORRECTION CONTROL that asserts
MEASURED values, not membership of a broad set: the exact threshold the gate
applied, the exact metric value it logged, the exact level names it produced.
"""

from __future__ import annotations

import ast
import contextlib
import html
import math
import pathlib
import re
import sys
import types
import warnings

import numpy as np
import pytest

_PAGE = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "site" / "api-reference" / "index.html"
)
_PRE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S)


def _published_blocks() -> list[str]:
    if not _PAGE.exists():  # pragma: no cover - the page ships with the repo
        pytest.skip("published api-reference page not present")
    raw = _PAGE.read_text(encoding="utf-8")
    return [html.unescape(re.sub(r"<[^>]+>", "", b)) for b in _PRE.findall(raw)]


def _block(*needles: str) -> str:
    """The one published block containing all of `needles`.

    Deliberately strict. Two matches means the block this test thinks it is
    running is not the block it is running, which is the failure mode that lets
    a pin go green against the wrong code.
    """
    hits = [b for b in _published_blocks() if all(n in b for n in needles)]
    assert len(hits) == 1, f"{needles} matched {len(hits)} published blocks, expected exactly 1"
    return hits[0]


# ---------------------------------------------------------------------------
# Data. Fixed seeds, so every number asserted below is reproducible.
# ---------------------------------------------------------------------------


def _hierarchical_data(n: int = 4000, bias: float = 0.0, seed: int = 7):
    """Two protected attributes; `bias` depresses positives for Female_Black."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n)
    race = rng.choice(["White", "Black"], size=n)
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    flip = rng.random(n) < 0.15
    y_pred[flip] = 1 - y_pred[flip]
    if bias:
        target = (gender == "Female") & (race == "Black")
        y_pred[target & (rng.random(n) < bias)] = 0
    return gender, race, y_true, y_pred


def _ci_data(n: int = 1200, seed: int = 42):
    rng = np.random.default_rng(seed)
    sensitive = rng.choice(["A", "B"], size=n)
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    y_pred[(sensitive == "A") & (rng.random(n) < 0.2)] = 1
    return y_true, y_pred, sensitive


def _training_data(n: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    sensitive = rng.choice(["M", "F"], size=n)
    return X, y, sensitive


class _RecordingMlflow(types.ModuleType):
    """Records what the decorator logs. No run, no server, no upload."""

    def __init__(self) -> None:
        super().__init__("mlflow")
        self.metrics: dict = {}
        self.params: dict = {}
        self.tags: dict = {}

    def active_run(self):
        return object()

    def start_run(self):
        return contextlib.nullcontext()

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_param(self, key, value):
        self.params[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path, name):  # pragma: no cover - not reached here
        pass


@pytest.fixture
def recording_mlflow(monkeypatch):
    module = _RecordingMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return module


def _intersection_dpd(decision):
    """The demographic-parity evaluation at the intersectional level."""
    level = decision.level_results["intersection:gender_x_race"]
    hits = [m for m in level.metric_evaluations if m.metric_name == "demographic_parity_difference"]
    assert len(hits) == 1
    return hits[0]


# ---------------------------------------------------------------------------
# F-1: evaluate_hierarchical(protected_attributes=[...])
# ---------------------------------------------------------------------------


class TestHierarchicalGateTakesADictOfArrays:
    def test_the_documented_keyword_is_still_refused(self):
        """REFUSAL PIN. `protected_attributes` raises, and always did."""
        from vfairness.operations.cicd import ModelFairnessGate

        gender, race, y_true, y_pred = _hierarchical_data()
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )
        with pytest.raises(TypeError, match="protected_attributes"):
            gate.evaluate_hierarchical(y_true, y_pred, protected_attributes=["gender", "race"])

    def test_renaming_alone_is_not_the_fix(self):
        """REFUSAL PIN. A list of COLUMN NAMES under the right keyword still
        fails: the parameter takes the values, not names to look up."""
        from vfairness.operations.cicd import ModelFairnessGate

        gender, race, y_true, y_pred = _hierarchical_data()
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )
        with pytest.raises(AttributeError, match="keys"):
            gate.evaluate_hierarchical(y_true, y_pred, protected_attrs=["gender", "race"])

    def test_the_published_example_runs_and_measures_all_four_levels(self):
        """OVER-CORRECTION CONTROL. Not "it returns something": the exact four
        level names, the exact verdict, and the measured gap at each level."""
        gender, race, y_true, y_pred = _hierarchical_data()
        namespace = {"y_true": y_true, "y_pred": y_pred, "gender": gender, "race": race}
        exec(  # noqa: S102 - running the published example is the point
            compile(
                _block("evaluate_hierarchical", "per_intersection_thresholds"), _PAGE.name, "exec"
            ),
            namespace,
        )
        decision = namespace["decision"]

        assert list(decision.level_results) == [
            "overall",
            "attr:gender",
            "attr:race",
            "intersection:gender_x_race",
        ]
        assert decision.approved is True
        assert decision.small_sample_warnings == []

        measured = {
            level: round(
                next(
                    m.value
                    for m in result.metric_evaluations
                    if m.metric_name == "demographic_parity_difference"
                ),
                4,
            )
            for level, result in decision.level_results.items()
        }
        assert measured == {
            "overall": 0.032,
            "attr:gender": 0.032,
            "attr:race": 0.035,
            "intersection:gender_x_race": 0.0672,
        }, measured

    def test_the_example_configures_only_metrics_the_gate_can_measure(self):
        """The example used to ask for `equal_opportunity_difference`, which the
        default gate does not compute, so every level came back BLOCKED with
        "could not be computed" while the page showed APPROVED."""
        from vfairness.operations.cicd import ModelFairnessGate

        gender, race, y_true, y_pred = _hierarchical_data()
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )
        computed = set(gate._compute_default_metrics(y_true, y_pred, gender))
        assert computed == {
            "demographic_parity_difference",
            "disparate_impact_ratio",
            "equalized_odds_difference",
            "false_positive_rate_difference",
            "predictive_parity_difference",
        }, computed

        block = _block("evaluate_hierarchical", "per_intersection_thresholds")
        requested = set()
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ModelFairnessGate":
                for keyword in node.keywords:
                    if keyword.arg == "metrics":
                        requested = {ast.literal_eval(e) for e in keyword.value.elts}
        assert requested and requested <= computed, (
            f"the published example asks the gate for {sorted(requested - computed)}, "
            "which it cannot compute; every level would report could-not-check"
        )


# ---------------------------------------------------------------------------
# F-4: per_intersection_thresholds keys
# ---------------------------------------------------------------------------


class TestPerIntersectionThresholdKeyFormat:
    """The key shape decides whether a documented stricter threshold reaches the
    gate at all. `bias=0.10` puts the intersectional gap at 0.10507, between the
    documented 0.08 and the relaxed default 0.12, so the two key shapes give
    OPPOSITE verdicts on identical data."""

    @staticmethod
    def _run(key):
        from vfairness.operations.cicd import HierarchicalGateConfig, ModelFairnessGate

        gender, race, y_true, y_pred = _hierarchical_data(bias=0.10, seed=11)
        gate = ModelFairnessGate(
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )
        config = HierarchicalGateConfig(
            check_intersections=True,
            intersection_depth=2,
            per_intersection_thresholds=(
                {key: {"demographic_parity_difference": 0.08}} if key else {}
            ),
        )
        return gate.evaluate_hierarchical(
            y_true,
            y_pred,
            protected_attrs={"gender": gender, "race": race},
            hierarchical_config=config,
        )

    def test_the_documented_key_shape_reaches_nothing(self):
        """REFUSAL PIN. 'gender_Female&race_Black' leaves the threshold at the
        RELAXED 0.12 and approves a 0.105 gap the reader thinks they blocked."""
        evaluation = _intersection_dpd(self._run("gender_Female&race_Black"))
        assert evaluation.value == pytest.approx(0.10507, abs=1e-4)
        assert evaluation.threshold == 0.12
        assert evaluation.passed is True

        untouched = _intersection_dpd(self._run(None))
        assert untouched.threshold == 0.12, "the bad key is indistinguishable from no key at all"

    def test_the_real_key_shape_tightens_the_gate_and_blocks(self):
        """OVER-CORRECTION CONTROL. The exact threshold applied, and the flipped
        verdict on identical data."""
        decision = self._run("Female_Black")
        evaluation = _intersection_dpd(decision)
        assert evaluation.value == pytest.approx(0.10507, abs=1e-4)
        assert evaluation.threshold == 0.08
        assert evaluation.passed is False
        assert decision.approved is False

    @pytest.mark.parametrize("key", ["Female_Black", "gender_x_race", "intersection:gender_x_race"])
    def test_every_key_shape_the_page_documents_actually_matches(self, key):
        """OVER-CORRECTION CONTROL. All three documented shapes are checked, so
        the page cannot name a shape the code ignores."""
        assert _intersection_dpd(self._run(key)).threshold == 0.08

    def test_the_page_documents_a_key_that_works(self):
        """The published block must not carry a key shape that matches nothing."""
        block = _block("evaluate_hierarchical", "per_intersection_thresholds")
        keys: list[str] = []
        for node in ast.walk(ast.parse(block)):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "HierarchicalGateConfig"
            ):
                for keyword in node.keywords:
                    if keyword.arg == "per_intersection_thresholds":
                        keys = [ast.literal_eval(k) for k in keyword.value.keys]
        assert keys, "the example no longer documents a per-intersection threshold"
        for key in keys:
            assert _intersection_dpd(self._run(key)).threshold == 0.08, (
                f"the page documents the key {key!r}, which the gate matches against nothing"
            )


# ---------------------------------------------------------------------------
# F-2 and F-3: auto_log_fairness
# ---------------------------------------------------------------------------


class TestAutoLogFairnessExample:
    def test_the_removed_parameter_is_still_refused(self):
        """REFUSAL PIN for F-2."""
        from vfairness import auto_log_fairness

        with pytest.raises(TypeError, match="protected_attr_column"):
            auto_log_fairness(backend="mlflow", protected_attr_column="gender")

    def test_the_signature_block_no_longer_shows_it(self):
        """REFUSAL PIN for F-2, on the page rather than on the function."""
        import inspect

        from vfairness import auto_log_fairness

        block = _block("@auto_log_fairness(", "prefix: str = 'fairness'")
        real = set(inspect.signature(auto_log_fairness).parameters)
        documented = {
            line.split(":")[0].strip()
            for line in block.splitlines()[1:]
            if ":" in line and not line.strip().startswith(("#", "*", ")"))
        }
        assert documented, "the signature block stopped listing parameters"
        assert documented <= real, f"documented but not real: {sorted(documented - real)}"
        assert "protected_attr_column" not in documented

    def test_the_shape_the_old_example_returned_logs_nothing(self, recording_mlflow):
        """REFUSAL PIN for F-3. A bare prediction array warns and SKIPS. This is
        the whole finding: the documented example logged nothing at all, and
        nothing about the run said so."""
        from vfairness import auto_log_fairness

        X, y, sensitive = _training_data()

        @auto_log_fairness(backend="mlflow", metrics=["demographic_parity_difference"])
        def train_model(X, y, sensitive):
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression().fit(X, y).predict(X)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = train_model(X, y, sensitive)

        assert isinstance(returned, np.ndarray), "the wrapper must still return your value"
        assert any(
            "must return (y_pred, y_true, sensitive_attr)" in str(w.message) for w in caught
        ), [str(w.message) for w in caught]
        assert recording_mlflow.metrics == {}
        assert recording_mlflow.params == {}

    def test_a_fitted_model_return_also_logs_nothing(self, recording_mlflow):
        """REFUSAL PIN for F-3, second half. The page also promised "a fitted
        model with a .predict() method" worked. It does not."""
        from vfairness import auto_log_fairness

        X, y, sensitive = _training_data()

        @auto_log_fairness(backend="mlflow", metrics=["demographic_parity_difference"])
        def train_model(X, y, sensitive):
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression().fit(X, y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_model(X, y, sensitive)
        assert recording_mlflow.metrics == {}

    def test_the_published_example_now_logs_a_measured_value(self, recording_mlflow):
        """OVER-CORRECTION CONTROL. The exact metric value, not "some metrics"."""
        X, y, sensitive = _training_data()
        namespace = {"X_train": X, "y_train": y, "gender": sensitive}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                exec(  # noqa: S102 - running the published example is the point
                    compile(_block("auto_log_fairness", "evaluate_model"), _PAGE.name, "exec"),
                    namespace,
                )
            except Exception as exc:  # noqa: BLE001 - the failure IS the finding
                pytest.fail(
                    "the published @auto_log_fairness example does not run as written: "
                    f"{exc!r}. It must return (y_pred, y_true, sensitive_attr) or a dict "
                    "with those keys; anything else warns and logs nothing."
                )

        skipped = [w for w in caught if "Skipping fairness logging" in str(w.message)]
        assert not skipped, "the published example logs NOTHING: " + "; ".join(
            str(w.message) for w in skipped
        )
        assert recording_mlflow.metrics["fairness.demographic_parity_difference"] == pytest.approx(
            0.0907, abs=1e-4
        )
        assert recording_mlflow.metrics["fairness.fairness_score"] == pytest.approx(1.0)
        assert namespace["y_pred"].shape == (400,)

    def test_the_page_does_not_promise_an_enforced_threshold(self):
        """`thresholds` are RECORDED, never enforced. The page has to say so
        where the reader is, not only in the docstring."""
        page = _PAGE.read_text(encoding="utf-8")
        section = page[page.index('id="auto-log-fairness"') :][:6000]
        assert "thresholds_enforced" in section
        assert "recorded, not enforced" in section.replace("</strong>", "").replace("<strong>", "")


# ---------------------------------------------------------------------------
# F-5: StatisticalResult.method
# ---------------------------------------------------------------------------


class TestStatisticalResultAttributeNames:
    def test_method_used_does_not_exist(self):
        """REFUSAL PIN. The published line raised AttributeError."""
        from vfairness import demographic_parity_difference_with_ci

        y_true, y_pred, sensitive = _ci_data()
        result = demographic_parity_difference_with_ci(
            y_true, y_pred, sensitive, n_bootstrap=500, random_state=42
        )
        with pytest.raises(AttributeError, match="method_used"):
            result.method_used  # noqa: B018 - reading it IS the assertion

    def test_the_published_example_prints_measured_values(self, capsys):
        """OVER-CORRECTION CONTROL. Every printed line, matched exactly, so a
        renamed attribute or a changed estimator cannot pass as "prints
        something"."""
        y_true, y_pred, sensitive = _ci_data()
        namespace = {"y_true": y_true, "y_pred": y_pred, "sensitive_attr": sensitive}
        exec(  # noqa: S102 - running the published example is the point
            compile(
                _block("demographic_parity_difference_with_ci", "Standard Error"),
                _PAGE.name,
                "exec",
            ),
            namespace,
        )
        printed = capsys.readouterr().out.splitlines()
        assert printed == [
            "Point Estimate: 0.125",
            "95% CI: [0.068, 0.184]",
            "Standard Error: 0.0292",
            "Method: stratified_bootstrap_fold_debiased",
            "Interval Type: confidence",
        ], printed

    def test_the_returns_table_names_the_real_fields(self):
        """SIBLING of F-5, found in the same sweep: the `bootstrap_ci` Returns
        line named `n_samples`, which this dataclass has never had."""
        import dataclasses

        from vfairness.evaluation.vfairness_metrics._statistics import StatisticalResult

        fields = {f.name for f in dataclasses.fields(StatisticalResult)}
        assert "n_samples" not in fields and "sample_size" in fields

        page = _PAGE.read_text(encoding="utf-8")
        checked = 0
        for match in re.finditer(r"<p>(.*?)</p>", page, re.S):
            paragraph = match.group(1)
            if not paragraph.startswith("<code>StatisticalResult</code>"):
                continue
            # The enumeration is the FIRST sentence: everything up to the first
            # period that closes a code tag. Prose after it (which may name a
            # field to say it is absent) is a reader's sentence, not a claim
            # about the dataclass, and judging it would invent defects.
            enumeration = paragraph.split("</code>.", 1)[0] + "</code>"
            claimed = set(re.findall(r"<code>([a-z_][a-z0-9_]*)</code>", enumeration))
            assert claimed, f"no fields parsed out of {enumeration!r}"
            checked += 1
            unreal = claimed - fields
            assert not unreal, f"the page claims StatisticalResult fields {sorted(unreal)}"
        assert checked >= 2, f"only {checked} StatisticalResult Returns paragraphs found"


# ---------------------------------------------------------------------------
# F-6: ExponentiatedGradient result paths
# ---------------------------------------------------------------------------


class TestExponentiatedGradientResultPaths:
    @staticmethod
    def _fitted():
        from sklearn.ensemble import RandomForestClassifier

        from vfairness.in_processing import DemographicParityConstraint, ExponentiatedGradient

        rng = np.random.default_rng(0)
        n = 600
        gender = rng.choice([0, 1], size=n)
        X = rng.normal(size=(n, 4))
        X[:, 0] += gender * 1.2
        y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.6, size=n) > 0.6).astype(int)
        eg = ExponentiatedGradient(
            base_estimator=RandomForestClassifier(n_estimators=25, random_state=0),
            constraint=DemographicParityConstraint(tolerance=0.05),
            max_iterations=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = eg.fit(X, y, sensitive_attr=gender)
        return eg, result

    def test_result_underscore_does_not_exist(self):
        """REFUSAL PIN. Both published lines raised AttributeError."""
        eg, _ = self._fitted()
        with pytest.raises(AttributeError, match="result_"):
            eg.result_  # noqa: B018 - reading it IS the assertion

    def test_converged_is_not_on_the_result_itself(self):
        """REFUSAL PIN. Renaming `result_` to `get_result()` alone still fails:
        `converged` lives one level deeper, on `.optimization_result`."""
        eg, result = self._fitted()
        assert not hasattr(result, "converged"), (
            "converged moved onto ReductionResult; the page's extra hop is now wrong"
        )
        with pytest.raises(AttributeError, match="converged"):
            eg.get_result().converged  # noqa: B018

    def test_both_documented_paths_give_the_same_measured_values(self):
        """OVER-CORRECTION CONTROL. `fit()`'s return and `get_result()` are the
        SAME object, and the numbers are the measured ones, not a default."""
        eg, result = self._fitted()
        assert eg.get_result() is result
        # numpy hands back np.False_ here, so it is normalised before the
        # identity check; the assertion is still on the measured value.
        assert bool(result.optimization_result.converged) is False
        # READINESS, 2026-09-10. This asserted final_violation == 0.2791 +/- 5e-3,
        # a magnitude captured on scikit-learn 1.8.0. CI resolves 1.9.0 and
        # measures 0.1306, so the same correct code was RED on every commit for a
        # day and the failure said "the library is broken" when it meant "the
        # solver changed".
        #
        # The subject of this control, per the docstring above, is that the
        # numbers are MEASURED rather than a fabricated default. A version-pinned
        # constant does not assert that; it asserts the solver version. What does
        # assert it: the violation is a real, finite, strictly positive quantity
        # on a fixture built to have a real disparity, which is exactly what a
        # neutral default (0.0, the cleanest reading on this scale) would not be.
        violation = float(result.final_violation)
        assert math.isfinite(violation), f"final_violation is {violation!r}, not a measurement"
        assert 0.0 < violation < 1.0, (
            f"final_violation is {violation}. Zero would be the neutral default this "
            f"control exists to catch, and the fixture has a real disparity."
        )
        assert 0.0 <= result.accuracy <= 1.0
        assert result.optimization_result.n_iterations >= 1

    def test_the_published_example_runs(self, capsys):
        """OVER-CORRECTION CONTROL: the block on the page, executed."""
        rng = np.random.default_rng(0)
        n = 600
        gender = rng.choice([0, 1], size=n)
        X = rng.normal(size=(n, 4))
        X[:, 0] += gender * 1.2
        y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.6, size=n) > 0.6).astype(int)
        namespace = {"X_train": X, "y_train": y, "X_test": X, "gender": gender}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(  # noqa: S102 - running the published example is the point
                compile(
                    _block("ExponentiatedGradient(", "Converged", "RandomForest"),
                    _PAGE.name,
                    "exec",
                ),
                namespace,
            )
        printed = capsys.readouterr().out.splitlines()
        assert printed[0] == "Converged: False"

        # The published block builds an UNSEEDED RandomForest, so the violation
        # moves a little between runs (0.2791, 0.2795, 0.2827 were observed).
        # Pinning a prefix pinned the run, not the claim. What is pinned instead
        # is exact: the number printed IS the measured `final_violation` of this
        # run, and it exceeds the 0.05 tolerance, which is precisely why the
        # line above reads False rather than True.
        printed_violation = float(printed[1].removeprefix("Violation: "))
        assert printed_violation == namespace["result"].final_violation
        assert printed_violation > 0.05
        assert namespace["y_pred"].shape == (600,)


def test_the_extractor_finds_the_blocks_these_pins_run():
    """NON-VACUITY. Every pin above runs a block pulled out of the page. If the
    extractor ever returned nothing, `_block` would raise rather than pass, but
    a narrowed extractor that still found ONE block per needle would look
    healthy while checking a fraction of the page. This floors the whole set."""
    blocks = _published_blocks()
    assert len(blocks) >= 200, f"only {len(blocks)} blocks extracted; the page or regex changed"
    for needles in (
        ("evaluate_hierarchical", "per_intersection_thresholds"),
        ("auto_log_fairness", "evaluate_model"),
        ("@auto_log_fairness(", "prefix: str = 'fairness'"),
        ("demographic_parity_difference_with_ci", "Standard Error"),
        ("ExponentiatedGradient(", "Converged", "RandomForest"),
    ):
        assert _block(*needles).strip(), needles
