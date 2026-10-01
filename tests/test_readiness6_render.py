"""READINESS-6, the render layer: a report that could not be produced must not
be indistinguishable from one that was.

Two defects, both proven by execution on 2026-09-10.

1. Ten adapter render wrappers caught every exception, warned, and returned "".
   `engine.raise_jinja2_missing` records the decision taken 2026-08-28 for the
   missing-BACKEND case, in its own words: "their docstrings document the return
   as 'SVG markup string', so a caller doing open(p, "w").write(svg) wrote a
   ZERO-BYTE file and read it as this run's report ... An empty string is not
   markup, and 'could not render' must not be indistinguishable from 'rendered
   nothing worth showing'." That fix funnelled 16 adapters through a raise and
   deliberately left the general render-FAILURE path alone. It should not have:
   the reasoning is identical, and `render_svg` raises THROUGH these wrappers,
   so the refusal was caught here anyway.

   Measured: a real failure (incomplete template data) returned '' with one
   warning, and THREE failed renders produced ONE warning, because Python shows
   a warning once per process. A batch job warns once and then emits nothing,
   silently, for every report after it.

2. `adapters_robustness` read a REFUSED permutation row as a graded pass.
   `p_value = float(raw_p) if raw_p is not None else None` lets NaN through, and
   NaN is how this library says a test refused to run. `nan < 0.05` is False, so
   the row was counted as a pass in the 0.4-weighted permutation component.

   Measured: one refused row rendered "stability score 1.00", a perfect score
   for a test that never ran; a refused row beside a real breach rendered 0.50,
   halving the visible severity.
"""

from __future__ import annotations

import re
import warnings

import pytest

RENDER_WRAPPER_MODULES = [
    "adapters_discovery",
    "adapters_experimentation",
    "adapters_ranking",
    "adapters_regression",
    "adapters_robustness",
    "adapters_validation",
]


def _module(name):
    import importlib

    return importlib.import_module(f"vfairness.rendering.{name}")


class TestAFailedRenderIsRefusedNotAnsweredWithAnEmptyString:
    def test_no_wrapper_returns_an_empty_string_on_failure(self):
        """Source-level, across all six files at once, so a new adapter that
        copies the old wrapper is caught even if nothing calls it yet."""
        import ast
        import pathlib

        offenders = []
        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "vfairness" / "rendering"
        for name in RENDER_WRAPPER_MODULES:
            path = root / f"{name}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Return)
                        and isinstance(inner.value, ast.Constant)
                        and inner.value.value == ""
                    ):
                        offenders.append(f"{name}.py:{node.lineno}")
        assert not offenders, (
            "a render failure is answered with an empty string, which a caller "
            f"writes out as a zero-byte report: {offenders}"
        )

    def test_the_wrapper_raises_and_names_the_cause(self):
        ar = _module("adapters_ranking")
        with pytest.raises(Exception) as excinfo:
            ar._render_and_save("ranking_fairness", {}, None)
        # The cause names the missing key, which the old warning never did.
        assert "verdict" in str(excinfo.value), excinfo.value

    def test_a_working_render_is_untouched(self):
        """OVER-CORRECTION CONTROL. Every adapter that can render a demo still
        renders one, and it is real markup rather than a stub."""
        import inspect

        import vfairness.rendering as rnd

        rendered = 0
        for name in sorted(n for n in dir(rnd) if n.endswith("_to_svg")):
            fn = getattr(rnd, name)
            if "example" not in inspect.signature(fn).parameters:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                svg = fn(example=True)
            assert isinstance(svg, str) and len(svg) > 500, name
            assert svg.lstrip().startswith("<"), name
            rendered += 1
        assert rendered >= 8, f"only {rendered} demo adapters exercised"


class TestARefusedPermutationRowIsNotAPass:
    @staticmethod
    def _score(rows):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svg = _module("adapters_robustness").robustness_testing_to_svg(permutation_results=rows)
        match = re.search(r"[Ss]core[^0-9]{0,14}([0-9]+\.[0-9]+)", svg)
        return (match.group(1) if match else None), svg

    REFUSED = {
        "metric": "equal_opportunity",
        "p_value": float("nan"),
        "significant_at_05": None,
        "method": "permutation (not run: null distribution collapsed)",
    }
    CLEAN = {
        "metric": "predictive_parity",
        "p_value": 0.60,
        "significant_at_05": False,
        "method": "permutation",
    }
    BREACH = {
        "metric": "demographic_parity",
        "p_value": 0.0396,
        "significant_at_05": True,
        "method": "permutation",
    }

    def test_a_refused_row_alone_scores_nothing(self):
        score, svg = self._score([self.REFUSED])
        assert score is None, f"a test that never ran rendered a stability score of {score}"
        assert "COULD NOT CHECK" in svg.upper()
        assert "neither robust nor fragile" in svg

    def test_a_refused_row_does_not_dilute_a_real_breach(self):
        score, _ = self._score([self.BREACH, self.REFUSED])
        assert score == "0.00", (
            f"the refused row was counted as a pass and lifted the score to {score}"
        )

    def test_measured_rows_still_score_exactly_as_before(self):
        """OVER-CORRECTION CONTROL, both ends of the scale."""
        assert self._score([self.CLEAN])[0] == "1.00"
        assert self._score([self.BREACH])[0] == "0.00"
        assert self._score([self.CLEAN, self.BREACH])[0] == "0.50"

    def test_a_row_graded_from_its_p_value_alone_still_grades(self):
        """The branch `_finite_or_none` actually governs.

        PIN FINDING, found by sabotage and reported rather than skipped: every
        control above supplies `significant_at_05`, and the adapter prefers a
        row's own verdict over its p-value, so none of them reached the p-value
        branch at all. Making `_finite_or_none` return None for EVERYTHING left
        the suite green. These rows carry a p-value and no verdict, which is the
        only shape that exercises it.
        """
        clean = {"metric": "predictive_parity", "p_value": 0.60, "method": "permutation"}
        breach = {"metric": "demographic_parity", "p_value": 0.0396, "method": "permutation"}
        assert self._score([clean])[0] == "1.00"
        assert self._score([breach])[0] == "0.00"
        assert self._score([clean, breach])[0] == "0.50"
        # And the refusal still refuses on this same branch.
        refused = {
            "metric": "equal_opportunity",
            "p_value": float("nan"),
            "method": "permutation (not run)",
        }
        assert self._score([refused])[0] is None

    def test_a_none_p_value_and_a_nan_p_value_are_treated_alike(self):
        """They mean the same thing and used to be graded differently."""
        absent = dict(self.REFUSED, p_value=None)
        assert self._score([absent])[0] is None
        assert self._score([self.REFUSED])[0] is None
        assert self._score([self.BREACH, absent])[0] == self._score([self.BREACH, self.REFUSED])[0]
