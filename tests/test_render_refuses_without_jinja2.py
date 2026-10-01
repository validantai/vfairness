"""A renderer without its template engine REFUSES; it never answers "".

Audit finding, 2026-08-28. In a core-only install (``pip install vfairness``,
no extras) 28 of the 44 public ``*_to_svg`` adapters raised a named ImportError
from ``engine.render_svg``. The other 16 ran

    warnings.warn("Jinja2 not available for SVG rendering")
    return ""

Three things are wrong with that, and none of them is style:

1. Every one of those docstrings documents the return as "SVG markup string",
   and ``fairness_detailed_report_to_svg``'s own example passes
   ``save_path='fairness.svg'``. A caller doing ``open(p, "w").write(svg)``
   wrote a ZERO-BYTE file and read it as this run's report.
2. Python's default warning filter shows a given warning once per process, so a
   batch job rendering many reports warned once and then produced nothing at
   all, silently, for every report after the first.
3. The same library answered the same condition two different ways, so no
   caller could write one correct handler.

The tests below are enumerated from the SOURCE, not from a hand-written list:
any adapter that guards on ``JINJA2_AVAILABLE`` is covered automatically, so a
new one added with the old pattern turns this file red.
"""

import inspect

import pytest

from vfairness import rendering
from vfairness.rendering import engine


def _guarded_adapters():
    """Every public ``*_to_svg`` whose body checks ``JINJA2_AVAILABLE``."""
    found = []
    for name in sorted(n for n in dir(rendering) if n.endswith("_to_svg")):
        if name.startswith("_"):
            continue
        func = getattr(rendering, name)
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):  # pragma: no cover - compiled or wrapped
            continue
        if "JINJA2_AVAILABLE" in source:
            found.append(name)
    return found


GUARDED = _guarded_adapters()


def _dummy_args(func):
    """Placeholder values for the required positionals.

    The guard is the first statement in every one of these functions, so the
    values are never read. Passing junk is the point: it proves the refusal does
    not depend on the input being good.
    """
    args = []
    for param in inspect.signature(func).parameters.values():
        if param.kind is param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
            args.append({})
    return args


def test_the_expected_adapters_are_the_ones_under_test():
    """A count, so a silently-emptied enumeration cannot make this file vacuous."""
    assert len(GUARDED) == 16, GUARDED
    assert "fairness_detailed_report_to_svg" in GUARDED
    assert "threshold_optimization_to_svg" in GUARDED


@pytest.mark.parametrize("name", GUARDED)
def test_a_missing_engine_raises_instead_of_returning_an_empty_string(name, monkeypatch):
    func = getattr(rendering, name)
    module = inspect.getmodule(func)
    monkeypatch.setattr(module, "JINJA2_AVAILABLE", False)

    with pytest.raises(ImportError) as excinfo:
        func(*_dummy_args(func))

    message = str(excinfo.value)
    assert "Jinja2 is required for SVG report rendering" in message
    # The hint names the EXTRA this package declares, not the bare distribution:
    # `pip install jinja2` installs the dependency without recording that the
    # library wanted it (audit note, 2026-08-22, unfixed until now).
    assert "vfairness[rendering]" in message


@pytest.mark.parametrize("name", GUARDED)
def test_the_same_call_does_not_raise_that_error_when_the_engine_is_present(name):
    """The over-correction control. With Jinja2 installed (this venv) the
    missing-backend refusal must not fire, whatever else the junk input does."""
    func = getattr(rendering, name)

    try:
        func(*_dummy_args(func))
    except ImportError as exc:  # pragma: no cover - would be the regression
        pytest.fail(f"{name} raised the missing-engine ImportError with Jinja2 present: {exc}")
    except Exception:
        # Any other complaint about the junk input is this test's business to
        # ignore; only the ImportError above is under test here.
        pass


def test_the_engine_itself_raises_the_same_error():
    """One message, one place. Sixteen adapters and render_svg now share it."""
    assert "vfairness[rendering]" in engine.JINJA2_MISSING_MESSAGE

    with pytest.raises(ImportError, match=r"vfairness\[rendering\]"):
        engine.raise_jinja2_missing()


def test_no_adapter_still_answers_a_missing_engine_with_an_empty_string():
    """Source-level backstop for the pattern itself, so it cannot be
    reintroduced in a new adapter that this file's enumeration would otherwise
    happily cover with a passing test."""
    offenders = []
    for name in sorted(
        n for n in dir(rendering) if n.endswith("_to_svg") and not n.startswith("_")
    ):
        source = inspect.getsource(getattr(rendering, name))
        if "Jinja2 not available for SVG rendering" in source:
            offenders.append(name)

    assert offenders == [], f"these adapters still warn-and-return-'' : {offenders}"


class TestHealthyRenderingIsUntouched:
    """The other half of the control: real input still produces real markup."""

    REPORT = {
        "assessment": {"fairness_score": 0.82, "passed_metrics": ["demographic_parity_difference"]},
        "metrics": {"demographic_parity_difference": 0.04},
        "group_statistics": {"a": {"count": 120}, "b": {"count": 118}},
    }

    def test_a_detailed_report_still_renders(self):
        svg = rendering.fairness_detailed_report_to_svg(self.REPORT)

        assert svg.startswith("<svg") or "<svg" in svg
        assert svg.rstrip().endswith("</svg>")
        assert len(svg) > 5000

    def test_an_export_still_writes_the_file_it_drew(self, tmp_path):
        out = tmp_path / "threshold.svg"

        svg = rendering.threshold_optimization_to_svg(
            {"constraint_type": "demographic_parity"}, save_path=str(out)
        )

        assert svg.rstrip().endswith("</svg>")
        assert out.read_text() == svg
