"""The published API reference must describe the constructor the code has.

THE DEFECT THIS EXISTS FOR. ``docs/site/api-reference/index.html`` is the page
the README links as "every public function and class", and the ``FairnessAnalyzer``
entry is the main way into the library. Its parameter table said of ``task_type``:

    One of 'classification', 'regression', 'ranking'. Auto-detected if None.

The runtime accepts two of those and refuses the third by name with
``ConfigurationError``. A reader following the most visible page in the project,
on its main class, was told about a mode that does not exist. Ranking fairness is
real, but it is measured by standalone functions taking ``(rankings, groups)``,
not by an analyzer task type. In the same table, ``backend`` appeared in the
rendered signature and had no row at all: the only constructor argument silently
omitted.

There is no generator behind that page. No markdown source produces it, so it is
hand-edited and nothing regenerates or re-derives it. This file is the check that
was missing.

THE RULES, all derived by EXECUTION rather than from a list of words:

1. The parameter names in the table are exactly the constructor's keyword
   parameters. Adding or removing one without touching the page is red. This is
   what catches an omitted row.
2. Every value the constructor ACCEPTS is quoted in that parameter's description.
   A value added in code and not documented is red.
3. Every quoted literal in a description that the constructor REFUSES must sit in
   a description that says so. This is the ``'ranking'`` case exactly: it was
   quoted as an option, is refused, and nothing on the page said so.

The accepted set is read out of the library's own refusal message ("Accepted
values are 'a', 'b'"), so it comes from the code path a user actually hits rather
than from a constant this file might get wrong.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Dict, List, Set

import pytest

pytest.importorskip("numpy")

LIB_ROOT = Path(__file__).resolve().parents[1]
PAGE = LIB_ROOT / "docs" / "site" / "api-reference" / "index.html"

# The `<div class="api-param">` blocks inside the FairnessAnalyzer entry.
_PARAM_RE = re.compile(
    r'<div class="api-param">\s*'
    r'<div class="api-param-header">\s*'
    r'<span class="api-param-name">([^<]+)</span>.*?'
    r'<div class="api-param-desc">(.*?)</div>',
    re.DOTALL,
)
# A string literal offered as a value: <code>'classification'</code>.
_LITERAL_RE = re.compile(r"<code>'([^']+)'</code>")
# The signature the page renders above the table.
_SIGNATURE_RE = re.compile(
    r'<pre><code class="language-python">FairnessAnalyzer\((.*?)\)</code></pre>', re.DOTALL
)

_REFUSAL_CUES = (
    "refused",
    "raises",
    "not implemented",
    "is not an analyzer task type",
    "ConfigurationError",
)

# Values whose refusal is the point of documenting them. Parameters not listed
# here are checked with rule 3 all the same; this tuple only records which
# descriptions are EXPECTED to carry a refusal cue, so the cue cannot be deleted
# while the mention stays.
_PARAMS_THAT_NAME_A_REFUSED_VALUE = ("backend",)


def _analyzer_section(html: str) -> str:
    """The FairnessAnalyzer entry, not the other 300 api-param blocks on the page."""
    start = html.index('<span class="api-function-name">class FairnessAnalyzer</span>')
    end = html.index("<h4>Methods</h4>", start)
    return html[start:end]


def documented_params(html: str) -> Dict[str, str]:
    section = _analyzer_section(html)
    params = {name.strip(): desc for name, desc in _PARAM_RE.findall(section)}
    assert params, "no api-param blocks parsed out of the FairnessAnalyzer entry"
    return params


def rendered_signature_params(html: str) -> List[str]:
    section = _analyzer_section(html)
    match = _SIGNATURE_RE.search(section)
    assert match, "the FairnessAnalyzer entry no longer renders a constructor signature"
    names: List[str] = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        name = line.split(":")[0].split("=")[0].strip().rstrip(",").strip()
        if name and name != "*":
            names.append(name)
    return names


def constructor_params() -> List[str]:
    from vfairness import FairnessAnalyzer

    return [p for p in inspect.signature(FairnessAnalyzer.__init__).parameters if p != "self"]


_ACCEPTED_RE = re.compile(r"Accepted values are ([^.]+)\.")


def accepted_values(param: str) -> Set[str]:
    """What the constructor accepts, read out of its own refusal message.

    Probing beats reading a module constant: this is the code path a user hits,
    so a validator that stopped consulting its constant would be caught here.
    """
    import numpy as np

    from vfairness import FairnessAnalyzer
    from vfairness.exceptions import ConfigurationError

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    groups = np.array(["a", "b"] * 30)
    try:
        FairnessAnalyzer(y, y, groups, **{param: "__probe_invalid_value__"})
    except ConfigurationError as exc:
        match = _ACCEPTED_RE.search(str(exc))
        assert match, f"{param}: refusal message no longer lists its accepted values: {exc}"
        return {v.strip().strip("'\"") for v in match.group(1).split(",")}
    raise AssertionError(
        f"{param}: the constructor accepted '__probe_invalid_value__', so this file cannot "
        "establish what it accepts, and neither can a reader"
    )


def _is_accepted(param: str, value: str) -> bool:
    import numpy as np

    from vfairness import FairnessAnalyzer
    from vfairness.exceptions import ConfigurationError

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    groups = np.array(["a", "b"] * 30)
    try:
        FairnessAnalyzer(y, y, groups, **{param: value})
    except ConfigurationError:
        return False
    return True


# --- The pins ----------------------------------------------------------------

CHECKED_PARAMS = ("task_type", "missing_strategy", "backend")


def test_the_page_still_carries_a_parameter_table_to_check() -> None:
    """Guard against a vacuous run before any rule below is trusted."""
    params = documented_params(PAGE.read_text(encoding="utf-8"))
    assert len(params) >= 8, sorted(params)
    for name in CHECKED_PARAMS:
        assert name in params, f"{name} has no row in the FairnessAnalyzer parameter table"


def test_the_table_documents_exactly_the_constructor_parameters() -> None:
    """Rule 1. ``backend`` was in the rendered signature and had no row."""
    documented = set(documented_params(PAGE.read_text(encoding="utf-8")))
    actual = set(constructor_params())
    assert documented == actual, (
        f"the parameter table and FairnessAnalyzer.__init__ disagree.\n"
        f"  undocumented: {sorted(actual - documented)}\n"
        f"  documented but gone from the signature: {sorted(documented - actual)}"
    )


def test_the_rendered_signature_matches_the_constructor_too() -> None:
    """The code block above the table is the first thing a reader copies."""
    assert rendered_signature_params(PAGE.read_text(encoding="utf-8")) == constructor_params()


@pytest.mark.parametrize("param", CHECKED_PARAMS)
def test_every_accepted_value_is_documented(param: str) -> None:
    """Rule 2. A value the code accepts and the page never mentions is a gap."""
    desc = documented_params(PAGE.read_text(encoding="utf-8"))[param]
    quoted = set(_LITERAL_RE.findall(desc))
    missing = sorted(accepted_values(param) - quoted)
    assert not missing, (
        f"the page's {param} row does not mention {missing}, which the constructor accepts"
    )


@pytest.mark.parametrize("param", CHECKED_PARAMS)
def test_no_refused_value_is_offered_as_an_option(param: str) -> None:
    """Rule 3, and the ``task_type='ranking'`` defect exactly.

    A literal the page quotes and the code refuses may stay on the page, but the
    page has to say it is refused. Silence is what made 'ranking' read as a mode
    the library has.
    """
    desc = documented_params(PAGE.read_text(encoding="utf-8"))[param]
    refused = sorted(v for v in set(_LITERAL_RE.findall(desc)) if not _is_accepted(param, v))
    if not refused:
        return
    assert any(cue in desc for cue in _REFUSAL_CUES), (
        f"the page's {param} row quotes {refused} as though they were options; the constructor "
        f"refuses them and the row never says so"
    )


@pytest.mark.parametrize("param", _PARAMS_THAT_NAME_A_REFUSED_VALUE)
def test_a_row_that_names_a_refused_value_keeps_saying_it_is_refused(param: str) -> None:
    """Over-correction control for rule 3: the cue may not be dropped.

    Rule 3 is satisfiable two ways, by adding the cue or by deleting the
    mention. Deleting the mention of ``'fairlearn'`` would leave a user who
    passes it with an unexplained error, so this pins that the row still names
    it AND still says what happens.
    """
    desc = documented_params(PAGE.read_text(encoding="utf-8"))[param]
    quoted = set(_LITERAL_RE.findall(desc))
    refused = {v for v in quoted if not _is_accepted(param, v)}
    assert refused, f"the {param} row no longer names any refused value; did the mention go?"
    assert any(cue in desc for cue in _REFUSAL_CUES), desc


def test_the_analyzer_entry_does_not_claim_a_ranking_task() -> None:
    """The prose above the table said the class covers "ranking tasks" as well.

    Rule 3 reads the parameter row; this reads the sentence that introduces the
    class, which made the same claim in words rather than as a quoted value.
    """
    from vfairness import FairnessAnalyzer

    section = _analyzer_section(PAGE.read_text(encoding="utf-8"))
    intro = section[: section.index('<div class="api-params">')]
    ranking_methods = [
        m
        for m in dir(FairnessAnalyzer)
        if not m.startswith("_") and ("rank" in m or "exposure" in m)
    ]
    if ranking_methods:
        return  # the class grew ranking methods; the old sentence would be true again
    assert not re.search(r"regression,? and ranking tasks", intro), (
        "the FairnessAnalyzer entry says the class covers ranking tasks. It has no ranking "
        "method and refuses task_type='ranking'."
    )


# --- The checks, checked -----------------------------------------------------


class TestTheseChecksAreThemselvesChecked:
    """Reinstate the shipped defect, verbatim, and require each rule to go red."""

    SHIPPED_TASK_TYPE_DESC = (
        "One of <code>'classification'</code>, <code>'regression'</code>, "
        "<code>'ranking'</code>. Auto-detected if <code>None</code>."
    )

    def _page_with(self, old: str, new: str) -> str:
        html = PAGE.read_text(encoding="utf-8")
        assert old in html, (
            f"the text this control replaces is gone, so it would test nothing: {old[:60]!r}"
        )
        return html.replace(old, new, 1)

    def test_the_shipped_ranking_description_is_caught(self) -> None:
        current = documented_params(PAGE.read_text(encoding="utf-8"))["task_type"]
        html = self._page_with(current, self.SHIPPED_TASK_TYPE_DESC)
        desc = documented_params(html)["task_type"]
        assert "ranking" in _LITERAL_RE.findall(desc)
        assert not _is_accepted("task_type", "ranking")
        assert not any(cue in desc for cue in _REFUSAL_CUES), (
            "the reinstated description accidentally carries a refusal cue, so this control "
            "would pass on the defect"
        )

    def test_a_deleted_parameter_row_is_caught(self) -> None:
        section = _analyzer_section(PAGE.read_text(encoding="utf-8"))
        match = re.search(
            r'<div class="api-param">\s*<div class="api-param-header">\s*'
            r'<span class="api-param-name">backend</span>.*?</div>\s*</div>',
            section,
            re.DOTALL,
        )
        assert match, "the backend row is not shaped as expected any more"
        html = self._page_with(match.group(0), "")
        assert "backend" not in documented_params(html)
        assert "backend" in constructor_params()

    def test_an_undocumented_accepted_value_is_caught(self) -> None:
        desc = "<code>'exclude'</code> only."
        quoted = set(_LITERAL_RE.findall(desc))
        assert sorted(accepted_values("missing_strategy") - quoted) == ["as_group", "error"]

    def test_the_accepted_set_comes_from_the_running_code(self) -> None:
        assert accepted_values("task_type") == {"classification", "regression"}
        assert accepted_values("backend") == {"auto", "native"}
        assert accepted_values("missing_strategy") == {"exclude", "as_group", "error"}

    def test_a_value_the_code_accepts_is_not_reported_as_refused(self) -> None:
        assert _is_accepted("task_type", "classification")
        assert _is_accepted("backend", "native")
        assert not _is_accepted("backend", "fairlearn")
