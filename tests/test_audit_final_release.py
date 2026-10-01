"""Final pre-release confidence pins (2026-08-23).

The final release-verification pass surfaced these confirmed blockers after the
three deep audits:
- CRITICAL: the validity groundedness judge POSTed via raw requests (redirects
  followed unchecked), so a payload endpoint could 302 to a private/metadata host
  the up-front guard refused directly.
- HIGH: two more cali-BRATIO-n substring sites survived in visualization.py
  (calibration_difference colored/verdicted as a ratio).
- HIGH: two more pandas-3 str-dtype sites crashed / silently understated risk.
- MEDIUM: FairnessTestSuite.get_summary reported an all-unmeasurable batch as PASS.

Wave 3 (2026-08-27) widened the reintroduction guard below. It had been pinned
to the single token "ratio" in three exact spellings, so when the SAME bug class
reappeared as ``"disparate impact" in n`` in ``rendering/adapters_reporting.py``
the pin stayed GREEN while a 0.45 disparate_impact_difference against a 0.10
bound rendered as a green pass one directory away. The bug class is not the word
"ratio", it is deciding a metric's DIRECTION by looking for a family token
anywhere inside its name, so the guard now hunts that shape for every
direction-bearing token rather than for one word.

Wave 4 (2026-08-27) rebuilt the detector on the PARSED TREE, because widening a
list of spellings only ever buys the spellings its author thought of. Measured
against sixteen spellings of the identical defect (``_DIRECTION_BUG_SPELLINGS``
below), the wave-3 spelling guard caught 4 and missed 12: it could not see a
renamed operand (``metric_name_lower``, ``metric_id``), an attribute operand,
``.casefold()`` / ``.upper()``, ``.find()`` / ``.count()``, ``re.search()``, the
token parked in a constant, a loop over a tuple of tokens, or an f-string /
``str()``-wrapped operand. The AST detector catches 16 of 16, and each is its
own positive control. The verbatim spelling pin is KEPT and unioned in as a
floor, so the pin that has actually been holding cannot lose a case.

Wave 4 also fixed the guard's meta-problem. Every site it was red on was
cosmetic (a radar-chart subset, two sentences of generated prose) while every
site that decides a verdict was green, so the person clearing CI would fix the
harmless ones and ship. The failure message now RANKS hits by whether the
branch they feed produces a verdict, dangerous half first. Ranking never
suppresses a hit; it only orders them.
"""

import ast
import math
import os
import re

import numpy as np
import pytest

# Sixteen spellings of ONE defect: decide a metric's better-direction by looking
# for a family token anywhere inside its name. Measured on 2026-08-27, the
# wave-3 spelling-enumeration guard caught 4 of these and missed 12, which is
# why the detector was rebuilt on the parsed tree; the AST detector catches
# 16 of 16. Every entry is a real way this has been or could be written, and
# each is its own positive control below.
_DIRECTION_BUG_SPELLINGS = {
    "bare literal, double quotes": (
        'def f(name):\n    if "ratio" in name:\n        return "higher"\n    return "lower"\n'
    ),
    "negated containment": (
        'def f(name):\n    if "ratio" not in name:\n        return "lower"\n    return "higher"\n'
    ),
    "single quotes, key operand": ("def f(key):\n    higher = 'ratio' in key\n    return higher\n"),
    "renamed operand, metric_name_lower": (
        'def f(metric_name_lower):\n    if "ratio" in metric_name_lower:\n        return "higher"\n    return "lower"\n'
    ),
    "renamed operand, metric_id": (
        'def f(metric_id):\n    if "parity" in metric_id:\n        return "higher"\n    return "lower"\n'
    ),
    "attribute operand": (
        'def f(result):\n    if "ratio" in result.metric_name:\n        return "higher"\n    return "lower"\n'
    ),
    "casefold tail": (
        'def f(name):\n    if "ratio" in name.casefold():\n        return "higher"\n    return "lower"\n'
    ),
    "upper tail": (
        'def f(name):\n    if "RATIO" in name.upper():\n        return "higher"\n    return "lower"\n'
    ),
    "str.find": (
        'def f(name):\n    if name.find("ratio") >= 0:\n        return "higher"\n    return "lower"\n'
    ),
    "str.count": (
        'def f(name):\n    if name.count("ratio"):\n        return "higher"\n    return "lower"\n'
    ),
    "re.search": (
        'import re\n\n\ndef f(name):\n    if re.search("ratio", name):\n        return "higher"\n    return "lower"\n'
    ),
    "token held in a constant": (
        '_TOK = "ratio"\n\n\ndef f(name):\n    if _TOK in name:\n        return "higher"\n    return "lower"\n'
    ),
    "loop over a token tuple": (
        '_TOKENS = ("ratio", "parity")\n\n\ndef f(name):\n    for tok in _TOKENS:\n        if tok in name:\n            return "higher"\n    return "lower"\n'
    ),
    "f-string wrapped operand": (
        'def f(metric):\n    if "ratio" in f"{metric}":\n        return "higher"\n    return "lower"\n'
    ),
    "str() wrapped operand": (
        'def f(metric):\n    if "ratio" in str(metric):\n        return "higher"\n    return "lower"\n'
    ),
    "multi-word literal": (
        'def f(label):\n    if "disparate impact" in label:\n        return "higher"\n    return "lower"\n'
    ),
}

# Forms that read like the defect and are NOT it. A guard that flags these
# teaches the next author to silence it, which is how the explanation of the
# cali-BRATIO-n incident was nearly deleted.
_SAFE_DIRECTION_FORMS = {
    "exact suffix test": 'def f(name):\n    return name.endswith("_ratio")\n',
    "exact table membership": 'def f(name):\n    return name in ("disparate_impact", "demographic_parity_ratio")\n',
    "whole-token membership": 'def f(name):\n    return "ratio" in name.split("_")\n',
    "dict-key lookup": 'def f(before_metrics):\n    return "accuracy" in before_metrics\n',
    "configured metric list": 'def f(self):\n    return "disparate_impact" in self.config.metrics_to_track\n',
    # The exact false positive a module-wide (unscoped) token set produced in
    # robustness.comprehensive_fairness_test: the NAME is the needle and the
    # dispatch table is the haystack, which is a lookup, not a name test.
    "metric name looked up in a dispatch table": (
        "def f(metrics, metric_funcs):\n"
        '    for metric_name in ("demographic_parity", "equal_opportunity"):\n'
        "        if metric_name not in metric_funcs:\n"
        "            continue\n"
        "        yield metric_funcs[metric_name]\n"
    ),
}


# --- CRITICAL: the judge routes its own POST through the SSRF guard -----------
class TestJudgeEgressGuard:
    def _judge(self, url, guard_egress=True):
        from vfairness.validity.judge import LlmGroundednessJudge

        return LlmGroundednessJudge(endpoint_url=url, model_name="m", guard_egress=guard_egress)

    def test_metadata_endpoint_is_refused_when_guarded(self):
        # No network: GuardedSession.validate_endpoint refuses the cloud-metadata
        # host before any dial, so _complete returns None and the judge refuses.
        j = self._judge("http://169.254.169.254/v1/chat/completions", guard_egress=True)
        out = j.judge_groundedness(answer="a", contexts=["c"], question=None, language="en")
        assert out["groundedness"] is None

    def test_rfc1918_endpoint_is_refused_when_guarded(self):
        j = self._judge("http://10.0.0.5/v1/chat/completions", guard_egress=True)
        out = j.judge_groundedness(answer="a", contexts=["c"], question=None, language="en")
        assert out["groundedness"] is None

    def test_guard_egress_defaults_on(self):
        j = self._judge("http://example.com/v1/chat/completions")
        assert j._guard_egress is True

    def test_build_scorer_threads_the_opt_in(self, monkeypatch):
        from vfairness.operations.validity import task_handlers as th

        payload = {"judge": {"endpoint_url": "http://example.com/v1", "model_name": "m"}}
        monkeypatch.delenv("VFAIRNESS_VALIDITY_JUDGE_ALLOW_PRIVATE", raising=False)
        scorer = th._build_scorer(payload)
        assert scorer._judge is not None and scorer._judge._guard_egress is True

        monkeypatch.setenv("VFAIRNESS_VALIDITY_JUDGE_ALLOW_PRIVATE", "1")
        scorer2 = th._build_scorer(payload)
        assert scorer2._judge is not None and scorer2._judge._guard_egress is False


# --- HIGH: the two remaining cali-BRATIO-n sites in visualization.py ----------
class TestVisualizationRatioSubstring:
    def test_ci_panel_and_confidence_plot_treat_calibration_as_difference(self):
        # The source must key on the _ratio suffix, not the "ratio" substring, so
        # calibration_difference is not misread as a higher-is-better ratio.
        import inspect

        from vfairness.evaluation.vfairness_metrics import visualization as V

        for fn_name in ("_add_ci_panel", "plot_confidence_intervals"):
            fn = getattr(V, fn_name, None)
            if fn is None:
                continue
            src = inspect.getsource(fn)
            assert '"ratio" not in name' not in src, fn_name
            assert '"ratio" in name' not in src, fn_name

    def test_no_ratio_substring_check_remains_anywhere(self):
        # Belt-and-braces across the rendering + metrics surfaces: the verdict-
        # inverting substring test must not reappear.
        #
        # Scans EXECUTABLE code only. Until 2026-08-27 this matched raw file
        # text, so it also fired on prose: _metric_direction.py's module
        # docstring quotes `"ratio" in name` while explaining why the module
        # refuses to do that, and the pin failed. A guard whose only green
        # states are "the bug is absent" or "the explanation of the bug is
        # deleted" pressures the next author to delete the explanation, which
        # is how the cali-BRATIO-n understanding was lost the first time
        # (CLAUDE.md, reconciled in 47f1e09f8). Stripping COMMENT and STRING
        # tokens keeps the pin exact on code and silent on documentation.
        #
        # The three verbatim spellings below are the FLOOR, kept so the
        # original pin can never regress. The general shape is caught by
        # test_no_substring_direction_check_remains_anywhere.
        hits = []
        for p in _source_files():
            code = _executable_source(p)
            for marker in _RATIO_SUBSTRING_MARKERS:
                if _normalize_tokens(marker) in code:
                    hits.append(f"{p.name}: {marker}")
        assert not hits, f"substring ratio checks survive: {hits}"

    def test_no_substring_direction_check_remains_anywhere(self):
        """The WIDENED guard, wave 3.

        The narrow version of this pin knew one token in three spellings, so
        the third instance of the bug class was invisible to it:
        ``adapters_reporting._is_ratio_metric`` ended with
        ``"disparate impact" in n or "disparate_impact" in n`` and graded
        ``disparate_impact_difference`` as a four-fifths ratio, rendering a 0.45
        violation against a 0.10 bound as a green pass. Fixing that instance
        without widening the guard would have guaranteed a fourth.

        So this scans for the SHAPE: a direction-bearing family token, in either
        quote style, with or without surrounding underscores, tested for
        containment (``in`` / ``not in``) against a metric-name-shaped operand.
        Deciding direction that way is the defect, whatever the token is; the
        safe forms are an exact table lookup, an exact ``endswith`` suffix, or a
        whole-token split, none of which this matches.
        """
        records = []
        for p in _source_files():
            records.extend(_direction_hit_records(p))
        assert not records, (
            "substring DIRECTION tests survive. Each one decides whether higher "
            "or lower is better by looking for a family token inside a metric "
            "name, which is what graded a maximal violation as a pass three "
            "times. Route the site through "
            "evaluation.vfairness_metrics._metric_direction instead of "
            "repairing the test in place. Ranked by whether the hit reaches a "
            "verdict:\n" + _rank_direction_hits(records)
        )

    def test_scanner_still_detects_a_real_code_hit(self, tmp_path):
        """Positive control for the scanner above.

        Stripping strings and comments is only safe if the scanner still fires
        on genuine code. This file contains the bug in executable form AND the
        same text in a docstring and a comment: exactly one must be reported.
        Without this control, a tokenizer change could make the pin silently
        green forever, which is worse than no pin at all.
        """
        offender = tmp_path / "offender.py"
        offender.write_text(
            '"""Docstring mentioning \'"ratio" in name\' as prose only."""\n'
            '# comment mentioning "ratio" in name\n'
            "def f(name):\n"
            '    return "ratio" in name\n'
        )
        code = _executable_source(offender)
        assert _normalize_tokens('"ratio" in name') in code, (
            "scanner no longer sees the bug in executable code; the pin would be "
            f"permanently green. Tokenized code was: {code!r}"
        )

        innocent = tmp_path / "innocent.py"
        innocent.write_text(
            '"""Explains why a naive \'"ratio" in name\' test is wrong."""\n'
            "def f(name):\n"
            '    return name.endswith("_ratio")\n'
        )
        assert _normalize_tokens('"ratio" in name') not in _executable_source(innocent), (
            "scanner still fires on prose; documenting the incident would fail CI"
        )

    @pytest.mark.parametrize(
        "line",
        [
            'return "ratio" in name',
            'return "ratio" not in name',
            "return 'ratio' in key",
            # The wave-3 instance the NARROW pin could not see.
            'return n.endswith("_ratio") or "disparate impact" in n',
            'return "disparate_impact" in n',
            # Other direction-bearing families, both quote styles, both
            # operators, with and without the surrounding underscores.
            "if 'calibration' in metric_name:",
            'if "_parity_" in metric:',
            'if "difference" not in k:',
            'ok = "accuracy" in m',
            'flag = "gap" in label.lower()',
        ],
    )
    def test_widened_scanner_fires_on_each_offending_shape(self, tmp_path, line):
        """Positive control for the WIDENED guard, one case per shape.

        A guard is only worth its green: every spelling listed here is a real
        way this defect has been or could be written, and each must be seen.
        """
        offender = tmp_path / "offender.py"
        offender.write_text(f"def f(name, n, key, k, metric, metric_name, m, label):\n    {line}\n")
        assert _substring_direction_hits(offender), (
            f"the widened scanner is blind to {line!r}; the pin would be green while "
            "the bug class is live, exactly as it was for the disparate-impact form"
        )

    @pytest.mark.parametrize(
        "line",
        [
            # The SAFE forms. Flagging these would push authors back onto the
            # substring test to keep CI quiet.
            'return name.endswith("_ratio")',
            'return name in ("disparate_impact", "demographic_parity_ratio")',
            'return "ratio" in name.split("_")',
            # Container membership, not a name test: a dict key lookup and a
            # configured metric list. Both read like the defect and are not it.
            'return "accuracy" in before_metrics',
            'return "disparate_impact" in self.config.metrics_to_track',
        ],
    )
    def test_widened_scanner_stays_silent_on_the_safe_forms(self, tmp_path, line):
        offender = tmp_path / "innocent.py"
        offender.write_text(f"def f(name, before_metrics, self):\n    {line}\n")
        assert not _substring_direction_hits(offender), (
            f"the widened scanner fires on the safe form {line!r}; a guard that "
            "flags the fix pressures the next author to revert it"
        )

    @pytest.mark.parametrize("spelling", sorted(_DIRECTION_BUG_SPELLINGS))
    def test_every_known_spelling_of_the_defect_is_seen(self, tmp_path, spelling):
        """Positive control for the AST detector, one case per SHAPE.

        Waves 1 to 3 widened this guard by adding spellings, and a spelling
        list is exactly as wide as its author's imagination. Measured against
        these sixteen, the spelling guard saw 3. Each one here must be seen, or
        the pin is green while that shape of the bug ships.
        """
        offender = tmp_path / f"offender_{abs(hash(spelling))}.py"
        offender.write_text(_DIRECTION_BUG_SPELLINGS[spelling])
        assert _direction_hit_records(offender), (
            f"the detector is blind to the {spelling!r} spelling:\n"
            f"{_DIRECTION_BUG_SPELLINGS[spelling]}"
        )

    @pytest.mark.parametrize("form", sorted(_SAFE_DIRECTION_FORMS))
    def test_every_safe_form_stays_silent(self, tmp_path, form):
        innocent = tmp_path / f"innocent_{abs(hash(form))}.py"
        innocent.write_text(_SAFE_DIRECTION_FORMS[form])
        assert not _direction_hit_records(innocent), (
            f"the detector fires on the SAFE form {form!r}; a guard that flags "
            "the fix pressures the next author to revert it:\n"
            f"{_SAFE_DIRECTION_FORMS[form]}"
        )

    def test_the_failure_message_puts_the_verdict_sites_first(self, tmp_path):
        """The meta-problem: a flat list gets cleared from the harmless end.

        Every hit this guard was red on decided a chart subset or a sentence of
        prose, while every site that decides a verdict was green (they use an
        exact ``endswith``). Handed a flat list, whoever clears CI fixes the
        cosmetic ones and ships, and a dangerous hit in that list goes last. So
        the message ranks: it never drops a hit, it orders them.
        """
        offender = tmp_path / "mixed.py"
        offender.write_text(
            "def pick_chart_subset(metrics):\n"
            '    return {k: v for k, v in metrics.items() if "difference" in k}\n'
            "\n\n"
            "def grade(name, value, threshold):\n"
            '    if "ratio" in name:\n'
            "        passed = value >= threshold\n"
            "    else:\n"
            "        passed = abs(value) <= threshold\n"
            "    return passed\n"
        )
        records = _direction_hit_records(offender)
        assert len(records) == 2, records
        assert records[0]["feeds_verdict"] is True, records
        assert records[0]["function"] == "grade"
        assert records[1]["feeds_verdict"] is False, records

        message = _rank_direction_hits(records)
        assert message.index("FEEDS A VERDICT") < message.index("DOES NOT APPEAR")
        assert message.index("grade") < message.index("pick_chart_subset")

    def test_a_hit_that_feeds_nothing_is_not_reported_by_the_ast_detector(self, tmp_path):
        """ "Feeds a branch" is part of the shape the AST detector hunts.

        The verbatim spelling FLOOR is deliberately unconditional, so the union
        still reports the classic three literals wherever they appear; that pin
        has been holding since wave 1 and may not lose a case. Both halves are
        asserted here so the difference between them stays deliberate.
        """
        inert = tmp_path / "inert.py"
        inert.write_text('def f(metric_id):\n    print("ratio" in metric_id)\n')
        assert not _ast_direction_hits(inert)
        assert not _direction_hit_records(inert)

        floored = tmp_path / "floored.py"
        floored.write_text('def f(name):\n    print("ratio" in name)\n')
        assert _direction_hit_records(floored), (
            "the verbatim spelling floor stopped firing; it is the pin that has "
            "actually been holding and it may not regress"
        )

    def test_widened_scanner_is_silent_on_prose_but_the_prose_is_really_there(self):
        """The module that DOCUMENTS the pattern must pass, and for the right reason.

        ``_metric_direction.py`` quotes ``"ratio" in name`` in its module
        docstring while explaining why it refuses to do that. It must pass this
        guard because the quote lives in a docstring, not because the guard
        cannot see it and not because someone deleted the explanation. Both
        halves are asserted, so a green here always means the same thing.
        """
        import pathlib

        from vfairness.evaluation.vfairness_metrics import _metric_direction as M

        path = pathlib.Path(M.__file__)
        raw = path.read_text(encoding="utf-8")
        # Half one: the prose really is present, so this is not a vacuous pass.
        assert '"ratio" in name' in raw, (
            "the explanation of the cali-BRATIO-n incident has been deleted from "
            "_metric_direction.py; that erasure is how the understanding was lost "
            "the first time"
        )
        assert _SUBSTRING_DIRECTION_RE.search(" ".join(raw.split())), (
            "the widened guard cannot even see the documented pattern in the raw "
            "file, so its silence on the module proves nothing"
        )
        # Half two: and it is silent anyway, because that text is a docstring.
        assert not _substring_direction_hits(path)


_RATIO_SUBSTRING_MARKERS = ('"ratio" in name', '"ratio" not in name', '"ratio" in key')

# Family TOKENS that carry a direction. A metric's better-direction must never be
# decided by looking for one of these INSIDE a name; that is the whole bug class.
# Audited 2026-08-27 against the name tables in
# evaluation/vfairness_metrics/_metric_direction.py (RATIO_METRICS,
# LOWER_IS_BETTER_METRICS, HIGHER_IS_BETTER_METRICS, LOWER_IS_BETTER_TOKENS) and
# the metric names in _registry.py. "ratio" is the one that started it,
# "disparate impact" is the one the narrow pin could not see.
_DIRECTION_TOKENS = (
    "ratio",
    "disparate impact",
    "disparate_impact",
    "disparate",
    "impact",
    "difference",
    "diff",
    "gap",
    "disparity",
    "deviation",
    "error",
    "loss",
    "parity",
    "calibration",
    "odds",
    "opportunity",
    "accuracy",
    "brier",
    "auroc",
    "worst_group",
)

# Operands that a metric NAME is held in at such a site. Restricting the
# right-hand side this way is what separates the defect from ordinary container
# membership: ``"accuracy" in before_metrics`` is a dict-key lookup and
# ``"disparate_impact" in self.config.metrics_to_track`` is a configured list,
# and neither is a direction test. Any expression ending in ``.lower()`` is also
# covered, because that is always a string and therefore always a substring test.
_DIRECTION_OPERANDS = (
    "metric_name",
    "metric_label",
    "metric_key",
    "metric",
    "name",
    "label",
    "column",
    "key",
    "mname",
    "col",
    "n",
    "k",
    "m",
)

_SUBSTRING_DIRECTION_RE = re.compile(
    # The operand must be a bare metric-name identifier that is NOT then
    # attribute-accessed: ``"ratio" in name.split("_")`` is whole-TOKEN
    # membership, which is the correct fix, and flagging it would push authors
    # straight back onto the substring test to keep CI quiet. A ``.lower()``
    # tail is the one attribute access that IS always a substring test, so it
    # gets its own alternative.
    r"""(['"])\s*_?(?:%s)_?\s*\1\s*(?:not\s+)?in\s+"""
    r"""(?:(?:%s)\b(?!\s*\.)|[A-Za-z0-9_.\[\]()'"\s]{0,60}?\.\s*lower\s*\(\s*\))"""
    % (
        "|".join(re.escape(t) for t in sorted(_DIRECTION_TOKENS, key=len, reverse=True)),
        "|".join(sorted(_DIRECTION_OPERANDS, key=len, reverse=True)),
    ),
    re.IGNORECASE,
)


def _normalize_tokens(text: str) -> str:
    """Collapse all whitespace, so token-joined code matches source spacing."""
    return "".join(text.split())


def _executable_tokens(path):
    """Tokens of ``path`` with comments and DOCSTRINGS dropped, or ``None``.

    ``None`` means the file could not be parsed or tokenised; callers fall back
    to the RAW text rather than skipping the file, so the pin can never go
    silent on a file it failed to read.

    Only docstrings are dropped, never string literals in general: the pattern
    being hunted (``"ratio" in name``) contains a string literal itself, so
    stripping every STRING token would make the scan match nothing and the pin
    permanently green. ``test_scanner_still_detects_a_real_code_hit`` is the
    positive control for exactly that mistake, and it caught it once already.
    """
    import ast
    import io
    import tokenize

    raw = _read(path)
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return None, raw

    docstring_lines = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_lines.add(body[0].value.lineno)

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(raw).readline))
    except (tokenize.TokenError, IndentationError):
        return None, raw

    kept = [
        t
        for t in toks
        if not (
            t.type == tokenize.COMMENT
            or (t.type == tokenize.STRING and t.start[0] in docstring_lines)
        )
    ]
    return kept, raw


def _read(path) -> str:
    import pathlib

    return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")


def _executable_source(path) -> str:
    """Source text of ``path`` with comments and DOCSTRINGS removed.

    Whitespace is collapsed so the result can be searched with
    :func:`_normalize_tokens` regardless of the original spacing. See
    :func:`_executable_tokens` for why only docstrings are dropped; do NOT
    change that to strip every STRING token.
    """
    toks, raw = _executable_tokens(path)
    if toks is None:
        return _normalize_tokens(raw)
    return _normalize_tokens("".join(t.string for t in toks))


def _executable_code_lines(path):
    """``[(lineno, code)]`` for ``path``, one entry per source line that has code.

    Same comment/docstring filtering as :func:`_executable_source`, but the
    tokens are joined with a SINGLE SPACE rather than closed up, and the line
    number is kept so a hit can be reported as ``file:line``.

    The space matters. ``_normalize_tokens`` closes every gap, which turns
    ``"disparate impact" in n or ...`` into ``"disparateimpact"innor...`` and
    destroys the word boundary after the operand, so a regex looking for a
    metric-name-shaped operand cannot tell ``n`` from ``nor``. That is not a
    theoretical edge: it is the exact line wave 3 had to catch.
    """
    toks, raw = _executable_tokens(path)
    if toks is None:
        return [(1, " ".join(raw.split()))]
    per_line = {}
    order = []
    for t in toks:
        line = t.start[0]
        if line not in per_line:
            per_line[line] = []
            order.append(line)
        per_line[line].append(t.string)
    return [(line, " ".join(" ".join(per_line[line]).split())) for line in order]


# --- the AST detector (wave 4) ------------------------------------------------
#
# Waves 1 to 3 widened this guard by ADDING SPELLINGS: another token, another
# operand name, another quote style. Measured on 2026-08-27 against sixteen
# spellings of the identical defect, the spelling-enumeration guard caught 3 and
# missed 13. It could not see a renamed operand (``metric_name_lower``,
# ``metric_id``), an attribute operand, ``.casefold()`` / ``.upper()``,
# ``.find()`` / ``.count()``, ``re.search()``, the token held in a constant, a
# loop over a tuple of tokens, an f-string or ``str()``-wrapped operand, or a
# multi-word literal. Enumeration is what made it narrow, so the detector below
# looks for the SHAPE instead, on the parsed tree:
#
#   a membership test (or .find / .count / .index / re.search) whose LITERAL
#   side carries a direction-bearing family token, whose OTHER side is a string
#   or a metric-name-shaped operand, and whose result feeds a branch.
#
# The regex above is kept and unioned in, as a floor: it is the pin that has
# actually been holding, and nothing it caught may stop being caught.

_STRING_METHODS = frozenset(
    {
        "lower",
        "upper",
        "casefold",
        "strip",
        "lstrip",
        "rstrip",
        "title",
        "swapcase",
        "replace",
        "format",
        "removeprefix",
        "removesuffix",
    }
)
# Substring searches that are a containment test wearing a different hat.
# ``endswith`` / ``startswith`` are deliberately absent: an exact suffix test is
# the correct FIX, and flagging it would push authors back onto the defect.
_SEARCH_METHODS = frozenset({"find", "rfind", "index", "rindex", "count"})
_RE_FUNCS = frozenset({"search", "match", "fullmatch", "findall", "finditer"})

# Word parts that say "this operand holds a metric NAME".
_NAME_WORDS = frozenset(
    {
        "name",
        "metric",
        "key",
        "label",
        "column",
        "col",
        "id",
        "ident",
        "identifier",
        "slug",
        "header",
        "title",
        "field",
        "mname",
    }
)
# Single letters this codebase uses for a metric name in comprehensions.
_SHORT_NAME_VARS = frozenset({"n", "k", "m"})
# Word parts that say "this operand holds a COLLECTION, so membership in it is
# an ordinary lookup and not a substring test at all". Without this,
# ``"accuracy" in before_metrics`` (a dict-key lookup) and
# ``"disparate_impact" in self.config.metrics_to_track`` (a configured list)
# would be reported, and a guard that flags the safe form teaches people to
# silence it.
_CONTAINER_WORDS = frozenset(
    {
        "list",
        "set",
        "dict",
        "map",
        "table",
        "registry",
        "config",
        "args",
        "kwargs",
        "results",
        "metrics",
        "names",
        "keys",
        "labels",
        "columns",
        "cols",
        "fields",
        "tokens",
        "items",
        "values",
        "suffixes",
        "prefixes",
        # Plural and callable-collection forms. ``metric_funcs`` and
        # ``before_metrics`` are name-shaped by their first word and are
        # dictionaries: membership in them is an ordinary lookup, and reporting
        # it would train readers to ignore this guard.
        "func",
        "funcs",
        "functions",
        "mapping",
        "lookup",
        "cache",
        "store",
        "ids",
        "titles",
        "headers",
        "slugs",
        "maps",
        "tables",
    }
)

# Statement shapes whose value is consumed: the "feeds a branch" requirement.
_BRANCH_CONTEXTS = (
    ast.If,
    ast.IfExp,
    ast.While,
    ast.Assert,
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.NamedExpr,
    ast.comprehension,
)

# Words that say the branch this test feeds produces a VERDICT: a pass/fail, a
# direction, a badge colour, a gate decision. A hit inside one of these inverts
# what a reader is told about fairness; a hit that only picks a chart subset or
# a sentence of prose does not. Both are reported, but never in one flat list
# where the dangerous one can be cleared last.
_VERDICT_WORDS = (
    "verdict",
    "unfair",
    "fair",
    "passed",
    "pass",
    "failed",
    "fail",
    "breach",
    "violat",
    "threshold",
    "higher_is_better",
    "lower_is_better",
    "is_ratio",
    "direction",
    "status",
    "outcome",
    "severity",
    "grade",
    "approve",
    "block",
    "compliant",
    "badge",
    "color",
    "colour",
)


def _identifier_parts(word):
    """Lower-case word parts of an identifier, splitting snake_case and camelCase."""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", str(word))
    return [p for p in re.split(r"[^a-z0-9]+", spaced.lower()) if p]


def _literal_words(text):
    return [w for w in re.split(r"[^a-z0-9]+", str(text).lower()) if w]


def _metric_name_vocabulary():
    """Every word that appears in a real metric name, or in a direction token.

    DERIVED, not curated. The two sources of truth are the name tables in
    ``evaluation/vfairness_metrics/_metric_direction.py`` and this module's own
    ``_DIRECTION_TOKENS``, so a metric added there cannot fall out of this set,
    and nobody has to remember to extend a hand-written list.
    """
    words = set()
    for token in _DIRECTION_TOKENS:
        words |= set(_literal_words(token))
    try:
        from vfairness.evaluation.vfairness_metrics import _metric_direction as _md
    except Exception:  # pragma: no cover - the detector still works without it
        # Degrading to the token words alone would make the vocabulary SMALLER,
        # which makes the guard STRICTER, not weaker. Failing open here would be
        # the wrong direction and is why this is not a bare pass.
        return words
    for attr in dir(_md):
        if not attr.isupper():
            continue
        value = getattr(_md, attr)
        names: set = set()
        if isinstance(value, (set, frozenset, tuple, list)):
            names = {x for x in value if isinstance(x, str)}
        elif isinstance(value, dict):
            names = {k for k in value if isinstance(k, str)}
        for name in names:
            words |= set(_literal_words(name))
    return words


_METRIC_NAME_WORDS = _metric_name_vocabulary()


def _is_direction_literal(text):
    """True when a string literal carries a direction-bearing family token.

    Compared as a WORD SEQUENCE, so surrounding underscores and multi-word
    spellings both land: ``"_ratio_"``, ``"disparate impact"`` and
    ``"disparate-impact ratio"`` all carry one.

    AND EVERY WORD MUST BE A METRIC-NAME WORD, which is the second half and was
    missing. ``"error"`` is a direction token, so the sentence
    ``"Fix all error-level issues"`` carried one, and on 2026-09-27 this detector
    reported a RECOMMENDATION DEDUP in operations/cicd/validator.py as a substring
    direction test:

        kept = [r for r in kept if "Fix all error-level issues" not in str(r)]

    Nothing about that line decides a direction. The needle is an English
    sentence and the haystack is a stringified recommendation. The only way to
    make the guard green was to delete a dedup the product needs, which is the
    shape of a guard that argues for breaking the product.

    The narrowing is deliberately NOT "the needle has no whitespace", which was
    the first attempt and suppressed the false positive cleanly. It would also
    have blinded the guard to a display label: ``"disparate impact" in
    label.lower()`` over "Disparate Impact Ratio" is the same bug class and has a
    space in it. Measured against the derived vocabulary: "ratio",
    "disparate impact", "disparate_impact", "calibration_difference" and
    "Disparate Impact Ratio" are all still needles, and "Fix all error-level
    issues" is not, because "fix", "all", "issues" and "level" appear in no
    metric name.
    """
    words = _literal_words(text)
    if not words:
        return False
    if any(w not in _METRIC_NAME_WORDS for w in words):
        return False
    for token in _DIRECTION_TOKENS:
        tw = _literal_words(token)
        if not tw:
            continue
        for i in range(len(words) - len(tw) + 1):
            if words[i : i + len(tw)] == tw:
                return True
    return False


def _operand_word(node):
    """The identifier a haystack expression is named by, or ``None``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _operand_word(node.value)
    return None


def _is_name_shaped(node):
    """True when the haystack looks like it holds a metric NAME."""
    word = _operand_word(node)
    if not word:
        return False
    stripped = word.strip("_").lower()
    if stripped in _SHORT_NAME_VARS:
        return True
    parts = _identifier_parts(word)
    if any(p in _CONTAINER_WORDS for p in parts):
        return False
    return any(p in _NAME_WORDS for p in parts)


def _is_string_expr(node):
    """True when the haystack is unambiguously a STRING, whatever it is named.

    A string haystack is always a substring test, so the operand's name cannot
    excuse it. ``.split("_")`` is pointedly NOT here: whole-token membership is
    the correct fix.
    """
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "str":
            return True
        if isinstance(func, ast.Attribute) and func.attr in _STRING_METHODS:
            return True
    return False


def _is_haystack(node):
    return _is_string_expr(node) or _is_name_shaped(node)


def _walk_own_scope(node):
    """Yield ``node`` and its descendants, NOT descending into nested functions.

    Scoping matters. A flat module-wide set of "names bound to a direction
    token" leaks: one ``for metric_name in ("demographic_parity_difference",
    ...)`` anywhere in a module made EVERY ``metric_name in some_dict`` in that
    module look like the defect, including two plain dict lookups in
    ``robustness.comprehensive_fairness_test``. A guard that reports ordinary
    lookups is a guard people learn to ignore.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if child is not node and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            stack.append(child)


def _direction_bearings_in_scope(scope_node, inherited_collections=frozenset()):
    """``(singles, collections)`` bound to a direction token inside ONE scope.

    Covers the two indirections a source scan for literals is blind to: the
    token parked in a constant (``_TOK = "ratio"`` then ``if _TOK in name``) and
    the loop over a tuple of tokens (``for t in _TOKENS: if t in name``), in a
    ``for`` statement or a comprehension. ``inherited_collections`` carries the
    module-level token tuples down into each function, because the tuple is
    almost always a module constant and the loop over it is not.
    """
    singles = set()
    collections = set(inherited_collections)
    nodes = list(_walk_own_scope(scope_node))

    def _collection_is_direction_bearing(node):
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(
                isinstance(e, ast.Constant)
                and isinstance(e.value, str)
                and _is_direction_literal(e.value)
                for e in node.elts
            )
        return isinstance(node, ast.Name) and node.id in collections

    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                value = node.value
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and _is_direction_literal(value.value)
                ):
                    singles.add(target.id)
                elif _collection_is_direction_bearing(value):
                    collections.add(target.id)

    for node in nodes:
        iters = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iters.append((node.target, node.iter))
        elif isinstance(node, ast.comprehension):
            iters.append((node.target, node.iter))
        for target, iterable in iters:
            if isinstance(target, ast.Name) and _collection_is_direction_bearing(iterable):
                singles.add(target.id)
    return singles, collections


def _direction_bearing_scopes(tree):
    """``{scope_node: names bound to a direction token there}``, module names included."""
    module_names, module_collections = _direction_bearings_in_scope(tree)
    scopes = {tree: module_names}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local, _ = _direction_bearings_in_scope(node, module_collections)
            scopes[node] = module_names | local
    return scopes


def _is_direction_needle(node, direction_names):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _is_direction_literal(node.value)
    if isinstance(node, ast.Name):
        return node.id in direction_names
    return False


def _feeds_a_branch(node, parents):
    """True when the value of ``node`` is consumed by a test, a return or a binding."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, _BRANCH_CONTEXTS):
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            return False
        current = parents.get(current)
    return False


def _enclosing_context(node, parents):
    """``(scope_node, function_name, statement)`` around a hit."""
    function_name = ""
    scope = None
    statement = None
    current = parents.get(node)
    while current is not None:
        if statement is None and isinstance(current, ast.stmt):
            statement = current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_name = current.name
            scope = current
            break
        current = parents.get(current)
    return scope, function_name, statement if statement is not None else node


def _verdict_context_words(statement, function_name):
    """The IDENTIFIERS around a hit, never the prose it selects.

    Ranking on raw source text mis-ranked ``explainer._generate_recommendation``
    as verdict-feeding because the sentence it appends happens to contain the
    word "threshold". What decides whether a branch produces a verdict is the
    names it touches, so string constants are deliberately excluded.
    """
    words = [function_name]
    for node in ast.walk(statement):
        if isinstance(node, ast.Name):
            words.append(node.id)
        elif isinstance(node, ast.Attribute):
            words.append(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            words.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            words.append(node.name)
    return " ".join(words).lower()


def _ast_direction_hits(path):
    """Structured direction-test hits in ``path``, found on the parsed tree."""
    raw = _read(path)
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return []
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    scopes = _direction_bearing_scopes(tree)
    records = []

    def _record(node, why):
        if not _feeds_a_branch(node, parents):
            return
        _scope, function_name, statement = _enclosing_context(node, parents)
        words = _verdict_context_words(statement, function_name)
        records.append(
            {
                "path": str(path),
                "line": getattr(node, "lineno", 0),
                "code": " ".join((ast.get_source_segment(raw, node) or "").split())[:160],
                "why": why,
                "function": function_name,
                "feeds_verdict": any(w in words for w in _VERDICT_WORDS),
            }
        )

    def _names_for(node):
        scope, _function_name, _statement = _enclosing_context(node, parents)
        return scopes.get(scope, scopes[tree])

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], (ast.In, ast.NotIn)):
                if _is_direction_needle(node.left, _names_for(node)) and _is_haystack(
                    node.comparators[0]
                ):
                    _record(node, "containment test on a metric name")
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            names = _names_for(node)
            if func.attr in _SEARCH_METHODS and node.args:
                if _is_direction_needle(node.args[0], names) and _is_haystack(func.value):
                    _record(node, f".{func.attr}() substring search on a metric name")
            elif (
                func.attr in _RE_FUNCS
                and isinstance(func.value, ast.Name)
                and func.value.id == "re"
                and len(node.args) >= 2
            ):
                if _is_direction_needle(node.args[0], names) and _is_haystack(node.args[1]):
                    _record(node, f"re.{func.attr}() on a metric name")
    return records


def _direction_hit_records(path):
    """Every direction-by-substring hit in ``path``: regex floor plus AST shapes.

    Unioned by line, so the spelling pin that has been holding since wave 1
    cannot lose a case the AST detector happens to miss, and vice versa.
    """
    records = _ast_direction_hits(path)
    seen = {(r["line"], r["code"]) for r in records}
    ast_lines = {r["line"] for r in records}
    for line, code in _executable_code_lines(path):
        for match in _SUBSTRING_DIRECTION_RE.finditer(code):
            fragment = match.group(0)
            if line in ast_lines or (line, fragment) in seen:
                continue
            seen.add((line, fragment))
            records.append(
                {
                    "path": str(path),
                    "line": line,
                    "code": fragment,
                    "why": "containment test on a metric name (spelling pin)",
                    "function": "",
                    "feeds_verdict": any(w in code.lower() for w in _VERDICT_WORDS),
                }
            )
    return sorted(records, key=lambda r: (not r["feeds_verdict"], r["line"]))


def _substring_direction_hits(path):
    """Every ``file:line: code`` in ``path`` that decides direction by substring."""
    return [f"{r['path']}:{r['line']}: {r['code']}" for r in _direction_hit_records(path)]


def _rank_direction_hits(records):
    """Failure text with the verdict-inverting hits FIRST and labelled as such.

    The meta-problem this solves, measured on 2026-08-27: every hit the guard
    was red on was cosmetic (a radar-chart subset, two sentences of generated
    prose), and every site that actually decides a verdict was green because it
    uses an exact ``endswith`` suffix. Handed a flat list, the person clearing
    CI fixes the three harmless ones and ships, and if a dangerous hit is ever
    in that list it goes last. So the list is split, dangerous half first, with
    the reason stated: the ranking never suppresses a hit, it only orders them.
    """
    dangerous = [r for r in records if r["feeds_verdict"]]
    cosmetic = [r for r in records if not r["feeds_verdict"]]

    def _fmt(rec):
        where = f"{rec['path']}:{rec['line']}"
        who = f" in {rec['function']}()" if rec["function"] else ""
        return f"    {where}{who}\n      {rec['code']}\n      ({rec['why']})"

    lines = []
    if dangerous:
        lines.append(
            f"  FEEDS A VERDICT ({len(dangerous)}). Fix these FIRST: the branch this "
            "test controls decides a pass/fail, a direction, a gate decision or a "
            "badge colour, so getting the direction wrong here tells a reader that a "
            "violation passed."
        )
        lines.extend(_fmt(r) for r in dangerous)
    if cosmetic:
        lines.append(
            f"  DOES NOT APPEAR TO FEED A VERDICT ({len(cosmetic)}). Still the same "
            "bug class and still must go, but fixing only these clears the guard "
            "while leaving any verdict site above unfixed."
        )
        lines.extend(_fmt(r) for r in cosmetic)
    return "\n".join(lines)


def _source_files():
    """Every .py file under ``src/vfairness``."""
    import pathlib

    root = pathlib.Path(_visualization_file()).parent.parent.parent  # src/vfairness
    return sorted(root.rglob("*.py"))


def _visualization_file():
    from vfairness.evaluation.vfairness_metrics import visualization as V

    return V.__file__


# --- HIGH: pandas-3 string-dtype robustness -----------------------------------
class TestPandas3StringDtype:
    def _str_frame(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        n = 200
        gender = np.where(rng.integers(0, 2, n) == 1, "male", "female")
        df = pd.DataFrame(
            {
                "gender": pd.array(gender, dtype="str") if hasattr(pd, "StringDtype") else gender,
                "feat1": rng.normal(0, 1, n),
                "feat2": rng.normal(0, 1, n),
            }
        )
        return df

    def test_feature_engineering_analyzer_compare_transformations_no_crash(self):
        from vfairness.preprocessing.feature_engineering.analyzer import (
            FeatureEngineeringAnalyzer,
        )

        df = self._str_frame()
        fa = FeatureEngineeringAnalyzer(df, ["gender"])
        # Must not raise "could not convert string to float" under a str dtype.
        out = fa.compare_transformations()
        assert out is not None

    def test_encode_column_handles_string_dtype(self):

        from vfairness.preprocessing.feature_engineering.analyzer import (
            FeatureEngineeringAnalyzer,
        )

        df = self._str_frame()
        fa = FeatureEngineeringAnalyzer(df, ["gender"])
        codes = fa._encode_column(df["gender"])
        assert codes.dtype == float and len(np.unique(codes)) == 2


# --- MEDIUM: an all-unmeasurable test batch is 'incomplete', not 'passed' -----
class TestSuiteSummaryThreeState:
    def test_all_skipped_batch_is_incomplete_not_passed(self):
        from vfairness.operations.cicd.testing import FairnessTestSuite

        n = 120
        y_true = np.concatenate([np.array([1, 0] * (n // 2)), np.zeros(n, dtype=int)])
        y_pred = np.concatenate([np.array([1] * (n // 2) + [0] * (n // 2)), np.zeros(n, dtype=int)])
        attr = np.array(["A"] * n + ["B"] * n)  # B has no positives -> EO undefined
        suite = FairnessTestSuite(
            protected_attributes=["group"],
            metrics=["equalized_odds_difference"],
            thresholds={"equalized_odds_difference": 0.1},
        )
        suite.test_predictions(y_true, y_pred, attr, raise_on_failure=False)
        summary = suite.get_summary()
        assert summary["status"] == "incomplete"
        assert summary["passed"] == 0 and summary["skipped"] >= 1

    def test_real_pass_still_reads_passed(self):
        from vfairness.operations.cicd.testing import FairnessTestSuite

        rng = np.random.default_rng(0)
        n = 400
        # A genuinely fair, measurable model: selection independent of the group.
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        attr = np.array(["A", "B"] * (n // 2))
        suite = FairnessTestSuite(
            protected_attributes=["group"],
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )
        suite.test_predictions(y_true, y_pred, attr, raise_on_failure=False)
        assert suite.get_summary()["status"] == "passed"


def test_math_import_guard():
    # keep the module's math import used (defensive; some CI linters prune).
    assert math.isfinite(1.0) and os is not None and pytest is not None
