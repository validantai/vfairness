"""Guard: the metric computation core must not swallow errors silently (VB-SEC-4).

An audit-evidence tool must never hide a computation error. This test walks the
AST of the core computation subpackages and fails when an exception handler's
entire body discards the error and substitutes an answer.

WIDENED 2026-08-27 (register #20). The guard previously matched only the
``pass`` SPELLING, while its own docstring claimed it "protects the core where a
swallowed error would corrupt an audit verdict". It did not: an AST scan for
handlers whose body is a single ``return``/``continue``/``break`` found ten more
in the same three subpackages, including a live one that mattered.
``association_strength`` ended ``except Exception: return 0.0``, and a returned
zero is indistinguishable from a genuine measurement of no association. Executed
on a column that is a PERFECT proxy for race but stored as list-valued objects (a
JSON column, entirely realistic): association 0.0, ``identify_proxy_features``
returned ``[]``, and not one warning was raised. The same proxy as plain strings
scores 1.0. A pin that only catches the spelling of the bug it was written
against is the same failure as no pin at all.

The matched shapes are now: ``pass``; ``return`` of a constant, ``None``, or a
collection literal; ``continue``; and ``break``. A handler that warns, logs,
re-raises, or returns a computed value is NOT flagged: the objection is to
discarding the error and inventing an answer, not to recovering from it.

Optional-backend import guards (``except ImportError: pass`` and
``except ModuleNotFoundError: pass``) are allowed: they degrade a missing
optional dependency, they do not hide a computation result. Best-effort
subsystems outside the core (rendering fallbacks, reporting, pulse
orchestration, exploratory bias-detection scans) are intentionally out of
scope here and are tracked separately; this guard protects the core where a
swallowed error would corrupt an audit verdict.
"""

import ast
import os

import vfairness

# The subpackages whose results feed a fairness verdict directly.
CORE_SUBPACKAGES = ("evaluation", "in_processing", "post_processing")
_ALLOWED_EXC = {"ImportError", "ModuleNotFoundError"}

# Known-open occurrences. The key is (relative_path, exception_expression,
# shape), NOT the file alone and NOT the line number.
#
# File granularity was tried first and was WRONG. Exempting a whole file meant
# the file with the worst instance of this bug could never fail the guard again:
# reinstating `except Exception: return 0.0` in discovery.py left the guard
# green, because discovery.py was exempt for an unrelated dtype probe. That is
# the same "reports green while the defect is live" failure the widening exists
# to remove, so the exemption has to name the exact handler shape it blesses.
#
# The line number is deliberately excluded: pinning it would expire the entry on
# an unrelated edit above it and train the next person to re-bless it without
# reading it.
#
# The guard fails on ANYTHING not listed here, so this is a ratchet: existing
# debt is visible and cannot grow. Every entry is recorded in
# docs/audits/fourth-iteration-audit-2026-08-27.md under #20. Removing an entry
# is the fix; adding one needs a reason as specific as these.
_KNOWN_OPEN = {
    # Owner mid-flight (wave 3) on 2026-08-27; not mine to touch. Triage with
    # that work. The dict-literal returns are the `{'error': str(e)}` shape,
    # which at least carries the message rather than inventing a value.
    ("evaluation/vfairness_metrics/analyzer.py", "(TypeError, ValueError)", "return False"): (
        "dtype/shape probe; 'not that type' is a classification, not a discarded error"
    ),
    ("evaluation/vfairness_metrics/analyzer.py", "Exception", "return dict literal"): (
        "returns {'error': str(e)}, which surfaces the message; owned by a "
        "concurrent workstream, not yet triaged"
    ),
    # `np.issubdtype(s.dtype, np.number)` raising means the series has no usable
    # numeric dtype. "Not numeric" is the correct classification and the caller
    # routes to the nominal branch, which is the right branch.
    (
        "evaluation/vfairness_metrics/discovery.py",
        "(TypeError, AttributeError)",
        "return False",
    ): "_num dtype probe: an exception here genuinely means 'not numeric'",
    ("post_processing/calibration/methods.py", "np.linalg.LinAlgError", "break"): (
        "terminates an iterative fit on a singular matrix; the caller reports "
        "the iterations actually completed, so nothing is invented"
    ),
    # TWO ABSENCE CLASSIFIERS added by the first-examination wave, 2026-09-30, both
    # answering "is this value ABSENT?". When `pd.isna` cannot answer (a list, an
    # array, an arbitrary object) they return False, i.e. "this is content, keep
    # it". That is the deliberate direction and the opposite of a fabrication:
    # returning True would MANUFACTURE ABSENCE and silently drop a caller's data,
    # which is the mirror of the defect fixed throughout that wave. The same day a
    # literal 'None' demographic label was deliberately kept and disclosed rather
    # than dropped, for the same reason. Each is keyed to its exact handler shape,
    # so reinstating a substitution anywhere else in these files still fails.
    (
        "evaluation/vfairness_metrics/explainer.py",
        "(ImportError, TypeError, ValueError)",
        "return False",
    ): (
        "absence classifier: an object pd.isna cannot answer is content, not absence; "
        "True here would manufacture absence and drop data"
    ),
    (
        "post_processing/calibration/visualization.py",
        "(TypeError, ValueError)",
        "return False",
    ): (
        "absence classifier for group labels: an unanswerable object is a real "
        "label; True would mint absence and drop a group from the comparison"
    ),
    # REMOVED, BGL5 wave 4 (2026-09-30):
    #   ("post_processing/calibration/tradeoffs.py", "Exception", "return dict literal")
    # That handler is ``mitigation_pareto``'s catch-all. It now warns before it
    # returns, and its returned dict carries the four disclosure counts the two
    # other return paths of that function promise on "every return path", held at
    # None where a count was never reached rather than absent. The entry is DELETED
    # rather than left in place because this guard's own message says a stale
    # exemption re-blesses the next regression in that file silently.
}


def _src_root():
    return os.path.dirname(os.path.abspath(vfairness.__file__))


def _discarding_shape(stmt):
    """Name the shape if ``stmt`` discards the error and substitutes an answer.

    An AST question, not a text one: the offending forms differ only in the node
    type, and a string scan would have to guess at every spelling of a returned
    constant. Returns None for anything that warns, logs, re-raises, or returns
    a computed value.
    """
    if isinstance(stmt, ast.Pass):
        return "pass"
    if isinstance(stmt, ast.Continue):
        return "continue"
    if isinstance(stmt, ast.Break):
        return "break"
    if isinstance(stmt, ast.Return):
        value = stmt.value
        if value is None:
            return "return None"
        if isinstance(value, ast.Constant):
            return f"return {value.value!r}"
        if isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            return f"return {type(value).__name__.lower()} literal"
    return None


def _silent_swallows_in(path):
    tree = ast.parse(open(path).read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = node.body
        if len(body) != 1:
            # More than one statement means something else happens: a warn, a
            # log, a re-raise. Not this guard's concern.
            continue
        shape = _discarding_shape(body[0])
        if shape is None:
            continue
        exc = ast.unparse(node.type) if node.type is not None else "bare"
        # Allow pure optional-import guards.
        names = (
            {exc}
            if node.type is None
            else set(ast.unparse(node.type).replace("(", "").replace(")", "").split(","))
        )
        names = {n.strip() for n in names}
        if not names <= _ALLOWED_EXC:
            out.append((node.lineno, exc, shape))
    return out


def _scan_core():
    """Every discarding handler in the core, as (relpath, lineno, exc, shape)."""
    root = _src_root()
    found = []
    for sub in CORE_SUBPACKAGES:
        base = os.path.join(root, sub)
        for dirpath, _, files in os.walk(base):
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, root).replace(os.sep, "/")
                for lineno, exc, shape in _silent_swallows_in(p):
                    found.append((rel, lineno, exc, shape))
    return found


def test_core_has_no_silent_error_swallow():
    offenders = [
        f"{rel}:{lineno} except {exc}: {shape}"
        for rel, lineno, exc, shape in _scan_core()
        if (rel, exc, shape) not in _KNOWN_OPEN
    ]
    assert not offenders, (
        "An exception handler in the metric core discards the error and "
        "substitutes an answer. A returned constant is indistinguishable from a "
        "real measurement, which is how a perfect proxy feature came to report "
        "zero association. Warn, log, or re-raise instead:\n  " + "\n  ".join(offenders)
    )


def test_known_open_allowlist_has_no_stale_entries():
    """The allowlist is a ratchet, so it must shrink and never quietly persist.

    An entry that no longer matches anything means the code was fixed and the
    exemption should go. Left in place, it would silently re-bless a future
    regression in that file.
    """
    scanned = {(rel, exc, shape) for rel, _, exc, shape in _scan_core()}
    stale = sorted(f"{r} | except {e}: {sh}" for r, e, sh in set(_KNOWN_OPEN) - scanned)
    assert not stale, (
        "These handlers are exempted in _KNOWN_OPEN but no longer exist. The "
        "code was fixed; delete the entries, or a future regression in that "
        "file gets silently re-blessed:\n  " + "\n  ".join(stale)
    )


# --- positive control for the widened scanner --------------------------------
#
# The widening is only worth anything if it actually fires on each new shape and
# stays silent on the recovering ones. Without this, a refactor of
# _discarding_shape could make the guard inert and it would still report green,
# which is the exact failure the widening was written to fix.

import textwrap  # noqa: E402

import pytest  # noqa: E402

_OFFENDING = [
    "        pass",
    "        return 0.0",
    "        return 0",
    "        return False",
    "        return None",
    "        return",
    "        return {}",
    "        return []",
    "        return ()",
    "        return 'unknown'",
]

_SAFE = [
    "        raise",
    "        raise ValueError('boom') from exc",
    "        warnings.warn('could not measure'); return float('nan')",
    "        return fallback_computed(x)",
    "        return float('nan')",
    "        logger.exception('failed'); return 0.0",
]


def _scan_source(tmp_path, handler_body, exc="Exception"):
    src = textwrap.dedent(
        """\
        def f(x):
            try:
                return risky(x)
            except {exc} as exc:
        {body}
        """
    ).format(exc=exc, body=handler_body)
    p = tmp_path / "sample.py"
    p.write_text(src)
    return _silent_swallows_in(str(p))


@pytest.mark.parametrize("body", _OFFENDING)
def test_widened_scanner_fires_on_each_discarding_shape(tmp_path, body):
    hits = _scan_source(tmp_path, body)
    assert hits, f"scanner missed a discarding handler: {body.strip()!r}"


@pytest.mark.parametrize("body", _SAFE)
def test_widened_scanner_stays_silent_on_recovering_shapes(tmp_path, body):
    """Recovering from an error is fine. Discarding it and inventing an answer is not."""
    hits = _scan_source(tmp_path, body)
    assert not hits, f"scanner wrongly flagged a recovering handler: {body.strip()!r}"


@pytest.mark.parametrize("exc", ["ImportError", "ModuleNotFoundError"])
def test_optional_import_guards_stay_allowed(tmp_path, exc):
    assert not _scan_source(tmp_path, "        pass", exc=exc)


def test_the_original_pass_shape_is_still_caught(tmp_path):
    """The widening must not have lost the case the guard was written for."""
    assert _scan_source(tmp_path, "        pass")


def test_scanner_reports_the_shape_not_just_the_line(tmp_path):
    """The failure message has to say WHAT was substituted, or triage is guesswork."""
    hits = _scan_source(tmp_path, "        return 0.0")
    assert hits and hits[0][2] == "return 0.0", hits
