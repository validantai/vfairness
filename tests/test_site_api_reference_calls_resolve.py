"""Every call the published API reference shows must be callable as shown.

The site under `docs/site/` is NOT generated from `docs/*.md` by anything in
this repo, so the two drift independently and both have shipped falsehoods. This
file closes the half that a reader actually copies: the code.

It does not run the examples (most need a model, a frame or a network), which is
why an execution sweep kept reporting NameErrors that mean nothing. It resolves
them instead: every `from vfairness... import X` must import, and every call to
one of those imported names must use keyword arguments that exist in the real
signature. That is exactly the class of defect found on 2026-09-07, all of which
raise TypeError the moment a reader runs them:

    ReportConfig(tier=..., format=...)        -> default_tier / default_format
    DashboardConfig(time_window=...)          -> default_time_window
    simulate_threshold_change(metric=...)     -> metric_name
    log_fairness_to_mlflow(log_params=True)   -> no such parameter
    assert_fairness(protected_attr=...)       -> imported from the wrong module;
                                                 TWO functions share that name
                                                 with different signatures.
"""

from __future__ import annotations

import ast
import html
import importlib
import inspect
import pathlib
import re

import pytest

_SITE = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "site" / "api-reference" / "index.html"
)


# WIDENED 2026-09-07 from the api-reference page alone to EVERY published page
# and every shipped markdown file.
#
# The original scope was the page the docs audit happened to be looking at. That
# left 40 files unchecked, and re-running the audit's own class of finding across
# all of them found one more the same day, in the sample assessment, which is the
# page a prospective user reads to see what an audit looks like:
#
#     FairnessAwareBCELoss(fairness_loss=..., fairness_weight=...)
#     TypeError: unexpected keyword argument 'fairness_loss'
#     real parameters: fairness_metric, lambda_fairness, reduction, ...
#
# Both keyword names were invented. Neither has ever existed.
#
# ROADMAP.md is excluded, and only ROADMAP.md. A roadmap is SUPPOSED to sketch
# APIs that do not exist yet, and it has its own guard
# (test_roadmap_complete_sections_resolve.py) enforcing the narrower rule that
# applies there: a block under a heading claiming the work is COMPLETE must
# resolve or be labelled a superseded sketch.
_LIB_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SITE_DIR = _LIB_ROOT / "docs" / "site"
_ROADMAP = _LIB_ROOT / "docs" / "ROADMAP.md"

_HTML_BLOCK = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S)

# FENCE PAIRING, fixed 2026-09-10. The previous pattern was
#
#     re.compile(r"```(?:python|py)?\n(.*?)```", re.S)
#
# and it silently lost most of the markdown it was pointed at, because the
# language tag is OPTIONAL in it. A bare closing ``` therefore matches the
# OPENING half of the pattern just as well as an opener does, so the scan pairs
# fences by position rather than by role, and one unmatched opener inverts the
# pairing for the whole rest of the file.
#
# In docs/API_REFERENCE.md the inverting opener is line 3086, ```bash. The
# alternation has no 'bash' branch, so that opener is skipped, the scan resumes
# at the block's CLOSING fence, and from there every real code block is read as
# a terminator while the PROSE between blocks is captured as code. The prose
# then fails ast.parse and _defects() continues past it without a word, so the
# loss is invisible: 112 of that file's 211 blocks were never parsed.
#
# Measured over the whole documented set, on the blocks that survive the
# "mentions vfairness and imports something" filter:
#
#     old pattern  104 markdown blocks
#     this pattern 200 markdown blocks     (+86 API_REFERENCE, +9 LIBRARY_OVERVIEW, +1 README)
#
# One of the 96 is the FairRegressor example, which documented
# method='reweighting' for weeks after the code began raising ValueError on it.
# Blocks containing a FairRegressor(...) call: 0 under the old pattern, 1 under
# this one.
#
# The fix is to make the roles unambiguous: an opener carries an infostring and
# starts a line, a closer is a ``` at the start of a line. Every language is
# accepted rather than an allow-list of two, so the next ```bash cannot do this
# again; a non-Python block fails ast.parse and is skipped exactly as before.
_MD_BLOCK = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)


def _documented_files() -> list[pathlib.Path]:
    files = sorted(_SITE_DIR.rglob("*.html")) + sorted(_SITE_DIR.rglob("*.md"))
    files += sorted(_LIB_ROOT / "docs" / f for f in ("API_REFERENCE.md", "API_STABILITY.md"))
    files += sorted(p for p in (_LIB_ROOT / "docs").glob("*.md") if p != _ROADMAP)
    files.append(_LIB_ROOT / "README.md")
    # de-duplicate, keep only what exists, and never the audit records
    seen, out = set(), []
    for f in files:
        if f in seen or not f.exists() or "audits" in f.parts or f == _ROADMAP:
            continue
        seen.add(f)
        out.append(f)
    return out


def _blocks_of(path: pathlib.Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".md":
        out = _MD_BLOCK.findall(raw)
    else:
        out = [html.unescape(re.sub(r"<[^>]+>", "", b)) for b in _HTML_BLOCK.findall(raw)]
    return [b for b in out if "vfairness" in b and "import" in b]


def _blocks() -> list[str]:
    if not _SITE_DIR.exists():  # pragma: no cover - the site is part of the repo
        pytest.skip("published site not present")
    out: list[str] = []
    for f in _documented_files():
        out.extend(_blocks_of(f))
    return out


def _missing_required(where: str, node: ast.Call, sig: inspect.Signature) -> list[str]:
    """A documented call that OMITS a parameter with no default.

    ADDED 2026-09-10. Everything above checks that the names a call SUPPLIES
    exist. Nothing checked the names it LEAVES OUT, and a required parameter is
    the other half of the same TypeError. Measured on
    docs/site/api-reference/index.html: the custom-handler example built

        MetricExplanation(metric_name=..., definition=..., interpretation_guide=...,
                          value=..., evaluation=..., recommendation=..., severity=...)

    with no ``benchmark_context``, which is declared with no default. The
    example crashed unconditionally, on every reader, on the first line they
    copied, and every name it passed was real, so the kwarg check saw nothing.

    Deliberately conservative, because a documentation example is not required
    to be a complete program:

      * skipped entirely when the callable takes ``*args``: the positional
        count cannot be reasoned about from the call site.
      * skipped when the call uses ``*seq`` or ``**mapping``: the missing
        argument may well be in there.
      * counts positional arguments against POSITIONAL_ONLY and
        POSITIONAL_OR_KEYWORD parameters in order, exactly as Python binds them.

    That leaves the unambiguous case: a fully explicit call to a signature with
    no varargs, missing a parameter that has no default.
    """
    params = list(sig.parameters.values())
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return []
    if any(kw.arg is None for kw in node.keywords):  # **mapping
        return []
    if any(isinstance(a, ast.Starred) for a in node.args):  # *seq
        return []

    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    supplied = {p.name for p in positional[: len(node.args)]}
    supplied |= {kw.arg for kw in node.keywords if kw.arg}

    missing = [
        p.name
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind is not p.VAR_KEYWORD
        and p.name not in supplied
    ]
    if not missing:
        return []
    # The caller only reaches here for an ast.Name callee, but narrow again
    # rather than asserting it: ast.unparse is a correct answer for any other
    # callee shape, so widening costs nothing and the check stays total.
    called = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
    return [
        f"{where}: {called}(...) is documented without {missing}, which "
        f"{'have' if len(missing) > 1 else 'has'} no default; the example raises "
        f"TypeError as written. Real signature: {sig}"
    ]


def _defects() -> list[str]:
    bad: list[str] = []
    for path in _documented_files():
        for block in _blocks_of(path):
            where = path.relative_to(_LIB_ROOT).as_posix()
            try:
                tree = ast.parse(block)
            except SyntaxError:
                # A deliberate fragment, not a program. Nothing to resolve.
                continue
            env: dict[str, object] = {}
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("vfairness")
                ):
                    try:
                        mod = importlib.import_module(node.module)
                    except Exception:
                        # An optional extra that is not installed here. The import
                        # itself is checked by test_api_reference_page_truth.
                        continue
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        obj = getattr(mod, alias.name, None)
                        if obj is None:
                            bad.append(
                                f"{where}: documented symbol does not exist: {node.module}.{alias.name}"
                            )
                        else:
                            env[alias.asname or alias.name] = obj
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                target = env.get(node.func.id)
                if target is None:
                    continue
                try:
                    sig = inspect.signature(target)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                    continue
                names = {p.name for p in sig.parameters.values()}
                for kw in node.keywords:
                    if kw.arg and kw.arg not in names:
                        bad.append(
                            f"{where}: {node.func.id}(... {kw.arg}=...) is documented, but "
                            f"{kw.arg!r} is not a parameter; real: {sorted(names)}"
                        )
                bad.extend(_missing_required(where, node, sig))
    # stable, de-duplicated
    seen: set[str] = set()
    return [d for d in bad if not (d in seen or seen.add(d))]


def test_every_documented_call_uses_parameters_that_exist():
    defects = _defects()
    assert not defects, "the published API reference documents calls that raise:\n  " + "\n  ".join(
        defects
    )


def test_the_scan_actually_reaches_every_documented_page():
    """NON-VACUITY. A resolver that silently found nothing would pass forever.

    Two floors, because widening the scan created a second way to be vacuous:
    the block count could stay healthy while the file list quietly collapsed to
    the one page this guard started on.
    """
    files = _documented_files()
    assert len(files) >= 25, f"only {len(files)} documented files found; the sweep narrowed"

    blocks = _blocks()
    assert len(blocks) > 200, f"only {len(blocks)} runnable blocks found; the extractor is broken"

    # A THIRD floor, on the markdown half alone. The two above were both
    # comfortably green while _MD_BLOCK was pairing fences by position and
    # losing 96 of the 200 markdown blocks, because the HTML half is much the
    # larger contributor and hid the collapse inside the total. Measured
    # 2026-09-10: 200 markdown blocks. Reverting to the old pattern yields 104,
    # which this refuses.
    md_blocks = sum(len(_blocks_of(f)) for f in files if f.suffix == ".md")
    assert md_blocks >= 190, (
        f"only {md_blocks} markdown blocks extracted; _MD_BLOCK is mis-pairing "
        f"fences again (the old positional pattern yields 104)"
    )

    # The pages that have each carried a defect of exactly this class.
    with_blocks = {f.relative_to(_LIB_ROOT).as_posix() for f in files if _blocks_of(f)}
    for expected in (
        "docs/site/api-reference/index.html",
        "docs/site/sample-assessment/index.html",
        "docs/site/getting-started/index.html",
        "docs/API_REFERENCE.md",
    ):
        assert expected in with_blocks, f"{expected} contributes no blocks; it is not being scanned"

    # ROADMAP.md is excluded ON PURPOSE and has its own guard. If that exclusion
    # ever silently widens, this says so.
    assert _ROADMAP.exists(), "ROADMAP.md is gone; its own guard should be reviewed"
    assert _ROADMAP not in files, "ROADMAP.md must stay excluded here"


# ---------------------------------------------------------------------------
# WIDENED 2026-09-09: a documented parameter NAME, wherever it appears.
#
# Everything above parses CALL examples only, `Foo(a=1)`. That is a blind spot
# the width of the page. A parameter is documented three ways here, and two of
# them went unchecked:
#
#     1. a call example      Foo(a=1)                                <- checked
#     2. a signature block   Foo(\n    a: bool = True\n) -> Bar      <- was NOT
#     3. a parameter table   <span class="api-param-name">a</span>   <- was NOT
#
# Measured the day two fix lanes removed parameters that had been accepted and
# ignored: the call check caught 1 of the 8 stale spots on this page. Forms 2
# and 3 carried the other 7, among them
# `compute_feature_correlations(include_pvalues=...)` in BOTH its signature
# block and its parameter table, and `log_fairness_to_mlflow`, whose parameter
# table still named `analyzer` and `log_params` after the docstring at the top
# of this file had recorded `log_params` as fixed in the call examples. Fixing
# what a checker enumerates and calling the class closed is the failure this
# whole audit is about, so the enumeration is widened rather than the list.
#
# SCOPE, stated plainly so a green run is not read as more than it is. This
# compares documented parameter NAMES against the real signature. It does NOT
# check:
#   * the VALUES an option accepts. `grid_search=False`,
#     `method='partial'`, `fallback_strategy='borrow'` and
#     `causal_criterion='direct_effect'` are all names that still exist and are
#     now refused at construction; a name check cannot see that.
#   * what a parameter DOES. `target_correlation` still exists, but it is
#     verified against the measurement instead of steering the output.
#   * the type or the default shown next to the name.
#   * prose, tables of methods, mermaid diagrams.
# Those are read by a person, or pinned by the behavioural test that came with
# each fix. A pass here means "every documented parameter name exists on the
# real signature", and nothing wider.
# ---------------------------------------------------------------------------

import functools
import pkgutil
import warnings

_SIGNATURE_HEAD = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.S)
_DECORATOR_AT = re.compile(r"^@\s*")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_API_FUNCTION_DIV = re.compile(
    r'<div class="api-function">(.*?)(?=<div class="api-function">|\Z)', re.S
)
_API_FUNCTION_NAME = re.compile(r'<span class="api-function-name">([^<]*)</span>')
_API_PARAM_NAME = re.compile(r'<span class="api-param-name">([^<]*)</span>')


@functools.lru_cache(maxsize=1)
def _symbol_index() -> dict:
    """Public classes and functions defined anywhere in vfairness, by bare name.

    A name can map to more than one object (two modules export an
    `assert_fairness` with different signatures), so the value is a tuple and a
    parameter is only reported missing when it is missing from ALL of them.
    """
    import vfairness

    modules = [vfairness]
    for found in pkgutil.walk_packages(
        vfairness.__path__, "vfairness.", onerror=lambda _name: None
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                modules.append(importlib.import_module(found.name))
        except Exception:
            # An optional extra that is not installed in this environment.
            # test_api_reference_page_truth is the guard for import paths.
            continue

    index: dict[str, list] = {}
    for module in modules:
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            if not (getattr(obj, "__module__", "") or "").startswith("vfairness"):
                continue
            bucket = index.setdefault(name, [])
            if obj not in bucket:
                bucket.append(obj)
    return {name: tuple(objs) for name, objs in index.items()}


def _real_parameter_names(obj) -> set | None:
    """The parameters the real object takes, or None when it cannot be pinned.

    None means "do not judge this one": either the signature is unreadable, or
    it ends in **kwargs and therefore accepts any keyword.
    """
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return {p.name for p in sig.parameters.values() if p.name != "self"}


def _name_defects(symbol: str, documented: list, where: str, form: str) -> list:
    """Documented names that exist on none of the objects called `symbol`."""
    candidates = _symbol_index().get(symbol)
    if not candidates:
        return []  # not a vfairness symbol we can resolve; nothing to say
    real = [names for names in (_real_parameter_names(o) for o in candidates) if names is not None]
    if not real:
        return []
    out = []
    for name in documented:
        if not _IDENTIFIER.match(name):
            continue  # '&nbsp;' in a "no parameters" row, or prose in the span
        if all(name not in names for names in real):
            out.append(
                f"{where}: {form} {symbol}(... {name} ...) is documented, but "
                f"{name!r} is not a parameter; real: {sorted(real[0])}"
            )
    return out


def _parameter_list(text: str):
    """The text between the first `(` and the `)` that closes it, or None."""
    open_at = text.find("(")
    if open_at < 0:
        return None
    depth = 0
    for i in range(open_at, len(text)):
        char = text[i]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i]
    return None


def _signature_block(text: str):
    """(symbol, documented parameter names) for a `Name(...)` block, else None.

    WIDENED 2026-09-09, twice, after both blind spots were measured on the
    api-reference page:

    1. A leading `@` is stripped. `@auto_log_fairness(...)` is rendered exactly
       like every other signature block on the page and is the ONLY form in
       which a decorator's parameters are documented, but `_SIGNATURE_HEAD`
       needs an identifier first, so the whole block was skipped in silence.
       That is where `protected_attr_column` survived being removed from the
       real function.
    2. Only the PARAMETER LIST is parsed, not the whole rendered line. The old
       form appended the body to the source line, so `-> StatisticalResult  #
       point_estimate = p1 - p2` swallowed the `:` in its comment and the block
       failed `ast.parse`. Six blocks on this page and one in API_REFERENCE.md
       were dropped that way, every one of them a real signature. Measured
       before and after: 75 blocks resolved, then 83, with none left skipped.

    A block that still does not parse returns None and is COUNTED by
    `_parameter_name_defects`, so a silent collapse shows up as a floor failure
    in `test_the_parameter_name_scan_is_not_vacuous` rather than as a green run.
    """
    text = _DECORATOR_AT.sub("", text.strip())
    head = _SIGNATURE_HEAD.match(text)
    if not head:
        return None
    params = _parameter_list(text)
    if params is None:
        return None
    try:
        tree = ast.parse("def _signature(\n" + params + "\n): ...")
    except SyntaxError:
        return None  # a fragment, an elided body, not a signature
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef):
        return None
    args = fn.args
    names = [a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return head.group(1), [n for n in names if n != "self"]


def _param_table_symbol(raw_name: str) -> str:
    """`class GroupThresholdOptimizer` and `def identify_paths` name a symbol."""
    name = html.unescape(raw_name).strip()
    for prefix in ("class ", "def "):
        if name.startswith(prefix):
            name = name[len(prefix) :].strip()
    return name


def _parameter_name_defects() -> tuple[list, int, int]:
    """Defects, plus how many signature blocks and parameter tables resolved."""
    defects: list[str] = []
    n_signatures = 0
    n_tables = 0

    for path in _documented_files():
        where = path.relative_to(_LIB_ROOT).as_posix()
        raw = path.read_text(encoding="utf-8", errors="replace")

        if path.suffix == ".md":
            blocks = _MD_BLOCK.findall(raw)
        else:
            blocks = [html.unescape(re.sub(r"<[^>]+>", "", b)) for b in _HTML_BLOCK.findall(raw)]

        for block in blocks:
            if "import" in block:
                continue  # a runnable example; the call check above owns it
            parsed = _signature_block(block)
            if parsed is None:
                continue
            symbol, documented = parsed
            if symbol in _symbol_index():
                n_signatures += 1
            defects += _name_defects(symbol, documented, where, "signature block")

        if path.suffix == ".md":
            continue
        for div in _API_FUNCTION_DIV.findall(raw):
            match = _API_FUNCTION_NAME.search(div)
            if not match:
                continue
            symbol = _param_table_symbol(match.group(1))
            documented = []
            for span in _API_PARAM_NAME.findall(div):
                # One span sometimes groups several parameters that share a
                # description, e.g. "y_true, y_pred, sensitive_attr". Each is
                # still a claim, so each is checked.
                for part in html.unescape(span).split(","):
                    part = part.strip().strip("`").lstrip("*").strip()
                    if part:
                        documented.append(part)
            if documented and symbol in _symbol_index():
                n_tables += 1
            defects += _name_defects(symbol, documented, where, "parameter table")

    seen: set[str] = set()
    unique = [d for d in defects if not (d in seen or seen.add(d))]
    return unique, n_signatures, n_tables


def test_every_documented_parameter_name_exists():
    defects, _, _ = _parameter_name_defects()
    assert not defects, (
        "the published documentation names parameters that do not exist:\n  " + "\n  ".join(defects)
    )


def test_the_parameter_name_scan_is_not_vacuous():
    """A name checker that resolves nothing passes forever. Three floors.

    1. the symbol index is populated,
    2. signature blocks and parameter tables are both being reached,
    3. the comparison itself still fires on a planted defect.
    """
    index = _symbol_index()
    assert len(index) >= 400, f"only {len(index)} vfairness symbols indexed; the walk narrowed"

    _, n_signatures, n_tables = _parameter_name_defects()
    # Raised from 60 to 78 on 2026-09-09 when the parser stopped dropping the
    # `@decorator(...)` form and blocks whose trailing comment swallowed the
    # body: 75 resolved before, 83 after. Leaving the floor at 60 would have let
    # that widening be undone without a red run.
    assert n_signatures >= 78, f"only {n_signatures} signature blocks resolved; the parser broke"

    # A count floor is too coarse to see ONE form disappear: neutering the
    # decorator strip takes 83 blocks to 82 and the floor stays green. Measured
    # on 2026-09-09 by doing exactly that. So both widenings are pinned by the
    # form they recovered, not by the total.
    decorated = _signature_block(
        "@auto_log_fairness(\n    backend: str = 'mlflow',\n    prefix: str = 'fairness'\n)"
    )
    assert decorated == ("auto_log_fairness", ["backend", "prefix"]), decorated
    trailing_comment = _signature_block(
        "proportion_z_test(\n"
        "    p1: float,    # Proportion in group 1\n"
        "    n1: int       # Sample size of group 1\n"
        ") -> float        # Two-sided p-value"
    )
    assert trailing_comment == ("proportion_z_test", ["p1", "n1"]), trailing_comment
    assert n_tables >= 40, f"only {n_tables} parameter tables resolved; the extractor broke"

    # The comparison fires. Without this a bug in _real_parameter_names (a bare
    # `return None`, say) would make every page pass and look identical to a
    # clean run.
    planted = _signature_block(
        "compute_feature_correlations(\n"
        "    df: pd.DataFrame,\n"
        "    protected_attributes: List[str],\n"
        "    include_pvalues: bool = True\n"
        ") -> FeatureCorrelationMatrix"
    )
    assert planted is not None, "the signature parser stopped parsing signatures"
    symbol, documented = planted
    assert "include_pvalues" in documented
    fired = _name_defects(symbol, documented, "planted", "signature block")
    assert len(fired) == 1 and "include_pvalues" in fired[0], (
        f"the parameter-name comparison did not fire on a known-removed parameter; got {fired!r}"
    )
    # and it does not fire on a parameter that does exist
    assert not _name_defects(symbol, ["df", "method"], "planted", "signature block")


# ---------------------------------------------------------------------------
# WIDENED 2026-09-09 (second pass): METHOD calls and documented ATTRIBUTE reads.
#
# Six defects were found by hand on the api-reference page that same day, and
# this guard, which is the guard for exactly that class, was green through all
# six. Its blind spots, measured rather than guessed:
#
#   * `_defects()` only looks at `ast.Call` whose `func` is a bare `ast.Name`.
#     Every METHOD call is therefore invisible, and
#     `gate.evaluate_hierarchical(protected_attributes=[...])` sat on the page
#     raising TypeError with the guard green.
#   * Nothing looked at ATTRIBUTE ACCESS at all. `result.method_used` (the real
#     name is `method`, on the most copied function in the library) and
#     `eg.result_.converged` (there is no `result_`; it is `get_result()`, and
#     `converged` lives one level deeper on `.optimization_result`) both hid
#     there.
#
# Both are closed below by resolving what a local NAME holds inside one block:
#
#     gate   = ModelFairnessGate(...)          -> the class
#     result = eg.fit(...)                     -> the return annotation
#     result = demographic_parity_difference_with_ci(...)   -> ditto
#
# and then checking the methods called on it and the attributes read from it.
#
# SCOPE, stated as plainly as the section above, so a green run is not read as
# more than it is. This resolves a name only from an assignment IN THE SAME
# BLOCK, and only when that assignment is unambiguous. It does NOT check:
#   * a name bound anywhere other than a single `x = <call>` (a loop variable, a
#     `with ... as`, tuple unpacking, a function parameter). Those are dropped
#     on purpose: a name that is rebound means something different at different
#     points and judging it would be guesswork.
#   * a class whose MRO defines `__getattr__`, which legitimately accepts any
#     attribute name. Every `torch.nn.Module` subclass is in this group, so the
#     loss classes are NOT covered here.
#   * a class whose source cannot be read or parsed, and anything reached
#     through a builtin (`dict`, `list`) rather than a vfairness type.
#   * VALUES, prose, and what an attribute MEANS. F-3 (the auto-log decorator
#     documented a return shape the wrapper does not accept, so the example
#     logged nothing) and F-4 (a `per_intersection_thresholds` key in a shape
#     the gate matches against nothing, so a documented stricter threshold was
#     silently ignored) are BOTH invisible to any name checker, and are pinned
#     by executing the published block instead, in tests/test_readiness_docs.py.
#
# Nothing here was widened past what could be shipped without noise: across all
# 322 runnable blocks in the corpus these two checks report zero false
# positives, measured before shipping, and the Enum carve-out below is the one
# case that needed it.
# ---------------------------------------------------------------------------

import enum
import sys
import textwrap


def _annotation_class(annotation, module) -> type | None:
    """The class an annotation names, or None when it does not name one."""
    if annotation is inspect.Signature.empty or annotation is None:
        return None
    if isinstance(annotation, str):
        resolved = getattr(module, annotation.strip(), None)
        return resolved if inspect.isclass(resolved) else None
    return annotation if inspect.isclass(annotation) else None


def _return_class(func) -> type | None:
    """The vfairness class a function is annotated to return, else None."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    module = sys.modules.get(getattr(func, "__module__", "") or "")
    if module is None:
        return None
    cls = _annotation_class(sig.return_annotation, module)
    if cls is not None and (getattr(cls, "__module__", "") or "").startswith("vfairness"):
        return cls
    return None


@functools.lru_cache(maxsize=None)
def _instance_attributes(cls: type) -> frozenset | None:
    """Every name an instance of `cls` can carry, or None when unknowable.

    None is the third state and it matters: it means "this guard cannot judge
    this class", not "the attribute is absent". Returning an empty set instead
    would report every read on a torch module as a defect.
    """
    if any("__getattr__" in vars(base) for base in cls.__mro__):
        return None
    names = set(dir(cls))
    if issubclass(cls, enum.Enum):
        # A documented `rec.decision.name` reads a MEMBER, and a member carries
        # `.name` / `.value` that `dir()` of the Enum CLASS does not list.
        return frozenset(names | {"name", "value"})
    for base in cls.__mro__:
        names |= set(getattr(base, "__annotations__", {}) or {})
    try:
        source = textwrap.dedent(inspect.getsource(cls))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                names.add(target.attr)
    return frozenset(names)


def _attribute_class(cls: type, name: str) -> type | None:
    """The class of `cls().name` when an annotation says so, else None."""
    for base in cls.__mro__:
        annotations = getattr(base, "__annotations__", {}) or {}
        if name in annotations:
            module = sys.modules.get(getattr(base, "__module__", "") or "")
            if module is None:
                continue
            return _annotation_class(annotations[name], module)
    member = getattr(cls, name, None)
    if isinstance(member, property) and member.fget is not None:
        return _return_class(member.fget)
    return None


def _imported_env(tree: ast.AST) -> dict:
    """`from vfairness... import X` names bound in one documented block."""
    env: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vfairness")
        ):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = importlib.import_module(node.module)
        except Exception:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            obj = getattr(module, alias.name, None)
            if obj is not None:
                env[alias.asname or alias.name] = obj
    return env


def _rebound_names(tree: ast.AST) -> set:
    """Names bound anywhere OTHER than a single plain assignment.

    A `for level, result in ...` loop variable called `result` is a different
    thing from a `result = f(...)` two blocks up, and judging it against the
    latter would invent defects. Anything in here is dropped.
    """
    out: set[str] = set()

    def collect(target):
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                out.add(node.id)

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            collect(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    collect(item.optional_vars)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in node.generators:
                collect(gen.target)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                out.add(arg.arg)
            if args.vararg:
                out.add(args.vararg.arg)
            if args.kwarg:
                out.add(args.kwarg.arg)
        elif isinstance(node, ast.Assign) and (
            len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)
        ):
            for target in node.targets:
                collect(target)
    return out


def _local_types(tree: ast.AST, env: dict) -> dict:
    """`name -> class` for every name a block assigns exactly one known type to."""
    found: dict[str, type | None] = {}
    ambiguous: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        cls: type | None = None
        value = node.value
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                obj = env.get(value.func.id)
                if inspect.isclass(obj):
                    cls = obj
                elif inspect.isfunction(obj):
                    cls = _return_class(obj)
            elif isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name):
                owner = found.get(value.func.value.id)
                if owner is not None:
                    method = getattr(owner, value.func.attr, None)
                    if inspect.isfunction(method):
                        cls = _return_class(method)
        if target.id in found and found[target.id] is not cls:
            ambiguous.add(target.id)
        found[target.id] = cls
    rebound = _rebound_names(tree)
    return {
        name: cls
        for name, cls in found.items()
        if cls is not None and name not in ambiguous and name not in rebound
    }


def _method_and_attribute_defects() -> tuple[list, int, int]:
    """Defects, plus how many method calls and attribute steps were resolved."""
    defects: list[str] = []
    n_methods = 0
    n_attributes = 0

    for path in _documented_files():
        where = path.relative_to(_LIB_ROOT).as_posix()
        for block in _blocks_of(path):
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue
            env = _imported_env(tree)
            local = _local_types(tree, env)

            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                base = node.func.value
                if not isinstance(base, ast.Name):
                    continue
                cls: type | None = local.get(base.id)
                if cls is None:
                    imported = env.get(base.id)
                    if isinstance(imported, type):
                        cls = imported
                if cls is None:
                    continue
                n_methods += 1
                method = getattr(cls, node.func.attr, None)
                if method is None:
                    allowed = _instance_attributes(cls)
                    if allowed is not None and node.func.attr not in allowed:
                        defects.append(
                            f"{where}: {cls.__name__}.{node.func.attr}(...) is documented, "
                            f"but that method does not exist"
                        )
                    continue
                if not (inspect.isfunction(method) or inspect.ismethod(method)):
                    continue
                try:
                    sig = inspect.signature(method)
                except (TypeError, ValueError):
                    continue
                if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                    continue
                names = {p.name for p in sig.parameters.values()}
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg not in names:
                        defects.append(
                            f"{where}: {cls.__name__}.{node.func.attr}(... {keyword.arg}=...) is "
                            f"documented, but {keyword.arg!r} is not a parameter; "
                            f"real: {sorted(names - {'self'})}"
                        )

            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                chain: list[str] = []
                cursor: ast.AST = node
                while isinstance(cursor, ast.Attribute):
                    chain.append(cursor.attr)
                    cursor = cursor.value
                if not isinstance(cursor, ast.Name):
                    continue
                chain.reverse()
                cls = local.get(cursor.id)
                if cls is None:
                    continue
                shown = cursor.id
                for attr in chain:
                    allowed = _instance_attributes(cls)
                    if allowed is None:
                        break
                    n_attributes += 1
                    if attr not in allowed:
                        defects.append(
                            f"{where}: {shown}.{attr} is documented, but "
                            f"{cls.__name__} has no attribute {attr!r}"
                        )
                        break
                    shown = f"{shown}.{attr}"
                    nxt = _attribute_class(cls, attr)
                    if nxt is None:
                        break
                    cls = nxt

    seen: set[str] = set()
    unique: list[str] = []
    for defect in defects:
        if defect not in seen:
            seen.add(defect)
            unique.append(defect)
    return unique, n_methods, n_attributes


def test_every_documented_method_call_and_attribute_read_resolves():
    defects, _, _ = _method_and_attribute_defects()
    assert not defects, (
        "the published documentation shows method calls or attribute reads that raise:\n  "
        + "\n  ".join(defects)
    )


def test_the_method_and_attribute_scan_is_not_vacuous():
    """A resolver that resolves nothing passes forever. Floors, then controls.

    The floors are not decoration. `_instance_attributes` returned None for
    every class while this check was being built, because the class source was
    being de-indented with the wrong helper, and the whole attribute half was
    silently disabled: 0 reports, indistinguishable from a clean corpus.
    """
    _, n_methods, n_attributes = _method_and_attribute_defects()
    assert n_methods >= 250, f"only {n_methods} method calls resolved; the resolver narrowed"
    assert n_attributes >= 450, (
        f"only {n_attributes} attribute steps resolved; the resolver narrowed"
    )

    # The comparisons fire, on the two defects that were live on this page.
    live = "\n".join(
        (
            "from vfairness.operations.cicd import ModelFairnessGate",
            "gate = ModelFairnessGate(metrics=['demographic_parity_difference'])",
            "d = gate.evaluate_hierarchical(y_true, y_pred, protected_attributes=['gender'])",
        )
    )
    tree = ast.parse(live)
    env = _imported_env(tree)
    local = _local_types(tree, env)
    assert local["gate"].__name__ == "ModelFairnessGate", local
    real = set(inspect.signature(local["gate"].evaluate_hierarchical).parameters)
    assert "protected_attributes" not in real and "protected_attrs" in real, sorted(real)

    from vfairness.evaluation.vfairness_metrics._statistics import StatisticalResult

    allowed = _instance_attributes(StatisticalResult)
    assert allowed is not None, "StatisticalResult became unjudgeable; the guard is off"
    assert "method_used" not in allowed, "the planted attribute defect stopped being a defect"
    assert "method" in allowed, "the real attribute vanished from the resolver"

    from vfairness.in_processing import ExponentiatedGradient

    eg_attrs = _instance_attributes(ExponentiatedGradient)
    assert eg_attrs is not None
    assert "result_" not in eg_attrs, "the planted attribute defect stopped being a defect"
    assert {"get_result", "fit", "predict"} <= eg_attrs, "the real API vanished from the resolver"

    # ... and the third state is really a third state: a class the guard cannot
    # judge returns None rather than an empty set, which would report every read
    # on it as a defect.
    class _Dynamic:
        def __getattr__(self, name):  # pragma: no cover - shape only
            return 1

    assert _instance_attributes(_Dynamic) is None


# ===========================================================================
# Result KEYS. Widened 2026-09-10 (READINESS-5).
#
# The checks above resolve what a documented example CALLS and what it reads
# off an object. Neither looks at what it reads out of a returned DICT, and
# that is where the api-reference page had been shipping a line that raises:
#
#     result = simulate_threshold_change(store, metric_name=..., ...)
#     print(f"Affected groups: {result['affected_groups']}")
#
# The real key is `groups_impacted`. The call resolved, every keyword existed,
# and the very next line was a KeyError for anyone who copied it.
#
# Scope, stated so nobody reads more into a pass than it earns: a function
# qualifies only when EVERY `return` in its body is a dict literal whose keys
# are all string literals. That is enough for the dict-returning reporting and
# probe functions the docs actually demonstrate, and it means the key set is
# read off the code rather than guessed. Anything else returns None from
# _returned_dict_keys and is skipped, which the floor below keeps honest.
# ===========================================================================


def _returned_dict_keys(func) -> frozenset | None:
    """Every string key this function can return, or None if not knowable."""
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    definition = next(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        return None
    keys: set[str] = set()
    returns = [n for n in ast.walk(definition) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return None
    for node in returns:
        if not isinstance(node.value, ast.Dict):
            return None
        for key in node.value.keys:
            # `**base` in a dict literal contributes an unknown key set.
            if key is None:
                return None
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return None
            keys.add(key.value)
    return frozenset(keys)


def _result_key_defects() -> tuple[list, int]:
    """Defects, plus how many documented result keys were actually resolved."""
    defects: list[str] = []
    checked = 0

    for path in _documented_files():
        where = path.relative_to(_LIB_ROOT).as_posix()
        for block in _blocks_of(path):
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue
            env = _imported_env(tree)
            if not env:
                continue
            rebound = _rebound_names(tree)

            # name -> the key set of the vfairness function it was bound from.
            bound: dict[str, frozenset] = {}
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                    continue
                target = node.targets[0]
                if not (isinstance(target, ast.Name) and target.id not in rebound):
                    continue
                call = node.value
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
                    continue
                func = env.get(call.func.id)
                if func is None or not inspect.isfunction(func):
                    continue
                keys = _returned_dict_keys(func)
                if keys:
                    bound[target.id] = keys

            for node in ast.walk(tree):
                if not isinstance(node, ast.Subscript):
                    continue
                value, index = node.value, node.slice
                if not (isinstance(value, ast.Name) and value.id in bound):
                    continue
                if not (isinstance(index, ast.Constant) and isinstance(index.value, str)):
                    continue
                checked += 1
                if index.value not in bound[value.id]:
                    defects.append(
                        "{0}: {1}[{2!r}] -- no such key; the function returns {3}".format(
                            where, value.id, index.value, sorted(bound[value.id])
                        )
                    )

    return defects, checked


def test_every_documented_result_key_exists():
    defects, _ = _result_key_defects()
    assert not defects, (
        "the published documentation reads keys that are not in the returned dict, "
        "so the line raises KeyError for anyone who copies it:\n  " + "\n  ".join(defects)
    )


def test_the_result_key_scan_is_not_vacuous():
    """A scanner that reaches nothing passes forever.

    This one is especially easy to disable by accident: narrow
    _returned_dict_keys and every function becomes "not knowable", so the whole
    check skips in silence and reports a clean corpus. The floor is the guard,
    and the control below proves the comparison itself still bites.
    """
    _, checked = _result_key_defects()
    assert checked >= 8, (
        f"only {checked} documented result key(s) were resolved; the scanner narrowed "
        "and is no longer looking at the corpus"
    )

    # Control: the key set really is read off the function, and a wrong key
    # really is rejected. Both directions, on the function whose documented
    # example was the defect that prompted this check.
    from vfairness.operations.reporting import simulate_threshold_change

    keys = _returned_dict_keys(simulate_threshold_change)
    assert keys is not None, "the key set could not be read from the real function"
    assert "groups_impacted" in keys
    assert "affected_groups" not in keys, (
        "the key the docs used to show is now real, so this control proves nothing"
    )
