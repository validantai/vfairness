"""Deserialization / dynamic-execution safety tests (VB-SEC-6).

vfairness must never silently deserialize untrusted input. joblib/pickle,
skops(trusted=True) and torch.load(weights_only=False) run arbitrary code at
load time, so any such call reachable from a payload boundary is a remote-code-
execution sink. numpy.load(allow_pickle=True) is the same trap.

This module has two halves:

1. A source scan over vfairness.* that fails the build if a new unguarded
   pickle/joblib/torch/numpy load, an unsafe yaml.load, or an eval()/exec() on
   user input appears in the metric/loader code paths.
2. A behavioral test that the one real untrusted-model loader
   (``vfairness.xai.sidecar_cli._load_model``) refuses to deserialize without an
   explicit opt-in and still works for callers who do opt in.
"""

import ast
import base64
import io
import os
from pathlib import Path

import joblib
import pytest

from vfairness.xai import sidecar_cli

# ── locate the installed source tree ─────────────────────────────────────────

_PKG_ROOT = Path(sidecar_cli.__file__).resolve().parent.parent  # .../vfairness
# Scan the whole installed source tree; nothing is exempt from the
# deserialization/exec ban except the audited, opt-in-gated loader in
# sidecar_cli._load_model (see _GUARDED_FUNCTIONS below).
_SKIP_DIRS = {"__pycache__"}


def _iter_source_files():
    for path in _PKG_ROOT.rglob("*.py"):
        parts = set(path.relative_to(_PKG_ROOT).parts)
        if parts & _SKIP_DIRS:
            continue
        yield path


# ── AST helpers ──────────────────────────────────────────────────────────────


def _call_dotted_name(node: ast.Call) -> str:
    """Return the dotted attribute chain for a call target, or '' if not one."""
    parts = []
    cur = node.func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _kw(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_false(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


# The sidecar's _load_model is the audited, opt-in-gated loader. Its joblib/
# skops calls are allowed to live inside that function only.
_GUARDED_FUNCTIONS = {("xai/sidecar_cli.py", "_load_model")}


def _rel(path: Path) -> str:
    return str(path.relative_to(_PKG_ROOT)).replace(os.sep, "/")


def _enclosing_func(tree: ast.Module, target: ast.AST):
    """Name of the nearest enclosing function def for a node, or None."""
    found = {"name": None}

    class _V(ast.NodeVisitor):
        def _walk_func(self, fn):
            for child in ast.walk(fn):
                if child is target:
                    found["name"] = fn.name
                    return True
            return False

        def visit_FunctionDef(self, fn):
            self._walk_func(fn)

        def visit_AsyncFunctionDef(self, fn):
            self._walk_func(fn)

    _V().visit(tree)
    return found["name"]


# Risky (module, funcname) pairs whose *direct* import must also be caught, e.g.
# `from pickle import loads as _l; _l(x)` -- the scan matches dotted call
# targets, so without alias resolution an aliased import would bypass every rule.
_RISKY_FUNCS = {
    ("pickle", "load"),
    ("pickle", "loads"),
    ("cloudpickle", "load"),
    ("cloudpickle", "loads"),
    ("dill", "load"),
    ("dill", "loads"),
    ("marshal", "load"),
    ("marshal", "loads"),
    ("joblib", "load"),
    ("numpy", "load"),
    ("torch", "load"),
    ("jsonpickle", "decode"),
    ("jsonpickle", "loads"),
    ("pandas", "read_pickle"),
    ("shelve", "open"),
}


def _alias_map(tree: ast.Module) -> dict:
    """Local name -> 'module.func' for risky funcs pulled in via `from ... import`.

    Lets the dotted-name rules catch `from joblib import load as L; L(x)` style
    imports, which would otherwise slip past the scan.
    """
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if (node.module, alias.name) in _RISKY_FUNCS:
                    out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


# ── source scan ──────────────────────────────────────────────────────────────


def test_no_unguarded_deserialization_or_exec_in_source():
    """Fail if an unguarded RCE-class load or an eval/exec appears in source.

    Guarded exceptions: the joblib/skops calls inside sidecar_cli._load_model,
    which are behind an explicit trust_input / env opt-in (VB-SEC-6).
    """
    violations = []

    for path in _iter_source_files():
        rel = _rel(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _alias_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _call_dotted_name(node)
            # Resolve a bare call imported directly, e.g.
            # `from pickle import loads as X; X(...)` -> `pickle.loads`, so the
            # canonical rules below apply to aliased imports too.
            if dotted and "." not in dotted and dotted in aliases:
                dotted = aliases[dotted]
            base = dotted.split(".")[-1] if dotted else ""
            root = dotted.split(".")[0] if dotted else ""

            # eval()/exec() on anything: banned outright in source.
            if dotted in ("eval", "exec"):
                violations.append(f"{rel}:{node.lineno} {dotted}(...)")
                continue

            # pickle/cloudpickle/dill/marshal .load/.loads and joblib.load.
            is_pickle_load = root in ("pickle", "cloudpickle", "dill", "marshal") and base in (
                "load",
                "loads",
            )
            is_joblib_load = dotted.endswith("joblib.load") or dotted == "joblib.load"
            if is_pickle_load or is_joblib_load:
                fn = _enclosing_func(tree, node)
                if (rel, fn) in _GUARDED_FUNCTIONS:
                    continue
                violations.append(f"{rel}:{node.lineno} {dotted}(...) [pickle/joblib/marshal]")
                continue

            # Other RCE-class deserializers (none used today; this keeps them
            # from being introduced without review): pandas.read_pickle,
            # jsonpickle.decode/loads, shelve.open.
            if (
                (root in ("pd", "pandas") and base == "read_pickle")
                or (root == "jsonpickle" and base in ("decode", "loads"))
                or dotted.endswith("shelve.open")
            ):
                violations.append(f"{rel}:{node.lineno} {dotted}(...) [pickle-class]")
                continue

            # skops.io.load(trusted=True) fully disables the allowlist.
            prefix = dotted.split(".")[:-1]
            is_skops_load = base == "load" and ("sio" in prefix or "skops" in prefix)
            if is_skops_load and _is_true(_kw(node, "trusted")):
                fn = _enclosing_func(tree, node)
                if (rel, fn) in _GUARDED_FUNCTIONS:
                    continue
                violations.append(f"{rel}:{node.lineno} skops load(trusted=True)")
                continue

            # torch.load without weights_only=True is an RCE sink.
            if dotted.endswith("torch.load"):
                wo = _kw(node, "weights_only")
                if wo is None or _is_false(wo):
                    violations.append(f"{rel}:{node.lineno} torch.load without weights_only=True")
                continue

            # numpy.load with allow_pickle=True on untrusted input.
            if dotted in ("np.load", "numpy.load"):
                ap = _kw(node, "allow_pickle")
                if _is_true(ap):
                    violations.append(f"{rel}:{node.lineno} np.load(allow_pickle=True)")
                continue

            # yaml.load without SafeLoader is unsafe.
            if dotted.endswith("yaml.load"):
                loader = _kw(node, "Loader")
                loader_name = ""
                if isinstance(loader, ast.Attribute):
                    loader_name = loader.attr
                elif isinstance(loader, ast.Name):
                    loader_name = loader.id
                if "Safe" not in loader_name:
                    violations.append(f"{rel}:{node.lineno} yaml.load without SafeLoader")
                continue

    assert not violations, (
        "Unguarded deserialization / dynamic-execution sites found:\n  " + "\n  ".join(violations)
    )


def test_scan_actually_covered_source_files():
    """Guard against the scan silently matching zero files (path drift)."""
    files = list(_iter_source_files())
    assert len(files) > 50, f"source scan only saw {len(files)} files; path may be wrong"
    assert any(_rel(p) == "xai/sidecar_cli.py" for p in files)


# ── behavioral: the guarded loader ───────────────────────────────────────────


def _joblib_blob(obj) -> str:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_load_model_refuses_without_opt_in(monkeypatch):
    monkeypatch.delenv(sidecar_cli.TRUST_INPUT_ENV, raising=False)
    blob = _joblib_blob({"hello": "world"})
    with pytest.raises(ValueError) as exc:
        sidecar_cli._load_model(blob, "joblib")
    msg = str(exc.value).lower()
    # The error must name the risk so the caller understands why.
    assert "remote-code-execution" in msg or "arbitrary code" in msg
    assert "trust_input" in msg


def test_load_model_works_with_payload_opt_in(monkeypatch):
    monkeypatch.delenv(sidecar_cli.TRUST_INPUT_ENV, raising=False)
    blob = _joblib_blob({"hello": "world"})
    loaded = sidecar_cli._load_model(blob, "joblib", trust_input=True)
    assert loaded == {"hello": "world"}


def test_load_model_works_with_env_opt_in(monkeypatch):
    monkeypatch.setenv(sidecar_cli.TRUST_INPUT_ENV, "1")
    blob = _joblib_blob([1, 2, 3])
    loaded = sidecar_cli._load_model(blob, "joblib")
    assert loaded == [1, 2, 3]


def test_env_opt_in_is_falsey_by_default(monkeypatch):
    monkeypatch.setenv(sidecar_cli.TRUST_INPUT_ENV, "0")
    assert sidecar_cli._env_opt_in() is False
    monkeypatch.setenv(sidecar_cli.TRUST_INPUT_ENV, "false")
    assert sidecar_cli._env_opt_in() is False
    monkeypatch.setenv(sidecar_cli.TRUST_INPUT_ENV, "yes")
    assert sidecar_cli._env_opt_in() is True


def test_main_reports_error_when_not_opted_in(monkeypatch, capsys):
    """End-to-end: an unopted payload with a model blob fails cleanly, no RCE."""
    monkeypatch.delenv(sidecar_cli.TRUST_INPUT_ENV, raising=False)
    payload = {
        "csv_data": "a,b\n1,2\n3,4\n",
        "feature_columns": ["a", "b"],
        "row_index": 0,
        "model_base64": _joblib_blob({"x": 1}),
        "model_format": "joblib",
    }
    import json

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    sidecar_cli.main()
    out = capsys.readouterr().out
    result = json.loads(out.strip().splitlines()[-1])
    assert result["success"] is False
    assert "trust_input" in result["error"] or "remote-code-execution" in result["error"].lower()
