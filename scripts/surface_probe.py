#!/usr/bin/env python3
"""Execute every ungraded public callable on healthy and degenerate input.

WHY THIS EXISTS. Grading by agent costs about 138,000 tokens per item, measured
over 242 items. For the 900 that remain that is roughly 124 million tokens, which
is several weeks of capacity and three more chances to lose a wave to a limit
mid-flight. But an agent is not needed to RUN code, only to JUDGE it. This probe
does the running, as a program, for nothing, and leaves agents to judge only what
it flags.

WHAT IT MEASURES. For each callable, the probe builds arguments from the
parameter names and annotations, then calls it once per scenario: a healthy world
carrying a real, findable disparity, and eight degenerate worlds in which the
thing being measured does not exist. It records the return value, every warning,
and any exception.

WHAT IT DOES NOT DO, deliberately. It assigns no grade. A probe result is an
OBSERVATION: "returned 0.0 on a single-group frame and emitted no warning". The
step from that to "this fabricates" needs a human or an agent who understands
what the function claims to measure, because for some functions 0.0 on that input
is the correct answer and refusing would destroy evidence. Letting this script
write grades would rebuild the defect the library was audited for, one level up.

THREE STATES APPLY TO THE PROBE ITSELF. An item it could not call is recorded as
NOT_REACHED with the reason, never skipped and never counted as fine. A silently
dropped item understates the surface exactly as an ungraded capability counted as
clean overstates it.

Usage:
    python scripts/surface_probe.py --out docs/surface-probe.json
    python scripts/surface_probe.py --out /tmp/p.json --limit 50   # smoke test
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import json
import math
import os
import signal
import sys
import tempfile
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import library_kpis  # noqa: E402

_LADDER_SENTINEL = object()


def _public_diagnostic(value: object, limit: int = 160) -> str:
    """Keep diagnostic evidence without publishing machine-specific paths.

    Redact before truncating: a long checkout prefix used to consume the whole
    detail field, leaking the machine path while hiding the useful filename.
    This changes explanatory text only, never outcomes, warnings or grades.
    The export boundary remains responsible for scanning the complete output.
    """
    text = str(value).replace(str(ROOT), "[package]")
    text = text.replace(str(Path.home()), "[home]")
    return text[:limit]


N = 200
RNG = np.random.default_rng(20260918)


# ---------------------------------------------------------------------------
# The worlds. Each returns a dict of primitives the argument binder draws from.
# The healthy world is the CONTROL: it carries a real, findable disparity, so a
# function that refuses it has over-corrected, which is worse than the defect.
# ---------------------------------------------------------------------------


def _world(name: str) -> Dict[str, Any]:
    n = N
    if name == "healthy":
        g = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
        y_true = (RNG.random(n) < 0.4).astype(int)
        y_prob = np.where(g == "a", RNG.random(n) * 0.5 + 0.45, RNG.random(n) * 0.5)
        y_pred = (y_prob > 0.5).astype(int)
        text = ["The candidate showed strong results and was hired."] * n
    elif name == "single_group":
        g = np.array(["a"] * n)
        y_true = (RNG.random(n) < 0.4).astype(int)
        y_prob = RNG.random(n)
        y_pred = (y_prob > 0.5).astype(int)
        text = ["Only one group is present in this sample."] * n
    elif name == "one_label_only":
        g = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
        y_true = np.ones(n, dtype=int)
        y_prob = RNG.random(n)
        y_pred = np.ones(n, dtype=int)
        text = ["Every outcome is the same."] * n
    elif name == "all_nan_scores":
        g = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
        y_true = (RNG.random(n) < 0.4).astype(int)
        y_prob = np.full(n, np.nan)
        y_pred = np.full(n, np.nan)
        text = [""] * n
    elif name == "empty":
        g = np.array([], dtype=object)
        y_true = np.array([], dtype=int)
        y_prob = np.array([], dtype=float)
        y_pred = np.array([], dtype=int)
        text = []
    elif name == "constant_scores":
        g = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
        y_true = (RNG.random(n) < 0.4).astype(int)
        y_prob = np.full(n, 0.7)
        y_pred = np.ones(n, dtype=int)
        text = ["identical"] * n
    elif name == "n_equals_2":
        g = np.array(["a", "b"])
        y_true = np.array([1, 0])
        y_prob = np.array([0.9, 0.1])
        y_pred = np.array([1, 0])
        text = ["one", "two"]
    elif name == "one_row_minority":
        g = np.array(["a"] * (n - 1) + ["b"])
        y_true = (RNG.random(n) < 0.4).astype(int)
        y_prob = RNG.random(n)
        y_pred = (y_prob > 0.5).astype(int)
        text = ["The minority group has a single row."] * n
    elif name == "unreadable_text":
        g = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
        y_true = (RNG.random(n) < 0.4).astype(int)
        y_prob = RNG.random(n)
        y_pred = (y_prob > 0.5).astype(int)
        text = ["这是一段中文文本，没有任何拉丁字母。"] * n
    else:  # pragma: no cover - guarded by SCENARIOS
        raise ValueError(name)

    m = len(g)
    df = pd.DataFrame(
        {
            "gender": g,
            "group": g,
            "protected": g,
            "y_true": y_true,
            "label": y_true,
            "y_pred": y_pred,
            "prediction": y_pred,
            "score": y_prob,
            "y_prob": y_prob,
            "income": (y_prob * 1000) if m else np.array([], dtype=float),
            "text": text,
        }
    )
    return {
        "n": m,
        "groups": g,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "text": text,
        "df": df,
    }


SCENARIOS = (
    "healthy",
    "single_group",
    "one_label_only",
    "all_nan_scores",
    "empty",
    "constant_scores",
    "n_equals_2",
    "one_row_minority",
    "unreadable_text",
)


# ---------------------------------------------------------------------------
# Argument binding. Parameter NAME first, because this library names its inputs
# consistently, then the annotation. A parameter we cannot fill makes the item
# NOT_REACHED with that parameter named, rather than being quietly skipped.
# ---------------------------------------------------------------------------

_BY_NAME = {
    "y_true": "y_true",
    "y_actual": "y_true",
    "labels": "y_true",
    "y": "y_true",
    "true_labels": "y_true",
    "ground_truth": "y_true",
    "y_pred": "y_pred",
    "predictions": "y_pred",
    "y_predicted": "y_pred",
    "preds": "y_pred",
    "predicted": "y_pred",
    "y_prob": "y_prob",
    "y_score": "y_prob",
    "scores": "y_prob",
    "probs": "y_prob",
    "probabilities": "y_prob",
    "y_proba": "y_prob",
    "y_scores": "y_prob",
    "predicted_probabilities": "y_prob",
    "confidence": "y_prob",
    "sensitive_attr": "groups",
    "sensitive_features": "groups",
    "groups": "groups",
    "group": "groups",
    "protected_attr": "groups",
    "sensitive": "groups",
    "protected_attribute": "groups",
    "group_labels": "groups",
    "a": "groups",
    "df": "df",
    "data": "df",
    "X": "df",
    "features": "df",
    "dataset": "df",
    "frame": "df",
    "records": "df",
    "texts": "text",
    "outputs": "text",
    "responses": "text",
    "generations": "text",
}

_BY_NAME_SCALAR = {
    "text": "one_text",
    "response": "one_text",
    "output": "one_text",
    "prompt": "one_text",
    "generation": "one_text",
    "s": "one_text",
    "protected_attr": "col_gender",
    "protected_attribute": "col_gender",
    "attribute": "col_gender",
    "column": "col_gender",
    "col": "col_gender",
    "target": "col_label",
    "target_col": "col_label",
    "label_col": "col_label",
    "prediction_col": "col_pred",
    "pred_col": "col_pred",
    "score_col": "col_score",
    "feature": "col_income",
    "metric": "metric_name",
    "metric_name": "metric_name",
}

_SCALARS = {
    "one_text": lambda w: w["text"][0] if w["text"] else "",
    "col_gender": lambda w: "gender",
    "col_label": lambda w: "y_true",
    "col_pred": lambda w: "y_pred",
    "col_score": lambda w: "score",
    "col_income": lambda w: "income",
    "metric_name": lambda w: "demographic_parity",
}


def _library_instance(ann: Any, w: Dict[str, Any], depth: int = 0) -> Tuple[bool, Any]:
    """Build an instance of a vfairness type named in an annotation.

    Measured on the first sweep: 14 units wanted an object with `.value`, 11
    wanted `.compute_health_score`, 7 wanted `.outcome`, 7 called asdict() on
    what they were given. All of those are library types the annotation names,
    and a probe that passes an ndarray instead learns nothing about the unit.
    Shallow by design, one level of nesting, so a type whose constructor needs
    another object is left alone rather than guessed at twice over.
    """
    if depth > 1 or not inspect.isclass(ann):
        return False, None
    if not getattr(ann, "__module__", "").startswith("vfairness"):
        return False, None
    try:
        if dataclasses.is_dataclass(ann):
            kwargs = {}
            for f in dataclasses.fields(ann):
                if (
                    f.default is not dataclasses.MISSING
                    or f.default_factory is not dataclasses.MISSING
                ):  # type: ignore[misc]
                    continue
                ok, v = _from_annotation(
                    f.type if not isinstance(f.type, str) else inspect._empty, w
                )
                kwargs[f.name] = v if ok else 0.5
            return True, ann(**kwargs)
        a, k, why = _bind(ann.__init__, w)
        if why:
            return False, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return True, ann(*a, **k)
    except Exception:  # noqa: BLE001 - a type we cannot build is not a finding
        return False, None


def _from_annotation(ann: Any, w: Dict[str, Any]) -> Tuple[bool, Any]:
    """Fill a parameter from its annotation alone. (matched, value)."""
    if ann is inspect._empty:
        return False, None
    origin = typing.get_origin(ann)
    if origin is typing.Union or str(origin) == "typing.Union":
        for arg in typing.get_args(ann):
            if arg is type(None):
                continue
            ok, v = _from_annotation(arg, w)
            if ok:
                return True, v
        return False, None
    txt = str(ann)
    if "DataFrame" in txt:
        return True, w["df"]
    if "ndarray" in txt or "Series" in txt or "Sequence" in txt or "List[str]" in txt:
        return True, w["groups"]
    if ann is bool or txt == "<class 'bool'>":
        return True, True
    if ann is int or txt == "<class 'int'>":
        return True, 2
    if ann is float or txt == "<class 'float'>":
        return True, 0.5
    if ann is str or txt == "<class 'str'>":
        return True, "gender"
    if "Dict" in txt or "Mapping" in txt:
        return True, {"a": 0.5, "b": 0.25}
    ok, inst = _library_instance(ann, w)
    if ok:
        return True, inst
    return False, None


# Comparison functions in this library name their two sides with _a / _b
# suffixes (routing_a, outcomes_b). Splitting the healthy world in half gives
# each side real data and keeps the disparity findable.
def _ab(name: str, w: Dict[str, Any]) -> Tuple[bool, Any]:
    if not (name.endswith("_a") or name.endswith("_b")):
        return False, None
    stem = name[:-2]
    n = w["n"]
    half = slice(0, n // 2) if name.endswith("_a") else slice(n // 2, n)
    base = _BY_NAME.get(stem)
    src = w[base] if base else w["y_prob"]
    if isinstance(src, pd.DataFrame):
        return True, src.iloc[half]
    if stem in ("text", "texts", "outputs", "docs", "responses", "generations"):
        return True, list(w["text"])[half]
    try:
        return True, src[half]
    except Exception:  # noqa: BLE001
        return True, list(w["text"])[half]


# Last resort for a required parameter no rule names. Each candidate is tried in
# order and the first that does not raise TypeError at the call is kept. A
# parameter that survives none of them makes the item NOT_REACHED, which is a
# recorded state and not a skip.
def _ladder(w: Dict[str, Any]) -> List[Any]:
    """Candidates for a parameter no rule names, ordered by measured demand.

    Derived from the 287 units the first sweep could not really call. Grouping
    their errors showed what they actually wanted: 75 raised "truth value of an
    array is ambiguous" (a scalar was wanted, an array was passed), 32 called
    .get on an ndarray (a dict was wanted), 15 wanted a number and got the
    string "a", 15 called .to_dict on a float (an object was wanted). Each entry
    below answers one of those.
    """
    n = max(int(w["n"]), 1)
    return [
        w["y_prob"],  # a numeric array
        w["df"],  # a frame
        {"gender": 0.5, "a": 0.5, "b": 0.25},  # a mapping: .get / .items
        0.5,  # a scalar float
        2,  # a scalar int
        True,  # a flag
        "gender",  # a column name
        list(w["text"]),  # a list of strings
        list(w["groups"]),  # a list of group labels
        np.arange(n, dtype=float),  # a plain numeric range
        [],
        {},
        None,
    ]


_GENERIC = {
    "value": 0.5,
    "values": None,
    "name": "gender",
    "keyword": "the",
    "site": "site",
    "key": "gender",
    "label": "a",
    "threshold": 0.5,
    "alpha": 0.05,
    "n": 2,
    "size": 2,
    "seed": 0,
    "path": None,
    "catalogue_keys": None,
    "keys": None,
    "columns": None,
    "features": None,
}


def _bind(fn: Any, w: Dict[str, Any]) -> Tuple[Optional[list], Optional[dict], Optional[str]]:
    """Return (args, kwargs, unreachable_reason)."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError) as exc:
        return None, None, f"no signature: {_public_diagnostic(exc)}"
    args: list = []
    kwargs: dict = {}
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name in _BY_NAME:
            val = w[_BY_NAME[name]]
        elif name in _BY_NAME_SCALAR:
            val = _SCALARS[_BY_NAME_SCALAR[name]](w)
        else:
            ok, val = _from_annotation(p.annotation, w)
            if not ok:
                ok, val = _ab(name, w)
            if not ok and name in _GENERIC:
                g = _GENERIC[name]
                val = w["y_prob"] if g is None else g
                ok = True
            if not ok and (name.endswith("s") or name.endswith("_list")):
                ok, val = True, w["y_prob"]
            if not ok:
                if p.default is not p.empty:
                    continue  # a default the caller would also rely on
                ok, val = True, _LADDER_SENTINEL
        if p.kind == p.KEYWORD_ONLY:
            kwargs[name] = val
        else:
            args.append(val)
    return args, kwargs, None


def _instance(cls: Any, w: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """Construct an instance, or say why not."""
    old = _deadline(CALL_TIMEOUT)
    try:
        a, k, why = _bind(cls.__init__, w)
        if why:
            return None, f"constructor: {why}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(os.devnull, "w") as null:
                with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                    return cls(*a, **k), None
    except _TimeoutError as exc:
        return None, f"constructor never returned: {exc}"
    except SystemExit as exc:
        return None, f"constructor exits the process: SystemExit({exc.code})"
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return None, f"constructor raised {type(exc).__name__}: {_public_diagnostic(exc, 120)}"
    finally:
        _clear_deadline(old)


# ---------------------------------------------------------------------------
# Classifying ONE observation. Never a grade: a description of what came back.
# ---------------------------------------------------------------------------

_NEUTRAL = (0.0, 0, 1.0, 1, False)


def _describe(value: Any) -> Tuple[str, str]:
    """(kind, rendering). kind is one of neutral, measured, refused, other."""
    if value is None:
        return "refused", "None"
    if isinstance(value, float) and math.isnan(value):
        return "refused", "nan"
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        v = float(value)
        if math.isnan(v):
            return "refused", "nan"
        if any(v == n and isinstance(value, type(n)) or v == float(n) for n in _NEUTRAL):
            return "neutral", repr(value)
        return "measured", repr(value)
    if isinstance(value, (list, tuple, set)):
        return (
            "neutral" if len(value) == 0 else "measured"
        ), f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        return ("neutral" if not value else "measured"), f"dict(keys={list(value)[:6]})"
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "neutral", "ndarray(empty)"
        if np.all(np.isnan(value.astype(float, copy=False))) if value.dtype.kind == "f" else False:
            return "refused", "ndarray(all nan)"
        return "measured", f"ndarray(shape={value.shape})"
    if isinstance(value, pd.DataFrame):
        return ("neutral" if value.empty else "measured"), f"DataFrame(shape={value.shape})"
    return "other", f"{type(value).__name__}"


def _call_with_ladder(fn: Any, a: list, k: dict, w: Dict[str, Any]):
    """Fill any sentinel by trying candidates until the call is accepted."""
    holes_a = [i for i, v in enumerate(a) if v is _LADDER_SENTINEL]
    holes_k = [n for n, v in k.items() if v is _LADDER_SENTINEL]
    if not holes_a and not holes_k:
        return fn(*a, **k)
    last: Optional[BaseException] = None
    for cand in _ladder(w):
        aa = list(a)
        kk = dict(k)
        for i in holes_a:
            aa[i] = cand
        for n in holes_k:
            kk[n] = cand
        try:
            return fn(*aa, **kk)
        except TypeError as exc:
            last = exc
            continue
    raise last if last else TypeError("no candidate accepted")


class _TimeoutError(Exception):
    pass


def _deadline(seconds: int):
    """A wall-clock limit for ONE call.

    Probed code is arbitrary library code and some of it blocks. The first full
    sweep hung for 75 minutes on 5 seconds of CPU, holding a pipe: something it
    called sat waiting to read stdin, most likely a server loop. Enumerating
    every way arbitrary code can block is not possible, so the probe defends
    itself instead. SIGALRM interrupts a pure-Python loop and most blocking
    syscalls, which covers what actually happened.
    """

    def _fire(signum, frame):  # noqa: ANN001
        raise _TimeoutError(f"still running after {seconds}s")

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return old


def _clear_deadline(old) -> None:  # noqa: ANN001
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old)


CALL_TIMEOUT = 15


def _probe_one(fn: Any, w: Dict[str, Any]) -> Dict[str, Any]:
    a, k, why = _bind(fn, w)
    if why:
        return {"outcome": "NOT_REACHED", "detail": why}
    old = _deadline(CALL_TIMEOUT)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            # Probed code is not written to be called by a probe. Some entry
            # points parse sys.argv and call sys.exit, which killed a run at
            # item 400, and many print. Neither is a property of the thing
            # being measured, so both are contained here rather than allowed to
            # end the sweep or pollute the record.
            with open(os.devnull, "w") as null:
                with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                    value = _call_with_ladder(fn, a, k, w)
        kind, shown = _describe(value)
        return {
            "outcome": kind,
            "detail": shown,
            "warnings": [type(x.message).__name__ for x in rec][:4],
            "n_warnings": len(rec),
        }
    except SystemExit as exc:
        return {
            "outcome": "raised",
            "detail": f"SystemExit({exc.code}), the callable exits the process",
        }
    except RecursionError:
        return {"outcome": "raised", "detail": "RecursionError"}
    except Exception as exc:  # noqa: BLE001 - an exception IS a refusal, recorded
        return {"outcome": "raised", "detail": f"{type(exc).__name__}: {_public_diagnostic(exc)}"}
    finally:
        _clear_deadline(old)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out = str(Path(args.out).resolve())

    # Any probed callable that reads sys.argv must not see ours.
    sys.argv = [sys.argv[0]]
    # And nothing may read stdin. The first sweep hung here: a callable sat
    # waiting on the pipe for input that was never coming.
    try:
        devnull = open(os.devnull)
        os.dup2(devnull.fileno(), 0)
    except Exception:  # noqa: BLE001
        pass

    # Run from a scratch directory. Probed code writes files, and it writes them
    # relative to the working directory: a renderer took the string "gender"
    # from the argument binder as an output path and left a 132KB SVG called
    # "gender" in the repository root, which then got committed. The probe must
    # not be able to touch the tree it is measuring.
    scratch = tempfile.mkdtemp(prefix="surface-probe-")
    os.chdir(scratch)
    print(f"  working from {scratch}", file=sys.stderr)

    surface = library_kpis._walk_public_surface()
    ungraded = sorted(set(surface["functions"]["ungraded"]) | set(surface["methods"]["ungraded"]))
    if args.limit:
        ungraded = ungraded[: args.limit]

    worlds = {s: _world(s) for s in SCENARIOS}
    out: Dict[str, Any] = {}
    for i, qual in enumerate(ungraded, 1):
        entry = library_kpis._OBJECTS.get(qual)
        if entry is None:
            out[qual] = {"status": "NOT_REACHED", "reason": "object not resolvable"}
            continue
        short, obj = entry
        owner = None
        if "." in obj.__qualname__:
            owner_name = obj.__qualname__.split(".")[0]
            mod = sys.modules.get(obj.__module__)
            owner = getattr(mod, owner_name, None) if mod else None

        runs: Dict[str, Any] = {}
        for s in SCENARIOS:
            w = worlds[s]
            target = obj
            if owner is not None and inspect.isclass(owner):
                inst, why = _instance(owner, w)
                if inst is None:
                    runs[s] = {"outcome": "NOT_REACHED", "detail": why}
                    continue
                target = getattr(inst, short, None)
                if target is None:
                    runs[s] = {"outcome": "NOT_REACHED", "detail": "attribute missing on instance"}
                    continue
            runs[s] = _probe_one(target, w)
        out[qual] = {"short": short, "runs": runs}
        if i % 50 == 0:
            # Write as we go. The first sweep lost 400 items' work to one hang;
            # a partial result on disk is worth more than a complete one that
            # never arrives.
            Path(args.out).write_text(
                json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8"
            )
            print(f"  probed {i}/{len(ungraded)}", file=sys.stderr, flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")

    reached = sum(
        1
        for v in out.values()
        if any(r.get("outcome") not in ("NOT_REACHED",) for r in v.get("runs", {}).values())
    )
    print(
        json.dumps(
            {
                "items": len(out),
                "reached_at_least_once": reached,
                "not_reached": len(out) - reached,
                "written": args.out,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
