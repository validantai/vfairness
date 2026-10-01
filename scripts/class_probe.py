#!/usr/bin/env python3
"""Execute every ungraded CLASS, and settle the ones a program honestly can.

WHY THIS EXISTS. `scripts/surface_probe.py` probes functions and methods. It has
never probed a class, so all 392 classes on the public surface sat in "nobody has
looked", 25% of the library, including 167 plain dataclasses whose whole job is to
hold what somebody else measured. Paying an agent 69,000 tokens to decide that a
result container does not compute a fairness number is paying agent prices for
machine work.

WHAT A PROGRAM MAY CONCLUDE HERE, and the one discriminator that makes it honest.
The question for a class is narrow: **does constructing it produce a number
derived from the data?** If it does not, the fabrication defect cannot live in the
constructor, and NOT A MEASUREMENT is a measured statement rather than a guess. If
it does, this script says NEEDS JUDGEMENT and stops. It never grades a computing
constructor.

The discriminator is not the class's type and not its name. Both lie: a dataclass
can have a `__post_init__` that computes, and a class called `...Result` can run
an analysis in its constructor. It is this:

    construct the class on the healthy world and on each degenerate world,
    then compare the state it holds that did NOT come in through its arguments.

  - nothing untraceable to an argument, on every world  -> it stores, it does not
    compute. NOT A MEASUREMENT.
  - untraceable state, but byte-identical on every world -> a constant default,
    not a measurement of anything. NOT A MEASUREMENT.
  - untraceable state that CHANGES with the data        -> the constructor derives
    a number from the data. NEEDS JUDGEMENT, never graded here.

Clause 1 of `grade_from_probe.py` applies unchanged and for the same reason: if
construction failed on the HEALTHY world we called it wrong, and its refusals on
degenerate worlds say nothing. Those are recorded NOT EXAMINED, not graded.

Run `--selftest` first. It builds three classes: a container, a computing
constructor and one that fabricates a neutral value exactly the way this library's
real defects do, and fails unless the rule separates them. A rule that cannot
fail looks exactly like a rule that passed.

Usage:
    python scripts/class_probe.py --selftest
    python scripts/class_probe.py --out docs/class-probe.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import library_kpis  # noqa: E402
import surface_probe as SP  # noqa: E402

NOT_A_MEASUREMENT = "NOT A MEASUREMENT"
NEEDS_JUDGEMENT = "NEEDS JUDGEMENT"
NOT_EXAMINED = "NOT EXAMINED"

# At least this many degenerate worlds must have been attempted before any
# conclusion is drawn, mirroring clause 2 of grade_from_probe.py.
MIN_DEGENERATE = 4


def _numberish(value: Any) -> bool:
    """Does this value carry a number the constructor could have invented?

     WIDENED after the first full run, which found 217 containers and NOT ONE
     computing constructor. That was the rule being blind, not the library being
     clean: it only looked at scalar attributes, so `self.metrics = {"gap": 0.0}`
    : a fabricated disparity in the shape this library actually ships it: read
     as a container storing its arguments. The hole was in the flattering
     direction, which is the defect this whole programme exists to remove, so
     collections of numbers count too.
    """
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return True
    # numpy scalars are the reason this is not a bare isinstance check: a
    # canonical predicate in this repo once discarded every np.float32 it was
    # handed and reported the result as caution.
    if hasattr(value, "dtype") and getattr(value, "shape", None) == ():
        return True
    if isinstance(value, dict):
        return any(_numberish(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_numberish(v) for v in value)
    return False


def _render(value: Any) -> str:
    """A bounded, order-stable rendering, so two worlds can be compared."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                f"{k!r}:{_render(v)}" for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))
            )[:400]
            + "}"
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else list(value)
        return "[" + ",".join(_render(v) for v in items[:20])[:400] + "]"
    return repr(value)[:200]


def _signature(obj: Any, passed: list) -> dict:
    """Public state that did NOT arrive through the constructor's arguments."""
    out: dict[str, str] = {}
    state = getattr(obj, "__dict__", None)
    if not isinstance(state, dict):
        return out
    for name, value in state.items():
        if name.startswith("_") or not _numberish(value):
            continue
        if any(_same(value, p) for p in passed):
            continue
        out[name] = _render(value)
    return out


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - arrays and frames refuse to be compared
        return False


def _flatten(args: list, kwargs: dict) -> list:
    out = list(args) + list(kwargs.values())
    for value in list(out):
        if isinstance(value, dict):
            out.extend(value.values())
        elif isinstance(value, (list, tuple)):
            out.extend(value)
    return out


def probe_class(cls: Any, worlds: dict) -> dict:
    runs: dict[str, dict] = {}
    for name, world in worlds.items():
        try:
            args, kwargs, why = SP._bind(cls.__init__, world)
        except Exception as exc:  # noqa: BLE001
            runs[name] = {"outcome": "NOT_REACHED", "detail": f"bind failed: {type(exc).__name__}"}
            continue
        if why:
            runs[name] = {"outcome": "NOT_REACHED", "detail": why}
            continue
        old = SP._deadline(SP.CALL_TIMEOUT)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with open(os.devnull, "w") as null:
                    with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                        obj = cls(*args, **kwargs)
            runs[name] = {
                "outcome": "constructed",
                "derived": _signature(obj, _flatten(args, kwargs)),
                "n_warnings": len(caught),
            }
        except SP._TimeoutError:
            runs[name] = {"outcome": "hung", "detail": "constructor never returned"}
        except SystemExit as exc:
            runs[name] = {"outcome": "raised", "detail": f"SystemExit({exc.code})"}
        except Exception as exc:  # noqa: BLE001 - a refusal, recorded
            runs[name] = {
                "outcome": "raised",
                "detail": f"{type(exc).__name__}: {SP._public_diagnostic(exc, 120)}",
            }
        finally:
            SP._clear_deadline(old)
    return {"runs": runs, **grade(runs)}


def grade(runs: dict) -> dict:
    healthy = runs.get("healthy") or {}
    if healthy.get("outcome") != "constructed":
        return {
            "grade": NOT_EXAMINED,
            "reason": "could not be constructed on the healthy world, so nothing was learned",
        }
    degenerate = [v for k, v in runs.items() if k != "healthy"]
    attempted = [v for v in degenerate if v.get("outcome") in ("constructed", "raised", "hung")]
    if len(attempted) < MIN_DEGENERATE:
        return {
            "grade": NOT_EXAMINED,
            "reason": f"only {len(attempted)} degenerate worlds could be attempted",
        }
    built = [v for v in [healthy, *degenerate] if v.get("outcome") == "constructed"]
    derived = [v.get("derived") or {} for v in built]
    if all(not d for d in derived):
        return {
            "grade": NOT_A_MEASUREMENT,
            "reason": "constructing it stores its arguments and derives no number of its own",
        }
    first = derived[0]
    if all(d == first for d in derived):
        return {
            "grade": NOT_A_MEASUREMENT,
            "reason": f"the only state it derives is constant across every world ({sorted(first)}), so it measures nothing",
        }
    changing = sorted({k for d in derived for k in d if any(d.get(k) != o.get(k) for o in derived)})
    return {
        "grade": NEEDS_JUDGEMENT,
        "reason": f"the constructor derives data-dependent state ({changing}); whether it refuses honestly needs judgement",
    }


def selftest() -> int:
    """Three classes, one rule, and it must separate them."""

    class Container:
        def __init__(self, y_true, y_pred, sensitive_features):
            self.y_true, self.y_pred = y_true, y_pred
            self.sensitive_features = sensitive_features
            self.version = 2

    class Computes:
        def __init__(self, y_true, y_pred, sensitive_features):
            self.gap = float(len(set(map(str, sensitive_features))))

    class Fabricates:
        """The real defect shape: a neutral value where nothing was measurable."""

        def __init__(self, y_true, y_pred, sensitive_features):
            seen = set(map(str, sensitive_features))
            self.disparity = 0.0 if len(seen) < 2 else float(len(seen)) / 3.0

    class FabricatesInsideADict:
        """The same fabrication one container deep, which the first rule missed."""

        def __init__(self, y_true, y_pred, sensitive_features):
            seen = set(map(str, sensitive_features))
            self.metrics = {"gap": 0.0 if len(seen) < 2 else float(len(seen)) / 3.0}

    worlds = {s: SP._world(s) for s in SP.SCENARIOS}
    expected = {
        "Container": NOT_A_MEASUREMENT,
        "Computes": NEEDS_JUDGEMENT,
        "Fabricates": NEEDS_JUDGEMENT,
        "FabricatesInsideADict": NEEDS_JUDGEMENT,
    }
    failures = []
    for cls in (Container, Computes, Fabricates, FabricatesInsideADict):
        got = probe_class(cls, worlds)
        want = expected[cls.__name__]
        flag = "ok " if got["grade"] == want else "FAIL"
        print(f"  {flag} {cls.__name__:<12} -> {got['grade']:<16} {got['reason'][:70]}")
        if got["grade"] != want:
            failures.append(f"{cls.__name__}: wanted {want}, got {got['grade']}")
    if failures:
        for f in failures:
            print(f"SELFTEST FAILED: {f}")
        return 1
    print("  selftest passed: the rule refuses to grade a fabricating constructor")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.argv = [sys.argv[0]]
    with contextlib.suppress(Exception):
        os.dup2(open(os.devnull).fileno(), 0)
    if args.selftest:
        return selftest()
    if not args.out:
        ap.error("--out is required unless --selftest")
    out_path = Path(args.out).resolve()

    surface = library_kpis._walk_public_surface()
    classes = sorted(surface["classes"]["ungraded"])
    if args.limit:
        classes = classes[: args.limit]

    # Probed constructors write files relative to the working directory. The
    # function probe learned this the expensive way: a renderer took a column
    # name for an output path and left a 132KB file in the repository root.
    os.chdir(tempfile.mkdtemp(prefix="class-probe-"))
    worlds = {s: SP._world(s) for s in SP.SCENARIOS}

    out: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for i, qual in enumerate(classes, 1):
        entry = library_kpis._OBJECTS.get(qual)
        if entry is None:
            out[qual] = {"grade": NOT_EXAMINED, "reason": "object not resolvable", "runs": {}}
        else:
            out[qual] = probe_class(entry[1], worlds)
        counts[out[qual]["grade"]] = counts.get(out[qual]["grade"], 0) + 1
        if i % 50 == 0:
            print(f"  {i}/{len(classes)}", file=sys.stderr)
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(classes)} classes probed: {counts}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
