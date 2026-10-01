#!/usr/bin/env python3
"""Which public code units does the TEST SUITE execute? Measured, not assumed.

WHY THIS EXISTS, and it is the largest single correction to this library's published
readiness. The release gate's B1 asks whether every public code unit "has been
executed and its result recorded", and it counted three sources: the capability
census, the grading waves, and the surface probe. It never counted the test suite.

Measured 2026-09-25: the suite executes **1,346 of 1,564** public code units. B1 was
reporting 920. The 426 units in the gap were not unexamined; they were examined by
10,552 passing tests that the gate did not look at, while the probe, built
specifically to execute code nobody had run, was credited for 459.

That is a measurement error in the UNFLATTERING direction, and it mattered: it put
"close B1" at roughly 1,100 hand-built fixtures spread over 343 owners, which is a
seven-to-twelve-million-token programme and therefore a bar nobody would reach. With
the suite counted, **166** units have never been executed by anything.

WHAT THIS EVIDENCE DOES AND DOES NOT SUPPORT, because the distinction is the whole
point of the three-state ledger:

  IT SUPPORTS "executed and its result recorded" (B1). A unit whose body lines run
  under a green suite has been executed, and an assertion recorded the outcome.

  IT DOES NOT SUPPORT "CHECKED". Being executed is not being judged for the
  fabrication class: a unit called incidentally inside another test has run, and
  nobody has asked whether it refuses honestly when nothing is measurable. Those
  units stay NOT CHECKED, with a far better REASON than "never executed", which is
  what the ledger used to say about 528 of them.

FAIL CLOSED ON THE SUITE'S OWN RESULT. A coverage run whose suite had failures is
not evidence that anything was verified, so the outcome is recorded here and
`--check` refuses a run with failures. Lines executed while a test was failing are
lines that ran on the way to a wrong answer.

HOW TO PRODUCE THE INPUT (about 14 minutes):

    python -m pytest -q --cov=vfairness --cov-report=json:/tmp/cov.json
    python scripts/suite_coverage.py --coverage /tmp/cov.json --out docs/suite-coverage.json

Usage:
    python scripts/suite_coverage.py --coverage COV.json --out docs/suite-coverage.json
    python scripts/suite_coverage.py --check      # exit 1 if the record is stale or red
"""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import re
import sys
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import library_kpis as K  # noqa: E402

OUT = ROOT / "docs" / "suite-coverage.json"


def _suite_outcome(log: pathlib.Path | None) -> dict:
    """The suite's own result, read from the pytest summary line if we have it."""
    if log is None or not log.exists():
        return {"recorded": False, "why": "no pytest log was supplied with the coverage run"}
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(\d+) passed", text)
    f = re.search(r"(\d+) failed", text)
    e = re.search(r"(\d+) error", text)
    return {
        "recorded": True,
        "passed": int(m.group(1)) if m else None,
        "failed": int(f.group(1)) if f else 0,
        "errors": int(e.group(1)) if e else 0,
    }


def _body_lines(source_lines: list[str], start: int) -> set:
    """The lines INSIDE a unit, excluding its definition header and decorators.

    Needed because a ``def`` or ``class`` header, its decorators and its default
    expressions all execute at IMPORT time. Intersecting a whole span with
    import-time coverage therefore matches every unit in the library and could not
    have disagreed: the first version of `_import_time_split` reported that all 218
    untouched units "execute at import", functions included, which is true only of
    their headers. The question worth asking is whether any line in the BODY ran.
    """
    header = start
    for offset, text in enumerate(source_lines):
        if text.strip().startswith(("def ", "class ", "async def ")):
            header = start + offset
            break
    # A signature can span several lines, so the body starts after the line that
    # closes it with a colon.
    for offset in range(header - start, len(source_lines)):
        stripped = source_lines[offset].rstrip()
        if stripped.endswith(":") and not stripped.strip().startswith("#"):
            return set(range(start + offset + 1, start + len(source_lines)))
    return set(range(header + 1, start + len(source_lines)))


# Run IN A SUBPROCESS, with coverage started before anything imports vfairness.
# In-process is not an option: this module imports library_kpis at the top, and that
# walks the public surface, which imports vfairness. By the time any function here
# runs, the library is already imported and coverage prints
# "Module vfairness was previously imported, but not measured" and then measures
# nothing. The first version did exactly that and recorded
# `at_import=0, nowhere=157`, which is the OPPOSITE of the truth and would have been
# published as a measurement. The warning was on stderr the whole time.
_IMPORT_SPLIT_CHILD = r"""
import json, pathlib, sys, warnings, pkgutil, inspect
warnings.filterwarnings("ignore")
import coverage
meter = coverage.Coverage(source=["vfairness"], data_file=None)
meter.start()
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "src"))
import library_kpis as K          # imports vfairness, now UNDER coverage
K._walk_public_surface()
import vfairness
for mod in pkgutil.walk_packages(vfairness.__path__, "vfairness."):
    try:
        __import__(mod.name)
    except Exception:
        pass
meter.stop()
data = meter.get_data()
lines_by_file = {
    str(pathlib.Path(name).resolve()): set(data.lines(name) or ())
    for name in data.measured_files()
}
quals = json.loads(pathlib.Path(sys.argv[2]).read_text())
at_import, nowhere = [], []
for qual in quals:
    entry = K._OBJECTS.get(qual)
    obj = entry[1] if entry else None
    if obj is None:
        nowhere.append(qual); continue
    try:
        src = str(pathlib.Path(inspect.getsourcefile(obj)).resolve())
        src_lines, start = inspect.getsourcelines(obj)
    except Exception:
        nowhere.append(qual); continue
    header = start
    for off, text in enumerate(src_lines):
        if text.strip().startswith(("def ", "class ", "async def ")):
            header = start + off
            break
    body = set(range(header + 1, start + len(src_lines)))
    for off in range(header - start, len(src_lines)):
        stripped = src_lines[off].rstrip()
        if stripped.endswith(":") and not stripped.strip().startswith("#"):
            body = set(range(start + off + 1, start + len(src_lines)))
            break
    (at_import if (lines_by_file.get(src, set()) & body) else nowhere).append(qual)
print(json.dumps({"at_import": at_import, "nowhere": nowhere}))
"""


def _import_time_split(untouched: list[str]) -> dict:
    """Of the units the suite never touched, which ones run at IMPORT?

    WHY THIS EXISTS. "untouched" reads as "nothing in the library ever runs this",
    and for a class that is false. A class body, including every enum member, executes
    once when its module is imported, and pytest-cov begins measuring only after the
    library has been imported, so those lines appear in no coverage record no matter
    how much of the library depends on them. 36 public enums read as never executed
    while the suite was using them throughout.

    BODY LINES ONLY. A ``def``/``class`` header and its decorators always execute at
    import, so intersecting a whole source span with import-time coverage could not
    have disagreed: the first attempt reported all 218 untouched units as executing at
    import, functions included, which is true only of their headers.

    The split is recorded for the reason a reader needs and is deliberately NOT fed to
    the release gate: executing a class body proves the class is constructible, which
    is not "its result was recorded", so it cannot satisfy B1. What it does is replace
    a false description with a true one.
    """
    scratch = None
    try:
        import subprocess  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(untouched, fh)
            scratch = fh.name
        proc = subprocess.run(
            [sys.executable, "-c", _IMPORT_SPLIT_CHILD, str(ROOT), scratch],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {
                "measured": False,
                "why": (
                    "the import-time measurement subprocess did not report; this is a "
                    "could-not-check, not a split of zero"
                ),
                "stderr_tail": (proc.stderr or "").strip()[-400:],
            }
        split = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001 - a failed measurement is not a result
        return {
            "measured": False,
            "why": f"the import-time measurement could not run: {type(exc).__name__}",
        }
    finally:
        if scratch:
            pathlib.Path(scratch).unlink(missing_ok=True)

    at_import = split["at_import"]
    nowhere = split["nowhere"]
    return {
        "measured": True,
        "_what_this_means": (
            "Of the units the suite never touched, these execute at IMPORT time (every "
            "class body does) versus nowhere at all. Measured in a subprocess so that "
            "coverage starts before vfairness is imported. Recorded to describe the "
            "untouched pile accurately; it is NOT counted towards B1, because running a "
            "class body is not recording a result."
        ),
        "n_executed_at_import": len(at_import),
        "n_executed_nowhere": len(nowhere),
        "executed_at_import": at_import,
        "executed_nowhere": nowhere,
    }


def measure(coverage_path: pathlib.Path, log: pathlib.Path | None = None) -> dict:
    cov = json.loads(coverage_path.read_text(encoding="utf-8"))
    by_file: dict[pathlib.Path, set] = {}
    for path, data in cov["files"].items():
        by_file[pathlib.Path(path).resolve()] = set(data.get("executed_lines") or ())

    surface = K._walk_public_surface()
    universe: list[str] = []
    for kind in ("functions", "classes", "methods"):
        universe += surface[kind]["ungraded"] + surface[kind]["graded_names"]

    executed: list[str] = []
    untouched: list[str] = []
    unlocatable: list[str] = []
    for qual in sorted(set(universe)):
        entry = K._OBJECTS.get(qual)
        obj = entry[1] if entry else None
        if obj is None:
            unlocatable.append(qual)
            continue
        try:
            src = pathlib.Path(inspect.getsourcefile(obj)).resolve()
            lines, start = inspect.getsourcelines(obj)
        except Exception:  # noqa: BLE001 - a unit we cannot locate is not evidence
            unlocatable.append(qual)
            continue
        span = set(range(start, start + len(lines)))
        (executed if (by_file.get(src, set()) & span) else untouched).append(qual)

    return {
        "_what_this_is": (
            "Which public code units the test suite EXECUTES, measured from a coverage "
            "run. Supports B1 (executed and recorded); it does NOT support CHECKED, "
            "because executing a unit is not judging whether it refuses honestly. "
            "Generated by scripts/suite_coverage.py."
        ),
        "measured_on": date.today().isoformat(),
        "suite": _suite_outcome(log),
        "surface_total": surface["total"]["total"],
        "n_executed": len(executed),
        "n_untouched": len(untouched),
        "n_unlocatable": len(unlocatable),
        # "untouched" alone reads as "nothing runs this", which is false for every
        # class in the pile. See _import_time_split.
        "untouched_split": _import_time_split(untouched),
        "executed": executed,
        "untouched": untouched,
        "unlocatable": unlocatable,
    }


def load() -> dict:
    """The recorded measurement, or an empty record. Never invents coverage."""
    if not OUT.exists():
        return {"executed": [], "n_executed": 0, "suite": {"recorded": False}}
    return json.loads(OUT.read_text(encoding="utf-8"))


def usable(record: dict | None = None) -> bool:
    """May this record be counted as evidence? A red suite may not."""
    record = record if record is not None else load()
    suite = record.get("suite") or {}
    if not suite.get("recorded"):
        return False
    return not suite.get("failed") and not suite.get("errors")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", help="coverage json from --cov-report=json:PATH")
    ap.add_argument("--log", help="the pytest output of that same run, for its pass/fail line")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        record = load()
        if not record.get("executed"):
            print("STALE: docs/suite-coverage.json missing or empty")
            return 1
        if not usable(record):
            print(f"UNUSABLE: the recorded suite run was not green: {record.get('suite')}")
            return 1
        surface = K._walk_public_surface()["total"]["total"]
        if record.get("surface_total") != surface:
            print(
                f"STALE: recorded against a surface of {record.get('surface_total')}, "
                f"now {surface}. Re-run the coverage measurement."
            )
            return 1
        print(
            f"suite coverage current: {record['n_executed']} of {surface} units executed, "
            f"{record['n_untouched']} untouched"
        )
        return 0

    if not args.coverage:
        ap.error("--coverage is required unless --check")
    record = measure(pathlib.Path(args.coverage), pathlib.Path(args.log) if args.log else None)
    pathlib.Path(args.out).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(
        f"{record['n_executed']} of {record['surface_total']} public code units are executed "
        f"by the test suite; {record['n_untouched']} are untouched"
    )
    print(f"suite outcome: {record['suite']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
