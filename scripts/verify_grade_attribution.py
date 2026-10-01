#!/usr/bin/env python3
"""Does each grading row's NAMED test file actually execute the unit it grades?

WHY THIS EXISTS. A grading row's ``test_file`` is the evidence behind a published
CHECKED badge. A row naming a test that never touches the unit is a claim with
nothing behind it, which is the defect this library is audited for, one level up and
in the surface built to prevent it.

It happened. Grading wave 4 recorded a ``test_file`` for all 95 units it graded, and
the generator assigned a default per module with a silent fallback for modules it did
not know. Eleven rows named a file that does not execute them: five had no wave-4
test at all (two base-class methods whose tests exercised overriding subclasses, two
dataclass serialisers that were inspected and never constructed, and a worker entry
point carried entirely by the fallback) and six named the wrong one of five files.
All eleven were live on the status page as CHECKED.

WHAT THIS CHECK CANNOT SEE, stated because a check that hides its blind spots is
worse than no check. Three kinds of unit are invisible to line coverage, and their
silence here is NOT evidence of anything:

  1. CLASS BODIES. A class body, including every enum member, executes once when its
     module is imported, which is before pytest-cov starts measuring. No assertion
     can make those lines appear in a coverage report. 36 public enums are in this
     category and their evidence is the assertion in the named file, not a line.
  2. LINES EXCLUDED BY CONFIGURATION. ``raise NotImplementedError`` is in this
     project's ``exclude_lines``, so an abstract base whose whole body is that raise
     reports as unexecuted however thoroughly it is tested.
  3. ``# pragma: no cover``. Entry points carry it, so a test that drives one still
     leaves no trace here.

Each is reported as its own category. Only "no line evidence and no blind spot to
explain it" is a finding.

Usage:
    python scripts/verify_grade_attribution.py            # report
    python scripts/verify_grade_attribution.py --check    # exit 1 on a real finding
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import inspect
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

PYTHON = str(ROOT / ".venv" / "bin" / "python3")
if not pathlib.Path(PYTHON).exists():  # pragma: no cover - venv layout differs in CI
    PYTHON = sys.executable


def _wave_files() -> list[pathlib.Path]:
    base = {"surface-grading-from-probe.json"}
    return sorted(p for p in (ROOT / "docs").glob("surface-grading-*.json") if p.name not in base)


def _named_files(record: dict) -> list[str]:
    """Every tests/ path mentioned in a row's test_file string."""
    # STRIP THE ``::test_name`` SUFFIX BEFORE asking whether it is a .py path. The
    # first version tested `token.endswith(".py")` on the raw token, so every
    # `tests/file.py::test_case` reference was silently dropped and only rows whose
    # test_file was a bare path were examined at all: 89 of them, with grading wave 4's
    # 95 rows, every one of which names a specific test, examined not at all. A
    # verifier that quietly checks a tenth of its input reports a clean bill.
    raw = (record.get("test_file") or "").strip()

    # AND THE WITHHELD PIN, which is the one that usually DISCRIMINATES. A row whose
    # deciding evidence is an internal harness cannot name it in ``test_file``, because
    # that string propagates into the published ledger and the export deletes the
    # harness, leaving a reader a dead link. Such rows name it in
    # ``test_file_withheld_from_export`` instead.
    #
    # THAT FIELD WAS READ BY NOTHING. Measured 2026-09-30: no script, no generator, no
    # document and no test consumed it, and two independent examiners in a row read only
    # ``test_file`` and reported the row as naming no evidence for the unit. A correct
    # record no consumer reads is the exact defect this repository is audited for,
    # committed in the evidence register itself. This verifier runs in the MONOREPO,
    # where the withheld file exists and can be executed, so it is the right consumer:
    # only the public export lacks it, and the export has its own rule.
    withheld = (record.get("test_file_withheld_from_export") or "").strip()
    if withheld:
        raw = f"{raw} {withheld}"

    # A ROW THAT DECLARES NO DIRECT TEST IS NOT CLAIMING ONE. Some rows begin "none
    # needed for the body..." or "none: already pinned by ..." and then MENTION other
    # test files in the explanation. Reading those mentions as the row's evidence made
    # this check accuse two honest rows of naming a test that does not execute them,
    # which is the same error it exists to catch, pointed the other way. A row that
    # opens with "none" is reported under its own heading and is never a finding here;
    # whether its reasoning holds is a judgement, not a coverage question.
    if (record.get("test_file") or "").strip().lower().startswith("none") and not withheld:
        return []

    text = raw.replace(";", " ").replace(",", " ")
    paths = set()
    for token in text.split():
        if not token.startswith("tests/"):
            continue
        head = token.split("::")[0].strip().rstrip(".")
        if head.endswith(".py"):
            paths.add(head)
    return sorted(paths)


def _body_lines(obj) -> tuple[pathlib.Path, set]:
    src = pathlib.Path(inspect.getsourcefile(obj)).resolve()
    lines, start = inspect.getsourcelines(obj)
    header = start
    for offset, text in enumerate(lines):
        if text.strip().startswith(("def ", "class ", "async def ")):
            header = start + offset
            break
    for offset in range(header - start, len(lines)):
        stripped = lines[offset].rstrip()
        if stripped.endswith(":") and not stripped.strip().startswith("#"):
            return src, set(range(start + offset + 1, start + len(lines)))
    return src, set(range(header + 1, start + len(lines)))


def measure() -> dict:
    rows: dict[str, dict] = {}
    for path in _wave_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        for qual, record in (data.get("items") or {}).items():
            if _named_files(record):
                rows[qual] = record

    files = sorted({f for r in rows.values() for f in _named_files(r)})
    executed: dict[str, dict] = {}
    excluded: dict[str, dict] = {}
    for test_file in files:
        if not (ROOT / test_file).exists():
            executed[test_file] = {}
            excluded[test_file] = {}
            continue
        out = pathlib.Path(tempfile.mkdtemp()) / "cov.json"
        subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:warnings",
                test_file,
                "--cov=vfairness",
                f"--cov-report=json:{out}",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=1800,
        )
        if not out.exists():
            executed[test_file] = {}
            excluded[test_file] = {}
            continue
        data = json.loads(out.read_text())
        executed[test_file] = {
            str(pathlib.Path(p).resolve()): set(d.get("executed_lines") or ())
            for p, d in data["files"].items()
        }
        excluded[test_file] = {
            str(pathlib.Path(p).resolve()): set(d.get("excluded_lines") or ())
            for p, d in data["files"].items()
        }

    import library_kpis as K  # noqa: PLC0415 - after sys.path is set

    K._walk_public_surface()
    ledger = json.loads((ROOT / "docs" / "capability-status.json").read_text())["units"]

    verdicts: dict[str, list] = collections.defaultdict(list)
    for qual, record in sorted(rows.items()):
        named = _named_files(record)
        missing_files = [f for f in named if not (ROOT / f).exists()]
        if missing_files:
            verdicts["the named test file does not exist"].append((qual, ",".join(missing_files)))
            continue
        entry = K._OBJECTS.get(qual)
        obj = entry[1] if entry else None
        if obj is None:
            verdicts["unit not on the current surface"].append((qual, ""))
            continue
        try:
            src, body = _body_lines(obj)
        except Exception:  # noqa: BLE001 - a unit we cannot locate is not a finding
            verdicts["source could not be located"].append((qual, ""))
            continue

        if any(executed[f].get(str(src), set()) & body for f in named):
            verdicts["confirmed by line coverage"].append((qual, ""))
            continue

        kind = (ledger.get(qual) or {}).get("kind")
        if kind == "class":
            verdicts["blind: class body runs at import"].append((qual, ""))
        elif any(excluded[f].get(str(src), set()) & body for f in named):
            verdicts["blind: body excluded from coverage"].append((qual, ""))
        else:
            elsewhere = [f for f in files if executed[f].get(str(src), set()) & body]
            if elsewhere:
                verdicts["FINDING: names the wrong file"].append((qual, ",".join(elsewhere)))
            else:
                verdicts["FINDING: no test executes it"].append((qual, ""))

    return {
        "n_rows": len(rows),
        "files": files,
        "verdicts": {k: v for k, v in verdicts.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    result = measure()
    print(f"{result['n_rows']} graded rows name {len(result['files'])} test files\n")
    for label, items in sorted(result["verdicts"].items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(items):4d}  {label}")
    findings = [
        (label, items)
        for label, items in result["verdicts"].items()
        if label.startswith("FINDING") or label.startswith("the named test file does not exist")
    ]
    if findings:
        print("\nFINDINGS, each one a published badge with nothing behind it:")
        for label, items in findings:
            for qual, extra in items:
                print(f"  [{label}] {qual}" + (f" -> actually {extra}" if extra else ""))
    # PUBLISH THE RESULT, do not leave it in a terminal. This run takes forty
    # minutes, so nothing can call it on every build, and the page that describes
    # it therefore carried its numbers as prose: "across 415 graded rows it
    # confirms 348 and reports 40 as invisible", typed once and never revisited.
    # Measured 2026-09-29 the real figures were 721, 677 and 39, and the sentence
    # "it reports zero rows whose named test does not reach them" was false, with
    # three rows naming a test of the adjacent thing. A snapshot with its own date
    # beside it is honest; a number nobody can trace to a run is not.
    out = ROOT / "docs" / "grade-attribution.json"
    snapshot = {
        "_what_this_is": (
            "The last run of scripts/verify_grade_attribution.py. It runs each test "
            "file a grading row names and looks for the graded unit's own lines in "
            "the coverage report, because a row naming a test that never executes "
            "the unit is a published badge with nothing behind it. It takes about "
            "forty minutes, so this is a dated SNAPSHOT and the date travels with "
            "the numbers."
        ),
        "measured_on": _dt.date.today().isoformat(),
        "rows": result["n_rows"],
        "test_files": len(result["files"]),
        "verdicts": {label: len(items) for label, items in result["verdicts"].items()},
        "findings": [
            {"unit": qual, "label": label, "actually": extra}
            for label, items in findings
            for qual, extra in items
        ],
    }
    if not args.check:
        out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {out.relative_to(ROOT)}")
    if args.check:
        return 1 if findings else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
