#!/usr/bin/env python3
"""The published KPIs, measured rather than typed.

Every figure the public site shows about this library is computed here. The rule
that produced this file: a hardcoded figure rots, and a rotted figure on a page
that claims to summarise a measurement is worse than no figure, because it reads
as verified. This repository has already shipped a stale chunk count, a test
count stale by 3,700, and a tile row reading "95 + 116" beside a "215" because
the first two counted NAMES and the third counted dispatch keys.

THREE STATES apply to the KPIs themselves. A figure this script cannot measure
is emitted as ``None`` with a reason, never as a stale number and never omitted
silently, so the page can say "not measured" where a reader sees it.

Two figures come from CI rather than from here, because they need a full test
run against the full dependency matrix. They carry their own date and commit in
docs/site/data/coverage-measurement.json and are reported "as of" that date.

Usage:
    python scripts/library_kpis.py            # print the KPI block as JSON
    python scripts/library_kpis.py --check    # exit 1 if the site data is stale
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
EVIDENCE = ROOT / "docs" / "beta-go-live-census-2026-09-11.json"
CI_MEASURED = ROOT / "docs" / "site" / "data" / "coverage-measurement.json"
GRADING = ROOT / "docs" / "surface-grading-2026-09-18.json"
GRADING_FROM_PROBE = ROOT / "docs" / "surface-grading-from-probe.json"
# Later dated waves. A wave supersedes an EARLIER wave's grade for the units it
# names, and only those. Without this the readiness gate could never report a
# defect as closed: it read the 2026-09-18 file alone, so on 2026-09-25 it printed
# 95 open defects beside a status ledger that had published 93, two surfaces
# disagreeing about the same measurement. A later PROVEN row is honoured only when
# it carries both a pin and a sabotage, the same bar scripts/capability_status.py
# applies; a row without them is reported, never silently used.
# DISCOVERED BY NAME, and the pattern must not be narrower than the naming scheme.
# It was `surface-grading-2026-09-2[0-9].json`, which cannot match a second wave on
# one day: `surface-grading-2026-09-25b.json` was written, both registries ignored it
# silently, and the ledger reprinted the previous numbers as though the wave did not
# exist. A registry that misses a file is worse than one that errors, because the
# output still looks like an answer. This matches every dated wave and excludes only
# the base wave, which is read separately as GRADING, and the probe-derived file.
_BASE_WAVES = {GRADING.name, GRADING_FROM_PROBE.name}
LATER_WAVES = sorted(
    p for p in (ROOT / "docs").glob("surface-grading-*.json") if p.name not in _BASE_WAVES
)

STAMP = "Beta Go-Live proof status"

GATE_CHECKS = ROOT / "docs" / "gate-checks.json"


def gate_measured_on() -> str:
    """The ONE date every generated page prints as "measured" or "generated".

    It is the date the two beta-gate checks were last EXECUTED, read from
    docs/gate-checks.json, and never the clock of whoever runs a --check.

    WHY. Every generator stamped date.today() into what it publishes, and every
    --check compared the whole published block, the stamp included. So a tree
    regenerated on 2026-10-01 read as STALE on 2026-10-02 with not one figure
    changed: the public CI job "Published documents match the repo" reported
    library-stats, the register and three markdown pages stale for no reason
    but the calendar. A check that fails on a timer teaches its reader to ignore
    it, which is worse than no check. Every figure is still compared; only the
    date stops being a function of the wall clock.

    It is also the TRUER date. ./scripts/refresh-docs.sh executes the gate checks
    first and writes that day into gate-checks.json, so this is the day the
    published figures were measured. A --check run later does not re-measure the
    gate, it recomputes the figures and proves they still match, which is not a
    new measurement date.

    Falls back to today only when the file is absent or unreadable, which is a
    tree where the gate checks never ran and release_gate.py already reports
    them as could not check.
    """
    try:
        with GATE_CHECKS.open(encoding="utf-8") as fh:
            stamped = json.load(fh).get("date")
        return date.fromisoformat(stamped).isoformat()
    except (OSError, ValueError, TypeError, AttributeError):
        return date.today().isoformat()


# THE EXTRAS THAT CHANGE THE SURFACE, and what each one is probed by. Each probe
# names a module the extra in pyproject.toml actually installs. `dashboard` was
# probed by `streamlit`, which no extra declares and no vfairness module imports,
# so the published record said whether a developer happened to have streamlit on
# the machine: true on the workstation that generated the ledger, false on every
# CI runner that installs the declared extras, and the library-stats check could
# never pass there. The dashboard extra declares plotly.
SURFACE_EXTRAS = (
    ("mcp", "mcp"),
    ("viz", "matplotlib"),
    ("xai", "shap"),
    ("training", "torch"),
    ("causal", "dowhy"),
    ("dashboard", "plotly"),
)


def missing_extras(published: dict | None) -> list[str]:
    """Extras the PUBLISHED measurement had importable and this environment lacks.

    The surface walk depends on them: without `mcp` the twelve functions of
    vfairness.mcp.server cannot be imported, and without torch the eight
    adversarial-training units in in_processing.loss_functions.adversarial are not
    defined while three placeholder stubs are. Measured 2026-10-02 on the public
    tree with only the dev, rendering, viz, monitoring and causal extras: 1,563
    units (1,560 checked, 3 not checked) against the 1,580 the full install
    measures. That is not a stale ledger, it is a different package being
    measured, and reporting it as STALE invites a regeneration that would publish
    a partial install's count as the library's. A check that cannot see what was
    published is a could-not-check, never a pass and never a staleness verdict.
    """
    import importlib.util as _ilu

    published = published or {}
    return sorted(
        name for name, mod in SURFACE_EXTRAS if published.get(name) and _ilu.find_spec(mod) is None
    )


# qualified name -> (short name, object), filled by _walk_public_surface.
_OBJECTS: dict[str, tuple[str, object]] = {}


def _package():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import vfairness
    from vfairness._proof_status import CAPABILITY_PROOF, COUNTS_AS_OF, MEASURED_ON

    return vfairness, CAPABILITY_PROOF, MEASURED_ON, COUNTS_AS_OF


def _graded_names(proof) -> set[str]:
    known = set(proof)
    for record in proof.values():
        known.update(record.get("shared_with") or ())
    return known


def _is_graded(name: str, obj: object, known: set[str]) -> bool:
    return name in known or STAMP in (inspect.getdoc(obj) or "")


def _walk_public_surface() -> dict:
    """Every public callable in the package, under ONE definition.

    THE DEFINITION, fixed on 2026-09-17 and not to be quietly changed again.
    Three counts were in circulation at once, and all three were called "the
    library": 215 census rows, 823 top-level callables, and 211 distinct names.
    They were three different denominators, so every comparison between them
    read as arithmetic that fails. From here there is one denominator:

      FUNCTION   a module-level def in any non-private vfairness module
      CLASS      a class defined in any non-private vfairness module
      METHOD     a non-underscore method, counted at the class that DEFINES it
                 so an inherited method is not counted once per subclass
      CAPABILITY one dispatch key in the capability registry. A capability is
                 not a fourth kind of object: it NAMES one or more of the above.
                 215 rows resolve to 222 objects.

    Private modules (a path segment starting with underscore) and the test tree
    are excluded, because they are not surface a user can call.

    A module that cannot be imported is reported in ``import_failures``, never
    dropped. An uncounted module would understate the surface, which is the same
    defect as an ungraded capability counted as fine.
    """
    import importlib
    import pkgutil
    import warnings

    vfairness, proof, _on, _asof = _package()
    known = _graded_names(proof)

    modules = [vfairness]
    failures: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(vfairness.__path__, "vfairness."):
            parts = info.name.split(".")
            if any(seg.startswith("_") for seg in parts[1:]) or "tests" in parts:
                continue
            try:
                modules.append(importlib.import_module(info.name))
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                # THREE STATES for an import, not two. "This module is broken"
                # and "this module needs an optional extra nobody installed"
                # are different facts, and reporting the second as the first
                # calls a working feature a fault. vfairness.mcp.server was
                # published as "a module that could not be imported" for two
                # days; it refuses cleanly with an install hint and works when
                # the extra is present, which was proved by driving it over the
                # protocol end to end.
                text = str(exc)
                optional = isinstance(exc, ImportError) and (
                    "optional" in text.lower() or "pip install" in text.lower()
                )
                failures.append(
                    {
                        "module": info.name,
                        "error": type(exc).__name__,
                        "kind": "optional extra not installed" if optional else "import failed",
                        "detail": text.splitlines()[0][:200] if text else "",
                    }
                )

    funcs: dict[str, tuple[str, object]] = {}
    classes: dict[str, tuple[str, object]] = {}
    methods: dict[str, tuple[str, object]] = {}

    for mod in modules:
        for name, obj in vars(mod).items():
            if name.startswith("_"):
                continue
            origin = getattr(obj, "__module__", "") or ""
            if not origin.startswith("vfairness"):
                continue
            if inspect.isfunction(obj):
                funcs.setdefault(f"{origin}.{obj.__qualname__}", (name, obj))
            elif inspect.isclass(obj):
                key = f"{origin}.{obj.__qualname__}"
                if key in classes:
                    continue
                classes[key] = (name, obj)
                for mname, m in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if mname.startswith("_"):
                        continue
                    if m.__qualname__.split(".")[0] != obj.__name__:
                        continue
                    methods.setdefault(f"{m.__module__}.{m.__qualname__}", (mname, m))

    # Keep the resolved objects. Re-deriving them from a dotted string cannot
    # work for a method: "pkg.mod.Class.method".rpartition(".") yields a module
    # path that is really a class, so every method came back unresolvable and
    # the triage reported 683 unknowns that it had just been holding.
    _OBJECTS.clear()
    for d in (funcs, classes, methods):
        for qual, (name, obj) in d.items():
            _OBJECTS[qual] = (name, obj)

    def bucket(d: dict) -> dict:
        # BOTH lists are published, not just the counts. A panel that shows a
        # number nobody can expand is a claim; a list is a thing a reader can
        # check. The owner's requirement, verbatim: "explain or name every
        # single item in there so that we can cross-reference that."
        graded, ungraded = [], []
        for qual, (name, obj) in sorted(d.items()):
            (graded if _is_graded(name, obj, known) else ungraded).append(qual)
        return {
            "total": len(d),
            "graded": len(graded),
            "graded_names": graded,
            "ungraded": ungraded,
        }

    # WHICH EXTRAS WERE INSTALLED. The measured surface depends on it: with the
    # optional mcp extra absent the package presents 1,546 code units across
    # 204 modules, and with it present 1,558 across 205. A surface count with no
    # record of the environment that produced it cannot be reproduced or
    # compared, and every figure downstream inherits that.
    import importlib.util as _ilu

    extras = {name: _ilu.find_spec(mod) is not None for name, mod in SURFACE_EXTRAS}

    f, c, m = bucket(funcs), bucket(classes), bucket(methods)
    total = f["total"] + c["total"] + m["total"]
    graded = f["graded"] + c["graded"] + m["graded"]
    return {
        "functions": f,
        "classes": c,
        "methods": m,
        "total": {"total": total, "graded": graded, "ungraded": total - graded},
        "capabilities": {"rows": len(proof), "objects": graded},
        "modules_scanned": len(modules),
        "import_failures": failures,
        "optional_extras_present": extras,
    }


_RETURNS_A_MEASUREMENT = re.compile(
    r"\b(float|int|bool|ndarray|Series|DataFrame|Result|Report|Score|Metric|"
    r"Disparity|Dict\[str,\s*(float|int|bool|Any)\])"
)
_RENDERS_OR_CONVERTS = re.compile(
    r"^(plot_|save_|print_|set_|get_palette|preview_|render_|to_|from_|as_|build_|make_fig)"
)


def work_list(surface_data: dict) -> dict:
    """What grading the rest would actually involve, sorted by what it returns.

    A CLASS is excluded here: a class does not itself return a measurement, its
    methods do. So the work list is functions plus methods.

    This is a TRIAGE, not a verdict. "Renders or converts" is a candidate for
    NOT A MEASUREMENT, it is not that grade yet. Nothing may take a grade from
    this heuristic: a human or an agent has to run it. Letting the heuristic
    stand as the answer would rebuild the exact defect this library was audited
    for, one level up.
    """
    vfairness, proof, _on, _asof = _package()
    wanted = set(surface_data["functions"]["ungraded"]) | set(surface_data["methods"]["ungraded"])
    seen: dict[str, str] = {}

    if not _OBJECTS:
        _walk_public_surface()
    for qual in sorted(wanted):
        entry = _OBJECTS.get(qual)
        if entry is None:
            seen[qual] = "could not resolve, needs a human"
            continue
        short, obj = entry
        try:
            ann = str(inspect.signature(obj).return_annotation)
        except Exception:  # noqa: BLE001
            ann = ""
        if _RENDERS_OR_CONVERTS.match(short):
            seen[qual] = "renders, prints or converts"
        elif ann in ("None", "<class 'NoneType'>", "<class 'str'>", "str"):
            seen[qual] = "returns nothing or text"
        elif _RETURNS_A_MEASUREMENT.search(ann):
            seen[qual] = "returns a number or a result"
        elif ann in ("", "<Signature empty>") or "inspect._empty" in ann:
            seen[qual] = "no return annotation, needs a human"
        else:
            seen[qual] = "unclassified, needs a human"

    counts: dict[str, int] = {}
    for v in seen.values():
        counts[v] = counts.get(v, 0) + 1
    return {"total": len(seen), "by_return": counts, "items": seen}


def size() -> dict:
    """How big the library is. Not a quality claim, a scale claim.

    Lines of code is deliberately kept here and deliberately kept OFF the
    hardening page. On a page explaining what the library is, scale is context a
    reader needs. Beside a coverage figure it is noise at best and an implied
    boast at worst.
    """
    import pkgutil

    vfairness, _p, _on, _asof = _package()
    manifest = ROOT / "vfairness-manifest.json"
    stats = {}
    if manifest.exists():
        stats = (json.loads(manifest.read_text(encoding="utf-8")) or {}).get("stats", {})
    sub = sorted(
        m.name
        for m in pkgutil.iter_modules(vfairness.__path__)
        if m.ispkg and not m.name.startswith("_") and m.name != "tests"
    )
    return {
        "lines_of_code": stats.get("lines_of_code"),
        "svg_templates": stats.get("svg_templates"),
        "sub_packages": {"count": len(sub), "names": sub},
    }


def unpublished_tests() -> list:
    """Test files and folders the public export drops, read from its exclude list.

    COUNT WHAT SHIPS (2026-10-01). The published test figures used to count every
    test in the development repository, including a few internal-only files the
    export removes, so the public repository measured 16,275 tests where the page
    said 16,452 and its own freshness check failed. Counting only what ships gives
    one figure that both copies reproduce: here the excluded files are read from
    the export script's EXCLUDES and left out; in the public repository that
    script is absent and so are the files.
    """
    script = ROOT.parent / "scripts" / "export-vfairness-to-public.sh"
    if not script.exists():
        return []
    text = script.read_text(encoding="utf-8")
    m = re.search(r"^EXCLUDES=\((.*?)^\)", text, re.S | re.M)
    if not m:
        raise RuntimeError("could not read EXCLUDES from the export script")
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for entry in re.findall(r'"([^"]+)"', line):
            if entry == "tests" or entry.startswith("tests/"):
                path = ROOT / entry
                if path.exists():
                    out.append(path)
    return out


def test_count() -> tuple[int | None, str]:
    """Tests collected in the PUBLISHED tree. None plus a reason when it cannot run.

    One definition both repositories compute identically (2026-10-01). Collecting
    in the development repository counted 16,343 even with the unpublished test
    files ignored, while the public copy collects 16,275: some tests are
    parametrised over documents and scripts the export does not ship. So here the
    export tree is built with the export's own code (from the committed HEAD) into
    a temporary folder and collected there; in the public repository, where the
    export script does not exist, the tree IS the published tree and is collected
    in place.
    """
    import shutil
    import tempfile

    py = ROOT / ".venv" / "bin" / "python"
    exe = str(py) if py.exists() else sys.executable
    export = ROOT.parent / "scripts" / "export-vfairness-to-public.sh"
    where, cleanup = ROOT, None
    if export.exists():
        cleanup = tempfile.mkdtemp(prefix="vfairness-published-tree-")
        where = Path(cleanup) / "tree"
        try:
            built = subprocess.run(
                [str(export), "--emit-tree", str(where)],
                cwd=str(ROOT.parent),
                capture_output=True,
                text=True,
                timeout=900,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            shutil.rmtree(cleanup, ignore_errors=True)
            return None, f"could not build the published tree: {exc}"
        if built.returncode != 0 or not (where / "tests").is_dir():
            shutil.rmtree(cleanup, ignore_errors=True)
            return None, f"could not build the published tree (exit {built.returncode})"
    try:
        r = subprocess.run(
            [
                exe,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                f"pythonpath={where / 'src'}",
            ],
            cwd=str(where),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return None, f"collection did not run: {exc}"
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
    # The summary line is wrapped in "=" padding: "==== 9536 tests collected ====".
    # Splitting on whitespace hands back the padding, so match the number.
    m = re.findall(r"(\d+)\s+tests?\s+collected", r.stdout)
    if m:
        return int(m[-1]), "collected in the published tree"
    return None, "collection produced no count"


def defects() -> dict:
    _v, proof, _on, _asof = _package()
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    found = (ev.get("totals") or {}).get("reproduced_public")
    # Deduplicate: one finding can belong to two dispatch keys on one object.
    open_now = 0
    for key, r in proof.items():
        group = sorted([key, *(r.get("shared_with") or [])])
        if group[0] == key:
            open_now += r["open_defects"]
    return {
        "found": found,
        "open": open_now,
        "fixed": (found - open_now) if isinstance(found, int) else None,
    }


def ci_measured() -> dict:
    """Figures that need a full CI run. Dated, because the fix for staleness is
    a date and not a refresh."""
    if not CI_MEASURED.exists():
        return {"coverage_pct": None, "reason": "no CI measurement recorded"}
    return json.loads(CI_MEASURED.read_text(encoding="utf-8"))


def graded_items() -> dict:
    """Every graded code unit, with later waves applied. THE one source.

    Two functions used to merge these separately and they disagreed: on
    2026-09-25 the readiness gate's B2 split read the 2026-09-18 file alone and
    printed 86 open while G3, reading the merged view, printed 93. A readiness
    verdict assembled from two different countings of the same evidence is the
    defect this gate exists to catch, committed inside the gate.
    """
    if not GRADING.exists():
        return {}
    items = dict(json.loads(GRADING.read_text(encoding="utf-8")).get("items", {}))
    for path in LATER_WAVES:
        for name, rec in (json.loads(path.read_text(encoding="utf-8")).get("items") or {}).items():
            if rec.get("grade") == "PROVEN" and not (rec.get("test_file") and rec.get("sabotage")):
                continue
            items[name] = rec
    return items


def waves() -> list[dict]:
    """One row per grading wave, so the reader can see the SHAPE of the work.

    The totals elsewhere on the page say how much was graded. They cannot say
    how it was graded, and the two differ in a way that matters to whether the
    grades are worth anything: wave 1 and wave 2 were audited as they went and
    are 100 percent independently checked, while the BGL3 wave graded 342 units
    in one day and none of them has been argued with yet. A reader looking only
    at the merged total sees "872 graded" and cannot tell those apart.

    So each wave is published with its OWN audit and overturn counts. Every
    figure is read out of the wave file, and a wave whose file is absent is
    omitted rather than invented.

    The counts here are per FILE and are not merged, which is why they do not
    sum to the merged totals: a unit graded in wave 1 and re-graded by BGL3
    appears in both rows. That is stated on the page rather than smoothed over,
    because the alternative is a table of numbers that quietly disagrees with
    the one above it.
    """
    out = []
    for path in [GRADING, *LATER_WAVES, GRADING_FROM_PROBE]:
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        items = doc.get("items") or {}
        withdrawn = doc.get("_withdrawn") or {}
        if not items and withdrawn:
            # A WAVE WHOSE GRADES WERE ALL WITHDRAWN still belongs in the table.
            # The probe-derived wave published 39 SEMI-PROVEN grades until
            # 2026-09-27 and now publishes none, because every one of them claimed
            # "measured on healthy input" over a healthy run the probe had recorded
            # as a refusal. Skipping the file makes those 39 vanish from the only
            # surface that shows a reader where each grade came from, and a figure
            # that silently disappears is worse than one that says it was wrong.
            out.append(
                {
                    "file": path.name,
                    "label": "probe (withdrawn)",
                    "measured_on": doc.get("measured_on"),
                    "rows": 0,
                    "by_grade": {"WITHDRAWN": len(withdrawn)},
                    "audited": 0,
                    "overturned": 0,
                    "found_fabricating": 0,
                    "fixed": 0,
                    "sabotaged": 0,
                    "retired": len(withdrawn),
                }
            )
            continue
        if not items:
            continue
        labels = sorted({r.get("wave") for r in items.values() if r.get("wave")})
        by_grade: dict[str, int] = {}
        for r in items.values():
            by_grade[r["grade"]] = by_grade.get(r["grade"], 0) + 1
        out.append(
            {
                "file": path.name,
                "label": ", ".join(labels) or "probe",
                "measured_on": doc.get("measured_on"),
                "rows": len(items),
                "by_grade": by_grade,
                "audited": sum(1 for r in items.values() if r.get("audited")),
                "overturned": sum(1 for r in items.values() if r.get("audit_overturned")),
                "found_fabricating": sum(1 for r in items.values() if r.get("fabricated")),
                "fixed": sum(1 for r in items.values() if r.get("fixed")),
                "sabotaged": sum(1 for r in items.values() if r.get("sabotage")),
                "retired": len(doc.get("_retired") or {}),
            }
        )
    return out


def grading() -> dict:
    """The wider-surface grading waves, beyond the 215 capability census.

    Reported separately from the census on purpose. The census graded
    CAPABILITIES, the things that produce a fairness number or a verdict, under
    a four-pass method. These waves grade the rest of the callable surface under
    a five-state scheme, and the fifth state, NOT A MEASUREMENT, does not exist
    in the census at all. Adding the two totals together would invent a number
    that neither method produced.

    An item whose grade no auditor confirmed is counted in `unaudited`, never in
    the grade totals as though it were settled. In wave 2 an entire audit pass
    was lost to a session limit, and 26 of the 129 grades it later produced were
    overturned: a claim nobody checked is not a measurement.

    TWO DIFFERENT QUESTIONS ABOUT OVERTURNS, and reporting only one of them
    understated the method's fallibility by a factor of three.

    ``audit_overturned`` counts rows whose CURRENT record carries the flag. That
    is the right answer to "is the grade published today a second opinion?", and
    it is the wrong answer to "how often has this method disagreed with itself?",
    because a later wave REPLACES the row and the flag goes with it. Measured on
    2026-09-27: the 2026-09-18 wave recorded 124 overturns, later waves
    superseded 89 of those rows, and the merged view reported 35. A reader
    judging whether these grades are worth anything needs the 124.

    ``audit_overturned_ever`` is therefore taken as a union over every wave file
    including the superseded ones, and it can only grow. ``audited`` is NOT
    treated that way and must not be: an audit of a grade that a later wave
    replaced does not carry over to the replacement, so a superseded row is
    correctly unaudited again.
    """
    if not GRADING.exists():
        return {"available": False, "reason": "no grading evidence recorded"}
    d = json.loads(GRADING.read_text(encoding="utf-8"))
    base = d.get("items", {})
    items = graded_items()
    superseded: dict[str, str] = {}
    unsupported_later: list[str] = []
    for path in LATER_WAVES:
        later = json.loads(path.read_text(encoding="utf-8"))
        for name, rec in (later.get("items") or {}).items():
            if rec.get("grade") == "PROVEN" and not (rec.get("test_file") and rec.get("sabotage")):
                unsupported_later.append(f"{name} ({path.name})")
            elif name in base:
                superseded[name] = (
                    f"{base[name]['grade']} -> {rec['grade']} ({later['measured_on']})"
                )
    # Gathered from the FILES, not from the merged view, so a row a later wave
    # replaced still counts toward the history of the method arguing with itself.
    overturned_ever: set[str] = set()
    audited_ever: set[str] = set()
    for path in (GRADING, GRADING_FROM_PROBE, *LATER_WAVES):
        if not path.exists():
            continue
        for name, rec in (json.loads(path.read_text(encoding="utf-8")).get("items") or {}).items():
            if rec.get("audit_overturned"):
                overturned_ever.add(name)
            if rec.get("audited"):
                audited_ever.add(name)

    counts: dict[str, int] = {}
    unaudited = overturned = fabricating = fixed = 0
    for r in items.values():
        counts[r["grade"]] = counts.get(r["grade"], 0) + 1
        unaudited += not r.get("audited")
        overturned += bool(r.get("audit_overturned"))
        fabricating += bool(r.get("fabricated"))
        fixed += bool(r.get("fixed"))
    # Probe-assigned grades are counted, and kept SEPARATE, because they were
    # not produced the same way. An agent ran the code, judged it, wrote a test
    # and had a second agent try to refute it. The probe ran the code and
    # nothing else. Both are grades; only one of them was argued with.
    probe_graded: dict = {}
    if GRADING_FROM_PROBE.exists():
        probe_graded = (json.loads(GRADING_FROM_PROBE.read_text(encoding="utf-8")) or {}).get(
            "items", {}
        )
    for name, rec in probe_graded.items():
        counts[rec["grade"]] = counts.get(rec["grade"], 0) + 1
        unaudited += 1

    return {
        "available": True,
        # The NEWEST wave that contributed, not the base one. The figures below
        # merge every wave, so dating them by the base file understates them by
        # however long the campaign has been running.
        "measured_on": max(
            [d.get("measured_on") or ""]
            + [
                (json.loads(p.read_text(encoding="utf-8")) or {}).get("measured_on") or ""
                for p in LATER_WAVES
                if p.exists()
            ]
        )
        or d.get("measured_on"),
        "later_waves": [p.name for p in LATER_WAVES],
        "superseded": superseded,
        "unsupported_later_claims": unsupported_later,
        "items": len(items) + len(probe_graded),
        "by_agent": len(items),
        "by_probe_alone": len(probe_graded),
        "by_grade": counts,
        "audited": len(items) - unaudited,
        "unaudited": unaudited,
        "audit_overturned": overturned,
        "audit_overturned_ever": len(overturned_ever),
        # The DENOMINATOR for the overturn rate, and it has to be a union for the
        # same reason the numerator does. audited + superseded is NOT it: that
        # leaves out every row that was audited, upheld, and later re-graded, 43
        # of them on 2026-09-27, so the rate it produces is too high.
        "audited_ever": len(audited_ever),
        "audit_overturned_superseded": len(overturned_ever) - overturned,
        "found_fabricating": fabricating,
        "fixed": fixed,
        "states": d.get("states", {}),
    }


def _second_round() -> dict:
    """The second-round audit register, published so the page can render it LIVE.

    The dashboard at the top of the hardening page has to be right whenever someone
    opens it, not right on the day it was written, so its figures come from this file
    at runtime like every other number on that page. Generated by
    scripts/bgl6_register.py from the audit files themselves; an absent register
    publishes available False rather than zeros, because "no findings" and "nobody
    counted" are different facts and only one of them is good news.
    """
    path = ROOT / "docs" / "bgl6-audit-register.json"
    if not path.exists():
        return {"available": False, "recorded": None, "closed": None, "open": None}
    doc = json.loads(path.read_text(encoding="utf-8"))
    totals = doc.get("totals") or {}
    return {
        "available": True,
        "recorded": totals.get("recorded"),
        "closed": totals.get("closed"),
        "open": totals.get("open"),
        "generated_on": doc.get("generated_on"),
        "batches": {
            tag: {
                "subject": b.get("subject", ""),
                "recorded": b.get("recorded"),
                "closed": b.get("closed"),
                "open": b.get("open"),
            }
            for tag, b in (doc.get("batches") or {}).items()
        },
    }


def _gate() -> dict:
    """The published readiness verdict. Imported late to avoid a cycle, since
    release_gate.py builds its criteria from this module."""
    try:
        import release_gate
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "COULD NOT MEASURE", "reason": str(exc)[:160]}
    crit = release_gate._criteria()
    beta = release_gate._beta_criteria()
    return {
        # GENERATED, because the page used to carry a hand-typed "Read 2026-09-18"
        # beside numbers that refresh themselves on every page load. The figures
        # were current and the date was eight days old, which is the worse way round:
        # it invites a reader to discount a measurement that is in fact live.
        # From gate-checks.json, not the clock: see gate_measured_on().
        "measured_on": gate_measured_on(),
        "verdict": "READY" if all(c["passes"] for c in crit) else "NOT READY",
        "criteria": crit,
        "beta_verdict": "BETA READY" if all(c["passes"] for c in beta) else "NOT BETA READY",
        "beta_criteria": beta,
    }


# The vocabulary that decides whether a test NAME is about honest refusal. It is
# published here, and the count is generated, because the hardening page briefly
# carried "854 and 1,499 of 6,735" from two vocabularies that existed nowhere:
# no artifact in the repository produced either number. A figure a reader cannot
# reproduce is exactly the defect this library is audited for, printed on the
# page that describes the audit.
REFUSAL_VOCABULARY = (
    "fabricat",
    "verdict",
    "could_not",
    "not_assess",
    "unassess",
    "no_silent",
    "swallow",
    "honest",
    "three_state",
    "unscored",
    "refus",
    "nan",
    "neutral",
    "measured",
)
REFUSAL_FILE_VOCABULARY = (
    "fabricat",
    "verdict",
    "honest",
    "readiness",
    "bgl",
    "surface_grade",
    "cramers",
    "swallow",
    "no_silent",
)


def test_composition() -> dict:
    """What the suite is made of, by file and by test name.

    A count of tests says how much checking happens; it says nothing about WHAT
    is checked. This splits the suite so the honesty audit can be seen as one
    strand among several rather than standing in for the whole programme.

    The split is by NAME and it is a heuristic, which is why the vocabulary is
    published beside the number instead of being buried in the script.
    """
    import re as _re

    dropped = unpublished_tests()
    files = sorted(
        f
        for f in (ROOT / "tests").rglob("test_*.py")
        if not any(f == d or d in f.parents for d in dropped)
    )
    names: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        names += _re.findall(r"^\s*(?:async\s+)?def (test_\w+)", text, _re.M)
    by_name = sum(1 for n in names if any(v in n.lower() for v in REFUSAL_VOCABULARY))
    by_file = sum(1 for f in files if any(v in f.name.lower() for v in REFUSAL_FILE_VOCABULARY))
    return {
        "test_files": len(files),
        "test_function_definitions": len(names),
        "refusal_named_tests": by_name,
        "refusal_named_files": by_file,
        "vocabulary": list(REFUSAL_VOCABULARY),
        "file_vocabulary": list(REFUSAL_FILE_VOCABULARY),
        "note": (
            "Split by name, a heuristic, so the vocabulary is published with the "
            "figure. It says which tests are ABOUT honest refusal, not which are "
            "good."
        ),
    }


#: The assessment path proper: the three areas a user is buying when they buy a
#: fairness assessment. A subset of core_areas, narrower on purpose, and named here
#: rather than in the page because a boundary drawn in prose cannot be recounted.
ASSESSMENT_PATH_AREAS = frozenset({"Fairness metrics", "Bias detection", "Explainability (XAI)"})


def attribution() -> dict:
    """The last run of scripts/verify_grade_attribution.py, as it published itself.

    A DATED SNAPSHOT AND NOT A LIVE FIGURE, because that run takes about forty
    minutes and nothing can call it per build. The date travels with the numbers
    for exactly that reason. Before this existed the page carried them as prose:
    "across 415 graded rows it confirms 348 and reports 40 as invisible", typed
    once and never revisited, and measured 2026-09-29 the real figures were 721,
    677 and 39, with the sentence "it reports zero rows whose named test does not
    reach them" false by three rows.
    """
    path = ROOT / "docs" / "grade-attribution.json"
    if not path.exists():
        return {"available": False, "why": "scripts/verify_grade_attribution.py has not run"}
    d = json.loads(path.read_text(encoding="utf-8"))
    v = d.get("verdicts") or {}
    blind = sum(n for label, n in v.items() if label.startswith("blind"))
    return {
        "available": True,
        "measured_on": d.get("measured_on"),
        "rows": d.get("rows"),
        "test_files": d.get("test_files"),
        "confirmed": v.get("confirmed by line coverage"),
        "blind": blind,
        "off_surface": v.get("unit not on the current surface"),
        "findings": len(d.get("findings") or []),
    }


def _ledger() -> dict:
    """The published three-state status of every public code unit.

    Generated by scripts/capability_status.py and read here so the hardening page
    can render the same counts the status grid does, from one source, rather than
    from numbers typed into the page by hand.
    """
    path = ROOT / "docs" / "capability-status.json"
    if not path.exists():
        return {"available": False, "why": "docs/capability-status.json has not been generated"}
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = data.get("counts") or {}
    total = sum(counts.values()) or None
    # The core assessment path, counted here so the page never types it. Zero open
    # defects in the core is a weaker claim than it reads, and it is only honest
    # beside the number nobody has examined, so both travel together.
    # Read from the ledger, never redeclared here: one definition of the boundary.
    core_areas = set(data.get("core_areas") or data.get("tier1_areas") or [])
    preview_areas = set(data.get("preview_areas") or [])
    core: dict[str, int] = {}
    preview: dict[str, int] = {}
    path_counts: dict[str, int] = {}
    for rec in (data.get("units") or {}).values():
        if rec.get("area") in core_areas:
            core[rec["status"]] = core.get(rec["status"], 0) + 1
        elif rec.get("area") in preview_areas:
            preview[rec["status"]] = preview.get(rec["status"], 0) + 1
        if rec.get("area") in ASSESSMENT_PATH_AREAS:
            path_counts[rec["status"]] = path_counts.get(rec["status"], 0) + 1
    return {
        "available": True,
        "total": total,
        "core_areas": sorted(core_areas),
        "core_total": sum(core.values()) or None,
        "core_checked": core.get("CHECKED", 0),
        "core_fix_pending": core.get("FIX PENDING", 0),
        "core_not_checked": core.get("NOT CHECKED", 0),
        "core_to_close": core.get("FIX PENDING", 0) + core.get("NOT CHECKED", 0),
        "preview_areas": sorted(preview_areas),
        "preview_total": sum(preview.values()) or None,
        "preview_checked": preview.get("CHECKED", 0),
        # PUBLISHED BECAUSE THE ROW DID NOT ADD UP (2026-09-28). core had a
        # fix_pending key and preview did not, so the published preview numbers were
        # 182 total against 84 checked plus 90 not checked, which is 174. The eight
        # missing units are the preview units with a KNOWN OPEN DEFECT, and the
        # quality page rendered a dash in that cell, which a reader takes as none.
        # Daniel spotted it by adding up the row. tests/test_kpi_scope_rows_add_up.py
        # now refuses a scope whose parts do not sum to its own total, which is the
        # check that was missing rather than the number.
        "preview_fix_pending": preview.get("FIX PENDING", 0),
        "preview_not_checked": preview.get("NOT CHECKED", 0),
        # THE ASSESSMENT PATH, the narrow claim inside the core claim. The page
        # carried "390 of the 1,569 code units, 251 examined and clean, 139 never
        # examined" as three typed digits with no generator behind them, and the
        # measured total was 391 by the time anybody added them up. Counted here so
        # the page cannot type them, and the area list travels with the number
        # because "the assessment path" is a boundary somebody drew, not a fact.
        "path_areas": sorted(ASSESSMENT_PATH_AREAS),
        "path_total": sum(path_counts.values()) or None,
        "path_checked": path_counts.get("CHECKED", 0),
        "path_fix_pending": path_counts.get("FIX PENDING", 0),
        "path_not_checked": path_counts.get("NOT CHECKED", 0),
        "preview_to_close": preview.get("FIX PENDING", 0) + preview.get("NOT CHECKED", 0),
        # A COUNT OF ZERO IS A MEASUREMENT, NOT AN ABSENCE, and the default is what
        # says so. scripts/capability_status.py builds `counts` by tallying the
        # statuses it finds, so a status nothing holds has no key at all.
        #
        # Measured 2026-09-29, the day FIX PENDING reached zero for the first time:
        # the key vanished, `.get()` answered None, and both
        # scripts/beta_go_live_docs.py and scripts/bgl6_register.py died with
        # "TypeError: unsupported format string passed to NoneType.__format__". The
        # whole campaign is about a neutral value standing in for one nobody
        # measured; this is its mirror, an absence standing in for a real zero, in
        # the tooling that reports it. Three states still exist above this: an
        # UNAVAILABLE ledger returns available=False, which is the could-not-check.
        "checked": counts.get("CHECKED", 0),
        "fix_pending": counts.get("FIX PENDING", 0),
        "not_checked": counts.get("NOT CHECKED", 0),
        "evidence_dates": data.get("evidence_dates") or [],
    }


def build() -> dict:
    _v, proof, measured_on, counts_as_of = _package()
    batches: dict[str, int] = {}
    for r in proof.values():
        batches[r["batch"]] = batches.get(r["batch"], 0) + 1
    n, reason = test_count()
    surf = _walk_public_surface()
    return {
        # PAGE ONE, what the library is and how big it is.
        "size": size(),
        "surface": surf,
        # PAGE TWO, how much of that surface was actually checked.
        "hardening": {
            "proof": {b: batches.get(b, 0) for b in ("BGL-A", "BGL-B", "BGL-C", "BGL-D")},
            "defects": defects(),
            "tests": {"collected": n, "note": reason},
            "ci": ci_measured(),
            "remaining": work_list(surf)["by_return"],
            "test_composition": test_composition(),
            "grading": grading(),
            # Per WAVE, unmerged, so a reader can see that the biggest wave is
            # also the least checked. The merged total above cannot show that.
            "waves": waves(),
        },
        "ledger": _ledger(),
        "release_gate": _gate(),
        "second_round": _second_round(),
        "attribution": attribution(),
        "measured_on": measured_on,
        "counts_as_of": counts_as_of,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    kpis = build()
    if not args.check:
        print(json.dumps(kpis, indent=2))
        return 0
    site = ROOT / "docs" / "site" / "data" / "library-stats.json"
    published = (json.loads(site.read_text(encoding="utf-8")) or {}).get("kpis")
    absent = missing_extras(((published or {}).get("surface") or {}).get("optional_extras_present"))
    if absent:
        print(
            "COULD NOT CHECK: docs/site/data/library-stats.json was measured with the "
            f"optional extras {absent} importable and this environment lacks them, so "
            "it cannot measure the surface that was published. Install the extras the "
            "published-docs CI job installs and run the check again.",
            file=sys.stderr,
        )
        return 2
    if published != kpis:
        print("STALE: docs/site/data/library-stats.json kpis block", file=sys.stderr)
        print("  run ./scripts/refresh-manifest.sh", file=sys.stderr)
        return 1
    print("site KPIs are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
