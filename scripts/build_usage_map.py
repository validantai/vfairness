"""Generate the vfairness ↔ Navigator Library Usage Map.

Scans every Navigator-class entry point IN THIS REPOSITORY for vfairness
usage, joins it against the library manifest (capabilities, classes,
functions, SVG templates) and emits a JSON file consumed by the docs Usage
Map page and the inline-badge script.

Navigator-class entry points discovered automatically:

  * Capability demonstrations:   notebooks/*.ipynb (one track per notebook)

THE SCAN IS ANCHORED ON THE LIBRARY ROOT, AND THAT IS A PUBLISH DECISION
(2026-09-10). This file and the JSON it writes are both published: they ship
in the public export and they are served, unauthenticated, from the docs
site. Until this date the anchor was ``parents[2]``, the private development
monorepo one level ABOVE the library, so the map enumerated four entry
points that exist only up there (a predictive pipeline module and its
utilities, and two gallery SVG generators) and prefixed every notebook with
the private subtree name. Measured on the shipped map that day: 54 of 330
rows named one of those four, the docs page rendered ``tracks[].path``
verbatim inside a ``<code>`` element, and the same bytes were live on the
public site. The generator could not run in the public repository either,
because the layout it hardcoded is not the layout over there.

The four filenames are not written out here on purpose: they are on the
export leak denylist, so naming them would abort the export on this very
file. The export script's LEAK_PATTERNS_RECON block carries them in full,
with this incident as the reason.

A path that does not resolve in THIS repository is therefore refused rather
than emitted: see ``TrackSpec.rel`` and ``discover_tracks``. The rule is not
"drop the four": it is that this map may only describe what a reader holding
this repository can open.

Run:
    python scripts/build_usage_map.py
    python scripts/build_usage_map.py --check   # CI-friendly
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# The library repository root: the directory holding pyproject.toml, notebooks/
# and docs/. In the development monorepo that is the library SUBTREE; in the
# published repository it is the repository root. Anchoring here is what makes
# every path this script emits mean the same thing in both places.
LIBRARY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = LIBRARY_ROOT / "vfairness-manifest.json"
OUTPUT_PATH = LIBRARY_ROOT / "docs" / "site" / "data" / "usage_map.json"

STEP_HEADER_RE = re.compile(r"^\s*#\s*Step\s+([0-9]+[a-z]?)\s*[-–—]\s*(.+?)\s*$", re.IGNORECASE)
STEP_METHOD_RE = re.compile(r"^step([0-9]+[a-z]?)_(.+)$")
NB_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
LIBRARY_ONLY_RE = re.compile(r"#\s*usage-map:\s*library-only", re.IGNORECASE)


# ── Track configuration ──────────────────────────────────────────────────
# Each track is an entry point that exercises the library. Categories group
# tracks that target the same kind of AI system (Predictive, Generative,
# Agentic, …) or the same library surface (Rendering, Operations, …).


@dataclass
class TrackSpec:
    id: str
    label: str
    category: str
    path: Path
    kind: str  # 'python' | 'notebook'

    @property
    def rel(self) -> str:
        """The track path, relative to the library root.

        REFUSES rather than falls back. The previous fallback returned
        ``str(self.path)``, an ABSOLUTE path on the machine that ran the
        build, for any entry point outside the anchor, and that string was
        then written into the published JSON and rendered on a public page as
        if it were a repository-relative location. A path this repository
        cannot resolve is an unmeasured value; it does not get a neutral
        default here.
        """
        try:
            return str(self.path.relative_to(LIBRARY_ROOT))
        except ValueError as exc:
            raise ValueError(
                f"track {self.id!r} points at {self.path}, which is outside the "
                f"library root {LIBRARY_ROOT}. The usage map may only name paths a "
                f"reader holding this repository can open."
            ) from exc


CATEGORY_ORDER = [
    "Predictive AI · End-to-End",
    "Pre-Training Bias Auditing",
    "Preprocessing",
    "In-Processing",
    "Post-Processing",
    "Evaluation & Metrics",
    "Monitoring",
    "Reporting",
    "Experimentation",
    "Operations · CI/CD",
    "Rendering · SVG",
    "Library Validation",
]

# Filename → (label, category). Patterns are evaluated in order; first match wins.
NOTEBOOK_CATEGORY = [
    (re.compile(r"_0_library_validation"), "Library Validation", "Library Validation"),
    (re.compile(r"_0_svg_rendering"), "SVG Rendering Demo", "Rendering · SVG"),
    (re.compile(r"_1_bias_detection"), "Bias Detection Demo", "Pre-Training Bias Auditing"),
    (re.compile(r"_1_feature_engineering"), "Feature Engineering Demo", "Preprocessing"),
    (re.compile(r"_2_training"), "Fair Training Demo", "In-Processing"),
    (re.compile(r"_3_calibration"), "Calibration Demo", "Post-Processing"),
    (re.compile(r"_4_metrics"), "Metrics Demo", "Evaluation & Metrics"),
    (re.compile(r"_5_monitoring"), "Monitoring Demo", "Monitoring"),
    (re.compile(r"_6_reporting"), "Reporting Demo", "Reporting"),
    (re.compile(r"_7_experimentation"), "Experimentation Demo", "Experimentation"),
    (re.compile(r"_8_cicd"), "CI/CD Gates Demo", "Operations · CI/CD"),
    (re.compile(r"_8_workflow_integration"), "Workflow Integration Demo", "Operations · CI/CD"),
]


def discover_tracks() -> List[TrackSpec]:
    """Every Navigator-class entry point that lives inside the library root.

    Entry points ABOVE the library root are deliberately not discovered. In
    the development monorepo they exist (a predictive pipeline and two SVG
    generators sit beside the library), and scanning them used to put their
    paths into a map that is published and served publicly. They are not part
    of what a reader of this repository holds, so this map does not describe
    them; the surface it measures is stated on the page that renders it.
    """
    out: List[TrackSpec] = []

    # One track per demonstration notebook.
    nb_dir = LIBRARY_ROOT / "notebooks"
    if nb_dir.exists():
        for nb in sorted(nb_dir.glob("*.ipynb")):
            label, category = _classify_notebook(nb.name)
            out.append(
                TrackSpec(
                    id="track-nb-" + nb.stem.replace("_", "-"),
                    label=label,
                    category=category,
                    path=nb,
                    kind="notebook",
                )
            )

    # Fail closed. Discovery is the only place a path enters this map, so it is
    # the only place the anchor can be checked once for every track.
    for t in out:
        t.rel  # raises for anything outside LIBRARY_ROOT

    return out


def _classify_notebook(filename: str) -> Tuple[str, str]:
    for pat, label, cat in NOTEBOOK_CATEGORY:
        if pat.search(filename):
            return label, cat
    return filename, "Library Validation"


# ── Manifest helpers ─────────────────────────────────────────────────────


def load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def collect_public_symbols(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    symbols: Dict[str, Dict[str, Any]] = {}
    for cap_key, cap in manifest.get("capabilities", {}).items():
        name = cap.get("name")
        if not name:
            continue
        symbols[name] = {
            "name": name,
            "module": cap.get("module_path", ""),
            "stage": cap.get("pipeline_stage", ""),
            "kind": cap.get("kind", ""),
            "capability": cap_key,
            "svg_templates": list(cap.get("svg_templates", []) or []),
        }
    for fn in manifest.get("function_names", []):
        symbols.setdefault(
            fn,
            {
                "name": fn,
                "module": "",
                "stage": "",
                "kind": "function",
                "capability": "",
                "svg_templates": [],
            },
        )
    for cls in manifest.get("class_names", []):
        symbols.setdefault(
            cls,
            {
                "name": cls,
                "module": "",
                "stage": "",
                "kind": "class",
                "capability": "",
                "svg_templates": [],
            },
        )
    return symbols


def all_svg_templates(manifest: Dict[str, Any]) -> List[str]:
    raw = manifest.get("svg_templates", [])
    if isinstance(raw, dict):
        return sorted(raw.keys())
    return sorted(list(raw))


def annotated_library_only(symbols: Dict[str, Dict[str, Any]]) -> Set[str]:
    annotated: Set[str] = set()
    src_root = LIBRARY_ROOT / "src" / "vfairness"
    for py in src_root.rglob("*.py"):
        try:
            text = py.read_text()
            tree = ast.parse(text, filename=str(py))
        except Exception:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name not in symbols:
                    continue
                window = lines[max(0, node.lineno - 2) : node.lineno]
                if any(LIBRARY_ONLY_RE.search(L) for L in window):
                    annotated.add(node.name)
    return annotated


# ── Python file scanner ──────────────────────────────────────────────────


def _step_markers_from_source(source: str) -> List[Tuple[int, str, str]]:
    markers: List[Tuple[int, str, str]] = []
    for i, line in enumerate(source.splitlines(), start=1):
        m = STEP_HEADER_RE.match(line)
        if m:
            num, label = m.group(1), m.group(2).strip()
            markers.append((i, f"step_{num}", f"Step {num} · {label}"))
    markers.sort()
    return markers


def _step_methods(tree: ast.Module) -> List[Tuple[int, int, str, str]]:
    out: List[Tuple[int, int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            m = STEP_METHOD_RE.match(node.name)
            if not m:
                continue
            num = m.group(1)
            label = m.group(2).replace("_", " ").title()
            end = getattr(node, "end_lineno", node.lineno)
            out.append((node.lineno, end, f"step_{num}", f"Step {num} · {label}"))
    return out


def _step_for_line(markers, methods, lineno: int) -> Tuple[str, str]:
    for s, e, sid, label in methods:
        if s <= lineno <= e:
            return sid, label
    cur = ("module", "Module scope")
    for line, sid, label in markers:
        if line <= lineno:
            cur = (sid, label)
        else:
            break
    return cur


def scan_python(track: TrackSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    source = track.path.read_text()
    try:
        tree = ast.parse(source, filename=str(track.path))
    except SyntaxError as e:
        print(f"  [warn] skipping {track.rel}: {e}", file=sys.stderr)
        return [], []

    markers = _step_markers_from_source(source)
    methods = _step_methods(tree)

    imports: Dict[str, Dict[str, Any]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vfairness"):
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = {
                    "qualified": f"{node.module}.{alias.name}",
                    "module": node.module,
                    "import_step": _step_for_line(markers, methods, node.lineno),
                }
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("vfairness"):
                    local = alias.asname or alias.name
                    imports[local] = {
                        "qualified": alias.name,
                        "module": alias.name,
                        "import_step": _step_for_line(markers, methods, node.lineno),
                    }

    usages: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int]] = set()

    def record(name: str, lineno: int, ref_kind: str) -> None:
        info = imports.get(name)
        if not info:
            return
        key = (name, lineno)
        if key in seen:
            return
        seen.add(key)
        sid, slabel = _step_for_line(markers, methods, lineno)
        if sid == "module" and not methods and info["import_step"][0] != "module":
            sid, slabel = info["import_step"]
        usages.append(
            {
                "track_id": track.id,
                "track_label": track.label,
                "track_category": track.category,
                "file": track.rel,
                "line": lineno,
                "step_id": sid,
                "step_label": slabel,
                "symbol_local": name,
                "qualified": info["qualified"],
                "module": info["module"],
                "ref_kind": ref_kind,
            }
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                record(f.id, node.lineno, "call")
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                record(f.value.id, node.lineno, "method")
        elif isinstance(node, ast.Name) and not isinstance(getattr(node, "ctx", None), ast.Store):
            if node.id in imports:
                record(node.id, node.lineno, "reference")

    steps = [{"id": sid, "label": label} for _, sid, label in markers]
    steps += [{"id": sid, "label": label} for _, _, sid, label in methods]
    # de-dup preserving order
    seen_s = set()
    deduped = []
    for s in steps:
        if s["id"] in seen_s:
            continue
        seen_s.add(s["id"])
        deduped.append(s)
    return usages, deduped


# ── Notebook scanner ─────────────────────────────────────────────────────


def scan_notebook(track: TrackSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    try:
        nb = json.loads(track.path.read_text())
    except Exception as e:
        print(f"  [warn] cannot read notebook {track.rel}: {e}", file=sys.stderr)
        return [], []

    # Concatenate all code cells into one synthetic source, with per-line
    # provenance pointing back to (cell_index, line_in_cell). Markdown cells
    # contribute step labels via their nearest preceding section header.
    code_lines: List[str] = []
    line_provenance: List[Tuple[int, int]] = []  # synthetic_line -> (cell_idx, cell_line)
    current_step: Tuple[str, str] = ("nb-intro", "Introduction")
    line_step: List[Tuple[str, str]] = [("nb-intro", "Introduction")]  # idx by synthetic_line-1
    step_order: List[Dict[str, str]] = []
    seen_steps: Set[str] = set()

    def set_step(label: str) -> None:
        nonlocal current_step
        clean = re.sub(r"\s+", " ", label).strip()[:80]
        slug = re.sub(r"[^a-z0-9]+", "-", clean.lower()).strip("-")[:50] or "section"
        sid = f"nb-{slug}"
        if sid not in seen_steps:
            seen_steps.add(sid)
            step_order.append({"id": sid, "label": clean})
        current_step = (sid, clean)

    for idx, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type")
        src = cell.get("source", "")
        if isinstance(src, list):
            # nbformat says items should end with \n except possibly the last,
            # but some malformed notebooks omit trailing newlines mid-list and
            # naive concatenation glues a comment onto the next statement.
            normalized = []
            for i, s in enumerate(src):
                if not s.endswith("\n") and i < len(src) - 1:
                    s = s + "\n"
                normalized.append(s)
            src = "".join(normalized)

        if ctype == "markdown":
            # Extract first heading from this cell to refresh the step label.
            for m in NB_HEADER_RE.finditer(src):
                # Skip the very first H1 of the notebook to keep step labels at ## level.
                level = len(m.group(1))
                if level <= 2:
                    set_step(m.group(2))
                    break
            continue
        if ctype != "code":
            continue

        # Strip cell magics and shell escapes which break the AST parser.
        cell_src = "\n".join(
            "" if (L.lstrip().startswith("%") or L.lstrip().startswith("!")) else L
            for L in src.split("\n")
        )

        for j, line in enumerate(cell_src.split("\n"), start=1):
            code_lines.append(line)
            line_provenance.append((idx, j))
            line_step.append(current_step)

    if not code_lines:
        return [], step_order

    synthetic = "\n".join(code_lines)
    try:
        tree = ast.parse(synthetic, filename=str(track.path))
    except SyntaxError as e:
        print(f"  [warn] notebook AST parse failed for {track.rel}: {e}", file=sys.stderr)
        return [], step_order

    imports: Dict[str, Dict[str, Any]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vfairness"):
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = {
                    "qualified": f"{node.module}.{alias.name}",
                    "module": node.module,
                }
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("vfairness"):
                    local = alias.asname or alias.name
                    imports[local] = {"qualified": alias.name, "module": alias.name}

    usages: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int]] = set()

    def line_to_step(lineno: int) -> Tuple[str, str]:
        if 0 < lineno <= len(line_step):
            return line_step[lineno - 1]
        return ("nb-intro", "Introduction")

    def line_to_cell(lineno: int) -> int:
        if 0 < lineno <= len(line_provenance):
            return line_provenance[lineno - 1][0]
        return -1

    def record(name: str, lineno: int, ref_kind: str) -> None:
        info = imports.get(name)
        if not info:
            return
        key = (name, lineno)
        if key in seen:
            return
        seen.add(key)
        sid, slabel = line_to_step(lineno)
        cell = line_to_cell(lineno)
        usages.append(
            {
                "track_id": track.id,
                "track_label": track.label,
                "track_category": track.category,
                "file": track.rel,
                "line": -1,
                "cell": cell,
                "step_id": sid,
                "step_label": slabel,
                "symbol_local": name,
                "qualified": info["qualified"],
                "module": info["module"],
                "ref_kind": ref_kind,
            }
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                record(f.id, node.lineno, "call")
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                record(f.value.id, node.lineno, "method")
        elif isinstance(node, ast.Name) and not isinstance(getattr(node, "ctx", None), ast.Store):
            if node.id in imports:
                record(node.id, node.lineno, "reference")

    return usages, step_order


# ── Build ────────────────────────────────────────────────────────────────


def build() -> Dict[str, Any]:
    manifest = load_manifest()
    symbols = collect_public_symbols(manifest)
    svg_template_pool = all_svg_templates(manifest)
    tracks = discover_tracks()

    track_records: List[Dict[str, Any]] = []
    all_usages: List[Dict[str, Any]] = []
    for t in tracks:
        usages, steps = scan_notebook(t) if t.kind == "notebook" else scan_python(t)
        all_usages.extend(usages)
        track_records.append(
            {
                "id": t.id,
                "label": t.label,
                "category": t.category,
                "path": t.rel,
                "kind": t.kind,
                "steps": steps,
                "usage_count": len(usages),
                "symbol_count": len({u["symbol_local"] for u in usages}),
            }
        )

    # Symbols that were never in the manifest but ARE used by a track.
    for u in all_usages:
        bare = u["qualified"].rsplit(".", 1)[-1]
        if bare not in symbols:
            symbols[bare] = {
                "name": bare,
                "module": u["module"].removeprefix("vfairness.")
                if u["module"].startswith("vfairness.")
                else u["module"],
                "stage": "",
                "kind": "imported",
                "capability": "",
                "svg_templates": _infer_svg_template(bare),
            }

    library_only = annotated_library_only(symbols)

    # SVG template inference for *_to_svg adapters even when listed in capabilities.
    for name, meta in symbols.items():
        if not meta["svg_templates"]:
            meta["svg_templates"] = _infer_svg_template(name)

    rows: List[Dict[str, Any]] = []
    for name, meta in sorted(symbols.items()):
        usages_for_sym = [u for u in all_usages if u["qualified"].rsplit(".", 1)[-1] == name]
        if usages_for_sym:
            status = "active"
        elif name in library_only:
            status = "library-only"
        else:
            status = "unused"
        tracks_used = sorted(
            {(u["track_id"], u["track_label"], u["track_category"]) for u in usages_for_sym}
        )
        categories_used = sorted({u["track_category"] for u in usages_for_sym})
        steps_used = []
        seen_steps = set()
        for u in usages_for_sym:
            key = (u["track_id"], u["step_id"])
            if key in seen_steps:
                continue
            seen_steps.add(key)
            steps_used.append(
                {
                    "track_id": u["track_id"],
                    "step_id": u["step_id"],
                    "step_label": u["step_label"],
                    "track_label": u["track_label"],
                }
            )
        rows.append(
            {
                "symbol": name,
                "module": meta["module"],
                "stage": meta["stage"],
                "kind": meta["kind"],
                "capability": meta["capability"],
                "svg_templates": meta["svg_templates"],
                "status": status,
                "categories": categories_used,
                "tracks": [{"id": t[0], "label": t[1], "category": t[2]} for t in tracks_used],
                "steps": steps_used,
                "usages": usages_for_sym,
                "use_count": len(usages_for_sym),
            }
        )

    # SVG template index — which tracks actually produce each template.
    svg_index: List[Dict[str, Any]] = []
    for tpl in svg_template_pool:
        caps_producing = sorted([r["symbol"] for r in rows if tpl in r["svg_templates"]])
        producers = [r for r in rows if tpl in r["svg_templates"]]
        tracks_producing = sorted(
            {
                (t["id"], t["label"], t["category"])
                for r in producers
                if r["status"] == "active"
                for t in r["tracks"]
            }
        )
        svg_index.append(
            {
                "template": tpl,
                "symbols": caps_producing,
                "active_tracks": [
                    {"id": t[0], "label": t[1], "category": t[2]} for t in tracks_producing
                ],
                "active": bool(tracks_producing),
            }
        )

    # ── Canonical Navigator pipelines ──
    # Each pipeline ("predictive AI" / "generative AI") has its own ordered set
    # of audit steps. Each step lists the library module prefixes it normally
    # invokes — we use those prefixes to attribute symbols to steps. Empty steps
    # are still rendered (faded) so the user can see where the library has gaps.
    PREDICTIVE_PIPELINE = [
        {
            "id": "p1-validate",
            "label": "1. Data Validation",
            "modules": ["preprocessing.bias_detection"],
        },
        {
            "id": "p2-detect",
            "label": "2. Bias Detection",
            "modules": ["preprocessing.bias_detection", "explainer"],
        },
        {
            "id": "p3-feature",
            "label": "3. Feature Engineering",
            "modules": ["preprocessing.feature_engineering"],
        },
        {"id": "p4-train", "label": "4. Fair Training", "modules": ["in_processing"]},
        {
            "id": "p5-calibrate",
            "label": "5. Calibration",
            "modules": [
                "post_processing.calibration",
                "post_processing.reweighting",
                "post_processing.threshold_optimization",
            ],
        },
        {
            "id": "p6-metrics",
            "label": "6. Metrics & Robustness",
            "modules": ["evaluation.vfairness_metrics"],
        },
        {"id": "p7-cicd", "label": "7. CI/CD Gating", "modules": ["operations.cicd"]},
        {
            "id": "p8-monitor",
            "label": "8. Production Monitoring",
            "modules": ["operations.monitoring"],
        },
        {
            "id": "p9-report",
            "label": "9. Multi-Tier Reporting",
            "modules": ["operations.reporting"],
        },
    ]
    GENERATIVE_PIPELINE = [
        {
            "id": "g1-validate",
            "label": "1. Output Data Validation",
            "modules": ["preprocessing.bias_detection", "llm"],
        },
        {
            "id": "g2-detect",
            "label": "2. Bias Detection",
            "modules": ["preprocessing.bias_detection", "explainer", "llm"],
        },
        {"id": "g3-probes", "label": "3. Counterfactual Probes", "modules": ["llm"]},
        {
            "id": "g4-calibrate",
            "label": "4. Output Calibration",
            "modules": ["post_processing.calibration", "llm"],
        },
        {
            "id": "g5-metrics",
            "label": "5. Metrics & Robustness",
            "modules": ["evaluation.vfairness_metrics", "llm"],
        },
        {
            "id": "g6-regression",
            "label": "6. Regression Fairness",
            "modules": ["evaluation.vfairness_metrics.regression"],
        },
        {"id": "g7-cicd", "label": "7. CI/CD Gating", "modules": ["operations.cicd"]},
        {
            "id": "g8-monitor",
            "label": "8. Production Monitoring",
            "modules": ["operations.monitoring"],
        },
        {
            "id": "g9-report",
            "label": "9. Multi-Tier Reporting",
            "modules": ["operations.reporting"],
        },
    ]

    def _row_matches_step(r, step) -> bool:
        mod = r.get("module") or ""
        mod = mod.removeprefix("vfairness.")
        for prefix in step["modules"]:
            if mod == prefix or mod.startswith(prefix + "."):
                return True
        return False

    def _build_pipeline_flow(pipeline):
        """Return (steps[], step_to_module[], module_to_svg[], step_symbols{})."""
        step_to_module: Dict[Tuple[str, str, str], int] = {}
        module_to_svg: Dict[Tuple[str, str], int] = {}
        step_symbols: Dict[str, set] = {s["id"]: set() for s in pipeline}
        active_modules: set = set()

        for r in rows:
            if r["status"] != "active":
                continue
            mod_top = (r["module"].split(".")[0] or "(other)").removeprefix("vfairness.")
            for step in pipeline:
                if _row_matches_step(r, step):
                    key = (step["id"], step["label"], mod_top)
                    step_to_module[key] = step_to_module.get(key, 0) + 1
                    step_symbols[step["id"]].add(r["symbol"])
                    active_modules.add(mod_top)

        for r in rows:
            if r["status"] != "active":
                continue
            mod_top = (r["module"].split(".")[0] or "(other)").removeprefix("vfairness.")
            if mod_top not in active_modules:
                continue
            for tpl in r["svg_templates"]:
                # Restrict to symbols matching at least one pipeline step.
                if not any(_row_matches_step(r, step) for step in pipeline):
                    continue
                module_to_svg[(mod_top, tpl)] = module_to_svg.get((mod_top, tpl), 0) + 1

        steps_out = []
        for step in pipeline:
            syms = sorted(step_symbols[step["id"]])
            steps_out.append(
                {
                    "id": step["id"],
                    "label": step["label"],
                    "modules": step["modules"],
                    "symbols": syms,
                    "active": bool(syms),
                    "symbol_count": len(syms),
                }
            )

        return {
            "steps": steps_out,
            "step_to_module": [
                {"step_id": k[0], "step_label": k[1], "module": k[2], "weight": v}
                for k, v in sorted(
                    step_to_module.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][2])
                )
            ],
            "module_to_svg": [
                {"module": k[0], "svg": k[1], "weight": v}
                for k, v in sorted(module_to_svg.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
        }

    pipelines = {
        "predictive": {
            "label": "Predictive AI",
            "tagline": "Binary classifiers, regressors, ranking systems, CV models.",
            **_build_pipeline_flow(PREDICTIVE_PIPELINE),
        },
        "generative": {
            "label": "Generative AI",
            "tagline": "Foundation LLMs, prompt-based agents, content-generation systems.",
            **_build_pipeline_flow(GENERATIVE_PIPELINE),
        },
    }

    # Per-row nav_steps (so matrix can filter by step) — union of both pipelines.
    for r in rows:
        steps_for_row = []
        for pipe in (PREDICTIVE_PIPELINE, GENERATIVE_PIPELINE):
            for step in pipe:
                if _row_matches_step(r, step):
                    steps_for_row.append({"id": step["id"], "label": step["label"]})
        # Dedup preserving order.
        seen = set()
        kept = []
        for s in steps_for_row:
            if s["id"] in seen:
                continue
            seen.add(s["id"])
            kept.append(s)
        r["nav_steps"] = kept

    total = len(rows)
    active = sum(1 for r in rows if r["status"] == "active")

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "library_version": manifest.get("version", "unknown"),
        "summary": {
            "total_symbols": total,
            "active": active,
            "library_only": sum(1 for r in rows if r["status"] == "library-only"),
            "unused": sum(1 for r in rows if r["status"] == "unused"),
            "coverage_pct": round(100.0 * active / total, 1) if total else 0.0,
            "tracks_scanned": len(tracks),
            "total_usages": len(all_usages),
            "svg_templates_total": len(svg_template_pool),
            "svg_templates_active": sum(1 for s in svg_index if s["active"]),
        },
        "categories": [
            {
                "label": c,
                "track_count": sum(1 for t in track_records if t["category"] == c),
                "active_symbols": len(
                    {r["symbol"] for r in rows if r["status"] == "active" and c in r["categories"]}
                ),
            }
            for c in CATEGORY_ORDER
            if any(t["category"] == c for t in track_records)
        ],
        "tracks": track_records,
        "rows": rows,
        "svg_templates": svg_index,
        "pipelines": pipelines,
    }


def _infer_svg_template(name: str) -> List[str]:
    """If a symbol looks like `<template>_to_svg`, expose that template name."""
    if name.endswith("_to_svg"):
        return [name[: -len("_to_svg")]]
    return []


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Fail if unannotated unused symbols are found."
    )
    args = parser.parse_args()

    data = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2))

    s = data["summary"]
    print(f"  Tracks scanned:    {s['tracks_scanned']}")
    print(f"  Categories:        {len(data['categories'])}")
    print(f"  Symbols:           {s['total_symbols']}")
    print(f"  Active:            {s['active']}  ({s['coverage_pct']}%)")
    print(f"  Library-only:      {s['library_only']}")
    print(f"  Unused:            {s['unused']}")
    print(
        f"  SVG templates:     {s['svg_templates_active']} / {s['svg_templates_total']} produced by tracks"
    )
    print(f"  Total usages:      {s['total_usages']}")
    print(f"  Wrote:             {OUTPUT_PATH.relative_to(LIBRARY_ROOT)}")

    if args.check and s["unused"] > 0:
        unused = [r["symbol"] for r in data["rows"] if r["status"] == "unused"]
        print(f"\n  [check] {len(unused)} unannotated unused symbol(s):", file=sys.stderr)
        for sym in unused[:20]:
            print(f"          - {sym}", file=sys.stderr)
        if len(unused) > 20:
            print(f"          ... and {len(unused) - 20} more", file=sys.stderr)
        print(
            "\n  Annotate with `# usage-map: library-only` above the def/class, or wire it into a track.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
