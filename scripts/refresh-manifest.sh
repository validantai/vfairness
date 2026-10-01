#!/usr/bin/env bash
# Regenerates vfairness-manifest.json from the capability registry.
# Run this after any change to _registry.py or adding new classes.
#
# Usage:
#   ./scripts/refresh-manifest.sh
#
# This script is also called automatically by the platform's build process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$REPO_ROOT/vfairness-manifest.json"

cd "$REPO_ROOT"

# PREFER THE PROJECT VENV, because several published figures depend on WHICH
# optional extras are importable and this script used bare `python`, meaning
# whatever happened to be first on PATH.
#
# Measured 2026-09-27: run with the system python it reported
# "docs/site/data/library-stats.json is up to date" while
# `.venv/bin/python scripts/library_kpis.py --check` reported STALE on the same
# file, and both were right about their own interpreter. The published block
# described an environment with the dashboard extra and without mcp, xai or
# causal, which is not the environment the wheel is tested in; the surface total
# differed by nine units and vfairness.mcp.server appeared as an import failure.
# A refresher that reports success having measured the wrong interpreter is the
# same defect this repository audits its own library for.
#
# The script's own error message below already hinted at it ("is vfairness
# importable by 'python'? try the project venv"), which fires only when the
# import fails outright. It succeeded here and produced the wrong numbers.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PATH="$REPO_ROOT/.venv/bin:$PATH"
  export PATH
fi

# Generate fresh manifest into a temp file. The '|| true' keeps set -e from
# aborting here so the validity guard below can run and report clearly.
python -m vfairness._manifest > "$MANIFEST.tmp" 2>/dev/null || true

# REFUSE to publish an empty or invalid regeneration. A 'python' that cannot
# import vfairness (e.g. a base conda interpreter) prints nothing yet can exit 0;
# that previously overwrote the good manifest with an empty file and broke every
# downstream consumer (the platform sync, the docs stat box, the KG seed). Never
# replace a good manifest with a broken one: keep the existing one and fail loud.
if ! python -c "import json,sys; d=json.load(open('$MANIFEST.tmp')); sys.exit(0 if d.get('capabilities') else 1)" 2>/dev/null; then
  rm -f "$MANIFEST.tmp"
  echo "ERROR: manifest regeneration produced empty/invalid output (is vfairness importable by 'python'? try the project venv). Kept the existing manifest." >&2
  exit 1
fi

# Only update if content changed (avoids unnecessary git noise)
if ! cmp -s "$MANIFEST.tmp" "$MANIFEST" 2>/dev/null; then
  mv "$MANIFEST.tmp" "$MANIFEST"
  echo "vfairness-manifest.json updated ($(python -c "import json; m=json.load(open('$MANIFEST')); print(m['stats']['total_capabilities'])" 2>/dev/null || echo '?') capabilities)"
else
  rm "$MANIFEST.tmp"
  echo "vfairness-manifest.json is up to date"
fi

# Project the manifest stats into the docs site so the changelog page can look
# up the true, current figures (version / lines-of-code / counts) at runtime.
# Keeps the public stat box from ever going stale -- it is regenerated here, not
# hand-edited. See docs/site/concepts/changelog.html (fetch of ../data/library-stats.json).
SITE_STATS="$REPO_ROOT/docs/site/data/library-stats.json"
python - "$MANIFEST" "$SITE_STATS" <<'PY'
import json, sys
manifest_path, out_path = sys.argv[1], sys.argv[2]
m = json.load(open(manifest_path))
s = m["stats"]
# Derive the maturity label from the packaging classifier so the stat box can
# never disagree with pyproject.toml (it used to stamp a literal "Alpha").
import re, pathlib
pyproject = pathlib.Path(manifest_path).parent / "pyproject.toml"
mstat = re.search(r'Development Status :: \d+ - (\w+)', pyproject.read_text())
# The capability LIST, grouped by package, so the getting-started page can show
# what the 215 actually ARE rather than only how many there are. A number nobody
# can expand is a claim; a list is a thing a reader can check. Derived from the
# manifest, never hand-written, so it cannot drift from the registry.
_PACKAGE_LABELS = {
    "preprocessing": "Pre-processing: audit and repair the data",
    "in_processing": "In-processing: train with fairness constraints",
    "post_processing": "Post-processing: calibrate and adjust outputs",
    "evaluation": "Evaluation: measure fairness",
    "operations": "Operations: gate, monitor, report",
    "llm": "LLM fairness",
    "agents": "Agent fairness",
    "multi_agent": "Multi-agent fairness",
    "xai": "Explainability",
    "vision": "Vision fairness",
    "explainer": "Plain-language explanation",
}
groups = {}
for key, entry in m["capabilities"].items():
    pkg = entry["module_path"].split(".")[0]
    g = groups.setdefault(pkg, {"label": _PACKAGE_LABELS.get(pkg, pkg), "items": []})
    g["items"].append({"name": entry["name"], "key": key, "kind": entry["kind"]})
for g in groups.values():
    g["items"].sort(key=lambda i: i["name"].lower())
    g["count"] = len(g["items"])
capability_groups = [
    dict(package=pkg, **groups[pkg])
    for pkg in sorted(groups, key=lambda p: (-groups[p]["count"], p))
]

sys.path.insert(0, str(pathlib.Path(manifest_path).parent / "scripts"))
import library_kpis  # noqa: E402

kpis = library_kpis.build()

# The Beta Go-Live GRADE of each capability, attached to the capability it
# belongs to. The counts alone (BGL-A 156, BGL-B 57 ...) are four numbers a
# reader cannot cross-reference against anything; with the batch on the item,
# the page can NAME which capability holds which grade. Read from the same
# CAPABILITY_PROOF the counts are derived from, so a name and a count can never
# disagree. library_kpis.build() has already put src/ on sys.path.
from vfairness._proof_status import CAPABILITY_PROOF  # noqa: E402

for _g in capability_groups:
    for _item in _g["items"]:
        _item["batch"] = CAPABILITY_PROOF[_item["key"]]["batch"]

# Severity of the 157 findings AS THE CENSUS RECORDED THEM. Deliberately not
# inside the "kpis" block: the staleness check in library_kpis.py compares that
# block field for field against what it can measure, so a key added to it here
# would make every published copy report STALE and send the next person chasing
# a phantom.
_census = json.loads(
    (pathlib.Path(manifest_path).parent / "docs" / "beta-go-live-census-2026-09-11.json")
    .read_text(encoding="utf-8")
)
defect_severity_at_census = (_census.get("totals") or {}).get("open_defects_by_severity") or {}

out = {
    "version": m["version"],
    "status": mstat.group(1) if mstat else "Beta",
    "lines_of_code": s["lines_of_code"],
    "capabilities": s["total_capabilities"],
    "classes": s["classes"],
    "functions": s["functions"],
    "svg_templates": s["svg_templates"],
    # classes + functions is LESS than capabilities, and that is not an error:
    # both are sets of NAMES, and four names serve two dispatch keys each.
    "capability_groups": capability_groups,
    "proof": s.get("proof", {}),
    # The severity split of the findings at census time, when all 157 were open.
    # Labelled "at the census" wherever it is shown: 155 have since been fixed,
    # so reading it as a current split would be wrong by 155.
    "defect_severity_at_census": defect_severity_at_census,
    # THE PUBLISHED KPIs. Measured by scripts/library_kpis.py, never typed. The
    # tile row on getting-started once read "95 registered functions" and "116
    # registered classes" beside "215 capabilities": all three were correct and
    # counted different things (names versus dispatch keys), so a reader could
    # only conclude the page could not add up. Every figure now comes from one
    # measurement with one definition attached, including the figures that say
    # what was NOT graded.
    "kpis": kpis,
    "_note": "Generated from vfairness-manifest.json by scripts/refresh-manifest.sh. Do not edit by hand.",
}
# Only rewrite when the content actually changed (same no-churn intent as the
# manifest guard above; the output is deterministic now that there is no
# timestamp, so an unchanged run leaves the file, and git, untouched).
new_content = json.dumps(out, indent=2) + "\n"
try:
    old_content = open(out_path).read()
except FileNotFoundError:
    old_content = None
loc = out["lines_of_code"]
if new_content != old_content:
    with open(out_path, "w") as f:
        f.write(new_content)
    print(f"docs/site/data/library-stats.json updated (LOC total={loc['total']}, code={loc['code']})")
else:
    print("docs/site/data/library-stats.json is up to date")
PY
