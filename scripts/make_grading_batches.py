#!/usr/bin/env python3
"""Split the ungraded public surface into FILE-DISJOINT batches for grading.

File-disjoint is the whole point. Two agents editing one file in a shared
checkout produce a merge nobody reviewed, and this campaign has already had an
agent's sabotage read mid-write by another agent as a SyntaxError in prose.

Batches are sized by WORK, not by item count: an item whose triage says it
returns a number needs a fixture, a degenerate run, a fix, a pin and a sabotage,
while a renderer usually needs one execution to confirm what it returns. So a
measuring item is weighted heavily and a renderer lightly.

The triage carried into each batch is a GUESS ABOUT RETURN TYPES and is labelled
as such in the batch file, because an agent that treats it as a grade rebuilds
the defect this library was audited for, one level up.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import library_kpis  # noqa: E402

WEIGHT = {
    "returns a number or a result": 5,
    "unclassified, needs a human": 3,
    "no return annotation, needs a human": 3,
    "returns nothing or text": 1,
    "renders, prints or converts": 1,
    "could not resolve, needs a human": 3,
}
TARGET = 34  # weight per batch


def source_file(qual: str) -> str | None:
    entry = library_kpis._OBJECTS.get(qual)
    if entry is None:
        return None
    try:
        f = inspect.getsourcefile(entry[1])
    except Exception:  # noqa: BLE001
        return None
    if not f:
        return None
    try:
        return str(Path(f).resolve().relative_to(ROOT))
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--only-measuring",
        action="store_true",
        help="wave 1: only files that hold at least one measuring item",
    )
    args = ap.parse_args()

    surface = library_kpis._walk_public_surface()
    triage = library_kpis.work_list(surface)["items"]

    by_file: dict[str, list[dict]] = {}
    unresolved: list[str] = []
    for qual, category in sorted(triage.items()):
        rel = source_file(qual)
        if rel is None:
            unresolved.append(qual)
            continue
        by_file.setdefault(rel, []).append({"item": qual, "triage": category})

    files = []
    for rel, items in by_file.items():
        measuring = sum(1 for i in items if i["triage"] == "returns a number or a result")
        if args.only_measuring and not measuring:
            continue
        weight = sum(WEIGHT.get(i["triage"], 3) for i in items)
        files.append({"file": rel, "items": items, "weight": weight, "measuring": measuring})
    # Heaviest first, so one enormous file becomes its own batch instead of
    # dragging a batch far over target at the end.
    files.sort(key=lambda f: -f["weight"])

    batches: list[dict] = []
    for f in files:
        placed = False
        for b in batches:
            if b["weight"] + f["weight"] <= TARGET:
                b["files"].append(f["file"])
                b["items"].extend(f["items"])
                b["weight"] += f["weight"]
                b["measuring"] += f["measuring"]
                placed = True
                break
        if not placed:
            batches.append(
                {
                    "files": [f["file"]],
                    "items": list(f["items"]),
                    "weight": f["weight"],
                    "measuring": f["measuring"],
                }
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for n, b in enumerate(batches):
        name = f"g{n:03d}"
        names.append(name)
        (out / f"{name}.json").write_text(
            json.dumps(
                {
                    "batch": name,
                    "files_you_own": b["files"],
                    "items": b["items"],
                    "measuring_items": b["measuring"],
                    "weight": b["weight"],
                    "WARNING": (
                        "The `triage` field on each item is a GUESS made from the return "
                        "type annotation. It is NOT a grade and must never be copied into "
                        "one. Assign NOT A MEASUREMENT only after running the item and "
                        "seeing what it returns. A renderer that turns out to compute a "
                        "disparity is the most valuable thing you can report."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "batches": names,
                "count": len(names),
                "items": sum(len(b["items"]) for b in batches),
                "measuring": sum(b["measuring"] for b in batches),
                "unresolved": len(unresolved),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
