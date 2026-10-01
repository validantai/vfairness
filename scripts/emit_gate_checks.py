#!/usr/bin/env python3
"""Run the two beta-gate checks and write their result to docs/gate-checks.json.

Check 1, correct on normal data: every measuring capability that has an outside
reference implementation must agree with it on clean data
(tests/test_reference_parity_registry.py holds the rows).

Check 2, honest on broken data: every measuring capability must refuse when its
input holds nothing measurable and measure when it does
(scripts/broken_data_check.py).

The page renders this file; nobody types its figures. Counted per REGISTRY ENTRY,
so the scope is the registry's own count of measuring capabilities. A check that
could not run (a reference library missing) is written as could-not-check, never
as a pass.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "docs" / "gate-checks.json"


def _entries():
    from vfairness._registry import CAPABILITY_REGISTRY

    return [e["name"] for e in CAPABILITY_REGISTRY.values() if e["pipeline_stage"] == "evaluation"]


def check1() -> dict:
    path = ROOT / "tests" / "test_reference_parity_registry.py"
    spec = importlib.util.spec_from_file_location("_parity_rows", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except BaseException as exc:  # noqa: BLE001 - pytest.skip raises a BaseException
        return {"state": "could_not_check", "reason": f"{type(exc).__name__}: {exc}"[:200]}
    agree, differ, rows = [], [], []
    for name, ours, ref, source, tol in mod.ROWS:
        ok = True
        for fx in mod.FIXTURES.values():
            try:
                mine, theirs = ours(fx), ref(fx)
                import numpy as np

                nudged = np.asarray(mine, dtype=float) + max(3 * tol, 1e-3)
                ok &= mod._agree(mine, theirs, tol) and not mod._agree(nudged, theirs, tol)
            except Exception:  # noqa: BLE001
                ok = False
        (agree if ok else differ).append(name)
        rows.append({"name": name, "reference": source, "tolerance": tol, "agree": ok})
    entries = _entries()
    with_ref = {r[0] for r in mod.ROWS}
    return {
        "state": "ran",
        "entries": len(entries),
        "with_reference": sum(1 for n in entries if n in with_ref),
        "agree": sum(1 for n in entries if n in agree),
        "differ": sorted(differ),
        "no_reference": sum(1 for n in entries if n not in with_ref),
        "rows": rows,
        "no_reference_names": sorted({n for n in entries if n not in with_ref}),
        "fixtures": sorted(mod.FIXTURES),
    }


def check2() -> dict:
    import broken_data_check as B

    res = B.run()
    caps = res["capabilities"]
    entries = _entries()
    count = {
        v: sum(1 for n in entries if caps[n]["verdict"] == v)
        for v in ("PASS", "FABRICATES", "OVER-REFUSES", "NOT A MEASUREMENT", "NOT COVERED")
    }
    return {
        "state": "ran",
        "entries": len(entries),
        "pass": count["PASS"],
        "fabricates": count["FABRICATES"],
        "over_refuses": count["OVER-REFUSES"],
        "not_a_measurement": count["NOT A MEASUREMENT"],
        "not_covered": count["NOT COVERED"],
        "failing": sorted(
            n for n in set(entries) if caps[n]["verdict"] in ("FABRICATES", "OVER-REFUSES")
        ),
        "not_covered_names": sorted(n for n in set(entries) if caps[n]["verdict"] == "NOT COVERED"),
        "verdicts": {n: caps[n]["verdict"] for n in sorted(set(entries))},
    }


def main() -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    c1, c2 = check1(), check2()
    gate_open = (
        c1.get("state") == "ran"
        and not c1["differ"]
        and c1["agree"] == c1["with_reference"]
        and c2["pass"] + c2["not_a_measurement"] == c2["entries"]
    )
    data = {
        "date": _dt.date.today().isoformat(),
        "check1_correct_on_normal_data": c1,
        "check2_honest_on_broken_data": c2,
        "gate_passes": gate_open,
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps({k: v for k, v in data.items()}, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
