#!/usr/bin/env python3
"""Generate the data behind the two interactive demos on the docs landing page.

WHY A GENERATOR. The landing page shows a live-looking run (the hero) and a
threshold explorer. Neither may show a number the library did not produce: a
fairness library whose own front page invents figures has lost the argument
before it starts. So every value either demo displays is computed HERE, by
vfairness, on a real public dataset, and written to
docs/site/data/landing-demos.json. The browser only looks values up; it does
no fairness arithmetic of its own.

THE HERO CODE IS EXECUTED, NOT PARAPHRASED. ``HERO_CODE`` is the exact text the
page types out, and it is run with ``exec`` below, so the printed lines the page
shows are the stdout of that very code. Editing the snippet changes the output
on the next run; the two cannot drift apart.

The dataset is the UCI Adult test split with a logistic-regression model's
scores, the same file published on the Test Objects page.

Usage:
    python scripts/build_landing_demos.py           # write the JSON
    python scripts/build_landing_demos.py --check   # exit 1 if the JSON is stale
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"
DATA = SITE / "test-objects" / "data" / "predictive" / "adult_test_with_predictions.csv"
OUT = SITE / "data" / "landing-demos.json"

HERO_CODE = """import pandas as pd
from vfairness import FairnessAnalyzer

df = pd.read_csv("adult_test_with_predictions.csv")
analyzer = FairnessAnalyzer(
    y_true=df.y_true, y_pred=df.y_pred,
    sensitive_attr=df.race + " / " + df.sex,
    min_group_size=50,
)
report = analyzer.get_report()
print(report["assessment"]["summary"])"""

# Thresholds the explorer offers. A coarse grid for the two-slider mode keeps
# the file small; the shared-threshold mode is finer.
UNIFORM_GRID = [round(x, 3) for x in np.arange(0.05, 0.951, 0.025)]
SPLIT_GRID = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]


def _f(x):
    """JSON-safe float: NaN and inf become None (could not check), never 0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 4)


def _hero(df: pd.DataFrame) -> dict:
    from vfairness import FairnessAnalyzer  # noqa: F401  (the snippet imports it too)

    cwd = os.getcwd()
    os.chdir(DATA.parent)  # the snippet reads the CSV by its bare file name
    buf = io.StringIO()
    scope: dict = {}
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
            warnings.simplefilter("ignore")
            exec(compile(HERO_CODE, "<hero>", "exec"), scope)  # noqa: S102
    finally:
        os.chdir(cwd)
    report = scope["report"]
    info = report["data_info"]
    stats = report["group_stats"]
    sizes = info["group_sizes"]
    withheld = set(info.get("invalid_groups") or [])
    groups = []
    for name in sorted(sizes, key=lambda g: -int(sizes[g])):
        row = {"name": name, "n": int(sizes[name])}
        if name in withheld or name not in stats:
            row["state"] = "withheld"
        else:
            row["state"] = "measured"
            row["rate"] = _f(stats[name]["positive_rate"])
        groups.append(row)
    measured = [g for g in groups if g["state"] == "measured"]
    hi = max(measured, key=lambda g: g["rate"])
    lo = min(measured, key=lambda g: g["rate"])
    expl = (
        (report.get("explanations") or {})
        .get("metrics", {})
        .get("demographic_parity_difference", {})
    )
    a = report["assessment"]
    return {
        "code": HERO_CODE,
        "stdout": buf.getvalue().rstrip("\n").split("\n"),
        "groups": groups,
        "floor": int(scope["analyzer"].min_group_size),
        "gap": {
            "metric": "demographic_parity_difference",
            "value": _f(report["metrics"]["demographic_parity_difference"]),
            "threshold": _f(report["thresholds_used"]["demographic_parity_difference"]),
            "severity": expl.get("severity"),
            "high": hi["name"],
            "low": lo["name"],
        },
        "passed": len(a.get("passed_metrics") or []),
        "failed": len(a.get("failed_metrics") or []),
        "not_assessable": len(a.get("not_assessable_metrics") or []),
    }


def _evaluate(df: pd.DataFrame, pred: np.ndarray) -> dict:
    from vfairness import FairnessAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = FairnessAnalyzer(
            y_true=df.y_true.values, y_pred=pred, sensitive_attr=df.sex.values
        ).get_report(include_ci=False)
    gs = r["group_stats"]
    m = r["metrics"]
    a = r["assessment"]
    return {
        "dp": _f(m.get("demographic_parity_difference")),
        "eo": _f(m.get("equal_opportunity_difference")),
        "acc": _f((pred == df.y_true.values).mean()),
        "sel": {g: _f(s["positive_rate"]) for g, s in gs.items()},
        "tpr": {g: _f(s["tpr"]) for g, s in gs.items()},
        "passed": len(a.get("passed_metrics") or []),
        "failed": len(a.get("failed_metrics") or []),
        "na": len(a.get("not_assessable_metrics") or []),
    }


def _threshold(df: pd.DataFrame) -> dict:
    from vfairness.post_processing.threshold_optimization.optimizer import (
        GroupThresholdOptimizer,
    )

    p = df.y_prob.values
    female = (df.sex == "Female").values
    uniform = []
    for t in UNIFORM_GRID:
        row = _evaluate(df, (p >= t).astype(int))
        row["t"] = t
        uniform.append(row)
    split = {}
    for tf in SPLIT_GRID:
        for tm in SPLIT_GRID:
            pred = np.where(female, p >= tf, p >= tm).astype(int)
            split[f"{tf:.2f}|{tm:.2f}"] = _evaluate(df, pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GroupThresholdOptimizer(constraint="demographic_parity", tolerance=0.05).fit(
            y_true=df.y_true.values, y_prob=p, sensitive_attr=df.sex.values
        )
    res = opt.result_ if hasattr(opt, "result_") else opt.result
    th = {str(k): _f(v) for k, v in res.group_thresholds.items()}
    pred = np.where(female, p >= th["Female"], p >= th["Male"]).astype(int)
    fixed = _evaluate(df, pred)
    fixed.update(
        {
            "thresholds": th,
            "constraint": "demographic_parity",
            "tolerance": 0.05,
            "feasible": res.is_feasible,
        }
    )
    # Score distribution per group, split by the true outcome, as shares of the
    # group: what the explorer draws under the threshold line. Counting, not a
    # fairness metric, so it is plain numpy.
    edges = np.linspace(0.0, 1.0, 41)
    hist = {}
    for g in ["Female", "Male"]:
        m = (df.sex == g).values
        n = int(m.sum())
        pos, _ = np.histogram(p[m & (df.y_true.values == 1)], bins=edges)
        neg, _ = np.histogram(p[m & (df.y_true.values == 0)], bins=edges)
        hist[g] = {"pos": [round(c / n, 5) for c in pos], "neg": [round(c / n, 5) for c in neg]}
    return {
        "hist": {"edges": [round(e, 3) for e in edges], **hist},
        "attr": "sex",
        "groups": ["Female", "Male"],
        "n": {g: int((df.sex == g).sum()) for g in ["Female", "Male"]},
        "thresholds": {
            "dp": _f(0.1),
            "eo": _f(0.05),
        },
        "uniform": uniform,
        "split_grid": SPLIT_GRID,
        "split": split,
        "optimizer": fixed,
    }


def build() -> dict:
    import vfairness

    df = pd.read_csv(DATA)
    return {
        "$comment": "GENERATED by scripts/build_landing_demos.py. Every number is vfairness output on the dataset below; the browser only looks values up. Do not edit by hand.",
        "vfairness_version": vfairness.__version__,
        "dataset": {
            "name": "UCI Adult (census income), test split",
            "rows": int(len(df)),
            "model": "logistic regression",
            "file": "test-objects/data/predictive/adult_test_with_predictions.csv",
        },
        "hero": _hero(df),
        "threshold": _threshold(df),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("STALE: docs/site/data/landing-demos.json; run scripts/build_landing_demos.py")
            return 1
        print("landing-demos.json is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
