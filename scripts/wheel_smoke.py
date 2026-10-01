"""Clean-room smoke test for the built wheel (VB-REL-3).

Exercises the public API of the INSTALLED package. Run it from OUTSIDE the
source tree (for example from a scratch directory) so ``import vfairness``
resolves to the installed wheel in site-packages, never to ``./src``. This
catches packaging defects that source-tree tests cannot: a missing data file,
a wrong entry point, a package path that was not shipped, or a missing runtime
dependency.

Uses only the core runtime dependencies (numpy, pandas, scipy, scikit-learn),
so it needs no optional extras. The assertions mirror the doctested snippets in
``docs/examples.md``.

Usage:
    pip install dist/vfairness-*.whl
    cd "$(mktemp -d)" && python /path/to/scripts/wheel_smoke.py
"""

import numpy as np

import vfairness as vf
from vfairness import FairnessAnalyzer


def main() -> None:
    print("vfairness", vf.__version__)

    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))

    # Core metric: a non-negative demographic-parity spread in [0, 1].
    dp = vf.demographic_parity_difference(y_true, y_pred, groups)
    assert 0.0 <= float(dp) <= 1.0, f"demographic_parity_difference out of range: {dp}"

    # Unified analyzer and its structured report.
    analyzer = FairnessAnalyzer(y_true, y_pred, groups)
    assert analyzer.task_type == "classification", analyzer.task_type
    report = analyzer.get_report(include_ci=False)
    assert "demographic_parity_difference" in report["metrics"]
    assert report["data_info"]["n_samples"] == 200
    assert "methodology_version" in report, "report is missing methodology_version stamp"

    # Group inspection.
    assert sorted(analyzer.groups) == ["A", "B"], analyzer.groups

    print("SMOKE OK: import, metric, analyzer, report, and groups all work from the wheel")


if __name__ == "__main__":
    main()
