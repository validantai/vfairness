"""Every shipped notebook must still run against the current library.

The notebooks are the project's teaching artifact: thirteen of them, 415 code
cells, and they are what a newcomer opens to see how the library actually works.
Nothing checked them. Measured 2026-09-07, FOUR of the thirteen were broken by
API changes the notebooks were never updated for:

  2_training_demo      `.difference` on a metric result. The module-level metric
                       functions return a plain float now; per-group rates moved
                       to the analyzer report's `group_stats`. Also
                       GroupThresholdOptimizer.thresholds_ -> result_.group_thresholds,
                       PredictionReweighter.adjustment_factors_ -> adjustments_,
                       RejectionOptionClassifier's constructor, and
                       ReweightingAnalyzer.full_analysis's parameters.
  3_calibration_demo   the recommendation cache became a per-context DICT, so
                       assigning None to it broke the next lookup; and
                       CalibrationDisparityResult carries `group_ece`, not
                       `group_metrics`.
  6_reporting_demo     formatted `change_pct` with a float spec when it is None.
                       The library is RIGHT there: a percent change from zero
                       alerts is undefined, and returning 0.0 would hide a real
                       0 -> N increase. The notebook was wrong.
  1_feature_engineering_demo  referenced two variable names that never existed.

A broken teaching notebook is worse than a broken test. A test failure is seen by
someone who can fix it; a notebook failure is seen by someone deciding whether to
trust the library at all, and their first experience is a traceback.

These are marked `slow` so `pytest -m "not slow"` stays fast, but they run by
default and in CI.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

NOTEBOOKS = sorted((pathlib.Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


# Executed by a FRESH interpreter, one per notebook. Kept flush-left and passed
# with `python -c`: an indented heredoc plus a line continuation is what broke the
# first attempt with IndentationError on its own second line.
_RUNNER = """
import json, sys, warnings, io, contextlib
import matplotlib
matplotlib.use("Agg")
cells = json.load(open(sys.argv[1]))
ns = {"__name__": "__main__"}
for number, code in enumerate(cells, 1):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                exec(compile(code, "cell%d" % number, "exec"), ns)
    except Exception as exc:
        first = next((l for l in code.split(chr(10)) if l.strip()), "")
        print(json.dumps({"cell": number, "type": type(exc).__name__,
                          "msg": str(exc).split(chr(10))[0][:200],
                          "first": first[:120]}))
        sys.exit(1)
sys.exit(0)
"""


def _code_cells(path: pathlib.Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Jupyter magics and shell escapes are not library code and cannot run
        # under exec(); stripping them is what a reader's kernel effectively
        # does for `%matplotlib inline`.
        code = "\n".join(
            line for line in src.split("\n") if not line.strip().startswith(("%", "!"))
        )
        if code.strip():
            out.append(code)
    return out


def test_notebooks_are_present():
    """NON-VACUITY. An empty glob would make every test below pass silently."""
    assert len(NOTEBOOKS) >= 13, f"only {len(NOTEBOOKS)} notebooks found under notebooks/"


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs_end_to_end(path: pathlib.Path, tmp_path: pathlib.Path):
    """Run every code cell in order in one namespace, as a kernel would.

    IN A SUBPROCESS, and that is not incidental. The first version of this file
    exec'd the cells in the pytest process and turned two unrelated tests red in
    the full suite while passing in isolation: the notebooks call process-global
    setters (branding, matplotlib state, module caches), and a notebook is
    entitled to do that. A kernel gets a fresh interpreter, so this gets one too.
    The cost is about two seconds of startup per notebook.
    """
    pytest.importorskip("matplotlib", reason="notebooks plot; matplotlib is a viz extra")

    cells = _code_cells(path)
    assert cells, f"{path.name} has no runnable code cells"

    runner = _RUNNER

    # READINESS-6, 2026-09-10. This wrote `.{stem}.cells.json` INTO
    # `notebooks/`, i.e. into the source tree, and removed it in a `finally`.
    # That covers a normal exit and an exception and does not cover a hard
    # interrupt, and the pattern was not in `.gitignore` either, so an
    # interrupted run left an untracked dotfile in the package directory that
    # the next `git add -A` would commit.
    #
    # The runner takes an ABSOLUTE path (`json.load(open(sys.argv[1]))`), and
    # the subprocess `cwd` is set separately, so this file never needed to be
    # next to the notebook at all. pytest's `tmp_path` removes the class rather
    # than the symptom: nothing is ever written into the repo, no cleanup can be
    # skipped, and each parametrised run gets its own directory.
    tmp = tmp_path / f"{path.stem}.cells.json"
    tmp.write_text(json.dumps(cells), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", runner, str(tmp)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=path.parent.parent,
    )

    if proc.returncode != 0:
        detail = proc.stdout.strip().split("\n")[-1] if proc.stdout.strip() else ""
        try:
            info = json.loads(detail)
            where = (
                f"code cell {info['cell']} of {len(cells)} raised "
                f"{info['type']}: {info['msg']}\n  cell begins: {info['first']}"
            )
        except (ValueError, KeyError):
            where = f"exited {proc.returncode}\n{proc.stderr[-1500:]}"
        pytest.fail(
            f"{path.name} {where}\n"
            f"  This notebook is shipped as a worked example. Fix the notebook "
            f"against the current API, or fix the API."
        )
