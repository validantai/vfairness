"""The landing-page demos may only show numbers vfairness produced.

docs/site/data/landing-demos.json feeds the live hero and the threshold
explorer, and the page only looks values up in it. These tests re-run the
library and require the committed file to agree, so a hand edit, a stale file
or a page whose code no longer matches the run turns the build red.

What it does not establish: it samples the explorer (a few thresholds and the
optimizer), it does not recompute all 398 cells; `build_landing_demos.py
--check` does that and is slower.
"""

import html
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_landing_demos.py"
DATA = ROOT / "docs" / "site" / "data" / "landing-demos.json"
PAGE = ROOT / "docs" / "site" / "index.html"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("build_landing_demos", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def committed():
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df(gen):
    return pd.read_csv(gen.DATA)


def test_the_page_types_the_code_that_was_run(gen, committed):
    page = PAGE.read_text(encoding="utf-8")
    m = re.search(r'<pre class="hero-run__code"><code>(.*?)</code></pre>', page, re.S)
    assert m, "the hero code block is missing from the landing page"
    assert html.unescape(m.group(1)) == gen.HERO_CODE
    assert committed["hero"]["code"] == gen.HERO_CODE


def test_the_hero_output_is_what_the_code_prints(gen, committed, df):
    fresh = gen._hero(df)
    assert fresh["stdout"] == committed["hero"]["stdout"]
    assert fresh["groups"] == committed["hero"]["groups"]
    assert fresh["gap"] == committed["hero"]["gap"]
    # The point of the hero: one group is withheld and named, not drawn.
    withheld = [g for g in committed["hero"]["groups"] if g["state"] == "withheld"]
    assert withheld and all("rate" not in g for g in withheld)
    assert withheld[0]["name"] in committed["hero"]["stdout"][0]


@pytest.mark.parametrize("t", [0.2, 0.5, 0.8])
def test_explorer_cells_are_library_output(gen, committed, df, t):
    row = next(r for r in committed["threshold"]["uniform"] if abs(r["t"] - t) < 1e-9)
    fresh = gen._evaluate(df, (df.y_prob.values >= t).astype(int))
    for k in ("dp", "eo", "acc", "sel", "tpr", "passed", "failed", "na"):
        assert fresh[k] == row[k], (t, k)


def test_the_optimizer_result_is_the_optimizers(gen, committed, df):
    from vfairness.post_processing.threshold_optimization.optimizer import GroupThresholdOptimizer

    opt = GroupThresholdOptimizer(constraint="demographic_parity", tolerance=0.05).fit(
        y_true=df.y_true.values, y_prob=df.y_prob.values, sensitive_attr=df.sex.values
    )
    th = {str(k): gen._f(v) for k, v in opt.result_.group_thresholds.items()}
    assert th == committed["threshold"]["optimizer"]["thresholds"]
    female = (df.sex == "Female").values
    pred = np.where(female, df.y_prob.values >= th["Female"], df.y_prob.values >= th["Male"])
    fresh = gen._evaluate(df, pred.astype(int))
    for k in ("dp", "eo", "acc", "passed", "failed"):
        assert fresh[k] == committed["threshold"]["optimizer"][k], k


def test_the_file_names_the_library_version_that_made_it(committed):
    import vfairness

    assert committed["vfairness_version"] == vfairness.__version__
