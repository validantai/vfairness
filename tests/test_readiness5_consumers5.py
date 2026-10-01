"""Readiness lane consumers5: two consumers that discarded a verdict the
producer had already computed honestly.

Every test below is either a REFUSAL PIN (the surface must answer
could-not-check, never a fabricated measurement) or an OVER-CORRECTION CONTROL
(a genuine measurement must still be reported, as the exact numbers it reports
today). The findings, all reproduced by execution on 2026-09-10:

1. ``simulate_threshold_change`` discarded ``alert_determined``, the column
   sitting in the very frame it reads. Two graded clean rows plus three rows
   nobody ever compared to a threshold (values 0.10 / 0.12 / 0.15) reported
   ``projected_alerts: 3`` and ``groups_impacted: ["never_compared"]``, a
   graded projection over records the alert rule never evaluates. It is a
   public export and also drives the Dash what-if panel and the standalone
   HTML one, so it reached a reader three ways.

1b. The Dash what-if callback formatted ``change_pct`` with ``:+.1f`` and
   compared it to 0. The producer already returns ``None`` there whenever the
   baseline is 0 alerts, so the callback raised TypeError instead of
   rendering. The standalone HTML's JavaScript fell back to
   ``projected - current``, which is ``0`` for two nulls in JavaScript, and
   then threw on ``groups_impacted.length``.

1c. Same file, same shape: the Dash health banner and the standalone HTML
   health badge formatted ``hs.score`` with ``:.0f``. ``compute_health_score``
   answers ``score=None, status="not_assessed"`` for a window it could not
   assess, so both raised TypeError. ``reports._narrate_health`` already
   handles that third state (C-04); these two consumers of the same object did
   not.

2. ``diagnose_local_attribution`` returned ``adversarial_flag: False`` when the
   Slack scaffolding probe never ran: an empty background produced exactly the
   verdict a real 120-row probe produces on a clean model, silently, with no
   note. The only could-not-check it did produce, ``adversarial_confidence:
   None``, has NO field in the ``XaiDiagnostics`` dataclass it claims to match
   and is read by nothing.

2b. Same function: an unmeasurable removal curve (nan) was clamped to 0.0 by
   ``max(0.0, nan)`` and then GRADED, so an explanation nobody could measure
   came back ``faithfulness: 0.0, faithfulness_grade: "weak"``.
"""

import dataclasses
import json
import math
import re
import shutil
import subprocess
import warnings
from datetime import datetime, timedelta

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
    diagnose_local_attribution,
    removal_curve_auc,
)
from vfairness.operations.reporting import interactive as interactive_mod
from vfairness.operations.reporting.interactive import (
    InteractiveDashboard,
    simulate_threshold_change,
)
from vfairness.operations.reporting.store import MetricsStore, StoredMetricRecord
from vfairness.xai.schemas import XaiDiagnostics


def _store(rows):
    """``rows`` are ``(value, alert, group)``; ``alert=None`` means the record
    was never compared to a threshold, which is what the store's
    ``alert_determined`` column reports as False."""
    store = MetricsStore()
    now = datetime.now()
    for value, alert, group in rows:
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(days=1),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=float(value),
                group=group,
                alert=alert,
            )
        )
    return store


# ---------------------------------------------------------------------------
# 1. simulate_threshold_change: the discarded alert determination
# ---------------------------------------------------------------------------


class TestThresholdSimulationRefusal:
    """VALUES FLIPPED 2026-09-10 (READINESS-5), assertions untouched.

    Every simulation fixture in this file was written against a simulator that
    projected an alert for each value BELOW the proposed threshold, whatever
    the metric was. demographic_parity is a violation MAGNITUDE, so that
    comparison was inverted: 0.10 against a bound of 0.5 is an excellent
    reading and was being projected as an alert, while 0.95 was not. The
    simulator now asks _metric_direction which way the metric passes, so these
    fixtures say 0.95 where they used to say 0.10, and the other way round.

    Not one asserted count, percentage, group list or record tally changed,
    and neither did what each test pins: which rows are eligible to be
    projected at all, never which comparison decides it.
    """

    def test_rows_nobody_compared_are_not_projected_alerts(self):
        """REFUSAL PIN. The three undetermined rows sit below the proposed
        threshold; none of them may be counted, and the group holding only
        them may not be named as impacted."""
        store = _store(
            [
                (0.10, False, "a"),
                (0.12, False, "a"),
                (0.95, None, "never_compared"),
                (0.97, None, "never_compared"),
                (0.99, None, "never_compared"),
            ]
        )
        res = simulate_threshold_change(store, "demographic_parity", 0.5)
        assert res["projected_alerts"] == 0
        assert res["current_alerts"] == 0
        assert res["change_abs"] == 0
        assert res["groups_impacted"] == []
        assert res["records_simulated"] == 2
        assert res["records_not_simulated"] == 3
        assert res["not_simulated_groups"] == ["never_compared"]

    def test_window_with_nothing_graded_is_could_not_check(self):
        """REFUSAL PIN. No baseline exists, so no count may be reported. An
        empty ``groups_impacted`` list would itself read as "no group is
        impacted", which is a measurement."""
        store = _store([(0.10, None, "a"), (0.12, None, "a"), (0.15, None, "b")])
        res = simulate_threshold_change(store, "demographic_parity", 0.5)
        assert res["current_alerts"] is None
        assert res["projected_alerts"] is None
        assert res["change_abs"] is None
        assert res["change_pct"] is None
        assert res["groups_impacted"] is None
        assert res["records_simulated"] == 0
        assert res["records_not_simulated"] == 3
        assert res["not_simulated_groups"] == ["a", "b"]

    def test_a_value_that_could_not_be_computed_is_not_a_clean_row(self):
        """REFUSAL PIN, sibling. ``nan < threshold`` is False in pandas, so an
        uncomputable value counted as "would not alert"."""
        store = _store([(0.95, False, "a"), (float("nan"), False, "b")])
        res = simulate_threshold_change(store, "demographic_parity", 0.5)
        assert res["projected_alerts"] == 1
        assert res["groups_impacted"] == ["a"]
        assert res["records_simulated"] == 1
        assert res["records_not_simulated"] == 1
        assert res["not_simulated_groups"] == ["b"]


class TestThresholdSimulationControl:
    """OVER-CORRECTION CONTROLS. A real simulation over rows that WERE graded
    must report exactly the numbers it reports today."""

    def test_zero_baseline_still_reports_the_absolute_increase(self):
        store = _store([(v, False, "a") for v in (0.95, 0.96, 0.94, 0.97)])
        res = simulate_threshold_change(store, "demographic_parity", 0.9)
        assert res["current_alerts"] == 0
        assert res["projected_alerts"] == 4
        assert res["change_abs"] == 4
        assert res["change_pct"] is None
        assert res["records_simulated"] == 4
        assert res["records_not_simulated"] == 0

    def test_zero_baseline_zero_projected_is_still_zero_percent(self):
        store = _store([(v, False, "a") for v in (0.55, 0.60)])
        res = simulate_threshold_change(store, "demographic_parity", 0.9)
        assert res["current_alerts"] == 0
        assert res["projected_alerts"] == 0
        assert res["change_abs"] == 0
        assert res["change_pct"] == 0.0

    def test_nonzero_baseline_still_keeps_the_percent(self):
        store = _store([(0.95, True, "a"), (0.96, False, "a")])
        res = simulate_threshold_change(store, "demographic_parity", 0.9)
        assert res["current_alerts"] == 1
        assert res["projected_alerts"] == 2
        assert res["change_abs"] == 1
        assert res["change_pct"] == 100.0
        assert res["groups_impacted"] == ["a"]
        assert res["records_simulated"] == 2

    def test_mixed_window_still_measures_the_graded_half(self):
        store = _store(
            [
                (0.95, True, "a"),
                (0.90, True, "a"),
                (0.10, False, "b"),
                (0.60, None, "never_compared"),
            ]
        )
        res = simulate_threshold_change(store, "demographic_parity", 0.5)
        assert res["current_alerts"] == 2
        assert res["projected_alerts"] == 2
        assert res["change_abs"] == 0
        assert res["change_pct"] == 0.0
        assert res["groups_impacted"] == ["a"]
        assert res["records_simulated"] == 3
        assert res["records_not_simulated"] == 1


# ---------------------------------------------------------------------------
# 1b. The two rendering surfaces the same result reaches
# ---------------------------------------------------------------------------


class _Node:
    """Stand-in for a dash html/dcc component: records children and kwargs."""

    def __init__(self, children=None, **kwargs):
        self.children = children
        self.kwargs = kwargs

    def text(self):
        parts = []
        child = self.children
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, _Node):
            parts.append(child.text())
        elif isinstance(child, (list, tuple)):
            for item in child:
                parts.append(item.text() if isinstance(item, _Node) else str(item))
        return " ".join(p for p in parts if p)


class _Namespace:
    def __getattr__(self, name):
        return _Node


class _RecordingApp:
    def __init__(self, *args, **kwargs):
        self.layout = None
        self.recorded = {}

    def callback(self, *args, **kwargs):
        def decorate(fn):
            self.recorded[fn.__name__] = fn
            return fn

        return decorate


class _FakeDash:
    Dash = _RecordingApp


def _dash_callbacks(monkeypatch, store):
    """Build the REAL Dash app with dash's own component classes stubbed, and
    hand back its registered callbacks. dash is an optional dependency and is
    not installed here, so the layout objects are doubles; the callback bodies
    executed below are the shipped ones."""
    monkeypatch.setattr(interactive_mod, "dash", _FakeDash(), raising=False)
    monkeypatch.setattr(interactive_mod, "html", _Namespace(), raising=False)
    monkeypatch.setattr(interactive_mod, "dcc", _Namespace(), raising=False)
    monkeypatch.setattr(interactive_mod, "Input", lambda *a, **k: ("in", a), raising=False)
    monkeypatch.setattr(interactive_mod, "Output", lambda *a, **k: ("out", a), raising=False)
    monkeypatch.setattr(interactive_mod, "_DASH_AVAILABLE", True, raising=False)
    return InteractiveDashboard(store).create_dash_app().recorded


class TestDashWhatIfPanel:
    def test_could_not_check_is_rendered_not_raised(self, monkeypatch):
        """REFUSAL PIN. The old body raised TypeError formatting None, so the
        panel showed nothing at all rather than the state it was handed."""
        store = _store([(0.10, None, "a"), (0.12, None, "a")])
        rendered = _dash_callbacks(monkeypatch, store)["update_whatif"](0.5, "demographic_parity")
        text = rendered.text()
        assert "Could not check" in text
        assert "no baseline" in text
        assert "Not simulated: 2 record(s)" in text
        assert "Change:" not in text
        assert "Projected alerts:" not in text

    def test_control_a_measured_simulation_still_renders_its_numbers(self, monkeypatch):
        """OVER-CORRECTION CONTROL on the same surface."""
        store = _store([(0.95, True, "a"), (0.96, False, "a")])
        rendered = _dash_callbacks(monkeypatch, store)["update_whatif"](0.9, "demographic_parity")
        text = rendered.text()
        assert "Current alerts: 1" in text
        assert "Projected alerts: 2" in text
        assert "+1 alert(s) (+100.0%)" in text
        assert "Groups impacted: a" in text
        assert "Could not check" not in text


def _app_script(html):
    """The last <script> block of the standalone export: the dashboard's own
    JavaScript, after the inlined plotly bundle."""
    script = html.rsplit("<script>", 1)[1].split("</script>")[0]
    assert script.lstrip().startswith("// Pre-rendered data"), "wrong <script> block extracted"
    return script


_JS_RUNNER = """
const fs = require('fs');
const code = fs.readFileSync(process.argv[2], 'utf8');
const nodes = {};
function node(id) {
  if (!nodes[id]) nodes[id] = {id: id, innerHTML: '', textContent: ''};
  return nodes[id];
}
global.document = {
  getElementById: node,
  querySelectorAll: () => [],
  addEventListener: () => {},
};
global.window = { dispatchEvent: () => {} };
global.Event = function () {};
global.Plotly = { react: () => {} };
eval(code);
updateWhatIf(process.argv[3]);
console.log(JSON.stringify({whatif: node('whatif-result').innerHTML}));
"""


def _run_shipped_js(tmp_path, html, threshold):
    script = tmp_path / "app.js"
    script.write_text(_app_script(html), encoding="utf-8")
    runner = tmp_path / "runner.js"
    runner.write_text(_JS_RUNNER, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(runner), str(script), threshold],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)["whatif"]


def _whatif_payload(html):
    return json.loads(re.search(r"var whatifData = (\{.*?\});\n", html, re.S).group(1))


class TestStandaloneHtmlWhatIfPanel:
    def test_payload_carries_the_third_state(self):
        """REFUSAL PIN, executed in Python: the JSON the page ships must carry
        nulls, not zeros, for a window nothing graded."""
        pytest.importorskip("plotly")
        store = _store([(0.10, None, "a"), (0.12, None, "a")])
        payload = _whatif_payload(InteractiveDashboard(store).to_standalone_html())
        entry = payload["0.5"]
        assert entry["current_alerts"] is None
        assert entry["projected_alerts"] is None
        assert entry["change_abs"] is None
        assert entry["groups_impacted"] is None
        assert entry["records_not_simulated"] == 2

    def test_control_payload_carries_a_measured_simulation(self):
        """OVER-CORRECTION CONTROL, same surface, exact numbers."""
        pytest.importorskip("plotly")
        store = _store([(0.95, True, "a"), (0.96, False, "a")])
        payload = _whatif_payload(InteractiveDashboard(store).to_standalone_html())
        entry = payload["0.9"]
        assert entry["current_alerts"] == 1
        assert entry["projected_alerts"] == 2
        assert entry["change_abs"] == 1
        assert entry["change_pct"] == 100.0
        assert entry["groups_impacted"] == ["a"]

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
    def test_shipped_javascript_renders_the_third_state(self, tmp_path):
        """REFUSAL PIN, executed: the page's own updateWhatIf, run in node.
        ``null - null`` is 0 in JavaScript, so the old delta fallback rendered
        "Change: 0 alert(s)" and then threw on ``groups_impacted.length``."""
        pytest.importorskip("plotly")
        store = _store([(0.10, None, "a"), (0.12, None, "a")])
        html = InteractiveDashboard(store).to_standalone_html()
        out = _run_shipped_js(tmp_path, html, "0.5")
        assert "Could not check" in out
        assert "Not simulated: 2 record(s)" in out
        assert "Change:" not in out
        assert "alert(s)" not in out

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
    def test_control_shipped_javascript_still_renders_measured_numbers(self, tmp_path):
        """OVER-CORRECTION CONTROL, executed in node."""
        pytest.importorskip("plotly")
        store = _store([(0.95, True, "a"), (0.96, False, "a")])
        html = InteractiveDashboard(store).to_standalone_html()
        out = _run_shipped_js(tmp_path, html, "0.9")
        assert "Current alerts: 1" in out
        assert "Projected alerts: 2" in out
        assert "+1 alert(s) (+100%)" in out
        assert "Groups impacted: a" in out
        assert "Could not check" not in out


# ---------------------------------------------------------------------------
# 1c. The unassessed health score, in both rendering surfaces
# ---------------------------------------------------------------------------


class TestUnassessedHealthScore:
    def test_standalone_html_says_not_assessed_instead_of_raising(self):
        """REFUSAL PIN. `{None:.0f}` raised TypeError, so the export died on
        exactly the store the what-if panel is most likely pointed at."""
        pytest.importorskip("plotly")
        store = _store([(0.10, None, "a"), (0.12, None, "a")])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            html = InteractiveDashboard(store).to_standalone_html()
        badge = re.search(r'<span class="health-badge">(.*?)</span>', html).group(1)
        assert badge == "not assessed"
        assert "/100" not in badge

    def test_control_a_real_health_score_still_renders_as_a_number(self):
        """OVER-CORRECTION CONTROL: a scored window keeps its badge."""
        pytest.importorskip("plotly")
        store = _store([(0.95, False, "a"), (0.96, False, "a")])
        html = InteractiveDashboard(store).to_standalone_html()
        badge = re.search(r'<span class="health-badge">(.*?)</span>', html).group(1)
        assert re.fullmatch(r"\d+/100", badge), badge

    def test_dash_banner_says_not_assessed_instead_of_raising(self, monkeypatch):
        """REFUSAL PIN on the Dash surface: same object, same third state."""
        store = _store([(0.10, None, "a"), (0.12, None, "a")])
        callbacks = _dash_callbacks(monkeypatch, store)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            banner = callbacks["update_charts"]("demographic_parity", "30d", ["a"])[0]
        assert "not assessed" in banner.text()

    def test_control_dash_banner_still_renders_a_real_score(self, monkeypatch):
        store = _store([(0.95, False, "a"), (0.96, False, "a")])
        callbacks = _dash_callbacks(monkeypatch, store)
        banner = callbacks["update_charts"]("demographic_parity", "30d", ["a"])[0]
        assert re.search(r"\d+/100", banner.text())
        assert "not assessed" not in banner.text()


# ---------------------------------------------------------------------------
# 2. diagnose_local_attribution: the fabricated adversarial verdict
# ---------------------------------------------------------------------------


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _clean_model(X):
    X = np.asarray(X, dtype=float)
    return _sigmoid(3.0 * X[:, 0] + 0.2 * X[:, 1])


def _scaffolded_model(X):
    """Slack-style scaffold: one answer on the data manifold, another off it."""
    X = np.asarray(X, dtype=float)
    return np.where(np.abs(X).max(axis=1) > 1.0, 1.0, 0.0)


def _one_output_model(arr):
    """A predict_fn that answers a single scalar for a batch. The probe feeds
    it 200 perturbations, so it raises: the probe's failure path."""
    row = np.asarray(arr, dtype=float).ravel()
    return np.array([0.25 * row[0] + 0.25 * row[1]])


class TestAdversarialProbeRefusal:
    @pytest.mark.parametrize(
        "background",
        [
            pytest.param(np.empty((0, 3)), id="empty-background"),
            pytest.param(np.zeros((3, 3)), id="fewer-rows-than-the-probe-needs"),
        ],
    )
    def test_a_probe_that_never_ran_is_not_a_negative_verdict(self, background):
        """REFUSAL PIN. False is the verdict "the probe ran and found no
        scaffolding". A probe that never ran must say so."""
        with pytest.warns(UserWarning, match="adversarial probe did not run"):
            out = diagnose_local_attribution(
                _clean_model, [2.0, 0.5, 0.5], [3.0, 0.2, 0.0], background, n_seeds=4
            )
        assert out["adversarial_flag"] is None
        assert out["adversarial_flag"] is not False
        assert out["adversarial_confidence"] is None
        assert out["adversarial_reason"].startswith("COULD NOT CHECK")
        assert any("adversarial probe did not run" in note for note in out["notes"])

    def test_a_probe_that_failed_is_not_a_negative_verdict(self):
        """REFUSAL PIN. The except branch appended a note and left the
        fabricated False standing."""
        with pytest.warns(UserWarning, match="adversarial probe did not run"):
            out = diagnose_local_attribution(
                _one_output_model, [1.0, 1.0], [2.0, 1.0], np.zeros((10, 2)), n_seeds=3
            )
        assert out["adversarial_flag"] is None
        assert out["adversarial_reason"].startswith("COULD NOT CHECK")
        json.dumps(out)  # still JSON-safe for the task envelope

    def test_could_not_check_survives_the_schema_it_claims_to_match(self):
        """REFUSAL PIN on the schema. ``adversarial_confidence`` is not a field
        of ``XaiDiagnostics``, so a could-not-check carried only there is
        dropped on the way to the reader. The state must ride in fields the
        dataclass actually has."""
        schema_fields = {f.name for f in dataclasses.fields(XaiDiagnostics)}
        assert "adversarial_confidence" not in schema_fields
        assert "faithfulness_grade" not in schema_fields

        with pytest.warns(UserWarning):
            out = diagnose_local_attribution(
                _clean_model, [2.0, 0.5, 0.5], [3.0, 0.2, 0.0], np.empty((0, 3)), n_seeds=4
            )
        carried = XaiDiagnostics(**{k: v for k, v in out.items() if k in schema_fields})
        assert carried.adversarial_flag is None
        assert carried.adversarial_reason.startswith("COULD NOT CHECK")
        assert any("adversarial probe did not run" in note for note in carried.notes)


class TestAdversarialProbeControl:
    """OVER-CORRECTION CONTROLS: a probe that DID run must still deliver its
    verdict, with the numbers it delivers today."""

    def test_a_scaffolded_model_is_still_flagged(self):
        background = np.random.default_rng(11).normal(0.0, 0.1, size=(120, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = diagnose_local_attribution(
                _scaffolded_model, [3.0, 3.0, 3.0], [1.0, 0.0, 0.0], background, n_seeds=6
            )
        assert out["adversarial_flag"] is True
        assert out["adversarial_confidence"] == 1.0
        assert "fired on 6/6 seeds" in out["adversarial_reason"]
        assert out["notes"] == []

    def test_a_clean_model_is_still_cleared(self):
        background = np.random.default_rng(1).normal(0, 1, size=(120, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = diagnose_local_attribution(
                _clean_model, [1.0, 0.0, 0.0], [3.0, 0.2, 0.0], background, n_seeds=6
            )
        assert out["adversarial_flag"] is False
        assert out["adversarial_confidence"] == 1.0
        assert out["adversarial_reason"] is None
        assert out["notes"] == []


class TestFaithfulnessGrade:
    def test_an_uncomputable_removal_curve_is_not_a_weak_explanation(self):
        """REFUSAL PIN, sibling. ``max(0.0, nan)`` returns 0.0, which grades
        "weak": a real verdict on a curve that was never computed."""
        auc = removal_curve_auc(
            _clean_model,
            np.array([2.0, 0.5, 0.5]),
            np.array([3.0, 0.2, 0.0]),
            np.full(3, np.nan),
        )
        assert math.isnan(auc)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = diagnose_local_attribution(
                _clean_model, [2.0, 0.5, 0.5], [3.0, 0.2, 0.0], np.empty((0, 3)), n_seeds=4
            )
        assert out["faithfulness"] is None
        assert out["faithfulness_grade"] is None
        assert any("faithfulness unavailable" in note for note in out["notes"])

    def test_control_a_real_removal_curve_is_still_measured_and_graded(self):
        """OVER-CORRECTION CONTROL, exact numbers on a deterministic model."""
        assert (
            removal_curve_auc(
                lambda arr: np.array([0.25 * arr.ravel()[0] + 0.25 * arr.ravel()[1]]),
                np.array([1.0, 1.0]),
                np.array([2.0, 1.0]),
                np.zeros(2),
            )
            == 0.5
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = diagnose_local_attribution(
                _one_output_model,
                [1.0, 1.0],
                [2.0, 1.0],
                np.zeros((10, 2)),
                reruns=[[2.0, 1.0], [2.02, 1.01]],
                n_steps=2,
                n_seeds=3,
            )
        assert out["faithfulness"] == 0.5
        assert out["faithfulness_grade"] == "strong"
        assert out["stability"] == 0.0075

    def test_control_the_zero_and_one_clamps_still_hold(self):
        """OVER-CORRECTION CONTROL: a MEASURED 0.0 is still 0.0 and still
        grades "weak"; an overshoot is still clamped to 1.0.

        The attribution vector in the first half was `[1.0, 1.0]` until
        2026-09-27. That is FULLY TIED and was never the subject: this test is
        about the CLAMPS, and any two attributions would have done. A tied vector
        defines no masking order, so removal_curve_auc now refuses it with a
        warning instead of scoring the identity permutation, and the 0.0 here has
        to come from the no-effect model rather than from the tie. Measured:
        [1.0, 1.0] gives nan plus that warning, [2.0, 1.0] gives 0.0 in silence.
        """
        x = np.array([1.0, 1.0])
        assert (
            removal_curve_auc(lambda arr: np.array([0.5]), x, np.array([2.0, 1.0]), np.zeros(2))
            == 0.0
        )

        def _overshoot(arr):
            return np.array([0.05 if arr.ravel()[0] == 1.0 else 0.95])

        assert (
            removal_curve_auc(
                _overshoot, np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), np.zeros(3)
            )
            == 1.0
        )
