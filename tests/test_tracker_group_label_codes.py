"""Group names are hidden by default when vfairness writes to an experiment tracker.

Decided 2026-10-24: ``log_fairness_to_mlflow`` (and ``log_fairness_to_wandb``, for
parity) code every group name by default, because the names of protected groups
are sensitive and a tracker run is read by many people. The code is a KEYED hash
(HMAC-SHA256 under a team secret), never a plain hash: "Female" is found again by
hashing a short list of likely names.

What this file holds the writers to, each as an executed check:

* the code is the published formula, stable under one key, different under
  another, identical for a str key and its UTF-8 bytes, and a shared code is
  refused rather than merging two groups;
* no key (neither ``label_key`` nor ``VFAIRNESS_GROUP_LABEL_KEY``) refuses with
  NOTHING written: no metric, param, tag or artifact;
* in coded mode no raw group name appears ANYWHERE in what is written (metric
  keys, params, tags, the artifact), checked with distinctive names, and a
  numeric-looking label ("1") never appears in a group-carrying position;
* the order of groups does not leak: lists are in code order and an effect-size
  pair restated for the reversed direction equals what the producer itself
  computes for that direction;
* ``group_labels="names"`` writes exactly what the writers wrote before coding
  existed (a snapshot taken from that code), plus one tag;
* against a REAL MLflow (a SQLite store in a temporary folder), the run read back
  through ``mlflow.get_run`` and its downloaded artifact holds codes, not names.
  That test needs the ``mlops`` extra and is skipped without it.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import hmac
import json
import re
import sys
import types
import warnings

import numpy as np
import pytest

import vfairness
from vfairness import FairnessAnalyzer, _group_codes
from vfairness._group_codes import ENV_VAR, group_label_codes, key_id, label_code
from vfairness.evaluation.vfairness_metrics.classification import compute_effect_sizes
from vfairness.evaluation.vfairness_metrics.integrations import (
    auto_log_fairness,
    log_fairness_to_mlflow,
    log_fairness_to_wandb,
)
from vfairness.evaluation.vfairness_metrics.regression import compute_regression_effect_sizes

KEY = "test key: a team secret"
CODE = re.compile(r"^g_[0-9a-f]{12}$")
PAIR = re.compile(r"^g_[0-9a-f]{12}_vs_g_[0-9a-f]{12}$")

# Distinctive on purpose, so a substring search over everything written cannot be
# fooled by a coincidence. "Smallvillean" is below min_group_size, so it lands in
# invalid_groups, the excluded-levels note and the insufficient-evidence records.
DISTINCTIVE = ("Zorblaxian", "Quibbletonian", "Mxyzptlkian", "Smallvillean")


# Fake trackers that record EVERY write, including the artifact's bytes.


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class _FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.metrics: dict = {}
        self.params: dict = {}
        self.tags: dict = {}
        self.artifacts: list = []
        self.calls: list = []

    def active_run(self):
        return object()

    def log_metric(self, key, value):
        self.calls.append("metric")
        self.metrics[key] = value

    def log_param(self, key, value):
        self.calls.append("param")
        self.params[key] = value

    def set_tag(self, key, value):
        self.calls.append("tag")
        self.tags[key] = value

    def log_artifact(self, path, name):
        self.calls.append("artifact")
        self.artifacts.append((name, _read(path)))


class _FakeArtifact:
    def __init__(self, name, type):  # noqa: A002 - mirrors wandb.Artifact
        self.name = name
        self.type = type
        self.files: list = []

    def add_file(self, path, name):
        self.files.append((name, _read(path)))


class _Summary(dict):
    def __init__(self, calls):
        super().__init__()
        self._calls = calls

    def __setitem__(self, key, value):
        self._calls.append("summary")
        super().__setitem__(key, value)


class _FakeWandbRun:
    def __init__(self, calls):
        self.summary = _Summary(calls)


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.calls: list = []
        self.run = _FakeWandbRun(self.calls)
        self.logged: dict = {}
        self.artifacts: list = []
        self.Artifact = _FakeArtifact

    def log(self, data):
        self.calls.append("log")
        self.logged.update(data)

    def log_artifact(self, artifact):
        self.calls.append("artifact")
        self.artifacts.append(artifact)


@pytest.fixture
def fake_mlflow(monkeypatch):
    module = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return module


@pytest.fixture
def fake_wandb(monkeypatch):
    module = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


@pytest.fixture
def no_env_key(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _labels_column(names_and_sizes):
    return np.array([name for name, size in names_and_sizes for _ in range(size)])


def _analyzer(task="classification", names=DISTINCTIVE, sizes=(250, 250, 90, 10), seed=0):
    rng = np.random.default_rng(seed)
    attr = _labels_column(list(zip(names, sizes)))
    n = len(attr)
    if task == "regression":
        y_true = rng.normal(size=n)
        return FairnessAnalyzer(
            y_true, y_true + rng.normal(scale=0.3, size=n), attr, task_type="regression"
        )
    return FairnessAnalyzer(
        rng.integers(0, 2, n), rng.integers(0, 2, n), attr, y_prob=rng.random(n)
    )


@functools.lru_cache(maxsize=None)
def _built_report(task, names, sizes):
    analyzer = _analyzer(task, names=names, sizes=sizes)
    return _quiet(
        analyzer.get_report,
        include_ci=True,
        n_bootstrap=30,
        random_state=0,
        include_explanations=True,
    )


def _full_report(task="classification", names=DISTINCTIVE, sizes=(250, 250, 90, 10)):
    """A real report with every section: intervals, effect sizes, explanations.

    Built once per shape and handed out as a deep copy, so a test that edits its
    report cannot change another test's.
    """
    return copy.deepcopy(_built_report(task, tuple(names), tuple(sizes)))


def _everything_written_mlflow(fake):
    parts = list(fake.metrics) + list(fake.params) + list(fake.tags)
    parts += [str(v) for v in fake.params.values()] + [str(v) for v in fake.tags.values()]
    parts += [content for _name, content in fake.artifacts]
    parts += [name for name, _content in fake.artifacts]
    return parts


def _everything_written_wandb(fake):
    parts = list(fake.logged) + [str(v) for v in fake.logged.values()]
    parts += list(fake.run.summary) + [str(v) for v in fake.run.summary.values()]
    for artifact in fake.artifacts:
        parts.append(artifact.name)
        parts += [content for _name, content in artifact.files]
    return parts


def _group_positions(artifact):
    """Every string that sits in a group-carrying position of a coded artifact."""
    found = []
    found += list(artifact.get("group_stats", {}))
    found += list(artifact.get("residual_bias", {}))
    info = artifact.get("data_info", {})
    for field in ("valid_groups", "invalid_groups", "groups_dropped"):
        found += list(info.get(field, []))
    found += list(info.get("group_sizes", {}))
    for record in artifact.get("assessment", {}).get("insufficient_evidence_groups", []):
        found.append(record["group"])
    validation = artifact.get("statistical_validation", {})
    found += list(validation.get("per_group_size_recommendation", {}))
    found += list(validation.get("per_group_methods", {}))
    for entry in artifact.get("metrics_with_ci", {}).values():
        if isinstance(entry, dict):
            found += list(entry.get("metadata", {}).get("group_sizes", {}))
    return found


# 1. The code itself.


def test_the_code_is_the_published_formula():
    expected = (
        "g_"
        + hmac.new(
            KEY.encode("utf-8"), b"vfairness.group-label.v1\x00" + b"Female", hashlib.sha256
        ).hexdigest()[:12]
    )
    assert label_code(KEY.encode("utf-8"), "Female") == expected
    assert group_label_codes(["Female"], label_key=KEY) == {"Female": expected}
    assert CODE.match(expected)


def test_the_key_id_is_the_published_formula():
    expected = hmac.new(
        KEY.encode("utf-8"), b"vfairness.group-label.key-id.v1", hashlib.sha256
    ).hexdigest()[:8]
    assert key_id(KEY.encode("utf-8")) == expected


def test_codes_are_stable_under_one_key_and_differ_under_another():
    first = group_label_codes(["Female", "Male", 1], label_key=KEY)
    again = group_label_codes(["Male", "Female", "1"], label_key=KEY)
    other = group_label_codes(["Female", "Male", 1], label_key="another secret")
    assert first == again
    assert set(first) == {"Female", "Male", "1"}
    assert all(first[name] != other[name] for name in first)
    assert len(set(first.values())) == 3


def test_a_str_key_and_its_utf8_bytes_give_the_same_codes():
    secret = "clé secrète"
    assert group_label_codes(["Female"], label_key=secret) == group_label_codes(
        ["Female"], label_key=secret.encode("utf-8")
    )


def test_the_environment_variable_is_read_and_label_key_wins(monkeypatch):
    monkeypatch.setenv(ENV_VAR, KEY)
    assert group_label_codes(["Female"]) == group_label_codes(["Female"], label_key=KEY)
    assert group_label_codes(["Female"], label_key="other") != group_label_codes(["Female"])


def test_no_key_anywhere_is_refused_with_the_fix(no_env_key):
    with pytest.raises(ValueError, match=ENV_VAR) as raised:
        group_label_codes(["Female"])
    message = str(raised.value)
    assert "label_key" in message and "group_labels='names'" in message


@pytest.mark.parametrize("bad", ["", b""])
def test_an_empty_key_is_refused(no_env_key, bad):
    with pytest.raises(ValueError, match="empty"):
        group_label_codes(["Female"], label_key=bad)


def test_a_key_that_is_not_text_is_refused():
    with pytest.raises(TypeError, match="label_key"):
        group_label_codes(["Female"], label_key=12345)


def test_a_shared_code_is_refused_never_merged(monkeypatch):
    monkeypatch.setattr(_group_codes, "label_code", lambda key, label: "g_000000000000")
    with pytest.raises(ValueError, match="same code"):
        group_label_codes(["Female", "Male"], label_key=KEY)


# 2. Refusals write NOTHING.


def test_mlflow_without_a_key_refuses_and_writes_nothing(fake_mlflow, no_env_key):
    report = _full_report()
    with pytest.raises(ValueError, match=ENV_VAR):
        log_fairness_to_mlflow(report)
    assert fake_mlflow.calls == []
    assert not (fake_mlflow.metrics or fake_mlflow.params or fake_mlflow.tags)
    assert fake_mlflow.artifacts == []


def test_wandb_without_a_key_refuses_and_writes_nothing(fake_wandb, no_env_key):
    report = _full_report()
    with pytest.raises(ValueError, match=ENV_VAR):
        log_fairness_to_wandb(report)
    assert fake_wandb.calls == []
    assert not (fake_wandb.logged or fake_wandb.run.summary or fake_wandb.artifacts)


def test_a_shared_code_in_the_writer_writes_nothing(fake_mlflow, monkeypatch):
    monkeypatch.setattr(_group_codes, "label_code", lambda key, label: "g_000000000000")
    with pytest.raises(ValueError, match="same code"):
        log_fairness_to_mlflow(_full_report(), label_key=KEY)
    assert fake_mlflow.calls == []


@pytest.mark.parametrize("bad", ["coded ", "hashed", None, True])
def test_an_unknown_mode_is_refused(fake_mlflow, fake_wandb, bad):
    report = _full_report()
    with pytest.raises(ValueError, match="group_labels"):
        log_fairness_to_mlflow(report, group_labels=bad, label_key=KEY)
    with pytest.raises(ValueError, match="group_labels"):
        log_fairness_to_wandb(report, group_labels=bad, label_key=KEY)
    with pytest.raises(ValueError, match="group_labels"):
        auto_log_fairness(group_labels=bad)
    assert fake_mlflow.calls == [] and fake_wandb.calls == []


def _evaluate(y_true, y_pred, attr):
    return y_pred, y_true, attr


def test_auto_log_without_a_key_warns_and_writes_nothing(fake_mlflow, no_env_key):
    analyzer = _analyzer()
    decorated = auto_log_fairness(
        backend="mlflow", thresholds={"demographic_parity_difference": 0.1}
    )(_evaluate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decorated(analyzer.y_true, analyzer.y_pred, analyzer.sensitive_attr)
    assert any(ENV_VAR in str(w.message) for w in caught)
    # Not even the thresholds: the key is resolved before anything is recorded.
    assert fake_mlflow.calls == []


def test_auto_log_codes_by_default(fake_mlflow, monkeypatch):
    monkeypatch.setenv(ENV_VAR, KEY)
    analyzer = _analyzer()
    decorated = auto_log_fairness(
        backend="mlflow", thresholds={"demographic_parity_difference": 0.1}
    )(_evaluate)
    _quiet(decorated, analyzer.y_true, analyzer.y_pred, analyzer.sensitive_attr)
    written = " ".join(_everything_written_mlflow(fake_mlflow))
    assert fake_mlflow.tags["fairness.group_labels"] == "coded"
    assert fake_mlflow.params["fairness.threshold.demographic_parity_difference"] == 0.1
    assert any(".group.g_" in name for name in fake_mlflow.metrics)
    for name in DISTINCTIVE:
        assert name not in written


def test_auto_log_names_mode_needs_no_key(fake_wandb, no_env_key):
    analyzer = _analyzer()
    decorated = auto_log_fairness(backend="wandb", group_labels="names")(_evaluate)
    _quiet(decorated, analyzer.y_true, analyzer.y_pred, analyzer.sensitive_attr)
    assert fake_wandb.run.summary["fairness/group_labels"] == "names"
    assert "fairness/group/Zorblaxian/size" in fake_wandb.logged


# 3. The negative check: no raw group name in anything a coded run writes.


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_a_coded_mlflow_run_holds_no_group_name(fake_mlflow, task):
    report = _full_report(task)
    # The raw report really does carry every name, in prose too.
    raw = json.dumps(report, default=str)
    assert all(name in raw for name in DISTINCTIVE)

    logged = _quiet(log_fairness_to_mlflow, report, label_key=KEY)

    for part in _everything_written_mlflow(fake_mlflow) + [json.dumps(logged, default=str)]:
        for name in DISTINCTIVE:
            assert name not in part, f"{name!r} reached the tracker in {part[:120]!r}"

    codes = group_label_codes(DISTINCTIVE, label_key=KEY)
    for name in DISTINCTIVE[:3]:  # the three groups above min_group_size
        assert f"fairness.group.{codes[name]}.size" in fake_mlflow.metrics
    assert fake_mlflow.params["fairness.groups"] == str(sorted(codes[n] for n in DISTINCTIVE[:3]))
    assert fake_mlflow.params["fairness.groups_excluded"] == str([codes["Smallvillean"]])
    assert fake_mlflow.tags["fairness.group_labels"] == "coded"
    assert fake_mlflow.tags["fairness.group_label_key_id"] == key_id(KEY.encode("utf-8"))
    left_out = fake_mlflow.tags["fairness.left_out_by_group_coding"]
    assert "assessment.summary" in left_out and "explanations" in left_out
    assert fake_mlflow.tags["fairness.n_left_out_by_group_coding"] == 2

    [(folder, content)] = fake_mlflow.artifacts
    assert folder == "fairness_report"
    artifact = json.loads(content)
    assert "summary" not in artifact["assessment"] and "explanations" not in artifact
    assert artifact["group_label_coding"]["key_id"] == key_id(KEY.encode("utf-8"))
    positions = _group_positions(artifact)
    assert positions and all(CODE.match(p) for p in positions)
    assert artifact["effect_sizes"] and all(PAIR.match(p) for p in artifact["effect_sizes"])
    for metric in fake_mlflow.metrics:
        if ".effect_size." in metric:
            assert PAIR.match(metric.split(".")[2])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_a_coded_wandb_run_holds_no_group_name(fake_wandb, task):
    report = _full_report(task)
    _quiet(log_fairness_to_wandb, report, label_key=KEY)
    for part in _everything_written_wandb(fake_wandb):
        for name in DISTINCTIVE:
            assert name not in part, f"{name!r} reached W&B in {part[:120]!r}"
    codes = group_label_codes(DISTINCTIVE, label_key=KEY)
    assert f"fairness/group/{codes['Zorblaxian']}/size" in fake_wandb.logged
    assert fake_wandb.run.summary["fairness/group_labels"] == "coded"
    assert fake_wandb.run.summary["fairness/groups_excluded"] == str([codes["Smallvillean"]])
    [artifact] = fake_wandb.artifacts
    [(file_name, content)] = artifact.files
    assert file_name == "fairness_report.json"
    assert all(CODE.match(p) for p in _group_positions(json.loads(content)))


def test_a_numeric_looking_label_never_sits_in_a_group_position(fake_mlflow):
    """ "1" is checked only where a group is written: elsewhere it is a count."""
    report = _full_report(names=("1", "Zorblaxian", "0", "Smallvillean"))
    _quiet(log_fairness_to_mlflow, report, label_key=KEY)
    for metric in fake_mlflow.metrics:
        parts = metric.split(".")
        if parts[1] == "group":
            assert CODE.match(parts[2]), metric
        if parts[1] == "effect_size":
            assert PAIR.match(parts[2]), metric
    for param in ("fairness.groups", "fairness.groups_excluded"):
        listed = json.loads(fake_mlflow.params[param].replace("'", '"'))
        assert listed and all(CODE.match(g) for g in listed)
    for param in ("fairness.group_stats_not_measured", "fairness.group_stats_suppressed"):
        listed = json.loads(fake_mlflow.params[param].replace("'", '"'))
        assert all(CODE.match(entry.split(".")[0]) for entry in listed)
    artifact = json.loads(fake_mlflow.artifacts[0][1])
    positions = _group_positions(artifact)
    assert positions and all(CODE.match(p) for p in positions)
    assert "1" not in positions and "0" not in positions


def test_suppressed_group_stats_are_named_by_code(fake_mlflow):
    _quiet(log_fairness_to_mlflow, _full_report(), label_key=KEY, include_group_stats=False)
    suppressed = fake_mlflow.params["fairness.group_stats_suppressed"]
    assert suppressed != "[]"
    assert all(name not in suppressed for name in DISTINCTIVE)
    assert not any(".group." in metric for metric in fake_mlflow.metrics)


# 4. Order does not leak, and a restated pair is the producer's own other side.


def _key_that_flips(first, second):
    """A key under which ``first`` (the producer's group1) gets the HIGHER code."""
    for i in range(200):
        key = f"flip key {i}"
        codes = group_label_codes([first, second], label_key=key)
        if codes[first] > codes[second]:
            return key
    raise AssertionError("no flipping key in 200 tries")  # pragma: no cover


def _binary_data(seed=3, n1=180, n2=220):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n1 + n2)
    y_pred = np.concatenate([rng.random(n1) < 0.35, rng.random(n2) < 0.55]).astype(int)
    return y_true, y_pred, n1, n2


def test_a_restated_classification_pair_equals_the_producer_read_the_other_way():
    y_true, y_pred, n1, n2 = _binary_data()
    # The producer orders groups by name: Alpha before Beta, so it computes
    # Alpha vs Beta. Renaming Alpha to Zeta makes it compute Beta vs Alpha.
    forward = _quiet(
        compute_effect_sizes, y_true, y_pred, np.array(["Alpha"] * n1 + ["Beta"] * n2)
    )["Alpha_vs_Beta"]
    reverse = _quiet(compute_effect_sizes, y_true, y_pred, np.array(["Zeta"] * n1 + ["Beta"] * n2))[
        "Beta_vs_Zeta"
    ]

    key = _key_that_flips("Alpha", "Beta")
    codes = group_label_codes(["Alpha", "Beta"], label_key=key)
    coded, _left, unexpected = _group_codes.code_report(
        {
            "effect_sizes": {"Alpha_vs_Beta": forward},
            "data_info": {"valid_groups": ["Alpha", "Beta"]},
        },
        key.encode("utf-8"),
    )
    assert unexpected == []
    [(pair, restated)] = coded["effect_sizes"].items()
    assert pair == f"{codes['Beta']}_vs_{codes['Alpha']}"

    assert restated["cohens_d_positive_rate"] == reverse["cohens_d_positive_rate"]
    assert restated["cohens_h_positive_rate"] == reverse["cohens_h_positive_rate"]
    for field in ("risk_ratio", "odds_ratio"):
        assert restated[field] == pytest.approx(reverse[field], rel=1e-12)
    for field in ("group1_positive_rate", "group2_positive_rate", "group1_size", "group2_size"):
        assert restated[field] == reverse[field]
    assert restated["interpretation"] == reverse["interpretation"]


def test_a_restated_regression_pair_equals_the_producer_read_the_other_way():
    rng = np.random.default_rng(5)
    n1, n2 = 150, 170
    y_true = rng.normal(size=n1 + n2)
    y_pred = y_true + np.concatenate([rng.normal(0.2, 0.4, n1), rng.normal(-0.1, 0.5, n2)])
    forward = _quiet(
        compute_regression_effect_sizes,
        y_true,
        y_pred,
        np.array(["Alpha"] * n1 + ["Beta"] * n2),
    )["Alpha_vs_Beta"]
    reverse = _quiet(
        compute_regression_effect_sizes,
        y_true,
        y_pred,
        np.array(["Zeta"] * n1 + ["Beta"] * n2),
    )["Beta_vs_Zeta"]
    key = _key_that_flips("Alpha", "Beta")
    coded, _left, unexpected = _group_codes.code_report(
        {"effect_sizes": {"Alpha_vs_Beta": forward}, "group_stats": {"Alpha": {}, "Beta": {}}},
        key.encode("utf-8"),
    )
    assert unexpected == []
    [restated] = coded["effect_sizes"].values()
    for field in (
        "cohens_d_predictions",
        "cohens_d_residuals",
        "mean_pred_diff",
        "mean_residual_diff",
    ):
        assert restated[field] == pytest.approx(reverse[field], rel=1e-12, abs=1e-15)
    for field in ("group1_size", "group2_size", "interpretation", "interpretation_residuals"):
        assert restated[field] == reverse[field]


def test_a_pair_already_in_code_order_is_written_unchanged():
    y_true, y_pred, n1, n2 = _binary_data()
    forward = _quiet(
        compute_effect_sizes, y_true, y_pred, np.array(["Alpha"] * n1 + ["Beta"] * n2)
    )["Alpha_vs_Beta"]
    for i in range(200):
        key = f"keep key {i}"
        codes = group_label_codes(["Alpha", "Beta"], label_key=key)
        if codes["Alpha"] < codes["Beta"]:
            break
    coded, _left, _unexpected = _group_codes.code_report(
        {
            "effect_sizes": {"Alpha_vs_Beta": forward},
            "data_info": {"valid_groups": ["Alpha", "Beta"]},
        },
        key.encode("utf-8"),
    )
    assert coded["effect_sizes"] == {f"{codes['Alpha']}_vs_{codes['Beta']}": forward}


def test_the_tracker_writes_the_restated_ratio(fake_mlflow):
    key = _key_that_flips("Alpha", "Beta")
    codes = group_label_codes(["Alpha", "Beta"], label_key=key)
    report = {
        "task_type": "classification",
        "metrics": {},
        "group_stats": {"Alpha": {"size": 100}, "Beta": {"size": 100}},
        "effect_sizes": {
            "Alpha_vs_Beta": {
                "cohens_h_positive_rate": -0.25,
                "risk_ratio": (0.5, 0.25, 0.8),
                "odds_ratio": (0.0, float("nan"), float("nan")),
            }
        },
        "assessment": {},
        "data_info": {},
    }
    _quiet(log_fairness_to_mlflow, report, label_key=key, log_artifacts=False)
    pair = f"{codes['Beta']}_vs_{codes['Alpha']}"
    assert fake_mlflow.metrics[f"fairness.effect_size.{pair}.cohens_h"] == 0.25
    assert fake_mlflow.metrics[f"fairness.effect_size.{pair}.risk_ratio"] == 2.0
    # 1 / 0.0 is a measured total exclusion read from the other side, written as inf.
    assert fake_mlflow.metrics[f"fairness.effect_size.{pair}.odds_ratio"] == float("inf")


def test_groups_and_records_are_listed_in_code_order():
    report = _full_report()
    coded, _left, _unexpected = _group_codes.code_report(report, KEY.encode("utf-8"))
    for listed in (
        coded["data_info"]["valid_groups"],
        list(coded["group_stats"]),
        list(coded["data_info"]["group_sizes"]),
        list(coded["statistical_validation"]["per_group_methods"]),
    ):
        assert listed == sorted(listed)


# 5. Pair keys come from the producer's structure, never from splitting on "_vs_".


def test_a_pair_key_is_matched_to_the_groups_that_built_it():
    report = {
        "group_stats": {"x_vs_y": {}, "z": {}},
        "effect_sizes": {"x_vs_y_vs_z": {"cohens_h_positive_rate": 0.1}},
    }
    coded, left_out, _unexpected = _group_codes.code_report(report, b"k")
    codes = group_label_codes(["x_vs_y", "z"], label_key=b"k")
    assert list(coded["effect_sizes"]) == ["_vs_".join(sorted((codes["x_vs_y"], codes["z"])))]
    assert left_out == []


def test_an_ambiguous_pair_key_is_left_out_and_named_without_its_text():
    # "a_vs_b" + "c" and "a" + "b_vs_c" both build "a_vs_b_vs_c".
    report = {
        "group_stats": {"a_vs_b": {}, "c": {}, "a": {}, "b_vs_c": {}},
        "effect_sizes": {"a_vs_b_vs_c": {"cohens_h_positive_rate": 0.1}},
    }
    with pytest.warns(UserWarning, match="could not be coded"):
        coded, left_out = _code_via_writer_helper(report)
    assert coded["effect_sizes"] == {}
    assert len(left_out) == 1 and "more than one pair" in left_out[0]
    assert "a_vs_b" not in left_out[0]


def _code_via_writer_helper(report):
    from vfairness.evaluation.vfairness_metrics.integrations import _code_for_tracker

    return _code_for_tracker(report, "coded", b"k", "test")


def test_a_pair_key_no_group_pair_builds_is_left_out(fake_mlflow):
    report = {
        "task_type": "classification",
        "effect_sizes": {"F_vs_M": {"cohens_h_positive_rate": 0.1}},
    }
    with pytest.warns(UserWarning, match="could not be coded"):
        log_fairness_to_mlflow(report, label_key=KEY, log_artifacts=False)
    assert not any("effect_size" in metric for metric in fake_mlflow.metrics)
    assert "F_vs_M" not in fake_mlflow.tags["fairness.left_out_by_group_coding"]
    assert fake_mlflow.tags["fairness.n_left_out_by_group_coding"] == 1


# 6. Fail closed on what coding does not know.


def test_unknown_group_fields_and_text_naming_a_group_are_left_out_and_named():
    report = _full_report()
    report["data_info"]["worst_group"] = "Zorblaxian"
    report["data_info"]["privileged_group"] = 1  # a numeric label is just a number
    report["group_analysis"] = {"privileged_group": {"group": "Quibbletonian"}}
    report["metrics_with_ci"]["demographic_parity_difference"]["metadata"]["note"] = (
        "Mxyzptlkian had the widest interval"
    )
    report["multiple_testing_correction"] = {"by_group": {0: 0.1}, "method": "fdr"}
    coded, left_out, unexpected = _group_codes.code_report(report, KEY.encode("utf-8"))
    text = json.dumps(coded, default=str)
    for name in DISTINCTIVE:
        assert name not in text
    assert "worst_group" not in coded["data_info"]
    assert "privileged_group" not in coded["data_info"]
    assert "group_analysis" not in coded
    assert coded["multiple_testing_correction"] == {"method": "fdr"}
    assert len(unexpected) == 5 and len(left_out) == 7
    assert coded["data_info"]["n_groups"] == report["data_info"]["n_groups"]


def test_two_labels_with_the_same_text_are_refused():
    report = {"group_stats": {1: {"size": 5}, "1": {"size": 6}}}
    with pytest.raises(ValueError, match="same text"):
        _group_codes.code_report(report, b"k")


def test_a_hand_built_report_with_unreadable_entries_keeps_its_disclosure(fake_mlflow):
    report = {
        "task_type": "classification",
        "metrics": {},
        "metrics_with_ci": {"demographic_parity_difference": (0.42, 0.31, 0.53)},
        "group_stats": {"Zorblaxian": (0.3,), "Quibbletonian": {"size": 10}},
        "assessment": {},
        "data_info": {},
    }
    with pytest.warns(UserWarning, match="not in a shape"):
        log_fairness_to_mlflow(report, label_key=KEY, log_artifacts=False)
    codes = group_label_codes(["Zorblaxian"], label_key=KEY)
    unreadable = fake_mlflow.params["fairness.unreadable_entries"]
    assert f"group_stats.{codes['Zorblaxian']} (tuple)" in unreadable
    assert "Zorblaxian" not in unreadable
    assert fake_mlflow.params["fairness.n_intervals_not_measured"] == 4


def test_a_long_left_out_list_is_cut_with_a_count():
    entries = [f"section.field_{i} (why)" for i in range(500)]
    text = _group_codes.left_out_text(entries, limit=300)
    assert len(text) <= 300 and text.endswith("more")


# 7. Names mode writes exactly what the writers wrote before coding existed.

# Captured by running the writers as they were before this change on _GOLDEN_REPORT.
_GOLDEN_REPORT = {
    "task_type": "classification",
    "metrics": {"demographic_parity_difference": 0.21, "equalized_odds_difference": float("nan")},
    "metrics_with_ci": {
        "demographic_parity_difference": {
            "point_estimate": 0.21,
            "lower_bound": 0.1,
            "upper_bound": 0.3,
            "standard_error": 0.05,
        }
    },
    "effect_sizes": {
        "Female_vs_Male": {
            "cohens_d_positive_rate": -0.4,
            "cohens_h_positive_rate": -0.4,
            "risk_ratio": (0.6, 0.4, 0.9),
            "odds_ratio": (0.5, 0.3, 0.8),
        }
    },
    "group_stats": {
        "Female": {"size": 150, "positive_rate": 0.41, "tpr": 0.55},
        "Male": {"size": 150, "positive_rate": 0.62, "tpr": float("nan")},
    },
    "assessment": {"fairness_score": 0.5, "summary": "1/2 metrics within thresholds"},
    "data_info": {
        "n_samples": 310,
        "n_groups": 3,
        "valid_groups": ["Female", "Male"],
        "invalid_groups": ["Other"],
        "group_sizes": {"Female": 150, "Male": 150, "Other": 10},
    },
}

_GOLDEN_MLFLOW_METRICS = {
    "fairness.demographic_parity_difference": 0.21,
    "fairness.demographic_parity_difference.ci_lower": 0.1,
    "fairness.demographic_parity_difference.ci_upper": 0.3,
    "fairness.demographic_parity_difference.std_error": 0.05,
    "fairness.demographic_parity_difference.value": 0.21,
    "fairness.effect_size.Female_vs_Male.cohens_d": -0.4,
    "fairness.effect_size.Female_vs_Male.cohens_h": -0.4,
    "fairness.effect_size.Female_vs_Male.odds_ratio": 0.5,
    "fairness.effect_size.Female_vs_Male.risk_ratio": 0.6,
    "fairness.fairness_score": 0.5,
    "fairness.group.Female.positive_rate": 0.41,
    "fairness.group.Female.size": 150.0,
    "fairness.group.Female.tpr": 0.55,
    "fairness.group.Male.positive_rate": 0.62,
    "fairness.group.Male.size": 150.0,
}
_GOLDEN_MLFLOW_PARAMS = {
    "fairness.effect_sizes_not_measured": "[]",
    "fairness.group_stats_not_measured": "['Male.tpr']",
    "fairness.group_stats_status": "logged",
    "fairness.group_stats_suppressed": "[]",
    "fairness.groups": "['Female', 'Male']",
    "fairness.groups_excluded": "['Other']",
    "fairness.intervals_not_measured": "[]",
    "fairness.metrics_not_measured": "['equalized_odds_difference']",
    "fairness.n_effect_sizes_not_measured": 0,
    "fairness.n_group_stats_not_measured": 1,
    "fairness.n_group_stats_suppressed": 0,
    "fairness.n_groups": 3,
    "fairness.n_groups_excluded": 1,
    "fairness.n_intervals_not_measured": 0,
    "fairness.n_metrics_not_measured": 1,
    "fairness.n_rows_excluded_by_min_group_size": 10,
    "fairness.n_samples": 310,
    "fairness.n_unreadable_entries": 0,
    "fairness.task_type": "classification",
    "fairness.unreadable_entries": "[]",
}
_GOLDEN_WANDB_SUMMARY_FROM_NOTES = {
    "fairness/effect_sizes_not_measured": "[]",
    "fairness/group_stats_not_measured": "['Male/tpr']",
    "fairness/groups_excluded": "['Other']",
    "fairness/n_rows_excluded_by_min_group_size": 10,
    "fairness/n_samples": 310,
    "fairness/n_groups": 3,
}


def test_names_mode_writes_what_the_mlflow_writer_wrote_before(fake_mlflow, no_env_key):
    _quiet(log_fairness_to_mlflow, _GOLDEN_REPORT, group_labels="names", log_artifacts=False)
    assert fake_mlflow.metrics == _GOLDEN_MLFLOW_METRICS
    assert fake_mlflow.params == _GOLDEN_MLFLOW_PARAMS
    assert fake_mlflow.tags == {
        "fairness.fairness_score_status": "assessed",
        "fairness.library": "vfairness",
        "fairness.library_version": vfairness.__version__,
        "fairness.group_labels": "names",  # the one new tag
    }


def test_names_mode_writes_what_the_wandb_writer_wrote_before(fake_wandb, no_env_key):
    _quiet(log_fairness_to_wandb, _GOLDEN_REPORT, group_labels="names", log_artifacts=False)
    expected_logged = {
        key.replace(".", "/", 1).replace(".", "/"): value
        for key, value in _GOLDEN_MLFLOW_METRICS.items()
    }
    expected_logged["fairness/fairness_score_status"] = "assessed"
    assert fake_wandb.logged == expected_logged
    for key, value in _GOLDEN_WANDB_SUMMARY_FROM_NOTES.items():
        assert fake_wandb.run.summary[key] == value
    assert fake_wandb.run.summary["fairness/group_labels"] == "names"
    assert "fairness/group_label_key_id" not in fake_wandb.run.summary


def test_names_mode_uploads_the_report_itself(fake_mlflow, no_env_key):
    report = {k: v for k, v in _GOLDEN_REPORT.items() if k not in ("metrics", "group_stats")}
    _quiet(log_fairness_to_mlflow, report, group_labels="names")
    artifact = json.loads(fake_mlflow.artifacts[0][1])
    assert artifact["assessment"]["summary"] == "1/2 metrics within thresholds"
    assert artifact["data_info"]["valid_groups"] == ["Female", "Male"]
    assert "Female_vs_Male" in artifact["effect_sizes"]
    assert "group_label_coding" not in artifact


# 8. A REAL MLflow: the run read back holds codes, never names.


def test_a_real_mlflow_run_holds_codes_not_names(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_VAR, KEY)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    analyzer = _analyzer()
    with mlflow.start_run() as run:
        _quiet(log_fairness_to_mlflow, analyzer)
    stored = mlflow.get_run(run.info.run_id).data
    folder = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="fairness_report", dst_path=str(tmp_path / "dl")
    )
    artifact_texts = [path.read_text() for path in sorted((tmp_path / "dl").rglob("*.json"))]
    assert folder and len(artifact_texts) == 1

    everything = list(stored.metrics) + list(stored.params) + list(stored.params.values())
    everything += list(stored.tags) + list(stored.tags.values()) + artifact_texts
    for part in everything:
        for name in DISTINCTIVE:
            assert name not in part, f"{name!r} is stored in the real run: {part[:120]!r}"

    codes = group_label_codes(DISTINCTIVE)  # the same key, from the environment
    for name in DISTINCTIVE[:3]:
        assert f"fairness.group.{codes[name]}.size" in stored.metrics
    assert stored.tags["fairness.group_labels"] == "coded"
    assert stored.tags["fairness.group_label_key_id"] == key_id(KEY.encode("utf-8"))
    decode = {code: name for name, code in codes.items()}
    artifact = json.loads(artifact_texts[0])
    assert sorted(decode[c] for c in artifact["data_info"]["valid_groups"]) == sorted(
        DISTINCTIVE[:3]
    )
    assert [decode[c] for c in artifact["data_info"]["invalid_groups"]] == ["Smallvillean"]


def test_a_real_mlflow_run_without_a_key_stays_empty(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ENV_VAR, raising=False)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    with mlflow.start_run() as run:
        with pytest.raises(ValueError, match=ENV_VAR):
            log_fairness_to_mlflow(_analyzer())
    stored = mlflow.get_run(run.info.run_id).data
    assert stored.metrics == {} and stored.params == {}
    assert not any(tag.startswith("fairness.") for tag in stored.tags)
    assert mlflow.artifacts.list_artifacts(run_id=run.info.run_id) == []
