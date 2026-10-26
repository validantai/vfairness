"""vfairness._register: fingerprints and evidence bundles for the Validant Model Register.

The manifest digest must equal, byte for byte, what the Validant platform's
browser code computes, so both sides are tested against the same vectors
(tests/fixtures/register_identity_vectors.json, computed with hashlib alone).
"""

import base64
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vfairness._register import (
    BUNDLE_FORMAT,
    Fingerprint,
    build_evidence_bundle,
    dataset_fingerprint,
    file_fingerprint,
    folder_fingerprint,
    manifest_digest,
    model_fingerprint,
    read_evidence_bundle,
    write_evidence_bundle,
)

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "register_identity_vectors.json").read_text(
        encoding="utf-8"
    )
)


def _files(vector):
    return [(f["path"], base64.b64decode(f["base64"])) for f in vector["files"]]


@pytest.mark.parametrize("vector", VECTORS["manifest"], ids=lambda v: v["name"])
def test_manifest_digest_reproduces_the_shared_vectors(vector):
    assert manifest_digest(_files(vector)) == vector["digest"]


@pytest.mark.parametrize("vector", VECTORS["manifest"], ids=lambda v: v["name"])
def test_a_folder_on_disk_gives_the_same_digest(tmp_path, vector):
    for path, content in _files(vector):
        target = tmp_path.joinpath(*path.replace("\\", "/").split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    fp = folder_fingerprint(tmp_path)
    assert fp == Fingerprint(digest=vector["digest"], files=len(vector["files"]))
    assert model_fingerprint(tmp_path) == fp


def test_negative_control_sorting_by_code_point_would_differ():
    # Python's str sort orders by code point, JavaScript's by UTF-16 unit: the
    # rule is UTF-8 byte order, and vector 2 is where the three disagree.
    vector = VECTORS["manifest"][1]
    entries = sorted(
        ((p, hashlib.sha256(b).hexdigest()) for p, b in _files(vector)),
        key=lambda e: [ord(c) for c in e[0].encode("utf-16-be").decode("latin-1")],
    )
    text = "".join(f"{p}\0{s}\n" for p, s in entries).encode("utf-8")
    assert hashlib.sha256(text).hexdigest() != vector["digest"]


def test_order_does_not_matter_and_one_byte_does():
    files = _files(VECTORS["manifest"][0])
    assert manifest_digest(list(reversed(files))) == VECTORS["manifest"][0]["digest"]
    path, content = files[2]
    files[2] = (path, bytes([content[0] ^ 1]) + content[1:])
    assert manifest_digest(files) != VECTORS["manifest"][0]["digest"]


def test_precomputed_file_digests_give_the_same_manifest():
    files = _files(VECTORS["manifest"][0])
    assert (
        manifest_digest([(p, hashlib.sha256(b).hexdigest()) for p, b in files])
        == VECTORS["manifest"][0]["digest"]
    )


def test_refusals():
    with pytest.raises(ValueError, match="no files"):
        manifest_digest([])
    with pytest.raises(ValueError, match="twice"):
        manifest_digest([("a", b"1"), ("./a", b"2")])
    with pytest.raises(ValueError, match="not inside"):
        manifest_digest([("../etc/passwd", b"1")])
    with pytest.raises(ValueError, match="SHA-256"):
        manifest_digest([("a", "not-a-digest")])


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks on this platform")
def test_a_symbolic_link_in_the_folder_is_refused(tmp_path):
    (tmp_path / "MLmodel").write_text("flavors: {}\n")
    outside = tmp_path.parent / f"outside-{tmp_path.name}.bin"
    outside.write_bytes(b"secret")
    try:
        os.symlink(outside, tmp_path / "model.bin")
    except OSError:
        pytest.skip("cannot create symlinks here")
    with pytest.raises(ValueError, match="Symbolic link"):
        folder_fingerprint(tmp_path)


def test_file_and_dataset_fingerprints_are_plain_sha256(tmp_path):
    data = b"age,approved\n31,1\n"
    f = tmp_path / "holdout.csv"
    f.write_bytes(data)
    want = hashlib.sha256(data).hexdigest()
    assert file_fingerprint(f).digest == want
    assert dataset_fingerprint(f).digest == want
    assert dataset_fingerprint(data).digest == want
    assert model_fingerprint(f).digest == want


def test_a_missing_path_is_not_mistaken_for_a_uri(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_fingerprint(tmp_path / "nope")


def _fp():
    return Fingerprint(digest=VECTORS["manifest"][0]["digest"], files=3)


def test_bundle_shape_and_round_trip(tmp_path):
    stamp = datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)
    bundle = build_evidence_bundle(
        _fp(),
        dataset_sha256=hashlib.sha256(b"holdout").hexdigest(),
        mlflow={"model_uri": "models:/credit@candidate", "version": 5},
        results={"fairness_score": 0.91},
        vfairness_version="0.1.0",
        created_at=stamp,
    )
    assert bundle["format"] == BUNDLE_FORMAT
    assert bundle["identity"] == {
        "method": "sha256",
        "digest": VECTORS["manifest"][0]["digest"],
        "files": 3,
    }
    assert bundle["mlflow"] == {
        "model_uri": "models:/credit@candidate",
        "model_id": None,
        "run_id": None,
        "version": "5",
    }
    assert bundle["created_at"] == "2026-10-24T10:00:00Z"
    path = write_evidence_bundle(bundle, tmp_path / "credit.evidence.json")
    assert read_evidence_bundle(path) == bundle


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dataset_sha256": "ABC"}, "dataset_sha256"),
        ({"system_type": "robot"}, "system_type"),
        ({"pathway": "endpoint"}, "identified in the browser"),
        ({"mlflow": {"tracking_uri": "http://internal"}}, "unexpected mlflow keys"),
        ({"results": {"score": math.nan}}, "Out of range float"),
    ],
)
def test_bundle_refusals(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_evidence_bundle(_fp(), vfairness_version="0.1.0", **kwargs)


def test_bundle_needs_a_sha256_identity():
    with pytest.raises(ValueError, match="SHA-256 Fingerprint"):
        build_evidence_bundle(Fingerprint(digest="0" * 64, files=1, method="org_hmac"))
    with pytest.raises(ValueError, match="identity.digest"):
        build_evidence_bundle(Fingerprint(digest="xyz", files=1))


def test_real_mlflow_model_versions(tmp_path, monkeypatch):
    """Two registered versions in a real MLflow (local SQLite store) get two digests; the same version twice, one."""
    mlflow = pytest.importorskip("mlflow")
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    import numpy as np

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    for c in (1.0, 0.1):
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                sklearn_linear.LogisticRegression(C=c).fit(X, y),
                name="model",
                registered_model_name="register-test",
                serialization_format="skops",
            )
    v1 = model_fingerprint("models:/register-test/1")
    v1_again = model_fingerprint("models:/register-test/1")
    v2 = model_fingerprint("models:/register-test/2")
    assert v1 == v1_again
    assert v1.digest != v2.digest
    assert v1.files >= 2
