"""The audit artifact's sha256 must be a CONTENT address.

Finding (xai/in-proc audit, 2026-08-27, MEDIUM). The module promised "a
deterministic JSON blob that the regulator can replay byte-stably given the same
model and the same seed", and the returned sha256 is the natural key in
``xai_audit_artifacts.id``. Two wall-clock fields were inside the hashed
payload: the bundle's ``generated_at`` and every ``Explanation.timestamp``. So
the digest differed on every call by construction. Two calls one second apart on
identical inputs gave a938ebe9... and 12b746eb..., a field-level diff isolating
exactly those two fields.

Consequences: a regulator replaying the artifact could never obtain a matching
hash, so the reproducibility the artifact exists to attest was unverifiable; and
a content-addressed natural key that never repeats produces duplicate rows
instead of colliding on identical content.

It survived because ``build_audit_artifact_bundle`` has zero callers in ``src/``
or ``tests/`` and no determinism test, inside a package with no reachable test at
all until 4e58c88. This file is that missing test.
"""

from __future__ import annotations

import time

import pytest

from vfairness.xai.schemas import Attribution, Explanation
from vfairness.xai.storage.audit_artifact import (
    DIGEST_EXCLUDED_PATHS,
    build_audit_artifact_bundle,
)


def _explanation(*, contribution: float = -0.4, timestamp: str | None = None) -> Explanation:
    kwargs = dict(
        method="shap.TreeExplainer",
        instance_id="applicant-1",
        subject_id="subject-loan-xgb-v3",
        model_hash="sha256:abc",
        data_hash="sha256:def",
        base_value=-0.4,
        prediction=-1.1,
        attributions=[
            Attribution(feature="credit_util", contribution=contribution),
            Attribution(feature="income", contribution=-0.3),
        ],
        units="log-odds",
        library_versions={"shap": "0.46.0"},
    )
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return Explanation(**kwargs)


def _build(explanation: Explanation, *, seed: int = 42, params: dict | None = None):
    return build_audit_artifact_bundle(
        subject_id="subject-loan-xgb-v3",
        explanations=[explanation],
        decomposition=None,
        params=params if params is not None else {"nsamples": 100},
        library_versions={"shap": "0.46.0"},
        seed=seed,
    )


# --- the missing test: identical content, identical digest -------------------


def test_two_builds_of_the_same_content_collide():
    """The test whose absence let the defect survive."""
    b1, sha1 = _build(_explanation())
    time.sleep(0.002)  # guarantee the wall clock moves between the two builds
    b2, sha2 = _build(_explanation())

    # Not a vacuous pass: the clock genuinely advanced, so the digests are
    # colliding despite differing timestamps rather than because none differ.
    assert b1["generated_at"] != b2["generated_at"], (
        "the clock did not advance between builds, so this test would prove nothing"
    )
    assert sha1 == sha2, (
        f"identical content must produce an identical content address; got {sha1} vs {sha2}"
    )


def test_explanation_timestamp_is_excluded_from_the_digest():
    """Two explanations differing ONLY in their timestamp are the same content."""
    sha_a = _build(_explanation(timestamp="2020-01-01T00:00:00+00:00"))[1]
    sha_b = _build(_explanation(timestamp="2099-12-31T23:59:59+00:00"))[1]
    assert sha_a == sha_b


# --- negative cases: a real content change MUST move the digest --------------


def test_changed_attribution_changes_the_digest():
    """Without this, 'the shas match' could be satisfied by hashing nothing."""
    sha_a = _build(_explanation(contribution=-0.4))[1]
    sha_b = _build(_explanation(contribution=-0.41))[1]
    assert sha_a != sha_b, "a changed attribution must change the content address"


def test_changed_seed_changes_the_digest():
    assert _build(_explanation(), seed=42)[1] != _build(_explanation(), seed=43)[1]


def test_changed_params_change_the_digest():
    sha_a = _build(_explanation(), params={"nsamples": 100})[1]
    sha_b = _build(_explanation(), params={"nsamples": 200})[1]
    assert sha_a != sha_b


# --- the excluded fields stay VISIBLE to the reader --------------------------


def test_wall_clock_fields_remain_in_the_artifact():
    """Excluded from the hash, not removed: a regulator wants to see them."""
    bundle, sha = _build(_explanation())

    assert "generated_at" in bundle, "the artifact must still say when it was produced"
    assert bundle["explanations"][0]["timestamp"], "the explanation keeps its own timestamp"
    assert bundle["content_sha256"] == sha


def test_bundle_declares_what_the_digest_does_not_cover():
    """A digest is only checkable if the verifier knows what it covers."""
    bundle, _ = _build(_explanation())
    assert bundle["digest_excludes"] == list(DIGEST_EXCLUDED_PATHS)
    assert "generated_at" in bundle["digest_excludes"]
    assert "explanations[].timestamp" in bundle["digest_excludes"]


def test_digest_is_a_hex_sha256():
    _, sha = _build(_explanation())
    assert len(sha) == 64
    int(sha, 16)  # raises if not hex


@pytest.mark.parametrize("field_name", ["generated_at", "content_sha256", "digest_excludes"])
def test_bundle_carries_its_provenance_fields(field_name):
    bundle, _ = _build(_explanation())
    assert field_name in bundle
