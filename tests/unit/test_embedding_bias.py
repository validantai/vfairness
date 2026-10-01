"""WEAT / SEAT embedding-bias test suite (Workstream E).

Builds synthetic embeddings with a KNOWN planted stereotype (career words near
'male' terms, family words near 'female' terms) and asserts WEAT recovers a
large positive effect size with significance. A neutral control (random vectors)
must NOT flag bias.
"""

import numpy as np
import pytest

from vfairness.llm.embedding_bias import EmbeddingBiasDetector, WEATResult

CAREER = [
    "executive",
    "management",
    "professional",
    "corporation",
    "salary",
    "office",
    "business",
    "career",
]
FAMILY = ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]
MALE = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
FEMALE = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]


def _biased_embeddings(dim=24, seed=0):
    """Career~male, family~female, on a shared axis (the planted stereotype)."""
    rng = np.random.default_rng(seed)
    axis = np.zeros(dim)
    axis[0] = 1.0  # the "gender" direction
    emb = {}

    def place(words, sign):
        for w in words:
            noise = rng.normal(0, 0.15, dim)
            emb[w] = sign * 1.0 * axis + noise

    place(MALE, +1)
    place(CAREER, +1)  # male & career on the + side
    place(FEMALE, -1)
    place(FAMILY, -1)  # female & family on the - side
    return emb


def _neutral_embeddings(dim=128, seed=1):
    # Higher dimensionality drives spurious cosine structure toward zero, the
    # honest "no planted association" control.
    rng = np.random.default_rng(seed)
    return {w: rng.normal(0, 1, dim) for w in CAREER + FAMILY + MALE + FEMALE}


# ---------------------------------------------------------------------------


def test_requires_exactly_one_source():
    with pytest.raises(ValueError):
        EmbeddingBiasDetector()
    with pytest.raises(ValueError):
        EmbeddingBiasDetector(embeddings={}, embed_fn=lambda w: np.zeros((len(w), 2)))


def test_weat_detects_planted_stereotype():
    det = EmbeddingBiasDetector(embeddings=_biased_embeddings())
    res = det.weat(CAREER, FAMILY, MALE, FEMALE)
    assert isinstance(res, WEATResult)
    assert res.effect_size > 0.8, f"expected large positive effect, got {res.effect_size}"
    assert res.p_value < 0.05
    assert res.severity in ("high", "critical")
    assert res.method == "WEAT"


def test_weat_detects_negative_direction_stereotype():
    # Swapping the attribute sets flips the sign of the effect. A two-sided
    # permutation test must still flag this as significant bias; a one-sided
    # test would wrongly report p ~ 1.0 / severity "info" and miss it.
    det = EmbeddingBiasDetector(embeddings=_biased_embeddings())
    res = det.weat(CAREER, FAMILY, FEMALE, MALE)  # attribute order reversed
    assert res.effect_size < -0.8, f"expected large negative effect, got {res.effect_size}"
    assert res.p_value < 0.05, f"two-sided test must flag negative bias, got p={res.p_value}"
    assert res.severity in ("high", "critical")


def test_weat_neutral_embeddings_no_bias():
    # A single random draw can be unlucky, so assess the control across many
    # seeds: independent embeddings must NOT be systematically flagged as
    # significant bias. The significant-rate should sit near alpha (0.05), not
    # near 1.0 as it does for the planted-stereotype embeddings.
    n_seeds = 30
    n_flagged = 0
    for seed in range(n_seeds):
        det = EmbeddingBiasDetector(embeddings=_neutral_embeddings(seed=seed))
        res = det.weat(CAREER, FAMILY, MALE, FEMALE)
        if res.severity in ("high", "critical"):
            n_flagged += 1
    # Generously allow up to ~20% false positives (small word sets are noisy);
    # the planted-stereotype case flags ~100% of the time, so this cleanly
    # separates signal from noise.
    assert n_flagged <= 0.20 * n_seeds, f"{n_flagged}/{n_seeds} neutral runs flagged"


def test_embed_fn_path():
    emb = _biased_embeddings()
    det = EmbeddingBiasDetector(embed_fn=lambda words: np.asarray([emb[w] for w in words]))
    res = det.weat(CAREER, FAMILY, MALE, FEMALE)
    assert res.effect_size > 0.8


def test_missing_word_raises():
    det = EmbeddingBiasDetector(embeddings=_biased_embeddings())
    with pytest.raises(KeyError):
        det.weat(["not_a_word"], FAMILY, MALE, FEMALE)


def test_empty_set_raises():
    det = EmbeddingBiasDetector(embeddings=_biased_embeddings())
    with pytest.raises(ValueError):
        det.weat([], FAMILY, MALE, FEMALE)


def test_seat_tags_method_only_when_sentence_resolved():
    det = EmbeddingBiasDetector(embeddings=_biased_embeddings())
    # Plain word vectors: SEAT is not honest, so it stays WEAT with a note.
    word_level = det.seat(CAREER, FAMILY, MALE, FEMALE)
    assert word_level.method == "WEAT"
    assert any("sentence-resolved" in n for n in word_level.notes)
    # Sentence-resolved inputs: the SEAT label is honest.
    sentence_level = det.seat(CAREER, FAMILY, MALE, FEMALE, sentence_resolved=True)
    assert sentence_level.method == "SEAT"


def test_to_dict_json_friendly():
    import json

    det = EmbeddingBiasDetector(embeddings=_biased_embeddings())
    d = det.weat(CAREER, FAMILY, MALE, FEMALE).to_dict()
    expected = {
        "effect_size",
        "p_value",
        "test_statistic",
        "n_target_a",
        "n_target_b",
        "n_attribute_a",
        "n_attribute_b",
        "severity",
        "interpretation",
        "method",
        "notes",
    }
    assert set(d) == expected
    json.dumps(d)
