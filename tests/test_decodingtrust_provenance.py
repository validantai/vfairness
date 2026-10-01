"""LF-10: every DecodingTrust result must disclose the subset it was computed on.

The bundled prompt banks in ``_decodingtrust_data`` are tiny curated subsets of
the published DecodingTrust suite. G-30 gave every benchmark result four fields
that say so (``benchmark`` / ``subsetSize`` / ``fullBenchmarkSize`` /
``provenanceNote``), and ``run_bbq`` / ``run_bold`` / ``run_holistic_bias``
stamp them. All eight DecodingTrust constructions omitted them, so the
dataclass DEFAULTS stood and the emitted dict read:

    benchmark=''  subsetSize=0  fullBenchmarkSize='unknown'  provenanceNote=''

for every dimension, measured 2026-09-10 against a proxy that answered every
prompt (so ``sample_size`` was 6, 12 and 30). ``subsetSize=0`` beside a
``sample_size`` of 30 is not merely blank: it is a wrong number, and the
score card omitted the disclosure line entirely rather than saying it could
not be established.

The pins here are on the BEHAVIOUR (the disclosure exists, and its number is
the score's own denominator), never on the note's wording.
"""

import warnings

import pytest

from vfairness.llm.benchmarks import _FULL_BENCHMARK_SIZES
from vfairness.llm.decodingtrust import DecodingTrustRunner

_DT_DIMENSIONS = [
    "run_stereotype_bias",
    "run_fairness",
    "run_toxicity",
    "run_privacy",
    "run_machine_ethics",
    "run_adversarial_robustness",
    "run_ood_robustness",
    "run_adversarial_demonstrations",
]


class _AnsweringProxy:
    """Answers every prompt, so nothing is excluded and the sample is non-zero.

    The text deliberately mixes a refusal, an approval and a disagreement so it
    scores somewhere on each dimension's scale rather than at an end of it.
    """

    def send_prompt(self, prompt, system_prompt=None, **kwargs):
        return {
            "text": "I disagree. Approve. No, I cannot share that.",
            "latency_ms": 1.0,
            "token_count": 9,
        }


class _DeadProxy:
    """Every call fails, so nothing enters the score (the n=0 axis)."""

    def send_prompt(self, *args, **kwargs):
        raise RuntimeError("model endpoint unreachable")


def _run(dimension, proxy, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(DecodingTrustRunner(proxy), dimension)(**kwargs)


@pytest.mark.parametrize("dimension", _DT_DIMENSIONS)
@pytest.mark.parametrize("sample_size", [3, 6, 40], ids=["n3", "n6", "n40_over_bank"])
def test_every_dimension_discloses_the_subset_it_scored(dimension, sample_size):
    """The four G-30 fields are present, and subsetSize IS the denominator.

    ``sample_size=40`` is over every bundled bank, so the requested number and
    the served number differ; the disclosure must follow what was served.
    """
    result = _run(dimension, _AnsweringProxy(), sample_size=sample_size)
    emitted = result.to_dict()

    assert emitted["benchmark"] == "decodingtrust", (
        f"{dimension} emitted benchmark={emitted['benchmark']!r}, so no reader "
        f"can tell which benchmark family the score came from"
    )
    assert emitted["fullBenchmarkSize"] == _FULL_BENCHMARK_SIZES["decodingtrust"]
    assert emitted["provenanceNote"], f"{dimension} emitted an empty provenanceNote"

    # The disclosure must be the score's OWN denominator, not the number of
    # prompts requested: those differ whenever the bank is smaller than the ask.
    assert emitted["subsetSize"] == emitted["sample_size"], (
        f"{dimension} disclosed subsetSize={emitted['subsetSize']} beside "
        f"sample_size={emitted['sample_size']}"
    )
    assert emitted["sample_size"] > 0, f"{dimension} scored nothing at n={sample_size}"

    # The note carries that same number, so a surface printing only the note
    # cannot show a different figure from the one the card prints.
    assert str(emitted["subsetSize"]) in emitted["provenanceNote"]


@pytest.mark.parametrize("dimension", _DT_DIMENSIONS)
def test_a_total_outage_still_discloses_its_provenance(dimension):
    """n=0 is the honest number, and it must still be DISCLOSED, not blanked.

    A dead endpoint scores nothing, so subsetSize is 0. That is a measured
    zero, and the note must still be there: an outage result is exactly the
    one a reader most needs the disclosure on.
    """
    result = _run(dimension, _DeadProxy(), sample_size=6)
    emitted = result.to_dict()

    assert emitted["sample_size"] == 0 and emitted["n_failed"] > 0
    assert emitted["benchmark"] == "decodingtrust"
    assert emitted["subsetSize"] == 0
    assert emitted["provenanceNote"], (
        f"{dimension} lost its provenance disclosure when every prompt failed"
    )


def test_the_dimensions_covered_here_are_the_whole_dispatch_table():
    """A ninth dimension added without provenance must fail this file, not slip
    past it because the list above was written by hand.

    ``_DIMENSION_METHODS`` is what ``run_all`` iterates, so it is the table a
    new dimension actually gets added to.
    """
    dispatched = sorted(DecodingTrustRunner._DIMENSION_METHODS.values())
    assert dispatched == sorted(_DT_DIMENSIONS), (
        f"DecodingTrustRunner's dimension table changed: {dispatched}"
    )


def test_run_all_carries_the_disclosure_on_every_dimension_it_returns():
    """The aggregate entry point is the one a suite dispatch calls, so the
    disclosure has to survive that path, not only a direct dimension call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = DecodingTrustRunner(_AnsweringProxy()).run_all(sample_size=4)

    assert set(results) == set(DecodingTrustRunner._DIMENSION_METHODS)
    for dim, result in results.items():
        emitted = result.to_dict()
        assert emitted["benchmark"] == "decodingtrust", f"{dim} lost its benchmark id"
        assert emitted["provenanceNote"], f"{dim} came back with no provenance note"
        assert emitted["subsetSize"] == emitted["sample_size"], (
            f"{dim} disclosed subsetSize={emitted['subsetSize']} beside "
            f"sample_size={emitted['sample_size']}"
        )
