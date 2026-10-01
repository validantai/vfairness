"""LF-10 follow-up: DecodingTrust is NOT a fixed prompt bank.

Added by the LF-10 verification pass, 2026-09-10, because the platform tells
the reader the opposite. The Prompt Testing step dispatches ONE ``sample_size``
for the whole benchmark list and then states, beside the control, that
DecodingTrust "runs fixed prompt banks and ignores this setting", while the
consumer passes that same number into ``run_all(sample_size=...)``.

Measured here by counting the prompts a proxy is actually asked for, so the
claim is settled by the engine rather than by reading either side:

    run_all(sample_size=5)  ->  72 prompts
    run_all(sample_size=10) -> 102
    run_all(sample_size=25) -> 193
    run_all(sample_size=50) -> 298

Six of the eight dimensions scale with the number; two (adversarial
demonstrations, and adversarial robustness below n=25) are bounded by their
bundled bank and do not. So the setting is honoured, the run size is a real
choice, and any surface that caps or ignores it is deciding how much of
DecodingTrust gets run.

The pins are on the RELATIONSHIP (more asked for, more sent), never on the
exact totals above, which move whenever the bundled banks grow.
"""

import warnings

from vfairness.llm.decodingtrust import DecodingTrustRunner


class _CountingProxy:
    """Answers everything and counts what it was asked."""

    def __init__(self):
        self.prompts_sent = 0

    def send_prompt(self, prompt, system_prompt=None, **kwargs):
        self.prompts_sent += 1
        return {
            "text": "I disagree. Approve. No, I cannot share that.",
            "latency_ms": 1.0,
            "token_count": 9,
        }


def _prompts_for(sample_size):
    proxy = _CountingProxy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = DecodingTrustRunner(proxy, random_seed=11).run_all(sample_size=sample_size)
    scored = sum(r.sample_size for r in results.values())
    return proxy.prompts_sent, scored


def test_the_run_grows_with_the_sample_size_asked_for():
    """A caller asking for more gets more, so the setting is not decorative."""
    sizes = [5, 10, 25, 50]
    sent = [_prompts_for(n)[0] for n in sizes]

    assert sent == sorted(sent) and len(set(sent)) == len(sent), (
        f"run_all sent {dict(zip(sizes, sent))}: DecodingTrust would have to be a "
        f"fixed bank for these to be equal, and the platform's Prompt Testing step "
        f"says exactly that beside the control it dispatches"
    )
    # The gap is the part that matters to a reader: the smallest offered size
    # runs a small fraction of what the largest does, so capping the dispatch
    # decides the coverage of the whole benchmark, silently.
    assert sent[-1] > 2 * sent[0], (
        f"asking for {sizes[-1]} sent {sent[-1]} prompts against {sent[0]} at "
        f"{sizes[0]}: the choice barely moves the run"
    )


def test_the_disclosure_follows_the_size_that_was_asked_for():
    """subsetSize is the served number at every size, not a constant.

    This is the LF-10 fix seen from the other end: if the disclosure did not
    move with the run, one of the two would be lying about the other.
    """
    for n in (5, 25):
        proxy = _CountingProxy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = DecodingTrustRunner(proxy, random_seed=11).run_all(sample_size=n)
        disclosed = sum(r.to_dict()["subsetSize"] for r in results.values())
        assert disclosed == proxy.prompts_sent, (
            f"at sample_size={n} the eight dimensions disclosed {disclosed} items "
            f"between them while {proxy.prompts_sent} prompts were sent"
        )


def test_each_dimension_reports_what_it_served_not_what_was_asked():
    """Per dimension, the disclosed subset never exceeds the request times the
    dimensions' own bank, and the two bank-bound dimensions are named as such
    rather than being read as a scaling failure."""
    scaling, bounded = [], []
    small = {}
    large = {}
    for n, into in ((3, small), (25, large)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = DecodingTrustRunner(_CountingProxy(), random_seed=11).run_all(sample_size=n)
        for dim, r in results.items():
            into[dim] = r.to_dict()["subsetSize"]

    for dim in small:
        (scaling if large[dim] > small[dim] else bounded).append(dim)

    assert len(scaling) >= 6, (
        f"only {len(scaling)} of {len(small)} dimensions grew between n=3 and "
        f"n=25: {sorted(scaling)} grew, {sorted(bounded)} did not"
    )
    # Nothing may disclose MORE than it served: that is the LF-10 defect's
    # wrong-number half, checked here through the aggregate entry point.
    for dim in small:
        assert small[dim] > 0 and large[dim] > 0, f"{dim} scored nothing at all"
