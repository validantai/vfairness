"""
Embedding Bias Detection (WEAT / SEAT), Workstream E.

Measures bias baked into word / sentence embeddings using the Word Embedding
Association Test (Caliskan, Bryson & Narayanan, 2017, *Science*) and its
sentence-level extension SEAT (May et al., 2019).

The test asks: are two TARGET concept sets (e.g. {career words} vs {family
words}) differentially associated with two ATTRIBUTE sets (e.g. {male terms} vs
{female terms}) in the embedding space? A large positive effect size means the
first target set is more associated with the first attribute set than the second
, i.e. the embedding encodes the stereotype.

Design (matches the library's minimal-deps rule):
  * Works on ANY embedding source via an injected ``embed(words) -> ndarray``
    callable, OR a precomputed ``{word: vector}`` dict. No model download, no
    transformers/torch dependency at import time.
  * Pure numpy + scipy for the permutation test.

This module is NOT imported from the package __init__ (keeps `import vfairness`
light); import explicitly:

    from vfairness.llm.embedding_bias import EmbeddingBiasDetector
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vfairness.evaluation.vfairness_metrics._statistics import (
    detectability,
    min_attainable_p_permutation,
)

#: The threshold _grade calls significant, and the bar a design must clear.
ALPHA = 0.05


@dataclass
class WEATResult:
    """Result of one WEAT/SEAT run."""

    effect_size: float  # Cohen's d style; |d|>0.5 medium, >0.8 large
    p_value: float  # permutation-test p (two-sided on |statistic|)
    test_statistic: float  # sum of per-word association differences
    n_target_a: int
    n_target_b: int
    n_attribute_a: int
    n_attribute_b: int
    severity: str  # info|low|medium|high|critical
    interpretation: str
    method: str = "WEAT"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmbeddingBiasDetector:
    """WEAT / SEAT bias detector for embeddings.

    Parameters
    ----------
    embeddings : dict[str, ndarray], optional
        Precomputed word->vector map.
    embed_fn : callable, optional
        ``embed_fn(list[str]) -> ndarray (n, dim)`` to compute embeddings on
        demand. Exactly one of ``embeddings`` / ``embed_fn`` must be given.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: embedding_bias_detector. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        embeddings: Optional[Dict[str, "np.ndarray"]] = None,
        embed_fn: Optional[Callable[[Sequence[str]], "np.ndarray"]] = None,
    ):
        if (embeddings is None) == (embed_fn is None):
            raise ValueError("Provide exactly one of `embeddings` or `embed_fn`.")
        self._embeddings = embeddings
        self._embed_fn = embed_fn

    # vector access

    def _vectors(self, words: Sequence[str]) -> "np.ndarray":
        if self._embed_fn is not None:
            vecs = np.asarray(self._embed_fn(list(words)), dtype=float)
            if vecs.ndim != 2 or len(vecs) != len(words):
                raise ValueError("embed_fn must return one row per input word.")
            return vecs
        # __init__ enforces exactly one of embed_fn / embeddings is set, so
        # reaching here (embed_fn is None) guarantees embeddings is not None.
        assert self._embeddings is not None
        missing = [w for w in words if w not in self._embeddings]
        if missing:
            raise KeyError(
                f"Words not in embedding map: {missing[:5]}" + (" ..." if len(missing) > 5 else "")
            )
        return np.asarray([self._embeddings[w] for w in words], dtype=float)

    # math

    @staticmethod
    def _cosine_matrix(A: "np.ndarray", B: "np.ndarray") -> "np.ndarray":
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return An @ Bn.T

    @staticmethod
    def _association(w_vec_idx: int, sim_a: "np.ndarray", sim_b: "np.ndarray") -> float:
        # s(w, A, B) = mean cos(w, a) - mean cos(w, b)
        return float(np.mean(sim_a[w_vec_idx]) - np.mean(sim_b[w_vec_idx]))

    # WEAT

    def weat(
        self,
        target_a: Sequence[str],
        target_b: Sequence[str],
        attribute_a: Sequence[str],
        attribute_b: Sequence[str],
        n_permutations: int = 10000,
        random_state: int = 42,
    ) -> WEATResult:
        """Run WEAT.

        target_a/target_b : the two concept sets being compared (e.g. career vs family).
        attribute_a/attribute_b : the two demographic attribute sets (e.g. male vs female terms).

        A positive effect size means target_a is more associated with
        attribute_a (and target_b with attribute_b) than chance.
        """
        for name, s in [
            ("target_a", target_a),
            ("target_b", target_b),
            ("attribute_a", attribute_a),
            ("attribute_b", attribute_b),
        ]:
            if len(s) == 0:
                raise ValueError(f"{name} must be non-empty.")

        Xt = self._vectors(target_a)
        Yt = self._vectors(target_b)
        A = self._vectors(attribute_a)
        B = self._vectors(attribute_b)

        # Per-word association s(w, A, B) for every target word.
        sx_a = self._cosine_matrix(Xt, A)
        sx_b = self._cosine_matrix(Xt, B)
        sy_a = self._cosine_matrix(Yt, A)
        sy_b = self._cosine_matrix(Yt, B)
        s_x = np.mean(sx_a, axis=1) - np.mean(sx_b, axis=1)  # (n_x,)
        s_y = np.mean(sy_a, axis=1) - np.mean(sy_b, axis=1)  # (n_y,)

        test_statistic = float(np.sum(s_x) - np.sum(s_y))

        # Effect size: (mean_x - mean_y) / pooled std of all per-word assocs.
        #
        # `or 1e-12` fired exactly when there was nothing to compare. A zero
        # spread means every target word has the identical association with the
        # two attribute sets, which is the signature of an embedding source that
        # returned constant or zero vectors (a dead backend), and dividing a
        # zero numerator by 1e-12 produced a clean effect_size of 0.0 to sit
        # beside the permutation test's p of 1.0. Measured 2026-09-10 with an
        # embed_fn returning all-zero vectors: effect_size=0.0, p=1.0, severity
        # "info", "No statistically significant association bias", and not one
        # warning. An undefined standardised effect is NaN, which the existing
        # could_not_check branch of _grade already knows how to say.
        all_s = np.concatenate([s_x, s_y])
        spread = float(np.std(all_s, ddof=1)) if len(all_s) > 1 else 0.0
        degenerate = not math.isfinite(spread) or spread <= 0.0
        if degenerate:
            zero_norm = int(
                np.count_nonzero(np.linalg.norm(np.concatenate([Xt, Yt, A, B]), axis=1) == 0.0)
            )
            warnings.warn(
                f"EmbeddingBiasDetector.weat: every target word has the identical "
                f"association with the two attribute sets (spread {spread}), so the "
                f"standardised effect size has a zero denominator and is undefined. "
                f"Reporting effect_size=nan (could not check), NOT 0.0, which reads as a "
                f"measured absence of association. "
                + (
                    f"{zero_norm} of {len(Xt) + len(Yt) + len(A) + len(B)} supplied "
                    f"vectors have zero length, which is what a dead embedding backend "
                    f"looks like."
                    if zero_norm
                    else "Check that the embedding source returns distinct vectors."
                ),
                UserWarning,
                stacklevel=2,
            )
            effect_size = float("nan")
        else:
            effect_size = (float(np.mean(s_x)) - float(np.mean(s_y))) / spread

        p_value, min_attainable = self._permutation_p(
            s_x, s_y, test_statistic, n_permutations, random_state
        )
        detectable, note = detectability(min_attainable, n_family=1, alpha=ALPHA)
        if detectable is False:
            warnings.warn(
                f"EmbeddingBiasDetector.weat: with {len(target_a)} and {len(target_b)} "
                f"target words the permutation test could NOT have reached {ALPHA} for "
                f"any embedding at all. {note} Grading this run 'could_not_check', NOT "
                f"'no significant association bias'.",
                UserWarning,
                stacklevel=2,
            )
        elif detectable is None:
            # BGL5 A-llm-3: this used to say "could NOT have reached ALPHA for any
            # embedding at all", which is a claim about the DESIGN, for a run in
            # which the test itself did not produce a number. A could-not-compute
            # is not a design-floor finding.
            warnings.warn(
                f"EmbeddingBiasDetector.weat: the permutation test produced no p-value "
                f"at all (p={p_value}), so whether this design could ever reach {ALPHA} "
                f"is unknown too. {note} Grading this run 'could_not_check', NOT 'no "
                f"significant association bias'. Check that the embedding source returns "
                f"finite vectors.",
                UserWarning,
                stacklevel=2,
            )

        severity, interpretation = self._grade(effect_size, p_value, detectable, note)
        return WEATResult(
            effect_size=effect_size,
            p_value=p_value,
            test_statistic=test_statistic,
            n_target_a=len(target_a),
            n_target_b=len(target_b),
            n_attribute_a=len(attribute_a),
            n_attribute_b=len(attribute_b),
            severity=severity,
            interpretation=interpretation,
            method="WEAT",
            notes=([note] if note else []),
        )

    # SEAT is WEAT applied to SENTENCE embeddings (templated words). The math is
    # identical; the SEAT label is only honest when the inputs are sentence
    # resolved (the embed_fn / embeddings map turns full sentence templates into
    # vectors). On plain word vectors this is WEAT, so we keep method="WEAT" and
    # attach a clarifying note rather than stamp a misleading SEAT badge.
    def seat(self, *args, sentence_resolved: bool = False, **kwargs) -> WEATResult:
        """Sentence Encoder Association Test, WEAT on sentence embeddings.

        Pass sentence-templated strings as the target/attribute sets, and set
        ``sentence_resolved=True`` once the embed_fn / embeddings map resolves
        those sentences to vectors. When the inputs are plain word vectors this
        is WEAT (not SEAT): the method stays "WEAT" and a note is added.
        """
        result = self.weat(*args, **kwargs)
        if sentence_resolved:
            result.method = "SEAT"
        else:
            result.notes.append(
                "Requested as SEAT but inputs were not flagged sentence-resolved; "
                "this is WEAT computed on the supplied word vectors, not a "
                "sentence-encoder association test."
            )
        return result

    # permutation test

    def _permutation_p(
        self, s_x, s_y, observed, n_permutations, random_state
    ) -> Tuple[float, Optional[float]]:
        """Two-sided permutation p: P(|statistic| >= |observed|) under random
        partitions of the pooled target associations. Two-sided so that a
        stereotype running in EITHER direction is detected; a one-sided test
        would silently miss a strong negative-direction association (it would
        report p ~ 1.0, severity "info"). Uses exact enumeration for small
        sets, Monte-Carlo sampling otherwise.

        Returns ``(p_value, min_attainable_p)``. The second value is the DESIGN
        FLOOR: WEAT's null is the set of ways to split the pooled target words
        in two, and with 3 words against 3 there are only C(6,3) = 20 of them,
        so the smallest p the test can return is 2/20 = 0.1, above the 0.05 it
        is graded against. Measured 2026-09-10 on a 3-vs-3 embedding built with
        a planted stereotype: effect size +1.83 (far past Cohen's "large"),
        p=0.100, severity "info", captioned "No statistically significant
        association bias". Caliskan's own WEAT word sets run 8 to 25 words per
        side for exactly this reason.
        """
        pooled = np.concatenate([s_x, s_y])
        # BGL5 A-llm-3 (2026-09-27). A TEST THAT COULD NOT RUN MAY NOT PUT A
        # NUMBER ON THE RECORD, and this one put the most extreme number the
        # scale has. Every comparison below is `value >= abs_observed`, and with
        # a NaN observed statistic each one is False, so `count` stayed 0 and the
        # exact branch returned a finite p of 0.0. Measured on 8 words a side
        # with an embedding map of all-NaN vectors, and again with a single NaN
        # vector among healthy ones, and again with all-inf vectors:
        #
        #   effect_size nan, test_statistic nan, severity 'could_not_check',
        #   p_value 0.0   <- and WEATResult.to_dict() carried that 0.0 out
        #
        # p = 0 is arithmetically impossible for this test: the observed split is
        # itself one of the enumerated partitions, so count is at least 1
        # whenever the statistic is finite. After: (nan, None), which
        # detectability reads as could-not-check and _grade reports as
        # 'could_not_check' with a warning, and the all-zero backend still
        # returns its real p of 1.0.
        if not np.all(np.isfinite(pooled)) or not math.isfinite(float(observed)):
            return float("nan"), None
        n = len(pooled)
        k = len(s_x)
        total_sum = float(np.sum(pooled))
        abs_observed = abs(observed) - 1e-12

        # statistic for a partition = sum(A) - sum(B) = 2*sum(A) - total
        def stat_from_idx(idx):
            return 2.0 * float(np.sum(pooled[list(idx)])) - total_sum

        from math import comb

        n_combos = comb(n, k)
        if n_combos <= 20000:  # exact
            count = 0
            ceiling = abs(float(observed))
            ties = 0
            stats_abs = []
            for idx in itertools.combinations(range(n), k):
                value = abs(stat_from_idx(idx))
                stats_abs.append(value)
                if value >= abs_observed:
                    count += 1
            ceiling = max([ceiling] + stats_abs)
            ties = sum(1 for v in stats_abs if v >= ceiling - 1e-12)
            # The observed split is itself one of the enumerated partitions, so
            # its p can never fall below the share that ties the most extreme
            # value the statistic can take here.
            return count / n_combos, ties / n_combos

        rng = np.random.default_rng(random_state)
        count = 0
        draws = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            perm_idx = rng.choice(n, size=k, replace=False)
            draws[i] = abs(stat_from_idx(perm_idx))
            if draws[i] >= abs_observed:
                count += 1
        ceiling = max(float(draws.max()), abs(float(observed)))
        ties = int(np.sum(draws >= ceiling - 1e-12))
        min_attainable = (1.0 + ties) / (1.0 + n_permutations)
        resample_floor = min_attainable_p_permutation(n_permutations)
        if resample_floor is not None:
            min_attainable = max(min_attainable, resample_floor)
        return (count + 1) / (n_permutations + 1), min_attainable

    # grading

    @staticmethod
    def _grade(
        effect_size: float,
        p_value: float,
        detectable: Optional[bool] = True,
        detectability_note: str = "",
    ):
        mag = abs(effect_size)
        # `nan < 0.05` is False, so an untestable comparison fell onto the
        # not-significant side and was captioned "No statistically significant
        # association bias" - the same shape as H-13, where the larger the
        # untested gap, the more reassuring the sentence became.
        if not math.isfinite(p_value) or not math.isfinite(effect_size):
            warnings.warn(
                f"embedding association test: effect size {effect_size} / p-value "
                f"{p_value} is not finite, so significance was NOT tested. Reporting "
                f"could_not_check rather than 'no significant bias'.",
                UserWarning,
                stacklevel=2,
            )
            return (
                "could_not_check",
                f"COULD NOT CHECK: the association test did not produce a usable result "
                f"(effect size {effect_size}, p={p_value}). This is neither a pass nor a "
                f"fail.",
            )
        # A test that RAN but could never have reached ALPHA is not a
        # not-significant finding: see _permutation_p for the measurement.
        if detectable is not True:
            return (
                "could_not_check",
                f"COULD NOT CHECK: the association measures {effect_size:+.2f} "
                f"(p={p_value:.3f}), but the word sets are too small for this "
                f"permutation test to reach {ALPHA} for ANY embedding. "
                f"{detectability_note} This is neither a pass nor a fail; use larger "
                f"target sets (Caliskan et al. use 8 to 25 words per side).",
            )

        significant = p_value < ALPHA
        if not significant:
            return (
                "info",
                f"No statistically significant association bias "
                f"(effect size {effect_size:+.2f}, p={p_value:.3f}).",
            )
        if mag >= 1.0:
            sev = "critical"
        elif mag >= 0.8:
            sev = "high"
        elif mag >= 0.5:
            sev = "medium"
        elif mag >= 0.2:
            sev = "low"
        else:
            sev = "info"
        direction = (
            "the first concept set with the first attribute group"
            if effect_size > 0
            else "the first concept set with the second attribute group"
        )
        return (
            sev,
            f"Significant stereotypical association (effect size {effect_size:+.2f}, "
            f"p={p_value:.3f}): the embedding links {direction}. "
            f"|d|>=0.5 medium, >=0.8 large (Cohen).",
        )
