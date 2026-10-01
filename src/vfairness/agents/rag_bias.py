"""
RAG retrieval bias detection for AI agents.

Measures whether retrieval-augmented generation (RAG) pipelines introduce
or amplify bias by comparing retrieved document sets and generated outputs
across demographic groups. Bias can enter at the retrieval stage (different
documents retrieved) or the generation stage (different outputs despite
similar documents).
"""

import hashlib
import logging
import math
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


def _is_blank(text: object) -> bool:
    """True when this completion carries no text at all.

    A prompt that came back empty, whitespace-only or as a newline produced NO
    output; it is not a short output. Used by
    :meth:`RAGBiasAnalyzer.analyze_output` to keep "nothing was generated"
    apart from "both groups were answered the same way", which on a [0, 1]
    disparity scale are both 0.0 unless something separates them.
    """
    return not str(text).strip()


@dataclass
class RAGBiasResult(SerializableMixin):
    """Result of a RAG bias analysis comparing two demographic groups.

    Attributes:
        query_id: Unique identifier for this analysis.
        demographic_a: Label for the first demographic group.
        demographic_b: Label for the second demographic group.
        retrieval_disparity: Jaccard distance between retrieved doc sets.
        output_disparity: Divergence score between generated outputs.
        is_retrieval_biased: Whether retrieval disparity exceeds threshold,
            or None when the disparity was not measured (three states, never
            two: biased / not biased / could not check).
        is_output_biased: Whether output disparity exceeds threshold, or None
            when the disparity was not measured.
    """

    query_id: str
    demographic_a: str
    demographic_b: str
    retrieval_disparity: float
    output_disparity: float
    is_retrieval_biased: Optional[bool]
    is_output_biased: Optional[bool]
    metadata: RunMetadata = field(default_factory=RunMetadata)


class RAGBiasAnalyzer:
    """Analyzer for bias in RAG pipelines.

    Compares retrieval results and generated outputs between demographic
    groups to detect bias at both the retrieval and generation stages.

    Args:
        alpha: Statistical significance level, reserved for significance
            testing. NOT used as a disparity threshold.
        similarity_method: "trigram" (default) or "embedding".
        disparity_threshold: Raw disparity level in [0, 1] above which
            retrieval / output disparity is flagged as biased. Default 0.1.

    Note:
        Earlier releases reused ``alpha`` (a significance level) as the raw
        disparity threshold in ``full_analysis``, which flagged bias at a
        meaninglessly low 0.05 disparity by default. Bias flags now use
        ``disparity_threshold`` (default 0.1); ``alpha`` is kept for
        significance only.

    Example:
        >>> analyzer = RAGBiasAnalyzer(disparity_threshold=0.1)
        >>> docs_a = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
        >>> docs_b = [{"id": "doc2"}, {"id": "doc4"}, {"id": "doc5"}]
        >>> retrieval_disp = analyzer.analyze_retrieval(docs_a, docs_b)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: rag_bias_analyzer. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        similarity_method: str = "trigram",
        disparity_threshold: float = 0.1,
    ) -> None:
        self.alpha = alpha
        self.similarity_method = similarity_method
        self.disparity_threshold = disparity_threshold
        logger.info(
            "RAGBiasAnalyzer initialized: alpha=%.3f, method=%s, disparity_threshold=%.3f",
            alpha,
            similarity_method,
            disparity_threshold,
        )
        self._embedder = None
        if similarity_method == "embedding":
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                import warnings

                warnings.warn(
                    "sentence-transformers not installed. Falling back to trigram similarity. "
                    "Install with: pip install sentence-transformers",
                    UserWarning,
                )
                self.similarity_method = "trigram"

    def analyze_retrieval(
        self,
        docs_a: list[dict],
        docs_b: list[dict],
    ) -> float:
        """Compute retrieval set divergence using Jaccard distance.

        Compares the sets of document IDs retrieved for each demographic
        group. Documents without an ``id`` key are identified by a hash of
        their content, so distinct unlabeled documents never collide.

        Args:
            docs_a: Retrieved documents for demographic group A.
            docs_b: Retrieved documents for demographic group B.

        Returns:
            Jaccard distance (1 - Jaccard similarity). Range [0, 1] where
            0 means identical retrieval sets and 1 means no overlap.
        """
        if len(docs_a) == 0 and len(docs_b) == 0:
            # 0.0 on this scale means "identical retrieval sets", the cleanest
            # answer available. Two EMPTY sets are not identical retrieval;
            # nothing was retrieved for either group. Measured 2026-09-08: this
            # returned exactly the same 0.0 as a genuinely identical pair.
            warnings.warn(
                "RAGBiasAnalyzer.analyze_retrieval: neither group retrieved any "
                "documents, so retrieval similarity was not measured. Returning nan, "
                "not 0.0, because 0.0 means identical retrieval.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        ids_a = {self._doc_key(d) for d in docs_a}
        ids_b = {self._doc_key(d) for d in docs_b}

        intersection = len(ids_a & ids_b)
        union = len(ids_a | ids_b)

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

    def analyze_output(
        self,
        outputs_a: list[str],
        outputs_b: list[str],
    ) -> float:
        """Compare output texts between demographic groups.

        Uses character-level n-gram overlap as a lightweight proxy for
        output similarity. For production use, consider replacing with
        embedding-based similarity.

        Args:
            outputs_a: Generated text outputs for demographic group A.
            outputs_b: Generated text outputs for demographic group B.

        Returns:
            Output disparity score in [0, 1]. Higher values indicate
            greater divergence between group outputs. Returns nan when
            NEITHER group produced an output, because on this scale 0.0
            means identical outputs. A group that produced nothing while the
            other answered is a measurement, not a refusal: those pairs score
            1.0, which is the finding.
        """
        n_nonblank_a = sum(1 for text in outputs_a if not _is_blank(text))
        n_nonblank_b = sum(1 for text in outputs_b if not _is_blank(text))

        # BGL-S2 (2026-09-16): this guard tested len(list) == 0, not whether
        # anything was GENERATED, so the docstring promise above was broken by
        # the commonest failure of a generation stage: prompts that came back
        # blank. Measured 2026-09-16: analyze_output(['', '', ''], ['', '', ''])
        # returned 0.0 with zero warnings, as did ['   '] * 3 and ['\n'] * 3,
        # byte-identical to two genuinely identical real answers (verified
        # equal), and full_analysis then shipped output_disparity=0.0 with
        # is_output_biased=False. Three blank completions IS neither group
        # producing an output. Placed ABOVE the embedding branch as well as the
        # trigram one, because both paths share the precondition: an embedder
        # returns a vector for the empty string too, and the cosine of two
        # nothings is the same fabricated agreement.
        if n_nonblank_a == 0 and n_nonblank_b == 0:
            # Same defect as the empty-retrieval case above, one stage later:
            # 0.0 here means "identical outputs", and it was returned for a run
            # that generated nothing for either group. Nothing generated is not
            # agreement, and 0.0 was indistinguishable from two genuinely
            # identical answers. Measured 2026-09-10.
            supplied = len(outputs_a) + len(outputs_b)
            detail = (
                "neither group produced any outputs"
                if supplied == 0
                else f"all {supplied} supplied completion(s) are empty or whitespace"
            )
            warnings.warn(
                f"RAGBiasAnalyzer.analyze_output: {detail}, so output similarity was "
                f"not measured. Returning nan, not 0.0, because 0.0 means identical "
                f"outputs and would read as two groups being answered the same way.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        if len(outputs_a) == 0 or len(outputs_b) == 0:
            return 1.0

        if self.similarity_method == "embedding" and self._embedder is not None:
            emb_a = self._embedder.encode(outputs_a)
            emb_b = self._embedder.encode(outputs_b)
            mean_a = np.mean(emb_a, axis=0)
            mean_b = np.mean(emb_b, axis=0)
            cos_sim = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b))
            return 1.0 - float(cos_sim)

        # Compute average pairwise character trigram Jaccard distance.
        #
        # A pair of two BLANK completions is excluded from the mean and
        # counted, never scored 0.0: _trigram_distance is a pure string
        # metric, where two empty strings really are identical, and the
        # question of whether an output EXISTS is this method's to answer, not
        # the string metric's. A blank against a real answer is NOT excluded:
        # one group being served while the other was not is the finding, and
        # it keeps its 1.0.
        distances = []
        n_unmeasurable_pairs = 0
        for text_a in outputs_a:
            for text_b in outputs_b:
                if _is_blank(text_a) and _is_blank(text_b):
                    n_unmeasurable_pairs += 1
                    continue
                dist = self._trigram_distance(text_a, text_b)
                distances.append(dist)

        if not distances:
            warnings.warn(
                f"RAGBiasAnalyzer.analyze_output: all "
                f"{n_unmeasurable_pairs} completion pair(s) had no text on either "
                f"side, so output similarity was not measured. Returning nan, not "
                f"0.0, because 0.0 means identical outputs.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

        if n_unmeasurable_pairs:
            warnings.warn(
                f"RAGBiasAnalyzer.analyze_output: {n_unmeasurable_pairs} of "
                f"{n_unmeasurable_pairs + len(distances)} completion pair(s) were "
                f"blank on both sides and were excluded from the mean rather than "
                f"scored 0.0 (identical). The disparity rests on the "
                f"{len(distances)} pair(s) where at least one group produced text.",
                UserWarning,
                stacklevel=2,
            )

        return float(np.mean(distances))

    def full_analysis(
        self,
        queries_a: list,
        queries_b: list,
        retrieved_a: list[dict],
        retrieved_b: list[dict],
        outputs_a: list[str],
        outputs_b: list[str],
    ) -> RAGBiasResult:
        """Run full RAG bias analysis across retrieval and output stages.

        Args:
            queries_a: Queries submitted for demographic group A.
            queries_b: Queries submitted for demographic group B.
            retrieved_a: Documents retrieved for group A.
            retrieved_b: Documents retrieved for group B.
            outputs_a: Generated outputs for group A.
            outputs_b: Generated outputs for group B.

        Returns:
            RAGBiasResult with retrieval and output disparity metrics. A
            stage whose disparity could not be measured reports its bias
            flag as None (could not check) rather than False. When exactly
            one group submitted queries, neither stage is measured: both
            disparities are nan and both flags None.
        """
        logger.info("full_analysis: n_queries_a=%d, n_queries_b=%d", len(queries_a), len(queries_b))

        # CHECK 2 (2026-10-01). The bare stage methods cannot tell a group that
        # was queried and got nothing back (a real, total disparity) from a group
        # that was never queried at all, because they only see what came back.
        # This method DOES see the queries, and it ignored them. Measured that
        # day on a run where every query belonged to group A, so group B's
        # queries, documents and outputs were all empty lists:
        #   retrieval_disparity 1.0, output_disparity 1.0,
        #   is_retrieval_biased True, is_output_biased True, no warning
        # the strongest bias finding this scale has, about a group that never
        # entered the pipeline. When exactly one side has queries there is no
        # second group to compare, so neither stage is measured. Both sides
        # empty is left to the stage methods as before: callers that do not
        # record queries pass [] for both, and the stages already refuse a run
        # that retrieved and generated nothing.
        if (len(queries_a) == 0) != (len(queries_b) == 0):
            unqueried = "group_a" if len(queries_a) == 0 else "group_b"
            warnings.warn(
                f"RAGBiasAnalyzer.full_analysis: {unqueried} submitted no queries while "
                f"the other group submitted {max(len(queries_a), len(queries_b))}, so "
                f"there is no second group to compare. Reporting both disparities as "
                f"nan and both bias flags as None (could not check), NOT 1.0 and True, "
                f"which is what a group that was queried and served nothing returns.",
                UserWarning,
                stacklevel=2,
            )
            return RAGBiasResult(
                query_id=str(uuid.uuid4()),
                demographic_a="group_a",
                demographic_b="group_b",
                retrieval_disparity=float("nan"),
                output_disparity=float("nan"),
                is_retrieval_biased=None,
                is_output_biased=None,
                metadata=RunMetadata(
                    parameters={
                        "alpha": self.alpha,
                        "similarity_method": self.similarity_method,
                        "disparity_threshold": self.disparity_threshold,
                        "n_queries_a": len(queries_a),
                        "n_queries_b": len(queries_b),
                        "not_assessed_reason": f"{unqueried} submitted no queries",
                    },
                ),
            )

        retrieval_disparity = self.analyze_retrieval(retrieved_a, retrieved_b)
        output_disparity = self.analyze_output(outputs_a, outputs_b)

        # Bias flags compare raw [0, 1] disparities against
        # disparity_threshold. alpha is a significance level and must not
        # double as a disparity cutoff (it previously did, flagging bias at
        # any disparity above 0.05).
        retrieval_threshold = self.disparity_threshold
        output_threshold = self.disparity_threshold

        # A disparity that was never measured must not be graded. `nan > t` is
        # False for every threshold, so a RAG run that retrieved nothing and
        # generated nothing reported is_retrieval_biased=False and
        # is_output_biased=False: a clean bill of health for a run that
        # produced no evidence at all. Measured 2026-09-10.
        retrieval_biased: Optional[bool]
        if math.isnan(retrieval_disparity):
            warnings.warn(
                "RAGBiasAnalyzer.full_analysis: retrieval disparity was not "
                "measured, so is_retrieval_biased is None (could not check), "
                "not False. `nan > threshold` is False, which reads as an "
                "unbiased retrieval stage.",
                UserWarning,
                stacklevel=2,
            )
            retrieval_biased = None
        else:
            retrieval_biased = bool(retrieval_disparity > retrieval_threshold)

        output_biased: Optional[bool]
        if math.isnan(output_disparity):
            warnings.warn(
                "RAGBiasAnalyzer.full_analysis: output disparity was not "
                "measured, so is_output_biased is None (could not check), not "
                "False. `nan > threshold` is False, which reads as an unbiased "
                "generation stage.",
                UserWarning,
                stacklevel=2,
            )
            output_biased = None
        else:
            output_biased = bool(output_disparity > output_threshold)

        logger.info(
            "full_analysis complete: retrieval_disparity=%.3f, output_disparity=%.3f",
            retrieval_disparity,
            output_disparity,
        )
        return RAGBiasResult(
            query_id=str(uuid.uuid4()),
            demographic_a="group_a",
            demographic_b="group_b",
            retrieval_disparity=float(retrieval_disparity),
            output_disparity=float(output_disparity),
            is_retrieval_biased=retrieval_biased,
            is_output_biased=output_biased,
            metadata=RunMetadata(
                parameters={
                    "alpha": self.alpha,
                    "similarity_method": self.similarity_method,
                    "disparity_threshold": self.disparity_threshold,
                },
            ),
        )

    @staticmethod
    def _doc_key(doc: dict) -> str:
        """Stable identity for a retrieved document.

        Prefers the explicit ``id``. Unlabeled documents fall back to a
        content hash (never a positional index, which made distinct docs at
        the same rank collide and report zero disparity). Documents whose
        content cannot be hashed deterministically fall back to object
        identity, which keeps distinct objects distinct.
        """
        doc_id = doc.get("id")
        if doc_id is not None:
            return f"id:{doc_id}"
        try:
            content = repr(sorted(doc.items()))
        except TypeError:
            return f"obj:{id(doc)}"
        # Not a security hash: a content-dedup key. usedforsecurity=False keeps
        # security scanners (bandit B324) from flagging a non-security SHA1.
        digest = hashlib.sha1(content.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()
        return f"sha1:{digest}"

    @staticmethod
    def _trigram_distance(text_a: str, text_b: str) -> float:
        """Compute character trigram Jaccard distance between two strings.

        Strings shorter than 3 characters have no trigrams, so they use an
        exact-match check first and a character-level Jaccard fallback
        (previously two distinct short strings reported distance 0.0).
        """
        if text_a == text_b:
            return 0.0
        if not text_a or not text_b:
            return 1.0
        if len(text_a) < 3 or len(text_b) < 3:
            chars_a = set(text_a)
            chars_b = set(text_b)
            union = len(chars_a | chars_b)
            if union == 0:
                return 0.0
            return 1.0 - (len(chars_a & chars_b) / union)

        trigrams_a = {text_a[i : i + 3] for i in range(len(text_a) - 2)}
        trigrams_b = {text_b[i : i + 3] for i in range(len(text_b) - 2)}

        if not trigrams_a and not trigrams_b:
            return 0.0

        intersection = len(trigrams_a & trigrams_b)
        union = len(trigrams_a | trigrams_b)

        if union == 0:
            return 0.0

        return 1.0 - (intersection / union)
