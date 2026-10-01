"""VA-50 / VA-10: the groundedness (validity) scorer.

The generative counterpart to the fairness scorers: does an answer stay grounded
in, and faithful to, its retrieved sources? Unlike the ``TextScorer`` protocol
(a bare ``text -> float``), groundedness needs the retrieved context, so it
returns a ``GroundednessResult`` dataclass.

FAIL-CLOSED BY CONSTRUCTION. The scorer follows a degradation ladder with a hard
floor:

    owned sidecar detector (VA-21)  ->  self-hosted LLM judge (VA-10)  ->  REFUSE

It never falls to a lexical placeholder in a shipped grade. When no rung is
wired, ``score()`` returns a result with ``available=False`` and ``value=None``
(NOT 0.0), and warns once. This mirrors the ``SidecarUnavailableWarning``
"zeros are NOT real scores" contract in ``vfairness.llm.scorers``: a missing
scorer must read as "not measured", never as "clean".

The interim judge (VA-10) is a self-hosted permissive model (Mistral Small 3.2,
European/Apache, chosen 2026-08-15) behind the existing ``LLMJudgeScorer`` over
``LLMApiProxy``; wiring it is a follow-up. The owned detector (VA-21) is a
fine-tuned EuroBERT/ModernBERT head hosted out of process via the sidecar.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from vfairness.llm._base import SerializableMixin

# The frozen VG metric ids. These MUST match the consuming platform's
# validity metric contract (VG-001..VG-007). Do not renumber.
VG_GROUNDEDNESS = "VG-001"
VG_FAITHFULNESS = "VG-002"
VG_CONTEXT_PRECISION = "VG-003"
VG_CONTEXT_RECALL = "VG-004"
VG_HALLUCINATION_RATE = "VG-005"
VG_CITATION_ACCURACY = "VG-006"
VG_ANSWER_CORRECTNESS = "VG-007"


class GoldUnsupportedWarning(RuntimeWarning):
    """``score(gold=...)`` was given a reference answer that NOTHING here reads.

    R6-2 (2026-09-10): ``gold`` was in the published signature (and in
    docs/API_REFERENCE.md) and appeared nowhere else in this module. Measured:
    ``gold=None``, ``gold='Paris.'``, ``gold='COMPLETELY WRONG GOLD ANSWER'`` and
    ``gold=12345`` all returned the identical result (value=0.5, faithfulness=0.5,
    note='stub') from the same stub judge. It is not forwarded to the judge (the
    ``GroundednessJudge`` protocol has no such argument) and is not stored on the
    result, so a caller supplying reference answers for VG-004 (context recall) or
    VG-007 (answer correctness) got neither, and nothing said so.

    It is REFUSED here rather than accepted and dropped. The refusal is a warning
    plus a line in ``GroundednessResult.note``, NOT an exception, because the
    documented task envelope ``vfairness_validity_run`` accepts an optional
    ``reference_answer`` per record and forwards it here
    (``operations/validity/task_handlers.py``). Measured: with a raise in this
    position, that handler dies on the first record carrying one and takes the
    VG-001 measurement of every other record in the batch with it, uncaught, with
    no TaskResult at all. The note travels into that handler's JSON output, so the
    refusal reaches a reader even where a warning would not. Promote it if you
    want the strict behaviour::

        warnings.simplefilter("error", GoldUnsupportedWarning)
    """


#: Appended to ``GroundednessResult.note`` whenever ``gold`` is supplied (R6-2).
GOLD_REFUSAL_NOTE = (
    "gold= REFUSED: answer correctness (VG-007) and context recall (VG-004) are "
    "not implemented; the reference answer was NOT used and nothing here scored it"
)


class ValidityScorerUnavailableWarning(RuntimeWarning):
    """No groundedness scorer is wired, so ``score()`` refuses rather than fabricate.

    Custom category (like ``SidecarUnavailableWarning``) so strict-warnings test
    runs can filter it without muting all RuntimeWarnings. Emitted once per
    scorer instance, never at import time.
    """


@runtime_checkable
class GroundednessJudge(Protocol):
    """Minimal contract for a judge rung (VA-10) or an owned detector (VA-21).

    Returns a dict with at least ``groundedness`` in [0, 1] (or None to refuse),
    and optionally ``faithfulness``, ``context_precision``, ``unsupported_spans``,
    ``model_id`` and ``note``.
    """

    def judge_groundedness(
        self, *, answer: str, contexts: List[str], question: Optional[str], language: str
    ) -> Dict[str, Any]: ...


@dataclass
class GroundednessResult(SerializableMixin):
    """One record's validity verdict. ``value=None`` means NOT measured.

    ``available=False`` marks a fail-closed result: nothing here is a real score,
    and it must never feed the grade or seal.
    """

    value: Optional[float] = None  # VG-001 groundedness in [0, 1]
    # Per-record claim-level unsupported fraction (1 - value). NOT the aggregate VG-005:
    # aggregate_validity publishes VG-005 as the SHARE OF ANSWERS with unsupported
    # content (answer incidence), which is not the mean of this per-record field.
    hallucination_rate: Optional[float] = None
    faithfulness: Optional[float] = None  # VG-002
    context_precision: Optional[float] = None  # VG-003
    positive_class: str = "unsupported"
    supported: Optional[bool] = None
    unsupported_spans: List[str] = field(default_factory=list)  # evidence for VG-006
    language: str = "unknown"
    scorer_tier: str = "none"  # 'owned_sidecar' | 'llm_judge' | 'none'
    model_id: Optional[str] = None
    available: bool = False  # False => value is not a real measurement
    note: str = ""


class GroundednessScorer:
    """Context-aware, fail-closed groundedness scorer.

    Pass an owned ``sidecar`` (VA-21) and/or an LLM ``judge`` (VA-10). With
    neither, ``score()`` refuses. Inspect ``.available`` before trusting results.

    ``score(gold=...)`` is REFUSED, not honoured: see ``GoldUnsupportedWarning``.
    """

    def __init__(
        self,
        judge: Optional[GroundednessJudge] = None,
        sidecar: Optional[GroundednessJudge] = None,
    ) -> None:
        self._judge = judge
        self._sidecar = sidecar
        self._warned = False
        self._warned_gold = False

    @property
    def available(self) -> bool:
        return self._sidecar is not None or self._judge is not None

    def score(
        self,
        answer: str,
        contexts: List[str],
        *,
        question: Optional[str] = None,
        gold: Optional[str] = None,
        language: str = "unknown",
    ) -> GroundednessResult:
        """Score one answer against its contexts. ``gold`` is refused, see below.

        Args:
            answer: The generated answer to check.
            contexts: The retrieved passages the answer must stay inside.
            question: Optional prompt, passed to the rung for VG-003.
            gold: NOT IMPLEMENTED. A reference answer would feed VG-004 /
                VG-007, which no rung here computes. Supplying it warns
                (``GoldUnsupportedWarning``) and stamps ``GOLD_REFUSAL_NOTE``
                into ``result.note``; it never silently changes a score
                (R6-2, 2026-09-10).
            language: Language tag recorded on the result.
        """
        # R6-2: refuse the unread parameter at the point of use, on EVERY return
        # path below, before any of them can hand back a result that looks like
        # it took the reference answer into account.
        if gold is not None:
            self._warn_gold_unsupported()

        if not answer or not str(answer).strip():
            return self._tag_gold(
                GroundednessResult(language=language, available=False, note="empty answer"), gold
            )

        # G10, 2026-09-30. TOTAL LOSS OF THE SOURCES WAS NOT REFUSED WHILE TOTAL LOSS
        # OF THE ANSWER WAS. Groundedness is defined only relative to the retrieved
        # passages the answer must stay inside (see the module docstring), so with no
        # passage there is no quantity to measure. The empty-ANSWER door above was
        # closed and this, the other half of the same relation, was open: measured
        # before this change, with a rung that answers 1.0,
        # ``score("The capital is Paris.", [])`` returned available=True, value=1.0,
        # hallucination_rate=0.0, supported=True, i.e. a PERFECT groundedness
        # measurement against ZERO sources, and ``[""]`` / ``["  "]`` did the same.
        # A retrieval outage over a 100-record batch therefore aggregated to VG-001
        # mean 1.0 and a VG-005 share of 0.0, a clean pass computed from nothing.
        # ``contexts=None`` failed closed only by accident, as a TypeError raised
        # INSIDE the rung, so the note blamed the rung for the caller's input.
        #
        # Normalise once here and refuse when nothing usable survives, ABOVE the rung
        # dispatch, so the precondition is checked on every rung (owned sidecar, LLM
        # judge) rather than once per branch. The rungs then receive the normalised
        # list, never a None or a list of blanks.
        usable_contexts = _usable_contexts(contexts)
        if not usable_contexts:
            self._warn_unavailable()
            return self._tag_gold(
                GroundednessResult(
                    language=language,
                    available=False,
                    note=(
                        "COULD NOT CHECK: no retrieved context was supplied, so "
                        "groundedness (VG-001) is undefined for this record. This is "
                        "NOT a grounded result and must not feed the grade or seal."
                    ),
                ),
                gold,
            )

        # Rung 1: owned sidecar detector (VA-21). Not yet wired.
        if self._sidecar is not None:
            return self._tag_gold(
                self._run_rung(
                    self._sidecar, "owned_sidecar", answer, usable_contexts, question, language
                ),
                gold,
            )

        # Rung 2: self-hosted LLM judge (VA-10).
        if self._judge is not None:
            return self._tag_gold(
                self._run_rung(
                    self._judge, "llm_judge", answer, usable_contexts, question, language
                ),
                gold,
            )

        # Rung 3: REFUSE. Fail-closed: no fabricated number.
        self._warn_unavailable()
        return self._tag_gold(
            GroundednessResult(
                language=language,
                scorer_tier="none",
                available=False,
                note=(
                    "No groundedness scorer wired (no owned detector, no LLM judge). "
                    "This is not a measurement."
                ),
            ),
            gold,
        )

    @staticmethod
    def _tag_gold(result: GroundednessResult, gold: Optional[str]) -> GroundednessResult:
        """Record the gold refusal on the result itself (R6-2).

        A warning does not survive into ``handle_validity_run``'s JSON, and the
        note does; a refusal a reader never sees is the same defect one layer up.
        """
        if gold is not None:
            result.note = (
                f"{result.note}; {GOLD_REFUSAL_NOTE}" if result.note else GOLD_REFUSAL_NOTE
            )
        return result

    def _warn_gold_unsupported(self) -> None:
        """Warn once per scorer, like ``_warn_unavailable`` (a batch is one signal)."""
        if not self._warned_gold:
            self._warned_gold = True
            warnings.warn(
                "GroundednessScorer.score(gold=...) is NOT implemented: the "
                "reference answer is not forwarded to any rung and no VG-004 "
                "(context recall) or VG-007 (answer correctness) score is "
                "produced. The returned result measures groundedness only. "
                'Raise on it with warnings.simplefilter("error", '
                "GoldUnsupportedWarning).",
                GoldUnsupportedWarning,
                stacklevel=3,
            )

    def _run_rung(
        self,
        rung: GroundednessJudge,
        tier: str,
        answer: str,
        contexts: List[str],
        question: Optional[str],
        language: str,
    ) -> GroundednessResult:
        try:
            verdict = rung.judge_groundedness(
                answer=answer, contexts=contexts, question=question, language=language
            )
        except Exception as exc:  # a rung failure must fail closed, never fabricate.
            self._warn_unavailable()
            return GroundednessResult(
                language=language,
                scorer_tier=tier,
                available=False,
                note=f"{tier} error, refusing: {exc}",
            )

        val = verdict.get("groundedness")
        if val is None:
            return GroundednessResult(
                language=language,
                scorer_tier=tier,
                available=False,
                note=f"{tier} returned no score, refusing",
            )
        # Central finite + [0,1] guard for EVERY rung (not just the LLM judge, which
        # pre-sanitizes). A non-numeric/NaN/out-of-range score from a future owned
        # sidecar (VA-21) must fail closed here, never crash the batch (float("abc"))
        # nor poison the mean (nan) nor pass an out-of-range value (1.5 -> hallucination
        # rate -0.5). This is the single place the invariant is enforced.
        #
        # Reject a boolean FIRST: Python bool is a subclass of int and float(True)==1.0,
        # so {"groundedness": True} from any rung would clamp to a perfect 1.0 fail-open
        # (a garbage reply sealing a PASS). Mirrors the same rejection in judge.py; that
        # guard only covers the LLM-judge rung, this covers every rung including the
        # not-yet-wired owned sidecar (VA-21).
        if isinstance(val, bool):
            self._warn_unavailable()
            return GroundednessResult(
                language=language,
                scorer_tier=tier,
                available=False,
                note=f"{tier} returned a non-numeric (boolean) score, refusing",
            )
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = float("nan")
        if not math.isfinite(val):
            self._warn_unavailable()
            return GroundednessResult(
                language=language,
                scorer_tier=tier,
                available=False,
                note=f"{tier} returned a non-finite score, refusing",
            )
        # Refuse a finite but out-of-range score (e.g. 47.0, 1.4). It is equally proof
        # the rung did not follow the [0,1] rubric, and clamping UP to the most
        # favorable bound (1.0) is an asymmetric fail-open: NaN/Infinity already fail
        # closed here, so a finite out-of-range value must too. Only absorb tiny
        # floating-point overshoot below. Mirrors judge.py.
        if val < -1e-6 or val > 1.0 + 1e-6:
            self._warn_unavailable()
            return GroundednessResult(
                language=language,
                scorer_tier=tier,
                available=False,
                note=f"{tier} returned an out-of-range score, refusing",
            )
        val = max(0.0, min(1.0, val))
        # Bind ONCE, then coerce every element to str. The field is List[str], and a
        # scorer is free to hand back a list of dicts or numbers; the previous form
        # also isinstance-checked a SEPARATE .get() call from the one it used, so the
        # guard proved nothing about the value actually stored. Mirrors the same
        # hardening on unsupported_claims in validity/judge.py.
        spans_raw = verdict.get("unsupported_spans")
        spans = [str(s) for s in spans_raw] if isinstance(spans_raw, list) else []
        # G10, 2026-09-30. The [0,1] guard above covered `groundedness` ONLY, so the
        # two SECONDARY metric fields, VG-002 (faithfulness) and VG-003 (context
        # precision), reached the result through a bare float() with no finiteness,
        # no range and no boolean check. Measured before this change, with a rung
        # returning {"groundedness": 0.9, "faithfulness": nan,
        # "context_precision": 47.0}: available=True, faithfulness=nan and
        # context_precision=47.0 on the result; `to_json()` emitted the bare token
        # `NaN`, which is not valid JSON (a strict parser rejects the WHOLE
        # `vfairness_validity_run` envelope, since the handler publishes every record
        # via `r.to_dict()`), and `_opt_float(True)` returned a perfect 1.0 from a
        # boolean, the exact fail-open the groundedness guard above exists to close.
        # `aggregate_validity` is shielded today only by the VA-74 emission
        # allowlist, so the nan would start poisoning `_mean` on the day the
        # allowlist widens. An unusable value is now NOT MEASURED (None) and the
        # reason is stamped into the note, never dropped silently.
        faithfulness, faith_reason = _unit_float_or_refuse(
            verdict.get("faithfulness"), "faithfulness (VG-002)"
        )
        context_precision, ctx_reason = _unit_float_or_refuse(
            verdict.get("context_precision"), "context_precision (VG-003)"
        )
        # `.get(key, default)` does NOT fire when the key is PRESENT holding None, so
        # `str(verdict.get("note", ""))` minted the literal note "None" for
        # {"note": None}. Read it None-aware instead.
        raw_note = verdict.get("note")
        note = "" if raw_note is None else str(raw_note)
        for reason in (faith_reason, ctx_reason):
            if reason:
                note = f"{note}; {reason}" if note else reason
        return GroundednessResult(
            value=val,
            hallucination_rate=1.0 - val,
            faithfulness=faithfulness,
            context_precision=context_precision,
            supported=val >= 0.5,
            unsupported_spans=spans,
            language=language,
            scorer_tier=tier,
            model_id=verdict.get("model_id"),
            available=True,
            note=note,
        )

    def _warn_unavailable(self) -> None:
        if not self._warned:
            self._warned = True
            warnings.warn(
                "No groundedness scorer available (or the wired rung failed); "
                "score() is refusing and returning value=None. This is NOT a "
                "clean / grounded result. Wire an owned detector (VA-21) or an LLM "
                "judge (VA-10), or inspect .available before trusting results.",
                ValidityScorerUnavailableWarning,
                stacklevel=3,
            )


def _usable_contexts(contexts: Any) -> List[str]:
    """The retrieved passages that actually carry text, or an empty list (G10).

    Absence has many doors here: ``None``, ``[]``, ``[""]``, ``["  "]``, and a
    non-iterable handed in by a caller. All of them mean the same thing, that there
    is no source to be grounded in, and they must all reach the same refusal rather
    than one of them crashing inside a rung. A non-string passage is coerced (a rung
    may be handed dicts by a caller) but only counts if it leaves visible text; the
    literal strings ``'None'`` and ``'nan'`` are NOT treated as absent, because a
    real passage may legitimately contain them.
    """
    if contexts is None or isinstance(contexts, (str, bytes)):
        # A bare string is a caller error, not one passage: honour it as one passage
        # only when it carries text, and never iterate it character by character.
        text = "" if contexts is None else str(contexts)
        return [text] if text.strip() else []
    try:
        items = list(contexts)
    except TypeError:
        return []
    out: List[str] = []
    for c in items:
        text = c if isinstance(c, str) else str(c)
        if text.strip():
            out.append(text)
    return out


def _unit_float_or_refuse(v: Any, field: str) -> "tuple[Optional[float], Optional[str]]":
    """Coerce a secondary rung score to a [0, 1] float, or refuse it with a reason.

    Returns ``(value, None)`` when the rung supplied a usable number, ``(None, None)``
    when it supplied nothing at all (an absent optional metric is not a refusal), and
    ``(None, reason)`` when it supplied something that cannot be a score. The reason
    is the third state: a reader of ``GroundednessResult.note`` sees that the field is
    None BECAUSE the rung's value was rejected, not because the rung stayed quiet.

    Applies the same three rejections the ``groundedness`` guard applies: a boolean
    (``float(True) == 1.0`` would read as a perfect score), a non-numeric or non-finite
    value, and a finite value outside [0, 1] (clamping up to 1.0 is a fail-open).
    """
    if v is None:
        return None, None
    if isinstance(v, bool):
        return None, f"{field} NOT MEASURED: rung returned a boolean, refusing"
    try:
        num = float(v)
    except (TypeError, ValueError):
        return None, f"{field} NOT MEASURED: rung returned a non-numeric value, refusing"
    if not math.isfinite(num):
        return None, f"{field} NOT MEASURED: rung returned a non-finite value, refusing"
    if num < -1e-6 or num > 1.0 + 1e-6:
        return None, f"{field} NOT MEASURED: rung returned {num!r}, outside [0, 1], refusing"
    return max(0.0, min(1.0, num)), None


def groundedness_scorer_status(scorer: GroundednessScorer) -> Dict[str, Any]:
    """Report the active rung and its quality tier, like ``scorer_status()``."""
    if scorer._sidecar is not None:
        tier, quality = "owned_sidecar", "production"
    elif scorer._judge is not None:
        tier, quality = "llm_judge", "interim"
    else:
        tier, quality = "none", "unavailable"
    return {"tier": tier, "quality": quality, "available": scorer.available}
