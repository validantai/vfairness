"""What model actually answered, recorded rather than assumed (LF-14).

WHY THIS MODULE EXISTS
----------------------
Every behavioural finding this library produces is a statement about the model
it was measured on. A refusal-rate disparity, a toxicity gap, a counterfactual
sensitivity: none of them carries over to a different model, and a provider can
serve a different build under a stable name without telling anyone. So a stored
result whose model identity is unknown has no shelf life, and a stored result
whose identity is *assumed* is worse, because it looks like it has one.

Two facts, kept apart, always:

``configured_model``
    The name the assessor asked for. It is a SETTING. It says what was
    requested and nothing about what served the request.

``reported_model``
    The name the endpoint put on its own response, captured by
    :meth:`vfairness.llm.api_proxy.LLMApiProxy._extract_reported_model`. It is
    ``None`` when the endpoint reported none, and it is NEVER filled in from
    the configured name: doing that would manufacture agreement out of nothing,
    and a reader seeing the two match would take it for confirmation.

Even when they agree, that is the endpoint's own self-report, not an
independent check of what served the request. This module says so in the
statements it produces rather than leaving the caller to remember it.

WHAT THE ENGINE CAN SEE THAT THE PLATFORM CANNOT
------------------------------------------------
The platform's connection test makes ONE request. A run here makes twenty-five
per prompt. If the reported name changes partway through a batch, the provider
swapped the model underneath a measurement, and every statistic computed over
those responses is a statistic over two different systems. That is not a
housekeeping detail; it is the strongest identity signal a run can produce, and
:func:`identity_from_run` is the only place it can be noticed.

Nothing here calls an endpoint, derives a fingerprint, or decides a verdict.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .._not_assessed import NOT_ASSESSED, warn_not_assessed

__all__ = [
    "NOT_ASSESSED",
    "ModelIdentity",
    "IdentityComparison",
    "clean_model_name",
    "identity_from_run",
    "compare_model_identity",
    "describe_model_identity",
]


#: Unicode general categories whose characters put no mark on the page: controls,
#: format characters (the zero-width space, the zero-width non-joiner, the BOM,
#: the soft hyphen, the word joiner), separators, surrogates, private use and
#: unassigned codepoints. A string made only of these RENDERS AS NOTHING.
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Cn", "Co", "Cs", "Zl", "Zp", "Zs"})


def _renders_as_nothing(trimmed: str) -> bool:
    """True when a non-empty string puts no visible mark on the page.

    ``str.strip()`` removes whitespace as Python defines it, which does NOT
    include the Unicode FORMAT characters, so a name of U+200B, U+200C, U+FEFF and
    U+00AD survives it: four characters long, truthy, and invisible.

    The codepoints are named in prose and never pasted as glyphs, the rule
    ``scorers._ASCII_FOLD`` states for the same reason: a reader cannot tell one
    invisible character from another, or from none at all, by eye.
    """
    return all(unicodedata.category(ch) in _INVISIBLE_CATEGORIES for ch in trimmed)


def clean_model_name(value: Any) -> Optional[str]:
    """A recorded model name, or ``None``. Blank and non-string are ``None``.

    A whitespace-only name is not a name. Treating it as one would put an empty
    string where a reader expects an identity and let it compare equal to
    another empty string, which reads as a match.

    A BLANK-RENDERING NAME GOT THROUGH THE ONLY GATE THERE IS (audit wave 2,
    2026-09-29). ``str.strip()`` strips whitespace, and the Unicode FORMAT
    characters are not whitespace, so a name made of U+200B, U+200C, U+FEFF and
    U+00AD survived it: four characters, truthy, and invisible to every reader.
    This function is the one gate between the endpoint's JSON and the identity
    claim, and the docstring above says what it exists to prevent. Measured on
    four responses whose reported name was those four characters:

        n_reporting            0 of 4  ->  4 of 4
        is_consistent          None    ->  True
        the "none of the N responses carried a model name" warning
                               fired   ->  silent
        coverage_statement     "so what answered was not recorded"
                               ->  'The endpoint reported "" on all 4 responses'
        compare_model_identity state='could_not_check'
                               ->  state='same', basis='endpoint_report',
                                   needs_retest=False

    So an endpoint that named nothing was recorded as having named something, and
    a comparison against a later run reported the model unchanged and in no need
    of a retest, which is the exact claim this module exists to refuse.

    A name that renders blank IS NOT SANITISED INTO ONE THAT DOES NOT. Only a
    name that is ENTIRELY invisible is refused; an ordinary name carrying a stray
    zero-width character keeps every codepoint the endpoint sent, so two such
    names compare as DIFFERENT and the run is flagged rather than quietly
    equated. That direction is loud, which is the safe one, and rewriting a
    recorded identity to make it match would be this defect again in the mirror.
    """
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed or _renders_as_nothing(trimmed):
        return None
    return trimmed


@dataclass(frozen=True)
class ModelIdentity:
    """The identity recorded with one run.

    Attributes:
        configured_model: The requested model name, or ``None``.
        reported_model: The name the endpoint reported, or ``None`` when it
            reported none. Never backfilled from ``configured_model``.
        version: A version pin or fingerprint string recorded with the run.
            This module does not compute one.
        reported_names: Every DISTINCT name seen across the run's responses, in
            first-seen order. Length above one means the endpoint did not serve
            one model for the whole run.
        n_responses: How many responses were examined.
        n_reporting: How many of them carried a model name at all.
    """

    configured_model: Optional[str] = None
    reported_model: Optional[str] = None
    version: Optional[str] = None
    reported_names: Tuple[str, ...] = field(default_factory=tuple)
    n_responses: int = 0
    n_reporting: int = 0

    @property
    def is_consistent(self) -> Optional[bool]:
        """Did every reporting response name the same model?

        Three states. ``None`` means no response reported a name, so there is
        nothing to be consistent about; that is not the same as consistent.

        G12, 2026-09-30. THE ANSWER WAS READ OFF A COUNT AND NOT OFF THE NAMES.
        With ``n_reporting`` above zero and ``reported_names`` empty, which is
        what a record reconstituted from a stored row looks like when the names
        column was not persisted, ``len(()) == 1`` answered **False**, i.e. "the
        endpoint did not serve one model for this run" -- a positive finding
        about an endpoint, asserted from a record holding no name at all. The two
        fields can disagree, and when they do the names are the evidence and the
        count is a claim about it, so the absence of names decides. Measured
        before this change: ``ModelIdentity(n_responses=1, n_reporting=1,
        reported_names=()).is_consistent`` -> False, and ``coverage_statement``
        on the same record raised ``IndexError: tuple index out of range`` from a
        report surface.
        """
        if self.n_reporting == 0 or not self.reported_names:
            return None
        return len(self.reported_names) == 1

    @property
    def coverage_statement(self) -> str:
        """What a report should say about how much of the run was identified.

        G12, 2026-09-30. Every sentence below is built from ``reported_names`` and
        the two counts TOGETHER, and nothing checked that they agree. A record
        whose counts claim names it does not carry took the last branch and
        raised ``IndexError: tuple index out of range`` on a report surface; a
        record claiming more reporting responses than it examined got the
        "No responses were examined" sentence while asserting three that did.
        Neither is a statement a reader can act on, so the contradiction is now
        named instead of being resolved in favour of whichever field a branch
        happened to read. ``identity_from_run`` cannot build either state, so no
        real run's wording changes; a record rebuilt from a stored row can.
        """
        if self.n_reporting > self.n_responses or (
            self.n_reporting > 0 and not self.reported_names
        ):
            return (
                f"COULD NOT CHECK: this identity record contradicts itself "
                f"({self.n_reporting} of {self.n_responses} responses recorded as naming a "
                f"model, carrying {len(self.reported_names)} distinct name(s)), so how much "
                f"of the run was identified cannot be stated. Rebuild it from the run's "
                f"responses with identity_from_run."
            )
        if self.n_responses == 0:
            return "No responses were examined, so no model identity was recorded for this run."
        if self.n_reporting == 0:
            return (
                f"None of the {self.n_responses} responses carried a model name, so what answered "
                f"was not recorded. A matching configured name would not establish it."
            )
        if len(self.reported_names) > 1:
            names = ", ".join(f'"{n}"' for n in self.reported_names)
            return (
                f"THE ENDPOINT DID NOT SERVE ONE MODEL FOR THIS RUN: {len(self.reported_names)} "
                f"different model names appeared across {self.n_reporting} of {self.n_responses} "
                f"responses ({names}). Any statistic computed over these responses is computed "
                f"over more than one system, and the run should be repeated before it is reported."
            )
        if self.n_reporting < self.n_responses:
            return (
                f'The endpoint reported "{self.reported_names[0]}" on {self.n_reporting} of '
                f"{self.n_responses} responses and named nothing on the rest. The unnamed "
                f"responses are not evidence that the same model served them."
            )
        return (
            f'The endpoint reported "{self.reported_names[0]}" on all {self.n_responses} '
            f"responses. That is the endpoint's own self-report, not an independent check of "
            f"what served the requests."
        )


def _iter_responses(run: Any) -> Iterable[Mapping[str, Any]]:
    """Yield response mappings from either shape ``send_batch`` can hand back."""
    if isinstance(run, Mapping):
        responses = run.get("responses")
        if isinstance(responses, Sequence) and not isinstance(responses, (str, bytes)):
            for item in responses:
                if isinstance(item, Mapping):
                    yield item
            return
        yield run
        return
    if isinstance(run, Sequence) and not isinstance(run, (str, bytes)):
        for item in run:
            yield from _iter_responses(item)


def identity_from_run(
    responses: Any,
    *,
    configured_model: Optional[str] = None,
    version: Optional[str] = None,
) -> ModelIdentity:
    """Build the model identity for one run from the responses it produced.

    Accepts a single response dict, a list of them, or the list-of-prompts
    shape :meth:`LLMApiProxy.send_batch` returns.

    A run whose responses name more than one model gets a warning, because that
    is a measurement over two systems and no downstream statistic can repair
    it. A run where nobody reported a name gets one too: the identity is
    ``None``, and silence there is exactly what makes a stale result look fresh.

    Args:
        responses: The run's responses, in any of the three shapes above.
        configured_model: The name that was requested, recorded as its own
            separate fact.
        version: A version pin or fingerprint recorded with the run.

    Returns:
        A :class:`ModelIdentity`. ``reported_model`` is set only when exactly
        one distinct name was seen; with several, it stays ``None`` and
        ``reported_names`` carries all of them, because there is no single
        answer to give.
    """
    seen: List[str] = []
    n_responses = 0
    n_reporting = 0
    for response in _iter_responses(responses):
        n_responses += 1
        name = clean_model_name(response.get("reported_model"))
        if name is None:
            continue
        n_reporting += 1
        if name not in seen:
            seen.append(name)

    if n_responses and len(seen) > 1:
        warn_not_assessed(
            "identity_from_run",
            measured=n_reporting,
            total=n_responses,
            unit=f"responses named a model, and they named {len(seen)} DIFFERENT ones",
            requirement="one run must be served by one model for its statistics to describe a system",
            reporting=f"reported_model=None and reported_names={tuple(seen)!r}",
            instead_of="the first name seen",
        )
    elif n_responses and n_reporting == 0:
        warn_not_assessed(
            "identity_from_run",
            measured=0,
            total=n_responses,
            unit="responses carried a model name",
            requirement="the model that answered can only be recorded if the endpoint names it",
            reporting="reported_model=None (could not check)",
            instead_of="the configured model name",
        )

    return ModelIdentity(
        configured_model=clean_model_name(configured_model),
        reported_model=seen[0] if len(seen) == 1 else None,
        version=clean_model_name(version),
        reported_names=tuple(seen),
        n_responses=n_responses,
        n_reporting=n_reporting,
    )


@dataclass(frozen=True)
class IdentityComparison:
    """The answer to "is this still the model that result was measured on?".

    ``state`` is the machine-readable field a re-test trigger branches on, so
    it carries the qualification rather than leaving it to ``statement``:

    ``"same"``
        Every recorded part matched AND the endpoint's own echo was one of
        them. This is the only state that says anything about what ANSWERED.
    ``"changed"``
        A part that could be compared differs. Re-test.
    ``NOT_ASSESSED``
        Nothing comparable, or only the configuration was comparable, or a part
        exists on one side only. Never a match, never a change.
    """

    state: str
    changed_fields: Tuple[str, ...] = ()
    compared: Tuple[str, ...] = ()
    not_compared: Tuple[str, ...] = ()
    one_sided: Tuple[str, ...] = ()
    basis: str = "none"
    statement: str = ""

    @property
    def needs_retest(self) -> bool:
        """Is a stored behavioural finding still attributable to this model?

        True for ``"changed"``. Deliberately FALSE for the not-assessed state:
        an unknown identity is not evidence of a change, and re-running every
        result whose endpoint never named itself would be a permanent loop. The
        report says the identity could not be established instead, which is the
        honest handling.
        """
        return self.state == "changed"


_FIELDS: Tuple[str, ...] = ("configured_model", "reported_model", "version")
_LABELS: Dict[str, str] = {
    "configured_model": "configured model name",
    "reported_model": "model name reported by the endpoint",
    "version": "version or fingerprint",
}


def compare_model_identity(
    recorded: Optional[ModelIdentity],
    current: Optional[ModelIdentity],
) -> IdentityComparison:
    """Compare the identity stored with a result against the current one.

    Mirrors the platform's ``compareModelIdentity`` deliberately, so the two
    halves of LF-14 cannot answer the same question differently. In particular
    a matching CONFIGURED name is not a match: it says the setting did not move.

    Values are compared as exact strings after trimming. A provider that
    changes only the case of its own name is reported as changed, which puts
    the difference in front of a person rather than resolving it here.
    """
    changed: List[str] = []
    compared: List[str] = []
    not_compared: List[str] = []
    one_sided: List[str] = []
    matched_parts: List[str] = []
    one_sided_notes: List[str] = []

    for name in _FIELDS:
        before = clean_model_name(getattr(recorded, name, None))
        after = clean_model_name(getattr(current, name, None))
        if before is None and after is None:
            not_compared.append(name)
            continue
        if before is None or after is None:
            not_compared.append(name)
            one_sided.append(name)
            side = "this run" if before is not None else "the current configuration"
            other = "the current configuration" if before is not None else "this run"
            value = before if before is not None else after
            one_sided_notes.append(
                f' The {_LABELS[name]} was recorded on {side} as "{value}" but not on {other}, '
                f"so the two could not be compared."
            )
            continue
        compared.append(name)
        if before != after:
            changed.append(name)
        else:
            matched_parts.append(f'{_LABELS[name]} "{after}"')

    never = [f for f in not_compared if f not in one_sided]
    never_note = (
        " The "
        + ", ".join(_LABELS[f] for f in never)
        + (" was" if len(never) == 1 else " were")
        + " recorded on neither side, so "
        + ("it was" if len(never) == 1 else "they were")
        + " not compared."
        if never
        else ""
    )
    one_sided_note = "".join(one_sided_notes)

    if changed:
        what = "; ".join(
            f'{_LABELS[f]} "{clean_model_name(getattr(recorded, f))}" is now '
            f'"{clean_model_name(getattr(current, f))}"'
            for f in changed
        )
        return IdentityComparison(
            state="changed",
            changed_fields=tuple(changed),
            compared=tuple(compared),
            not_compared=tuple(not_compared),
            one_sided=tuple(one_sided),
            basis="endpoint_report" if "reported_model" in compared else "configuration_only",
            statement=(
                f"The model identity changed since this run: {what}. A behavioural finding is a "
                f"statement about the model it was measured on, so it does not carry over to the "
                f"model configured now." + one_sided_note + never_note
            ),
        )

    if not compared:
        return IdentityComparison(
            state=NOT_ASSESSED,
            compared=(),
            not_compared=tuple(not_compared),
            one_sided=tuple(one_sided),
            basis="none",
            statement=(
                "No part of the model identity was recorded on both sides, so it is not known "
                "whether the model changed. That is a limit on this evidence, not a sign that "
                "the model is unchanged." + one_sided_note
            ),
        )

    matched = "; ".join(matched_parts)

    # NOTHING BELOW MAY RETURN "same" UNLESS THE ENDPOINT'S OWN ECHO WAS PART
    # OF THE MATCH. Anything else compares SETTINGS, and reporting that as a
    # match is a could-not-check wearing a verified badge.
    if one_sided:
        return IdentityComparison(
            state=NOT_ASSESSED,
            compared=tuple(compared),
            not_compared=tuple(not_compared),
            one_sided=tuple(one_sided),
            basis="endpoint_report" if "reported_model" in compared else "configuration_only",
            statement=(
                f"No part of the model identity that could be compared changed: {matched}. "
                f"Whether the model changed is still not established, because part of the record "
                f"exists on only one side." + one_sided_note + never_note
            ),
        )

    if "reported_model" not in compared:
        return IdentityComparison(
            state=NOT_ASSESSED,
            compared=tuple(compared),
            not_compared=tuple(not_compared),
            one_sided=(),
            basis="configuration_only",
            statement=(
                f"The configuration did not move: {matched}. The endpoint did not report a model "
                f"name on both sides, so nothing here establishes that the same model answered. "
                f"A provider can serve a different build under a stable name, so this is a "
                f"could-not-check, not a match." + never_note
            ),
        )

    return IdentityComparison(
        state="same",
        compared=tuple(compared),
        not_compared=tuple(not_compared),
        one_sided=(),
        basis="endpoint_report",
        statement=(
            f"No recorded part of the model identity changed: {matched}. The endpoint reported "
            f"the same model name on both sides, which is the endpoint's own self-report rather "
            f"than an independent check of what served the request." + never_note
        ),
    )


def describe_model_identity(identity: Optional[ModelIdentity]) -> str:
    """One line naming the model, which never prints a setting as an observation."""
    configured = clean_model_name(getattr(identity, "configured_model", None))
    reported = clean_model_name(getattr(identity, "reported_model", None))
    version = clean_model_name(getattr(identity, "version", None))
    names = tuple(getattr(identity, "reported_names", ()) or ())

    if len(names) > 1:
        listed = ", ".join(f'"{n}"' for n in names)
        base = f"more than one model answered this run ({listed})"
    elif configured is not None and reported is not None:
        base = (
            f'"{configured}" (configured, and the endpoint reported the same name)'
            if configured == reported
            else f'configured "{configured}", but the endpoint reported "{reported}"'
        )
    elif reported is not None:
        base = f'"{reported}" (reported by the endpoint; no configured name was recorded)'
    elif configured is not None:
        base = (
            f'"{configured}" (the configured name; the endpoint reported none, so the model '
            f"that answered was not recorded)"
        )
    else:
        return "model identity not recorded"

    return f"{base}, version {version}" if version else base
