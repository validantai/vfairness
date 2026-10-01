"""Ask the library, from code, whether a capability has been checked.

WHY THIS IS IN THE PACKAGE and not only on the website. The three-state status of
every public capability is published at https://vfairness.validant.ai/status/, and
a reader deciding whether to call something can look it up there. That is no use
to the two people who need it most:

* a user who wants their OWN pipeline to fail when it depends on something this
  library has not verified, and
* a user on an older version, for whom today's website is the wrong answer.

So the ledger ships inside the wheel and this module reads it. The answer is about
the version you have installed, not about whatever the site says today.

    >>> import vfairness
    >>> vfairness.status("demographic_parity_difference").state
    'CHECKED'
    >>> vfairness.status().counts["NOT CHECKED"]        # the whole surface
    299                                 # this number is per version; 1,567 units
                                        # on 2026-09-27. Do not read it as fixed.

WHAT THE THREE STATES MEAN, in one line each, and the detail on every row says more:

    CHECKED      graded and assessed; if it was found inventing a value that was
                 fixed. NOT a correctness certificate for the number it returns.
    FIX PENDING  graded and assessed, and a defect is still open. Do not rely on
                 it in the beta.
    NOT CHECKED  no evidence either way. We are blind here.

USING IT AS A GATE. `require_checked` raises on anything that is not CHECKED, which
is the form a user's CI wants:

    vfairness.require_checked("demographic_parity_difference", "equalized_odds_difference")

A NAME THAT COVERS SEVERAL UNITS reports the WORST of them, in the order
FIX PENDING > NOT CHECKED > CHECKED, the same rule the published badges use. A
class whose nineteen methods are checked and whose twentieth has an open defect is
reported FIX PENDING, because that is the thing the caller needs to know. THAT
HOLDS HOWEVER THE NAME IS SPELLED: the short name, the fully qualified name and
the class object itself are one question and get one answer. Until 2026-09-27 the
last two skipped the fold and answered CHECKED for a class with an unchecked
method, while the published badge said NOT CHECKED (see ``_resolve``).

AN UNKNOWN NAME RAISES. It does not return "not checked": those are different
answers, and conflating them would let a typo read as a finding about the library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Optional

__all__ = ["CapabilityStatus", "SurfaceStatus", "status", "require_checked"]

_DATA = "_capability_status.json"
_WORST_FIRST = ("FIX PENDING", "NOT CHECKED", "CHECKED")
_cache: Optional[dict] = None


class UnknownCapabilityError(LookupError):
    """The name is not on this version's public surface.

    Deliberately not a "NOT CHECKED" answer: a misspelled name would then read as
    a finding about the library rather than a mistake in the question.
    """


@dataclass(frozen=True)
class CapabilityStatus:
    """What is published about one name."""

    name: str
    state: str
    detail: str
    units: tuple = field(default_factory=tuple)
    counts: dict = field(default_factory=dict)
    evidence_dates: tuple = field(default_factory=tuple)
    scope: str = "core"

    @property
    def checked(self) -> bool:
        """True only for CHECKED. NOT CHECKED is not a pass."""
        return self.state == "CHECKED"

    @property
    def preview(self) -> bool:
        """Outside what the beta certifies.

        SEPARATE FROM `checked`, and deliberately so. A preview capability can be
        perfectly well checked, and a core one can be unexamined; merging the two
        would lose whichever answer the caller needed. `checked` says whether
        anybody verified this call refuses honestly. `preview` says whether the
        beta stands behind it.
        """
        return self.scope == "preview"

    def __str__(self) -> str:  # pragma: no cover - convenience
        tail = ", preview: not yet fully covered by the beta" if self.preview else ""
        return f"{self.name}: {self.state}{tail} ({self.detail})"


@dataclass(frozen=True)
class SurfaceStatus:
    """What is published about the whole public surface."""

    total: int
    counts: dict
    evidence_dates: tuple
    library_version: str

    def share(self, state: str) -> float:
        """Fraction of the public surface in *state*.

        A STATE THIS OBJECT DOES NOT HOLD IS AN ERROR, NOT A ZERO (2026-09-25).
        This method was ``self.counts.get(state, 0) / self.total``, and the keys
        are upper case while ``__str__`` prints them lower case. So the obvious
        reading of "874 checked, 626 not checked" was ``share("checked")``, and
        that returned a confident **0.0** for the 874 units it had just printed:
        a fabricated zero, indistinguishable from a surface where nothing had
        been checked, produced by the ``.get(key, default)`` idiom this library
        audits other code for.

        Names are now matched without regard to case or to the separator, and an
        unrecognised state raises instead of answering. There is no honest
        numeric answer to "what share is in a state that does not exist".

        A DECLARED STATE WITH NO MEMBERS IS 0.0, NOT AN ERROR (2026-09-27). The
        vocabulary was read off ``self.counts``, which holds only the states that
        currently have members, so emptying one made the API deny that the state
        exists at all. It happened the moment the BGL3 campaign closed the last
        open defect: ``counts`` lost its "FIX PENDING" key, and
        ``share("FIX PENDING")`` began raising "'FIX PENDING' is not a published
        status state. The three states are ['CHECKED', 'NOT CHECKED']", a message
        that says three while listing two.

        That is this library's own defect class turned on its own API: a fact
        about the published vocabulary, which is fixed, inferred from transient
        data. Zero units in a state is a MEASUREMENT of that state, and the
        happiest possible one here. ``__str__`` already had it right, printing
        ``self.counts.get(s, 0)`` over ``_WORST_FIRST``, so the two accessors
        disagreed with each other.

        A SURFACE OF ZERO UNITS HAS NO SHARE, AND AN UNACCOUNTED REMAINDER IS NOT
        ZERO (2026-09-27, second pass). The two branches above were right and left a
        third fabricated zero standing behind them. Measured before this change:

            SurfaceStatus(total=0, counts={}).share("CHECKED")              -> 0.0
            SurfaceStatus(total=0, counts={}).share("NOT CHECKED")          -> 0.0
            SurfaceStatus(total=0, counts={"CHECKED": 5}).share("CHECKED")  -> 0.0
            SurfaceStatus(total=1567, counts={"CHECKED": 1307})
                .share("NOT CHECKED")                                      -> 0.0
            and the three shares of that last object then summed to 0.834, not 1.0

        Every one of those raises ``ValueError`` now. 0/0 is a could-not-answer, and
        a 0.0 for the 260 units that are in no state at all asserts a tally nobody
        supplied: it is indistinguishable from a real surface in which none of them
        is in that state, which is the defect class this module audits other code
        for. The two answers that ARE measurements are kept deliberately: a DECLARED
        state with no members, over a surface whose tally accounts for every unit,
        still returns 0.0 (the live ledger's FIX PENDING, 0 of 1567), and a populated
        state still returns its real fraction (CHECKED, 1268/1567 -> 0.80919, the
        same value as before this edit).

        WHERE THE LINE IS DRAWN, said plainly so nobody has to guess. On an
        inconsistent object, a state that IS in ``counts`` still answers
        (1307/1567 -> 0.834): both numbers were supplied and dividing them invents
        nothing, though a reader should treat it as a fraction of the stated total
        and no more. It is the ABSENT state that had to stop answering, because
        there 0.0 is a tally of units nobody counted.
        """

        def _norm(text: str) -> str:
            return str(text).strip().upper().replace("-", " ").replace("_", " ")

        wanted = _norm(state)
        if not any(_norm(key) == wanted for key in (*_WORST_FIRST, *self.counts)):
            raise ValueError(
                f"{state!r} is not a published status state. The published states are "
                f"{list(_WORST_FIRST)}. This is a could-not-answer, not a share of 0.0."
            )
        if self.total <= 0:
            raise ValueError(
                f"this surface holds {self.total} code units, so there is no population "
                f"to take a share of and share({state!r}) is a could-not-answer, not "
                "0.0. Zero units IN A STATE is a measurement; zero units in the SURFACE "
                "is the absence of one."
            )
        for key, count in self.counts.items():
            if _norm(key) == wanted:
                return count / self.total
        tallied = sum(self.counts.values())
        if tallied != self.total:
            raise ValueError(
                f"{state!r} is declared and absent from counts, but counts accounts for "
                f"{tallied} of {self.total} code units, so {self.total - tallied} are in "
                "no state at all. Answering 0.0 would assert a tally of those that "
                "nothing supplied, so this is a could-not-answer. Regenerate the ledger, "
                "or read `counts` directly if a partial tally is what you meant."
            )
        # Declared, empty, and the tally accounts for every unit on the surface. Zero
        # of them is a real share, and the happiest one this object can report.
        return 0.0

    def __str__(self) -> str:  # pragma: no cover - convenience
        parts = ", ".join(f"{self.counts.get(s, 0)} {s.lower()}" for s in _WORST_FIRST[::-1])
        return f"{self.total} public code units: {parts}"


def _load() -> dict:
    global _cache
    if _cache is None:
        # Read lazily: the ledger is a few hundred kilobytes and no import should
        # pay for it. resources.files keeps this working from a zipped wheel.
        with resources.files(__package__).joinpath(_DATA).open("r", encoding="utf-8") as fh:
            _cache = json.load(fh)
    return _cache


def _resolve(data: dict, name: str) -> list:
    """Every unit a name could mean. Exact qualified name first, then the tail."""
    # `states_by_unit` IS the unit index: the shipped payload interns its detail
    # strings and carries no second copy of the key set, so there is one place a
    # unit can be known from and no way for two of them to disagree.
    units = data["states_by_unit"]
    # THE FOLD BELOW USED TO SIT UNDER `if name in units: return [name]`, WHICH MADE
    # IT DEAD FOR THE TWO SPELLINGS A CALLER REACHES FOR FIRST (2026-09-27). A guard
    # below a dispatch that already returned cannot fire. Measured on the shipped
    # ledger before this change:
    #     _resolve(data, 'vfairness.evaluation.vfairness_metrics.analyzer'
    #                    '.FairnessAnalyzer')          -> 1 unit,  status CHECKED
    #     _resolve(data, 'FairnessAnalyzer')           -> 15 units, status NOT CHECKED
    #     status(vfairness.FairnessAnalyzer).state     -> 'CHECKED'
    #     require_checked(vfairness.FairnessAnalyzer)  -> None, i.e. the gate PASSED
    # while FairnessAnalyzer.report_to_svg is NOT CHECKED in that same ledger, and
    # scripts/stamp_status_badges.index_by_name folds unconditionally, so the badge
    # said NOT CHECKED for the name the installed API called CHECKED. 98 ledger rows
    # are in that shape today. After this change all three spellings resolve 15
    # units and answer NOT CHECKED, and require_checked(vfairness.FairnessAnalyzer)
    # raises RuntimeError. A fully qualified name still means only itself and its
    # descendants, never its same-tail namesakes.
    hits = [name] if name in units else [q for q in units if q.split(".")[-1] == name]
    if not hits:
        return []
    # A CLASS answers for the methods reached through it, exactly as the published
    # badge does. Without this the website and this function disagree about the same
    # name: the badge for FairnessAnalyzer folds in its methods and reports the
    # worst, so a class with one defective method reads FIX PENDING there, and
    # reporting only the class row here would have said CHECKED. Two surfaces
    # answering the same question differently is the defect this whole programme
    # exists to remove, so the rule lives in one shape and is applied in both.
    folded = list(hits)
    for qual in hits:
        folded += [q for q in units if q.startswith(qual + ".")]
    return sorted(set(folded))


def status(name: Any = None) -> Any:
    """Published status of *name*, or of the whole surface when called with none."""
    data = _load()
    dates = tuple(data.get("evidence_dates") or ())
    if name is None:
        return SurfaceStatus(
            total=len(data["states_by_unit"]),
            counts=dict(data.get("counts") or {}),
            evidence_dates=dates,
            library_version=str(data.get("library_version", "")),
        )

    if not isinstance(name, str):
        # A callable or class: ask for its defining name, which is the key the
        # ledger uses. Falling back to __name__ keeps a bound method answerable.
        module = getattr(name, "__module__", "")
        qual = getattr(name, "__qualname__", None) or getattr(name, "__name__", None)
        if not qual:
            raise UnknownCapabilityError(f"cannot work out a capability name from {name!r}")
        name = f"{module}.{qual}" if module.startswith("vfairness") else qual

    units = _resolve(data, name)
    if not units:
        raise UnknownCapabilityError(
            f"{name!r} is not on the public surface of vfairness "
            f"{data.get('library_version', '')}. This is not a status: a name nobody "
            "can resolve is a question about the wrong thing. Check the spelling, or "
            "see https://vfairness.validant.ai/status/ for the full list."
        )

    states = [data["states_by_unit"][q] for q in units]
    state = next(s for s in _WORST_FIRST if s in states)
    counts = {s: states.count(s) for s in _WORST_FIRST if states.count(s)}
    if len(units) == 1:
        detail = data["details"][data["detail_by_unit"][units[0]]]
    else:
        detail = f"{len(units)} code units behind this name: " + ", ".join(
            f"{n} {s.lower()}" for s, n in counts.items()
        )
    return CapabilityStatus(
        name=name,
        state=state,
        detail=detail,
        units=tuple(sorted(units)),
        counts=counts,
        evidence_dates=dates,
        # Preview only when EVERY unit behind the name is preview. A name covering
        # a mix is core, because the beta does stand behind part of what it reaches
        # and calling the whole thing preview would understate that.
        scope=(
            "preview"
            if units and all(u in set(data.get("preview_units") or ()) for u in units)
            else "core"
        ),
    )


def require_checked(*names: str) -> None:
    """Raise unless every name is CHECKED. The form a caller's CI wants.

    Raises ``UnknownCapabilityError`` for a name that does not exist and ``RuntimeError``
    listing every name that is not CHECKED, so one call reports all of them rather
    than stopping at the first.

    AN EMPTY CALL IS REFUSED (2026-09-27). ``require_checked()`` returned ``None``,
    the same answer it gives when every named capability is CHECKED, after
    verifying nothing at all. The realistic route in is the splatted form this
    gate is designed for, ``require_checked(*config["capabilities"])``: a config
    that failed to load, a filter that matched nothing or a typo'd key yields an
    empty tuple, and a gate that passes when it examined zero capabilities is the
    exact collapse of "nothing was examined" into "nothing was found" that the
    three published states exist to prevent. There is no honest pass over an
    empty set, so this is a ``ValueError`` and not a silent success.
    """
    if not names:
        raise ValueError(
            "vfairness.require_checked: no capability names were given, so nothing "
            "was verified. That is a could-not-check, not a pass. Pass the names "
            "your pipeline depends on, e.g. require_checked('demographic_parity_"
            "difference'); if the names come from config, check that it loaded."
        )
    bad = []
    for name in names:
        result = status(name)
        if not result.checked:
            # result.name, not the raw argument: a class OBJECT reached this gate
            # for the first time on 2026-09-27 (it used to pass silently), and
            # f"{name}" would have put "<class 'vfairness...FairnessAnalyzer'>" in
            # the message where a capability name belongs. For a string argument
            # result.name IS that string, so nothing else changes.
            bad.append(f"{result.name}: {result.state} ({result.detail})")
    if bad:
        raise RuntimeError(
            "vfairness.require_checked: "
            + f"{len(bad)} of {len(names)} capabilities are not CHECKED in this "
            + "version:\n  "
            + "\n  ".join(bad)
        )
