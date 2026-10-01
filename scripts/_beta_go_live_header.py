"""
vfairness Beta Go-Live proof ledger.

WHAT THIS IS. For every capability in ``CAPABILITY_REGISTRY``, this file records
how far its **honesty** has actually been proved: not whether it computes a
number, but whether it refuses to invent one when nothing is measurable.

WHY IT EXISTS. The library's one standing rule is THREE STATES, NEVER TWO:
measured / failed / could-not-check. A capability that quietly substitutes a
neutral default (0.0, 1.0, ``True``, ``"pass"``) for a value nobody measured
reads, to every downstream grader, chart and report, exactly like a measurement.
Before the 0.1.0 beta we measured the whole surface for that defect. This ledger
is the result, and it is deliberately written so that a capability nobody has
checked CANNOT look like one that passed.

The ledger is the same shape as the thing it audits. A capability with no
evidence is ``BGL-C UNPROVEN``. It is never silently promoted, and there is no
default value: ``scripts/beta_go_live_ledger.py`` fails closed, and
``tests/test_beta_go_live_proof_ledger.py`` refuses any entry whose claim is not
backed by a file that exists.

HOW IT WAS MEASURED. A harness executed every registered capability twice: once
on HEALTHY input carrying a real, findable group disparity, and once on each
degenerate input where the thing it claims to measure does not exist at all
(one group only; every label identical; every score NaN; zero rows; every score
identical so no ranking exists; two rows; a group with a single row; text with
no lexicon word in it). Any capability that answered confidently in the second
case went to an independent judge, then to an adversarial verifier told to
refute the judge, then to a third pass whose only job was to re-run the claim at
the PUBLIC entry point and report the real returned value. Only findings that
survived all three are recorded as defects here.

Full method, batch definitions and the summary table: docs/BETA_GO_LIVE_PLAN.md
"""

from __future__ import annotations

from typing import TypedDict


class ProofBatch(TypedDict):
    """Definition of one rung on the proof ladder."""

    label: str  # short human name
    means: str  # what a reader may conclude
    requires: str  # what a capability must satisfy to be in this batch
    may_not_conclude: str  # what this batch explicitly does NOT establish


class ProofRecord(TypedDict):
    """Proof status of a single registered capability."""

    batch: str  # one of PROOF_BATCHES
    shared_with: list  # other dispatch keys resolving to the SAME live object
    reached: bool  # was it executed on degenerate input at all
    judged: int  # entry points independently adjudicated
    open_defects: int  # confirmed, three-pass-verified fabrications still open
    severities: list  # severities of those open defects
    pins: list  # test files pinning its could-not-check path
    fix_minutes: int  # estimated minutes to close open_defects (0 if none)


# The date the surface was measured. Every claim in CAPABILITY_PROOF is a claim
# about the code as of this date; re-run scripts/beta_go_live_ledger.py after any
# change to a registered capability.
MEASURED_ON = "2026-09-11"
MEASURED_AT_COMMIT = "8559dda"

# ``COUNTS_AS_OF`` is emitted below by the generator and is a DIFFERENT date from
# ``MEASURED_ON``. The census reproduced 157 defects; what the batches show is
# what remains after the fix waves since. Anything rendering these counts must
# date them with COUNTS_AS_OF, not with MEASURED_ON: a surface that shows a
# count beside the census date tells a reader the census found that count, and
# both halves of that sentence are true while the sentence is false. Found twice
# in one day, once here and once on the platform's coverage card.

PROOF_BATCHES: dict[str, ProofBatch] = {
    "BGL-A": {
        "label": "PROVEN",
        "means": (
            "Executed on healthy input and on the degenerate inputs relevant to "
            "what it claims to measure. In each of those it EITHER refused "
            "(returned NaN, None, an explicit could-not-check, or raised) OR "
            "returned a value an independent judge showed to be a true "
            "measurement for that input. Nothing it returned survived three "
            "passes as a fabrication, and at least one test in the suite names "
            "it next to a refusal assertion."
        ),
        "requires": (
            "reached=True AND judged>=1 AND open_defects==0 AND at least one test "
            "file, present in the repository, that names the capability within "
            "twenty lines of a refusal assertion."
        ),
        "may_not_conclude": (
            "Three things, each of which stays open at 0.1.0. (1) That the NUMBER "
            "it returns on healthy data is statistically correct: this ladder "
            "measures honesty about the ABSENCE of a measurement, not the accuracy "
            "of one. (2) That the pin covers every degenerate scenario: the pin "
            "test is a proximity match, so it proves a refusal is asserted "
            "somewhere near this capability, not that THIS scenario is the one "
            "asserted. (3) That the pin has been sabotage-checked. A guard that "
            "cannot fail looks identical to one that passed, and only a reinstated "
            "defect tells them apart. Promoting BGL-A to a sabotage-verified rung "
            "is the exit criterion of Stage 4 in docs/BETA_GO_LIVE_PLAN.md."
        ),
    },
    "BGL-B": {
        "label": "SEMI-PROVEN",
        "means": (
            "Executed on healthy and degenerate input and judged clean today, but "
            "nothing in the suite holds it there. The behaviour is observed, not "
            "protected: the next refactor can reintroduce the defect and every "
            "gate will stay green."
        ),
        "requires": (
            "reached (generically, or by a hand-written fixture) AND judged>=1 AND "
            "open_defects==0 AND no test file names it near a refusal assertion."
        ),
        "may_not_conclude": (
            "That it will still be honest after the next change. A BGL-B capability "
            "is one commit away from BGL-D with no alarm."
        ),
    },
    "BGL-C": {
        "label": "UNPROVEN",
        "means": (
            "No execution evidence on degenerate input. Either no fixture could be "
            "constructed generically, or it needs a collaborator (a live LLM proxy, "
            "a fitted pipeline, a torch model) the harness does not have. Its "
            "honesty is UNKNOWN, which is not the same as suspect and is emphatically "
            "not the same as clean."
        ),
        "requires": "no adjudicated execution on degenerate input.",
        "may_not_conclude": (
            "Anything at all about whether it fabricates. This batch exists so that "
            "'not checked' can never be read as 'checked and fine'."
        ),
    },
    "BGL-D": {
        "label": "DEFECT OPEN",
        "means": (
            "At least one entry point was proved, by three independent passes ending "
            "in a re-run at the public API, to hand back a confident value where "
            "nothing was measurable. The capability still computes correctly on "
            "healthy data; what is broken is what it says when it cannot."
        ),
        "requires": "open_defects>=1, each reproduced at the public entry point.",
        "may_not_conclude": (
            "That the capability is unusable. It means its could-not-check path is "
            "wrong, and a caller who hits that path is told a number that was never "
            "measured."
        ),
    },
}


PROOF_BLOCK_BEGIN = "Beta Go-Live proof status"
PROOF_BLOCK_END = "(end Beta Go-Live proof status)"


def strip_proof_block(doc: str | None) -> str:
    """Return ``doc`` without its generated Beta Go-Live block.

    Anything that PARSES a docstring must strip this block first, because the
    block is machine-written prose that can collide with a heuristic aimed at
    human prose. That is not hypothetical: ``tests/test_docs_truth.py`` detects a
    paper citation with ``\\b(19|20)\\d\\d\\b``, and the stamp's own date,
    ``(2026-09-11)``, matched it. Every stamped metric suddenly "cited a paper"
    and was required to carry a divergence note.

    The right fix was not to weaken the citation rule, which would let a real
    citation slip past, nor to disguise the date, which a reader needs. It was to
    take the generated block out of the text a prose heuristic reads.
    """
    if not doc:
        return ""
    start = doc.find(PROOF_BLOCK_BEGIN)
    if start < 0:
        return doc
    end = doc.find(PROOF_BLOCK_END, start)
    if end < 0:
        return doc[:start].rstrip() + "\n"
    return (
        doc[:start].rstrip() + "\n" + doc[end + len(PROOF_BLOCK_END) :].lstrip("\n")
    ).rstrip() + "\n"
