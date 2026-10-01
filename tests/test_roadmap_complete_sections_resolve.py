"""A roadmap section headed COMPLETE must not show code that raises.

A roadmap is allowed, and expected, to sketch APIs that do not exist yet. That
is what a roadmap IS, and `tests/test_site_api_reference_calls_resolve.py`
deliberately does not police this file for that reason.

The defect this closes is narrower and worse. When a gap is actually BUILT, the
heading gets "(COMPLETE)" and the Current State gets a tick, and the
Implementation Plan underneath keeps the PRE-BUILD sketch, which the real
implementation no longer matches. The reader is then told the feature is done
and handed code that raises, which is a stronger claim than the library
supports.

Measured on 2026-09-07, Gap 3 "Real-time Monitoring and Drift Detection
(COMPLETE)" carried four such sketches:

    FairnessMonitor(metrics=..., protected_attrs=..., alert_thresholds=...)
        -> TypeError; the shipped constructor takes window_size / alert_threshold
           / config / custom_metrics, and there is no .update() or
           .get_historical() at all
    FairnessDriftDetector(reference_data=..., method='psi')
        -> TypeError; the shipped detector works on a metric SERIES via
           set_baseline() / check_drift(), and has no .detect()
    FairnessAlertManager(...)   -> does not exist (FairnessAlertPrioritizer +
                                   AdaptiveThresholdManager do)
    FairnessABTest(...)         -> does not exist (FairnessExperiment does)

Gap 2 already had the right convention for this: keep the sketch as a design
record, and label it. So the rule here is not "every block must run", it is:

    inside a section headed COMPLETE, a block must either RESOLVE, or be
    labelled with the file's own marker, "supersedes the sketch below".

which keeps the historical sketches readable without letting them masquerade as
working code.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re

import pytest

_ROADMAP = pathlib.Path(__file__).resolve().parents[1] / "docs" / "ROADMAP.md"

_EXEMPT = "supersedes the sketch below"
_COMPLETE = re.compile(r"^##\s+Gap\b.*\((?:COMPLETE|SHIPPED|LIVE)\)", re.I)
_GAP = re.compile(r"^##\s+\S")
# Any heading resets the prose window, so a "superseded" label exempts the phase
# it introduces and NOT every later block in the same gap. Scoping it to the gap
# made all 8 blocks exempt and the resolver vacuous, which is what the
# non-vacuity test below caught.
_ANY_HEADING = re.compile(r"^#{2,6}\s+\S")
_FENCE = re.compile(r"^```\s*(?:python|py)?\s*$")


def _blocks_in_complete_sections() -> list[tuple[str, str, bool]]:
    """(gap heading, code, exempt) for every python block under a COMPLETE gap."""
    if not _ROADMAP.exists():  # pragma: no cover - the file is part of the repo
        pytest.skip("ROADMAP.md not present")
    out: list[tuple[str, str, bool]] = []
    heading, complete, prose, buf = "", False, [], None
    for line in _ROADMAP.read_text(encoding="utf-8").split("\n"):
        if buf is None and _ANY_HEADING.match(line):
            prose = []
            if _GAP.match(line):
                heading, complete = line.strip(), bool(_COMPLETE.match(line))
            else:
                heading = f"{gap} / {line.strip()}" if (gap := heading.split(" / ")[0]) else heading
        if buf is None and line.startswith("```"):
            # Only python fences carry callable code; ``` json / bash / text do not.
            buf = [] if _FENCE.match(line) else None
            if buf is None:
                prose.append(line)
                continue
            # The label may sit anywhere between the gap heading and the fence.
            out.append((heading, "", _EXEMPT in "\n".join(prose)))
            continue
        if buf is not None:
            if line.startswith("```"):
                if complete:
                    h, _, ex = out[-1]
                    out[-1] = (h, "\n".join(buf), ex)
                else:
                    out.pop()
                buf = None
            else:
                buf.append(line)
            continue
        prose.append(line)
    return [b for b in out if b[1].strip()]


def _defects() -> list[str]:
    bad: list[str] = []
    for heading, code, exempt in _blocks_in_complete_sections():
        if exempt or "vfairness" not in code or "import" not in code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue  # a fragment, not a program
        env: dict[str, object] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("vfairness")
            ):
                try:
                    mod = importlib.import_module(node.module)
                except Exception:
                    continue  # an optional extra that is not installed here
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    obj = getattr(mod, alias.name, None)
                    if obj is None:
                        bad.append(
                            f"{heading}: documented as done, but "
                            f"{node.module}.{alias.name} does not exist"
                        )
                    else:
                        env[alias.asname or alias.name] = obj
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            target = env.get(node.func.id)
            if target is None:
                continue
            try:
                sig = inspect.signature(target)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                continue
            names = {p.name for p in sig.parameters.values()}
            for kw in node.keywords:
                if kw.arg and kw.arg not in names:
                    bad.append(
                        f"{heading}: documented as done, but "
                        f"{node.func.id}(... {kw.arg}=...) raises TypeError; "
                        f"real parameters: {sorted(names)}"
                    )
    seen: set[str] = set()
    return [d for d in bad if not (d in seen or seen.add(d))]


def test_complete_sections_do_not_document_code_that_raises():
    defects = _defects()
    assert not defects, (
        "ROADMAP.md calls these gaps done and then shows code that raises.\n"
        "Either correct the block, or label it '"
        + _EXEMPT
        + "' as Gap 2 and Gap 3 do:\n  "
        + "\n  ".join(defects)
    )


def _label_lines() -> list[str]:
    """The '**Status:** ... supersedes the sketch below' paragraphs."""
    return [
        ln
        for ln in _ROADMAP.read_text(encoding="utf-8").split("\n")
        if _EXEMPT in ln and ln.lstrip().startswith("**Status:**")
    ]


def _package_symbols() -> set[str]:
    import pkgutil

    import vfairness

    names: set[str] = set(dir(vfairness))
    for m in pkgutil.walk_packages(vfairness.__path__, "vfairness."):
        try:
            names |= set(dir(importlib.import_module(m.name)))
        except Exception:
            continue  # an optional extra that is not installed here
    return names


# Classes the labels name as the real replacement. Prose words in backticks
# (`llm/counterfactual.py`, `analyze_all()`) are not symbols, so match only
# bare CamelCase identifiers.
_CLASS = re.compile(r"`([A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*)`")

# A label may also say a name does NOT exist, and that claim must stay true too:
# if the class later ships, the label is the thing telling readers it did not.
# Two phrasings are in use, and in both the name sits IMMEDIATELY beside the
# phrase, so anchor on the phrase and take the nearest backticked name. A looser
# span match pairs "does not exist" with whichever name came first in the
# sentence, which reported `TextScorer` and `FairnessExperiment` as absent when
# both ship.
_NO_SUCH = re.compile(r"[Tt]here is no `([A-Z][A-Za-z0-9]*)`")


def _absent_claims(line: str) -> set[str]:
    out = set(_NO_SUCH.findall(line))
    for head in line.split("does not exist")[:-1]:
        names = _CLASS.findall(head)
        if names:
            out.add(names[-1])
    return out


def test_superseded_labels_name_something_that_exists():
    """A label that says "shipped as X" is a claim, and it can rot too.

    This is what keeps the file honest once every block under a COMPLETE gap is
    an exempt sketch: the exemption is only earned by naming the real thing, so
    the labels are checked even though the code they replace is not run.
    """
    symbols = _package_symbols()
    bad = []
    for line in _label_lines():
        absent = _absent_claims(line)
        for name in _CLASS.findall(line):
            if name in absent:
                if name in symbols:
                    bad.append(f"label says {name!r} does not exist, but it does: {line[:90]}...")
            elif name not in symbols:
                bad.append(f"label points at {name!r}, which is not in the package: {line[:90]}...")
    assert not bad, "ROADMAP.md status labels have gone stale:\n  " + "\n  ".join(bad)


def test_the_scan_actually_checks_something():
    """NON-VACUITY, in both directions.

    If every block in a COMPLETE section is exempt, the resolver above checks
    nothing, and that is the current state of this file: all of them are
    superseded sketches. So the coverage requirement is placed on the LABELS
    instead, which is the claim a reader actually acts on. Both halves have to
    be doing work, or this file passes forever while checking nothing.
    """
    blocks = _blocks_in_complete_sections()
    assert blocks, "no python blocks found under a COMPLETE gap; the extractor is broken"
    assert [b for b in blocks if b[2]], "the 'superseded sketch' marker is unused"

    labels = _label_lines()
    assert len(labels) >= 6, f"only {len(labels)} status labels found; the extractor is broken"
    named = {n for line in labels for n in _CLASS.findall(line)}
    assert len(named) >= 8, f"the labels only name {len(named)} classes; nothing is being resolved"


def test_a_complete_gap_does_not_list_whats_missing():
    """ "(COMPLETE)" and "What's Missing" cannot both be true of one gap.

    Gap 2 and Gap 3 carried both until 2026-09-07: each was headed COMPLETE, and
    each still listed the four course topics under "What's Missing" even though
    all four had shipped. The list itself was fine, it just had the wrong title,
    so it is now "Course coverage this gap closes". The remaining gaps keep
    "What's Missing", because for them it is accurate.

    If a gap genuinely still has missing pieces, the fix is to drop COMPLETE
    from its heading, not to retitle the list.
    """
    gap, bad = "", []
    for line in _ROADMAP.read_text(encoding="utf-8").split("\n"):
        if line.startswith("## Gap"):
            gap = line.strip()
        if line.strip().lower().startswith("### what's missing") and _COMPLETE.match(gap):
            bad.append(gap)
    assert not bad, (
        "these gaps claim to be complete and still list missing work:\n  " + "\n  ".join(bad)
    )
