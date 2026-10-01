"""Pulse image / text-to-image fairness path (Stage B2).

Operates on a table that carries a per-image demographic LABEL column
(e.g. ``detected_race`` produced upstream / by the vision sidecar) and,
optionally, a reference-distribution. REUSES vfairness.vision skew / NDKL /
bias-amplification (pure math, verified). If only image paths are given it
calls the sidecar-gated classifier, which honestly degrades to
``available: False`` rather than fabricate demographics. Never raises.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pandas as pd

#: Whole-token spellings of "this image carries no demographic label", matched
#: case-insensitively against the ENTIRE stripped cell and never as a substring.
#: pandas' own missing values (None, NaN, pd.NA, NaT) are caught by ``isna()``
#: before this set is consulted; these are the literal strings that survive a
#: hand-built frame or a CSV whose missing markers were quoted. "na" is
#: deliberately absent: pandas already reads a bare NA as missing, and a
#: three-letter token is too close to a real label to drop on a guess.
_UNLABELLED_TOKENS = frozenset({"", "nan", "none", "null", "n/a", "<na>", "nat"})


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _labelled(series: pd.Series) -> Tuple[List[str], int, int]:
    """The images that actually carry a demographic label, and the two counts.

    R-11, 2026-09-10. This used to be ``df[demo_col].astype(str).tolist()``, and
    ``astype(str)`` renders a missing demographic as the LITERAL STRING "nan"
    (or "None", or "<NA>"), which every metric downstream then reads as one
    perfectly ordinary group. Measured on 100 images the classifier labelled
    NONE::

        vision.skew: {"available": true, "perGroup": {"nan": 0.0},
                      "maxSkew": 0.0, "mostOverrepresented": "nan"}
        vision.ndkl: {"available": true, "value": 0.0}
        vision.summary: "Within representation tolerance on the perceived labels..."

    MaxSkew 0.0 and NDKL 0.0 are the BEST attainable score on both scales, and
    they were awarded to an image set with no demographic information in it at
    all: a single fabricated group is perfectly represented against itself.

    ``isna()`` runs first so pandas' own missing values need no string guessing;
    the token set only catches the literal spellings that survive into an object
    column.
    """
    total = int(len(series))
    present = series[series.notna()]
    labels: List[str] = []
    for value in present.astype(str).tolist():
        if value.strip().lower() in _UNLABELLED_TOKENS:
            continue
        labels.append(value)
    return labels, len(labels), total


def _ndkl_block(nd: object) -> Dict[str, Any]:
    """The NDKL panel: three states, with ndkl's own coverage carried through.

    ``vfairness.vision.ndkl`` returns an ``NDKLValue``, a float that REFUSES
    with ``nan`` (one observed group and no reference distribution; a
    reference giving zero share to a group that appears in the set; no
    non-empty ranking) and that carries ``n_rankings_supplied`` /
    ``n_rankings_measured`` / ``n_rankings_empty`` beside the number.

    BGL-S2b, 2026-09-17. This panel tested only ``nd is not None``, so every
    one of those refusals was published as ``{"available": True, "value":
    nan}``. Measured at this public entry on 30 images all labelled 'white':
    ``ndkl={'available': True, 'value': nan}``; and with a reference giving
    'black' a zero share while ten images carry it, the same shape appeared
    under ``available: True`` beside two real findings. A reader keying on
    ``ndkl.available`` cannot tell that from a measurement, which is the
    refusal being correct one layer down and invisible one layer up. The
    coverage counts are lifted onto the panel as well, because ``json.dumps``
    and every arithmetic operation drop them from the float subclass.
    """
    if nd is None:
        return {
            "available": False,
            "reason": "NDKL could not be computed on this label set.",
        }
    coverage = {
        key: int(value)
        for key, value in (
            ("rankingsSupplied", getattr(nd, "n_rankings_supplied", None)),
            ("rankingsMeasured", getattr(nd, "n_rankings_measured", None)),
            ("rankingsEmpty", getattr(nd, "n_rankings_empty", None)),
        )
        if value is not None
    }
    try:
        value = float(nd)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value):
        return {
            "available": False,
            "reason": (
                "NDKL REFUSED this label set: it returned NaN, which "
                "vfairness.vision.ndkl does when the divergence is undefined. "
                "The three cases are one observed group with no reference "
                "distribution to diverge from, a reference giving zero share to "
                "a group that is present (infinite KL), and no non-empty ranking "
                "at all. This is COULD NOT CHECK, not a divergence of zero."
            ),
            **coverage,
        }
    return {"available": True, "value": value, **coverage}


def _summary(
    *,
    n_images: int,
    n_labelled: int,
    n_unlabelled: int,
    demo_col: str,
    findings: List[Dict[str, Any]],
    skew: Dict[str, Any],
    ndkl_panel: Dict[str, Any],
    ref_label: str,
    unlabelled_reason: str,
) -> str:
    """The one sentence a reader of this page takes away. Three states, not two.

    THE ALL-CLEAR WAS SAID WHENEVER THERE WAS NO FINDING, and "no finding" is
    also what a REFUSED measurement produces: ``findings`` is appended to only
    inside ``if s.get("available")``, so a skew panel that answered
    ``available: False`` left the list empty and the summary read "Within
    representation tolerance on the perceived labels".

    R-11 (2026-09-10) closed this for the TOTAL case, and its comment here said
    "the all-clear sentence is the LAST thing that may be said about an
    unlabelled set". It guarded ``n_labelled == 0`` and nothing else. Measured on
    this repo, grade wave G06, 2026-09-30, three sets that ARE partly labelled:

        50 'white' + 50 NaN   skew available False
                              ("only one group was observed ('white') and no
                               reference distribution was supplied, so
                               representation skew is undefined. This is NOT a
                               pass."), ndkl available False,
                              summary "Within representation tolerance on the
                              perceived labels, against a uniform reference..."
        1 image, 1 label      the same three
        100 x the string 'na' the same three

    So the skew panel said in as many words "This is NOT a pass" and the summary
    beside it said it was. PARTIAL loss passing where TOTAL loss is refused,
    inside one payload.

    The deciding fact is whether the measurement that WOULD have produced a
    finding ran at all, so the sentence is gated on ``skew.available`` and
    carries the panel's own reason when it did not. The all-clear keeps its exact
    wording where skew was genuinely measured and found nothing, which is the
    over-correction control: refusing every clean image set would be worse than
    the defect.
    """
    if n_labelled == 0:
        return (
            f"NOT ASSESSED: none of the {n_images} image(s) carries a demographic "
            f"label in '{demo_col}', so no representation measurement was made. "
            f"This is not a finding of balanced representation."
        )
    tail = f" {unlabelled_reason}" if n_unlabelled else ""
    if findings:
        return f"{len(findings)} image-representation finding(s)." + tail
    if not (isinstance(skew, dict) and skew.get("available")):
        reason = str((skew or {}).get("reason") or "").strip()
        refused = ["representation skew"]
        if not (isinstance(ndkl_panel, dict) and ndkl_panel.get("available")):
            refused.append("NDKL")
        return (
            f"COULD NOT CHECK: {' and '.join(refused)} produced no measurement on "
            f"the {n_labelled} labelled image(s), so there is no representation "
            f"reading to be within tolerance of. This is NOT a finding of balanced "
            f"representation." + (f" {reason[0].upper() + reason[1:]}" if reason else "") + tail
        )
    return (
        "Within representation tolerance on the perceived labels, against " + ref_label + "."
    ) + tail


def vision_probe_pulse(
    df: pd.DataFrame, inputs: Dict[str, Any], demo_col: str, domain: str, jurisdiction: str
) -> Dict[str, Any]:
    """Image-set representation fairness from a demographic-label column."""
    from vfairness.vision import bias_amplification, ndkl, representation_severity, skew

    labels, n_labelled, n_images = _labelled(df[demo_col])
    n_unlabelled = n_images - n_labelled
    unlabelled_reason = (
        f"{n_unlabelled} of {n_images} image(s) carry no demographic label in "
        f"'{demo_col}'. Representation skew and NDKL are computed over the "
        f"{n_labelled} that do; the rest are UNASSESSED, not evenly distributed."
    )
    ref = inputs.get("reference_distribution") or None
    if n_labelled == 0:
        # NOT a label set. Every metric below refuses rather than scoring the
        # one manufactured group; a 0.0 from them is the best score on the
        # scale and would be handed to an image set nobody could label.
        no_labels = {
            "available": False,
            "reason": (
                f"None of the {n_images} image(s) carries a demographic label in "
                f"'{demo_col}', so there is no distribution to measure. This is "
                f"COULD NOT CHECK, not balanced representation."
            ),
        }
        s: Dict[str, Any] = dict(no_labels)
        nd = None
        amp: Dict[str, Any] = dict(no_labels)
    else:
        s = _safe(lambda: skew(labels, ref), {"available": False})
        # NDKL failure must read as UNAVAILABLE, never as 0.0: defaulting to
        # zero rendered "no divergence" when the metric simply crashed.
        nd = _safe(lambda: ndkl(labels, ref), None)
        amp = (
            _safe(lambda: bias_amplification(labels, ref), {"available": False})
            if isinstance(ref, dict)
            else {"available": False}
        )
    # Built once, above the finding text AND the panel below, because both read
    # the same three-state answer and they must not disagree.
    ndkl_panel = _ndkl_block(nd)
    ref_label = (
        "the supplied reference distribution"
        if isinstance(ref, dict)
        else "a uniform reference (no reference distribution supplied)"
    )
    findings: List[Dict[str, Any]] = []
    if s.get("available"):
        # Pass the full skew dict so the gate keys on the WORST absolute skew
        # (max of |MaxSkew|, |MinSkew|), not MaxSkew alone. Otherwise a
        # severely under-represented or excluded group (large negative MinSkew)
        # is never flagged. See vfairness.vision.representation_severity.
        sev = representation_severity(s)
        if sev in ("warn", "critical"):
            findings.append(
                {
                    "type": "image_representation_skew",
                    "severity": sev,
                    "attribute": demo_col,
                    "plain": (
                        f"The image set reads as demographically skewed "
                        f"on its PERCEIVED labels "
                        f"(MaxSkew {s.get('maxSkew')}, over-represented: "
                        f"{s.get('mostOverrepresented')}; "
                        f"under-represented: "
                        f"{s.get('mostUnderrepresented')}"
                        + (
                            f"; NDKL {nd}"
                            if ndkl_panel.get("available")
                            else "; NDKL unavailable (refused, see vision.ndkl.reason)"
                        )
                        + f", measured against {ref_label}). "
                        "Generated/representative sets this skewed "
                        "encode representation bias."
                    ),
                    "statisticalTest": None,
                    "groupDistributions": {},
                    "representationRatios": {},
                    "underrepresentedGroups": [],
                    "overrepresentedGroups": [],
                }
            )
    if isinstance(amp, dict) and amp.get("available") and amp.get("maxAmplification", 0) >= 0.1:
        findings.append(
            {
                "type": "image_bias_amplification",
                "severity": "warn",
                "attribute": demo_col,
                "plain": (
                    f"The model amplifies the real-world imbalance for "
                    f"'{amp.get('worstGroup')}' by "
                    f"{amp.get('maxAmplification')} vs the reference "
                    f"distribution (Seshadri et al. 2023)."
                ),
                "statisticalTest": None,
                "groupDistributions": {},
                "representationRatios": {},
                "underrepresentedGroups": [],
                "overrepresentedGroups": [],
            }
        )

    from vfairness.operations.reporting import build_assurance_verdict

    assurance = _safe(
        lambda: build_assurance_verdict(
            schema={"pii_leakage": [], "mismatches": [], "refuse": False},
            per_variable=[],
            metrics=[],
            bias=findings,
            proxies={},
            statistical={},
            intersectional={},
            disparity_matrix={},
            recommended={},
            domain=domain,
            jurisdiction=jurisdiction,
            has_truth=False,
        ),
        {
            "overall": "Disclaimer",
            "blocksDeployment": False,
            "oneLineVerdict": "Vision assurance verdict could not be assembled.",
            "findings": [],
            "recommendations": [],
            "metricsDeferred": {},
            "auditTrail": {},
        },
    )
    tone = (
        "critical"
        if assurance.get("overall") == "Adverse"
        else "warn"
        if assurance.get("overall") == "Qualified"
        else "pass"
        if assurance.get("overall") == "Unqualified"
        # Disclaimer = nothing was assessable; render as could-not-
        # assess, never as neutral/pass (canonical tone vocabulary).
        else "disclaimer"
        if assurance.get("overall") == "Disclaimer"
        else "neutral"
    )
    from vfairness.operations.pulse.orchestrator import (
        build_legal_framework_block,
        build_scope_block,
    )

    return {
        "success": True,
        "data": {
            "sourceKind": "image_set",
            "assurance": assurance,
            "verdict": {
                "tone": tone,
                "headline": assurance.get("oneLineVerdict", ""),
                "summary": (
                    "Image representation fairness on "
                    "PERCEIVED demographic labels produced by "
                    "an automated classifier, measured against "
                    f"{ref_label} (vfairness.vision skew / "
                    "NDKL / amplification)."
                ),
                "ranked": [],
            },
            "vision": {
                "available": bool(s.get("available")),
                "demographicColumn": demo_col,
                # G-16: when the CLIP sidecar produced the labels ON
                # this run, say so and carry its disclosure block;
                # pre-labeled manifests keep the original wording.
                "labelsSource": (
                    "classified_by_sidecar_clip_zero_shot"
                    if inputs.get("_vision_classifier")
                    else "perceived_by_classifier"
                ),
                "classifier": (
                    dict(inputs["_vision_classifier"])
                    if isinstance(inputs.get("_vision_classifier"), dict)
                    else None
                ),
                "reference": ("supplied" if isinstance(ref, dict) else "uniform"),
                "skew": s,
                "ndkl": ndkl_panel,
                "amplification": amp,
                # How much of the image set the numbers above actually rest on.
                # Three states, and the reader sees all three: fully labelled,
                # partly labelled (the count is stated), or nothing labelled.
                "labelCoverage": {
                    "images": n_images,
                    "labelled": n_labelled,
                    "unlabelled": n_unlabelled,
                    "complete": n_unlabelled == 0,
                },
                "summary": _summary(
                    n_images=n_images,
                    n_labelled=n_labelled,
                    n_unlabelled=n_unlabelled,
                    demo_col=demo_col,
                    findings=findings,
                    skew=s,
                    ndkl_panel=ndkl_panel,
                    ref_label=ref_label,
                    unlabelled_reason=unlabelled_reason,
                ),
            },
            "legalFramework": build_legal_framework_block(domain, jurisdiction),
            "scope": build_scope_block(
                "image_set",
                covered=[
                    "Representation skew (MaxSkew) and rank divergence (NDKL) "
                    "of PRE-LABELED demographic columns, measured against " + ref_label,
                    "Bias amplification vs the reference distribution (only "
                    "when a reference is supplied)",
                ]
                + (
                    [
                        "Live image classification: the vision sidecar (CLIP "
                        "zero-shot) produced the perceived labels on this run; "
                        "model, axis and per-image outcome counts are disclosed "
                        "in vision.classifier",
                    ]
                    if inputs.get("_vision_classifier")
                    else []
                ),
                not_covered=[
                    *(
                        [
                            f"The {n_unlabelled} image(s) of {n_images} carrying no "
                            f"demographic label in '{demo_col}': they are in no skew, "
                            f"NDKL or amplification number on this page"
                        ]
                        if n_unlabelled
                        else []
                    ),
                    (
                        "The classifier's OWN demographic error skew: perceived "
                        "labels from a zero-shot model are not ground truth, and "
                        "its error rate can itself differ by group"
                        if inputs.get("_vision_classifier")
                        else "Live image classification: labels are perceived "
                        "demographics from an upstream classifier whose own error "
                        "can be demographically skewed; no vision sidecar ran on "
                        "this dispatch"
                    ),
                    "Text-to-image counterfactual probing and per-image quality disparity",
                ],
                power_note=(
                    "Skew and NDKL describe THIS label set against "
                    "the stated reference; they say nothing about "
                    "labels the classifier got wrong."
                ),
            ),
            "perVariable": [],
            "metrics": [],
            "bias": findings,
            "proxies": {
                "available": False,
                "notApplicable": True,
                "reason": "Proxy reconstruction applies to tabular features, not image labels.",
                "proxies": [],
                "chains": [],
            },
            "statistical": {
                "available": False,
                "notApplicable": True,
                "reason": "The tabular robustness battery does not apply to representation counts.",
                "perAttribute": [],
            },
            "intersectional": {"skipped": True, "reason": "Image-set triage."},
            "causal": {"nodes": [], "edges": [], "overview": "Not applicable to an image set."},
            "recommendedDefinition": {},
            "interventions": [],
            "nextStep": (
                "Convert to a full assessment for per-image "
                "FairFace/CLIP classification + T2I counterfactual "
                "probing (vision sidecar)."
            ),
        },
    }
