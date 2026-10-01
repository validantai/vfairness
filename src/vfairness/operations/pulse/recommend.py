"""Canonical recommenders for Pulse: fairness definition + interventions.

Rule-based and citable -- no model, no fabrication. These were previously
hand-rolled inside the consumer handler; centralised here so Pulse and the
Navigator give the same, defensible guidance.

Grounding:
* Fairness-definition choice by task/domain: Barocas, Hardt & Narayanan,
  *Fairness and Machine Learning* (ch. on classification criteria);
  EEOC Uniform Guidelines (US four-fifths) vs EU/UK proportionality
  (Directives 2000/43/EC, 2000/78/EC; Equality Act 2010); Turing AI
  Fairness Module 1-2.
* Works WITHOUT ground truth: demographic parity / four-fifths and
  representation are label-free; equal-opportunity / equalized-odds /
  predictive-parity require labels and are recommended only when present.
* Bias -> intervention mapping: Turing Module 2/4 intervention families
  (pre / in / post processing).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Split a human definition label into words on every run of non-alphanumerics,
# so "False-positive parity / predictive parity" becomes
# ["false", "positive", "parity", "predictive", "parity"].
_LABEL_WORD_RE = re.compile(r"[^a-z0-9]+")


def _label_words(label: Any) -> List[str]:
    """Lowercase WHOLE words of a definition label, punctuation removed."""
    return [w for w in _LABEL_WORD_RE.split(str(label).lower()) if w]


def _label_has(label: Any, phrase: str) -> bool:
    """True when ``phrase`` occurs in ``label`` as whole words, in order.

    Whole-token, never substring. This is the cali-BRATIO-n rule (CLAUDE.md,
    reconciled in 47f1e09f8) applied to definition LABELS: ``"calibration" in
    label.lower()`` also matches "multicalibration" and "recalibration error",
    and ``"parity" in label.lower()`` matches any label that merely mentions
    parity in passing. Both of those tests drove user-facing recommendation
    prose here, so a loose match puts the wrong paragraph, including the
    Kleinberg/Chouldechova impossibility trade-off, in front of a reader.

    A multi-word phrase must match as a consecutive run of words, so
    "predictive parity" matches "False-positive parity / predictive parity" but
    not a label that happens to contain both words far apart.

    Pinned by tests/test_pulse_recommend_tokens.py.
    """
    words = _label_words(label)
    wanted = _label_words(phrase)
    if not wanted or len(wanted) > len(words):
        return False
    return any(words[i : i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1))


def recommend_fairness_definition(
    domain: str,
    jurisdiction: str,
    has_ground_truth: bool,
    harm_direction: Optional[str] = None,
    scores_exposed: bool = False,
    base_rates_differ: Optional[bool] = None,
) -> Dict[str, Any]:
    """Pick a primary + secondary fairness definition for the case.

    Returns a dict the UI renders directly: primary, secondaries[],
    rationale, how_to_apply, label_free (bool).

    G-19 routing (all optional, backwards compatible):
      * ``harm_direction``: 'punitive' (a positive decision harms the
        person it lands on: fraud flag, risk score) emphasizes the
        false-positive / predictive-parity family; 'assistive' (the
        decision grants a benefit: hire, loan, treatment) emphasizes
        equal opportunity (true-positive-rate parity).
      * ``scores_exposed``: scores shown to decision-makers add
        calibration-within-groups to the applied definitions.
      * ``base_rates_differ``: when calibration and error-rate parity
        are both applied AND group base rates differ, the returned
        ``tradeoff`` states the Kleinberg/Chouldechova impossibility
        and names which emphasis was chosen and why. None means the
        base rates could not be measured (no ground truth).

    The routed emphasis is surfaced as ``headlineMetric`` (one of
    'selection_rate' | 'equal_opportunity' | 'predictive_parity' |
    'calibration'); the frontend reorders/flags metric cards from it.
    ``definitionsApplied`` / ``rejected`` / ``tradeoff`` are ADDITIVE to
    the existing keys, never replacements.
    """
    d = (domain or "").strip().lower()
    j = (jurisdiction or "").strip().lower()

    # Domain priors. The primary is a NORMATIVE choice -- which harm matters
    # most in this context -- and is independent of what happens to be
    # computable on the uploaded data (Turing fairness-definition module:
    # the primary definition is chosen from the harm model, then propagates
    # through measurement, Pareto and interventions; computability only
    # changes HOW it is measured, never WHICH one is the objective).
    label_dependent_primary = False
    if any(k in d for k in ("lend", "credit", "loan", "mortgage", "insurance")):
        primary = "Equal opportunity"
        secondaries = ["Demographic parity / four-fifths", "Calibration within groups"]
        label_dependent_primary = True
        rationale = (
            "Lending must not deny applicants who would actually qualify "
            "(repay) because of their group: equal opportunity is the "
            "normative objective. Demographic parity / four-fifths is the "
            "legal adverse-impact screen, monitored as a secondary signal."
        )
    elif any(k in d for k in ("hir", "recruit", "employ", "promotion")):
        primary = "Equal opportunity"
        secondaries = ["Demographic parity / four-fifths"]
        label_dependent_primary = True
        rationale = (
            "Hiring must not reject qualified candidates because of their "
            "group: equal opportunity (qualified-applicant parity) is the "
            "normative objective. The adverse-impact (four-fifths) test is "
            "the legal screen, monitored as a secondary signal."
        )
    elif any(k in d for k in ("health", "clinic", "medical", "diagnos", "triage")):
        primary = "Equalized odds"
        secondaries = ["Calibration within groups", "Equal opportunity"]
        label_dependent_primary = True
        rationale = (
            "In clinical decisions both false negatives and false positives "
            "carry asymmetric harm, so error rates must match across groups."
        )
    elif any(k in d for k in ("recidiv", "justice", "bail", "parole", "police")):
        primary = "Equalized odds"
        secondaries = ["Calibration within groups", "False-positive parity"]
        label_dependent_primary = True
        rationale = (
            "Liberty-affecting decisions: equal false-positive rates across "
            "groups is the central concern (cf. COMPAS debate)."
        )
    else:
        primary = "Demographic parity / four-fifths"
        secondaries = ["Equal opportunity"]
        rationale = (
            "No strong domain prior; start from the disparity headline and "
            "refine once the harm model is clearer."
        )

    # Jurisdiction framing (the *threshold* is jurisdiction-specific).
    if any(k in j for k in ("us", "u.s", "united states", "america")):
        legal = (
            "US: read the disparity against the EEOC four-fifths (80%) rule, "
            "paired with a statistical-significance test (the confidence "
            "interval here provides it)."
        )
    elif any(k in j for k in ("eu", "europe", "european")):
        legal = (
            "EU: there is no fixed numeric threshold. A statistically "
            "significant disparity triggers a proportionality / objective-"
            "justification assessment (Directives 2000/43/EC, 2000/78/EC; "
            "EU AI Act Art. 10)."
        )
    elif any(k in j for k in ("uk", "united kingdom", "britain", "england")):
        legal = (
            "UK: no fixed threshold; the Equality Act 2010 uses a "
            "particular-disadvantage + proportionality test."
        )
    else:
        legal = (
            "The US four-fifths rule is not universal. The statistically "
            "significant disparity shown is the jurisdiction-neutral signal "
            "that warrants review."
        )

    # Computability changes HOW the primary is measured, never WHICH
    # definition is the objective. A label-dependent primary with no
    # ground truth stays the normative target; it is approximated now and
    # measured directly once labels exist.
    primary_computable = (not label_dependent_primary) or has_ground_truth
    if not label_dependent_primary:
        how = (
            f"{primary} is the objective and is directly computable here "
            f"(it needs only the inputs and the model's decision). {legal}"
        )
    elif has_ground_truth:
        how = (
            f"{primary} is the objective and is computable here: this "
            f"dataset has the outcome column it needs. {legal} Also track "
            f"{secondaries[0].lower()} as a secondary signal."
        )
    else:
        proxy = (
            secondaries[0]
            if secondaries and _label_has(secondaries[0], "parity")
            else "the selection-rate (four-fifths) gap"
        )
        how = (
            f"{primary} remains the primary, normative objective for this "
            "use case even though it cannot be computed directly here: this "
            "dataset has no ground-truth outcomes. Do not switch the "
            f"objective. For now, approximate it with {proxy.lower()} and the "
            "disparity headline (a same-direction lower-bound proxy), and "
            "treat closing that gap as progress toward "
            f"{primary.lower()}. {legal} To measure {primary.lower()} "
            "directly, collect ground-truth labels or run a controlled "
            "audit; optimisation can still proceed against the proxy in the "
            "meantime."
        )

    # ──────────────────────────────────────────────────────────────────
    # G-19: harm-direction routing + honest applied/rejected/trade-off
    # record. Everything below is ADDITIVE to the legacy keys.
    # ──────────────────────────────────────────────────────────────────
    hd = (harm_direction or "").strip().lower()
    if hd not in ("assistive", "punitive"):
        hd = ""

    # Default headline from the domain prior, then routed by harm model.
    headline_metric = (
        "equal_opportunity"
        if primary in ("Equal opportunity", "Equalized odds")
        else "selection_rate"
    )
    applied: List[Dict[str, str]] = [
        {
            "definition": primary,
            "why": "Domain prior for this use case: " + rationale,
        }
    ]
    rejected: List[Dict[str, str]] = []

    if hd == "punitive":
        headline_metric = "predictive_parity"
        applied.append(
            {
                "definition": "False-positive parity / predictive parity",
                "why": (
                    "You indicated a positive decision harms the person "
                    "it lands on (punitive). The first-order check is "
                    "that no group is wrongly flagged more often, so the "
                    "false-positive-rate / predictive-parity family "
                    "leads the headline."
                ),
            }
        )
        rejected.append(
            {
                "definition": "Equal opportunity (true-positive-rate parity)",
                "why": (
                    "De-emphasized for a punitive system: missing a true "
                    "positive spares the person, while a false positive "
                    "is the harm event. It remains visible as a "
                    "secondary check, not the headline."
                ),
            }
        )
    elif hd == "assistive":
        headline_metric = "equal_opportunity"
        applied.append(
            {
                "definition": "Equal opportunity (true-positive-rate parity)",
                "why": (
                    "You indicated the decision grants a benefit "
                    "(assistive). The first-order harm is qualified "
                    "people being missed, so true-positive-rate parity "
                    "leads the headline."
                ),
            }
        )
        rejected.append(
            {
                "definition": "False-positive parity / predictive parity",
                "why": (
                    "De-emphasized for an assistive system: a false "
                    "positive grants a benefit to someone unqualified, "
                    "a lesser harm than denying a qualified person. It "
                    "remains visible as a secondary check."
                ),
            }
        )

    if scores_exposed:
        applied.append(
            {
                "definition": "Calibration within groups",
                "why": (
                    "Scores are exposed to decision-makers, so a given "
                    "score must mean the same thing for every group; "
                    "calibration is applied alongside the headline "
                    "definition."
                ),
            }
        )

    if not has_ground_truth:
        rejected.append(
            {
                "definition": "Directly measured error-rate parity",
                "why": (
                    "This dataset has no ground-truth outcome column, so "
                    "error-rate definitions stay the objective but are "
                    "approximated through the selection-rate gap until "
                    "labels are collected. Nothing was measured that the "
                    "data cannot support."
                ),
            }
        )

    # De-duplicate by definition name, first mention wins.
    seen_defs: List[str] = []
    deduped: List[Dict[str, str]] = []
    for a in applied:
        if a["definition"] not in seen_defs:
            seen_defs.append(a["definition"])
            deduped.append(a)
    applied = deduped

    def _is_error_rate(name: str) -> bool:
        return any(
            _label_has(name, k)
            for k in ("opportunity", "odds", "false-positive", "predictive parity")
        )

    emphasis_phrase = {
        "predictive_parity": "the false-positive / predictive-parity "
        "family (wrongful flags are the harm)",
        "equal_opportunity": "equal opportunity, i.e. true-positive-rate "
        "parity (missed qualified people are the "
        "harm)",
        "calibration": "calibration within groups",
        "selection_rate": "the selection-rate (demographic-parity) reading",
    }[headline_metric]
    calibration_applied = any(_label_has(a["definition"], "calibration") for a in applied)
    error_applied = any(_is_error_rate(a["definition"]) for a in applied)

    if calibration_applied and error_applied and base_rates_differ:
        tradeoff = (
            "Calibration and error-rate parity cannot all hold at once "
            "when groups have different base rates (Kleinberg, "
            "Mullainathan and Raghavan 2017; Chouldechova 2017), and "
            "this data shows differing base rates. Both are reported, "
            "but the headline emphasis chosen here is "
            + emphasis_phrase
            + "; calibration is monitored as a secondary check rather "
            "than enforced jointly."
        )
    elif calibration_applied and error_applied:
        confirm = (
            "group base rates could not be measured on this run (no ground-truth column)"
            if base_rates_differ is None
            else "no material base-rate difference was confirmed on this data"
        )
        tradeoff = (
            "Calibration and error-rate parity can only hold together "
            "when group base rates match; " + confirm + ", so both are "
            "applied without a forced choice. If base rates diverge, "
            "the headline emphasis (" + emphasis_phrase + ") takes "
            "priority and calibration becomes the secondary check."
        )
    else:
        tradeoff = (
            "Headline emphasis is " + emphasis_phrase + "; the other "
            "definitions listed are tracked as secondary signals, not "
            "traded against the headline on this run."
        )

    return {
        "primary": primary,
        "secondaries": secondaries,
        "rationale": rationale,
        "howToApply": how,
        # True only when the PRIMARY itself needs no labels at all --
        # never True for a label-dependent primary, even when ground
        # truth happens to be present (that case is what
        # primaryComputableHere says; labelFree must stay the negation
        # of primaryNeedsLabels or the two keys contradict each other).
        "labelFree": not label_dependent_primary,
        "primaryNeedsLabels": bool(label_dependent_primary),
        "primaryComputableHere": bool(primary_computable),
        # G-19 additions (additive; the frontend keys card ordering and
        # verdict framing off headlineMetric).
        "definitionsApplied": applied,
        "rejected": rejected,
        "tradeoff": tradeoff,
        "headlineMetric": headline_metric,
        "harmDirection": hd or None,
    }


# Detected-bias-type -> recommended control family. Each entry: what to do,
# which vfairness mechanism implements it, and the pipeline stage.
_INTERVENTION_MAP = {
    "representation": {
        "title": "Rebalance representation",
        "control": "Reweighting or resampling under-represented groups",
        "mechanism": "vfairness.post_processing.reweighting / pre-processing resample",
        "stage": "pre-processing",
        "how": "Reweight training rows so each group's effective size meets the "
        "minimum, or collect more data for the named small groups before relying "
        "on the model.",
    },
    "proxy": {
        "title": "Cut proxy leakage",
        "control": "Suppress or transform proxy features",
        "mechanism": "vfairness.preprocessing.feature_engineering.FeatureSuppressor",
        "stage": "pre-processing",
        "how": "Remove or decorrelate the named proxy columns; dropping the "
        "protected attribute alone is not enough while proxies remain.",
    },
    "historical": {
        "title": "Break the historical pattern",
        "control": "Relabel / reweight outcomes encoding past discrimination",
        "mechanism": "vfairness.post_processing.reweighting + label audit",
        "stage": "pre-processing",
        "how": "Audit the labels that encode the historical pattern; reweight or "
        "re-source them so the model does not learn the inherited bias.",
    },
    "disparity": {
        "title": "Equalize the decision",
        "control": "Group-aware threshold optimization",
        "mechanism": "vfairness.post_processing.threshold_optimization",
        "stage": "post-processing",
        "how": "Tune per-group decision thresholds to equalize the chosen "
        "fairness metric while holding overall accuracy.",
    },
    "calibration": {
        "title": "Calibrate within groups",
        "control": "Per-group calibration",
        "mechanism": "vfairness.post_processing.calibration",
        "stage": "post-processing",
        "how": "Fit a per-group calibration map so a given score means the same "
        "thing for every group.",
    },
}


#: Bias-type synonyms that route to an intervention family. Whole-word, in
#: order, exactly like _label_has. The KEY is the intervention; the values are
#: the detected-category names that legitimately mean it.
#:
#: READINESS-6, 2026-09-10. The mapping was ``k in str(bt).lower()``, which is
#: the cali-BRATIO-n substring test this file's own ``_label_has`` docstring
#: forbids, three hundred lines above. MEASURED: 'multicalibration' and
#: 'recalibration_error' both routed to the `calibration` intervention ("fit a
#: per-group calibration map"), which is not what either of them asks for.
#: Meanwhile 'intersectional', 'label_bias' and 'measurement' matched nothing
#: and silently returned only the always-on `disparity` fallback, with no
#: signal anywhere that the detected bias type had not been recognised.
_BIAS_TYPE_SYNONYMS: Dict[str, List[str]] = {
    "representation": [
        "representation",
        "representation bias",
        "sample size",
        "underrepresentation",
        "coverage",
    ],
    "proxy": ["proxy", "proxy leakage", "redundant encoding", "surrogate"],
    "historical": [
        "historical",
        "historical bias",
        "label bias",
        "label_bias",
        "legacy bias",
        "inherited bias",
    ],
    "disparity": [
        "disparity",
        "outcome disparity",
        "demographic parity",
        "adverse impact",
        "selection rate",
    ],
    "calibration": ["calibration", "calibration bias", "miscalibration", "score calibration"],
}

#: Bias types this mapper knows it does NOT have an intervention family for.
#: Naming them beats a silent fallback: the reader is told the detection was
#: understood and that the control is out of scope here.
_BIAS_TYPES_WITHOUT_CONTROL: Dict[str, str] = {
    "intersectional": (
        "Intersectional bias needs a control chosen per intersecting group; no single "
        "pre/in/post-processing family applies. Re-run the disparity controls per "
        "intersection rather than on the marginal attributes."
    ),
    "measurement": (
        "Measurement bias sits in how the feature or outcome was recorded, upstream of "
        "any model control. It is fixed in instrumentation, not by reweighting or "
        "thresholding."
    ),
    "multicalibration": (
        "Multicalibration is a stronger requirement than per-group calibration (it holds "
        "across a whole family of overlapping subgroups). The per-group calibration "
        "control does not satisfy it; see the multicalibration boosting literature."
    ),
    "recalibration_error": (
        "A recalibration error is a defect in a fitted calibration map, not a bias family. "
        "Refit or re-validate the existing map before adding another control."
    ),
}


def recommend_interventions(bias_types: List[str]) -> List[Dict[str, str]]:
    """Map the bias types Pulse detected to concrete controls + how-to.

    `bias_types` is the set of detected categories (e.g. from BiasDetector:
    'representation', 'proxy', 'historical', 'disparity', 'calibration').
    Always returns at least the disparity control so the user has a path.

    Matching is WHOLE-WORD through ``_label_has`` (see its docstring for the
    cali-BRATIO-n rule this fixes). A bias type that matches no intervention
    family is no longer dropped in silence: it comes back as a row with
    ``biasType`` set to what was detected, ``matched: False`` and a plain
    reason, so the reader can tell "we have no control for this" from "we
    found nothing".
    """
    seen: List[str] = []
    out: List[Dict[str, str]] = []
    unmatched: List[str] = []
    for bt in list(bias_types) + ["disparity"]:
        key = None
        for candidate, phrases in _BIAS_TYPE_SYNONYMS.items():
            if any(_label_has(bt, phrase) for phrase in phrases):
                key = candidate
                break
        if key is None:
            label = str(bt)
            if label not in unmatched:
                unmatched.append(label)
            continue
        if key not in seen:
            seen.append(key)
            out.append({"biasType": key, "matched": "true", **_INTERVENTION_MAP[key]})
    for label in unmatched:
        words = _label_words(label)
        note = next(
            (
                text
                for name, text in _BIAS_TYPES_WITHOUT_CONTROL.items()
                if _label_words(name) == words
            ),
            (
                "No intervention family in this library maps to this bias type. It was "
                "detected and is reported here so it is not lost; choosing a control for "
                "it is a decision, not a lookup."
            ),
        )
        out.append(
            {
                "biasType": label,
                "matched": "false",
                "title": f"No mapped control for '{label}'",
                "control": "Not mapped",
                "mechanism": "",
                "stage": "unmapped",
                "how": note,
            }
        )
    return out
