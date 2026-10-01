"""Canonical Pulse orchestrator -- the single source that assembles a
valuable, statistically-rigorous fairness read from a dataset.

One library function so Pulse (the task consumer) and the Navigator share
*identical* logic; the divergent hand-rolled consumer handler is the root
cause of the recent breakages and is replaced by a thin shim that calls
:func:`run_pulse`.

Pipeline (each stage wrapped -- a partial failure degrades, never raises):

  1. prepare_protected_attributes  (Phase-1 canonical binning: DOB->age,
     identifiers excluded, quantile bins, sibling preference)
  2. data quality                  (build_quality_report / DataBiasValidator)
  3. PER protected variable: FairnessAnalyzer.compute_all_metrics with
     BOOTSTRAP CONFIDENCE INTERVALS (statistical rigor) + group sizes;
     significance = CI excludes 0 / clears the four-fifths line
  4. bias taxonomy + proxies + intersectional  (BiasDetector.full_audit)
  5. causal overview               (build_causal_skeleton -> direct vs
     indirect/proxy paths, optional DoWhy identifiability)
  6. recommended fairness definition (works without ground truth)
  7. recommended interventions / controls (mapped from detected bias)
  8. per-variable verdict + worst-case roll-up (Kearns subgroup approach)

Result is a JSON-serialisable superset of the legacy contract (legacy keys
kept so the current UI keeps working; new keys: ``perVariable``,
``interventions``, richer ``causal``) -- the frontend redesign consumes the
new keys.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# The library's ONE text normaliser/tokeniser, shared with every lexicon in
# vfairness.llm.scorers. Imported at module scope on purpose: vfairness/
# __init__.py loads `.llm` before `.operations`, so scorers is always already
# initialised by the time this module executes, and a lazy import inside
# _label_quality would be swallowed by its own _safe() wrapper on failure.
from vfairness._triage import is_measured
from vfairness.llm.scorers import identifier_tokens

_MIN_GROUP_DEFAULT = 20  # Audit-grade default: 20 keeps small
# protected cells visible while still
# producing usable selection-rate
# estimates (Wilson 95% half-width on a
# p=0.5 cell at n=20 is ±0.22, tight
# enough to flag adverse impact, loose
# enough to read with a low_n_warning).
# The historical Turing M3 number (30)
# is for academic *reporting*; the audit
# context wants small protected cells
# visible by default. Callers can still
# override via inputs.minGroupSize.
_MIN_GROUP = _MIN_GROUP_DEFAULT  # legacy alias; still referenced where the
# CONSTANT is the right anchor (e.g. the
# "minReliable" diagnostics field). For the
# per-run gate, prefer the runtime override
# read from inputs.minGroupSize in run_pulse
# (see _resolve_min_group below).


def _resolve_min_group(inputs: Dict[str, Any]) -> int:
    """Read the per-run min_group_size override from the caller's inputs.

    The library default is 20 (_MIN_GROUP_DEFAULT: the audit-grade
    default that keeps small protected cells visible; the Turing M3
    academic reporting minimum of 30 is deliberately NOT the default,
    see the constant's comment). A caller can override per run via
    ``inputs.minGroupSize``
    (camelCase) or ``inputs.min_group_size``. Values are clamped to
    [5, 200] to keep statistical sanity: below 5 the rate estimate is
    dominated by noise; above 200 you'll filter every cell on most real
    datasets and the analysis becomes vacuous. Returns the default if no
    override or the value isn't a positive int.
    """
    v = inputs.get("minGroupSize")
    if v is None:
        v = inputs.get("min_group_size")
    try:
        n = int(v) if v is not None else _MIN_GROUP_DEFAULT
    except (TypeError, ValueError):
        return _MIN_GROUP_DEFAULT
    if n < 5:
        return 5
    if n > 200:
        return 200
    return n


_FOUR_FIFTHS = 0.80
_BOOTSTRAP = 1000  # CI rigor
_CI = 0.95

# G-25 individual-fairness thresholds (documented, never silent):
#   * consistency floor 0.80: the kNN consistency metric convention
#     (Zemel et al. 2013): below 0.8, more than one in five of each
#     individual's nearest neighbours on the NON-protected features
#     receives a different decision, i.e. similar people are treated
#     differently. Severity: critical below 0.60 (a coin-flip-like
#     neighbourhood), warn otherwise.
#   * Theil between-group share ceiling 0.20: when more than 20% of
#     the inequality in who receives the favorable outcome lies BETWEEN
#     protected groups rather than within them, group membership (not
#     individual variation) is a first-order driver of the outcome.
#     Severity: critical above 0.50 (group membership dominates), warn
#     otherwise.
_IF_MIN_ROWS = 100  # below this, kNN agreement is noise
_IF_K = 10  # neighbours per point (standard k)
_IF_SAMPLE_CAP = 2000  # kNN evaluated on a seeded subsample cap
_IF_CONSISTENCY_FLOOR = 0.80
_IF_CONSISTENCY_CRITICAL = 0.60
_IF_BETWEEN_SHARE_CEILING = 0.20
_IF_BETWEEN_SHARE_CRITICAL = 0.50

# G-25 remainder, counterfactual flip threshold (documented):
#   * counterfactual fairness asks for INVARIANCE under a protected-
#     attribute flip on an otherwise identical row (Kusner et al. 2017),
#     and Pulse models are deterministic at scoring time, so ANY nonzero
#     flip rate is direct evidence the decision depends on the protected
#     attribute (warn). Critical at >= 5%: one decision in twenty
#     changing on the protected attribute alone.
_CF_FLIP_CRITICAL = 0.05

# G-38 label-quality knobs (documented):
#   * label base-rate skew reuses the four-fifths band as an EVIDENTIARY
#     screen on the recorded favorable-outcome rate per group (it is not
#     a legal threshold here; labels are not selections). Critical below
#     0.5: the reference group's favorable label is recorded at more
#     than double the rate.
#   * annotator agreement: flagged when the lowest group's inter-
#     annotator agreement sits >= 10 points below the highest group's
#     (labels are less reliable exactly for that group).
_LQ_SKEW_RATIO = 0.8
_LQ_SKEW_CRITICAL = 0.5
_LQ_AGREEMENT_GAP = 0.10
#: Column-name tokens that mark a human labelling pass. Matched as WHOLE
#: identifier tokens (see ``_lq_annotator_columns``), never as substrings.
#:
#: READINESS-6, 2026-09-10. This was ("annotator", "rater", "coder") matched
#: with ``t in str(c).lower()``. MEASURED: a frame carrying ``encoder_output``
#: and ``decoder_state`` selected BOTH of them ("coder" is a substring of
#: en-CODER and de-CODER), passed the ``len(ann_cols) >= 2`` gate, and computed
#: an inter-annotator agreement statistic and Cohen's kappa over two
#: neural-network layer outputs. In the same measurement a genuinely
#: multi-annotated frame using ``judge_1 / judge_2 / reviewer_a / worker_id``
#: selected NOTHING and took the notApplicable branch. Wrong in both
#: directions at once, from one substring test.
_LQ_ANNOTATOR_TOKENS = (
    "annotator",
    "annotation",
    "rater",
    "coder",
    "judge",
    "reviewer",
    "labeler",
    "labeller",
    "labelers",
    "grader",
    "assessor",
    "worker",
    "turker",
)

#: A label column carries a small set of repeated labels. Above this many
#: distinct values it is an identifier or free text, not something two people
#: can agree on: ``worker_id`` passes the NAME test and must still be refused.
_LQ_MAX_LABEL_CARDINALITY = 20

# G-44: ethical-stance tags per metric (T1-11). Every fairness metric
# embeds a moral worldview (Friedler, Scheidegger & Venkatasubramanian
# 2016/2021): WAE ("we're all equal") metrics require equal outcome
# rates and treat label differences as structural bias; WYSIWYG ("what
# you see is what you get") metrics trust the recorded labels as ground
# truth and require equal treatment CONDITIONAL on them. Neither is
# "correct"; picking one is a values decision, which is why the tag is
# surfaced instead of silently choosing. Explicit per-name entries only
# (the exact keys FairnessAnalyzer.compute_all_metrics emits); an
# unmapped metric carries no tag rather than a guessed one.
_METRIC_STANCE: Dict[str, Dict[str, str]] = {
    "demographic_parity_difference": {
        "basis": "group",
        "worldview": "WAE",
        "note": (
            "Requires equal favorable-outcome rates across groups, "
            "regardless of the recorded labels: differences in the "
            "labels are treated as structural bias to correct."
        ),
    },
    "demographic_parity_ratio": {
        "basis": "group",
        "worldview": "WAE",
        "note": (
            "The ratio form of demographic parity (the four-fifths "
            "screen reads this): equal outcome rates regardless of "
            "recorded labels."
        ),
    },
    "equalized_odds_difference": {
        "basis": "group",
        "worldview": "WYSIWYG",
        "note": (
            "Trusts the recorded outcomes as ground truth and "
            "requires equal error rates conditional on them; "
            "inherits any bias in how the labels were made."
        ),
    },
    "equal_opportunity_difference": {
        "basis": "group",
        "worldview": "WYSIWYG",
        "note": (
            "Equal true-positive rates among the label-qualified; "
            "trusts the labels to say who is qualified."
        ),
    },
    "predictive_parity_difference": {
        "basis": "group",
        "worldview": "WYSIWYG",
        "note": ("Equal precision conditional on the recorded outcome; trusts the labels."),
    },
    "mae_parity_difference": {
        "basis": "group",
        "worldview": "WYSIWYG",
        "note": ("Equal mean absolute error against recorded values; trusts the recorded values."),
    },
    "rmse_parity_difference": {
        "basis": "group",
        "worldview": "WYSIWYG",
        "note": (
            "Equal root-mean-square error against recorded values; trusts the recorded values."
        ),
    },
}

# G-35 cohort-error-analysis knobs (documented):
#   * cohorts are level-crossings of feature PAIRS (Azure error-tree
#     style, depth 2), categorical features with <= 8 levels only;
#   * a cohort needs n >= 30 to be read (below that the error rate is
#     sampling noise);
#   * the retained evaluation set is capped at the 50 worst cohorts;
#   * a finding fires when the worst cohort's error rate is at least
#     DOUBLE the baseline error rate (lift >= 2; critical at >= 3).
_COHORT_MIN_N = 30
_COHORT_MAX_LEVELS = 8
_COHORT_MAX_FEATURES = 12  # lowest-cardinality features enter pairs
_COHORT_MAX_COHORTS = 50
_COHORT_LIFT_WARN = 2.0
_COHORT_LIFT_CRITICAL = 3.0

# G-42 trimmed-mean robustness: when dropping each group's extreme 5% of
# scores (2.5% per tail) moves the selection-rate gap by more than 20%
# relative, the headline gap partially rides on extreme-score rows
# (data-entry outliers or a thin tail) and the row gets a robustnessNote.
_TRIM_TAIL = 0.025
_TRIM_DIVERGENCE = 0.20
#: A relative change needs a baseline. Below this absolute selection-rate gap,
#: a "the gap moved by more than 20 percent" note is a ratio of nearly nothing
#: against nearly nothing, so no note is emitted. READINESS-6, 2026-09-10.
_TRIM_MIN_BASELINE = 0.005


def _safe(fn, default, *, degraded=None, label=None):
    """Run `fn`; on ANY exception return `default` so a section never
    crashes Pulse. When `degraded` (a list) is supplied, a swallowed
    failure is RECORDED -- a silently-substituted default must never be
    indistinguishable from a real clean result. The caller folds recorded
    degradations into the verdict so a collapsed load-bearing stage can
    never read as a pass.
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 -- a section must never crash Pulse
        if degraded is not None:
            degraded.append(
                {
                    "stage": str(label or getattr(fn, "__name__", "section")),
                    "error": f"{type(e).__name__}: {e}"[:300],
                }
            )
        return default


def _jsonify(o: Any) -> Any:
    """Recursively coerce numpy scalars/arrays to plain JSON types.

    FairnessAnalyzer / pandas leak np.bool_ / np.float64 which Deno's JSON
    and Python's json both reject. The result MUST be serialisable end to
    end (consumer -> submit-result -> browser), so sanitise once at the top.
    """
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonify(v) for v in o]
    if isinstance(o, np.generic):
        v = o.item()
        # NaN AND +/-Infinity -> None. Non-finite floats (e.g. an inf
        # disparity ratio when a reference group's selection rate is 0)
        # serialise as `NaN`/`Infinity` tokens that strict JSON parsers
        # (the browser, the consumer transport) reject -- "Out of range
        # float values are not JSON compliant". Coerce once, here.
        return None if isinstance(v, float) and not math.isfinite(v) else v
    if isinstance(o, float):
        return None if not math.isfinite(o) else o  # NaN/Inf -> None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    # pandas / numpy containers can leak from library calls. This runs
    # OUTSIDE _safe, so an un-handled object here fails the WHOLE Pulse
    # (the consumer JSON-encodes the result). Coerce defensively; never let
    # the result be unserialisable.
    if isinstance(o, pd.DataFrame):
        return _jsonify(o.reset_index().to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return _jsonify({str(k): v for k, v in o.to_dict().items()})
    if isinstance(o, (pd.Index, np.ndarray)):
        return _jsonify(list(o))
    if hasattr(o, "to_dict"):
        try:
            return _jsonify(o.to_dict())
        except Exception:  # noqa: BLE001
            return str(o)
    import dataclasses

    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _jsonify(dataclasses.asdict(o))
    if hasattr(o, "__dict__"):
        return _jsonify(vars(o))
    return str(o)


def _coerce_binary(s: pd.Series) -> np.ndarray:
    """Length-preserving 0/1 coercion.

    MUST NOT drop rows -- the result is index-aligned with the group /
    sensitive columns elsewhere. Null-prediction rows are removed once,
    up-front, in run_pulse so every array stays the same length and the
    same rows. (A previous dropna() here silently misaligned groups.)
    """
    if s.dtype == bool:
        return s.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        u = set(pd.unique(num.dropna()))
        if u and u <= {0, 1}:
            return num.fillna(0).astype(int).to_numpy()
        thr = float(np.nanmedian(num)) if num.notna().any() else 0.5
        return (num.fillna(thr - 1) >= thr).astype(int).to_numpy()
    low = s.astype("string").str.strip().str.lower()
    truthy = {
        "yes",
        "true",
        "1",
        "y",
        "approved",
        "hired",
        "selected",
        "positive",
        "accept",
        "accepted",
        "pass",
    }
    return low.isin(truthy).astype(int).to_numpy()


def _pick(df: pd.DataFrame, names) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    for c in df.columns:
        cl = c.lower()
        if any(cl == n or cl.endswith("_" + n) or cl.startswith(n + "_") for n in names):
            return c
    return None


def _tone(gap: float) -> str:
    return "critical" if gap >= 0.20 else "warn" if gap >= 0.10 else "pass"


# Canonical tone ranks. The library emits several severity vocabularies
# (warn/critical here; low/medium/high/severe/info from the bias detector;
# high from the proxy outcome screen). They MUST all map to a rank, and an
# UNRECOGNISED token must never silently rank 0 ("pass") -- that
# downgraded real high/medium/severe findings to pass inside _worst_tone.
_TONE_RANK = {
    "pass": 0,
    "ok": 0,
    "none": 0,
    "info": 0,
    "warn": 1,
    "low": 1,
    "medium": 1,
    "moderate": 1,
    "watch": 1,
    "review": 1,
    "critical": 2,
    "high": 2,
    "severe": 2,
    "strong": 2,
    "significant": 2,
}


def _tone_rank(t: str) -> int:
    """Rank a tone/severity token. Unknown -> 1 (warn), never 0 (pass):
    an unrecognised severity is, by definition, still a flagged finding."""
    return _TONE_RANK.get(str(t).strip().lower().split(".")[-1], 1)


def _worst_tone(*tones: str) -> str:
    """The most severe of several tones (audit-honest: never downgrade).
    Collapses any non-canonical token to the nearest canonical tone."""
    rank = max((_tone_rank(t) for t in tones), default=0)
    return {0: "pass", 1: "warn", 2: "critical"}[rank]


def _legal_screen_framework(domain: str, jurisdiction: str) -> Dict[str, Any]:
    """Return the disparity-severity framework that applies to a Pulse
    run given its (domain, jurisdiction) inputs.

    Why this exists: the four-fifths (0.80) ratio is a LEGAL trigger
    under EEOC Uniform Guidelines (US, employment) -- elsewhere it is
    either a useful auditor heuristic, an informational reference, or
    not applicable. A universal Pulse must respect the framework the
    caller is auditing under, not assume US/employment for every run.

    The returned dict always carries the observed-ratio thresholds (so
    the GUI can still show "below 0.80") but ``mode`` controls whether
    those thresholds *alone* trigger severity or whether they are gated
    on statistical confirmation:

        ``ratio_drives_severity``  -- EEOC-style: ratio alone triggers
        ``ratio_is_heuristic``     -- ratio shown; severity needs
                                      ratio + statistical confirmation
                                      OR a meaningful effect size

    ``framework`` is a human-readable string with a citation so the
    Pulse output can name the basis on which the verdict was given --
    an auditor must be able to point at WHICH rule was applied.

    Returns
    -------
    dict with keys ``framework, ratio_warn, ratio_critical, mode``.
    """
    dl = (domain or "").strip().lower()
    jl = (jurisdiction or "").strip().lower()

    EMPLOYMENT = (
        "employ",
        "hir",
        "recruit",
        "promot",
        "termin",
        "layoff",
        "lay-off",
        "discharge",
        "workforce",
        "staff",
    )
    LENDING = ("lend", "credit", "loan", "mortgage", "underwrit", "fintech", "bnpl")
    HOUSING = ("hous", "rent", "tenant", "lease")
    HEALTH = ("health", "medic", "clinic", "patient", "hospital", "diagnos", "triage", "screen")
    EDUCATION = ("educat", "admission", "school", "univers", "college", "scholar")
    INSURANCE = ("insur", "actuar", "underwriting risk")

    def is_us():
        return (
            jl in ("", "us", "usa", "united states", "united states of america")
            or "us " in jl + " "
        )

    # US + employment: EEOC 4/5ths is the bright-line legal screen.
    if any(k in dl for k in EMPLOYMENT) and is_us():
        return {
            "framework": "EEOC Uniform Guidelines, 4/5ths rule (US, employment, 29 CFR §1607.4(D))",
            "ratio_warn": 0.80,
            "ratio_critical": 0.50,
            "mode": "ratio_drives_severity",
        }
    # EU + employment: Race Equality Directive 2000/43, Framework Dir 2000/78.
    # No bright-line ratio; the four-fifths gap is a useful flag for audits
    # but legal trigger is statistical disparate impact + justification.
    if any(k in dl for k in EMPLOYMENT) and (
        "eu" in jl or "europ" in jl or "uk" in jl or "united kingdom" in jl or "england" in jl
    ):
        return {
            "framework": "EU Equality Directives / UK Equality Act 2010 "
            "(employment) - ratio is informational; legal "
            "trigger is statistical disparate-impact testing",
            "ratio_warn": 0.80,
            "ratio_critical": 0.50,
            "mode": "ratio_is_heuristic",
        }
    # Lending: ECOA Reg B (US) / Consumer Credit Directive (EU). No bright
    # ratio; both jurisdictions look at statistical disparity + justification.
    if any(k in dl for k in LENDING):
        return {
            "framework": (
                "ECOA Reg B disparate-impact testing (US lending) "
                if is_us()
                else "EU Consumer Credit Directive disparate-impact testing (lending)"
            )
            + ": ratio shown; severity driven by statistical "
            "tests + effect size",
            "ratio_warn": 0.80,
            "ratio_critical": 0.50,
            "mode": "ratio_is_heuristic",
        }
    # Housing (US FHA), healthcare, education, insurance -- all
    # statistical-disparate-impact frameworks, no bright-line ratio.
    for keys, label in (
        (HOUSING, "Fair Housing Act / EU housing-discrimination law"),
        (HEALTH, "Civil Rights Act §1557 (HHS) / EU patient-rights directives"),
        (EDUCATION, "Title VI / EU non-discrimination in education"),
        (INSURANCE, "ACA / state insurance commissioners / EU IDD"),
    ):
        if any(k in dl for k in keys):
            return {
                "framework": (
                    f"{label}: statistical disparate-impact analysis; ratio shown alongside stats"
                ),
                "ratio_warn": 0.80,
                "ratio_critical": 0.50,
                "mode": "ratio_is_heuristic",
            }
    # Default / unknown / generic: report the ratio as a heuristic flag with
    # statistical confirmation gating severity. Honest default for an audit
    # tool that doesn't know which framework the caller is operating under.
    return {
        "framework": (
            f"Generic adverse-impact heuristic "
            f"(domain={domain or 'unspecified'}, "
            f"jurisdiction={jurisdiction or 'unspecified'}); "
            "no specific framework matched, so the ratio is shown alongside "
            "statistical tests"
        ),
        "ratio_warn": 0.80,
        "ratio_critical": 0.50,
        "mode": "ratio_is_heuristic",
    }


def _disparity_tone(
    four_fifths: float,
    significant: bool,
    worst_rate: float,
    best_rate: float,
    framework: Optional[Dict[str, Any]] = None,
    effect_size_h: Optional[float] = None,
) -> str:
    """Universal disparity-severity rule, framework-aware.

    Inputs:
        four_fifths: observed worst/best selection-rate ratio
        significant: post-FDR statistical confirmation flag
        worst_rate, best_rate: the rates the ratio was computed from
        framework: from ``_legal_screen_framework(domain, jurisdiction)``;
                   None falls back to the generic heuristic (ratio +
                   statistical confirmation).
        effect_size_h: Cohen's h on the proportion gap, if available

    Rules:
        - Zero-rate worst group with a non-zero best group is critical
          UNIVERSALLY (exclusion is a regulator-actionable finding in
          every framework that recognises adverse impact).
        - ``ratio_drives_severity`` (EEOC US-employment): ratio alone
          triggers severity tiers, regardless of significance. This is
          the audit-correct EEOC reading -- the legal trigger is the
          observed ratio, not the p-value. Significance is conveyed
          separately via the "not confirmed" tag on the row.
        - ``ratio_is_heuristic`` (everything else): the ratio is shown
          but severity requires ratio + (statistical confirmation OR
          meaningful effect size) so we don't escalate noise on a
          framework that doesn't recognise the 0.80 bright line.
    """
    if best_rate <= 0:
        return "pass"
    # Universal: a non-zero best group plus a zero worst group is exclusion.
    if worst_rate <= 0.0:
        return "critical"
    if framework is None:
        framework = {"ratio_warn": 0.80, "ratio_critical": 0.50, "mode": "ratio_is_heuristic"}
    ratio_warn = float(framework.get("ratio_warn", 0.80))
    ratio_critical = float(framework.get("ratio_critical", 0.50))
    mode = framework.get("mode", "ratio_is_heuristic")
    if mode == "ratio_drives_severity":
        if four_fifths < ratio_critical:
            return "critical"
        if four_fifths < ratio_warn:
            return "warn"
        return "pass"
    # ratio_is_heuristic: gate the escalation on statistical signal.
    h = abs(effect_size_h or 0.0)
    has_stat_support = bool(significant) or h >= 0.20
    if four_fifths < ratio_critical and (significant or h >= 0.50):
        return "critical"
    if four_fifths < ratio_warn and has_stat_support:
        return "warn"
    return "pass"


# Backwards-compat alias: older call sites and any external integrations
# importing _ff_tone keep working with the legacy signature, but now they
# resolve through the universal helper with the generic-heuristic mode.
def _ff_tone(four_fifths: float, significant: bool, worst_rate: float, best_rate: float) -> str:
    return _disparity_tone(four_fifths, significant, worst_rate, best_rate)


def _as_mapping(o: Any) -> Dict[str, Any]:
    """Coerce a dict / dataclass / plain object to a plain dict so the
    snake/camel pickers work. intersectional_disparity_analysis returns
    GroupAdvantage *dataclass instances* (not dicts) when called in-process;
    the Navigator only ever saw dicts because its dispatch JSON-serialises
    first. Pulse calls the engine directly, so coerce here."""
    if isinstance(o, dict):
        return o
    import dataclasses

    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return dict(getattr(o, "__dict__", {}) or {})


def _scalar(v: Any) -> Any:
    """Enums (e.g. Severity.HIGH) -> their bare name; pass others through."""
    if v is None or isinstance(v, (str, int, float, bool, list, dict)):
        return v
    name = getattr(v, "name", None) or getattr(v, "value", None)
    return str(name if name is not None else v).split(".")[-1].lower()


def _norm_int_group(g: Any) -> Optional[Dict[str, Any]]:
    """Snake/camel-tolerant -> Pulse camelCase intersectional group."""
    if g is None:
        return None
    g = _as_mapping(g)
    if not g:
        return None

    def pick(*k):
        return next((g[x] for x in k if x in g and g[x] is not None), None)

    # The two relative rates are THREE-STATE, and `pick(...) or 1.0` collapsed
    # them to two. `pick` already answers None for absent-or-None, so the
    # `or 1.0` added nothing for the missing case and instead fired
    # ADDITIONALLY on a MEASURED 0.0. On this scale 0.0 and 1.0 are opposite
    # ends: relative_to_best == 0.0 means the group received zero positive
    # outcomes (total exclusion, the strongest disparate-impact reading there
    # is, which is why intersectional.py tracks zero_selection_alerts), while
    # 1.0 means exact parity with the best group. The single worst measurement
    # was reported as the single best.
    #
    # A genuinely MISSING ratio stays None rather than becoming 1.0, because
    # emitting parity for "we did not measure this" is the same defect in a
    # quieter form. The consumer renders could-not-check from None.
    #
    # BGL-S2c (2026-09-17). The test below read `if value is None`, and that is
    # the WRONG half of the third state for this producer: intersectional.py
    # sets these two ratios to float("nan"), not None, whenever their
    # denominator is empty (overall or best selection rate of 0). So the
    # could-not-check warning never fired, and the row went out carrying a bare
    # NaN, which json.dumps(..., allow_nan=False) refuses outright. Measured at
    # the engine boundary (120 rows, nobody selected, so overall_rate is 0.0):
    #
    #   relativeToOverall: nan, relativeToBest: nan, warnings: []
    #   json.dumps(out, allow_nan=False) -> ValueError
    #
    # `is_measured` answers False for None, NaN and the infinities alike, which
    # is the whole question being asked here. The emitted value is None, the
    # JSON null the consumer already renders as could-not-check.
    rel_overall = pick("relative_to_overall", "relativeToOverall")
    rel_best = pick("relative_to_best", "relativeToBest")
    unreported = [
        field
        for field, value in (
            ("relativeToOverall", rel_overall),
            ("relativeToBest", rel_best),
        )
        if not is_measured(value)
    ]
    if not is_measured(rel_overall):
        rel_overall = None
    if not is_measured(rel_best):
        rel_best = None
    group_label = _scalar(pick("group", "label", "name")) or ""

    # The engine's own per-cell reasons (GroupAdvantage.unmeasured), carried
    # across the boundary so the null below is never a bare blank: the reader
    # gets the field name AND why it has no number. Keys are re-cased to match
    # the camelCase fields they describe in this row, so one absent figure is
    # named once rather than twice under two spellings.
    def _camel(name: str) -> str:
        head, *rest = str(name).split("_")
        return head + "".join(part.title() for part in rest)

    cell_unmeasured = {_camel(k): v for k, v in (pick("unmeasured") or {}).items()}
    if unreported:
        warnings.warn(
            f"Intersectional group {group_label!r} reported no measured "
            f"{' and no measured '.join(unreported)}; emitted as None (not "
            "measured) rather than 1.0, which would have claimed parity.",
            UserWarning,
            stacklevel=2,
        )
        for field in unreported:
            cell_unmeasured.setdefault(
                field,
                "this ratio has an empty denominator, so it was never measured",
            )

    # Same treatment for the FPR: a cell with no actual negatives carries NaN
    # here, and `pick(...) or 0.0` let it straight through because NaN is
    # truthy. Emitted as None with its reason named, never as a rate of 0.0
    # (which reads as a PERFECT false positive rate, the best possible result).
    fpr_raw = pick("false_positive_rate", "falsePositiveRate")
    fpr_out: Optional[float] = fpr_raw if fpr_raw is not None else 0.0
    if fpr_raw is not None and not is_measured(fpr_raw):
        cell_unmeasured.setdefault(
            "falsePositiveRate",
            "no actual negatives in this cell, so the false positive rate "
            "has an empty denominator and was never measured",
        )
        fpr_out = None

    return {
        "group": group_label,
        "positiveRate": pick("positive_rate", "positiveRate") or 0.0,
        "groundTruthRate": pick("ground_truth_rate", "groundTruthRate") or 0.0,
        "falsePositiveRate": fpr_out,
        "predictionDelta": pick("prediction_delta", "predictionDelta") or 0.0,
        "size": pick("size") or 0,
        "relativeToOverall": rel_overall,
        "relativeToBest": rel_best,
        # Field name -> why it has no number. Empty when every figure in this
        # row is a real measurement, so a consumer can tell the two apart
        # without reading this source.
        "unmeasured": cell_unmeasured,
        "disparityContribution": pick("disparity_contribution", "disparityContribution") or 0.0,
        "severity": _scalar(pick("severity")) or "info",
        # Confidence caveat: True when the cell's n is below the low-n
        # warning threshold (see data_treatment.low_n_warning_threshold).
        "lowNWarning": bool(pick("low_n_warning", "lowNWarning") or False),
    }


def _norm_int_finding(f: Any) -> Dict[str, Any]:
    fd = _as_mapping(f) if f is not None else {}

    def pick(*k):
        return next((fd[x] for x in k if x in fd and fd[x] is not None), None)

    ci = _as_mapping(pick("confidence_interval", "confidenceInterval")) or None

    # BGL-S2c (2026-09-17). This mapping is a field-by-field rebuild, so the
    # `unmeasured` map that intersectional.py now attaches to each finding was
    # simply not listed and was DROPPED at the boundary: the engine measured
    # honestly, named the reason, and the Pulse envelope handed the platform
    # `metricValues: {..., 'false_positive_rate': nan}` with nothing at all
    # saying that NaN was never a measurement. Measured on the 250-row fixture
    # with one cell that has no actual negatives: the engine emits one finding
    # carrying unmeasured={'false_positive_rate': ...}; the normalised finding
    # had no such key.
    #
    # Any metricValue that is a non-finite number is nulled here (JSON has no
    # NaN: json.dumps(..., allow_nan=False) refused the whole payload) and
    # named in the same map, so could-not-check crosses as an explicit null
    # WITH its reason, never as a silent blank and never as a number.
    metric_values = dict(pick("metric_values", "metricValues") or {})
    unmeasured = dict(pick("unmeasured") or {})
    for key, value in list(metric_values.items()):
        # A bool is not a measurement and `is_measured` rejects it, but a
        # boolean flag in metric_values is not a could-not-check either, so it
        # is left exactly as it is rather than nulled.
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not is_measured(value):
            unmeasured.setdefault(
                key,
                f"{key} is not a finite number, so it was never measured for this finding",
            )
            metric_values[key] = None

    return {
        "type": _scalar(pick("type")) or "over_prediction",
        "severity": _scalar(pick("severity")) or "info",
        "groups": pick("groups") or [],
        "metricValues": metric_values,
        # Field name -> why it has no number, empty when the finding is made
        # entirely of real measurements.
        "unmeasured": unmeasured,
        "description": pick("description") or "",
        "pValue": pick("p_value", "pValue"),
        "pValueCorrected": pick("p_value_corrected", "pValueCorrected"),
        "statisticallySignificant": pick("statistically_significant", "statisticallySignificant"),
        "nGroupsTested": pick("n_groups_tested", "nGroupsTested"),
        "effectSizeH": pick("effect_size_h", "effectSizeH"),
        "effectSizeInterpretation": pick("effect_size_interpretation", "effectSizeInterpretation"),
        "minimumDetectableEffect": pick("minimum_detectable_effect", "minimumDetectableEffect"),
        "powerWarning": pick("power_warning", "powerWarning"),
        "confidenceInterval": (
            {
                "lower": ci.get("lower", 0.0),
                "upper": ci.get("upper", 0.0),
                "level": ci.get("level", 0.95),
                "method": ci.get("method", "unknown"),
            }
            if isinstance(ci, dict)
            else None
        ),
    }


def _normalise_intersectional(raw: Dict[str, Any], label_free: bool) -> Dict[str, Any]:
    """Map intersectional_disparity_analysis() output into the camelCase
    Pulse contract, mirroring the Navigator's IntersectionalAnalysisResult
    so the platform renders one shared shape."""
    inter = raw.get("intersectional_analysis") or raw.get("intersectionalAnalysis") or {}

    def g(*k):
        return next((inter[x] for x in k if x in inter and inter[x] is not None), None)

    summ = raw.get("statistical_summary") or raw.get("statisticalSummary") or {}
    # Transparency triple lives at top level (mirrored under the
    # intersectional sub-key for back-compat). Promote to the Pulse shape so
    # the GUI can render a Data Treatment panel + Zero Invites callout.
    excluded = (
        raw.get("excluded_groups")
        or raw.get("excludedGroups")
        or inter.get("excluded_groups")
        or inter.get("excludedGroups")
        or []
    )
    zero_alerts = (
        raw.get("zero_selection_alerts")
        or raw.get("zeroSelectionAlerts")
        or inter.get("zero_selection_alerts")
        or inter.get("zeroSelectionAlerts")
        or []
    )
    dtreat = (
        raw.get("data_treatment")
        or raw.get("dataTreatment")
        or inter.get("data_treatment")
        or inter.get("dataTreatment")
        or {}
    )
    return {
        "skipped": False,
        "labelFree": label_free,
        "privilegedGroup": _norm_int_group(g("privileged_group", "privilegedGroup")),
        "disadvantagedGroup": _norm_int_group(g("disadvantaged_group", "disadvantagedGroup")),
        "allGroups": [
            x for x in (_norm_int_group(z) for z in (g("all_groups", "allGroups") or [])) if x
        ],
        "overallRate": g("overall_rate", "overallRate") or 0.0,
        "overallGroundTruthRate": g("overall_ground_truth_rate", "overallGroundTruthRate"),
        "maxDisparity": g("max_disparity", "maxDisparity") or 0.0,
        "maxGroundTruthDisparity": g("max_ground_truth_disparity", "maxGroundTruthDisparity"),
        "disparitySeverity": g("disparity_severity", "disparitySeverity") or "info",
        "outcomePolarity": g("outcome_polarity", "outcomePolarity"),
        "insights": raw.get("insights") or [],
        "recommendations": raw.get("recommendations") or [],
        "findings": [_norm_int_finding(x) for x in (raw.get("findings") or [])],
        "statisticalSummary": {
            "totalFindings": summ.get("total_findings", summ.get("totalFindings", 0)),
            "significantFindings": summ.get(
                "significant_findings", summ.get("significantFindings", 0)
            ),
            "correctionMethod": summ.get("correction_method", summ.get("correctionMethod", "none")),
            "alpha": summ.get("alpha", 0.05),
        }
        if summ
        else None,
        # Transparency (Pulse contract addition v2.0)
        # excludedGroups: cells that fell below the size gate, kept so the
        # GUI can say "we did NOT analyse these, here's why".
        # zeroInviteAlerts: cells where positive_count == 0 and n >= floor,
        # surfaced regardless of size gate (the audit-grade headline).
        # dataTreatment: full audit log of thresholds + counts + rates.
        "excludedGroups": [
            {
                "group": str(e.get("group", "")),
                "size": int(e.get("size", 0) or 0),
                "positiveCount": int(e.get("positive_count", e.get("positiveCount", 0)) or 0),
                "positiveRate": float(e.get("positive_rate", e.get("positiveRate", 0.0)) or 0.0),
                "groundTruthRate": float(
                    e.get("ground_truth_rate", e.get("groundTruthRate", 0.0)) or 0.0
                ),
                "reason": str(e.get("reason", "")),
            }
            for e in excluded
            if isinstance(e, dict)
        ],
        "zeroInviteAlerts": [
            {
                "group": str(z.get("group", "")),
                "size": int(z.get("size", 0) or 0),
                "positiveCount": int(z.get("positive_count", z.get("positiveCount", 0)) or 0),
                "groundTruthRate": float(
                    z.get("ground_truth_rate", z.get("groundTruthRate", 0.0)) or 0.0
                ),
                "analysed": bool(z.get("analysed", False)),
                "reason": str(z.get("reason", "")),
            }
            for z in zero_alerts
            if isinstance(z, dict)
        ],
        "dataTreatment": {
            "minGroupSize": int(dtreat.get("min_group_size", dtreat.get("minGroupSize", 0)) or 0),
            "lowNWarningThreshold": int(
                dtreat.get("low_n_warning_threshold", dtreat.get("lowNWarningThreshold", 0)) or 0
            ),
            "zeroSelectionFloor": int(
                dtreat.get("zero_selection_floor", dtreat.get("zeroSelectionFloor", 0)) or 0
            ),
            "outcomePolarity": str(
                dtreat.get("outcome_polarity", dtreat.get("outcomePolarity", "positive_favorable"))
                or "positive_favorable"
            ),
            "nTotalCellsSeen": int(
                dtreat.get("n_total_cells_seen", dtreat.get("nTotalCellsSeen", 0)) or 0
            ),
            "nCellsIncluded": int(
                dtreat.get("n_cells_included", dtreat.get("nCellsIncluded", 0)) or 0
            ),
            "nCellsExcludedSmall": int(
                dtreat.get("n_cells_excluded_small", dtreat.get("nCellsExcludedSmall", 0)) or 0
            ),
            "nCellsWarningLowN": int(
                dtreat.get("n_cells_warning_low_n", dtreat.get("nCellsWarningLowN", 0)) or 0
            ),
            "nZeroSelectionAlerts": int(
                dtreat.get("n_zero_selection_alerts", dtreat.get("nZeroSelectionAlerts", 0)) or 0
            ),
            "overallRate": float(dtreat.get("overall_rate", dtreat.get("overallRate", 0.0)) or 0.0),
            "overallGroundTruthRate": float(
                dtreat.get("overall_ground_truth_rate", dtreat.get("overallGroundTruthRate", 0.0))
                or 0.0
            ),
        }
        if dtreat
        else None,
    }


def _intersectional(
    work: pd.DataFrame,
    usable: List[str],
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray],
    polarity: str,
    min_group: int = _MIN_GROUP_DEFAULT,
) -> Dict[str, Any]:
    """Navigator-grade intersectional disparity (dual-lens) over the
    cross-product of the chosen protected attributes -- e.g. age x sex x
    ethnicity yields subgroups like "25-34 · female · Chinese". Uses the
    SAME engine the Navigator's vfairness_intersectional_disparity dispatch
    uses (intersectional_disparity_analysis), so the two surfaces agree.
    Needs >=2 assessable attributes to be intersectional."""
    if len(usable) < 2:
        return {
            "skipped": True,
            "reason": "Intersectional analysis needs at least two protected "
            "attributes (for example age and sex). Only one assessable "
            "attribute was selected, so there are no intersections to read.",
        }
    from itertools import combinations

    from vfairness.evaluation.vfairness_metrics.intersectional import (
        intersectional_disparity_analysis,
    )

    n = min(len(work), len(y_pred))
    has_truth = y_true is not None
    # No ground truth: feed predictions as the reference so the subgroup
    # selection-rate ranking + disparity still compute (delta collapses to
    # 0 and the section is flagged labelFree -- exactly the Navigator's
    # label-free posture).
    yt = y_true[:n] if y_true is not None else y_pred[:n]
    single = {a: work[a].astype("string").fillna("missing").to_numpy()[:n] for a in usable}

    def _combo(cols: List[str]) -> np.ndarray:
        c = work[cols[0]].astype("string").fillna("missing")
        for a in cols[1:]:
            c = c.str.cat(work[a].astype("string").fillna("missing"), sep=" · ")
        return c.to_numpy()[:n]

    def _run(cols: List[str], mgs: int) -> Dict[str, Any]:
        return intersectional_disparity_analysis(
            yt,
            y_pred[:n],
            _combo(cols),
            single_attributes=single,
            min_group_size=mgs,
            outcome_polarity=polarity,
        )

    def _n_groups(raw: Dict[str, Any]) -> int:
        ia = raw.get("intersectional_analysis") or raw.get("intersectionalAnalysis") or {}
        return len(ia.get("all_groups") or ia.get("allGroups") or [])

    def _singleaxis_dominance(raw: Dict[str, Any]) -> Tuple[float, int, int]:
        """Quantify whether the deepest cross is "intersectional" in name
        only -- i.e. whether the variance across surviving cells is
        explained almost entirely by ONE attribute.

        Method: one-way variance decomposition. For each attribute that
        varies, compute SSbetween / SStotal on the selection-rate column
        (the eta-squared of grouping by that single attribute). The
        maximum across attributes is the "single-axis dominance" score.
        Score near 1.0 means one attribute carries essentially all the
        variance; score near 0.0 means variance is spread evenly across
        attributes (a genuinely intersectional finding).

        Returns (dominance_score, n_varying_attrs, n_cells).

        Dataset-agnostic: works on any set of cells, no thresholds
        baked into the data shape. The caller compares dominance against
        a fixed cutoff (0.70 by default) to decide whether to reject the
        deepest cross.
        """
        ia = raw.get("intersectional_analysis") or raw.get("intersectionalAnalysis") or {}
        gs = ia.get("all_groups") or ia.get("allGroups") or []
        labels: List[str] = []
        rates: List[float] = []
        for g in gs:
            if isinstance(g, dict):
                lab = g.get("group") or g.get("label") or ""
                r = g.get("positive_rate", g.get("positiveRate"))
            else:
                lab = getattr(g, "group", "")
                r = getattr(g, "positive_rate", None)
            if isinstance(lab, str) and lab and isinstance(r, (int, float)):
                labels.append(lab)
                rates.append(float(r))
        if len(labels) < 2:
            return 0.0, 0, len(labels)
        parts = [tuple(label.split(" · ")) for label in labels]
        ncols = min(len(p) for p in parts)
        rates_arr = np.asarray(rates, dtype=float)
        overall_mean = float(rates_arr.mean())
        ss_total = float(((rates_arr - overall_mean) ** 2).sum())
        if ss_total <= 1e-12:
            # All cells have the same rate -- no variance to decompose;
            # the cross can't tell us anything. Treat as fully dominated.
            return 1.0, 0, len(labels)
        n_varying = 0
        max_eta2 = 0.0
        for i in range(ncols):
            vals = [p[i] for p in parts]
            unique = set(vals)
            if len(unique) < 2:
                continue
            n_varying += 1
            ss_between = 0.0
            for v in unique:
                mask = [j for j, x in enumerate(vals) if x == v]
                if not mask:
                    continue
                grp_mean = float(rates_arr[mask].mean())
                ss_between += len(mask) * (grp_mean - overall_mean) ** 2
            eta2 = ss_between / ss_total
            if eta2 > max_eta2:
                max_eta2 = eta2
        return max_eta2, n_varying, len(labels)

    # 1. Try the deepest combination first (every chosen attribute crossed).
    #    With many attributes every cell is tiny, so step the minimum group
    #    size down before giving up -- a 12-person "young Chinese woman"
    #    cell is still worth surfacing (flagged low-reliability), which is
    #    the whole point of intersectional analysis.
    # Size ladder starts at the caller-resolved min_group (defaults to 20,
    # _MIN_GROUP_DEFAULT,
    # overridable via inputs.minGroupSize) and steps down to keep the analysis
    # alive on sparse intersections. Duplicates removed; preserved descending.
    SIZE_LADDER = sorted(
        {s for s in (min_group, 15, 10, 5) if s >= 5},
        reverse=True,
    )
    best: Optional[Dict[str, Any]] = None
    best_cols: List[str] = usable
    best_mgs = min_group
    # Track WHY a deepest-cross result was rejected so the GUI can name
    # the fallback honestly ("deepest cross only varied on one attribute").
    deepest_singleaxis_note: Optional[str] = None
    for mgs in SIZE_LADDER:
        raw = _run(usable, mgs)
        if _n_groups(raw) < 2:
            continue
        # ANTI-SINGLE-AXIS GUARD (information-theoretic). The previous
        # heuristic ("count varying attrs < 2") was easily fooled by
        # surviving cells that varied on 3 noise attributes while the
        # real disparity was carried by a 4th. Replaced with a one-way
        # variance decomposition (eta-squared) per attribute: if ONE
        # attribute explains >= 70% of the cell-to-cell variance in
        # selection rates, the cross is single-axis dominated regardless
        # of how many attributes happen to take different values.
        # Threshold is dataset-agnostic.
        eta2, n_varying, _ncells = _singleaxis_dominance(raw)
        if len(usable) >= 3 and (eta2 >= 0.70 or n_varying < 2):
            deepest_singleaxis_note = (
                f"Crossing every selected attribute produced cells whose "
                f"variance is {int(eta2 * 100)}% explained by a single "
                f"attribute, so the headline would read as 'intersectional' "
                f"but is statistically single-axis. Falling through to the "
                f"widest 2-way cross-attribute pair so the surfaced finding "
                f"is a genuine intersection."
            )
            continue
        best, best_mgs = raw, mgs
        break

    # 2. Full cross-product still too sparse: fall back to the 2-way
    #    intersection with the WIDEST real disparity (the Navigator renders
    #    every pair; here we surface the one that matters most).
    if best is None and len(usable) >= 2:
        best_disp = -1.0
        for pair in combinations(usable, 2):
            for mgs in SIZE_LADDER:
                raw = _run(list(pair), mgs)
                if _n_groups(raw) >= 2:
                    ia = (
                        raw.get("intersectional_analysis")
                        or raw.get("intersectionalAnalysis")
                        or {}
                    )
                    disp = float(ia.get("max_disparity") or ia.get("maxDisparity") or 0.0)
                    if disp > best_disp:
                        best, best_disp = raw, disp
                        best_cols, best_mgs = list(pair), mgs
                    break

    # 3. Nothing computable even at the smallest floor. Do NOT return a
    #    misleading "0 subgroups / 0 pt gap" and do NOT hand-wave -- quantify
    #    exactly why, with the statistical KPI that drives the call.
    if best is None:
        sizes = pd.Series(_combo(usable)).value_counts()
        total_sub = int(sizes.shape[0])
        largest = int(sizes.iloc[0]) if total_sub else 0
        at = {t: int((sizes >= t).sum()) for t in (30, 15, 10, 5)}
        # 95% margin of error on a SINGLE subgroup's selection rate at the
        # largest available cell, worst case p=0.5: ±1.96·sqrt(0.25/n).
        # This is the KPI: even the biggest cell's rate is ± this much, so a
        # between-group gap is indistinguishable from sampling noise.
        moe = (1.96 * (0.25 / largest) ** 0.5) if largest > 0 else 1.0
        # Rows needed per subgroup for a ±5pt rate (the usual readable bar).
        need = int(round((1.96**2) * 0.25 / (0.05**2)))  # ~385
        reason = (
            f"The {len(usable)} attributes you crossed form {total_sub:,} "
            f"combined subgroups over {n:,} rows. The largest has only "
            f"{largest} people, and {at[5]} subgroup(s) reach even 5; none "
            f"reach the {min_group}-person reporting minimum. At the largest "
            f"cell the 95% margin of error on a single subgroup's selection "
            f"rate is already ±{round(moe * 100)} points, so any "
            f"between-subgroup gap here is within sampling noise and cannot "
            f"be called real. Reliable intersectional reading needs roughly "
            f"{need} rows per subgroup: cross only the two or three "
            f"highest-priority attributes, or collect more data."
        )
        return {
            "skipped": True,
            "reason": reason,
            "diagnostics": {
                "rows": int(n),
                "subgroups": total_sub,
                "largestSubgroup": largest,
                "subgroupsAtLeast": {str(k): v for k, v in at.items()},
                "marginOfErrorLargest": float(moe),
                "minReliable": int(min_group),
                "rowsNeededPerSubgroup": need,
            },
        }

    out = _normalise_intersectional(best, label_free=not has_truth)
    out["crossedAttributes"] = best_cols
    out["minGroupSizeUsed"] = best_mgs
    # Echo the configured floor (the engine may have stepped down to a smaller
    # ladder rung) so the GUI can show both the policy and the effective value.
    out["minGroupSizeRequested"] = int(min_group)
    if deepest_singleaxis_note and best_cols != usable:
        # The deepest cross was rejected for being single-attribute-varying;
        # tell the user honestly which pair we used instead and why.
        out["note"] = deepest_singleaxis_note + (" Pair used: " + " × ".join(best_cols) + ".")
    if best_cols != usable:
        out["note"] = (
            "Crossing all selected attributes left every subgroup too small "
            "to read, so this shows the two attributes with the widest "
            "intersectional gap: " + " × ".join(best_cols) + "."
        )
    elif best_mgs < min_group:
        out["note"] = (
            f"Some intersectional subgroups are small (down to {best_mgs} "
            "people). They are shown because intersectional risk concentrates "
            "in small cells, but read the smallest ones with caution."
        )
    return out


def _per_variable(
    df: pd.DataFrame,
    attr: str,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray],
    min_group: int = _MIN_GROUP_DEFAULT,
    framework: Optional[Dict[str, Any]] = None,
    reference_override: Optional[str] = None,
) -> Dict[str, Any]:
    """One protected variable: per-group rate, gap, four-fifths, CI, n.

    ``framework`` controls how the four-fifths ratio elevates the row's
    tone, per ``_legal_screen_framework(domain, jurisdiction)``. Default
    (None) is the generic-heuristic mode: ratio + statistical
    confirmation. EEOC-US-employment uses the bright-line ratio rule.

    ``reference_override`` (G-32) is a caller-requested reference group
    for THIS attribute (from inputs.reference_group). When the requested
    group exists in the data it becomes the comparison baseline instead
    of the default most-populous reliable group; honored or not, the
    request is disclosed via ``referenceRequested`` /
    ``referenceOverrideHonored`` and the top-level ``referenceGroups``
    block.
    """
    g = df[attr].astype("string").fillna("missing").to_numpy()
    if len(g) != len(y_pred):
        m = min(len(g), len(y_pred))
        g, y_pred = g[:m], y_pred[:m]
    from vfairness.evaluation.vfairness_metrics import group_reliability

    sizes = {str(lab): int((g == lab).sum()) for lab in pd.unique(g)}
    rel = group_reliability(sizes)  # shared tiering (one source of truth)
    groups: List[Dict[str, Any]] = []
    for lab in pd.unique(g):
        mask = g == lab
        n = int(mask.sum())
        rate = float(y_pred[mask].mean()) if n else 0.0
        r = rel.get(str(lab), {"tier": "invalid", "interpretable": False, "note": ""})
        groups.append(
            {
                "label": str(lab),
                "n": n,
                "rate": rate,
                "tier": r["tier"],
                "interpretable": r["interpretable"],
                "note": r["note"],
                # Back-compat: "reliable" == n>=30 (reliable/caution tiers).
                "reliable": r["tier"] in ("reliable", "caution"),
            }
        )
    groups.sort(key=lambda d: d["rate"], reverse=True)
    rates = [gr for gr in groups if gr["n"] > 0]
    if len(rates) < 2:
        return {
            "attribute": attr,
            "groups": groups,
            "assessable": False,
            "reason": "Needs at least two groups with data.",
        }

    # Reference group = the MAJORITY (most-populous) reliable group, i.e. the
    # population baseline a regulator compares against. Never a tiny high-rate
    # group: ranking by rate alone made e.g. a 50-person 'Egypt' the reference
    # and produced a noisy, misleading headline gap. Restrict ref/worst to
    # reliable groups (n >= min) when at least two exist; small groups are
    # still reported and flagged, just not used to drive the headline.
    reliable = [gr for gr in rates if gr["n"] >= min_group]
    # Never let an 'invalid' (n<10, non-interpretable) group drive the
    # headline ref/worst -- its rate is noise. Fall back through tiers.
    interpretable = [gr for gr in rates if gr["interpretable"]]
    pool = reliable if len(reliable) >= 2 else interpretable if len(interpretable) >= 2 else rates
    ref = max(pool, key=lambda d: d["n"])
    worst = min(pool, key=lambda d: d["rate"])
    if worst["label"] == ref["label"]:
        # The majority is itself the worst-served group: compare it against
        # the best-served reliable group instead so the gap is meaningful.
        ref = max(pool, key=lambda d: d["rate"])
    # G-32: honor a caller-requested reference group when it exists in the
    # data. The requested group becomes the comparison baseline; the worst
    # group is then the lowest-rate group other than the reference. The
    # request is disclosed downstream whether honored or not.
    reference_requested = (
        str(reference_override).strip() if reference_override is not None else None
    )
    reference_honored = False
    if reference_requested:
        match = next((gr for gr in rates if gr["label"] == reference_requested), None)
        if match is None:  # forgiving case-insensitive fallback
            match = next(
                (gr for gr in rates if gr["label"].lower() == reference_requested.lower()), None
            )
        if match is not None:
            ref = match
            reference_honored = True
            others = [gr for gr in pool if gr["label"] != ref["label"]]
            if not others:
                others = [gr for gr in rates if gr["label"] != ref["label"]]
            if others:
                worst = min(others, key=lambda d: d["rate"])
    gap = ref["rate"] - worst["rate"]
    # EEOC four-fifths screen uses the highest-rate group as the benchmark,
    # independent of which group we narrate against.
    best_rate = max((d["rate"] for d in pool), default=0.0)
    four_fifths = (worst["rate"] / best_rate) if best_rate > 0 else 0.0

    # Bootstrap CI on the gap (statistical rigor: is it real or noise?).
    rng = np.random.default_rng(0)
    boots = []
    rmask = g == ref["label"]
    wmask = g == worst["label"]
    rp, wp = y_pred[rmask], y_pred[wmask]
    if len(rp) >= 5 and len(wp) >= 5:
        for _ in range(_BOOTSTRAP):
            b = (
                rp[rng.integers(0, len(rp), len(rp))].mean()
                - wp[rng.integers(0, len(wp), len(wp))].mean()
            )
            boots.append(b)
        lo, hi = np.quantile(boots, [(1 - _CI) / 2, 1 - (1 - _CI) / 2])
        significant = bool(lo > 0)  # CI excludes 0 -> a real disparity
        # Two-sided bootstrap p-value for H0: gap == 0, with +1 smoothing
        # so it can never be exactly 0. This feeds the BH-FDR correction
        # APPLIED ACROSS ALL ATTRIBUTES in run_pulse -- a single
        # uncorrected per-attribute test inflates the family-wise false
        # positive rate (~34% over 8 attributes) and is the dominant
        # spurious-Adverse driver. `significant` here is provisional and
        # is overwritten by the corrected decision downstream.
        ba = np.asarray(boots, dtype=float)
        nb = len(ba)
        p_two = 2.0 * min((np.sum(ba <= 0) + 1) / (nb + 1), (np.sum(ba >= 0) + 1) / (nb + 1))
        p_value = float(min(1.0, p_two))
        # Selection-aware omnibus gate (audit fix pv-1): `worst` is the
        # MINIMUM-rate group of k, so the ref-vs-worst bootstrap above is an
        # implicit max over k-1 comparisons and is anti-conservative under
        # the null (max-gap inflation, ~2.4x per AgentFairBench). Require a
        # k-group omnibus chi-square over ALL interpretable groups to reject
        # before the selected pair may count as significant; a pair picked
        # for being extreme cannot drive the verdict when the group-level
        # test sees no signal. Additive: p_value/CI are unchanged, only the
        # `significant` flag is gated.
        if significant and len(pool) > 2:

            def _omnibus() -> bool:
                from scipy.stats import chi2_contingency

                rows = []
                for gr in pool:
                    m = g == gr["label"]
                    yp = np.asarray(y_pred[m], dtype=float)
                    pos = int(np.nansum(yp >= 0.5))
                    neg = int(m.sum()) - pos
                    if pos + neg > 0:
                        rows.append([pos, neg])
                tbl = np.asarray(rows, dtype=float)
                if tbl.shape[0] < 2 or tbl.sum(axis=0).min() == 0:
                    return True  # degenerate table: do not over-gate
                return float(chi2_contingency(tbl)[1]) < (1.0 - _CI)

            significant = bool(_safe(_omnibus, True))
    else:
        # Too few rows in the reference or worst group to bootstrap a CI.
        # The previous code set lo=hi=gap, which made significant=(gap>0)
        # ALWAYS TRUE for any non-zero gap on a tiny group -- the exact
        # opposite of the intended "a small group must never inflate
        # severity". A disparity that cannot be tested is NOT significant.
        lo, hi = float("nan"), float("nan")
        significant = False
        p_value = None

    # Cohen's h on the worst/best proportion gap. h = 2 * (arcsin(sqrt(p1))
    # - arcsin(sqrt(p2))). h is the standard effect-size statistic for two
    # proportions; magnitudes: 0.2 = small, 0.5 = medium, 0.8 = large
    # (Cohen 1988). This feeds the ratio_is_heuristic branch of
    # _disparity_tone so non-EEOC frameworks (lending, housing, etc.)
    # can elevate tone on a confirmed-large effect even when BH-FDR
    # significance is borderline on a small sample.
    try:
        p_worst = max(0.0, min(1.0, float(worst["rate"])))
        p_best = max(0.0, min(1.0, float(best_rate)))
        effect_h = abs(2.0 * (np.arcsin(np.sqrt(p_best)) - np.arcsin(np.sqrt(p_worst))))
    except Exception:  # noqa: BLE001
        effect_h = 0.0

    return {
        "attribute": attr,
        "assessable": True,
        "groups": groups,
        "referenceGroup": ref["label"],
        "worstGroup": worst["label"],
        "gap": gap,
        "ciLow": None if lo != lo else float(lo),
        "ciHigh": None if hi != hi else float(hi),
        "fourFifthsRatio": four_fifths,
        "pValue": p_value,
        "significant": significant,
        # Snapshot of the per-attribute bootstrap-CI verdict BEFORE
        # BH-FDR correction. _fdr_correct() may overwrite `significant`
        # with the family-wise corrected value; this field never moves,
        # so the GUI can name which of the two checks rejected the gap.
        "significantRaw": significant,
        # Tone = the WORST of two honest severities:
        #  (a) the practical size of the absolute gap (_tone), and
        #  (b) the four-fifths / adverse-impact screen.
        # (b) is essential: a reliable group selected 0% of the time is
        # total exclusion (ratio 0.00) yet may have only a small ABSOLUTE
        # gap from a low base rate -- _tone(gap) alone rendered that as a
        # mild "pass/warn" bar, trivializing the single most severe finding.
        # Only escalate on a CONFIRMED disparity (significant) so noise from
        # a wide-CI small group never inflates severity.
        # Tone is the worst of:
        #   (a) gap-size severity (size-only heuristic)
        #   (b) framework-aware adverse-impact severity from the
        #       observed four-fifths ratio (EEOC-style ratio rule for
        #       US employment; ratio+stats for other contexts)
        # The framework is resolved at run_pulse top level from the
        # caller's domain/jurisdiction inputs and threaded down.
        "tone": _worst_tone(
            _tone(gap),
            _disparity_tone(
                four_fifths,
                significant,
                worst["rate"],
                best_rate,
                framework=framework,
                effect_size_h=effect_h,
            ),
        ),
        "effectSizeH": float(effect_h),
        # G-32 disclosure fields: what the caller asked for and whether
        # the engine could honor it (folded into referenceGroups).
        "referenceRequested": reference_requested,
        "referenceOverrideHonored": bool(reference_honored),
        "smallGroups": [
            gr["label"]
            for gr in groups
            if gr["tier"] in ("underpowered", "caution") and gr["n"] > 0
        ],
        "invalidGroups": [gr["label"] for gr in groups if gr["tier"] == "invalid" and gr["n"] > 0],
    }


def _best_protected_link(
    work: pd.DataFrame, feature: str, protected_attrs: List[str], min_strength: float = 0.10
) -> Optional[Dict[str, Any]]:
    """Find the protected attribute that a candidate proxy ``feature``
    correlates most strongly with, so an outcome-disparity finding can
    name WHICH protected trait the feature stands in for.

    Dataset-agnostic: uses Cramer's V for categorical-vs-categorical,
    point-biserial-like eta-squared for numeric-vs-categorical (and
    swap for symmetry), and absolute Spearman for numeric-vs-numeric.
    Returns the best match {attr, strength, method} above ``min_strength``,
    or None if nothing meaningful correlates. Designed to be the missing
    link that turns "zip_minority_majority drives outcome disparity"
    into "zip_minority_majority drives outcome disparity AND is a stand-
    in for race_ethnicity (Cramer's V = 0.42)" -- the narrative an
    auditor needs for the planted-bias mechanism to be visible.
    """
    if feature not in work.columns:
        return None
    fcol = work[feature]
    if fcol.isna().all():
        return None
    f_is_num = pd.api.types.is_numeric_dtype(fcol) and (fcol.nunique(dropna=True) > 12)
    best: Optional[Dict[str, Any]] = None
    for attr in protected_attrs:
        if attr == feature or attr not in work.columns:
            continue
        acol = work[attr]
        if acol.isna().all():
            continue
        a_is_num = pd.api.types.is_numeric_dtype(acol) and (acol.nunique(dropna=True) > 12)
        try:
            if not f_is_num and not a_is_num:
                # Categorical x categorical: Cramer's V (uniform 0..1).
                ct = pd.crosstab(
                    fcol.astype("string").fillna("missing"), acol.astype("string").fillna("missing")
                )
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                from scipy.stats import chi2_contingency

                chi2, _p, _dof, _ = chi2_contingency(ct, correction=False)
                n = int(ct.values.sum())
                k = min(ct.shape) - 1
                if n <= 0 or k <= 0:
                    continue
                strength = float(np.sqrt(chi2 / (n * k)))
                method = "cramers_v"
            elif f_is_num and not a_is_num:
                # Numeric feature vs categorical attribute: eta-squared
                # (one-way ANOVA proportion of variance explained).
                vals = pd.to_numeric(fcol, errors="coerce")
                grp = acol.astype("string").fillna("missing")
                mask = vals.notna() & grp.notna()
                vals, grp = vals[mask], grp[mask]
                if len(vals) < 4 or grp.nunique() < 2:
                    continue
                overall_mean = float(vals.mean())
                ss_total = float(((vals - overall_mean) ** 2).sum())
                if ss_total <= 1e-12:
                    continue
                ss_between = 0.0
                for v, mask_g in grp.groupby(grp).groups.items():
                    g_vals = vals.loc[mask_g]
                    if len(g_vals) == 0:
                        continue
                    ss_between += len(g_vals) * (float(g_vals.mean()) - overall_mean) ** 2
                strength = float(np.sqrt(ss_between / ss_total))
                method = "eta_squared"
            elif not f_is_num and a_is_num:
                # Symmetric: swap and reuse eta-squared.
                vals = pd.to_numeric(acol, errors="coerce")
                grp = fcol.astype("string").fillna("missing")
                mask = vals.notna() & grp.notna()
                vals, grp = vals[mask], grp[mask]
                if len(vals) < 4 or grp.nunique() < 2:
                    continue
                overall_mean = float(vals.mean())
                ss_total = float(((vals - overall_mean) ** 2).sum())
                if ss_total <= 1e-12:
                    continue
                ss_between = 0.0
                for v, mask_g in grp.groupby(grp).groups.items():
                    g_vals = vals.loc[mask_g]
                    if len(g_vals) == 0:
                        continue
                    ss_between += len(g_vals) * (float(g_vals.mean()) - overall_mean) ** 2
                strength = float(np.sqrt(ss_between / ss_total))
                method = "eta_squared"
            else:
                # Numeric x numeric: absolute Spearman.
                from scipy.stats import spearmanr

                vals_f = pd.to_numeric(fcol, errors="coerce")
                vals_a = pd.to_numeric(acol, errors="coerce")
                mask = vals_f.notna() & vals_a.notna()
                if mask.sum() < 8:
                    continue
                rho, _ = spearmanr(vals_f[mask], vals_a[mask])
                if rho is None or (isinstance(rho, float) and np.isnan(rho)):
                    continue
                strength = float(abs(rho))
                method = "spearman_abs"
        except Exception:  # noqa: BLE001 -- linker is best-effort
            continue
        if best is None or strength > best["strength"]:
            best = {"attr": attr, "strength": strength, "method": method}
    if best is None or best["strength"] < min_strength:
        return None
    return best


def _lq_annotator_columns(work: pd.DataFrame) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Columns that really are a human labelling pass, plus what was refused.

    Two gates, both needed (READINESS-6, 2026-09-10):

    * NAME, on whole identifier tokens. ``encoder_output`` and
      ``decoder_state`` were selected by the previous substring test because
      "coder" sits inside en-CODER, and Cohen's kappa was then computed over
      two neural-network layer outputs. ``identifier_tokens`` splits
      snake_case and camelCase, so "coder" matches ``coder_2`` and
      ``coderName`` and nothing else.
    * SHAPE. A name is not evidence. ``worker_id`` passes the name gate and is
      an identifier, not a label; two people cannot agree on it. A label
      column carries between 2 and _LQ_MAX_LABEL_CARDINALITY repeated values,
      and the retained columns must share at least one value with each other,
      because an agreement statistic over disjoint label vocabularies is
      arithmetic without a meaning.

    Returns ``(kept, rejected)`` where rejected is [(column, why)], so the
    notApplicable branch can say what it looked at instead of reporting a
    bare "none".
    """
    wanted = set(_LQ_ANNOTATOR_TOKENS)
    named = [c for c in work.columns if wanted & set(identifier_tokens(c))]
    kept: List[str] = []
    rejected: List[Tuple[str, str]] = []
    values: Dict[str, set] = {}
    for col in named:
        series = work[col].dropna()
        vals = set(series.astype("string").tolist())
        if len(vals) < 2:
            rejected.append((str(col), "only one distinct value"))
            continue
        if len(vals) > _LQ_MAX_LABEL_CARDINALITY:
            rejected.append(
                (str(col), f"{len(vals)} distinct values, an identifier rather than a label")
            )
            continue
        if len(vals) * 2 > len(series):
            # A label REPEATS: several units carry the same verdict. A column
            # whose values barely repeat is an id or a free-text note, and it
            # sails past the absolute cap above on any small frame (worker_id
            # over 20 rows has exactly 20 distinct values, and 20 > 20 is
            # False). Each gate is pinned separately in
            # tests/test_readiness6_textmatch.py, because a gate that is only
            # ever reached after another one already refused is untestable and
            # would rot.
            rejected.append(
                (
                    str(col),
                    f"{len(vals)} distinct values over {len(series)} rows, so its "
                    "values barely repeat: an identifier or free text, not a label",
                )
            )
            continue
        kept.append(str(col))
        values[str(col)] = vals
    # Overlapping vocabularies: keep only columns that share a value with at
    # least one other retained column.
    overlapping = [c for c in kept if any(o != c and values[c] & values[o] for o in kept)]
    for c in kept:
        if c not in overlapping:
            rejected.append((c, "its label values overlap no other annotator column"))
    return overlapping, rejected


def _proxy_outcome_disparity(
    work: pd.DataFrame,
    y_pred: np.ndarray,
    candidates: List[str],
    protected_attrs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Adverse impact carried *by a declared proxy column itself*.

    A redlining-style proxy may not correlate with any protected attribute
    in the realised sample (e.g. a minority-majority ZIP flag that is ~50/50
    and statistically independent of race) yet still drive an outcome gap:
    decisions differ sharply across its levels. The feature->protected
    screen is blind to this; this is the feature->decision screen that
    surfaces it. Low-cardinality declared proxy candidates only.

    When ``protected_attrs`` is supplied, we ALSO compute the strongest
    correlation between the feature and any protected attribute and
    emit a linked row tagged ``method='outcome_disparity_linked'``. This
    turns "zip_minority_majority drives an outcome gap" into
    "zip_minority_majority drives an outcome gap AND stands in for
    race_ethnicity (Cramer's V = 0.42)" -- the narrative an auditor
    needs to read the mechanism, not just the symptom.
    """
    out: List[Dict[str, Any]] = []
    n = min(len(work), len(y_pred))
    if n == 0:
        return out
    yp = np.asarray(y_pred[:n], dtype=float)
    # Two passes: one worst-vs-best contrast is examined per candidate
    # column, so the size of the test family is only known after the
    # scan. Bonferroni over that count keeps the "confirmed" wording
    # honest -- an uncorrected max-selected z at 1.96 would call noise
    # "confirmed p<0.05" once several candidates are screened.
    examined = 0
    hits: List[Dict[str, Any]] = []
    # Columns whose worst/best ratio cleared the four-fifths screen. They enter
    # no hypothesis family (nothing is tested on them); they are kept so the
    # "N comparisons examined" wording can name the screen it applied.
    screened: List[Dict[str, Any]] = []
    for col in candidates or []:
        if col not in work.columns:
            continue
        s = work[col].iloc[:n]
        # Numeric features with many unique values are binned into
        # tertiles so the outcome-disparity check works on continuous
        # signals (employment gap months, photo attractiveness score,
        # years experience etc.) -- not only categorical declared proxies.
        # Without this branch a planted B8/B9-style numeric proxy is
        # invisible to the disparity screen.
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 12:
            try:
                s = pd.qcut(s, q=3, duplicates="drop", labels=["low", "mid", "high"])
            except Exception:  # noqa: BLE001 -- qcut fails on heavy ties
                continue
        levels = [lv for lv in pd.unique(s.dropna())]
        if not (2 <= len(levels) <= 12):
            continue
        rates: List[Tuple[Any, float, int]] = []
        for lv in levels:
            m = (s == lv).to_numpy()
            cnt = int(m.sum())
            if cnt >= _MIN_GROUP:
                rates.append((lv, float(yp[m].mean()), cnt))
        if len(rates) < 2:
            continue
        best = max(r[1] for r in rates)
        worst = min(r[1] for r in rates)
        if best <= 0:
            continue
        ratio = worst / best
        if ratio >= 0.80:
            # READINESS-6, 2026-09-10. `examined += 1` used to sit ABOVE this
            # screen, so a column that exits here, with no z-test computed and
            # no hypothesis tested, still inflated the Bonferroni divisor
            # below. MEASURED on zip_flag at 24% vs 44% approval over 50+50
            # people (four-fifths 0.55, raw z p=0.03477):
            #     0 benign columns -> divisor  1 -> pAdj 0.0348 -> REPORTED
            #     1 benign column  -> divisor  2 -> pAdj 0.0695 -> SUPPRESSED
            #    10 benign columns -> divisor 11 -> pAdj 0.3825 -> SUPPRESSED
            # A real adverse-impact finding was erased by columns that were
            # never tested, and nothing recorded that a 0.55 ratio had been
            # found and dropped. The divisor is now the number of z-tests
            # actually in the family, and every screened-out ratio is kept in
            # `screened` so a suppression can be told from a clean result.
            screened.append({"col": str(col), "ratio": float(ratio)})
            continue  # within the four-fifths screen -> not adverse
        examined += 1  # one worst-vs-best contrast enters the family
        lo = min(rates, key=lambda r: r[1])
        hi = max(rates, key=lambda r: r[1])
        # A min/max selection-rate ratio over several levels is a
        # high-variance extremum: on small groups it drops below 0.80 by
        # sampling noise alone. Require the worst-vs-best gap to be
        # statistically significant (pooled two-proportion z) before
        # flagging, and never assign 'critical' from an unconfirmed point
        # ratio. Severities are canonical (warn/critical) only.
        n_lo, n_hi = lo[2], hi[2]
        s_lo = lo[1] * n_lo
        s_hi = hi[1] * n_hi
        p_pool = (s_lo + s_hi) / (n_lo + n_hi) if (n_lo + n_hi) else 0.0
        se = (p_pool * (1.0 - p_pool) * (1.0 / n_lo + 1.0 / n_hi)) ** 0.5
        z = abs(hi[1] - lo[1]) / se if se > 0 else 0.0
        # Two-sided normal p for the pooled two-proportion z; the
        # Bonferroni gate over the examined count runs after the scan,
        # once the family size is known.
        p_raw = math.erfc(z / math.sqrt(2.0)) if z > 0 else 1.0
        hits.append(
            {"col": str(col), "ratio": float(ratio), "lo": lo, "hi": hi, "pRaw": float(p_raw)}
        )
    for h in hits:
        p_adj = min(1.0, h["pRaw"] * max(examined, 1))
        if p_adj >= 0.05:
            # Not confirmed once corrected. It is NOT flagged as a finding
            # (severity "info", significant False, so no verdict roll-up can
            # escalate on it), but it is recorded: a four-fifths ratio below
            # the screen that was found and then dropped is exactly the fact
            # the previous code discarded in silence.
            out.append(
                {
                    "protectedAttribute": "(decision outcome)",
                    "feature": h["col"],
                    "strength": float(1.0 - h["ratio"]),
                    "method": "outcome_disparity_unconfirmed",
                    "fourFifthsRatio": float(h["ratio"]),
                    "significant": False,
                    "severity": "info",
                    "pValue": h["pRaw"],
                    "pValueAdjusted": float(p_adj),
                    "comparisonsExamined": examined,
                    "detail": (
                        f"Decisions differ by '{h['col']}' (four-fifths "
                        f"{h['ratio']:.2f}, below the 0.80 screen), but the "
                        f"worst-vs-best gap is not statistically confirmed once "
                        f"Bonferroni-corrected over {examined} tested "
                        f"comparison{'s' if examined != 1 else ''} "
                        f"(p={h['pRaw']:.4g}, adjusted {p_adj:.4g}). Recorded, "
                        "not flagged: this is an unconfirmed lead, and its "
                        "absence from the findings is a power limit rather than "
                        "evidence that the column is clean."
                    ),
                }
            )
            continue
        col, ratio = h["col"], h["ratio"]
        lo, hi = h["lo"], h["hi"]
        out.append(
            {
                "protectedAttribute": "(decision outcome)",
                "feature": col,
                "strength": float(1.0 - ratio),
                "method": "outcome_disparity",
                "fourFifthsRatio": float(ratio),
                "significant": True,
                "severity": "critical" if ratio < 0.50 else "warn",
                "pValue": h["pRaw"],
                "pValueAdjusted": float(p_adj),
                "comparisonsExamined": examined,
                "detail": (
                    f"Decisions differ by '{col}': "
                    f"{lo[0]} selected {lo[1] * 100:.1f}% vs "
                    f"{hi[0]} {hi[1] * 100:.1f}% (four-fifths {ratio:.2f}, "
                    f"difference confirmed, Bonferroni-corrected p<0.05 "
                    f"over {examined} comparison"
                    f"{'s' if examined != 1 else ''}). "
                    f"A stand-in column that itself produces an outcome gap."
                ),
            }
        )
        # NARRATIVE LINK: emit a second row that names which protected
        # attribute this proxy actually correlates with, so the GUI can
        # render "X stands in for Y" rather than the symptom-only
        # "(decision outcome)" finding. Best-effort -- skipped silently
        # if no protected attr clears the correlation floor.
        if protected_attrs:
            link = _best_protected_link(work, col, protected_attrs)
            if link is not None:
                out.append(
                    {
                        "protectedAttribute": link["attr"],
                        "feature": col,
                        "strength": float(link["strength"]),
                        "method": "outcome_disparity_linked",
                        "linkageMethod": link["method"],
                        "fourFifthsRatio": float(ratio),
                        "significant": True,
                        "severity": "critical" if ratio < 0.50 else "warn",
                        "detail": (
                            f"'{col}' drives an outcome gap (four-fifths "
                            f"{ratio:.2f}) AND correlates with protected "
                            f"attribute '{link['attr']}' "
                            f"({link['method']} = {link['strength']:.2f}). "
                            f"Mechanism: a stand-in feature laundering the "
                            f"protected trait into the decision."
                        ),
                    }
                )
    return out


def _proxies(
    work: pd.DataFrame,
    usable: List[str],
    exclude_cols: Optional[List[str]] = None,
    y_pred: Optional[np.ndarray] = None,
    proxy_candidates: Optional[List[str]] = None,
    pred_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Proxy screen: which features stand in for a protected attribute (the
    mechanism behind "drop the column and it's still biased"). Uses the SAME
    top-level vfairness functions the Navigator's vfairness_proxy_analysis
    handler wraps -- one source of truth:

      * identify_proxy_variables  -> univariate (Pearson / Cramer's V / MI)
      * find_proxy_chains         -> indirect A -> B -> protected
      * multivariate_proxy_leakage-> systemic: can the attribute be
                                     reconstructed from ALL other features?
    """
    from vfairness import find_proxy_chains, identify_proxy_features, multivariate_proxy_leakage

    proxies: List[Dict[str, Any]] = []
    chains: List[Dict[str, Any]] = []

    # Univariate screen: identify_proxy_features (discovery.py) -- the SAME
    # function the Navigator's vfairness_proxy_analysis handler uses, so the
    # two paths report identical proxies. Returns a list of
    # {column, correlation, abs_correlation, correlation_type, risk_level}.
    for attr in usable:
        pr = _safe(lambda a=attr: identify_proxy_features(work, a), [])
        for p in pr if isinstance(pr, list) else _as_mapping(pr).get("proxies") or []:
            pm = _as_mapping(p)
            col = pm.get("column") or pm.get("feature") or pm.get("name")
            if not col:
                continue
            proxies.append(
                {
                    "protectedAttribute": attr,
                    "feature": str(col),
                    "strength": pm.get("abs_correlation")
                    if pm.get("abs_correlation") is not None
                    else pm.get("correlation"),
                    "mutualInformation": pm.get("mutual_information"),
                    "cramersV": pm.get("cramers_v"),
                    "pValue": pm.get("pvalue"),
                    "method": _scalar(pm.get("correlation_type") or pm.get("method")),
                    "severity": _scalar(pm.get("risk_level") or pm.get("severity")) or "warn",
                }
            )
    for attr in usable:
        for ch in _safe(lambda a=attr: find_proxy_chains(work, a, max_chain_length=3), []) or []:
            cm = _as_mapping(ch)
            chains.append(
                {
                    "protectedAttribute": attr,
                    "path": cm.get("chain") or cm.get("path"),
                    "strength": cm.get("indirect_correlation") or cm.get("strength"),
                }
            )

    # Self-proxies (date_of_birth ~ age) are noise; keep genuine stand-ins.
    prot_lower = {a.lower() for a in usable}
    proxies = [p for p in proxies if str(p["feature"]).lower() not in prot_lower]
    proxies.sort(
        key=lambda d: (isinstance(d.get("strength"), (int, float)), abs(d.get("strength") or 0.0)),
        reverse=True,
    )
    # Feature->decision screen: a declared proxy that itself drives an
    # outcome gap (redlining footprint). Surfaces proxies the
    # feature->protected correlation screen is structurally blind to.
    outcome_proxies: List[Dict[str, Any]] = []
    if y_pred is not None and proxy_candidates:
        # Broaden the candidate set: the declared proxy_candidates list
        # from the column typology covers categorical stand-ins, but for
        # planted-bias mechanisms like photo-attractiveness-laundering
        # the carrier is a numeric score-like feature classified as
        # "job-relevant" by the typology. Run the disparity screen over
        # ALL numeric features too (excluding protected / declared id /
        # decision-target columns) so the carrier surfaces.
        protected_set = {a.lower() for a in usable}
        excluded_set = {(c or "").lower() for c in (exclude_cols or [])}
        broadened: List[str] = list(proxy_candidates or [])
        for c in work.columns:
            cl = c.lower()
            if cl in protected_set or cl in excluded_set:
                continue
            if c in broadened:
                continue
            ser = work[c]
            if not pd.api.types.is_numeric_dtype(ser):
                continue
            if ser.nunique(dropna=True) < 3:
                continue
            broadened.append(c)
        outcome_proxies = _safe(
            lambda: _proxy_outcome_disparity(
                work,
                y_pred,
                [c for c in broadened if c.lower() not in prot_lower],
                protected_attrs=list(usable),
            ),
            [],
        )
        seen = {(p.get("feature"), p.get("method")) for p in proxies}
        for op in outcome_proxies:
            if (op["feature"], op["method"]) not in seen:
                proxies.append(op)
        proxies.sort(
            key=lambda d: (
                isinstance(d.get("strength"), (int, float)),
                abs(d.get("strength") or 0.0),
            ),
            reverse=True,
        )

    # T2.1: STRONG PROXY CROSSWALK
    # identify_proxy_features() above can miss (feature -> protected)
    # links when the feature correlates with outcome strongly but with
    # the protected attribute only weakly -- the canonical "redlining"
    # pattern (zip_minority_majority weakly correlates with
    # race_ethnicity but strongly with hiring outcome). Without an
    # explicit cross-link the proxy stays tagged "(decision outcome)"
    # and the audit narrative cannot say "zip stands in for race".
    #
    # This pass walks every outcome-driven proxy feature and computes
    # its association with each protected attribute using the strongest
    # statistic for the (feature, protected) type pair:
    #   numeric    + categorical -> eta-squared (ANOVA)
    #   categorical + categorical -> Cramer's V
    #   numeric    + numeric      -> abs(Spearman rho)
    # Permissive thresholds (eta2/V >= 0.02 or |rho| >= 0.15) -- in
    # proxy mechanisms the feature->protected correlation is by
    # construction weaker than feature->outcome. Severity tiers by
    # strength. Universal: operates on raw column statistics, no
    # framework-specific assumptions.
    def _eta_squared(x_num: pd.Series, y_cat: pd.Series) -> float:
        """Eta-squared for a (numeric, categorical) pair.

        THREE STATES, for the same reason as ``_cramers_v`` below: eta2 is a
        ratio of sums of squares, so a feature that does not vary at all makes
        it 0/0, and a pair with too few aligned rows or a single observed
        level was never computed. Both used to answer 0.0, the crosswalk's own
        word for "measured, not a proxy".
        """
        try:
            pair = pd.concat([x_num, y_cat], axis=1).dropna()
            if len(pair) < 10:
                warnings.warn(
                    f"Eta-squared not computed for a pair with {len(pair)} "
                    f"aligned non-null rows (minimum 10). Returning nan (could "
                    f"not check), never 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")
            xv = pd.to_numeric(pair.iloc[:, 0], errors="coerce").to_numpy()
            yv = pair.iloc[:, 1].astype("string").to_numpy()
            valid = ~np.isnan(xv)
            xv, yv = xv[valid], yv[valid]
            if xv.size < 10 or len(set(yv)) < 2:
                warnings.warn(
                    f"Eta-squared not computed: {int(xv.size)} usable numeric "
                    f"rows over {len(set(yv))} observed group level(s); it needs "
                    f"10 rows and at least 2 levels. Returning nan (could not "
                    f"check), never 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")
            grand = float(xv.mean())
            ss_total = float(((xv - grand) ** 2).sum())
            if ss_total <= 1e-12:
                warnings.warn(
                    "Eta-squared is undefined: the numeric side of the pair has "
                    "no variance at all, so the statistic is 0/0. Returning nan "
                    "(could not check), never 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")
            ss_between = 0.0
            for lvl in set(yv):
                mask = yv == lvl
                if mask.sum() == 0:
                    continue
                gm = float(xv[mask].mean())
                ss_between += float(mask.sum()) * (gm - grand) ** 2
            return min(1.0, ss_between / ss_total)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Eta-squared could not be computed for this pair "
                f"({type(exc).__name__}: {exc}). Returning nan (could not "
                f"check), never 0.0.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

    def _cramers_v(x_cat: pd.Series, y_cat: pd.Series) -> float:
        """Cramer's V for a (feature, protected attribute) pair.

        THREE STATES. Every refusal below used to answer 0.0, and 0.0 on this
        scale means "measured, no association", which is the crosswalk's own
        word for "not a proxy". A pair with six aligned rows, or a pair where
        one side never varies, is not a pair without proxy risk: nobody
        measured it. nan is returned instead, the caller records the pair in
        ``crosswalkNotAssessed`` and the summary says so, because nan >= 0.015
        is False and an undisclosed nan is indistinguishable from a clean 0.0
        at the reader's end.
        """
        try:
            pair = pd.concat([x_cat, y_cat], axis=1).dropna()
            if len(pair) < 10:
                warnings.warn(
                    f"Cramer's V not computed for a pair with {len(pair)} "
                    f"aligned non-null rows (minimum 10). Returning nan (could "
                    f"not check), never 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")
            ct = pd.crosstab(pair.iloc[:, 0].astype("string"), pair.iloc[:, 1].astype("string"))
            n = float(ct.values.sum())
            if n <= 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
                warnings.warn(
                    f"Cramer's V is undefined for a {ct.shape[0]}x{ct.shape[1]} "
                    f"contingency table over {n} observations: k = min(rows, "
                    f"columns) - 1 is 0, so V = sqrt(chi2 / (n * 0)). With fewer "
                    f"than two observed levels on a side there is no association "
                    f"to measure. Returning nan (could not check), never 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")
            row_t = ct.sum(axis=1).to_numpy().astype(float)
            col_t = ct.sum(axis=0).to_numpy().astype(float)
            exp = np.outer(row_t, col_t) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = np.where(exp > 0, ((ct.values - exp) ** 2) / exp, 0.0).sum()
            denom = n * (min(ct.shape) - 1)
            if denom <= 0:
                warnings.warn(
                    f"Cramer's V is undefined: the normaliser n * (min(shape) - 1) "
                    f"is {denom}. Returning nan (could not check), never 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
                return float("nan")
            return float(np.sqrt(chi2 / denom))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Cramer's V could not be computed for this pair "
                f"({type(exc).__name__}: {exc}). Returning nan (could not "
                f"check), never 0.0, which would read as a measured absence of "
                f"association.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")

    def _is_numeric_col(s: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(s):
            return int(s.dropna().nunique()) > 2
        return False

    # Crosswalk feature universe: every non-protected column. We do NOT
    # honour ``exclude_cols`` here -- those are "shouldn't feed a model"
    # columns (PII, target, model output, unknown classification). The
    # crosswalk's WHOLE PURPOSE is to detect that supposedly-innocent
    # columns leak protected information; excluding a column flagged as
    # PII (e.g. photo_attractiveness_score, derived-from-photo) would
    # hide the exact mechanism a regulator needs to see. The only
    # exclusions are the protected attributes themselves (self-link
    # noise) and the actual target/decision column (would be a leak by
    # definition). The pred_col guard keeps the outcome itself out.
    excluded_lower = set(prot_lower)
    if pred_col:
        excluded_lower.add(str(pred_col).lower())
    if label_col:
        excluded_lower.add(str(label_col).lower())
    crosswalk_feats = sorted({str(c) for c in work.columns if str(c).lower() not in excluded_lower})
    already = {(str(p.get("feature")), str(p.get("protectedAttribute") or "")) for p in proxies}
    new_links: List[Dict[str, Any]] = []
    # THIRD STATE, carried to the surface. A pair whose association could not
    # be computed is NOT a pair below the threshold: it never reached one. It
    # lands here, is named in the summary and is returned as
    # ``crosswalkNotAssessed`` so a reader can tell it apart from a measured
    # 0.0 without reading this source.
    crosswalk_unmeasured: List[Dict[str, Any]] = []
    for feat in crosswalk_feats:
        if feat not in work.columns:
            continue
        fcol = work[feat]
        f_is_num = _is_numeric_col(fcol)
        for pa in usable:
            if pa not in work.columns or (feat, pa) in already:
                continue
            pcol = work[pa]
            p_is_num = _is_numeric_col(pcol)
            # nan, not 0.0. On this scale 0.0 is a MEASUREMENT ("no
            # association", i.e. not a proxy), so seeding the variable with it
            # meant every branch that declined to compute anything published
            # the crosswalk's own all-clear for that pair.
            strength = float("nan")
            method = ""
            reason = ""
            notes: List[str] = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if f_is_num and not p_is_num:
                        strength = _eta_squared(fcol, pcol)
                        method = "eta_squared"
                    elif not f_is_num and p_is_num:
                        strength = _eta_squared(pcol, fcol)
                        method = "eta_squared"
                    elif not f_is_num and not p_is_num:
                        strength = _cramers_v(fcol, pcol)
                        method = "cramers_v"
                    else:
                        method = "spearman"
                        pair = pd.concat(
                            [
                                pd.to_numeric(fcol, errors="coerce"),
                                pd.to_numeric(pcol, errors="coerce"),
                            ],
                            axis=1,
                        ).dropna()
                        if len(pair) < 10:
                            reason = (
                                f"only {len(pair)} aligned non-null row(s), below "
                                f"the minimum of 10 for a rank correlation"
                            )
                        else:
                            rho = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
                            if pd.notna(rho):
                                strength = float(abs(rho))
                            else:
                                reason = (
                                    "Spearman rho is undefined for this pair: at "
                                    "least one side does not vary"
                                )
                notes = list(dict.fromkeys(str(w.message) for w in caught))
            except Exception as exc:  # noqa: BLE001
                strength = float("nan")
                reason = f"{type(exc).__name__}: {exc}"
            if not is_measured(strength):
                crosswalk_unmeasured.append(
                    {
                        "protectedAttribute": pa,
                        "feature": feat,
                        "method": method or "unknown",
                        "strength": None,
                        "reason": reason
                        or "; ".join(notes)
                        or "the association could not be computed for this pair",
                    }
                )
                continue
            # Permissive threshold. eta2/Cramer's V floor of 0.015 is
            # above the noise level (random ~ 0.001 on n=2700) but loose
            # enough to catch real-but-weak proxy mechanisms like photo
            # laundering (B9 on the recruitment dataset clocked 0.017).
            # Severity tiers preserve their original cutoffs so the GUI
            # treats the floor entries as "warn", not "critical".
            if (method in ("eta_squared", "cramers_v") and strength >= 0.015) or (
                method == "spearman" and strength >= 0.12
            ):
                severity = (
                    "critical"
                    if strength >= 0.30
                    else "high"
                    if strength >= 0.15
                    else "medium"
                    if strength >= 0.07
                    else "warn"
                )
                new_links.append(
                    {
                        "protectedAttribute": pa,
                        "feature": feat,
                        "strength": float(strength),
                        "method": method,
                        "severity": severity,
                        "linkSource": "crosswalk",
                    }
                )
    if new_links:
        proxies.extend(new_links)
        proxies.sort(
            key=lambda d: (
                isinstance(d.get("strength"), (int, float)),
                abs(d.get("strength") or 0.0),
            ),
            reverse=True,
        )
    # END T2.1

    # SYNTHESISE feature -> protected -> outcome chains from the proxies
    # we just computed. find_proxy_chains() returns nothing on many real
    # datasets because it requires a specific structural signal; for the
    # mechanism narrative an auditor needs ("ZIP stands in for race,
    # which drives the decision"), the operationally useful chain is
    # the union of (feature -> protected) and (protected -> outcome)
    # signals we already have. Emit them as explicit 3-node paths so
    # the GUI can render "feature -> protected -> outcome" plainly.
    seen_chains = {tuple(c.get("path") or []) for c in chains if c.get("path")}
    for p in proxies:
        pa = str(p.get("protectedAttribute") or "")
        feat = str(p.get("feature") or "")
        if not pa or pa == "(decision outcome)" or not feat:
            continue
        p_strength = p.get("strength")
        if not isinstance(p_strength, (int, float)) or abs(p_strength) < 0.15:
            continue
        # Build "feat -> pa -> (decision outcome)" path; (decision
        # outcome) is the standard sentinel the engine uses elsewhere
        # for the verdict node so the GUI already knows how to render it.
        path = [feat, pa, "(decision outcome)"]
        if tuple(path) in seen_chains:
            continue
        seen_chains.add(tuple(path))
        chains.append(
            {
                "protectedAttribute": pa,
                "path": path,
                "strength": float(abs(p_strength)),
                "method": "synthesised_from_proxy",
                "detail": (
                    f"'{feat}' is a proxy for '{pa}' "
                    f"(strength {float(abs(p_strength)):.2f}); '{pa}' has an "
                    f"adverse-impact outcome gap. Removing '{feat}' alone "
                    f"will not de-bias the decision: the protected trait "
                    f"reaches the outcome through this chain."
                ),
            }
        )

    high = [p for p in proxies if p["severity"] in ("high", "critical", "severe")]

    # Systemic leakage: collective reconstruction of each attribute. Exclude
    # identity / target / decision / oracle columns (passed from the column
    # typology in E2) so leakage is measured on legitimate features only.
    leak_raw = _safe(
        lambda: multivariate_proxy_leakage(work, usable, exclude_columns=exclude_cols), []
    )
    leakage = [
        _as_mapping(r.to_dict() if hasattr(r, "to_dict") else r)
        for r in (leak_raw if isinstance(leak_raw, list) else [])
    ]
    systemic = [rec for rec in leakage if rec.get("systemic_leakage")]

    parts: List[str] = []
    if proxies:
        parts.append(
            f"{len(proxies)} feature(s) act as a stand-in for a protected "
            f"attribute" + (f" ({len(high)} high-risk)" if high else "")
        )
    if systemic:
        worst = max(systemic, key=lambda rec: rec.get("auc") or 0.0)
        parts.append(
            f"systemic leakage: {worst.get('protected_attribute')} is "
            f"reconstructable from the other features (AUC "
            f"{worst.get('auc'):.2f}): removing single columns will not "
            f"de-bias the model"
        )
    summary = (
        ("; ".join(parts) + ".")
        if parts
        else "No strong proxy features or systemic leakage detected for the assessed attributes."
    )
    if crosswalk_unmeasured:
        # The sentence above is a clean bill of health for everything the
        # crosswalk covered. It must not be read as covering the pairs nobody
        # could measure, so it says how many there were and where they are.
        summary += (
            f" {len(crosswalk_unmeasured)} feature / attribute pair(s) COULD NOT "
            f"BE MEASURED by the proxy crosswalk and are listed under "
            f"crosswalkNotAssessed; an unassessed pair is not a pair without "
            f"proxy risk."
        )
    # Smart cap: never drop a (feature, protected_attribute) linkage
    # because of a blanket top-N limit -- that was the bug that hid
    # B4 (zip->race), B8 (gap->gender) and B9 (photo->race) on the
    # recruitment dataset. Strategy:
    #   * keep ALL entries whose protectedAttribute is a real protected
    #     attr (these are the audit-narrative linkages we need)
    #   * cap "(decision outcome)" entries (symptom-only, no mechanism)
    #     at 10 -- they're useful but not the headline
    # The combined output is sorted by strength so the GUI's top-N
    # display still leads with the strongest findings. Universal: the
    # rule never drops a planted-mechanism linkage on any dataset.
    usable_set = {str(a) for a in usable}
    protected_links = [p for p in proxies if str(p.get("protectedAttribute") or "") in usable_set]
    outcome_only = [
        p for p in proxies if str(p.get("protectedAttribute") or "") == "(decision outcome)"
    ]
    other = [p for p in proxies if p not in protected_links and p not in outcome_only]
    capped_outcome = sorted(
        outcome_only,
        key=lambda d: (isinstance(d.get("strength"), (int, float)), abs(d.get("strength") or 0.0)),
        reverse=True,
    )[:10]
    proxies_final = sorted(
        protected_links + capped_outcome + other,
        key=lambda d: (isinstance(d.get("strength"), (int, float)), abs(d.get("strength") or 0.0)),
        reverse=True,
    )
    return {
        "available": bool(proxies_final or chains or leakage),
        "proxies": proxies_final,
        "chains": chains[:20],
        "leakage": leakage,
        # THREE STATES on the crosswalk: measured links are in "proxies", pairs
        # measured below the threshold are simply absent, and pairs that could
        # NOT be measured are here with the reason. Never fold the third into
        # the second.
        "crosswalkNotAssessed": crosswalk_unmeasured[:50],
        "crosswalkNotAssessedCount": len(crosswalk_unmeasured),
        "summary": summary,
    }


def _statistical(
    work: pd.DataFrame, usable: List[str], y_pred: np.ndarray, y_true: Optional[np.ndarray]
) -> Dict[str, Any]:
    """Significance + robustness battery. Permutation-based
    comprehensive_fairness_test per attribute and subgroup_robustness_audit
    -- the SAME top-level vfairness entry points the Navigator's
    vfairness_statistical_validation / vfairness_robustness_test handlers
    use. Tells the user whether a gap is real and whether it is fragile."""
    from vfairness import (
        comprehensive_fairness_test,
        sequential_fairness_test,
        subgroup_robustness_audit,
    )

    yt = y_true if y_true is not None else y_pred
    per_attr: List[Dict[str, Any]] = []
    for attr in usable:
        sens = work[attr].astype("string").fillna("missing").to_numpy()
        n = min(len(sens), len(yt), len(y_pred))
        res = _safe(
            lambda s=sens, n=n: comprehensive_fairness_test(
                yt[:n], y_pred[:n], s[:n], n_permutations=_BOOTSTRAP
            ),
            None,
        )

        # Anytime-valid Wald SPRT between the two largest groups -- lets the
        # audit conclude (or not) without alpha-spending on repeated looks.
        def _seq(s=sens, n=n):
            vc = pd.Series(s[:n]).value_counts()
            if len(vc) < 2:
                return None
            g1, g2 = vc.index[0], vc.index[1]
            return sequential_fairness_test(y_pred[:n][s[:n] == g1], y_pred[:n][s[:n] == g2])

        seq = _as_mapping(_safe(_seq, None))
        rm = _as_mapping(res)
        if rm:
            # comprehensive_fairness_test -> {metric_tests, p_values,
            # significant_metrics, adjusted_p_values, correction_method,
            # any_significant, n_tests}. Significance is FDR-adjusted across
            # the metric family already (one source of truth).
            adj = [p for p in (rm.get("adjusted_p_values") or []) if isinstance(p, (int, float))]
            sig_metrics = rm.get("significant_metrics") or []
            # READINESS-6, 2026-09-10. This was `bool(rm.get("any_significant"))`.
            # robustness.comprehensive_fairness_test deliberately builds
            # any_significant as THREE states and says so in its own comment:
            # False means "tests ran and none was significant", None means
            # "nothing was testable", "which is not the same claim and must not
            # be reported as one". bool(None) is False, so this line reported it
            # as one. MEASURED with a single-group sensitive attribute, where
            # all three metrics are unmeasurable: robustness returned
            # any_significant=None, n_tests=0, not_assessable=[all three], and
            # this block wrote significant=False, a correct measurement no
            # reader can see. None now survives to the consumer, alongside the
            # counts that say why.
            any_sig = rm.get("any_significant")
            not_assessable = [str(m) for m in (rm.get("not_assessable") or [])]
            per_attr.append(
                {
                    "attribute": attr,
                    "pValue": (min(adj) if adj else None),
                    "significant": (None if any_sig is None else bool(any_sig)),
                    "significantMetrics": [str(m) for m in sig_metrics],
                    "nTests": rm.get("n_tests"),
                    "notAssessableMetrics": not_assessable,
                    "nNotAssessable": len(not_assessable),
                    "assessed": any_sig is not None,
                    "notAssessedReason": (
                        None
                        if any_sig is not None
                        else (
                            "No fairness metric was computable for this attribute"
                            + (f" ({', '.join(not_assessable)})" if not_assessable else "")
                            + ", so no test ran. This is COULD NOT CHECK, not "
                            "'no significant difference'."
                        )
                    ),
                    "test": _scalar(rm.get("correction_method")) or "permutation",
                    "sequential": (seq or None),
                }
            )
    codes = {a: np.asarray(pd.Categorical(work[a]).codes, dtype=int) for a in usable}
    robustness = _as_mapping(
        _safe(lambda: subgroup_robustness_audit(y_pred, codes, y_true=yt), None)
    )
    return {
        "available": bool(per_attr or robustness),
        "perAttribute": per_attr,
        "robustness": robustness or None,
        "correction": "benjamini_hochberg_fdr",
        "nPermutations": _BOOTSTRAP,
        "inferenceMethods": [
            "permutation (FDR-corrected)",
            "Wilson per-group CI",
            "Empirical-Likelihood simultaneous bound (see disparity grid)",
            "anytime-valid Wald SPRT (per attribute)",
        ],
    }


def _pareto(
    work: pd.DataFrame,
    df: pd.DataFrame,
    primary: str,
    ref_label: Optional[str],
    worst_label: Optional[str],
    label_col: Optional[str],
    has_truth: bool,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Accuracy / fairness trade-off by sweeping the decision threshold of a
    real model SCORE. Honest by construction: only computed when a
    continuous score exists (a binary decision has no trade-off to trace),
    and the accuracy axis is shown only when ground-truth labels exist
    (otherwise the gap-vs-selection-rate trade-off is shown, never a faked
    accuracy)."""
    score_col = _pick(
        df, ("probability", "proba", "prob", "score", "y_score", "y_prob", "confidence")
    )
    if score_col is None or score_col not in work.columns:
        return {
            "available": False,
            "reason": "A Pareto trade-off needs the model's continuous score or "
            "probability, not just the final yes/no decision. None was "
            "found in this data, so there is no threshold to sweep.",
        }
    s = pd.to_numeric(work[score_col], errors="coerce")
    if s.notna().mean() < 0.5 or s.nunique() < 5:
        return {
            "available": False,
            "reason": f'"{score_col}" is not a usable continuous score (too few '
            "distinct values), so the threshold sweep cannot be drawn.",
        }
    s = s.to_numpy()
    n = min(len(s), len(y_pred))
    s, base = s[:n], y_pred[:n].astype(int)
    grp = work[primary].astype("string").fillna("missing").to_numpy()[:n]
    if not ref_label or not worst_label:
        return {
            "available": False,
            "reason": "No assessable primary variable to measure the gap against.",
        }
    a_mask = grp == ref_label
    b_mask = grp == worst_label
    if not a_mask.any() or not b_mask.any():
        return {
            "available": False,
            "reason": "Reference / worst group not present alongside the score.",
        }
    lab = None
    if has_truth and label_col and label_col in work.columns:
        lab = _coerce_binary(work[label_col])[:n]

    def _gap(pred):
        return abs(float(pred[a_mask].mean()) - float(pred[b_mask].mean()))

    thresholds = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 17)))
    pts = []
    for t in thresholds:
        pred = (s >= t).astype(int)
        acc = float((pred == lab).mean()) if lab is not None and len(lab) == n else None
        pts.append(
            {"label": f"thr {t:.2f}", "threshold": float(t), "accuracy": acc, "gap": _gap(pred)}
        )
    cur_acc = float((base == lab).mean()) if lab is not None and len(lab) == n else None
    current = {
        "label": "this system",
        "threshold": None,
        "accuracy": cur_acc,
        "gap": _gap(base),
        "current": True,
    }

    # Pareto-optimal set: no other point is better on BOTH axes (lower gap
    # and >= accuracy). Without labels, frontier is by gap alone.
    def _dominated(p, q):
        if p["accuracy"] is None or q["accuracy"] is None:
            return q["gap"] < p["gap"] - 1e-9
        return (
            q["gap"] <= p["gap"] + 1e-9
            and q["accuracy"] >= p["accuracy"] - 1e-9
            and (q["gap"] < p["gap"] - 1e-9 or q["accuracy"] > p["accuracy"] + 1e-9)
        )

    for p in pts:
        p["onFrontier"] = not any(_dominated(p, q) for q in pts if q is not p)
    cur_on = not any(_dominated(current, q) for q in pts)
    current["onFrontier"] = cur_on

    return {
        "available": True,
        "labelFree": lab is None,
        "scoreColumn": score_col,
        "attribute": primary,
        "referenceGroup": ref_label,
        "worstGroup": worst_label,
        "points": pts + [current],
        "currentOnFrontier": bool(cur_on),
    }


_EN_MARKERS = frozenset(
    (
        "the",
        "and",
        "of",
        "to",
        "is",
        "was",
        "that",
        "with",
        "for",
        "this",
        "have",
        "are",
        "not",
        "you",
        "they",
        "from",
        "were",
        "which",
        "their",
        "there",
        "would",
        "could",
        "should",
        "been",
        "will",
        "what",
        "when",
    )
)
_EN_MARKER_MIN_SHARE = 0.08


def _english_marker_share(texts: List[str], sample: int = 80) -> float:
    """Share of word tokens that are high-frequency English function words.

    The generative triage scorers are English keyword/lexicon scorers; on
    non-English text they read near zero on every dimension, which would
    render as "no disparity" (a false clean reading). English prose scores
    well above 0.15 on this marker share; German/French/Spanish text lands
    near 0.02-0.06. Returns -1.0 when there is too little text to call a
    language at all (the caller must not treat that as non-English)."""
    import re as _re

    words: List[str] = []
    for t in texts[:sample]:
        words.extend(_re.findall(r"[a-zA-Z']+", str(t).lower()))
        if len(words) >= 4000:
            break
    if len(words) < 20:
        return -1.0
    return sum(1 for w in words if w in _EN_MARKERS) / float(len(words))


def _bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (monotone), pure
    numpy, array-in/array-out. Used for the run-wide FAMILY-WISE
    correction on the generative path (the family is every attribute x
    pair x metric p-value in the run) and for the continuous/rank
    per-attribute family. Same math as the canonical library helper
    (benjamini_hochberg_correction in vfairness_metrics._statistics);
    kept here as a dependency-light scalar helper because the library
    version returns a full MultipleTestingResult envelope."""
    p = np.asarray(pvals, dtype=float)
    n = int(p.size)
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(n, dtype=float)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = p[order[i]] * n / rank
        cummin = min(cummin, val)
        adj[order[i]] = min(cummin, 1.0)
    return adj


def _generative_pulse(
    df: pd.DataFrame,
    inputs: Dict[str, Any],
    text_col: str,
    requested: List[str],
    domain: str,
    jurisdiction: str,
    progress_cb=None,
) -> Dict[str, Any]:
    """Generative (prompts+outputs) Pulse path. REUSES the existing
    vfairness.llm.OutputAnalyzer (12 metrics, NO API) per protected
    attribute, then maps into the SAME assurance-verdict shape so the
    redesigned UX renders unchanged. Never raises.

    G-06 brings this path up to the tabular battery's bar:
      - ALL group pairs per attribute (<= 6 groups; one-vs-rest above 6)
      - run-wide FAMILY-WISE BH-FDR across every (attribute, pair,
        metric) p-value; significance decisions use only these values
      - per-attribute minimum-detectable-effect power disclosure
      - a text-specific dataQuality section (empty / duplicate texts,
        per-group counts)
      - per-attribute progress via ``progress_cb`` (same signature as
        run_pulse's _p: (stage_index, label, pct)); telemetry can never
        affect the analysis or raise
      - degradations recorded exactly like the tabular battery
    """
    from vfairness.evaluation.vfairness_metrics import classify_column_roles

    degradations: List[Dict[str, Any]] = []
    schema = _safe(
        lambda: classify_column_roles(df, declared_protected=requested),
        {"pii_leakage": [], "mismatches": [], "refuse": False},
        degraded=degradations,
        label="schema_roles",
    )
    groups_cols = [c for c in requested if c in df.columns] or [
        c for c in ("group", "demographic", "persona", "segment") if c in df.columns
    ]
    findings: List[Dict[str, Any]] = []
    generative: List[Dict[str, Any]] = []

    _MIN_TEXTS = 5  # per-side floor (unchanged)
    _MAX_PAIRWISE_GROUPS = 6  # above this, one-vs-rest bounds the family

    def _emit(stage_index: int, label: str, pct: int) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage_index, label, pct)
        except Exception:  # noqa: BLE001, telemetry is never fatal
            pass

    def _analyze(attr: str) -> None:
        from itertools import combinations

        from vfairness.llm import OutputAnalyzer

        s = df[attr].astype("string").fillna("missing")
        vc = s.value_counts()
        labels = [str(lab) for lab in vc.index if int(vc[lab]) >= _MIN_TEXTS]
        if len(labels) < 2:
            return
        analyzer = OutputAnalyzer()
        comparisons: List[Dict[str, Any]] = []
        pair_iter: List[Tuple[str, Optional[str]]]
        if len(labels) <= _MAX_PAIRWISE_GROUPS:
            # ALL group pairs: the old top-2-only read was blind to any
            # disparity not involving the two most-populous groups.
            mode = "all_pairs"
            pair_iter = [(ga, gb) for ga, gb in combinations(labels, 2)]
        else:
            # High cardinality: one-vs-rest on every group with n >= 5
            # keeps the comparison family bounded (k, not k(k-1)/2).
            mode = "one_vs_rest"
            pair_iter = [(ga, None) for ga in labels]
        for ga, gb in pair_iter:
            ma = (s == ga).to_numpy()
            if gb is None:
                mb = ~ma
                gb_label = "rest"
            else:
                mb = (s == gb).to_numpy()
                gb_label = gb
            ta = df.loc[ma, text_col].astype(str).tolist()
            tb = df.loc[mb, text_col].astype(str).tolist()
            if len(ta) < _MIN_TEXTS or len(tb) < _MIN_TEXTS:
                continue
            # correction_method corrects WITHIN this one call (across the
            # 12 metrics of this single pair). The run-wide family-wise
            # BH pass below supersedes it for the significance decision;
            # the per-call correction is kept so each comparison's
            # p-values stay individually honest when read in isolation.
            res = analyzer.analyze_all(ta, tb, ga, gb_label, correction_method="benjamini_hochberg")
            metrics = [
                _as_mapping(r.to_dict() if hasattr(r, "to_dict") else r) for r in (res or [])
            ]
            comparisons.append(
                {"groupA": ga, "groupB": gb_label, "nA": len(ta), "nB": len(tb), "metrics": metrics}
            )
        if not comparisons:
            return
        # Power disclosure (G-06): minimum detectable effect at the
        # smallest analyzed group, d = 2.8 * sqrt(2/n) (the standard
        # two-sided alpha=.05 / power=.80 approximation). A clean read
        # below that d is weak evidence, never clearance.
        n_smallest = min(int(vc[lab]) for lab in labels)
        d_detectable = 2.8 * math.sqrt(2.0 / float(n_smallest))
        entry: Dict[str, Any] = {
            "attribute": attr,
            "nGroups": len(labels),
            "mode": mode,
            "comparisons": comparisons,
            "minDetectableEffect": {
                "nSmallestGroup": n_smallest,
                "d": float(d_detectable),
                "plain": (
                    f"At the smallest group size n={n_smallest} only "
                    f"effects of about d >= {d_detectable:.2f} are "
                    "detectable; a clean read here is weak evidence, "
                    "not clearance."
                ),
            },
        }
        # Legacy mirror: the first comparison is the two most-populous
        # groups (value_counts order), exactly the pair the old
        # top-2-only contract carried. Kept so an older renderer that
        # reads groupA/groupB/nA/nB/metrics still shows the headline
        # pair. Same dict objects as comparisons[0], so the family-wise
        # pass updates both views at once.
        entry.update(
            {
                "groupA": comparisons[0]["groupA"],
                "groupB": comparisons[0]["groupB"],
                "nA": comparisons[0]["nA"],
                "nB": comparisons[0]["nB"],
                "metrics": comparisons[0]["metrics"],
            }
        )
        generative.append(entry)

    # English-lexicon scorer guard: every OutputAnalyzer dimension is an
    # English keyword/lexicon scorer. On non-English text they all read
    # near zero, which would render as a clean "no disparity" -- a false
    # clean for any German/French/... artifact. Refuse to score instead
    # of scoring wrongly.
    scorer_refusal: Optional[str] = None
    marker_share = _safe(
        lambda: _english_marker_share(df[text_col].dropna().astype(str).tolist()), 1.0
    )
    if 0.0 <= marker_share < _EN_MARKER_MIN_SHARE:
        scorer_refusal = (
            "The outputs do not look like English text (English "
            f"function-word share {marker_share:.0%}). The triage scorers "
            "are English-lexicon based and would silently read 'no "
            "disparity' on non-English text, so Pulse refuses to score "
            "rather than issue a false clean reading. Re-run with English "
            "outputs, or use the full assessment once multilingual "
            "scoring lands."
        )

    if scorer_refusal is None:
        k = max(1, len(groups_cols))
        for i, a in enumerate(groups_cols):
            # Per-attribute progress mapped over stage indices 2..6
            # (run_pulse emits stage 1 before and stage 7 after this
            # call), pct 35..90.
            stage = 2 + min(4, (i * 5) // k)
            pct = 35 + int(55.0 * i / k)
            _emit(stage, f"Scoring generative outputs across groups: {a}", pct)
            _safe(lambda a=a: _analyze(a), None, degraded=degradations, label=f"generative[{a}]")

    # G-06 FAMILY-WISE correction: ONE BH-FDR family across every
    # (attribute, pair, metric) p-value in the whole run. The per-call
    # correction inside _analyze only controls the 12-metric family of a
    # single pair; with all pairs across all attributes the run-wide
    # false-discovery rate would inflate exactly like the uncorrected
    # tabular battery once did. Re-adjusting the already-within-call-
    # adjusted p-values is conservative (never anti-conservative), which
    # is the honest side to err on. Significance decisions below use
    # ONLY the family-wise values.
    #: Filled in by _familywise_fdr, read by the summary and the scope note.
    power_report: Dict[str, Any] = {
        "familySize": 0,
        "excludedZeroPower": [],
        "notDetectable": [],
        "couldNotCheckDetectability": [],
        "unconfirmedLargeEffects": [],
    }

    def _is_zero_power(m: Dict[str, Any]) -> bool:
        """True when this metric row records a NON-test, not a negative result.

        vfairness.llm.output_analysis short-circuits to ``p_value = 1.0`` when
        the two score arrays are identical (``np.array_equal``), which is the
        right call: there is nothing to test. But 1.0 is also the largest p a
        test can return, so such a row joins the Benjamini-Hochberg family and
        raises the bar for every member that DID measure something, while being
        incapable of ever being a discovery itself.

        AND THE SIGNATURE ALONE CLOSED ON A REAL NEGATIVE (audit wave 4,
        2026-09-30). A Mann-Whitney test returns p exactly 1.0 whenever the two
        group means are equal, reporting delta 0.0 and effect 0.0 on the same row,
        so the four numbers below are ALSO the signature of a genuine tested
        negative. For refusal_rate, toxicity, stereotype and representation, which
        are discrete, equal counts between two groups is common and is the
        outcome a fair model is supposed to produce. Measured on four texts per
        group with the refusals interleaved (scores A [1,0,1,0] against B
        [0,1,0,1], so np.array_equal is False and a real test ran): this returned
        True, the row was excluded from the family, and the payload told the
        reader "This scorer returned the same value for every text in BOTH
        groups", which was false of that data.

        So the analyzer now RECORDS which door produced the p-value and this
        reads that fact, exactly as its sibling
        ``llm.output_analysis._is_zero_power`` does. The signature is kept only
        for a row carrying no provenance (an older stored payload), where it is
        the answer this function has always given.

        The signature is all four together: p exactly 1.0, delta exactly 0,
        effect exactly 0, and the two group values equal.
        """
        # The VALUE, not the key: a row can carry the key holding None.
        source = ((m.get("metadata") or {}).get("parameters") or {}).get("p_value_source")
        if source == "identical_samples_short_circuit":
            return True
        if source is not None:
            return False
        p = m.get("p_value")
        if not (isinstance(p, (int, float)) and float(p) == 1.0):
            return False
        delta, eff = m.get("delta"), m.get("effect_size")
        a, b = m.get("group_a_value"), m.get("group_b_value")
        return (
            isinstance(delta, (int, float))
            and float(delta) == 0.0
            and isinstance(eff, (int, float))
            and float(eff) == 0.0
            and isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and float(a) == float(b)
        )

    def _declares_zero_power(m: Dict[str, Any]) -> bool:
        """True when the ROW ITSELF says it performed no comparison.

        BGL5, 2026-09-28. ``_is_zero_power`` above requires ``p`` to be EXACTLY
        1.0, which was the signature ``analyze_all`` handed over while it left the
        short circuit's own p-value in place. It does not any more: it clears the
        p-value and names the state in ``not_assessed_reason``, so that signature
        stopped matching and this detector went quiet for its main producer.

        The rows still LEFT the family, because a None p-value cannot join it, so
        no number moved and nothing failed. They left through the ``unscored`` arm
        instead, and ``unscored`` carries no reason, so ``excludedZeroPower`` came
        back EMPTY and the payload no longer said why the family had shrunk.
        Measured on 5 refusals against 5 answers: familySize 5, with toxicity,
        stereotype and representation each carrying
        ``not_assessed_reason='identical_scores_nothing_to_test'``, and
        ``excludedZeroPower`` ``[]``. A correct exclusion that no reader can see.

        Checked ABOVE the p-value dispatch, so it no longer matters which layer
        noticed first.
        """
        if m.get("zero_power") is True:
            return True
        return m.get("not_assessed_reason") == "identical_scores_nothing_to_test"

    def _familywise_fdr() -> None:
        # The library's SHARED discrete-floor helpers (added to
        # evaluation/vfairness_metrics/_statistics.py for exactly this): a
        # local copy would drift from the nine other detectors that read them.
        from vfairness.evaluation.vfairness_metrics._statistics import (
            detectability,
            min_attainable_p_mannwhitney,
        )

        # The PRODUCER's sentence for this state, imported rather than copied.
        from vfairness.llm.output_analysis import ZERO_POWER_REASON

        refs: List[Dict[str, Any]] = []
        unscored: List[Dict[str, Any]] = []
        zero_power: List[Dict[str, Any]] = []
        sizes: Dict[int, Tuple[Any, Any]] = {}
        for entry in generative:
            for comp in entry.get("comparisons") or []:
                for m in comp.get("metrics") or []:
                    sizes[id(m)] = (comp.get("nA"), comp.get("nB"))
                    p = m.get("p_value")
                    if _declares_zero_power(m):
                        zero_power.append(m)
                    elif not (isinstance(p, (int, float)) and math.isfinite(float(p))):
                        unscored.append(m)
                    elif _is_zero_power(m):
                        zero_power.append(m)
                    else:
                        refs.append(m)
        # A metric with no finite p-value cannot join the family, so it cannot be
        # a discovery either. Say so explicitly (None, not a missing key) and clear
        # the flag: significance must rest ONLY on the family-wise pass, and leaving
        # the per-call flag standing on an unmeasurable metric let it reach
        # _build_findings as an UNCORRECTED "significant" result.
        for m in unscored:
            m["p_value_family_adjusted"] = None
            m["is_significant"] = False
        # READINESS-6, 2026-09-10. Same rule, one step further in. A row whose
        # scorer returned the SAME value for every text on both sides is a
        # non-test wearing a p-value, and it was inflating the family exactly
        # like a NaN p would have. MEASURED on 5 refusals against 5 answers
        # (100% vs 0%), where the keyword toxicity, stereotype, regard and
        # representation scorers all read a flat 0.0 on both sides: the family
        # was m=10 with four such members, the BH rank-1 bar sat at 0.005, and
        # the refusal_rate row (group_a_value 1.0, group_b_value 0.0,
        # effect_size 3.14, "Cohen's h: large") came out adjusted 0.10519,
        # is_significant False, and the run reported "No significant
        # generative-output disparity detected" with zero findings.
        for m in zero_power:
            m["p_value_family_adjusted"] = None
            m["is_significant"] = False
            m["zero_power"] = True
            # RELAYED, NOT RESTATED (audit wave 4, 2026-09-30). This was a verbatim
            # copy of the analyzer's ZERO_POWER_REASON, and its first sentence was
            # false: the state is reached when the two groups' score ARRAYS are
            # equal element for element, which a scorer that read 1.0, 0.0, 1.0,
            # 0.0 in each group satisfies without having read one value. The
            # analyzer's wording was corrected there; importing it is what keeps
            # this reader-visible field from drifting from its producer again.
            m["zero_power_reason"] = ZERO_POWER_REASON
        power_report["excludedZeroPower"] = [str(m.get("metric") or "") for m in zero_power]
        if not refs:
            return
        adj = _bh_adjust([float(m["p_value"]) for m in refs])
        m_family = len(refs)
        power_report["familySize"] = m_family
        for m, ap in zip(refs, adj):
            m["p_value_family_adjusted"] = float(ap)
            # Inclusive boundary, matching the tabular _fdr_correct.
            m["is_significant"] = bool(float(ap) <= 0.05)
            # THREE STATES on the design, next to the result. "not significant"
            # and "this design could never have been significant" are different
            # readings and must not share a rendering.
            n_a, n_b = sizes.get(id(m), (None, None))
            floor = (
                min_attainable_p_mannwhitney(int(n_a), int(n_b))
                if isinstance(n_a, int) and isinstance(n_b, int)
                else None
            )
            can_fire, note = detectability(floor, m_family)
            m["min_attainable_p_value"] = floor
            m["detectable"] = can_fire
            m["detectability_note"] = note
            name = str(m.get("metric") or "")
            if can_fire is False:
                power_report["notDetectable"].append(name)
            elif can_fire is None:
                power_report["couldNotCheckDetectability"].append(name)
            eff = m.get("effect_size")
            if (
                not m["is_significant"]
                and isinstance(eff, (int, float))
                and math.isfinite(float(eff))
                and abs(float(eff)) >= 0.8
            ):
                # The symptom that made this defect visible: a LARGE effect the
                # family-wise pass could not confirm. Recorded by name so the
                # summary cannot state a clean read over the top of it.
                power_report["unconfirmedLargeEffects"].append(
                    {
                        "metric": name,
                        "effectSize": float(eff),
                        "effectInterpretation": str(m.get("effect_size_interpretation") or ""),
                        "groupAValue": m.get("group_a_value"),
                        "groupBValue": m.get("group_b_value"),
                        "pValue": float(m["p_value"]),
                        "pValueFamilyAdjusted": float(ap),
                    }
                )

    _safe(_familywise_fdr, None, degraded=degradations, label="generative_family_fdr")

    # Findings are built AFTER the family-wise pass so severity never
    # rests on a pre-correction significance flag.
    def _build_findings() -> None:
        for entry in generative:
            attr = entry.get("attribute")
            for comp in entry.get("comparisons") or []:
                ga, gb = comp.get("groupA"), comp.get("groupB")
                for rm in comp.get("metrics") or []:
                    if not rm.get("is_significant"):
                        continue
                    eff = abs(float(rm.get("effect_size") or 0.0))
                    sev = "critical" if eff >= 0.8 else "high" if eff >= 0.5 else "warn"
                    plain = (
                        f"LLM outputs differ on '{rm.get('metric')}' "
                        f"between {ga} and {gb} "
                        f"(delta {float(rm.get('delta') or 0):.2f}, "
                        f"{rm.get('effect_size_interpretation') or 'effect'}; "
                        "p<0.05 after family-wise BH-FDR across all "
                        "pairs and metrics)."
                    )
                    # G-23: a refusal-rate gap is a quality-of-service
                    # harm, not a content difference; say so plainly.
                    is_qos = str(rm.get("metric")) == "refusal_rate"
                    if is_qos:
                        plain += (
                            " A model refusing or deflecting for one "
                            "group while answering for another is a "
                            "quality-of-service harm invisible to "
                            "content metrics: the affected group simply "
                            "is not served."
                        )
                    finding: Dict[str, Any] = {
                        "type": "generative_output_disparity",
                        "severity": sev,
                        "attribute": attr,
                        "plain": plain,
                        "statisticalTest": None,
                        "groupDistributions": {},
                        "representationRatios": {},
                        "underrepresentedGroups": [],
                        "overrepresentedGroups": [],
                    }
                    if is_qos:
                        finding["qualityOfService"] = True
                    findings.append(finding)

    _safe(_build_findings, None, degraded=degradations, label="generative_findings")

    # G-06 text data-quality: rows, empty/whitespace-only outputs,
    # duplicate outputs, per-group text counts. The tabular
    # build_quality_report reads feature columns; free text needs its
    # own read. Tone warns when > 20% of outputs are empty or any group
    # sits below the 5-text comparability floor.
    def _text_quality() -> Dict[str, Any]:
        texts = df[text_col].astype("string")
        stripped = texts.fillna("").str.strip()
        rows = int(len(df))
        empty = int((stripped == "").sum())
        non_empty = stripped[stripped != ""]
        duplicates = int(non_empty.duplicated().sum())
        per_group: Dict[str, Dict[str, int]] = {}
        small_groups: List[str] = []
        for a in groups_cols:
            gvc = df[a].astype("string").fillna("missing").value_counts()
            per_group[a] = {str(gk): int(gv) for gk, gv in gvc.items()}
            small_groups.extend(f"{a}={gk}" for gk, gv in gvc.items() if int(gv) < _MIN_TEXTS)
        empty_share = (empty / rows) if rows else 1.0
        tone_q = "warn" if (empty_share > 0.20 or small_groups) else "pass"
        if empty_share > 0.20:
            headline = (
                f"{empty} of {rows} outputs ({empty_share:.0%}) "
                "are empty or whitespace-only; the scored sample "
                "is smaller than the file suggests."
            )
        elif small_groups:
            headline = (
                f"Some groups have fewer than {_MIN_TEXTS} texts "
                "and could not be compared: " + ", ".join(small_groups[:6]) + "."
            )
        else:
            headline = f"{rows} outputs; {empty} empty, {duplicates} duplicate texts."
        return {
            "tone": tone_q,
            "headline": headline,
            "rows": rows,
            "columns": int(df.shape[1]),
            "textColumn": text_col,
            "emptyTexts": empty,
            "duplicateTexts": duplicates,
            "perGroupCounts": per_group,
            "checks": [
                {
                    "label": "Empty / whitespace-only outputs",
                    "value": empty,
                    "tone": "warn" if empty_share > 0.20 else "pass",
                },
                {"label": "Duplicate output texts", "value": duplicates, "tone": "pass"},
                {
                    "label": f"Groups below the {_MIN_TEXTS}-text floor",
                    "value": len(small_groups),
                    "tone": "warn" if small_groups else "pass",
                },
            ],
        }

    data_quality = _safe(
        _text_quality,
        {
            "tone": "warn",
            "headline": "Text data quality could not be confirmed.",
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "checks": [],
        },
        degraded=degradations,
        label="text_data_quality",
    )

    # G-23: surface the analyzer's coverage explicitly. dimensionsCovered
    # lists the metric names that ACTUALLY scored in this run (the 11
    # standard OutputAnalyzer dimensions, plus llm_judge when a judge
    # endpoint is configured); no dimension is claimed that did not run.
    def _dimensions_covered() -> List[str]:
        seen: List[str] = []
        for entry in generative:
            for comp in entry.get("comparisons") or []:
                for m in comp.get("metrics") or []:
                    name = str(m.get("metric") or "")
                    if name and name not in seen:
                        seen.append(name)
        return seen

    dimensions_covered = _safe(
        _dimensions_covered, [], degraded=degradations, label="dimensions_covered"
    )

    # G-23: explicit quality-of-service block. Refusal detection uses the
    # SAME library refusal scorer the refusal_rate metric uses (pattern-
    # based; a text counts as a refusal at score >= 0.5, i.e. hard, soft,
    # policy and partial refusals). The toxicity caveat is a standing
    # honesty note on the keyword-tier toxicity dimension.
    _TOXICITY_CAVEAT = (
        "keyword-tier toxicity classifiers over-flag identity terms; "
        "treat flags as leads, not verdicts"
    )

    def _quality_of_service() -> Dict[str, Any]:
        from vfairness.llm.scorers import DEFAULT_REFUSAL_SCORER

        by_group: Dict[str, Dict[str, float]] = {}
        for a in groups_cols:
            s = df[a].astype("string").fillna("missing")
            rates: Dict[str, float] = {}
            for lab in s.unique():
                mask = (s == lab).to_numpy()
                texts = [t for t in df.loc[mask, text_col].astype(str).tolist() if t and t.strip()]
                if not texts:
                    continue
                scores = DEFAULT_REFUSAL_SCORER.score_batch(texts)
                rates[str(lab)] = float(
                    np.mean([1.0 if float(sc) >= 0.5 else 0.0 for sc in scores])
                )
            if rates:
                by_group[a] = rates
        return {
            "refusalRateByGroup": by_group,
            "refusalDetection": (
                "pattern-based refusal scorer over the supplied outputs; "
                "a text counts as a refusal at score >= 0.5 (hard, soft, "
                "policy and partial refusal categories)"
            ),
            "toxicityCaveat": _TOXICITY_CAVEAT,
        }

    if scorer_refusal is None:
        quality_of_service = _safe(
            _quality_of_service,
            {"refusalRateByGroup": {}, "toxicityCaveat": _TOXICITY_CAVEAT},
            degraded=degradations,
            label="quality_of_service",
        )
    else:
        # Non-English text: the pattern scorer is English too, so refusal
        # rates would be as false-clean as the content metrics. Refuse.
        quality_of_service = {
            "refusalRateByGroup": {},
            "toxicityCaveat": _TOXICITY_CAVEAT,
            "reason": scorer_refusal,
        }

    from vfairness.operations.reporting import build_assurance_verdict

    assurance = _safe(
        lambda: build_assurance_verdict(
            schema=schema,
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
            "oneLineVerdict": "Generative assurance verdict could not be assembled.",
            "findings": [],
            "recommendations": [],
            "metricsDeferred": {},
            "auditTrail": {},
        },
    )
    if scorer_refusal is not None:
        assurance = {
            "overall": "Disclaimer",
            "blocksDeployment": False,
            "oneLineVerdict": scorer_refusal,
            "findings": [],
            "recommendations": [],
            "metricsDeferred": {},
            "auditTrail": {"engine": "vfairness"},
        }
    # Canonical tone vocabulary: pass | warn | critical | disclaimer.
    # "Disclaimer" (nothing was assessable) must NEVER fall through to a
    # pass tone -- a run that analyzed zero group pairs is not clean.
    tone = (
        "critical"
        if assurance.get("overall") == "Adverse"
        else "warn"
        if assurance.get("overall") == "Qualified"
        else "pass"
        if assurance.get("overall") == "Unqualified"
        else "disclaimer"
    )
    # A swallowed failure means the generative analysis is INCOMPLETE,
    # same honesty rule as the tabular battery: a run with collapsed
    # stages must never present as a clean pass.
    if degradations:
        try:
            if tone == "pass":
                tone = "warn"
            if isinstance(assurance, dict) and assurance.get("overall") in ("", "Unqualified"):
                assurance["overall"] = "Disclaimer"
                assurance["oneLineVerdict"] = (
                    "Incomplete analysis: "
                    f"{len(degradations)} stage(s) could not be computed, "
                    "so a clean opinion cannot be issued. See degradations."
                )
        except Exception:  # noqa: BLE001, post-processing is never fatal
            pass

    # Scope power note from the RUN-WIDE smallest analyzed group (G-06).
    n_min = min(
        (
            int(e["minDetectableEffect"]["nSmallestGroup"])
            for e in generative
            if isinstance(e.get("minDetectableEffect"), dict)
        ),
        default=None,
    )
    if n_min:
        d_min = 2.8 * math.sqrt(2.0 / float(n_min))
        power_note = (
            f"Run-wide smallest analyzed group is n={n_min}: only effects "
            f"of about d >= {d_min:.2f} are detectable at that size, so a "
            "clean read there is weak evidence, not clearance."
        )
    else:
        power_note = (
            "At small per-group sample sizes only large "
            "disparities are detectable; a clean read at "
            "n close to the 5-text floor is weak evidence."
        )
    # READINESS-6, 2026-09-10. The d-based sentence above is a two-sample power
    # approximation that ignores the multiple-comparison correction entirely,
    # and on the measured 100%-refusal-vs-0% case it read "only effects of about
    # d >= 1.77 are detectable" while the effect actually present was a
    # d-equivalent of 3.14 and STILL could not be reported. A power disclosure
    # that understates the bar is worse than none, so the family-wise bar and
    # the discrete floors that were computed against it are appended here.
    if power_report.get("familySize"):
        power_note += (
            " Significance is decided AFTER family-wise BH-FDR across "
            f"{power_report['familySize']} measured metric(s) in this run, so a "
            f"single metric must reach p <= {0.05 / float(power_report['familySize']):.4g} "
            "on its own evidence; the d above does not account for that."
        )
    if power_report.get("notDetectable"):
        power_note += (
            " NOT DETECTABLE at this sample size, whatever the data: "
            + ", ".join(sorted(set(power_report["notDetectable"])))
            + "."
        )
    if power_report.get("excludedZeroPower"):
        power_note += (
            " Excluded from the family as non-tests (the scorer returned one "
            "constant value across both groups): "
            + ", ".join(sorted(set(power_report["excludedZeroPower"])))
            + "."
        )

    # A finding-free run is only a clean read when the tests that ran COULD
    # have found something. Where a large effect was measured and the
    # family-wise pass could not confirm it, or where a member's design could
    # never clear the bar, that is could-not-check and must not present as a
    # pass. Same escalation shape as the degradations block above.
    unconfirmed = list(power_report.get("unconfirmedLargeEffects") or [])
    # An EMPTY family is the loudest could-not-check of all: comparisons were
    # set up and not one of them produced a test. Found while building the
    # control for this fix, on a frame whose scorers read a constant for every
    # text: familySize 0, findings 0, and the old sentence would have said "No
    # significant generative-output disparity detected".
    nothing_tested = bool(generative) and not power_report.get("familySize")
    power_limited = bool(unconfirmed or power_report.get("notDetectable") or nothing_tested)
    if not findings and power_limited and scorer_refusal is None:
        try:
            if tone == "pass":
                tone = "warn"
            reason = (
                "{0} metric(s) showed a large effect the family-wise test could "
                "not confirm ({1})".format(
                    len(unconfirmed),
                    ", ".join(sorted({str(u["metric"]) for u in unconfirmed})),
                )
                if unconfirmed
                else "{0} metric(s) could not have reached significance at this "
                "sample size under any data".format(len(power_report["notDetectable"]))
                if power_report.get("notDetectable")
                else "not one metric produced a test at all"
            )
            if isinstance(assurance, dict):
                if assurance.get("overall") in ("", "Unqualified"):
                    assurance["overall"] = "Disclaimer"
                assurance["powerLimited"] = True
                assurance["oneLineVerdict"] = (
                    str(assurance.get("oneLineVerdict") or "").rstrip()
                    + " No generative-output disparity was CONFIRMED, and this run "
                    "could not have confirmed one: " + reason + "."
                ).strip()
        except Exception:  # noqa: BLE001, post-processing is never fatal
            pass

    if findings:
        generative_summary: Optional[str] = None  # built inline below
    elif power_limited:
        parts = [
            "No generative-output disparity reached family-wise significance, "
            "and this is NOT a clean read."
        ]
        if unconfirmed:
            parts.append(
                " {0} metric(s) measured a large effect that the corrected test "
                "could not confirm: {1}.".format(
                    len(unconfirmed),
                    "; ".join(
                        "{0} ({1} vs {2}, {3}, adjusted p={4:.3g})".format(
                            u["metric"],
                            u["groupAValue"],
                            u["groupBValue"],
                            u["effectInterpretation"] or "large effect",
                            u["pValueFamilyAdjusted"],
                        )
                        for u in unconfirmed[:4]
                    ),
                )
            )
        if power_report.get("notDetectable"):
            parts.append(
                " {0} metric(s) could not have been reported at this sample size "
                "under ANY data: {1}.".format(
                    len(power_report["notDetectable"]),
                    ", ".join(sorted(set(power_report["notDetectable"]))),
                )
            )
        if nothing_tested:
            parts.append(
                " NOT ONE metric produced a test in this run: every scorer either "
                "returned no p-value or read a single constant value across both "
                "groups, so nothing was compared."
            )
        parts.append(" Raise the number of texts per group before reading this as fair service.")
        generative_summary = "".join(parts)
    else:
        generative_summary = "No significant generative-output disparity detected."

    n_comparisons = sum(len(e.get("comparisons") or []) for e in generative)
    return {
        "success": True,
        "data": _jsonify(
            {
                "sourceKind": "prompts_outputs",
                "schema": schema,
                "degradations": degradations,
                "dataQuality": data_quality,
                "assurance": assurance,
                "verdict": {
                    "tone": tone,
                    "headline": assurance.get("oneLineVerdict", ""),
                    "summary": "Generative-output fairness read "
                    "(vfairness OutputAnalyzer, 12 metrics per pair, "
                    "family-wise BH-FDR corrected).",
                    "ranked": [],
                },
                "generative": {
                    "available": bool(generative) and scorer_refusal is None,
                    "textColumn": text_col,
                    "perAttribute": generative,
                    # G-23: explicit coverage + quality-of-service.
                    "dimensionsCovered": dimensions_covered,
                    "qualityOfService": quality_of_service,
                    # READINESS-6: what the family could and could not have
                    # found, next to what it did find. Never a missing key.
                    "familyPower": power_report,
                    **({"reason": scorer_refusal} if scorer_refusal else {}),
                    "summary": (
                        scorer_refusal
                        or (
                            f"{len(findings)} significant output "
                            f"disparity finding(s) across "
                            f"{n_comparisons} comparison(s) in "
                            f"{len(generative)} attribute(s)."
                            if findings
                            else generative_summary
                        )
                    ),
                },
                # Universal: every modality emits the legal/disparity framework
                # so the GUI can name the basis on which severity was assigned.
                "legalFramework": build_legal_framework_block(domain, jurisdiction),
                "scope": build_scope_block(
                    "prompts_outputs",
                    covered=[
                        "Deterministic English-lexicon screening of the supplied "
                        "outputs across ALL group pairs per attribute up to 6 "
                        "groups (one-vs-rest on every group with n >= 5 above "
                        "that), 12 dimensions per comparison, with family-wise "
                        "BH-FDR across every attribute, pair and metric in the "
                        "run",
                        "Per-attribute minimum-detectable-effect power disclosure",
                        "Non-English text is refused, never silently scored",
                        "Explicit quality-of-service read: per-group refusal "
                        "rates over the supplied outputs, with the toxicity and "
                        "regard dimensions surfaced by name (keyword-tier "
                        "toxicity flags are leads, not verdicts)",
                    ],
                    not_covered=[
                        "Intersectional group combinations",
                        "Transformer-grade toxicity / regard scoring and human evaluation",
                        "Counterfactual augmentation and live endpoint probing "
                        "(use the model-endpoint pathway)",
                    ],
                    power_note=power_note,
                ),
                "perVariable": [],
                "metrics": [],
                "bias": findings,
                "proxies": {
                    "available": False,
                    "notApplicable": True,
                    "reason": "Proxy reconstruction applies to tabular features, not free text.",
                    "proxies": [],
                    "chains": [],
                },
                "statistical": {
                    "available": False,
                    "notApplicable": True,
                    "reason": "The tabular robustness battery does not "
                    "apply; the text metrics carry their own FDR "
                    "correction.",
                    "perAttribute": [],
                },
                "intersectional": {
                    "skipped": True,
                    "reason": "Generative path: per-token "
                    "intersectional analysis not run in triage.",
                },
                "causal": {
                    "nodes": [],
                    "edges": [],
                    "overview": "Not applicable to free-text outputs.",
                },
                "recommendedDefinition": {},
                "interventions": [],
                "nextStep": (
                    "Convert to a full assessment to run the deep LLM "
                    "battery (counterfactual, DecodingTrust)."
                ),
            }
        ),
    }


def build_legal_framework_block(domain: str, jurisdiction: str) -> Dict[str, Any]:
    """Canonical camelCase legalFramework block, ONE shape for every
    pathway (tabular + all probes). The GUI names the legal basis that
    severity rests on; three divergent shapes (camelCase, hardcoded,
    raw snake_case) was a renderer hazard."""
    fw = _legal_screen_framework(domain, jurisdiction)
    return {
        "framework": fw.get("framework"),
        "ratioWarn": fw.get("ratio_warn"),
        "ratioCritical": fw.get("ratio_critical"),
        "mode": fw.get("mode"),
        "domain": domain,
        "jurisdiction": jurisdiction,
    }


_SCOPE_COMMON = [
    "Pulse is a rapid triage reading: not a full assessment, not an NYC "
    "LL144 independent bias audit, not an EU AI Act conformity assessment, "
    "and not a FRIA. It produces an auditor-ready evidence package.",
    "Severity tones are statistical / heuristic bands (including the "
    "four-fifths rule as the UGESP evidentiary standard), never legal "
    "thresholds.",
    "Anything listed as not covered is UNTESTED, not clean; the absence "
    "of a finding is not clearance.",
]


def build_scope_block(
    pathway: str, covered: List[str], not_covered: List[str], power_note: str = ""
) -> Dict[str, Any]:
    """Engine-emitted scope-and-limitations block. The UI and the report
    render it verbatim: the engine owns the honesty text, so no surface
    can imply more coverage than was tested."""
    return {
        "level": "triage",
        "pathway": pathway,
        "covered": list(covered),
        "notCovered": list(not_covered),
        "powerNote": power_note,
        "disclosures": list(_SCOPE_COMMON),
    }


def _rows_missing_a_protected_value(
    raw: pd.DataFrame, prepared: pd.DataFrame, axes: List[str]
) -> pd.Series:
    """Per row: is ANY of these protected axes unrecorded? Indexed like ``prepared``.

    THE ONE PLACE this package decides what "no protected value was recorded"
    means, so the twenty-odd ``.astype("string").fillna("missing")`` call sites
    below cannot each answer it differently. The rule is the library's own
    ``pd.isna``, the one ``_validation.handle_missing_values`` uses for
    ``missing_strategy='exclude'``; it is not re-derived here.

    Three shapes of absence, and the third is why this reads the RAW frame:

    * ``pd.isna`` on the prepared column. Covers ``None``, ``float('nan')``,
      ``pd.NA`` and ``pd.NaT`` wherever the preparation left them in place.
    * A whitespace-only string, NORMALISED INTO the same absence rather than
      tested as a separate rule. ``pd.isna("")`` is False, so a blank protected
      cell published 'material disparity for ""' as the headline verdict.
    * ``pd.isna`` on the SAME ROW of the caller's own column. The preparation
      mints a level: ``protected_binning`` bands a date of birth into age bands
      through ``.fillna("missing")``, so a ``pd.NaT`` arrives here as the
      readable string "missing" and no test of the prepared column can see it.
      Consulted only when the axis kept its name (a binned-sibling substitution
      renames it), and it is an addition to the prepared test, never a
      replacement: an axis derived from several columns is still caught by the
      first two rules.

    THE LITERAL STRING 'None' IS NOT ABSENCE HERE, deliberately. It is
    indistinguishable from a category somebody chose, and this library already
    made that call in the other direction on purpose: its ``__missing__``
    sentinel is spelled that way precisely so it can never swallow a genuine
    level spelled "missing" (``tests/test_readiness_classification.py::
    TestMissingAttributeIsNotAGroup::
    test_a_real_group_named_missing_is_still_a_real_group``). A protected value
    reaching Pulse as ``str(None)`` is a defect in whatever wrote the file, and
    guessing at it here would start deleting real groups.

    Returns an all-False Series when there is nothing to check, so a complete
    frame is untouched and a clean run reads exactly as it did before.
    """
    absent = pd.Series(False, index=prepared.index)
    if not axes:
        return absent
    for axis in axes:
        try:
            col = prepared[axis]
        except KeyError:
            continue
        here = col.isna()
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            blank = col.map(lambda v: isinstance(v, str) and not v.strip())
            here = here | blank.fillna(False).astype(bool)
        if axis in raw.columns:
            try:
                raw_absent = raw[axis].isna().reindex(prepared.index, fill_value=False)
                here = here | raw_absent.astype(bool)
            except Exception:  # noqa: BLE001
                # A duplicate column label or a non-alignable index. The prepared
                # test above still stands; this half is an addition to it.
                pass
        absent = absent | here.astype(bool)
    return absent


def _na(reason: str) -> Dict[str, Any]:
    """Not-applicable section stub: distinguishes 'does not apply to this
    pathway' from 'tried and collapsed' (which records a degradation)."""
    return {"available": False, "notApplicable": True, "reason": reason}


def _kind_refusal(
    source_kind: str,
    reason: str,
    next_step: str,
    domain: str,
    jurisdiction: str,
    rows: int = 0,
    columns: int = 0,
) -> Dict[str, Any]:
    """Honest refusal when a DECLARED source_kind's requirements are
    missing. Falling through silently analyzed the wrong thing (or died on
    'No prediction column'), misreading the user's explicit declaration; a
    declared kind either runs or refuses by name."""
    return {
        "success": True,
        "data": _jsonify(
            {
                "sourceKind": source_kind,
                "schema": {},
                "degradations": [],
                "assurance": {
                    "overall": "Disclaimer",
                    "blocksDeployment": False,
                    "oneLineVerdict": reason,
                    "findings": [],
                    "recommendations": [],
                    "metricsDeferred": {},
                    "auditTrail": {"engine": "vfairness"},
                },
                "verdict": {
                    "tone": "disclaimer",
                    "headline": "Could not assess as declared.",
                    "summary": reason,
                    "ranked": [],
                },
                "scope": build_scope_block(source_kind, [], ["Nothing was assessed on this run."]),
                "legalFramework": build_legal_framework_block(domain, jurisdiction),
                "perVariable": [],
                "metrics": [],
                "bias": [],
                "dataQuality": {
                    "tone": "warn",
                    "headline": reason,
                    "rows": int(rows),
                    "columns": int(columns),
                    "checks": [],
                },
                "proxies": _na("Nothing was assessed on this run."),
                "statistical": _na("Nothing was assessed on this run."),
                "intersectional": {"skipped": True, "reason": reason},
                "causal": {
                    "nodes": [],
                    "edges": [],
                    "overview": "Nothing was assessed on this run.",
                },
                "recommendedDefinition": {},
                "interventions": [],
                "nextStep": next_step,
            }
        ),
    }


def _detect_output_type(df: pd.DataFrame, pred_col: str, inputs: Dict[str, Any]) -> str:
    """Classify the model-output column: 'binary' | 'continuous' | 'rank'.

    Order of precedence (G-11):
      1. explicit caller override ``inputs.output_type`` / ``outputType``
      2. binary: numeric values within {0, 1} (the existing contract)
      3. rank: a column NAMED rank/position/ranking with integer values,
         or a dense integer permutation 0..n-1 / 1..n (one value per row)
      4. continuous: numeric with > 10 distinct values
      5. everything else stays 'binary' (categorical yes/no coercion),
         keeping the historical binary path byte-identical.
    """
    override = str(inputs.get("output_type") or inputs.get("outputType") or "").strip().lower()
    if override in ("binary", "continuous", "rank"):
        return override
    s = pd.to_numeric(df[pred_col], errors="coerce")
    vals = s.dropna()
    if len(vals) == 0 or s.notna().mean() < 0.5:
        return "binary"  # string yes/no style column -> _coerce_binary
    u = pd.unique(vals)
    if len(u) and set(u) <= {0, 1}:
        return "binary"
    is_int = bool(np.all(np.mod(vals.to_numpy(dtype=float), 1) == 0))
    name = str(pred_col).lower()
    if is_int and any(k in name for k in ("rank", "position", "ranking")):
        return "rank"
    if is_int and len(u) > 2:
        lo, hi = int(vals.min()), int(vals.max())
        dense = len(u) == hi - lo + 1
        # Permutation-like ranking: dense integers starting at 0/1 with
        # exactly one value per row. Ranking tables with ties/repeats
        # (multiple queries) must use the name-based route above or the
        # explicit output_type override.
        if dense and lo in (0, 1) and (hi - lo + 1) == len(vals):
            return "rank"
    if vals.nunique() > 10:
        return "continuous"
    return "binary"


def _nonbinary_tabular_pulse(
    df: pd.DataFrame,
    work: pd.DataFrame,
    usable: List[str],
    inputs: Dict[str, Any],
    schema: Dict[str, Any],
    data_quality: Dict[str, Any],
    data_preparation: Dict[str, Any],
    degradations: List[Dict[str, Any]],
    framework: Dict[str, Any],
    domain: str,
    jurisdiction: str,
    min_group: int,
    pred_col: str,
    label_col: Optional[str],
    has_truth: bool,
    output_type: str,
    progress_cb=None,
) -> Dict[str, Any]:
    """Tabular Pulse for CONTINUOUS (regression-style) and RANK model
    outputs (G-11). The binary battery silently median-thresholded such
    columns into a fake yes/no decision; this path instead runs the
    library's real batteries per protected attribute:

      continuous -> regression parity (vfairness_metrics.regression):
        group mean-outcome gaps with bootstrap CIs on the min-max
        normalized prediction, mean_prediction_difference_with_ci, and
        MAE/RMSE error parity when a numeric ground-truth column exists.
      rank -> ranking exposure (vfairness_metrics.ranking):
        position-discounted exposure parity + NDKL per attribute.

    Rows keep the exact perVariable shape the renderer expects
    (attribute, assessable, gap, tone, significant, worstGroup,
    referenceGroup, ciLow, ciHigh, fourFifthsRatio); fields that do
    not apply (fourFifthsRatio, effectSizeH) are None, never faked.
    ``framework`` is accepted for signature parity with the binary path;
    the four-fifths screen it parameterises is rate-based and does not
    apply here. Never raises (_safe pattern throughout).
    """

    def _emit(stage_index: int, label: str, pct: int) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage_index, label, pct)
        except Exception:  # noqa: BLE001, telemetry is never fatal
            pass

    raw = pd.to_numeric(work[pred_col], errors="coerce")
    valid = raw.notna()
    if not bool(valid.all()):
        # Rows whose prediction did not parse as a number cannot feed a
        # numeric battery; drop once, keeping every array row-aligned
        # (same discipline as the binary path's up-front notna filter).
        work = work[valid].reset_index(drop=True)
        raw = raw[valid].reset_index(drop=True)
    vals = raw.to_numpy(dtype=float)

    polarity = str(
        inputs.get("outcome_polarity") or inputs.get("outcomePolarity") or "positive_favorable"
    )
    polarity_flipped = polarity == "positive_unfavorable"
    data_preparation["outcomePolarity"] = polarity
    data_preparation["outputType"] = output_type

    positions = None
    if output_type == "continuous":
        lo = float(np.min(vals)) if len(vals) else 0.0
        hi = float(np.max(vals)) if len(vals) else 0.0
        span = hi - lo
        y_norm = ((vals - lo) / span) if span > 0 else np.zeros(len(vals))
        # G-11 polarity guard: continuous outputs are never 1-y flipped
        # (that is a binary transform). The flip is a SIGN/DIRECTION
        # flip only: negated values, used solely for the direction-of-
        # favorability reading. On the min-max normalized scale,
        # negation is exactly 1 - y_norm.
        y_fav = (1.0 - y_norm) if polarity_flipped else y_norm
        data_preparation["continuousNormalization"] = (
            f'"{pred_col}" min-max scaled from [{lo:g}, {hi:g}] to [0, 1] '
            "so group gaps read as normalized mean differences"
        )
        if polarity_flipped:
            data_preparation["predictionsInvertedForAnalysis"] = False
            data_preparation["continuousPolarityHandling"] = (
                "Positive prediction declared adverse: values were "
                "negated for the direction-of-favorability reading only "
                "(no 1-y binary flip; magnitudes unchanged)"
            )
    else:  # rank
        base = float(np.min(vals)) if len(vals) else 0.0
        positions = (vals - base).astype(int)  # 0 = top of the ranking
        expo = 1.0 / np.log2(positions + 2.0)
        expo_max = float(np.max(expo)) if len(expo) else 0.0
        if expo_max > 0:
            expo = expo / expo_max
        y_fav = expo  # per-item normalized exposure (1.0 = top slot)
        data_preparation["rankExposure"] = (
            f'"{pred_col}" read as ranking positions (top = most '
            "exposure); per-item exposure = 1/log2(position + 2), "
            "normalized to [0, 1]"
        )

    from vfairness.evaluation.vfairness_metrics import group_reliability

    def _row(attr: str) -> Dict[str, Any]:
        g = work[attr].astype("string").fillna("missing").to_numpy()
        n = min(len(g), len(y_fav))
        gv = g[:n]
        yv = np.asarray(y_fav[:n], dtype=float)
        sizes = {str(lab): int((gv == lab).sum()) for lab in pd.unique(gv)}
        rel = group_reliability(sizes)
        groups: List[Dict[str, Any]] = []
        for lab in pd.unique(gv):
            mask = gv == lab
            cnt = int(mask.sum())
            r = rel.get(str(lab), {"tier": "invalid", "interpretable": False, "note": ""})
            groups.append(
                {
                    "label": str(lab),
                    "n": cnt,
                    "rate": float(yv[mask].mean()) if cnt else 0.0,
                    "tier": r["tier"],
                    "interpretable": r["interpretable"],
                    "note": r["note"],
                    "reliable": r["tier"] in ("reliable", "caution"),
                }
            )
        groups.sort(key=lambda d: d["rate"], reverse=True)
        rates = [gr for gr in groups if gr["n"] > 0]
        if len(rates) < 2:
            return {
                "attribute": attr,
                "groups": groups,
                "assessable": False,
                "outputType": output_type,
                "reason": "Needs at least two groups with data.",
            }
        # Same headline ref/worst discipline as the binary path: the
        # reference is the most-populous reliable group, never a tiny
        # noisy one; small groups are reported and flagged, not used to
        # drive the headline.
        reliable = [gr for gr in rates if gr["n"] >= min_group]
        interpretable = [gr for gr in rates if gr["interpretable"]]
        pool = (
            reliable if len(reliable) >= 2 else interpretable if len(interpretable) >= 2 else rates
        )
        ref = max(pool, key=lambda d: d["n"])
        worst = min(pool, key=lambda d: d["rate"])
        if worst["label"] == ref["label"]:
            ref = max(pool, key=lambda d: d["rate"])
        gap = float(ref["rate"] - worst["rate"])

        # Bootstrap CI + two-sided p on the normalized gap (mirrors the
        # binary _per_variable inference so downstream reads are uniform).
        rng = np.random.default_rng(0)
        rp = yv[gv == ref["label"]]
        wp = yv[gv == worst["label"]]
        if len(rp) >= 5 and len(wp) >= 5:
            boots = np.empty(_BOOTSTRAP, dtype=float)
            for i in range(_BOOTSTRAP):
                boots[i] = (
                    rp[rng.integers(0, len(rp), len(rp))].mean()
                    - wp[rng.integers(0, len(wp), len(wp))].mean()
                )
            lo_, hi_ = np.quantile(boots, [(1 - _CI) / 2, 1 - (1 - _CI) / 2])
            significant = bool(lo_ > 0)
            nb = len(boots)
            p_two = 2.0 * min(
                (np.sum(boots <= 0) + 1) / (nb + 1), (np.sum(boots >= 0) + 1) / (nb + 1)
            )
            p_value = float(min(1.0, p_two))
            ci_low: Optional[float] = float(lo_)
            ci_high: Optional[float] = float(hi_)
        else:
            # A disparity that cannot be tested is NOT significant.
            ci_low = ci_high = None
            significant = False
            p_value = None

        detail: Dict[str, Any] = {}
        lib_min_group = max(5, min(int(min_group), 30))
        if output_type == "continuous":

            def _lib_mpd():
                from vfairness.evaluation.vfairness_metrics.regression import (
                    mean_prediction_difference_with_ci,
                )

                res = mean_prediction_difference_with_ci(
                    yv,
                    yv,
                    gv,
                    min_group_size=lib_min_group,
                    n_bootstrap=300,
                    method="bootstrap",
                    random_state=0,
                )
                return {
                    "metric": "mean_prediction_difference",
                    "value": float(res.point_estimate),
                    "ciLow": float(res.lower_bound),
                    "ciHigh": float(res.upper_bound),
                    "unit": "normalized_prediction",
                }

            detail["regression"] = _safe(
                _lib_mpd, None, degraded=degradations, label=f"regression_parity[{attr}]"
            )
            if has_truth and label_col and label_col in work.columns:

                def _lib_err():
                    yt = pd.to_numeric(work[label_col], errors="coerce").to_numpy(dtype=float)[:n]
                    mask = np.isfinite(yt)
                    if int(mask.sum()) < 10 or pd.Series(yt[mask]).nunique() <= 2:
                        return None  # not a regression ground truth
                    from vfairness.evaluation.vfairness_metrics.regression import (
                        mae_parity_difference,
                        rmse_parity_difference,
                    )

                    # Error parity on the RAW scale: errors are only
                    # honest in the units the model actually predicts.
                    return {
                        "maeParityDifference": float(
                            mae_parity_difference(
                                yt[mask], vals[:n][mask], gv[mask], min_group_size=lib_min_group
                            )
                        ),
                        "rmseParityDifference": float(
                            rmse_parity_difference(
                                yt[mask], vals[:n][mask], gv[mask], min_group_size=lib_min_group
                            )
                        ),
                    }

                detail["errorParity"] = _safe(
                    _lib_err, None, degraded=degradations, label=f"regression_error_parity[{attr}]"
                )
        else:

            def _lib_rank():
                from vfairness.evaluation.vfairness_metrics.ranking import (
                    exposure_parity_difference,
                    normalized_discounted_kl_divergence,
                )

                pos = positions[:n]
                return {
                    "exposureParityDifference": float(
                        exposure_parity_difference(pos, gv, min_group_size=5)
                    ),
                    "ndkl": float(normalized_discounted_kl_divergence(pos, gv, min_group_size=5)),
                }

            detail["ranking"] = _safe(
                _lib_rank, None, degraded=degradations, label=f"ranking_exposure[{attr}]"
            )

        return {
            "attribute": attr,
            "assessable": True,
            "groups": groups,
            "referenceGroup": ref["label"],
            "worstGroup": worst["label"],
            "gap": gap,
            "ciLow": ci_low,
            "ciHigh": ci_high,
            # Four-fifths is a selection-RATE screen: inapplicable to
            # continuous / rank outputs. None, never a fake ratio.
            "fourFifthsRatio": None,
            "pValue": p_value,
            "significant": significant,
            "significantRaw": significant,
            "tone": _tone(gap) if significant else "pass",
            "effectSizeH": None,
            "outputType": output_type,
            "gapUnit": (
                "normalized_mean_difference"
                if output_type == "continuous"
                else "normalized_exposure_difference"
            ),
            "smallGroups": [
                gr["label"]
                for gr in groups
                if gr["tier"] in ("underpowered", "caution") and gr["n"] > 0
            ],
            "invalidGroups": [
                gr["label"] for gr in groups if gr["tier"] == "invalid" and gr["n"] > 0
            ],
            **detail,
        }

    _emit(2, "Computing outcome-level parity across protected attributes", 40)
    per_variable = [
        _safe(
            lambda a=a: _row(a),
            {
                "attribute": a,
                "assessable": False,
                "reason": "Could not assess.",
                "outputType": output_type,
            },
            degraded=degradations,
            label=f"per_variable[{a}]",
        )
        for a in usable
    ]

    # Family-wise BH-FDR across the assessed attributes (same rule as
    # the binary battery's _fdr_correct). fourFifthsRatio is
    # inapplicable here, so the corrected tone is gap-based only.
    def _fdr() -> None:
        idx = [
            i
            for i, r in enumerate(per_variable)
            if isinstance(r, dict)
            and r.get("assessable")
            and isinstance(r.get("pValue"), (int, float))
        ]
        if len(idx) < 2:
            return
        adj = _bh_adjust([float(per_variable[i]["pValue"]) for i in idx])
        for j, i in enumerate(idx):
            ap = float(adj[j])
            per_variable[i]["pValueAdjusted"] = ap
            per_variable[i]["significant"] = bool(ap <= (1.0 - _CI))
            per_variable[i]["tone"] = (
                _tone(float(per_variable[i].get("gap") or 0.0))
                if per_variable[i]["significant"]
                else "pass"
            )

    _safe(_fdr, None, degraded=degradations, label="per_variable_fdr")

    _emit(5, "Ranking disparities and composing the verdict", 80)
    assessable = [v for v in per_variable if v.get("assessable")]
    ranked = sorted(assessable, key=lambda v: v.get("gap", 0.0), reverse=True)
    unit_phrase = (
        "average predicted outcome (0-100 normalized scale)"
        if output_type == "continuous"
        else "ranking exposure (0-100 normalized scale)"
    )
    if not ranked:
        tone = "warn"
        headline = "Could not compute a reliable fairness verdict on this data."
        summary = "No protected variable had enough grouped data to assess."
    else:
        worst_row = ranked[0]
        tone = worst_row["tone"]
        wl, rl = worst_row["worstGroup"], worst_row["referenceGroup"]
        gp = round(float(worst_row.get("gap") or 0.0) * 100)
        if tone == "critical":
            headline = f'Not fit to deploy as-is: material disparity for "{wl}".'
        elif tone == "warn":
            headline = f'Deployable only with monitoring: watch-level gap for "{wl}".'
        else:
            headline = "Within fairness budget on the assessed attributes."
        sig = "a statistically significant" if worst_row.get("significant") else "an unconfirmed"
        ci_txt = ""
        if isinstance(worst_row.get("ciLow"), (int, float)) and isinstance(
            worst_row.get("ciHigh"), (int, float)
        ):
            ci_txt = (
                f"; 95% CI {worst_row['ciLow'] * 100:.0f} to {worst_row['ciHigh'] * 100:.0f} pts"
            )
        summary = (
            f'Largest gap is in "{worst_row["attribute"]}": "{wl}" trails '
            f'"{rl}" by {gp} points of {unit_phrase} '
            f"({sig} difference{ci_txt})."
        )
    verdict = {
        "tone": tone,
        "headline": headline,
        "summary": summary,
        "ranked": [
            {
                "attribute": v["attribute"],
                "worstGroup": v["worstGroup"],
                "gap": v["gap"],
                "tone": v["tone"],
                "significant": v["significant"],
            }
            for v in ranked
        ],
    }

    from vfairness.operations.pulse.recommend import (
        recommend_fairness_definition,
        recommend_interventions,
    )

    recommendation = _safe(
        lambda: recommend_fairness_definition(domain, jurisdiction, has_truth),
        {},
        degraded=degradations,
        label="recommended_definition",
    )
    interventions = _safe(
        lambda: recommend_interventions([]), [], degraded=degradations, label="interventions"
    )

    def _assurance():
        from vfairness.operations.reporting import build_assurance_verdict

        return build_assurance_verdict(
            schema=schema,
            per_variable=per_variable,
            metrics=[],
            bias=[],
            proxies={},
            statistical={},
            intersectional={},
            disparity_matrix={},
            recommended=recommendation,
            domain=domain,
            jurisdiction=jurisdiction,
            has_truth=has_truth,
            artifact_hash=str(inputs.get("artifact_hash") or inputs.get("artifactHash") or "")
            or None,
        )

    _emit(7, "Composing the assurance verdict and recommendations", 96)
    assurance = _safe(
        _assurance,
        {
            "overall": "Disclaimer",
            "blocksDeployment": False,
            "oneLineVerdict": "The assurance verdict could not be assembled.",
            "findings": [],
            "recommendations": [],
            "metricsDeferred": {},
            "auditTrail": {},
        },
        degraded=degradations,
        label="assurance",
    )

    # Same honesty rule as the binary battery: a swallowed failure in a
    # load-bearing stage means the analysis is INCOMPLETE and must never
    # present as a clean pass.
    if degradations:
        try:
            verdict["tone"] = _worst_tone(str(verdict.get("tone") or "pass"), "warn")
            if isinstance(assurance, dict):
                ov = str(assurance.get("overall") or "")
                if ov in ("", "Unqualified"):
                    assurance["overall"] = "Disclaimer"
                    assurance["oneLineVerdict"] = (
                        "Incomplete analysis: "
                        f"{len(degradations)} stage(s) could not be "
                        "computed, so a clean opinion cannot be issued. "
                        "See degradations."
                    )
                else:
                    assurance["oneLineVerdict"] = (
                        str(assurance.get("oneLineVerdict") or "")
                        + " (Note: parts of the analysis could not be "
                        "computed; treat the verdict as a lower bound.)"
                    )
        except Exception:  # noqa: BLE001, post-processing is never fatal
            pass

    battery = (
        "regression-parity battery (group mean-outcome gaps with "
        "bootstrap CIs on the normalized prediction; MAE/RMSE error "
        "parity when a numeric ground-truth column exists)"
        if output_type == "continuous"
        else "ranking-exposure battery (position-discounted exposure parity "
        "and NDKL representation divergence)"
    )
    scope = build_scope_block(
        "tabular",
        covered=[
            f"Model output detected as {output_type}: the {battery} ran "
            "per protected attribute, with family-wise BH-FDR across "
            "attributes",
            "Per-group sizes and reliability tiers",
        ],
        not_covered=[
            "The binary decision battery (selection rates, four-fifths "
            "screen, bias taxonomy, proxy / leakage screen, "
            "intersectional subgroups, causal sketch): it needs a "
            "yes/no decision column",
            "Threshold-based mitigation sweeps (no decision threshold "
            "exists until a cut-off is chosen)",
        ],
        power_note=(
            f"Groups below n >= {min_group} did not drive the headline "
            "reading; small groups are listed per attribute row."
        ),
    )

    return {
        "success": True,
        "data": _jsonify(
            {
                "sourceKind": "tabular",
                "outputType": output_type,
                "degradations": degradations,
                "dataQuality": data_quality,
                "dataPreparation": data_preparation,
                "schema": schema,
                "verdict": verdict,
                "assurance": assurance,
                "scope": scope,
                "legalFramework": build_legal_framework_block(domain, jurisdiction),
                "perVariable": per_variable,
                "adjustedDisparity": [],
                "metrics": [],
                "bias": [],
                "temporalDrift": _na(
                    "Temporal drift screening runs on the binary decision battery."
                ),
                "specification": _na(
                    "Specification screening runs on the binary decision battery."
                ),
                "disparityMatrix": {},
                "proxies": _na("Proxy / leakage screening needs the binary decision battery."),
                "statistical": _na(
                    f"The {output_type} battery carries its own bootstrap and "
                    "BH-FDR inference (see perVariable)."
                ),
                # G-25 / G-35: the kNN-consistency, Theil and error-cohort reads
                # are defined on binary decisions; honestly not applicable here.
                "individualFairness": _na(
                    "The kNN consistency and Theil inequality reading is "
                    f"defined for binary decisions; this output is "
                    f"{output_type}."
                ),
                "cohorts": _na(
                    "Error-cohort analysis runs on the binary decision battery "
                    "against ground-truth outcomes."
                ),
                "labelQuality": _na(
                    "Label-quality screening runs on the binary decision "
                    "battery against ground-truth outcomes."
                ),
                "intersectional": {
                    "skipped": True,
                    "reason": (
                        "Intersectional subgroup analysis is not yet wired for "
                        f"{output_type} outputs."
                    ),
                },
                "pareto": {
                    "available": False,
                    "reason": (
                        "No decision threshold exists for a continuous or rank "
                        "output until a cut-off is chosen, so there is no "
                        "trade-off curve to sweep."
                    ),
                },
                "causal": {
                    "nodes": [],
                    "edges": [],
                    "overview": (
                        "Causal sketch is not built for continuous or rank outputs in triage."
                    ),
                },
                "recommendedDefinition": recommendation,
                "interventions": interventions,
                "nextStep": (
                    "Convert this Pulse into a tracked assessment to choose a "
                    "decision rule and run the full binary battery on it."
                ),
            }
        ),
    }


def run_pulse(df: pd.DataFrame, inputs: Dict[str, Any], progress=None) -> Dict[str, Any]:
    """Assemble the full Pulse result. Never raises.

    `progress`, when supplied, is an optional callback
    ``progress(stage, label, stage_index, total_stages, pct)`` used to
    report real per-stage telemetry to the caller (the Pulse consumer
    relays it to the live phase indicator). It defaults to None and is
    invoked through `_p`, which swallows every error -- progress
    reporting can never affect the analysis or raise. The Navigator does
    not call run_pulse, so this is an additive, single-caller change.
    """
    # Frontend PULSE_PHASES order (8 stages). Stage 0 ("loading") is
    # emitted by the consumer handler before run_pulse; stages 1-7 here.
    _TOTAL_STAGES = 8

    # Resolve the per-run min-group-size. The library default is 20
    # (_MIN_GROUP_DEFAULT, audit-grade); callers can override per audit via
    # inputs.minGroupSize (or min_group_size). See _resolve_min_group for
    # the clamp/parse rules. Propagated to every helper that gates by size.
    min_group = _resolve_min_group(inputs)
    # Resolve the disparity-severity framework ONCE from the caller's
    # (domain, jurisdiction). Threaded into _per_variable + _fdr_correct
    # so every tone decision uses the same framework. The framework
    # dict is also emitted on the PulseResult so the GUI can name the
    # legal basis for an auditor ("EEOC 4/5ths" vs "EU disparate-impact
    # testing" vs "generic heuristic"). This is what makes Pulse
    # universal across domains/jurisdictions instead of US-employment-
    # specific.
    framework = _legal_screen_framework(
        str(inputs.get("domain") or ""), str(inputs.get("jurisdiction") or "")
    )

    # Phase-4 regulatory packaging lives in its own module; the wiring
    # here stays thin. Options are total-parsed once (camelCase twins
    # accepted): harm_direction, scores_exposed, reference_group,
    # target_ratio.
    from vfairness.operations.pulse import regulatory as _regulatory

    pulse_options = _regulatory.parse_pulse_options(inputs)

    def _p(stage_index: int, label: str, pct: int) -> None:
        if progress is None:
            return
        try:
            progress("pulse", label, stage_index, _TOTAL_STAGES, pct)
        except Exception:  # noqa: BLE001 -- telemetry is never fatal
            pass

    domain = str(inputs.get("domain") or "")
    jurisdiction = str(inputs.get("jurisdiction") or "")
    requested = list(inputs.get("protected_attributes") or inputs.get("protectedAttributes") or [])

    # A2 input-contract router: prompts+outputs artefact -> generative path
    # (reuses OutputAnalyzer). Default = tabular. Misroute-safe: needs an
    # explicit source_kind OR both a prompt-like and a long free-text output
    # column present (the tabular audit file has neither).
    _src = str(inputs.get("source_kind") or inputs.get("sourceKind") or "").lower()
    # The frontend's explicit "Generative AI" selection arrives as
    # system_type. Honor it as a routing hint when source_kind is absent,
    # so a deliberate user choice is never silently coerced to predictive.
    # Still misroute-safe: the generative path below also requires a text
    # output column, so a mistaken generative pick on a tabular file falls
    # through to the predictive audit instead of erroring.
    _sys = str(inputs.get("system_type") or inputs.get("systemType") or "").lower()
    _txt = _pick(df, ("output", "response", "completion", "generation", "answer", "model_output"))
    _prm = _pick(df, ("prompt", "input", "question", "query", "instruction"))
    _gen_declared = _src in ("prompts_outputs", "text", "generative")
    _is_gen = (
        _gen_declared
        or (_src == "" and _sys == "generative" and _txt is not None)
        or (
            _src == ""
            and _txt is not None
            and _prm is not None
            and not pd.api.types.is_numeric_dtype(df[_txt])
            and df[_txt].astype(str).str.len().mean() > 25
        )
    )
    # A DECLARED kind either runs or refuses by name. The wizard always
    # sends an explicit source_kind, so silent fall-through to the tabular
    # path would misread the user's declaration (and then die on 'No
    # prediction column' after minutes of waiting).
    if _gen_declared and _txt is None and not inputs.get("llm_config"):
        return _kind_refusal(
            "prompts_outputs",
            "This run was declared as prompts + outputs, but no response "
            "text column was found (expected one of: output, response, "
            "completion, generation, answer, model_output). It was NOT "
            "silently analyzed as a tabular file.",
            "Add a response/output text column (and ideally a prompt "
            "column) to the CSV, then re-run.",
            domain,
            jurisdiction,
            rows=int(len(df)),
            columns=int(df.shape[1]),
        )
    if _is_gen and _txt is not None:
        _p(1, "Scoring generative outputs across groups", 30)
        out = _safe(
            lambda: _generative_pulse(
                df, inputs, _txt, requested, domain, jurisdiction, progress_cb=_p
            ),
            None,
        )
        if out is not None:
            _p(7, "Composing the assurance verdict and recommendations", 96)
            # G-34: recheck + legalContextAsOf (the pathway-agnostic
            # regulatory pieces); never fatal.
            return _regulatory.attach_lightweight_regulatory(out, inputs)

    # B1: live LLM-endpoint artefact -> reuse CounterfactualTester via the
    # shared llm_probe_pulse (one source of truth). Honest: no endpoint
    # configured => an explicit Disclaimer, never a fabricated verdict.
    _llm_cfg = inputs.get("llm_config") or (
        {
            "endpoint_url": inputs.get("endpoint"),
            "api_format": inputs.get("api_format", "openai"),
            "auth_token": inputs.get("auth_token"),
            "model_name": inputs.get("model_name"),
        }
        if inputs.get("endpoint")
        else None
    )
    if _llm_cfg or _src in ("endpoint", "llm", "model_endpoint"):
        from vfairness.operations.pulse.llm_probe import llm_probe_pulse

        _p(1, "Probing the live endpoint with counterfactual prompts", 30)
        out = _safe(
            lambda: llm_probe_pulse(_llm_cfg or {}, domain=domain, jurisdiction=jurisdiction),
            {
                "success": True,
                "data": {
                    "sourceKind": "llm_endpoint",
                    "assurance": {
                        "overall": "Disclaimer",
                        "blocksDeployment": False,
                        "oneLineVerdict": "LLM probe could not run.",
                        "findings": [],
                        "recommendations": [],
                        "metricsDeferred": {},
                        "auditTrail": {},
                    },
                    "verdict": {"tone": "neutral", "headline": "", "summary": "", "ranked": []},
                },
            },
        )
        # G-34: recheck + legalContextAsOf. The suite echo strips the
        # endpoint auth token (and every other secret) by construction.
        return _regulatory.attach_lightweight_regulatory(out, inputs)

    # B4: agent / multi-agent trace artefact -> reuse ToolBiasAuditor /
    # ActionBiasAnalyzer via the shared agent_probe_pulse. Misroute-safe:
    # needs an explicit source_kind OR a tool/action/route column AND a
    # group column (a feature CSV has neither).
    _act = _pick(
        df, ("tool", "tool_name", "action", "route", "delegate", "agent_action", "selected_tool")
    )
    _grp = next((c for c in (requested or []) if c in df.columns), None) or _pick(
        df, ("group", "demographic", "persona", "segment", "cohort")
    )
    _agent_declared = _src in ("agent", "agent_traces", "traces", "multiagent", "agentic")
    _is_agent = _agent_declared or (_src == "" and _act is not None and _grp is not None)
    if _agent_declared and (_act is None or _grp is None):
        missing = []
        if _act is None:
            missing.append(
                "a tool/action column (tool, tool_name, action, route, "
                "delegate, agent_action, selected_tool)"
            )
        if _grp is None:
            missing.append(
                "a demographic group column (one of the selected protected "
                "attributes, or group / demographic / persona / segment / "
                "cohort)"
            )
        return _kind_refusal(
            "agent_traces",
            "This run was declared as agent traces, but the table is "
            "missing " + " and ".join(missing) + ". It was NOT silently "
            "analyzed as a tabular file.",
            "Export traces with one row per episode carrying the acting "
            "group and the tool/action taken, then re-run.",
            domain,
            jurisdiction,
            rows=int(len(df)),
            columns=int(df.shape[1]),
        )
    if _is_agent and _act is not None and _grp is not None:
        from vfairness.operations.pulse.agent_probe import agent_probe_pulse

        _p(1, "Auditing tool and action distribution across groups", 30)
        out = _safe(lambda: agent_probe_pulse(df, inputs, _act, _grp, domain, jurisdiction), None)
        if out is not None:
            _p(7, "Composing the assurance verdict and recommendations", 96)
            return _regulatory.attach_lightweight_regulatory(out, inputs)

    # B2: image-set artefact -> vfairness.vision skew/NDKL/amplification on
    # a per-image demographic-label column (FairFace classification itself
    # is the sidecar-gated step; see vfairness.vision). Misroute-safe.
    _demo = _pick(
        df,
        (
            "detected_race",
            "face_race",
            "image_demographic",
            "detected_gender",
            "face_gender",
            "demographic_label",
            "image_race",
            "image_gender",
        ),
    )
    _img_declared = _src in ("image", "image_set", "t2i", "vision", "images")
    _is_img = _img_declared or (_src == "" and _demo is not None)
    if _img_declared and _demo is None:
        # G-16: an unlabeled manifest is auto-labeled by the vision
        # classifier (perceived demographics, disclosed) when it carries
        # image links and the classifier is available. Reaching this
        # refusal means neither a label column NOR usable auto-labeling
        # was present on THIS run -- state that honestly instead of
        # denying the capability (audit fix ux-5).
        _img_cols = ("image_url", "image", "url", "image_path", "path", "file")
        _has_links = any(str(c).strip().lower() in _img_cols for c in df.columns)
        if _has_links:
            reason = (
                "This run was declared as an image set and the manifest "
                "carries image links, but no perceived-demographic labels "
                "reached the reading: the vision classifier that labels an "
                "unlabeled manifest did not run on this dispatch (the "
                "analysis classifier was unavailable, or every image "
                "failed to fetch). It was NOT silently analyzed as a "
                "tabular file."
            )
            nxt = (
                "Re-run once the analysis classifier is available, or "
                "add a per-image demographic-label column (e.g. "
                "detected_gender) produced by your own classifier."
            )
        else:
            reason = (
                "This run was declared as an image set, but the manifest "
                "carries neither a per-image demographic-label column "
                "(e.g. detected_gender, detected_race) nor an image-link "
                "column the vision classifier could label (image_url, "
                "image, url, image_path). It was NOT silently analyzed as "
                "a tabular file."
            )
            nxt = (
                "Add an image-link column (Pulse will auto-label the "
                "perceived demographics) or a demographic-label column "
                "from your own classifier, then re-run."
            )
        return _kind_refusal(
            "image_set",
            reason,
            nxt,
            domain,
            jurisdiction,
            rows=int(len(df)),
            columns=int(df.shape[1]),
        )
    if _is_img and _demo is not None:
        from vfairness.operations.pulse.vision_probe import vision_probe_pulse

        _p(1, "Reading representation skew across image labels", 30)
        out = _safe(lambda: vision_probe_pulse(df, inputs, _demo, domain, jurisdiction), None)
        if out is not None:
            _p(7, "Composing the assurance verdict and recommendations", 96)
            return _regulatory.attach_lightweight_regulatory(out, inputs)

    pred_col = _pick(df, ("prediction", "predicted", "pred", "y_pred", "score", "decision"))

    # The model's PREDICTION for fairness is the hard yes/no decision, not a
    # continuous score. _pick can grab "model_score" (continuous) ahead of
    # "invite_decision" (the actual decision) purely from column order, which
    # silently washes out group differences (every group's median-thresholded
    # score looks similar) -- a root cause of missed biases. If the picked
    # column is not binary-ish but a binary decision column exists, prefer it.
    def _binaryish(col: str) -> bool:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() >= 0.5:
            u = set(pd.unique(s.dropna()))
            return bool(u) and u <= {0, 1}
        u = set(df[col].astype("string").str.strip().str.lower().dropna())
        return 0 < len(u) <= 3

    if pred_col is not None and not _binaryish(pred_col):
        decision_like = [
            c
            for c in df.columns
            if any(
                k in c.lower()
                for k in (
                    "decision",
                    "prediction",
                    "approved",
                    "hired",
                    "selected",
                    "invite",
                    "outcome_pred",
                )
            )
            and _binaryish(c)
        ]
        if decision_like:
            pred_col = decision_like[0]

    label_col = _pick(df, ("label", "y_true", "target", "outcome", "ground_truth", "y"))
    # G-11: ranking artefacts carry no prediction/decision column; a
    # rank/position column IS the model output. Only reached when the
    # binary picker found nothing, so binary inputs are untouched.
    if pred_col is None:
        _rank_col = _pick(df, ("rank", "position", "ranking"))
        if (
            _rank_col is not None
            and _rank_col != label_col
            and pd.to_numeric(df[_rank_col], errors="coerce").notna().mean() >= 0.5
        ):
            pred_col = _rank_col
    if pred_col is None:
        return {
            "success": False,
            "error": "No prediction/decision column found. Pulse needs "
            "the model's output column (e.g. 'prediction').",
        }
    has_truth = label_col is not None and label_col != pred_col

    # 1. canonical preparation (Phase 1)
    from vfairness.preprocessing import prepare_protected_attributes

    prep = prepare_protected_attributes(df, requested)
    work = prep.frame
    # The protected-value absence mask is taken HERE, while ``work`` still
    # carries the caller's index, so it can be read from the RAW column too; see
    # the block below ``usable`` for why that matters.
    _prepared_usable = [a for a in prep.usable if a in work.columns and a != pred_col]
    #: Per axis, then the union. Computed once, while the index still matches
    #: ``df``, and sliced with the frame afterwards: recomputing it against a
    #: reset index would silently lose the raw half of the test.
    _absent_by_axis = {a: _rows_missing_a_protected_value(df, work, [a]) for a in _prepared_usable}
    _no_protected = _rows_missing_a_protected_value(df, work, _prepared_usable)
    # Drop rows with no model decision ONCE, up-front, and reset the index so
    # every derived array (y_pred, y_true, group/sensitive columns) is the
    # same length and the SAME rows. Row alignment is load-bearing.
    if pred_col in work.columns:
        _keep_decision = work[pred_col].notna()
        work = work[_keep_decision].reset_index(drop=True)
        _no_protected = _no_protected[_keep_decision.to_numpy()]
        _absent_by_axis = {
            a: m[_keep_decision.to_numpy()].reset_index(drop=True)
            for a, m in _absent_by_axis.items()
        }
    _no_protected = _no_protected.reset_index(drop=True)
    usable = [a for a in prep.usable if a in work.columns and a != pred_col]

    # A ROW WITH NO PROTECTED VALUE IS NOT A PROTECTED GROUP (grade wave G06,
    # 2026-09-30). Dropped ONCE here, beside the no-decision drop above and for
    # the same reason: row alignment is load-bearing, and the exclusion has to
    # be the same rows for every section below.
    #
    # This closes a SECOND, DISAGREEING COPY of a judgement this library had
    # already made. ``_validation.handle_missing_values`` defaults to
    # ``missing_strategy='exclude'``, names its opt-in level ``__missing__``
    # rather than a str()-minted one, and its own comment says a phantom group
    # made out of absent values "could drive the reported disparity";
    # ``selection_rate_disparity_matrix`` was fixed for exactly that on
    # 2026-09-09 and ``validate_inputs``'s intersectional branch on 2026-09-30.
    # Every section below this line instead wrote ``.astype("string")
    # .fillna("missing")`` for itself, twenty-odd times, so the rows carrying NO
    # protected attribute became a demographic group literally named "missing".
    # Measured here before this drop, on 400 rows labelled "A" and 400 whose
    # protected value is absent (identical for pd.NA, np.nan and None):
    #
    #   verdict.headline  'Not fit to deploy as-is: material disparity for
    #                      "missing".'
    #   perVariable       groups [A n=400 rate 0.635, missing n=400 rate 0.280],
    #                     worstGroup 'missing', gap 0.355, pValue 0.002
    #   and the minted level also reached disparityMatrix, assurance,
    #   individualFairness, causal and regulatoryExports, with NO warning.
    #
    # In the SAME payload, ``dataPreparation.binning`` carried the binner's own
    # note: '"group" has only one distinct value (\'A\'), so it forms a single
    # group. No between-group fairness comparison exists for it.' The binner
    # counts levels with ``dropna=True`` and was right; the verdict was built on
    # a group the same payload said did not exist, and it is the verdict a reader
    # reads. A four-fifths ratio and an adverse-impact headline are claims about
    # a protected class, and "we have no record of this person's protected
    # attribute" is not one.
    #
    # A WHITESPACE-ONLY STRING IS NORMALISED INTO THE SAME ABSENCE rather than
    # tested separately, so this stays ONE rule (``pd.isna``) rather than
    # becoming the seventh disagreeing copy. Measured before: a blank protected
    # value published 'material disparity for ""'. The literal string 'None' is
    # NOT treated as absent: it is indistinguishable from a category somebody
    # chose, which is the same reason the library refuses to let its
    # ``__missing__`` sentinel swallow a genuine level spelled "missing".
    #
    # IT NEVER EMPTIES THE FRAME. If excluding would leave no row at all (two
    # attributes absent on disjoint halves), the rows are KEPT and the affected
    # axes leave ``usable`` instead, which routes the run into the existing
    # "No assessable protected attribute" refusal. A guard that empties the
    # frame is the wrong guard, and so is one that publishes an adverse-impact
    # headline about a phantom because the alternative was awkward: when every
    # single row is missing something, no axis can be read on complete rows and
    # the honest answer is that nothing was assessed.
    # THE RAW COLUMN IS CONSULTED, NOT ONLY THE PREPARED ONE, because the
    # preparation itself mints a level: ``protected_binning`` bands a date of
    # birth into age bands and writes ``.fillna("missing")``, so a DOB column
    # half full of ``pd.NaT`` arrives here holding the literal STRING "missing"
    # and ``isna()`` is False for it. Measured before, on 400 dated rows and 400
    # ``pd.NaT``: groups [under_18 n=400, missing n=400], headline 'material
    # disparity for "missing"'. See ``_rows_missing_a_protected_value``.
    protected_rows_excluded = 0
    protected_exclusion_blocked: List[str] = []
    if usable:
        n_absent = int(_no_protected.sum())
        if n_absent:
            if n_absent < len(work):
                work = work[(~_no_protected).to_numpy()].reset_index(drop=True)
                protected_rows_excluded = n_absent
            else:
                protected_exclusion_blocked = [
                    a for a in usable if bool(_absent_by_axis.get(a, pd.Series(dtype=bool)).any())
                ]
                usable = [a for a in usable if a not in protected_exclusion_blocked]

    # Transparency: every transformation applied before a single metric was
    # computed (DOB->age banding, ZIP roll-up, quantile bands, and the
    # value-identity de-dup that drops a duplicate age axis) plus every
    # column excluded and why. Audit-grade tools must never transform data
    # silently -- this is surfaced in the UI's "How your data was prepared".
    data_preparation = {
        "binning": [{"label": str(lbl), "detail": str(det)} for (lbl, det) in (prep.notes or [])],
        "excluded": [
            {"column": str(col), "reason": str(rsn)} for (col, rsn) in (prep.excluded or [])
        ],
        "assessed": list(usable),
        #: Rows dropped because no protected value was recorded for them. 0 on a
        #: complete frame, so a clean run reads exactly as it did before.
        "rowsExcludedNoProtectedValue": protected_rows_excluded,
    }
    if protected_rows_excluded:
        data_preparation["binning"].append(
            {
                "label": "Coverage",
                "detail": (
                    f"{protected_rows_excluded} row(s) carry no recorded value for "
                    f"{', '.join(chr(34) + str(a) + chr(34) for a in usable)} (blank, "
                    "null or missing) and were EXCLUDED from every fairness comparison "
                    "below. They are not a demographic group: a four-fifths ratio and "
                    "an adverse-impact finding are claims about a protected class, and "
                    "an absent attribute is not one. Nothing here says those rows were "
                    "treated fairly or unfairly; they were not assessed. Supply the "
                    "attribute for them, or analyse missingness as its own question."
                ),
            }
        )
    if protected_exclusion_blocked:
        names = ", ".join(str(a) for a in protected_exclusion_blocked)
        data_preparation["excluded"].extend(
            {
                "column": str(a),
                "reason": (
                    f'"{a}" is unrecorded on some rows, and every row in the frame is '
                    "missing at least one of the chosen protected attributes, so no "
                    "complete row survives to compare protected groups on. Excluded "
                    "from the assessment rather than analysed with a group standing for "
                    "the absent value: a four-fifths ratio and an adverse-impact "
                    "finding are claims about a protected class, and an absent "
                    "attribute is not one. Nothing here says these rows were treated "
                    "fairly or unfairly; they were not assessed. Assess one attribute "
                    "at a time, or supply the attribute for the rows that lack it."
                ),
            }
            for a in protected_exclusion_blocked
        )
        data_preparation["assessed"] = list(usable)
        data_preparation["binning"].append(
            {
                "label": "Coverage",
                "detail": (
                    f"Every row in this frame is missing at least one of {names}, so "
                    "excluding the unrecorded rows would leave nothing to compare. "
                    f"{'Those attribute(s) were' if len(protected_exclusion_blocked) > 1 else 'That attribute was'}"
                    " excluded from the assessment instead. Read this run as "
                    "unassessed on them, not clean."
                ),
            }
        )

    # 1b. column-role typology -- identity/PII, oracle, model output,
    #     protected, proxy candidate, job-relevant. Same vfairness function
    #     the Navigator profiling step uses (one source of truth). Drives
    #     the PII/oracle exclusion set so leakage + metrics never train on
    #     an identifier or the answer key.
    def _schema():
        from vfairness.evaluation.vfairness_metrics import classify_column_roles

        return classify_column_roles(
            df,
            declared_protected=requested,
            declared_prediction=pred_col,
            declared_target=(label_col if has_truth else None),
        )

    # Load-bearing-stage failures are RECORDED (not silently defaulted) so
    # the verdict can never read as a clean pass when an analysis stage
    # actually collapsed. Created before the FIRST recordable stage.
    degradations: List[Dict[str, Any]] = []
    # Consumer-side model scoring (operations/pulse/scoring.py via the
    # consumer shim) reports its dropped/imputed features here so they
    # feed the same verdict-downgrade machinery as in-engine stages.
    for _d in inputs.get("_scoring_degradations") or []:
        if isinstance(_d, dict) and _d.get("stage"):
            degradations.append(_d)

    # G-44: protected-encoding audit (T1-06). A binary or flattened
    # encoding of a protected attribute is itself a modeling decision
    # that can erase people (sex recorded as two values, race flattened
    # to two buckets). This is a DISCLOSURE, not a finding: two recorded
    # values can be exactly what the source system captured, so the
    # audit names what the encoding cannot see instead of accusing.
    def _encoding_audit() -> List[Dict[str, str]]:
        audit: List[Dict[str, str]] = []
        sexish = ("sex", "gender")
        raceish = ("race", "ethnic", "nationality", "national_origin")
        for attr in usable:
            vals = work[attr].dropna().astype(str)
            levels = sorted(vals.unique().tolist())
            if len(levels) != 2:
                continue
            name_l = str(attr).lower()
            shown = "'{0}' / '{1}'".format(levels[0], levels[1])
            if any(t in name_l for t in sexish):
                detail = (
                    "'{0}' is recorded with exactly two values ({1}). "
                    "Non-binary people are invisible to every reading on "
                    "this attribute; disparities affecting them cannot "
                    "appear in this report.".format(attr, shown)
                )
            elif any(t in name_l for t in raceish):
                detail = (
                    "'{0}' is recorded with exactly two values ({1}). A "
                    "two-bucket encoding flattens distinct groups "
                    "together; disparities between the groups inside a "
                    "bucket cannot appear in this report.".format(attr, shown)
                )
            else:
                detail = (
                    "'{0}' is recorded with exactly two values ({1}); "
                    "any finer structure within these categories is "
                    "invisible to this reading.".format(attr, shown)
                )
            audit.append({"attribute": str(attr), "levels": shown, "detail": detail})
        return audit

    data_preparation["encodingAudit"] = _safe(
        _encoding_audit, [], degraded=degradations, label="protected_encoding_audit"
    )

    _p(1, "Checking data quality and column roles", 12)
    schema = _safe(
        _schema,
        {
            "roles": [],
            "pii_leakage": [],
            "oracle_columns": [],
            "model_output": [],
            "proxy_candidates": [],
            "mismatches": [],
            "refuse": False,
            "refuse_reason": "",
        },
        degraded=degradations,
        label="schema_roles",
    )
    # Columns that must never feed a feature-based model: identity/PII, the
    # oracle/target, and the model's own output. Fed into the E1 leakage
    # test and any feature-based stage.
    exclude_cols = sorted(
        set(
            (schema.get("pii_leakage") or [])
            + (schema.get("oracle_columns") or [])
            + (schema.get("model_output") or [])
            + (schema.get("unknown") or [])
        )
    )

    # 2. data quality (canonical translator)
    def _dq():
        from vfairness.operations.cicd import build_quality_report

        return build_quality_report(
            df, requested, outcome_column=(label_col if has_truth else None)
        )

    data_quality = _safe(
        _dq,
        {
            "tone": "warn",
            "headline": "Data quality could not be confirmed.",
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "checks": [],
        },
        degraded=degradations,
        label="data_quality",
    )

    # Refuse-fast: an unrecoverable schema error (declared target missing, or
    # the dataset is all identifiers / answer keys). Still returns a VALID
    # PulseResult -- the user gets a clear verdict, never a blank failure.
    if schema.get("refuse"):
        return {
            "success": True,
            "data": _jsonify(
                {
                    "sourceKind": "tabular",
                    "degradations": degradations,
                    "scope": build_scope_block(
                        "tabular", [], ["Analysis refused before any battery ran (schema gate)."]
                    ),
                    "legalFramework": build_legal_framework_block(domain, jurisdiction),
                    "dataQuality": data_quality,
                    "schema": schema,
                    "assurance": {
                        "overall": "Disclaimer",
                        "blocksDeployment": True,
                        "oneLineVerdict": schema.get("refuse_reason", ""),
                        "findings": [],
                        "recommendations": [],
                        "metricsDeferred": {},
                        "auditTrail": {},
                    },
                    "verdict": {
                        "tone": "critical",
                        "headline": "Cannot assess this artifact as provided.",
                        "summary": schema.get("refuse_reason", ""),
                        "ranked": [],
                    },
                    "perVariable": [],
                    "metrics": [],
                    "bias": [],
                    "causal": {},
                    "proxies": {"available": False, "proxies": [], "chains": []},
                    "disparityMatrix": {},
                    "statistical": {"available": False, "perAttribute": []},
                    "intersectional": {"skipped": True, "reason": schema.get("refuse_reason", "")},
                    "recommendedDefinition": {},
                    "interventions": [],
                    "nextStep": "Fix the schema issue above, then re-run Pulse.",
                }
            ),
        }

    if not usable:
        # NAME THE REASON THIS RUN REFUSED (grade wave G06, 2026-09-30). The
        # generic sentence below is about identifiers and cardinality, and it is
        # simply untrue of an attribute that WAS groupable and left because no
        # complete row carried it. A reader who acts on "pick a categorical
        # attribute" when the attribute was already categorical looks in the
        # wrong place, and this is the surface the refusal is read from.
        if protected_exclusion_blocked:
            _no_attr_summary = (
                f"{', '.join(str(a) for a in protected_exclusion_blocked)}: every row in "
                "the frame is missing at least one of the chosen protected attributes, so "
                "there is no complete row to compare protected groups on. Nothing was "
                "assessed, which is not the same as nothing being found. Supply the "
                "attribute for the rows that lack it, or assess one attribute at a time."
            )
            _no_attr_not_covered = [
                "No protected attribute had a complete row to assess, so no battery ran."
            ]
        else:
            _no_attr_summary = (
                "Every chosen attribute was an identifier or could not be grouped. "
                "Pick a categorical or age/income-style attribute."
            )
            _no_attr_not_covered = ["No assessable protected attribute, so no battery ran."]
        return {
            "success": True,
            "data": _jsonify(
                {
                    "sourceKind": "tabular",
                    "degradations": degradations,
                    "scope": build_scope_block("tabular", [], _no_attr_not_covered),
                    "legalFramework": build_legal_framework_block(domain, jurisdiction),
                    "dataQuality": data_quality,
                    "schema": schema,
                    # Carried here too, so the exclusion that CAUSED this refusal
                    # is readable in the payload that reports it.
                    "dataPreparation": data_preparation,
                    "verdict": {
                        "tone": "critical",
                        "headline": "No assessable protected attribute.",
                        "summary": _no_attr_summary,
                        "ranked": [],
                    },
                    "perVariable": [],
                    "metrics": [],
                    "bias": [],
                    "causal": {},
                    "proxies": {"available": False, "proxies": [], "chains": []},
                    "disparityMatrix": {},
                    "statistical": {"available": False, "perAttribute": []},
                    "intersectional": {
                        "skipped": True,
                        "reason": "No assessable protected attribute, so intersectional "
                        "subgroups cannot be formed.",
                    },
                    "recommendedDefinition": {},
                    "interventions": [],
                    "nextStep": "Pick an assessable attribute and re-run.",
                }
            ),
        }

    # G-11: regression + ranking routing. Binary inputs stay byte-
    # identical (the detector returns 'binary' for every 0/1 or yes/no
    # column and the code below is unchanged); CONTINUOUS and RANK
    # outputs are routed to the library's regression-parity / ranking-
    # exposure batteries instead of being silently median-thresholded
    # into a fake binary decision (the old _coerce_binary behavior,
    # which washed out real group differences).
    output_type = _safe(
        lambda: _detect_output_type(df, pred_col, inputs),
        "binary",
        degraded=degradations,
        label="output_type_detection",
    )
    if output_type in ("continuous", "rank"):
        out = _nonbinary_tabular_pulse(
            df=df,
            work=work,
            usable=usable,
            inputs=inputs,
            schema=schema,
            data_quality=data_quality,
            data_preparation=data_preparation,
            degradations=degradations,
            framework=framework,
            domain=domain,
            jurisdiction=jurisdiction,
            min_group=min_group,
            pred_col=pred_col,
            label_col=label_col,
            has_truth=has_truth,
            output_type=output_type,
            progress_cb=_p,
        )
        # Non-binary tabular runs still get the modality-agnostic
        # regulatory pieces: recheck, legalContextAsOf, column-level
        # legal admissibility and the historical-context catalog. The
        # LL144 impact-ratio table needs selection RATES, which do not
        # exist for continuous/rank outputs, so it is honestly omitted.
        return _regulatory.attach_lightweight_regulatory(
            out, inputs, domain=domain, jurisdiction=jurisdiction, columns=list(df.columns)
        )

    y_pred = _coerce_binary(work[pred_col])
    y_true = _coerce_binary(work[label_col]) if has_truth else None

    # Outcome polarity: for systems where the positive prediction is an
    # ADVERSE event (fraud flag, risk score, rejection), a raw selection-
    # rate reading would name the over-flagged group as "favored" -- an
    # inverted verdict. Flip ONCE here so 1 == favorable for every stage
    # downstream; the flip is disclosed in dataPreparation and the
    # intersectional engine receives positive_favorable to avoid double
    # handling.
    polarity = str(
        inputs.get("outcome_polarity") or inputs.get("outcomePolarity") or "positive_favorable"
    )
    polarity_flipped = polarity == "positive_unfavorable"
    if polarity_flipped:
        y_pred = 1 - y_pred
        if y_true is not None:
            y_true = 1 - y_true
    data_preparation["outcomePolarity"] = polarity
    data_preparation["predictionsInvertedForAnalysis"] = bool(polarity_flipped)

    # 3. per-variable metrics with bootstrap CIs. G-32: a caller-requested
    #    reference group (inputs.reference_group[attr]) is threaded through
    #    so the comparison baseline honors the request when the group
    #    exists; the disclosure lands in the referenceGroups block below.
    _reference_requests = pulse_options.get("reference_group") or {}
    _p(2, "Computing fairness metrics across every protected attribute", 35)
    per_variable = [
        _safe(
            lambda a=a: _per_variable(
                work,
                a,
                y_pred,
                y_true,
                min_group=min_group,
                framework=framework,
                reference_override=_reference_requests.get(a),
            ),
            {"attribute": a, "assessable": False, "reason": "Could not assess."},
            degraded=degradations,
            label=f"per_variable[{a}]",
        )
        for a in usable
    ]

    # Benjamini-Hochberg FDR correction ACROSS every assessed attribute.
    # Each _per_variable did an independent test; with k attributes the
    # family-wise false-positive probability is ~1-(1-alpha)^k (~34% at
    # k=8). The intersectional path already BH-corrects -- the headline
    # disparate-impact path must too, or it is the dominant spurious
    # "Adverse" driver. Reuse the canonical monotone BH (one source of
    # truth) and overwrite the provisional per-attribute `significant`.
    def _fdr_correct(rows):
        idx = [
            i
            for i, r in enumerate(rows)
            if isinstance(r, dict)
            and r.get("assessable")
            and isinstance(r.get("pValue"), (int, float))
        ]
        if len(idx) < 2:
            return  # nothing to correct (0/1 test): leave as-is
        from vfairness.evaluation.vfairness_metrics._statistics import benjamini_hochberg_correction

        pvals = np.array([rows[i]["pValue"] for i in idx], dtype=float)
        res = benjamini_hochberg_correction(pvals, alpha=1.0 - _CI)
        adj = np.asarray(res.adjusted_p_values, dtype=float)
        for j, i in enumerate(idx):
            ap = float(adj[j])
            rows[i]["pValueAdjusted"] = ap
            # `<=` (not the canonical strict `<`) so a hypothesis exactly
            # at the FDR boundary is still rejected (BH 1995 is inclusive).
            rows[i]["significant"] = bool(ap <= (1.0 - _CI))
        # Recompute the four-fifths tone with the corrected significance so
        # severity never rests on an uncorrected test.
        for i in idx:
            r = rows[i]
            wr = next(
                (
                    gr["rate"]
                    for gr in (r.get("groups") or [])
                    if gr.get("label") == r.get("worstGroup")
                ),
                None,
            )
            br = max(
                (
                    gr["rate"]
                    for gr in (r.get("groups") or [])
                    if isinstance(gr.get("rate"), (int, float))
                ),
                default=0.0,
            )
            if wr is not None:
                r["tone"] = _worst_tone(
                    _tone(r.get("gap") or 0.0),
                    _disparity_tone(
                        r.get("fourFifthsRatio") or 0.0,
                        bool(r["significant"]),
                        float(wr),
                        float(br),
                        framework=framework,
                        effect_size_h=float(r.get("effectSizeH") or 0.0),
                    ),
                )

    _safe(lambda: _fdr_correct(per_variable), None, degraded=degradations, label="per_variable_fdr")

    # CONFOUND FLAGGING (T2.3, dataset-agnostic, stratification-based).
    # For each protected attribute X with an adverse-impact tone, check
    # whether its disparity is largely mediated by another protected
    # attribute Y. The Cochran-Mantel-Haenszel-style stratified gap pools
    # the within-Y-stratum X-disparity weighted by stratum size. When the
    # stratified gap shrinks by >= 70% relative to the marginal gap, X is
    # tagged ``confoundedBy: [{attribute, reductionPct, ...}]`` so the
    # GUI can render the dependency ("religion's disparity is mediated
    # by race") instead of presenting the marginal finding as primary.
    def _stratified_gap(
        attr_x: str, worst_x: str, best_x: str, attr_y: str
    ) -> Optional[Dict[str, float]]:
        if attr_x not in work.columns or attr_y not in work.columns:
            return None
        n_ = min(len(work), len(y_pred))
        if n_ == 0:
            return None
        xs = work[attr_x].astype("string").fillna("missing").to_numpy()[:n_]
        ys = work[attr_y].astype("string").fillna("missing").to_numpy()[:n_]
        yp = np.asarray(y_pred[:n_], dtype=float)
        m_w = xs == worst_x
        m_b = xs == best_x
        if m_w.sum() < 20 or m_b.sum() < 20:
            return None
        marginal_gap = float(yp[m_b].mean() - yp[m_w].mean())
        if marginal_gap <= 0:
            return None
        weighted_gap = 0.0
        total_w = 0
        for y_val in np.unique(ys):
            mask_y = ys == y_val
            mw = mask_y & m_w
            mb = mask_y & m_b
            if mw.sum() < 10 or mb.sum() < 10:
                continue
            stratum_n = int(mask_y.sum())
            gap = float(yp[mb].mean() - yp[mw].mean())
            weighted_gap += gap * stratum_n
            total_w += stratum_n
        if total_w == 0:
            return None
        stratified_gap = weighted_gap / total_w
        return {
            "marginalGap": marginal_gap,
            "stratifiedGap": stratified_gap,
            "reductionPct": float(
                100 * (1.0 - stratified_gap / marginal_gap) if marginal_gap > 0 else 0.0
            ),
        }

    # adjustedDisparity rows accumulate alongside the confound flags so a
    # single stratification pass covers both T2.3 (confound flag) and T3
    # (residualised demographic parity).
    adjusted_disparity: List[Dict[str, Any]] = []

    def _flag_confounds(rows):
        idx = [
            i
            for i, r in enumerate(rows)
            if isinstance(r, dict)
            and r.get("assessable")
            and r.get("tone") in ("warn", "critical")
            and r.get("worstGroup")
            and r.get("referenceGroup")
        ]
        for i in idx:
            r = rows[i]
            attr_x = r["attribute"]
            groups = r.get("groups") or []
            best_x = max(
                (g for g in groups if isinstance(g.get("rate"), (int, float))),
                key=lambda g: g["rate"],
                default=None,
            )
            worst_x = r["worstGroup"]
            if best_x is None or best_x.get("label") == worst_x:
                continue
            best_lbl = best_x["label"]
            confounds: List[Dict[str, Any]] = []
            stratified_by_y: List[Tuple[str, float, float]] = []  # (Y, adjusted_gap, reduction_pct)
            marginal_gap_x: Optional[float] = None
            for attr_y in usable:
                if attr_y == attr_x:
                    continue
                stats_ = _stratified_gap(attr_x, worst_x, best_lbl, attr_y)
                if stats_ is None:
                    continue
                marginal_gap_x = stats_["marginalGap"]
                stratified_by_y.append((attr_y, stats_["stratifiedGap"], stats_["reductionPct"]))
                if stats_["reductionPct"] >= 70.0:
                    confounds.append(
                        {
                            "attribute": attr_y,
                            "reductionPct": stats_["reductionPct"],
                            "marginalGap": stats_["marginalGap"],
                            "stratifiedGap": stats_["stratifiedGap"],
                            "method": "stratification_mantel_haenszel_pool",
                        }
                    )
            if confounds:
                confounds.sort(key=lambda d: -d["reductionPct"])
                r["confoundedBy"] = confounds[:3]
            # T3: adjustedDisparity record. The adjusted gap is the
            # MINIMUM (most-deconfounded) of the per-Y stratified gaps.
            # If even after the strongest single-Y mediation the gap is
            # still material (>= 5 pp), the disparity is robust; otherwise
            # it's largely explained by another protected trait and the
            # row should be read with that caveat.
            if stratified_by_y and marginal_gap_x is not None:
                strongest = min(stratified_by_y, key=lambda t: t[1])
                adjusted_gap = float(strongest[1])
                adj_tone = (
                    "critical"
                    if adjusted_gap >= 0.20
                    else "warn"
                    if adjusted_gap >= 0.05
                    else "pass"
                )
                adjusted_disparity.append(
                    {
                        "attribute": attr_x,
                        "marginalGap": float(marginal_gap_x),
                        "adjustedGap": adjusted_gap,
                        "strongestMediator": strongest[0],
                        "reductionPct": float(strongest[2]),
                        "tone": adj_tone,
                        # significant = the adjusted-after-deconfounding gap
                        # still clears the EEOC-style 5-pp materiality bar.
                        "significant": adjusted_gap >= 0.05,
                        "method": "stratification_mantel_haenszel_min_pooled",
                        "mediatorsScanned": [t[0] for t in stratified_by_y],
                    }
                )

    _safe(
        lambda: _flag_confounds(per_variable),
        None,
        degraded=degradations,
        label="per_variable_confound_flag",
    )

    # G-42: trimmed-mean divergence flag. When a continuous score exists,
    # recompute each assessed attribute's selection-rate gap after
    # dropping each group's extreme 5 percent of scores (2.5% per tail).
    # When the trimmed gap diverges from the raw gap by more than 20%
    # relative, the row gets a robustnessNote: part of the headline gap
    # rides on extreme-score rows (data-entry outliers or a thin tail).
    # Purely additive: tones and significance never move here.
    def _trimmed_divergence() -> None:
        score_col = _pick(
            df, ("probability", "proba", "prob", "score", "y_score", "y_prob", "confidence")
        )
        if not score_col or score_col not in work.columns:
            return
        sp = pd.to_numeric(work[score_col], errors="coerce")
        if sp.notna().mean() < 0.5 or sp.nunique() < 5:
            return
        sv = sp.to_numpy(dtype=float)
        for r in per_variable:
            if not isinstance(r, dict) or not r.get("assessable"):
                continue
            raw_gap = r.get("gap")
            # Gaps below one point have no meaningful relative reading.
            if not isinstance(raw_gap, (int, float)) or abs(raw_gap) < 0.01:
                continue
            g = work[r["attribute"]].astype("string").fillna("missing").to_numpy()
            n = min(len(g), len(y_pred), len(sv))

            def _trimmed_rate(lbl, g=g, n=n):
                mask = g[:n] == lbl
                s_g = sv[:n][mask]
                y_g = np.asarray(y_pred[:n], dtype=float)[mask]
                fin = np.isfinite(s_g)
                s_g, y_g = s_g[fin], y_g[fin]
                if len(s_g) < 40:  # a 2.5% tail needs rows to trim
                    return None
                lo, hi = np.quantile(s_g, [_TRIM_TAIL, 1.0 - _TRIM_TAIL])
                keep = (s_g >= lo) & (s_g <= hi)
                if int(keep.sum()) < 20:
                    return None
                return float(y_g[keep].mean())

            t_ref = _trimmed_rate(r.get("referenceGroup"))
            t_worst = _trimmed_rate(r.get("worstGroup"))
            if t_ref is None or t_worst is None:
                continue
            trimmed_gap = float(t_ref - t_worst)
            r["trimmedGap"] = trimmed_gap
            # READINESS-6, 2026-09-10. `rel = |trimmed - raw| / max(|raw|, 1e-9)`
            # is a RELATIVE change against a baseline that is routinely near
            # zero, which is the shape that prints "-20,000,000,000% improvement"
            # elsewhere. When the raw gap is ~0, any movement at all clears the
            # 20% bar and the note fires saying the gap "moves from 0 to 0".
            #
            # A relative change needs a baseline big enough to be a baseline. The
            # absolute floor below is deliberately the same order as the smallest
            # gap anyone would act on: a movement of less than half a percentage
            # point is not outlier sensitivity worth telling a reader about,
            # whatever it is as a ratio of nearly nothing.
            if abs(float(raw_gap)) < _TRIM_MIN_BASELINE:
                continue
            rel = abs(trimmed_gap - float(raw_gap)) / abs(float(raw_gap))
            if rel > _TRIM_DIVERGENCE:
                r["robustnessNote"] = (
                    "Outlier sensitivity: after dropping each group's "
                    f"extreme 5 percent of '{score_col}' scores, the "
                    "selection-rate gap moves from "
                    f"{float(raw_gap) * 100:.0f} to "
                    f"{trimmed_gap * 100:.0f} points "
                    f"({rel * 100:.0f}% relative change). Part of the "
                    "headline gap rides on extreme-score rows; check "
                    "those rows for data-entry outliers before acting "
                    "on the exact gap size."
                )

    _safe(_trimmed_divergence, None, degraded=degradations, label="trimmed_gap_robustness")

    # rigorous metric cards via FairnessAnalyzer (bootstrap CIs) -- computed
    # for EVERY assessable attribute, not just the first. A model can be fair
    # on gender and severely biased on disability/national-origin; scanning
    # only usable[0] is exactly why Pulse missed most implanted biases. Same
    # FairnessAnalyzer + compute_effect_sizes the Navigator's
    # vfairness_compute_metrics handler uses (one source of truth).
    LABEL_DEPENDENT = {
        "equalized_odds_difference",
        "equal_opportunity_difference",
        "predictive_parity_difference",
        "mae_parity_difference",
        "rmse_parity_difference",
    }

    def _cards_for(attr: str) -> List[Dict[str, Any]]:
        from vfairness.evaluation.vfairness_metrics.analyzer import (
            FairnessAnalyzer,
            MetricResult,
        )

        sens = work[attr].astype("string").fillna("missing").to_numpy()
        n = min(len(sens), len(y_pred))
        yt = y_true[:n] if y_true is not None else y_pred[:n]
        an = FairnessAnalyzer(yt, y_pred[:n], sens[:n], min_group_size=_MIN_GROUP)
        allm = an.compute_all_metrics(include_ci=True, n_bootstrap=_BOOTSTRAP, confidence_level=_CI)
        # Effect sizes (Cohen's h on the disparity) so a tiny-but-significant
        # gap is not over-sold and a large practical gap is never buried --
        # exactly the helper the Navigator metrics handler calls.
        eff: Dict[str, Any] = {}
        try:
            from vfairness.evaluation.vfairness_metrics.classification import compute_effect_sizes

            eff = compute_effect_sizes(yt, y_pred[:n], sens[:n]) or {}
        except Exception:  # noqa: BLE001 -- effect size is additive, never fatal
            eff = {}
        out = []
        for name, res in allm.items():
            val = res.value if isinstance(res, MetricResult) else res
            ci = getattr(res, "confidence_interval", (float("nan"), float("nan")))
            needs_truth = name in LABEL_DEPENDENT
            uncomputable_no_truth = needs_truth and not has_truth
            is_nan = val != val
            computable = (not uncomputable_no_truth) and (not is_nan)
            if uncomputable_no_truth:
                note = (
                    "Cannot be calculated: this compares decisions against "
                    "the real outcome, and this dataset has no ground-truth "
                    "labels. Re-run with an outcome column to see it."
                )
            elif is_nan:
                note = (
                    "Cannot be calculated on this data (no group cleared "
                    "the minimum size, or the metric is undefined here)."
                )
            else:
                note = ""
            es = eff.get(name) if isinstance(eff, dict) else None
            out.append(
                {
                    "key": f"{attr}::{name}",
                    "metric": name,
                    "label": name.replace("_", " ").title(),
                    "attribute": attr,
                    "labelDependent": needs_truth,
                    # G-44: the metric's ethical stance (worldview it encodes);
                    # None for a metric not in the explicit map, never a guess.
                    "stance": _METRIC_STANCE.get(name),
                    "value": None if not computable else float(val),
                    "ciLow": None if (not computable or ci[0] != ci[0]) else float(ci[0]),
                    "ciHigh": None if (not computable or ci[1] != ci[1]) else float(ci[1]),
                    "effectSize": (
                        float(es)
                        if isinstance(es, (int, float))
                        else (es if isinstance(es, dict) else None)
                    ),
                    "computable": computable,
                    "note": note,
                }
            )
        return out

    metrics: List[Dict[str, Any]] = []
    for a in usable:
        metrics.extend(
            _safe(lambda a=a: _cards_for(a), [], degraded=degradations, label=f"metric_cards[{a}]")
        )
    # Worst disparity first so the real issue leads, not alphabetical noise.
    metrics.sort(key=lambda m: abs(m.get("value") or 0.0), reverse=True)

    # 4. bias taxonomy + proxies + intersectional + recommendations
    def _bias():
        from vfairness.preprocessing.bias_detection.detector import BiasDetector

        det = BiasDetector(work, protected_attributes=usable, outcome_column=pred_col)
        rep = det.full_audit(include_intersectional=True)
        return rep.to_dict() if hasattr(rep, "to_dict") else {}

    _p(3, "Detecting bias patterns and group disparity", 52)
    bias_report = _safe(_bias, {}, degraded=degradations, label="bias_taxonomy")
    bias_findings: List[Dict[str, Any]] = []
    bias_types: List[str] = []
    # Every taxonomy category the BiasDetector report exposes -- historical,
    # representation, disparity, proxy, AND aggregation / measurement / label
    # / sampling / temporal when present. The old hard-coded 4-category,
    # 6-per-category cap silently dropped most implanted mechanisms; iterate
    # all *_findings and keep up to 40 per category (effectively all, with a
    # blow-up guard on pathological high-dimensional inputs).
    finding_kinds = [
        k for k in (bias_report or {}) if isinstance(k, str) and k.endswith("_findings")
    ]
    # Stable, meaningful order; any extra kinds the library adds still run.
    _kind_order = [
        "historical_findings",
        "representation_findings",
        "disparity_findings",
        "proxy_findings",
        "aggregation_findings",
        "measurement_findings",
        "label_findings",
        "sampling_findings",
        "temporal_findings",
    ]
    finding_kinds.sort(key=lambda k: (_kind_order.index(k) if k in _kind_order else 99, k))
    for kind in finding_kinds:
        for f in (bias_report.get(kind) or [])[:40]:
            t = kind[: -len("_findings")]
            bias_types.append(t)
            fd = f if isinstance(f, dict) else {}

            # Disparity findings (StatisticalDisparityResult.to_dict) carry no
            # description/message/summary -- the old code fell through to
            # str(f), which dumped the raw dict ("{'feature': ..., 'pvalue':
            # 0.0, ...}") into the user-facing `plain` text and got truncated
            # mid-word at [:300]. Build a clean sentence AND a structured
            # statisticalTest the UI can render as a visualization instead.
            stat_test = None
            msg: str
            if fd.get("disparity_type") and fd.get("test_name"):
                priv = fd.get("privileged_group")
                disadv = fd.get("disadvantaged_group")
                feat = fd.get("feature") or "this feature"
                attr_name = fd.get("protected_attribute") or "the protected attribute"
                eff = fd.get("effect_interpretation") or ""
                pv = fd.get("pvalue")
                p_phrase = (
                    "p < 0.001"
                    if isinstance(pv, (int, float)) and pv < 0.001
                    else f"p = {pv:.3f}"
                    if isinstance(pv, (int, float))
                    else ""
                )
                msg = (
                    f"{feat} differs across {attr_name} "
                    f"({fd.get('test_name')}"
                    f"{', ' + str(eff).strip() + ' effect' if eff else ''}"
                    f"{', ' + p_phrase if p_phrase else ''}). "
                    + (
                        f"Most favoured: {priv}; least favoured: {disadv}."
                        if priv and disadv
                        else ""
                    )
                ).strip()
                stat_test = {
                    "feature": fd.get("feature"),
                    "protectedAttribute": fd.get("protected_attribute"),
                    "disparityType": fd.get("disparity_type"),
                    "testName": fd.get("test_name"),
                    "testStatistic": fd.get("test_statistic"),
                    "pValue": pv,
                    "significance": fd.get("significance"),
                    "effectSize": fd.get("effect_size"),
                    "effectSizeType": fd.get("effect_size_type"),
                    "effectInterpretation": eff or None,
                    "privilegedGroup": priv,
                    "disadvantagedGroup": disadv,
                    "disparityMagnitude": fd.get("disparity_magnitude"),
                }
            else:
                raw_msg = fd.get("description") or fd.get("message") or fd.get("summary")
                if not raw_msg:
                    # Last-resort: never serialize a raw dict into the UI.
                    msg = (
                        f"{t.capitalize()} bias detected"
                        + (f" in {fd['attribute']}" if fd.get("attribute") else "")
                        + "."
                    )
                else:
                    msg = str(raw_msg)

            # Group membership names can arrive either as bare strings or as
            # {"group": "...", ...} dicts depending on the finding kind.
            def _group_names(items: Any) -> List[str]:
                out: List[str] = []
                for g in items or []:
                    if isinstance(g, dict):
                        name = g.get("group") or g.get("label") or g.get("name")
                        if name is not None:
                            out.append(str(name))
                    else:
                        out.append(str(g))
                return out

            # Severity: prefer an explicit `severity`; otherwise fall back to
            # the finding's own risk vocabulary. Historical patterns
            # (HistoricalPatternResult.to_dict) carry NO `severity` key -- they
            # expose `risk_level` (critical/high/medium/low). The old code
            # `fd.get("severity", "warn")` therefore hardcoded EVERY historical
            # finding to "warn", silently downgrading HIGH/CRITICAL patterns
            # such as Geographic Redlining so they ranked below ordinary
            # warnings and were visually buried in the bias taxonomy. Pass the
            # real risk word through (the frontend sevRank understands
            # critical/high/medium/low) so redlining surfaces at its true
            # severity in the Historical bias section.
            _sev_raw = fd.get("severity") or fd.get("risk_level") or fd.get("risk") or "warn"
            # Resolve a structured historical-discrimination pattern for this
            # finding when its (attribute, domain) matches a documented
            # precedent (race+lending => redlining, age+hiring => ADEA, ...).
            # Findings of kind "historical" that already carry a pattern_type
            # populate the envelope from their own dict so the UI never has
            # to keyword-match evidence text for redlining etc. either.
            _attr_for_pat = fd.get("attribute") or fd.get("feature")
            _hist_pat = None
            try:
                from vfairness.preprocessing.bias_detection.historical import (
                    _classify_attribute,
                    attribute_historical_pattern,
                )

                _hist_pat = attribute_historical_pattern(_attr_for_pat, domain, jurisdiction)
            except Exception:
                _hist_pat = None

                def _classify_attribute(  # local no-op fallback
                    name: Optional[str],
                ) -> Optional[str]:
                    return None

            if t == "historical" and fd.get("pattern_type"):
                # Native HistoricalPatternResult: surface its own label /
                # context / citations so the UI flag is populated even when
                # there is no (attribute, domain) entry in the lookup.
                _hist_pat = {
                    "pattern_id": (
                        str(fd.get("pattern_type") or "").strip().lower().replace(" ", "_")
                        or "historical_pattern"
                    ),
                    "label": fd.get("pattern_type") or "Historical pattern",
                    "summary": (fd.get("historical_context") or fd.get("description") or ""),
                    "citations": fd.get("citations") or [],
                    "attribute_class": (_classify_attribute(_attr_for_pat) or "unknown"),
                    "domain": str(domain or "").strip().lower(),
                    "jurisdiction": jurisdiction or "",
                    **(
                        {"affected_groups": fd.get("affected_groups")}
                        if fd.get("affected_groups")
                        else {}
                    ),
                }
            bias_findings.append(
                {
                    "type": t,
                    "plain": str(msg)[:300],
                    "severity": str(_sev_raw).lower().split(".")[-1],
                    # Structured statistical test for disparity findings so the UI
                    # renders a visualization, not a raw dict. None for other kinds.
                    "statisticalTest": stat_test,
                    # Per-variable detail so the UI can group by bias type and
                    # render the group distribution graphically on hover. Kept
                    # camelCase to match the rest of the Pulse contract.
                    # Historical findings key the column as `feature`, not
                    # `attribute`; fall back so the UI ties e.g. Geographic
                    # Redlining to `zip_code` instead of rendering it unattributed.
                    "attribute": _attr_for_pat,
                    "groupDistributions": fd.get("group_distributions") or {},
                    "representationRatios": fd.get("representation_ratios") or {},
                    "underrepresentedGroups": _group_names(fd.get("underrepresented_groups")),
                    "overrepresentedGroups": _group_names(fd.get("overrepresented_groups")),
                    # Structured historical-pattern envelope (None when the
                    # combination has no documented precedent). The frontend keys
                    # the "Historical pattern" channel off this flag instead of
                    # regex-matching evidence text, so race/age/gender findings
                    # in regulated domains no longer fall through to disparity-
                    # only labelling.
                    "historicalPattern": _hist_pat,
                }
            )

    # 4a. selection-rate disparity matrix (jurisdiction-NEUTRAL). Pure
    #     descriptive group x group ratio + difference + per-group CI. The
    #     legal interpretation (US four-fifths vs EU proportionality) is
    #     applied later by the verdict engine, never baked into the metric.
    def _disp():
        from vfairness.evaluation.vfairness_metrics import selection_rate_disparity_matrix

        out = {}
        for a in usable:
            sens = work[a].astype("string").fillna("missing").to_numpy()
            n = min(len(sens), len(y_pred))
            out[a] = selection_rate_disparity_matrix(y_pred[:n], sens[:n])
        return out

    disparity_matrix = _safe(_disp, {}, degraded=degradations, label="disparity_matrix")

    # 4b. proxy / correlation screen (same vfairness functions the Navigator
    #     proxy handler uses). Folds into bias_types so the intervention
    #     engine recommends proxy mitigation when stand-ins exist.
    _p(4, "Screening for proxy and redundant-encoding leakage", 65)
    proxies = _safe(
        lambda: _proxies(
            work,
            usable,
            exclude_cols=exclude_cols,
            y_pred=y_pred,
            proxy_candidates=list(schema.get("proxy_candidates") or []),
            pred_col=pred_col,
            label_col=label_col,
        ),
        {
            "available": False,
            "proxies": [],
            "chains": [],
            "summary": "Proxy analysis could not be computed.",
        },
        degraded=degradations,
        label="proxies",
    )
    if proxies.get("proxies"):
        bias_types.append("proxy")

    # 4c. temporal / distribution-shift bias (static-snapshot risk screen;
    #     Suresh & Guttag deployment bias, EU AI Act Art. 15(4)).
    def _tdrift():
        from vfairness.preprocessing.bias_detection import detect_temporal_drift

        # Use the BINNED frame (work), not raw df: PSI must be measured on
        # age *bands* etc., never raw dates/ages. Raw near-unique columns
        # produce a degenerate PSI (e.g. 12.08 on raw date_of_birth) that
        # falsely escalates to "critical".
        return detect_temporal_drift(work, usable, prediction=pred_col)

    temporal_drift = _safe(
        _tdrift,
        {"available": False, "reason": "Temporal drift unavailable."},
        degraded=degradations,
        label="temporal_drift",
    )
    if temporal_drift.get("available"):
        for dd in temporal_drift.get("drift") or []:
            if dd.get("severity") in ("warn", "critical"):
                bias_types.append("temporal")
                bias_findings.append(
                    {
                        "type": "temporal",
                        "plain": dd.get("plain", ""),
                        "severity": dd.get("severity", "warn"),
                        "attribute": dd.get("attribute"),
                        "statisticalTest": None,
                        "groupDistributions": {},
                        "representationRatios": {},
                        "underrepresentedGroups": [],
                        "overrepresentedGroups": [],
                    }
                )

    # 4d. specification / learning bias (Jacobs & Wallach 2021): construct
    #     validity, target leakage, proxy target, circular evaluation.
    def _spec():
        from vfairness.preprocessing.bias_detection import detect_specification_bias

        return detect_specification_bias(
            df,
            outcome=(label_col if has_truth else None),
            prediction=pred_col,
            exclude_columns=exclude_cols,
        )

    specification = _safe(
        _spec,
        {"available": False, "reason": "Specification screen failed."},
        degraded=degradations,
        label="specification",
    )
    if specification.get("available"):
        for sf in specification.get("findings") or []:
            bias_types.append("specification")
            bias_findings.append(
                {
                    "type": "specification",
                    "plain": sf.get("plain", ""),
                    "severity": sf.get("severity", "warn"),
                    "attribute": None,
                    "statisticalTest": None,
                    "groupDistributions": {},
                    "representationRatios": {},
                    "underrepresentedGroups": [],
                    "overrepresentedGroups": [],
                }
            )

    # 4c. statistical significance + robustness battery (permutation tests +
    #     subgroup robustness -- same entry points as the Navigator's
    #     statistical_validation / robustness_test handlers).
    _p(5, "Running statistical robustness and confidence intervals", 78)
    statistical = _safe(
        lambda: _statistical(work, usable, y_pred, y_true),
        {"available": False, "perAttribute": []},
        degraded=degradations,
        label="statistical",
    )

    # 4e. group calibration stage (G-12). When the model exposes a
    #     continuous score/probability (same picker the Pareto sweep
    #     uses) AND real ground-truth labels exist, per-group Expected
    #     Calibration Error is a first-class fairness read: a score that
    #     means different things for different groups breaks every
    #     downstream threshold, even when selection rates look clean.
    #     Uses the library's calibration_disparity
    #     (post_processing.calibration) per protected attribute.
    def _calibration():
        score_col = _pick(
            df, ("probability", "proba", "prob", "score", "y_score", "y_prob", "confidence")
        )
        if not score_col or score_col not in work.columns:
            return {
                "available": False,
                "reason": (
                    "No continuous score/probability column found, so group "
                    "calibration cannot be read."
                ),
            }
        if y_true is None:
            return {
                "available": False,
                "reason": (
                    "Calibration compares scores against real outcomes; no "
                    "ground-truth column was provided."
                ),
            }
        sp = pd.to_numeric(work[score_col], errors="coerce")
        if sp.notna().mean() < 0.5 or sp.nunique() < 5:
            return {
                "available": False,
                "reason": (
                    f'"{score_col}" is not a usable continuous score, so '
                    "group calibration cannot be read."
                ),
            }
        s = sp.to_numpy(dtype=float)
        # Scores must be probabilities in [0, 1]; min-max rescale a raw
        # score once and disclose it (per-group ECE gaps survive a
        # monotone rescale far better than a silent refusal would).
        rescaled = False
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            return {"available": False, "reason": (f'"{score_col}" has no finite values.')}
        s_lo, s_hi = float(finite.min()), float(finite.max())
        if s_lo < 0.0 or s_hi > 1.0:
            span = s_hi - s_lo
            s = ((s - s_lo) / span) if span > 0 else np.zeros_like(s)
            rescaled = True
        # Calibration compares the RAW score to the RAW outcome. y_true
        # was polarity-normalized above (1 == favorable); undo the flip
        # so an adverse-positive system's score is read against the
        # event it actually predicts.
        y_cal = (1 - y_true) if polarity_flipped else y_true
        from vfairness.post_processing.calibration.metrics import (
            calibration_disparity,
        )

        per_attr: List[Dict[str, Any]] = []
        for attr in usable:
            sens = work[attr].astype("string").fillna("missing").to_numpy()
            n = min(len(sens), len(s), len(y_cal))
            mask = np.isfinite(s[:n])
            if int(mask.sum()) < 10:
                continue
            yt_ = y_cal[:n][mask]
            sp_ = np.clip(s[:n][mask], 0.0, 1.0)
            sv_ = sens[:n][mask]
            res = _safe(
                lambda yt_=yt_, sp_=sp_, sv_=sv_: calibration_disparity(
                    yt_, sp_, sv_, n_bins=10, min_group_size=min_group
                ),
                None,
                degraded=degradations,
                label=f"calibration[{attr}]",
            )
            if res is None:
                continue
            group_ece = dict(getattr(res, "group_ece", None) or {})
            if len(group_ece) < 2:
                continue
            counts = pd.Series(sv_).value_counts()
            cal_groups = [
                {"group": str(gk), "ece": float(ge), "n": int(counts.get(gk, 0))}
                for gk, ge in group_ece.items()
            ]
            cal_groups.sort(key=lambda d: d["ece"], reverse=True)
            max_gap = float(getattr(res, "ece_disparity", 0.0) or 0.0)
            cal_tone = "critical" if max_gap >= 0.10 else "warn" if max_gap >= 0.05 else "pass"
            worst_g = str(
                getattr(res, "most_miscalibrated_group", "")
                or (cal_groups[0]["group"] if cal_groups else "")
            )
            best_g = str(getattr(res, "least_miscalibrated_group", "") or "")
            worst_e = float(group_ece.get(worst_g, 0.0) or 0.0)
            plain = (
                f'For "{attr}", the score is least trustworthy for '
                f'"{worst_g}": its average calibration error is '
                f"{worst_e * 100:.0f} points versus the best-calibrated "
                f'group ("{best_g}"), a between-group gap of '
                f"{max_gap * 100:.0f} points. The same score means "
                "different things for different groups, so any single "
                "threshold treats them unequally."
            )
            per_attr.append(
                {
                    "attribute": attr,
                    "groups": cal_groups,
                    "maxGap": max_gap,
                    "tone": cal_tone,
                    "mostMiscalibratedGroup": worst_g,
                    "leastMiscalibratedGroup": best_g,
                    "recommendations": [
                        str(x) for x in (getattr(res, "recommendations", None) or [])
                    ][:4],
                    "plain": plain,
                }
            )
        if not per_attr:
            return {
                "available": False,
                "reason": (
                    "No protected attribute had two groups above the "
                    f"n >= {min_group} floor with a readable score."
                ),
            }
        worst_row = max(per_attr, key=lambda r_: r_["maxGap"])
        return {
            "available": True,
            "scoreColumn": score_col,
            "scoreRescaled": rescaled,
            "method": (
                "per-group Expected Calibration Error, 10 uniform "
                "bins; gap = max between-group ECE difference "
                "(warn >= 0.05, critical >= 0.10)"
            ),
            "perAttribute": per_attr,
            "summary": (
                f"Worst calibration gap: {worst_row['maxGap'] * 100:.0f} "
                f'ECE points on "{worst_row["attribute"]}" (group '
                f'"{worst_row["mostMiscalibratedGroup"]}").'
            ),
        }

    calibration = _safe(
        _calibration,
        {"available": False, "reason": "Calibration stage could not run."},
        degraded=degradations,
        label="calibration",
    )
    # Significant calibration gaps are first-class findings: the verdict
    # and the assurance engine must see them (a parity-clean selection
    # rate can still hide a score that lies about one group). Type
    # "calibration_disparity"; the "calibration" bias type also routes
    # the per-group calibration intervention.
    if calibration.get("available"):
        for _c in calibration.get("perAttribute") or []:
            if _c.get("tone") in ("warn", "critical"):
                bias_types.append("calibration")
                bias_findings.append(
                    {
                        "type": "calibration_disparity",
                        "plain": _c.get("plain", ""),
                        "severity": _c.get("tone", "warn"),
                        "attribute": _c.get("attribute"),
                        "statisticalTest": None,
                        "groupDistributions": {},
                        "representationRatios": {},
                        "underrepresentedGroups": [],
                        "overrepresentedGroups": [],
                    }
                )

    # 4e-bis. G-38: label-quality / annotation-bias screen. The battery
    #     otherwise treats the ground-truth labels as gospel; this stage
    #     reads the LABELS themselves:
    #       (a) recorded favorable-outcome base rate per protected group
    #           (a skew here means any model trained on these labels
    #           inherits it; the reading honestly cannot distinguish
    #           historical bias in how outcomes were defined or recorded
    #           from real group differences, and says so);
    #       (b) inter-annotator agreement by group, only when the upload
    #           carries two or more annotator/rater/coder columns (raw
    #           agreement + Cohen's kappa; a group whose labels the
    #           raters disagree on has less reliable ground truth).
    #     Counterfactual label tests (T2-12: re-labeling with protected
    #     descriptors swapped) need a labeling pass and are named as not
    #     covered in triage.
    def _label_quality() -> Dict[str, Any]:
        if not has_truth or y_true is None:
            return {
                "available": False,
                "notApplicable": True,
                "reason": (
                    "Label-quality screening reads the ground-truth "
                    "labels; no real outcome column was provided."
                ),
            }
        n_rows = int(min(len(work), len(y_true)))
        y_lab = np.asarray(y_true[:n_rows], dtype=float)

        # (a) favorable-label base rate per group.
        per_attr: List[Dict[str, Any]] = []
        for attr in usable:
            gv = work[attr].astype("string").fillna("missing").to_numpy()[:n_rows]
            rows: List[Dict[str, Any]] = []
            for lab in pd.unique(gv):
                mask = gv == lab
                n_g = int(mask.sum())
                if n_g < min_group:
                    continue
                rows.append(
                    {"group": str(lab), "n": n_g, "labelBaseRate": float(y_lab[mask].mean())}
                )
            if len(rows) < 2:
                continue
            ref = max(rows, key=lambda r_: r_["n"])
            for r_ in rows:
                r_["ratioVsReference"] = (
                    (r_["labelBaseRate"] / ref["labelBaseRate"])
                    if ref["labelBaseRate"] > 0
                    else None
                )
            non_ref = [
                r_ for r_ in rows if r_ is not ref and isinstance(r_.get("ratioVsReference"), float)
            ]
            worst = min(non_ref, key=lambda r_: r_["ratioVsReference"]) if non_ref else None
            per_attr.append(
                {
                    "attribute": attr,
                    "referenceGroup": ref["group"],
                    "groups": rows,
                    "worstGroup": (worst["group"] if worst else None),
                    "worstRatio": (worst["ratioVsReference"] if worst else None),
                }
            )

        # (b) inter-annotator agreement, gated on real annotator columns.
        ann_cols, ann_rejected = _lq_annotator_columns(work)
        if len(ann_cols) < 2:
            reason = (
                "Inter-annotator agreement needs two or more "
                "annotator/rater columns in the upload; this "
                "artifact carries " + (f"one ('{ann_cols[0]}')" if ann_cols else "none") + "."
            )
            if ann_rejected:
                # Say what was looked at and dropped. A silent "none" here is
                # how the substring version's opposite failure hid: columns
                # were selected, or refused, with nothing recorded either way.
                reason += " Named like annotator columns but not label-shaped: " + "; ".join(
                    f"'{c}' ({why})" for c, why in ann_rejected[:6]
                )
            agreement: Dict[str, Any] = {
                "available": False,
                "notApplicable": True,
                "reason": reason,
                "rejectedColumns": [{"column": c, "reason": why} for c, why in ann_rejected],
            }
        else:

            def _kappa(a: np.ndarray, b: np.ndarray) -> float:
                po = float((a == b).mean())
                pe = 0.0
                for v in np.unique(np.concatenate([a, b])):
                    pe += float((a == v).mean()) * float((b == v).mean())
                return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0

            pairs = [
                (ann_cols[i], ann_cols[j])
                for i in range(len(ann_cols))
                for j in range(i + 1, len(ann_cols))
            ]
            av = {
                c: work[c].astype("string").fillna("missing").to_numpy()[:n_rows] for c in ann_cols
            }
            overall_agree = float(np.mean([(av[a] == av[b]).mean() for a, b in pairs]))
            overall_kappa = float(np.mean([_kappa(av[a], av[b]) for a, b in pairs]))
            by_group: List[Dict[str, Any]] = []
            for attr in usable:
                gv = work[attr].astype("string").fillna("missing").to_numpy()[:n_rows]
                rows = []
                for lab in pd.unique(gv):
                    mask = gv == lab
                    if int(mask.sum()) < min_group:
                        continue
                    rows.append(
                        {
                            "group": str(lab),
                            "n": int(mask.sum()),
                            "agreement": float(
                                np.mean([(av[a][mask] == av[b][mask]).mean() for a, b in pairs])
                            ),
                        }
                    )
                if len(rows) >= 2:
                    by_group.append({"attribute": attr, "groups": rows})
            agreement = {
                "available": True,
                "annotatorColumns": [str(c) for c in ann_cols],
                "overallAgreement": overall_agree,
                "overallKappa": overall_kappa,
                "byGroup": by_group,
            }

        return {
            "available": True,
            "labelBaseRates": per_attr,
            "annotatorAgreement": agreement,
            "thresholds": {
                "skewRatio": _LQ_SKEW_RATIO,
                "skewCritical": _LQ_SKEW_CRITICAL,
                "agreementGap": _LQ_AGREEMENT_GAP,
            },
            "notCovered": (
                "Counterfactual label tests (re-labeling with protected "
                "descriptors swapped, T2-12) need a labeling pass and "
                "are not run in triage."
            ),
            "method": (
                "Recorded favorable-outcome base rate per protected "
                f"group (groups below n >= {min_group} excluded; largest "
                "group as reference; the four-fifths band reused as an "
                "evidentiary screen, not a legal threshold) plus raw "
                "inter-annotator agreement and Cohen's kappa when the "
                "upload carries two or more annotator columns."
            ),
            "plain": (
                "This screen reads the labels themselves: a model "
                "trained on skewed or unreliable labels inherits their "
                "bias no matter how fair its own arithmetic is."
            ),
        }

    label_quality = _safe(
        _label_quality,
        {"available": False, "reason": "The label-quality stage could not be computed."},
        degraded=degradations,
        label="label_quality",
    )

    def _lq_findings() -> None:
        if not label_quality.get("available"):
            return
        for block in label_quality.get("labelBaseRates") or []:
            ratio = block.get("worstRatio")
            if not isinstance(ratio, (int, float)) or ratio >= _LQ_SKEW_RATIO:
                continue
            # Two-proportion z-test worst vs reference (audit fix lq-1): a
            # raw base-rate ratio at the n>=20 floor is dominated by
            # sampling noise, so the finding may only reach "critical" when
            # the difference is statistically significant. Otherwise it is
            # surfaced at "warn". statisticalTest carries the real p so the
            # verdict roll-up gate escalates only on a confirmed skew.
            rows = {r_["group"]: r_ for r_ in (block.get("groups") or [])}
            w = rows.get(block.get("worstGroup"))
            rf = rows.get(block.get("referenceGroup"))
            p_val = None
            if w and rf:

                def _two_prop() -> float:
                    from scipy.stats import norm

                    nw, nr = int(w["n"]), int(rf["n"])
                    pw, pr = float(w["labelBaseRate"]), float(rf["labelBaseRate"])
                    xw, xr = pw * nw, pr * nr
                    pooled = (xw + xr) / (nw + nr)
                    se = (pooled * (1 - pooled) * (1.0 / nw + 1.0 / nr)) ** 0.5
                    if se == 0:
                        return 1.0
                    z = (pw - pr) / se
                    return float(2.0 * norm.sf(abs(z)))

                p_val = _safe(_two_prop, None, degraded=degradations, label="label_baserate_ztest")
            sig = isinstance(p_val, float) and p_val < 0.05
            bias_types.append("label_baserate_skew")
            bias_findings.append(
                {
                    "type": "label_baserate_skew",
                    "plain": (
                        "The RECORDED outcomes are themselves skewed: "
                        f"'{block.get('worstGroup')}' carries the favorable "
                        f"label at {ratio * 100:.0f}% of the "
                        f"'{block.get('referenceGroup')}' rate on "
                        f"'{block.get('attribute')}'"
                        + (f" (two-proportion p {p_val:.3g})" if isinstance(p_val, float) else "")
                        + ". The reading cannot "
                        "distinguish historical bias in how outcomes were "
                        "defined or recorded from real group differences; "
                        "either way, any model trained on these labels "
                        "inherits the skew."
                    ),
                    "severity": ("critical" if (ratio < _LQ_SKEW_CRITICAL and sig) else "warn"),
                    "attribute": block.get("attribute"),
                    "statisticalTest": (
                        {"test": "two_proportion_z", "pValue": p_val} if p_val is not None else None
                    ),
                    "significant": bool(sig),
                    "groupDistributions": {},
                    "representationRatios": {},
                    "underrepresentedGroups": [],
                    "overrepresentedGroups": [],
                }
            )
        agr = label_quality.get("annotatorAgreement") or {}
        if agr.get("available"):
            for block in agr.get("byGroup") or []:
                agr_rows = block.get("groups") or []
                if len(agr_rows) < 2:
                    continue
                hi = max(agr_rows, key=lambda r_: r_["agreement"])
                lo = min(agr_rows, key=lambda r_: r_["agreement"])
                gap = float(hi["agreement"]) - float(lo["agreement"])
                if gap < _LQ_AGREEMENT_GAP:
                    continue
                bias_types.append("annotator_disagreement")
                bias_findings.append(
                    {
                        "type": "annotator_disagreement",
                        "plain": (
                            "The annotators disagree more about "
                            f"'{lo['group']}' than about '{hi['group']}' on "
                            f"'{block.get('attribute')}' (agreement "
                            f"{lo['agreement'] * 100:.0f}% vs "
                            f"{hi['agreement'] * 100:.0f}%; documented gap "
                            f"threshold {_LQ_AGREEMENT_GAP:.0%}). The ground "
                            "truth is least reliable exactly for that "
                            "group, so every label-dependent metric reads "
                            "it with extra uncertainty."
                        ),
                        "severity": "warn",
                        "attribute": block.get("attribute"),
                        "statisticalTest": None,
                        "groupDistributions": {},
                        "representationRatios": {},
                        "underrepresentedGroups": [],
                        "overrepresentedGroups": [],
                    }
                )

    _safe(_lq_findings, None, degraded=degradations, label="label_quality_findings")

    # 4f. G-25: individual fairness. Two label-free reads on the binary
    #     decision itself:
    #       (a) kNN consistency (Zemel et al. 2013): the share of each
    #           individual's k=10 nearest neighbours ON THE NON-PROTECTED
    #           FEATURES that receive the same decision. Low consistency
    #           means similar people are treated differently, a harm that
    #           group-level parity metrics are structurally blind to.
    #       (b) Theil-T inequality decomposition of the favorable-outcome
    #           distribution into between-group vs within-group shares
    #           per protected attribute (pure numpy).
    #     The COUNTERFACTUAL half of individual fairness (re-score every
    #     row with the protected attribute flipped through predict_fn)
    #     needs a live model; it is NOT faked here. When the inputs carry
    #     a model / endpoint config it is named as notCovered in scope.
    def _individual_fairness() -> Dict[str, Any]:
        n_rows = int(min(len(work), len(y_pred)))
        if n_rows < _IF_MIN_ROWS:
            return {
                "available": False,
                "notApplicable": True,
                "reason": (
                    "Individual-fairness reading needs at least "
                    f"{_IF_MIN_ROWS} rows for a stable k-nearest-"
                    "neighbour agreement; this artifact has "
                    f"{n_rows}."
                ),
            }
        # Similarity space: non-protected, non-decision features only.
        # PII / oracle / model-output columns are excluded (they must
        # not define who counts as 'similar'); unknown-role columns stay.
        never = {str(c).lower() for c in usable}
        never.add(str(pred_col).lower())
        if label_col:
            never.add(str(label_col).lower())
        for key in ("pii_leakage", "oracle_columns", "model_output"):
            never.update(str(c).lower() for c in (schema.get(key) or []))
        feat_spec: List[Tuple[str, str]] = []
        for c in work.columns:
            if str(c).lower() in never:
                continue
            s = work[c]
            if (
                pd.api.types.is_numeric_dtype(s)
                and s.notna().mean() >= 0.5
                and s.nunique(dropna=True) > 1
            ):
                feat_spec.append(("num", str(c)))
            else:
                nu = int(s.astype("string").nunique(dropna=True))
                if 2 <= nu <= 20:  # genuine category, not text or an id
                    feat_spec.append(("cat", str(c)))
        if not feat_spec:
            return {
                "available": False,
                "notApplicable": True,
                "reason": (
                    "No usable non-protected feature columns to "
                    "define similarity on, so the kNN consistency "
                    "read cannot run."
                ),
            }
        idx = np.arange(n_rows)
        sampled = False
        if n_rows > _IF_SAMPLE_CAP:
            idx = np.sort(np.random.default_rng(0).choice(n_rows, _IF_SAMPLE_CAP, replace=False))
            sampled = True
        cols: List[np.ndarray] = []
        for kind, c in feat_spec:
            if kind == "num":
                v = pd.to_numeric(work[c], errors="coerce").to_numpy(dtype=float)[:n_rows][idx]
                finite = v[np.isfinite(v)]
                fill = float(np.median(finite)) if finite.size else 0.0
                v = np.where(np.isfinite(v), v, fill)
                sd = float(v.std())
                cols.append(((v - float(v.mean())) / sd) if sd > 0 else np.zeros_like(v))
            else:
                sv_ = work[c].astype("string").fillna("missing").to_numpy()[:n_rows][idx]
                for lev in sorted({str(x) for x in sv_}):
                    cols.append((sv_ == lev).astype(float))
        X = np.column_stack(cols)
        yp_ = np.asarray(y_pred[:n_rows], dtype=float)[idx]
        m = int(len(idx))
        k = min(_IF_K, m - 1)
        sq = np.einsum("ij,ij->i", X, X)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.fill_diagonal(d2, np.inf)
        nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        agree = (yp_[nn] == yp_[:, None]).mean(axis=1)
        consistency = float(agree.mean())
        per_group: Dict[str, Dict[str, float]] = {}
        for attr in usable:
            gv = work[attr].astype("string").fillna("missing").to_numpy()[:n_rows][idx]
            row: Dict[str, float] = {}
            for lab in pd.unique(gv):
                mask = gv == lab
                if int(mask.sum()) >= 5:
                    row[str(lab)] = float(agree[mask].mean())
            if row:
                per_group[attr] = row
        # (b) Theil-T decomposition on the FULL sample (descriptive).
        # For a binary favorable outcome the total Theil-T collapses to
        # ln(1/mu); between-group = sum_g (n_g/n)(mu_g/mu) ln(mu_g/mu).
        y_all = np.asarray(y_pred[:n_rows], dtype=float)
        mu = float(y_all.mean())
        theil_rows: List[Dict[str, Any]] = []
        if mu > 0.0:
            t_total = float(np.log(1.0 / mu)) if mu < 1.0 else 0.0
            for attr in usable:
                gv = work[attr].astype("string").fillna("missing").to_numpy()[:n_rows]
                t_between = 0.0
                for lab in pd.unique(gv):
                    mask = gv == lab
                    n_g = int(mask.sum())
                    if n_g == 0:
                        continue
                    mu_g = float(y_all[mask].mean())
                    if mu_g > 0:
                        t_between += (n_g / float(n_rows)) * (mu_g / mu) * float(np.log(mu_g / mu))
                t_between = min(max(0.0, float(t_between)), t_total)
                theil_rows.append(
                    {
                        "attribute": attr,
                        "total": t_total,
                        "betweenGroups": t_between,
                        "withinGroups": t_total - t_between,
                        "betweenShare": ((t_between / t_total) if t_total > 0 else 0.0),
                    }
                )
        if theil_rows:
            theil = dict(max(theil_rows, key=lambda r_: r_["betweenShare"]))
            theil["perAttribute"] = theil_rows
        else:
            theil = {
                "attribute": None,
                "total": 0.0,
                "betweenGroups": 0.0,
                "withinGroups": 0.0,
                "betweenShare": 0.0,
                "perAttribute": [],
                "note": (
                    "No favorable outcomes in the sample, so "
                    "there is no outcome distribution to "
                    "decompose."
                ),
            }
        plain = (
            f"On the non-protected features, {consistency * 100:.0f}% of "
            f"each person's {k} most similar peers receive the same "
            f"decision (consistency {consistency:.2f}; documented floor "
            f"{_IF_CONSISTENCY_FLOOR:.2f})."
        )
        if theil.get("attribute"):
            plain += (
                f" {float(theil['betweenShare']) * 100:.0f}% of the "
                "inequality in who receives the favorable outcome lies "
                f"between '{theil['attribute']}' groups (documented "
                f"ceiling {_IF_BETWEEN_SHARE_CEILING:.0%})."
            )
        return {
            "available": True,
            "consistencyScore": consistency,
            "k": int(k),
            "nEvaluated": m,
            "sampledForKnn": bool(sampled),
            "featuresUsed": [c for _kind, c in feat_spec],
            "perGroupConsistency": per_group,
            "theil": theil,
            "thresholds": {
                "consistencyFloor": _IF_CONSISTENCY_FLOOR,
                "betweenShareCeiling": _IF_BETWEEN_SHARE_CEILING,
            },
            # G-44: individual-basis reads carry their own stance tag.
            "stance": {
                "basis": "individual",
                "worldview": "consistency",
                "note": (
                    "Treat similar individuals similarly (Dwork et "
                    "al. 2012); label-free, so it neither assumes "
                    "equal group outcomes nor trusts the recorded "
                    "labels."
                ),
            },
            "method": (
                f"kNN consistency (k={k}, numeric features standardized, "
                "categoricals one-hot, protected / PII / oracle / "
                "decision columns excluded from the similarity space) "
                "plus a Theil-T between-vs-within-group decomposition of "
                "the favorable-outcome distribution per protected "
                "attribute"
            ),
            "plain": plain,
        }

    individual_fairness = _safe(
        _individual_fairness,
        {"available": False, "reason": "The individual-fairness stage could not be computed."},
        degraded=degradations,
        label="individual_fairness",
    )

    # G-25 remainder: the counterfactual flip pass ran consumer-side inside
    # scoring.py (the only place with predict_fn access) and travels on the
    # same inputs side-channel as scoring degradations. Attach it to the
    # individualFairness block so the counterfactual half lives next to the
    # kNN-consistency and Theil halves it completes.
    _cf = inputs.get("_counterfactual_flips")
    if isinstance(_cf, dict) and _cf:
        individual_fairness["counterfactual"] = _cf

    def _if_findings() -> None:
        if not individual_fairness.get("available"):
            return
        cs = individual_fairness.get("consistencyScore")
        if isinstance(cs, (int, float)) and cs < _IF_CONSISTENCY_FLOOR:
            worst_note = ""
            pg = individual_fairness.get("perGroupConsistency") or {}
            flat = [(f"{a}={g}", v) for a, row_ in pg.items() for g, v in row_.items()]
            if flat:
                wg = min(flat, key=lambda t_: t_[1])
                worst_note = f" It is lowest for {wg[0]} ({wg[1]:.2f})."
            bias_types.append("individual_fairness")
            bias_findings.append(
                {
                    "type": "individual_fairness",
                    "plain": (
                        "Similar individuals receive different decisions: "
                        f"only {cs * 100:.0f}% of each person's "
                        f"{individual_fairness.get('k')} nearest neighbours "
                        "on the non-protected features share their decision "
                        f"(consistency {cs:.2f}; documented floor "
                        f"{_IF_CONSISTENCY_FLOOR:.2f})."
                        + worst_note
                        + " Group-level parity metrics cannot see this harm."
                    ),
                    "severity": ("critical" if cs < _IF_CONSISTENCY_CRITICAL else "warn"),
                    "attribute": None,
                    "statisticalTest": None,
                    "groupDistributions": {},
                    "representationRatios": {},
                    "underrepresentedGroups": [],
                    "overrepresentedGroups": [],
                }
            )
        th = individual_fairness.get("theil") or {}
        share = th.get("betweenShare")
        if (
            isinstance(share, (int, float))
            and share > _IF_BETWEEN_SHARE_CEILING
            and th.get("attribute")
        ):
            bias_types.append("inequality_index")
            bias_findings.append(
                {
                    "type": "inequality_index",
                    "plain": (
                        f"{share * 100:.0f}% of the inequality in who "
                        "receives the favorable outcome lies BETWEEN "
                        f"'{th.get('attribute')}' groups rather than within "
                        "them (Theil decomposition: total "
                        f"{float(th.get('total') or 0.0):.3f}, between "
                        f"{float(th.get('betweenGroups') or 0.0):.3f}; "
                        f"documented ceiling {_IF_BETWEEN_SHARE_CEILING:.0%})."
                        " Group membership, not individual variation, is a "
                        "first-order driver of the outcome."
                    ),
                    "severity": ("critical" if share > _IF_BETWEEN_SHARE_CRITICAL else "warn"),
                    "attribute": th.get("attribute"),
                    "statisticalTest": None,
                    "groupDistributions": {},
                    "representationRatios": {},
                    "underrepresentedGroups": [],
                    "overrepresentedGroups": [],
                }
            )

    _safe(_if_findings, None, degraded=degradations, label="individual_fairness_findings")

    # G-25 remainder findings: a nonzero flip rate is direct evidence the
    # decision depends on the protected attribute (documented threshold
    # rationale at _CF_FLIP_CRITICAL). Runs on the attached counterfactual
    # block regardless of whether the kNN half was applicable.
    def _cf_findings() -> None:
        cf = individual_fairness.get("counterfactual") or {}
        if not cf.get("available"):
            return
        for row in cf.get("perAttribute") or []:
            rate = row.get("flipRate")
            if not isinstance(rate, (int, float)) or rate <= 0:
                continue
            attr = row.get("attribute")
            bias_types.append("counterfactual_flip")
            bias_findings.append(
                {
                    "type": "counterfactual_flip",
                    "plain": (
                        f"Flipping '{attr}' alone (every other value held "
                        f"identical) changes the model's decision for "
                        f"{rate * 100:.1f}% of rows. A counterfactually fair "
                        "model changes for none: this is direct evidence the "
                        "decision depends on the protected attribute, not "
                        "merely on correlated features."
                    ),
                    "severity": ("critical" if rate >= _CF_FLIP_CRITICAL else "warn"),
                    "attribute": attr,
                    "statisticalTest": None,
                    "groupDistributions": {},
                    "representationRatios": {},
                    "underrepresentedGroups": [],
                    "overrepresentedGroups": [],
                }
            )

    _safe(_cf_findings, None, degraded=degradations, label="counterfactual_flip_findings")

    # 4g. G-35: error-cohort analysis (Azure error-tree style, depth 2).
    #     Only meaningful against real outcomes: without ground truth it
    #     reports available: False, never a fabricated error rate.
    def _cohort_analysis() -> Dict[str, Any]:
        if not has_truth or y_true is None:
            return {
                "available": False,
                "reason": (
                    "Cohort error analysis compares decisions against real "
                    "outcomes; no ground-truth column was provided."
                ),
            }
        n_rows = int(min(len(work), len(y_pred), len(y_true)))
        if n_rows < 2 * _COHORT_MIN_N:
            return {
                "available": False,
                "reason": (
                    f"Cohort error analysis needs at least "
                    f"{2 * _COHORT_MIN_N} rows; this artifact has "
                    f"{n_rows}."
                ),
            }
        err = (np.asarray(y_pred[:n_rows]) != np.asarray(y_true[:n_rows])).astype(float)
        baseline = float(err.mean())
        skip = {str(pred_col).lower()}
        if label_col:
            skip.add(str(label_col).lower())
        skip.update(str(c).lower() for c in exclude_cols)
        cand: List[Tuple[int, str]] = []
        for c in work.columns:
            if str(c).lower() in skip:
                continue
            s = work[c].iloc[:n_rows].astype("string").fillna("missing")
            nu = int(s.nunique())
            if 2 <= nu <= _COHORT_MAX_LEVELS:
                cand.append((nu, str(c)))
        cand.sort()
        features = [c for _nu, c in cand[:_COHORT_MAX_FEATURES]]
        if len(features) < 2:
            return {
                "available": False,
                "reason": (
                    "The cohort scan needs at least two categorical "
                    f"features with {_COHORT_MAX_LEVELS} or fewer levels; "
                    "this artifact has fewer."
                ),
            }
        from itertools import combinations as _combos

        arrays = {
            c: work[c].iloc[:n_rows].astype("string").fillna("missing").to_numpy() for c in features
        }
        found: List[Dict[str, Any]] = []
        seen_total = 0
        for f1, f2 in _combos(features, 2):
            a1, a2 = arrays[f1], arrays[f2]
            for l1 in pd.unique(a1):
                m1 = a1 == l1
                if int(m1.sum()) < _COHORT_MIN_N:
                    continue
                for l2 in pd.unique(a2):
                    m_ = m1 & (a2 == l2)
                    n_c = int(m_.sum())
                    if n_c < _COHORT_MIN_N:
                        continue
                    seen_total += 1
                    er = float(err[m_].mean())
                    found.append(
                        {
                            "cohort": f"{f1}={l1} & {f2}={l2}",
                            "features": [str(f1), str(f2)],
                            "n": n_c,
                            "errorRate": er,
                            "baselineErrorRate": baseline,
                            "lift": ((er / baseline) if baseline > 0 else None),
                        }
                    )
        found.sort(key=lambda d_: (d_["errorRate"], d_["n"]), reverse=True)
        found = found[:_COHORT_MAX_COHORTS]
        return {
            "available": True,
            "baselineErrorRate": baseline,
            "nCohortsSeen": int(seen_total),
            "nCohortsRetained": len(found),
            "worst": found[:5],
            "method": (
                "depth-2 error-tree scan: level crossings of feature "
                f"pairs (categorical features with <= "
                f"{_COHORT_MAX_LEVELS} levels, cohorts with n >= "
                f"{_COHORT_MIN_N}, worst {_COHORT_MAX_COHORTS} "
                "retained), misclassification rate vs the baseline "
                "error rate"
            ),
        }

    cohorts = _safe(
        _cohort_analysis,
        {"available": False, "reason": "The cohort-analysis stage could not be computed."},
        degraded=degradations,
        label="cohort_analysis",
    )

    def _cohort_findings() -> None:
        if not cohorts.get("available"):
            return
        worst_c = (cohorts.get("worst") or [None])[0]
        base = float(cohorts.get("baselineErrorRate") or 0.0)
        if not worst_c or base <= 0:
            return
        lift = worst_c.get("lift")
        if (
            not isinstance(lift, (int, float))
            or lift < _COHORT_LIFT_WARN
            or int(worst_c.get("n") or 0) < _COHORT_MIN_N
        ):
            return
        # Multiplicity gate (audit fix coh-1): the worst cohort is the max
        # over up to hundreds of depth-2 cohorts, so a raw lift threshold
        # with no test flags a false pocket on clean data. Binomial-test
        # the worst cohort's error count against the baseline and correct
        # for the number of cohorts SCANNED (Bonferroni on the max
        # statistic); only fire when the excess survives the correction.
        n_c = int(worst_c.get("n") or 0)
        errors = int(round(float(worst_c.get("errorRate") or 0.0) * n_c))
        n_seen = int(cohorts.get("nCohortsSeen") or 1) or 1

        def _binom() -> float:
            from scipy.stats import binomtest

            return float(binomtest(errors, n_c, min(1.0, base), alternative="greater").pvalue)

        p_raw = _safe(_binom, None, degraded=degradations, label="cohort_binom")
        if p_raw is None:
            return
        p_corr = min(1.0, p_raw * n_seen)
        if p_corr >= 0.05:
            return  # a max-of-many pocket that does not survive correction
        bias_types.append("cohort_error")
        bias_findings.append(
            {
                "type": "cohort_error_concentration",
                "plain": (
                    f"Errors concentrate in the cohort {worst_c['cohort']} "
                    f"(n={worst_c['n']}): "
                    f"{float(worst_c['errorRate']) * 100:.0f}% of its "
                    f"decisions are wrong versus {base * 100:.0f}% overall "
                    f"({float(lift):.1f}x; binomial p {p_corr:.3g} after "
                    f"correcting for {n_seen} cohorts scanned). A concentrated "
                    "error pocket like this localizes specification or proxy "
                    "bias: the model is systematically wrong for a describable "
                    "subpopulation, which per-attribute averages hide."
                ),
                "severity": ("critical" if lift >= _COHORT_LIFT_CRITICAL else "warn"),
                "attribute": None,
                "statisticalTest": {
                    "test": "binomial_vs_baseline",
                    "pValue": p_corr,
                    "correction": "bonferroni_cohorts_scanned",
                },
                "significant": True,
                "groupDistributions": {},
                "representationRatios": {},
                "underrepresentedGroups": [],
                "overrepresentedGroups": [],
            }
        )

    _safe(_cohort_findings, None, degraded=degradations, label="cohort_findings")

    # 5. causal overview
    def _causal():
        from vfairness.operations.pulse.causal_skeleton import (
            build_causal_skeleton,
            domain_dag_template,
        )

        c = build_causal_skeleton(work, usable, pred_col).to_dict()
        # Defensible domain-template DAG instantiated from the E2 column
        # roles (template > learned). Read-only here; editable in Navigator.
        c["template"] = domain_dag_template(domain, schema)
        return c

    _p(6, "Mapping causal structure and intersectional subgroups", 88)
    causal = _safe(
        _causal,
        {
            "nodes": [],
            "edges": [],
            "paths": [],
            "overview": "Causal overview unavailable.",
            "template": {"available": False},
        },
        degraded=degradations,
        label="causal",
    )

    # 6. fairness definition recommendation. G-19: harm_direction routes
    #    the headline emphasis (punitive -> predictive-parity family,
    #    assistive -> equal opportunity), scores_exposed adds calibration,
    #    and the measured base-rate divergence drives the honest
    #    impossibility acknowledgment in `tradeoff`.
    from vfairness.operations.pulse.recommend import (
        recommend_fairness_definition,
        recommend_interventions,
    )

    recommendation = _safe(
        lambda: recommend_fairness_definition(
            domain,
            jurisdiction,
            has_truth,
            harm_direction=pulse_options.get("harm_direction"),
            scores_exposed=bool(pulse_options.get("scores_exposed")),
            base_rates_differ=_regulatory.base_rates_differ(work, usable, y_true),
        ),
        {},
        degraded=degradations,
        label="recommended_definition",
    )

    # 7. interventions / controls
    interventions = _safe(
        lambda: recommend_interventions(bias_types),
        [],
        degraded=degradations,
        label="interventions",
    )

    # 7b. intersectional disparity (Navigator-grade dual-lens over the
    #     cross-product of the assessable attributes -- e.g. "young Chinese
    #     women"). Same engine the Navigator uses; never raises.
    # Predictions were already polarity-normalized at coercion time, so the
    # shared intersectional engine always receives positive_favorable
    # (passing the raw declaration through would flip a second time).
    intersectional = _safe(
        lambda: _intersectional(
            work, usable, y_pred, y_true, "positive_favorable", min_group=min_group
        ),
        {"skipped": True, "reason": "Intersectional analysis could not be computed on this data."},
        degraded=degradations,
        label="intersectional",
    )

    # 8. verdict: the WORST-SEVERITY assessable variable drives the headline
    #    (Kearns worst-CASE: subgroup fairness is the MAX violation over
    #    subgroups, not the largest-magnitude one -- Kearns et al. 2018).
    #    A row's tone is NOT monotonic in the absolute gap: _disparity_tone
    #    escalates a reliable group selected 0% of the time (total exclusion,
    #    four-fifths 0) to 'critical' at a SMALL absolute gap, while a larger
    #    but benign gap stays 'warn'. Selecting ranked[0]-by-gap therefore
    #    hid a critical total-exclusion behind a bigger warn gap and rendered
    #    the reassuring "Deployable only with monitoring" headline. So the
    #    overall tone is the MAX tone across ALL assessable attributes, and
    #    the narrative row is the most-severe attribute (largest gap only as
    #    a within-tone tiebreak). `ranked` stays gap-sorted for the display
    #    list ONLY.
    assessable = [v for v in per_variable if v.get("assessable")]
    ranked = sorted(assessable, key=lambda v: v.get("gap", 0.0), reverse=True)
    if not ranked:
        worst = None
        tone = "warn"
        headline = "Could not compute a reliable fairness verdict on this data."
        summary = "No protected variable had enough grouped data to assess."
    else:
        tone = _worst_tone(*[str(v.get("tone") or "pass") for v in assessable])
        worst = max(
            assessable,
            key=lambda v: (_tone_rank(str(v.get("tone") or "pass")), v.get("gap", 0.0)),
        )
        wl, rl = worst["worstGroup"], worst["referenceGroup"]
        gp = round(worst["gap"] * 100)
        if tone == "critical":
            headline = f'Not fit to deploy as-is: material disparity for "{wl}".'
        elif tone == "warn":
            headline = f'Deployable only with monitoring: watch-level gap for "{wl}".'
        else:
            headline = "Within fairness budget on the assessed attributes."
        sig = "a statistically significant" if worst["significant"] else "an unconfirmed"
        # Guard the CI clause like the non-binary path does: small groups can
        # yield ciLow/ciHigh = None, and formatting None crashed the whole run.
        ci_txt = ""
        if isinstance(worst.get("ciLow"), (int, float)) and isinstance(
            worst.get("ciHigh"), (int, float)
        ):
            ci_txt = f"; 95% CI {worst['ciLow'] * 100:.0f} to {worst['ciHigh'] * 100:.0f} pts"
        ff_txt = ""
        if isinstance(worst.get("fourFifthsRatio"), (int, float)):
            ff_txt = f"; four-fifths ratio {worst['fourFifthsRatio']:.2f}"
        summary = (
            f'Most severe disparity is in "{worst["attribute"]}": "{wl}" is selected '
            f'{gp} points less than "{rl}" ({sig} difference{ci_txt}{ff_txt}).'
        )

    verdict = {
        "tone": tone,
        "headline": headline,
        "summary": summary,
        "ranked": [
            {
                "attribute": v["attribute"],
                "worstGroup": v["worstGroup"],
                "gap": v["gap"],
                "tone": v["tone"],
                "significant": v["significant"],
            }
            for v in ranked
        ],
    }

    _pv0 = next((v for v in per_variable if v.get("assessable")), None)
    # The trade-off sweep must analyze the attribute the headline is about --
    # the WORST-severity attribute selected above -- not the largest-gap row
    # (`ranked[0]`, which can be a benign warn gap hiding a critical exclusion)
    # nor whichever usable attribute happened to be listed first.
    pareto_attr = (
        worst.get("attribute")
        if worst is not None and worst.get("attribute") in usable
        else usable[0]
    )
    _pvw = worst if worst is not None and worst.get("attribute") == pareto_attr else _pv0

    # Mitigation-sweep Pareto: real post-processing strategies (global +
    # group thresholds for DP and equal opportunity), Pareto frontier via
    # the shared vfairness helper. Falls back to the threshold-only sweep
    # when no continuous score exists.
    def _mitigation():
        from vfairness.post_processing import mitigation_pareto

        score_col = _pick(
            df, ("probability", "proba", "prob", "score", "y_score", "y_prob", "confidence")
        )
        if not score_col or score_col not in work.columns:
            raise ValueError("no continuous score")
        sp = pd.to_numeric(work[score_col], errors="coerce")
        if sp.notna().mean() < 0.5 or sp.nunique() < 5:
            raise ValueError("score not continuous")
        prim = pareto_attr
        sens = work[prim].astype("string").fillna("missing").to_numpy()
        n = min(len(sens), len(sp))
        return mitigation_pareto(
            (y_true[:n] if y_true is not None else None), sp.to_numpy()[:n], sens[:n]
        )

    if polarity_flipped:
        # Both sweeps threshold the RAW score column; with predictions
        # inverted for an adverse-positive system the frontier would be
        # computed against the wrong direction. Skip honestly rather than
        # emit a misleading trade-off (a polarity-aware sweep is a planned
        # extension).
        pareto = {
            "available": False,
            "reason": (
                "The accuracy / fairness trade-off sweep is not yet "
                "polarity-aware, so it is skipped for systems where the "
                "positive prediction is the adverse outcome."
            ),
        }
    else:
        pareto = _safe(_mitigation, None)
        if pareto and pareto.get("available") and not pareto.get("attribute"):
            # Name the axis the sweep ran over so the UI and the report can
            # say which attribute the trade-off applies to.
            pareto["attribute"] = pareto_attr
        if not pareto or not pareto.get("available"):
            pareto = _safe(
                lambda: _pareto(
                    work,
                    df,
                    pareto_attr,
                    _pvw.get("referenceGroup") if _pvw else None,
                    _pvw.get("worstGroup") if _pvw else None,
                    label_col,
                    has_truth,
                    y_pred,
                ),
                {
                    "available": False,
                    "reason": "The accuracy / fairness trade-off could not be computed.",
                },
                degraded=degradations,
                label="pareto",
            )

    # 9. assurance-audit verdict -- the structured, audit-grade opinion
    #    (Unqualified/Qualified/Adverse/Disclaimer) + routed findings &
    #    recommendations + jurisdiction overlay + audit trail. Same engine
    #    the Navigator pre-fill consumes. Never raises.
    def _assurance():
        from vfairness.operations.reporting import build_assurance_verdict

        return build_assurance_verdict(
            schema=schema,
            per_variable=per_variable,
            metrics=metrics,
            bias=bias_findings,
            proxies=proxies,
            statistical=statistical,
            intersectional=intersectional,
            disparity_matrix=disparity_matrix,
            recommended=recommendation,
            domain=domain,
            jurisdiction=jurisdiction,
            has_truth=has_truth,
            artifact_hash=str(inputs.get("artifact_hash") or inputs.get("artifactHash") or "")
            or None,
        )

    _p(7, "Composing the assurance verdict and recommendations", 96)
    assurance = _safe(
        _assurance,
        {
            "overall": "Disclaimer",
            "blocksDeployment": False,
            "oneLineVerdict": "The assurance verdict could not be assembled.",
            "findings": [],
            "recommendations": [],
            "metricsDeferred": {},
            "auditTrail": {},
        },
        degraded=degradations,
        label="assurance",
    )

    # A swallowed failure in any load-bearing stage means the analysis is
    # INCOMPLETE -- it must never present as a clean pass. Downgrade an
    # otherwise-clean opinion to a Disclaimer (insufficient basis), bump
    # the plain verdict tone to at least 'warn', and surface what failed.
    if degradations:
        try:
            if isinstance(verdict, dict):
                verdict["tone"] = _worst_tone(str(verdict.get("tone") or "pass"), "warn")
            if isinstance(assurance, dict):
                ov = str(assurance.get("overall") or "")
                if ov in ("", "Unqualified"):
                    assurance["overall"] = "Disclaimer"
                    assurance["oneLineVerdict"] = (
                        "Incomplete analysis: "
                        f"{len(degradations)} stage(s) could not be computed, "
                        "so a clean opinion cannot be issued. See degradations."
                    )
                else:
                    assurance["oneLineVerdict"] = (
                        str(assurance.get("oneLineVerdict") or "")
                        + " (Note: parts of the analysis could not be "
                        "computed; treat the verdict as a lower bound.)"
                    )
        except Exception:  # noqa: BLE001 -- post-processing is never fatal
            pass

    # 10. regulatory packaging (Phase 4: G-13 admissibility, G-18 LL144 +
    #     Art. 10 exports, G-24 ISO 24027 crosswalk, G-31 historical
    #     context, G-32 targets + reference disclosure, G-34 recheck).
    #     Bulk logic lives in regulatory.py; every stage is _safe so a
    #     collapse degrades and is recorded, never crashes the run.
    legal_admissibility = _safe(
        lambda: _regulatory.legal_admissibility(list(df.columns), domain, jurisdiction),
        _regulatory.uncovered_admissibility(jurisdiction, "legal admissibility stage collapsed"),
        degraded=degradations,
        label="legal_admissibility",
    )

    def _reg_score():
        # Continuous score for the LL144 above-median scoring rate; None
        # when no usable score column exists (the export then reports
        # null, never an invented rate).
        sc = _pick(df, ("probability", "proba", "prob", "score", "y_score", "y_prob", "confidence"))
        if not sc or sc not in work.columns:
            return None
        sp = pd.to_numeric(work[sc], errors="coerce")
        if sp.notna().mean() < 0.5 or sp.nunique() < 5:
            return None
        return sp.to_numpy(dtype=float)

    _score_arr = _safe(_reg_score, None)

    regulatory_exports = _safe(
        lambda: _regulatory.build_regulatory_exports(
            work=work,
            usable=usable,
            y_pred=y_pred,
            df=df,
            requested=requested,
            domain=domain,
            jurisdiction=jurisdiction,
            score=_score_arr,
            artifact_label=str(inputs.get("artifact_label") or inputs.get("artifactLabel") or ""),
            artifact_hash=str(inputs.get("artifact_hash") or inputs.get("artifactHash") or "")
            or None,
            schema=schema,
            data_quality=data_quality,
            verdict=verdict,
            bias_findings=bias_findings,
            interventions=interventions,
            has_truth=has_truth,
            calibration_available=bool(calibration.get("available")),
            output_type="binary",
        ),
        _regulatory.empty_exports("The regulatory export stage could not be computed."),
        degraded=degradations,
        label="regulatory_exports",
    )

    # BGL5 A-operations-3-b, 2026-09-29. This read
    # `bool((regulatory_exports.get("ll144") or {}).get("applicable"))`, and that
    # `bool()` is the flattening build_recheck's three states exist to survive.
    # `regulatory_exports` falls back to `_regulatory.empty_exports(...)` a few
    # lines up when the export stage collapses, and that block carries
    # `applicable=False` meaning "not evaluated", so the expression answered a
    # plain False: the statute screen ran and nothing matched. Measured before,
    # on the collapsed block this very function substitutes:
    #   _ll144_screen_state(collapsed)                         -> None
    #   bool((collapsed.get("ll144") or {}).get("applicable"))  -> False
    #   published recheck.basis -> "No annual audit statute matched this run, so
    #       the 6-month window is Pulse's default hygiene interval", warnings []
    # which is a legal finding nobody made, sealed into the payload at the
    # "recheck" key below. After: the basis reads "Whether an annual audit
    # statute applies could not be determined on this run ... this is NOT a
    # finding that none applies". The window is 6 months either way, so nothing
    # a caller schedules changes; what changes is that the payload stops
    # asserting the statute was tested. `_ll144_screen_state` is deliberately
    # module-private in regulatory.py (a public name adds a row to the measured
    # capability surface); a cross-module private helper is the house pattern.
    recheck = _safe(
        lambda: _regulatory.build_recheck(
            inputs, _regulatory._ll144_screen_state(regulatory_exports)
        ),
        {},
        degraded=degradations,
        label="recheck",
    )

    historical_context = _safe(
        lambda: _regulatory.historical_context(domain),
        {"available": False, "reason": "The historical-context stage could not be computed."},
        degraded=degradations,
        label="historical_context",
    )

    targets = _safe(
        lambda: _regulatory.build_targets(per_variable, pulse_options.get("target_ratio")),
        {
            "targetRatio": None,
            "perAttribute": [],
            "note": "The four-fifths target stage could not be computed.",
        },
        degraded=degradations,
        label="targets",
    )

    reference_groups = _safe(
        lambda: _regulatory.build_reference_groups(_reference_requests, per_variable),
        {},
        degraded=degradations,
        label="reference_groups",
    )

    scope = build_scope_block(
        "tabular",
        covered=[
            "Selection-rate disparity with bootstrap confidence intervals per protected attribute",
            "Full bias-taxonomy screen, proxy / leakage battery, and the "
            "statistical robustness battery (BH-FDR corrected)",
            f"Intersectional subgroups at n >= {min_group} (smaller cells "
            "excluded and listed transparently)",
        ]
        + (
            ["Ground-truth checks (equalized odds; calibration where a score exists)"]
            if has_truth
            else []
        )
        + (
            [
                "Per-group score calibration (Expected Calibration Error "
                "with the between-group gap reading)"
            ]
            if calibration.get("available")
            else []
        )
        + (
            [
                "Individual-fairness read: kNN decision consistency "
                f"(k={_IF_K} nearest neighbours on the non-protected "
                "features) and the Theil between-group inequality "
                "decomposition"
            ]
            if individual_fairness.get("available")
            else []
        )
        + (
            [
                "Counterfactual flip test: the supplied model re-scored "
                "with each protected model input flipped, reporting the "
                "share of decisions that change"
            ]
            if (individual_fairness.get("counterfactual") or {}).get("available")
            else []
        )
        + (
            [
                "Label-quality screen: recorded favorable-outcome base "
                "rate per group"
                + (
                    " and inter-annotator agreement by group"
                    if (label_quality.get("annotatorAgreement") or {}).get("available")
                    else ""
                )
            ]
            if label_quality.get("available")
            else []
        )
        + (
            [
                "Error-cohort scan over feature-pair cohorts (worst "
                "misclassification pockets vs the baseline error rate)"
            ]
            if cohorts.get("available")
            else []
        ),
        not_covered=[
            "Causal identification: the causal view is a structural sketch, correlational only",
            "Regression / multiclass / ranking outputs (binary decisions only in this release)",
            "Deployment, human-factors and systemic bias (NIST SP 1270): "
            "reachable only through the full assessment questionnaire",
        ]
        + (
            []
            if has_truth
            else [
                "Ground-truth-dependent checks (equalized odds, calibration): "
                "no real outcome column was provided",
            ]
        )
        + (
            [
                "Counterfactual individual fairness (re-scoring each row "
                "with the protected attribute flipped through the supplied "
                "model): "
                + str(
                    (individual_fairness.get("counterfactual") or {}).get("reason")
                    or "the flip pass did not run on this dispatch"
                ),
            ]
            if (
                (inputs.get("model_base64") or inputs.get("endpoint_url"))
                and not (individual_fairness.get("counterfactual") or {}).get("available")
            )
            else []
        ),
        power_note=(
            "Findings reflect this sample only; groups below the "
            f"n >= {min_group} floor were excluded from per-group "
            "readings and are listed in the transparency sections."
        ),
    )
    # One honest scope line when the regulatory packaging actually ran.
    if isinstance(regulatory_exports, dict) and regulatory_exports.get("legalContextAsOf"):
        scope["covered"].append(
            "Regulatory packaging: column-level legal admissibility, the "
            "NYC LL144 impact-ratio table and the EU AI Act Art. 10 "
            "section draft (each only where applicable), the ISO/IEC TR "
            "24027 metric crosswalk, and a dated recheck window"
        )

    return {
        "success": True,
        "data": _jsonify(
            {
                "sourceKind": "tabular",
                "degradations": degradations,
                "dataQuality": data_quality,
                "dataPreparation": data_preparation,
                "schema": schema,
                "verdict": verdict,
                "assurance": assurance,
                "scope": scope,
                # Disparity-severity framework actually applied to this run.
                # GUIs MUST surface this so an auditor sees which legal basis
                # the tone/severity decisions rest on (EEOC 4/5ths vs EU
                # disparate-impact vs generic heuristic, etc.).
                "legalFramework": build_legal_framework_block(domain, jurisdiction),
                "perVariable": per_variable,
                # T3 / T2.3 output: per-attribute disparity after stratifying
                # by each other protected attribute (most-deconfounded gap is
                # the "adjusted" reading). Rows with significant=False here have
                # marginal disparities that are largely confounded by another
                # trait; rows with significant=True have robust adjusted gaps.
                "adjustedDisparity": adjusted_disparity,
                "metrics": metrics,
                "bias": bias_findings,
                "temporalDrift": temporal_drift,
                "specification": specification,
                "disparityMatrix": disparity_matrix,
                "proxies": proxies,
                "statistical": statistical,
                # G-12: per-group score calibration (ECE) with between-group
                # gaps; significant gaps also appear in `bias` as
                # calibration_disparity findings.
                "calibration": calibration,
                # G-38: label base-rate skew + inter-annotator agreement by
                # group; breaches appear in `bias` as label_baserate_skew /
                # annotator_disagreement findings.
                "labelQuality": label_quality,
                # G-25: kNN decision consistency + Theil between/within-group
                # inequality decomposition (plus the counterfactual flip block
                # when the supplied model took a protected input); breaches
                # appear in `bias` as individual_fairness / inequality_index /
                # counterfactual_flip findings.
                "individualFairness": individual_fairness,
                # G-35: depth-2 error-cohort scan (only against real outcomes);
                # a 2x-baseline worst cohort appears in `bias` as
                # cohort_error_concentration.
                "cohorts": cohorts,
                "intersectional": intersectional,
                "pareto": pareto,
                "causal": causal,
                "recommendedDefinition": recommendation,
                "interventions": interventions,
                # Phase-4 regulatory packaging (G-13/G-18/G-24/G-31/G-32/G-34).
                "legalAdmissibility": legal_admissibility,
                "regulatoryExports": regulatory_exports,
                "recheck": recheck,
                "historicalContext": historical_context,
                "targets": targets,
                "referenceGroups": reference_groups,
                "nextStep": (
                    "Convert this Pulse into a tracked assessment, then apply the "
                    "recommended controls below before relying on the system."
                ),
            }
        ),
    }
