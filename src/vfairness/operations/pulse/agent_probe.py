"""Pulse agent / multi-agent trace path (Stage B4).

Given agent execution traces (a table with a group column plus a
tool/action/route column), REUSES the existing vfairness agent detectors
(ToolBiasAuditor, ActionBiasAnalyzer, DelegationRoutingAuditor) and maps
their output into the SAME assurance-verdict shape. One source of truth:
identical engine to the Navigator's vfairness_agent_* steps. Never raises.

Statistics regime (Pulse close plan G-09 + G-33):

- ALL adequately sampled groups are compared: every pair when there are
  at most 5 groups, one-vs-rest per group beyond that (not just the two
  most-populous groups).
- An omnibus chi-square on the full group x tool contingency table gates
  every tool/action finding, and p-values are corrected family-wise with
  Benjamini-Hochberg across ALL comparisons. This guards the
  AgentFairBench max-gap inflation problem: with many groups the largest
  pairwise gap grows even under the null, so no finding may fire on a
  single uncorrected pairwise test.
- Effect size per comparison is Cramér's V and severity derives from it
  (never hardcoded); a bootstrap CI (1000 resamples, seed 7) is reported
  on the largest per-tool selection-rate difference.
- Sample adequacy is a contract field (``agent.sampleAdequacy``), not a
  Python warning; groups under the floor are excluded and named, and
  when nothing is assessable the verdict is an honest Disclaimer.
- Trajectory shape (step counts / outcome rates) is compared with
  permutation tests (1000 permutations, seed 7) + BH within that family
  when such columns exist (``agent.trajectory``).
- Delegation routing (``agent.delegation``) runs DelegationRoutingAuditor
  across groups when a route/delegation column distinct from the action
  column exists.

Phase 6 additions (all additive to the result contract, never gating):

- G-10 (``agent.ingestion``): when the frame is a raw OpenTelemetry
  GenAI span export or a Langfuse-style export rather than the
  per-episode table, it is flattened via
  ``vfairness.operations.pulse.traces`` and the ingestion form is
  disclosed ({form: per_episode | otel_spans | langfuse, notes}).
- G-21 (``agent.amplification`` / ``agent.groupthink``): when the traces
  carry per-agent output columns AND a system-level output column, the
  existing EmergentBiasDetector measures system-vs-component
  amplification across groups; GroupthinkDetector runs when two or more
  per-agent opinion columns plus an ordering column exist (its API takes
  per-round outputs, so rounds are equal-count ordered windows). Honest
  notApplicable blocks otherwise.
- G-37 (``agent.temporalDynamics``): when the traces carry a
  timestamp/ordering column, the per-window tool-selection-rate gap
  between the two most-populous adequate groups feeds the existing
  TemporalTracker (Mann-Kendall feedback loop, CUSUM, EWMA); drift in
  the disparity raises a finding. Honest notApplicable without a time
  column. The tracker's own API is reused, never reimplemented.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics._statistics import detectability

ALPHA = 0.05
SAMPLE_FLOOR = 20  # minimum episodes per group to be assessable

#: The tool name recorded for an episode whose trace shows no tool call. Stated
#: as text so it is counted, compared and printed like any other action, and so
#: it reads the same on every pandas version.
NO_ACTION_LABEL = "(no tool call)"
# Minimum cross-group-descriptor-matched episodes before the term-level
# memory screen fires (a lone word-boundary hit is not contamination).
_MEM_TERM_MIN_EPISODES = 3
MAX_GROUPS_ALL_PAIRS = 5
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
RESAMPLE_SEED = 7
REST_LABEL = "rest_of_groups"
MAX_EVIDENCE_ROWS = 5  # raw trace rows per side of a comparison (G-20)

_TRAJECTORY_STEP_COLS = ("steps", "n_steps", "turns", "episode_length")
_TRAJECTORY_OUTCOME_COLS = ("success", "resolved", "outcome", "escalated")
_DELEGATION_COLS = ("route", "delegate", "delegated_to", "handoff", "assigned_to")

# G-21: multi-agent depth column vocabulary (conservative on purpose;
# a false "system output" detection would fabricate an amplification
# reading, so only explicitly named columns qualify).
_SYSTEM_OUTPUT_COLS = (
    "system_output",
    "system_score",
    "system_decision",
    "final_output",
    "final_score",
    "final_decision",
)
_COMPONENT_PREFIXES = ("agent_", "component_")
_COMPONENT_SUFFIXES = ("_output", "_score", "_decision", "_opinion")
_AMPLIFICATION_NA_REASON = "multi-agent depth needs per-agent AND system outputs in the traces"

# G-37: ordering/timestamp column vocabulary and windowing bounds.
_TIME_COLS = ("timestamp", "time", "ts", "created_at", "step_index")
# Marks a default that was published because the detector RAISED, as opposed to a
# could-not-check the detector returned itself. It never reaches the payload: the
# section copies named keys out of these dicts rather than serialising them whole, and
# what it publishes instead is the `detectorsFailed` list.
_DETECTOR_FAILED = "_probe_detector_failed"

_MIN_TEMPORAL_WINDOWS = 3
_MAX_TEMPORAL_WINDOWS = 10
_ROWS_PER_WINDOW = 20
_MAX_GROUPTHINK_ROUNDS = 6

_INGESTION_PER_EPISODE = {
    "form": "per_episode",
    "notes": (
        "Per-episode trace table ingested as supplied (one row "
        "per episode; no span flattening needed)."
    ),
}


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _quiet(fn, default):
    """_safe plus warning suppression: sample-size inadequacy is reported
    through the agent.sampleAdequacy contract field, never as a Python
    warning leaking out of the probe."""

    def wrapped():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn()

    return _safe(wrapped, default)


def _quiet_recorded(fn, default) -> Tuple[Any, List[str]]:
    """_quiet that KEEPS what it suppresses. Same contract towards the
    caller (no Python warning escapes the probe), but the detector's own
    warnings come back as strings instead of being discarded.

    A detector warns exactly when it could not measure something, so
    swallowing the warning next to a `bool(None)` verdict silenced both
    channels at once: the structured answer said False and the human
    readable one said nothing. Whatever is captured here is surfaced on
    the section contract as ``detectorNotes``."""

    def wrapped() -> Tuple[Any, List[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = fn()
        # Category first (a bare "invalid value encountered in scalar
        # divide" does not say who said it), and deduplicated in order: a
        # numpy warning fires once per element pair and repetition carries
        # no information the first copy did not.
        return value, list(dict.fromkeys(f"{w.category.__name__}: {w.message}" for w in caught))

    return _safe(wrapped, (default, []))


def _measured_verdict(value: Optional[bool], measured: bool) -> Optional[bool]:
    """Three states, never two. ``bool(None)`` is False, and False on a
    fairness verdict reads as "checked, nothing found", which is the one
    thing a could-not-check must never be collapsed into. A verdict is
    None when the detector reports None AND when the quantity it rests on
    is not finite: a comparison against nan is not a measurement either."""
    if value is None or not measured:
        return None
    return bool(value)


def _finite_or_none(x: Any) -> Optional[float]:
    """NaN/Inf -> None, the coercion the orchestrator applies at its JSON
    boundary (``_jsonify``). An unmeasured quantity is reported as null,
    never as a number a reader would compare, rank or format.

    READINESS-6, 2026-09-10. The body was ``math.isfinite(x)`` on a parameter
    annotated ``float``, so anything else RAISED: None, a string and a list each
    gave TypeError rather than the null this function exists to produce. An
    annotation is not a check, and the orchestrator's JSON boundary is exactly
    where a value of the wrong shape arrives.

    The full canonical rule is right HERE, unlike in the rendering adapters. Its
    inputs are statistics this module just computed, not rows deserialised from
    JSON or CSV, so a numeric string is not a serialised measurement, it is a
    value of the wrong shape and nulling it is the honest answer. The rendering
    coercers accept "0.5" for the opposite and equally correct reason; the
    difference is pinned in tests/test_readiness6_flags.py.
    """
    return float(x) if is_measured(x) else None


def _bh_adjust(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg step-up adjusted p-values (pure numpy)."""
    p = np.asarray(list(pvalues), dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    scaled = p[order] * m / (np.arange(m) + 1.0)
    # Enforce monotonicity from the largest rank down (step-up).
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(x) for x in out]


def _cramers_v(table) -> float:
    """Cramér's V from a contingency table (uncorrected chi-square, so a
    perfect association yields V = 1.0).

    THREE STATES, never two. V = sqrt(chi2 / (n * k)) with k = min(r, c) - 1,
    so a table whose rows or columns collapse to a single observed level has
    k = 0 and the statistic is 0/0: UNDEFINED, not zero. Returning 0.0 there
    published "no association" for a pair where no association is definable.
    Measured 2026-09-17 at the public entry: run_pulse(source_kind="agent")
    on two groups of 30 episodes that only ever called one tool produced a
    2x1 contingency table and reported cramersV 0.0 with severity "info", a
    clean bill of health for a tool CHOICE that was never made. The empty and
    1xN tables behaved the same way and warned about nothing.

    nan is returned instead, with a UserWarning naming the shape and the
    reason. Every caller must branch on math.isfinite before grading,
    thresholding or formatting it: nan >= 0.5 is False, so an unguarded band
    test silently lands on the weakest word the scale has.
    """
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or t.size == 0:
        warnings.warn(
            f"Cramer's V is undefined for a table of shape {tuple(t.shape)}: "
            f"a contingency table needs two dimensions and at least one cell. "
            f"Returning nan (could not check), never 0.0, which would read as "
            f"a measured absence of association.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    supplied = tuple(int(d) for d in t.shape)
    t = t[t.sum(axis=1) > 0, :]
    if t.size:
        t = t[:, t.sum(axis=0) > 0]
    if t.ndim != 2 or t.shape[0] < 2 or t.shape[1] < 2:
        warnings.warn(
            f"Cramer's V is undefined for the {supplied[0]}x{supplied[1]} table "
            f"supplied: only {int(t.shape[0])}x{int(t.shape[1])} rows and "
            f"columns carry any observations, so k = min(rows, columns) - 1 is "
            f"0 and V = sqrt(chi2 / (n * 0)). With fewer than two observed "
            f"levels on a side there is no association to measure. Returning "
            f"nan (could not check), never 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    n = float(t.sum())
    k = min(t.shape) - 1
    if n <= 0 or k <= 0:
        warnings.warn(
            f"Cramer's V is undefined: the table totals {n} observations over "
            f"k = {k}. Returning nan (could not check), never 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    chi2 = float(stats.chi2_contingency(t, correction=False)[0])
    return float(np.sqrt(chi2 / (n * k)))


def _severity_from_v(v: Optional[float]) -> Optional[str]:
    """Severity derives from the Cramér's V effect size, never hardcoded:
    >= 0.5 critical, >= 0.3 warn, else info.

    THREE STATES, exactly like its Cohen's d sibling below. ``None`` when no
    effect size was measured (``None`` or a non-finite V), because every band
    test against nan is False and the function fell through to "info": the
    weakest word on the scale, handed out for an effect nobody measured. A
    caller that needs a word must take it from something it DID measure, and
    must never read this ``None`` as a small effect.
    """
    if v is None or not math.isfinite(v):
        return None
    if v >= 0.5:
        return "critical"
    if v >= 0.3:
        return "warn"
    return "info"


def _v_clause(comp: Dict[str, Any]) -> str:
    """The Cramér's V phrase for a finding sentence.

    Never formats an unmeasured effect size: ``f"{None:.2f}"`` raises, and
    ``_probe`` is wrapped in ``_safe``, so one TypeError here would swallow
    the WHOLE probe into a Disclaimer and delete every real finding in it.
    """
    if comp.get("cramersVMeasured") and comp.get("cramersV") is not None:
        return f"Cramér's V {float(comp['cramersV']):.2f}"
    return "Cramér's V undefined for this pair, so the effect size is unmeasured"


def _severity_from_d(d: Optional[float]) -> Optional[str]:
    """Severity for mean/rate gaps from |Cohen's d| (large/medium
    conventions): >= 0.8 critical, >= 0.5 warn, else info.

    THREE STATES. ``None`` when no effect size was measured (``None`` or a
    non-finite d), because ``abs(nan)`` is nan and every band test against
    it is False, so the function fell through to "info", rank 0, the
    weakest word the scale has. ActionBiasAnalyzer._cohens_d was changed on
    2026-09-08 to return nan for perfect separation with zero within-group
    variance, which is the STRONGEST gap there is, and grading it "info"
    inverted it. Callers must decide from something they did measure (the
    p-value, the raw mean gap, whether the groups are actually separated)
    and must never read this ``None`` as a small effect."""
    if d is None or not math.isfinite(d):
        return None
    a = abs(d)
    if a >= 0.8:
        return "critical"
    if a >= 0.5:
        return "warn"
    return "info"


def _bootstrap_rate_diff_ci(
    tools_a: Sequence[str], tools_b: Sequence[str], tool: str
) -> Tuple[float, float]:
    """95% bootstrap CI (1000 resamples, seed 7) on the selection-rate
    difference (rate_a minus rate_b) for one named tool. The tool is
    fixed at the observed argmax so the CI does not itself inherit the
    max-gap inflation it is meant to bound."""
    rng = np.random.default_rng(RESAMPLE_SEED)
    a = np.asarray(list(tools_a))
    b = np.asarray(list(tools_b))
    diffs = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        ra = float(np.mean(rng.choice(a, size=a.size, replace=True) == tool))
        rb = float(np.mean(rng.choice(b, size=b.size, replace=True) == tool))
        diffs[i] = ra - rb
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def _tool_p_floor(n_a: int, n_b: int, k: int) -> Optional[float]:
    """Smallest Fisher p the per-tool 2x2 can return at these margins.

    A tool's table is [[used_a, not_used_a], [used_b, not_used_b]], so its
    column margin is k, the number of times the tool was used AT ALL. Fisher's
    exact conditions on both margins, and with k tiny there are almost no
    tables to condition over: the floor is the p of putting every one of the k
    uses in a single arm. Measured 2026-09-10 at n_A = n_B = 60, that is 1.0 at
    k=1, 0.4958 at k=2, 0.2437 at k=3, 0.1187 at k=4 and 0.0573 at k=5, so a
    tool used five or fewer times in the whole trace set has NO power against
    alpha 0.05, whatever the demographics of who got it.

    Returns None when the floor cannot be computed (could-not-check).
    """
    try:
        n_a, n_b, k = int(n_a), int(n_b), int(k)
    except (TypeError, ValueError):
        return None
    if n_a <= 0 or n_b <= 0 or k <= 0:
        return None
    if k >= n_a + n_b:
        # Every call used it: the table has an empty column and nothing to test.
        return 1.0
    from scipy import stats as _st

    candidates = []
    for used_a in (min(k, n_a), max(0, k - n_b)):
        used_b = k - used_a
        table = [[used_a, n_a - used_a], [used_b, n_b - used_b]]
        try:
            _, p = _st.fisher_exact(table)
        except Exception:  # pragma: no cover - defensive
            continue
        if math.isfinite(float(p)):
            candidates.append(float(p))
    return min(candidates) if candidates else None


def _permutation_pvalue(values_a, values_b) -> float:
    """Two-sided permutation test (1000 permutations, seed 7) on the
    absolute difference in means. Deterministic."""
    rng = np.random.default_rng(RESAMPLE_SEED)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    observed = abs(float(a.mean()) - float(b.mean()))
    pooled = np.concatenate([a, b])
    n_a = a.size
    hits = 0
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(pooled)
        d = abs(float(perm[:n_a].mean()) - float(perm[n_a:].mean()))
        if d >= observed - 1e-12:
            hits += 1
    return float((hits + 1) / (N_PERMUTATIONS + 1))


def _find_column(
    df: pd.DataFrame, names: Sequence[str], exclude: Sequence[Optional[str]]
) -> Optional[str]:
    """Case-insensitive exact column lookup, skipping excluded columns."""
    excluded = {str(c).lower() for c in exclude if c}
    lowered: Dict[str, str] = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl not in lowered:
            lowered[cl] = str(c)
    for n in names:
        c = lowered.get(n)
        if c is not None and c.lower() not in excluded:
            return c
    return None


def _numeric_series(series: pd.Series) -> pd.Series:
    """Coerce a metric column to numeric; map common boolean-ish strings
    when plain coercion loses most of the column."""
    x = pd.to_numeric(series, errors="coerce")
    if float(x.notna().mean()) >= 0.8:
        return x
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "success": 1.0,
                "fail": 0.0,
                "failure": 0.0,
                "resolved": 1.0,
                "unresolved": 0.0,
                "escalated": 1.0,
                "not_escalated": 0.0,
            }
        )
    )


def _finding(
    ftype: str,
    severity: str,
    attribute: Optional[str],
    plain: str,
    statistical_test: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "type": ftype,
        "severity": severity,
        "attribute": attribute,
        "plain": plain,
        "statisticalTest": statistical_test,
        "groupDistributions": {},
        "representationRatios": {},
        "underrepresentedGroups": [],
        "overrepresentedGroups": [],
    }


def _evidence_rows(
    group: str, values: Sequence[Any], key: str, limit: int = MAX_EVIDENCE_ROWS
) -> List[Dict[str, Any]]:
    """G-20: first ``limit`` raw trace rows for one side of a comparison.
    Deterministic and seed-free: rows are taken in table order. Plain
    dicts only, so the whole payload stays JSON-serializable."""
    return [{"group": str(group), key: str(v)} for v in list(values)[:limit]]


def _delegation_evidence(
    routes: Sequence[str],
    demographics: Sequence[str],
    groups: Sequence[str],
    limit: int = MAX_EVIDENCE_ROWS,
) -> List[Dict[str, Any]]:
    """G-20: first ``limit`` raw route rows PER GROUP behind a delegation
    finding, taken in table order (deterministic, seed-free)."""
    rows: List[Dict[str, Any]] = []
    for g in groups:
        n = 0
        for route, dg in zip(routes, demographics):
            if str(dg) == str(g):
                rows.append({"group": str(g), "route": str(route)})
                n += 1
                if n >= limit:
                    break
    return rows


def _build_comparisons(
    adequate: List[str], tool_lists: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """All pairs when at most MAX_GROUPS_ALL_PAIRS adequate groups; one
    group vs the pooled rest beyond that."""
    comparisons: List[Dict[str, Any]] = []
    if len(adequate) <= MAX_GROUPS_ALL_PAIRS:
        for i in range(len(adequate)):
            for j in range(i + 1, len(adequate)):
                ga, gb = adequate[i], adequate[j]
                comparisons.append(
                    {
                        "groupA": ga,
                        "groupB": gb,
                        "mode": "pairwise",
                        "toolsA": tool_lists[ga],
                        "toolsB": tool_lists[gb],
                    }
                )
    else:
        for g in adequate:
            rest: List[str] = []
            for h in adequate:
                if h != g:
                    rest.extend(tool_lists[h])
            comparisons.append(
                {
                    "groupA": g,
                    "groupB": REST_LABEL,
                    "mode": "one_vs_rest",
                    "toolsA": tool_lists[g],
                    "toolsB": rest,
                }
            )
    return comparisons


def _omnibus_block(adequate: List[str], tool_lists: Dict[str, List[str]]) -> Dict[str, Any]:
    """Omnibus chi-square on the full group x tool contingency table.
    Gates every tool/action finding."""
    counts = {g: Counter(tool_lists[g]) for g in adequate}
    all_tools = sorted(set().union(*[set(c) for c in counts.values()])) if counts else []
    if len(adequate) < 2 or len(all_tools) < 2:
        return {
            "available": True,
            "test": "chi_square_contingency",
            "chiSquare": 0.0,
            "pValue": 1.0,
            "dof": 0,
            "significant": False,
            "groups": list(adequate),
            "tools": all_tools,
            "note": "Fewer than two groups or two distinct tools; no "
            "selection disparity is definable.",
        }
    table = np.array([[counts[g].get(t, 0) for t in all_tools] for g in adequate], dtype=float)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {
        "available": True,
        "test": "chi_square_contingency",
        "chiSquare": float(chi2),
        "pValue": float(p),
        "dof": int(dof),
        "significant": bool(float(p) < ALPHA),
        "groups": list(adequate),
        "tools": all_tools,
    }


def _trajectory_section(
    df: pd.DataFrame,
    s: pd.Series,
    adequate: List[str],
    exclude: Sequence[Optional[str]],
    group_col: str,
    action_col: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Cheap deterministic trajectory-shape check: group means for a
    step-count column and group rates for an outcome column, permutation
    tests + BH within this family."""
    step_col = _find_column(df, _TRAJECTORY_STEP_COLS, exclude)
    outcome_col = _find_column(df, _TRAJECTORY_OUTCOME_COLS, exclude)
    if step_col is None and outcome_col is None:
        return (
            {
                "available": False,
                "notApplicable": True,
                "reason": "No step-count or outcome column in the traces "
                "(looked for steps / n_steps / turns / "
                "episode_length and success / resolved / "
                "outcome / escalated).",
                "perMetric": [],
            },
            [],
        )
    if len(adequate) < 2:
        return (
            {
                "available": False,
                "notApplicable": False,
                "reason": "Fewer than two adequately sampled groups; the "
                "trajectory columns exist but cannot be "
                "compared.",
                "perMetric": [],
            },
            [],
        )

    per_metric: List[Dict[str, Any]] = []
    family: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    # G-20: raw trace rows behind each comparison, keyed by comparison
    # identity so the findings loop below can attach them.
    evidence_by_comp: Dict[int, List[Dict[str, Any]]] = {}
    for col, metric, kind in ((step_col, "steps", "mean"), (outcome_col, "outcome", "rate")):
        if col is None:
            continue
        x = _numeric_series(df[col])
        values: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        for g in adequate:
            m = ((s == g) & x.notna()).to_numpy()
            arr = x[m].to_numpy(dtype=float)
            if arr.size >= 2:
                values[g] = arr
                masks[g] = m
        groups = [g for g in adequate if g in values]
        if len(groups) < 2:
            continue

        def _raw_rows(
            mask: np.ndarray, x=x, col=col, limit: int = MAX_EVIDENCE_ROWS
        ) -> List[Dict[str, Any]]:
            """First rows (table order, seed-free) of one comparison side:
            group + the step/outcome value, plus the tool/action value of
            the same trace row when the traces carry one."""
            rows: List[Dict[str, Any]] = []
            for i in np.nonzero(mask)[0][:limit]:
                i = int(i)
                row: Dict[str, Any] = {
                    "group": str(s.iloc[i]),
                    "column": str(col),
                    "value": float(x.iloc[i]),
                }
                if action_col is not None and action_col in df.columns:
                    row["tool"] = str(df[action_col].iloc[i])
                rows.append(row)
            return rows

        entry: Dict[str, Any] = {
            "metric": metric,
            "column": col,
            "kind": kind,
            "perGroup": {g: float(np.mean(values[g])) for g in groups},
            "comparisons": [],
        }
        if len(groups) <= MAX_GROUPS_ALL_PAIRS:
            specs = [
                (
                    groups[i],
                    groups[j],
                    values[groups[i]],
                    values[groups[j]],
                    masks[groups[i]],
                    masks[groups[j]],
                )
                for i in range(len(groups))
                for j in range(i + 1, len(groups))
            ]
        else:
            specs = [
                (
                    g,
                    REST_LABEL,
                    values[g],
                    np.concatenate([values[h] for h in groups if h != g]),
                    masks[g],
                    np.logical_or.reduce([masks[h] for h in groups if h != g]),
                )
                for g in groups
            ]
        for ga, gb, va, vb, ma, mb in specs:
            p = _safe(lambda va=va, vb=vb: _permutation_pvalue(va, vb), 1.0)
            from vfairness.agents import ActionBiasAnalyzer

            # nan, never 0.0: this default is reached only when the effect
            # size was NOT computed, and 0.0 there is "no effect", a
            # measurement. _quiet_recorded KEEPS the producer's own sentence
            # explaining why (it warns exactly when it could not measure),
            # instead of letting it escape the probe or discarding it.
            d, d_notes = _quiet_recorded(
                lambda va=va, vb=vb: float(ActionBiasAnalyzer._cohens_d(va, vb)),
                float("nan"),
            )
            # ActionBiasAnalyzer._cohens_d returns nan when the pooled
            # standard deviation is zero. MEASURE that condition here rather
            # than inferring it from the nan: both groups constant AND their
            # means apart is PERFECT SEPARATION, the largest gap the scale can
            # describe, not an absent one. pooled_std == 0 iff both
            # within-group variances are 0, so this is the producer's own test.
            separated = _safe(
                lambda va=va, vb=vb: bool(
                    float(np.var(va, ddof=1)) == 0.0
                    and float(np.var(vb, ddof=1)) == 0.0
                    and float(np.mean(va)) != float(np.mean(vb))
                ),
                False,
            )
            comp = {
                "groupA": ga,
                "groupB": gb,
                "meanA": float(np.mean(va)),
                "meanB": float(np.mean(vb)),
                "diff": float(np.mean(va) - np.mean(vb)),
                # None, not nan: an unmeasured quantity is reported as null,
                # never as a number a reader would compare or format.
                "effectSize": _finite_or_none(d),
                "effectSizeMeasured": bool(math.isfinite(d)),
                "perfectSeparation": separated,
                "pValue": float(p),
            }
            if d_notes:
                comp["effectSizeNotes"] = d_notes
            entry["comparisons"].append(comp)
            family.append((entry, comp, float(p)))
            evidence_by_comp[id(comp)] = _safe(
                lambda ma=ma, mb=mb: _raw_rows(ma) + _raw_rows(mb), []
            )
        per_metric.append(entry)

    findings: List[Dict[str, Any]] = []
    adjusted = _bh_adjust([p for _, _, p in family])
    for (entry, comp, _), adj in zip(family, adjusted):
        comp["pAdjusted"] = float(adj)
        comp["significant"] = bool(adj < ALPHA)
        if comp["significant"]:
            effect_size = comp["effectSize"]
            sev = _severity_from_d(effect_size)
            # THREE STATES. A missing Cohen's d is decided from what WAS
            # measured, never from the missing value itself. The p-value
            # beside it is measured and has already cleared BH correction,
            # so this branch is never reasoning in the dark.
            if sev is None and comp["perfectSeparation"]:
                # Every episode of each group carried the same value and the
                # two values differ: the standardised effect is undefined
                # because its denominator is zero, and the gap it would have
                # standardised is total. That is the top of the scale, not
                # the bottom. Grading it off the nan returned "info", rank 0,
                # which turned a maximal, significant disparity into
                # "Disclaimer / insufficient assessable data".
                sev = "critical"
                effect_clause = (
                    "Cohen's d undefined (neither group varies at all): the "
                    "two are perfectly separated, not similar"
                )
            elif sev is None:
                # No effect size and no separation to reason from: the third
                # state. The assurance grader ranks this word None, so the
                # finding is reported as unassessed and is never counted as a
                # clean or a minor result.
                sev = "insufficient_data"
                effect_clause = (
                    "Cohen's d could not be computed for this comparison, so no "
                    "effect-size severity was measured"
                )
            else:
                effect_clause = f"Cohen's d {float(effect_size):.2f}"
            unit = "steps per episode" if entry["kind"] == "mean" else "outcome rate"
            fd = _finding(
                "agent_trajectory_bias",
                sev,
                group_col,
                (
                    f"Trajectory gap on '{entry['column']}': {comp['groupA']} "
                    f"averages {comp['meanA']:.2f} vs {comp['meanB']:.2f} for "
                    f"{comp['groupB']} ({unit}; permutation test, BH-adjusted "
                    f"p {adj:.4g}, {effect_clause}). The "
                    f"agent's episodes unfold differently by demographic."
                ),
                {
                    "test": "permutation",
                    "permutations": N_PERMUTATIONS,
                    "seed": RESAMPLE_SEED,
                    "pValue": comp["pValue"],
                    "pAdjusted": float(adj),
                    "correction": "benjamini_hochberg",
                    "cohensD": comp["effectSize"],
                    "cohensDMeasured": comp["effectSizeMeasured"],
                    "perfectSeparation": comp["perfectSeparation"],
                },
            )
            # G-20: the literal trace rows behind this comparison.
            fd["evidence"] = evidence_by_comp.get(id(comp), [])
            findings.append(fd)
    section = {
        "available": True,
        "perMetric": per_metric,
        "method": (
            f"Permutation test ({N_PERMUTATIONS} "
            f"permutations, seed {RESAMPLE_SEED}) with "
            f"Benjamini-Hochberg correction within the "
            f"trajectory family."
        ),
    }
    return (section, findings)


def _delegation_section(
    df: pd.DataFrame, s: pd.Series, adequate: List[str], action_col: str, group_col: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[str]]:
    """G-33: delegation-routing audit across ALL adequate groups via the
    existing DelegationRoutingAuditor (its analyze() takes flat aligned
    route/demographic sequences, so per-episode rows fit natively)."""
    col = _find_column(df, _DELEGATION_COLS, exclude=(action_col, group_col))
    if col is None:
        return (
            {
                "available": False,
                "notApplicable": True,
                "reason": "no route/delegation column in the traces",
            },
            [],
            None,
        )
    if len(adequate) < 2:
        return (
            {
                "available": False,
                "notApplicable": False,
                "column": col,
                "reason": "Fewer than two adequately sampled groups; the "
                "delegation column exists but cannot be "
                "compared.",
            },
            [],
            col,
        )
    mask = s.isin(adequate)
    routes = df.loc[mask.to_numpy(), col].astype(str).tolist()
    demographics = s[mask].astype(str).tolist()

    from vfairness.multi_agent import DelegationRoutingAuditor

    res = _quiet(lambda: DelegationRoutingAuditor(alpha=ALPHA).analyze(routes, demographics), None)
    if res is None:
        return (
            {
                "available": False,
                "notApplicable": False,
                "column": col,
                "reason": "The delegation auditor needs at least two "
                "distinct routes and two groups; this column "
                "does not vary enough to audit.",
            },
            [],
            col,
        )

    v = float(res.cramers_v)
    v_measured = math.isfinite(v)
    severity = _severity_from_v(v)
    # THREE STATES on the routing effect size too. DelegationRoutingAuditor
    # still answers 0.0 for a table with k = 0 (multi_agent/delegation.py
    # _cramers_v), so this branch does not fire today; it is here so that the
    # moment that helper reports the honest nan this section cannot grade it
    # "info", a word that means measured and small.
    deleg_not_checked: List[str] = []
    if severity is None:
        severity = "insufficient_data"
        deleg_not_checked.append(
            "Cramer's V could not be computed for this routing table, so no "
            "effect-size severity was measured. This is not a small effect, "
            "it is an unmeasured one."
        )
    odds = float(res.odds_ratio)
    section = {
        "available": True,
        "column": col,
        "method": "delegation_routing_auditor",
        "auditor": "DelegationRoutingAuditor",
        "routes": list(res.routes),
        "groups": list(res.groups),
        "contingencyTable": {
            r: {g: int(n) for g, n in row.items()} for r, row in res.contingency_table.items()
        },
        "chiSquare": float(res.chi_square),
        # None, not nan: an unmeasured quantity is reported as null, never as
        # a number a reader would compare or format.
        "cramersV": _finite_or_none(v),
        "cramersVMeasured": v_measured,
        "couldNotCheck": deleg_not_checked,
        "pValue": float(res.p_value),
        # THREE STATES. The auditor returns None when the routing test could
        # never have reached alpha at these group sizes (100 percent segregated
        # routing with 3 decisions per group floors at p = 0.10), and
        # `bool(None)` printed that as a clean "not significant".
        "significant": res.is_significant,
        "detectable": res.detectable,
        "minAttainableP": res.min_attainable_p,
        "notAssessedReason": (res.detectability_note if res.is_significant is None else ""),
        "testUsed": str(res.test_used),
        "oddsRatio": (None if math.isnan(odds) else odds),
        "perRouteDisparity": {r: float(d) for r, d in res.per_route_disparity.items()},
        "severity": severity,
    }
    findings: List[Dict[str, Any]] = []
    if res.is_significant is True:
        worst_route = (
            max(res.per_route_disparity, key=lambda r: res.per_route_disparity[r])
            if res.per_route_disparity
            else ""
        )
        fd = _finding(
            "delegation_routing",
            severity,
            group_col,
            (
                f"Delegation routing depends on the demographic group "
                f"({res.test_used} p {float(res.p_value):.4g}, "
                f"{_v_clause(section)}); the largest per-route disparity is on "
                f"'{worst_route}'. Routing demographics to different "
                f"downstream handlers is an agentic fairness violation."
            ),
            {
                "test": str(res.test_used),
                "pValue": float(res.p_value),
                "chiSquare": float(res.chi_square),
                "cramersV": _finite_or_none(v),
                "cramersVMeasured": v_measured,
                "routes": list(res.routes),
                "groups": list(res.groups),
            },
        )
        # G-20: the literal route rows behind this finding, per group.
        fd["evidence"] = _safe(lambda: _delegation_evidence(routes, demographics, adequate), [])
        findings.append(fd)
    return (section, findings, col)


def _severity_from_amplification(factor: float) -> str:
    """Severity for emergent amplification derives from the detector's
    own amplification factor (system bias over max component bias):
    >= 3x critical, >= 1.5x (the detector's emergence threshold) warn,
    else info."""
    if factor >= 3.0:
        return "critical"
    if factor >= 1.5:
        return "warn"
    return "info"


def _severity_from_gap(gap: float) -> str:
    """Severity for a selection-rate gap (same convention as the
    orchestrator's disparity tones): >= 0.20 critical, >= 0.10 warn,
    else info."""
    a = abs(gap)
    if a >= 0.20:
        return "critical"
    if a >= 0.10:
        return "warn"
    return "info"


def _component_columns(df: pd.DataFrame, exclude: Sequence[Optional[str]]) -> List[str]:
    """Per-agent/component output columns: an explicit agent_*/component_*
    prefix AND an explicit *_output/*_score/*_decision/*_opinion suffix,
    numeric-coercible on at least 80 percent of rows. Conservative by
    design (G-21): never guess a component column."""
    excluded = {str(c).lower() for c in exclude if c}
    out: List[str] = []
    for c in df.columns:
        cl = str(c).lower()
        if cl in excluded or cl in _SYSTEM_OUTPUT_COLS:
            continue
        if not any(cl.startswith(p) for p in _COMPONENT_PREFIXES):
            continue
        if not any(cl.endswith(sfx) for sfx in _COMPONENT_SUFFIXES):
            continue
        x = _safe(lambda c=c: _numeric_series(df[c]), None)
        if x is not None and float(x.notna().mean()) >= 0.8:
            out.append(str(c))
    return out


def _ordering_values(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Ordering column as floats: numeric coercion first, ISO datetime
    parse as the fallback. NaN where unparseable."""
    t = pd.to_numeric(df[time_col], errors="coerce")
    if float(t.notna().mean()) >= 0.8:
        return t.astype(float)
    dt = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    seconds = [float(x.timestamp()) if pd.notna(x) else np.nan for x in dt]
    return pd.Series(seconds, index=df.index, dtype=float)


def _amplification_section(
    df: pd.DataFrame,
    s: pd.Series,
    adequate: List[str],
    exclude: Sequence[Optional[str]],
    group_col: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """G-21: system-vs-component amplification via the existing
    EmergentBiasDetector, run across ALL adequately sampled groups
    (every pair up to 5 groups, one-vs-rest beyond; the same comparison
    regime as the tool statistics). Only runs when the traces carry
    BOTH per-agent output columns and a system-level output column."""
    not_applicable: Tuple[Dict[str, Any], List[Dict[str, Any]]] = (
        {
            "available": False,
            "notApplicable": True,
            "reason": _AMPLIFICATION_NA_REASON,
            "perComparison": [],
        },
        [],
    )
    sys_col = _find_column(df, _SYSTEM_OUTPUT_COLS, exclude)
    comp_cols = _component_columns(df, list(exclude) + [sys_col])
    if sys_col is None or not comp_cols:
        return not_applicable
    sys_x = _numeric_series(df[sys_col])
    if sys_x is None or float(sys_x.notna().mean()) < 0.5:
        return not_applicable
    if len(adequate) < 2:
        return (
            {
                "available": False,
                "notApplicable": False,
                "systemColumn": sys_col,
                "componentColumns": comp_cols,
                "reason": "Fewer than two adequately sampled groups; "
                "the per-agent and system output columns "
                "exist but cannot be compared.",
                "perComparison": [],
            },
            [],
        )

    comp_x = {c: _numeric_series(df[c]) for c in comp_cols}
    valid_all = sys_x.notna()
    for c in comp_cols:
        valid_all = valid_all & comp_x[c].notna()

    if len(adequate) <= MAX_GROUPS_ALL_PAIRS:
        pairs = [
            (adequate[i], adequate[j], "pairwise")
            for i in range(len(adequate))
            for j in range(i + 1, len(adequate))
        ]
    else:
        pairs = [(g, REST_LABEL, "one_vs_rest") for g in adequate]

    from vfairness.multi_agent import EmergentBiasDetector

    comparisons: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    for ga, gb, mode in pairs:
        if gb == REST_LABEL:
            mask = (s.isin(adequate) & valid_all).to_numpy()
            labels = np.where(s[mask].to_numpy() == ga, ga, REST_LABEL)
        else:
            mask = (s.isin([ga, gb]) & valid_all).to_numpy()
            labels = s[mask].to_numpy()
        if len(set(labels.tolist())) < 2:
            continue
        res, notes = _quiet_recorded(
            lambda mask=mask, labels=labels: EmergentBiasDetector().analyze(
                {c: comp_x[c][mask].to_numpy(dtype=float) for c in comp_cols},
                sys_x[mask].to_numpy(dtype=float),
                labels,
            ),
            None,
        )
        if res is None:
            continue
        factor = float(res.amplification_factor)
        system_bias = float(res.system_bias)
        max_component = float(res.max_component_bias)
        # THREE STATES. Emergence is a COMPARISON of system bias against
        # the largest component bias; the detector reports None (plus a
        # warning) when it could not make that comparison. `bool(...)` turned
        # that into a clean "no emergent bias" which was then gated on,
        # counted into the section total and read as a census of what was
        # checked, while _quiet() threw away the one sentence that said
        # otherwise. The severity was graded off the same nan and came out
        # "info", a measured-looking floor for a comparison nobody made.
        #
        # What is measured is the two SIDES of the comparison, not the
        # amplification RATIO between them: since 2026-09-10 the detector
        # reports amplification_factor=nan when every component measured a bias
        # of exactly 0, because a ratio over zero is undefined, and it still
        # decides emergence there, from the bootstrap comparison. Gating on
        # isfinite(factor) would throw that real verdict away. Gating on the
        # system bias alone would go too far the other way: with no component
        # bias measurable at all (max_component_bias nan) there is nothing to
        # compare against and the comparison was NOT made.
        measured = math.isfinite(system_bias) and math.isfinite(max_component)
        is_emergent = _measured_verdict(res.is_emergent, measured)
        is_significant = _measured_verdict(res.is_significant, measured)
        could_not_check: List[str] = []
        if is_emergent is None:
            could_not_check.append(
                "system-versus-component amplification could not be measured, "
                "so no emergence verdict was reached for this comparison"
            )
        if is_significant is None:
            could_not_check.append(
                "the bootstrap comparison of system bias against the largest "
                "component bias could not be run, so significance is unknown"
            )
        # SEVERITY COMES FROM A MEASUREMENT. Grading it off the amplification
        # ratio alone is how a 0.001 output gap over a zero component baseline
        # became a CRITICAL finding in this report (measured 2026-09-10: the
        # detector returned the 1e6 sentinel and this line read it as 1e6-fold
        # amplification). When the ratio is undefined the gap itself is still
        # measured, so it is graded on the same scale the rest of the probe uses
        # for a selection-rate gap: 0.20 critical, 0.10 warn, below that info.
        if not measured:
            severity: Optional[str] = None
        elif math.isfinite(factor):
            severity = _severity_from_amplification(factor)
        else:
            severity = _severity_from_gap(abs(system_bias))
        comp: Dict[str, Any] = {
            "groupA": ga,
            "groupB": gb,
            "comparisonMode": mode,
            "n": int(mask.sum()),
            "systemBias": _finite_or_none(system_bias),
            "maxComponentBias": _finite_or_none(float(res.max_component_bias)),
            "amplificationFactor": _finite_or_none(factor),
            "amplificationDefined": bool(math.isfinite(factor)),
            "isEmergent": is_emergent,
            "pValue": _finite_or_none(float(res.p_value)),
            "isSignificant": is_significant,
            "severity": severity,
            "detectorNotes": notes,
        }
        if could_not_check:
            comp["couldNotCheck"] = could_not_check
        comparisons.append(comp)
        if is_emergent is True and is_significant is True:
            findings.append(
                _finding(
                    "emergent_amplification",
                    comp["severity"],
                    group_col,
                    (
                        f"The multi-agent SYSTEM shows a {comp['systemBias']:.3f} "
                        f"output gap between {ga} and {gb} while the worst "
                        f"individual component shows only "
                        f"{comp['maxComponentBias']:.3f} "
                        + (
                            f"({factor:.1f}x amplification; "
                            if math.isfinite(factor)
                            else "(amplification ratio undefined: every component "
                            "measured a bias of exactly 0; "
                        )
                        + f"bootstrap p {comp['pValue']:.4g}). Bias emerges from the "
                        f"agent interaction itself, so auditing components in "
                        f"isolation would miss it."
                    ),
                    {
                        "test": "emergent_bias_bootstrap",
                        "detector": "EmergentBiasDetector",
                        "pValue": comp["pValue"],
                        "systemBias": comp["systemBias"],
                        "maxComponentBias": comp["maxComponentBias"],
                        "amplificationFactor": factor,
                        "nBootstrap": 500,
                        "seed": 42,
                    },
                )
            )
    if not comparisons:
        return (
            {
                "available": False,
                "notApplicable": False,
                "systemColumn": sys_col,
                "componentColumns": comp_cols,
                "reason": "The emergent-bias detector could not run on "
                "these columns (no comparison had two groups "
                "with usable numeric outputs).",
                "perComparison": [],
            },
            [],
        )
    # The denominator is what was CHECKED, not what was attempted: an
    # unmeasured comparison used to sit in the total as evidence of no
    # amplification, so "0 of 1" read as a census of a check that never ran.
    n_emergent = sum(
        1 for c in comparisons if c["isEmergent"] is True and c["isSignificant"] is True
    )
    n_unchecked = sum(
        1 for c in comparisons if c["isEmergent"] is None or c["isSignificant"] is None
    )
    n_checked = len(comparisons) - n_unchecked
    detector_notes = list(dict.fromkeys(n for c in comparisons for n in c["detectorNotes"]))
    if n_checked == 0:
        return (
            {
                "available": False,
                "notApplicable": False,
                "detector": "EmergentBiasDetector",
                "systemColumn": sys_col,
                "componentColumns": comp_cols,
                "reason": f"None of the {len(comparisons)} group comparison(s) "
                "could be checked for emergent amplification: the "
                "detector could not measure system bias against the "
                "component biases. This is a could-not-check, not an "
                "absence of amplification.",
                "perComparison": comparisons,
                "comparisonsChecked": 0,
                "comparisonsUnchecked": n_unchecked,
                "detectorNotes": detector_notes,
            },
            findings,
        )
    summary = (
        f"{n_emergent} of {n_checked} checked group "
        f"comparison(s) show significant emergent "
        f"amplification (system bias exceeding every "
        f"component's bias)."
    )
    if n_unchecked:
        summary += (
            f" A further {n_unchecked} comparison(s) could not be checked "
            f"and are counted in neither figure."
        )
    section: Dict[str, Any] = {
        "available": True,
        "notApplicable": False,
        "detector": "EmergentBiasDetector",
        "systemColumn": sys_col,
        "componentColumns": comp_cols,
        "perComparison": comparisons,
        "comparisonsChecked": n_checked,
        "comparisonsUnchecked": n_unchecked,
        "detectorNotes": detector_notes,
        "summary": summary,
    }
    return (section, findings)


def _groupthink_section(
    df: pd.DataFrame, exclude: Sequence[Optional[str]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """G-21: groupthink/echo-chamber convergence via the existing
    GroupthinkDetector. Its API takes per-round agent output vectors;
    flat per-episode rows can only express rounds through an ordering
    column, so rounds are equal-count windows of the ordered rows. Two
    or more per-agent opinion columns AND an ordering column are
    required; otherwise an honest notApplicable is emitted."""
    comp_cols = _component_columns(df, exclude)
    if len(comp_cols) < 2:
        return (
            {
                "available": False,
                "notApplicable": True,
                "reason": "Groupthink convergence needs at least two "
                "per-agent opinion columns "
                "(agent_*/component_* with an "
                "_output/_score/_decision/_opinion suffix) "
                "in the traces.",
            },
            [],
        )
    time_col = _find_column(df, _TIME_COLS, exclude)
    if time_col is None:
        return (
            {
                "available": False,
                "notApplicable": True,
                "agents": comp_cols,
                "reason": "Groupthink convergence needs an ordering "
                "column (timestamp / time / ts / created_at "
                "/ step_index) to form deliberation rounds; "
                "the GroupthinkDetector API takes per-round "
                "agent outputs and cannot run on unordered "
                "flat rows.",
            },
            [],
        )
    t = _ordering_values(df, time_col)
    valid = t.notna()
    comp_x = {c: _numeric_series(df[c]) for c in comp_cols}
    for c in comp_cols:
        valid = valid & comp_x[c].notna()
    idx = np.argsort(t[valid.to_numpy()].to_numpy(dtype=float), kind="stable")
    n = int(idx.size)
    n_rounds = min(_MAX_GROUPTHINK_ROUNDS, n // 5)
    if n_rounds < 2:
        return (
            {
                "available": False,
                "notApplicable": False,
                "agents": comp_cols,
                "timeColumn": time_col,
                "reason": "Too few ordered rows to form at least two "
                "rounds (need 10 or more rows with every "
                "agent-opinion value present).",
            },
            [],
        )
    bounds = np.linspace(0, n, n_rounds + 1).astype(int)
    ordered = {c: comp_x[c][valid.to_numpy()].to_numpy(dtype=float)[idx] for c in comp_cols}
    rounds: List[Dict[str, List[float]]] = []
    for w in range(n_rounds):
        seg = slice(int(bounds[w]), int(bounds[w + 1]))
        if bounds[w + 1] - bounds[w] < 1:
            continue
        rounds.append({c: [float(v) for v in ordered[c][seg]] for c in comp_cols})
    if len(rounds) < 2:
        return (
            {
                "available": False,
                "notApplicable": False,
                "agents": comp_cols,
                "timeColumn": time_col,
                "reason": "Could not form two non-empty rounds from the ordered rows.",
            },
            [],
        )
    from vfairness.multi_agent import GroupthinkDetector

    res, notes = _quiet_recorded(lambda: GroupthinkDetector().analyze_convergence(rounds), None)
    if res is None:
        return (
            {
                "available": False,
                "notApplicable": False,
                "agents": comp_cols,
                "timeColumn": time_col,
                "reason": "The groupthink detector could not run on these rounds.",
            },
            [],
        )
    echo = float(res.echo_chamber_score)
    trajectory = [float(v) for v in res.convergence_trajectory]
    # THREE STATES, the same rule as the amplification section above.
    # has_groupthink is the detector's TREND across the per-round
    # convergence trajectory, so one non-finite round voids it (Kendall's
    # tau on a nan is nan and `nan > 0.3` is a silent False); is_significant
    # is the permutation test on the FINAL round, so a non-finite echo score
    # voids that one. `bool(...)` reported both as "no groupthink found",
    # and _quiet() suppressed the detector's own could-not-measure warning
    # in the same call, leaving no channel at all.
    trend_measured = bool(trajectory) and all(math.isfinite(v) for v in trajectory)
    echo_measured = math.isfinite(echo)
    has_groupthink = _measured_verdict(res.has_groupthink, trend_measured)
    is_significant = _measured_verdict(res.is_significant, echo_measured)
    could_not_check: List[str] = []
    if has_groupthink is None:
        could_not_check.append(
            "the convergence trend across the ordered rounds could not be "
            "measured, so no groupthink verdict was reached"
        )
        # The detector also reports None when the trend test could not have
        # reached its threshold on this many rounds (Kendall's tau on 3 rounds
        # floors at 0.3333). That reason is specific and actionable, namely
        # collect more ordered rows, so it travels rather than being
        # generalised away.
        if getattr(res, "trend_note", ""):
            could_not_check.append(str(res.trend_note))
    if is_significant is None:
        could_not_check.append(
            "the permutation test on the final round's convergence could not "
            "be run, so significance is unknown"
        )
        if getattr(res, "convergence_note", ""):
            could_not_check.append(str(res.convergence_note))
    section: Dict[str, Any] = {
        "available": has_groupthink is not None,
        "notApplicable": False,
        "detector": "GroupthinkDetector",
        "agents": comp_cols,
        "timeColumn": time_col,
        "rounds": len(rounds),
        "roundsFrom": (f"equal-count windows of the rows ordered by '{time_col}'"),
        "hasGroupthink": has_groupthink,
        "echoChamberScore": _finite_or_none(echo),
        "convergenceTrajectory": [_finite_or_none(v) for v in trajectory],
        "coalitions": [list(c) for c in res.coalition_structure],
        "pValue": _finite_or_none(float(res.p_value)),
        "isSignificant": is_significant,
        "detectorNotes": notes,
        "note": (
            "Convergence is the detector's own pairwise "
            "cosine similarity of per-agent output vectors "
            "per round; rounds here are ordered windows of "
            "the trace rows, not literal deliberation "
            "rounds, and are disclosed as such."
        ),
    }
    if could_not_check:
        section["couldNotCheck"] = could_not_check
    if has_groupthink is None:
        # available is False here, so the file's own could-not-check idiom
        # (available False + notApplicable False + reason) has to be complete.
        section["reason"] = (
            "The groupthink verdict could not be measured on these rounds; "
            "the trajectory below is reported as evidence for review, not as "
            "a finding of no convergence."
        )
    findings: List[Dict[str, Any]] = []
    if has_groupthink is True and is_significant is True:
        severity = "critical" if echo >= 0.9 else "warn"
        findings.append(
            _finding(
                "groupthink_convergence",
                severity,
                "agents",
                (
                    f"The per-agent outputs converge over the ordered trace "
                    f"windows (echo-chamber score {echo:.2f}, permutation p "
                    f"{float(res.p_value):.4g}). Converging agents stop "
                    f"correcting each other, which can lock in and amplify a "
                    f"biased consensus."
                ),
                {
                    "test": "groupthink_convergence_permutation",
                    "detector": "GroupthinkDetector",
                    "pValue": float(res.p_value),
                    "echoChamberScore": echo,
                    "rounds": len(rounds),
                    "nPermutations": 500,
                    "seed": 42,
                },
            )
        )
    return (section, findings)


def _temporal_section(
    df: pd.DataFrame,
    s: pd.Series,
    adequate: List[str],
    action_col: str,
    exclude: Sequence[Optional[str]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """G-37: temporal disparity dynamics via the existing
    TemporalTracker (agents/temporal.py). The per-window selection-rate
    gap for the most disparate tool between the two most-populous
    adequate groups is fed to the tracker's own Mann-Kendall
    feedback-loop test, CUSUM and EWMA drift detectors. CUSUM/EWMA are
    never reimplemented here."""
    time_col = _find_column(df, _TIME_COLS, exclude)
    if time_col is None:
        return (
            {
                "available": False,
                "notApplicable": True,
                "reason": "No timestamp/ordering column in the traces "
                "(looked for timestamp / time / ts / "
                "created_at / step_index).",
                "perWindow": [],
            },
            [],
        )
    if len(adequate) < 2:
        return (
            {
                "available": False,
                "notApplicable": False,
                "timeColumn": time_col,
                "reason": "Fewer than two adequately sampled groups; "
                "the ordering column exists but no disparity "
                "series can be formed.",
                "perWindow": [],
            },
            [],
        )
    t = _ordering_values(df, time_col)
    ga, gb = adequate[0], adequate[1]  # the two most-populous groups
    mask = (s.isin([ga, gb]) & t.notna()).to_numpy()
    if int(mask.sum()) < _MIN_TEMPORAL_WINDOWS * 4:
        return (
            {
                "available": False,
                "notApplicable": False,
                "timeColumn": time_col,
                "groups": [ga, gb],
                "reason": "Too few rows with a parseable ordering "
                "value to form at least three windows.",
                "perWindow": [],
            },
            [],
        )
    # Same explicit missing-action label as the main tool lists (pandas 3
    # keeps NaN a float through astype(str); see NO_ACTION_LABEL).
    _sub = df.loc[mask, action_col]
    sub_tools = (
        _sub.astype(object).where(_sub.notna().to_numpy(), NO_ACTION_LABEL).astype(str).to_numpy()
    )
    sub_groups = s[mask].to_numpy()
    sub_t = t[mask].to_numpy(dtype=float)
    order = np.argsort(sub_t, kind="stable")
    sub_tools = sub_tools[order]
    sub_groups = sub_groups[order]

    # Focal tool: the largest overall selection-rate gap between the two
    # groups; ties resolve lexicographically (deterministic).
    a_tools = sub_tools[sub_groups == ga]
    b_tools = sub_tools[sub_groups == gb]
    focal, focal_gap = None, 0.0
    for tool in sorted(set(sub_tools.tolist())):
        gap = float(np.mean(a_tools == tool)) - float(np.mean(b_tools == tool))
        if focal is None or abs(gap) > abs(focal_gap):
            focal, focal_gap = tool, gap
    if focal is None:
        return (
            {
                "available": False,
                "notApplicable": False,
                "timeColumn": time_col,
                "groups": [ga, gb],
                "reason": "No tool values found in the ordered rows.",
                "perWindow": [],
            },
            [],
        )

    n = int(sub_tools.size)
    n_windows = max(_MIN_TEMPORAL_WINDOWS, min(_MAX_TEMPORAL_WINDOWS, n // _ROWS_PER_WINDOW))
    bounds = np.linspace(0, n, n_windows + 1).astype(int)

    from vfairness.agents import TemporalTracker

    tracker = TemporalTracker()
    per_window: List[Dict[str, Any]] = []
    turn = 0
    for w in range(n_windows):
        seg = slice(int(bounds[w]), int(bounds[w + 1]))
        wa = (sub_tools[seg][sub_groups[seg] == ga] == focal).astype(float)
        wb = (sub_tools[seg][sub_groups[seg] == gb] == focal).astype(float)
        if wa.size == 0 or wb.size == 0:
            continue  # a window must contain both groups to be a turn
        tracker.record_turn(turn, wa, wb)
        per_window.append(
            {
                "window": w,
                "nA": int(wa.size),
                "nB": int(wb.size),
                "rateA": float(wa.mean()),
                "rateB": float(wb.mean()),
                "disparity": float(wa.mean() - wb.mean()),
            }
        )
        turn += 1
    if turn < _MIN_TEMPORAL_WINDOWS:
        return (
            {
                "available": False,
                "notApplicable": False,
                "timeColumn": time_col,
                "groups": [ga, gb],
                "tool": focal,
                "reason": (
                    f"Only {turn} ordered window(s) contain both "
                    f"groups; at least {_MIN_TEMPORAL_WINDOWS} "
                    f"are needed for drift detection."
                ),
                "perWindow": per_window,
            },
            [],
        )

    # THE DEFAULT HANDED TO _quiet IS WHAT GETS PUBLISHED IF THE DETECTOR RAISES, so
    # it must be the could-not-check reading and not the clean one. These three
    # defaults used to be has_feedback_loop=False, trend_direction="stable",
    # trend_strength=0.0, p_value=1.0, has_drift=False, max_cusum=0.0: every single
    # value the reassuring one. A detector that crashed was published as a drift test
    # that ran and found nothing.
    #
    # The detectors themselves are honest. Executed on two turns they return
    # has_feedback_loop=None, trend_direction='not_assessed', trend_strength=nan and
    # p_value=nan. What threw that away was this layer: `bool(None)` is False, and
    # False on a fairness verdict reads as "checked, nothing found".
    feedback = _quiet(
        lambda: tracker.detect_feedback_loop(alpha=ALPHA),
        {
            "has_feedback_loop": None,
            "trend_direction": "not_assessed",
            "trend_strength": float("nan"),
            "p_value": float("nan"),
            _DETECTOR_FAILED: True,
        },
    )
    # CUSUM slack/limit tuned for a rate-gap series bounded in [-1, 1];
    # the statistic itself is the tracker's, never reimplemented.
    cusum = _quiet(
        lambda: tracker.detect_drift_cusum(threshold=0.05, drift_limit=0.5),
        {
            "has_drift": None,
            "drift_point": None,
            "max_cusum": float("nan"),
            _DETECTOR_FAILED: True,
        },
    )
    ewma = _quiet(
        lambda: tracker.detect_drift_ewma(),
        {"has_drift": None, "drift_points": [], _DETECTOR_FAILED: True},
    )
    detector_failed = [
        name
        for name, d in (("feedbackLoop", feedback), ("cusum", cusum), ("ewma", ewma))
        if d.get(_DETECTOR_FAILED)
    ]
    # THREE STATES. True if any detector found drift; False only when all three
    # reached a verdict and none did; None when none of them reached one. `bool(None
    # or None or None)` was False, which is the answer that closes the question.
    verdicts = [
        _measured_verdict(d.get(key), True)
        for d, key in (
            (feedback, "has_feedback_loop"),
            (cusum, "has_drift"),
            (ewma, "has_drift"),
        )
    ]
    if any(v is True for v in verdicts):
        drift: Optional[bool] = True
    elif all(v is None for v in verdicts):
        drift = None
    else:
        drift = False
    first_gap = per_window[0]["disparity"]
    last_gap = per_window[-1]["disparity"]
    growing = (
        str(feedback.get("trend_direction")) == "increasing"
        or abs(last_gap) > abs(first_gap) + 1e-12
    )

    section = {
        "available": True,
        "notApplicable": False,
        "timeColumn": time_col,
        "tool": focal,
        "groups": [ga, gb],
        "windows": turn,
        "method": (
            f"TemporalTracker (Mann-Kendall feedback-loop trend "
            f"+ CUSUM + EWMA drift detection) on the per-window "
            f"'{focal}' selection-rate gap between {ga} and {gb} "
            f"over {turn} ordered windows of '{time_col}'."
        ),
        "driftDetected": drift,
        # Named here so a reader of the payload can tell a detector that REFUSED from
        # one that RAISED. Both leave the verdict null and only one of them means the
        # data was insufficient.
        "detectorsFailed": detector_failed,
        "feedbackLoop": {
            "hasFeedbackLoop": _measured_verdict(feedback.get("has_feedback_loop"), True),
            "trendDirection": str(feedback.get("trend_direction") or "not_assessed"),
            # _finite_or_none, not `or 0.0`. A trend strength of 0.0 is a MEASURED
            # absence of trend and nan is no measurement at all, and `nan or 0.0`
            # keeps nan only because nan happens to be truthy, which is not a
            # property to rely on.
            "trendStrength": _finite_or_none(feedback.get("trend_strength")),
            "pValue": _finite_or_none(feedback.get("p_value")),
        },
        "cusum": {
            "hasDrift": _measured_verdict(cusum.get("has_drift"), True),
            "driftPoint": cusum.get("drift_point"),
            "maxCusum": _finite_or_none(cusum.get("max_cusum")),
        },
        "ewma": {
            "hasDrift": _measured_verdict(ewma.get("has_drift"), True),
            "driftPoints": [int(p) for p in (ewma.get("drift_points") or [])],
        },
        "perWindow": per_window,
        "perGroup": {
            str(ga): [w["rateA"] for w in per_window],
            str(gb): [w["rateB"] for w in per_window],
        },
    }
    findings: List[Dict[str, Any]] = []
    if drift and growing:
        severity = _severity_from_gap(last_gap)
        # p = 0.0 is the MOST significant reading on the scale and is falsy,
        # so it must never be coalesced away with `or 1.0` (the LEAST
        # significant reading) inside a finding that says drift is growing.
        # Same `is not None` form as the section payload above; the prose
        # and the structured evidence read this one value.
        fb_p = float(feedback["p_value"] if feedback.get("p_value") is not None else 1.0)
        findings.append(
            _finding(
                "agent_temporal_drift",
                severity,
                time_col,
                (
                    f"The '{focal}' selection-rate gap between {ga} and {gb} "
                    f"drifts over time: {first_gap:+.2f} in the first window "
                    f"to {last_gap:+.2f} in the last "
                    f"(Mann-Kendall tau "
                    f"{float(feedback.get('trend_strength') or 0.0):.2f}, p "
                    f"{fb_p:.4g}; CUSUM max "
                    f"{float(cusum.get('max_cusum') or 0.0):.2f}). A growing "
                    f"disparity suggests a feedback loop rather than a static "
                    f"bias, and it will not correct itself."
                ),
                {
                    "test": "temporal_tracker_mann_kendall_cusum_ewma",
                    "detector": "TemporalTracker",
                    "tau": float(feedback.get("trend_strength") or 0.0),
                    "pValue": fb_p,
                    "maxCusum": float(cusum.get("max_cusum") or 0.0),
                    "windows": turn,
                },
            )
        )
    return (section, findings)


def _memory_section(
    df: pd.DataFrame, s: pd.Series, adequate: List[str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """G-43: memory/context-contamination screen (MemAudit / OWASP ASI06
    lineage), v1 count-based form on the reserved trace-contract fields.

    The schema-v1 flattener records per-episode memory_reads and
    memory_writes COUNTS (populated only when the span export actually
    carried memory events); the event payloads are not retained, so v1
    honestly reads reliance and exposure, never content:

      (a) memory-reliance disparity: the share of each group's episodes
          that read memory-carried context at all (chi-square across the
          adequately sampled groups). When one group's episodes lean on
          carried-over state more than another's, whatever bias that
          state accumulates lands on them unevenly.
      (b) cross-group carry-over exposure: ordered by timestamp, an
          episode that reads memory AFTER any earlier episode of a
          different group wrote memory is exposed to state written by
          the other group. This is a contamination SURFACE (the
          opportunity exists), not proven contamination; term-level
          flow needs the payloads and is named as not covered.
    """
    has_cols = "memory_reads" in df.columns or "memory_writes" in df.columns
    reads = (
        pd.to_numeric(df["memory_reads"], errors="coerce")
        if "memory_reads" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    writes = (
        pd.to_numeric(df["memory_writes"], errors="coerce")
        if "memory_writes" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    if not has_cols or (reads.notna().sum() == 0 and writes.notna().sum() == 0):
        return (
            {
                "available": False,
                "notApplicable": True,
                "reason": (
                    "The traces carry no memory read/write "
                    "events (the trace contract's memory fields "
                    "are populated only when the span export "
                    "contains them), so the memory-contamination "
                    "screen has nothing to read."
                ),
            },
            [],
        )
    if len(adequate) < 2:
        return (
            {
                "available": False,
                "notApplicable": False,
                "reason": (
                    "Memory events exist, but fewer than two "
                    "groups clear the sample floor, so no "
                    "reliance comparison can be formed."
                ),
            },
            [],
        )

    r = reads.fillna(0.0)
    w = writes.fillna(0.0)
    per_group: List[Dict[str, Any]] = []
    reading_counts: List[List[int]] = []
    for g in adequate:
        mask = (s == g).to_numpy()
        n_g = int(mask.sum())
        n_reading = int((r.to_numpy()[mask] > 0).sum())
        per_group.append(
            {
                "group": str(g),
                "n": n_g,
                "episodesReadingShare": (n_reading / n_g) if n_g else 0.0,
                "episodesWritingShare": (float((w.to_numpy()[mask] > 0).mean()) if n_g else 0.0),
                "meanReadsPerEpisode": float(r.to_numpy()[mask].mean()) if n_g else 0.0,
                "meanWritesPerEpisode": float(w.to_numpy()[mask].mean()) if n_g else 0.0,
            }
        )
        reading_counts.append([n_reading, n_g - n_reading])

    chi2_p = None
    if len(reading_counts) >= 2 and all(sum(row) > 0 for row in reading_counts):
        table = np.asarray(reading_counts, dtype=float)
        # chi-square needs variation in BOTH margins; an all-read or
        # no-read table is reliance parity by inspection.
        if table[:, 0].sum() > 0 and table[:, 1].sum() > 0:
            chi2_p = float(stats.chi2_contingency(table)[1])

    # (b) cross-group carry-over exposure, timestamp-ordered.
    exposure: Dict[str, Any]
    time_col = _find_column(df, _TIME_COLS, ())
    if time_col is None:
        exposure = {
            "available": False,
            "notApplicable": True,
            "reason": (
                "No timestamp/ordering column, so the carry-over order of episodes is unknown."
            ),
        }
    else:
        t = _ordering_values(df, time_col)
        order = np.argsort(t.fillna(np.inf).to_numpy(dtype=float), kind="stable")
        gv = s.astype(str).to_numpy()[order]
        rv = (r.to_numpy() > 0)[order]
        wv = (w.to_numpy() > 0)[order]
        writers_seen: set = set()
        exposed = {str(g): 0 for g in adequate}
        readers = {str(g): 0 for g in adequate}
        for i in range(len(gv)):
            g = gv[i]
            if rv[i] and g in exposed:
                readers[g] += 1
                if writers_seen - {g}:
                    exposed[g] += 1
            if wv[i]:
                writers_seen.add(g)
        exposure = {
            "available": True,
            "timeColumn": str(time_col),
            "perGroup": [
                {
                    "group": g,
                    "episodesReadingMemory": readers[g],
                    "exposedToOtherGroupsWrites": exposed[g],
                    "exposedShare": ((exposed[g] / readers[g]) if readers[g] else 0.0),
                }
                for g in (str(a) for a in adequate)
            ],
            "plain": (
                "An episode is counted as exposed when it reads "
                "memory after any earlier episode of a DIFFERENT "
                "group wrote memory: the opportunity for "
                "carried-over state to cross groups existed. This "
                "is a surface measure, not proven contamination."
            ),
        }

    findings: List[Dict[str, Any]] = []
    shares = [(p["group"], p["episodesReadingShare"]) for p in per_group]
    hi = max(shares, key=lambda x: x[1])
    lo = min(shares, key=lambda x: x[1])
    gap = float(hi[1]) - float(lo[1])
    if chi2_p is not None and chi2_p < ALPHA and gap >= 0.10:
        findings.append(
            _finding(
                "memory_reliance_disparity",
                _severity_from_gap(gap),
                None,
                (
                    f"Episodes for '{hi[0]}' lean on memory-carried context in "
                    f"{hi[1] * 100:.0f}% of cases vs {lo[1] * 100:.0f}% for "
                    f"'{lo[0]}' (chi-square p {chi2_p:.4g}). Whatever bias the "
                    "shared memory state accumulates lands on the "
                    "memory-reliant group unevenly."
                ),
                {
                    "test": "chi2_contingency",
                    "pValue": chi2_p,
                    "gap": gap,
                    "groups": [str(g) for g in adequate],
                },
            )
        )

    # Term-level pass: only when the flattener retained memory payload
    # text (additive v1 enrichment). The term list is the OBSERVED group
    # descriptors (>= 3 chars, word-boundary matched), so the screen
    # reads whether protected descriptors cross groups through memory;
    # it is honestly NOT a general PII sweep.
    terms: Dict[str, Any] = {"available": False}
    has_text = "memory_read_text" in df.columns and df["memory_read_text"].notna().any()
    if has_text:
        import re

        descriptors = {
            str(g): re.compile(r"\b" + re.escape(str(g)) + r"\b", re.IGNORECASE)
            for g in adequate
            if len(str(g).strip()) >= 3
        }
        if len(descriptors) < 2:
            terms = {
                "available": False,
                "reason": (
                    "Memory payload text is present, but the observed "
                    "group labels are too short to match as words, so "
                    "no descriptor screen can run on them."
                ),
            }
        else:
            read_txt = df["memory_read_text"].fillna("").astype(str)
            per_group_terms: List[Dict[str, Any]] = []
            worst_share, worst_group, worst_from = 0.0, None, None
            worst_exposed = 0
            for g in adequate:
                g = str(g)
                mask = (s.astype(str) == g).to_numpy()
                texts = read_txt.to_numpy()[mask]
                reading = [t for t in texts if t]
                if not reading:
                    per_group_terms.append(
                        {
                            "group": g,
                            "episodesWithPayload": 0,
                            "crossGroupDescriptorShare": 0.0,
                            "descriptorsSeen": [],
                        }
                    )
                    continue
                seen: Dict[str, int] = {}
                exposed_n = 0
                for t in reading:
                    hit = False
                    for other, rx in descriptors.items():
                        if other == g:
                            continue
                        if rx.search(t):
                            seen[other] = seen.get(other, 0) + 1
                            hit = True
                    if hit:
                        exposed_n += 1
                share = exposed_n / len(reading)
                per_group_terms.append(
                    {
                        "group": g,
                        "episodesWithPayload": len(reading),
                        "crossGroupDescriptorShare": share,
                        "exposedEpisodes": exposed_n,
                        "descriptorsSeen": sorted(seen.keys()),
                    }
                )
                if share > worst_share:
                    worst_share, worst_group = share, g
                    worst_from = sorted(seen.keys())
                    worst_exposed = exposed_n
            terms = {
                "available": True,
                "descriptorBasis": (
                    "the observed group labels of the adequately "
                    "sampled groups, word-boundary matched"
                ),
                "perGroup": per_group_terms,
            }
            # Fire only when enough episodes actually matched (a single
            # word-boundary hit is not evidence of systemic contamination),
            # and cap at "warn": this is a heuristic descriptor scan, not a
            # significance test, so it is surfaced for review but must not
            # by itself block deployment (audit fix mem-1). statisticalTest
            # is None so the verdict roll-up's significance gate keeps it
            # non-escalating.
            if (
                worst_group is not None
                and worst_share > 0
                and worst_exposed >= _MEM_TERM_MIN_EPISODES
            ):
                findings.append(
                    _finding(
                        "memory_term_contamination",
                        "warn",
                        None,
                        (
                            f"Protected descriptors cross groups through the "
                            f"memory layer: {worst_share * 100:.0f}% of "
                            f"'{worst_group}' episodes that read memory "
                            f"payloads ({worst_exposed} episode(s)) saw another "
                            f"group's descriptor "
                            f"({', '.join(worst_from or [])}) in the carried "
                            "context. Memory-carried demographic information "
                            "can steer downstream behaviour even when the "
                            "current episode never received it directly. This is "
                            "a descriptor-match screen, surfaced for review."
                        ),
                        None,
                    )
                )

    section = {
        "available": True,
        "perGroup": per_group,
        "relianceChi2P": chi2_p,
        "exposure": exposure,
        "termScreen": terms,
        "contentAudited": bool(terms.get("available")),
        "notCovered": (
            "The term screen matches the observed group descriptors "
            "only; a general PII or semantic sweep of memory payloads "
            "is full-assessment work."
            if terms.get("available")
            else "Term-level contamination (protected or personal terms "
            "flowing between users or episodes through memory): the "
            "export carried no memory payload text, so only the "
            "count-based screen ran."
        ),
        "method": (
            "Count-based screen on the trace contract's memory "
            "read/write fields (per-group reliance shares with a "
            "chi-square comparison, timestamp-ordered cross-group "
            "carry-over exposure)"
            + (
                ", plus a term-level descriptor scan over the retained "
                "memory payload text (bounded per episode)."
                if terms.get("available")
                else "."
            )
        ),
    }
    return (section, findings)


def _resolve_trace_frame(
    df: pd.DataFrame, inputs: Dict[str, Any], action_col: str, group_col: str
) -> Tuple[pd.DataFrame, str, str, Dict[str, Any]]:
    """G-10 ingestion hook, BEFORE any column use: when the frame does
    not carry the expected group+tool columns but is a raw span or
    observation export, flatten it via traces.maybe_flatten_spans and
    re-point the columns at the canonical flattened names. Returns
    (frame, action_col, group_col, ingestion_disclosure); the frame and
    columns are unchanged whenever flattening does not apply."""
    hint = None
    if isinstance(inputs, dict):
        h = inputs.get("_trace_ingestion")
        if isinstance(h, dict) and h.get("form"):
            hint = {"form": str(h.get("form")), "notes": str(h.get("notes") or "")}
    if action_col in df.columns and group_col in df.columns:
        return (df, action_col, group_col, hint or dict(_INGESTION_PER_EPISODE))
    from vfairness.operations.pulse.traces import maybe_flatten_spans

    flat = maybe_flatten_spans(df)
    if flat is None:
        return (
            df,
            action_col,
            group_col,
            hint
            or {
                "form": "per_episode",
                "notes": (
                    "The expected columns are missing and the frame "
                    "is not a recognizable span/observation export; "
                    "it was used as supplied."
                ),
            },
        )
    meta = dict(flat.attrs.get("ingestion") or {})
    act = "tool" if "tool" in flat.columns else action_col
    grp = "group" if "group" in flat.columns else group_col
    return (
        flat,
        act,
        grp,
        {
            "form": str(meta.get("form") or "otel_spans"),
            "notes": str(
                meta.get("notes") or "Flattened a raw span export into the per-episode trace table."
            ),
        },
    )


def _probe(
    df: pd.DataFrame,
    inputs: Dict[str, Any],
    action_col: str,
    group_col: str,
    domain: str,
    jurisdiction: str,
    ingestion: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # AN EPISODE WITH NO ACTING GROUP IS NOT A DEMOGRAPHIC GROUP (grade wave
    # G06, 2026-09-30). This line was ``.astype("string").fillna("missing")``,
    # which turned the episodes carrying no record of who was acting into a
    # group and then compared it with the real ones. Measured before, 60
    # episodes through ``ingest_and_run_pulse``, half labelled "A" and half whose
    # ``user.group`` trace attribute is absent, the absent half failing more
    # often:
    #
    #   "Adverse opinion: material fairness defects make this system unfit to
    #    deploy as-is. Trajectory gap on 'outcome': A averages 0.80 vs 0.30 for
    #    missing (outcome rate; permutation test, BH-adjusted p 0.001998,
    #    Cohen's d 1.14)."
    #
    # byte for byte the sentence a genuine second group "B" produces on the same
    # data. An adverse opinion is a claim about a protected class, and an absent
    # attribute is not one.
    #
    # THE SAME RULE THE REST OF THIS PACKAGE USES, not a new one: absence is
    # ``pd.isna`` plus a whitespace-only string, the library's
    # ``missing_strategy='exclude'`` default, and the SAME judgement
    # ``orchestrator.run_pulse`` applies once above its own sections (this
    # pathway returns from ``run_pulse`` before that block, which is why it
    # needs the rule here rather than inheriting it). A literal level somebody
    # chose, including one spelled "missing", is left alone.
    #
    # It never empties the table: with no episode carrying a group at all the
    # rows are KEPT and the single-group path below refuses the run, which is
    # already the honest answer.
    _s_raw = df[group_col].astype("string")
    _absent = _s_raw.isna() | _s_raw.map(lambda v: isinstance(v, str) and not v.strip()).fillna(
        False
    ).astype(bool)
    n_no_group = int(_absent.sum())
    if n_no_group and n_no_group < len(df):
        df = df.loc[(~_absent).to_numpy()].reset_index(drop=True)
        s = df[group_col].astype("string")
    else:
        s = _s_raw.fillna("missing")
        if not n_no_group:
            n_no_group = 0
    vc = s.value_counts()
    per_group = {str(g): int(n) for g, n in vc.items()}
    adequate = [str(g) for g in vc.index if int(vc[g]) >= SAMPLE_FLOOR]
    excluded = [g for g in per_group if g not in adequate]
    assessable = len(adequate) >= 2

    if not excluded:
        note = f"All groups meet the n >= {SAMPLE_FLOOR} floor."
    elif assessable:
        note = (
            "Excluded below-floor group(s): "
            + ", ".join(f"{g} (n={per_group[g]})" for g in excluded)
            + f". The floor is {SAMPLE_FLOOR} episodes per group."
        )
    else:
        note = (
            f"No two groups reach the n >= {SAMPLE_FLOOR} floor "
            "("
            + ", ".join(f"{g} (n={per_group[g]})" for g in per_group)
            + "); nothing is assessable."
        )
    if n_no_group:
        # The could-not-check has to be where the reader of the sample block
        # looks, beside the per-group counts it changed.
        note += (
            f" {n_no_group} episode(s) carry no recorded acting group (blank, null "
            "or missing) and were EXCLUDED from every comparison: they are not a "
            "demographic group, so nothing here says they were served fairly or "
            "unfairly. They were not assessed."
        )
    sample_adequacy = {
        "adequate": bool(per_group) and not excluded,
        "perGroup": per_group,
        "floor": SAMPLE_FLOOR,
        #: Episodes dropped because no acting group was recorded. 0 on a complete
        #: export, so a clean run reads exactly as it did before.
        "episodesExcludedNoGroup": n_no_group,
        "note": note,
    }

    findings: List[Dict[str, Any]] = []
    comparisons_out: List[Dict[str, Any]] = []
    omnibus: Dict[str, Any] = {
        "available": False,
        "significant": False,
        "pValue": None,
        "chiSquare": None,
        "dof": None,
    }
    # A missing action (an episode that invoked no tool) is made EXPLICIT before
    # the lists are built. `.astype(str)` used to do it by accident and only on
    # pandas 2, where a missing value becomes the text "nan". On pandas 3 the
    # column is StringDtype(na_value=nan), astype(str) leaves NaN a float, and
    # the sorted() over tool names below raised TypeError, which _safe turned
    # into "The agent-trace probe could not analyze these traces". Reproduced
    # 2026-10-01 with pandas 3.0.6 on an OTel export where some episodes call
    # no tool: every such export was reported as unassessable.
    _acts = df[action_col]
    _act_text = _acts.astype(object).where(_acts.notna().to_numpy(), NO_ACTION_LABEL).astype(str)
    tool_lists = {g: _act_text[(s == g).to_numpy()].tolist() for g in adequate}

    # G-20: raw trace rows behind each comparison, keyed by comparison
    # identity; family_size feeds the auditTrail.probe envelope.
    evidence_by_comp: Dict[int, List[Dict[str, Any]]] = {}
    family_size = 0
    tool_family: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    tool_not_assessed: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    action_family: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    if assessable:
        omnibus = _safe(lambda: _omnibus_block(adequate, tool_lists), omnibus)
        from vfairness.agents import ActionBiasAnalyzer, ToolBiasAuditor

        for spec in _build_comparisons(adequate, tool_lists):
            av, bv = spec["toolsA"], spec["toolsB"]
            ga, gb = spec["groupA"], spec["groupB"]
            tb = _quiet(
                lambda av=av, bv=bv: ToolBiasAuditor().analyze_tool_calls(
                    [{"tool": t} for t in av], [{"tool": t} for t in bv]
                ),
                [],
            )
            tools = []
            for r in tb or []:
                rd = r.to_dict() if hasattr(r, "to_dict") else dict(r or {})
                tools.append(rd)

            ca, cb = Counter(av), Counter(bv)
            pair_tools = sorted(set(ca) | set(cb))
            pair_table = [[ca.get(t, 0) for t in pair_tools], [cb.get(t, 0) for t in pair_tools]]
            # _safe(lambda: _cramers_v(pair_table), 0.0) used to stand here and
            # laundered BOTH a refusal and a raise into a confident zero.
            # _quiet_recorded keeps the probe's contract that no Python warning
            # escapes it while KEEPING the sentence that says why V could not be
            # computed, and its failure value is nan, which no band test can
            # mistake for a small measured effect.
            v, v_notes = _quiet_recorded(lambda: _cramers_v(pair_table), float("nan"))
            v_measured = is_measured(v)
            severity = _severity_from_v(v if v_measured else None)
            could_not_check: List[str] = []
            if severity is None:
                # Never "info": that word means measured and small. The only
                # honest word here is the one the assurance grader ranks as
                # unassessed (compliance._UNASSESSED_SEVERITIES), so the
                # comparison is reported as its own state and is counted
                # neither as a clean result nor as a minor one.
                severity = "insufficient_data"
                could_not_check.append(
                    f"Cramer's V could not be computed for {ga} vs {gb}: "
                    f"{len(pair_tools)} distinct action(s) were observed across "
                    f"the two groups, so the contingency table has fewer than "
                    f"two levels on a side and the effect size is undefined. "
                    f"This is an unmeasured effect, not a small one."
                )

            best_tool, best_diff = None, 0.0
            for t in pair_tools:
                diff = (ca.get(t, 0) / len(av)) - (cb.get(t, 0) / len(bv))
                if best_tool is None or abs(diff) > abs(best_diff):
                    best_tool, best_diff = t, diff
            largest = None
            if best_tool is not None:
                # The sibling of the Cramer's V default above: a (0.0, 0.0)
                # fallback publishes the TIGHTEST POSSIBLE interval around no
                # difference for a resampling that never ran. The bootstrap
                # cannot raise at this call site (both groups clear the n >= 20
                # floor, and it only fails on an empty side), so this is a
                # standing guard rather than a live defect, but the default it
                # replaces was a measurement either way.
                lo, hi = _safe(
                    lambda: _bootstrap_rate_diff_ci(av, bv, best_tool),
                    (float("nan"), float("nan")),
                )
                largest = {
                    "tool": best_tool,
                    "observed": float(best_diff),
                    "ciLow": _finite_or_none(lo),
                    "ciHigh": _finite_or_none(hi),
                    "resamples": N_BOOTSTRAP,
                    "seed": RESAMPLE_SEED,
                }

            # Overall action-distribution test between the two sides of
            # the comparison: ActionBiasAnalyzer.analyze_delegation is
            # the library's categorical-distribution comparison (full
            # chi-square on the action x side table).
            ab = _quiet(
                lambda av=av, bv=bv: ActionBiasAnalyzer(alpha=ALPHA).analyze_delegation(av, bv),
                None,
            )
            abd = ab if isinstance(ab, dict) else {}

            comp = {
                "groupA": ga,
                "groupB": gb,
                "comparisonMode": spec["mode"],
                "actionColumn": action_col,
                "nA": len(av),
                "nB": len(bv),
                "tools": tools,
                # None, not nan: an unmeasured quantity is reported as null,
                # never as a number a reader would compare or format.
                "cramersV": _finite_or_none(v),
                "cramersVMeasured": bool(v_measured),
                "severity": severity,
                "largestRateDiff": largest,
                "actionDistribution": abd,
            }
            if could_not_check:
                comp["couldNotCheck"] = could_not_check
            if v_notes:
                comp["effectSizeNotes"] = v_notes
            comparisons_out.append(comp)
            # G-20: first rows per side (table order, seed-free) as the
            # raw evidence behind any finding fired on this comparison.
            evidence_by_comp[id(comp)] = _evidence_rows(ga, av, "tool") + _evidence_rows(
                gb, bv, "tool"
            )
            # DESIGN POWER PER TOOL (readiness 6, 2026-09-10). Every tool row
            # took a slot in the Benjamini-Hochberg family, including tools that
            # could not have produced a finding under any demographics: with
            # n_A = n_B = 60, a tool used 5 times in total floors at p = 0.0573
            # and one used once floors at 1.0. Those slots are paid for by the
            # tests that CAN fire. Measured 2026-09-10 on a real 11.7-percent
            # versus 0 escalation disparity (raw p = 0.0130): reported at BH
            # 0.0390 with no rare tools in the family, SUPPRESSED at BH 0.0519
            # with one rare tool added and 0.0649 with two. The finding turned
            # on tools nobody could have tested. Real agent traces always carry
            # a long tail of rarely-used tools.
            #
            # Excluding them from the family is the same move `_testable` in
            # evaluation/vfairness_metrics/_statistics.py already makes for a
            # p-value that is not finite, and for the same reason: an untestable
            # hypothesis is not a hypothesis that passed.
            for rd in tools:
                name = rd.get("tool_name")
                k_uses = int(ca.get(name, 0)) + int(cb.get(name, 0))
                floor = _tool_p_floor(len(av), len(bv), k_uses)
                detectable, note = detectability(floor, n_family=1, alpha=ALPHA)
                rd["nUses"] = k_uses
                rd["minAttainableP"] = floor
                rd["detectable"] = detectable
                if detectable is True:
                    rd["assessed"] = True
                    tool_family.append((comp, rd))
                else:
                    # Never "not significant": this row is a could-not-check and
                    # says so in the report, with the reason a reader can act on.
                    rd["assessed"] = False
                    rd["p_adjusted"] = None
                    rd["family_significant"] = None
                    rd["is_significant"] = None
                    rd["notAssessedReason"] = (
                        f"'{name}' was used {k_uses} time(s) across "
                        f"{len(av)} + {len(bv)} episodes. {note}"
                    )
                    tool_not_assessed.append((comp, rd))
            if abd.get("overall_p_value") is not None:
                action_family.append((comp, abd))

        # Family-wise BH-FDR across ALL per-tool p-values from ALL
        # comparisons; findings fire only when BOTH the adjusted p-value
        # and the omnibus test are significant.
        gate = bool(omnibus.get("significant"))
        # The family is the tests that COULD fire; the rest are reported as
        # not assessed, above, and counted separately in the audit trail.
        family_size = len(tool_family) + len(action_family)
        for (comp, rd), adj in zip(
            tool_family, _bh_adjust([float(rd.get("p_value", 1.0)) for _, rd in tool_family])
        ):
            rd["p_adjusted"] = float(adj)
            rd["family_significant"] = bool(adj < ALPHA and gate)
            if rd["family_significant"]:
                fd = _finding(
                    "agent_tool_bias",
                    comp["severity"],
                    group_col,
                    (
                        f"The agent invokes '{rd.get('tool_name')}' at "
                        f"{float(rd.get('invocation_rate_a', 0)):.0%} for "
                        f"{comp['groupA']} vs "
                        f"{float(rd.get('invocation_rate_b', 0)):.0%} for "
                        f"{comp['groupB']} (BH-adjusted p {adj:.4g}, omnibus "
                        f"gated; {_v_clause(comp)}). "
                        f"Different tools/paths per demographic is an "
                        f"agentic fairness violation."
                    ),
                    {
                        "test": "per_tool_chi_square_or_fisher",
                        "pValue": float(rd.get("p_value", 1.0)),
                        "pAdjusted": float(adj),
                        "correction": "benjamini_hochberg",
                        "omnibusPValue": omnibus.get("pValue"),
                        "effectSize": {
                            "cramersV": comp["cramersV"],
                            "cramersVMeasured": comp["cramersVMeasured"],
                        },
                    },
                )
                # G-20: the literal trace rows behind this comparison.
                fd["evidence"] = evidence_by_comp.get(id(comp), [])
                findings.append(fd)
        for (comp, abd), adj in zip(
            action_family,
            _bh_adjust([float(abd.get("overall_p_value", 1.0)) for _, abd in action_family]),
        ):
            abd["p_adjusted"] = float(adj)
            abd["family_significant"] = bool(adj < ALPHA and gate)
            if abd["family_significant"]:
                fd = _finding(
                    "agent_action_bias",
                    comp["severity"],
                    group_col,
                    (
                        f"The overall action distribution differs "
                        f"significantly between {comp['groupA']} and "
                        f"{comp['groupB']} (chi-square, BH-adjusted p "
                        f"{adj:.4g}; {_v_clause(comp)}); "
                        f"the agent behaves differently by demographic."
                    ),
                    {
                        "test": "chi_square_contingency",
                        "pValue": float(abd.get("overall_p_value", 1.0)),
                        "pAdjusted": float(adj),
                        "correction": "benjamini_hochberg",
                        "omnibusPValue": omnibus.get("pValue"),
                        "effectSize": {
                            "cramersV": comp["cramersV"],
                            "cramersVMeasured": comp["cramersVMeasured"],
                        },
                    },
                )
                # G-20: the literal trace rows behind this comparison.
                fd["evidence"] = evidence_by_comp.get(id(comp), [])
                findings.append(fd)

    delegation, deleg_findings, deleg_col = _safe(
        lambda: _delegation_section(df, s, adequate, action_col, group_col),
        (
            {
                "available": False,
                "notApplicable": True,
                "reason": "no route/delegation column in the traces",
            },
            [],
            None,
        ),
    )
    findings.extend(deleg_findings)

    trajectory, traj_findings = _safe(
        lambda: _trajectory_section(
            df,
            s,
            adequate,
            exclude=(action_col, group_col, deleg_col),
            group_col=group_col,
            action_col=action_col,
        ),
        (
            {
                "available": False,
                "notApplicable": True,
                "reason": "No step-count or outcome column in the traces.",
                "perMetric": [],
            },
            [],
        ),
    )
    findings.extend(traj_findings)

    # G-21: system-vs-component amplification + groupthink convergence
    # (crash-safe, additive; each degrades to an honest notApplicable).
    _special_cols = (action_col, group_col, deleg_col, "trace_id")
    amplification, amp_findings = _safe(
        lambda: _amplification_section(df, s, adequate, _special_cols, group_col),
        (
            {
                "available": False,
                "notApplicable": True,
                "reason": _AMPLIFICATION_NA_REASON,
                "perComparison": [],
            },
            [],
        ),
    )
    findings.extend(amp_findings)

    groupthink, gt_findings = _safe(
        lambda: _groupthink_section(df, _special_cols),
        (
            {
                "available": False,
                "notApplicable": True,
                "reason": "Groupthink convergence needs at least two "
                "per-agent opinion columns in the traces.",
            },
            [],
        ),
    )
    findings.extend(gt_findings)

    # G-37: temporal disparity dynamics via TemporalTracker.
    temporal_dynamics, td_findings = _safe(
        lambda: _temporal_section(df, s, adequate, action_col, _special_cols),
        (
            {
                "available": False,
                "notApplicable": True,
                "reason": "No timestamp/ordering column in the traces.",
                "perWindow": [],
            },
            [],
        ),
    )
    findings.extend(td_findings)

    # G-43: memory-contamination screen on the reserved trace fields;
    # honest notApplicable when the export carried no memory events.
    memory, mem_findings = _safe(
        lambda: _memory_section(df, s, adequate),
        (
            {
                "available": False,
                "notApplicable": True,
                "reason": "The memory screen could not be computed.",
            },
            [],
        ),
    )
    findings.extend(mem_findings)

    if assessable:
        from vfairness.operations.reporting import build_assurance_verdict

        # The verdict has to know WHAT was assessed. With per_variable=[] the
        # BGL-3 guard in build_assurance_verdict (correctly) reads "nothing was
        # assessed" and returns a Disclaimer, so an adequately sampled agent
        # whose tool choice showed NO bias was reported as "Insufficient
        # assessable data": a measured null shown as a could-not-check.
        # Measured 2026-10-01: 6 groups x 120 episodes, omnibus p 0.996,
        # headline "Insufficient assessable data". The group attribute is now
        # passed as assessed, with its measured gap (the largest difference in
        # any one tool's invocation rate between adequate groups) and the
        # omnibus decision as its significance.
        def _largest_tool_rate_gap() -> Optional[float]:
            tools = sorted({t for g in adequate for t in tool_lists[g]})
            gaps = []
            for t in tools:
                rates = [
                    tool_lists[g].count(t) / len(tool_lists[g]) for g in adequate if tool_lists[g]
                ]
                if len(rates) >= 2:
                    gaps.append(max(rates) - min(rates))
            return float(max(gaps)) if gaps else None

        # Declared only when the omnibus test actually RAN: a frame where every
        # episode used one tool cannot be tested, and that stays a
        # could-not-check (Disclaimer). The entry carries scope only:
        # findingsFromBias tells the verdict that this attribute's findings are
        # the probe's own omnibus-gated, corrected ones in `bias`, so the raw
        # gap is shown but never turned into a second, uncorrected finding.
        agent_per_variable = (
            [
                {
                    "attribute": str(group_col),
                    "assessable": True,
                    "gap": _safe(_largest_tool_rate_gap, None),
                    "significant": omnibus.get("significant") is True,
                    "fourFifthsRatio": None,
                    "measure": "largest per-tool invocation-rate gap between groups",
                    "findingsFromBias": True,
                }
            ]
            # dof > 0: with one tool (or one group) the table has no degrees of
            # freedom, the "test" cannot disagree, and the probe itself notes that
            # no selection disparity is definable.
            if omnibus.get("available") is True and (omnibus.get("dof") or 0) > 0
            else []
        )

        assurance = _safe(
            lambda: build_assurance_verdict(
                schema={"pii_leakage": [], "mismatches": [], "refuse": False},
                per_variable=agent_per_variable,
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
                "oneLineVerdict": "Agent assurance verdict could not be assembled.",
                "findings": [],
                "recommendations": [],
                "metricsDeferred": {},
                "auditTrail": {},
            },
        )
    else:
        # Honest low-n disclaimer: every (or all but one) group is under
        # the sample floor, so nothing is assessable. Never a pass tone.
        assurance = {
            "overall": "Disclaimer",
            "blocksDeployment": False,
            "oneLineVerdict": (
                "We cannot form a fairness opinion on these traces: "
                + note
                + " Collect at least "
                + f"{SAMPLE_FLOOR} episodes per demographic group and "
                "re-run."
            ),
            "findings": [],
            "recommendations": [
                {
                    "id": "REC-001",
                    "priority": 1,
                    "action": (
                        f"Collect at least {SAMPLE_FLOOR} episodes per "
                        f"demographic group and re-run the triage."
                    ),
                    "addressesFindings": [],
                    "routeTo": "navigator_full_assessment",
                    "vfairnessFunction": "run_pulse",
                }
            ],
            "metricsDeferred": {"metrics": [], "rationale": ""},
            "auditTrail": {},
        }

    # G-20: full reproducibility envelope on the audit trail (additive,
    # applied AFTER the assurance verdict is built so the verdict logic
    # is untouched). The tool and action p-values are BH-corrected as TWO
    # SEPARATE families (not one joint family), so both sizes are disclosed
    # for exact reproducibility; familySize is the combined count (audit
    # fix agp-1: the old single-family label understated the actual, less
    # strict correction that ran).
    audit = assurance.get("auditTrail")
    if not isinstance(audit, dict):
        audit = {}
        assurance["auditTrail"] = audit
    audit["probe"] = {
        "groupsCompared": list(adequate),
        "comparisons": len(comparisons_out),
        "familySize": family_size,
        "toolFamilySize": len(tool_family),
        "toolTestsNotAssessed": len(tool_not_assessed),
        "actionFamilySize": len(action_family),
        "correctionScope": "tool and action corrected as separate BH families; "
        "per-tool tests whose Fisher p-floor cannot reach alpha are excluded "
        "from the family and reported as not assessed",
        "omnibus": {"chi2": omnibus.get("chiSquare"), "p": omnibus.get("pValue")},
        "sampleFloor": SAMPLE_FLOOR,
        "seeds": {"bootstrap": RESAMPLE_SEED, "permutation": RESAMPLE_SEED},
    }

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
    n_findings = len(findings)
    # "No significant bias detected" is a claim about tests that COULD have
    # detected something. Measured 2026-09-10, this line was printed while a
    # deny_loan row showed rate_a=0.0667 against rate_b=0.0 at p=0.1187, which
    # is the p-FLOOR for that tool's usage count, so no data could have produced a
    # finding on it. The count of tests that could not fire now travels with
    # the sentence.
    not_assessed_clause = (
        f" {len(tool_not_assessed)} per-tool test(s) could not fire at any data "
        f"(too few uses of that tool to reach p<{ALPHA:g}) and are reported as "
        f"not assessed, not as absence of bias."
        if tool_not_assessed
        else ""
    )
    summary = (
        f"{n_findings} agentic-bias finding(s) after omnibus gating "
        f"and family-wise Benjamini-Hochberg correction." + not_assessed_clause
        if n_findings
        else (
            "No significant tool/action selection bias detected across "
            "groups (omnibus-gated, family-wise corrected)." + not_assessed_clause
            if assessable
            else note
        )
    )
    return {
        "success": True,
        "data": {
            "sourceKind": "agent_traces",
            "assurance": assurance,
            "verdict": {
                "tone": tone,
                "headline": assurance.get("oneLineVerdict", ""),
                "summary": "Agentic fairness triage (vfairness "
                "ToolBiasAuditor / ActionBiasAnalyzer / "
                "DelegationRoutingAuditor).",
                "ranked": [],
            },
            "agent": {
                "available": bool(comparisons_out),
                "perComparison": comparisons_out,
                "omnibus": omnibus,
                "sampleAdequacy": sample_adequacy,
                "trajectory": trajectory,
                "delegation": delegation,
                # G-10: how the traces were ingested (per-episode
                # table vs flattened OTel/Langfuse span export).
                "ingestion": (dict(ingestion) if ingestion else dict(_INGESTION_PER_EPISODE)),
                # G-21 + G-37: multi-agent depth and temporal
                # dynamics; honest notApplicable when the traces do
                # not carry the needed columns.
                "amplification": amplification,
                "groupthink": groupthink,
                "temporalDynamics": temporal_dynamics,
                # G-43: count-based memory-contamination screen on the
                # reserved memory_reads/memory_writes fields.
                "memory": memory,
                "summary": summary,
            },
            # Universal contract: name the framework that applied to this
            # agent-trace triage. Resolved lazily so this module avoids a
            # hard import-time dependency on the orchestrator's helper.
            "legalFramework": (
                lambda: __import__(
                    "vfairness.operations.pulse.orchestrator",
                    fromlist=["build_legal_framework_block"],
                ).build_legal_framework_block(domain, jurisdiction)
            )(),
            "scope": (
                lambda: __import__(
                    "vfairness.operations.pulse.orchestrator",
                    fromlist=["build_scope_block"],
                ).build_scope_block(
                    "agent_traces",
                    covered=[
                        "Tool-selection and action-distribution disparity across "
                        "ALL adequately sampled demographic groups (every pair up "
                        "to 5 groups, one-vs-rest beyond), gated on an omnibus "
                        "group x tool chi-square with Benjamini-Hochberg "
                        "family-wise correction across all comparisons",
                        "Effect sizes (Cramér's V) drive finding severity; "
                        "bootstrap confidence intervals on the largest per-tool "
                        "selection-rate difference",
                        "Trajectory shape (episode step counts / outcome rates) "
                        "via permutation tests when such columns exist",
                        "Delegation / routing disparity across groups when a "
                        "route or delegation column exists "
                        "(DelegationRoutingAuditor)",
                        "Raw OpenTelemetry GenAI span exports and Langfuse-style "
                        "exports are flattened into the per-episode table "
                        "deterministically; the ingestion form is disclosed in "
                        "agent.ingestion",
                        "System-vs-component amplification "
                        "(EmergentBiasDetector) when the traces carry per-agent "
                        "AND system output columns; groupthink convergence "
                        "(GroupthinkDetector) when two or more per-agent "
                        "opinion columns plus an ordering column exist",
                        "Temporal disparity dynamics (TemporalTracker "
                        "Mann-Kendall / CUSUM / EWMA) when the traces carry a "
                        "timestamp or ordering column",
                    ]
                    + (
                        [
                            "Memory-contamination screen (count-based, G-43): "
                            "per-group memory-reliance shares and timestamp-ordered "
                            "cross-group carry-over exposure on the trace "
                            "contract's memory read/write fields"
                            + (
                                ", plus a term-level descriptor scan over the "
                                "retained memory payload text"
                                if memory.get("contentAudited")
                                else ""
                            ),
                        ]
                        if memory.get("available")
                        else []
                    ),
                    not_covered=(
                        [
                            "Memory-contamination screening: "
                            + str(memory.get("reason") or "the screen did not run on these traces"),
                        ]
                        if not memory.get("available")
                        else [
                            (
                                "A general PII or semantic sweep of memory payloads "
                                "(the term screen matches observed group descriptors "
                                "only)"
                            )
                            if memory.get("contentAudited")
                            else "Term-level memory contamination (protected or personal "
                            "terms flowing between users or episodes): the export "
                            "carried no memory payload text, so only the "
                            "count-based screen ran",
                        ]
                    )
                    + [
                        "Correspondence probing: UNTESTED in triage",
                        "Agent capability benchmarks say nothing about fairness",
                    ],
                    power_note=(
                        "Findings describe the supplied traces only; "
                        "trace volume per group bounds what is "
                        "detectable. Groups under the n >= "
                        f"{SAMPLE_FLOOR} floor are excluded and named "
                        "in agent.sampleAdequacy."
                    ),
                )
            )(),
            "perVariable": [],
            "metrics": [],
            "bias": findings,
            "proxies": {
                "available": False,
                "notApplicable": True,
                "reason": "Proxy reconstruction applies to tabular features, not trace tables.",
                "proxies": [],
                "chains": [],
            },
            "statistical": {
                "available": False,
                "notApplicable": True,
                "reason": "The tabular robustness battery does not apply to trace tables.",
                "perAttribute": [],
            },
            "intersectional": {"skipped": True, "reason": "Agent trace triage."},
            "causal": {"nodes": [], "edges": [], "overview": "Not applicable to agent traces."},
            "recommendedDefinition": {},
            "interventions": [],
            "nextStep": (
                "Convert to a full assessment for the deep agent "
                "battery (RAG retrieval bias, correspondence "
                "probing, term-level memory-contamination analysis "
                "on the raw event payloads)."
            ),
        },
    }


def agent_probe_pulse(
    df: pd.DataFrame,
    inputs: Dict[str, Any],
    action_col: str,
    group_col: str,
    domain: str,
    jurisdiction: str,
) -> Dict[str, Any]:
    """Tool/action-selection bias across ALL demographic groups in agent
    traces, plus trajectory shape, delegation routing, multi-agent depth
    (amplification / groupthink) and temporal dynamics when the traces
    carry the columns. Raw OTel/Langfuse span exports are flattened
    first (G-10) and the ingestion form is disclosed. Returns a
    PulseResult ``data`` payload (assurance + agent section). Never
    raises: any unexpected failure degrades to an honest Disclaimer
    payload instead of falling through the router."""
    # G-10 ingestion hook BEFORE any column use: flatten raw span or
    # observation exports; a per-episode table passes through untouched.
    resolved = _safe(lambda: _resolve_trace_frame(df, inputs, action_col, group_col), None)
    if resolved is None:
        frame, act, grp = df, action_col, group_col
        ingestion = {
            "form": "per_episode",
            "notes": ("Ingestion resolution failed; the frame was used as supplied."),
        }
    else:
        frame, act, grp, ingestion = resolved
    out = _safe(lambda: _probe(frame, inputs, act, grp, domain, jurisdiction, ingestion), None)
    if out is not None:
        return out
    return {
        "success": True,
        "data": {
            "sourceKind": "agent_traces",
            "assurance": {
                "overall": "Disclaimer",
                "blocksDeployment": False,
                "oneLineVerdict": "The agent-trace probe could not analyze these traces.",
                "findings": [],
                "recommendations": [],
                "metricsDeferred": {},
                "auditTrail": {},
            },
            "verdict": {
                "tone": "disclaimer",
                "headline": "The agent-trace probe could not analyze these traces.",
                "summary": "Agentic fairness triage.",
                "ranked": [],
            },
            "agent": {
                "available": False,
                "perComparison": [],
                "omnibus": {"available": False, "significant": False},
                "sampleAdequacy": {
                    "adequate": False,
                    "perGroup": {},
                    "floor": SAMPLE_FLOOR,
                    "note": "The probe failed before sampling could be assessed.",
                },
                "trajectory": {
                    "available": False,
                    "notApplicable": False,
                    "reason": "probe failure",
                    "perMetric": [],
                },
                "delegation": {
                    "available": False,
                    "notApplicable": False,
                    "reason": "probe failure",
                },
                "ingestion": dict(ingestion),
                "amplification": {
                    "available": False,
                    "notApplicable": False,
                    "reason": "probe failure",
                    "perComparison": [],
                },
                "groupthink": {
                    "available": False,
                    "notApplicable": False,
                    "reason": "probe failure",
                },
                "temporalDynamics": {
                    "available": False,
                    "notApplicable": False,
                    "reason": "probe failure",
                    "perWindow": [],
                },
                "summary": "The agent-trace probe failed; no comparison was run.",
            },
            "legalFramework": {},
            "scope": {},
            "perVariable": [],
            "metrics": [],
            "bias": [],
            "proxies": {
                "available": False,
                "notApplicable": True,
                "reason": "Proxy reconstruction applies to tabular features, not trace tables.",
                "proxies": [],
                "chains": [],
            },
            "statistical": {
                "available": False,
                "notApplicable": True,
                "reason": "The tabular robustness battery does not apply to trace tables.",
                "perAttribute": [],
            },
            "intersectional": {"skipped": True, "reason": "Agent trace triage."},
            "causal": {"nodes": [], "edges": [], "overview": "Not applicable to agent traces."},
            "recommendedDefinition": {},
            "interventions": [],
            "nextStep": (
                "Check that the trace table has one row per "
                "episode with a group column and a tool/action "
                "column, then re-run."
            ),
        },
    }
