"""
Framework-agnostic capture harness for multi-agent fairness testing.

The other analyzers in ``vfairness.multi_agent`` operate on numeric
arrays (per-agent outputs, per-turn outcomes, routing decisions).
Producing those arrays from a real multi-agent run typically requires
glue code per agent framework (autogen, crewai, langgraph, raw
function-calling). This module provides a *minimal* capture layer that
sits between user code and the analyzers.

Design:
- ``MultiAgentRunHarness`` is intentionally framework-agnostic. The
  user calls ``record_*`` methods from inside their own
  callbacks/hooks; the harness aggregates traces and produces the
  exact input shapes the analyzers expect.
- The harness is also a context manager so trace collection is scoped
  and exception-safe.

Inspirations:
- HELM (Liang et al. 2022): separation between scenario, runner, and
  metric.
- AutoGen tracing API (Wu et al. 2023): per-agent hook surface.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from vfairness._not_assessed import warn_not_assessed
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


#: The exact strings ``str()`` produces from an absence sentinel, and nothing
#: else. Kept as literals rather than derived, so the set a reader sees here is
#: the set the function tests.
#:
#:   str(None) -> 'None'                      str(pd.NA) -> '<NA>'
#:   str(float('nan')) -> 'nan'               str(pd.NaT) -> 'NaT'
#:   str(np.float64('nan')) -> 'nan'          str(np.datetime64('NaT')) -> 'NaT'
#:   str(np.ma.masked) -> '--'
#:
#: DELIBERATELY NOT HERE: 'NA', 'N/A', 'null', 'NULL', 'missing', 'unknown'. Those
#: are not what ``str()`` mints from an absent value, and each of them is a label a
#: caller can legitimately mean ('NA' is a region). Refusing them would be the
#: over-correction, and this predicate is the ONLY thing standing between a caller
#: and a refused run, so it narrows to the renderings that cannot mean anything else.
_ABSENCE_RENDERINGS = frozenset({"None", "nan", "NaN", "<NA>", "NaT", "--"})


def _label_not_recorded(value) -> bool:
    """True when a routing label is ABSENT rather than written down.

    ``None`` from a callback that had no group to report, any of the five other
    scalars a dataframe uses for a missing cell (float ``nan`` including
    ``np.float32``/``np.float64``, ``pd.NA``, ``pd.NaT``,
    ``np.datetime64('NaT')`` and ``np.ma.masked``), a string that renders blank,
    and a string that is EXACTLY what ``str()`` prints for one of those. Everything
    else is a label the caller meant, including ``0``, an ``Enum`` member and a
    numpy scalar, so ``str()`` still converts those.

    A TYPE-AND-VALUE TEST, not a falsy one. ``0`` and ``0.0`` are falsy and are
    perfectly good group labels, so ``if not value`` would refuse a real binary
    group; ``float("nan")`` is TRUTHY, so the same test would accept the shape
    that carries no information at all. The blank-string case is here because
    ``" "`` is a label that renders as nothing wherever a reader meets it, which
    is the same defect as ``'None'`` wearing a different mask.

    ABSENCE HAS SIX DOORS AND THIS GUARDED THREE (audit wave 4, 2026-09-30). The
    wave-2 version tested ``value is None``, ``isinstance(value, float) and value
    != value``, and a blank string. That refused ``None``, a Python/numpy float
    ``nan`` and ``''``/``'   '``, and ACCEPTED ``pd.NA`` (stored as the label
    ``'<NA>'``), ``pd.NaT`` and ``np.datetime64('NaT')`` (``'NaT'``),
    ``np.ma.masked`` (``'--'``) and the literal strings ``'None'`` and ``'nan'``.
    ``pd.NA`` is the missing value of EVERY pandas nullable dtype (``Int64``,
    ``string``, ``boolean``) and pandas is a hard dependency, so the dataframe-driven
    caller this docstring names as the source walked straight past the guard.
    Measured that day, the incident in :meth:`MultiAgentRunHarness.record_routing`
    reproduced verbatim with ``pd.NA`` in place of ``None``: ten routing calls
    against ten labelled ones gave groups ``['<NA>', 'a']``, cramers_v 1.0,
    odds_ratio inf, p 1.082508822446903e-05, is_significant True, per-route
    disparity 1.0, detectable True and ZERO warnings, which is the same maximal
    finding of routing discrimination against a group that does not exist.

    ``pd.isna`` is used for the scalar sentinels rather than a hand-written list of
    types, so a seventh sentinel pandas adds later arrives closed rather than open.
    It is asked only about SCALARS: on a list or an array it returns an elementwise
    array, and ``bool()`` of that raises, which would turn a malformed label into an
    exception from the wrong place.
    """
    if value is None:
        return True
    # np.ma.masked answers `pd.isna(...) -> masked`, whose bool() is False, so it
    # needs its own line: without it the one sentinel that is NOT a pandas value
    # would pass a pandas test.
    if value is np.ma.masked:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip() in _ABSENCE_RENDERINGS
    try:
        absent = pd.isna(value)
    except (TypeError, ValueError):  # pragma: no cover - a label pandas cannot judge
        absent = False
    if np.ndim(absent) == 0 and bool(absent):
        return True
    return False


@dataclass
class HarnessTrace(SerializableMixin):
    """Collected trace from a multi-agent run.

    Attributes:
        sample_groups: Per-sample group labels. Binary (0 or 1) is
            the documented case; any number of distinct labels is
            accepted and every one of them is measured.
        component_outputs: Per-agent output arrays, aligned with
            ``sample_groups``.
        system_outputs: System-level outputs aligned with
            ``sample_groups`` (one value per sample).
        pre_interaction_outputs: If recorded, per-agent outputs before
            any inter-agent communication for the same samples.
        post_interaction_outputs: If recorded, per-agent outputs after
            the interaction round for the same samples.
        routing_decisions: List of routing labels (one per task) the
            orchestrator chose.
        routing_demographics: Demographic label aligned with
            ``routing_decisions``.
        per_turn_group_a: For multi-turn dialogues, per-turn outcome
            time series for group A.
        per_turn_group_b: Per-turn outcome time series for group B.
        n_samples: Number of samples captured for the (group,
            component, system) trio.
    """

    sample_groups: List[int] = field(default_factory=list)
    component_outputs: Dict[str, List[float]] = field(default_factory=dict)
    system_outputs: List[float] = field(default_factory=list)
    pre_interaction_outputs: Dict[str, List[float]] = field(default_factory=dict)
    post_interaction_outputs: Dict[str, List[float]] = field(default_factory=dict)
    routing_decisions: List[str] = field(default_factory=list)
    routing_demographics: List[str] = field(default_factory=list)
    per_turn_group_a: List[float] = field(default_factory=list)
    per_turn_group_b: List[float] = field(default_factory=list)
    n_samples: int = 0
    metadata: RunMetadata = field(default_factory=RunMetadata)


class MultiAgentRunHarness:
    """Framework-agnostic capture surface for multi-agent fairness tests.

    Typical use:
        >>> with MultiAgentRunHarness() as h:
        ...     for sample in dataset:
        ...         # ... run your agents on `sample` ...
        ...         h.record_sample(
        ...             group=sample.protected_group,
        ...             component_outputs={"agent_a": a_out, "agent_b": b_out},
        ...             system_output=final,
        ...             pre_interaction={"agent_a": a_pre, "agent_b": b_pre},
        ...             post_interaction={"agent_a": a_post, "agent_b": b_post},
        ...         )
        ...     for task in routing_tasks:
        ...         h.record_routing(
        ...             route=orchestrator.choose(task),
        ...             demographic=task.demographic,
        ...         )
        >>> trace = h.trace

    The trace can then be fed into the analyzers via the convenience
    accessors :meth:`as_emergent_inputs`, :meth:`as_compositionality_input`,
    :meth:`as_collusion_inputs`, :meth:`as_delegation_inputs`, and
    :meth:`as_negotiation_inputs`.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: multi_agent_run_harness. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self) -> None:
        self.trace = HarnessTrace()

    def __enter__(self) -> "MultiAgentRunHarness":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Nothing to clean up; presence of an exception is the caller's problem.
        logger.info(
            "harness exit: n_samples=%d, n_routes=%d",
            self.trace.n_samples,
            len(self.trace.routing_decisions),
        )

    # Recording

    def record_sample(
        self,
        group: int,
        component_outputs: Dict[str, float],
        system_output: float,
        pre_interaction: Optional[Dict[str, float]] = None,
        post_interaction: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record one sample's worth of per-agent + system outputs.

        Args:
            group: Group label for this sample. Binary (0 or 1) is
                the documented case; more labels are accepted and
                every one of them is measured, but
                :meth:`system_bias` then warns and reports the
                worst-case pairwise gap rather than a binary
                contrast.
            component_outputs: One numeric output per agent for this
                sample.
            system_output: The system-level output for this sample.
            pre_interaction: Optional per-agent outputs measured before
                any inter-agent interaction. Required if you later want
                to run :class:`AdversarialCollusionDetector`.
            post_interaction: Optional per-agent outputs measured after
                the interaction round.

        Raises:
            ValueError: If the agent set is inconsistent with previously
                recorded samples.
        """
        component_set = set(component_outputs)
        # The precondition is "has anything been recorded yet", and asking
        # `if self.trace.component_outputs` could not see the one case that
        # matters: a FIRST sample recorded with no components at all leaves that
        # dict empty, so this check was skipped on every later call and the
        # per-agent arrays diverged from sample_groups in silence. Measured
        # 2026-09-27, record_sample(group=0, component_outputs={}) followed by
        # record_sample(group=1, component_outputs={"a": 0.9}) was accepted and
        # left n_samples=2 against a single recorded value for "a", after which
        # as_compositionality_input raised a numpy IndexError ("boolean index
        # did not match indexed array") naming neither the sample nor the agent.
        # n_samples is the count the per-agent arrays must match, so that is what
        # the guard asks about. This is the divergence the pre/post checks below
        # already refuse, by the same argument.
        if self.trace.n_samples > 0 and component_set != set(self.trace.component_outputs):
            raise ValueError(
                "Inconsistent agent set: previously recorded "
                f"{sorted(self.trace.component_outputs)} but received "
                f"{sorted(component_outputs)}."
            )

        # pre/post must (a) cover exactly the same agents as
        # component_outputs and (b) be provided consistently across
        # samples: otherwise the per-agent arrays diverge in length and
        # downstream analyzers fail in non-obvious ways.
        prior_has_pre = bool(self.trace.pre_interaction_outputs)
        prior_has_post = bool(self.trace.post_interaction_outputs)
        if pre_interaction is not None and set(pre_interaction) != component_set:
            raise ValueError(
                "pre_interaction must cover exactly the same agents as "
                f"component_outputs ({sorted(component_set)}); got "
                f"{sorted(pre_interaction)}."
            )
        if post_interaction is not None and set(post_interaction) != component_set:
            raise ValueError(
                "post_interaction must cover exactly the same agents as "
                f"component_outputs ({sorted(component_set)}); got "
                f"{sorted(post_interaction)}."
            )
        had_prior_samples = self.trace.n_samples > 0
        if prior_has_pre and pre_interaction is None:
            raise ValueError(
                "pre_interaction was provided on earlier samples; once "
                "started it must be provided on every record_sample call."
            )
        if prior_has_post and post_interaction is None:
            raise ValueError(
                "post_interaction was provided on earlier samples; once "
                "started it must be provided on every record_sample call."
            )
        if had_prior_samples and not prior_has_pre and pre_interaction is not None:
            raise ValueError(
                "pre_interaction was not provided on earlier samples; cannot "
                "introduce it partway through a run (per-agent arrays would "
                "diverge in length)."
            )
        if had_prior_samples and not prior_has_post and post_interaction is not None:
            raise ValueError(
                "post_interaction was not provided on earlier samples; cannot "
                "introduce it partway through a run (per-agent arrays would "
                "diverge in length)."
            )

        for name, val in component_outputs.items():
            self.trace.component_outputs.setdefault(name, []).append(float(val))
        self.trace.system_outputs.append(float(system_output))
        self.trace.sample_groups.append(int(group))
        self.trace.n_samples += 1

        if pre_interaction is not None:
            for name, val in pre_interaction.items():
                self.trace.pre_interaction_outputs.setdefault(name, []).append(float(val))
        if post_interaction is not None:
            for name, val in post_interaction.items():
                self.trace.post_interaction_outputs.setdefault(name, []).append(float(val))

    def record_routing(self, route: str, demographic: str) -> None:
        """Record a single orchestrator routing decision.

        Args:
            route: The route / agent the orchestrator chose.
            demographic: The demographic label of the originating task.

        Raises:
            ValueError: If either label is absent rather than written down
                (``None``, any of the five other missing-value scalars a
                dataframe carries, a string that is empty or only whitespace, or
                a string that is exactly what ``str()`` prints for one of them,
                such as ``'None'``, ``'nan'``, ``'<NA>'`` or ``'NaT'``). See
                :func:`_label_not_recorded`.

        ``str()`` MINTED A GROUP THAT DID NOT EXIST (audit wave 2, 2026-09-29).
        Both lines below used to be ``append(str(...))``, and ``str(None)`` is the
        five-character label ``'None'``, so a task whose demographic was never
        captured became a member of a demographic group named "None" and a task
        whose route was never captured became a route named "None". Reproduced,
        ten routing calls with ``demographic=None`` against ten with
        ``demographic="a"``, each group routed one way:

            groups=['None', 'a']  cramers_v=1.0  odds_ratio=inf
            p_value=1.082508822446903e-05  is_significant=True  warnings=0

        through ``DelegationRoutingAuditor().analyze``: a maximal, statistically
        significant finding of routing discrimination against a group that does
        not exist, with nothing anywhere saying so.

        THE SAME NUMBERS CAME BACK THROUGH ``pd.NA`` (audit wave 4, 2026-09-30),
        because the wave-2 predicate tested three shapes of absence and there are
        six. Reproduced verbatim with ``pd.NA`` substituted for ``None``:
        groups=['<NA>', 'a'], cramers_v=1.0, odds_ratio=inf, p=1.082508822446903e-05,
        is_significant=True, warnings=0. ``pd.NaT``, ``np.datetime64('NaT')``,
        ``np.ma.masked`` and the literal strings ``'None'`` and ``'nan'`` walked
        through the same gap. See :func:`_label_not_recorded`, which now closes all
        six doors and the renderings they print.

        REFUSED AT THE RECORDING BOUNDARY, on purpose, and this is the only option
        that neither invents nor hides. Recording the pair under an invented label
        corrupts the contingency table for every OTHER group too, because it moves
        the totals. Dropping the pair silently instead would be differential
        attrition, which this library refuses elsewhere by name. And the harness
        has no field in which to carry "this one task had no group", so there is no
        third state to put it in: the caller has to decide, at the call site, which
        is where it still knows what happened.

        ``str()`` IS KEPT for everything else, deliberately. An ``int``, an
        ``Enum`` or a numpy scalar used as a label is a real label that a caller
        legitimately expects to be stringified, so this narrows the refusal to the
        shapes that mean "nothing was recorded" rather than replacing the
        conversion.
        """
        # The messages name the ACTUAL rendering, str(value), and not a fixed
        # "'None' (or '')". A caller who passed pd.NA needs to read '<NA>' to
        # recognise the column it came out of; being told about 'None' sends them
        # looking for the wrong thing.
        if _label_not_recorded(route):
            raise ValueError(
                f"record_routing received route={route!r}, which is not a recorded "
                f"label. str() would turn it into a route literally named "
                f"{str(route)!r}, which is then counted, cross-tabulated and "
                f"significance tested as a real route. Pass the route the orchestrator "
                f"actually chose, or decide at this call site to skip the task; note "
                f"that skipping some tasks and not others is differential attrition and "
                f"has to be recorded as such."
            )
        if _label_not_recorded(demographic):
            raise ValueError(
                f"record_routing received demographic={demographic!r}, which is not "
                f"a recorded label. str() would turn it into a demographic group "
                f"literally named {str(demographic)!r}: measured on ten such calls "
                f"against ten labelled ones, DelegationRoutingAuditor reported "
                f"cramers_v=1.0, odds_ratio=inf, p=1.08e-05 and is_significant=True for "
                f"a group that does not exist. Pass the demographic of the originating "
                f"task, or decide at this call site to skip the task; note that skipping "
                f"some tasks and not others is differential attrition and has to be "
                f"recorded as such."
            )
        self.trace.routing_decisions.append(str(route))
        self.trace.routing_demographics.append(str(demographic))

    def record_turn(self, group_a_outcome: float, group_b_outcome: float) -> None:
        """Record one turn of a per-turn negotiation outcome series.

        Args:
            group_a_outcome: Outcome at this turn for group A.
            group_b_outcome: Outcome at this turn for group B.
        """
        self.trace.per_turn_group_a.append(float(group_a_outcome))
        self.trace.per_turn_group_b.append(float(group_b_outcome))

    # Convenience accessors → analyzer inputs

    def as_emergent_inputs(self):
        """Return the trio needed by :class:`EmergentBiasDetector.analyze`.

        The labels are returned AS RECORDED, so a run with more than two
        distinct group labels comes back with all of them. The two consumers
        then differ, on purpose, and a caller should know which one it is
        talking to: :meth:`system_bias` and :meth:`as_compositionality_input`
        measure every group and disclose the non-binary run through
        :meth:`_group_view` (the worst-case pairwise gap, with a warning), while
        :meth:`EmergentBiasDetector.analyze` is a binary comparison with no
        field in which to say which pair a verdict belongs to and raises
        ValueError for a third label rather than silently comparing the first
        two (BGL-5, 2026-09-27: before that it reported system_bias 0.0 and
        is_emergent False on a three-group frame whose whole gap sat in the
        third group). For a non-binary run, slice this trio per pair, or per
        group against the rest, and say which pair each verdict belongs to.
        """
        return (
            {k: np.asarray(v, dtype=float) for k, v in self.trace.component_outputs.items()},
            np.asarray(self.trace.system_outputs, dtype=float),
            np.asarray(self.trace.sample_groups),
        )

    # DO NOT REMOVE: `as_compositionality_input` and `system_bias` share one
    # precondition -- how many distinct group labels were actually recorded --
    # so the guard lives ABOVE both of them. Until 2026-09-11 each method took
    # np.unique(groups)[0] and [1] independently and silently dropped every
    # sample outside those two labels; a run whose THIRD group was the harmed
    # one reported a near-zero bias with no warning. Fixing one call site alone
    # would have moved the silent exclusion to its sibling.

    def _group_view(self, stacklevel: int = 3):
        """Return ``(groups, unique)``, disclosing a non-binary run.

        ``record_sample`` documents a binary group label but does not
        enforce one, so a caller can record three groups in silence.
        Every observed group is measured now; more than two of them is
        reported as a ``UserWarning`` because the returned number then
        means something different from the binary contrast.
        """
        groups = np.asarray(self.trace.sample_groups)
        unique = np.unique(groups)
        if len(unique) > 2:
            warnings.warn(
                f"{len(unique)} distinct group labels were recorded "
                f"({unique.tolist()}); record_sample documents a binary group "
                "label. Returning the worst-case pairwise gap across ALL "
                f"{len(unique)} groups: this is not the binary contrast the "
                "method name implies and is not comparable with a two-group "
                "run.",
                UserWarning,
                stacklevel=stacklevel,
            )
        return groups, unique

    @staticmethod
    def _worst_pairwise_gap(
        values: np.ndarray,
        groups: np.ndarray,
        unique: np.ndarray,
        site: str = "MultiAgentRunHarness",
    ) -> float:
        """Largest difference between any two observed group means.

        Identical to ``abs(mean_0 - mean_1)`` when exactly two groups
        were recorded, so binary runs are unchanged. A group whose
        outputs are not finite makes its mean NaN and the gap NaN:
        a could-not-check, never a small finite number.

        The NaN was already right; what was missing was the SENTENCE. Measured
        2026-09-27 with ten samples per group and group 1's system outputs all
        NaN, ``system_bias()`` returned ``nan`` and warned about nothing, so a
        caller that formats the number or compares it (``nan > threshold`` is
        False, which reads as a gap under the bar) had no way to know a whole
        group went unmeasured. The disclosure lives here rather than in either
        accessor because both call this and a warning in one of them would leave
        the other silent.
        """
        # G10, 2026-09-30. THE ALIGNMENT PRECONDITION LIVES HERE, ABOVE BOTH
        # CALLERS. `as_compositionality_input` and `system_bias` both mask `values`
        # with a boolean array built from `groups`, so both need the two to be the
        # same length, and a guard in one of them would leave the other leaking.
        # `record_sample` closes the door for a trace built through the recording
        # API, but `HarnessTrace` is a public dataclass and `harness.trace` is a
        # documented public attribute, so a hand-built or mutated trace reaches
        # here directly. Measured that day on such a trace (sample_groups of 10,
        # component_outputs {"a": [...3 values]}): `as_compositionality_input`
        # raised the raw numpy `IndexError: boolean index did not match indexed
        # array along axis 0; size of axis is 3 but size of corresponding boolean
        # axis is 10`, naming neither the agent nor the trace, while all three
        # SIBLING accessors name the mismatch (`EmergentBiasDetector` says
        # "Component 'a' output length (3) must match groups length (10)",
        # `DelegationRoutingAuditor` and `NegotiationFairnessTracker` likewise).
        # That unnamed IndexError is the exact symptom the 2026-09-27 note in
        # `record_sample` says it wanted gone. It is a ValueError, matching every
        # sibling, not a fabricated number: no masked comparison can silently
        # succeed on a mismatch, so nothing was being mis-measured, only
        # mis-explained.
        if values.shape[0] != groups.shape[0]:
            raise ValueError(
                f"{site}: recorded {values.shape[0]} value(s) against "
                f"{groups.shape[0]} group label(s). A group-conditional gap needs "
                "one value per labelled sample; these cannot be aligned, so no "
                "gap is reported rather than one computed from part of the run."
            )
        means = np.array([np.mean(values[groups == u]) for u in unique], dtype=float)
        n_unmeasurable = int(np.count_nonzero(~np.isfinite(means)))
        if n_unmeasurable:
            warn_not_assessed(
                site,
                measured=int(len(unique)) - n_unmeasurable,
                total=int(len(unique)),
                unit="recorded groups had a finite mean",
                requirement="a gap spans every group, so it needs a finite mean in each",
                reporting="nan",
                instead_of="a finite gap computed from the groups that survived",
                # 1 warn_not_assessed, 2 here, 3 the accessor, 4 its caller.
                stacklevel=4,
            )
        return float(np.max(means) - np.min(means))

    def as_compositionality_input(self) -> Dict[str, float]:
        """Compute per-agent group-conditional bias scores.

        Returns the ``component_biases`` dict shape that
        :meth:`CompositionalityAnalyzer.analyze` expects. Every recorded
        group is measured: with more than two of them the value is the
        worst-case pairwise gap over all of them and a ``UserWarning``
        says so; a group with non-finite outputs yields NaN.
        """
        groups = np.asarray(self.trace.sample_groups)
        if len(np.unique(groups)) < 2:
            raise ValueError("Need at least 2 distinct groups in the recorded samples.")
        groups, unique = self._group_view()
        # G10, 2026-09-30. AN EMPTY PER-AGENT DICT IS A COULD-NOT-CHECK, NOT A
        # CLEAN SHEET. `record_sample(group=..., component_outputs={}, ...)` is
        # accepted on purpose (see the note there), so a run whose per-agent
        # capture was never wired records samples and groups and NO components.
        # This method then returned `{}` in silence, and a caller rendering
        # `for agent, bias in h.as_compositionality_input().items()` gets a page
        # with no biased agent on it, which reads as "no agent is biased" rather
        # than "no agent was measured". The downstream
        # `CompositionalityAnalyzer.analyze` does refuse `{}` ("At least one
        # component bias score is required"), so the number never reached a
        # verdict; what was missing was the SENTENCE for anyone reading the dict
        # itself. Still returns `{}`, so no caller breaks.
        if not self.trace.component_outputs:
            warnings.warn(
                f"MultiAgentRunHarness.as_compositionality_input: {len(groups)} "
                "sample(s) were recorded but NO per-agent outputs, so there is no "
                "component bias to measure and this empty dict is a could-not-check, "
                "NOT a finding that no agent is biased. Pass component_outputs to "
                "record_sample.",
                UserWarning,
                stacklevel=2,
            )
        out: Dict[str, float] = {}
        for name, vals in self.trace.component_outputs.items():
            arr = np.asarray(vals, dtype=float)
            out[name] = self._worst_pairwise_gap(
                arr,
                groups,
                unique,
                site=f"MultiAgentRunHarness.as_compositionality_input[{name}]",
            )
        return out

    def system_bias(self) -> float:
        """Group-conditional bias for the system-level output.

        Every recorded group is measured: with more than two of them the
        value is the worst-case pairwise gap over all of them and a
        ``UserWarning`` says so; a group with non-finite outputs yields
        NaN rather than a finite number computed from the others.
        """
        groups = np.asarray(self.trace.sample_groups)
        if len(np.unique(groups)) < 2:
            raise ValueError("Need at least 2 distinct groups.")
        groups, unique = self._group_view()
        sys_arr = np.asarray(self.trace.system_outputs, dtype=float)
        return self._worst_pairwise_gap(
            sys_arr, groups, unique, site="MultiAgentRunHarness.system_bias"
        )

    def as_collusion_inputs(self):
        """Return the trio needed by :class:`AdversarialCollusionDetector.analyze`."""
        if not self.trace.pre_interaction_outputs or not self.trace.post_interaction_outputs:
            raise ValueError(
                "Collusion analysis requires pre_interaction and post_interaction "
                "to have been recorded on each sample."
            )
        return (
            {k: np.asarray(v, dtype=float) for k, v in self.trace.pre_interaction_outputs.items()},
            {k: np.asarray(v, dtype=float) for k, v in self.trace.post_interaction_outputs.items()},
            np.asarray(self.trace.sample_groups),
        )

    def as_delegation_inputs(self):
        """Return ``(routes, demographics)`` for :class:`DelegationRoutingAuditor.analyze`."""
        if not self.trace.routing_decisions:
            raise ValueError("No routing decisions recorded.")
        return list(self.trace.routing_decisions), list(self.trace.routing_demographics)

    def as_negotiation_inputs(self):
        """Return the per-turn pair for :class:`NegotiationFairnessTracker.analyze`."""
        if len(self.trace.per_turn_group_a) < 3:
            raise ValueError("Need at least 3 recorded turns for negotiation-fairness analysis.")
        return list(self.trace.per_turn_group_a), list(self.trace.per_turn_group_b)
