"""
Natural direct (NDE) and natural indirect (NIE) effect decomposition via DoWhy.

Wraps DoWhy's mediation analysis to split a total effect into:
    * direct effect of treatment on outcome holding mediator at its natural level
    * indirect effect transmitted through the mediator

This is the canonical fairness-mediation question:
"How much of the disparity is direct discrimination vs. discrimination passing
through M?"
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class MediationDecomposition:
    treatment: str
    outcome: str
    mediator: str
    total_effect: Optional[float] = None
    natural_direct_effect: Optional[float] = None
    natural_indirect_effect: Optional[float] = None
    proportion_mediated: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class MediationResult:
    decompositions: List[MediationDecomposition] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decompositions": [asdict(d) for d in self.decompositions],
            "warnings": list(self.warnings),
        }


def _import_dowhy():
    try:
        from dowhy import CausalModel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vfairness.operations.causal.mediate requires 'dowhy'. "
            "Install with: pip install dowhy networkx"
        ) from exc
    return CausalModel


#: Share of rows whose mediator / treatment / outcome value cannot be read
#: above which the triple is REFUSED rather than estimated on the readable
#: rows. Complete-case analysis on a small remainder is not a measurement of
#: the population, and the direction of the bias is unknowable from the data
#: that could be read.
MAX_UNREADABLE_SHARE = 0.20


def _coerce_column(s: pd.Series) -> Tuple[pd.Series, int, bool]:
    """Coerce one column for the NDE regression. Never invents a value.

    Returns ``(series, n_unreadable, factorized)``. On the numeric path the
    values ``pd.to_numeric`` cannot read stay NaN and are COUNTED; the caller
    drops those rows and says so. On the categorical path every distinct
    string gets an ordinal code, which is a modelling choice the caller
    discloses rather than a measurement.

    BGL-S2 (2026-09-16): this used to end ``.fillna(num.mean() ...)``. Any
    value ``pd.to_numeric`` could not read ('unknown', 'N/A', 'declined to
    state', a coded refusal) was replaced by the column MEAN and fed to the
    regression whose coefficient is REPORTED as ``natural_direct_effect``.
    Measured 2026-09-16 on a DGP with true NDE 1.0, NIE 6.0, proportion
    mediated 0.857: with half the mediator recorded as the string 'unknown'
    (and ZERO real NaNs, so ``dropna`` never saw them) the decomposition came
    back nde=5.4055, nie=1.3583, proportion_mediated=0.2008 with ``notes=[]``
    and ``warnings=[]``. The conclusion INVERTS, from "almost all of the
    disparity is laundered through the mediator" to "mostly direct". A
    variable whose coefficient is the reported effect is never mean-imputed.
    """
    num = pd.to_numeric(s, errors="coerce")
    n_readable = int(num.notna().sum())

    # Counted ONCE, ABOVE the branch selection, because both branches share
    # the precondition "how many of these cells hold no readable value".
    #
    # BGL-S2b (2026-09-17): the factorize branch returned the constant 0 here,
    # so MAX_UNREADABLE_SHARE could not fire once the unreadable values were
    # the MAJORITY - exactly the case it was written for. Measured on the same
    # DGP (true NDE 1.0, NIE 6.0, proportion mediated 0.857) with 201 of 400
    # mediator rows recorded as the string 'unknown': 199 readable is below
    # half, so the column fell through to factorize, 'unknown' became ordinal
    # code 0 beside 199 quantities, and the result was nde=6.8868,
    # prop=0.0267, warnings=[] - the same inverted conclusion the ceiling
    # exists to prevent, reached by the branch the ceiling could not see.
    #
    # A column no value of which reads as a number is CATEGORICAL by nature,
    # and "not a number" is not a defect in it: only a genuinely missing cell
    # is unreadable there. A column holding BOTH numbers and non-numbers is a
    # quantity polluted with markers, and every non-numeric cell in it is
    # unreadable, whichever side of the half-way line the readable ones fall.
    if n_readable:
        unreadable_mask = num.isna()
    else:
        unreadable_mask = s.isna()
    n_unreadable = int(unreadable_mask.sum())

    if n_readable >= len(num) // 2:
        return num, n_unreadable, False

    codes, _ = pd.factorize(s.astype(str), sort=True)
    coded = pd.Series(codes, index=s.index, dtype=float)
    # NaN, not a code: an unreadable cell has to leave the regression through
    # the same dropna the numeric branch uses, or the caller's share of
    # dropped rows says zero while a fifth of the column was never read.
    # ``factorize`` would otherwise give 'unknown' a code of its own and the
    # row would be fitted as though the value had been recorded.
    coded[unreadable_mask.to_numpy()] = float("nan")
    return coded, n_unreadable, True


def _constant_treatment_reason(data: pd.DataFrame, column: str, label: str) -> Optional[str]:
    """Why this column cannot support a contrast, or None if it can.

    BGL-S2 (2026-09-16): a treatment that never varies gives a rank-deficient
    design, and ``LinearRegression`` answers a rank-deficient system with the
    MINIMUM-NORM solution: coefficient 0.0. That 0.0 was reported as
    ``natural_direct_effect``, which made ``natural_indirect_effect`` equal to
    the whole total effect and ``proportion_mediated`` exactly 1.0, "every
    bit of the disparity passes through the mediator", from data that
    contains no comparison at all. Measured 2026-09-16 with a constant
    treatment: total=1.8718, nde=0.0, nie=1.8718, prop_mediated=1.0,
    notes=[], warnings=[]. Counted on the RAW values, never on a fitted
    statistic, because an accumulated statistic is not reliably exactly zero.

    BGL-S2b (2026-09-17): the same guard now covers the MEDIATOR, which was
    unguarded while the treatment was refused, from the identical rank-deficient
    mechanism. A constant mediator is collinear with the intercept, so the
    treatment coefficient absorbs the whole total effect. Measured with M = 1.0
    in every row: total=7.0757, nde=7.0757, nie=-5.3e-15,
    proportion_mediated=-7.5e-16, notes=[], warnings=[] - "none of the disparity
    is mediated", from a mediator carrying no information at all. A mediator
    recorded as 'unknown' in every row reaches production through exactly this
    path, because factorize codes one label as the constant 0.
    """
    if column not in data.columns:
        return None
    n_distinct = int(data[column].nunique(dropna=True))
    if n_distinct >= 2:
        return None
    if n_distinct == 0:
        return (
            f"Refused: {label} '{column}' has no usable value in any row, so there is "
            f"no contrast to decompose. Reporting nothing measured rather than the "
            f"minimum-norm coefficient 0.0, which would read as 'no direct effect'."
        )
    if label.startswith("mediator"):
        consequence = (
            "A constant mediator is collinear with the intercept, so the treatment "
            "coefficient absorbs the WHOLE total effect and the fit answers nde = "
            "total, nie = 0, proportion_mediated = 0, i.e. 'none of the disparity "
            "is mediated', from a mediator that carries nothing."
        )
    else:
        consequence = (
            "A constant treatment makes the design rank deficient and the fitted "
            "coefficient is the minimum-norm 0.0, which would be reported as 'no "
            "direct effect' and as proportion_mediated 1.0."
        )
    return (
        f"Refused: {label} '{column}' takes a single value in every row, so there is "
        f"no contrast to decompose. {consequence} Nothing was measured here."
    )


#: Rows that must differ from a column's most common value before the contrast
#: is treated as supported by the data. Same floor as the whole-frame minimum
#: this module already applies (``len(frame) < 5``), reused rather than invented.
MIN_ROWS_OFF_MODE = 5


def _thin_contrast_note(data: pd.DataFrame, column: str, label: str) -> Optional[str]:
    """Disclose a contrast carried by almost no rows, or None when it is fine.

    BGL-S2b (2026-09-17): ``nunique >= 2`` is satisfied by a SINGLE
    distinguishing row, and the refusal above therefore passes a column that
    supports no contrast worth the name. Measured on 399 rows at T=1 and one at
    T=0: total=-1.5989, nde=0.4982, prop=1.3116, notes=[] - a decomposition of
    a disparity estimated from one row, presented with nothing said about it.

    Counted as rows OFF the modal value rather than as the rarest level's size,
    so a continuous column (every value distinct, every count 1) is silent: what
    matters is how many rows disagree with the majority, not how finely the
    values are spread. This DISCLOSES; it does not refuse, because a small cell
    is still a measurement and the caller may know it is enough.
    """
    if column not in data.columns:
        return None
    counts = data[column].value_counts(dropna=True)
    if len(counts) < 2:
        return None
    n_used = int(counts.sum())
    n_off_mode = n_used - int(counts.iloc[0])
    if n_off_mode >= MIN_ROWS_OFF_MODE:
        return None
    mode = counts.index[0]
    mode = mode.item() if hasattr(mode, "item") else mode
    return (
        f"Thin contrast: only {n_off_mode} of {n_used} row(s) differ from the modal "
        f"value of {label} '{column}' ({mode!r}), below the {MIN_ROWS_OFF_MODE} "
        f"this module treats as a minimum elsewhere. The split below rests on those "
        f"{n_off_mode} row(s)."
    )


def decompose_mediation(
    gml: str,
    data: pd.DataFrame,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    mediators: Iterable[str],
) -> MediationResult:
    """
    For each (treatment, mediator, outcome) triple, return the NDE/NIE split.

    DoWhy estimates the total effect, then re-identifies under "mediation" mode
    to decompose it. We use linear_regression as the estimator because it is
    the only DoWhy mediation method that is reliable on small samples; richer
    estimators are exposed in Phase 3 once GCM lands.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: decompose_mediation. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    CausalModel = _import_dowhy()
    result = MediationResult()

    if data is None or data.empty:
        result.warnings.append("Mediation requires a non-empty dataset.")
        return result

    for t in treatments:
        for y in outcomes:
            for m in mediators:
                dec = MediationDecomposition(treatment=t, outcome=y, mediator=m)
                # Guarded ABOVE the estimator, not inside the regression
                # branch: the total effect is fitted on the same constant
                # treatment and is no more identified than the NDE is, so
                # refusing only the NDE would leave the same fabrication
                # standing one field over.
                # Both the treatment and the MEDIATOR have to vary: the
                # decomposition is a contrast in one held at the other, and
                # either one constant makes the design rank deficient. The
                # mediator was unguarded until BGL-S2b; see
                # _constant_treatment_reason.
                reason = _constant_treatment_reason(data, t, "treatment") or (
                    _constant_treatment_reason(data, m, "mediator")
                )
                if reason is not None:
                    dec.notes.append(reason)
                    result.warnings.append(f"{t} -> {m} -> {y}: {reason}")
                    result.decompositions.append(dec)
                    continue
                try:
                    model = CausalModel(
                        data=data,
                        treatment=t,
                        outcome=y,
                        graph=gml,
                    )
                    estimand_total = model.identify_effect(
                        proceed_when_unidentifiable=True,
                    )
                    total = model.estimate_effect(
                        estimand_total,
                        method_name="backdoor.linear_regression",
                    )
                    dec.total_effect = float(getattr(total, "value", float("nan")))

                    # NDE via direct regression: regress Y on T while
                    # controlling for the mediator AND the same backdoor
                    # adjustment set the total-effect estimate used. Without
                    # the backdoor covariates, total (confounder-adjusted) and
                    # NDE (unadjusted) mix two different adjustment sets and
                    # their difference is NOT the NIE: proportion_mediated
                    # was biased whenever confounders existed.
                    from sklearn.linear_model import LinearRegression

                    backdoor_vars = [
                        c
                        for c in (estimand_total.get_backdoor_variables() or [])
                        if c in data.columns and c not in (t, y, m)
                    ]
                    needed = [t, y, m] + backdoor_vars
                    df = data.dropna(subset=needed).copy()
                    if len(df) < 5:
                        dec.notes.append("Not enough rows for mediation after dropping NaNs.")
                    else:
                        # Coerce every column ONCE, keeping the count of values
                        # that could not be read. The unreadable rows are
                        # dropped, never mean-filled: see _coerce_column.
                        coerced: List[pd.Series] = []
                        unreadable: Dict[str, int] = {}
                        factorized: List[str] = []
                        for name in [t, m] + backdoor_vars + [y]:
                            series, n_bad, was_factorized = _coerce_column(df[name])
                            coerced.append(series.rename(name))
                            if n_bad:
                                unreadable[name] = n_bad
                            if was_factorized:
                                factorized.append(name)

                        frame = pd.concat(coerced, axis=1)
                        n_before = len(frame)
                        frame = frame.dropna()
                        n_dropped = n_before - len(frame)
                        share = (n_dropped / n_before) if n_before else 1.0

                        if factorized:
                            dec.notes.append(
                                "Treated as categorical (ordinal factorize codes), not as a "
                                "quantity: " + ", ".join(sorted(factorized))
                            )
                        if unreadable:
                            listing = ", ".join(
                                f"{col} ({n} of {n_before})"
                                for col, n in sorted(unreadable.items())
                            )
                            dec.notes.append(
                                f"Unreadable values were DROPPED, not imputed: {listing}. "
                                f"{n_dropped} of {n_before} row(s) left the NDE regression; "
                                f"it rests on the {len(frame)} row(s) that carry a value for "
                                f"every variable."
                            )

                        # The guards below share the precondition "the NDE
                        # regression frame supports the contrast", so they sit
                        # above the fit rather than inside it.
                        post_reason = _constant_treatment_reason(
                            frame, t, "treatment (after dropping unreadable rows),"
                        ) or _constant_treatment_reason(
                            frame, m, "mediator (after dropping unreadable rows),"
                        )
                        if share > MAX_UNREADABLE_SHARE:
                            note = (
                                f"Refused: {n_dropped} of {n_before} row(s) "
                                f"({share:.1%}) have a value that cannot be read in "
                                f"{', '.join(sorted(unreadable))}, above the "
                                f"{MAX_UNREADABLE_SHARE:.0%} ceiling. The decomposition "
                                f"is NOT reported on the remainder: whether the "
                                f"unreadable rows differ from the readable ones cannot "
                                f"be established from the readable ones. Nothing was "
                                f"measured for the direct and indirect split."
                            )
                            dec.notes.append(note)
                            result.warnings.append(f"{t} -> {m} -> {y}: {note}")
                        elif len(frame) < 5:
                            dec.notes.append(
                                f"Not enough rows for mediation after dropping unreadable "
                                f"values: {len(frame)} left of {n_before}."
                            )
                        elif post_reason is not None:
                            dec.notes.append(post_reason)
                            result.warnings.append(f"{t} -> {m} -> {y}: {post_reason}")
                        else:
                            for col, lbl in ((t, "treatment"), (m, "mediator")):
                                thin = _thin_contrast_note(frame, col, lbl)
                                if thin is not None:
                                    dec.notes.append(thin)

                            X = frame[[t, m] + backdoor_vars].values
                            Y = frame[y].values
                            reg = LinearRegression().fit(X, Y)
                            # Coefficient on T (index 0) holding M and the
                            # backdoor set fixed = NDE.
                            dec.natural_direct_effect = float(reg.coef_[0])
                            if backdoor_vars:
                                dec.notes.append(
                                    "NDE adjusted for backdoor set: " + ", ".join(backdoor_vars)
                                )

                            if (
                                dec.total_effect is not None
                                and math.isfinite(dec.total_effect)
                                and dec.natural_direct_effect is not None
                            ):
                                dec.natural_indirect_effect = (
                                    dec.total_effect - dec.natural_direct_effect
                                )
                                # BGL-S2b (2026-09-17): the floor was the ABSOLUTE
                                # 1e-9, which a fitted OLS coefficient practically
                                # never lands under, so the "undefined" branch was
                                # unreachable and a total effect that is numerical
                                # noise still produced a printed proportion. The
                                # floor is now relative to the spread the total
                                # effect explains in the outcome: a total effect
                                # that moves Y across the observed range of T by
                                # less than a millionth of Y's own spread is noise,
                                # at any scale.
                                y_spread = float(frame[y].std(ddof=0))
                                t_spread = float(frame[t].std(ddof=0))
                                explains = abs(dec.total_effect) * t_spread
                                floor = 1e-6 * y_spread if y_spread > 0 else 1e-9
                                if explains > floor:
                                    dec.proportion_mediated = (
                                        dec.natural_indirect_effect / dec.total_effect
                                    )
                                    # A ratio outside [0, 1] is not a SHARE, and it
                                    # was printed as one with notes=[]. Measured on
                                    # a DGP whose direct and indirect effects nearly
                                    # cancel (total -0.1736): proportion_mediated
                                    # came back -33.06. The number is arithmetically
                                    # right and reads as nonsense, so it is labelled
                                    # rather than deleted: the effects themselves are
                                    # still the measurement.
                                    prop = dec.proportion_mediated
                                    if not 0.0 <= prop <= 1.0:
                                        dec.notes.append(
                                            f"proportion_mediated is {prop:.4g}, OUTSIDE "
                                            f"[0, 1], so it is not a share of the "
                                            f"disparity: the direct and indirect effects "
                                            f"have opposite signs (inconsistent "
                                            f"mediation), or the total effect "
                                            f"({dec.total_effect:.4g}) sits near the point "
                                            f"where they cancel. Read the two effects, "
                                            f"not the ratio."
                                        )
                                else:
                                    dec.notes.append(
                                        f"proportion_mediated is undefined: the total "
                                        f"effect ({dec.total_effect:.4g}) is "
                                        f"indistinguishable from zero at the scale of "
                                        f"this data, so there is no disparity for a "
                                        f"share of it to be mediated."
                                    )
                            elif dec.total_effect is None or not math.isfinite(dec.total_effect):
                                dec.notes.append(
                                    "The total effect estimator returned no finite value, so "
                                    "the indirect effect and proportion mediated could not be "
                                    "derived from it. The direct effect above is measured."
                                )
                except Exception as exc:
                    dec.notes.append(f"Mediation raised: {exc}")

                result.decompositions.append(dec)

    return result
