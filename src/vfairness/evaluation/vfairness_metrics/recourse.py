"""Algorithmic recourse / counterfactual explanations (DiCE-style), model-agnostic.

Given an individual who received an undesired decision, finds nearby
counterfactual instances, that is minimal changes to the inputs that would flip
the decision across the threshold. This answers "what would have to change for a
different outcome?", the recourse side of the GDPR Art. 22 right to explanation.

Model-agnostic random + greedy search over any ``predict(X) -> scores`` callable
(no ``dice-ml`` / torch dependency): it samples candidate feature changes from
the observed data ranges (categorical-aware), keeps the candidates that flip the
prediction across the decision threshold, and ranks them by proximity (few,
small, plausible changes). Immutable features (age direction, protected
attributes) can be frozen so recourse stays actionable.

Not imported from the package __init__ (keeps ``import vfairness`` light):

    from vfairness.evaluation.vfairness_metrics.recourse import generate_recourse
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .attribution import _as_2d_array, _feature_names, _scores_from_predict


@dataclass
class FeatureChange:
    feature: str
    from_value: float
    to_value: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Recourse:
    changes: List[FeatureChange]
    n_changes: int  # sparsity (fewer is better)
    proximity: float  # normalised L1 distance (smaller is better)
    prediction_after: float
    sentence: str  # plain-language Wachter-style recourse text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changes": [c.to_dict() for c in self.changes],
            "n_changes": self.n_changes,
            "proximity": self.proximity,
            "prediction_after": self.prediction_after,
            "sentence": self.sentence,
        }


@dataclass
class RecourseResult:
    """The outcome of a recourse search.

    Attributes:
        found: True when at least one counterfactual flipped the decision, False
            when the search RAN and found none, and None when the search could
            not be evaluated at all (COULD NOT CHECK). The third state exists
            because this surface answers a GDPR Article 22 question and the
            other two are both claims about the model that an unscorable model
            does not support.
        n_scorable_candidates: How many of the candidates the model returned a
            finite score for. 0 with a non-zero ``n_candidates`` means nothing
            was actually evaluated.
    """

    found: Optional[bool]
    prediction_before: float
    threshold: float
    desired_outcome: str  # 'above_threshold' | 'below_threshold'
    counterfactuals: List[Recourse]
    n_candidates: int
    notes: List[str] = field(default_factory=list)
    n_scorable_candidates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "prediction_before": self.prediction_before,
            "threshold": self.threshold,
            "desired_outcome": self.desired_outcome,
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "n_candidates": self.n_candidates,
            "n_scorable_candidates": self.n_scorable_candidates,
            "notes": list(self.notes),
        }


def _column_sampler(col: "np.ndarray", rng: "np.random.Generator", n: int) -> "np.ndarray":
    """Draw n plausible values for a feature from its observed distribution.

    Low-cardinality columns are treated as categorical (sample observed values);
    continuous columns sample uniformly across the central observed range.

    A column with NO finite observed value has no distribution to draw from, so
    this returns NaN. It used to return ``np.zeros(n)``: zero is a real,
    plausible-looking value that was never observed, and a recourse sentence
    built on it tells an applicant to change a feature to a number the data
    never contained. ``generate_recourse`` drops such columns from the mutable
    set before it gets here and says so in its notes; the NaN is the fail-closed
    answer for any other caller.
    """
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return np.full(n, np.nan)
    uniques = np.unique(finite)
    if uniques.size <= 12:
        return rng.choice(uniques, size=n)
    lo, hi = np.quantile(finite, 0.05), np.quantile(finite, 0.95)
    if hi <= lo:
        lo, hi = finite.min(), finite.max()
    return rng.uniform(lo, hi, size=n)


def generate_recourse(
    predict,
    x_row: Any,
    background: Any,
    feature_names: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    max_changes: int = 3,
    n_candidates: int = 3000,
    top_k: int = 3,
    random_state: int = 42,
    immutable_features: Optional[Sequence[str]] = None,
) -> RecourseResult:
    """Find minimal counterfactual changes that flip an individual's decision.

    Parameters
    ----------
    predict : callable ``predict(X_2d) -> scores`` (probabilities preferred).
    x_row : the individual's feature vector.
    background : 2D dataset used for feature ranges / scales.
    threshold : decision threshold; the desired outcome is the opposite side.
    max_changes : maximum number of features any single recourse may change.
    immutable_features : feature names that may not change (frozen during search).

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

    Ledger row: recourse_generator. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    bg = _as_2d_array(background).astype(float)
    if bg.ndim != 2:
        raise ValueError("background must be 2D")
    row = _as_2d_array(x_row).astype(float).reshape(-1)
    n_feat = bg.shape[1]
    if row.shape[0] != n_feat:
        raise ValueError("x_row width must match background width")
    names = list(_feature_names(background, n_feat, feature_names))
    immutable = set(immutable_features or [])
    mutable_idx = [j for j, nm in enumerate(names) if nm not in immutable]
    if not mutable_idx:
        raise ValueError("at least one feature must be mutable")

    # A FEATURE WITH NO OBSERVED VALUE CANNOT SUPPLY A PLAUSIBLE ONE. The
    # sampler used to answer np.zeros for an all-NaN column, so a search could
    # propose "change this to 0" for a feature whose data contains no number at
    # all, and the recourse sentence would read exactly like a real one. Such
    # columns are dropped from the search and named in the notes instead.
    unobservable = [j for j in mutable_idx if not np.any(np.isfinite(bg[:, j]))]
    mutable_idx = [j for j in mutable_idx if j not in set(unobservable)]
    if not mutable_idx:
        raise ValueError(
            "every mutable feature has no finite observed value in `background`, so no "
            "plausible change could be sampled and no recourse search was run"
        )

    pred_before = float(_scores_from_predict(predict, row.reshape(1, -1))[0])
    wants_above = pred_before < threshold  # desired = cross to the other side
    desired = "above_threshold" if wants_above else "below_threshold"

    # Feature scales for proximity (median absolute deviation, range fallback).
    mad = np.median(np.abs(bg - np.median(bg, axis=0)), axis=0)
    rng_scale = bg.max(axis=0) - bg.min(axis=0)
    scale = np.where(mad > 1e-9, mad, np.where(rng_scale > 1e-9, rng_scale, 1.0))

    rng = np.random.default_rng(random_state)

    # Build a batch of candidate counterfactuals: each changes 1..max_changes
    # mutable features to plausible sampled values.
    candidates = np.tile(row, (n_candidates, 1))
    changed_mask = np.zeros((n_candidates, n_feat), dtype=bool)
    for i in range(n_candidates):
        k = int(rng.integers(1, max_changes + 1))
        cols = rng.choice(mutable_idx, size=min(k, len(mutable_idx)), replace=False)
        for j in cols:
            candidates[i, j] = _column_sampler(bg[:, j], rng, 1)[0]
            changed_mask[i, j] = True

    scores = np.asarray(_scores_from_predict(predict, candidates), dtype=float)
    scorable = np.isfinite(scores)
    n_scorable = int(scorable.sum())
    # `nan >= threshold` and `nan < threshold` are both False, so an unscorable
    # candidate silently reads as "did not flip". Mask it out explicitly, so the
    # count of candidates that were actually EVALUATED is a real number below.
    flipped = np.where(scorable, scores >= threshold if wants_above else scores < threshold, False)

    recourses: List[Recourse] = []
    seen = set()
    for i in np.where(flipped)[0]:
        changed_cols = np.where(changed_mask[i])[0]
        # Drop no-op "changes" (sampled value equals the original).
        keep_cols = [j for j in changed_cols if abs(candidates[i, j] - row[j]) > 1e-12]
        if not keep_cols:
            continue
        key = tuple(sorted((int(j), round(float(candidates[i, j]), 6)) for j in keep_cols))
        if key in seen:
            continue
        seen.add(key)
        proximity = float(np.sum([abs(candidates[i, j] - row[j]) / scale[j] for j in keep_cols]))
        changes = [
            FeatureChange(
                feature=names[j], from_value=float(row[j]), to_value=float(candidates[i, j])
            )
            for j in keep_cols
        ]
        recourses.append(
            Recourse(
                changes=changes,
                n_changes=len(keep_cols),
                proximity=proximity,
                prediction_after=float(scores[i]),
                sentence=_sentence(changes, desired),
            )
        )

    # Rank: fewest changes first, then smallest proximity.
    recourses.sort(key=lambda r: (r.n_changes, r.proximity))
    top = recourses[:top_k]

    notes: List[str] = []
    if immutable:
        notes.append(f"Held immutable: {', '.join(sorted(immutable))}.")
    if unobservable:
        notes.append(
            "Excluded from the search (no finite observed value in `background`, so no "
            f"plausible replacement could be sampled): {', '.join(names[j] for j in unobservable)}."
        )

    # AN UNSCORABLE MODEL HAS NOT TOLD THE APPLICANT ANYTHING, and until
    # 2026-09-10 it told them the strongest thing this function can say.
    # Measured that day: a model returning NaN for every candidate produced
    # found=False and the note "No counterfactual within the search budget
    # flipped the decision; the outcome is robust to small input changes, ...",
    # BYTE-IDENTICAL to a genuinely robust refusal that was scored 400 times.
    # This is the GDPR Article 22 surface: the sentence a rejected applicant is
    # given as the reason they cannot get a different answer. Three states.
    search_ran = np.isfinite(pred_before) and n_scorable > 0
    if not search_ran:
        if not np.isfinite(pred_before):
            reason = (
                "the model returned no finite score for this individual, so even which "
                f"side of the threshold they are on is unknown (desired_outcome={desired!r} "
                "is the default, not a measurement)"
            )
        else:
            reason = (
                f"the model returned no finite score for any of the {int(n_candidates)} candidates"
            )
        notes.append(
            f"COULD NOT CHECK: no recourse search was evaluated because {reason}. "
            "This is NOT a finding that the outcome is robust to small input "
            "changes, and it is not a finding that recourse is unavailable: "
            "nothing was measured."
        )
        found: Optional[bool] = None
    elif not top:
        found = False
        partial = (
            ""
            if n_scorable == len(scores)
            else (
                f" {len(scores) - n_scorable} of {len(scores)} candidates returned no "
                "finite score and were not evaluated."
            )
        )
        notes.append(
            "No counterfactual within the search budget flipped the decision; "
            "the outcome is robust to small input changes, or recourse needs "
            "changes beyond the observed data ranges." + partial
        )
    else:
        found = True

    return RecourseResult(
        found=found,
        prediction_before=pred_before,
        threshold=float(threshold),
        desired_outcome=desired,
        counterfactuals=top,
        n_candidates=int(n_candidates),
        notes=notes,
        n_scorable_candidates=n_scorable,
    )


def _sentence(changes: List[FeatureChange], desired: str) -> str:
    parts = [f"{c.feature} from {c.from_value:g} to {c.to_value:g}" for c in changes]
    if len(parts) == 1:
        change_text = parts[0]
    else:
        change_text = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    direction = "approved" if desired == "above_threshold" else "declined"
    return f"Had {change_text}, the decision would have been {direction}."
