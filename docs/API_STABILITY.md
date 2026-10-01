# API Stability

**vfairness** follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This document declares which parts of the library are a **frozen public surface**
covered by that guarantee, and the policy for changing them.

## Versioning intent

- **MAJOR** (`X`.0.0): may remove or change the behaviour of the frozen surface
  below.
- **MINOR** (0.`X`.0): adds functionality in a backward-compatible way. May
  **deprecate** a frozen symbol but not remove or repurpose it.
- **PATCH** (0.0.`X`): backward-compatible bug fixes only. Numeric outputs of a
  frozen metric may change only to fix a demonstrable correctness bug, which is
  called out in the [CHANGELOG](../CHANGELOG.md).

While the library is pre-1.0, we treat the surface below as stable in practice:
breaking changes are reserved for a MAJOR bump and always announced in the
changelog. The `TaskResult` envelope carries an explicit `schema_version` for
consumers that need to detect envelope changes independently of the package
version.

## The three-state contract

Before the symbol list, one behaviour that runs through all of it, because it
governs what every frozen return value means.

Every result in this library has three states, not two: assessed-pass,
assessed-fail, and **could-not-check**. Could-not-check is never collapsed into
either of the other two. A metric that was never computed is not a pass and it is
not a failure, and it gets no number, no verdict and no place in any count that
implies a measurement was taken.

On the frozen surface that means:

- **A frozen classification or regression metric returns `float('nan')` when it
  could not be computed**, in particular when fewer than two groups survive
  `min_group_size`. It does not return `0.0` or `1.0`, which are the
  perfect-parity readings. The `*_with_ci` variants return a `StatisticalResult`
  whose `point_estimate` and both bounds are NaN.
- **`report["assessment"]["fairness_score"]` is `Optional[float]`.** `None` is the
  third state and is not the same value as `0.0`.
- **`report["assessment"]["assessable"]` is `False`** when the data is degenerate,
  and every metric entry is then listed under `not_assessable_metrics`, never
  under `passed_metrics`.

Callers must rule out NaN and `None` before grading a value. Every comparison
against NaN is `False`, both `nan < t` and `nan > t`, so a guard phrased as
`if value > threshold: fail()` never fires and an unmeasured metric slides
through as though it had passed. That is a property of IEEE floats, not of this
library. The frozen surface returns NaN rather than a sentinel precisely so that
the value carries no false claim of its own; turning it into a correct verdict is
the caller's `math.isnan()` check.

## Frozen public surface

These symbols are imported directly from the top-level package
(`from vfairness import ...`) and are considered stable. Their names, call
signatures, and documented return shapes will not change without the deprecation
process below.

Every symbol listed below was verified present, importable and matching its
documented signature by execution on **2026-08-28**, against the working tree at
that date. Reproduce with:

```bash
python -c "
import importlib, inspect, vfairness
mods = {
    'classification': ['demographic_parity_difference', 'demographic_parity_ratio',
                       'equal_opportunity_difference', 'equalized_odds_difference',
                       'predictive_parity_difference', 'calibration_difference'],
    'regression': ['mae_parity_difference', 'rmse_parity_difference',
                   'mean_prediction_difference', 'r2_parity_difference', 'residual_bias'],
    'ranking': ['exposure_parity_difference', 'exposure_parity_ratio',
                'attention_weighted_rank_fairness', 'normalized_discounted_kl_divergence'],
}
for sub, names in mods.items():
    m = importlib.import_module('vfairness.evaluation.vfairness_metrics.' + sub)
    for n in names:
        print(n, inspect.signature(getattr(m, n)), 'top-level:', hasattr(vfairness, n))
"
```

### Analyzer and result types

- `FairnessAnalyzer` - the unified analysis entry point, including its
  constructor keyword arguments (`task_type`, `min_group_size`,
  `missing_strategy`, `backend`), its metric methods, `compute_all_metrics`, and
  `get_report`.
- `MetricResult` - the structured metric output dataclass and its `to_dict()`.
- `FairnessReport` - the documented key structure of the report `dict` returned by
  `get_report`, `classification_fairness_report` and `regression_fairness_report`,
  together with its nested structure types `AssessmentReport`, `DataInfo`,
  `ExplanationsReport`, `MetricStatusEntry` and `InsufficientEvidenceGroup`. These
  are static-only `TypedDict` annotations (the report is the same plain,
  JSON-serialisable `dict` it has always been); the **required** keys named in
  each are frozen, while the `NotRequired` keys appear only when their feature is
  requested (confidence intervals, explanations) or for a task type (regression).

  `AssessmentReport` is a total `TypedDict`, so all seven of its keys are
  required and therefore frozen: `fairness_score`, `assessable`,
  `passed_metrics`, `failed_metrics`, `not_assessable_metrics`,
  `insufficient_evidence_groups` and `summary`. The three that carry the
  could-not-check state (`assessable`, `not_assessable_metrics`,
  `insufficient_evidence_groups`) need no separate entry on this list: they were
  covered the moment `AssessmentReport` was frozen, and a consumer may rely on
  each being present on every report, empty rather than absent when there is
  nothing to report. `MetricStatusEntry.status` is frozen to the three values
  `"PASS"`, `"FAIL"` and `"NOT_ASSESSABLE"`; `threshold` is `None` for the third.

  `DataInfo` is declared `total=False`, so none of its keys are required and none
  are frozen. Its missing-data provenance fields (`original_size`, `final_size`,
  `n_excluded`, `missing_strategy`) are produced for every classification and
  regression report today, and `docs/API_REFERENCE.md` documents them, but they
  are not covered by the guarantee below. Read them defensively.

### Core classification metrics (`vfairness.evaluation.vfairness_metrics.classification`)

- `demographic_parity_difference`
- `demographic_parity_ratio`
- `equal_opportunity_difference`
- `equalized_odds_difference`
- `predictive_parity_difference`
- `calibration_difference`

### Core regression metrics (`vfairness.evaluation.vfairness_metrics.regression`)

- `mae_parity_difference`
- `rmse_parity_difference`
- `mean_prediction_difference`
- `r2_parity_difference`
- `residual_bias`

### Core ranking metrics (`vfairness.evaluation.vfairness_metrics.ranking`)

- `exposure_parity_difference`
- `exposure_parity_ratio`
- `attention_weighted_rank_fairness`
- `normalized_discounted_kl_divergence`

Each core metric takes the standard positional inputs
(`y_true, y_pred, sensitive_attr` for classification/regression;
`rankings, groups` for ranking) plus keyword-only options such as
`min_group_size`, and returns a Python `float`. Documented exceptions:
`calibration_difference` requires a fourth positional argument, `y_prob` (the
predicted probabilities the calibration bins are built from);
`attention_weighted_rank_fairness` returns a `RankingFairnessResult` dataclass
whose `.value` is the float; and `residual_bias` returns a `Dict[str, float]`
mapping each group name to its mean residual (`y_true - y_pred`), not a single
float. `normalized_discounted_kl_divergence` also accepts an optional third
positional argument, `target_distribution`. Exact numeric
behaviour is guarded by golden-file tests in `tests/test_golden_metrics.py`.

**The ranking metrics now honour the three-state contract, and this paragraph
used to say they did not.** It disclosed a real defect: all four returned the
perfect-parity value when fewer than two groups survived `min_group_size`, so a
comparison that never ran certified as fair. That was closed on 2026-09-06.

Re-measured on 2026-09-07 with the reproduction this paragraph itself specified,
105 items in one group and 15 in another at `min_group_size=30`:

| metric | was | is |
| --- | --- | --- |
| `exposure_parity_difference` | `0.0` | `nan` |
| `exposure_parity_ratio` | `1.0` | `nan` |
| `normalized_discounted_kl_divergence` | `0.0` | `nan` |
| `attention_weighted_rank_fairness` | `.value == 0.0`, `is_fair` True | `.value` `nan`, `is_fair` `None` |

The frozen surface is uniform again: every between-group metric in this library
returns NaN for a comparison that could not be made. **The group-size workaround
this section used to tell readers to apply is no longer needed**, and following
it now would only hide the honest NaN behind a guard of your own.

Under the versioning rules above the change is a correctness fix, permitted in a
PATCH, and it is called out in the CHANGELOG. It is stated for users at
`docs/API_REFERENCE.md`, section "Ranking Metrics".

### Task-result envelope

- `vfairness.result.TaskResult` and `TaskResult.to_dict()`. The serialized
  envelope keys (`schema_version`, `task_type`, `success`, and, when present,
  `data` / `error` / `warnings`) are stable; `schema_version` is bumped only on
  a breaking change to the envelope shape.

### Exceptions

The library raises a small, stable exception hierarchy rooted at `VfairnessError`
(all importable from the top-level package):

- `VfairnessError` - base class for every error the library raises on purpose.
- `InvalidDataError` - the inputs are malformed (wrong shape, length mismatch,
  non-binary labels, and similar).
- `InsufficientDataError` - there is not enough data to assess a group or metric.
- `ProtectedAttributeError` - a protected-attribute column is missing or unusable.
- `ConfigurationError` - an argument is out of range or an unknown option was
  requested (unknown metric, method, or exposure type).

For backward compatibility each subclass also inherits from the built-in it
replaces (all four extend `ValueError`), so existing `except ValueError` handlers
keep working. Catch `VfairnessError` to handle any library error, or a specific
subclass for finer control. A few internal lookups still raise a plain `KeyError`.

### Opt-in result cache

`FairnessAnalyzer(cache=True)` memoizes `compute_all_metrics()` and `get_report()`
by their arguments. Results are deep-copied on both store and return, so mutating
a returned report never corrupts the cache. Caveat: the cache key is the method
arguments only, not the input arrays, so it assumes the analyzer's `y_true`,
`y_pred`, and `sensitive_attr` are not mutated in place after construction; if you
do mutate them in place, cached results go stale (call `clear_cache()` or build a
new analyzer). The cache is off by default and is not part of the frozen surface.

### Output branding switch

- `set_branding(enabled: bool | None = True) -> None` - remove or restore the
  validant.ai mark on generated charts and reports. `None` clears the override
  so the `VFAIRNESS_BRANDING` environment variable decides again.
- `branding_enabled() -> bool` - the currently resolved state.

Resolution order is explicit call, then `VFAIRNESS_BRANDING`, then the branded
default. The `"branding"` key in a `render_svg` data dict overrides both for a
single chart. Which elements carry the mark is a rendering internal and may
change; that a `False` here removes all of them is the frozen guarantee, and
`tests/test_branding.py` enforces it across every template and report format.

## Disclosed behaviour changes on the frozen surface

The versioning rules above allow the numeric output of a frozen metric to change
only to fix a demonstrable correctness bug, and require that change to be called
out. The changes below are the ones a consumer pinned to an earlier revision will
notice, all of them recorded in the [CHANGELOG](../CHANGELOG.md) under
`[0.1.0] - UNRELEASED`. None of them removes, renames or repurposes a frozen
symbol, so no deprecation cycle is owed; each of them changes a value that was
previously fabricated.

- **Frozen metrics now return NaN where they previously returned a sentinel.**
  Degenerate cases that used to yield `0.0` or `1.0`, the two perfect-parity
  readings, now yield `float('nan')`. Code that compared the result against a
  threshold without checking for NaN previously saw a clean pass and now sees a
  value that fails every comparison in both directions. That is the intended
  direction of the change: the earlier value asserted a parity nobody had
  measured, and NaN asserts nothing. Callers still have to add the
  `math.isnan()` branch to turn it into a correct verdict.
- **`assessment["fairness_score"]` is `Optional[float]`.** It is `None` when
  nothing was assessable. Consumers that format it directly as a percentage need
  a `None` branch.
- **Ungraded metrics moved out of `passed_metrics`.** A metric that could not be
  compared is now listed under `not_assessable_metrics` with
  `status: "NOT_ASSESSABLE"`, and is excluded from `fairness_score` rather than
  counted as a pass. A count of `passed_metrics` therefore drops for reports that
  previously absorbed ungraded metrics into it, and that lower count is the
  accurate one.
- **`assessment["summary"]` gained a trailing data-provenance clause**, present
  on every report of the form `(data provenance: <final> of <original> rows
  assessed, <n> excluded, missing_strategy='<strategy>')`. `summary` is frozen as
  a required key holding human-readable prose, not as a fixed string, so anything
  parsing it was already outside the guarantee; read `data_info` for the values.

Nothing added during this work needs to be added to the frozen list. The new
three-state fields all live inside `AssessmentReport`, which was already frozen
and is a total `TypedDict`, so they are covered as required keys. The
could-not-check rendering behaviour lives in `vfairness.rendering`, which the
next section explicitly leaves uncovered.

## Not covered (may change without a major bump)

Everything outside the list above is internal or experimental, including but not
limited to: modules prefixed with an underscore (`_registry`, `_manifest`, ...),
the `rendering.adapters_*` modules and SVG template internals, the `operations`
orchestration internals (Pulse pipeline internals, scoring helpers), the `llm`,
`agents`, `multi_agent`, `xai`, `causal`, and `mcp` subsystems, and any symbol
documented as experimental. These evolve faster and should be pinned by exact
version if you depend on them.

One clarification about `vfairness.rendering`, which is the largest thing on that
list. Which template draws what, the adapter module layout, the SVG internals and
the exact wording on a canvas are all rendering internals and may change. The
three-state behaviour is not treated as one: an adapter handed an input that
measured nothing renders its could-not-check state or refuses, and does not
render a verdict. That behaviour is pinned by the adapter guards under `tests/`,
each of which pairs its could-not-check assertion with a healthy-input control so
that a fix which merely suppresses every verdict fails the suite. Checked on
2026-08-28 by calling all 44 public `*_to_svg` adapters with the emptiest input
each accepts: 39 rendered the NOT CHECKED panel, 5 raised, none produced a
verdict. Treat that as a property the library intends to keep rather than a
frozen signature; the symbols themselves remain uncovered by the guarantee above.

## Deprecation policy

When a frozen symbol must change:

1. The old symbol keeps working and emits a `DeprecationWarning` that names the
   replacement.
2. The deprecation ships for at least **one minor version** before removal.
3. The removal happens only in a subsequent MAJOR release and is recorded in the
   [CHANGELOG](../CHANGELOG.md).

Example:

```python
import warnings

def old_metric(*args, **kwargs):
    warnings.warn(
        "old_metric is deprecated since 0.2.0 and will be removed in 1.0.0; "
        "use new_metric instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_metric(*args, **kwargs)
```
