# vfairness Assessment Methodology

> **Status: RATIFIED (M1.1), 2026-08-27.** This document is the authored,
> reviewed methodology for the 0.1.0 beta, ratified at M1.0 on 2026-08-22 and
> amended to M1.1 on 2026-08-27 (a verdict rule changed; see the changelog).
> `vfairness._methodology.METHODOLOGY_VERSION` holds the version (`M1.1`), and
> every `FairnessAnalyzer.get_report()` stamps it into the report under
> `methodology_version`, so a rating records the methodology that produced it.
> The version changes only under the bump policy below, never on an ordinary
> code release.

## Why the methodology is versioned separately from the code

A rating is only defensible if it can be reproduced. Code changes constantly; the
*methodology* (which metrics, which thresholds, which verdict rules) must change
rarely and deliberately, and every published rating must say which methodology
version produced it. That is why the methodology carries its own version,
independent of `vfairness.__version__`.

- **Code version** (`vfairness.__version__`): changes on every release.
- **Methodology version** (`METHODOLOGY_VERSION`, this document): changes only
  when the assessment substance changes.

## Version-bump policy

Bump `METHODOLOGY_VERSION` when, and only when, one of these changes:

- a metric's mathematical definition,
- a decision threshold or reliability-tier boundary,
- a verdict rule (for example how the overall assessment or the
  insufficient-evidence verdict is decided),
- the set of metrics that feed a rating.

A pure code refactor, a performance change, or a bug fix that does not change any
produced number does NOT bump the methodology version. A bug fix that DOES change
a produced number (for example correcting a metric that returned a wrong value)
bumps it, because the substance of the rating changed. Record every bump in the
Methodology changelog below with the date and the reason.

## 1. Scope and inputs

An assessment consumes, per protected attribute:

- **`y_true`** - the ground-truth binary labels (classification) or continuous
  targets (regression). Optional for the pure allocation metric (demographic
  parity needs only predictions and groups).
- **`y_pred`** - the model's decisions (0/1) or predicted values.
- **`sensitive_attr`** - the protected attribute (one or more), categorical.
- **`y_prob`** (optional) - predicted probabilities, required for the
  calibration, AUROC-parity, multicalibration, and ICI families.
- **`strata` / `controls`** (optional) - a legitimate conditioning variable
  (conditional demographic disparity) or bona-fide risk covariates
  (`pricing_disparity`, `conditional_adverse_impact`).

An assessment does **not** consume, and never infers, a protected attribute the
caller did not supply; it does not impute missing labels; and it does not read
any network or environment state (a test-enforced zero-telemetry guarantee). Rows
with missing values are handled by an explicit `missing_strategy` (default
`exclude`), never silently coerced.

Task types: **classification** (binary decisions), **regression** (continuous
outcomes), and **ranking** (ordered exposure). Each has its own metric family and
its own report builder.

## 2. Metrics

Every metric is defined against its primary literature source and tested against
a hand-derived golden value and, where a reference library computes the same
quantity, a cross-library parity check (`tests/test_reference_parity.py`,
`tests/test_golden_metrics.py`). The exact per-metric formulas, signatures, and
ranges are in [API_REFERENCE.md](API_REFERENCE.md); deliberate deviations from
reference libraries are in [DIVERGENCES.md](DIVERGENCES.md). The families are:

- **Independence (allocation).** Demographic parity difference (max-min selection-
  rate spread across groups, fairlearn-aligned) and `disparate_impact_ratio` (the
  selection-rate ratio / EEOC four-fifths statistic). The four-fifths 0.8 cutoff
  is applied only in the US employment overlay (Section 5), never baked into the
  jurisdiction-neutral metric.
- **Separation (error parity).** Equal opportunity (TPR gap), equalized odds
  (max of TPR and FPR gaps, Hardt, Price & Srebro 2016), predictive equality
  (FPR gap), FNR parity, and accuracy parity. Every one of these returns NaN, not
  a false 0.0, when a group's rate is undefined (an empty denominator), so an
  unmeasurable group routes to insufficient evidence rather than sealing false
  parity.
- **Sufficiency (calibration / predictive value).** Predictive parity (PPV gap),
  negative predictive value parity, calibration difference, integrated
  calibration index (Austin & Steyerberg 2019), and multicalibration
  (Hebert-Johnson et al. 2018). ICI and calibration difference return NaN when
  fewer than two groups are assessable.
- **Continuous / decision-utility / integrity.** `pricing_disparity` (residual
  priced-outcome disparity after controlling legitimate risk factors, estimated
  jointly per Ross & Yinger 2002), `conditional_adverse_impact` (the binary
  counterpart, a joint-fit average-marginal-effect gap), `net_benefit_parity`
  (decision-curve net benefit across groups, Vickers & Elkin 2006),
  `conditional_demographic_disparity` (within-stratum selection-rate spread; see
  the DIVERGENCES note on its relationship to the Wachter/Mittelstadt/Russell
  composition statistic), and `multivariate_proxy_leakage` (reconstruct-the-
  attribute proxy audit).
- **Ranking.** Exposure and representation parity over ranked output, with the
  documented exposure-decay models (linear, geometric, log).

## 3. Statistical validation

A fairness gap is a sample estimate, not a fact; the methodology treats it as one.

- **Confidence intervals.** Every `*_with_ci` metric returns a bootstrap interval
  (stratified by group to preserve group proportions). There is no Bayesian
  estimator for a disparity metric in this library: below the small-sample
  threshold `method='auto'` doubles the resamples and warns, and `method='bayesian'`
  is accepted for API compatibility but warns that it is not implemented and runs
  that same doubled-resample bootstrap. `StatisticalResult.method` on the returned
  interval names the bootstrap that ran and never says `bayesian`. Genuine credible
  intervals exist for the quantities that have a conjugate posterior and are
  reached by their own functions, not by `method=`: `bayesian_proportion_ci` for one
  group's rate, `bayesian_difference_ci` for a two-group rate gap (used by the
  intersectional findings path), `bayesian_mean_ci` for a mean, and
  `get_group_metrics_with_ci(method='bayesian')` for the per-group rates.
  Custom-shape statistics (needing `y_prob`, strata, or continuous
  controls) get an interval via `bootstrap_over_index`, which resamples row
  indices stratified by group. Interval coverage is validated by Monte Carlo, and
  for the non-negative max-min spread statistics the interval used for the verdict
  is constructed so that it is not biased upward at the parity null (a naive
  percentile interval of a max of absolute differences has zero coverage at the
  null and would declare every fair model unfair).
- **Effect sizes.** Cohen's d / h and risk / odds ratios are available alongside
  the raw gap, so a statistically significant but negligible gap is legible as
  such.
- **Multiple-testing correction.** When several metrics or several groups are
  tested together, Benjamini-Hochberg (FDR) or Bonferroni correction is applied
  via `apply_multiple_testing_correction`; the sealed intersectional path uses
  Benjamini-Hochberg.
- **Reliability tiers.** Every per-group number carries a sample-size honesty
  flag, so a 31-person group is not read like a 3,000-person one. The boundaries
  (source: `_statistics.reliability_tier`, following the Fairlearn small-group
  convention and the Turing M3 minimum-reporting guidance) are:

  | Tier | Group size n | Treatment |
  |---|---|---|
  | reliable | n >= 100 | reported normally |
  | caution | 30 <= n < 100 | reported with a wider-CI caveat |
  | underpowered | 10 <= n < 30 | reported as indicative only |
  | invalid | n < 10 | shown but NOT interpreted and excluded from the verdict |

## 4. Verdicts

Every verdict has **three** states, never two: a pass, a fail, and an explicit
could-not-assess. The library exposes two verdict surfaces, and it is important to
be precise about which rule each uses, because they answer slightly different
questions.

Which side of a threshold is the *fair* side depends on the metric, and it is
decided once, for every surface, by `vfairness_metrics._metric_direction`
(M1.1). A violation magnitude (a difference, gap, disparity, error) passes at or
below its bound; a parity ratio (the four-fifths family) passes at or above it.
A metric whose direction that resolver cannot determine is **not graded at all**:
it is could-not-assess, never a pass. No surface re-derives the direction from
the metric name on its own.

**(a) The `FairnessAnalyzer` per-metric verdict is interval-based** (source:
`vfairness_metrics.analyzer._fairness_verdict`, which requires the metric name so
the direction is resolved rather than assumed). The verdict is read from the
whole confidence interval, not the point estimate, against the metric's own fair
band (`[0, threshold]` for a spread metric, `[threshold, ...)` for a ratio):

- **fair** - the WHOLE confidence interval lies within the band (affirmative
  evidence of practical equivalence; a one-sided equivalence / TOST reading).
- **unfair** - the WHOLE confidence interval lies outside the band (significant
  evidence of exceedance).
- **insufficient_evidence** - the interval straddles the threshold, the metric
  could not be measured (NaN), or its direction is unknown.

So a tight, large-sample gap just over the threshold reads **unfair**, while a
wide, small-sample gap of the same point value reads **insufficient_evidence**,
not a spurious pass or fail. The back-compatible boolean `is_fair` is exactly
`verdict == "fair"`.

**(b) The report builders classify against the threshold on the point estimate**
(source: `classification_fairness_report`, `regression_fairness_report`). Each
metric is **PASS** (within its own fair band), **FAIL** (outside it), or
**NOT_ASSESSABLE** (the value is NaN / undefined, its direction is unknown, or,
for regression, the outcome scale is degenerate). The `fairness_score` is
computed over assessable metrics only; NaN metrics are never counted as failures.
When no metric is assessable the score is **`None`** ("could not check"), never a
number: reporting `0.0` and reporting `1.0` are two opposite lies about the same
absence of evidence, so `fairness_score` is `Optional[float]` and a consumer must
check it before formatting it as a percentage. When fewer than two valid groups
survive filtering, the whole assessment is marked **NOT ASSESSABLE** rather than
certifying a vacuous `fairness_score = 1.0`, and groups below `min_group_size` are
surfaced as an explicit `insufficient_evidence_groups` list.

The two surfaces share the same NaN-is-not-a-pass and degenerate-data-is-not-a-
pass discipline; they differ in that (a) reads the interval and (b) reads the
point estimate against the threshold. A caller who needs the statistical
three-state reading uses the analyzer surface; a caller who needs the compact
report card uses the report surface. (Unifying the report card onto the interval
verdict is tracked as a future methodology change, because it would alter
published pass/fail counts and therefore must bump the methodology version.)

Common to both:

- A metric that cannot be computed on the data (an undefined group rate, a single
  assessable group, a degenerate label distribution) is could-not-assess, and
  every deployment-gate surface (`ModelFairnessGate`, `FairnessTestSuite`,
  `BiasMonitor`) fails closed on it rather than approving an unmeasured metric.
- A group in the `invalid` reliability tier is excluded from the verdict and
  surfaced as its own insufficient-evidence record, never silently dropped.

## 5. Jurisdiction overlays

Legal rules are applied as explicit, named overlays on top of the jurisdiction-
neutral metrics, never baked into them (source: `legal/` rule engine; the metric
functions carry no legal constant):

- **US employment (four-fifths / EEOC).** The 0.8 selection-rate-ratio rule is
  applied by the overlay to `disparate_impact_ratio`; the metric itself reports
  the raw ratio.
- **EU / CJEU objective justification.** `conditional_demographic_disparity`
  measures the within-stratum residual disparity the overlay reads.

The neutral metric and the legal reading are kept separate so the same measured
number can be read under more than one jurisdiction, and so a change in law does
not silently change a metric. Overlay application is tested to confirm the legal
constants do not leak into the neutral metric path.

## 6. Limitations

Stated plainly, because an honest limit is part of the methodology:

- **Observational, not causal.** Except where a metric is explicitly causal
  (`conditional_adverse_impact`, `pricing_disparity`, the causal module), the
  metrics describe associations in the supplied data; they cannot by themselves
  establish that the model caused a disparity or that a control is truly
  legitimate rather than a proxy.
- **Garbage in, garbage out on the protected attribute.** The assessment is only
  as good as the supplied `sensitive_attr`; a mislabeled or coarsened attribute
  produces a mislabeled or coarsened assessment. The library measures the groups
  it is given; it cannot discover a protected group that is absent from the data.
- **Small groups bound what can be claimed.** Below the `invalid` tier a rate is
  not interpretable; the honest output there is insufficient evidence, not a
  number.
- **Intersectional coverage is bounded by power.** Crossing attributes multiplies
  the number of cells and shrinks each; an unattended or underpowered intersection
  reduces the evidence a verdict can rest on rather than being read as fair.
- **The methodology does not set the fairness contract.** Which metric and
  threshold are contractually binding for a given use case is the owner's
  decision; the library recommends and measures, it does not mandate.

## Methodology changelog

| Methodology version | Date | Change |
|---|---|---|
| M1.1 | 2026-08-27 | Verdict rules only; no metric definition, threshold or metric-set change. (1) Every metric is now graded in ITS OWN direction, resolved once by `_metric_direction`: the ratio family (four-fifths / disparate impact) passes at or ABOVE its bound, where both verdict surfaces previously assumed a `[0, threshold]` band for every metric and so graded a ratio of 0.00 as a pass and perfect parity as a fail. (2) A metric whose direction cannot be resolved fails closed: it is could-not-assess and is excluded from the verdict and from the `fairness_score`, where it was previously graded as if lower were better and therefore usually passed. (3) A metric that cannot be measured returns NaN and is NOT_ASSESSABLE, never a 0.0 "perfect parity" pass. (4) With no assessable metric the `fairness_score` is `None`, not `0.0` or `1.0`. This moves published PASS/FAIL counts, which is why it is a bump and not a bug fix. Minor rather than major: only how a value is compared to its threshold changed. |
| M1.0 | 2026-08-22 | First ratified methodology: all six sections authored and reviewed against the implemented code for the 0.1.0 beta. Supersedes the M0.1-draft scaffold. Superseded in M1.1 on the verdict-direction rules in Section 4; the rest stands. |
| M0.1-draft | 2026-08-11 | Initial scaffold and versioning mechanism (not yet ratified). |
