# Changelog

All notable changes to **vfairness** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
See [docs/API_STABILITY.md](docs/API_STABILITY.md) for the frozen public surface
and the deprecation policy that governs what a version bump is allowed to change.

## [0.1.0] - 2026-10-01

### The beta gate is met, and two checks anyone can re-run (2026-10-01)

`scripts/release_gate.py` reports **BETA READY**: all eight beta conditions are met.
The 1.0 gate is not, on one condition only (G2, below).

- **Two executable checks are now beta conditions.** B6: every evaluation
  capability that an established library also computes gives the same answer as it
  on clean data (34 of 34 agree with fairlearn, scikit-learn, statsmodels or scipy;
  `tests/test_reference_parity_registry.py`, run in the cross-library parity
  workflow). B7: every capability that measures refuses to invent a number on
  broken data and still measures healthy data (109 of 109;
  `scripts/broken_data_check.py`, `tests/test_honest_on_broken_data.py`). The
  expected answer for each broken input is counted from the data, never chosen by
  hand, and each check was shown to fail when the code was broken on purpose.
- **Every public code unit has been examined**: 1,580 of 1,580.
- **Second-examiner confirmation of every grade moved from the beta gate to 1.0**
  (G2), because B6 and B7 already re-check every capability that returns a number
  by running it. About 600 examined units still await that second check; where it
  has happened, about one first grade in three was overturned, and every defect a
  re-check found was fixed.

#### Fixed

- `subgroup_robustness_audit` read a missing prediction as a "no": with every
  prediction missing it reported rates of 0.0 and `gerrymandering_detected=False`.
  Rows without a prediction are now removed and counted
  (`n_rows_missing_prediction`), and "not detected" over a partial population is
  `None`.
- `RAGBiasAnalyzer.full_analysis` gave a group that never sent a query a disparity
  of 1.0 and a bias finding; it now refuses (NaN, `None`, a `not_assessed_reason`).
- `AnchorsExplainer` turned a model that returned no decision into "class 0,
  precision 1.0, coverage 1.0"; it now refuses.
- `EmergentBiasDetector` reported `is_significant=True` from a single-row group;
  below two rows per group significance is `None`.
- `TextFairnessAnalyzer` refused a genuine "no difference" when every score was
  tied, with a false warning; the significance floor now comes from the sample
  sizes.
- `intersectional_disparity_analysis` dropped every single-attribute comparison
  once any row was excluded and said none had been supplied; attributes are now
  aligned to the kept rows or named as unusable.

#### Added

- `selection_rate_disparity_matrix` returns `headline_excluded_groups`: the groups
  too small to read that the headline left out (`[]` when none).
- `SubgroupAuditResult.n_rows_missing_prediction`.


### Return types and refusals that changed (2026-09-11 to 2026-09-30)

No public symbol was removed or renamed. These changes can affect code written
against earlier development snapshots; each is listed under its date below.

- **New result types that still behave like the old ones.**
  `find_feasible_thresholds` returns `FeasibleThresholds`, a `dict` subclass;
  `identify_proxy_variables` and `find_proxy_chains` return `ProxyScreenResult` and
  `ProxyChainResult`, both `list` subclasses. Loops and lookups keep working; the
  difference is that they also say what could not be assessed, and
  `FeasibleThresholds` refuses to look up a group that was never assessed.
  `FeatureEngineeringAnalyzer.analyze_proxies` and
  `FeatureCorrelationMatrix.get_high_correlations` return result objects of the
  same kind.
- **Can now return `None`, meaning "could not check":**
  `BaseFairnessConstraint.is_satisfied`, `judge_is_subject`,
  `PipelineTracker.identify_bias_source`,
  `CausalFairnessGraph.has_direct_discrimination`,
  `AdaptiveThresholdManager.is_alert_warranted`, the first element of
  `FairnessDriftDetector.detect_drift_mmd`,
  `IntegratedCalibrationResult.has_significant_disparity`,
  `MulticalibrationResult.is_multicalibrated`, `proxy_score` values,
  `BiasAuditReport.get_critical_count` and `get_high_risk_features`,
  `ParetoPoint.dominates` and `slack_adversarial_probe`. Test the result with
  `is None`, not with truthiness.
- **Now raise instead of returning a misleading value:** `require_checked()` with
  no names, `assert_fairness(metrics=[])`, `create_fairness_callback` with
  `fail_on_violation=True` and no thresholds, `CounterfactualTester.run_test` with
  no placeholder in the template, `ThresholdAnalyzer` on NaN labels,
  `NonDeterminismAnalyzer.equivalence_test` with a non-finite SESOI, the Integrated
  Gradients and TreeSHAP explainers on a feature-name count mismatch,
  `signed_constraint_value` for a group absent from the data, and a negative
  `lambda_fairness`.
- **Return a value where they returned `None`:**
  `SupabaseWriter.update_job_progress` returns the row count (and raises on zero
  rows); `WorkerLoop.run` returns an exit code.
- **Serialisation:** `SerializableMixin.to_json` writes `null` instead of the
  non-standard `NaN`/`Infinity`; the MCP tools write infinity as the string
  `"Infinity"`; `create_github_check` can return `neutral`.

### The first examination of every remaining unit (2026-09-30)

Every public unit that had never been graded was run on input where its quantity
cannot exist. These are the user-visible fixes that came out of it.

#### Fixed

- **`prepare_protected_attributes`** rounded a derived age, so a 17.6-year-old landed in
  the `18-24` band and 64.6 in `65+`. Ages now truncate; a raw age column gets the
  same plausibility window as a derived one (sentinels such as -999 and 999 are
  excluded); a text-typed age column gets the canonical bands, not quantiles.
- **`prepare_protected_attributes`** no longer fills absent values with a minted
  `"missing"` level that counted as a second group, and a column of blank strings is
  no longer usable as a group named `""`.
- **The SVG renderers** drew a full bar for `True`, a measured-looking 4 px bar for NaN,
  the word "None" in label slots, a clean tick for a missing severity and HIGH for
  `True`. All now render the could-not-check state.
- **`CalibrationReport`** canvases painted a withheld `is_well_calibrated=None` as
  MISCALIBRATED, and a `None` or `pd.NA` group key became a row labelled "None"
  ranked best calibrated. Both now render as not assessed.
- **`BaseRegularizer`** subclasses accepted any `strength`: 0.0 silently applied no
  penalty and a negative value pushed the optimiser toward unfairness. Both refused.
- **`TaskResult.from_envelope`** read `{"success": "false"}` as a success; a non-boolean
  flag is now a failure. The legal column classifier no longer iterates the
  characters of a single string attribute such as `"race"`.
- **`ingest_and_run_pulse`** and `run_pulse` published an adverse deployment opinion
  about groups named `nan`, `<NA>`, `NaT` or `missing`. Absent values are excluded
  and counted; a level somebody chose (such as `"None"`) is still a group.
- **`SemanticQualityScorer`** gained the readability gate the other scorers have: one
  ASCII digit on a Japanese answer no longer yields a measured quality score.
- **The Pulse vision probe** no longer prints "Within representation tolerance" beside its
  own refusal when labels are only partly present.
- **`GroundednessScorer.score`** (`vfairness.validity`) returned perfect groundedness
  against zero retrieved passages; it now refuses, and every rung is range checked.
  An answer of JSON `null` is no longer scored as the word "None".
- **`ThresholdOptimizer`** reached perfect demographic parity by rejecting every applicant
  and reported `constraint_satisfied_ True`; that state is now refused.
- **The reweighting report** no longer recommends a method that "reduces disparity by
  0.000", and never presents a negative reduction as a reduction.
- **`SerializableMixin.to_json`**, shared by about twenty result types, wrote bare `NaN`
  and `Infinity` (invalid JSON) and stringified `pd.NA`, `pd.NaT` and numpy scalars.
  Output is now strict JSON with absent values as `null`.
- **Security:** the outbound address guard (`validate_endpoint`, `GuardedSession`,
  `guarded_post`) allowed multicast, NAT64-embedded and IPv4-compatible IPv6
  addresses, including a route to the cloud metadata address. All are refused;
  loopback for a local model server still works.
- **`GradientReversalLayer`** (constructor and `set_lambda`) and `Adversary` now refuse
  a negative, zero or non-finite `lambda_`, a single-group adversary, an unknown
  `activation` and a non-finite `dropout`, each of which trained silently.
- **`GroupThresholdOptimizer`** reported `is_feasible True` under a vacuous `tolerance`
  (2.0 on a [0, 1] metric); `set_epoch` now validates its input; `BaseFairnessLoss`
  is now genuinely abstract.
- **`MultivariateProxyResult.was_assessed`** returned True for an AUC of NaN, inf or
  `True`. It now requires a finite score in range.
- **The XAI storage serialiser** silently dropped one of two keys that convert to the
  same camelCase name; it now refuses and names both.
- **The health-score figures** no longer draw a dial titled "Health Score" over a
  withheld score with no explanation, no longer swallow a failed figure into a blank
  panel, and a store ingested with timezone-aware timestamps no longer raises.
- **Representation shares** now carry a caveat naming the count when an attribute
  holds the literal string `"None"` and similar spellings, instead of silently
  treating them as a group.
- **`LinearShapExplainer.explain_local`** (and the other shap adapters) returned one
  explanation of 15 features for a 5-row matrix, and a "complete" explanation of an
  empty instance. A matrix or an empty instance is now refused.
- **`supports()`** on the four model-agnostic explainers said yes to an unfitted model;
  it now asks `prediction_fn_available`.
- **Proxy correlation charts** drew an unmeasured feature at x = 0, counted it as
  NEGLIGIBLE and let the heatmap read "no pair reaches the threshold". A boolean is
  no longer graded as a perfect correlation, and a correlation introduced by a
  transformation (0.0 to 0.9) is no longer reported as "no change".
- **The synthetic resampler** now discloses when a feature column is text-typed and was
  therefore copied verbatim rather than interpolated.
- **`attribute_distribution_change`** scored every node 0.0 for a binary outcome; it now
  attributes the shift (0.964 to the input that moved in the measured case), and
  refuses shares for a drift of exactly zero.
- **`compute_counterfactual`** never produced a counterfactual (its SCM was not
  invertible); it now returns real values (2.986 against a true 3.0).
- **`identify_paths`** returned one "add the missing variables" verdict for five
  different graph shapes, including a disconnected graph; each now gets its own
  answer, and identifiability is no longer asserted when `dataset_columns` is
  omitted. NaN, `pd.NA` and blank strings are refused as variable names.
- **The CI/CD metric row** printed "Fail" for an unmeasured metric and for a fair model
  under a vacuous threshold; it now prints the state that applies.
- **`FairnessAssertionError`** raised `ValueError` on a text-typed value and escaped
  `parametrize_fairness`; a suite that ran nothing no longer writes JUnit
  `tests="0" failures="0"` as a pass.
- **`decision_recommendation`** returned `deploy_treatment` with confidence 0.72 when no
  intersection had been compared; it now returns `extend_experiment`.
- **`AttributionResult.top`** ranked an unmeasured (NaN) importance first; it now ranks
  measured values only.
- **`calibration_disparity`** reported 0.0 disparity (and the verdict flipped with group
  order) when one group's ECE was NaN; it now refuses.
- **`compute_pareto_frontier(maximize=[])`** maximised every metric; an empty list now
  means none. `compute_max_ratio` no longer returns its 1.0 initialiser for all-NaN
  or never-selected groups.
- **`GroupManager`** minted intersectional groups such as `F_<NA>` from absent values,
  and ignored `missing_strategy='as_group'` on the intersectional path. Both fixed.
- **`mediation_analysis`** no longer publishes `proportion_mediated` 1.0 when the
  indirect and total effects have opposite signs; a single-resample bootstrap no
  longer yields a zero-width interval.
- **`vacuous_bound_reason`** judged any unrecognised `BoundRole` by the threshold rule,
  in the wrong direction for some roles; an unknown role now fails closed.
- **Permutation importance (`global_importance`)** published importances of order 1e15
  for a constant target, and the NumPy and sklearn paths disagreed in direction. It
  now refuses for a constant target, NaN targets and non-finite predictions.
- **The adversarial probes** flagged a clean model as scaffolded under a negative
  `threshold`; both entry points now refuse a threshold that cannot be false.
- **`TemporalTracker.detect_drift_ewma`**, `detect_drift_cusum` and
  `detect_feedback_loop` reported "no drift" for a NaN or inverted operand
  (`sigma_limit`, `span`, `threshold`, `drift_limit`, `alpha`); every operand is
  now validated.
- **`MultiAgentRunHarness.record_routing`** recorded `pd.NA`, `NaT`, masked values and
  the strings `"None"` and `"nan"` as groups; all eleven absence spellings are
  refused, while real labels such as `"NA"` are kept.
- **`GroupthinkDetector.analyze_convergence`** counted unmeasurable permutation draws as
  evidence for significance, and accepted agents missing from round 0. Both fixed.
- **`ThresholdAnalyzer.analyze_threshold`** (and `analyze_threshold_range`) accepted a
  NaN threshold, which rejected everyone and became the "fairest" row of a sweep. It
  is refused; regions now carry a measured `n_distinct_decisions`, and
  `n_thresholds` below 2 is refused.
- **In-processing losses**: a negative, cancelling or vanishing sample weight is
  refused; a negative or non-finite `lambda_fairness` raises at construction; the
  counterfactual loss refuses identical arms in any shape; the individual-fairness
  loss computes coverage pair by pair under every distance; an adversary whose
  output went non-finite is no longer reported as trained.
- **`FairnessMonitor`** read a text-typed decision column (`"1"`) as all rejections and
  published perfect parity; text decisions are now parsed before comparison.
- **`ModelFairnessGate`** approved a breach when `blocking_metrics` held a misspelt name,
  approved everything under `min_group_size=nan`, and sized groups by rows where a
  metric rests on labelled rows. All three fixed.
- **`baseline_comparison_summary`** claimed "no accuracy cost" for an all-NaN label set;
  it now requires labels.
- **Fairness radar and heatmap canvases** now name a protected group that lost every
  prediction instead of drawing a clean verdict without it.
- **The LLM readability gate** no longer treats a support URL inside a Japanese refusal
  as English text.
- **`BiasDetector`**'s categorical disparity test named the privileged group by
  alphabetical column order (with `admit`/`waitlist`, the 90%-admitted group was
  called disadvantaged). Direction is now withheld unless the favourable label is
  known; `positive_class_used` names it. Gap, p-value and Cramer's V are unchanged.
- **Bias detection** also stopped: reading an unrecorded outcome as a rejection, treating
  a text column as stable, accepting a benchmark that is not a share, reporting a
  gap outside [-1, 1], resolving ambiguous column names by alias length, and making a
  redlining finding for a place named `"None"`.
- **`ModelFairnessGate`** now refuses any release bound the metric's range makes
  unbreachable: a relaxed intersection threshold above 1.0, `improvement_margin`
  of -1e9 and `allow_degradation_margin` of 1e9 each approved a measured breach.
  `check_fairness_config` and the pre-commit hook use the same rule.
- **A gate's `min_group_size`** of 1 or less cannot flag any group. The verdict is kept,
  but `small_sample_check_ran` is now False with `small_sample_vacuous_minimum`
  set, on every surface.
- **`IntersectionalGateDecision`** now records `small_sample_unsized`: intersection cells
  whose size was never compared. Reports no longer print "every group was at or
  above the minimum" over cells of five people.
- **`BiasMonitor`** reported perfect equalized odds for a batch with some unlabelled
  rows; it now returns NaN with a warning naming the group.
- **`FairnessMonitor`** metrics refuse partly unlabelled groups (equal opportunity,
  equalized odds) and score columns (disparate impact, demographic parity), and
  `get_alert_summary` reports coverage per metric as well as per window.
- **`compute_regression_effect_sizes`** reported a prediction Cohen's d of -3.7e7 as
  "large"; values far outside any real effect now warn and are not labelled.

#### Added

- **`BoundRole`**, `metric_value_range` and `vacuous_bound_reason` (metric direction).
- **`prediction_fn_available`** (explainers base), `finite_or_nan` (feature
  engineering), `MetricsStore.window_now`, `WindowMetrics.uncompared_metrics`,
  `non_lexicon_script_share`.
- **`CalibrationDisparityResult.groups_without_measured_calibration`**,
  `IntegratedCalibrationResult.uncovered_groups` and
  `MulticalibrationResult.uncovered_groups`.
- **`adversarial_convergence_diagnostics(degenerate_constant_predictions=...)`**.

### Fixes found by the second examination of grades (2026-09-29)

#### Fixed

- **`cohens_d`** treated a computed, near-constant residual as spread and returned
  -6.9e15. Degeneracy is now judged relative to the data's magnitude, in all four
  copies (core, agents, LLM output analysis, bias detection);
  `compute_regression_effect_sizes` adds `interpretation_residuals`.
- **`permutation_test_equal_opportunity`** and `robust_fairness_comparison` silently
  dropped a NaN group label and understated the gap (0.50 reported as 0.05);
  `compute_robust_metrics` now returns such a level with NaN estimates.
- **`MetricsStore.compute_health_score`** scored 96.2 GREEN when the breaching records
  had no readable timestamp; it is now withheld, with
  `HealthScore.n_undatable_metrics`. `ingest_dataframe` checks timestamps at the door.
- **`MetricsStore.ingest_alert`** stored an undatable alert that vanished from every
  window. It is kept with `timestamp_measured False`; `get_alerts` names undatable
  records, and the health score withholds its alert component.
- **`MetricsStore.get_alerts`**, `get_metrics` and `get_drift_history` returned an empty
  result for an unreadable window bound; they now refuse. A row with no recorded group
  size reads `privacy_level 'unknown_size'`, not `'exact'`, under both privacy settings.
- **`ModelFairnessGate`** approved `require_improvement` with an absent baseline (four
  spellings of absence); it now blocks. `evaluate_hierarchical` fails closed when an
  attribute or the intersections were never evaluated, and now enforces
  `allow_degradation_margin` at intersection levels.
- **`IntersectionalGateDecision`** now carries `small_sample_check_ran` in the same three
  states as the flat decision, on every renderer.
- **`print_explanations`** dropped the whole statistical section; it is now printed, an
  empty section says why, and a missing severity no longer crashes the printout.
- **`explain_effect_size`** graded six of nine statistical measure types (p-value,
  standard error and others) as "info, no action needed"; ungraded types are now
  described without a grade.
- **`FairnessAnalyzer.compare_with_aequitas`** reported a one-group self-comparison as
  disparity 1.0; disparities are now NaN with fewer than two groups, aequitas' own
  table is kept under `aequitas_reported_disparities`, and no valid group refuses.
- **`compute_signed_test_log`** sealed an unparseable timestamp string (`"banana"`), a
  `NaT` and an epoch number as `declared_by_caller`; these now read
  `declared_but_unreadable` with a warning, and a `datetime` is read correctly.
- **`detect_temporal_drift`** graded critical drift over a single-valued time column, cut
  cohorts by row position, imputed unreadable timestamps into the middle, and scored
  PSI on an invented "missing" category. All fixed; new keys
  `nRowsWithUnreadableTime` and `attributesWithTooFewObservations`.
- **`cusum_drift`**, `page_hinkley` and `sequential_fairness_drift` reported "stable" for
  a `threshold` or `lambda_` no series could reach; they now return `None` with the
  reachable ceiling named.
- **`NonDeterminismAnalyzer.equivalence_test`** confirmed fairness on a 0.6 difference
  with `sesoi=inf`; inf, NaN, zero and negative SESOI now raise `ValueError`.
- **The LLM scorers** returned a measured 0.0 for a missing response (`None`) and
  raised on NaN; they now return NaN with reason `no_response`. An empty string
  keeps its measured 0.0.
- **The scorer readability gate** asked whether one ASCII token existed; it now requires
  half the letters to be readable, so a Japanese refusal beginning "AIとして" is no
  longer scored "did not refuse". The keyword sentiment, regard and framing scorers
  no longer score the wrong pole from one appended English word.
- **`ContextualStereotypeScorer.stage_coverage`** no longer reports the NLI stage as
  having read a blank text, and the sidecar-down warning now describes the NaN that
  scorers return.
- **DecodingTrust** (`run_toxicity`, `run_adversarial_robustness`,
  `run_adversarial_demonstrations`) uses the same share gate, and no longer reads an
  outcome verb about an object ("assign the reviews") as a rating of the applicant.
- **`OutputAnalyzer`** deleted missing responses before its attrition guard could count
  them; `n_missing_a` and `n_missing_b` now enter the comparison.
- **`clean_model_name`** accepted a name made only of invisible format characters, so
  `compare_model_identity` reported `same`; such a name is now absent.
- **`classify_column_roles`** ignored a misspelt `declared_prediction` and let a name
  heuristic override a present one; both fixed.
- **`scan_fairness_violations`** silently narrowed to one metric without `y_true`; it now
  publishes `metrics_not_attempted` and withholds the all-clear.
- **`compute_pearson_correlation_matrix`** published a coefficient beside a NaN p-value
  for a pair containing inf, and |r| = 1 for two-row overlaps at low
  `min_periods`. Both sides now read the same rows; overlaps below 3 are NaN.
- **`identify_proxy_variables`** reported `complete=True` over zero requested pairs; it now
  sets `not_screened_reason`, carried into `ProxyScreenFindings` and
  `FeatureAnalysisReport.proxy_screen_complete`.
- **`analyze_intersectional_correlations`** reported coverage `complete` when no requested
  feature was numeric; it now reports `not_assessed` with `features_not_numeric`.
- **Representation analysis** overwrote one spelling of a group with another (`'m'` and
  `'M'`) and divided by subnormal benchmarks. Spellings are summed and disclosed in
  `labels_folded_together`; an unusable benchmark is refused.
- **The target-leakage check** in the specification-bias scan correlated over imputed
  values, so a perfect leak passed; it now correlates over rows holding both values.
- **`impossibility_diagnostics`** flipped to "THEOREM APPLIES" when an unlabelled group
  was added; degeneracy is measured on the compared groups, with `None` when unknown.
- **`mitigation_pareto`** counted unlabelled rows as negatives (accuracy 0.865 instead
  of 0.495); they are dropped and counted as `nRowsUnlabelled`.
- **`recommend_calibration_strategy`** now names, in its rationale, a group whose ECE
  could not be measured.
- **`LimeExplainer`** reported near-zero attributions for barely perturbed features as
  measurements; it now publishes `barely_perturbable_features` and
  `background_relative_spread` and warns.
- **`removal_curve_auc`** returned different scores for the same explanation depending
  on column order when attributions tied; ties are now resolved or refused.
- **Reweighting (`target_parity`)** blamed missing labels on the protected attribute and
  counted NaN as a label class; it now reports `n_rows_without_a_recorded_label`.
- **The resampler** interpolated synthetic rows into cells with no recorded label (130
  rows became 150 or 720); unlabelled rows are now returned untouched and counted.
- **`FeatureSuppressor(strategy='noise')`** claimed to suppress a column whose noise was
  below float resolution, and a stale `could_not_check` carried into the next run.
- **`compute_adverse_action_reasons`** ignored `np.float32` SHAP values and could rank a
  proxy in the omitted driver's slot; numpy scalars are now read.
- **`build_assurance_verdict`**: one row with no computed gap erased every
  disparate-impact finding in the run; such rows are now named as unassessed.
- **`generate_iso42001_evidence_map`** counted nested placeholders as content (92.3% for
  an empty wizard); values are now checked to their leaves.
- **`run_pulse`** published "No annual audit statute matched this run" when the LL144
  screen had not run; the recheck basis now says it could not be determined.
- **`create_technical_view`** painted |r| = 1 correlations from two observations; the
  panel is withheld below the minimum sample size.
- **`fetch_holc_data`** returned error bodies and `None` as HOLC data; the shape is now
  checked, and the city lookup is case-insensitive.
- **`plot_confidence_intervals`**, `plot_effect_sizes` and `plot_group_comparison` crashed
  or drew a neutral value for unmeasured inputs; the dashboard twins drew a missing
  metric as 0.0% and a missing Cohen's d as `d = 0.00`; a NaN was shaded inside the
  "Acceptable range". All now mark the value as not measured, once per bar.
- **`log_fairness_to_mlflow`** and `log_fairness_to_wandb` now record
  `group_stats_status 'suppressed_by_caller'` and name unreadable entries in
  `unreadable_entries` instead of reporting nothing left out.
- **`bias_audit_to_svg`** drew measured 0.00 tiles for a report with no assessment record.
- **The threshold explanation** treated a region with no recorded status as "retrain the
  model"; such regions are now reported as not covered.
- **`GroupthinkDetector`**'s coalition refusal now gives a reason that is true for
  negative thresholds.
- **The causal task** handlers turned a `bytearray` or `memoryview` dataset into an empty
  frame of repr fragments; all three buffer types are decoded, and a zero-row frame
  is refused.
- **`WorkerLoop.run`** reported a clean stop when every job crashed; it now returns
  `EXIT_WORK_ALL_FAILED` (4).

#### Added

- **`cohens_d(constant_atol=...)`**, `lexicon_readable_share`, `lexicon_can_read`,
  `response_not_recorded`.

### `mypy src` is clean again (2026-09-28)

The Quality workflow had failed on every push to `main` since 2026-09-27 on 23 mypy
errors in 9 files. None was a hidden runtime defect: each None or type was already
excluded a few lines earlier, in a form mypy could not follow, and each guard is now
stated explicitly, with no casts and no ignores. Two looked like bugs and were not:
`statistical.py` reused the name `why` (a string, then a list) and now uses
`skip_reasons`; the LLM judge's weighted score now reads only ratings that are
present, which the NaN refusal above it already guaranteed.
`_unmeasurable_baseline_message` now accepts `Optional[float]`, which both gate call
sites pass. No behaviour changes.

### More from the second-round audit (2026-09-28)

#### Fixed

- **`ThresholdAnalyzer`** cast NaN labels to platform-dependent integers and reported a
  maximal equalized-odds violation for unlabelled groups; it now raises
  `InvalidDataError` at construction.
- **The Pulse payload's** `excludedZeroPower` lists the zero-power metrics again after
  the zero-power state moved upstream.
- **`GroupManager.get_invalid_groups`** now discloses that a minimum of 1 or less cannot
  exclude anything; the agent audit summary keeps a not-interpretable gap visible
  ("NOT INTERPRETABLE") instead of claiming none could be computed.
- **The temporal card's summary chip** no longer prints STABLE in green when no trend was
  fitted, and the threshold explanation no longer counts requested constraints as
  satisfiable ones.
- **193 further findings closed in one sweep.** User-visible examples:
  sklearn's `DummyClassifier` is treated as degenerate, an object column of NaN
  labels no longer passes a constraint, LLM judge marks out of 100 are no longer
  clipped to 1.0, builtin max/min no longer drop NaN groups, a jittered frozen
  background is refused, `check_fairness_config` rejects a 1e9 bound, and a
  DecodingTrust "no" in "There is no evidence" is no longer a denial.

#### Added

- **`AdverseActionReasons(features_examined=..., adverse_attributions_not_ranked=...)`**,
  `reads_as_another_latin_script_language`.

### The fix wave, audited a second time (2026-09-28)

The Beta Go-Live fix wave was audited again, by agents attacking the fixes rather than
the code. That audit recorded **59 claims**, each a test that asserts the value a unit
produces today with the measurement in its docstring. **24 are closed**, 35 remain
open and reproducible. The live count is generated into
[docs/QUALITY_AND_HARDENING.md](docs/QUALITY_AND_HARDENING.md) and onto the quality
page from `docs/bgl6-audit-register.json`, so it cannot drift from the audit files.

Every fix below was reproduced before it was made, carries its measured before and
after in the source, and was sabotaged in both directions: the guard shown red, and
the healthy case shown to survive.

#### Fixed

- **The release gate approved a deployment on an unusable margin.**
  `improvement < margin` has two operands and only the baseline was guarded. With
  `improvement_margin=float('nan')` and a 0.89 degradation in the metric's own
  declared direction: `approved=True`, "APPROVED - All fairness requirements met",
  `warnings=[]`, GitHub conclusion `success`. A margin guard now sits above both
  comparisons in BOTH entry points, and an infinite margin is refused for the same
  reason a NaN one is.

- **Equal opportunity and equalized odds reported perfect parity for a maximally
  discriminating model.** `_selected_share` is `(values == 1).mean()`, and
  `prediction_col` is documented as "binary 0/1 or scores", so a SCORE column
  contributed no selected row and both true-positive rates were 0.0. On 80 rows per
  group where group A's positive-label rows all score 0.9 and group B's all score
  0.1: `compute_equal_opportunity` 0.0 and `compute_equalized_odds` 0.0 with no
  warnings, while `compute_disparate_impact` read 0.111 and
  `compute_demographic_parity` 0.800 on the same frame. A share of SELECTED rows is
  undefined for a score column without a decision threshold, so it refuses.

- **Rows with no protected attribute were dropped in silence.** `groupby` omits
  them and the undersampled disclosure grouped the same way, so no predicate looked
  at the group column. On 150 rows where 50 carry no group value:
  `compute_disparate_impact` 1.0 and `compute_demographic_parity` 0.0, no warnings. A
  perfect four-fifths ratio over a population nobody chose.

- **A mostly unmonitored history reported as quiet.** `get_alert_summary` warned only
  when NOT ONE window carried a comparison. One clean window followed by nine whose
  predictions are all NaN returned `{}` with no warning, identical to a ten-window
  fully monitored clean history. Coverage is stated now whenever any window is
  uncompared.

- **The report card could not tell "never checked" from "checked and clean".**
  `small_sample_check_ran` is emitted by `to_dict` and rendered by
  `to_markdown_report`, and `FairnessReportCard` printed neither, so the card for a
  decision whose check never ran was byte-identical to one on data whose groups were
  all large enough. The card answers in three states now, in the same words as the
  report.

- **The signed test log left its reproducibility block outside the seal.** Replacing
  `reproducibility_info['data_hash']` with a different 64-character hash, or
  `library_version` with `9.9.9`, or the note with "Any result may be substituted
  freely", all left `content_hash` verifying, in a document the docstring calls a
  tamper-evident record. The block is inside the sealed payload now. **This changes
  the hash for a given run**: a log issued by an earlier version verifies against the
  rule that version used, and the two test helpers that carried their own copy of the
  recipe are updated. Proven by its negative cases: seven tampering attempts, seven
  refusals, and the honest document still verifies.

- **A Spanish or Dutch refusal scored the clean end of the scale.** The language gate
  needs no English function word anywhere AND one token of another Latin-script
  language, and the English half was an ANY over the whole text, so one homograph
  vetoed it: "No puedo ayudarte..." ('no') and "Het spijt me, ik kan je hier niet mee
  helpen." ('me') both scored a measured 0.0. At the public entry point, 25 Spanish
  refusals against 25 Spanish helps reported state `measured`, observed 0.0,
  `exceeds_noise` False. The gate compares the evidence on each side now and needs a
  strict majority; a tie stays measured so English text is never refused.

- **A model that decided nothing was certified perfectly fair.** "There is no 2024
  tax return in the file to support a determination." was read as a loan DENIAL,
  because the determiner rule required a letter after the space and a digit is a
  following word too. With that one reply on every prompt, `run_fairness(24)`
  published `overall_score` 0.0, "perfectly consistent across groups", every group
  rate 0.0, zero warnings, and `run_machine_ethics(12)` published "perfect ethical
  consistency". A sentence-final rating ("the quality of the documentation is low")
  was also read as a rating OF THE APPLICANT; a rating after a linking verb is now
  read only beside the rating noun, so "their recidivism risk is low" keeps its
  reading.

- **The adversarial probe cleared a model it had no power over.** Both twin guards
  refused only when NOT ONE background column could be perturbed. With one frozen
  column, `sigma = std + 1e-9` is the division guard rather than spread, so the gap
  along it is exactly 0.0 for any model: a deliberately scaffolded model came back
  `flag=False, confidence=1.0`, byte-identical to a clean one, and
  `diagnose_local_attribution` published that. Both twins refuse now, word for word.

- **A mitigation that did nothing reported success.** `LabelMassager` on a live
  0.5-against-0.6 rate gap computes an equalizing count of 0.5, floors it to zero,
  and returned the labels unchanged with `n_labels_flipped_ == 0` and no warning,
  indistinguishable from an already-equal frame. And with 20 of 40 rows carrying no
  protected value it flipped 6 labels without saying half the frame was ineligible.
  Both are disclosed now, via `unequalized_rate_gap_` and
  `n_call_rows_without_a_recorded_group_`.

- **A length is not a mean.** `ResidualTransformer` decided a group had a conditional
  mean with `len(group_data) > 0`, and twenty NaNs have length twenty. The group was
  recorded with a mean of `nan`, stayed in `_fitted_groups` (disabling transform's own
  refusal), counted toward the "fewer than 2 groups" disclosure (disabling that too),
  and the column was reported as residualized. It is excluded and named now, via
  `groups_without_a_conditional_mean_`, and transform no longer says such a group
  "was not seen at fit", which was false.

- **The tomek branch counted the absence of a record as a protected group.**
  `get_resampled_data` dispatches to `_tomek` before it computes `recorded`, so on 130
  rows with 20 missing gender values: `group_totals_before {'None': 20, ...}` and rows
  deleted and accounted to that pseudo-group, with no warning. Excluded from the
  totals, counted under its own key, and disclosed; the rows themselves are still
  returned.

- **An empty denominator reported as 0% ISO 42001 coverage.**
  `generate_iso42001_evidence_map` computed `... if assessable else 0.0`, so a map in
  which nothing is assessable published 0.0, the strongest adverse compliance
  statement it can make, from an assessment that never ran. It is `None` now with a
  basis that says NOT ASSESSED and that this is not 0% coverage. An empty wizard still
  earns a real 0.0 over the 13 controls the map CAN assess.

- **A blank generation scored as measured neutral regard.**
  `TransformerRegardScorer.score` returned 0.0 for a blank, and 0 on that scale is the
  class's own "neutral regard", so five empty generations came back as a measurement
  on both sides. Its keyword twin documents the refusal and honours it, so which
  answer the metric gave depended on whether `transformers` happened to be installed.

- **A zero-power row reported as a test that came back clean.** `analyze_all` marked a
  row whose scorer read one constant value on both sides only when a correction was
  requested, so an uncorrected run published p=1.0 and `is_significant=False` for a
  row that compared nothing. Whether the scorer read a constant is a fact about the
  data, so it is stated in every run, and `_is_zero_power` now recognises the declared
  state rather than a p-value of exactly 1.0 that no longer exists.

- **An evidence defect with correct code.** A grading record cited a sabotage that
  cannot reproduce: reverting `isinstance(value, (bool, np.bool_))` to
  `isinstance(value, bool)` and "2 failed". Re-run by reverting that exact clause:
  64 passed, nothing red, because `is_measured` refuses `np.bool_` twice over. The
  behaviour is right and stays; the RECORD is corrected and the clause now says it
  must never be cited as evidence again.

#### Added

- `docs/bgl6-audit-register.json`, generated by `scripts/bgl6_register.py` from the
  audit files, with the generated count on the quality page and in
  QUALITY_AND_HARDENING.md. Each audit file declares which of its tests still record
  an open defect, and `tests/test_bgl6_register_is_honest.py` refuses a name that is
  not a test in that file, so the published number cannot flatter itself.

### How much of your data the answer rests on (2026-09-27)

#### Added

- `data_info` on every `FairnessAnalyzer` report now carries `coverage`,
  `groups_dropped`, `n_groups_before` and `n_groups_after`, and `validate_inputs`
  warns when `missing_strategy='exclude'` removes rows. Purely additive:
  `invalid_groups`, `valid_groups`, `group_sizes` and `n_excluded` are unchanged.

  `n_excluded` said how many rows went. It did not say what share of the data every
  number in the report was computed from, nor whether a whole group went with them.
  Measured on 120 rows across three groups, with the never-selected group's attribute
  missing on every row: `demographic_parity_difference` moved from 0.475 to 0.050 and
  `demographic_parity_ratio` from 0.000 to 0.895, for the SAME predictions. A total
  exclusion read as a four-fifths pass, and the only trace was an integer in a dict.


### Every core capability examined (2026-09-27)

#### Fixed

- **`FairExplAIner.explain_metric`** graded inf as critical (or "near-perfect parity" on
  a ratio); any non-finite value is now could-not-check.
- **`ModelFairnessGate`** substituted `abs(baseline) - abs(value)` when a metric's
  direction was unknown, reversing the sign for larger-is-better metrics; such an
  improvement is now NaN.
- **`SurfaceStatus.share`** raised for a declared state with no members; it now returns
  0.0, and an undeclared state still raises.
- **`MetricsStore.compute_health_score`** included a hardcoded 100.0 drift stability for
  an empty drift table. `drift_stability` is now absent when unmeasured and the
  composite is renormalised; a window under 24 hours no longer forces a "stable"
  trend, and a first window's trend is `unknown`.
- **`analyze_intersectional_correlations`** reported a coefficient of variation of 0 for
  negative group means; `compute_pearson_correlation_matrix` returned p = 1.0 for
  untested pairs (now NaN); `check_geographic_redlining_risk` scored 0.0 for an
  all-null outcome; `compare_to_benchmark` graded zero observations;
  `prepare_protected_attributes` accepted an all-None column.
- **`run_fairness`** read decisions by substring ("I cannot make that determination" was
  a denial); it now uses whole words and negation. `run_machine_ethics` and
  `run_adversarial_demonstrations` no longer score unreadable answers.
- **`removal_curve_auc`** scored an all-NaN attribution vector as "strong";
  `diagnose_local_attribution` read an unprobed model as cleared;
  `get_group_rankings` dropped a shut-out group; `NonDeterminismAnalyzer` rejects
  `min_runs=0`.
- **Adversarial training**: `update_adversary` no longer counts a NaN network as trained,
  `get_adversary_accuracy` no longer returns 0.0 for unreadable labels, and
  `FairRepresentationLoss` no longer reports a task loss of 0.0.
- **`streaming_demographic_parity`** now warns when it excludes a group, and
  `stream_group_counts` no longer returns `{}` for all-NaN predictions.
- **`require_checked()`** with no names returned as if everything passed; it now raises.
  `is_measured(Decimal("0.5"))` is now True.
- **`EnsembleClassifier`** weights are normalised (two members predicting 1 no longer
  vote 0), a negative weight is refused, and `ExponentiatedGradient` no longer
  declares a reject-everyone model satisfied.
- **`BoundedGroupLossConstraint.compute_violation`** turned a 0.36 breach into compliance
  when labels were removed; `TrainableGroupCalibrator.calibration_loss`,
  `CalibrationAwareTrainer` and `fine_tune_calibration` reported 0.0 loss for NaN
  logits. All now refuse.
- **`FairClassifier.predict`** returned 0 for unscored rows; it now refuses like
  `predict_with_sensitive_attr`. `FairRegressor.fit` no longer drops sample weights
  silently, and `CounterfactualFairnessLoss` refuses an identical counterfactual.
- **All six assessability canvases** now show groups excluded for size;
  `TextFairnessAnalyzer.analyze` no longer drops a term with no text;
  `run_disparity_tests` names excluded groups.
- **`LLMJudgeScorer`** no longer accepts a boolean as a score; `IntersectionalGroup.
  from_attributes({})` no longer returns one empty group; `IntersectionalAnalyzer`
  reports unresolvable metrics; a PSI of 23 is no longer graded `pass`; NaN eta
  squared is no longer `large`.
- **`ModelFairnessGate.create_github_check`** returned `success` for a CONDITIONAL
  decision; it now returns `neutral`.
- **`build_assurance_verdict`** gave an Unqualified opinion over an empty assessed set,
  and `compute_adverse_action_reasons` returned a complete notice over zero
  features. Both fixed.
- **`cusum_drift`** reported no drift when sigma was zero and the mean sat far from an
  explicit target; `get_rolling_average` averaged booleans into a parity figure.
- **`selection_rate_disparity_matrix`** published `min_ratio` 1.0 when nobody was
  selected; regression `get_group_metrics` gave a constant-target group r2 0.0;
  `global_importance` returned 0.0 for every feature from one row.
- **`CounterfactualTester.run_test`** raises when the template has no placeholder, so no
  prompt varied; `sign_flip_paired_test` no longer hardcodes `detectable`.
- **`LimeExplainer`** and `AnchorsExplainer` refuse a background that cannot perturb;
  `XaiDiagnostics.adversarial_flag` defaults to `None`, not False.
- **The Pulse agent probe** published crashed or refusing detectors as clean verdicts; it
  now carries three states and `detectorsFailed`.
- **`get_impossibility_diagnosis`** ignored `min_group_size`; NaN thresholds no longer
  form a Pareto frontier; a non-contiguous feasible set is no longer printed as one
  interval; `calibration_vs_error_parity`, `mitigation_pareto` and
  `recommend_calibration_strategy` refuse single-group or NaN inputs.
- **`create_paired_artifacts`** refuses values that are all identical;
  `PipelineTracker` discloses stages never recorded.
- **Dashboards and reports**: the health gauge annotates unmeasured drift; an empty
  store no longer renders green; unscored alerts are not counted as LOW; missing
  trend groups and marginal "intersectional" heatmaps are disclosed;
  `generate_threshold_breach_report` and `generate_alert_report` handle NaN and
  missing values; a failed chart is reported, not blank.
- **MCP tools** return infinity as the string `"Infinity"` instead of `null`, so a total
  exclusion is not confused with a missing value.
- **`compare_methods`** and `generate_recommendation` no longer recommend from zero
  successful comparisons; `baseline_comparison_summary`, `FairnessTrainingReport.
  summary`, `evaluate_baseline`, `adversarial_convergence_diagnostics` and
  `sklearn_adversarial_debiasing` no longer report absences as results.
- **`assert_fairness(metrics=[])`** and `create_fairness_callback(fail_on_violation=True)`
  without thresholds now raise; `fairness_test` and `parametrize_fairness` fail when
  no case runs.
- **`explain_metric`** no longer grades a boolean; `explain_report` names an unreadable
  metric; the MLflow and W&B loggers write `metrics_not_measured` and
  `group_stats_not_measured`.
- **`CalibratedEqualizer`** and `RejectionOptionClassifier` no longer report a collapse
  onto one score, or a NaN threshold, as an improvement.
- **`analyze_delegation`**, `compute_selection_disparity` and `contingency_test` return
  `is_significant None` when the sample cannot reach 0.05, with
  `min_attainable_p_value` and `detectable_at_05`.
- **`TemporalFairnessAnalyzer.detect_trend`** returned "decreasing" for a NaN slope;
  `ece_confidence_intervals(n_bootstrap=0)`, `cv_calibration_stability`,
  `BaseReweighter.predict` and `GroupthinkDetector.detect_coalitions` no longer
  report absences as measurements.
- **`ReweightingTransformer`** disclaimed real intersectional weights; missing protected
  values are no longer a group; `FeatureSuppressor` no longer reuses a previous fit;
  `CorrelationReducer` no longer maps an unseen category to the reference.
- **`LLMApiProxy.send_prompt`** returned an error body as the generation; error and
  unparseable responses now carry a reason with `token_count=-1`.
  `OutputAnalyzer._compare` no longer returns p = 1.0 when the test raises.
- **`IntegratedGradientsExplainer`** and `TreeShapExplainer` raise when
  `feature_names` does not match the attribution count; `local_r_squared` returns NaN
  for unmeasurable fidelity; `slack_adversarial_probe` refuses a NaN model.
- **`TreeShapExplainer.supports`** and `LinearShapExplainer.supports` check the model,
  not only the caller's label; `KernelShapExplainer.supports` accepts sklearn models.
- **`CalibrationRecommendation.to_dict`** now emits `not_assessed_groups`;
  `plot_group_calibration` and `group_calibration_to_svg` disclose small groups.

#### Added

- **`ParetoPoint.is_rankable`**; `ContextualStereotypeScorer.stage_coverage`;
  `build_quality_report(unavailable_attributes=...)`;
  `streaming_selection_rates(min_group_size=...)`;
  `vfairness.status(name).preview`.

### Verification status, readable from code and from every page (2026-09-25)

#### Added

- `vfairness.status()` and `vfairness.require_checked()`, with `CapabilityStatus`,
  `SurfaceStatus` and `UnknownCapabilityError`. The three-state verification status of
  every public code unit now ships INSIDE the wheel and is readable at runtime, so
  a caller can fail their own pipeline when it depends on something this library has
  not verified, and can ask about the version they installed rather than about
  whatever the website says today. An unknown name raises rather than answering
  NOT CHECKED: a typo must not read as a finding about the library.
- A published three-state label on every one of the public code units, in the API
  reference, in `docs/API_REFERENCE.md`, in a full grid at `/status/`, and as a
  machine-readable feed. `scripts/capability_status.py --check` and
  `scripts/stamp_status_badges.py --check` fail the build when any surface
  disagrees with the ledger.
- `scripts/suite_coverage.py`, which measures which public code units the test
  suite executes. The readiness gate credited the capability census, the grading
  waves and the surface probe, and never the suite: measured, the suite executes
  1,346 of the units while the gate reported 920 examined in total. The record
  fails closed on the suite's own result, because lines executed on the way to a
  failing assertion verify nothing.

#### Fixed

- `sequential_fairness_test` computed a ONE-SIDED log-likelihood ratio for a
  two-sided question, so a real disparity in the unfavourable direction crossed the
  lower Wald boundary and returned `accept_h0_no_difference`. Measured at 100 per
  arm with group A at 50% and group B at 0%: the forward call accepted the null and
  the same data with the arms swapped rejected it. Now the two-sided Wald mixture,
  symmetric in the arms.
- `FairnessAnalyzer.compare_with_fairlearn` reported fairlearn's conventional `0.0`
  as a parity comparison on a single-group frame, where the same object's
  `compute_all_metrics` returns `nan` and warns.
- The sidecar scorers returned `0.0` for every text while the ML sidecar was
  unreachable, which on toxicity means NOT TOXIC, warning once per process. They
  now return `nan`; the return shape is unchanged.
- `fisher_exact_test` returned `1.0` for every degenerate-margin table, which reads
  as "no association found" where no odds ratio is estimable. Now `nan` with a
  warning, for all four shapes at once.
- `BiasAuditReport.to_svg` rendered "OVERALL RISK 0% MINIMAL" over an audit whose
  own record said every module ran and none could assess anything, from three
  separate causes plus a hardcoded template sentence that was false on that canvas.
- `FeatureEngineeringAnalyzer.get_explanation` graded an analysis 'info' whose
  screens had all been skipped for want of samples.
- `minimum_detectable_effect` returned a finite `1.0` under a comment reading
  "Cannot detect anything", so every downstream `isfinite` guard passed it;
  `power_warning` cleared a subgroup larger than its own dataset;
  `get_invalid_groups` answered "no group is too small" for zero groups and for a
  threshold that disables the check; `interpret_effect_size` claimed a direction at
  exactly zero; `FairnessMonitor.get_alert_summary`,
  `TemporalFairnessAnalyzer.get_metric_summary` and `MetricsStore.get_alerts`
  returned an empty result in silence where nothing had been monitored at all.

### Status and the last observed fabrications (2026-09-25)

#### Fixed

- **`FramingScorer`** returned the 0.5 midpoint for empty text while refusing real text
  without markers; empty text is now NaN.
- **`generate_iso42001_evidence_map`** reported 10% coverage for an empty wizard; the two
  controls it cannot determine are `manual` and excluded from the fraction.
- **`assign_clusters`** could assign every cluster to treatment; the correlation ratio
  returned 0.0 and p 1.0 for a 0/0 quantity; `SurfaceStatus.share` was case
  sensitive; `IntersectionalGateDecision.to_dict` omitted a field.
- **`plot_group_disparity_heatmap`** drew a 0.000 "pairwise" cell for a single group; it
  now refuses.

### Portable full-suite controls (2026-09-21)

- Update the full-suite checkout and Python setup actions to SHA-pinned Node 24
  releases, and pin its existing Ubuntu 24.04 runner image. Remove the deprecated
  action-runtime and pending OS-rollout warnings without suppressing diagnostics
  or changing the Python matrix, dependency installation, tests or coverage gate.
- Derive the two-group numerical controls from independent mathematical
  identities and native tensor primitives instead of another backend's rounded
  scalar values. Seeded float32 fixtures and the existing relative tolerance are
  unchanged; hand-calculated anchors and deliberate zero/half/double mutations
  verify that the controls still reject incorrect penalties.
- Isolate the missing-MCP import test from installed packages and preloaded
  modules. Keep the optional MCP backend installed in the full suite, preserve
  incompatible-version diagnostics, and exercise the ambient-package case with
  a successful import control. Production calculations and import guards are
  unchanged.

### Export evidence repair (2026-09-21)

- Keep the documented grading methodology, release plan, probe evidence and
  reproduction scripts in the public export admission manifest. Their inclusion
  does not assert that the library is ready to release.
- Redact local checkout paths in the committed probe's diagnostic text without
  changing observations, outcomes, warnings or grades. The probe now removes its
  package and home prefixes before truncating future exception details.
- Describe operational workflows by purpose in the quality page without exposing
  deployment host names or private checkout paths. Export denylist, secret-scan
  and positive-control checks remain enabled and unchanged.

**Released 2026-10-01.** Tagged `v0.1.0` on `validantai/vfairness` and published
to PyPI from that tag by the Trusted Publisher pipeline. The date on the heading
above is the publication date, filled in by `scripts/cut_release.py` at the cut
and not before: earlier revisions of this file asserted a release event twice
over that had not happened, and both dates were fiction.

**Everything continues to accumulate under this heading** until then. There is
deliberately no separate `[Unreleased]` section: while 0.1.0 is itself
unreleased, a second holding pen would only invite the same mistake again, of
work sitting under one heading while shipping inside another. Every change made
between now and the tag belongs here, because this is where a reader of the
published 0.1.0 will look for it, and PyPI is immutable: a later version's
changelog cannot retroactively say what 0.1.0 contained.

**What 0.1.0 will be, when it ships.** A plain SemVer 0.x first public beta
carrying the `Development Status :: 4 - Beta` trove classifier, installable with
a normal `pip install vfairness` and no `--pre` flag. The API is not frozen until
`1.0.0`, though the public surface in `docs/API_STABILITY.md` is already treated
as stable under the deprecation policy. Pushing the tag is what starts the
PyPI publish pipeline described in `docs/RELEASE_PIPELINE.md`.

### Grading wave 2 (2026-09-18)

#### Fixed

- **`AlertPayload.to_dict`**, `CalibrationReport.to_dict` and `.to_svg`,
  `FeatureAnalysisReport.to_dict` and `build_regulatory_exports` flattened a
  could-not-check into a number; they now keep the third state.
- **Adversarial debiasing** guarded its inputs but not the network's output, so batches
  after a NaN update counted as measured; the output is now checked.

#### Added

- **`UnmeasurableGroupWarning`**, `BiasAuditReport.empty_is_not_a_measurement`,
  `ProxyScreenFindings`, `GroupCalibrator.evaluate(metric_min_group_size=...)`,
  `TrainableGroupCalibrator.calibration_loss(min_group_size=...)`,
  `base_rates_differ(min_group_labelled=...)`.

### Beta Go-Live Stages 2 and 3, and grading wave 1 (2026-09-17)

#### Fixed

- **Stages 2 and 3** closed the remaining census findings across 99 capabilities,
  including the calibration methods, threshold optimisers, fairness losses and
  regularisers, reweighting, data balancing, feature transformers, drift detection,
  counterfactual and LLM analyzers, and the explainers.
- **`identify_proxy_variables`** returned a clean `[]` when the protected column was
  absent, unreadable, misspelt or filtered out; each case is now named.
  `auto_discovery`, `detect_historical_patterns` and `discover_intersectional_groups`
  list what they did not assess; `causal_fairness_graph` returns `None` with
  `assessable=False` for an unreadable graph.
- **`compute_feature_correlations`** (and `identify_proxy_features`,
  `association_strength`, `find_proxy_chains`) published Cramer's V 0.0 for an
  undefined table, so a column that determines gender was reported as no proxy.
- **`permutation_test`** only refused a powerless metric below 1000 rows; the design floor
  now applies at every size via `n_design_rows`.
- **`IntegratedGradientsExplainer`** let a cancelling explanation pass completeness; the
  tolerance no longer scales with attribution mass.
- **`RepresentationScorer`** scored non-Latin text 0.0 (no diversity), manufacturing a
  significant erasure finding; unreadable text is now NaN.
- **`KeywordSentimentScorer`** scored blank text as neutral 0.0, which beat a denigrated
  group; blank text is now NaN.
- **`EqualizedOddsConstraint.signed_constraint_value`** returned 0.0 for a group not in
  the data; it now raises `KeyError` naming the groups present.
- **`simulate_threshold_change`** reported "no change" for a metric absent from an empty
  store; it now refuses.
- **`FairnessAlertPrioritizer`** graded an alert CRITICAL when `drift_score` was NaN; such
  alerts are now UNSCORED. `create_alert(drift_score=...)` now reaches the priority.
- **`MetricsStore.compute_health_score`** reports measured alert frequency even when the
  composite is withheld.
- **The proxy detector** read an unmeasurable association as 0.0 and dropped the pair;
  it now refuses by name. `max_group_disparity` no longer returns 0.0 when one group
  is NaN.
- **Two f-strings that only parsed on Python 3.12** no longer break import on 3.11.

#### Added

- **`FeasibleThresholds`**, `ProxyScreenResult`, `ProxyChainResult`, `NDKLValue`,
  `PartialCoverageWarning`, `IntersectionalProvenanceWarning`,
  `validate_borrowing_strategy`, `IntersectionalCalibrator.get_group_provenance`,
  `MultivariateProxyResult.was_assessed`, `CausalFairnessGraph.not_assessable`,
  `SemanticQualityScorer.dimension_coverage`, `completeness_scale`.
- **New parameters**: `permutation_test(n_design_rows=...)`,
  `conditional_demographic_disparity_with_ci(method=...)`,
  `IntegratedGradientsExplainer.explain_local(completeness_tol=...)`,
  `HistogramBinning(min_bin_count=...)`, `FairRegressor(on_unseen_group=...)`,
  `FairRegressor.score(sensitive_attr=...)`, and `min_group_size` on
  `CalibratedEqualizer`, `DistributionMatcher` and `DisparateImpactRemover`.

### Fixed (Beta Go-Live Stage 1: the 27 critical fabrications, 2026-09-11)

- **Every critical finding from the census is closed**: 27 findings across 25
  capabilities, each re-checked by an independent agent told to refute the fix.
  All 27 came back fix-real, pin-able-to-fail, no over-correction. `BGL-D` drops
  from 124 capabilities to 99 and `BGL-A` rises from 67 to 89. Nothing that
  reaches a graded or rendered surface is still known to report a number nobody
  measured. Three of the worst, all reproduced at the public API before and after:

  - `exposure_parity_ratio` returned `0.5410755479122771` for an all-NaN score
    column, bit-for-bit identical to a genuinely ordered ranking, because the
    positions fell back to the caller's row order. That value is below the
    four-fifths threshold, so the library graded it FAIL and rendered a red badge;
    interleaving the same rows returned `0.8187`, which passes.
  - `ExponentiatedGradient.fit` on a single-valued sensitive attribute returned
    `final_violation=0.0`, `constraint_satisfied=True`, `converged=True` with no
    warning: a compliance certificate for a run in which no pair of groups existed.
  - `net_benefit_parity` returned `0.0` for a fully unscored cohort, identical to
    a real, equal, useful model, and `check_threshold` graded that **PASS**.

- **The audit pass found three further defects**, each now fixed and pinned:
  `data_bias_validation`'s fix had swallowed a CRITICAL `zero_outcome_rate`
  finding (removing a fabricated disparity took a real finding with it);
  `conditional_demographic_disparity` still scored a stratum that had lost a valid
  group's cell, diluting a measured 0.5000 to 0.2804; and the DecodingTrust
  stereotype lexicon could not read a plain refusal, while `"No, that is not
  correct."` scored **1.0, full agreement with the stereotype**, because both
  lexicons match as substrings and nothing handled negation.

- **`mypy src` is clean again.** The refusals introduced `float | None` where
  callers expected `float`. Fixed at the source rather than silenced: `_is_finite`
  is now a `TypeGuard[float]`, the five ranking guards narrow on `position_array`
  as well as on the reason (closing a real hole if the helper ever returns both
  as `None`), and `LimeExplainer._run_once`'s signature now admits the `Optional`
  it was already returning.

### Added (Beta Go-Live proof ledger, 2026-09-11)

- **Every one of the 215 registered capabilities now carries a proof batch**
  saying how far its honesty has been established: whether it refuses to invent a
  number when nothing is measurable. The grade is in the capability's own
  docstring (so `help()` shows it), in `vfairness-manifest.json` under
  `proof_status` and `stats.proof`, and in `vfairness._proof_status`.
  Batch definitions, the method and the staged plan:
  [docs/BETA_GO_LIVE_PLAN.md](docs/BETA_GO_LIVE_PLAN.md).

  | Batch | | Count |
  | --- | --- | ---: |
  | BGL-A | PROVEN, and a test holds it | 67 |
  | BGL-B | SEMI-PROVEN, nothing holds it | 18 |
  | BGL-C | UNPROVEN, honesty unknown | 6 |
  | BGL-D | DEFECT OPEN, proved to fabricate | 124 |

  Four batches rather than three, deliberately: three would have put "checked and
  fine" and "never checked" in one bucket, which is the exact defect being
  audited. `BGL-C` is a positive statement that nothing is known.

- **`vfairness._proof_status.strip_proof_block(doc)`**, for anything that parses a
  docstring. Needed immediately: `tests/test_docs_truth.py` detects a paper
  citation with `\b(19|20)\d\d\b`, and the stamp's own date, `(2026-09-11)`,
  matched it, so every stamped metric read as citing a paper. Stripping generated
  text from a prose scan was the fix; weakening the citation rule would have let a
  real citation through.

- **The evidence is committed, not summarised**:
  [docs/beta-go-live-census-2026-09-11.json](docs/beta-go-live-census-2026-09-11.json)
  records what was executed against every capability.
  `tests/test_beta_go_live_proof_ledger.py` (25 tests) refuses any ledger row
  that disagrees with it, any BGL-A without a pin file that exists, any BGL-C
  carrying evidence, and any docstring whose batch disagrees with the ledger.
  Regenerate with `scripts/beta_go_live_ledger.py`,
  `scripts/stamp_proof_status.py` and `scripts/beta_go_live_docs.py`; all three
  have a `--check` mode and all three are enforced by the suite.

### Fixed (a release-pipeline truth test that skipped in CI, 2026-09-11)

- **`tests/test_site_pipeline_truth.py` skipped entirely in CI**, all 70 checks,
  because it calls `importorskip("yaml")` at module level and PyYAML lives in the
  `cicd` extra, which the private `vfairness-tests.yml` did not install. The
  public-facing `.github/workflows/tests.yml` already did, so the two workflows
  disagreed about what gets tested and the file people read was not the file that
  ran. `cicd` is now installed in both.
- Three of those checks were **red at `8559dda`**: `release.yml` declares the job
  `suite-green-on-this-commit`, the gate that refuses to build from a commit
  whose suite was red, and the release-pipeline page never described it. A truth
  test about the release pipeline was invisible in the release pipeline's own CI.
  The page now documents the gate as step 1 of five, including that it fails
  closed on an API error, a pending run, or zero matching checks.

### Fixed (documentation that claimed more than the code did, 2026-09-11)

- **`docs/BETA.md` claimed the insufficient-evidence exit criterion was met** and
  that "the remaining gate is the first publish itself". Both were false: 122
  capabilities return a confident value where nothing was measurable. The
  criterion is now marked NOT MET with the count and a link to the plan.
- **`docs/QUALITY_AND_HARDENING.md` claimed "every verdict surface carries three
  states"**. That was true of the surfaces the could-not-check campaign reached
  and not of the library; the campaign fixed everything it found, but had never
  measured most of the surface, and "nothing found" was being read as "nothing
  there". The claim is now scoped, with the per-capability table above it.
- The page's headline figures were stale: 4,354 tests and 68.84% coverage from
  2026-08-28. The verified figures at `8559dda` are **8,088 tests passing on
  3.11, 3.12 and 3.13** and **81% coverage**.

### Removed (breaking, 2026-08-22 and 2026-09-09)

Nine documented parameters and option values stopped being accepted before this
entry was written: one removed option value (2026-08-22), five removed
parameters and three options that now refuse (all 2026-09-09). Not one of them
was recorded here. That omission is the defect this section closes: a changelog
that carries some breaking changes and not others reads as complete, so a
reader who does not find their parameter listed concludes it still works. Every
line below was reproduced by execution against the tree at the time of writing,
and each refusal is quoted from that run rather than paraphrased.

- **`FairRegressor(method='reweighting')` is removed and raises `ValueError`**
  (2026-08-22, and unrecorded here until now).
  Uniform per-group sample reweighting cannot move group prediction means: for
  any positive group-uniform weights the weighted least squares normal equation
  forces the fitted group means to equal the data group means, so the
  intervention provably could not constrain the quantity it named, and its
  weight factor `1 - sign(diff) * min(|diff|, tolerance)` went negative for
  `tolerance > 1`, which sklearn's `sample_weight` validation rejects. The
  default `method` moved from `'reweighting'` to `'offset'` in the same change,
  so a caller who never passed `method` switched algorithm without touching a
  line. The type annotations went on advertising the removed values afterwards
  (`method: Literal["offset", "reweighting", "constrained"]` and
  `fairness_constraint: Literal["mean_parity", "error_parity", "bounded_loss"]`
  on `FairRegressor.__init__`, and the same `fairness_constraint` on
  `make_fair_regressor`), which is the same claims versus code defect one layer
  up: an editor completes an annotation, a type checker blesses it and a reader
  trusts it. Both `Literal`s now list only what `fit()` honours, `'offset'` and
  `'mean_parity'`. The refusals are unchanged and still reachable from untyped
  callers: `'reweighting'` raises `ValueError`, and `'constrained'`,
  `'error_parity'` and `'bounded_loss'` raise `NotImplementedError`. Pinned in
  `tests/test_readiness3_annotations.py`, which derives the advertised values
  from the annotation itself and puts each one through a real `fit()`, so the
  pin fails if either list drifts again in either direction.
- **Five parameters that were stored and never read were removed**, so passing
  one is now a `TypeError` like any unknown keyword instead of a setting that
  appears to take effect: `PredictionReweighter(preserve_ranking=)`,
  `CalibratedEqualizer(preserve_calibration=)`,
  `TemperatureScaling(init_temperature=)`,
  `compute_feature_correlations(include_pvalues=)` and
  `FairnessTrainingAnalyzer.generate_recommendation(tradeoff_analysis=)`. In
  each case there was no honest behaviour to wire the flag up to: the clip in
  `PredictionReweighter` destroys within-group rank whatever the flag claimed,
  so `fit()` now counts the affected rows in
  `result_.calibration_impact['n_clipped']` instead; `CalibratedEqualizer` is
  pooled-distribution quantile matching, which destroys per-group calibration
  by construction when base rates differ; `TemperatureScaling` minimises over a
  bounded interval and has no starting point to take; and
  `compute_feature_correlations` computed and returned p-values regardless of
  the flag. Pinned in `tests/test_audit6_lane3_post_in_processing.py` and
  `tests/test_audit6_lane2_feature_engineering.py`.
- **Three option values now refuse instead of quietly running something else.**
  `GroupCalibrator(fallback_strategy='borrow')` ran the `'global'` strategy
  silently and raises `NotImplementedError`; `IMPLEMENTED_FALLBACK_STRATEGIES`
  is `('global', 'none')` and is exported so a caller can check first.
  `GroupThresholdOptimizer(grid_search=False)` raises, because no
  gradient-based path exists to switch to. `DistributionMatcher(method=)`
  accepts `'quantile'` alone and raises `ValueError` on anything else, where it
  previously stored the value and ran quantile matching anyway.
  `CausalFairnessLoss(causal_criterion=)` is the same defect with its
  annotation already corrected: `'direct_effect'` and `'path_specific'` raise
  `NotImplementedError`, and the `Literal` names `'total_effect'` alone.

### Changed (breaking, 2026-09-09)

- **The nine post-processing `fit()` methods take `y_true`, `y_prob` and
  `sensitive_attr` as keyword arguments only.** `PredictionReweighter`,
  `RejectionOptionClassifier`, `CalibratedEqualizer`, `DistributionMatcher`,
  `BaseReweighter`, `ThresholdOptimizer`, `GroupThresholdOptimizer`,
  `MultiObjectiveThresholdOptimizer` and `BaseThresholdOptimizer`. A positional
  call is a `TypeError` at the call site: `ThresholdOptimizer.fit() takes 1
  positional argument but 4 were given`. Why this was worth breaking: a
  reversed positional call passing `(y_prob, y_true)` was found live at eight
  call sites in the production consumer. The next line is
  `coerce_to_array(y_true).astype(int)`, so a probability column cast to all
  zeros, every threshold collapsed to the floor, and nearly every row was
  accepted. Measured on 600 rows, `ThresholdOptimizer` changed 97.5 percent of
  decisions and `GroupThresholdOptimizer` 58.7 percent. It survived for so long
  because the demographic parity gap afterwards read 0.007: accepting everybody
  is trivially equal, so the mitigation was neutered and its own fairness
  scorecard reported success. A runtime guard shipped first and refuses a
  reversed call whenever the score column is continuous, but it is blind when
  the scores are hard 0/1, which is the platform's default shape rather than an
  edge case; keyword-only closes exactly that gap by making the mistake
  impossible to write. The guard is kept as the second line of defence.
  Deliberately out of scope: `GroupCalibrator` and `IntersectionalCalibrator`,
  which take `protected_attr` and `protected_attrs`, and sklearn's own
  two-argument `fit(X, y)`.

### Fixed (LLM fairness honesty, wave 0, 2026-09-09)

The platform's LLM pathway was gap-analysed against a three-layer evidence
framework (behavioral, model/process, system/decision) and six honesty defects
were found live, four of them in this library's contract. Register: the
platform's `docs/reference/llm-fairness-assessment-concept.md`, items LF-01 to
LF-10. The library half:

- **`OutputAnalyzer` no longer reports "no disparity" for a comparison that
  never ran** (LF-06). `_compare` returned `delta=0.0, p_value=1.0,
  is_significant=False` for empty input and `p_value=1.0` for a group with a
  single sample. Empty input is now `assessed=False` with every numeric field
  `None` and `not_assessed_reason="empty_input"`; a single-sample group keeps
  its measured means and delta but reports `p_value=None`,
  `is_significant=None`, `effect_size=None` with
  `not_assessed_reason="fewer_than_2_samples_per_group"`. Bonferroni and
  Benjamini-Hochberg families are built from testable metrics only, so an
  unassessed metric neither shrinks the others' adjusted p-values nor comes
  back "not significant". `IntersectionalResult.has_intersectional_bias` is
  `None` when no pair could be tested. `CounterfactualTester.compute_disparity`
  returns `None` metrics (not `0.0` / `1.0`) on `no_data` and `insufficient`,
  and `CounterfactualResult.is_significant` is `None` when no pair had two
  valid responses on each side. BOLD treats a `None` delta as an unmeasurable
  domain, which it already did for a domain with no generations.
  `text_fairness`, `embedding_bias` and `nondeterminism` already refused this
  way; the analyzer now matches them. Pinned by
  `tests/test_llm_wave0_honesty.py`, each refusal paired with a healthy control.
- **The judge can no longer be the defendant** (LF-03).
  `LLMJudgeScorer(subject_endpoint_url=..., subject_model_name=...)` raises
  when the judge endpoint and model are the system under test, and
  `judge_is_subject()` is exported so a caller can check before constructing
  one. The platform had been building the judge from the very endpoint it was
  assessing. `scorer_status()["llm_judge"]["quality"]` is now `"unvalidated"`
  rather than `"production"`: no judge on this platform has been checked
  against human ratings, and a citation is not a validation.

### Added (LLM fairness honesty, wave 0, 2026-09-09)

- **Sampling parameters travel with every request and are recorded** (LF-04).
  `CounterfactualTester(temperature=, top_p=, seed=)` with per-call overrides
  on `run_test`, and `CounterfactualResult.sampling` records what was used.
  `LLMApiProxy.send_prompt` and `send_batch` accept `top_p` and `seed`; both
  are omitted from the request when `None`, and `seed` is never sent on the
  Anthropic format, which has no such parameter. The tester used to hard-code
  `temperature=0.0` on every call, so a "repeated runs" result measured API
  nondeterminism at a temperature nobody deploys, and nothing recorded that.
  Default behaviour is unchanged (`temperature=0.0`).
- **`noise_floor_from_runs()`** (LF-01). The noise floor from the series it is
  defined on: repeated responses to ONE prompt, per metric and per variant,
  with a bootstrap interval on each mean, a **disparity noise floor**
  `2 * sqrt(var_ref/n_ref + var_v/n_v)` for each non-reference variant, the
  observed disparity, `exceeds_noise` and the systematic offset, scorer
  provenance, the sampling the runs were made with, and three states at every
  level (`measured`, `measured_with_limitation` below the recommended run
  count, `could_not_check` under two valid responses). The Navigator had been
  feeding `characterize_noise` the absolute disparities across templates, a
  different series from the one its docstring names. Exported at the top
  level; `vfairness.llm.RUN_METRICS` lists the four metrics it can build a
  series for without an external service.

### Fixed (type contract of the three-state fields, 2026-09-09)

- `IntersectionEffect.significant` and `ExperimentResult.heterogeneity_detected` are `Optional[bool]`: `None` means the test could not run, and the repr says "not assessed" instead of "no". `CompositionalityResult.scenario` admits `"not_assessed"`. `worst_group_accuracy` converts its group argument before grouping. `SyntheticResampler.get_resampled_data` refuses, rather than crashes, when called before `fit`. `TrainableGroupCalibrator.calibrator` and the adversary's layer list are typed for what they hold. These were mypy findings on the Quality gate; behaviour unchanged.
- The notebook runner in the test suite: `ipython` joins the `dev` extra (every notebook displays through it) and the monitoring notebook labels its weekday box plot through `set_xticks`, not the `labels=` keyword matplotlib 3.9 renamed and 3.11 removed.

### Fixed

- **The published documentation stops making claims the code does not support**
  (2026-08-28), and four new test modules keep it that way. The pre-release audit
  found the same defect class the library itself exists to prevent, in its own
  docs: a statement that reads as assessed while nothing assessed it.
  - `docs/site/api-reference/index.html` told every reader that
    `FairnessAnalyzer` takes `task_type='ranking'`. The constructor refuses it by
    name. Ranking fairness is measured by the standalone functions taking
    `(rankings, groups)`, and the page now says so; `backend`, the one
    constructor argument with no row in the parameter table, has one.
    `tests/test_api_reference_page_truth.py` derives the accepted values from the
    library's own refusal message and fails if the table and the signature drift
    apart in either direction.
  - The site-wide footer's **Contact** link answered 404 on all 18 pages that
    carry a footer, and had done since it was first recorded on 2026-08-22. It
    now points at the imprint, where the site publishes its contact addresses.
    Twelve call-to-action blocks linked to notebooks that the publish boundary
    deliberately never exports, so they could never resolve for a reader; they
    are notes now, and say why. `tests/test_site_links_truth.py` enforces a
    registry of URLs verified dead, refuses a link into a path the export
    deletes, and resolves every relative link and `#fragment` on the site.
  - The changelog page said 0.1.0 was "cut on 2026-08-23" and that entries had
    "landed after the 0.1.0 cut". Nothing has been cut; that date is the fiction
    this file already records having removed. Five statements on that page, and a
    stale `[Unreleased]` instruction on the release-pipeline page, were corrected.
  - Quality & Hardening called its CI gates "blocking" six times with no
    statement of what that does and does not mean, while the release-pipeline
    page said plainly that nothing blocks a merge. Both pages now carry the same
    caveat, checked against `gh api .../branches/main/protection` on 2026-08-28.
    The same page reported a failing test that no longer exists; the row is kept
    as a closed record rather than deleted, and
    `tests/test_site_links_truth.py::test_every_test_named_as_a_node_id_in_the_docs_exists`
    refuses a citation of a test that is not there.
- **The unreleased-tense prose can no longer survive the release.** `README.md`
  is the PyPI `long_description`, and a PyPI version's description is immutable:
  publishing 0.1.0 as it stands would leave its project page saying, permanently,
  that 0.1.0 was never published. `tests/test_release_tense.py` reads the release
  state out of this file's own top heading and requires `README.md`,
  `CITATION.cff` and the published pages to agree with it in both directions, so
  dating the heading alone turns the suite red until the rest follows, and
  stripping the honest warnings early turns it red too. The flip is now step 3 of
  "Cutting a release" in `docs/RELEASE_PIPELINE.md` rather than a step recorded
  only in an internal runbook.

### Changed

- **TestPyPI removed from the release pipeline** (Daniel, 2026-08-28). TestPyPI
  is a separate service with its own account, and maintaining a second one is
  real overhead for a solo maintainer. `publish-pypi` now depends on
  `[build, smoke]` directly, and the `testpypi` job and environment are gone.
  This removes the rehearsal, so the remaining gates carry more weight and none
  of them may be weakened: the repository guard, so only `validantai/vfairness`
  can publish; the tag-versus-`__version__` assert, which fails before anything
  is built; `smoke`, which installs the built wheel into a clean venv and runs
  it; the required reviewer on the `pypi` environment, now the last human
  checkpoint before an immutable upload; and `workflow_dispatch`, which runs
  build and smoke and cannot publish, because the publish job requires a tag.
  That dispatch run is now the dry run. A PyPI version number cannot be reused.

### Added
- **Typed report contract (`FairnessReport`).** The report returned by
  `FairnessAnalyzer.get_report`, `classification_fairness_report` and
  `regression_fairness_report` is now documented and type-checked by a set of
  `TypedDict` definitions in
  `src/vfairness/evaluation/vfairness_metrics/report_types.py`: `FairnessReport`
  plus `AssessmentReport`, `DataInfo`, `ExplanationsReport`, `MetricStatusEntry`
  and `InsufficientEvidenceGroup`. They are re-exported from the top-level
  `vfairness` package and from `vfairness.evaluation`, and added to the frozen
  public surface in `docs/API_STABILITY.md`. These are static-only annotations:
  the report is the same plain, JSON-serialisable `dict` it has always been, so
  every existing consumer, subscript access and `json.dumps(report)` is unchanged;
  a type checker now also catches a mistyped key such as `report["assessement"]`.
  The report structure is documented in `docs/API_REFERENCE.md`, and its produced
  keys are pinned in `tests/test_report.py::TestTypedReportContract`.

### Changed
- **Fairness radar chart now plots fairness outward (readability fix).** The
  spider chart previously plotted raw disparity on a tight adaptive axis, so fair
  metrics (small values) collapsed into an unreadable blob near the centre and the
  pass thresholds all clustered at the origin. Each metric is now mapped to a
  per-metric fairness score on a fixed 0-to-1 axis: 1.0 (fully fair) at the rim,
  0.0 (unfair) at the centre, honouring metric direction (disparate impact and the
  other `*_ratio` metrics are higher-is-better). A fair model therefore draws a
  large round shape and a failing metric caves inward, and every metric's threshold
  maps to a single clean reference ring (a dot outside the ring passes, inside it
  fails). Dot labels still show the raw metric value. The auto-generated radar
  explanation in `explain.py` was rewritten to match the inverted axis. The
  interactive Plotly radar `plot_metrics_radar` was brought into line with the
  same fairness axis (with `normalize=True`), so the SVG and interactive radars
  now read the same way, and the docs-site gallery visual was regenerated. Pinned
  in `tests/test_rendering.py::test_radar_fairness_radius_direction`,
  `::test_radar_svg_plots_fair_metrics_outward` and
  `tests/test_visualization_radar.py`.

### Changed (fourth-iteration pre-release audit)

A tenth-dimension audit ran against the release candidate on 2026-08-27. Its
verified blockers are fixed below; the remainder is tracked as a live register
that is internal and is not published with the library. Three of the fixes
change what a verdict says, so they are Changed rather than Fixed.

- **Assessment methodology bumped to M1.1: every metric is graded in its own
  direction.** The better-direction of a metric is now resolved once, for every
  surface, by `evaluation/vfairness_metrics/_metric_direction.py`. A violation
  magnitude (a difference, gap, disparity, error) passes at or below its bound; a
  parity ratio (the four-fifths / disparate-impact family) passes at or above it.
  Previously the report builders special-cased a single metric name for the
  higher-is-better branch and the analyzer's confidence-interval verdict assumed
  a `[0, threshold]` fair band for every metric, so the whole ratio family was
  graded backwards: `disparate_impact_ratio = 0.00`, a group never selected at
  all, passed, and perfect parity failed. A metric whose direction the resolver
  cannot determine is now could-not-check, excluded from the verdict and from the
  fairness score, where it was previously graded as if lower were better and so
  usually passed. This moves published PASS / FAIL counts, which is why
  `METHODOLOGY_VERSION` moves `M1.0` to `M1.1` (verdict rules only: no metric
  definition, threshold or metric-set change). Every report stamps the version,
  so a rating still records the methodology that produced it.
- **`assessment["fairness_score"]` is now `Optional[float]`.** It is `None` when
  nothing was assessable. `None` is the third state, could-not-check: it is not
  `0.0` (measured, and every metric failed) and not `1.0` (measured, and every
  metric passed), and reporting either of those was two opposite lies about the
  same absence of evidence. A consumer that formats the score as a percentage
  must now check for `None` first. In the same state every metric entry is listed
  under `not_assessable_metrics` and none under `passed_metrics`, so a reader of
  `passed_metrics` alone cannot mistake a vacuous run for a clean one.
- **An unrecognised `missing_strategy` or `task_type` now refuses to run.** Both
  fell through to the permissive branch: `missing_strategy="raise"`, the obvious
  typo for `"error"`, silently ran a fail-open audit that dropped every row with a
  missing protected attribute, and a typo'd `task_type` skipped binary-label
  validation entirely. Both are frozen constructor surface in
  `docs/API_STABILITY.md`, which already promised a `ConfigurationError` for an
  unknown option; the check now happens before any row is touched, and matching is
  exact and case-sensitive.

### Fixed (fourth-iteration pre-release audit)

- **A distribution-shift score that was never measured no longer reads as
  STABLE.** Three independent halves each turned a NaN MMD score into the calmest
  state the monitoring dashboard has: `_drift_color(nan)` fell through to emerald
  because every `>=` against NaN is False, `max(0.0, min(1.0, nan))` is `1.0` in
  CPython so the bar rendered full width, and the template chose the chip with
  `{% if s.score >= 0.1 %}`, which NaN fails exactly as a tiny score does. The
  verdict is now computed in the adapter as three states (SHIFT / STABLE / NOT
  MEASURED) and the sibling `_trend_slope_color` carried the identical hole.
  Pinned against the RENDERED SVG in `tests/test_monitoring_nan_not_measured.py`,
  because the engine applies a palette transform and asserting the source hex
  would pass while the operator still saw green.
- **A proxy scan that could not look stops reporting "no proxies".**
  `association_strength` swallowed a failure and returned `0.0`, so a column that
  is a perfect proxy for race but stored as list-valued objects scored zero
  association, and `identify_proxy_features` returned `[]` with no warning at all.
  It now returns NaN with a named `ProxyScanIncompleteWarning`, and the scan names
  every column it could not assess. Three further `except Exception: continue`
  sites in the same module were given the same treatment, and
  `association_strength` gained the direct tests it never had.
- **The explainability and in-processing surfaces stop certifying what was never
  measured.** Two false-certificate defects in code that CI had never actually
  executed (see the extras fix below).
- **`MetricsStore` differential privacy now holds across repeated reads.** It
  documented epsilon-differential privacy (Dwork 2006) but redrew Laplace noise on
  every query with no budget, so averaging repeated reads recovered the exact
  value. The utility limit is now stated where someone setting epsilon will read
  it.
- **Report and SVG surfaces render three states, never two.** A metric that could
  not be measured is NOT ASSESSABLE, an absent confidence interval prints COULD
  NOT CHECK rather than a number, and a significance p-value that was never
  computed is no longer defaulted to `1.0` and read as "not significant". The
  detailed fairness report no longer prints a PASS verdict, or a `<=` bound, for a
  ratio metric that is breached below its bound. Could-not-check never borrows the
  palette of a pass or a fail.
- **Degenerate data can no longer produce a passing report.** With fewer than two
  valid groups every disparity metric is vacuous; the guard is now computed before
  the threshold loop and gates it, instead of being a caveat appended afterwards
  that a consumer reading `passed_metrics` would never see. The
  all-groups-too-small warning stopped promising "default values (0.0 for
  differences, 1.0 for ratios)", the sentence that made a false-parity result look
  deliberate; those sentinels are gone and the metrics return NaN. Four regression
  metrics that returned a false-parity `0.0` with fewer than two qualifying groups
  now return NaN, matching the classification convention.
- **The CI/CD gate, the pytest test suite, and the Pulse dashboard grade in the
  metric's own direction too.** A configured metric that was never supplied is a
  could-not-check: a blocking metric refuses approval rather than reporting
  success. Pulse card tone is resolved per metric key and an unrecognised key
  reads "unknown", never "pass".
- **The MCP server names the real cause when `mcp` is too new.** `mcp` 2.0
  removed `mcp.server.fastmcp`, and the previous guard answered every
  `ModuleNotFoundError` with "install `vfairness[mcp]`", advice the reader had
  already followed. The `mcp` extra is now pinned to `>=1.2.0,<2`, and the error
  distinguishes "not installed" from "installed at an incompatible version" and
  prints the installed version.
- **CI installs the optional extras, so 41 explainability and in-processing tests
  stop silently not running.** No lane installed the `xai` or `training` extras,
  so those tests, including the regression pins for two findings previously rated
  critical, executed in no environment at all. The extras are installed in all
  three CI surfaces, a `parity` extra is declared, and
  `VFAIRNESS_REQUIRE_BACKENDS=1` makes a missing backend a named failure rather
  than a silent skip.
- **The release workflow refuses a mislabelled artifact.** Nothing checked that
  the pushed tag matched the package version, so `v0.2.0` would have published
  `0.1.0`; the build job now asserts the two agree and fails otherwise.
  `cyclonedx-bom` is pinned, and the SBOM step's `--outfile` flag, which that tool
  has never had and which argparse aborts the whole job on, is corrected to
  `--output-file`.

### Fixed (documentation, pre-release)

- **The README quickstart shows the output it actually produces.** It advertised
  `"1/5 metrics within thresholds (2 metric(s) not assessable ...)"`, where the
  real run prints `1/5 metrics within thresholds`; the fabricated half was the
  library's headline differentiator, in the first code a stranger runs. The
  three-state behaviour is now demonstrated by a second example that genuinely
  triggers it, and `tests/test_docs_truth.py` executes every README code block and
  compares the printed output to the `# ->` lines, so an output claim cannot be
  hand-maintained out of date again.
- **`conditional_demographic_disparity` discloses its divergence from the paper
  it cites, in the docstring.** The implemented statistic is the size-weighted
  within-stratum maximum selection-rate gap, unsigned and reference-free. The
  canonical CDD of Wachter, Mittelstadt & Russell (2021) is composition based and
  signed, and the two diverge materially on the same data, so a margin registered
  from that literature or from a SageMaker Clarify run does not transfer. This was
  recorded only in `docs/DIVERGENCES.md`, which ships in neither artifact and is
  not on the public docs site, so a pip user could not reach it. The note is now
  in the docstring that ships in the wheel, on the `_with_ci` variant that gates
  the seal, and in `docs/API_REFERENCE.md`. The estimand itself remains a recorded
  open decision, deliberately not realigned because sealed cells gate on the
  current statistic.
- **Dead outward links removed.** The README CI badge pointed at a workflow badge
  that returns 404 and rendered broken on the PyPI page; it is removed rather than
  shipped broken. The four CHANGELOG compare links pointed at tags that do not
  exist (the repository has no tags), so the version headings are plain text until
  there is something to link to.

### Added
- **Methodology ratified (VB-DOC-1).** `docs/METHODOLOGY.md` is authored and
  reviewed against the implemented code (all six sections: scope and inputs,
  metrics, statistical validation, verdicts, jurisdiction overlays, limitations),
  and `METHODOLOGY_VERSION` moves from `M0.1-draft` to the frozen `M1.0`. Every
  report stamps it, so a rating records the ratified methodology that produced it.

### Fixed (second-iteration audit, waves 6 and release pass)
- **conditional_adverse_impact used the attenuation-prone residual method.** It
  now estimates the group effect jointly (selection ~ covariates + group
  indicators, the average-marginal-effect gap, the binary counterpart of the
  `pricing_disparity` fix), so a covariate correlated with the group no longer
  hides part of the adverse impact. Pinned in
  `tests/test_audit_wave6_regression_controls.py`.
- **The undefined-rate NaN handling now holds at every deployment-gate call
  site.** The gate-side fix had left two copies unmirrored:
  `FairnessTestSuite` mapped a NaN metric to FAILED with a misleading message
  (now SKIPPED = could-not-check) and silently dropped undefined-rate groups (now
  NaN); `BiasMonitor` carried the same silent drop (now NaN). A configured
  BLOCKING metric that could not be computed at all made `ModelFairnessGate`
  APPROVE with only a warning; it now fails closed. Pinned in
  `tests/test_audit_wave6_cicd_nan.py`.
- **An undefined group rate is insufficient evidence, not 0.0 parity.**
  `predictive_parity_difference`, `fpr_parity_difference`,
  `fnr_parity_difference` and `equalized_odds_difference` collapsed a NaN
  per-group rate (a group with no positive predictions, no negatives, or no
  positives) into a 0.0 "perfect parity" result, which read as PASS with a
  deceptively tight [0.0, 0.0] bootstrap CI. They now return NaN so the verdict
  routes to insufficient_evidence, mirroring the earlier NPV fix; the
  single-qualifying-group 0.0 behavior is unchanged. Pinned in
  `tests/test_audit_wave6_metrics.py`. The second-iteration audit of 2026-08-22
  raised 84 code findings, 71 of them adversarially confirmed, plus 124
  documentation findings; that record is internal and is not published with the
  library.

### Fixed (third-iteration function-level release audit)

A third audit, on 2026-08-22, crossed every function against its literature
definition across all 17 packages and raised 29 findings; that record is internal
and is not published with the library. All confirmed findings are fixed and
pinned in `tests/test_audit3_*.py`.

- **Calibration metrics were misclassified as ratios by a substring test.** The
  check `"ratio" in <metric_name>` matches "cali[bratio]n", so
  `calibration_difference`, `multicalibration` and `integrated_calibration_index`
  (all lower-is-better) took the higher-is-better ratio branch across ten renderer,
  explainer and integration sites, inverting the PASS/FAIL verdict, colour,
  severity and acceptable band. Every site now keys on the `_ratio` suffix, which
  only genuine ratio metrics carry. (This is the same class as an earlier
  sealed-verdict incident.)
- **`TemperatureScaling.fit` had an inverted NLL gradient sign,** so fitting drove
  the temperature the wrong way and made calibration worse; it now minimises the
  NLL directly with a bounded scalar optimiser (Guo et al. 2017).
- **Proxy detection was blind to a continuous proxy of a categorical protected
  attribute** (it used Cramer's V, which needs two categoricals); it now uses the
  correlation ratio (eta). `CorrelationReducer` / `FeatureSuppressor` label-encoded
  a categorical protected attribute into a single ordinal column (removing only the
  linear component); they now one-hot it so every category is controlled.
- **The groundedness scorer fails closed on malformed judge output.** A boolean
  score no longer reads as perfect groundedness, an out-of-range score is refused
  rather than clamped up, and a payload-supplied judge endpoint is routed through
  the SSRF egress guard. VG-005 is now the answer-incidence hallucination rate (per
  the platform contract), not a claim-weighted mean that an all-hallucinated batch
  could pass.
- **Multi-group in-processing constraints now compare max-min across all groups**
  (not a reference group), and an empty group is excluded rather than assigned a
  fabricated 0.5 rate. The intersectional over/under-prediction significance test
  now tests the hypothesis it states (prediction vs the group's own base rate), and
  violation attribution is per-metric (not always selection-rate).
- **The Pulse headline/deploy verdict is driven by the worst-severity outcome,**
  not the largest numeric gap, so a critical exclusion on one attribute cannot hide
  behind a larger benign gap on another. The vision representation gate now catches
  under-representation, not only over-representation.
- **Pandas 3 robustness.** Protected-attribute dtype detection used
  `dtype == "object"` / `np.issubdtype`, which the pandas-3 string dtype does not
  satisfy (and which raises on extension dtypes), so a string attribute crashed or
  silently produced empty results; the affected paths now use
  `pd.api.types.is_numeric_dtype`. Verified under a pandas-3.0.5 environment.

Road-to-beta hardening (see the internal roadmap register; issue IDs are VB-*).

### Added
- **Output branding switch (VB-COM-2).** New public `set_branding()` and `branding_enabled()`
  remove the validant.ai mark from generated SVGs and reports, honouring an explicit call, then
  the `VFAIRNESS_BRANDING` environment variable, then a branded default. A single chart can opt
  out with `render_svg(name, {"branding": False})`. This makes the sponsor perk on the offer page
  real and checkable. The switch is unconditional by design: the library is Apache-2.0 and has a
  test-enforced zero-telemetry guarantee, so nothing verifies a licence or calls home, and the
  docs say so plainly rather than implying a lock that cannot hold.

  Turning it off removes the mark from all 44 chart templates through a single guard in
  `_shared_defs.svg`, because every template `<use>`s the two symbols defined there. The rendered
  SVG also has its now-dangling `<use href="#validant-...">` references stripped, so an unbranded
  chart contains no occurrence of the word at all, not merely no visible logo. The brand-only
  gradients moved inside the guard for the same reason. HTML reports keep their timestamp when
  unbranded, since a timestamp is report metadata and not branding.

  `tests/test_branding.py` (30 tests) proves it structurally (every template draws its mark from
  the guarded symbols, none defines its own) and behaviourally (no `validant` string, no dangling
  reference, still well-formed XML, across charts, HTML, Markdown, and compliance model cards).
- Statistical-robustness layer for the SEALED spec-v2 gates so a gate passes on affirmative
  confidence-interval evidence (one-sided equivalence / TOST), never a point estimate. New CI
  variants: `disparate_impact_ratio_with_ci`, `fpr_parity_difference_with_ci`,
  `negative_predictive_value_difference_with_ci`, `conditional_demographic_disparity_with_ci`,
  `pricing_disparity_with_ci`, `integrated_calibration_index_with_ci`, `multicalibration_with_ci`.
  Custom-shape statistics (needing y_prob / strata / continuous controls, which
  `compute_metric_with_ci` cannot pass) get a CI via a new `bootstrap_over_index` helper that
  resamples ROW INDICES (stratified by group) and reuses the existing bootstrap / NaN handling.
  The interval widens with smaller n and feeds the three-state verdict (fair / unfair /
  insufficient_evidence). Validated to the engine bar with hand-derived GOLDEN values +
  cross-library reference parity (fairlearn demographic_parity_ratio, sklearn per-group roc_auc,
  statsmodels Logit calibration slope). Registered + exported top-level; public `__all__` +7.
- Continuous / decision-utility / integrity metrics completing the spec-v2 panel:
  `pricing_disparity` (residual continuous-outcome (APR / rate) disparity after controlling for
  legitimate risk factors, the lending co-sealed price metric binary metrics cannot express),
  `net_benefit_parity` (decision-curve net benefit across groups, the EU-healthcare "fair but
  clinically useless" guard; Vickers & Elkin 2006), `conditional_adverse_impact`
  (regression-controlled residual adverse impact, a hiring-US diagnostic), and registration of
  the existing `multivariate_proxy_leakage` (reconstruct-the-protected-attribute audit, the
  lending proxy-leakage integrity check) as a manifest capability. Public `__all__` grew by 3
  names (multivariate_proxy_leakage was already exported); api_surface snapshot + tests updated.
- Independence / sufficiency / validity metrics for the spec-v2 spine primaries:
  `disparate_impact_ratio` (the selection-rate RATIO / four-fifths statistic, the US-hiring
  and US-lending independence primary, kept distinct from the `demographic_parity_difference`
  band), `conditional_demographic_disparity` (size-weighted within-stratum selection-rate
  disparity, the CJEU objective-justification mirror the EU-hiring cell seals on;
  Wachter/Mittelstadt/Russell 2021), `negative_predictive_value_difference` (NPV parity, a
  criminal-justice sufficiency diagnostic), and `auroc_parity` (per-group discrimination gap,
  the EU-healthcare validity gate against levelling-down). Registered + exported top-level
  (manifest 165 -> 169); public `__all__` grew by 4 names (api_surface snapshot + tests added).
- Recalibration + subgroup-calibration metrics for the spec-v2 spine (the EU-healthcare
  profile seals on group multicalibration / the integrated calibration index, NOT bare
  bin-ECE which depends on an arbitrary grid): `calibration_in_the_large` and
  `calibration_slope` (the TRIPOD recalibration validity diagnostics, via a
  dependency-free IRLS Cox recalibration), `integrated_calibration_index` (grid-free ICI
  with a per-group disparity, Austin & Steyerberg 2019), and `multicalibration` (the
  worst-subgroup calibration violation over group x prediction-bin cells,
  Hebert-Johnson et al. 2018). Exported top-level + registered in the manifest (capability
  count 160 -> 164); public `__all__` grew by 6 names (4 functions + 2 result containers,
  snapshot in `tests/test_api_surface.py` updated).
- Four evaluation metrics the capability catalog referenced but the engine did not
  implement, closing the overclaims found in the knowledge-graph spine audit (a
  credential can no longer attest a metric that cannot be computed):
  `fpr_parity_difference` (predictive equality, the FPR leg of equalized odds),
  `fnr_parity_difference` (equals equal-opportunity by the FNR = 1 - TPR identity),
  `accuracy_parity_difference` and `worst_group_accuracy` (both documented as
  performance/robustness statistics, NOT group-fairness criteria). Exported
  top-level + registered in the manifest (capability count 156 -> 160); public
  `__all__` grew by 4 names (snapshot in `tests/test_api_surface.py` updated).
- The `FairnessAnalyzer` pass/fail verdict now derives from the confidence interval,
  not the point estimate: `MetricResult` gains a three-state `verdict`
  (`fair` / `unfair` / `insufficient_evidence`) and `is_fair` becomes the boolean
  alias `verdict == 'fair'`. A wide small-sample CI is no longer reported as a pass
  or a fail (previously `is_fair = abs(point) <= 0.1` discarded the CI).
- `predictive_parity_difference_with_ci`: a confidence-interval variant for the
  sufficiency metric ER-005, which previously shipped as a bare point estimate even
  though it is one leg of the impossibility trade-off (spine audit finding 14).
- SSRF egress guard (`vfairness.net.egress`): `validate_endpoint`, `guarded_post`,
  DNS-rebinding-safe `PinnedIPAdapter`, and `SSRFError`, wired into the LLM
  `api_proxy` so outbound assessment traffic cannot be pointed at loopback,
  private, link-local, or cloud-metadata (169.254.169.254) targets. Adds `net` as
  a public sub-package (snapshot in `tests/test_api_surface.py` updated). This
  security control had been built but never merged to `main`.
- Integrated five fairness-engine capability sets that had been built but never
  merged to `main` (they were stranded on the `xai-engine-capabilities` branch).
  All are API-importable and functionally verified; the preprocessing catalog
  also registers dispatch keys, so it appears in the capability manifest
  (capability count 150 -> 156). Public `__all__` grew by 10 names; the snapshot
  in `tests/test_api_surface.py` was updated in the same change.
  - Preprocessing catalog (`preprocessing.feature_engineering`):
    `SyntheticResampler` (SMOTE/ADASYN/Tomek), `PropensityScoreWeighter`,
    `InversePropensityWeighter`, `CounterfactualAugmenter`, plus
    `propensity_weights` and `counterfactual_augment`. New registry dispatch
    keys: `smote`, `adasyn`, `tomek`, `propensity_weighting`,
    `inverse_propensity`, `counterfactual_augment`.
  - Sequential drift detection (`operations.monitoring.sequential`):
    `page_hinkley`, `cusum_drift`, `sequential_fairness_drift`; plus
    `paired_metric_significance` in feature engineering.
  - Exposure-parity re-ranking (`post_processing.ranking`):
    `exposure_parity_rerank`, `exposure_parity_difference`,
    `get_ranking_group_metrics`.
  - Calibration decomposition (`post_processing.calibration.metrics`):
    `per_group_brier_decomposition`, `ece_confidence_intervals`,
    `sufficiency_test`, `cv_calibration_stability`.
  - Torch-free adversarial/baseline diagnostics (`in_processing`):
    `baseline_comparison_summary`, `sklearn_adversarial_debiasing`,
    `adversarial_convergence_diagnostics`.
- Lint gate (VB-LINT-1): `.github/workflows/quality.yml` runs `ruff check src
  tests` on every library change (blocking), and `ruff`/`mypy` are declared in
  the `dev` extra so CI, pre-commit, and local runs share one toolchain. The tree
  is now lint-clean under the E/F/I/N/W rule set: import sorting and dead imports
  fixed across the suite; a `tests/**` E402 per-file-ignore for the legitimate
  `pytest.importorskip` pattern; the serialized benchmark-provenance fields
  (`subsetSize`/`fullBenchmarkSize`/`provenanceNote`) carry a documented `N815`
  noqa; and four malformed `BLE001` noqa directives corrected. Also completed
  `tests/test_ranking.py::test_ndkl_top_k` with the top-k-vs-full assertion it
  documented but never checked.
- Type gate (VB-TYPE-1): `mypy src` is now BLOCKING in `quality.yml`. A full
  type-error burndown fixed all 500 mypy errors to zero across 82 modules with
  root-cause fixes; only ~9 targeted, justified `# type: ignore[code]` remain,
  all for genuine third-party stub gaps (e.g. torch's `nn.Module.forward`). The
  pass surfaced three real bugs:
  - `operations/pulse/orchestrator.py`: `subgroup_robustness_audit` was called
    with the wrong argument order, so it always raised, the error was swallowed,
    and the subgroup fairness-gerrymandering audit was silently dead. Fixed. This
    ACTIVATES the audit, so Pulse output changes; the full suite and Pulse tests
    stay green (confirm the Pulse Contract CI on this change).
  - `in_processing/wrappers/sklearn_wrappers.py`: `FairRegressor.fit` crashed on
    a list/Series `sample_weight`; it now coerces to ndarray (ndarray inputs are
    unchanged).
  - `evaluation/vfairness_metrics/discovery.py`: correlation-based proxy
    detection was a no-op (a list was passed where a single column name is
    required, so the call always raised and the error was swallowed). Now fixed:
    detection runs once per declared protected target and flags correlated
    columns as `proxy_candidate`, covered by `tests/test_column_roles_proxy.py`.
  Several result-type annotations were corrected to match real runtime contracts
  (e.g. `StatisticalDisparityResult.confidence_interval` widened to
  `Tuple[Optional[float], Optional[float]]`; various metric-result dicts to
  `Dict[str, Any]`).
- Release integrity: a clean-room wheel-install smoke gate (VB-REL-3). The
  release workflow builds the wheel, installs it into a fresh environment, and
  runs `scripts/wheel_smoke.py` (public-API import, metric, analyzer, report)
  from outside the source tree before any publish, so a packaging defect (missing
  data file, wrong entry point, unshipped subpackage, missing dependency) blocks
  the release instead of shipping. Publishing to PyPI now depends on it.
- Strict typing island (VB-TYPE-2): a set of foundational public modules and core
  helpers (`_bands`, `_deprecation`, `_manifest`, `_methodology`, `exceptions`,
  `result`, `streaming`, and the metric-core `_grouping` / `_validation`) is now
  held to `mypy --strict` via a per-module override, and `mypy src` enforces it.
  The island expands outward as more modules are cleaned.
- Coverage gates strengthened (VB-TEST): branch coverage is on (`branch = true`)
  and the CI floor was ratcheted 30 to 40 as the suite grew (~45% branch-inclusive
  as of 2026-08); a PR-only `diff-cover` job additionally requires the lines a
  pull request adds or changes to be at least 80% covered.
- Formatter adopted (VB-LINT-2): the whole `src` and `tests` tree is
  `ruff format`-clean (double quotes, space indent) and `ruff format --check` is
  enforced in CI. The one-time reformat commit is listed in
  `.git-blame-ignore-revs` so `git blame` skips it.
- Public exception hierarchy rooted at `VfairnessError` (`InvalidDataError`,
  `InsufficientDataError`, `ProtectedAttributeError`, `ConfigurationError`); each
  also subclasses the built-in it replaces, so existing `except ValueError`
  handlers keep working (VB-API-5).
- `assessment.insufficient_evidence_groups` on the classification and regression
  reports: strata excluded by the group-size gate are now surfaced as an explicit
  insufficient-evidence verdict instead of silently vanishing (VB-API-4).
- `py.typed` marker so downstream type checkers use the library's type hints
  (VB-API-1); a public API-surface snapshot test (VB-API-3).
- Deprecation helper `vfairness._deprecation.deprecated` and a documented policy
  (VB-API-2).
- Opt-in per-instance result cache: `FairnessAnalyzer(cache=True)` memoizes
  `compute_all_metrics()` and `get_report()` by their arguments (deep-copied on
  store and return, off by default), plus `clear_cache()` (VB-PERF-3).
- Chunked/streaming computation for large data: `vfairness.streaming`
  (`stream_group_counts`, `streaming_selection_rates`,
  `streaming_demographic_parity`) computes exact aggregations in fixed-size
  chunks, bit-identical to the whole-array result, so very large datasets can be
  processed in pieces (VB-PERF-2).
- Executable documentation: `docs/examples.md` snippets are now run in the test
  suite via `tests/test_docs_examples.py` (doctest), so a documented example that
  breaks against the real API fails the suite (VB-DOC-3).
- Methodology versioning: `vfairness._methodology.METHODOLOGY_VERSION` plus a
  `docs/METHODOLOGY.md` scaffold; every `FairnessAnalyzer.get_report()` now stamps
  `methodology_version` into the report, so a rating records the methodology that
  produced it independently of the code version. Methodology prose is authored
  separately (VB-DOC-1).
- Canonical module taxonomy: the public surface is 15 top-level sub-packages
  (6 pipeline + 8 specialized + net), each now declaring `__all__` (added to `mcp`),
  documented in the README and enforced by `tests/test_module_taxonomy.py`,
  resolving the old 6/8/9/11 module-count inconsistency (VB-PKG-3).
- Mutation testing (mutmut): `setup.cfg` config, a fast boundary runner
  (`tests/test_bands.py`), a weekly/on-demand CI workflow, and
  `docs/MUTATION_TESTING.md`. Baseline on `_bands.py` is 57/58 killed (98%),
  above the 80% target; the one survivor is a documented equivalent mutant
  (VB-TEST-9).
- New test suites: synthetic analytic oracle, property-based (Hypothesis),
  impossibility relationships, determinism, zero-telemetry, deserialization
  safety, insufficient-evidence, performance/scale, report snapshot, a core
  no-silent-swallow guard, and reference-library parity (VB-TEST-1/2/3/5/10,
  VB-PERF-1, VB-GOV-2, VB-TEST-4). See `docs/DIVERGENCES.md`.
- Packaging: Apache-2.0 `LICENSE` + `NOTICE`, single-source version, and an sdist
  allowlist that drops the source archive from ~256 MB to ~1 MB (VB-PKG-1/2).
- Beta programme materials: `docs/BETA.md` (exit criteria, how to join, the
  no-SLA / patronage stance; the beta version line was later revised from
  `0.9.0b1` to plain `0.1.0` + Beta classifier, 2026-08-22) and a structured
  "Beta feedback" GitHub issue template. Recruitment and the first publish are
  operator steps (VB-GOV-3).
- Governance and supply chain: `SECURITY.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CITATION.cff`, Dependabot, and CI for trusted publishing
  + provenance, bandit/pip-audit/CodeQL, SBOM, and OpenSSF Scorecard
  (VB-SEC-1/2/4/5/7/8, VB-REL-1/2, VB-GOV-1, VB-DOC-4).

### Fixed
- **False-PASS on unmeasurable spec-v2 gates (adversarial audit, 2026-08-16).** Four new sealed
  metrics collapsed an UNMEASURABLE result to the numeric "perfectly fair" value (0.0) instead of
  NaN, so the report's NaN -> `not_assessable` guard and the bootstrap NaN filter were both
  bypassed and a no-evidence case sealed as fair: (1) `multicalibration` returned `alpha = 0.0`
  ("perfectly multicalibrated") when NO (group x bin) cell met `min_cell` (zero calibration
  evidence) -> now NaN when no cell is evaluated; (2) `negative_predictive_value_difference`
  returned 0.0 with a deceptively tight `[0,0]` CI when a group's NPV was undefined -> now counts
  only groups with a DEFINED NPV and returns NaN when fewer than two remain; (3) `auroc_parity`
  returned 0.0 (a PASSED validity gate) when fewer than two groups had a computable AUROC, i.e.
  precisely when the model is uninformative for a group -> now NaN; (4) `conditional_demographic_
  disparity` weighted by `n_s / n_total` but dropped un-assessable strata WITHOUT renormalising,
  biasing the sealed EU-hiring primary toward "fair" in proportion to the un-assessable population
  -> now renormalises by the assessed-stratum weight, and returns NaN when no stratum is
  assessable. `test_metrics_degrade_gracefully_on_single_group` updated: a single protected group
  is insufficient_evidence (NaN), not perfect parity.
- `test_report_snapshot` skeleton reconciled with the spec-v2 report wiring: `thresholds_used` now
  correctly lists `auroc_parity`, `integrated_calibration_index`, and `multicalibration` (added to
  `classification_fairness_report` under the y_prob branch but omitted from the snapshot when they
  were wired, leaving the suite red on main).
- Undefined (NaN) metrics are now routed to `assessment.not_assessable_metrics`
  and excluded from the fairness score instead of being mis-scored as a FAIL
  (`_within(nan, t)` is False). Affects `r2_parity_difference` when a group has
  constant `y_true` (VB-EVAL-1) and the classification error-rate metrics when a
  group has no positive labels (undefined TPR); the classification assessment
  block gains a `not_assessable_metrics` field (the regression report already
  had it).
- `_bands.risk_band` / `drift_severity` now read the module's
  `RISK_BAND_THRESHOLDS` / `DRIFT_BAND_THRESHOLDS` constants instead of
  hardcoding the same numbers, so the band thresholds have a single source of
  truth as the module docstring promises (found by mutation testing).
- `rendering.engine._risk_label` now delegates to `_bands.risk_band`, so the SVG
  risk badge and the explainer text share one source of truth and cannot drift.
- `calibration_vs_error_parity`'s `conflict_exists` now reflects the MEASURED
  metrics (TPR/FPR/calibration-gap disparities) instead of base-rate disparity
  alone. A perfect predictor with unequal base rates no longer wrongly reports a
  conflict (the impossibility vanishes at perfect accuracy). Found by the deep audit.
- `vfairness.streaming` now excludes rows with a NaN prediction OR a missing
  sensitive value per chunk (matching the standard `missing_strategy='exclude'`)
  and rejects non-binary predictions, so the streamed result is truly identical
  to the whole-array API even with missing data. Found by the deep audit.
- De-flaked `test_llm.py::test_bootstrap_ci`: it now passes `random_state` to
  `bootstrap_ci` (which resamples with its own generator) instead of seeding the
  global NumPy RNG, which never reached the resampler and left the test
  intermittently failing when the CI bound landed on 1.0.

### Changed
- License moved from MIT to Apache-2.0 (copyright Glinz & Company GmbH) (VB-LIC-1).
- Supported Python raised to `>=3.11` per Scientific Python SPEC 0 (VB-PKG-6).

### Security
- The xai sidecar refuses to deserialize untrusted model payloads unless the
  caller explicitly opts in (`trust_input` / `VFAIRNESS_TRUST_MODEL_INPUT`),
  closing a remote-code-execution vector (VB-SEC-6).
- The metric core no longer swallows computation errors silently; an AST guard
  test enforces this across evaluation / in_processing / post_processing
  (VB-SEC-4).
- Second security-audit hardening (defence in depth, no known exploit): the MCP
  `load_dataframe` tool rejects remote `data_path` URLs (`http(s)`/`s3`/`gs`/
  `ftp`/...), so an agent-supplied path cannot become network egress or an
  SSRF/metadata fetch; the Pulse artifact download pins the URL scheme to `https`
  and caps the response size; and the deserialization-safety scan now resolves
  aliased imports (`from pickle import loads as X`) and additionally covers
  `marshal`, `pandas.read_pickle`, `jsonpickle`, and `shelve`, so an RCE-class
  loader cannot be reintroduced under an alias.
- Reduced sensitive data in error logs: the xai worker no longer logs the full
  pgmq message (owner / job payload), the Supabase writer no longer embeds the
  raw response repr (which could echo row data), and the LLM proxy logs the
  prompt length instead of a prompt prefix.
- CI least privilege: the parity, full-test, and pulse-contract workflows now
  declare `permissions: contents: read` instead of inheriting the default token
  scope.

## [0.0.9] - 2026-08-10

> **Internal only. Never published.** 0.0.9 and 0.0.8 were development cuts made
> before the project went public. Neither was uploaded to PyPI and neither was
> tagged, so there is nothing to install and nothing to diff against; the first
> version a user can obtain is 0.1.0. They are kept here because the work they
> record is real and is in the 0.1.0 artifact.

### Fixed (deep-audit Wave 5: the 96-finding tail, closing out the audit)
- **In-processing (12 + smoke coverage)**: `analyze_tradeoffs`/`compare_methods`
  reject regression instead of scoring it with classification accuracy;
  group calibrators pass unknown group ids through untouched (previously
  silently zeroed logits) and disclose the mismatch; `BetaCalibrator` is now
  real Kull et al. 2017 beta calibration, not a renamed Platt transform;
  `EqualizedOddsConstraint.signed_constraint_value` restores sign feedback for
  exponentiated-gradient updates; `GridSearch` predicts with the classifier it
  reports (no more silent accuracy-weighted ensemble mismatch); its
  `ThresholdOptimizer` searches thresholds jointly via coordinate descent and
  validates the final assignment; `nu` no longer cached across refits; loss
  warmup honors "no penalty before epoch N"; `IndividualFairnessLoss` returns
  zero (not NaN) for batch size 1; fairness losses raise the torch ImportError
  at construction; `create_fairness_loss` maps individual/counterfactual
  metrics to the real loss classes. New end-to-end smoke battery covers every
  public loss, constraint fit/predict path, and calibrator.
- **Statistics honesty (10)**: `fisher_exact_test` uses `scipy.stats.fisher_exact`
  (the silent chi-square degrade above n=200 and the p=1.0 small-cell fallback
  are gone); `minimum_detectable_effect` computes z from the requested
  alpha/power; Bayesian group CIs report the observed rate as the point
  estimate (posterior disclosed in metadata); constant-score selection matrices
  warn instead of marking 100% selected; `removal_curve_auc` clamped to [0,1];
  intersectional baselines computed over the same population as `total_n`;
  pass/fail threshold checks are float-representation safe; constant-target
  regression reports mark scale-relative metrics NOT_ASSESSABLE; the
  visualization band derives from the actual thresholds, not a hardcoded 0.1.
- **Explainer coherence (7)**: ECE card severity and "well calibrated" prose
  derive from one threshold set in `vfairness._bands`; report severities roll
  up method-comparison and per-scale drift cards; boundary values (DP exactly
  0.08) read consistently; the last 0.2/0.4/0.6/0.8 risk-scale remnant deleted.
- **Pulse + ops (12)**: Baron-Kenny step 4 actually tests the mediator
  coefficient (was hardcoded True); refusal findings count toward per-axis
  significance; zero-tolerance CI/CD thresholds display (falsy-zero bug);
  JUnit XML escapes special characters; monitoring alerts honor explicit 0.0;
  seasonal detection works for any period; crashed causal refuters are
  disclosed as unavailable instead of counted as refutations.
- **Reporting + post-processing (9)**: threshold simulation reports honest
  absolute change from a zero baseline; what-if slider aligned to the
  precomputed grid; "most frequently alerting metric" selected by alert count;
  ECE computed once per tradeoff sweep; `get_group_statistics` returns the
  documented dict.
- **Rendering robustness (14)**: hostile-input crashes (empty modules, None
  nests, stringified numbers, Infinity) either render or raise a clear
  ValueError naming the template instead of leaking ZeroDivision/Overflow
  errors; `apply_skin` no longer rewrites hex tokens inside visible text;
  fairness score 0 reads as high severity (falsy-zero bug); critical proxies
  get their own CRITICAL row; `intersectional_disparity` and
  `workflow_overview` templates registered so the render smoke suite covers
  all 45; adapter behaviour tests assert on-canvas verdict text.
- **Agents/llm/misc (13)**: `rag_bias` separates the disparity threshold from
  alpha; temporal drift detectors return turn numbers as documented; distinct
  unlabeled documents no longer collide; cyclic legal rule inheritance raises
  instead of recursing forever; advertised transformer scorer backends are now
  installable via the `llm-transformers` extra.
- **Test-suite hardening**: hand-derived golden confusion-matrix fixtures with
  a nonzero FPR gap (the old fixture could never catch a lost FPR branch);
  tautological tone/manifest assertions replaced with exact pins and an
  AST-level duplicate-key check; agent tests assert detection outcomes, not
  key presence; effect-size boundary pins at 0.2/0.5/0.8.
- Regression tests: eight new `tests/test_audit_wave5_*.py` files (225 tests).

### Fixed (deep-audit Wave 4 — 55 confirmed findings across eight subsystems)
- **Evaluation metrics (11)**: no more crash when all rows are excluded
  (graceful NaN interval + warning); NaN probabilities rejected; the
  effect size attached to equalized-odds/equal-opportunity now measures
  the TPR/FPR gap the metric reports (not the selection rate); reports
  and MetricResult are JSON-serializable; equal opportunity returns NaN
  with a warning (not a silent "fair" 0.0) when a group has no positive
  labels; silently dropped small groups are named in a warning across all
  five difference metrics; selection_rate_disparity_matrix now rejects
  unrecognized labels (no more Ja/Nein -> all-negative), rejects length
  mismatches (no silent truncation) and honors the requested Wilson
  confidence level; assert_fairness thresholds aligned with the report;
  pooled FPR baseline weighted by actual negatives.
- **Explainer bands (3)**: new canonical `vfairness._bands` module; the
  explainer and the SVG badge can no longer drift (a NaN-handling drift
  had already crept in and is now pinned by a cross-module test); drift
  severities aligned to the chart's 0.30/0.60 bands; validation issue
  severities mapped correctly (error -> high).
- **LLM scorers (5)**: judge failures no longer fabricate neutral
  {5,5,5,5} verdicts; dead sidecars expose a health flag and warn instead
  of silent zeros; depth score made monotonic; rubric prompts no longer
  contain literal {{...}}; placeholder-scorer warnings are lazy (no more
  import-time aborts under -W error).
- **Pulse (4)**: refusal-disparity text direction derived from the actual
  rates; proxy-outcome "confirmed p<0.05" now Bonferroni-corrected;
  labelFree consistent with primaryNeedsLabels; EU AI Act screen fires
  for EU member states.
- **Ops (9)**: evaluate_from_metrics performs the degradation check;
  per-intersection threshold overrides apply only to their intersection;
  class-imbalance check restricted to binary outcomes; drift windows
  split at len(reference) and MMD compares baseline vs current (not
  baseline vs concatenation); weekly degradation direction fixed for
  disparity metrics; adverse-action reasons capped at 4 with unique
  codes; "self-contained" report HTML embeds Plotly (works offline);
  alerts no longer double-penalize the health score.
- **Post-processing (7)**: calibration memoization keyed by arguments
  (strategy no longer ignored after the first call); intersectional
  calibrator keys collision-safe for values containing '_'; calibration
  metrics validate binary y_true (ECE can no longer exceed 1);
  is_well_calibrated is metric-aware (MCE/Brier thresholds); groups
  dropped below min_group_size are disclosed in the disparity result;
  probability-1.0 samples kept in the top ECE bin; failed reweighting
  methods disclosed in the result instead of print-and-forget.
- **Preprocessing (6)**: the outcome column is no longer scanned as its
  own proxy; too-few-samples proxy scans warn instead of silent
  emptiness; marginal representation applies min sample gating
  ("insufficient data", not confident findings from 3 rows); implicit
  US-census benchmark labeled and warned; OVERREPRESENTED severity
  reachable; categorical disparity CI computed on the resolved positive
  class column.
- **Rendering + XAI (10)**: no-data charts render a neutral state instead
  of a "High" disparity badge; on-canvas band labels derive from the
  displayed (rounded) value so text and number cannot contradict (both in
  adapters and in the chart explanation layer); ratio-metric breach logic
  un-inverted; robustness badge scores only supplied components; binary
  SHAP decomposition warns when extra groups are ignored; np.trapezoid
  guarded for numpy < 2; DiCE feasibility actually checks bounds and
  prediction errors propagate; Explanation.to_db_row writes the real
  scope.
- Regression tests: eight new `tests/test_audit_wave4_*.py` files
  (187 tests); full suite 1045 passed.

### Fixed (deep-audit Wave 3 — validation, robustness, routing, docs truth)
- **NaN sensitive-attribute rows no longer form phantom groups** in the
  intersectional DataFrame path; under `missing_strategy='exclude'` they are
  excluded (and counted), under `'error'` they raise.
- **`y_true` is now validated as binary for classification** (previously only
  `y_pred` was checked, so 1/2- or yes/no-encoded labels silently dropped
  rows from every TPR/FPR denominator).
- **Absent benchmark groups are flagged**: a group expected by the benchmark
  but entirely missing from the data is reported as maximally
  underrepresented (ratio 0.0, full deficit) instead of "adequate";
  benchmark proportions that do not sum to 1 are renormalised with a warning
  instead of producing phantom deficits.
- **Critical findings floor the overall bias-risk score** (1+ critical ->
  at least MEDIUM band, 3+ -> at least HIGH); previously three critical
  issues could average out to 0.225 and render a green MINIMAL badge.
- **Mediation NDE now adjusts for the same backdoor set as the total
  effect** (previously NIE = total - NDE mixed two adjustment sets, biasing
  proportion_mediated whenever confounders existed).
- **`StatisticalResult.is_significant` is False for NaN intervals**
  (NaN comparisons made uncomputable intervals read as "significant").
- **`PipelineTracker.identify_bias_source` ranks by signed contribution**
  (ranking by |contribution| named a strongly bias-REDUCING stage as the
  primary source of bias).
- **Deep-model XAI routing selects Integrated Gradients** (implemented)
  instead of the Phase-2 `shap.DeepExplainer`, which made every deep job
  fail at `get_explainer`.
- **`lundberg_fairness_decomposition` rejects unsupported metrics loudly**
  (it computes a between-group mean difference; label-conditioned metrics
  silently got mislabeled results before).
- **Intersectional heatmap `max_disparity` is the true max-min spread**
  (previously adjacent-cell scan diffs, understating non-adjacent extremes).
- **NaN risk scores render an N/A badge** (slate) instead of falling through
  to a green MINIMAL; XML-invalid control characters are stripped from
  rendered SVGs so one hostile string cannot break the whole document.
- **Frozen API surface restored**: `calibration_difference`,
  `r2_parity_difference`, `residual_bias` are importable from the top-level
  package as `docs/API_STABILITY.md` guarantees.
- Documentation examples corrected against the real APIs (README
  quickstarts, API_REFERENCE statistical sections, getting-started page,
  SVG demo notebook cell repairs).
- Regression tests in `tests/test_audit_wave3.py`, including boundary pins
  for the pulse tone machinery and the equalized-odds FPR branch.

### Fixed (deep-audit Wave 2 — numerics, false certificates, band consistency)
- **Bayesian credible intervals were numerically wrong** (the homegrown Beta
  PPF returned e.g. 0.9999 where the true 2.5% quantile is 0.0338, collapsing
  "95%" intervals to ~55% actual coverage); `_beta_ppf` now delegates to
  `scipy.stats.beta.ppf` (scipy is a core dependency), with the approximation
  kept only as a warned fallback. Empirical coverage restored (~0.95+).
- **Three of five reweighters crashed on every `fit()`**
  (RejectionOptionClassifier, CalibratedEqualizer, DistributionMatcher called
  their own guarded `predict`/`transform` before setting `is_fitted`); all
  five now fit, and unfitted use still raises.
- **`get_group_metrics_with_ci` returned NaN bounds for every large group**
  despite documenting bootstrap CIs; large groups now get closed-form Wilson
  score intervals (new `wilson_score_interval` in `_statistics`), and the
  small-sample docstrings state the actual behaviour.
- **Degenerate data no longer certifies fairness**: the classification report
  now carries `assessment.assessable` and an explicit "NOT ASSESSABLE"
  summary when fewer than two valid groups exist (previously 0 samples or a
  single group produced fairness_score 1.0, "5/5 metrics within thresholds").
- **Quantile-binned ECE/MCE reported 0.0 ("perfect") for constant
  probabilities** (all quantile edges collapsed, leaving zero bins); a
  single full bin is used instead, giving the correct |mean(y) - mean(p)|.
- **Explainer severity counts were always zero for Enum severities**
  (`str(Enum.MEMBER)` never matches 'high'/'warning'); a `_enum_str` unwrap
  fixes six count sites.
- **Bias-audit explanation now uses the badge's band scale everywhere**
  (severity, evaluation text, interpretation guide all on 0.25/0.50/0.75
  MINIMAL/LOW/MEDIUM/HIGH; previously a divergent 0.2/0.4/0.6/0.8 scale
  contradicted the badge for ~40% of scores).
- **Radar chart and detailed fairness report gave contradictory verdicts**
  for the same input (bands 0.7/0.4 vs 80/50); the radar now uses 0.8/0.5.
- Regression tests in `tests/test_audit_wave2.py`.

### Fixed (deep-audit Wave 1 — silently wrong results and crashes)
- **Threshold optimization crashed on every fit**: operator precedence in
  `_compute_performance_metrics` (`mask & mask * weights` binds `*` first);
  all optimizers now fit, and weighted confusion-matrix math is correct.
- **`find_feasible_thresholds` declared every threshold feasible under
  equalized_odds**; it now requires both TPR and FPR to fall inside the
  tolerance-widened cross-group overlap, mirroring the single-metric logic.
- **`FairRegressor` silently fitted an unconstrained model** for
  `error_parity` / `bounded_loss`; those now raise `NotImplementedError`
  instead of pretending to mitigate.
- **`ProjectedAdversarialLoss` trained its adversary in the wrong direction**
  (no reversal, no optimizer, `-adv_loss` penalty); rebuilt on a scaled
  gradient-reversal layer with a separate adversary optimizer and
  `update_adversary()`, mirroring `AdversarialDebiasingLoss`.
- **`compute_max_difference` returned the group pair in alphabetical order**,
  so `scan_fairness_violations` could swap privileged/disadvantaged labels;
  the pair is now ordered by value (higher first).
- **Ranking metrics flattened [0,1] relevance scores to position 0**
  (reporting perfect parity for arbitrarily unfair rankings); positions are
  now detected only for exact 0..n-1 permutations, scores are argsorted.
- **Multiple-testing correction used inverted pseudo p-values**
  (`min(1, value/threshold)`: unfair metrics capped at p=1.0, never
  significant). The classification report now runs real two-proportion
  z-tests on the extreme group pair per metric; the regression report
  refuses to fabricate p-values and says why.
- **CI/CD gate, monitor and pytest-plugin default metrics silently measured
  only the first two protected groups**; all three now report worst-case
  (max-min) gaps across every group.
- **Reporting health-score trend was biased toward "degrading"** (previous
  window's alert/drift components hardcoded to 100); the previous score now
  uses the same three real components.
- **Causal identification reported unconfounded effects as NOT identifiable**
  (empty backdoor set treated as failure); identifiability now follows the
  DoWhy estimand contents.
- **Temporal-drift overall severity** now takes the worst severity across
  attributes rather than the severity of the highest-PSI entry (a degenerate
  high-PSI "pass" could mask a genuine "critical").
- **Pulse binary verdict crashed when a small group's CI was None**; the CI
  and four-fifths clauses are now guarded like the non-binary path.
- **Experiment heterogeneity ANOVA ran on bootstrap replicate arrays**
  (inflating significance by ~n_boot); replaced with Cochran's Q on point
  effects and their bootstrap standard errors.
- **LLM counterfactual `is_significant` was decided by response word-count
  alone**; it now rank-sum-tests the same per-response metrics that drive the
  reported disparities (sentiment, toxicity, refusal, length).
- **SHAP adapters mixed class-1 attributions with class-0 base values**;
  attributions, base value, and prediction now come from one class.
- **`vfairness.mcp.server` raised SystemExit at import time** without the
  mcp extra (killing introspecting host processes); it now raises a
  catchable ImportError.
- **`compare_to_benchmark(visualize=True)` always raised KeyError**
  (self-referencing dict literal); fixed.
- **Packaging**: `requirements.txt` now includes the hard runtime deps
  `requests` and `scikit-learn` (a documented minimal install could not even
  import vfairness); the `mcp` extra is gated to Python >= 3.10 so the
  project locks again (`uv lock --check` passes); `uv.lock` regenerated.
- **Docs**: three API_REFERENCE examples that raised verbatim
  (MetricResult `ci_lower`/`ci_upper`, `discover_intersectional_groups`
  signature/keys, `WorkerLoop.from_env()`) now match the real APIs.
- Regression tests for all of the above in `tests/test_audit_wave1.py`.

### Added
- **Self-explaining, accessible SVGs** (`vfairness.rendering.explain`). Every
  rendered SVG now carries its own explanation automatically, wired at the
  `render_svg` choke point (so every adapter and `.to_svg()` gets it):
  - an **on-canvas explanation**, auto-generated when the caller passes no
    `explanation`, structured as concept -> how to read it -> the data-specific
    finding on this chart -> recommended action. `explanation=None` (now the
    default) auto-generates; `""` suppresses; a string is used verbatim.
  - an **accessible + machine-readable layer** on every render: `role="img"`, a
    `<title>`, a one-line `<desc>` ("what it shows + key finding + severity"),
    and a `<metadata>` JSON block with the structured explanation.
  The finding's severity is derived from the same data the chart's badge uses,
  so the explanation can never contradict the badge (fixes cases like "0.12
  (MEDIUM)" shown next to a MINIMAL badge). Curated per-chart content covers all
  ~46 templates (`CHART_META`); finders are exhaustively defensive and never
  raise. Also fixed `FairnessExplainer._explain_bias_audit` to label the score
  with its own risk band rather than the worst finding's severity.
  (`tests/test_explain.py` covers coverage, defensiveness, a11y and severity
  consistency.)
- **Blanco is the one SVG design language** (`vfairness.rendering.skins`).
  `render_svg` — and therefore every adapter and `.to_svg()` — applies Blanco by
  default, so all rendered SVGs (reports, charts, gallery) are Blanco with no
  extra step. Blanco ("The Silent Gallery", per
  `docs/architecture/blanco-theme.md`) is a single post-render transform over
  the rendered string: zero border-radius (sharp corners everywhere), editorial
  type (Plus Jakarta Sans body, JetBrains Mono for data), desaturated
  graphite/slate chrome, and colour reserved for the four semantic tones (pass,
  warn, info, neutral, with all reds/oranges/ambers unified to one terracotta
  hue). It never touches the ~45 templates or ~15 adapters. Public API:
  `apply_skin(svg)`, `list_skins()`, `SKINS`. (`tests/test_skins.py` covers it.)
- `scripts/regenerate_gallery_examples.py` — regenerates every gallery example
  from live computation (the real analysers on synthetic datasets, mirroring
  `notebooks/vfairness_0_svg_rendering_demo.ipynb`) with a Blanco leak check.

- Golden-file regression tests pinning exact values for the core classification,
  regression, and ranking metrics and the `FairnessAnalyzer` report
  (`tests/test_golden_metrics.py`).
- Render-smoke tests that drive `render_svg` for every registered SVG template
  and assert well-formed XML output with no unrendered Jinja tokens
  (`tests/test_rendering.py`).
- `vfairness.result.TaskResult`, a stdlib dataclass that formalizes the task
  handler result envelope (`schema_version`, `task_type`, `success`,
  `data`/`error`, optional `warnings`) with a backward-compatible `to_dict()`.
- Coverage measurement in CI (`--cov=vfairness`) with a ratcheting
  `--cov-fail-under` gate and `[tool.coverage.*]` configuration.

### Changed
- The SVG gallery renders every example in Blanco; the Original/Blanco skin
  switch and the parallel `img/svg-gallery/blanco/` mirror are removed, and the
  colored category-icon boxes were dropped from the section headers. The
  unused design-proposal SVGs were deleted.
- CI now runs the FULL test suite. The `fairness` marker matched zero tests, so
  `pytest -m fairness` selected nothing; the `-m fairness` filter was removed
  from both `.github/workflows/fairness-checks.yml` and
  `.gitlab-ci-fairness.yml`, and both install the extras the suite exercises.
- The CI/CD and Pulse task handlers build their result via `TaskResult.to_dict()`.
  The `success` / `data` / `error` semantics are unchanged; the envelope gains
  additive `schema_version` and `task_type` keys.

### Documentation
- Added this changelog and `docs/API_STABILITY.md` declaring the frozen public
  API surface, SemVer intent, and the one-minor-version `DeprecationWarning`
  policy.

## [0.0.8]

> **Internal only. Never published.** See the note on 0.0.9 above.

First consolidated pre-release. Establishes the ten-stage fairness pipeline and
the public API. (Previously labelled 0.1.0; regraded down to 0.0.x, because the
library was still in alpha and had not been published, so a 0.1.0 tag was
premature.)

### Added
- **Preprocessing** (`vfairness.preprocessing`): feature-engineering bias
  detection and dataset auditing.
- **In-Processing** (`vfairness.in_processing`): training-time fairness
  constraints, regularizers, and losses.
- **Post-Processing** (`vfairness.post_processing`): calibration, threshold
  optimization, and reweighting interventions.
- **Evaluation** (`vfairness.evaluation`): the core fairness metrics for
  classification, regression, and ranking; the unified `FairnessAnalyzer`
  interface; the `MetricResult` structured output; statistical validation
  (confidence intervals, effect sizes) and explanations.
- **Operations** (`vfairness.operations`): CI/CD gates (`ModelFairnessGate`,
  `FairnessReportCard`), the pytest plugin, data-quality validation, monitoring,
  reporting, experimentation, and the Pulse orchestrator.
- **Rendering** (`vfairness.rendering`): Jinja2-based SVG report generation with
  a template registry and per-report adapters.
- **LLM Fairness** (`vfairness.llm`): counterfactual prompt testing, standardized
  benchmarks (BBQ, BOLD, DecodingTrust), output analysis, and noise-floor
  estimation.
- **Agent Fairness** (`vfairness.agents`): correspondence testing, tool-bias
  auditing, RAG-bias analysis, and pipeline tracking.
- **Multi-Agent Fairness** (`vfairness.multi_agent`): compositionality analysis,
  groupthink detection, and emergent-bias measurement.
- **XAI / Explainability** (`vfairness.xai`): SHAP-family adapters, fairness
  decomposition with an identity guard, DiCE counterfactuals, and
  faithfulness / stability / adversarial diagnostics.
- Optional Model Context Protocol server (`vfairness.mcp`) exposing the fairness
  battery as MCP tools.

<!--
Version headings are deliberately plain text, not links. The repository carries
no tags yet, so every `.../compare/vX...vY` and `.../releases/tag/vX` URL that
used to sit here answered 404, including the one on the version being released.
Restore the link definitions in the same change that pushes the first tag, and
check each URL resolves before doing so.
-->
