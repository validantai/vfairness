# Beta Go-Live Plan

How every registered capability in vfairness is graded for honesty before the
0.1.0 beta, what each grade does and does not establish, and what remains to be
done.

**Read the scope first.** This page covers the **capability census**: the 215
rows of the capability registry, which resolve to 222 of the 1,558 code units on
the public surface. That census is complete and carries no open defect. It is not
the library. The other 1,336 code units are the subject of a separate campaign
under a different method, and that campaign has graded 420 of them so far, so
most of the surface still carries no grade at all. **That is where this library's
95 open defects are** (measured 2026-09-18). The two counts are never added: they measure different
things and a total of them is a number neither produced. See
[GRADING.md](GRADING.md) for both, and [RELEASE_PLAN.md](RELEASE_PLAN.md) for
what is left.

**Nothing has been released.** There is no `v0.1.0` tag and no artifact on PyPI
or TestPyPI. This page is one of the gates in front of that tag, not the gate.
The machine-checked gate is `scripts/release_gate.py`, and on 2026-10-01 it
reports **BETA READY**, all eight beta conditions met, this census among them.
The stricter 1.0 gate is not met yet.

## The one defect this plan exists to remove

> A value nobody measured, replaced by a neutral default, then graded, counted,
> compared, ranked, plotted or reported as if it were a measurement.

A fairness library is read by people deciding whether a model may ship. When it
cannot compute something and says `0.0` instead of saying so, that `0.0` is the
best possible score for a disparity metric. It flows into a grader, a badge, a
chart and an executive report, and every one of them reads it as evidence of
fairness. The failure is silent by construction: the number looks exactly like a
number that was measured.

The fix is always the same and it is not subtle: **three states, never two.**
Measured, failed, could-not-check. A could-not-check may never be collapsed into
a pass, a zero, a `False`, or a neutral score.

The reverse also counts. A value that *was* measured but is reported as
could-not-check throws away real evidence while reading as caution. Both
directions are graded here.

## Summary

<!-- BGL:summary:start -->
| Batch | Status | Capabilities | Share | What it means |
| --- | --- | ---: | ---: | --- |
| **BGL-A** | PROVEN | 158 | 73% | Refuses honestly when nothing is measurable, and a test holds it there. |
| **BGL-B** | SEMI-PROVEN | 57 | 27% | Refuses honestly today. Nothing stops that regressing. |
| **BGL-C** | UNPROVEN | 0 | 0% | Never executed on degenerate input. Honesty unknown. |
| **BGL-D** | DEFECT OPEN | 0 | 0% | Proved to report a number nobody measured. Fix pending. |
| | **Total** | **215** | 100% | |

The census of 2026-09-11 (commit `8559dda`) reproduced **157 defects** at the public API across the capabilities in the census. **As of 2026-09-17, 0 remain open** across 0 capabilities. Estimated remediation **0 engineer-hours**. Every count on this page is the state at 2026-09-17, not at the census.

**Stage 1 closed on 2026-09-11**: 27 critical findings fixed across 25 capabilities, every one re-checked by an independent agent told to refute it (fix real, pin able to fail, no over-correction). The audit itself found 3 further defects, which are fixed too. The counts above already reflect that.

**Stage 2 closed on 2026-09-17**: 100 findings fixed, of which 86 held under an independent audit told to refute them, clearing 86 capabilities. Suite 9154 passed, 0 failed. The fixes the audit REJECTED are still counted as open defects above, including several since repaired by hand: an unaudited fix is a claim, and this page counts measurements. The number understates progress on purpose.
<!-- BGL:summary:end -->

<!-- BGL:scope:start -->
**Scope of the 215.** The table above grades the 215 rows of the
capability census, which is not the whole exported surface. Measured at
render time:

| Exported callables | Count | Graded |
| --- | ---: | --- |
| Carry a proof batch | 215 | yes, one of BGL-A/B/C/D above |
| Plain functions with no batch | 90 | **no grade at all** |
| Classes with no batch | 114 | no grade; mostly result containers, configs and exceptions |

Those **90 functions were never executed on degenerate input by this
census**. They are NOT BGL-C: BGL-C is a measured statement about a capability
that was in scope and not reached, whereas these were never in scope. Nothing
on this page says anything about whether they refuse honestly, and a reader
who took the 100% above as covering the library would have been misled.

Grading them is the next census wave, not part of this one.

<details><summary>The ungraded exported functions</summary>

- `analyze_calibration_fairness_tradeoff`
- `analyze_intersectional_correlations`
- `apply_multiple_testing_correction`
- `assert_fairness`
- `assign_clusters`
- `attribute_historical_pattern`
- `auto_log_fairness`
- `bayesian_difference_ci`
- `bayesian_mean_ci`
- `branding_enabled`
- `calibration_vs_error_parity`
- `check_fairness_config`
- `classify_column_roles`
- `cohens_d`
- `compare_to_benchmark`
- `compute_constraint_violation`
- `compute_disparity_effect_sizes`
- `compute_pareto_frontier`
- `compute_proxy_correlations`
- `compute_regression_effect_sizes`
- `compute_robust_metrics`
- `contingency_test`
- `create_calibration_dashboard`
- `create_factorial_design`
- `create_fairness_callback`
- `create_fairness_dashboard`
- `detect_specification_bias`
- `detect_temporal_drift`
- `domain_historical_context`
- `empirical_likelihood_ci`
- `explain_fairness_report`
- `exposure_parity_rerank`
- `fairness_test`
- `get_available_styles`
- `get_group_metrics_with_ci`
- `get_group_rankings`
- `get_palettes`
- `get_ranking_group_metrics`
- `group_reliability`
- `identify_feature_proxies`
- `identify_proxy_features`
- `impossibility_diagnostics`
- `interpret_effect_size`
- `log_fairness_to_mlflow`
- `log_fairness_to_wandb`
- `mae_parity_difference_with_ci`
- `make_fair_classifier`
- `make_fair_regressor`
- `mean_prediction_difference_with_ci`
- `mitigation_pareto`
- `mmd_gaussian`
- `odds_ratio`
- `parametrize_fairness`
- `permutation_test_demographic_parity`
- `permutation_test_equal_opportunity`
- `plot_calibration_comparison`
- `plot_calibration_disparity`
- `plot_confidence_intervals`
- `plot_effect_sizes`
- `plot_fairness_metrics`
- `plot_fairness_report`
- `plot_group_calibration`
- `plot_group_comparison`
- `plot_group_disparity_heatmap`
- `plot_metrics_radar`
- `plot_reliability_diagram`
- `plot_tradeoff_curve`
- `preview_palette`
- `print_explanations`
- `print_report`
- `propensity_weights`
- `rank_fairness_issues`
- `recommend_calibration_strategy`
- `regression_fairness_report`
- `reliability_tier`
- `require_checked`
- `risk_ratio`
- `rmse_parity_difference_with_ci`
- `robust_fairness_comparison`
- `run_disparity_tests`
- `save_fairness_plots`
- `select_method`
- `selection_rate_disparity_matrix`
- `sequential_fairness_test`
- `set_branding`
- `simulate_threshold_change`
- `simultaneous_disparity_bounds`
- `status`
- `stratified_bootstrap_ci`
- `test_equalized_odds_chi_square`

</details>
<!-- BGL:scope:end -->

**What has happened to that scope since.** The block above is generated from the
census evidence and describes the census as it stood: the 88 functions and 111
classes with no proof batch were out of its scope, not cleared by it. They are no
longer untouched. A separate surface-grading campaign has since graded **420 code
units** under a five-state method, and a mechanical probe (`scripts/surface_probe.py`)
has executed the rest on nine input worlds. Neither belongs to this census and
neither is added to its totals. As of 2026-09-18 that campaign holds 209 proven,
22 semi-proven, 94 not-a-measurement and **95 defect open**, with 10 grades not
yet independently checked. The 100% in the table above is 100% of the 215, and
nothing else.

This table is generated from `src/vfairness/_proof_status.py` by
`scripts/beta_go_live_docs.py`. It cannot drift from the ledger: CI runs
`--check` and fails if it does.

## The batches

Four batches, because there are four genuinely different things to say. Three
would have forced "we checked it and it is fine" and "we never checked it" into
one bucket, which is the exact defect this whole plan is about.

<!-- BGL:definitions:start -->
### BGL-A — PROVEN

**What it means.** Executed on healthy input and on the degenerate inputs relevant to what it claims to measure. In each of those it EITHER refused (returned NaN, None, an explicit could-not-check, or raised) OR returned a value an independent judge showed to be a true measurement for that input. Nothing it returned survived three passes as a fabrication, and at least one test in the suite names it next to a refusal assertion.

**A capability is in this batch when.** reached=True AND judged>=1 AND open_defects==0 AND at least one test file, present in the repository, that names the capability within twenty lines of a refusal assertion.

**What you may NOT conclude.** Three things, each of which stays open at 0.1.0. (1) That the NUMBER it returns on healthy data is statistically correct: this ladder measures honesty about the ABSENCE of a measurement, not the accuracy of one. (2) That the pin covers every degenerate scenario: the pin test is a proximity match, so it proves a refusal is asserted somewhere near this capability, not that THIS scenario is the one asserted. (3) That the pin has been sabotage-checked. A guard that cannot fail looks identical to one that passed, and only a reinstated defect tells them apart. Promoting BGL-A to a sabotage-verified rung is the exit criterion of Stage 4 in docs/BETA_GO_LIVE_PLAN.md.

### BGL-B — SEMI-PROVEN

**What it means.** Executed on healthy and degenerate input and judged clean today, but nothing in the suite holds it there. The behaviour is observed, not protected: the next refactor can reintroduce the defect and every gate will stay green.

**A capability is in this batch when.** reached (generically, or by a hand-written fixture) AND judged>=1 AND open_defects==0 AND no test file names it near a refusal assertion.

**What you may NOT conclude.** That it will still be honest after the next change. A BGL-B capability is one commit away from BGL-D with no alarm.

### BGL-C — UNPROVEN

**What it means.** No execution evidence on degenerate input. Either no fixture could be constructed generically, or it needs a collaborator (a live LLM proxy, a fitted pipeline, a torch model) the harness does not have. Its honesty is UNKNOWN, which is not the same as suspect and is emphatically not the same as clean.

**A capability is in this batch when.** no adjudicated execution on degenerate input.

**What you may NOT conclude.** Anything at all about whether it fabricates. This batch exists so that 'not checked' can never be read as 'checked and fine'.

### BGL-D — DEFECT OPEN

**What it means.** At least one entry point was proved, by three independent passes ending in a re-run at the public API, to hand back a confident value where nothing was measurable. The capability still computes correctly on healthy data; what is broken is what it says when it cannot.

**A capability is in this batch when.** open_defects>=1, each reproduced at the public entry point.

**What you may NOT conclude.** That the capability is unusable. It means its could-not-check path is wrong, and a caller who hits that path is told a number that was never measured.

<!-- BGL:definitions:end -->

### Where the batch is written down

The same grade appears in four places, and they are mechanically kept in step:

| Surface | What it carries | Kept honest by |
| --- | --- | --- |
| The function's own docstring | the batch, its limits, and the ledger row | `tests/test_beta_go_live_proof_ledger.py` |
| `vfairness._proof_status` | the full record per capability | generated from the evidence file |
| `vfairness-manifest.json` | `proof_status` + `stats.proof` | `_manifest.generate_manifest()` |
| This page | the tables above | `scripts/beta_go_live_docs.py --check` |

A reader who types `help(exposure_parity_ratio)` sees the grade without knowing
this document exists. That was deliberate: a correct measurement no reader can
see is the same defect one layer up.

## How the 215 were measured

Four passes, each able to overrule the one before it. This is the census method,
applied to the capability registry on 2026-09-11. It is not the method used on
the wider surface, which grades code units in five states and is described in
[GRADING.md](GRADING.md).

**1. Probe (mechanical, complete).** Every registered capability was resolved and
executed on HEALTHY input carrying a real, findable group disparity, and then on
each degenerate input where the thing it claims to measure does not exist:

| Scenario | Why nothing is measurable |
| --- | --- |
| `single_group` | one group only, so no between-group comparison exists |
| `one_label_only` | every `y_true` is 1, so TPR, FPR and AUC are undefined |
| `all_nan_scores` | every score is NaN: nothing was scored |
| `empty` | zero rows |
| `constant_scores` | every score identical, so no ranking and no AUC exists |
| `n_equals_2` | two rows, below any statistical power |
| `one_row_minority` | one group has a single row |
| `unreadable_text` | text containing no lexicon word at all |

506 entry points were reached this way at the census of 2026-09-11. The healthy run is the control: a
capability that cannot measure healthy data either is a different finding, not
this one.

**2. Judge (independent).** Every row that answered confidently on degenerate
input went to an auditor who read the source and ruled FABRICATES, CORRECT,
PROBE_ARTEFACT or OTHER_DEFECT. CORRECT required naming the true value and why it
is right for that input, so that `accuracy_parity_difference` returning 0.0 when
every label is 1 is recorded as the true measurement it is.

**3. Adversarial verify.** Every claimed defect went to a second auditor told to
**refute** it, defaulting to refuted when uncertain, and required to paste real
execution output before confirming. 31 of 189 claims were refuted here and
dropped.

**4. Reproduce at the public entry.** A third pass re-ran every surviving claim
at the API a user actually calls and recorded the real returned value. This pass
exists because of a concrete miss: a claim that `net_benefit_parity` returned
`0.0` for unscored data looked false when checked with 20 rows per group, because
the `nan` came from the `min_group_size=30` gate and not from the NaN path at
all. At 60 rows per group it returned `0.0`, identical to a real, equal, useful
model, and the library's own `check_threshold` graded that **PASS**. That defect
is fixed; the pass that caught it is kept, because a fixture can supply the
refusal through a parameter nobody was thinking about.

Only findings that survived all four passes are recorded as BGL-D.

**The evidence is committed**, not summarised: see
[docs/beta-go-live-census-2026-09-11.json](beta-go-live-census-2026-09-11.json).
`tests/test_beta_go_live_proof_ledger.py` refuses any ledger row that disagrees
with it, so a number cannot be improved by editing the ledger.

## Where the work is

<!-- BGL:packages:start -->
| Package | A proven | B semi | C unproven | D defect | Total | Fix (h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents` | 5 | 1 | 0 | 0 | 6 | 0.0 |
| `evaluation` | 46 | 11 | 0 | 0 | 57 | 0.0 |
| `explainer` | 1 | 0 | 0 | 0 | 1 | 0.0 |
| `in_processing` | 20 | 15 | 0 | 0 | 35 | 0.0 |
| `llm` | 11 | 0 | 0 | 0 | 11 | 0.0 |
| `multi_agent` | 6 | 1 | 0 | 0 | 7 | 0.0 |
| `operations` | 21 | 3 | 0 | 0 | 24 | 0.0 |
| `post_processing` | 29 | 7 | 0 | 0 | 36 | 0.0 |
| `preprocessing` | 11 | 13 | 0 | 0 | 24 | 0.0 |
| `vision` | 4 | 0 | 0 | 0 | 4 | 0.0 |
| `xai` | 4 | 6 | 0 | 0 | 10 | 0.0 |
| **all** | **158** | **57** | **0** | **0** | **215** | **0** |
<!-- BGL:packages:end -->

Every fix column reads 0 because every census finding is closed. When the census
was taken the concentration was neither even nor random: `post_processing` and
`in_processing` carried the most open defects, because they are full of estimators
that fit a model and then answer questions about it, and an unfitted or unfittable
estimator has an enormous number of ways to return a plausible number.
`evaluation` has the most capabilities and the most proven ones, because it is
where the earlier could-not-check campaign concentrated.

### Severity of the 157 census findings, all since closed

The table below is the census of 2026-09-11 sized per finding. It is the record
of what was found, not a list of what is open: as of 2026-09-17 none of these is
open and the remediation total on the summary table is 0 hours. It is kept
because deleting the finding would delete the evidence.

<!-- BGL:severity:start -->
| Severity | Findings | Fix (h) | What it means |
| --- | ---: | ---: | --- |
| critical | 27 | 15.8 | reaches a graded, rendered or exported surface a reader treats as a verdict |
| high | 104 | 49.5 | fabricates a number a caller consumes, but not directly onto a badge |
| medium | 23 | 8.9 | fabricates, with another field nearby a careful reader could notice |
| low | 3 | 1.3 | cosmetic, or reachable only on a path the library discourages |
| **all** | **157** | **75.5** | |
<!-- BGL:severity:end -->

Three that show the shape, all reproduced at the public API on 2026-09-11 and all
since fixed. The behaviour described is what the code did at commit `8559dda`,
not what it does now; the reproductions are kept because they are the evidence
the finding rests on:

- `exposure_parity_ratio` with an all-NaN score column returns `0.5410755479122771`,
  **bit-for-bit identical** to the genuinely ordered healthy ranking, because the
  positions fall back to the caller's row order. That value is below the
  four-fifths threshold, so `check_threshold` returns FAIL and the SVG renders a
  red FAIL badge. Interleaving the same rows returns `0.8187`, which passes. A
  regulator-facing verdict derived from nothing but the order the rows arrived in.
- `ExponentiatedGradient.fit` on a single-valued sensitive attribute returns
  `final_violation=0.0`, `constraint_satisfied=True`, `converged=True`, with no
  warning: a compliance certificate for a run in which no pair of groups existed
  to compare. The library's own `_defined_spread` docstring states the rule being
  broken, and its three sibling constraints all refuse correctly on that input.
- `ReportGenerator.generate_executive_report` prints `## Health Score: 100/100
  (GREEN)` with a `Drift Stability | 100.0` row for a store in which **no drift
  test was ever run**. Zero drift records is the default state of every
  `MetricsStore`.

### The unproven capabilities

<!-- BGL:unproven:start -->
| Capability | Symbol | Module | Why no fixture |
| --- | --- | --- | --- |
<!-- BGL:unproven:end -->

The table is empty: there are no BGL-C capabilities today, which is what B5 on the
release gate reports. The definition is kept because it is what the empty table
means. A capability lands here when no generic fixture reached it with data, not
because anything suspicious was seen. Unknown is not clean, and it is not suspect
either.

## The plan

Five stages. Each has an exit criterion a machine checks, because a stage whose
exit is a judgement call is a stage that ends when someone is tired of it.

### Stage 1 — the critical findings — **DONE, 2026-09-11**
**Exit was:** every critical finding has a fix, a test, and a recorded sabotage
that went RED with the fix reverted; `proof_summary()["BGL-D"]` drops by the
number of capabilities cleared, and no capability moves to BGL-A without a pin.
**Met.** 27 findings closed across 25 capabilities, each re-checked by an
independent agent told to refute it: fix real, pin able to fail, no
over-correction, on all 27. Nothing that reaches a graded or rendered surface is
still known to fabricate.

The audit pass was worth more than the fixes. It found three further defects the
fix agents had missed, all now closed:

- **`data_bias_validation` deleted a real finding.** Routing "fewer than two
  defined rates" into a could-not-check `continue` swallowed the CRITICAL
  `zero_outcome_rate` issue the pre-fix code raised when one group had a measured
  rate of 0.0. Removing a fabricated disparity must never take a real finding
  with it: "no member of this group received a single positive outcome" is a fact
  about the rows that WERE read, and it needs no comparison group.
- **A second critical in `conditional_demographic_disparity`**, invisible until
  the first fix removed the number hiding it. A stratum that lost a valid group's
  cell was still scored on whichever groups survived, at full population weight.
  Measured: a group approving at 100% against 50%, valid overall but holding two
  rows in one stratum, took the metric from 0.5000 to 0.2804. The group most
  likely to lose a cell is the smallest one, which is the group a fairness audit
  exists to protect, so the error ran in the unsafe direction by construction.
- **The DecodingTrust stereotype lexicon could not read a refusal.** Removing the
  fabricated 0.3 exposed it: `"I disagree"`, `"I do not agree"` and `"Disagree"`
  all came back unscorable while `"Yes, absolutely"` scored a clean 1.0. Worse,
  both lexicons match as substrings, so `"That is not true"` scored **0.5** and
  `"No, that is not correct."` scored **1.0, full agreement with the
  stereotype** - a plain refusal published as maximal agreement on a
  trustworthiness number. The negation rule is now derived from the agreement
  lexicon itself, so a word added there is negation-aware for free.

Several share one root, so the order matters: the three ranking metrics
(`exposure_parity_ratio`, `exposure_parity_difference`,
`normalized_discounted_kl_divergence`, plus `attention_weighted_rank_fairness`)
all fail in `_rankings_to_positions`, and four constraint capabilities all fail at
`base.py:446`. Fixing the root is one change; each caller still needs its own
refusal path and its own sabotaged test, which is where the hours go.

### Stage 2 — the high findings. **DONE, 2026-09-17**
**Exit was:** same bar, applied to every high finding. **Estimated at 50 h.**
**Met.** 100 findings closed, of which 86 held under an independent audit told to
refute them, clearing 86 capabilities; suite 9154 passed, 0 failed. Fixes the
audit rejected stayed recorded as open defects until they were reworked and
re-audited, which is why the count moved later than the fixes did.

### Stage 3 — the medium and low findings. **DONE, 2026-09-17**
**Exit was:** same bar. **Estimated at 10 h.**
**Met.** The summary table at the top of this page is generated from the evidence
file and reports 0 open defects across all 215 capabilities and 0 remaining
remediation hours, which is the exit condition for stages 1 to 3 taken together.
One item, `integrated_gradients`, was repaired by hand and sabotage-verified
before an independent audit held it; it stayed counted as open until that audit
ran, on the rule that an unaudited fix is a claim.

### Stage 4 — promote BGL-A to sabotage-verified. **OPEN**
**Exit:** every BGL-A pin has a recorded sabotage that went RED. Today BGL-A
requires a pin that *exists* and *names* the capability beside a refusal
assertion; it does not prove the pin would fail if the defect came back. A guard
that cannot fail looks identical to one that passed, and this repository has
shipped four of those in a single day. **Estimate: 20 h**, and it is the stage
most likely to find new defects rather than confirm old ones. It covers all 158
capabilities now in BGL-A, so the estimate predates that count and is the least
certain figure on this page.

### Stage 5 — close BGL-C. **Exit condition met, 2026-09-17**
**Exit:** zero capabilities in BGL-C, each one moved by a hand-written fixture
that reaches it with data. **Estimated at 4 h.** The generated table reports
BGL-C at 0.

**Where the hours went.** The original total was about 100 engineer-hours: stages
1 to 3 as the remediation total the census sized per finding, plus roughly 24 for
stages 4 and 5. It was a real-hours figure built from the bottom up rather than a
guess at a round number: every estimate was made by the auditor who actually
reproduced that defect, and each covers the fix plus a test that has been
sabotage-checked. Stages 1, 2, 3 and the exit condition of stage 5 are met, and
the generated remediation total is now 0 hours. What remains on this page is stage
4. The estimates above are left in place as the record of what the work was
thought to cost; the stage figures move as findings close, because the table they
derive from is generated.

**This is not the remaining work on the library.** It is the remaining work on
the capability census. The wider surface carries 95 open defects and 10 unaudited
grades, and the plan for those is costed in tokens rather than hours in
[RELEASE_PLAN.md](RELEASE_PLAN.md): five phases, 15.6 to 20.4 million.

### What beta may claim at each point

Every sentence in this table is about **the 215 registered capabilities**. None of
them is a claim about the library as a whole, and reading one that way is the
error this page is written to prevent.

| After | vfairness may honestly say, of the 215 capabilities |
| --- | --- |
| the census | "every capability is graded, and the ones that fabricate are named" |
| **Stage 1 (done)** | **"nothing that reaches a graded or rendered surface is known to fabricate"** |
| **Stage 2+3 (done)** | **"no capability is known to report a number it did not measure"** |
| Stage 4 | "and every one of those refusals is held by a test proven able to fail" |
| **Stage 5 (exit met)** | **"and there is no capability we have not checked"** |

What none of those lines said on 2026-09-11, and what had to be said beside them:
95 code units outside the census were proved to report a value nobody measured,
86 of them in code that returns a fairness number or a verdict. All of them were
fixed by 2026-09-30, and on 2026-10-01 the release gate, which holds the two
together, reads BETA READY.

Whether to publish at all is a product decision, not a technical one, and the
gate is the input to it rather than the answer. The library is **useful today**:
a capability or code unit graded DEFECT OPEN still computes correctly on healthy
data. What is wrong is what it says when it cannot compute, and the honest
description of that is now attached to every graded capability rather than to a
footnote.

## Re-running any of this

```bash
python scripts/beta_go_live_ledger.py            # regenerate the ledger
python scripts/beta_go_live_ledger.py --check    # CI: is the ledger current
python scripts/stamp_proof_status.py             # re-stamp docstrings
python scripts/stamp_proof_status.py --check     # CI: is any stamp stale
python scripts/beta_go_live_docs.py              # regenerate the tables here
python scripts/beta_go_live_docs.py --check      # CI: are the tables stale
python -m pytest tests/test_beta_go_live_proof_ledger.py -q
python scripts/library_kpis.py                   # the wider-surface counts
python scripts/release_gate.py                   # the readiness verdict
```

The ledger is derived from the evidence file and the evidence file is derived
from execution. Nothing in that chain is typed by hand, which is why the
**generated tables** on this page can be trusted after the date at the top of it.
The prose between them is typed by hand and can go stale: the surface-grading and
release-gate figures quoted here are from `library_kpis.py` and
`release_gate.py` on 2026-09-18, and the fix for a number that has moved is to
re-run those two, not to trust this page.
