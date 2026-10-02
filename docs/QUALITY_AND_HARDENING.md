# Quality and Hardening

## Where we are

<!-- BGL:readiness:start -->
**READY TO CUT A BETA.** 8 of 8 beta criteria pass; 0 block it. _Generated 2026-10-02 from `scripts/release_gate.py`._

| | Criterion | Measured now | Closes when |
| --- | --- | --- | --- |
| PASS | `B1` Every public code unit has been EXECUTED and its result recorded | 1580 of 1580 executed and recorded |  |
| PASS | `B1b` Every public code unit's examination state is PUBLISHED where it is described | 1580 of 1580 carry a published state |  |
| PASS | `B2` No open defect in code that returns a fairness number or verdict | 0 open (0 measuring, 0 unclassified) |  |
| PASS | `B2b` No open defect recorded by the second-round audit | 0 open of 59 recorded claims |  |
| PASS | `B3` Every remaining open defect carries an assessed severity | 0 non-measuring open, severity unassessed |  |
| PASS | `B5` Capability census complete and clear | see G5 |  |
| PASS | `B6` Same answer as an established library, where one exists | 34 of 34 agree; 76 have no outside reference |  |
| PASS | `B7` No invented number on broken data | 109 of 109 measuring capabilities pass; 0 fabricate, 0 over-refuse, 0 not covered |  |

<!-- BGL:readiness:end -->

How vfairness is tested, audited, secured, and hardened for its first public beta.

Two kinds of work go into that.

**Most of it is ordinary testing**, and there is a lot of it: over ten thousand
tests across the metric, mitigation, LLM, XAI, monitoring, Pulse and rendering
paths. They check that the library does what it says.

**The rest is a hunt for one particular bug**, and it is the reason this page is
long. The bug happens when the library cannot work something out, quietly puts a
harmless-looking number in its place, and then presents that number as a result.
A `0.0` that reads as "no bias found", when what really happened is that nothing
could be measured at all.

Around those two sit a static type-checking burndown, a supply-chain and security
posture built into CI, and packaging and release-integrity checks. This page
inventories every measure, and each carries its internal issue id (`VB-*`) so it
maps back to the changelog. Both kinds of work are reported here with equal
weight, and so is everything still open.

The beta exit criteria are computed by `scripts/release_gate.py`. [BETA.md](BETA.md)
reports that gate rather than defining the bar. This page
is the detailed companion to both, and it is an inventory of what was built, not
a readiness verdict.

The guiding principle is honesty over optimism: the library should never
silently pass, silently drop a small group, or report a number it did not
actually compute.

## How a release happens

This page is the one place the whole picture lives: where the library stands, what
is still open, and how a version reaches a user. The step-by-step a maintainer
follows is in [Release Pipeline](RELEASE_PIPELINE.md); everything a reader needs to
judge the release is here.

**Nothing is uploaded from a laptop.** The only upload path is
`.github/workflows/release.yml`, triggered by pushing a `vX.Y.Z` tag, and it
publishes to PyPI only after a required human approval on the `pypi` environment.
There is no TestPyPI rehearsal: it was removed on 2026-08-28, so the gates below are
the only things standing between a tag and an immutable upload.

**Releases are cut only from the public repository `validantai/vfairness`**, and that
is enforced rather than documented. Every publishing job requires all three of a
pushed ref, a `v`-prefixed tag and that exact repository, so the copy of the workflow
in the private monorepo is structurally incapable of publishing. The event clause is
not redundant with the tag clause: a manual run can be aimed at a tag ref, and on the
ref clause alone it would reach the upload.

**Before the tag: the readiness gate.** `python scripts/release_gate.py` exits
non-zero while anything is open, and the table at the top of this page is generated
from it. It is not the test suite, and a failing criterion is not a failing test: the
suite passes. The gate measures how much of the surface has been adversarially
audited and how much nobody has examined. A release cut while it is red is a
decision, and it has to be a stated one.

**At release time, on the exact artifact:** the tag-versus-`__version__` assert,
`twine check`, `check-wheel-contents`, a CycloneDX SBOM of the built wheel and its
scope check, and a clean-room install smoke test. Build provenance attestations are
attached to the upload, requested explicitly on the publish step rather than left to
a default. The SBOM is published on the GitHub Release beside the wheel, so a
consumer can fetch it. Trusted Publishing (OIDC) means no long-lived PyPI token
exists to leak. `workflow_dispatch` is the dry run: it builds and smokes and cannot
publish, because both publishing jobs require a pushed tag.

**On every change, in the public repository:** `tests.yml` runs the full suite on
Python 3.11, 3.12 and 3.13 with the optional extras installed, branch coverage behind
a floor, and `VFAIRNESS_REQUIRE_BACKENDS=1` so a missing backend fails by name
instead of skipping silently. `quality.yml` runs ruff and mypy blocking, plus
`diff-cover` at 80% on the lines a pull request changes. `security.yml` runs Bandit,
`pip-audit --strict` and CodeQL, which genuinely runs there because the repository is
public. `fairness-checks.yml` runs the suite with every extra and then exercises the
library's own `ModelFairnessGate` on synthetic data, which demonstrates the gate and
is not an assessment of the library; its second job resolves the lowest declared
dependency versions so a wrong floor fails there rather than on a user's machine.
`secret-scan.yml` runs gitleaks over the full history from a pinned binary, and
carries a positive control: it plants a secret the ruleset may not allowlist and fails
if gitleaks misses it, because a clean result from a broken scanner looks exactly like
a clean result from a working one. Verified 2026-09-28 against the exported subtree's
`.github/workflows/`, which carries exactly those five plus `release.yml`.

One limit worth stating: whether any of them is a *required* status check is a
repository setting, not something a workflow file can assert, so this page does not
claim they block a merge.

## Why hardening this library costs more than building it

A Python function has to return something. There is no natural way to say "I do not
know" where a number is expected, `0.0` is the easiest thing to type, and it stops the
program crashing. Every neutral value this programme has removed was written by
somebody trying to be helpful.

![A cutaway technical plate of a ceiling smoke alarm. Smoke rises through the louvres
and into the sensing chamber, the alarm horn is silent, and the only saturated element
in the drawing is the pair of empty battery contacts. A small figure stands below,
unaware.](site/img/silent-failure-smoke-detector.png)

*Smoke in the chamber, a horn that will not sound, and nothing about the outside of the
device to tell you which of the two you have.*

**Think of a smoke detector.** Building one that beeps when you press the test button is
easy. The hard part is proving it beeps when there is real smoke, and that it does not
sit there silently when its battery is flat.

A fairness tool has the same shape. When it cannot measure bias and returns `0.0`, that
reads as "no bias found". So:

- the failure is **silent**: nothing crashes, and no error appears
- the failure always points the **reassuring** way
- **nobody reports it**, because a customer told "your AI is fine" is a happy customer

That combination is why these defects survived so long. A bug that produces a wrong
chart is reported in a day. A bug that produces a clean bill of health is never reported
at all.

### One function is one step to write and five to check

Writing a function is one step: make it work.

Checking one honestly is five:

1. **Build the situation where it cannot measure.** Often harder than the function
   itself, because it needs a fitted model, a report object, or a populated store.
2. **Run it** and look at what actually comes back.
3. **Judge whether that answer is honest.** This is the expensive step, and a machine
   cannot do it.
4. **Write a test** that locks the honest behaviour in.
5. **Deliberately break the code** to prove the test notices. A test that cannot fail
   looks exactly like a test that passed.

Multiplied across the public surface, that ratio is the whole of this page.

**Nothing has been released yet.** There is no `v0.1.0` tag, the repository
carries no tags at all, and no artifact has reached PyPI or TestPyPI. The
release pipeline below is built and gated but has never run to a publish.
`CHANGELOG.md` heads its section `[0.1.0] - UNRELEASED` for that reason.

**And it is not ready to be.** The verdict below is generated on every docs build
rather than typed, because this is the paragraph a reader consults to decide whether
to publish, and it is the worst place in the repository for a stale number. The
previous version said "Run on 2026-09-25 ... four of the six beta criteria pass ...
B2, open defects: 63 ... B4: 215 grades". Within two days B2 had reached 0 and gone
back up on an audit, B4 had passed 500 and come back down, and B1 had failed and
passed again.

<!-- BGL:gate:start -->
_Generated 2026-10-02 by `scripts/release_gate.py`._

**Beta bar: READY.** 8 of 8 criteria pass.

| | Criterion | Measured | Verdict |
| --- | --- | --- | --- |
| B1 | Every public code unit has been EXECUTED and its result recorded | 1580 of 1580 executed and recorded | passes |
| B1b | Every public code unit's examination state is PUBLISHED where it is described | 1580 of 1580 carry a published state | passes |
| B2 | No open defect in code that returns a fairness number or verdict | 0 open (0 measuring, 0 unclassified) | passes |
| B2b | No open defect recorded by the second-round audit | 0 open of 59 recorded claims | passes |
| B3 | Every remaining open defect carries an assessed severity | 0 non-measuring open, severity unassessed | passes |
| B5 | Capability census complete and clear | see G5 | passes |
| B6 | Same answer as an established library, where one exists | 34 of 34 agree; 76 have no outside reference | passes |
| B7 | No invented number on broken data | 109 of 109 measuring capabilities pass; 0 fabricate, 0 over-refuse, 0 not covered | passes |

**Full release bar: NOT READY.** 4 of 5 criteria pass.

| | Criterion | Measured | Verdict |
| --- | --- | --- | --- |
| G1 | Every public code unit carries a grade | 1580 of 1580 | passes |
| G2 | Every grade was confirmed by an independent check | 618 unaudited | **fails** |
| G3 | No open defect anywhere in the public surface | 0 open | passes |
| G4 | Every open defect has an assessed severity | 0 unassessed | passes |
| G5 | Capability census complete and clear | 0 open, 0 unproven | passes |

Reproduce with `python scripts/release_gate.py`. Every figure above is read from the recorded evidence, so a criterion cannot pass by assertion.
<!-- BGL:gate:end -->

## The second-round audit

<!-- BGL:second_round:start -->
The BGL5 fix wave was audited a **second time**, and that audit recorded **59 claims**. **59 are closed** and **0 are still open** (100% closed, as of 2026-10-02).

A claim is CLOSED when its defect is fixed and its test has been inverted into a pin, in the same commit. OPEN means the test still records the defective value the unit produces today, with the measurement in its docstring, so every open row below is reproducible rather than suspected.

| Batch | Subject | Recorded | Closed | Open |
| --- | --- | --- | --- | --- |
| `f01` | refusal scoring and the explainer's root predicate | 4 | 4 | 0 |
| `f02` | mitigation (label massager, residual transformer, suppressor, resampler, reweighting) and the DecodingTrust runners | 12 | 12 | 0 |
| `f03` | the CI/CD release gate and the fairness monitor | 8 | 8 | 0 |
| `f04` | XAI adversarial probes, threshold analysis and calibration | 13 | 13 | 0 |
| `f06` | compliance mapping, the signed test log and the report canvas | 11 | 11 | 0 |
| `f12` | data balancing, benchmark comparison, LIME and the task worker | 11 | 11 | 0 |
| **total** | | **59** | **59** | **0** |

The register is `docs/bgl6-audit-register.json`, generated by `scripts/bgl6_register.py` from the audit files, so this table cannot drift from them: each file declares which of its tests still record an open defect, and a name that is not a test in that file is refused.
<!-- BGL:second_round:end -->

**How to read B2 and B4 in particular.** B2 counts open defects in code that returns
a fairness number, a result, or a value nobody has classified. A defect counted there
is not necessarily a unit anybody has watched fabricate: the count also holds evidence
gaps, fixes an independent audit declined to certify, and units where the pins do not
cover enough input to call them proven. Those are different claims and the register in
`docs/audits/` says which each one is.

B4 counts grades that no independent examiner has attacked. **It rises when the
library gets better**, and that is the honest direction rather than a regression:
closing a defect produces a fix nobody has argued with, and reaching a unit for the
first time produces a grade nobody has argued with. The gate reports that instead of
netting it off against the progress, because a claim nobody checked is not a
measurement. The measured reason it matters is in the grading block further down: a
large minority of the grades that HAVE been attacked did not survive it.

**Every public code unit has been executed and its result recorded**, all 1,569 of
them, which is criterion B1. That total is the library's own size and it grows, so
the claim is re-measured rather than restated: reproduce it with
`python scripts/release_gate.py`, which prints the count it compared.

**How a unit counts as examined.** The gate takes the union of four sources, the
capability census, the grading waves, the three probes and the test suite, and names
any unit missing from it. It is a subset test, not a count: the sources are keyed by
qualified name and some of those names are re-export spellings the surface walk does
not count as units, so the union overshoots the total and a count comparison could
report PASS with real units unexamined.

**The evidence fails closed on the suite's own result.** The coverage record carries
the run's pass and fail counts and contributes nothing from a run that was not green,
because lines executed on the way to a failing assertion verified nothing.

**Line coverage cannot see every kind of unit, so the gap is measured.** A class body
runs once when its module is imported, before `pytest-cov` starts measuring, so no
assertion can put an enum's lines into a coverage report. That split is measured
separately, in a subprocess that starts coverage before the library is imported, and
recorded in `docs/suite-coverage.json` under `untouched_split`. It is deliberately
**not** fed to B1: running a class body is not recording a result. Of the units
coverage calls untouched, 132 as measured on 2026-09-25 and falling as the suite
grows, 43 execute nowhere and 89 are class bodies. The split lands exactly on the
class/function boundary, which is what makes it credible.

**The evidence behind each badge is checked, not taken on trust.** Every graded row
names a test file, and `scripts/verify_grade_attribution.py` runs each named file
under coverage and looks for the unit's body lines, because a row naming a test that
never executes the unit is a claim with nothing behind it. Across 415 graded rows it
confirms 348 by line coverage and reports 40 as blind, each with its reason: class
bodies, and bodies excluded by configuration. It reports zero rows whose named test
does not reach them.

**B1b is new, and it passes.** The beta tier's own definition has four clauses:
"nothing on the public surface is UNEXAMINED, nothing that can hand a user a false
clean bill is left open, everything still open is disclosed with a severity, and
**the documentation says which is which**". The fourth had no criterion until
2026-09-25, because until then there was no way to record a result for a unit
nobody had run. There is now, and all 1,569 units carry one, with guards that fail
the build if any published surface disagrees with the ledger. **The bar was not
lowered to achieve this**: B1 keeps its strict verdict and still fails.

Everything below describes measures that exist; this section is what the
measurement says about them.

**Read the unaudited count the right way: it went UP, from 39 to 112, on a day
when the library got better**, and it has gone up and down several times since, which
is why the live figure is generated rather than restated here. Every defect closed
adds a grade nobody has argued with yet, and every unit the probes reached for the
first time adds another. More evidence means more claims owing an independent check.
The gate reports that instead of netting it off against the progress, because a claim
nobody checked is not a measurement.

## Can I use this capability? One question, answered for all 1,569

Everything else on this page is about what was BUILT to harden the library.
This section is the part a user acts on, and it is the only place that answers
the question per capability rather than per programme.

Every public function, class and method carries one of three states, and the
badge appears wherever the capability is described: in the API reference, in the
markdown API reference, in its own docstring, and in the full grid at
[/status/](https://vfairness.validant.ai/status/).

<!-- STATUS:summary:start -->
| State | Whole public surface | Share | Core |
| --- | ---: | ---: | ---: |
| **CHECKED** | 1580 | 100.0% | 1398 |
| **FIX PENDING** | 0 | 0.0% | 0 |
| **NOT CHECKED** | 0 | 0.0% | 0 |
| | **1580** | 100% | **1398** |

Assembled by `scripts/capability_status.py` from evidence dated 2026-09-11, 2026-09-18, 2026-09-25, 2026-09-27, 2026-09-29, 2026-09-30, 2026-10-01; the newest rows are 2026-10-01. **Core** is what the beta promises: every area this project publishes a guide or a product page for, plus the top-level namespace, which is Agent fairness testing, Bias detection, CI/CD gates, Explainability (XAI), Fairness metrics, In-processing mitigation, LLM fairness testing, Monitoring, Post-processing mitigation, Preprocessing mitigation, Report rendering, Reporting and Top level (1398 code units). The rest is preview: documented and usable, and not covered by the beta bar.
<!-- STATUS:summary:end -->

- **CHECKED** means graded and assessed: it was executed on healthy input and on
  input where the thing it measures does not exist, and if it was found inventing
  a value that was fixed. It is **not** a correctness certificate for the number
  it returns. Each row states its strength: held by a test, verified once today
  with no test, or producing no verdict that could be fabricated.
- **FIX PENDING** means graded and assessed, and a defect is still open. Do not
  rely on it in the beta. Each row says which of three kinds it is, and those
  cost very different things to close: a fabrication still open, a fix an
  independent audit refused, or evidence too narrow to call it proven.
- **NOT CHECKED** means nobody has established anything either way. It is the
  absence of a measurement, not a failed one, and each row says why: never
  executed, unreachable without a hand-built fixture, called in a way that told
  us nothing, or executed with the evidence still unjudged.

## What the beta covers, and what it does not

**The rule is what we told you to use.** If a page on this site walks a reader
into a capability, that capability is part of what the beta promises and has to be
examined before the beta calls itself ready. It is not a judgement about which
parts matter most; it is a reading of what has been advertised. The rule cuts both
ways: anything we are not prepared to examine has to stop being presented as ready.

<!-- BGL:scope_table:start -->
| Scope | Units | Checked | Fix pending | Not checked |
| --- | ---: | ---: | ---: | ---: |
| **Core**, what the beta promises | 1,398 | 1,398 | 0 | **0** |
| **Preview**, documented but not certified | 182 | 182 | 0 | **0** |
| _total_ | 1,580 | 1,580 | 0 | 0 |
<!-- BGL:scope_table:end -->

**In core:** the assessment path itself (fairness metrics, bias detection,
explainability); everything that hands a user a verdict (report rendering,
reporting, CI/CD gates, monitoring); every surface with a product page of its own
(LLM testing, agent testing, and the pre-processing, in-processing and
post-processing mitigation families); and the top-level namespace.

**In preview, each for a stated reason:** the validity axis, deliberately
unfinished until its reference set exists; the MCP tools, reached through another
program rather than by writing vfairness code; the experimentation, causal and
pulse operations; and the vision, networking and legal helpers, which are plumbing
with no user-facing entry point. Preview means documented and usable, and not
covered by the beta bar. It does not mean broken.

**What it takes to ship: 498 core units are not yet clean**, 442 unexamined and 56
carrying a known open defect. That is the beta's remaining work and the honest
size of it. Every one is listed by name in the grid filtered to core.

This boundary used to be three areas. It was drawn by functional area rather than
by what reaches a reader, and two things fell through it: report rendering, which
draws the verdict a user looks at and is the worst-covered area in the library,
and CI/CD gates, which write a pass or a fail into a pipeline. Both are advertised
on the getting-started and sample-assessment pages, and neither was core.

**Inside core sits the assessment path itself: fairness metrics, bias detection
and explainability, 390 units.** Of those, 251 have been examined
and are clean, 0 carry an open defect, and 139 have not been examined at all.

**Read those two numbers together.** Zero open defects in the core does not mean
the core is verified. It means that of the units somebody examined, none was
found reporting a value it had not measured. The 139 nobody examined are not
evidence in either direction, and folding them into the clean figure would be the
exact defect this library is audited for, committed about its own core. Each one
says so in the grid, and the core path can be listed on its own at
[status/?area=core](site/status/index.html).

That zero is the bar this beta was held to, and it is a scoped claim: the
mitigation families, the LLM and agent testing surfaces and the operations
tooling still carry open defects, every one of them disclosed in the grid.

The full contract, the matrix by area, and what still has to change:
[CAPABILITY_STATUS.md](CAPABILITY_STATUS.md). The ledger itself is
`docs/capability-status.json`, regenerated by `scripts/capability_status.py`,
and `--check` fails the build when any published surface disagrees with it.

## Beta Go-Live: capability proof status

**Read this before anything else on this page, and read its scope with it.**
Everything below describes what was *built* to harden the library. This section
says how much of it has been *proved*, capability by capability.

It covers the 215 registered capabilities, which resolve to 222 of the 1,569 code
units on the public surface. Within that scope it is complete and clear. It is
**not** the number to trust for the library as a whole, and it was published as
though it were: for two days the status given for this library was "publishable,
the 215 capabilities have zero open defects", which was true and omitted the 95
open defects measured elsewhere on the surface. A claim scoped to the flattering
subset and presented as the whole is the defect this library exists to catch.

Every one of the 215 registered capabilities carries a Beta Go-Live batch saying
how far its honesty has actually been established: not whether it computes a
number, but whether it refuses to invent one when nothing is measurable.

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

**What has happened to that scope since.** The generated block above describes
the census as it stood: those 88 functions and 111 classes were outside its
scope, not cleared by it. They have not been left there. A separate
surface-grading campaign has graded a large part of the remainder under a
five-state method, and `scripts/surface_probe.py` has mechanically executed the
rest on nine input worlds. Neither belongs to this census and neither is added to
its totals, because they measure different things by different methods. The
current state of that campaign is generated below rather than typed here.

The figures in this paragraph used to be typed by hand and they rotted: the file
said "420 code units", "95 defect open" and "124 of 410 audited grades
overturned" for nine days and four waves after each of those stopped being true,
in a document whose subject is values that read as measurements and are not. The
dated sentences that follow are kept as history, which is what a date is for; the
live numbers are in the generated block. As of 2026-09-18 the campaign held 209
proven, 22 semi-proven, 94 not-a-measurement and 95 defect open, with 124 of 410
audited grades overturned when an independent agent was told to refute them. As
of 2026-09-25 it stood at 64 defect open, none of them an observed fabrication.
The 100% above is 100% of the 215.

<!-- BGL:grading:start -->
_Generated from the grading evidence. Newest wave: 2026-10-01._

| State | Count |
| --- | --: |
| Proven | 936 |
| Semi-Proven | 79 |
| Not A Measurement | 359 |
| Defect Open | 0 |
| Unproven | 0 |
| **Graded, total** | **1374** |

| How settled the grades are | Count |
| --- | --: |
| Current grade independently checked | 756 |
| **No independent check yet** | **618** |
| Overturned on audit, ever | 291 |
| ... of which the row was later re-graded | 236 |
| Caught fabricating, by execution | 417 |
| A fix recorded against the unit | 445 |

Every wave, unmerged, so the least-checked wave is visible rather than
averaged away. A unit graded twice appears in both rows, which is why
these do not sum to the totals above:

| Wave | Date | Graded | Caught fabricating | Fixed | Pin sabotaged | Audited | Overturned |
| --- | --- | --: | --: | --: | --: | --: | --: |
| wave1, wave2 | 2026-09-18 | 419 | 138 | 156 | 1 | 409 of 419 | 123 |
| wave3 | 2026-09-25 | 73 | 19 | 20 | 73 | **none** | not checked |
| wave4 | 2026-09-25 | 95 | 6 | 6 | 53 | **none** | not checked |
| wave5 | 2026-09-25 | 26 | 1 | 1 | 26 | **none** | not checked |
| bgl3 | 2026-09-27 | 342 | 178 | 183 | 342 | **none** | not checked |
| bgl4 | 2026-09-27 | 554 | 200 | 115 | 322 | 515 of 554 | 179 |
| bgl5 | 2026-09-27 | 192 | 192 | 192 | 192 | 1 of 192 | 1 |
| bgl4 | 2026-09-29 | 60 | 60 | 43 | 43 | 60 of 60 | 23 |
| bgl5 | 2026-09-29 | 19 | 19 | 19 | 19 | **none** | not checked |
| bgl4 | 2026-09-29 | 67 | 41 | 23 | 23 | 67 of 67 | 36 |
| bgl5 | 2026-09-29 | 28 | 28 | 28 | 28 | **none** | not checked |
| bgl4 | 2026-09-30 | 84 | 84 | 25 | 25 | 84 of 84 | 66 |
| bgl5 | 2026-09-30 | 77 | 77 | 77 | 77 | **none** | not checked |
| bgl5 | 2026-09-30 | 1 | 0 | 1 | 1 | **none** | not checked |
| grade-1 | 2026-09-30 | 481 | 0 | 0 | 481 | **none** | not checked |
| grade-1 | 2026-10-01 | 6 | 0 | 0 | 6 | **none** | not checked |
| probe (withdrawn) | 2026-09-18 | 0 | 0 | 0 | 0 | **none** | not checked |

| The surface those grades sit in | Count |
| --- | --: |
| Public code units in the library | 1580 |
| Graded by the capability census | 222 |
| Graded by the wider-surface waves | 1374 |
| **Carrying no grade from either** | **-16** |

Executed is a different and weaker measurement than graded, and the two are never added: the release gate's B1 counts a unit as executed once any of the census, the waves, the probes or the test suite has run it and recorded what came back. Reproduce both with `python scripts/release_gate.py`.

**618 unaudited grades is the largest thing open here.** 291 grades have been overturned when somebody independent attacked them, better than one in five of those checked, so an unaudited grade is a claim with evidence behind it rather than a settled measurement. The two overturn figures answer different questions: 55 rows carry an overturn AND still carry their audited grade, while 236 more were re-graded by a later wave, which resets them to unaudited and used to delete the record of the overturn with them.
<!-- BGL:grading:end -->

The graded and executed counts are in the generated block above rather than typed
here; the sentence that used to sit in this place said 817 graded of 1,564 and was
wrong on both numbers within two days. Grading wave 4 closed the last 94 units
never executed by any probe, on 2026-09-25, and found six live defects doing it. `assign_clusters` is the concrete example of why the two scopes must not be
confused. It sits in the census list of ungraded functions above, outside that
census's scope; wave 4 executed it for the first time and found it returning a
cluster randomisation with no control group in silence. Its answer today is CHECKED
and fixed, in the grid, which is the authoritative place to ask.
[GRADING.md](GRADING.md) explains both measurements;
[RELEASE_PLAN.md](RELEASE_PLAN.md) costs what is left.

The batch is on the capability itself, so `help(some_capability)` states it
without needing this page. It is also in `vfairness-manifest.json` under
`proof_status`, and in `src/vfairness/_proof_status.py`, all generated from one
committed evidence file.

**BGL-C is a positive statement, not an absence.** A capability nobody checked
says so, in those words, rather than looking like one that passed. That
distinction is the entire point: collapsing "not checked" into "fine" is the same
defect the library is being audited for, one layer up. BGL-C stands at 0 today,
which is a measurement about the 215 and says nothing about the code units that
carry no grade at all, of which there were 874 when this section was written and
**532** after grading wave 4 on 2026-09-25. Every one of them is published as NOT
CHECKED in the grid rather than left unsaid; see "Can I use this capability?" above.

**A BGL-D capability is not broken on real data.** It computes correctly when
there is something to compute. What is wrong is what it reports when there is
not, and until it is fixed a caller on that path is handed a number that was
never measured. BGL-D stands at 0 in the census. The equivalent state in the
surface campaign, DEFECT OPEN, stands at 95.

Batch definitions, the four-pass method that produced them, the per-package
breakdown and the staged remediation plan with hours are in
[BETA_GO_LIVE_PLAN.md](BETA_GO_LIVE_PLAN.md).

## How to check every number on this page

Numbers rot. Each figure below is stated with the date it was measured and the
command that produces it, so you can re-run it rather than trust the page.

```bash
python scripts/release_gate.py                   # the readiness verdict, both bars
python scripts/library_kpis.py                   # surface, grades, defects, coverage
python -m pytest --collect-only -q | tail -1     # tests collected locally
python -m pytest -q                              # the suite
ruff check src tests && ruff format --check src tests
mypy src
python -m build --sdist && ls -lh dist/          # distribution size
```

`release_gate.py` is the authority on whether this library is ready; it exits
non-zero while anything is open, and every readiness sentence on this page is
downstream of it. For test and coverage figures the authoritative live values are
in the CI run for the commit you are looking at, not in this prose.

## At a glance (measured 2026-09-11 unless stated)

- **Release gate: BETA READY, and NOT READY on the full 1.0 bar** (measured
  2026-10-01 by `scripts/release_gate.py`): all eight beta conditions are met and
  no defect is open anywhere on the public surface; the full gate waits for a
  second, independent examiner on part of the graded code. Every other figure in this list describes a
  measure that exists, not a verdict that it is sufficient.
- **8,088 tests passing on Python 3.11, 3.12 and 3.13**, plus 15 skipped and
  3 xfailed, with zero failures and the identical count on all three
  interpreters. Source: the full-suite CI run for commit `8559dda`
  ([run 34574015532](https://github.com/validantai/vfairness/actions/runs/34574015532)),
  which runs `pytest --cov=vfairness` on all three. CI installs
  `dev,rendering,viz,monitoring,causal,xai,training,parity,dashboard`.
  This figure and the coverage figure below were previously stated as 4,354 and
  68.84% from commit `31e182b` on 2026-08-28; both were real at the time and both
  had gone stale, which is what the section immediately above this one exists to
  stop happening to the capability counts.
- **81% coverage, line and branch combined** (41,685 statements, 15,058
  branches; 6,955 statements and 2,036 branches uncovered), against a CI floor
  of 55%. Branch coverage is on, so a half-tested
  `if`/`else` does not read as fully covered. A second, package-local workflow
  holds a floor of 48% over a narrower dependency set. Pull requests additionally
  face `diff-cover --fail-under=80`, so the lines a change adds must be tested
  even while the global figure is still climbing (VB-TEST).
- **`ruff check` and `ruff format --check` both pass** on `src`, `tests` and
  `scripts` at commit `8559dda`, and both are blocking gates (VB-LINT-1, VB-LINT-2).
- **`mypy src` is a blocking gate and is green in CI** on the dependency set the
  workflow installs. Reproducing that requires the same set: an environment that
  additionally carries the optional `torch` and `matplotlib` backends puts four
  more modules in scope, where mypy reports 11 errors across 5 modules today
  (measured 2026-09-06 at HEAD; it was 8 across 4 until the compliance three-state
  work put `operations/reporting/compliance.py` in scope). That is a limit of
  the gate's reach, not a claim that those modules are clean. See "What is still
  open". A `mypy --strict` island covers nine foundational modules and expands
  outward as more are cleaned (VB-TYPE-2).
- **242 deep-audit findings fixed, zero open, in the first audit campaign.**
  Later audit iterations run against later code and keep their own live registers
  in `docs/audits/`, some with open entries. "Zero open" describes that first
  campaign, not the library's whole audit history.
- **Mutation testing: 57 of 58 mutants killed (98%)** on `src/vfairness/_bands.py`,
  the band-boundary module, baseline recorded 2026-08-10 in
  [MUTATION_TESTING.md](MUTATION_TESTING.md). The scope is that one small pure
  module with a fast boundary-exhaustive runner, not the whole metric core; the
  single survivor is a documented equivalent mutant.
- **The source distribution builds to 1.4 MB** (`vfairness-0.1.0.tar.gz`, built
  2026-08-28), held there by an sdist allowlist (VB-PKG-2).
- No release publishes unless the built wheel installs into a clean environment
  and its public API runs (clean-room smoke test, VB-REL-3).
- Supply-chain controls: pip-audit and Bandit gate every library change,
  Dependabot updates pip dependencies and GitHub Actions weekly, and the
  tag-triggered release adds trusted publishing (OIDC), provenance attestations,
  and a CycloneDX SBOM. CodeQL code scanning and the OpenSSF Scorecard are wired
  up but skip while the repository is private (no GitHub Advanced Security);
  both re-arm automatically if it goes public.
- Apache-2.0, first-party, no service-level agreement (see the beta stance).

## Correctness and scientific validity

The metric core is checked against ground truth from several independent
directions, not just its own past output.

- **Reference-library parity.** Metrics are cross-checked against established
  libraries, and every intentional difference is documented in
  [DIVERGENCES.md](DIVERGENCES.md) rather than hidden (VB-TEST-4).
- **Synthetic analytic oracle.** Known-answer checks on constructed data where
  the correct fairness value is derivable by hand (VB-TEST-1).
- **Property-based tests.** Hypothesis generates adversarial inputs to probe
  invariants that fixed examples miss (VB-TEST-2).
- **Impossibility relationships.** Tests assert the known mathematical tensions
  between fairness criteria hold (and correctly vanish at perfect accuracy)
  (VB-TEST-3).
- **Determinism.** Repeated runs on the same input produce identical output; the
  bootstrap paths take an explicit `random_state` (VB-TEST-5).
- **Insufficient-evidence honesty, on the paths that carry it.** Strata below the
  group-size gate return an explicit insufficient-evidence verdict via
  `assessment.insufficient_evidence_groups` rather than being dropped in silence
  (VB-API-4). This line read "no code path silently passes or drops a small group"
  until 2026-09-18. That was a claim about every path in the library, and it did
  not hold: 95 code units were then proved by execution to report a value nobody
  measured, which is the failure the sentence ruled out. Those 95 are now fixed
  and pinned, and the live count is in the generated grading block above. The
  sentence is still wrong as originally written, because VB-API-4 is a mechanism
  that exists and works where it is wired in, not a property of every path, and
  the part of the library no wave has graded is where the next ones will be.
- **Undefined metrics are not failures.** A metric that cannot be computed
  (returns NaN, e.g. an error-rate metric when a group has no positive labels, or
  R2 when a group has constant `y_true`) is surfaced in
  `assessment.not_assessable_metrics` and excluded from the score, never reported
  as a fairness violation (VB-EVAL-1).
- **No silent swallow.** An AST-level guard test forbids the metric core from
  swallowing computation errors across evaluation, in-processing, and
  post-processing (VB-SEC-4).

## The could-not-check campaign

This is the largest single piece of hardening in the project's history, it ran
over sixteen waves across two days in August 2026, and it closed one defect
class.

### The defect

vfairness computed its fairness metrics correctly. In a large number of places
it then made a **claim about those numbers that the data did not support**. The
mechanism was one habit, repeated: where a value was missing, the code
substituted a default (usually `0`, sometimes a midpoint `0.5`, sometimes a
maximally reassuring `p = 1.0`, sometimes a severity string or a boolean
`False`), and then graded, counted, coloured, sorted by, or plotted that
substitute as though it were a measurement.

**Both directions occurred, and the second is not the safer error.** A fabricated
all-clear tells a reader a group was checked and cleared when it never was. A
fabricated breach reports a violation nobody measured, which sends someone
chasing a problem that does not exist and discredits the tool when they work out
why.

Each of the following was reproduced by execution before it was fixed:

- The CI/CD deployment gate was **inverted for the whole ratio family**. Under
  the four-fifths rule it approved a disparate impact of 0.00, a protected group
  never selected at all, and blocked perfect parity at 1.00. All four cases were
  wrong.
- A group dropped by the default `min_group_size` made the metrics return 0.0 and
  1.0, which reads as **perfect parity**, for what was the maximal violation.
- SVG exports rendered **"PASSED 5 / 5, FAIR RATE 100%"** for a report the engine
  itself had marked not assessable.
- Chart adapters **fabricated confidence intervals**, printing value plus or
  minus 0.02 as a computed band on every real report.
- `report_card_to_svg(None)` rendered **"DEPLOYMENT APPROVED"** from no data at
  all, because a demo fixture was reachable as a runtime fallback.
- `assert_fairness` treated a **NaN metric as a pass**, so a run that measured
  nothing satisfied a release gate.
- `reliability_diagram_to_svg` reported **"Well Calibrated, ECE = 0.000"** on no
  data.
- The analyzer **misstated its own row accounting**, reporting `original_size`
  105 and `n_excluded` 0 for a run that excluded 15 of 120 rows.

### What the library guarantees now: three states, never two

> **Scope correction, 2026-09-11, updated 2026-09-18.** Read this section as
> describing the surfaces this campaign REACHED, not the whole library. The Beta
> Go-Live census then executed all 215 registered capabilities on input where
> nothing they claim to measure exists and found **124 that did**. All 157
> defects it reproduced at the public API are now fixed, and the census reports 0
> open; the live per-batch counts are in the table at the top. That closes the
> census, not the library: surface grading has since found open defects outside
> it, on code this campaign never reached either (95 when first measured, **85 as
> of 2026-09-25**). The campaign below
> fixed everything it found; what it had not done was measure most of the
> surface, and "nothing found" was being read here as "nothing there". That is
> the same defect the campaign is about, applied to the campaign's own coverage,
> and the correction is only worth anything if it is applied again each time the
> measured scope grows.

On the verdict surfaces this campaign reached, **assessed-pass, assessed-fail and
could-not-check** are all carried, and could-not-check is never collapsed into
either of the other two.

- A row, group, metric or chart that measured nothing gets **no number, no badge,
  no colour, no plot point**, and no place in any count or headline implying it
  was measured.
- Verdicts are **graded over the graded subset**, so a partial run still produces
  a useful answer instead of refusing outright.
- An **unqualified all-clear is never rendered while anything is ungraded**.
- The **ungraded count is stated on the canvas itself**, in the headline band, not
  in a footnote.
- A default is not a measurement, a sentinel is not a measurement, and absent and
  zero are different claims.

Two of those behaviours, executed on 2026-08-28:

```python
>>> report_card_to_svg(None)          # renders, and says what it could not do
'... COULD NOT CHECK: no gate decision was supplied, so nothing was approved
here ... DEPLOYMENT: NOT CHECKED ...'

>>> classification_fairness_report(y_true, y_pred, groups)  # one group never selected
demographic_parity_ratio == 0.0  ->  in failed_metrics, not in passed_metrics
demographic_parity_ratio == 1.0  ->  in passed_metrics, not in failed_metrics
```

### How that was verified

The verification method is the part worth trusting, because the guarantee above
is only worth what the checks behind it are worth. Five things carried it.

**1. An exhaustive sweep, so coverage is a table and not a judgement.** Every
public `*_to_svg` adapter is called with the emptiest input it accepts, and the
result is read. There are 44 of them
(`[n for n in dir(vfairness.rendering) if n.endswith("_to_svg")]`, executed
2026-08-28), so "did we find them all" is answered by enumeration rather than by
recollection. The campaign's own runs of that sweep recorded the number of
adapters fabricating a verdict from nothing falling 13, then 4, then 0. Re-run
independently on 2026-08-28 against commit `31e182b`: **40 of the 44 render an
explicit could-not-check state** for the emptiest input they accept, and **none
renders a canvas without one**. The remaining 4 take a report dataclass with no defaults, so the emptiest
input has to be hand-built rather than produced by a generic harness; each of
those is pinned by its own named test, and those tests pass.

**2. A per-row sweep, one level down.** The chart-level rule does not imply the
row-level rule: an adapter whose chart-level state is correct can still invent an
individual row from defaults, so a green chart did not yet mean every row on it
was measured. The per-row sweep constructs an input where the chart carries real
data and **one row reports nothing but its identifier**, then reads the rendered
artifact.

**3. An AST scanner that counts the defect shape, not its spelling.** The scan
walks the parsed tree of `src/vfairness/rendering` and finds every
`something.get(key, <numeric or boolean literal>)`, which is the shape that turns
an absence into a measurement. Executed against two commits on 2026-08-28, it
reports **180 sites across 14 files at `e220ef4`** (before the campaign) and
**47 across 12 files at `31e182b`**. Both are reproducible from the repository at
those commits; run it yourself rather than believing the numbers here. The count
is a work item, not a clean bill: it is how the remaining sites are found and
counted, and it keeps falling.

The same principle is what the suite's own reintroduction guards are built on.
They walk the parsed tree rather than scanning for text, so a rename, an
attribute operand, a `.casefold()`, an `re.search()` or an f-string cannot hide
the defect from them. Where a site is deliberately exempt, the exemption is
**named in the test, with its reason**, and keyed by the defect rather than by
the file, so it cannot silently bless a different bug in the same file.

**4. Every guard was sabotaged before it was trusted.** After each fix, the exact
defect is reinstated, the named test is confirmed to go **red**, the healthy-input
controls are confirmed to stay **green**, and the source is restored
byte-identically. In several places the sabotage is kept permanently as a
positive control inside the suite, so a guard cannot quietly become inert.

**5. Healthy output was protected throughout.** A library that answers
could-not-check for everything would pass every failure test and be useless.
Every could-not-check assertion in these suites is paired with a healthy-input
control asserting that a genuine all-pass run keeps its green all-pass, so a
"fix" that simply suppresses verdicts everywhere fails the controls. The
published example gallery is regenerated from live computation
(`scripts/regenerate_gallery_examples.py`) and compared against the previous build
as **rasterised images**, not as markup, because at one commit the markup differed
from the shipped gallery in 12 files and the pixels in 4. That comparison is
currently a manual step in the campaign's process, not a CI check; see "What is
still open".

### The single most useful thing this campaign found

**Five guards in this repository were green while the defect they existed to
catch was live.** Sabotage is what found them, and in every case the cause was
the same: the guard enumerated the *spelling* of the bug rather than the bug.

1. The direction-bug guard scanned exactly one file, `_metric_direction.py`,
   while each recurrence of the bug appeared in a **different** file.
2. That guard was pinned to three exact spellings of the token `"ratio"`. When
   the same bug class reappeared one directory away as `"disparate impact" in n`,
   the pin stayed green while a 0.45 disparate-impact difference against a 0.10
   bound rendered as a green pass.
3. Measured against sixteen real spellings of that one defect, the
   spelling-enumeration guard caught **4 and missed 12**: it could not see a
   renamed operand, an attribute operand, `.casefold()`, `.find()`,
   `re.search()`, the token parked in a constant, a loop over a tuple of tokens,
   or an f-string-wrapped operand. Rebuilt on the parsed tree, it catches 16 of
   16, and each of the sixteen is its own positive control
   (`tests/test_audit_final_release.py`).
4. The silent-swallow guard's exemption list was keyed by **file**, so reinstating
   the exact defect in `discovery.py` left the guard green, because that file was
   already exempt for an unrelated reason. Re-keyed to
   `(file, exception, shape)`, the same sabotage now fails and names the line
   (`tests/test_no_silent_swallow_core.py`).
5. A guard that greps rendered markup for a string **passes on a chart where that
   string is painted over by another element**. A summary pill drawn at the wrong
   coordinate covered its siblings, so a scan holding one high-risk proxy
   displayed a single emerald "LOW: 2" as its headline, while "HIGH: 1" sat in
   the file the whole time. Only rasterising the image and looking at the picture
   finds that.

The transferable lesson: a guard is only worth its green. Assert on the drawn
text, then rasterise and look; key an exemption by the defect, not by the file;
and detect the shape on the parsed tree, because a list of spellings only ever
buys the spellings its author thought of.

## The deep-audit campaign

A two-workflow, multi-agent deep audit (one static, one empirical, each
adversarially verified) ran across the library and produced 242 findings. All
were fixed across five waves to zero open items, and the suite grew from 831 to
over 1,280 tests during that campaign. Its coverage was never enumerated code
unit by code unit, so "across the library" says where it looked and not what it
reached: the campaigns that followed found defects it had not, which is the
evidence that it did not reach everything. The suite has kept growing since: the
CI run for commit `31e182b` on 2026-08-28 executed 4,354 passing tests, and the
run for `8559dda` on 2026-09-11 executed 8,088. Later audit
iterations, run against later code, keep their own registers in `docs/audits/`
and are not covered by this campaign's "zero open". Representative fixes from
this one:

- **Statistics honesty.** `fisher_exact_test` now uses the real Fisher exact
  test (no silent chi-square degrade above n=200); Bayesian group intervals
  report the observed rate as the point estimate; a minimum-detectable-effect is
  computed from the requested alpha and power.
- **Calibration correctness.** `BetaCalibrator` is real Kull et al. 2017 beta
  calibration rather than a renamed Platt transform; group calibrators pass
  unknown group ids through untouched and disclose the mismatch instead of
  silently zeroing them.
- **Threshold optimization.** The optimizer that previously crashed on fit now
  searches thresholds jointly by coordinate descent and validates the final
  assignment.
- **Falsy-zero bugs.** A family of bugs where a legitimate `0` (a zero-tolerance
  threshold, a fairness score of 0, an explicit `0.0` alert bound) was treated
  as absent were fixed across CI/CD, monitoring, and rendering.
- **Rendering robustness.** Hostile inputs (empty modules, `None` nests,
  stringified numbers, infinities) either render or raise a clear `ValueError`
  naming the template, instead of leaking a `ZeroDivision` or overflow error.

## Code-quality gates

- **Lint gate (VB-LINT-1).** `ruff check src tests` runs on every library change
  and is blocking. The tree is clean under the `E`/`F`/`I`/`N`/`W` rule set.
- **Format gate (VB-LINT-2).** The whole `src` and `tests` tree is
  `ruff format`-clean (double quotes, space indent) and `ruff format --check` is
  blocking. The one-time reformat commit is recorded in `.git-blame-ignore-revs`
  so `git blame` skips it.
- **Type gate (VB-TYPE-1).** A full type-error burndown took `mypy src` from 500
  errors to zero across 82 modules with root-cause fixes, and `mypy src` is now
  blocking on the dependency set the workflow installs
  (`dev,rendering,viz,monitoring,causal`). That dependency set is also the gate's
  limit: with the optional deep-learning and plotting backends additionally
  installed, mypy sees four more modules and reports errors there today. See
  "What is still open". The type pass also surfaced and fixed real bugs (for example a
  subgroup-robustness audit and a correlation-based proxy detector that were each
  silently dead because a value was passed in the wrong argument position). The
  package ships a `py.typed` marker so downstream projects type-check against the
  library's own hints (VB-API-1).
- **Strict type island (VB-TYPE-2).** Foundational public modules and core
  helpers are additionally held to `mypy --strict` via a per-module override; the
  island expands outward as more modules are cleaned.
- **Public API-surface snapshot.** A test freezes the public surface so an
  accidental breaking change fails CI (VB-API-3).
- **Mutation testing (VB-TEST-9).** `mutmut` verifies the tests actually catch
  injected faults. Baseline recorded 2026-08-10: 57 of 58 mutants killed on
  `src/vfairness/_bands.py`, the band-boundary module, with the one survivor
  documented as an equivalent mutant. The configured scope is that single small
  pure module, chosen so a run finishes in about a minute; the workflow can widen
  `paths_to_mutate` on demand, and it is informational rather than blocking.

## The test suite

- 8,088 tests passed on Python 3.11, 3.12 and 3.13 in the CI run for commit
  `8559dda` on 2026-09-11, with 15 skipped, 3 xfailed and zero failures, the same
  count on all three interpreters. Collected locally at 2026-09-18 the suite is
  10,397 tests; that is a collection count from this working tree, not a CI pass
  count, and the two are different claims. A skip is invisible in a summary line,
  so CI sets `VFAIRNESS_REQUIRE_BACKENDS=1`, which turns a missing optional
  backend into a named failure rather than a silent skip.
- **Branch coverage and new-code gating (VB-TEST).** Coverage counts untaken
  branches, not just unexecuted lines. The measured figure is 81% line-and-branch
  at commit `8559dda` on 2026-09-11, against a CI floor of 55%. The floor is only
  ever raised, so
  it acts as a ratchet, and the gap between the floor and the measurement is the
  room a regression can move in without failing anything. A pull-request
  `diff-cover` gate separately requires the lines a change adds to be at least 80%
  covered, so new code is held to the target whatever the global figure does.
- **Executable documentation (VB-DOC-3).** Snippets in `docs/examples.md` run in
  the suite via doctest, so a documented example that breaks against the real API
  fails the build.
- **Report snapshots, performance and scale, zero-telemetry, and
  deserialization-safety** suites guard output stability, large-input behavior,
  the no-network guarantee during metric computation, and the absence of unsafe
  loaders (VB-TEST-10, VB-PERF-1, VB-GOV-2).

## Security and supply chain

- **Static and dependency scanning.** Bandit (source, medium and high severity)
  and pip-audit run on pull requests, on pushes that touch the library, and on a
  weekly schedule. pip-audit audits the resolved third-party dependency set from
  a non-editable `pip freeze` under `--strict`, so a genuinely unauditable
  dependency fails the build. A CodeQL job is configured in the same workflow but
  skips while the repository is private (code scanning needs GitHub Advanced
  Security); it re-arms automatically if the repo goes public (VB-SEC-4,
  VB-SEC-8).
- **Supply-chain posture.** Weekly Dependabot updates for pip dependencies and
  GitHub Actions, and a CycloneDX SBOM generated at release. An OpenSSF Scorecard
  workflow is committed but skips while the repository is private (Scorecard
  publishing and its SARIF upload both need a public repo with code scanning); it
  re-arms if the repo goes public (VB-SEC-7, VB-SEC-5).
- **Deserialization safety.** The xai sidecar refuses to deserialize untrusted
  model payloads unless the caller explicitly opts in, closing a
  remote-code-execution vector (VB-SEC-6).
- **Outbound egress guard (`vfairness.net`).** Model-endpoint calls made through
  `LLMApiProxy` pass an SSRF guard first: `validate_endpoint` parses the URL,
  resolves the host and vets every resolved address at construction time, and a
  refused target raises `SSRFError` (surfaced to the caller as `ValueError`). The
  session then mounts a DNS-rebinding-safe `PinnedIPAdapter` on both schemes, so
  the vetted IP is dialled while SNI and certificate verification stay on the
  hostname. The Pulse LLM probe builds an `LLMApiProxy` and is covered too.
  Every surface that posts to a caller-supplied URL goes through it: the LLM
  judge scorer and the XAI sidecar client both call
  `guarded_post` (`llm/scorers.py`, `xai/sidecar_cli.py`), and the groundedness
  judge uses a redirect-revalidating `GuardedSession`, on by default
  (`guard_egress=True`) with an explicit opt-out for a private address. As of
  2026-08-28, `grep -rn "requests\.post" src/vfairness/` matches only comments and
  the guard's own docstring, and `validate_endpoint` executed against
  `169.254.169.254`, `127.0.0.1` and `10.0.0.5` refuses all three with
  `SSRFError` while allowing a public HTTPS endpoint.
  Private (RFC1918), loopback, link-local, unique-local and
  cloud-metadata (169.254.169.254) destinations are refused, and IPv4-mapped IPv6
  is unwrapped so `::ffff:127.0.0.1` cannot smuggle a loopback past the check.
  `LLMApiProxy` takes an explicit `allow_loopback=False` opt-in that unlocks
  loopback only (127.0.0.0/8, ::1) for a local model server; cloud-metadata stays
  refused whatever the caller asks for.
- **Second security-audit hardening (defense in depth).** The MCP server's
  shared `load_dataframe` loader (called by every registered tool, so the check
  covers all seven) rejects remote `data_path` URLs so an agent-supplied path
  cannot become network egress or an SSRF fetch; the Pulse artifact download pins
  the scheme to `https` and caps response size; the deserialization scan resolves
  aliased imports and covers `marshal`, `pandas.read_pickle`, `jsonpickle`, and
  `shelve`; error logs no longer echo sensitive payloads; and CI workflows
  declare least-privilege (`contents: read`) token scope.

## Packaging and release integrity

- **Licensing.** Apache-2.0 with an accompanying `NOTICE`, single-source version
  (VB-LIC-1, VB-PKG-1).
- **Lean source distribution.** An sdist allowlist keeps the archive small
  instead of shipping the whole working tree. Built on 2026-08-28,
  `vfairness-0.1.0.tar.gz` is 1.4 MB; reproduce with `python -m build --sdist`
  (VB-PKG-2).
- **Build verification.** Every release runs `twine check` and
  `check-wheel-contents` on the built artifacts.
- **Clean-room install smoke test (VB-REL-3).** Before any publish, the built
  wheel is installed into a fresh environment and its public API is exercised
  from outside the source tree, so a missing data file, wrong entry point,
  unshipped subpackage, or missing dependency blocks the release.
- **Trusted publishing.** Tag-triggered release to PyPI via Trusted Publishing
  (OIDC, no stored token), with provenance attestations requested explicitly
  rather than left to the action's default, and a required human approval before
  the upload (VB-REL-1, VB-REL-2). There is no TestPyPI leg: it was removed on
  2026-08-28, so no rehearsal upload stands in front of the real one, and the
  gates that remain carry that weight instead.

## API stability and methodology

- **Frozen public surface.** The surface listed in
  [API_STABILITY.md](API_STABILITY.md) is already treated as stable: it changes
  only after a deprecation warning, and removals are reserved for `1.0.0`
  (VB-API-2).
- **Public exception hierarchy.** Errors are rooted at `VfairnessError` while
  still subclassing the built-in they replace, so existing handlers keep working
  (VB-API-5).
- **Methodology versioning.** Every `FairnessAnalyzer.get_report()` stamps a
  `methodology_version`, so a rating records the methodology that produced it
  independently of the code version (VB-DOC-1).
- **Canonical module taxonomy.** The public surface is 15 top-level sub-packages
  (6 pipeline, 8 specialized, and `net`, the one infrastructure package, which
  holds the egress guard), each declaring `__all__` and enforced by a test
  (VB-PKG-3).

## Continuous enforcement

The gates above are not one-time cleanups; they run in CI on every change.

**What "blocking" means here.** A blocking gate fails its workflow run: the job
goes red and the run is red. That is not the same thing as a merge blocker.
Neither repository configures required status checks on its default branch, so a
red run does not by itself stop a merge. Checked on 2026-08-28 with
`gh api repos/<owner>/<repo>/branches/main/protection` and `.../rulesets`: the
public repository has no branch protection and no rulesets at all, and the
private development repository has protection that blocks force-pushes and
deletion but sets `required_status_checks` to null. It is said here rather than
left to the word "blocking", because a reader deciding whether to trust this
library should know the difference between a gate that runs and a gate that
blocks. `docs/RELEASE_PIPELINE.md` says the same thing about the release
workflows.

| Workflow | What it enforces |
|---|---|
| Full test suite | The whole suite with branch coverage on Python 3.11 / 3.12 / 3.13, at a `--cov-fail-under=55` floor, with `VFAIRNESS_REQUIRE_BACKENDS=1` so a missing optional backend fails rather than skips |
| Quality | `ruff check`, `ruff format --check`, and `mypy src`, all blocking; a PR-only diff-cover check |
| Security | Bandit and pip-audit (a CodeQL job is present but skips while the repo is private) |
| Scorecard | OpenSSF supply-chain posture (skipped while the repo is private; re-arms if it goes public) |
| Mutation testing | `mutmut` on `_bands.py` (on demand and weekly, informational, does not block) |
| Cross-library parity | Metric agreement with reference libraries |
| Pulse contract | The Pulse assessment output contract |
| Changelog check | A changelog entry accompanies changes |
| Release | Build, verify, clean-room wheel-install smoke test, SBOM, and trusted publishing. Tag-triggered, and it has never yet run: there is no tag |

## What is still open

A page that admits a known limit is worth more than one that does not, and this
library's whole argument is that it says what it did not measure. The same rule
applies here. The first two entries are measured 2026-09-25; the rest were
verified 2026-08-28 and carry their own dates where they differ.

- **64 open defects on the public surface, none of them an observed fabrication,
  and none in the assessment path itself.** The split by what they return: 37 a
  fairness number or a result, 27 something nobody has classified, counted with
  them because not having worked out what a function returns is not evidence that
  it is harmless, and **zero that return neither**, which is what closed B3. The
  split by KIND is the one that matters: 25 are fixes an independent audit declined
  to certify and 39 are units nothing was ever seen fabricating whose pins do not
  cover enough input. Thirty-five were closed on 2026-09-25, each reproduced by
  execution at the public entry point and each cause sabotaged separately; the
  evidence is in `docs/surface-grading-2026-09-25.json`. What remains sits in the
  mitigation families, the LLM and agent surfaces, monitoring, reporting and the
  MCP tools, and 54 of the 67 are units the probe cannot reach at all, so they
  carry no evidence either way rather than an observed fabrication.
  Costed in [RELEASE_PLAN.md](RELEASE_PLAN.md).
- **Most of the public surface still carries no grade at all, and the grades that
  exist are mostly unaudited.** The live counts are in the generated grading block
  under "What has happened to that scope since"; typing them here is what let the
  previous version of this bullet claim 112 unaudited nine days after the figure
  had passed 500. An ungraded code unit is not a clean one, it is the absence of a
  measurement, and it is where every defect this campaign found was living. The
  unaudited count has risen at every wave, from 10 to 112 to 207 to the figure in
  that block, and **that is the honest direction rather than a regression**: each
  wave adds grades faster than auditors can attack them, and a grade nobody has
  argued with is published as a claim rather than counted with the settled ones.
  The measured reason to care is in the same block: better than one audited grade
  in five has been overturned.
  Counting execution rather than grading, **every unit has now been reached at
  least once**, which is what criterion B1 measures and prints; the figure was 919
  of 1,558 when this paragraph was written, and the denominator itself has moved
  since, which is why the live one is not repeated here.
  Four programs moved it on 2026-09-25, the first three at no agent cost: a class
  probe (392 classes had never
  been executed by any probe), an explainability probe built around a fitted
  model (79 of 81 XAI units had no grade because our fixtures could not reach
  them, not because anybody judged them risky), and an argument binder that
  builds the library's own types from the annotation naming them; then the test
  suite was counted at last, which is the correction described above, and grading
  wave 4 closed the 94 units that were left.
  `scripts/release_gate.py` printed 1,524 for B1 until 2026-09-18, because it
  added the graded and probe-reached counts instead of taking their union and the
  307 units in both were counted twice. That arithmetic reported the surface as
  better examined than it is. It now prints the union and a test pins it.
- **The per-row rule is met in most adapters, not yet library-wide.** The
  chart-level rule holds across all 44 public adapters. One level down, an
  adapter whose chart-level state is correct can still invent an individual row
  from a default. Each remaining site is recorded, with its reproduction and its
  current state, in `docs/audits/row-level-fabrication-register-2026-08-27.md`.
  That register is the live source of truth; this page is not.
- **The canvas and the accessible `<desc>` are two surfaces, and they drift.**
  Where a chart has learned the third state, the description a screen reader
  receives can still carry the older two-state phrasing or an ungraded row in its
  denominator. Roughly half the sites found after the original register entries
  were exactly this. Both surfaces are now checked, but the class is not closed.
- **Nothing ties the published example gallery to the code.** The gallery is
  regenerated by a manual script, so a fix can land in a template and the
  published picture can stay stale. That happened, and the artwork carried a
  false all-clear for a day after the code was fixed. A CI job that regenerates
  the gallery and fails on a pixel difference would close it; there is no such
  job today.
- **The type gate does not cover the whole tree.** `mypy src` is green on the
  dependency set CI installs. Add the optional `torch` and `matplotlib` backends
  and mypy reports 11 errors across 5 modules (measured 2026-09-06)
  (`in_processing/calibrators/group_calibrators.py`,
  `in_processing/loss_functions/{adversarial,counterfactual}.py`,
  `preprocessing/feature_engineering/visualization.py`). The gate should be
  widened to that set rather than the claim being softened.
- **The coverage gate is far below the coverage measurement.** Coverage measured
  81% line and branch at commit `8559dda` on 2026-09-11, up from 68.84% at
  `31e182b` on 2026-08-28. The CI floor is still 55%, so roughly 26 points of
  regression would pass the gate unnoticed; the ratchet has not been raised to
  follow the measurement. New code separately faces an 80% diff-cover gate. The
  figure moves with every run and is published with its date for that reason, and
  the number to act on is the floor, not the measurement.
- **Mutation testing covers one module.** The 98% figure is `_bands.py`, and
  widening it to the metric core is on-demand work, not a standing gate.
- **One test was failing on `main` on 2026-08-28. CLOSED.** A test named
  `test_changelog_unreleased_section_is_empty` (written here without its module
  path, because it no longer exists to point at) asserted
  a `[Unreleased]` changelog heading that the honest-versioning change replaced
  with `[0.1.0] - UNRELEASED`, and the full-suite workflow was red on that commit
  for that one reason. That test no longer exists:
  `grep -rn "def test_changelog_unreleased_section_is_empty" tests/` returns nothing
  (the name survives only in the docstring and the controls of the test that now
  refuses a citation of a test that is gone),
  and the test that replaced it,
  `tests/test_docs_truth.py::test_changelog_top_section_is_the_unreleased_version`,
  passes (`pytest tests/test_docs_truth.py` -> 14 passed, run 2026-08-28). The
  entry is kept rather than deleted, because the record of what was wrong is the
  point of this section.
- **Nothing has been released.** No tag, nothing on PyPI or TestPyPI. The release
  pipeline is built, gated and never yet run to a publish.

## What beta does and does not promise

Beta will mean ready for real users to rely on while the library keeps improving.
The beta gate passes: the release gate reports BETA READY as of 2026-10-01. Nothing
has been published yet. A beta may ship with known defects; it
may not ship with unknown ones, and it may not be quiet about the known ones,
which is why the counts on this page are stated at full weight rather than
summarised.

Passing the gate would mean one specific thing: this library does not report a
number it did not measure, and tests hold it to that. It would not mean the
statistics are correct, that a method suits your problem, or that any test covers
every input. Those are different claims and they need different evidence.

It is free and open source under Apache-2.0 and stays that way. It is a community
project with no service-level agreement: no guaranteed response time, no uptime
commitment, and no contractual assurance. APIs may still change before the
stable `1.0.0`, though the frozen surface above is already treated as stable.
Managed, independent, service-backed assurance lives on the validant.ai
platform, not in this library. See [BETA.md](BETA.md) for the gate and how to
join.
