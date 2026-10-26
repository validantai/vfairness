# vfairness 0.2.0: deployment policies

vfairness measures fairness in machine-learning and AI systems: group disparity
metrics for classification, regression and ranking, calibration, intersectional
analysis, LLM and agent testing, explainability, mitigation and monitoring.

```bash
pip install vfairness==0.2.0
```

This is the second public beta. The API may change before 1.0.0. It is a minor
version: new functionality, and nothing on the frozen public surface removed or
repurposed (`docs/API_STABILITY.md`).

## New

- **Deployment policies: required coverage before approval.** The gate knew what
  it was configured to check, not what a deployment requires, so every way of
  configuring it to check less still approved. A `CoveragePolicy` states what a
  deployment requires (attributes, named intersections, metrics with the loosest
  acceptable bound, its own minimum group size), and
  `ModelFairnessGate.evaluate_against_policy()` returns a `PolicyDecision` with one
  item per required group set and metric: assessed, insufficient evidence, omitted
  by configuration, or input absent. `permitted` is True only when every item is
  assessed and passes. Opt-in: results without a policy are unchanged.
  ([vfairness issue #1](https://github.com/validantai/vfairness/issues/1))
- **Every gate decision says what it evaluated.** `evaluated_scope` lists the
  levels that ran, the threshold each applied, and the group sets left out with the
  setting that left them out.
- **GitHub checks and report cards follow the policy.** `create_github_check()`
  concludes `failure` whenever a supplied policy is not met, and
  `FairnessReportCard` leads with the policy result.
- **Pre-release modules, outside the frozen surface and the grading:**
  `vfairness._register` fingerprints a model version where it lives and writes an
  evidence bundle for the Validant Model Register, without sending the model, the
  data or the results anywhere in clear; `vfairness._group_codes` rebuilds the codes
  of a coded tracker run on your own machine. Their names may change.

## Changes you may notice

- **Experiment trackers code group names by default.** `log_fairness_to_mlflow`,
  `log_fairness_to_wandb` and `auto_log_fairness` replace every group name with a
  keyed code (HMAC-SHA256). Set `VFAIRNESS_GROUP_LABEL_KEY` once, or pass
  `label_key`; without a key the writers refuse before writing anything. Pass
  `group_labels="names"` to keep the previous output.
- **A Pulse run on a text decision column needs its favourable value declared**
  (`decision_labels.favourable_values`). Pulse used to guess, and on
  "Invited"/"Rejected" it guessed wrong; undeclared, it now refuses by name.

## Fixed

- **pandas 3.1:** no deprecation warning from the library on healthy data.
  Results are unchanged.
- **Pulse:** a gap interval that holds at zero counts (Newcombe hybrid score
  instead of a percentile bootstrap that read 0 of 22 as a confirmed gap), every
  attribute graded Fail has a finding behind it, the worst pair is corrected for
  being chosen out of several, and an empty grid cell is no longer read as parity.

## What you can rely on

Every statement here is held by a test that fails the moment it stops being true.
Before the tag, `scripts/release_gate.py` reported **BETA READY**: all eight beta
conditions met.

- **Same answer as the established libraries.** Every capability that fairlearn,
  scikit-learn, statsmodels or scipy also computes gives the same number on clean
  data: 34 of 34.
- **No invented numbers.** When the data cannot support a number, every capability
  that measures says so instead of returning a harmless-looking value such as 0.0.
  109 of 109 pass.
- **Three answers, never two.** A result is measured, failed, or could not be
  checked, and the third is written into the result itself.
- **Every public code unit has been run and examined, and no known defect is
  open.**

## What this beta does not yet establish

- **A second examiner for every grade.** 637 grades have not yet been re-checked by
  an independent examiner; that is what the full 1.0 gate still waits for. Until
  then, those grades are evidence, not proof.
- **Correctness where no other library exists.** 76 capabilities have no outside
  implementation to compare against. They are shown not to invent numbers on broken
  data; their values on normal data rest on vfairness's own tests.
- **The pre-release modules** (`vfairness._register`, `vfairness._group_codes`) have
  not been graded.

## Upgrading from 0.1.0

- Logging to MLflow or Weights & Biases: set `VFAIRNESS_GROUP_LABEL_KEY`, or pass
  `group_labels="names"` to keep real names.
- Running Pulse on a text decision column: declare
  `inputs.decision_labels.favourable_values`.

## Where to read more

- What changed, in full: https://vfairness.validant.ai/concepts/changelog.html#v0.2.0
  and `CHANGELOG.md`
- Deployment policies: https://vfairness.validant.ai/api-reference/#deployment-policies
- Readiness, with every figure and what is still open:
  https://vfairness.validant.ai/quality-and-hardening/
- How releases are built and published: https://vfairness.validant.ai/release-pipeline/
