# vfairness 0.1.0: the first public beta

vfairness measures fairness in machine-learning and AI systems: group disparity
metrics for classification, regression and ranking, calibration, intersectional
analysis, LLM and agent testing, explainability, mitigation and monitoring.

```bash
pip install vfairness==0.1.0
```

This is a beta. The API may change before 1.0.0.

## What you can rely on

Every statement here is held by a test that fails the moment it stops being true.

- **Same answer as the established libraries.** Every capability that fairlearn,
  scikit-learn, statsmodels or scipy also computes gives the same number on clean
  data: 34 of 34, compared on two datasets to the ninth decimal.
- **No invented numbers.** When the data cannot support a number (one group only,
  no rows, missing scores, a group too small to read, a group with no positive
  outcomes), every capability that measures says so instead of returning a
  harmless-looking value such as 0.0 that reads as "no bias". 109 of 109 pass.
- **Three answers, never two.** A result is measured, failed, or could not be
  checked, and the third is written into the result itself rather than only into a
  warning.
- **Every public code unit has been run and examined** (1,580 of 1,580), and no known
  defect is open.

## What this beta does not yet establish

- **A second examiner for every grade.** Each code unit was examined once. A second,
  independent examiner has re-checked part of them; where that happened, about one
  first grade in three was overturned, and every defect found was fixed. The rest are
  re-checked before 1.0. Until then, those grades are evidence, not proof.
- **Correctness where no other library exists.** 76 capabilities (LLM, agent and
  multi-agent testing, ranking, explainers, some calibration and intersectional
  measures) have no outside implementation to compare against. They are shown not to
  invent numbers on broken data; their values on normal data rest on vfairness's own
  tests.

## Where to read more

- Readiness, with every figure and what is still open:
  https://vfairness.validant.ai/quality-and-hardening/
- How releases are built and published: https://vfairness.validant.ai/release-pipeline/
- Full change history: `CHANGELOG.md`

## About the earlier 0.0.1

The `0.0.1` on PyPI is an early name reservation from March 2026 that contains none
of this code. Install `0.1.0` or later.
