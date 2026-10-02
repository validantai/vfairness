<p align="center">
  <img src="https://raw.githubusercontent.com/validantai/vfairness/main/docs/img/vfairness-hero.png" width="900"
       alt="vfairness: measure and explain machine-learning fairness. An illustrative fairness-evidence chart of three between-group disparity metrics (0 = parity), each with a 95% uncertainty interval compared to a chosen tolerance: equal opportunity is WITHIN TOLERANCE (interval entirely below the tolerance), demographic parity EXCEEDS TOLERANCE (interval entirely above it), and the calibration error gap is INCONCLUSIVE (interval overlaps the tolerance).">
</p>

<h1 align="center">vfairness</h1>

<p align="center">
  <strong>Measure and explain machine-learning fairness, and say plainly when the data does not support a verdict.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-0.1.0%20beta-006686.svg" alt="Status: 0.1.0 beta">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Requires Python 3.11 or newer">
  <a href="https://github.com/validantai/vfairness/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://vfairness.validant.ai"><img src="https://img.shields.io/badge/docs-vfairness.validant.ai-006686.svg" alt="Documentation"></a>
</p>

**vfairness** is a Python library for auditing whether a machine-learning system
treats different groups fairly, for finding the sources of bias behind an unfair
result, and for intervening to reduce it. It is built for the ML engineers,
data scientists, auditors, and risk and compliance teams who have to stand
behind a fairness claim. It covers classification, regression, and ranking
models, and extends the same discipline to LLM, agent, and vision systems.

What sets it apart is honesty under uncertainty. A measured disparity is an
estimate, not a fact, so vfairness compares the whole confidence interval to a
configurable tolerance and returns a **three-state verdict** rather than a point
estimate dressed up as a yes or no: the disparity is *within tolerance*,
*exceeds* it, or the evidence is *inconclusive*. These are statistical evidence
states, not a final judgment of fairness (the API exposes them as `fair` /
`unfair` / `insufficient_evidence`); a disparity it cannot measure is reported
the same honest way, never as a silent pass.

> **Documentation, guides, and a live example gallery: [vfairness.validant.ai](https://vfairness.validant.ai)**

> ### First beta
>
> `0.1.0` is the first public beta, released 2026-10-02 and installable with
> `pip install vfairness`. It is a `0.x` release: the frozen public surface is
> stable and changes only through a deprecation cycle, while experimental
> surface may still move before `1.0.0`. See
> [Beta and API stability](#beta-and-api-stability).

## Why vfairness

- **Verdicts you can defend.** Where a metric carries an interval, the verdict is
  read from the whole interval rather than the point estimate, so a tight
  large-sample gap and a wide small-sample gap of the same size are treated
  differently, as they should be. A "within tolerance" result is then positive
  evidence of practical equivalence, not merely a failure to prove disparity.
  Bootstrap and Bayesian intervals, effect sizes and multiple-testing correction
  are all available, and the `*_with_ci` functions provide an interval for every
  metric that has one.

  **Know which route you are on.** `get_report(include_ci=True)` populates
  `metrics_with_ci` for three of the five default metrics
  (`demographic_parity_difference`, `equalized_odds_difference`,
  `equal_opportunity_difference`). `demographic_parity_ratio` and
  `predictive_parity_difference` come back as plain point estimates there, with
  no interval and no interval-derived verdict. For those two, call
  `disparate_impact_ratio_with_ci` and `predictive_parity_difference_with_ci`
  directly. This matters most for the four-fifths ratio, which is the statistic a
  regulator is most likely to ask about.
- **Fails closed, not silent.** An undefined group rate, or a run left with a
  single assessable group, does not return a number: the report comes back
  `assessable: False` with `fairness_score: None` and every metric listed under
  `not_assessable_metrics`, rather than a fabricated score. The two release
  gates behave the same way. `assert_fairness` refuses a metric whose value is
  NaN with "NOT MEASURABLE ... so the metric was never compared against
  threshold (fail closed)", and `ModelFairnessGate` blocks on a metric it could
  not compute rather than approving an unevaluated check.
- **The whole lifecycle, one library.** A six-stage pipeline covers
  preprocessing bias detection, in-processing constraints, post-processing
  calibration and thresholding, evaluation, operations (CI/CD gates, monitoring,
  drift), and reporting.
- **Beyond tabular models.** Dedicated surfaces for LLMs (counterfactual prompt
  testing, BBQ / BOLD / DecodingTrust benchmarks), agents and multi-agent
  systems, vision, and retrieval groundedness.
- **Explainability built in.** SHAP-family attributions, a fairness
  decomposition, counterfactuals, and faithfulness / stability diagnostics, so
  you can explain *why* a result is unfair, not only *that* it is.
- **Corroborated against the field.** A differential test suite checks vfairness
  against fairlearn and AIF360: where a metric is shared, the numbers agree
  within tolerance (the suite runs when those reference libraries are installed);
  where definitions legitimately differ, the difference is documented in
  [`docs/DIVERGENCES.md`](https://github.com/validantai/vfairness/blob/main/docs/DIVERGENCES.md).
- **Report-ready.** Self-contained SVG reports and dashboards render with no
  browser or JavaScript dependency.
- **Engineered for reliance.** A typed public API (`py.typed`), a large test
  suite with mutation and property-based testing, a versioned assessment
  methodology stamped into every report, and a security-reviewed egress guard.

## Installation

```bash
pip install vfairness
```

Requires Python 3.11 or newer. The `0.0.1` distribution that carried this name
before this release is our own early 4.5 KB placeholder from March 2026 with no
`FairnessAnalyzer` in it; it has been yanked, so a resolver will not select it.
If you pinned it, upgrade rather than reinstall.

The core installs on `numpy`, `pandas`, `scipy`, `scikit-learn`, and `requests`.
Optional capabilities are opt-in extras, so a lean install stays lean:

| Extra | Adds |
| --- | --- |
| `rendering` | SVG report and dashboard templates (Jinja2) |
| `viz` | Matplotlib / Seaborn charts |
| `dashboard` | Interactive Plotly dashboards |
| `training` | PyTorch fairness-aware losses and trainable calibrators |
| `llm` | LLM output scorers (VADER sentiment, profanity detection) |
| `causal` | Causal fairness graphs |
| `xai` | SHAP-family attributions and explainability diagnostics |
| `mcp` | Model Context Protocol server |
| `all` | Everything above |

```bash
pip install "vfairness[rendering,viz]"     # reports + charts
```

Requires Python 3.11 or newer.

## Quick start

```python
import numpy as np
from vfairness import FairnessAnalyzer, demographic_parity_difference_with_ci

# A hiring model that approves men more often than women.
rng = np.random.default_rng(7)
n = 2000
gender = rng.choice(["female", "male"], size=n)
y_true = rng.integers(0, 2, size=n)                  # the true outcome
selection_rate = np.where(gender == "male", 0.62, 0.38)
y_pred = (rng.random(n) < selection_rate).astype(int)  # the model's decisions

# One call for a full, statistically validated fairness report. On this data all
# five metrics are measurable, so the verdict is a plain pass/fail count, and the
# summary states the rows it was computed over.
analyzer = FairnessAnalyzer(y_true, y_pred, gender)
report = analyzer.get_report(include_ci=True)
print(report["assessment"]["summary"])
# -> 1/5 metrics within thresholds (data provenance: 2000 of 2000 rows assessed, 0 excluded for missing values, 0 in 0 group(s) withheld by the group-size floor, missing_strategy='exclude')

# Or a single metric, with the confidence interval that drives the verdict.
result = demographic_parity_difference_with_ci(y_true, y_pred, gender, random_state=0)
print(f"Demographic parity gap: {result.point_estimate:.3f}  "
      f"95% CI [{result.lower_bound:.3f}, {result.upper_bound:.3f}]  "
      f"significant={result.is_significant}")
# -> Demographic parity gap: 0.199  95% CI [0.157, 0.243]  significant=True

# The same metric through the analyzer carries the three-state verdict, and the
# verdict is read from the whole interval, not from the point estimate.
graded = analyzer.demographic_parity_difference(include_ci=True, random_state=0)
print(f"verdict={graded.verdict}")
# -> verdict=unfair

# On a 60-row slice the gap is smaller and the interval is far wider, so the
# same code declines to call it either way rather than reporting a pass.
small = FairnessAnalyzer(y_true[:60], y_pred[:60], gender[:60], min_group_size=5)
thin = small.demographic_parity_difference(include_ci=True, random_state=0)
lo, hi = thin.confidence_interval
print(f"60 rows: {thin.value:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  verdict={thin.verdict}")
# -> 60 rows: 0.029  95% CI [0.000, 0.292]  verdict=insufficient_evidence

# The third state at group level, on data that actually triggers it. Add 12
# nonbinary rows: too few to support a verdict, so instead of quietly dropping
# them or folding them into a pass, the report names them as insufficient
# evidence.
gender_3 = np.append(gender, ["nonbinary"] * 12)
y_true_3 = np.append(y_true, rng.integers(0, 2, size=12))
y_pred_3 = np.append(y_pred, rng.integers(0, 2, size=12))

assessment = FairnessAnalyzer(y_true_3, y_pred_3, gender_3).get_report()["assessment"]
print(assessment["summary"])
# -> 1/5 metrics within thresholds (1 group(s) with insufficient evidence, excluded from the verdict: ['nonbinary']) (data provenance: 2012 of 2012 rows assessed, 0 excluded for missing values, 12 in 1 group(s) withheld by the group-size floor, missing_strategy='exclude')
print(assessment["insufficient_evidence_groups"][0]["verdict"])
# -> insufficient_evidence
```

The [Getting Started guide](https://vfairness.validant.ai/getting-started/) walks
through regression and ranking audits, interventions, calibration, monitoring,
and CI/CD gates.

## What is inside

**Metric families**, each defined against its primary literature source and
tested against hand-derived values and cross-library references:

- **Independence** (allocation): demographic parity, disparate impact ratio
  (the four-fifths statistic).
- **Separation** (error parity): equal opportunity, equalized odds, predictive
  equality, false-negative-rate parity, accuracy parity.
- **Sufficiency** (calibration): predictive parity, negative predictive value
  parity, calibration difference, integrated calibration index,
  multicalibration.
- **Continuous, decision-utility, and integrity**: pricing disparity, net
  benefit parity, conditional adverse impact, and proxy-leakage audits.
- **Ranking**: exposure and representation parity over ranked output.

**Fifteen sub-packages**: the six-stage pipeline (`preprocessing`,
`in_processing`, `post_processing`, `evaluation`, `operations`, `rendering`),
eight specialized surfaces (`llm`, `agents`\*, `multi_agent`\*, `xai`,
`vision`\*, `legal`\*, `mcp`\*, `validity`\*), and a cross-cutting `net` egress
guard.

> \* **Open today; not guaranteed to remain part of this distribution.** The
> surfaces marked with an asterisk (`agents`, `multi_agent`, `vision`, `legal`,
> `mcp`, `validity`) are specialized, research-stage capabilities. As they
> mature into validated, production-grade tooling, some may be taken forward as
> part of the commercial Validant platform rather than this open-source
> distribution, and may therefore be withdrawn from, or not included in, future
> releases. We say so here rather than later, so that you can plan any
> dependency on them with that in mind.
>
> This cannot affect what has already been published: every release made
> available under Apache-2.0 stays available under those terms permanently,
> including the right to use, modify, fork, and redistribute it. The core
> pipeline, the metric families, and the `llm`, `xai`, and `net` surfaces are
> not covered by this notice and remain open source.

## Documentation

Everything is on **[vfairness.validant.ai](https://vfairness.validant.ai)**:

- [Getting Started](https://vfairness.validant.ai/getting-started/): install, first audit, interventions, monitoring.
- [API Reference](https://vfairness.validant.ai/api-reference/): every public function and class.
- [Concepts](https://vfairness.validant.ai/concepts/): the fairness definitions, the impossibility results, and how the library reasons about them.
- [Sample Assessments](https://vfairness.validant.ai/sample-assessment/): full walkthroughs for hiring, lending, LLM, and ranking audits.
- [SVG Gallery](https://vfairness.validant.ai/svg-gallery/): every report and chart template with live previews.
- [Business Guide](https://vfairness.validant.ai/business-guide/): why algorithmic fairness matters, in plain language.

## From the library to the platform

Fairness and explainability are genuinely hard: choosing the right metric for a
use case, reading a confidence interval correctly, meeting a regulatory
threshold, and turning a result into a decision are expert work. **vfairness is
the open engine underneath [validant.ai](https://validant.ai), a continuous
digital-trust assurance platform that makes that work approachable.**

Where the library gives you the primitives, the platform gives you the guided
path: a visual workspace that recommends the right fairness definition for your
use case and jurisdiction, runs the assessment for you, explains each result in
plain language, tracks trust over time, and produces independently verifiable,
tamper-evident assurance reports and credentials. It is built so a product
manager, a compliance lead, or a data scientist can each take the pathway that
fits them, without first becoming a fairness researcher.

The goal of publishing this library and building that platform is the same: to
make rigorous AI fairness and explainability something every team can actually
do, not a specialty reserved for a few. Use the library directly when you want
full control in code; reach for the platform at
[validant.ai](https://validant.ai) when you want the assessment, the
explanation, and the assurance handled for you.

## Beta and API stability

vfairness follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
`0.1.0` is the first beta, published to PyPI on 2026-10-02 from the `v0.1.0`
tag on `validantai/vfairness`. The API is
not frozen until `1.0.0`, so a `0.x` minor release may still change experimental
surface. The frozen public surface
(`FairnessAnalyzer`, `MetricResult`, and the core classification, regression, and
ranking metric functions) is already treated as stable and changes only through a
one-minor-version deprecation warning, with removals reserved for `1.0.0`. See
[`docs/API_STABILITY.md`](https://github.com/validantai/vfairness/blob/main/docs/API_STABILITY.md) and the
[changelog](https://github.com/validantai/vfairness/blob/main/CHANGELOG.md).

## Citation

If you use vfairness in your work, please cite the software (see
[`CITATION.cff`](https://github.com/validantai/vfairness/blob/main/CITATION.cff)) and the underlying paper:

> Glinz & Company GmbH. *The Architecture of Digital Trust: A Multi-Level
> Framework for Bridging the AI Value Gap.* 2026 IEEE Swiss Conference on Data Science and AI (SDS), pp. 60-67.
> doi:10.1109/SDS70563.2026.00016

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](https://github.com/validantai/vfairness/blob/main/CONTRIBUTING.md) and
the [Code of Conduct](https://github.com/validantai/vfairness/blob/main/CODE_OF_CONDUCT.md), and report security issues per
[`SECURITY.md`](https://github.com/validantai/vfairness/blob/main/SECURITY.md).

## License

Apache License 2.0. See [`LICENSE`](https://github.com/validantai/vfairness/blob/main/LICENSE) and [`NOTICE`](https://github.com/validantai/vfairness/blob/main/NOTICE).
