## Description

<!-- Brief description of the changes -->

## Fairness Checklist

Before merging, please verify the following:

### Data & Protected Attributes
- [ ] Protected attributes have been identified and documented
- [ ] Data distributions across groups have been reviewed
- [ ] Intersectional groups have been considered (e.g., gender x race)

### Metrics & Thresholds
- [ ] Fairness metrics have been computed (e.g., demographic parity, equalized odds)
- [ ] Thresholds have been documented and justified
- [ ] Confidence intervals have been reviewed where applicable

### Intersectional Analysis
- [ ] Intersectional disparity analysis has been performed
- [ ] Small-sample groups have been identified and acknowledged
- [ ] Per-intersection thresholds have been set where appropriate

### Testing
- [ ] Fairness tests pass (`pytest -m fairness`)
- [ ] CI/CD fairness gate passes
- [ ] No regressions in existing fairness metrics

### Documentation
- [ ] Model card includes fairness section
- [ ] Changes to fairness configuration are documented
- [ ] Any new thresholds or metrics are documented

## Fairness Gate Results

<!-- Paste the output of the fairness gate or report card here -->
<!-- You can generate this with: FairnessReportCard(decision).to_markdown() -->

```
<paste gate results>
```

## Additional Notes

<!-- Any additional context about fairness implications -->
