# Intentional metric divergences from reference libraries

vfairness is corroborated against the established fairness libraries (fairlearn,
AIF360, Aequitas). Where a metric shares a definition, the numbers agree within
tolerance and this is checked in CI (`tests/test_reference_parity.py` and
`scripts/emit_parity_evidence.py`). Where definitions legitimately differ, the
difference is documented here and each metric is tested against its own intended
definition, never blindly against another library.

## Demographic parity difference

- fairlearn `demographic_parity_difference` is the difference between the largest
  and the smallest group selection rate across all groups (a max minus min
  spread, always non-negative).
- AIF360 `statistical_parity_difference` is `P(Y_hat=1 | unprivileged) - P(Y_hat=1 | privileged)`
  (a signed difference against a chosen reference group).
- vfairness `demographic_parity_difference` reports the max minus min spread, so
  it aligns with fairlearn. For exactly two groups all three coincide in
  magnitude; the parity test uses a two-group case where they must agree.

## Reference group choice

Some AIF360 metrics require a designated privileged group and report a signed
value relative to it. vfairness prefers reference-free spreads (max minus min)
by default, and offers explicit privileged/disadvantaged identification
separately (`identify_privileged_groups`). A signed, reference-relative value
and a reference-free spread are different questions; both are valid.

## Ratio vs difference

Several metrics exist in both a difference form (0 is fair) and a ratio form
(1 is fair, or the four-fifths 0.8 threshold in the US employment overlay).
vfairness exposes both where applicable and does not silently convert one to the
other. The four-fifths rule is applied only in the US employment legal overlay,
never baked into the jurisdiction-neutral metric.

## Small groups

vfairness returns an explicit insufficient-evidence verdict for groups below the
reliability threshold rather than silently dropping them or emitting a point
estimate; reference libraries generally return a number regardless of group size.
Compare like for like: exclude such groups before a direct numeric comparison.

## Conditional demographic disparity (CDD)

- The canonical CDD of Wachter, Mittelstadt and Russell (2021), as
  operationalized for example in AWS SageMaker Clarify, is composition based
  and signed: within each stratum it compares the protected group's share of
  the REJECTED pool to its share of the ACCEPTED pool
  (`DD_k = P(group=d | rejected, k) - P(group=d | accepted, k)`), then
  size-weights across strata, so the sign names which group is disadvantaged.
- vfairness `conditional_demographic_disparity` currently reports the
  size-weighted within-stratum maximum SELECTION-RATE gap across protected
  groups: an unsigned, reference-free spread measured within the same strata.
  The two statistics answer related but different questions and can diverge
  materially on the same data (a 50 percent relative difference was measured in
  the 2026-08-22 audit), so a margin calibrated on the CDD literature or a
  Clarify run does not transfer to this statistic one to one.
- STATUS (2026-08-22): recorded as an open definition decision, not silently
  aligned, because sealed assessment cells gate on the current statistic and a
  formula change alters sealed verdicts. Until it is resolved, treat the
  vfairness value as a within-stratum selection-rate spread, and do not compare
  it numerically against composition-based CDD implementations. The comparison
  that produced the 50 percent figure above was executed on 2026-08-22; that
  audit record is internal and is not published with the library.
