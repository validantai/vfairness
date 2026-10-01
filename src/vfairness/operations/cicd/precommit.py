"""
Pre-commit hooks for vfairness.

Provides hooks for the pre-commit framework to enforce fairness
documentation standards in ML projects.

Hooks:
    vfairness-check-config: Verify fairness config files are valid.
    vfairness-check-model-card: Check model cards include fairness sections.

Usage in .pre-commit-config.yaml::

    repos:
      - repo: https://github.com/your-org/vfairness
        rev: v0.0.9
        hooks:
          - id: vfairness-check-config
          - id: vfairness-check-model-card
"""

import argparse
import json
import os
import sys
from typing import Optional, Sequence

REQUIRED_FAIRNESS_SECTIONS = [
    "fairness",
    "protected_attributes",
    "metrics",
    "bias",
    "demographic",
    "equit",
]

REQUIRED_CONFIG_KEYS = [
    "metrics",
    "thresholds",
]

# Metrics whose ATTAINABLE RANGE is known, so a bound outside it can be
# recognised as a gate that cannot gate. Matched on the EXACT normalised name,
# never as a substring: "ratio" is a substring of "cali[bratio]n_difference",
# and matching a family token anywhere inside a name is the cali-BRATIO-n bug
# class recorded in evaluation/vfairness_metrics/_metric_direction.py.
#
# A metric NOT in either table is deliberately left unchecked rather than
# guessed at: a gap between two continuous outcomes (a salary difference) is a
# legitimate lower-is-better magnitude far larger than 1. Three states, never
# two: in range, out of range, or range unknown.

# Differences between two RATES (probabilities), so the value lies in [0, 1] and
# a maximum at or above 1.0 can never be exceeded.
RATE_DIFFERENCE_METRICS = frozenset(
    {
        "demographic_parity_difference",
        "demographic_parity",
        "statistical_parity_difference",
        "statistical_parity",
        "equalized_odds_difference",
        "equalized_odds",
        "equal_opportunity_difference",
        "equal_opportunity",
        "predictive_parity_difference",
        "predictive_parity",
        "predictive_equality_difference",
        "predictive_equality",
        "true_positive_rate_difference",
        "false_positive_rate_difference",
        "false_negative_rate_difference",
        "true_negative_rate_difference",
        "selection_rate_difference",
        "accuracy_difference",
        "error_rate_difference",
        "disparate_impact_difference",
        # THE SIBLING TABLE CARRIED THESE AND THIS ONE DID NOT (2026-09-30).
        # ``_metric_direction.UNIT_INTERVAL_METRICS`` applies the identical rule at
        # RUNTIME, inside the gate, and the two tables have to hold the same names or
        # a bound one refuses is approved by the other. Each of these is a max - min
        # difference of per-group quantities that each live in [0, 1], so the
        # difference does too: what puts a name in THIS table rather than the one
        # below is that its bound is a MAXIMUM, not the statistic's provenance.
        "fpr_parity_difference",
        "fnr_parity_difference",
        "accuracy_parity_difference",
        "negative_predictive_value_difference",
        # A max difference of per-group AUROCs, each in [0, 1].
        "auroc_parity",
    }
)

# Min/max parity ratios (the four-fifths family). The numerator is the smallest
# group rate and the denominator the largest, so the value lies in [0, 1]:
# a required minimum at or below 0.0 is met by every possible value, and one
# above 1.0 cannot be met by any.
MIN_MAX_RATIO_METRICS = frozenset(
    {
        "disparate_impact_ratio",
        "disparate_impact",
        "adverse_impact_ratio",
        "exposure_parity_ratio",
        # Same statistic as disparate_impact_ratio under the library's other name.
        "demographic_parity_ratio",
        # NOT a ratio, and in this table on purpose: its bound is a required
        # MINIMUM on a quantity confined to [0, 1] (the smallest per-group
        # accuracy, Sagawa et al. 2020), so the two refusals below are exactly the
        # right ones for it. Same reason as the note in the table above: the branch
        # is chosen by the DIRECTION of the bound.
        "worst_group_accuracy",
    }
)


def _normalize_metric_name(metric_name: object) -> str:
    """Canonicalise a metric name for EXACT table lookups.

    Same rule as ``evaluation/vfairness_metrics/_metric_direction._normalize``,
    duplicated here on purpose: this hook must stay importable (and fast) with
    nothing but the standard library, because pre-commit runs it on every commit
    in an environment that may not carry the analysis extras.
    """
    name = str(metric_name).strip().lower()
    name = name.replace("-", " ").replace("_", " ")
    name = "_".join(name.split())
    if name.endswith("_with_ci"):
        name = name[: -len("_with_ci")]
    return name


def _unbreachable_bound_reason(metric_name: object, bound: float) -> Optional[str]:
    """Why no value of ``metric_name`` can be graded against ``bound``, or None.

    BGL5 AUDIT, measured 2026-09-27. Before this function existed:
        {"metrics": ["demographic_parity_difference"],
         "thresholds": {"demographic_parity_difference": 1e9}}
        -> check_fairness_config exited 0, and ModelFairnessGate built from that
           same config answered GateDecision(approved=True) for a measured
           demographic_parity_difference of 0.90.
    After: the same config exits 1 printing "bound of 1e+09 ... lies in [0, 1]",
    while the healthy {"demographic_parity_difference": 0.1} config still exits 0
    and the zero-tolerance {"demographic_parity_difference": 0.0} config still
    exits 0 (a real policy: any non-zero magnitude exceeds it).

    Only the bound's TYPE was validated before, so a number no data can reach
    was the same as a real bound to this checker. That is the unbounded-metric
    defect the block below already refuses, arriving as a number instead of as
    an absence.
    """
    name = _normalize_metric_name(metric_name)

    if name in RATE_DIFFERENCE_METRICS:
        if bound >= 1.0:
            return (
                f"is {bound:g}, and '{name}' is a difference between two quantities that "
                f"each lie in [0, 1], so it lies in [0, 1]: no value can exceed this bound, "
                f"so the gate it configures can never fail. Use a real tolerance (0.05 to "
                f"0.10 is usual)"
            )
        if bound < 0.0:
            return (
                f"is {bound:g}, and no magnitude can meet a negative maximum, so nothing is "
                f"ever graded against it (the gate reports could-not-check and refuses every "
                f"deployment). Use a bound in [0, 1)"
            )
        return None

    if name in MIN_MAX_RATIO_METRICS:
        if bound <= 0.0:
            return (
                f"is {bound:g}, and '{name}' is a required MINIMUM on a quantity that is "
                f"never negative (a min/max rate ratio, or a per-group accuracy floor): "
                f"every possible value meets it, the worst one included, so the gate it "
                f"configures can never fail. Use a real floor (0.80 is the four-fifths "
                f"rule)"
            )
        if bound > 1.0:
            return (
                f"is {bound:g}, and '{name}' is graded against a required minimum and lies "
                f"in [0, 1]: no value can reach this floor, so the gate it configures can "
                f"never pass. Use a floor in (0, 1]"
            )
        return None

    return None


def check_fairness_config(filenames: Sequence[str]) -> int:
    """Verify that fairness configuration files are valid.

    Checks JSON/YAML files matching common fairness config patterns. A config
    passes only when it describes a gate that can actually gate something:

    - ``metrics`` and ``thresholds`` are both present;
    - ``metrics`` is a non-empty list of metric names;
    - ``thresholds`` is a mapping whose values are real, finite numbers
      (``bool`` is refused explicitly: it is a subclass of ``int``);
    - every metric in ``metrics`` has a threshold, and every threshold names a
      metric in ``metrics``;
    - for a metric whose attainable range is KNOWN, the bound lies inside that
      range, so the gate it configures can actually be breached (see
      :func:`_unbreachable_bound_reason`; a bound on a metric whose range is not
      known is left unchecked rather than guessed at).

    The last rule is the one that matters most: a configured metric with no
    threshold is never compared against anything, and a threshold whose metric
    is not gated is never read. Both look like a working config in the file.

    Args:
        filenames: List of filenames to check.

    Returns:
        0 if all valid, 1 if any issues found.
    """
    retval = 0

    for filename in filenames:
        if not _is_fairness_config(filename):
            continue

        try:
            with open(filename, "r") as f:
                if filename.endswith(".json"):
                    config = json.load(f)
                elif filename.endswith((".yml", ".yaml")):
                    try:
                        import yaml
                    except ImportError:
                        # FAIL CLOSED. pre-commit judges a hook purely by its exit
                        # code, so the WARNING this used to print blocked nothing:
                        # the hook `continue`d and returned 0, silently approving
                        # every YAML fairness gate config, valid or not, on any
                        # install without PyYAML. PyYAML is now declared in the
                        # `cicd` extra (and in `all`), but a user can still run the
                        # console script from an environment that lacks it, and a
                        # config checker that cannot check must not report success.
                        # DO NOT turn this back into a warning-and-continue.
                        print(
                            f"FAIL: {filename} - cannot validate: PyYAML is not installed. "
                            "Install it with: pip install 'vfairness[cicd]' "
                            "(or pip install PyYAML)"
                        )
                        retval = 1
                        continue

                    config = yaml.safe_load(f)
                else:
                    continue

            if not isinstance(config, dict):
                print(f"FAIL: {filename}: config must be a JSON/YAML object")
                retval = 1
                continue

            # Check for required keys
            missing = []
            for key in REQUIRED_CONFIG_KEYS:
                if key not in config:
                    missing.append(key)

            if missing:
                print(f"FAIL: {filename}: missing required keys: {', '.join(missing)}")
                retval = 1
            else:
                # FAIL CLOSED. Measured 2026-09-10, every one of these exited 0
                # from this checker:
                #   {"metrics": ["demographic_parity_difference",
                #                "equalized_odds_difference"],
                #    "thresholds": {"demographic_parity_difference": 0.1}}
                #       -- a configured metric with no bound. Run through
                #          ModelFairnessGate that config APPROVED a model whose
                #          equalized_odds_difference was 0.90.
                #   {"metrics": [...], "thresholds": {}}   -- no bound at all.
                #   {"metrics": [],    "thresholds": {}}   -- a gate that gates
                #          nothing; the gate itself refuses this one.
                #   {"thresholds": {"x": true}} -- bool is a subclass of int, so
                #          `isinstance(val, (int, float))` accepted True as a
                #          numeric threshold.
                # The key presence check above only asks whether the two words
                # appear. A config checker that cannot tell a real gate config
                # from an empty one certifies the empty one.
                metrics_value = config.get("metrics")
                if not isinstance(metrics_value, list):
                    print(
                        f"FAIL: {filename}: 'metrics' must be a list of metric names, "
                        f"got {type(metrics_value).__name__}"
                    )
                    retval = 1
                    continue
                if not metrics_value:
                    print(
                        f"FAIL: {filename}: 'metrics' is empty, so this config gates "
                        f"nothing; a gate that checks no metric cannot approve anything"
                    )
                    retval = 1
                    continue
                bad_names = [m for m in metrics_value if not isinstance(m, str) or not m.strip()]
                if bad_names:
                    print(
                        f"FAIL: {filename}: 'metrics' must contain metric names as strings, "
                        f"got {bad_names!r}"
                    )
                    retval = 1
                    continue

                thresholds = config.get("thresholds")
                if not isinstance(thresholds, dict):
                    print(
                        f"FAIL: {filename}: 'thresholds' must be a mapping of metric name "
                        f"to bound, got {type(thresholds).__name__}"
                    )
                    retval = 1
                    continue

                # Validate thresholds are numeric. `bool` is excluded
                # EXPLICITLY: it is a subclass of int, so True passed the old
                # isinstance test and would reach the gate as the number 1.
                for metric, val in thresholds.items():
                    if isinstance(val, bool) or not isinstance(val, (int, float)):
                        print(
                            f"FAIL: {filename}: threshold for '{metric}' must be numeric, got {type(val).__name__}"
                        )
                        retval = 1
                    elif val != val or val in (float("inf"), float("-inf")):
                        print(
                            f"FAIL: {filename}: threshold for '{metric}' is {val}, which "
                            f"no value can be compared against"
                        )
                        retval = 1
                    else:
                        # The bound's RANGE, not only its type. See
                        # _unbreachable_bound_reason for the measured before and
                        # after: a demographic_parity_difference maximum of 1e9
                        # exited 0 here and then APPROVED a 0.90 disparity at the
                        # gate.
                        reason = _unbreachable_bound_reason(metric, float(val))
                        if reason is not None:
                            print(f"FAIL: {filename}: threshold for '{metric}' {reason}.")
                            retval = 1

                # Every gated metric needs a bound, and every bound must name a
                # gated metric. A metric with no threshold is never compared
                # (the gate reports could-not-check and refuses approval); a
                # threshold whose metric is not in `metrics` is never read at
                # all, so it is a bound the author believes is enforced and is
                # not. Both are silent in the config file and loud in
                # production.
                unbounded = [m for m in metrics_value if m not in thresholds]
                if unbounded:
                    print(
                        f"FAIL: {filename}: no threshold for {', '.join(unbounded)}; "
                        f"a configured metric with no bound is never compared against "
                        f"anything. Add it to 'thresholds', or remove it from 'metrics'."
                    )
                    retval = 1
                orphans = [m for m in thresholds if m not in metrics_value]
                if orphans:
                    print(
                        f"FAIL: {filename}: threshold(s) for {', '.join(map(str, orphans))} "
                        f"name no metric in 'metrics', so they are never applied. Add the "
                        f"metric to 'metrics', or remove the threshold."
                    )
                    retval = 1

        except (json.JSONDecodeError, Exception) as e:
            print(f"FAIL: {filename}: parse error: {e}")
            retval = 1

    return retval


def check_model_card(filenames: Sequence[str]) -> int:
    """Check that model card files include fairness-related sections.

    Looks for markdown files matching common model card patterns and
    verifies they contain fairness-related content.

    Args:
        filenames: List of filenames to check.

    Returns:
        0 if all valid, 1 if any issues found.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: check_model_card. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    retval = 0

    for filename in filenames:
        if not _is_model_card(filename):
            continue

        try:
            with open(filename, "r") as f:
                content = f.read().lower()

            has_fairness = any(section in content for section in REQUIRED_FAIRNESS_SECTIONS)

            if not has_fairness:
                print(
                    f"FAIL: {filename}: model card does not contain a fairness section. "
                    f"Please add documentation about fairness considerations, "
                    f"protected attributes, and bias evaluation."
                )
                retval = 1

        except Exception as e:
            print(f"FAIL: {filename}: read error: {e}")
            retval = 1

    return retval


def _is_fairness_config(filename: str) -> bool:
    """Check if a file looks like a fairness config file."""
    basename = os.path.basename(filename).lower()
    patterns = ["fairness", "bias", "gate"]
    extensions = (".json", ".yml", ".yaml")
    return any(p in basename for p in patterns) and filename.endswith(extensions)


def _is_model_card(filename: str) -> bool:
    """Check if a file looks like a model card."""
    basename = os.path.basename(filename).lower()
    patterns = ["model_card", "model-card", "modelcard"]
    return any(p in basename for p in patterns) and filename.endswith((".md", ".markdown"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for pre-commit hooks."""
    parser = argparse.ArgumentParser(
        description="vfairness pre-commit hooks for fairness documentation."
    )
    subparsers = parser.add_subparsers(dest="command")

    # check-config
    config_parser = subparsers.add_parser(
        "check-config",
        help="Verify fairness config files are valid.",
    )
    config_parser.add_argument("filenames", nargs="*")

    # check-model-card
    card_parser = subparsers.add_parser(
        "check-model-card",
        help="Check model cards include fairness sections.",
    )
    card_parser.add_argument("filenames", nargs="*")

    args = parser.parse_args(argv)

    if args.command == "check-config":
        return check_fairness_config(args.filenames)
    elif args.command == "check-model-card":
        return check_model_card(args.filenames)
    else:
        # READINESS-6, 2026-09-10. FAIL CLOSED, the same rule this module already
        # applies everywhere else. pre-commit judges a hook purely by its exit
        # code, so returning 0 here made a MISCONFIGURED hook, one whose `args:`
        # entry is missing or misspelt so no subcommand arrives, pass every
        # commit silently while checking nothing. That is the shape of issue #19
        # recorded in tests/test_packaging_hygiene.py, one level up: not a check
        # that answered wrongly, a check that never ran and reported success.
        #
        # The two subcommands keep returning 0 when they receive NO FILENAMES,
        # deliberately: pre-commit passes only the staged files matching the
        # hook's `files:` pattern, so an empty list genuinely means there was
        # nothing of that kind in this commit.
        print(
            "FAIL: no subcommand given, so nothing was checked. This hook is "
            "misconfigured and is failing closed rather than reporting a pass it "
            "did not earn. Expected 'check-config' or 'check-model-card'.",
            file=sys.stderr,
        )
        parser.print_help(sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
