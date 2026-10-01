# Mutation testing (VB-TEST-9)

Line and branch coverage tell you which code ran, not whether a test would
notice a wrong answer. For a fairness library, where a subtly wrong metric is
the worst failure, that distinction matters. Mutation testing deliberately
introduces small bugs (mutants) into the code and checks that the tests fail
(kill the mutant). A surviving mutant means a real test gap or an equivalent
mutant that cannot change behaviour.

We use [mutmut](https://mutmut.readthedocs.io/). Configuration is in
`setup.cfg` under `[mutmut]`.

## Running it

```bash
pip install "mutmut>=2,<3"
mutmut run          # mutate the configured scope, run the fast runner per mutant
mutmut results      # list survivors
mutmut show <id>    # see a specific surviving mutant
```

The configured scope is a small, pure module (`src/vfairness/_bands.py`) with a
fast, boundary-exhaustive runner (`tests/test_bands.py`) so a run finishes in
about a minute. Widen `paths_to_mutate` to cover more of the metric core, or
pass `--paths-to-mutate <path>`; broader runs are slow, so they run on demand
and weekly in CI (`.github/workflows/mutation.yml`), not as a blocking gate.

## Baseline (2026-08-10)

Scope `src/vfairness/_bands.py`: **57 of 58 mutants killed (98%)**, above the
80% target for the metric core.

The single survivor is an equivalent mutant: the `except Exception: return True`
guard in `_is_nan` (an unreachable defensive fallback for objects whose `!=`
raises). It cannot be killed by a behaviour test without contriving such an
object, so it is documented rather than chased.

Mutation testing also surfaced a real issue while establishing this baseline:
`risk_band` and `drift_severity` hardcoded their thresholds instead of using the
module's `RISK_BAND_THRESHOLDS` / `DRIFT_BAND_THRESHOLDS` constants, so mutating
those "single source of truth" constants changed nothing. The functions now read
the constants, which both removes the dead-constant smell and kills those mutants.

## Target

Ratchet toward >=80% mutation score on the core metric functions
(`evaluation`, `in_processing`, `post_processing`), the same way coverage
ratchets up. Triage each survivor: document equivalent mutants, add a test for
any real gap.
