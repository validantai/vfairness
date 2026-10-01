# Contributing to vfairness

Thanks for your interest in vfairness. It is free and open source under Apache-2.0, and it stays that way. Contributions are welcome.

## Ground rules

- Correctness is the product. This is a fairness measurement library, so a wrong metric is worse than a missing one. New or changed metrics need a test that pins the expected value (see `tests/test_synthetic_oracle.py` and `tests/test_property_invariants.py`).
- Keep changes focused and additive where possible. Read the surrounding code and preserve existing defensive checks.
- No network calls on the metric-computation path, and no telemetry (`tests/test_no_telemetry.py` guards this).

## Developer Certificate of Origin (DCO)

We use the DCO rather than a contributor license agreement. Sign off each commit to certify you wrote the change and can contribute it under the project license:

```
git commit -s -m "your message"
```

This adds a `Signed-off-by:` line. Contributions are accepted under Apache-2.0.

## Development setup

```
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU wheel; skip on macOS
pip install -e ".[dev,rendering,viz,monitoring,causal,xai,training,parity]"
VFAIRNESS_REQUIRE_BACKENDS=1 pytest -q -rs
```

That extras set matches the one CI installs, so a local run exercises the same surface. Install all of it: with the shorter `[dev,rendering,viz,monitoring,causal]` set that this file recommended until 2026-08-27, 60 tests skip rather than run, and the summary line looks identical either way. `VFAIRNESS_REQUIRE_BACKENDS=1` makes a missing backend a named failure instead of a silent subtraction, and `-rs` prints the reason for every remaining skip.

`uv sync --extra dev` is the lockfile-pinned alternative; add feature extras with further `--extra` flags.

Lint and types are pinned to match CI, so local findings are reproducible:

```
pip install ruff==0.16.2 mypy==2.3.0
```

Bump these in the repo-root `.github/workflows/quality.yml` and here in the same change.

Install the local lint hooks, from the repository root, so a bad commit fails before CI does:

```
sh scripts/githooks/install.sh
```

Use these instead of `pre-commit install`. The repo-root `.pre-commit-config.yaml` is present for reference, but installing it on a working tree shared by several concurrent agents races on the git index; the hooks in `scripts/githooks/` are the race-free equivalents.

## Before you open a pull request

- `VFAIRNESS_REQUIRE_BACKENDS=1 pytest -q -rs` passes locally. The bare `pytest` form
  is not equivalent: without the variable a missing optional backend silently skips
  its tests, so the checklist can be ticked on a run that never executed them.
- `ruff check src tests` is clean.
- `ruff format --check src tests` is clean (the whole tree is ruff-formatted).
- `mypy src` reports zero errors (it is a blocking gate, not advisory).
- New or changed lines are at least 80% covered: the PR-only diff-cover job enforces it.
- Public API changes update the snapshot in `tests/test_api_surface.py` and add a `CHANGELOG.md` entry.

## Support and sponsorship

vfairness is a community project with no service-level agreement. If it is useful to you or your organization, please consider sponsoring its development. Sponsorship funds maintenance; it does not buy private features or support guarantees. Commercial assurance and managed offerings live on the validant.ai platform, not in this library. Sponsor through [GitHub Sponsors](https://github.com/sponsors/validantai) (the validantai organization; the profile opens once GitHub approves it). Tiers and what each includes are on [Our Offering](https://vfairness.validant.ai/our-offering/).
