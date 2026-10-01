"""Packaging hygiene guards.

These pin four packaging defects found in the fourth-iteration audit
(docs/audits/fourth-iteration-audit-2026-08-27.md). Each one shipped inside the
wheel or the wheel METADATA, so none of them is visible from the library's own
behaviour and none of them would be caught by the rest of the suite:

* #7 / #16  the auto-loaded pytest11 entry point pointed four levels deep into
            the package, so pytest had to execute vfairness/__init__.py (numpy,
            pandas, scipy, sklearn, ~1500 modules) at the start of EVERY pytest
            session in ANY project that had vfairness installed. Measured 3.2s
            vs 0.35s on a one-assert throwaway project.
* #18       scikit-learn was floored at >=1.0, a version the project has never
            passed on: at the lowest installable resolution the
            ExponentiatedGradient mitigation misses its own demographic-parity
            constraint (violation 0.0516 > tolerance 0.05) and reports it as
            enforced.
* #19       `vfairness-precommit check-config` exited 0 on an invalid YAML gate
            config when PyYAML was absent, and PyYAML was declared in no
            dependency and no extra, `all` included. pre-commit judges a hook by
            its exit code alone, so the warning it printed blocked nothing.
* #22       the CI coverage gate sat far below measured coverage, so it ratcheted
            nothing.

The tests read pyproject.toml and the workflow directly, because that is where
the defects live.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SRC = ROOT / "src"
FAIRNESS_CHECKS_WORKFLOW = ROOT / ".github" / "workflows" / "fairness-checks.yml"
REQUIREMENTS_TXT = ROOT / "requirements.txt"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _requirement_floor(spec: str) -> tuple:
    """Return the >= floor of a PEP 508 requirement as a comparable tuple."""
    match = re.search(r">=\s*([0-9][0-9A-Za-z.\-_]*)", spec)
    assert match, f"requirement {spec!r} declares no >= floor"
    parts = []
    for chunk in match.group(1).split("."):
        digits = re.match(r"\d+", chunk)
        parts.append(int(digits.group(0)) if digits else 0)
    return tuple(parts)


def _declared_dependency(name: str) -> str:
    deps = _pyproject()["project"]["dependencies"]
    for dep in deps:
        if re.match(rf"^{re.escape(name)}\b", dep, flags=re.IGNORECASE):
            return dep
    raise AssertionError(f"{name} is not declared in [project].dependencies: {deps}")


# ---------------------------------------------------------------------------
# #7 / #16: the auto-loaded pytest plugin must not import the package
# ---------------------------------------------------------------------------


class TestPytestEntryPointIsLightweight:
    def test_entry_point_target_is_a_top_level_module(self):
        """A dotted target forces Python to execute vfairness/__init__.py first.

        This is the whole defect: pytest imports every pytest11 entry point
        unconditionally, and importing `vfairness.operations.cicd.pytest_plugin`
        means importing `vfairness` first.
        """
        target = _pyproject()["project"]["entry-points"]["pytest11"]["vfairness"]
        assert "." not in target, (
            f"pytest11 entry point {target!r} is nested in a package; pytest would have to "
            "import every parent package (and so numpy/pandas/scipy/sklearn) at startup in "
            "every project that installs vfairness. Point it at a top-level module."
        )
        assert (SRC / f"{target}.py").is_file(), (
            f"pytest11 entry point {target!r} has no module at src/{target}.py"
        )

    def test_plugin_module_has_no_vfairness_import_at_module_scope(self):
        """Static guard: the fast path stays fast only while module scope is clean."""
        target = _pyproject()["project"]["entry-points"]["pytest11"]["vfairness"]
        tree = ast.parse((SRC / f"{target}.py").read_text(encoding="utf-8"))
        offenders = []
        for node in tree.body:  # module scope only, not function bodies
            if isinstance(node, ast.Import):
                offenders += [a.name for a in node.names if a.name.split(".")[0] == "vfairness"]
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "vfairness":
                    offenders.append(node.module)
        assert not offenders, (
            f"{target}.py imports {offenders} at module scope; move them inside the "
            "hook/fixture bodies so an unrelated pytest run does not pay for them"
        )

    def test_importing_the_plugin_does_not_import_vfairness(self):
        """Behavioural guard, run in a clean interpreter.

        This is the one that actually reproduces the user-visible cost: if
        `vfairness` lands in sys.modules, so do numpy, pandas, scipy and sklearn,
        and every pytest run in the installing project pays for them.
        """
        target = _pyproject()["project"]["entry-points"]["pytest11"]["vfairness"]
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            f"import {target}\n"
            "heavy = sorted(m for m in ('vfairness', 'numpy', 'pandas', 'scipy', 'sklearn')\n"
            "               if m in sys.modules)\n"
            "print('HEAVY=' + ','.join(heavy))\n"
            "print('MODULES=%d' % len(sys.modules))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        lines = dict(line.split("=", 1) for line in proc.stdout.strip().splitlines())
        assert lines["HEAVY"] == "", (
            f"importing the pytest11 entry point pulled in {lines['HEAVY']}; that cost is paid "
            "at the start of every pytest session in every project that installs vfairness"
        )
        # Measured 2026-08-27 on this interpreter: 50 modules bare, 262 after
        # importing the top-level shim, 2811 after importing the old nested path.
        assert int(lines["MODULES"]) < 800, (
            f"importing the pytest11 entry point loaded {lines['MODULES']} modules"
        )

    def test_wheel_ships_the_top_level_plugin_module(self):
        """`packages = [...]` alone would leave the entry point dead in the wheel."""
        target = _pyproject()["project"]["entry-points"]["pytest11"]["vfairness"]
        wheel_cfg = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]
        forced = wheel_cfg.get("force-include", {})
        assert forced.get(f"src/{target}.py") == f"{target}.py", (
            "the top-level pytest plugin module is not mapped into the wheel; "
            f"expected force-include['src/{target}.py'] == '{target}.py', got {forced!r}"
        )

    def test_legacy_plugin_import_path_still_works(self):
        """The documented `-p vfairness.operations.cicd.pytest_plugin` path is kept."""
        from vfairness.operations.cicd import pytest_plugin

        for name in (
            "pytest_configure",
            "pytest_collection_modifyitems",
            "pytest_terminal_summary",
            "fairness_gate",
            "assert_fairness_fn",
        ):
            assert hasattr(pytest_plugin, name), f"{name} no longer re-exported"


# ---------------------------------------------------------------------------
# #19: the shipped pre-commit hook must fail closed
# ---------------------------------------------------------------------------

_BLOCK_YAML = (
    "import sys\n"
    "class _Blocker:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'yaml' or name.startswith('yaml.'):\n"
    "            raise ImportError('PyYAML blocked for this test')\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Blocker())\n"
    "sys.modules.pop('yaml', None)\n"
)


def _run_check_config(path: Path, block_yaml: bool = False):
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        + (_BLOCK_YAML if block_yaml else "")
        + "from vfairness.operations.cicd.precommit import main\n"
        f"sys.exit(main(['check-config', {str(path)!r}]))\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)


class TestPrecommitCheckConfigFailsClosed:
    def test_missing_pyyaml_on_a_yaml_config_exits_non_zero(self, tmp_path):
        """A checker that cannot check must not report success.

        pre-commit reads the exit code and nothing else, so the old
        warning-and-continue meant every YAML gate config passed unchecked on a
        clean install.
        """
        cfg = tmp_path / "fairness_gate.yaml"
        cfg.write_text('metrics: [dp]\nthresholds:\n  dp: "not-a-number"\n', encoding="utf-8")

        proc = _run_check_config(cfg, block_yaml=True)

        assert proc.returncode != 0, (
            "check-config exited 0 with PyYAML absent; the hook approved an unvalidated "
            f"config. stdout:\n{proc.stdout}"
        )
        assert "PyYAML" in proc.stdout, (
            f"the failure must name the missing dependency. stdout:\n{proc.stdout}"
        )

    def test_missing_pyyaml_also_fails_on_an_otherwise_valid_config(self, tmp_path):
        """Fail closed means fail on VALID input too: the answer is unknown, not yes."""
        cfg = tmp_path / "fairness_gate.yaml"
        cfg.write_text("metrics: [dp]\nthresholds:\n  dp: 0.1\n", encoding="utf-8")

        proc = _run_check_config(cfg, block_yaml=True)

        assert proc.returncode != 0, (
            f"check-config exited 0 without being able to parse. stdout:\n{proc.stdout}"
        )

    def test_invalid_yaml_config_exits_non_zero(self, tmp_path):
        pytest.importorskip("yaml", reason="PyYAML is the thing under test here")
        cfg = tmp_path / "fairness_gate.yaml"
        cfg.write_text('metrics: [dp]\nthresholds:\n  dp: "not-a-number"\n', encoding="utf-8")

        proc = _run_check_config(cfg)

        assert proc.returncode != 0, f"invalid config passed. stdout:\n{proc.stdout}"
        assert "must be numeric" in proc.stdout

    def test_valid_yaml_config_exits_zero(self, tmp_path):
        pytest.importorskip("yaml", reason="PyYAML is the thing under test here")
        cfg = tmp_path / "fairness_gate.yaml"
        cfg.write_text("metrics: [dp]\nthresholds:\n  dp: 0.1\n", encoding="utf-8")

        proc = _run_check_config(cfg)

        assert proc.returncode == 0, f"valid config rejected. stdout:\n{proc.stdout}"

    def test_pyyaml_is_declared_in_an_installable_extra(self):
        extras = _pyproject()["project"]["optional-dependencies"]
        assert any(re.match(r"(?i)^pyyaml\b", dep) for dep in extras.get("cicd", [])), (
            "PyYAML must be declared in the `cicd` extra; the shipped "
            "vfairness-precommit console script cannot check YAML configs without it"
        )
        assert any("cicd" in dep for dep in extras["all"]), (
            "the `cicd` extra must be part of `all`, or `pip install vfairness[all]` still "
            "cannot validate a YAML fairness gate config"
        )


# ---------------------------------------------------------------------------
# #18: the declared dependency floors must be floors the project passes on
# ---------------------------------------------------------------------------

# The oldest release of each project that has a cp311 wheel; anything below this
# cannot be installed at all under requires-python >= 3.11, so declaring it is a
# claim about an environment that cannot exist.
FIRST_CP311_RELEASE = {
    "numpy": (1, 23, 2),
    "scipy": (1, 9, 2),
    "pandas": (1, 5, 0),
    "scikit-learn": (1, 1, 3),
}

# The floors measured on 2026-08-27 on Python 3.11, one variable at a time with
# the others held constant. Two are load-bearing:
#
#   scikit-learn 1.1.3 / 1.2.2 / 1.3.2 fail
#     tests/test_audit_wave6_reductions.py::TestExponentiatedGradientEnforces::
#     test_eg_dp_satisfies_constraint_on_separable_bias
#     with violation 0.0516 > 0.05: the ExponentiatedGradient mitigation reports
#     its own demographic-parity constraint as enforced when it is not.
#     1.4.0 is the oldest that passes.
#
#   pandas 1.5.0 / 1.5.3 / 2.0.3 / 2.1.4 fail the two agent memory-contamination
#     screens in tests/test_pulse_deferred_tail.py: below 2.2.0 the screen raises
#     IndexError inside agent_probe._memory_section and degrades to
#     "could not be computed". 2.2.0 fixes that but emits a pyarrow
#     DeprecationWarning at import, which breaks
#     tests/test_audit_wave4_llm.py::TestPlaceholderWarningsLazy::
#     test_import_survives_error_warnings, so the floor is 2.2.1.
#
# Raising a floor here is fine; LOWERING one means re-running those batteries at
# the new floor first.
TESTED_FLOORS = {
    "numpy": (1, 23, 2),
    "scipy": (1, 9, 2),
    "pandas": (2, 2, 1),
    "scikit-learn": (1, 4, 0),
}


class TestDependencyFloors:
    @pytest.mark.parametrize("name", sorted(TESTED_FLOORS))
    def test_floor_is_at_least_the_tested_floor(self, name):
        floor = _requirement_floor(_declared_dependency(name))
        assert floor >= TESTED_FLOORS[name], (
            f"{name} is floored at {floor}, below the tested floor {TESTED_FLOORS[name]}. "
            "A resolver in a constrained environment can legitimately install that version, "
            "and nothing in CI exercises it."
        )

    @pytest.mark.parametrize("name", sorted(FIRST_CP311_RELEASE))
    def test_floor_is_installable_under_requires_python(self, name):
        floor = _requirement_floor(_declared_dependency(name))
        assert floor >= FIRST_CP311_RELEASE[name], (
            f"{name}>={floor} predates the first release with a cp311 wheel "
            f"{FIRST_CP311_RELEASE[name]}, but requires-python is "
            f"{_pyproject()['project']['requires-python']}. The floor is unreachable, so it "
            "documents a support claim that was never true."
        )

    def test_requirements_txt_mirrors_the_declared_floors(self):
        """requirements.txt ships in the sdist, so a stale copy is a second claim.

        It drifted the moment the pyproject floors were corrected: it still said
        numpy>=1.21.0 / pandas>=1.3.0 / scipy>=1.7.0 / scikit-learn>=1.0 while the
        wheel METADATA said otherwise, which is two different support statements on
        one release.
        """
        declared = {}
        for dep in _pyproject()["project"]["dependencies"]:
            name = re.split(r"[<>=!~\[; ]", dep, maxsplit=1)[0]
            declared[name.lower()] = _requirement_floor(dep)

        mirrored = {}
        for raw in REQUIREMENTS_TXT.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0]
            mirrored[name.lower()] = _requirement_floor(line)

        assert mirrored == declared, (
            "requirements.txt and [project].dependencies disagree.\n"
            f"  pyproject.toml:   {declared}\n"
            f"  requirements.txt: {mirrored}"
        )

    def test_lowest_direct_resolution_is_exercised_in_ci(self):
        """A declared floor that CI never installs is an assertion, not a contract."""
        workflow = FAIRNESS_CHECKS_WORKFLOW.read_text(encoding="utf-8")
        assert "lowest-direct" in workflow, (
            "no lowest-direct resolution job in fairness-checks.yml; CI installs whatever pip "
            "resolves today, so the declared floors are never tested"
        )


# ---------------------------------------------------------------------------
# #14: the metadata that renders on the PyPI project page
# ---------------------------------------------------------------------------


class TestProjectMetadata:
    def test_every_declared_url_is_https_and_concrete(self):
        urls = _pyproject()["project"]["urls"]
        for label, url in urls.items():
            assert url.startswith("https://"), f"{label} URL is not https: {url}"
            for placeholder in ("your-org", "example.com", "TODO", "localhost"):
                assert placeholder not in url, f"{label} URL is a placeholder: {url}"

    def test_the_links_a_pypi_visitor_clicks_are_declared(self):
        urls = _pyproject()["project"]["urls"]
        for label in ("Homepage", "Documentation", "Repository", "Issues", "Changelog"):
            assert label in urls, f"project.urls is missing {label}; it renders on the PyPI page"

    def test_repository_url_points_at_the_public_source_repository(self):
        urls = _pyproject()["project"]["urls"]
        assert urls["Repository"] == "https://github.com/validantai/vfairness"
        assert urls["Issues"].startswith(urls["Repository"]), (
            "Issues must live under the same repository the Source button opens"
        )


# ---------------------------------------------------------------------------
# #22: the coverage gate must actually ratchet
# ---------------------------------------------------------------------------

# Measured on 2026-08-27 over the full suite: 52.38% line+branch coverage with the
# local extras set, and 51.67% recomputed with the causal subsystem's covered lines
# removed, which approximates the extras fairness-checks.yml installs. The gate
# sits under that with a few points of headroom for platform variance, so ordinary
# work is not blocked while a real regression still trips it. Raise this together
# with the workflow whenever the measured figure moves up; never lower it to make a
# red run green.
COVERAGE_GATE_MINIMUM = 48


class TestCoverageGateRatchets:
    def test_ci_gate_is_close_to_measured_coverage(self):
        workflow = FAIRNESS_CHECKS_WORKFLOW.read_text(encoding="utf-8")
        matches = re.findall(r"--cov-fail-under=(\d+)", workflow)
        assert matches, "fairness-checks.yml declares no --cov-fail-under gate"
        for value in matches:
            assert int(value) >= COVERAGE_GATE_MINIMUM, (
                f"--cov-fail-under={value} is far below measured coverage, so the gate cannot "
                "catch a regression. Re-measure and re-baseline it instead of lowering this."
            )
