"""
Local-first vfairness MCP server (stdio).

Exposes the vfairness fairness battery as Model Context Protocol tools,
resources, and prompts, so any MCP client (Claude Desktop, Cursor, an IDE
assistant, a custom agent) can run rigorous fairness analysis on local data.

Local-first: every computation runs in this process on the file paths you pass.
Nothing is uploaded; the tools are read-only and analytical. This is the
on-ramp described in docs/reference/mcp-tools-redesign.md (Track 2).

Run:
    vfairness-mcp                 # after: pip install "vfairness[mcp]"
    python -m vfairness.mcp

Then point an MCP client at that command (stdio transport).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# The version range this module's only import (mcp.server.fastmcp) exists in.
# Keep in lockstep with the `mcp` extra in pyproject.toml.
_MCP_REQUIREMENT = "mcp>=1.2.0,<2"

# Re-raise as ImportError, NOT SystemExit: SystemExit derives from
# BaseException and escapes `except Exception`, so an import-time SystemExit
# killed any host process (pkgutil sweeps, autodoc, pytest collection) that
# merely introspected this module without the extra installed. The friendly
# install hint lives in main() for the CLI path.
_MCP_INSTALL_HINT = (
    "The vfairness MCP server needs the optional 'mcp' dependency "
    f"({_MCP_REQUIREMENT}, Python >= 3.10).\n"
    'Install it with:  pip install "vfairness[mcp]"'
)


def _installed_mcp_version() -> str:
    """Best-effort version of whatever `mcp` distribution is on the path."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("mcp")
    except PackageNotFoundError:
        pass
    except Exception:  # pragma: no cover - metadata backends vary
        return "unknown"
    # A source checkout or a namespace shim can be importable without
    # distribution metadata; fall back to the module attribute.
    try:
        import mcp as _mcp_pkg

        return str(getattr(_mcp_pkg, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        return "unknown"


def _mcp_incompatible_hint(missing: str) -> str:
    """Message for: `mcp` IS installed, but not a version that has FastMCP.

    Naming the real cause matters. The previous guard answered every
    ModuleNotFoundError with "install vfairness[mcp]", which is advice the
    reader has already followed when the true problem is that mcp 2.x dropped
    mcp.server.fastmcp. That sent people chasing a packaging ghost.
    """
    return (
        f"The vfairness MCP server needs {missing}, but the installed 'mcp' "
        f"package (version {_installed_mcp_version()}) does not provide it. "
        f"That module was removed in mcp 2.0, so this 'mcp' is incompatible "
        f"(too new) rather than missing.\n"
        f'Install a supported version with:  pip install "{_MCP_REQUIREMENT}"\n'
        f'or reinstall the extra, which now pins the same range:  pip install "vfairness[mcp]"'
    )


try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without a usable extra
    # Distinguish "the extra is not installed" from "the extra is installed at
    # an incompatible version". Only the first is fixed by installing it.
    _absent = getattr(exc, "name", "") or ""
    if _absent == "mcp" or _absent.startswith("mcp."):
        if _absent == "mcp":
            raise ImportError(_MCP_INSTALL_HINT) from exc
        raise ImportError(_mcp_incompatible_hint(_absent)) from exc
    raise

# Imported after the mcp-extra availability guard above, so a missing optional
# dependency raises the friendly ImportError instead of a bare ModuleNotFound.
from vfairness.mcp import tools  # noqa: E402

mcp = FastMCP("vfairness")


# --- Tools -----------------------------------------------------------------


@mcp.tool()
def triage_dataset(
    protected_attributes: List[str],
    target_column: str,
    data_path: Optional[str] = None,
    csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan a local dataset for bias across the protected attributes. Start here.

    Provide the data as a local CSV path (data_path) or inline CSV text (csv).
    """
    df = tools.load_dataframe(data_path=data_path, csv=csv)
    return tools.triage_dataset(df, protected_attributes, target_column)


@mcp.tool()
def measure_fairness(
    protected_attributes: List[str],
    prediction_column: str,
    target_column: str,
    data_path: Optional[str] = None,
    csv: Optional[str] = None,
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """Compute the vfairness fairness-metric battery (demographic parity,
    equalized odds, equal opportunity, predictive parity, per-group rates) for
    each protected attribute.

    min_group_size is the floor below which a group is too small to measure; the
    metric is then NOT MEASURABLE rather than a number. It is exposed here, and
    defaults to the library's own 30, because the underlying tool hardwired 2
    until 2026-09-10: a group of TWO people returned a demographic parity
    difference of 0.010 with zero warnings, where the library default on the
    same input returns NaN with six. The sibling tool below already exposed it.
    """
    df = tools.load_dataframe(data_path=data_path, csv=csv)
    return tools.measure_fairness(
        df,
        protected_attributes,
        prediction_column,
        target_column,
        min_group_size=min_group_size,
    )


@mcp.tool()
def detect_proxies(
    protected_attributes: List[str],
    data_path: Optional[str] = None,
    csv: Optional[str] = None,
    correlation_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Find features that act as proxies for the protected attributes: feature
    correlations and multi-hop proxy chains. Fully offline."""
    df = tools.load_dataframe(data_path=data_path, csv=csv)
    return tools.detect_proxies(
        df, protected_attributes, correlation_threshold=correlation_threshold
    )


@mcp.tool()
def analyze_intersectional(
    protected_attributes: List[str],
    prediction_column: str,
    target_column: str,
    data_path: Optional[str] = None,
    csv: Optional[str] = None,
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """Intersectional disparity across combinations of protected attributes
    (for example race x gender). Needs at least two protected attributes."""
    df = tools.load_dataframe(data_path=data_path, csv=csv)
    return tools.analyze_intersectional(
        df, protected_attributes, prediction_column, target_column, min_group_size=min_group_size
    )


@mcp.tool()
def suggest_mitigation(
    protected_attributes: List[str],
    target_column: str,
    data_path: Optional[str] = None,
    csv: Optional[str] = None,
    score_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommend bias mitigations: a pre-processing feature analysis (which
    proxies to suppress or transform) and, when a probability score column is
    given, post-processing threshold/calibration trade-offs."""
    df = tools.load_dataframe(data_path=data_path, csv=csv)
    return tools.suggest_mitigation(
        df, protected_attributes, target_column, score_column=score_column
    )


@mcp.tool()
def explain_decision(
    prediction_column: str,
    data_path: Optional[str] = None,
    csv: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Explain which features the model's decisions track (association of each
    feature with the prediction column). Model-free: it explains the observed
    predictions in the CSV, not a SHAP attribution of a specific model. Pair
    with detect_proxies to see whether the top drivers are also proxies."""
    df = tools.load_dataframe(data_path=data_path, csv=csv)
    return tools.explain_decision(df, prediction_column, feature_columns=feature_columns)


@mcp.tool()
def audit_agent(
    group_column: str,
    action_column: str,
    data_path: Optional[str] = None,
    csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit an agent / LLM decision log for action-allocation fairness: for
    each action the agent takes, does the rate differ across groups? Expects a
    log where each row is one decision with a group column and a categorical
    action (or outcome / tool-choice) column."""
    df = tools.load_dataframe(data_path=data_path, csv=csv)
    return tools.audit_agent(df, group_column, action_column)


# --- Resources -------------------------------------------------------------


@mcp.resource("vfairness://glossary")
def glossary() -> str:
    """Plain-language definitions of the fairness metrics and concepts."""
    return tools.glossary_markdown()


@mcp.resource("vfairness://legal-framing")
def legal_framing() -> str:
    """Informational legal framing (US 4/5ths rule, EU AI Act). Not legal advice."""
    return tools.jurisdictions_markdown()


# --- Prompts ---------------------------------------------------------------


@mcp.prompt()
def fairness_triage(data_path: str, protected_attributes: str, target_column: str) -> str:
    """Template: a full fairness triage of a local dataset."""
    return (
        f"Run a full fairness triage on the dataset at '{data_path}'. "
        f"Protected attributes: {protected_attributes}. Outcome column: {target_column}. "
        "Use triage_dataset first, then measure_fairness and detect_proxies, and if there "
        "are two or more protected attributes also analyze_intersectional. Summarize the "
        "top three fairness risks in plain language and cite the actual numbers."
    )


@mcp.prompt()
def explain_in_plain_language(audience: str = "a non-technical stakeholder") -> str:
    """Template: explain the latest fairness results for a given reader."""
    return (
        f"Explain the fairness results above in plain language for {audience}. "
        "Define each metric briefly (you can read the vfairness://glossary resource), say "
        "whether each disparity is practically meaningful, and avoid implying legal conclusions."
    )


def main() -> None:
    """Console entry point (vfairness-mcp): run the stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
