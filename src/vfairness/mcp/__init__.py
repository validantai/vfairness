"""
vfairness MCP server package (local-first).

`tools` holds the pure tool logic (no `mcp` dependency, importable and testable
on its own). `server` wraps it with FastMCP and requires the optional `mcp`
extra: install with ``pip install "vfairness[mcp]"`` and run ``vfairness-mcp``.

See docs/reference/mcp-tools-redesign.md (Track 2).
"""

# This package re-exports nothing at the top level: callers import the
# submodules directly (`vfairness.mcp.tools` / `vfairness.mcp.server`). The
# empty __all__ makes that public/private boundary explicit (VB-PKG-3).
__all__: list = []
