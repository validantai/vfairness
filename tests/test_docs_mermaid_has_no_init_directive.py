"""No Mermaid diagram on the docs site may carry its own %%{init}%% directive.

On 2026-10-01 the swim-lane diagram on the Concepts page failed to render on
6 of 8 page loads, leaving its raw source on the page, and failed on 0 of 8
once its directive was removed. The directive forced a font the site
no longer loads and a theme of its own. Every diagram takes its theme from the
one place that sets it for the whole site (initMermaid in js/main.js), so a
per-diagram directive is never needed and has been shown to break rendering.
"""

from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "docs" / "site"


def test_no_diagram_overrides_the_site_mermaid_config():
    offenders = [
        str(p.relative_to(SITE))
        for p in sorted(SITE.rglob("*.html"))
        if "%%{init" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert not offenders, f"per-diagram %%{{init}}%% directives: {offenders}"
