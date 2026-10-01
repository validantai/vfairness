"""The published gallery artwork must be what this code renders today.

A-03, the last open row of docs/audits/row-level-fabrication-register-2026-08-27.md:
"Nothing ties the artwork to the code." The register recorded the consequence
precisely. At `927abdf` the gallery was 12 files stale in markup and 4 once
rasterised, and two live rendering defects were VISIBLE on the published page
while every test was green, because no test and no workflow ever compared the
92 committed SVGs against a fresh render. `grep -rn "gallery" .github/workflows/`
returned nothing.

That matters more here than stale artwork usually would. These files are the
product page: vfairness.validant.ai/svg-gallery/ is what a prospective user
looks at to decide whether the library reports honestly. A chart that shows an
all-clear the code no longer draws is a claim about the library, published, with
nothing to catch it.

WHY THIS IS A TEST AND NOT A WORKFLOW. A workflow only guards the repository it
lives in, and the register's own note is that the check must survive the export
into the public repository. Both `scripts/regenerate_gallery_examples.py` and the
92 images ship, so as a test it runs in both places, on every push, in the same
job as everything else.

WHY IT DRIVES THE GENERATOR RATHER THAN REIMPLEMENTING IT. The fixtures are the
hard part: 46 templates each need a plausible data dict. Importing `build_data`
and `build_specs` from the script means the comparison cannot drift from the
thing it checks. A reimplementation would be a second source of truth, and the
one that is wrong is the one nobody notices.

THE TIMESTAMP. Every rendered SVG stamps its own render time in the header, so a
byte comparison always fails and would have to be ignored, which is how a check
like this ends up disabled. It is normalised out, and both directions of that
normalisation are proved by controls at the bottom: a differing timestamp must
compare EQUAL, and a differing anything-else must NOT.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
GALLERY = LIB_ROOT / "docs" / "site" / "img" / "svg-gallery"
GENERATOR = LIB_ROOT / "scripts" / "regenerate_gallery_examples.py"

# The one line that legitimately differs between two renders of the same code.
#
# Deliberately NOT anchored on the x coordinate. The first version of this regex
# hardcoded `x="644"` and reported pareto_frontier as stale on its very first
# run: that chart is 708px wide instead of 644, so its stamp sits at x="744". An
# over-specific guard that cries wolf gets switched off, which is worse than not
# having it. Both fill colours are matched for the same reason; missing the
# second made a "real change" list come out ten times too long.
#
# What keeps it from being too broad is the payload: a full ISO date and time,
# inside a text element carrying one of the two stamp colours. A finding, a
# label or a metric value cannot match that shape.
_STAMP = re.compile(
    r'(<text[^>]*fill="#(?:93a0ab|aab2bb)"[^>]*>)'
    r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}"
    r"(</text>)"
)


def _normalise(svg: str) -> str:
    return _STAMP.sub(r"\1RENDER-TIME\2", svg)


def _generator():
    if not GENERATOR.exists():  # pragma: no cover - it ships with the repo
        pytest.skip("gallery generator not present")
    spec = importlib.util.spec_from_file_location("_gallery_generator", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_gallery_generator"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rendered() -> dict:
    """{filename: svg} for every template, both variants, rendered right now."""
    gen = _generator()
    specs = gen.build_specs(gen.build_data())
    out: dict[str, str] = {}
    for name, fn in specs.items():
        out[f"{name}.svg"] = fn(None)  # None: render_svg writes the explanation
        out[f"{name}_no_explanation.svg"] = fn("")  # "": panel suppressed
    return out


def test_every_published_gallery_file_matches_a_fresh_render(rendered):
    stale, missing = [], []
    for filename, fresh in sorted(rendered.items()):
        published = GALLERY / filename
        if not published.exists():
            missing.append(filename)
            continue
        if _normalise(published.read_text(encoding="utf-8")) != _normalise(fresh):
            stale.append(filename)
    assert not missing, (
        "these templates render but are not published, so the gallery is missing "
        "artwork the code produces:\n  " + "\n  ".join(missing)
    )
    assert not stale, (
        "the published artwork no longer matches what this code renders. Run\n"
        "  python scripts/regenerate_gallery_examples.py\n"
        "and commit the result, in the SAME change as the rendering edit:\n  " + "\n  ".join(stale)
    )


def test_no_published_file_is_orphaned(rendered):
    """The mirror direction: artwork on the page that no template produces."""
    orphans = sorted(p.name for p in GALLERY.glob("*.svg") if p.name not in rendered)
    assert not orphans, (
        "these files are published but no template renders them, so the page shows "
        "artwork with no code behind it:\n  " + "\n  ".join(orphans)
    )


def test_the_comparison_actually_covers_the_gallery(rendered):
    """NON-VACUITY. An empty spec map would pass both tests above in silence."""
    assert len(rendered) >= 80, f"only {len(rendered)} renders produced; the generator broke"
    published = list(GALLERY.glob("*.svg"))
    assert len(published) >= 80, f"only {len(published)} files found in {GALLERY}"


def test_the_normaliser_ignores_the_timestamp_and_nothing_else():
    """Both directions, because a normaliser that is too broad disables the test.

    The first assertion is what lets the check run at all. The second is what
    stops it from being a check that always passes.
    """
    stamp = '<text x="644" y="32" text-anchor="end" font-size="9" fill="#93a0ab">{}</text>'
    a = "<svg>" + stamp.format("2026-08-28 15:35") + "<text>Low disparity</text></svg>"
    b = "<svg>" + stamp.format("2026-09-07 14:57") + "<text>Low disparity</text></svg>"
    assert _normalise(a) == _normalise(b), "a differing render time must compare equal"

    c = "<svg>" + stamp.format("2026-09-07 14:57") + "<text>HIGH disparity</text></svg>"
    assert _normalise(b) != _normalise(c), "a differing finding must NOT be normalised away"

    # The other fill colour in use must be normalised too.
    other = '<text x="644" y="32" text-anchor="end" font-size="9" fill="#aab2bb">{}</text>'
    d = "<svg>" + other.format("2026-08-28 15:35") + "</svg>"
    e = "<svg>" + other.format("2026-09-07 14:57") + "</svg>"
    assert _normalise(d) == _normalise(e), "the second timestamp colour is not being matched"

    # A WIDER chart puts its stamp at a different x. Anchoring on x="644" made
    # this guard report pareto_frontier (708px wide, stamp at x="744") as stale
    # on its first run, which is the false positive that gets a guard disabled.
    wide = '<text x="744" y="32" text-anchor="end" font-size="9" fill="#93a0ab">{}</text>'
    f = "<svg>" + wide.format("2026-08-28 15:35") + "</svg>"
    g = "<svg>" + wide.format("2026-09-07 15:00") + "</svg>"
    assert _normalise(f) == _normalise(g), "a wide chart's stamp is not being matched"

    # And a date that is NOT a render stamp must survive, or the guard would
    # normalise away a genuine change to any dated content on a chart.
    dated = '<svg><text x="36" y="90" fill="#0f172a">2026-08-28 15:35</text></svg>'
    assert "2026-08-28" in _normalise(dated), "a non-stamp date must not be normalised"
