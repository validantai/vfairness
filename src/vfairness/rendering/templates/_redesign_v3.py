#!/usr/bin/env python3
"""
SVG Template Redesign v3.0: Glass card shadows, uniform headers, title spacing.

Changes applied to ALL 29 templates:
1. Shadow: near-invisible → visible glass-style (stdDeviation=3, flood-opacity=0.10, flood-color=#94a3b8)
2. Title spacing: move first content row down 14px to fit multi-line explanations
3. Increase viewBox height by 14px to absorb the shift
4. Version comment: v2.0 → v3.0

Special fixes for training_report.svg and training_analysis_report.svg:
- Remove colored header bars (dark / green gradient)
- Replace with standard centered dark title on white background
"""

import glob
import os
import re

TEMPLATES_DIR = os.path.dirname(os.path.abspath(__file__))


def transform_standard(svg):
    """Transform standard v2.0 templates (27 templates)."""

    # 1. Shadow: upgrade to visible glass-style
    # Standard shadow
    svg = svg.replace(
        '<feDropShadow dx="0" dy="0.5" stdDeviation="1" flood-opacity="0.05"/>',
        '<feDropShadow dx="0" dy="1" stdDeviation="3" flood-color="#94a3b8" flood-opacity="0.10"/>',
    )
    # training_analysis_report sh filter
    svg = svg.replace(
        '<feDropShadow dx="0" dy="0.5" stdDeviation="0.8" flood-opacity="0.04"/>',
        '<feDropShadow dx="0" dy="1" stdDeviation="3" flood-color="#94a3b8" flood-opacity="0.10"/>',
    )
    # training_analysis_report sh2 filter
    svg = svg.replace(
        '<feDropShadow dx="0" dy="0.3" stdDeviation="0.5" flood-opacity="0.03"/>',
        '<feDropShadow dx="0" dy="0.5" stdDeviation="2" flood-color="#94a3b8" flood-opacity="0.06"/>',
    )
    # Catch any dy="2" variants from training_report
    svg = svg.replace(
        '<feDropShadow dx="0" dy="2" stdDeviation="1" flood-opacity="0.05"/>',
        '<feDropShadow dx="0" dy="1" stdDeviation="3" flood-color="#94a3b8" flood-opacity="0.10"/>',
    )

    # 2. Increase viewBox height by 14px
    # Match the FIRST viewBox in the <svg> tag only (not in <symbol>)
    def bump_viewbox(m):
        full = m.group(0)
        # Only modify if it's the opening <svg> tag's viewBox
        w, h = int(m.group(1)), int(m.group(2))
        # Skip small viewBoxes (symbols, icons)
        if w < 100 or h < 100:
            return full
        return full.replace(f'viewBox="0 0 {w} {h}"', f'viewBox="0 0 {w} {h + 14}"')

    # Only match the first <svg> tag's viewBox
    svg = re.sub(
        r'(<svg[^>]*?)viewBox="0 0 (\d+) (\d+)"',
        lambda m: (
            m.group(1) + f'viewBox="0 0 {m.group(2)} {int(m.group(3)) + 14}"'
            if int(m.group(2)) >= 100 and int(m.group(3)) >= 100
            else m.group(0)
        ),
        svg,
        count=1,  # Only the first match (the outer <svg>)
    )

    # Also increase the outer background rect height by 14
    svg = re.sub(
        r'(<rect[^>]*width="\d+"[^>]*height=")(\d+)(")',
        lambda m: (
            m.group(1) + str(int(m.group(2)) + 14) + m.group(3)
            if int(m.group(2)) >= 400  # Only large rects (the outer background)
            else m.group(0)
        ),
        svg,
        count=1,  # Only the first large rect
    )

    # 3. Move explanation text down 2px (y="56" → y="58")
    svg = svg.replace('explanation %}<text x="300" y="56"', 'explanation %}<text x="300" y="58"')

    # 4. Shift first content row from y="68" to y="82"
    # This creates 24px more space below the title for explanations
    svg = re.sub(r'y="68"', 'y="82"', svg)

    # 5. Update version comment
    svg = svg.replace("Design System v2.0", "Design System v3.0")

    return svg


def fix_training_report(svg):
    """Fix training_report.svg: remove dark header, normalize to standard title style."""

    # Remove the dark header-bg style
    svg = svg.replace(".header-bg { fill: #1a1a2e; }", ".header-bg { fill: none; }")

    # Remove the header rect itself
    svg = re.sub(
        r'<!-- Header -->\s*<rect class="header-bg"[^/]*/>\s*', "<!-- Header -->\n    ", svg
    )

    # Change title to dark text, centered
    svg = svg.replace(
        '<text class="title" x="20" y="38" fill="#ffffff">{{ title }}</text>',
        '<text x="450" y="36" text-anchor="middle" font-size="16" font-weight="700" fill="#1e293b" font-family="\'Inter\',\'Segoe UI\',system-ui,sans-serif">{{ title }}</text>',
    )

    # Move explanation after new title, centered
    svg = re.sub(
        r'{% if explanation %}<text x="450" y="52" text-anchor="middle" font-size="10" fill="#64748b">',
        '{% if explanation %}<text x="450" y="50" text-anchor="middle" font-size="10" fill="#64748b">',
        svg,
    )

    # Move timestamp to right-aligned, dark text
    svg = svg.replace(
        '<text class="small" x="880" y="38" text-anchor="end" fill="#adb5bd">{{ timestamp }}</text>',
        '<text x="880" y="36" text-anchor="end" font-size="9" fill="#94a3b8">{{ timestamp }}</text>',
    )

    # Add font-family to outer SVG if missing
    if "font-family" not in svg.split(">")[0]:
        svg = svg.replace(
            'viewBox="0 0 900',
            "font-family=\"'Inter','Segoe UI',system-ui,sans-serif\" viewBox=\"0 0 900",
        )

    return svg


def fix_training_analysis_report(svg):
    """Fix training_analysis_report.svg: remove green gradient header, normalize title."""

    # Remove the two header gradient rects
    svg = re.sub(
        r'<rect x="0" y="0" width="900" height="62" rx="14" fill="url\(#headerGrad\)"/>\s*'
        r'<rect x="0" y="14" width="900" height="48" fill="url\(#headerGrad\)"/>\s*',
        "",
        svg,
    )

    # Replace white title with standard centered dark title
    svg = svg.replace(
        '<text x="28" y="38" font-size="18" font-weight="700" fill="#fff">{{ title }}</text>',
        '<text x="450" y="36" text-anchor="middle" font-size="16" font-weight="700" fill="#1e293b">{{ title }}</text>',
    )

    # Fix timestamp (was light green on dark background)
    svg = svg.replace(
        '<text x="842" y="28" text-anchor="end" font-size="9" fill="#a7f3d0">{{ timestamp }}</text>',
        '<text x="860" y="28" text-anchor="end" font-size="9" fill="#94a3b8">{{ timestamp }}</text>',
    )
    svg = svg.replace(
        '<text x="842" y="42" text-anchor="end" font-size="9" fill="#a7f3d0">Task: {{ task_type }} | Constraint: {{ constraint_type }}</text>',
        '<text x="860" y="40" text-anchor="end" font-size="9" fill="#94a3b8">{{ task_type }} | {{ constraint_type }}</text>',
    )

    # Move validant icon position to standard
    svg = svg.replace(
        '<use href="#validant-icon" x="852" y="12" width="36" height="36" opacity="0.6"/>',
        '<use href="#validant-icon" x="856" y="8" width="32" height="32" opacity="0.5"/>',
    )

    # Move explanation to standard position
    svg = svg.replace(
        '{% if explanation %}<text x="450" y="70" text-anchor="middle" font-size="10" fill="#64748b">{{ explanation }}</text>{% endif %}',
        '{% if explanation %}<text x="450" y="50" text-anchor="middle" font-size="10" fill="#64748b">{{ explanation }}</text>{% endif %}',
    )

    return svg


def process_all():
    templates = sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.svg")))
    results = {}

    for path in templates:
        name = os.path.basename(path)
        with open(path, "r") as f:
            original = f.read()

        svg = original

        # Apply special fixes FIRST for the two training templates
        if name == "training_report.svg":
            svg = fix_training_report(svg)
        elif name == "training_analysis_report.svg":
            svg = fix_training_analysis_report(svg)

        # Apply standard transforms to ALL templates
        svg = transform_standard(svg)

        if svg != original:
            with open(path, "w") as f:
                f.write(svg)
            results[name] = "MODIFIED"
        else:
            results[name] = "UNCHANGED"

    print(f"\nProcessed {len(results)} templates:")
    modified = sum(1 for v in results.values() if v == "MODIFIED")
    print(f"  Modified: {modified}")
    print(f"  Unchanged: {len(results) - modified}")
    for name, status in sorted(results.items()):
        icon = "~" if status == "MODIFIED" else "."
        print(f"  {icon} {name}")


if __name__ == "__main__":
    process_all()
