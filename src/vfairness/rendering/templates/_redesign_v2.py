#!/usr/bin/env python3
"""
SVG Template Redesign Script: v1.0 → v2.0
============================================
Transforms all 29 SVG templates to the new crisp, minimal design:
  1. Flat white background (remove bluish gradient)
  2. Very subtle shadows → thin border strokes on cards
  3. Increased viewBox height for more breathing room
  4. Title / explanation shifted down for top whitespace
  5. Version comment updated
"""

import glob
import os
import re

TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))


def transform_template(filepath):
    """Apply v2.0 design system changes to a single SVG template."""
    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # 1. Replace gradient background fill with flat white
    content = content.replace('fill="url(#bg)"', 'fill="#ffffff"')
    content = content.replace('fill="url(#bgGrad)"', 'fill="#ffffff"')

    # 2. Remove the unused bg / bgGrad gradient definitions
    #    (keep validant gradients and headerGrad)
    # Standard bg gradient
    content = re.sub(
        r'\s*<linearGradient id="bg"[^>]*>\s*'
        r"<stop[^/]*/>\s*"
        r"<stop[^/]*/>\s*"
        r"</linearGradient>",
        "",
        content,
    )
    # training_analysis_report bgGrad
    content = re.sub(
        r'\s*<linearGradient id="bgGrad"[^>]*>\s*'
        r"<stop[^/]*/>\s*"
        r"<stop[^/]*/>\s*"
        r"</linearGradient>",
        "",
        content,
    )

    # 3. Replace heavy shadow filters with very subtle versions
    # Standard shadow: stdDeviation=4 → 1, flood-opacity=0.14 → 0.05
    content = re.sub(
        r'<filter id="shadow">\s*'
        r"<feDropShadow[^/]*/>\s*"
        r"</filter>",
        '<filter id="shadow">'
        '<feDropShadow dx="0" dy="0.5" stdDeviation="1" flood-opacity="0.05"/>'
        "</filter>",
        content,
    )
    # training_analysis_report: sh
    content = re.sub(
        r'<filter id="sh">\s*'
        r"<feDropShadow[^/]*/>\s*"
        r"</filter>",
        '<filter id="sh">'
        '<feDropShadow dx="0" dy="0.5" stdDeviation="0.8" flood-opacity="0.04"/>'
        "</filter>",
        content,
    )
    # training_analysis_report: sh2
    content = re.sub(
        r'<filter id="sh2">\s*'
        r"<feDropShadow[^/]*/>\s*"
        r"</filter>",
        '<filter id="sh2">'
        '<feDropShadow dx="0" dy="0.3" stdDeviation="0.5" flood-opacity="0.03"/>'
        "</filter>",
        content,
    )

    # 4. Add thin border stroke to card rects that use shadow filters
    #    (only where stroke is not already present)
    for filt in ["shadow", "sh", "sh2"]:
        pattern = rf'fill="#fff" filter="url\(#{filt}\)"'
        # Only add stroke if not already present
        if re.search(pattern, content):
            content = re.sub(
                pattern,
                f'fill="#fff" stroke="#eaecef" stroke-width="0.5" filter="url(#{filt})"',
                content,
            )

    # 5. Add thin border to outer background rect for crispness
    # The outer rect: <rect width="W" height="..." rx="12" fill="#ffffff"/>
    # Add a very faint border
    content = re.sub(
        r'(rx="12" fill="#ffffff"/>)',
        r'rx="10" fill="#ffffff" stroke="#f0f0f0" stroke-width="0.5"/>',
        content,
        count=1,  # Only the first (background) rect
    )

    # 6. Increase viewBox height by 30 for bottom breathing room
    def increase_static_viewbox(m):
        w, h = int(m.group(1)), int(m.group(2))
        return f'viewBox="0 0 {w} {h + 30}"'

    # Static viewBox (most templates)
    content = re.sub(
        r'viewBox="0 0 (\d+) (\d+)"',
        increase_static_viewbox,
        content,
    )

    # Dynamic viewBox (alert_timeline: {{ 200 + entries|length * 40 }})
    content = re.sub(
        r'viewBox="0 0 (\d+) \{\{ (\d+)',
        lambda m: f'viewBox="0 0 {m.group(1)} {{{{ {int(m.group(2)) + 30}',
        content,
    )

    # 7. Increase outer background rect height to match viewBox
    # For static height backgrounds: match the first <rect with fill="#ffffff"
    # that has a plain integer height
    bg_rect_found = False

    def increase_bg_rect_static(m):
        nonlocal bg_rect_found
        if bg_rect_found:
            return m.group(0)
        bg_rect_found = True
        w, h = m.group(1), int(m.group(2))
        return f'<rect width="{w}" height="{h + 30}"'

    content = re.sub(
        r'<rect width="(\d+)" height="(\d+)"',
        increase_bg_rect_static,
        content,
    )

    # For dynamic height (alert_timeline):
    # <rect width="780" height="{{ 200 + entries|length * 40 }}"
    content = re.sub(
        r'(<rect width="\d+" height="\{\{ )(\d+)( \+)',
        lambda m: f"{m.group(1)}{int(m.group(2)) + 30}{m.group(3)}",
        content,
        count=1,
    )

    # 8. Move title and explanation down for more top whitespace
    # Title: y="36" → y="42" (specific to title pattern)
    content = re.sub(
        r'(<text x="\d+" y=")36(" text-anchor="middle" font-size="16" font-weight="700" fill="#1e293b">)',
        r"\g<1>42\2",
        content,
    )

    # Explanation: y="50" → y="56"
    content = re.sub(
        r'(<text x="\d+" y=")50(" text-anchor="middle" font-size="10" fill="#64748b">\{\{ explanation \}\})',
        r"\g<1>56\2",
        content,
    )

    # Validant icon: y="12" → y="16" (a bit lower to match title)
    content = re.sub(
        r'(<use href="#validant-icon" x="\d+" y=")12(" width="32" height="32")',
        r"\g<1>16\2",
        content,
    )

    # 9. Shift first content row down by 12px (y="56" → y="68")
    #    Only for the FIRST card rect at y="56" after the header
    # Match: <rect x="..." y="56" ..., first occurrence only
    first_content_shifted = False

    def shift_first_content(m):
        nonlocal first_content_shifted
        if first_content_shifted:
            return m.group(0)
        first_content_shifted = True
        return f'{m.group(1)}y="68"'

    content = re.sub(
        r'(<rect x="\d+" )y="56"',
        shift_first_content,
        content,
    )

    # 10. Update design system version comment
    content = content.replace(
        "vfairness SVG Design System v1.0",
        "vfairness SVG Design System v2.0",
    )

    # 11. Reduce outer corner radius from rx="12" to rx="10"
    #     (already done in step 5 for background rect)
    #     Also reduce card corner radii from rx="10" to rx="8"
    #     for crisper look
    # Card rects already have rx="10", keep them: they look fine.
    # The outer rect was changed to rx="10" in step 5.

    # Write if changed
    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False


def main():
    svg_files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "*.svg")))
    print(f"Found {len(svg_files)} SVG templates in {TEMPLATE_DIR}")
    print("=" * 60)

    changed = 0
    unchanged = 0
    for fpath in svg_files:
        name = os.path.basename(fpath)
        if transform_template(fpath):
            print(f"  [UPDATED] {name}")
            changed += 1
        else:
            print(f"  [no-op]   {name}")
            unchanged += 1

    print("=" * 60)
    print(f"Done: {changed} updated, {unchanged} unchanged")


if __name__ == "__main__":
    main()
