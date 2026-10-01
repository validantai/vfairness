#!/usr/bin/env python3
"""
Fix 6 SVG templates that have a different structure from the standard ones
and were not fully transformed by the initial redesign script.

Templates fixed:
  1. fairness_detailed_report.svg : bg-grad, card-shadow, header-grad group
  2. reweighting_comparison_report.svg: same group
  3. threshold_optimization_report.svg: same group
  4. method_comparison.svg: shadow filter with extended attrs, #f8fafc background
  5. tradeoff_analysis.svg: same as method_comparison
  6. training_report.svg: shadow filter with extended attrs, plus bg-grad if present
"""

import re
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent


# Helpers


def remove_bg_grad(content: str) -> str:
    """Remove the bg-grad linearGradient definition (2 stop colors)."""
    # Match the entire <linearGradient id="bg-grad" ...> ... </linearGradient> block
    pattern = r'\s*<linearGradient\s+id="bg-grad"[^>]*>.*?</linearGradient>'
    return re.sub(pattern, "", content, flags=re.DOTALL)


def replace_bg_grad_fill(content: str) -> str:
    """Replace fill="url(#bg-grad)" with fill="#ffffff"."""
    return content.replace('fill="url(#bg-grad)"', 'fill="#ffffff"')


def fix_card_shadow_filter(content: str) -> str:
    """
    In the card-shadow filter, change:
      stdDeviation="5" or stdDeviation="4" -> stdDeviation="1"
      flood-opacity="0.14" -> flood-opacity="0.05"
    Only within the <filter id="card-shadow"> block.
    """

    def _replace_card_shadow(m):
        block = m.group(0)
        block = re.sub(r'stdDeviation="[45]"', 'stdDeviation="1"', block)
        block = block.replace('flood-opacity="0.14"', 'flood-opacity="0.05"')
        return block

    return re.sub(
        r'<filter\s+id="card-shadow">.*?</filter>',
        _replace_card_shadow,
        content,
        flags=re.DOTALL,
    )


def replace_f8fafc_background(content: str) -> str:
    """
    Replace the background rect fill="#f8fafc" with fill="#ffffff".
    Only targets the full-width background rect (has rx="18" or is the first
    big rect after <defs>).
    """
    # The background rects look like:
    #   <rect width="700" height="590" rx="18" fill="#f8fafc"/>
    # We target rect elements that set the full background.
    return re.sub(
        r'(<rect\s+width="\d+"\s+height="\d+"\s+rx="\d+"\s+)fill="#f8fafc"',
        r'\1fill="#ffffff"',
        content,
    )


def fix_shadow_filter_extended(content: str) -> str:
    """
    Replace <filter id="shadow" x="-2%" y="-2%" width="104%" height="104%">
    with    <filter id="shadow">
    And inside that filter block change:
      stdDeviation="2" -> stdDeviation="1"
      flood-opacity="0.14" -> flood-opacity="0.05"
    """

    def _replace_shadow(m):
        block = m.group(0)
        # Strip the extended attributes from the opening tag
        block = re.sub(
            r'<filter\s+id="shadow"\s+x="[^"]*"\s+y="[^"]*"\s+width="[^"]*"\s+height="[^"]*">',
            '<filter id="shadow">',
            block,
        )
        block = re.sub(r'stdDeviation="[24]"', 'stdDeviation="1"', block)
        block = block.replace('flood-opacity="0.14"', 'flood-opacity="0.05"')
        return block

    return re.sub(
        r'<filter\s+id="shadow"[^>]*>.*?</filter>',
        _replace_shadow,
        content,
        flags=re.DOTALL,
    )


# Template-specific fixers


def fix_group1(filepath: Path) -> None:
    """Fix fairness_detailed_report, reweighting_comparison_report,
    threshold_optimization_report."""
    content = filepath.read_text()
    original = content

    content = replace_bg_grad_fill(content)
    content = remove_bg_grad(content)
    content = fix_card_shadow_filter(content)

    if content != original:
        filepath.write_text(content)
        print(f"  FIXED: {filepath.name}")
    else:
        print(f"  NO CHANGE: {filepath.name}")


def fix_group2(filepath: Path) -> None:
    """Fix method_comparison, tradeoff_analysis."""
    content = filepath.read_text()
    original = content

    content = replace_f8fafc_background(content)
    content = fix_shadow_filter_extended(content)

    if content != original:
        filepath.write_text(content)
        print(f"  FIXED: {filepath.name}")
    else:
        print(f"  NO CHANGE: {filepath.name}")


def fix_training_report(filepath: Path) -> None:
    """Fix training_report: hybrid of both patterns."""
    content = filepath.read_text()
    original = content

    content = fix_shadow_filter_extended(content)
    content = replace_bg_grad_fill(content)
    content = remove_bg_grad(content)

    if content != original:
        filepath.write_text(content)
        print(f"  FIXED: {filepath.name}")
    else:
        print(f"  NO CHANGE: {filepath.name}")


# Main


def main() -> None:
    print("Fixing special SVG templates...\n")

    # Group 1: bg-grad + card-shadow + header-grad templates
    print("Group 1, bg-grad / card-shadow / header-grad:")
    for name in [
        "fairness_detailed_report.svg",
        "reweighting_comparison_report.svg",
        "threshold_optimization_report.svg",
    ]:
        fix_group1(TEMPLATE_DIR / name)

    # Group 2: shadow with extended attrs + #f8fafc background
    print("\nGroup 2, shadow (extended attrs) / #f8fafc background:")
    for name in [
        "method_comparison.svg",
        "tradeoff_analysis.svg",
    ]:
        fix_group2(TEMPLATE_DIR / name)

    # Group 3: training_report: hybrid
    print("\nGroup 3, training_report (hybrid):")
    fix_training_report(TEMPLATE_DIR / "training_report.svg")

    print("\nDone.")


if __name__ == "__main__":
    main()
