#!/usr/bin/env python3
"""Fix symbol/marker viewBoxes that were accidentally modified by the redesign script."""

import glob
import os
import re

TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))


def fix_template(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # Fix symbol viewBoxes: "0 0 24 54" → "0 0 24 24" (icon symbols)
    content = re.sub(
        r'(<symbol[^>]*viewBox=")0 0 24 54(")',
        r"\g<1>0 0 24 24\2",
        content,
    )

    # Fix validant icon viewBox: "0 0 200 230" → "0 0 200 200"
    content = re.sub(
        r'(<symbol id="validant-icon" viewBox=")0 0 200 230(")',
        r"\g<1>0 0 200 200\2",
        content,
    )

    # Fix marker viewBoxes: "0 0 10 40" → "0 0 10 10"
    content = re.sub(
        r'(<marker[^>]*viewBox=")0 0 10 40(")',
        r"\g<1>0 0 10 10\2",
        content,
    )

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False


def main():
    svg_files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "*.svg")))
    print(f"Fixing viewBox issues in {len(svg_files)} templates...")
    fixed = 0
    for fpath in svg_files:
        name = os.path.basename(fpath)
        if fix_template(fpath):
            print(f"  [FIXED] {name}")
            fixed += 1
    print(f"Done: {fixed} files fixed")


if __name__ == "__main__":
    main()
