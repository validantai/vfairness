# Logo Assets for vfairness Documentation

This directory contains logo assets for the vfairness documentation site.

## Files Included

- `validant-logo.svg` - Full logo with text (for header)
- `validant-logo-white.svg` - White version for dark backgrounds
- `validant-icon.svg` - Icon only (square format)
- `validant-icon-white.svg` - White icon for dark backgrounds

## Usage

### In HTML (SVG)
```html
<img src="img/validant-logo.svg" alt="validant.ai" height="40">
```

### For PNG versions
If PNG versions are needed, convert the SVG files using:
```bash
# Using ImageMagick
convert -density 300 validant-logo.svg validant-logo.png

# Using Inkscape
inkscape validant-logo.svg --export-type=png --export-filename=validant-logo.png
```

## Brand Colors

- Primary Dark: `#1a3a4f`
- Primary: `#2c5f7c`
- Teal: `#0096a7`
- Cyan: `#00b4d8`
- Light Cyan: `#48cae4`

## Note

To use the actual validant.ai corporate logos, replace these placeholder SVG files with the official logo assets provided in the project root directory.
