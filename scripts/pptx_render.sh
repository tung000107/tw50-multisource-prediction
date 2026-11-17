#!/usr/bin/env bash
# Render a pptx to PDF + per-page PNG with the user-space LibreOffice (no sudo needed).
# usage: scripts/pptx_render.sh <deck.pptx> <out_dir> [dpi]
set -euo pipefail
SOFFICE="$HOME/opt/libreoffice/opt/libreoffice25.8/program/soffice"
PPTX="$1"; OUT="$2"; DPI="${3:-80}"; mkdir -p "$OUT"
"$SOFFICE" --headless --norestore --convert-to pdf --outdir "$OUT" "$PPTX" >/dev/null
PDF="$OUT/$(basename "${PPTX%.*}").pdf"; pdftoppm -r "$DPI" -png "$PDF" "$OUT/p"
echo "pdf: $PDF"; echo "pages: $(ls "$OUT"/p-*.png | wc -l)"
