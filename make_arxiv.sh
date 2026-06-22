#!/usr/bin/env bash
# Assemble a self-contained arXiv submission tarball from paper/.
# The bundle is flat (main.tex at the root) and references figures/ as a
# relative subdirectory, so it extracts and compiles on arXiv without any
# parent-directory paths. Only the figure PDFs are shipped; the PNG
# previews and the compiled main.pdf are left out to keep the source small.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER_DIR="$ROOT_DIR/paper"

# Build first so main.bbl is current.
"$ROOT_DIR/build_paper.sh"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/figures"
cp "$PAPER_DIR/main.tex" "$PAPER_DIR/main.bbl" "$PAPER_DIR/references.bib" \
   "$PAPER_DIR/kore.sty" "$PAPER_DIR/kore.bst" "$STAGE/"
cp "$PAPER_DIR/figures/"*.pdf "$STAGE/figures/"

OUT="$ROOT_DIR/arxiv-submission.tar.gz"
tar -czf "$OUT" -C "$STAGE" .
echo "Wrote $OUT"
tar -tzf "$OUT" | sort
