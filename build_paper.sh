#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/paper"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  latexmk -c main.tex
  # Keep main.bbl: it ships with the self-contained arXiv source.
  rm -f main.blg main.run.xml main.synctex.gz
elif command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
else
  echo "error: neither latexmk (TeX Live / MacTeX) nor tectonic is on PATH" >&2
  echo "       install one of: brew install --cask mactex-no-gui  OR  brew install tectonic" >&2
  exit 1
fi
