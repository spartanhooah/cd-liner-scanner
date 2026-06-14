#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

echo "Installing system dependencies..."
sudo pacman -S --needed sane python

echo "Adding $USER to the scanner group..."
sudo usermod -aG scanner "$USER"

echo "Creating virtual environment at $VENV..."
python -m venv "$VENV"

echo "Installing Python dependencies..."
"$VENV/bin/pip" install --quiet Pillow deskew img2pdf rapidfuzz

echo ""
echo "Done. Log out and back in for scanner group membership to take effect."
echo "Then run:  $SCRIPT_DIR/run.sh"
