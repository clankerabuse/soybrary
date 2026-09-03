#!/usr/bin/env bash
# One-time setup for Soybrary (gallery + scraper).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

if ! command -v "${PYTHON}" >/dev/null; then
    echo "Python not found. Install Python 3.10+ and try again."
    exit 1
fi

if ! "${PYTHON}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10+ is required."
    exit 1
fi

if [[ ! -d "${VENV}" ]]; then
    echo "==> Creating virtual environment at ${VENV}/"
    "${PYTHON}" -m venv "${VENV}"
else
    echo "==> Using existing virtual environment at ${VENV}/"
fi

echo "==> Installing Python dependencies"
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install -r requirements.txt

echo "==> Installing Playwright Chromium (required for scraping)"
"${VENV}/bin/playwright" install chromium

echo ""
echo "Setup complete."
echo ""
echo "  Start gallery:   ./start.sh"
echo "                   (or: ${VENV}/bin/python server.py)"
echo ""
echo "  Run scraper:     ${VENV}/bin/python scraper.py"
echo "                   (or use the Scrape button in the gallery)"
echo ""
echo "  config.json ships with sensible defaults; edit it to customize."
echo "  Scraping runs headless — no browser window is opened during a scrape."
