#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
exec streamlit run frontend/app.py --server.port "${UI_PORT:-8501}"
