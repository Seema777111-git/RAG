#!/usr/bin/env bash
# Creates .venv, installs dependencies and creates .env. Safe to re-run.
#   PYTHON=python3.11 scripts/setup.sh      choose the interpreter
#   TORCH_CPU=1 scripts/setup.sh            install the much smaller CPU-only PyTorch first
set -euo pipefail
cd "$(dirname "$0")/.."

pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
      echo "$candidate"; return
    fi
  done
}

PYTHON_BIN="$(pick_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.11 or newer is required but was not found. Install it, or run: PYTHON=/path/to/python scripts/setup.sh" >&2
  exit 1
fi
echo "Using $($PYTHON_BIN --version) ($PYTHON_BIN)"

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [ "${TORCH_CPU:-0}" = "1" ]; then
  echo "Installing CPU-only PyTorch ..."
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

python -m pip install -r requirements-dev.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

cat <<'MSG'

Setup complete.
  1. Put your key in .env:            ANTHROPIC_API_KEY=sk-ant-...
  2. Terminal 1 - start the API:      make api        (or scripts/run_api.sh)
  3. Terminal 2 - start the UI:       make ui         (or scripts/run_ui.sh)
  4. Open http://localhost:8501, upload a PDF in the Documents tab, then ask questions.
MSG
