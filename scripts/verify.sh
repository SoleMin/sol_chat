#!/usr/bin/env bash
# Project verification entry point. The Stop hook runs this before the agent
# may declare completion. Add your project's checks below; keep it fast.
set -euo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"

# Harness self-check: typecheck the hook scripts (skipped until npm install)
if [ -d node_modules ]; then
  npx tsc --noEmit
fi

# Python unit tests (no API key / model download needed; skipped until pip install)
PYTHON="${PYTHON:-python}"
if command -v "$PYTHON" >/dev/null 2>&1 && "$PYTHON" -c "import pytest, fastapi, openai" >/dev/null 2>&1; then
  "$PYTHON" -m pytest -q
else
  echo "verify: skipped pytest (run: pip install -r requirements.txt)"
fi

echo "verify: OK"
exit 0
