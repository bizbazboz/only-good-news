#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
API_BASE_URL="${API_BASE_URL:-http://localhost:${BACKEND_PORT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Install Node.js/npm first."
  exit 1
fi

if [[ ! -d "${FRONTEND_DIR}" ]]; then
  echo "Missing frontend directory: ${FRONTEND_DIR}"
  exit 1
fi

echo "Starting backend on :${BACKEND_PORT}"
"${PYTHON_BIN}" "${ROOT_DIR}/api.py" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "${FRONTEND_DIR}" && npm install)
fi

echo "Starting frontend on :${FRONTEND_PORT} (API_BASE_URL=${API_BASE_URL})"
(
  cd "${FRONTEND_DIR}"
  API_BASE_URL="${API_BASE_URL}" \
  NEXT_PUBLIC_API_BASE_URL="${API_BASE_URL}" \
  npm run dev -- --port "${FRONTEND_PORT}"
) &
FRONTEND_PID=$!

echo ""
echo "Backend:  ${API_BASE_URL}"
echo "Frontend: http://localhost:${FRONTEND_PORT}"
echo "Press Ctrl+C to stop both."
echo ""

wait -n "${BACKEND_PID}" "${FRONTEND_PID}"
