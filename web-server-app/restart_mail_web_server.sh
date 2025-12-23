#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

PIDFILE="${PIDFILE:-/tmp/mail_web_server.pid}"
LOGFILE="${LOGFILE:-/tmp/mail_web_server.log}"

stop_existing() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping existing server (pid $pid)…"
      kill "$pid" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        if kill -0 "$pid" 2>/dev/null; then
          sleep 0.2
        else
          break
        fi
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "Process still running; sending SIGKILL…"
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$PIDFILE" || true
  fi

  # Best-effort cleanup if pidfile was missing/stale.
  pkill -f "python3 .*mail_web_server\.py" 2>/dev/null || true
}

start_server() {
  echo "Compiling…"
  bash ./check_mail_web_server.sh

  echo "Starting server on http://${HOST}:${PORT} …"
  : >"$LOGFILE"
  nohup env HOST="$HOST" PORT="$PORT" python3 mail_web_server.py >"$LOGFILE" 2>&1 &
  local pid=$!
  echo "$pid" >"$PIDFILE"

  # Wait for health endpoint
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS "http://${HOST}:${PORT}/healthz" >/dev/null 2>&1; then
      echo "OK: server is healthy (pid $pid)"
      echo "URL: http://${HOST}:${PORT}/"
      echo "Log: $LOGFILE"
      return 0
    fi
    sleep 0.2
  done

  echo "ERROR: server did not become healthy. Tail of log:"
  tail -n 80 "$LOGFILE" || true
  return 1
}

stop_existing
start_server
