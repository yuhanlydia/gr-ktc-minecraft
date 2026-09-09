#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

curl -fsS -X POST http://127.0.0.1:3000/stop >/dev/null 2>&1 || true

stopped_pids=()
for name in mineflayer minecraft-server; do
  pid_file="${LOG_DIR}/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(tr -cd '0-9' <"${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
      stopped_pids+=("${pid}")
    fi
  fi
done

# Do not let an immediate restart mistake a gracefully terminating JVM for a
# live server. Minecraft can take several seconds to flush chunks on SIGTERM.
for pid in "${stopped_pids[@]}"; do
  for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
done
