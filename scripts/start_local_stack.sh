#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="${ROOT_DIR}/runtime/server-1.19"
MINEFLAYER_DIR="${ROOT_DIR}/third_party/voyager/voyager/env/mineflayer"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

server_running=false
if [[ -f "${LOG_DIR}/minecraft-server.pid" ]]; then
  server_pid="$(tr -cd '0-9' <"${LOG_DIR}/minecraft-server.pid")"
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    server_running=true
  fi
fi
if [[ "${server_running}" != true ]]; then
  : >"${LOG_DIR}/minecraft-server.log"
  (
    cd "${SERVER_DIR}"
    exec java -Xms2G -Xmx3G -jar "${SERVER_DIR}/server.jar" nogui
  ) >"${LOG_DIR}/minecraft-server.log" 2>&1 &
  echo $! >"${LOG_DIR}/minecraft-server.pid"
fi

for _ in $(seq 1 120); do
  if grep -q 'Done (' "${LOG_DIR}/minecraft-server.log" 2>/dev/null; then
    break
  fi
  sleep 1
done
grep -q 'Done (' "${LOG_DIR}/minecraft-server.log"

if ! curl -fsS http://127.0.0.1:3000/health >/dev/null 2>&1; then
  (
    cd "${MINEFLAYER_DIR}"
    exec "${MINEFLAYER_DIR}/node_modules/.bin/node" index.js 3000
  ) >"${LOG_DIR}/mineflayer.log" 2>&1 &
  echo $! >"${LOG_DIR}/mineflayer.pid"
fi

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:3000/health >/dev/null 2>&1; then
    curl -fsS http://127.0.0.1:3000/health
    printf '\n'
    exit 0
  fi
  sleep 1
done

echo "Mineflayer bridge did not become healthy" >&2
exit 1
