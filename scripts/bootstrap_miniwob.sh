#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
browsergym_root="$repo_root/third_party/browsergym"
miniwob_root="$repo_root/third_party/miniwob-plusplus"
browsergym_pin=9e779f087de9a65668b6974d11f9ce9816026e96
miniwob_pin=7fd85d71a4b60325c6585396ec4f48377d049838
python_bin=${PYTHON_BIN:-python}

checkout_pin() {
  local repository=$1
  local destination=$2
  local pin=$3
  if [[ ! -d "$destination/.git" ]]; then
    git clone "$repository" "$destination"
  fi
  git -C "$destination" fetch origin "$pin"
  git -C "$destination" checkout --detach "$pin"
  test "$(git -C "$destination" rev-parse HEAD)" = "$pin"
}

checkout_pin https://github.com/ServiceNow/BrowserGym.git "$browsergym_root" "$browsergym_pin"
checkout_pin https://github.com/Farama-Foundation/miniwob-plusplus.git "$miniwob_root" "$miniwob_pin"

if "$python_bin" -m pip --version >/dev/null 2>&1; then
  "$python_bin" -m pip install \
    -e "$browsergym_root/browsergym/core" \
    -e "$browsergym_root/browsergym/miniwob"
elif command -v uv >/dev/null 2>&1; then
  uv pip install --python "$python_bin" \
    -e "$browsergym_root/browsergym/core" \
    -e "$browsergym_root/browsergym/miniwob"
else
  echo "Neither pip nor uv is available for $python_bin" >&2
  exit 1
fi
"$python_bin" -m playwright install-deps chromium
"$python_bin" -m playwright install chromium

miniwob_url="file://$miniwob_root/miniwob/html/miniwob/"
printf 'BrowserGym: %s\n' "$browsergym_pin"
printf 'MiniWoB++: %s\n' "$miniwob_pin"
printf 'MINIWOB_URL=%s\n' "$miniwob_url"
