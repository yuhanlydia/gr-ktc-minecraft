#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOYAGER_DIR="${PROJECT_DIR}/third_party/voyager"
VOYAGER_COMMIT="55e45a880755d0c8c66ca7fb5fe7962ac8974f89"
PATCH_FILE="${PROJECT_DIR}/patches/voyager-local-1.19.patch"

if [[ ! -d "${VOYAGER_DIR}/.git" ]]; then
  mkdir -p "${PROJECT_DIR}/third_party"
  git clone https://github.com/MineDojo/Voyager.git "${VOYAGER_DIR}"
fi

git -C "${VOYAGER_DIR}" fetch origin "${VOYAGER_COMMIT}"
git -C "${VOYAGER_DIR}" checkout --detach "${VOYAGER_COMMIT}"

if git -C "${VOYAGER_DIR}" apply --check "${PATCH_FILE}"; then
  git -C "${VOYAGER_DIR}" apply "${PATCH_FILE}"
elif git -C "${VOYAGER_DIR}" apply --reverse --check "${PATCH_FILE}"; then
  echo "Voyager local patch is already applied."
else
  echo "Voyager tree is neither pristine nor already patched; refusing to overwrite it." >&2
  exit 1
fi

python -m pip install -e "${VOYAGER_DIR}"
(
  cd "${VOYAGER_DIR}/voyager/env/mineflayer"
  npm install
  npm --prefix mineflayer-collectblock run build
)

echo "Voyager ${VOYAGER_COMMIT} with GR-KTC local 1.19 patch is ready."
