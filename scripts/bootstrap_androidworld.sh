#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ANDROIDWORLD_ROOT:-$ROOT/third_party/android_world}"
PIN="e3fea3ccc69787570e282c99573298f1c3019a34"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python is required" >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET")"
if [[ ! -d "$TARGET/.git" ]]; then
  git clone https://github.com/google-research/android_world.git "$TARGET"
fi

git -C "$TARGET" fetch origin "$PIN"
git -C "$TARGET" checkout --detach "$PIN"
ACTUAL="$(git -C "$TARGET" rev-parse HEAD)"
if [[ "$ACTUAL" != "$PIN" ]]; then
  echo "AndroidWorld checkout mismatch: $ACTUAL != $PIN" >&2
  exit 1
fi

python -m pip install -r "$TARGET/requirements.txt"
python -m pip install "$TARGET"

echo
echo "AndroidWorld pinned successfully: $ACTUAL"
echo "Checkout: $TARGET"
echo
echo "Before the first benchmark run, start the Pixel 6 / API 33 AndroidWorld emulator."
echo "The runner defaults to console port 5554 and searches these adb paths:"
echo "  ~/Android/Sdk/platform-tools/adb"
echo "  ~/Library/Android/sdk/platform-tools/adb"
echo "or pass --adb-path explicitly."
echo
echo "First-time app setup is performed by adding --perform-emulator-setup to the smoke run."
echo "This project uses local Qwen and does NOT require OPENAI_API_KEY/GCP_API_KEY."
