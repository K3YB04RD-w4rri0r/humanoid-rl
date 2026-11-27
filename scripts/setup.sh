#!/usr/bin/env bash
# Download Booster's K1 model (booster_assets robots/K1) at the tested commit.
set -euo pipefail
cd "$(dirname "$0")/.."

# Only robots/K1 (21 MB of the repo's 367 MB).
if [ ! -d third_party/booster_assets ]; then
  git clone --filter=blob:none --no-checkout https://github.com/BoosterRobotics/booster_assets.git third_party/booster_assets
  git -C third_party/booster_assets sparse-checkout set robots/K1
fi
git -C third_party/booster_assets checkout -q 12a97516f57469078707afb587a5f84c1cb110d1
