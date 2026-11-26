#!/usr/bin/env bash
# Download the K1 models (booster_assets robots/K1, booster-mjlab K1 MJCF) at the tested commits.
set -euo pipefail
cd "$(dirname "$0")/.."

# Only robots/K1 (21 MB of the repo's 367 MB).
if [ ! -d third_party/booster_assets ]; then
  git clone --filter=blob:none --no-checkout https://github.com/BoosterRobotics/booster_assets.git third_party/booster_assets
  git -C third_party/booster_assets sparse-checkout set robots/K1
fi
git -C third_party/booster_assets checkout -q 12a97516f57469078707afb587a5f84c1cb110d1

# K1 model used by the get-up task (robot.xml + meshes, MJCF from RCSSServerMJ, MIT).
if [ ! -d third_party/booster-mjlab ]; then
  git clone --filter=blob:none --no-checkout https://github.com/66Leslie/booster-mjlab.git third_party/booster-mjlab
  git -C third_party/booster-mjlab sparse-checkout set src/mjlab_playground/asset_zoo/robots/booster_k1/xmls third_party_licenses
fi
git -C third_party/booster-mjlab checkout -q 3c5b8e18f12eb7e3d20a2bd39f17529b13ef6c92
