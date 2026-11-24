#!/usr/bin/env bash
# Download the Booster K1 MuJoCo model from https://github.com/BoosterRobotics/booster_assets

set -euo pipefail

COMMIT=12a97516f57469078707afb587a5f84c1cb110d1
BASE=https://raw.githubusercontent.com/BoosterRobotics/booster_assets/$COMMIT/robots/K1
DEST="${1:-$(dirname "$0")/../src/data/assets/K1}"

mkdir -p "$DEST/meshes"
curl -fsSL "$BASE/K1_22dof.xml" -o "$DEST/K1_22dof.xml"
for f in $(grep -o 'file="[^"]*"' "$DEST/K1_22dof.xml" | cut -d'"' -f2); do
  curl -fsSL "$BASE/meshes/$f" -o "$DEST/meshes/$f"
done
echo "K1 model: $DEST/K1_22dof.xml ($(ls "$DEST/meshes" | wc -l) meshes, booster_assets @ ${COMMIT:0:7})"
