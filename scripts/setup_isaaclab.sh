#!/usr/bin/env bash
# Pull the Isaac Lab 2.3.2 image and clone booster_train and booster_assets at the tested commits.
set -euo pipefail
cd "$(dirname "$0")/.."
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
clone() { [ -d "third_party/$1" ] || git clone "https://github.com/BoosterRobotics/$1.git" "third_party/$1"; git -C "third_party/$1" checkout -q "$2"; }
clone booster_train  83037fde138c49cd8d6459a4b592538e6e979ffb
clone booster_assets 12a97516f57469078707afb587a5f84c1cb110d1
