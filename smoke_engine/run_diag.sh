#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build_diag}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSMOKE_ENABLE_CUDA=OFF \
  -DSMOKE_ENABLE_VERBOSE_DIAGNOSTICS=ON \
  "$@"
cmake --build "$BUILD_DIR" -j"$JOBS"
"./$BUILD_DIR/SmokeEngine"
