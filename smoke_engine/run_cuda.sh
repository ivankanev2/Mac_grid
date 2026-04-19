#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
  echo "run_cuda.sh: nvcc was not found in PATH. Install the CUDA toolkit or use ./run.sh." >&2
  exit 1
fi

BUILD_DIR=${BUILD_DIR:-build_cuda}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSMOKE_ENABLE_CUDA=ON \
  -DSMOKE_ENABLE_VERBOSE_DIAGNOSTICS=OFF \
  "$@"
cmake --build "$BUILD_DIR" --config Release -j"$JOBS"
if [ -f "./$BUILD_DIR/Release/SmokeEngine.exe" ]; then
  "./$BUILD_DIR/Release/SmokeEngine.exe"
elif [ -f "./$BUILD_DIR/SmokeEngine.exe" ]; then
  "./$BUILD_DIR/SmokeEngine.exe"
elif [ -f "./$BUILD_DIR/Debug/SmokeEngine.exe" ]; then
  echo "run_cuda.sh: warning: Release executable not found; running Debug build." >&2
  "./$BUILD_DIR/Debug/SmokeEngine.exe"
else
  "./$BUILD_DIR/SmokeEngine"
fi
