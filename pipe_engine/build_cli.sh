#!/bin/bash
# ============================================================
# build_cli.sh — Build the headless CLI mesh generator only
# Run from the pipe_engine/ directory:
#   cd pipe_engine && ./build_cli.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"

# `--fresh` clears stale CMake cache/config state, which matters when the
# workspace has been moved or copied from another machine.
cmake --fresh -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPIPE_BUILD_GUI=OFF \
    -DPIPE_BUILD_CLI=ON

cmake --build "$BUILD_DIR" --parallel

echo ""
echo "=== CLI Build complete ==="
echo "  $BUILD_DIR/PipeEngine                       # generate demo pipes"
echo "  $BUILD_DIR/PipeEngine examples/simple_L.pipe   # from blueprint"
