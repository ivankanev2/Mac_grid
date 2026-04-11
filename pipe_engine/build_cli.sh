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
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPIPE_BUILD_GUI=OFF \
    -DPIPE_BUILD_CLI=ON

cmake --build . --parallel

echo ""
echo "=== CLI Build complete ==="
echo "  $BUILD_DIR/PipeEngine                       # generate demo pipes"
echo "  $BUILD_DIR/PipeEngine examples/simple_L.pipe   # from blueprint"
