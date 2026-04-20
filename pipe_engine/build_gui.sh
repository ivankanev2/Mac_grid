#!/bin/bash
# ============================================================
# build_gui.sh — Build and launch the Pipe Engine live viewer
# Run from the pipe_engine/ directory:
#   cd pipe_engine && ./build_gui.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_gui"

echo "=== Pipe Engine GUI Build ==="
echo "Build dir: $BUILD_DIR"

mkdir -p "$BUILD_DIR"

# `--fresh` clears stale CMake cache/config state, which matters when the
# workspace has been moved or copied from another machine.
cmake --fresh -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPIPE_BUILD_GUI=ON \
    -DPIPE_BUILD_CLI=ON

cmake --build "$BUILD_DIR" --parallel

echo ""
echo "=== Build complete ==="
echo ""
echo "To run the live viewer:"
echo "  $BUILD_DIR/PipeEngineGUI"
echo ""
echo "To load a blueprint on startup:"
echo "  $BUILD_DIR/PipeEngineGUI examples/simple_L.pipe"
echo ""

# Optionally launch immediately
if [[ "$1" == "--run" ]]; then
    cd "$BUILD_DIR"
    ./PipeEngineGUI "$2"
fi
