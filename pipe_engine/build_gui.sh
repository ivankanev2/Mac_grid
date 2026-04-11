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
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPIPE_BUILD_GUI=ON \
    -DPIPE_BUILD_CLI=ON

cmake --build . --parallel

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
