#!/usr/bin/env bash
# Build pipe_fluid_engine's GUI viewer in ./build_gui
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR=build_gui
mkdir -p "${BUILD_DIR}"

# Prefer Ninja if available
GEN="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GEN="Ninja"
fi

cmake -S . -B "${BUILD_DIR}" -G "${GEN}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DPIPEFLUID_BUILD_GUI=ON

cmake --build "${BUILD_DIR}" --parallel

echo ""
echo "Done. Run with:"
echo "  ./${BUILD_DIR}/PipeFluidEngine"
echo "or with a blueprint:"
echo "  ./${BUILD_DIR}/PipeFluidEngine examples/demo_L.pipe"
