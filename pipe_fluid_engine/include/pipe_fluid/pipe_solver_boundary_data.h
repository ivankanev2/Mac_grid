#pragma once

#include "pipe_fluid/pipe_boundary_field.h"

#include <cstdint>
#include <vector>

namespace pipe_fluid {

struct PipeSolverBoundaryData {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;

    std::vector<uint8_t> solidMask;
    std::vector<uint8_t> waterSolidMask;
    std::vector<uint8_t> openingMask;

    std::vector<float> uOpen;
    std::vector<float> vOpen;
    std::vector<float> wOpen;

    std::vector<float> wallNx;
    std::vector<float> wallNy;
    std::vector<float> wallNz;

    std::vector<PipeBoundaryTerminal> terminals;

    float minFaceOpen = 1.0f;
    int faceOpenCountLt099 = 0;
    int faceOpenCountLt050 = 0;
    int faceOpenCountClosed = 0;

    [[nodiscard]] bool valid() const noexcept {
        const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
        const std::size_t nu = static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
        const std::size_t nv = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1) * static_cast<std::size_t>(nz);
        const std::size_t nw = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1);
        return nx > 0 && ny > 0 && nz > 0 &&
               solidMask.size() == n && waterSolidMask.size() == n && openingMask.size() == n &&
               wallNx.size() == n && wallNy.size() == n && wallNz.size() == n &&
               uOpen.size() == nu && vOpen.size() == nv && wOpen.size() == nw;
    }

    [[nodiscard]] int idx(int i, int j, int k) const noexcept {
        return i + nx * (j + ny * k);
    }
};

PipeSolverBoundaryData buildSolverBoundaryData(const PipeBoundaryField& field);

} // namespace pipe_fluid