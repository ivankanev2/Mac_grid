#include "mac_smoke_sim.h"
#include <cmath>

static inline float jacobiUpdate(float b, float sumN, int count, float alphaInvDx2) {
    return (b + alphaInvDx2 * sumN) / (1.0f + alphaInvDx2 * (float)count);
}

void MAC2D::diffuseVelocityImplicit() {
    if (viscosity <= 0.0f || diffuseIters <= 0) return;

    const float invDx2 = 1.0f / (dx * dx);
    const float alphaInvDx2 = (viscosity * dt) * invDx2; // (ν dt)/dx^2
    if (alphaInvDx2 <= 0.0f) return;

    // Reuse u0/v0 as temp buffers (they already exist and are sized correctly)
    if (u0.size() != u.size()) u0 = u;
    if (v0.size() != v.size()) v0 = v;

    auto isFixedU = [&](int i, int j) {
        if (i == 0 || i == nx) return true; // outer walls
        // u face between (i-1,j) and (i,j)
        return isSolid(i - 1, j) || isSolid(i, j);
    };

    auto isFixedV = [&](int i, int j) {
        if (j == 0) {
            // bottom is wall except inside valve inflow
            return !(valveOpen && inValve(i));
        }
        if (j == ny) {
            // top is wall if closed; if openTop, treat as non-fixed (we re-apply BC after)
            return !openTop;
        }
        // v face between (i,j-1) and (i,j)
        return isSolid(i, j - 1) || isSolid(i, j);
    };

    // --- diffuse U ---
    for (int it = 0; it < diffuseIters; ++it) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                int id = idxU(i, j);
                if (isFixedU(i, j)) { u0[id] = u[id]; continue; }

                float sumN = 0.0f;
                int count = 0;

                if (i - 1 >= 0)  { sumN += u[idxU(i - 1, j)]; count++; }
                if (i + 1 <= nx) { sumN += u[idxU(i + 1, j)]; count++; }

                // free-slip at top/bottom for u diffusion: only count in-domain neighbors
                if (j - 1 >= 0)  { sumN += u[idxU(i, j - 1)]; count++; }
                if (j + 1 < ny)  { sumN += u[idxU(i, j + 1)]; count++; }

                float xNew = jacobiUpdate(u[id], sumN, count, alphaInvDx2);
                u0[id] = (1.0f - diffuseOmega) * u[id] + diffuseOmega * xNew;
            }
        }
        u.swap(u0);
        applyBoundary();
        if (valveOpen) applyValveBC();
    }

    // --- diffuse V ---
    for (int it = 0; it < diffuseIters; ++it) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idxV(i, j);
                if (isFixedV(i, j)) { v0[id] = v[id]; continue; }

                float sumN = 0.0f;
                int count = 0;

                if (i - 1 >= 0) { sumN += v[idxV(i - 1, j)]; count++; }
                if (i + 1 < nx) { sumN += v[idxV(i + 1, j)]; count++; }

                if (j - 1 >= 0)  { sumN += v[idxV(i, j - 1)]; count++; }
                if (j + 1 <= ny) { sumN += v[idxV(i, j + 1)]; count++; }

                float xNew = jacobiUpdate(v[id], sumN, count, alphaInvDx2);
                v0[id] = (1.0f - diffuseOmega) * v[id] + diffuseOmega * xNew;
            }
        }
        v.swap(v0);
        applyBoundary();
        if (valveOpen) applyValveBC();
    }
}

// Implicit scalar diffusion with optional dissipation, fuck this function
void MAC2D::diffuseScalarImplicit(std::vector<float>& phi,
                                 std::vector<float>& phi0,
                                 float diffusivity,
                                 float dissipation)
{
    if (diffusivity <= 0.0f && dissipation == 0.0f) return;

    // RHS must be the ORIGINAL field (before diffusion)
    if (phi0.size() != phi.size()) phi0.resize(phi.size());
    phi0 = phi; // b

    const float alpha = (diffusivity * dt) / (dx * dx);
    if (alpha <= 0.0f) {
        // only dissipation
        if (dissipation != 0.0f) {
            float k = std::max(0.0f, 1.0f - dissipation * dt);
            for (size_t i = 0; i < phi.size(); ++i) phi[i] *= k;
        }
        return;
    }

    std::vector<float> xOld;
    xOld.resize(phi.size());

    auto idx = [&](int i, int j) { return idxP(i, j); };

    for (int it = 0; it < diffuseIters; ++it) {
        xOld = phi; // Jacobi uses previous iterate

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id = idx(i, j);
                if (isSolid(i, j)) continue;

                float sum = 0.0f;
                int count = 0;

                if (i > 0     && !isSolid(i - 1, j)) { sum += xOld[idx(i - 1, j)]; count++; }
                if (i + 1 < nx && !isSolid(i + 1, j)) { sum += xOld[idx(i + 1, j)]; count++; }
                if (j > 0     && !isSolid(i, j - 1)) { sum += xOld[idx(i, j - 1)]; count++; }
                if (j + 1 < ny && !isSolid(i, j + 1)) { sum += xOld[idx(i, j + 1)]; count++; }

                const float denom = 1.0f + alpha * (float)count;
                const float xNew  = (phi0[id] + alpha * sum) / denom;

                // weighted Jacobi relaxation
                phi[id] = (1.0f - diffuseOmega) * xOld[id] + diffuseOmega * xNew;
            }
        }
    }

    // Optional dissipation after diffusion (physical-ish “decay”)
    if (dissipation != 0.0f) {
        float k = std::max(0.0f, 1.0f - dissipation * dt);
        for (size_t i = 0; i < phi.size(); ++i) phi[i] *= k;
    }
}



void MAC2D::addForces(float buoyancy, float gravity) {
    // add vertical force to v faces using smoke as buoyancy
    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i < nx; i++) {
            float x = (i + 0.5f) * dx;
            float y = (j) * dx;

            float t = sampleCellCentered(temp, x, y);
            float theta = t - ambientTemp;      // temperature deviation

            v[idxV(i, j)] += dt * (buoyancy * theta + gravity);
        }
    }

    // --- NEW: simple global velocity damping (viscous drag) ---
    if (velDamping > 0.0f) {
        // exponential decay factor per time step
        float factor = std::exp(-velDamping * dt);

        // damp u faces
        for (float& uu : u) {
            uu *= factor;
        }

        // damp v faces
        for (float& vv : v) {
            vv *= factor;
        }
    }
}

// Vorticity confinement
void MAC2D::addVorticityConfinement(float eps) {
    if (eps <= 0.0f) return;

    const int Nc = nx * ny;
    std::vector<float> omega(Nc, 0.0f);
    std::vector<float> mag(Nc, 0.0f);
    std::vector<float> fx(Nc, 0.0f);
    std::vector<float> fy(Nc, 0.0f);

    auto idx = [&](int i, int j) { return idxP(i, j); };

    // --- 1) vorticity at cell centers: ω = dv/dx - du/dy
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i, j)) { omega[idx(i,j)] = 0.0f; continue; }

            // dv/dx at center using v averaged to center rows
            // v is at (i+0.5, j) and (i+0.5, j+1)
            float vL = 0.5f * (v[idxV(i, j)]     + v[idxV(i, j + 1)]);
            float vR = 0.5f * (v[idxV(std::min(i + 1, nx - 1), j)] +
                               v[idxV(std::min(i + 1, nx - 1), j + 1)]);

            // du/dy at center using u averaged to center columns
            // u is at (i, j+0.5) and (i+1, j+0.5)
            float uB = 0.5f * (u[idxU(i, j)]     + u[idxU(i + 1, j)]);
            float uT = 0.5f * (u[idxU(i, std::min(j + 1, ny - 1))] +
                               u[idxU(i + 1, std::min(j + 1, ny - 1))]);

            float dv_dx = (vR - vL) / dx;
            float du_dy = (uT - uB) / dx;

            float w = dv_dx - du_dy;
            omega[idx(i,j)] = w;
            mag[idx(i,j)] = std::fabs(w);
        }
    }

    // --- 2) N = ∇|ω| / |∇|ω||, force = eps * dx * (N × ω)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i, j)) continue;

            int im1 = std::max(i - 1, 0);
            int ip1 = std::min(i + 1, nx - 1);
            int jm1 = std::max(j - 1, 0);
            int jp1 = std::min(j + 1, ny - 1);

            float dmx = (mag[idx(ip1, j)] - mag[idx(im1, j)]) / (2.0f * dx);
            float dmy = (mag[idx(i, jp1)] - mag[idx(i, jm1)]) / (2.0f * dx);

            float len = std::sqrt(dmx*dmx + dmy*dmy) + 1e-6f;
            float Nx = dmx / len;
            float Ny = dmy / len;

            float w = omega[idx(i,j)];

            // 2D: N x (0,0,w) = (Ny*w, -Nx*w, 0)
            fx[idx(i,j)] = eps * dx * (Ny * w);
            fy[idx(i,j)] = eps * dx * (-Nx * w);
        }
    }

    // --- 3) Apply to faces (average adjacent cell forces onto the face)
    // u faces: between cell (i-1,j) and (i,j)
    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            bool solidL = isSolid(i - 1, j);
            bool solidR = isSolid(i, j);
            if (solidL || solidR) continue;

            float f = 0.5f * (fx[idx(i - 1, j)] + fx[idx(i, j)]);
            u[idxU(i, j)] += dt * f;
        }
    }

    // v faces: between cell (i,j-1) and (i,j)
    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            bool solidB = isSolid(i, j - 1);
            bool solidT = isSolid(i, j);
            if (solidB || solidT) continue;

            float f = 0.5f * (fy[idx(i, j - 1)] + fy[idx(i, j)]);
            v[idxV(i, j)] += dt * f;
        }
    }
}

void MAC2D::addVelocityImpulse(float cx, float cy, float radius, float strength) {
    // Add a circular rotational impulse: u and v get +/- tangential components
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) continue;
            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float rx = x - cx;
            float ry = y - cy;
            float r2 = rx*rx + ry*ry;
            if (r2 <= radius*radius) {
                float r = std::sqrt(r2) + 1e-9f;
                // tangential unit vector = (-ry/r, rx/r)
                float tux = -ry / r;
                float tvy =  rx / r;
                float w = (1.0f - (r / radius)) * strength; // falloff
                // add u on the two vertical faces around this cell
                u[idxU(i, j)] += dt * 0.5f * w * tux;
                u[idxU(std::min(i+1, nx), j)] += dt * 0.5f * w * tux;
                // add v on the two horizontal faces
                v[idxV(i, j)] += dt * 0.5f * w * tvy;
                v[idxV(i, std::min(j+1, ny))] += dt * 0.5f * w * tvy;
            }
        }
    }
}

void MAC2D::coolAndDiffuseTemperature() {
    const float invDx2 = 1.0f / (dx * dx);

    // simple explicit diffusion + cooling in one pass
    std::vector<float> newTemp = temp;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { newTemp[id] = 0.0f; continue; }

            float T = temp[id];

            // --- diffusion term κ ∇²T (optional)
            float lap = 0.0f;
            if (tempDiffusivity > 0.0f) {
                float center = T;
                float sum = 0.0f;
                int count = 0;

                if (i > 0     && !isSolid(i-1,j)) { sum += temp[idxP(i-1,j)]; count++; }
                if (i+1 < nx && !isSolid(i+1,j)) { sum += temp[idxP(i+1,j)]; count++; }
                if (j > 0     && !isSolid(i,j-1)) { sum += temp[idxP(i,j-1)]; count++; }
                if (j+1 < ny && !isSolid(i,j+1)) { sum += temp[idxP(i,j+1)]; count++; }

                lap = (sum - count * center) * invDx2;
            }

            // --- cooling toward ambient: dT/dt = -k (T - Tamb)
            float coolTerm = -tempCoolRate * (T - ambientTemp);

            float dTdt = tempDiffusivity * lap + coolTerm;

            newTemp[id] = T + dt * dTdt;
            newTemp[id] = clampf(newTemp[id], 0.0f, 1.0f);
        }
    }

    temp.swap(newTemp);
}

void MAC2D::addHeatSource(float cx, float cy, float radius, float amount) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (isSolid(i, j)) continue;
            float x = (i + 0.5f) * dx;
            float y = (j + 0.5f) * dx;
            float rx = x - cx;
            float ry = y - cy;
            if (rx*rx + ry*ry <= radius*radius) {
                int id = idxP(i,j);
                temp[id] = clampf(temp[id] + amount, 0.0f, 1.0f);
            }
        }
    }
}

void MAC2D::updateAge(float dtLocal) {
    // AGE_MAX seconds maps to age==1.0
    const float AGE_MAX = 3.0f;
    const float invAge = 1.0f / AGE_MAX;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = idxP(i,j);
            if (isSolid(i,j)) { age[id] = 0.0f; continue; }
            if (smoke[id] > 1e-4f) {
                age[id] = clampf(age[id] + dtLocal * invAge, 0.0f, 1.0f);
            } else {
                age[id] = 0.0f;
            }
        }
    }
}
