#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

struct MAC2D {
    int nx, ny;
    float dx;
    float dt;

    // Staggered velocity
    std::vector<float> u; // (nx+1) * ny
    std::vector<float> v; // nx * (ny+1)

    // Cell-centered
    std::vector<float> p;     // nx * ny
    std::vector<float> smoke; // nx * ny

    // Scratch buffers
    std::vector<float> u0, v0, smoke0;
    std::vector<float> div;   // nx * ny
    std::vector<float> rhs;   // nx * ny (same as div in this simple setup)

    std::vector<uint8_t> solid; // to determine whether is solid or fluid cell, 1 for solid 0 for fluid

    MAC2D(int NX, int NY, float DX, float DT)
        : nx(NX), ny(NY), dx(DX), dt(DT)
    {
        u.resize((nx+1)*ny, 0.0f);
        v.resize(nx*(ny+1), 0.0f);
        p.resize(nx*ny, 0.0f);
        smoke.resize(nx*ny, 0.0f);

        u0 = u; v0 = v; smoke0 = smoke;
        div.resize(nx*ny, 0.0f);
        rhs.resize(nx*ny, 0.0f);

        solid.resize(nx*ny, 0); // we start without solids

        //to mark the outside walls as solid
        for (int i = 0; i < nx; ++i) { solid[idxP(i,0)] = 1; solid[idxP(i,ny-1)] = 1; }
        for (int j = 0; j < ny; ++j) { solid[idxP(0,j)] = 1; solid[idxP(nx-1,j)] = 1; }
    }

    // ---- Index helpers ----
    inline int idxP(int i, int j) const { return i + nx*j; }
    inline int idxU(int i, int j) const { return i + (nx+1)*j; }      // i in [0..nx], j in [0..ny-1]
    inline int idxV(int i, int j) const { return i + nx*j; }          // i in [0..nx-1], j in [0..ny]
    inline bool isSolid(int i,int j) const { return solid[idxP(i,j)] != 0; }
    inline void worldToCell(float x, float y, int &i, int &j) const {
    // cell centers at (i+0.5)*dx -> invert
    float fx = x / dx - 0.5f;
    float fy = y / dx - 0.5f;
    i = (int)std::floor(fx);
    j = (int)std::floor(fy);
    if (i < 0) i = 0; if (i >= nx) i = nx-1;
    if (j < 0) j = 0; if (j >= ny) j = ny-1;
}
    float maxAbsDiv() const {
    float m = 0.0f;
    for (int k = 0; k < nx*ny; ++k) m = std::max(m, std::abs(div[k]));
    return m;
}

    // Clamp helper
    static inline float clampf(float x, float a, float b) { return std::max(a, std::min(b, x)); }

    // ---- Sample cell-centered field (smoke, pressure) at world position (x,y) using bilinear ----
    float sampleCellCentered(const std::vector<float>& f, float x, float y) const {
        // Cell centers are at ( (i+0.5)*dx, (j+0.5)*dx )
        // Convert world -> "cell index space" where center aligns nicely:
        float fx = x/dx - 0.5f;
        float fy = y/dx - 0.5f;

        int i0 = (int)std::floor(fx);
        int j0 = (int)std::floor(fy);
        float tx = fx - i0;
        float ty = fy - j0;

        // clamp indices to valid cells
        i0 = (int)clampf((float)i0, 0.0f, (float)(nx-1));
        j0 = (int)clampf((float)j0, 0.0f, (float)(ny-1));
        int i1 = std::min(i0+1, nx-1);
        int j1 = std::min(j0+1, ny-1);

        float a = f[idxP(i0,j0)];
        float b = f[idxP(i1,j0)];
        float c = f[idxP(i0,j1)];
        float d = f[idxP(i1,j1)];

        float ab = a*(1-tx) + b*tx;
        float cd = c*(1-tx) + d*tx;
        return ab*(1-ty) + cd*ty;
    }


    // ---- Sample u (x-velocity on vertical faces) at world position (x,y) ----
    float sampleU(const std::vector<float>& fu, float x, float y) const {

        int ci, cj;
        worldToCell(x, y, ci, cj);
        if (isSolid(ci, cj)) return 0.0f;
        // u lives at ( i*dx, (j+0.5)*dx )
        float fx = x/dx;        // face-aligned in x
        float fy = y/dx - 0.5f; // centered in y

        int i0 = (int)std::floor(fx);
        int j0 = (int)std::floor(fy);
        float tx = fx - i0;
        float ty = fy - j0;

        i0 = (int)clampf((float)i0, 0.0f, (float)nx);      // u has i in [0..nx]
        j0 = (int)clampf((float)j0, 0.0f, (float)(ny-1));  // j in [0..ny-1]
        int i1 = std::min(i0+1, nx);
        int j1 = std::min(j0+1, ny-1);

        float a = fu[idxU(i0,j0)];
        float b = fu[idxU(i1,j0)];
        float c = fu[idxU(i0,j1)];
        float d = fu[idxU(i1,j1)];

        float ab = a*(1-tx) + b*tx;
        float cd = c*(1-tx) + d*tx;
        return ab*(1-ty) + cd*ty;
    }

    // ---- Sample v (y-velocity on horizontal faces) at world position (x,y) ----
    float sampleV(const std::vector<float>& fv, float x, float y) const {

        int ci, cj;
        worldToCell(x, y, ci, cj);
        if (isSolid(ci, cj)) return 0.0f;
        // v lives at ( (i+0.5)*dx, j*dx )
        float fx = x/dx - 0.5f; // centered in x
        float fy = y/dx;        // face-aligned in y

        int i0 = (int)std::floor(fx);
        int j0 = (int)std::floor(fy);
        float tx = fx - i0;
        float ty = fy - j0;

        i0 = (int)clampf((float)i0, 0.0f, (float)(nx-1));  // i in [0..nx-1]
        j0 = (int)clampf((float)j0, 0.0f, (float)ny);      // v has j in [0..ny]
        int i1 = std::min(i0+1, nx-1);
        int j1 = std::min(j0+1, ny);

        float a = fv[idxV(i0,j0)];
        float b = fv[idxV(i1,j0)];
        float c = fv[idxV(i0,j1)];
        float d = fv[idxV(i1,j1)];

        float ab = a*(1-tx) + b*tx;
        float cd = c*(1-tx) + d*tx;
        return ab*(1-ty) + cd*ty;
    }

    // ---- Velocity at cell center (for advecting smoke etc.) ----
    void velAt(float x, float y, const std::vector<float>& fu, const std::vector<float>& fv, float& outUx, float& outVy) const {
        outUx = sampleU(fu, x, y);
        outVy = sampleV(fv, x, y);
    }

    // ---- Add buoyancy to v based on smoke density ----
    void addForces(float buoyancy = 2.0f, float gravity = 0.0f) {
        // v is on horizontal faces: (i+0.5, j)
        // We'll add vertical acceleration using smoke at nearby cell center.
        for (int j = 0; j <= ny; j++) {
            for (int i = 0; i < nx; i++) {
                float x = (i + 0.5f) * dx;
                float y = (j) * dx;

                // sample smoke at this face position (cell-centered field)
                float s = sampleCellCentered(smoke, x, y);

                // buoyancy pushes upward (positive v)
                v[idxV(i,j)] += dt * (buoyancy * s + gravity);
            }
        }
    }

    // ---- Enforce solid box boundary: no flow through walls ----
    void applyBoundary() {
    // left/right walls: u = 0 at i=0 and i=nx
    for (int j = 0; j < ny; j++) {
        u[idxU(0,j)]  = 0.0f;
        u[idxU(nx,j)] = 0.0f;
    }
    // bottom/top walls: v = 0 at j=0 and j=ny
    for (int i = 0; i < nx; i++) {
        v[idxV(i,0)]  = 0.0f;
        v[idxV(i,ny)] = 0.0f;
    }

    // additionally: any face next to a solid cell should be zero (no-through-wall)
    // u faces sit between cell (i-1,j) and (i,j)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            bool leftSolid  = (i-1 >= 0)    ? isSolid(i-1, j) : true; // outside -> solid
            bool rightSolid = (i < nx)      ? isSolid(i, j)   : true;
            if (leftSolid || rightSolid) u[idxU(i,j)] = 0.0f;
        }
    }
    // v faces sit between cell (i,j-1) and (i,j)
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            bool botSolid = (j-1 >= 0)  ? isSolid(i, j-1) : true;
            bool topSolid = (j < ny)    ? isSolid(i, j)   : true;
            if (botSolid || topSolid) v[idxV(i,j)] = 0.0f;
        }
    }
}

    // ---- Semi-Lagrangian advection for u and v ----
    void advectVelocity() {
    u0 = u;
    v0 = v;

    // Advect u faces
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i <= nx; i++) {
            float x = (i) * dx;
            float y = (j + 0.5f) * dx;

            float ux, vy;
            velAt(x, y, u0, v0, ux, vy);

            // Backtrace
            float x0 = x - dt * ux;
            float y0 = y - dt * vy;
            x0 = clampf(x0, 0.0f, nx*dx);
            y0 = clampf(y0, 0.0f, ny*dx);

            u[idxU(i,j)] = sampleU(u0, x0, y0);

            // If this face lies adjacent to a solid cell, zero it (no-through)
            bool leftSolid  = (i-1 >= 0) ? isSolid(i-1, j) : true;
            bool rightSolid = (i   < nx) ? isSolid(i,     j) : true;
            if (leftSolid || rightSolid) {
                u[idxU(i,j)] = 0.0f;
            }
        }
    }

    // Advect v faces
    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i < nx; i++) {
            float x = (i + 0.5f) * dx;
            float y = (j) * dx;

            float ux, vy;
            velAt(x, y, u0, v0, ux, vy);

            float x0 = x - dt * ux;
            float y0 = y - dt * vy;
            x0 = clampf(x0, 0.0f, nx*dx);
            y0 = clampf(y0, 0.0f, ny*dx);

            v[idxV(i,j)] = sampleV(v0, x0, y0);

            // If this face lies adjacent to a solid cell, zero it
            bool botSolid = (j-1 >= 0) ? isSolid(i, j-1) : true;
            bool topSolid = (j   < ny) ? isSolid(i, j)   : true;
            if (botSolid || topSolid) {
                v[idxV(i,j)] = 0.0f;
            }
        }
    }

    // final clamp to domain and solids
    applyBoundary();
}

    // ---- Compute divergence at cell centers ----
    void computeDivergence() {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (isSolid(i,j)) { div[idxP(i,j)] = 0.0f; continue; }

            // u faces around cell (i,j): left = u(i,j), right = u(i+1,j)
            float uL = u[idxU(i,  j)];
            float uR = u[idxU(i+1,j)];
            float vB = v[idxV(i,  j)];
            float vT = v[idxV(i,  j+1)];

            // If neighbor is solid, enforce no-through-wall by treating the face as zero-normal flow.
            // This is the key correctness fix.
            if (i > 0     && isSolid(i-1,j)) uL = 0.0f;
            if (i < nx-1  && isSolid(i+1,j)) uR = 0.0f;
            if (j > 0     && isSolid(i,j-1)) vB = 0.0f;
            if (j < ny-1  && isSolid(i,j+1)) vT = 0.0f;

            div[idxP(i,j)] = (uR - uL + vT - vB) / dx;
        }
    }
}

    // ---- Solve Poisson: Laplacian(p) = (1/dt)*div ----
    // Simple Gauss-Seidel iterations (prototype-level).
    void solvePressure(int iters = 80) {
    std::fill(p.begin(), p.end(), 0.0f);

    // rhs = div/dt
    for (int k = 0; k < nx*ny; k++) rhs[k] = div[k] / dt;

    // Gauss-Seidel on fluid cells only
    for (int it = 0; it < iters; it++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (isSolid(i,j)) { p[idxP(i,j)] = 0.0f; continue; }

                float sum = 0.0f;
                int count = 0;

                // For each neighbor: if neighbor is fluid, include it in stencil.
                if (i > 0 && !isSolid(i-1,j)) { sum += p[idxP(i-1,j)]; count++; }
                if (i+1 < nx && !isSolid(i+1,j)) { sum += p[idxP(i+1,j)]; count++; }
                if (j > 0 && !isSolid(i,j-1)) { sum += p[idxP(i,j-1)]; count++; }
                if (j+1 < ny && !isSolid(i,j+1)) { sum += p[idxP(i,j+1)]; count++; }

                // If surrounded by solids (shouldn’t happen often), skip.
                if (count == 0) { p[idxP(i,j)] = 0.0f; continue; }

                float b = rhs[idxP(i,j)];
                p[idxP(i,j)] = (sum - b * dx * dx) / (float)count;
            }
        }
    }
}

    // ---- Subtract pressure gradient from velocities ----
    void project() {
    computeDivergence();
    solvePressure(80);

    // u faces between (i-1,j) and (i,j)
    for (int j = 0; j < ny; j++) {
        for (int i = 1; i < nx; i++) {
            bool solidL = isSolid(i-1,j);
            bool solidR = isSolid(i,  j);

            if (solidL && solidR) { u[idxU(i,j)] = 0.0f; continue; }
            if (solidL || solidR) { u[idxU(i,j)] = 0.0f; continue; } // no-through-wall

            float gradp = (p[idxP(i,j)] - p[idxP(i-1,j)]) / dx;
            u[idxU(i,j)] -= dt * gradp;
        }
    }

    // v faces between (i,j-1) and (i,j)
    for (int j = 1; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            bool solidB = isSolid(i,j-1);
            bool solidT = isSolid(i,j);

            if (solidB && solidT) { v[idxV(i,j)] = 0.0f; continue; }
            if (solidB || solidT) { v[idxV(i,j)] = 0.0f; continue; }

            float gradp = (p[idxP(i,j)] - p[idxP(i,j-1)]) / dx;
            v[idxV(i,j)] -= dt * gradp;
        }
    }

    applyBoundary();
}

    // ---- Advect smoke (cell-centered) with semi-Lagrangian ----
    void advectSmoke(float dissipation = 0.995f) {
        smoke0 = smoke;

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                float x = (i + 0.5f) * dx;
                float y = (j + 0.5f) * dx;

                float ux, vy;
                velAt(x, y, u, v, ux, vy);

                float x0 = x - dt * ux;
                float y0 = y - dt * vy; 
                x0 = clampf(x0, 0.0f, nx*dx);
                y0 = clampf(y0, 0.0f, ny*dx);

                // find which cell that point is inside
                int si, sj;
                worldToCell(x0, y0, si, sj);

                float s = 0.0f;
                if (!isSolid(si, sj)) {
                    s = sampleCellCentered(smoke0, x0, y0);
                } else {
                    s = 0.0f; // inside solid -> no smoke comes out
                }

                smoke[idxP(i,j)] = dissipation * s;
            }
        }
    }

    // ---- Simple smoke source injector ----
    void addSmokeSource(float cx, float cy, float radius, float amount) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                float x = (i + 0.5f) * dx;
                float y = (j + 0.5f) * dx;
                float dx0 = x - cx;
                float dy0 = y - cy;
                float r2 = dx0*dx0 + dy0*dy0;
                if (r2 <= radius*radius && !isSolid(i,j)) {
                    smoke[idxP(i,j)] = std::min(1.0f, smoke[idxP(i,j)] + amount);
                }
            }
        }
    }

    void addSolidCircle(float cx, float cy, float r) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float x = (i + 0.5f) * dx;
                float y = (j + 0.5f) * dx;
                float dx0 = x - cx;
                float dy0 = y - cy;
                if (dx0*dx0 + dy0*dy0 <= r*r) {
                    solid[idxP(i,j)] = 1;
                    smoke[idxP(i,j)] = 0.0f;
                }
            }
        }
    }

    // ---- Write a grayscale PPM of smoke (better than raw pressure) ----
    void writePPM(const std::string& path, int scale = 6) const {
        int W = nx * scale;
        int H = ny * scale;

        std::vector<unsigned char> img(W*H*3, 0);

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int i = x / scale;
                int j = y / scale;

                float s = smoke[idxP(i, ny-1-j)]; // flip Y for image
                s = clampf(s, 0.0f, 1.0f);

                // map smoke to nice contrast
                float c = std::pow(s, 0.6f);
                unsigned char g = (unsigned char)(clampf(c,0.0f,1.0f) * 255.0f);

                int k = (x + W*y)*3;
                img[k+0] = g;
                img[k+1] = g;
                img[k+2] = g;
            }
        }

        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) { std::perror("fopen"); return; }
        std::fprintf(f, "P6\n%d %d\n255\n", W, H);
        std::fwrite(img.data(), 1, img.size(), f);
        std::fclose(f);
    }

    // ---- One full step ----
    void step() {
        // (A) inject smoke near bottom
        addSmokeSource(0.5f*nx*dx, 0.2f*ny*dx, 0.10f*nx*dx, 0.08f);

        // (B) forces (buoyancy makes smoke rise)
        addForces(/*buoyancy=*/3.5f, /*gravity=*/0.0f);
        applyBoundary();

        // (C) move velocity with itself
        advectVelocity();
        applyBoundary();

        // (D) incompressibility (the "magic" step)
        project();
        computeDivergence();
        std::cout << "max |div| after project: " << maxAbsDiv() << "\n";


        // (E) advect smoke with final velocity field
        advectSmoke();
    }
};

int main(int argc, char** argv) {
    int iteration = 2;
    int nx = 96, ny = 96;
    float dx = 1.0f / nx;
    float dt = 0.02f;

    int frames = 180; // ~7.5 seconds at 24fps if you export at 24fps

    MAC2D sim(nx, ny, dx, dt);

    sim.addSolidCircle(0.5f, 0.55f, 0.12f);

    // output dirs
    std::string baseDir  = "iterations/smoke_" + std::to_string(iteration);
    std::string frameDir = baseDir + "/frames";

    // make folders
    std::string mkdirCmd = "mkdir -p " + frameDir;
    std::system(mkdirCmd.c_str());

for (int f = 0; f < frames; f++) {
    sim.step();
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/frame_%04d.ppm", frameDir.c_str(), f);
    sim.writePPM(buf, /*scale=*/6);

    std::cout << "wrote " << buf << "\n";
}

        std::cout << "Done. Convert to video:\n";
        std::cout << "ffmpeg -framerate 24 -i " << frameDir
          << "/frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p "
          << baseDir << "/smoke.mp4\n";
    return 0;
}