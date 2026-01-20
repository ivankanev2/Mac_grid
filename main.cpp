#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

static inline int idx(int x, int y, int w) { return x + y * w; }

struct MacGrid2D {
    int nx, ny;
    float dx;           // cell size
    float rho = 1.0f;   // density

    // u on vertical faces: (nx+1) * ny
    std::vector<float> u;
    // v on horizontal faces: nx * (ny+1)
    std::vector<float> v;
    // pressure at centers: nx * ny
    std::vector<float> p;

    MacGrid2D(int nx_, int ny_, float dx_=1.0f)
        : nx(nx_), ny(ny_), dx(dx_),
          u((nx+1)*ny, 0.0f),
          v(nx*(ny+1), 0.0f),
          p(nx*ny, 0.0f) {}

    float& U(int i, int j) { return u[idx(i, j, nx+1)]; }
    float& V(int i, int j) { return v[idx(i, j, nx)]; }
    float& P(int i, int j) { return p[idx(i, j, nx)]; }

    float Uc(int i, int j) const {
        // cell-centered u from faces
        return 0.5f * (u[idx(i, j, nx+1)] + u[idx(i+1, j, nx+1)]);
    }
    float Vc(int i, int j) const {
        return 0.5f * (v[idx(i, j, nx)] + v[idx(i, j+1, nx)]);
    }

    float divergence(int i, int j) const {
        // div = (u_{i+1/2}-u_{i-1/2} + v_{j+1/2}-v_{j-1/2}) / dx
        float du = u[idx(i+1, j, nx+1)] - u[idx(i, j, nx+1)];
        float dv = v[idx(i, j+1, nx)] - v[idx(i, j, nx)];
        return (du + dv) / dx;
    }

    void applyBoundaryNoSlip() {
        // simple: zero normal velocity at domain boundaries
        // u on left/right
        for (int j = 0; j < ny; ++j) {
            U(0, j) = 0.0f;
            U(nx, j) = 0.0f;
        }
        // v on bottom/top
        for (int i = 0; i < nx; ++i) {
            V(i, 0) = 0.0f;
            V(i, ny) = 0.0f;
        }
    }

    void addVortex(float cx, float cy, float strength) {
        // add a simple swirling velocity field around (cx, cy) in cell-center coords
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float x = (i + 0.5f) * dx;
                float y = (j + 0.5f) * dx;
                float rx = x - cx;
                float ry = y - cy;
                float r2 = rx*rx + ry*ry + 1e-6f;
                float invr = 1.0f / std::sqrt(r2);
                // tangential direction (-ry, rx)
                float ux = -ry * invr * strength;
                float vy =  rx * invr * strength;

                // write into staggered faces (simple “splat” to nearby faces)
                // u faces at (i, j) and (i+1, j)
                U(i, j) += 0.5f * ux;
                U(i+1, j) += 0.5f * ux;
                // v faces at (i, j) and (i, j+1)
                V(i, j) += 0.5f * vy;
                V(i, j+1) += 0.5f * vy;
            }
        }
    }

    void pressureSolveProject(float dt, int iters=80) {
        // Solve Poisson: ∇²p = (rho/dt) * div(u)
        // Jacobi iteration with simple boundary (Neumann-ish by clamping indices)

        std::vector<float> pnew(nx*ny, 0.0f);

        auto Pget = [&](int i, int j)->float {
            i = std::clamp(i, 0, nx-1);
            j = std::clamp(j, 0, ny-1);
            return p[idx(i,j,nx)];
        };

        float scale = rho / dt;

        for (int k = 0; k < iters; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    float b = scale * divergence(i, j); // RHS
                    float sum = Pget(i-1,j) + Pget(i+1,j) + Pget(i,j-1) + Pget(i,j+1);
                    // dx^2 * b moves to other side; Laplacian stencil denominator 4
                    pnew[idx(i,j,nx)] = (sum - dx*dx * b) * 0.25f;
                }
            }
            p.swap(pnew);
        }

        // Subtract pressure gradient from velocities (projection)
        // u(i,j) -= dt/rho * (p(i,j) - p(i-1,j))/dx   (u face between cells i-1 and i)
        for (int j = 0; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                float gradp = (P(i, j) - P(i-1, j)) / dx;
                U(i, j) -= (dt / rho) * gradp;
            }
        }
        // v(i,j) -= dt/rho * (p(i,j) - p(i,j-1))/dx
        for (int j = 1; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float gradp = (P(i, j) - P(i, j-1)) / dx;
                V(i, j) -= (dt / rho) * gradp;
            }
        }

        applyBoundaryNoSlip();
    }
};

// write a grayscale PPM image
static void writePPMGray(const std::string& path, int w, int h, const std::vector<uint8_t>& img) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w*h; ++i) {
        uint8_t c = img[i];
        f.write((char*)&c, 1);
        f.write((char*)&c, 1);
        f.write((char*)&c, 1);
    }
}

// draw arrows over a grayscale base (very simple, coarse)
static void drawVelocityArrows(std::vector<uint8_t>& img, int w, int h, const MacGrid2D& g, int step=8) {
    auto setpix = [&](int x, int y, uint8_t c){
        if (x<0||y<0||x>=w||y>=h) return;
        img[idx(x,y,w)] = c;
    };

    for (int j = 0; j < g.ny; j += step) {
        for (int i = 0; i < g.nx; i += step) {
            float ux = g.Uc(i,j);
            float vy = g.Vc(i,j);

            int x0 = (int)((i + 0.5f) * (w / (float)g.nx));
            int y0 = (int)((j + 0.5f) * (h / (float)g.ny));

            float s = 6.0f; // arrow scale in pixels
            int x1 = (int)std::lround(x0 + ux * s);
            int y1 = (int)std::lround(y0 + vy * s);

            // draw a simple line (Bresenham-ish)
            int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
            int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
            int err = dx + dy;
            int x = x0, y = y0;
            while (true) {
                setpix(x, y, 255);
                if (x == x1 && y == y1) break;
                int e2 = 2 * err;
                if (e2 >= dy) { err += dy; x += sx; }
                if (e2 <= dx) { err += dx; y += sy; }
            }
        }
    }
}

int main() {
    const int nx = 96, ny = 96;
    MacGrid2D g(nx, ny, 1.0f);

    g.applyBoundaryNoSlip();

    // add an initial swirling flow
    g.addVortex(nx*0.5f, ny*0.5f, 2.0f);

    const float dt = 1.0f;
    const int frames = 120;

    for (int f = 0; f < frames; ++f) {
        // enforce incompressibility (projection)
        g.pressureSolveProject(dt, 80);

        // visualize pressure
        std::vector<uint8_t> img(nx*ny, 0);

        // normalize pressure to [0,255] for display
        float pmin = g.p[0], pmax = g.p[0];
        for (float v : g.p) { pmin = std::min(pmin, v); pmax = std::max(pmax, v); }
        float inv = (pmax - pmin) > 1e-6f ? 1.0f / (pmax - pmin) : 0.0f;

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float pn = (g.P(i,j) - pmin) * inv;
                img[idx(i, ny-1-j, nx)] = (uint8_t)std::clamp((int)std::lround(pn * 255.0f), 0, 255);
            }
        }

        drawVelocityArrows(img, nx, ny, g, 8);

        char buf[256];
        std::snprintf(buf, sizeof(buf), "out/frame_%04d.ppm", f);
        std::system("mkdir -p out");
        writePPMGray(buf, nx, ny, img);

        std::cout << "wrote " << buf << "\n";
    }

    std::cout << "Done. Convert to video:\n"
              << "ffmpeg -framerate 24 -i out/frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p macgrid.mp4\n";
    return 0;
}