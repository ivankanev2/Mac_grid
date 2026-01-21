// main.cpp
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef __APPLE__
  #define GL_SILENCE_DEPRECATION
  #include <OpenGL/gl3.h>
#else
  #include <GL/gl.h>
#endif

#include "mac_smoke_sim.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

static const int NX = 96;
static const int NY = 96;

enum class PaintTool { AddSolid, EraseSolid, AddSmoke };
PaintTool paintTool = PaintTool::AddSolid;

bool circleMode = true;
float brushRadius = 0.06f;          // in sim space (0..1)
float rectHalfSize = 0.06f;         // half-size for rectangle

float smokeBrushRadius = 0.05f;
float smokeBrushAmount = 0.15f;

// Debug overlay toggles
static bool showDivOverlay = false;
static bool showVelOverlay = false;
static bool showVortOverlay = false;

// Tuning
static float divScale = 8.0f;     // larger = more sensitive heatmap
static float divAlpha = 0.75f;    // overlay opacity
static int   velStride = 6;       // draw arrow every N cells
static float velScale  = 0.35f;   // arrow length multiplier

static float vortScale = 8.0f;    // larger = more sensitive heatmap
static float vortAlpha = 0.75f;   // overlay opacity

float dtMax = 0.02f;     // cap so it doesn’t get too large when still
float dtMin = 0.001f;    // optional safety
float cfl   = 0.9f;      // 0.5–1.0 is typical

static float vortEps = 2.0f;

// simple clamp (avoids std::clamp editor issues)
static inline float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static GLuint makeSmokeTexture(int w, int h) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    std::vector<uint8_t> blank(w * h * 3, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, blank.data());
    return tex;
}

static GLuint makeOverlayTextureRGBA(int w, int h) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    std::vector<uint8_t> blank(w * h * 4, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, blank.data());
    return tex;
}

static void uploadDivOverlay(GLuint tex, int w, int h,
                             const std::vector<float>& div,
                             const std::vector<uint8_t>& solid,
                             float scale, float alpha)
{
    std::vector<uint8_t> img(w * h * 4, 0);

    for (int j = 0; j < h; ++j) {
        int srcJ = (h - 1 - j); // SAME flip as your smoke upload
        for (int i = 0; i < w; ++i) {
            int srcIdx = i + w * srcJ;
            int dstIdx = i + w * j;

            if (solid[srcIdx]) {
                img[dstIdx*4 + 3] = 0; // transparent on solids
                continue;
            }

            float d = div[srcIdx] * scale;   // roughly map into [-1..1]
            if (d < -1.0f) d = -1.0f;
            if (d >  1.0f) d =  1.0f;

            float m = std::fabs(d);          // magnitude 0..1
            float a = m * alpha;             // opacity

            uint8_t A = (uint8_t)std::lround(clamp01(a) * 255.0f);
            uint8_t R = (d > 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;
            uint8_t B = (d < 0.0f) ? (uint8_t)std::lround(m * 255.0f) : 0;

            img[dstIdx*4 + 0] = R;
            img[dstIdx*4 + 1] = 0;
            img[dstIdx*4 + 2] = B;
            img[dstIdx*4 + 3] = A;
        }
    }

    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
}

static void uploadVortOverlay(GLuint tex, int w, int h, const std::vector<float>& omega, const std::vector<uint8_t>& solid, float scale, float alpha) {
    std::vector<uint8_t> img(w*h*4,0);
    for (int j=0;j<h;++j) {
        int srcJ = (h-1-j);
        for (int i=0;i<w;++i) {
            int sidx = i + w*srcJ;
            int didx = i + w*j;
            if (solid[sidx]) { img[didx*4+3]=0; continue; }
            float v = omega[sidx] * scale;
            v = std::max(-1.0f, std::min(1.0f, v));
            float m = std::fabs(v);
            uint8_t A = (uint8_t)std::lround(clamp01(m * alpha) * 255.0f);
            uint8_t R = (v > 0.0f) ? (uint8_t)std::lround(m*255.0f) : 0;
            uint8_t B = (v < 0.0f) ? (uint8_t)std::lround(m*255.0f) : 0;
            img[didx*4+0] = R;
            img[didx*4+1] = 0;
            img[didx*4+2] = B;
            img[didx*4+3] = A;
        }
    }
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,img.data());
}

static void drawArrow(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 col) {
    dl->AddLine(a, b, col, 1.0f);

    ImVec2 d(b.x - a.x, b.y - a.y);
    float len = std::sqrt(d.x*d.x + d.y*d.y);
    if (len < 1e-5f) return;
    d.x /= len; d.y /= len;

    float head = 6.0f;
    ImVec2 left(-d.y, d.x);

    ImVec2 p1(b.x - d.x*head + left.x*head*0.5f, b.y - d.y*head + left.y*head*0.5f);
    ImVec2 p2(b.x - d.x*head - left.x*head*0.5f, b.y - d.y*head - left.y*head*0.5f);

    dl->AddLine(b, p1, col, 1.0f);
    dl->AddLine(b, p2, col, 1.0f);
}

static void uploadSmoke(GLuint tex, int w, int h,
                        const std::vector<float>& s,
                        const std::vector<uint8_t>& solid)
{
    std::vector<uint8_t> img(w * h * 3);

    for (int j = 0; j < h; ++j) {
        int srcJ = (h - 1 - j);   // <-- FLIP Y so smoke is not upside-down
        for (int i = 0; i < w; ++i) {
            int srcIdx = i + w * srcJ; // read from flipped row
            int dstIdx = i + w * j;    // write normally

            uint8_t r=0, g=0, b=0;

            if (solid[srcIdx]) {
                r = 40; g = 90; b = 200;  // solids color
            } else {
                float v = s[srcIdx];
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                uint8_t gray = (uint8_t)std::lround(std::pow(v, 0.6f) * 255.0f);
                r = g = b = gray;
            }

            img[dstIdx*3 + 0] = r;
            img[dstIdx*3 + 1] = g;
            img[dstIdx*3 + 2] = b;
        }
    }

    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, img.data());
}

static void printGlError(const char* where) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) {
        std::printf("GL error at %s: 0x%x\n", where, (unsigned)e);
    }
}

static inline float uAtCellCenter(const MAC2D& sim, int i, int j) {
    // u is on vertical faces: (i, j) and (i+1, j)
    // clamp i to valid interior faces
    int i0 = std::max(0, std::min(i, sim.nx - 1));
    int j0 = std::max(0, std::min(j, sim.ny - 1));
    float uL = sim.u[i0 + (sim.nx + 1) * j0];
    float uR = sim.u[(i0 + 1) + (sim.nx + 1) * j0];
    return 0.5f * (uL + uR);
}

static inline float vAtCellCenter(const MAC2D& sim, int i, int j) {
    // v is on horizontal faces: (i, j) and (i, j+1)
    int i0 = std::max(0, std::min(i, sim.nx - 1));
    int j0 = std::max(0, std::min(j, sim.ny - 1));
    float vB = sim.v[i0 + sim.nx * j0];
    float vT = sim.v[i0 + sim.nx * (j0 + 1)];
    return 0.5f * (vB + vT);
}

static void computeStats(const MAC2D& sim, float& outMaxDiv, float& outAvgAbsDiv, float& outMaxSpeed) {
    outMaxDiv = 0.0f;
    outAvgAbsDiv = 0.0f;
    outMaxSpeed = 0.0f;

    const int N = sim.nx * sim.ny;
    for (int j = 0; j < sim.ny; ++j) {
        for (int i = 0; i < sim.nx; ++i) {
            int idx = sim.idxP(i,j);

            float d = sim.div[idx];
            float ad = std::fabs(d);
            outMaxDiv = std::max(outMaxDiv, ad);
            outAvgAbsDiv += ad;

            float uc = uAtCellCenter(sim, i, j);
            float vc = vAtCellCenter(sim, i, j);
            float sp = std::sqrt(uc*uc + vc*vc);
            outMaxSpeed = std::max(outMaxSpeed, sp);
        }
    }
    outAvgAbsDiv /= (float)N;
}

int main() {
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    GLFWwindow* win = glfwCreateWindow(1100, 800, "Smoke Engine", nullptr, nullptr);
    if (!win) return 1;

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    GLuint smokeTex = makeSmokeTexture(NX, NY);
    GLuint divTex = makeOverlayTextureRGBA(NX, NY);
    GLuint vortTex = makeOverlayTextureRGBA(NX, NY);

    // --- Create our MAC grid simulator ---
    float dx = 1.0f / NX;
    float dt = 0.02f;
    MAC2D sim(NX, NY, dx, dt);
    // add the same solid obstacle we used in the offline demo
    sim.addSolidCircle(0.5f, 0.55f, 0.12f);

    bool playing = true;

    // --- probe info (hovered cell) ---
    bool hasProbe = false;
    int probeI = 0, probeJ = 0;
    float probeSmoke = 0.0f, probeDiv = 0.0f, probeU = 0.0f, probeV = 0.0f, probeSpeed = 0.0f;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        if (playing) {
        for (int sub = 0; sub < 2; ++sub) {
            float maxSpeed = sim.maxFaceSpeed();
            float dt = cfl * sim.dx / (maxSpeed + 1e-6f);

            // clamp dt
            if (dt > dtMax) dt = dtMax;
            if (dt < dtMin) dt = dtMin;

            sim.setDt(dt);
            sim.step(vortEps);
        }
    }

        uploadSmoke(smokeTex, sim.nx, sim.ny, sim.smoke, sim.solid);
        printGlError("uploadSmoke");

        if (showDivOverlay) {
        uploadDivOverlay(divTex, sim.nx, sim.ny, sim.div, sim.solid, divScale, divAlpha);
        printGlError("uploadDivOverlay");
    }
        if (showVortOverlay) {
        std::vector<float> vort(sim.nx * sim.ny);
        sim.computeVorticity(vort);           
        uploadVortOverlay(vortTex, sim.nx, sim.ny, vort, sim.solid, vortScale, vortAlpha);
    }

        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls");
        ImGui::Checkbox("Play", &playing);
        ImGui::Separator();
        ImGui::Text("Click mode");
        if (ImGui::RadioButton("Add obstacle", paintTool == PaintTool::AddSolid)) paintTool = PaintTool::AddSolid;
        ImGui::SameLine();
        if (ImGui::RadioButton("Erase obstacle", paintTool == PaintTool::EraseSolid)) paintTool = PaintTool::EraseSolid;
        if (ImGui::RadioButton("Add smoke", paintTool == PaintTool::AddSmoke)) paintTool = PaintTool::AddSmoke;

        ImGui::Checkbox("Circle", &circleMode);
        ImGui::SameLine();
        ImGui::Text("Rect");
        if (ImGui::Button("Reset Smoke")) {
        sim.reset();
        }

        ImGui::SliderFloat("Obstacle radius", &brushRadius, 0.01f, 0.20f);
        ImGui::SliderFloat("Rect half-size", &rectHalfSize, 0.01f, 0.30f);
        ImGui::SliderFloat("Smoke radius", &smokeBrushRadius, 0.01f, 0.20f);
        ImGui::SliderFloat("Smoke amount", &smokeBrushAmount, 0.01f, 0.5f);
        ImGui::End();

        ImGui::Begin("Data / Debug");

    if (ImGui::BeginTabBar("DebugTabs")) {

        if (ImGui::BeginTabItem("Formulas")) {
            ImGui::TextWrapped("MAC grid (staggered): u on vertical faces, v on horizontal faces, smoke/pressure at cell centers.");

            ImGui::Separator();
            ImGui::Text("Divergence (cell center):");
            ImGui::BulletText("div(i,j) = ( u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j) ) / dx");

            ImGui::Text("Pressure solve (Poisson):");
            ImGui::BulletText("Laplace(p) = div / dt   (solved iteratively)");

            ImGui::Text("Projection (make incompressible):");
            ImGui::BulletText("u -= dt * dp/dx,   v -= dt * dp/dy");

            ImGui::Text("Smoke advection (semi-Lagrangian):");
            ImGui::BulletText("smoke(x) = smoke0( x - dt * vel(x) )");

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Stats")) {
            float maxDiv=0, avgAbsDiv=0, maxSpeed=0;
            computeStats(sim, maxDiv, avgAbsDiv, maxSpeed);

            ImGui::Text("Grid: %d x %d", sim.nx, sim.ny);
            ImGui::Text("dx: %.6f   dt: %.6f", sim.dx, sim.dt);
            ImGui::Separator();
            ImGui::Text("max |div|: %.6e", maxDiv);
            ImGui::Text("avg |div|: %.6e", avgAbsDiv);
            ImGui::Text("max speed: %.6f", maxSpeed);

            ImGui::Separator();
            ImGui::SliderFloat("Vorticity eps", &vortEps, 0.0f, 8.0f);
            ImGui::Checkbox("Vorticity heatmap", &showVortOverlay);
            ImGui::SliderFloat("Vort scale", &vortScale, 0.1f, 50.0f);
            ImGui::SliderFloat("Vort alpha", &vortAlpha, 0.0f, 1.0f);

            if (ImGui::Button("Impulse (test swirl)")) {
                sim.addVelocityImpulse(0.5f, 0.5f, 0.12f, 3.0f);
            }

            if (ImGui::Button("Step with/without vort (A/B)")) {
                // IMPORTANT: ensure playing is false or you’ll also step in the main loop
                bool wasPlaying = playing;
                playing = false;

                float saved = vortEps;

                vortEps = saved;
                sim.setDt(dtMax);     // optional: stabilize A/B comparison
                sim.step(vortEps);           // or sim.step(vortEps) if you did Option A

                vortEps = 0.0f;
                sim.step(vortEps);

                vortEps = saved;
                playing = wasPlaying;
            }

            ImGui::Separator();
            ImGui::Checkbox("Divergence heatmap", &showDivOverlay);
            ImGui::SliderFloat("Div scale", &divScale, 0.1f, 50.0f);
            ImGui::SliderFloat("Div alpha", &divAlpha, 0.0f, 1.0f);

            ImGui::Separator();
            ImGui::Checkbox("Velocity arrows", &showVelOverlay);
            ImGui::SliderInt("Vel stride", &velStride, 2, 16);
            ImGui::SliderFloat("Vel scale", &velScale, 0.05f, 2.0f);

            ImGui::Separator();
            ImGui::Text("Adaptive dt (CFL)");
            ImGui::SliderFloat("CFL", &cfl, 0.1f, 1.5f);
            ImGui::SliderFloat("dtMax", &dtMax, 0.001f, 0.05f);
            ImGui::SliderFloat("dtMin", &dtMin, 0.0001f, 0.01f);
            ImGui::Text("current dt: %.6f", sim.dt);
            ImGui::Text("max face speed: %.6f", sim.maxFaceSpeed());

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Probe")) {
            if (!hasProbe) {
                ImGui::Text("Hover the Smoke View to inspect a cell.");
            } else {
                ImGui::Text("Cell (i,j): (%d, %d)", probeI, probeJ);
                ImGui::Text("smoke: %.4f", probeSmoke);
                ImGui::Text("div:   %.6e", probeDiv);
                ImGui::Text("u_c:   %.6f", probeU);
                ImGui::Text("v_c:   %.6f", probeV);
                ImGui::Text("|v|:   %.6f", probeSpeed);
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

ImGui::End();

        ImGui::Begin("Smoke View");

        // Debug-friendly pixel-accurate draw (scale up by an integer)
        float scale = 5.0f;
        ImGui::Image((ImTextureID)(intptr_t)smokeTex, ImVec2(NX * scale, NY * scale));
        ImVec2 p0 = ImGui::GetItemRectMin();
        ImVec2 p1 = ImGui::GetItemRectMax();

        ImDrawList* dl = ImGui::GetWindowDrawList();

        // 1) Heatmap overlay texture
        if (showDivOverlay) {
            dl->AddImage((ImTextureID)(intptr_t)divTex, p0, p1);
        }

        // 2) Vorticity heatmap overlay texture
        if (showVortOverlay) {
            dl->AddImage((ImTextureID)(intptr_t)vortTex, p0, p1);
        }

        

        // 3) Velocity arrows overlay
        if (showVelOverlay) {
            float W = (p1.x - p0.x);
            float H = (p1.y - p0.y);
            float cellW = W / sim.nx;
            float cellH = H / sim.ny;

            // Your upload flips rows (srcJ = ny-1-j),
            // so screen row j corresponds to sim row simJ = ny-1-j.
            for (int j = 0; j < sim.ny; j += velStride) {
                int simJ = (sim.ny - 1 - j);

                for (int i = 0; i < sim.nx; i += velStride) {
                    int idx = sim.idxP(i, simJ);
                    if (sim.solid[idx]) continue;

                    float uc = uAtCellCenter(sim, i, simJ);
                    float vc = vAtCellCenter(sim, i, simJ);

                    float cx = p0.x + (i + 0.5f) * cellW;
                    float cy = p0.y + (j + 0.5f) * cellH;

                    // screen y goes down, sim y goes up => negate vc
                    float dx =  uc * velScale * cellW;
                    float dy = -vc * velScale * cellH;

                    drawArrow(dl, ImVec2(cx, cy), ImVec2(cx + dx, cy + dy),
                            IM_COL32(0, 255, 0, 180));
                }
            }
        }

        bool hovered = ImGui::IsItemHovered();

        // --- PROBE (hover-only) ---
        hasProbe = false;
        if (hovered) {
            ImVec2 m = ImGui::GetMousePos();
            float uu = (m.x - p0.x) / (p1.x - p0.x);
            float vv = (m.y - p0.y) / (p1.y - p0.y);

            float sx = uu;
            float sy = vv; // keep this consistent with your current orientation

            if (sx >= 0 && sx <= 1 && sy >= 0 && sy <= 1) {
                probeI = std::max(0, std::min((int)(sx * sim.nx), sim.nx - 1));
                probeJ = std::max(0, std::min((int)(sy * sim.ny), sim.ny - 1));
                int idx = sim.idxP(probeI, probeJ);

                probeSmoke = sim.smoke[idx];
                probeDiv   = sim.div[idx];
                probeU     = uAtCellCenter(sim, probeI, probeJ);
                probeV     = vAtCellCenter(sim, probeI, probeJ);
                probeSpeed = std::sqrt(probeU*probeU + probeV*probeV);
                hasProbe = true;
            }
        }

        bool strokeOnce = (paintTool == PaintTool::AddSmoke) ? ImGui::IsMouseClicked(ImGuiMouseButton_Left)
                                                             : ImGui::IsMouseDown(ImGuiMouseButton_Left);
        if (hovered && strokeOnce) {
            ImVec2 m = ImGui::GetMousePos();
            float u = (m.x - p0.x) / (p1.x - p0.x); // 0..1
            float v = (m.y - p0.y) / (p1.y - p0.y); // 0..1


            // because we flipped the Image UVs (0,1)->(1,0), we also flip v for sim coords
            float sx = u;
            float sy = 1.0f - v;

            if (sx >= 0 && sx <= 1 && sy >= 0 && sy <= 1) {
                switch (paintTool) {
                case PaintTool::AddSolid:
                    if (circleMode) {
                        sim.addSolidCircle(sx, sy, brushRadius);
                    } else {
                        // rectangle: easiest is “paint” by stamping many circles OR add a real function
                        // We'll do a quick stamp grid (works fine for now):
                        float hs = rectHalfSize;
                        for (float yy = sy-hs; yy <= sy+hs; yy += sim.dx) {
                            for (float xx = sx-hs; xx <= sx+hs; xx += sim.dx) {
                                sim.addSolidCircle(xx, yy, sim.dx*0.75f);
                            }
                        }
                    }
                    break;
                case PaintTool::EraseSolid: {
                    // Erase: quick direct edit of the solid mask (keep border walls)
                    int i = (int)(sx * sim.nx);
                    int j = (int)(sy * sim.ny);
                    int rad = std::max(1, (int)(brushRadius / sim.dx));

                    for (int y = j-rad; y <= j+rad; ++y) {
                        for (int x = i-rad; x <= i+rad; ++x) {
                            if (x <= 0 || x >= sim.nx-1 || y <= 0 || y >= sim.ny-1) continue; // keep walls
                            float dx = (x + 0.5f) / sim.nx - sx;
                            float dy = (y + 0.5f) / sim.ny - sy;
                            if (dx*dx + dy*dy <= brushRadius*brushRadius) {
                                sim.solid[sim.idxP(x,y)] = 0;
                            }
                        }
                    }
                    break;
                }
                case PaintTool::AddSmoke:
                    sim.addSmokeSource(sx, sy, smokeBrushRadius, smokeBrushAmount);
                    break;
                }
            }
        }

        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
