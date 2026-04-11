#pragma once
#include "../Geometry/mesh_generator.h"
#include "camera.h"

#ifdef __APPLE__
#  include <OpenGL/gl3.h>
#else
#  include <GL/gl.h>
#endif

#include <string>
#include <vector>
#include <iostream>
#include <cstring>

// ============================================================================
// MeshRenderer: uploads a TriMesh to the GPU and renders it with a simple
//               Phong shading model.
//
// Uses OpenGL 3.2 Core Profile (same as smoke_engine).
// VAO layout:
//   location 0 → vec3 position
//   location 1 → vec3 normal
// ============================================================================

static const char* PIPE_VERT_SRC = R"GLSL(
#version 150 core

in  vec3 aPos;
in  vec3 aNormal;

uniform mat4 uProj;
uniform mat4 uView;
uniform mat3 uNormalMat;

out vec3 vWorldPos;
out vec3 vNormal;

void main() {
    vec4 worldPos = vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal   = normalize(uNormalMat * aNormal);
    gl_Position = uProj * uView * worldPos;
}
)GLSL";

static const char* PIPE_FRAG_SRC = R"GLSL(
#version 150 core

in  vec3 vWorldPos;
in  vec3 vNormal;

uniform vec3  uCameraPos;
uniform vec3  uPipeColor;      // base pipe colour
uniform vec3  uLightDir;       // world-space, unit length
uniform float uAmbient;
uniform float uSpecular;
uniform float uShininess;
uniform bool  uWireframe;

out vec4 fragColor;

void main() {
    if (uWireframe) {
        fragColor = vec4(uPipeColor * 1.3, 1.0);
        return;
    }

    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir);
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), uShininess);

    vec3 ambient  = uAmbient  * uPipeColor;
    vec3 diffuse  = diff      * uPipeColor;
    vec3 specular = uSpecular * spec * vec3(1.0);

    vec3 color = ambient + diffuse + specular;
    fragColor  = vec4(color, 1.0);
}
)GLSL";

// ---- Grid / axis lines -------------------------------------------------------

static const char* GRID_VERT_SRC = R"GLSL(
#version 150 core
in  vec3 aPos;
uniform mat4 uProj;
uniform mat4 uView;
void main() { gl_Position = uProj * uView * vec4(aPos, 1.0); }
)GLSL";

static const char* GRID_FRAG_SRC = R"GLSL(
#version 150 core
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }
)GLSL";

// ============================================================================

class MeshRenderer {
public:
    // Render settings
    struct Settings {
        float pipeColor[3]   = {0.72f, 0.45f, 0.20f};  // warm steel/brass
        float bgColor[4]     = {0.13f, 0.13f, 0.15f, 1.f};
        float lightDir[3]    = {-0.6f, -1.0f, -0.5f};
        float ambient        = 0.20f;
        float specular       = 0.55f;
        float shininess      = 64.f;
        bool  wireframe      = false;
        bool  showGrid       = true;
        bool  showAxes       = true;
    } settings;

    OrbitCamera camera;

    bool init() {
        if (!compilePipeShader()) return false;
        if (!compileGridShader()) return false;
        buildGrid();
        buildAxes();
        return true;
    }

    // Upload (or re-upload) a mesh to the GPU.
    void uploadMesh(const TriMesh& mesh) {
        m_triCount = (int)mesh.triangles.size();

        // Build interleaved buffer: [pos.x, pos.y, pos.z, n.x, n.y, n.z] per vertex
        std::vector<float> vbuf;
        vbuf.reserve(mesh.vertices.size() * 6);
        for (auto& v : mesh.vertices) {
            vbuf.push_back(v.pos.x);    vbuf.push_back(v.pos.y);    vbuf.push_back(v.pos.z);
            vbuf.push_back(v.normal.x); vbuf.push_back(v.normal.y); vbuf.push_back(v.normal.z);
        }

        std::vector<uint32_t> ibuf;
        ibuf.reserve(mesh.triangles.size() * 3);
        for (auto& t : mesh.triangles) {
            ibuf.push_back(t.a); ibuf.push_back(t.b); ibuf.push_back(t.c);
        }

        // Compute bounding sphere for auto-fit
        Vec3 centre{0,0,0};
        for (auto& v : mesh.vertices) centre += v.pos;
        centre = centre * (1.f / (float)mesh.vertices.size());
        float maxR = 0;
        for (auto& v : mesh.vertices) maxR = std::max(maxR, (v.pos - centre).length());
        camera.fitToBounds(centre, maxR);

        if (m_vao == 0) {
            glGenVertexArrays(1, &m_vao);
            glGenBuffers(1, &m_vbo);
            glGenBuffers(1, &m_ibo);
        }

        glBindVertexArray(m_vao);

        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, vbuf.size() * sizeof(float), vbuf.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ibuf.size() * sizeof(uint32_t), ibuf.data(), GL_DYNAMIC_DRAW);

        GLsizei stride = 6 * sizeof(float);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));

        glBindVertexArray(0);
    }

    // Call once per frame, with the viewport size.
    void render(int vpW, int vpH) {
        glViewport(0, 0, vpW, vpH);
        glClearColor(settings.bgColor[0], settings.bgColor[1],
                     settings.bgColor[2], settings.bgColor[3]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        float aspect = (float)vpW / std::max(1.f, (float)vpH);
        float proj[16], view[16], norm3[9];
        camera.buildProjMatrix(proj, aspect);
        camera.buildViewMatrix(view);
        camera.buildNormalMatrix(norm3);
        Vec3 camPos = camera.position();

        if (settings.showGrid) drawGrid(proj, view);
        if (settings.showAxes) drawAxes(proj, view);
        if (m_vao && m_triCount > 0) drawMesh(proj, view, norm3, camPos);
    }

    void cleanup() {
        if (m_vao) { glDeleteVertexArrays(1, &m_vao); m_vao = 0; }
        if (m_vbo) { glDeleteBuffers(1, &m_vbo); m_vbo = 0; }
        if (m_ibo) { glDeleteBuffers(1, &m_ibo); m_ibo = 0; }
        if (m_gridVao) { glDeleteVertexArrays(1, &m_gridVao); m_gridVao = 0; }
        if (m_gridVbo) { glDeleteBuffers(1, &m_gridVbo); m_gridVbo = 0; }
        if (m_axesVao) { glDeleteVertexArrays(1, &m_axesVao); m_axesVao = 0; }
        if (m_axesVbo) { glDeleteBuffers(1, &m_axesVbo); m_axesVbo = 0; }
        if (m_pipeShader) { glDeleteProgram(m_pipeShader); m_pipeShader = 0; }
        if (m_gridShader) { glDeleteProgram(m_gridShader); m_gridShader = 0; }
    }

private:
    // GPU objects
    GLuint m_vao = 0, m_vbo = 0, m_ibo = 0;
    GLuint m_gridVao = 0, m_gridVbo = 0;
    GLuint m_axesVao = 0, m_axesVbo = 0;
    GLuint m_pipeShader = 0, m_gridShader = 0;
    int    m_triCount = 0;
    int    m_axesVertCount = 0;

    // ---- Pipe mesh draw ----------------------------------------------------
    void drawMesh(const float* proj, const float* view, const float* norm3, const Vec3& camPos) {
        glUseProgram(m_pipeShader);

        setUniformMat4("uProj",      proj);
        setUniformMat4("uView",      view);
        setUniformMat3("uNormalMat", norm3);

        glUniform3f(glGetUniformLocation(m_pipeShader, "uCameraPos"),
                    camPos.x, camPos.y, camPos.z);
        glUniform3fv(glGetUniformLocation(m_pipeShader, "uPipeColor"), 1,
                     settings.pipeColor);
        glUniform3fv(glGetUniformLocation(m_pipeShader, "uLightDir"), 1,
                     settings.lightDir);
        glUniform1f(glGetUniformLocation(m_pipeShader, "uAmbient"),   settings.ambient);
        glUniform1f(glGetUniformLocation(m_pipeShader, "uSpecular"),  settings.specular);
        glUniform1f(glGetUniformLocation(m_pipeShader, "uShininess"), settings.shininess);
        glUniform1i(glGetUniformLocation(m_pipeShader, "uWireframe"), settings.wireframe ? 1 : 0);

        glBindVertexArray(m_vao);

        if (settings.wireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
        glDrawElements(GL_TRIANGLES, m_triCount * 3, GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glBindVertexArray(0);
        glUseProgram(0);
    }

    // ---- Grid --------------------------------------------------------------
    void buildGrid() {
        const int HALF = 5;
        const float STEP = 0.5f;
        std::vector<float> lines;
        auto addLine = [&](float x0, float y0, float z0, float x1, float y1, float z1) {
            lines.insert(lines.end(), {x0,y0,z0, x1,y1,z1});
        };

        for (int i = -HALF; i <= HALF; ++i) {
            float p = i * STEP;
            addLine(p, 0, -HALF*STEP,  p, 0,  HALF*STEP);
            addLine(-HALF*STEP, 0, p,   HALF*STEP, 0, p);
        }

        glGenVertexArrays(1, &m_gridVao);
        glGenBuffers(1, &m_gridVbo);
        glBindVertexArray(m_gridVao);
        glBindBuffer(GL_ARRAY_BUFFER, m_gridVbo);
        glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), nullptr);
        glBindVertexArray(0);
        m_gridLineVerts = (int)lines.size() / 3;
    }

    void drawGrid(const float* proj, const float* view) {
        glUseProgram(m_gridShader);
        setUniformMat4("uProj", proj);
        setUniformMat4("uView", view);
        glUniform3f(glGetUniformLocation(m_gridShader, "uColor"), 0.28f, 0.28f, 0.32f);
        glBindVertexArray(m_gridVao);
        glDrawArrays(GL_LINES, 0, m_gridLineVerts);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    // ---- Axes --------------------------------------------------------------
    void buildAxes() {
        float axes[] = {
            // X — red
            0,0,0,  1,0,0,   0.5f,0,0,  1,0.2f,0.2f,
            // Y — green
            0,0,0,  0,1,0,   0,0.5f,0,  0.2f,1,0.2f,
            // Z — blue
            0,0,0,  0,0,1,   0,0,0.5f,  0.2f,0.2f,1,
        };
        // We'll store these as colored lines, using a separate draw call per axis
        // For simplicity, just store all 6 verts (positions only) and draw with fixed colors
        float axesPos[] = {
            0,0,0,  0.5f,0,0,
            0,0,0,  0,0.5f,0,
            0,0,0,  0,0,0.5f,
        };
        glGenVertexArrays(1, &m_axesVao);
        glGenBuffers(1, &m_axesVbo);
        glBindVertexArray(m_axesVao);
        glBindBuffer(GL_ARRAY_BUFFER, m_axesVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(axesPos), axesPos, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), nullptr);
        glBindVertexArray(0);
        m_axesVertCount = 6;
    }

    void drawAxes(const float* proj, const float* view) {
        glUseProgram(m_gridShader);
        setUniformMat4("uProj", proj);
        setUniformMat4("uView", view);
        glBindVertexArray(m_axesVao);
        // X axis - red
        glUniform3f(glGetUniformLocation(m_gridShader, "uColor"), 0.9f, 0.2f, 0.2f);
        glDrawArrays(GL_LINES, 0, 2);
        // Y axis - green
        glUniform3f(glGetUniformLocation(m_gridShader, "uColor"), 0.2f, 0.9f, 0.2f);
        glDrawArrays(GL_LINES, 2, 2);
        // Z axis - blue
        glUniform3f(glGetUniformLocation(m_gridShader, "uColor"), 0.2f, 0.4f, 0.9f);
        glDrawArrays(GL_LINES, 4, 2);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    int m_gridLineVerts = 0;

    // ---- Shader helpers ----------------------------------------------------
    bool compilePipeShader() {
        m_pipeShader = compileProgram(PIPE_VERT_SRC, PIPE_FRAG_SRC);
        if (!m_pipeShader) { std::cerr << "[MeshRenderer] Pipe shader failed\n"; return false; }
        // bind attribute locations
        glBindAttribLocation(m_pipeShader, 0, "aPos");
        glBindAttribLocation(m_pipeShader, 1, "aNormal");
        glLinkProgram(m_pipeShader);
        return true;
    }

    bool compileGridShader() {
        m_gridShader = compileProgram(GRID_VERT_SRC, GRID_FRAG_SRC);
        if (!m_gridShader) { std::cerr << "[MeshRenderer] Grid shader failed\n"; return false; }
        glBindAttribLocation(m_gridShader, 0, "aPos");
        glLinkProgram(m_gridShader);
        return true;
    }

    static GLuint compileShader(GLenum type, const char* src) {
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[1024]; glGetShaderInfoLog(sh, 1024, nullptr, buf);
            std::cerr << "[Shader] " << buf << "\n";
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    }

    static GLuint compileProgram(const char* vert, const char* frag) {
        GLuint vs = compileShader(GL_VERTEX_SHADER, vert);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, frag);
        if (!vs || !fs) { glDeleteShader(vs); glDeleteShader(fs); return 0; }
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char buf[1024]; glGetProgramInfoLog(prog, 1024, nullptr, buf);
            std::cerr << "[Program] " << buf << "\n";
        }
        glDeleteShader(vs);
        glDeleteShader(fs);
        return ok ? prog : 0;
    }

    void setUniformMat4(const char* name, const float* m) {
        GLint loc = glGetUniformLocation(m_pipeShader, name);
        if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, m);
    }

    void setUniformMat3(const char* name, const float* m) {
        GLint loc = glGetUniformLocation(m_pipeShader, name);
        if (loc >= 0) glUniformMatrix3fv(loc, 1, GL_FALSE, m);
    }
};
