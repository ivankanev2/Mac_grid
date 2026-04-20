#pragma once
#include "../Geometry/pipe_network.h"
#include "../Geometry/mesh_generator.h"
#include "../Geometry/obj_exporter.h"
#include "../Blueprint/blueprint_parser.h"
#include "../Renderer/mesh_renderer.h"

#include "imgui.h"
#include <string>
#include <vector>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;

// ============================================================================
// PipeUI: all ImGui panels for the live pipe engine viewer.
// ============================================================================

struct PipeUI {
    // ---- Shared state ------------------------------------------------------
    PipeNetwork     network;
    MeshGenerator   generator;
    TriMesh         mesh;
    bool            meshDirty = true;   // rebuild mesh on next frame?

    // Blueprint text editor buffer
    char  blueprintBuf[4096] = {};
    bool  blueprintEdited    = false;
    std::string statusMsg;
    float statusTimer = 0.f;

    // Segment builder UI state
    enum class AddMode { None, Straight, Bend };
    AddMode addMode       = AddMode::None;
    float   straightLen   = 0.5f;
    float   bendDir[3]    = {1.f, 0.f, 0.f};
    float   bendRadius    = 0.15f;
    bool    hasPipeStarted = false;
    float   startPos[3]   = {0, 0, 0};
    float   startDir[3]   = {0, 0, 1};
    float   innerRadius   = 0.05f;
    float   outerRadius   = 0.06f;

    // Export
    char exportDir[512]  = "output";
    bool exportedThisFrame = false;

    // Callback: called whenever the mesh is rebuilt
    std::function<void(const TriMesh&)> onMeshRebuilt;

    // ---- Init --------------------------------------------------------------
    void init() {
        std::snprintf(blueprintBuf, sizeof(blueprintBuf),
            "# Type a blueprint here, then click 'Apply'\n"
            "name my_pipe\n"
            "inner_radius 0.05\n"
            "outer_radius 0.06\n"
            "start 0 0 0\n"
            "direction 0 0 1\n"
            "straight 1.0\n"
            "bend90 +x 0.15\n"
            "straight 0.8\n");
        generator.ringSlices    = 32;
        generator.samplesPerMetre = 50.f;
        rebuildFromBlueprint();
    }

    // ---- Per-frame update --------------------------------------------------
    void update(float dt) {
        if (statusTimer > 0.f) statusTimer -= dt;

        if (meshDirty) {
            rebuildMesh();
            meshDirty = false;
        }
    }

    // ---- Rebuild helpers ---------------------------------------------------
    void rebuildMesh() {
        if (network.numSegments() == 0) {
            mesh.clear();
        } else {
            mesh = generator.generatePipeMesh(network);
        }
        if (onMeshRebuilt) onMeshRebuilt(mesh);
    }

    void rebuildFromBlueprint() {
        network = BlueprintParser::parseString(std::string(blueprintBuf));
        meshDirty = true;
        hasPipeStarted = network.numSegments() > 0;
        setStatus("Blueprint applied — " + std::to_string(network.numSegments()) + " segments");
    }

    void setStatus(const std::string& msg) {
        statusMsg  = msg;
        statusTimer = 3.f;
    }

    // ---- Main panel layout -------------------------------------------------
    void drawAll(MeshRenderer& renderer) {
        drawMenuBar();
        drawBlueprintPanel();
        drawSegmentBuilderPanel();
        drawNetworkInfoPanel();
        drawRenderSettingsPanel(renderer);
        drawExportPanel();
        drawStatusBar();
    }

private:
    // ---- Menu bar ----------------------------------------------------------
    void drawMenuBar() {
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("New Network")) {
                    network = PipeNetwork{};
                    hasPipeStarted = false;
                    meshDirty = true;
                    setStatus("New network created");
                }
                if (ImGui::MenuItem("Load Blueprint...")) {
                    // TODO: file picker — for now tell user to paste in the editor
                    setStatus("Paste your blueprint in the editor panel");
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Export OBJ + STL")) {
                    exportMesh();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Quit")) {
                    // Signal quit via glfwSetWindowShouldClose — handled in main
                    m_wantsQuit = true;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Blueprint Editor",   nullptr, &m_showBlueprint);
                ImGui::MenuItem("Segment Builder",    nullptr, &m_showBuilder);
                ImGui::MenuItem("Network Info",       nullptr, &m_showNetInfo);
                ImGui::MenuItem("Render Settings",    nullptr, &m_showRenderSettings);
                ImGui::MenuItem("Export",             nullptr, &m_showExport);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Presets")) {
                if (ImGui::MenuItem("L-pipe (horizontal)"))  loadPreset(PRESET_L);
                if (ImGui::MenuItem("S-bend"))               loadPreset(PRESET_S);
                if (ImGui::MenuItem("U-bend (180°)"))        loadPreset(PRESET_U);
                if (ImGui::MenuItem("Industrial riser"))     loadPreset(PRESET_RISER);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
    }

    // ---- Blueprint text editor --------------------------------------------
    void drawBlueprintPanel() {
        if (!m_showBlueprint) return;
        ImGui::SetNextWindowSize({420, 340}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Blueprint Editor", &m_showBlueprint)) {
            ImGui::TextDisabled("Edit the .pipe blueprint then click Apply");
            ImGui::Separator();

            ImVec2 editorSize = {ImGui::GetContentRegionAvail().x,
                                 ImGui::GetContentRegionAvail().y - 38.f};
            if (ImGui::InputTextMultiline("##bp", blueprintBuf, sizeof(blueprintBuf),
                                          editorSize, ImGuiInputTextFlags_AllowTabInput)) {
                blueprintEdited = true;
            }

            bool canApply = blueprintEdited;
            if (!canApply) ImGui::BeginDisabled();
            if (ImGui::Button("Apply##bp", {100,0})) {
                rebuildFromBlueprint();
                blueprintEdited = false;
            }
            if (!canApply) ImGui::EndDisabled();

            ImGui::SameLine();
            if (ImGui::Button("Revert##bp", {80,0})) {
                blueprintEdited = false;
                // reload from current network (rebuild text)
                setStatus("Reverted (text reset)");
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu segments)", network.numSegments());
        }
        ImGui::End();
    }

    // ---- Interactive segment builder ----------------------------------------
    void drawSegmentBuilderPanel() {
        if (!m_showBuilder) return;
        ImGui::SetNextWindowSize({320, 320}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Segment Builder", &m_showBuilder)) {
            // Start position / direction (only editable before first segment)
            if (!hasPipeStarted) {
                ImGui::SeparatorText("Start");
                ImGui::DragFloat3("Position##sp", startPos, 0.01f);
                ImGui::DragFloat3("Direction##sd", startDir, 0.01f);
                ImGui::DragFloat("Inner radius (m)##ir", &innerRadius, 0.001f, 0.005f, 1.f);
                ImGui::DragFloat("Outer radius (m)##or", &outerRadius, 0.001f, 0.006f, 1.1f);
                innerRadius = std::min(innerRadius, outerRadius - 0.001f);

                if (ImGui::Button("Set start", {-1, 0})) {
                    network = PipeNetwork{};
                    network.defaultInnerRadius = innerRadius;
                    network.defaultOuterRadius = outerRadius;
                    network.begin({startPos[0], startPos[1], startPos[2]},
                                  {startDir[0], startDir[1], startDir[2]});
                    hasPipeStarted = true;
                    setStatus("Start set — add segments");
                }
                ImGui::Separator();
            } else {
                // Show current cursor info
                ImGui::SeparatorText("Current end-point");
                Vec3 c = network.cursor, d = network.cursorDir;
                ImGui::Text("pos  (%.3f, %.3f, %.3f)", c.x, c.y, c.z);
                ImGui::Text("dir  (%.3f, %.3f, %.3f)", d.x, d.y, d.z);
                ImGui::Separator();
            }

            if (hasPipeStarted) {
                ImGui::SeparatorText("Add Segment");

                // Straight
                if (ImGui::CollapsingHeader("Straight", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat("Length (m)##sl", &straightLen, 0.01f, 0.01f, 50.f);
                    if (ImGui::Button("Add straight##s", {-1, 0})) {
                        network.addStraight(straightLen);
                        meshDirty = true;
                        setStatus("Added straight " + formatF(straightLen) + " m");
                    }
                }

                // Bend
                if (ImGui::CollapsingHeader("Bend", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat3("New direction##bd", bendDir, 0.01f, -1.f, 1.f);
                    ImGui::DragFloat("Bend radius (m)##br", &bendRadius, 0.005f, 0.05f, 2.f);

                    ImGui::Text("Quick:");
                    ImGui::SameLine();
                    if (ImGui::SmallButton("+X")) { bendDir[0]=1;bendDir[1]=0;bendDir[2]=0; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("-X")) { bendDir[0]=-1;bendDir[1]=0;bendDir[2]=0; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("+Y")) { bendDir[0]=0;bendDir[1]=1;bendDir[2]=0; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("-Y")) { bendDir[0]=0;bendDir[1]=-1;bendDir[2]=0; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("+Z")) { bendDir[0]=0;bendDir[1]=0;bendDir[2]=1; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("-Z")) { bendDir[0]=0;bendDir[1]=0;bendDir[2]=-1; }

                    if (ImGui::Button("Add bend##b", {-1, 0})) {
                        network.addBend({bendDir[0], bendDir[1], bendDir[2]}, bendRadius);
                        meshDirty = true;
                        setStatus("Added bend toward ("
                            + formatF(bendDir[0]) + ","
                            + formatF(bendDir[1]) + ","
                            + formatF(bendDir[2]) + ")");
                    }
                }

                ImGui::Separator();
                if (ImGui::Button("Undo last segment", {-1, 0})) {
                    if (!network.segments.empty()) {
                        network.segments.pop_back();
                        // Recompute cursor from remaining segments
                        if (!network.segments.empty()) {
                            auto& last = *network.segments.back();
                            network.cursor    = last.endPoint();
                            network.cursorDir = last.tangent(1.f);
                        } else {
                            hasPipeStarted = false;
                        }
                        meshDirty = true;
                        setStatus("Removed last segment");
                    }
                }

                if (ImGui::Button("Clear all", {-1, 0})) {
                    network = PipeNetwork{};
                    hasPipeStarted = false;
                    meshDirty = true;
                    setStatus("Cleared");
                }
            }
        }
        ImGui::End();
    }

    // ---- Network info -------------------------------------------------------
    void drawNetworkInfoPanel() {
        if (!m_showNetInfo) return;
        ImGui::SetNextWindowSize({280, 220}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Network Info", &m_showNetInfo)) {
            ImGui::Text("Name:     %s", network.name.c_str());
            ImGui::Text("Segments: %zu", network.numSegments());
            ImGui::Text("Length:   %.3f m", network.totalLength());
            ImGui::Text("Inner R:  %.4f m", network.defaultInnerRadius);
            ImGui::Text("Outer R:  %.4f m", network.defaultOuterRadius);
            ImGui::Separator();
            ImGui::Text("Mesh:");
            ImGui::Text("  Vertices:  %zu", mesh.vertices.size());
            ImGui::Text("  Triangles: %zu", mesh.triangles.size());
            ImGui::Separator();

            if (network.numSegments() > 0 && ImGui::CollapsingHeader("Segments")) {
                for (size_t i = 0; i < network.numSegments(); ++i) {
                    auto& s = *network.segments[i];
                    const char* t = (s.type == SegmentType::Straight) ? "Str" : "Bend";
                    Vec3 sp = s.startPoint(), ep = s.endPoint();
                    ImGui::Text("[%zu] %s  %.3fm", i, t, s.arcLength());
                    ImGui::TextDisabled("    (%.2f,%.2f,%.2f)→(%.2f,%.2f,%.2f)",
                        sp.x,sp.y,sp.z, ep.x,ep.y,ep.z);
                }
            }
        }
        ImGui::End();
    }

    // ---- Render settings ---------------------------------------------------
    void drawRenderSettingsPanel(MeshRenderer& renderer) {
        if (!m_showRenderSettings) return;
        ImGui::SetNextWindowSize({280, 260}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Render Settings", &m_showRenderSettings)) {
            ImGui::SeparatorText("Pipe Material");
            ImGui::ColorEdit3("Pipe color",  renderer.settings.pipeColor);
            ImGui::ColorEdit3("Light dir",   renderer.settings.lightDir);
            ImGui::SliderFloat("Ambient",    &renderer.settings.ambient,   0.f, 1.f);
            ImGui::SliderFloat("Specular",   &renderer.settings.specular,  0.f, 1.f);
            ImGui::SliderFloat("Shininess",  &renderer.settings.shininess, 2.f, 256.f);
            ImGui::Separator();
            ImGui::SeparatorText("Viewport");
            ImGui::Checkbox("Wireframe",  &renderer.settings.wireframe);
            ImGui::Checkbox("Show grid",  &renderer.settings.showGrid);
            ImGui::Checkbox("Show axes",  &renderer.settings.showAxes);
            ImGui::ColorEdit4("Background", renderer.settings.bgColor);
            ImGui::Separator();
            ImGui::SeparatorText("Camera");
            ImGui::SliderFloat("Yaw",   &renderer.camera.yawDeg,   -180.f, 180.f);
            ImGui::SliderFloat("Pitch", &renderer.camera.pitchDeg,  -89.f,  89.f);
            ImGui::SliderFloat("Zoom",  &renderer.camera.distance,    0.1f,  20.f);
            if (ImGui::Button("Reset camera")) {
                renderer.camera.yawDeg  = 35.f;
                renderer.camera.pitchDeg = 20.f;
                renderer.camera.distance = 3.f;
                renderer.camera.target  = {0,0,0};
            }
            ImGui::Separator();
            ImGui::SeparatorText("Mesh Quality");
            bool qualityChanged = false;
            if (ImGui::SliderInt("Ring slices", &generator.ringSlices, 6, 64))  qualityChanged = true;
            float spm = generator.samplesPerMetre;
            if (ImGui::SliderFloat("Samples/m",  &spm, 5.f, 100.f)) {
                generator.samplesPerMetre = spm;
                qualityChanged = true;
            }
            if (qualityChanged) meshDirty = true;
        }
        ImGui::End();
    }

    // ---- Export panel ------------------------------------------------------
    void drawExportPanel() {
        if (!m_showExport) return;
        ImGui::SetNextWindowSize({320, 140}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Export", &m_showExport)) {
            ImGui::InputText("Directory", exportDir, sizeof(exportDir));
            if (ImGui::Button("Export OBJ + STL", {-1, 0})) {
                exportMesh();
            }
        }
        ImGui::End();
    }

    void exportMesh() {
        if (mesh.vertices.empty()) {
            setStatus("Nothing to export — add segments first");
            return;
        }
        fs::create_directories(exportDir);
        std::string base = std::string(exportDir) + "/" + network.name;
        ObjExporter::writeOBJ(mesh, base + ".obj", network.name);
        ObjExporter::writeSTL(mesh, base + ".stl", network.name);
        setStatus("Exported to " + base + ".obj / .stl");
    }

    // ---- Status bar --------------------------------------------------------
    void drawStatusBar() {
        ImGuiIO& io = ImGui::GetIO();
        float barH = ImGui::GetFrameHeight() + 4.f;
        ImGui::SetNextWindowPos({0, io.DisplaySize.y - barH});
        ImGui::SetNextWindowSize({io.DisplaySize.x, barH});
        ImGui::SetNextWindowBgAlpha(0.75f);
        ImGui::Begin("##statusbar", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        if (statusTimer > 0.f) {
            ImGui::Text("%s", statusMsg.c_str());
        } else {
            ImGui::TextDisabled("Pipe Engine  |  Left-drag: orbit   Right-drag: pan   Scroll: zoom   Ctrl+Z: undo");
        }
        ImGui::End();
    }

    // ---- Presets -----------------------------------------------------------
    static constexpr const char* PRESET_L =
        "name L_pipe\ninner_radius 0.05\nouter_radius 0.06\nstart 0 0 0\ndirection 0 0 1\n"
        "straight 1.0\nbend90 +x 0.15\nstraight 0.8\n";

    static constexpr const char* PRESET_S =
        "name S_bend\ninner_radius 0.05\nouter_radius 0.06\nstart 0 0 0\ndirection 0 0 1\n"
        "straight 0.5\nbend90 +x 0.12\nstraight 0.3\nbend90 +z 0.12\nstraight 0.5\n";

    static constexpr const char* PRESET_U =
        "name U_bend\ninner_radius 0.04\nouter_radius 0.05\nstart 0 0 0\ndirection 0 0 1\n"
        "straight 0.6\nbend -z 0.20\nstraight 0.6\n";

    static constexpr const char* PRESET_RISER =
        "name industrial_riser\ninner_radius 0.075\nouter_radius 0.090\n"
        "start 0 0 0\ndirection 1 0 0\n"
        "straight 0.8\nbend90 +y 0.20\nstraight 1.5\nbend90 +x 0.20\nstraight 0.6\n";

    void loadPreset(const char* text) {
        std::strncpy(blueprintBuf, text, sizeof(blueprintBuf)-1);
        blueprintEdited = false;
        rebuildFromBlueprint();
    }

    // ---- Visibility flags --------------------------------------------------
    bool m_showBlueprint      = true;
    bool m_showBuilder        = true;
    bool m_showNetInfo        = true;
    bool m_showRenderSettings = true;
    bool m_showExport         = false;

public:
    bool m_wantsQuit = false;

    static std::string formatF(float f) {
        char buf[32]; std::snprintf(buf, sizeof(buf), "%.3f", f);
        return buf;
    }
};
