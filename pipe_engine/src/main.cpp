#include "../Geometry/pipe_network.h"
#include "../Geometry/mesh_generator.h"
#include "../Geometry/obj_exporter.h"
#include "../Blueprint/blueprint_parser.h"
#include "../Voxelizer/voxelizer.h"

#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================================
// Pipe Engine — CLI demo
//
// Usage:
//   pipe_engine                         → generates default example pipes
//   pipe_engine <blueprint.pipe>        → parses blueprint and generates mesh
//   pipe_engine <blueprint.pipe> <out>  → custom output directory
// ============================================================================

// Build a demo L-shaped pipe programmatically.
PipeNetwork buildDemoLPipe() {
    PipeNetwork net;
    net.name = "demo_L_pipe";
    net.defaultInnerRadius = 0.05f;
    net.defaultOuterRadius = 0.06f;

    net.begin({0, 0, 0}, {0, 0, 1});  // start at origin, heading +Z
    net.addStraight(1.0f);             // 1 m straight section
    net.addBend90({1, 0, 0}, 0.15f);   // 90° turn toward +X
    net.addStraight(0.8f);             // 0.8 m straight section

    return net;
}

// Build a more complex S-bend pipe.
PipeNetwork buildDemoSPipe() {
    PipeNetwork net;
    net.name = "demo_S_pipe";
    net.defaultInnerRadius = 0.04f;
    net.defaultOuterRadius = 0.05f;

    net.begin({0, 0, 0}, {0, 0, 1});
    net.addStraight(0.5f);
    net.addBend90({1, 0, 0}, 0.12f);   // turn right
    net.addStraight(0.3f);
    net.addBend90({0, 0, 1}, 0.12f);   // turn back forward
    net.addStraight(0.5f);

    return net;
}

// Build a U-bend (pipe goes forward, turns 180°, comes back).
PipeNetwork buildDemoUPipe() {
    PipeNetwork net;
    net.name = "demo_U_pipe";
    net.defaultInnerRadius = 0.03f;
    net.defaultOuterRadius = 0.04f;

    net.begin({0, 0, 0}, {0, 0, 1});
    net.addStraight(0.6f);
    net.addBend({0, 0, -1}, 0.20f);   // 180° turn
    net.addStraight(0.6f);

    return net;
}

// Build a vertical riser (pipe goes up with bends).
PipeNetwork buildDemoRiser() {
    PipeNetwork net;
    net.name = "demo_riser";
    net.defaultInnerRadius = 0.06f;
    net.defaultOuterRadius = 0.075f;

    net.begin({0, 0, 0}, {1, 0, 0});   // heading +X
    net.addStraight(0.8f);
    net.addBend90({0, 1, 0}, 0.18f);    // turn upward
    net.addStraight(1.2f);              // vertical section
    net.addBend90({1, 0, 0}, 0.18f);    // turn horizontal again
    net.addStraight(0.5f);

    return net;
}

void generateAndExport(const PipeNetwork& net, const std::string& outDir) {
    std::cout << "\n========================================\n";
    net.printSummary();

    if (!net.validate()) {
        std::cerr << "  WARNING: Network has connectivity gaps!\n";
    }

    MeshGenerator gen;
    gen.ringSlices = 32;
    gen.samplesPerMetre = 50.f;

    TriMesh mesh = gen.generatePipeMesh(net);
    std::cout << "  Mesh: " << mesh.vertices.size() << " vertices, "
              << mesh.triangles.size() << " triangles\n";

    fs::create_directories(outDir);

    std::string objPath = outDir + "/" + net.name + ".obj";
    std::string stlPath = outDir + "/" + net.name + ".stl";

    ObjExporter::writeOBJ(mesh, objPath, net.name);
    ObjExporter::writeSTL(mesh, stlPath, net.name);
}

int main(int argc, char* argv[]) {
    std::cout << "=== Pipe Engine v0.1 ===\n";

    std::string outDir = "output";

    if (argc >= 2) {
        // Parse a blueprint file
        std::string blueprintPath = argv[1];
        if (argc >= 3) outDir = argv[2];

        std::cout << "Parsing blueprint: " << blueprintPath << "\n";
        PipeNetwork net = BlueprintParser::parse(blueprintPath);

        if (net.numSegments() == 0) {
            std::cerr << "ERROR: No segments parsed from blueprint.\n";
            return 1;
        }

        generateAndExport(net, outDir);
    }
    else {
        // Generate demo pipes
        std::cout << "No blueprint specified — generating demo pipes.\n\n";

        generateAndExport(buildDemoLPipe(),  outDir);
        generateAndExport(buildDemoSPipe(),  outDir);
        generateAndExport(buildDemoUPipe(),  outDir);
        generateAndExport(buildDemoRiser(),  outDir);

        // Also demonstrate the voxelizer on the L-pipe
        std::cout << "\n--- Voxelizer demo (L-pipe) ---\n";
        PipeVoxelizer vox;
        vox.cellSize = 0.01f;
        vox.padding  = 0.05f;
        VoxelGrid grid = vox.voxelize(buildDemoLPipe());
        grid.printStats();
    }

    std::cout << "\nDone. Output in: " << outDir << "/\n";
    return 0;
}
