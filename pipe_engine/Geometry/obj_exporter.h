#pragma once
#include "mesh_generator.h"
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstdint>

// ============================================================================
// OBJ / STL exporter for TriMesh.
// ============================================================================

class ObjExporter {
public:
    // Export to Wavefront OBJ (positions + normals + texture coords + faces).
    static bool writeOBJ(const TriMesh& mesh, const std::string& path,
                         const std::string& objectName = "pipe") {
        std::ofstream f(path);
        if (!f.is_open()) {
            std::cerr << "[ObjExporter] Failed to open " << path << "\n";
            return false;
        }

        f << "# Pipe Engine — generated mesh\n";
        f << "# Vertices: " << mesh.vertices.size()
          << "  Triangles: " << mesh.triangles.size() << "\n";
        f << "o " << objectName << "\n\n";

        f << std::fixed << std::setprecision(6);

        // Positions
        for (auto& v : mesh.vertices) {
            f << "v " << v.pos.x << " " << v.pos.y << " " << v.pos.z << "\n";
        }
        f << "\n";

        // Normals
        for (auto& v : mesh.vertices) {
            f << "vn " << v.normal.x << " " << v.normal.y << " " << v.normal.z << "\n";
        }
        f << "\n";

        // Texture coords
        for (auto& v : mesh.vertices) {
            f << "vt " << v.u << " " << v.v << "\n";
        }
        f << "\n";

        // Faces (OBJ is 1-indexed)
        for (auto& tri : mesh.triangles) {
            uint32_t a = tri.a + 1, b = tri.b + 1, c = tri.c + 1;
            f << "f " << a << "/" << a << "/" << a
              << " " << b << "/" << b << "/" << b
              << " " << c << "/" << c << "/" << c << "\n";
        }

        f.close();
        std::cout << "[ObjExporter] Wrote " << path
                  << " (" << mesh.vertices.size() << " verts, "
                  << mesh.triangles.size() << " tris)\n";
        return true;
    }

    // Export to binary STL.
    static bool writeSTL(const TriMesh& mesh, const std::string& path,
                         const std::string& header = "pipe_engine") {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[ObjExporter] Failed to open " << path << "\n";
            return false;
        }

        // 80-byte header
        char hdr[80] = {};
        std::snprintf(hdr, sizeof(hdr), "%s", header.c_str());
        f.write(hdr, 80);

        // Triangle count
        uint32_t triCount = (uint32_t)mesh.triangles.size();
        f.write(reinterpret_cast<const char*>(&triCount), 4);

        // Triangles
        for (auto& tri : mesh.triangles) {
            const auto& va = mesh.vertices[tri.a];
            const auto& vb = mesh.vertices[tri.b];
            const auto& vc = mesh.vertices[tri.c];

            // Compute face normal
            Vec3 e1 = vb.pos - va.pos;
            Vec3 e2 = vc.pos - va.pos;
            Vec3 fn = e1.cross(e2).normalized();

            // Normal (3 floats)
            f.write(reinterpret_cast<const char*>(&fn.x), 4);
            f.write(reinterpret_cast<const char*>(&fn.y), 4);
            f.write(reinterpret_cast<const char*>(&fn.z), 4);

            // Vertex A
            f.write(reinterpret_cast<const char*>(&va.pos.x), 4);
            f.write(reinterpret_cast<const char*>(&va.pos.y), 4);
            f.write(reinterpret_cast<const char*>(&va.pos.z), 4);
            // Vertex B
            f.write(reinterpret_cast<const char*>(&vb.pos.x), 4);
            f.write(reinterpret_cast<const char*>(&vb.pos.y), 4);
            f.write(reinterpret_cast<const char*>(&vb.pos.z), 4);
            // Vertex C
            f.write(reinterpret_cast<const char*>(&vc.pos.x), 4);
            f.write(reinterpret_cast<const char*>(&vc.pos.y), 4);
            f.write(reinterpret_cast<const char*>(&vc.pos.z), 4);

            // Attribute byte count
            uint16_t attr = 0;
            f.write(reinterpret_cast<const char*>(&attr), 2);
        }

        f.close();
        std::cout << "[ObjExporter] Wrote STL " << path
                  << " (" << triCount << " tris)\n";
        return true;
    }
};
