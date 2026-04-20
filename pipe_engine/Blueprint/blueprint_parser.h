#pragma once
#include "../Geometry/pipe_network.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cctype>

// ============================================================================
// BlueprintParser: reads a simple text-based pipe specification and builds a
//                  PipeNetwork.
//
// Blueprint format (.pipe file):
//   Lines starting with '#' are comments.
//   Blank lines are ignored.
//
//   HEADER section:
//     name <string>
//     inner_radius <float>      # metres
//     outer_radius <float>      # metres
//     start <x> <y> <z>        # starting position
//     direction <dx> <dy> <dz> # initial direction (will be normalised)
//
//   SEGMENT commands:
//     straight <length>
//     bend <dx> <dy> <dz> <bend_radius>
//       — redirects toward direction (dx,dy,dz) with given curvature radius
//     bend90 <axis>  [bend_radius]
//       — 90-degree bend.  axis = +x, -x, +y, -y, +z, -z
//
// Example:
//   name simple_L_pipe
//   inner_radius 0.05
//   outer_radius 0.06
//   start 0 0 0
//   direction 0 0 1
//   straight 1.0
//   bend90 +x 0.15
//   straight 0.8
// ============================================================================

class BlueprintParser {
public:
    static PipeNetwork parse(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            std::cerr << "[BlueprintParser] Cannot open " << filepath << "\n";
            return {};
        }

        PipeNetwork net;
        Vec3 startPos{0,0,0};
        Vec3 startDir{0,0,1};
        bool begun = false;

        std::string line;
        int lineNum = 0;
        while (std::getline(f, line)) {
            ++lineNum;
            // strip comments and whitespace
            auto comment = line.find('#');
            if (comment != std::string::npos) line = line.substr(0, comment);
            line = trim(line);
            if (line.empty()) continue;

            std::istringstream ss(line);
            std::string cmd;
            ss >> cmd;
            toLower(cmd);

            if (cmd == "name") {
                ss >> net.name;
            }
            else if (cmd == "inner_radius") {
                ss >> net.defaultInnerRadius;
            }
            else if (cmd == "outer_radius") {
                ss >> net.defaultOuterRadius;
            }
            else if (cmd == "start") {
                ss >> startPos.x >> startPos.y >> startPos.z;
            }
            else if (cmd == "direction") {
                ss >> startDir.x >> startDir.y >> startDir.z;
            }
            else if (cmd == "straight") {
                ensureBegun(net, startPos, startDir, begun);
                float length = 1.f;
                ss >> length;
                net.addStraight(length);
            }
            else if (cmd == "bend") {
                ensureBegun(net, startPos, startDir, begun);
                float dx, dy, dz, br = 0.15f;
                ss >> dx >> dy >> dz;
                if (!(ss >> br)) br = 0.15f;
                net.addBend({dx, dy, dz}, br);
            }
            else if (cmd == "bend90") {
                ensureBegun(net, startPos, startDir, begun);
                std::string axis;
                float br = 0.15f;
                ss >> axis;
                if (!(ss >> br)) br = 0.15f;
                Vec3 dir = parseAxis(axis);
                net.addBend(dir, br);
            }
            else {
                std::cerr << "[BlueprintParser] Unknown command '" << cmd
                          << "' at line " << lineNum << "\n";
            }
        }

        return net;
    }

    // Convenience: parse from a string (for testing).
    static PipeNetwork parseString(const std::string& content) {
        // Write to a temp buffer and parse line by line
        PipeNetwork net;
        Vec3 startPos{0,0,0};
        Vec3 startDir{0,0,1};
        bool begun = false;

        std::istringstream file(content);
        std::string line;
        while (std::getline(file, line)) {
            auto comment = line.find('#');
            if (comment != std::string::npos) line = line.substr(0, comment);
            line = trim(line);
            if (line.empty()) continue;

            std::istringstream ss(line);
            std::string cmd;
            ss >> cmd;
            toLower(cmd);

            if (cmd == "name")             { ss >> net.name; }
            else if (cmd == "inner_radius") { ss >> net.defaultInnerRadius; }
            else if (cmd == "outer_radius") { ss >> net.defaultOuterRadius; }
            else if (cmd == "start")        { ss >> startPos.x >> startPos.y >> startPos.z; }
            else if (cmd == "direction")    { ss >> startDir.x >> startDir.y >> startDir.z; }
            else if (cmd == "straight") {
                ensureBegun(net, startPos, startDir, begun);
                float length = 1.f; ss >> length;
                net.addStraight(length);
            }
            else if (cmd == "bend") {
                ensureBegun(net, startPos, startDir, begun);
                float dx, dy, dz, br = 0.15f;
                ss >> dx >> dy >> dz;
                if (!(ss >> br)) br = 0.15f;
                net.addBend({dx, dy, dz}, br);
            }
            else if (cmd == "bend90") {
                ensureBegun(net, startPos, startDir, begun);
                std::string axis; float br = 0.15f;
                ss >> axis;
                if (!(ss >> br)) br = 0.15f;
                net.addBend(parseAxis(axis), br);
            }
        }
        return net;
    }

private:
    static void ensureBegun(PipeNetwork& net, const Vec3& pos, const Vec3& dir, bool& begun) {
        if (!begun) {
            net.begin(pos, dir);
            begun = true;
        }
    }

    static Vec3 parseAxis(const std::string& s) {
        if (s == "+x" || s == "x")  return { 1, 0, 0};
        if (s == "-x")              return {-1, 0, 0};
        if (s == "+y" || s == "y")  return { 0, 1, 0};
        if (s == "-y")              return { 0,-1, 0};
        if (s == "+z" || s == "z")  return { 0, 0, 1};
        if (s == "-z")              return { 0, 0,-1};
        std::cerr << "[BlueprintParser] Unknown axis '" << s << "', defaulting to +x\n";
        return {1, 0, 0};
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end   = s.find_last_not_of(" \t\r\n");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }

    static void toLower(std::string& s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return std::tolower(c); });
    }
};
