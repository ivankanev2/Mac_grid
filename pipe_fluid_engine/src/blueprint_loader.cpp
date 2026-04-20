#include "pipe_fluid/blueprint_loader.h"

#include "pipe_network.h"        // pipe_engine/Geometry
#include "blueprint_parser.h"    // pipe_engine/Blueprint

#include <exception>
#include <fstream>

namespace {

pipe_fluid::LoadResult validateLoadedBlueprint(const std::string& path,
                                               const PipeNetwork& network) {
    pipe_fluid::LoadResult r;
    if (network.segments.empty()) {
        r.ok = false;
        r.error = path.empty()
            ? "blueprint parse failed: blueprint contains no pipe segments"
            : std::string("blueprint parse failed: no pipe segments were loaded from '") +
                  path + "'";
        return r;
    }
    if (!network.validate()) {
        r.ok = false;
        r.error = path.empty()
            ? "blueprint parse failed: pipe network connectivity validation failed"
            : std::string("blueprint parse failed: pipe network connectivity validation failed for '") +
                  path + "'";
        return r;
    }
    r.ok = true;
    return r;
}

} // namespace

namespace pipe_fluid {

LoadResult loadBlueprintFile(const std::string& path, PipeNetwork& network) {
    LoadResult r;

    std::ifstream f(path);
    if (!f.is_open()) {
        r.ok = false;
        r.error = std::string("blueprint load failed: cannot open '") + path + "'";
        return r;
    }

    try {
        network = BlueprintParser::parse(path);
        return validateLoadedBlueprint(path, network);
    } catch (const std::exception& e) {
        r.ok = false;
        r.error = std::string("blueprint parse failed: ") + e.what();
    } catch (...) {
        r.ok = false;
        r.error = "blueprint parse failed: unknown error";
    }
    return r;
}

LoadResult loadBlueprintString(const std::string& content, PipeNetwork& network) {
    LoadResult r;
    try {
        network = BlueprintParser::parseString(content);
        return validateLoadedBlueprint("", network);
    } catch (const std::exception& e) {
        r.ok = false;
        r.error = std::string("blueprint parse failed: ") + e.what();
    } catch (...) {
        r.ok = false;
        r.error = "blueprint parse failed: unknown error";
    }
    return r;
}

} // namespace pipe_fluid
