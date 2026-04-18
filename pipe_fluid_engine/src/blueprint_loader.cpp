#include "pipe_fluid/blueprint_loader.h"

#include "pipe_network.h"        // pipe_engine/Geometry
#include "blueprint_parser.h"    // pipe_engine/Blueprint

#include <exception>

namespace pipe_fluid {

LoadResult loadBlueprintFile(const std::string& path, PipeNetwork& network) {
    LoadResult r;
    try {
        network = BlueprintParser::parse(path);
        r.ok = true;
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
        r.ok = true;
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
