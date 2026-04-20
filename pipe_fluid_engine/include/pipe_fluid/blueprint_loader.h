#pragma once
// ============================================================================
// blueprint_loader
//
// Thin wrapper around pipe_engine's BlueprintParser that loads a pipe
// blueprint (.pipe) from disk and returns a PipeNetwork ready to be
// voxelized into a fluid scene.
//
// We keep this wrapper separate so pipe_fluid_engine consumers don't have
// to include Blueprint/blueprint_parser.h directly, and so we can add
// fluid-specific overrides later (e.g. default cell size, padding hints).
// ============================================================================

#include <string>

struct PipeNetwork;   // Geometry/pipe_network.h

namespace pipe_fluid {

struct LoadResult {
    bool ok = false;
    std::string error;
};

// Loads a blueprint file from `path` into `network`. Returns ok=false with
// a human-readable error on parse failure.
LoadResult loadBlueprintFile(const std::string& path, PipeNetwork& network);

// Parses blueprint content from a string (same format as .pipe files).
LoadResult loadBlueprintString(const std::string& content, PipeNetwork& network);

} // namespace pipe_fluid
