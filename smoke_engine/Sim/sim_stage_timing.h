#pragma once

struct SimStageTimings {
    float setupMs = 0.0f;
    float advectVelocityMs = 0.0f;
    float forcesMs = 0.0f;
    float diffuseVelocityMs = 0.0f;
    float vorticityMs = 0.0f;
    float projectMs = 0.0f;
    float advectScalarsMs = 0.0f;
    float diffuseScalarsMs = 0.0f;
    float particleToGridMs = 0.0f;
    float liquidMaskMs = 0.0f;
    float extrapolateMs = 0.0f;
    float gridToParticlesMs = 0.0f;
    float advectParticlesMs = 0.0f;
    float reseedMs = 0.0f;
    float rasterizeMs = 0.0f;
    float boundaryMs = 0.0f;
    float statsMs = 0.0f;
    float totalMs = 0.0f;

    void reset() { *this = SimStageTimings{}; }
};
