#pragma once

#include "../mac_water3d.h"

struct MACWater3DCudaBackend;

#if SMOKE_ENABLE_CUDA
MACWater3DCudaBackend* water3dCreateCudaBackend();
void water3dDestroyCudaBackend(MACWater3DCudaBackend* backend);
void water3dCudaReset(MACWater3DCudaBackend* backend, MACWater3D& sim);
void water3dCudaSetParams(MACWater3DCudaBackend* backend, MACWater3D& sim);
void water3dCudaSetVoxelSolids(MACWater3DCudaBackend* backend, MACWater3D& sim);
void water3dCudaAddWaterSourceSphere(MACWater3DCudaBackend* backend, MACWater3D& sim,
                                     const MACWater3D::Vec3& center, float radius,
                                     const MACWater3D::Vec3& velocity);
void water3dCudaStep(MACWater3DCudaBackend* backend, MACWater3D& sim);
void water3dCudaDownloadHostAll(MACWater3DCudaBackend* backend, MACWater3D& sim);
void water3dCudaDownloadHostVolume(MACWater3DCudaBackend* backend, MACWater3D& sim);
void water3dCudaDownloadHostParticles(MACWater3DCudaBackend* backend, MACWater3D& sim);
void water3dCudaDownloadHostDebugField(MACWater3DCudaBackend* backend, MACWater3D& sim,
                                      MACWater3D::DebugField field);
#else
inline MACWater3DCudaBackend* water3dCreateCudaBackend() { return nullptr; }
inline void water3dDestroyCudaBackend(MACWater3DCudaBackend*) {}
inline void water3dCudaReset(MACWater3DCudaBackend*, MACWater3D&) {}
inline void water3dCudaSetParams(MACWater3DCudaBackend*, MACWater3D&) {}
inline void water3dCudaSetVoxelSolids(MACWater3DCudaBackend*, MACWater3D&) {}
inline void water3dCudaAddWaterSourceSphere(MACWater3DCudaBackend*, MACWater3D&,
                                            const MACWater3D::Vec3&, float,
                                            const MACWater3D::Vec3&) {}
inline void water3dCudaStep(MACWater3DCudaBackend*, MACWater3D&) {}
inline void water3dCudaDownloadHostAll(MACWater3DCudaBackend*, MACWater3D&) {}
inline void water3dCudaDownloadHostVolume(MACWater3DCudaBackend*, MACWater3D&) {}
inline void water3dCudaDownloadHostParticles(MACWater3DCudaBackend*, MACWater3D&) {}
inline void water3dCudaDownloadHostDebugField(MACWater3DCudaBackend*, MACWater3D&,
                                              MACWater3D::DebugField) {}
#endif
