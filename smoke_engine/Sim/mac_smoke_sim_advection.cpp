#include "mac_smoke_sim.h"
#include <cmath>

void MAC2D::advectScalar(std::vector<float>& phi,
                         std::vector<float>& phi0,
                         float dissipation)
{
    if (useMacCormack) advectScalarMacCormack(phi, phi0, dissipation);
    else               advectScalarSemiLagrangian(phi, phi0, dissipation);
}

float MAC2D::smokeL2Diff(const std::vector<float>& a, const std::vector<float>& b) const {
    double s = 0.0;
    size_t N = a.size();
    for (size_t k = 0; k < N; ++k) {
        double d = (double)a[k] - (double)b[k];
        s += d * d;
    }
    return (float)std::sqrt(s / (double)N);
}

float MAC2D::compareAdvectors(float dissipation)
{
    std::vector<float> smoke_before = smoke;
    std::vector<float> phi0_copy    = smoke0;

    std::vector<float> mac = smoke_before;
    std::vector<float> mac_phi0 = phi0_copy;
    advectScalarMacCormack(mac, mac_phi0, dissipation);

    std::vector<float> sl = smoke_before;
    std::vector<float> sl_phi0 = phi0_copy;
    advectScalarSemiLagrangian(sl, sl_phi0, dissipation);

    float l2 = smokeL2Diff(mac, sl);
    return l2;
}
