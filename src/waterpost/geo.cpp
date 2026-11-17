#include "geo.hpp"

#include "error.hpp"

#include <cmath>

namespace ai3d::waterpost
{
namespace
{

constexpr double A = 6378137.0;
constexpr double F = 1 / 298.257223563;

} // namespace

Geodetic ecef_to_geodetic(double x, double y, double z)
{
    const double e2 = F * (2 - F), b = A * (1 - F), ep2 = (A * A - b * b) / (b * b);
    const double p = std::hypot(x, y);
    const double theta = std::atan2(z * A, p * b);
    Geodetic g;
    g.latitude = std::atan2(z + ep2 * b * std::pow(std::sin(theta), 3), p - e2 * A * std::pow(std::cos(theta), 3));
    g.longitude = std::atan2(y, x);
    const double n = A / std::sqrt(1 - e2 * std::sin(g.latitude) * std::sin(g.latitude));
    g.height = p / std::cos(g.latitude) - n;
    return g;
}

std::array<double, 2> utm_forward(double latitude, double longitude, int zone, bool south)
{
    const double k0 = 0.9996;
    const double lon0 = (zone * 6 - 183) * M_PI / 180;
    const double n = F / (2 - F);
    const double a_hat = A / (1 + n) * (1 + n * n / 4 + n * n * n * n / 64);
    const double alpha[3] = {n / 2 - 2 * n * n / 3 + 5 * n * n * n / 16, 13 * n * n / 48 - 3 * n * n * n / 5,
                             61 * n * n * n / 240};
    const double c = 2 * std::sqrt(n) / (1 + n);
    const double t = std::sinh(std::atanh(std::sin(latitude)) - c * std::atanh(c * std::sin(latitude)));
    const double xi = std::atan2(t, std::cos(longitude - lon0));
    const double eta = std::atanh(std::sin(longitude - lon0) / std::sqrt(1 + t * t));
    double east = eta, north = xi;
    for (int j = 0; j < 3; ++j)
    {
        east += alpha[j] * std::cos(2 * (j + 1) * xi) * std::sinh(2 * (j + 1) * eta);
        north += alpha[j] * std::sin(2 * (j + 1) * xi) * std::cosh(2 * (j + 1) * eta);
    }
    return {500000 + k0 * a_hat * east, (south ? 10000000.0 : 0.0) + k0 * a_hat * north};
}

int utm_zone_of(double longitude)
{
    const double degrees = longitude * 180 / M_PI;
    return static_cast<int>(std::floor((degrees + 180) / 6)) % 60 + 1;
}

void parse_pose_crs(const std::string& text, int& zone, bool& south)
{
    south = false;
    if (text.empty() || text == "auto")
    {
        zone = 0;
        return;
    }
    if (text == "none")
    {
        zone = -1;
        return;
    }
    if (text.size() == 10 && (text.rfind("EPSG:326", 0) == 0 || text.rfind("EPSG:327", 0) == 0))
    {
        zone = std::stoi(text.substr(8));
        south = text[7] == '7';
        if (zone >= 1 && zone <= 60)
        {
            return;
        }
    }
    fail_value("--pose-crs expects auto, none or EPSG:326NN / EPSG:327NN (UTM): " + text);
}

} // namespace ai3d::waterpost
