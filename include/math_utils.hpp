#include <Eigen/Dense>
#include <vector>

namespace petliukh::optimization_methods::math_utils {

Eigen::ParametrizedLine<double, 2>
param_line_from_coeffs(const Eigen::Vector3d& coeffs);

Eigen::Vector2d
point_at_x(const Eigen::ParametrizedLine<double, 2>& line, double x);

double closest_line_point_y_diff(
        const Eigen::Vector2d& point,
        const Eigen::ParametrizedLine<double, 2>& line);

bool is_point_above_line(
        const Eigen::Vector2d& point,
        const Eigen::ParametrizedLine<double, 2>& line);

bool is_point_below_line(
        const Eigen::Vector2d& point,
        const Eigen::ParametrizedLine<double, 2>& line);

bool is_point_on_line_approx(
        const Eigen::Vector2d& point,
        const Eigen::ParametrizedLine<double, 2>& line, double eps = 1e-2);

bool is_point_onabove_line(
        const Eigen::Vector2d& point,
        const Eigen::ParametrizedLine<double, 2>& line);

bool is_point_onbelow_line(
        const Eigen::Vector2d& point,
        const Eigen::ParametrizedLine<double, 2>& line);

bool satisfies_eq_approx(
        const Eigen::Vector3d& coeffs, const Eigen::Vector2d& point,
        double eps = 1e-2);

bool is_point_valid(const Eigen::Vector2d& point);

Eigen::Vector2d get_centroid(const std::vector<Eigen::Vector2d>& points);

void sort_clockwise(std::vector<Eigen::Vector2d>& points);

}  // namespace petliukh::optimization_methods::math_utils
