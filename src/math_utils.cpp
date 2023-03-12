#include <math_utils.hpp>
#include <muParser.h>

namespace petliukh::optimization_methods::math_utils {
using pline2 = Eigen::ParametrizedLine<double, 2>;
using Eigen::Vector2d, Eigen::Vector3d;
using std::vector;

pline2 param_line_from_coeffs(const Vector3d& coeffs) {
    double c = coeffs(2) / coeffs(1);
    Vector2d dir = Vector2d(coeffs(1), -coeffs(0)).normalized();
    return { Vector2d(0, c), dir };
}

Vector2d point_at_x(const pline2& line, double x) {
    double t = (x - line.origin()(0)) / line.direction()(0);
    return line.pointAt(t);
}

double closest_line_point_y_diff(const Vector2d& point, const pline2& line) {
    double proj_len = (point - line.origin()).dot(line.direction());
    Vector2d proj_point = line.origin() + proj_len * line.direction();
    return point.y() - proj_point.y();
}

bool is_point_above_line(const Vector2d& point, const pline2& line) {
    return closest_line_point_y_diff(point, line) > 0;
}

bool is_point_below_line(const Vector2d& point, const pline2& line) {
    return closest_line_point_y_diff(point, line) < 0;
}

bool is_point_on_line_approx(
        const Vector2d& point, const pline2& line, double eps) {
    double proj_len = (point - line.origin()).dot(line.direction());
    Vector2d proj_point = line.origin() + proj_len * line.direction();
    return (point - proj_point).norm() < eps;
}

bool is_point_onabove_line(const Vector2d& point, const pline2& line) {
    return is_point_on_line_approx(point, line)
            || is_point_above_line(point, line);
}

bool is_point_onbelow_line(const Vector2d& point, const pline2& line) {
    return is_point_on_line_approx(point, line)
            || is_point_below_line(point, line);
}

bool satisfies_eq_approx(
        const Vector3d& coeffs, const Vector2d& point, double eps) {
    return abs(coeffs(0) * point(0) + coeffs(1) * point(1) - coeffs(2)) < eps;
}

bool is_point_valid(const Vector2d& point) {
    return point(0) != INFINITY && point(1) != INFINITY && point(0) != -INFINITY
            && point(1) != -INFINITY;
}

Vector2d get_centroid(const vector<Vector2d>& points) {
    Vector2d centroid = Vector2d::Zero();
    for (const Vector2d& point : points)
        centroid += point;
    centroid /= points.size();
    return centroid;
}

void sort_clockwise(vector<Vector2d>& points) {
    Vector2d center = get_centroid(points);
    sort(points.begin(), points.end(),
         [&](const Vector2d& a, const Vector2d& b) {
             double angle_a = atan2(a.y() - center.y(), a.x() - center.x());
             double angle_b = atan2(b.y() - center.y(), b.x() - center.x());
             return angle_a < angle_b;
         });
}

}  // namespace petliukh::optimization_methods::math_utils
