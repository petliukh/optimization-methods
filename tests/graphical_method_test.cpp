#include "graphical_method.hpp"
#include "math_utils.hpp"

#include <gtest/gtest.h>

using gmet = petliukh::optimization_methods::graphical_method;
using pline2 = Eigen::ParametrizedLine<double, 2>;
using std::vector, std::string;
using namespace Eigen;
using namespace petliukh::optimization_methods::math_utils;

bool is_approx(double a, double b, double eps = 1e-3) {
    return abs(a - b) < eps;
}

TEST(graphical_method, param_line_from_coeffs) {
    Vector3d coeffs(1, 2, 3);  // x + 2y = 3 -> y = 1.5 - 0.5x
    pline2 line = param_line_from_coeffs(coeffs);

    EXPECT_EQ(line.origin().x(), 0);
    EXPECT_TRUE(is_approx(line.origin().y(), 1.5));

    // (b, -a) = (2, -1) - direction vector
    // || (2, -1) || = sqrt(5) = 2.236
    // normalized: (2, -1) / 2.236 = (0.894, -0.447)
    EXPECT_TRUE(is_approx(line.direction().x(), 0.894));
    EXPECT_TRUE(is_approx(line.direction().y(), -0.447));

    Vector3d coeffs2(-5, 2, 8);  // -5x + 2y = 8 -> y = 2.5x + 4
    pline2 line2 = param_line_from_coeffs(coeffs2);

    EXPECT_EQ(line2.origin().x(), 0);
    EXPECT_TRUE(is_approx(line2.origin().y(), 4));

    // (b, -a) = (2, 5) - direction vector
    // || (2, 5) || = sqrt(29) = 5.385
    // normalized: (2, 5) / 5.385 = (0.371, 0.928)
    EXPECT_TRUE(is_approx(line2.direction().x(), 0.371));
    EXPECT_TRUE(is_approx(line2.direction().y(), 0.928));
}

TEST(graphical_method, point_at_x) {
    Vector3d coeffs(1, 2, 3);  // x + 2y = 3 -> y = 1.5 - 0.5x
    pline2 line = param_line_from_coeffs(coeffs);

    Vector2d point = point_at_x(line, 1);
    Vector2d point2 = point_at_x(line, 2);
    Vector2d point3 = point_at_x(line, 3);
    Vector2d point4 = point_at_x(line, 4);

    EXPECT_EQ(point.x(), 1);
    EXPECT_EQ(point.y(), 1);

    EXPECT_EQ(point2.x(), 2);
    EXPECT_EQ(point2.y(), 0.5);

    EXPECT_EQ(point3.x(), 3);
    EXPECT_EQ(point3.y(), 0);

    EXPECT_EQ(point4.x(), 4);
    EXPECT_EQ(point4.y(), -0.5);
}

TEST(graphical_method, on_above_below) {
    Vector3d coeffs(1, 2, 3);  // x + 2y = 3 -> y = 1.5 - 0.5x
    pline2 line = param_line_from_coeffs(coeffs);

    EXPECT_TRUE(is_point_above_line(Vector2d(1, 2), line));
    EXPECT_TRUE(is_point_above_line(Vector2d(-1, 3), line));
    EXPECT_FALSE(is_point_above_line(Vector2d(1, 1), line));
    EXPECT_FALSE(is_point_above_line(Vector2d(-1, 1), line));

    EXPECT_TRUE(is_point_below_line(Vector2d(-1, 1), line));
    EXPECT_TRUE(is_point_below_line(Vector2d(1, 0.5), line));
    EXPECT_FALSE(is_point_below_line(Vector2d(1, 2), line));
    EXPECT_FALSE(is_point_below_line(Vector2d(1, 1), line));

    EXPECT_TRUE(is_point_on_line_approx(Vector2d(1, 1), line));
    EXPECT_TRUE(is_point_on_line_approx(Vector2d(-1, 2), line));
    EXPECT_FALSE(is_point_on_line_approx(Vector2d(1, 2), line));
    EXPECT_FALSE(is_point_on_line_approx(Vector2d(-1, 1), line));

    EXPECT_TRUE(is_point_onabove_line(Vector2d(1, 1), line));
    EXPECT_TRUE(is_point_onabove_line(Vector2d(-1, 3), line));
    EXPECT_FALSE(is_point_onabove_line(Vector2d(1, 0), line));
    EXPECT_FALSE(is_point_onabove_line(Vector2d(-1, 1), line));

    EXPECT_TRUE(is_point_onbelow_line(Vector2d(-1, 2), line));
    EXPECT_TRUE(is_point_onbelow_line(Vector2d(-1, 1), line));
    EXPECT_FALSE(is_point_onbelow_line(Vector2d(1, 2), line));
    EXPECT_FALSE(is_point_onbelow_line(Vector2d(-1, 4), line));
}

TEST(graphical_method, meets_conditions) {
    MatrixXd A(4, 3);
    vector<string> signs = { "<=", "<=", ">=", ">=" };

    // clang-format off
    A << 1, -1, 1, // x - y <= 1
         1, 1, 2, // x + y <= 2
         1, -2, 0, // x - 2y >= 0
         2, 2, 1; // 2x + 2y >= 1
    // clang-format on
    gmet gm(A, signs);

    EXPECT_TRUE(gm.meets_conditions(Vector2d(0.6, 0.1)));
    EXPECT_TRUE(gm.meets_conditions(Vector2d(0.8, 0.3)));
    EXPECT_TRUE(gm.meets_conditions(Vector2d(1.2, 0.4)));
    EXPECT_TRUE(gm.meets_conditions(Vector2d(1.3, 0.6)));

    EXPECT_TRUE(gm.meets_conditions(Vector2d(0.333, 0.166)));
    EXPECT_TRUE(gm.meets_conditions(Vector2d(0.75, -0.25)));
    EXPECT_TRUE(gm.meets_conditions(Vector2d(1.33, 0.66)));
    EXPECT_TRUE(gm.meets_conditions(Vector2d(1.5, 0.5)));

    EXPECT_FALSE(gm.meets_conditions(Vector2d(0, 0)));
    EXPECT_FALSE(gm.meets_conditions(Vector2d(1, 1)));
    EXPECT_FALSE(gm.meets_conditions(Vector2d(-1, 1)));
    EXPECT_FALSE(gm.meets_conditions(Vector2d(2, 1)));
}

TEST(graphical_method, intersections) {
    MatrixXd A(4, 3);
    vector<string> signs = { "<=", "<=", ">=", ">=" };

    // clang-format off
    A << 1, -1, 1, // x - y <= 1
         1, 1, 2, // x + y <= 2
         1, -2, 0, // x - 2y >= 0
         2, 2, 1; // 2x + 2y >= 1
    // clang-format on
    gmet gm(A, signs);

    vector<Vector2d> intersections = gm.get_lines_intersection_points();
    EXPECT_EQ(intersections.size(), 5);

    intersections = gm.get_lines_intersection_points(true);
    EXPECT_EQ(intersections.size(), 4);
}

TEST(graphical_method, eval_expression) {
    MatrixXd A(4, 3);
    vector<string> signs = { "<=", "<=", ">=", ">=" };
    string func = "x - 2*y";

    // clang-format off
    A << 1, -1, 1, // x - y <= 1
         1, 1, 2, // x + y <= 2
         1, -2, 0, // x - 2y >= 0
         2, 2, 1; // 2x + 2y >= 1
    // clang-format on
    gmet gm(A, signs, func);
    vector<Vector2d> intersections = gm.get_lines_intersection_points(true);

    for (Vector2d& point : intersections) {
        double expected = point.x() - 2 * point.y();
        double actual = gm.eval_func_at(point);
        EXPECT_NEAR(expected, actual, 1e-3);
    }
}

TEST(graphical_method, minmax) {
    MatrixXd A(4, 3);
    vector<string> signs = { "<=", "<=", ">=", ">=" };
    string func = "x - 2*y";

    // clang-format off
    A << 1, -1, 1, // x - y <= 1
         1, 1, 2, // x + y <= 2
         1, -2, 0, // x - 2y >= 0
         2, 2, 1; // 2x + 2y >= 1
    // clang-format on
    gmet gm(A, signs, func);
    vector<Vector2d> intersections = gm.get_lines_intersection_points(true);
    Vector2d minmax = gm.get_min_max_values(intersections);

    double min = minmax(0);
    double max = minmax(1);

    EXPECT_NEAR(0, min, 1e-2);
    EXPECT_NEAR(1.25, max, 1e-2);
}
