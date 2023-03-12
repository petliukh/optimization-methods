#include "graphical_method.hpp"

#include "math_utils.hpp"

using Eigen::MatrixXd, Eigen::Vector2d, Eigen::Vector3d, Eigen::VectorXd;
using std::min_element, std::max_element;
using std::string, std::vector, std::to_string;
using hplane2 = Eigen::Hyperplane<double, 2>;
using pline2 = Eigen::ParametrizedLine<double, 2>;
using namespace petliukh::optimization_methods::math_utils;

namespace petliukh::optimization_methods {

graphical_method::graphical_method(
        const MatrixXd& A, const vector<string>& signs, string func)
    : m_inequations(A), m_signs(signs), m_func(func) {
    init_param_lines_and_hplanes();
}

double*
graphical_method::create_var_implicit(const char* var_name, void* val_ptr) {
    static double val_buf[100];
    static int val_cnt = -1;

    val_cnt++;
    val_buf[val_cnt] = 0;

    if (val_cnt >= 99)
        throw mu::ParserError("Variable buffer overflow.");
    else
        return &val_buf[val_cnt];
}

double graphical_method::eval_func_at(const VectorXd& point) {
    m_parser.SetExpr(m_func);
    m_parser.SetVarFactory(create_var_implicit);
    m_parser.Eval();
    auto vars = m_parser.GetVar();
    if (vars.size() != point.size())
        throw mu::ParserError("Invalid number of variables.");
    int i = 0;

    for (auto& [var_name, val_ptr] : vars) {
        *val_ptr = point[i];
        i++;
    }

    return m_parser.Eval();
}

Vector2d
graphical_method::get_min_max_values(const vector<Vector2d>& intersections) {
    vector<double> values;
    values.reserve(intersections.size());

    for (const Vector2d& ipt : intersections) {
        double val = eval_func_at(ipt);
        values.push_back(val);
    }

    double min_val = *min_element(values.begin(), values.end());
    double max_val = *max_element(values.begin(), values.end());

    return Vector2d(min_val, max_val);
}

bool graphical_method::meets_conditions(const Vector2d& point) const {
    if (!is_point_valid(point))
        return false;

    for (int i = 0; i < m_inequations.rows(); i++) {
        const string& sign = m_signs[i];
        double y_sign = m_inequations(i, 1) > 0 ? 1 : -1;

        if (satisfies_eq_approx(m_inequations.row(i), point))
            continue;

        double diff
                = y_sign * closest_line_point_y_diff(point, m_param_lines[i]);

        if ((sign == ">" || sign == ">=") && !(diff > 0))
            return false;
        else if ((sign == "<" || sign == "<=") && !(diff < 0))
            return false;
    }

    return true;
}

void graphical_method::init_param_lines_and_hplanes() {
    m_param_lines.reserve(m_inequations.rows());
    m_hplanes.reserve(m_inequations.rows());

    for (int i = 0; i < m_inequations.rows(); i++) {
        m_param_lines.push_back(param_line_from_coeffs(m_inequations.row(i)));
        m_hplanes.push_back(hplane2(m_param_lines.back()));
    }
}

vector<Vector2d>
graphical_method::get_lines_intersection_points(bool check_conditions) const {
    int in_count = m_inequations.rows() * (m_inequations.rows() - 1) / 2;
    vector<Vector2d> intersections;
    intersections.reserve(in_count);
    int k = 0;

    for (int i = 0; i < m_inequations.rows(); i++) {
        for (int j = i + 1; j < m_inequations.rows(); ++j) {
            Vector2d ipt = m_param_lines[i].intersectionPoint(m_hplanes[j]);

            if (!is_point_valid(ipt)
                || (check_conditions && !meets_conditions(ipt)))
                continue;

            intersections.push_back(ipt);
            k++;
        }
    }

    return intersections;
}

const vector<pline2>& graphical_method::get_param_lines() const {
    return m_param_lines;
}

vector<string> graphical_method::get_labels() const {
    vector<string> labels;
    labels.reserve(m_inequations.rows());

    for (int i = 0; i < m_inequations.rows(); i++) {
        Vector3d coeffs = m_inequations.row(i);
        string label = to_string(coeffs(0)) + "x + " + to_string(coeffs(1))
                + "y " + m_signs[i] + " " + to_string(coeffs(2));
        labels.push_back(label);
    }

    return labels;
}

}  // namespace petliukh::optimization_methods
