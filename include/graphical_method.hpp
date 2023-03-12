#include <Eigen/Dense>
#include <muParser.h>
#include <string>
#include <vector>

namespace petliukh::optimization_methods {

class graphical_method {
public:
    static double* create_var_implicit(const char* var_name, void* val_ptr);

    graphical_method(
            const Eigen::MatrixXd& A, const std::vector<std::string>& signs,
            std::string func = "");

    bool meets_conditions(const Eigen::Vector2d& point) const;

    std::vector<Eigen::Vector2d>
    get_lines_intersection_points(bool check_conditions = false) const;

    const std::vector<Eigen::ParametrizedLine<double, 2>>&
    get_param_lines() const;

    std::vector<std::string> get_labels() const;

    double eval_func_at(const Eigen::VectorXd& point);

    Eigen::Vector2d
    get_min_max_values(const std::vector<Eigen::Vector2d>& intersections);

private:
    void init_param_lines_and_hplanes();

    Eigen::MatrixXd m_inequations;  // matrix of line equation coefficient rows
    std::vector<std::string> m_signs;
    std::string m_func;
    std::vector<Eigen::ParametrizedLine<double, 2>> m_param_lines;
    std::vector<Eigen::Hyperplane<double, 2>> m_hplanes;
    mu::Parser m_parser;
};

}  // namespace petliukh::optimization_methods
