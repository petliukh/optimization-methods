#pragma once

#include "graphical_method.hpp"
#include "math_utils.hpp"

#include <Eigen/Core>
#include <QMainWindow>
#include <QPainter>
#include <QPointF>
#include <qwt_plot_curve.h>
#include <qwt_plot_shapeitem.h>
#include <qwt_scale_map.h>
#include <qwt_symbol.h>
#include <qwt_text.h>
#include <vector>

QT_BEGIN_NAMESPACE
namespace Ui {
class main_window;
}
QT_END_NAMESPACE

class main_window : public QMainWindow {
    Q_OBJECT

public:
    main_window(QWidget* parent = nullptr);
    ~main_window();

private slots:
    Eigen::MatrixXd get_matrix();

    std::vector<std::string> get_signs();

    std::vector<Eigen::Vector2d>
    get_endpoints(const std::vector<Eigen::ParametrizedLine<double, 2>>&
                          parametrized_lines);

    std::string get_func();

    void shade_polygon_area_n_scatter(std::vector<Eigen::Vector2d>& points);

    void draw_lines(
            const std::vector<Eigen::ParametrizedLine<double, 2>>&
                    parametrized_lines,
            std::vector<std::string>& labels);

    void on_draw_btn_clicked();

    void on_col_spinbox_valueChanged(int arg1);

    void on_row_spinbox_valueChanged(int arg1);

    void on_findminmax_btn_clicked();

private:
    Ui::main_window* ui;
    Eigen::Vector2d x_range;
};
