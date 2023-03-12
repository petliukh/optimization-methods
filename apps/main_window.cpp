#include "main_window.h"

#include "ui_main_window.h"

#include <sstream>

using Eigen::MatrixXd;
using Eigen::Vector2d;
using pline2 = Eigen::ParametrizedLine<double, 2>;
using gmet = petliukh::optimization_methods::graphical_method;
using std::vector, std::string, std::stringstream, std::copy_if;
using namespace petliukh::optimization_methods::math_utils;

class QwtPlotCurveWithTitle : public QwtPlotCurve {
public:
    explicit QwtPlotCurveWithTitle(const QString& title = QString())
        : QwtPlotCurve(title) {
    }
    explicit QwtPlotCurveWithTitle(const QwtText& title) : QwtPlotCurve(title) {
    }

protected:
    virtual void drawCurve(
            QPainter* p, int style, const QwtScaleMap& xMap,
            const QwtScaleMap& yMap, const QRectF& canvasRect, int from,
            int to) const {
        QwtPlotCurve::drawCurve(p, style, xMap, yMap, canvasRect, from, to);
        QPointF point = sample(from);
        p->drawText(
                xMap.transform(point.x()), yMap.transform(point.y()),
                title().text());
    }
};

main_window::main_window(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::main_window) {
    ui->setupUi(this);
    x_range = Vector2d(-5, 5);
}

main_window::~main_window() {
    delete ui;
}

MatrixXd main_window::get_matrix() {
    int cols = ui->inequations_table->columnCount() - 1;
    int rows = ui->inequations_table->rowCount();
    MatrixXd mtx(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; ++j) {
            mtx(i, j) = ui->inequations_table->item(i, j)->text().toDouble();
        }
    }

    return mtx;
}

vector<string> main_window::get_signs() {
    int cols = ui->inequations_table->columnCount();
    int rows = ui->inequations_table->rowCount();
    vector<string> signs;
    signs.reserve(rows);

    for (int i = 0; i < rows; i++) {
        signs.push_back(
                ui->inequations_table->item(i, cols - 1)->text().toStdString());
    }

    return signs;
}

string main_window::get_func() {
    return ui->function_text_edit->toPlainText().toStdString();
}

vector<Vector2d> main_window::get_endpoints(const vector<pline2>& param_lines) {
    vector<Vector2d> endpoints;
    endpoints.reserve(param_lines.size());

    for (auto& line : param_lines) {
        endpoints.push_back(point_at_x(line, x_range(0)));
        endpoints.push_back(point_at_x(line, x_range(1)));
    }
    return endpoints;
}

void main_window::shade_polygon_area_n_scatter(vector<Vector2d>& points) {
    QPolygonF polygon;

    for (auto& point : points) {
        polygon << QPointF(point.x(), point.y());
    }

    QwtPlotShapeItem* shape = new QwtPlotShapeItem();
    shape->setPolygon(polygon);
    shape->setPen(QPen(Qt::NoPen));
    shape->setBrush(QBrush(QColor(255, 255, 255, 100)));
    shape->attach(ui->qwt_plot);

    QwtSymbol* symbol = new QwtSymbol(QwtSymbol::Ellipse);
    QVector<QPointF> sqpoints;
    QwtPlotCurve* curve = new QwtPlotCurve();

    symbol->setSize(10);
    symbol->setPen(QPen(Qt::red));
    symbol->setBrush(QBrush(Qt::red));
    curve->setSymbol(symbol);

    for (auto& point : points) {
        sqpoints << QPointF(point.x(), point.y());
    }

    curve->setSamples(sqpoints);
    curve->attach(ui->qwt_plot);
}

void main_window::draw_lines(
        const vector<pline2>& parametrized_lines, vector<string>& labels) {
    for (int i = 0; i < parametrized_lines.size(); i++) {
        Vector2d p1 = point_at_x(parametrized_lines[i], x_range(0));
        Vector2d p2 = point_at_x(parametrized_lines[i], x_range(1));

        QwtPlotCurveWithTitle* curve
                = new QwtPlotCurveWithTitle(QString::fromStdString(labels[i]));
        curve->setPen(QPen(Qt::white));
        curve->setSamples(QVector<QPointF>{ QPointF(p1(0), p1(1)),
                                            QPointF(p2(0), p2(1)) });
        curve->attach(ui->qwt_plot);
    }
    ui->qwt_plot->replot();
}

void main_window::on_col_spinbox_valueChanged(int arg1) {
    ui->inequations_table->setColumnCount(arg1);
}

void main_window::on_row_spinbox_valueChanged(int arg1) {
    ui->inequations_table->setRowCount(arg1);
}

void main_window::on_draw_btn_clicked() {
    MatrixXd mtx = get_matrix();
    vector<string> signs = get_signs();
    gmet gm(mtx, signs);

    vector<Vector2d> intersections = gm.get_lines_intersection_points(true);
    vector<pline2> param_lines = gm.get_param_lines();
    vector<string> labels = gm.get_labels();

    vector<Vector2d> endpoints = get_endpoints(param_lines);
    vector<Vector2d> endpoints_mc;

    copy_if(endpoints.begin(), endpoints.end(),
            std::back_inserter(endpoints_mc),
            [=](Vector2d& point) { return gm.meets_conditions(point); });

    if (endpoints_mc.size() > 0) {
        intersections.insert(
                intersections.end(), endpoints_mc.begin(), endpoints_mc.end());
    }

    sort_clockwise(intersections);

    ui->qwt_plot->detachItems();
    ui->qwt_plot->replot();
    draw_lines(param_lines, labels);
    shade_polygon_area_n_scatter(intersections);
}

void main_window::on_findminmax_btn_clicked() {
    try {
        string func = get_func();
        MatrixXd mtx = get_matrix();
        vector<string> signs = get_signs();
        gmet gm(mtx, signs, func);
        vector<Vector2d> intersections = gm.get_lines_intersection_points(true);

        if (intersections.size() == 0) {
            ui->result_text_edit->clear();
            ui->result_text_edit->setText("No intersections found");
            return;
        }

        Vector2d minmax = gm.get_min_max_values(intersections);

        stringstream ss;
        ss << "Intersections:\n";
        for (Vector2d& point : intersections) {
            ss << "x: " << point(0) << "\ty: " << point(1);
            ss << "\t; Value at x: " << gm.eval_func_at(point) << "\n";
        }
        ss << "\nMin: " << minmax(0) << "\nMax: " << minmax(1);

        ui->result_text_edit->clear();
        ui->result_text_edit->setText(QString::fromStdString(ss.str()));
    } catch (mu::Parser::exception_type& e) {
        ui->result_text_edit->clear();
        ui->result_text_edit->setText(QString::fromStdString(e.GetMsg()));
    }
}
