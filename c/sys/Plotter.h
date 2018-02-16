//
// Created by thallock on 12/23/17.
//

#ifndef C_PLOTTER_H
#define C_PLOTTER_H

#include <string>
#include <fstream>
#include <list>

#include "../opt/Optimization.h"
#include "../base/Bounds.h"

class Contour {
public:
    const std::string& name;
    Function *func;
    const std::string& color;
    const arma::vec& contourLevels;

    Contour(
            const std::string& name,
            Function *func,
            const std::string& color,
            const arma::vec& contourLevels
    )
            : name{name}
            , func{func}
            , color{color}
            , contourLevels{contourLevels}
    {}
};

class PlotInternals;

class Plot {
    PlotInternals *internals;
    Bounds bounds;
    // not used yet
    double padding;
    std::list<Contour> contours;
    const double numberOfPoints;

    void writeContours();
public:
    Plot(PlotInternals *internals,
         const double padding,
         int n
    )
      : internals{internals}
      , bounds{n}
      , padding{padding}
      , contours{}
      , numberOfPoints{100}
    {}

    ~Plot();

    void addFunction(
            const std::string& name,
            Function* func,
            const std::string& color = "black",
            const arma::vec& contourLevels = {}
    );

    void addPoints(
            const std::string& name,
            const arma::mat& rows,
            const std::string& options = "bx"
    );

    void addPoint(
            const std::string& name,
            const arma::vec& point,
            const std::string& options = "bx"
    );

    void addArrow(
            const std::string& name,
            const arma::vec& from,
            const arma::vec& direction,
            const std::string& color = "green"
    );
};

class PlotFactory {
    int image_count;
    std::string plot_directory;

public:
    PlotFactory(const std::string& dir)
            : plot_directory{dir}
            , image_count{0} {}

    Plot createPlot(
            const std::string& name,
            const double padding = 0.5,
            int n = 2
    );
};


PlotFactory& get_plot_factory_instance();


#endif //C_PLOTTER_H
