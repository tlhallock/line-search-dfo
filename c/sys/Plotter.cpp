//
// Created by thallock on 12/23/17.
//

#include "Plotter.h"

#include <sstream>
#include <iomanip>
#include <cstdio>

#define RAPIDJSON_NAMESPACE rj
#include <rapidjson/rapidjson.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/filewritestream.h>
#include <ctime>

#define DEFINE_WRITER rj::PrettyWriter<rj::FileWriteStream>& writer = internals->writer;


class PlotInternals {
public:
    FILE *fp;
    char writeBuffer[65536];
    rj::FileWriteStream os;
    rj::PrettyWriter<rj::FileWriteStream> writer;

    PlotInternals(const std::string& outputFile)
      : fp{fopen(outputFile.c_str(), "w")}
      , os{fp, writeBuffer, sizeof(writeBuffer)}
      , writer{os}
    {}

    ~PlotInternals() {
        fclose(fp);
    }
};

void Plot::writeContours() {
    DEFINE_WRITER

    for(auto it = contours.begin(); it != contours.end(); ++it) {
        Contour& c = *it;

        writer.StartObject();

        writer.Key("name");
        writer.String(c.name.c_str());

        writer.Key("type");
        writer.String("contour");

        writer.Key("params");
        writer.StartObject();
        writer.Key("color");
        writer.String(c.color.c_str());
        writer.Key("contours");
        writer.StartArray();
        for (int i=0;i<c.contourLevels.n_elem;i++) {
            writer.Double(c.contourLevels(i));
        }
        writer.EndArray();
        writer.EndObject();

        writer.Key("value");

        if (bounds.lower.n_elem != 2) {
            std::cout << "this is a problem" << std::endl;
        }

        const double xmax = bounds.upper[0];
        const double xmin = bounds.lower[0];
        const double ymax = bounds.upper[1];
        const double ymin = bounds.lower[1];
        const double xlength = xmax - xmin;
        const double ylength = ymax - ymin;

        writer.StartArray();
        arma::vec eval(2);
        int i, j; double x, y;
        int nx;
        int ny;
        for (i=0, x = xmin; x <= xmax; x += xlength / numberOfPoints, i++) {
            nx++;
            ny = 0;
            for (j=0, y = ymin; y <= ymax; y += ylength / numberOfPoints, j++) {
                eval(0) = x;
                eval(1) = y;
                double z = c.func->evaluate(eval);
                writer.StartObject();
                writer.Key("i");
                writer.Int(i);
                writer.Key("j");
                writer.Int(j);
                writer.Key("x");
                writer.Double(x);
                writer.Key("y");
                writer.Double(y);
                writer.Key("f");
                writer.Double(z);
                writer.EndObject();
                ny++;
            }
        }
        writer.EndArray();

        writer.Key("rows");
        writer.Int(nx);

        writer.Key("cols");
        writer.Int(ny);

        writer.EndObject();
    }
}

void Plot::addFunction(
        const std::string& name,
        Function* func,
        const std::string& color,
        const arma::vec& contourLevels
) {
    contours.push_back(Contour(
            name,
            func,
            color,
            contourLevels
    ));
}

void Plot::addPoints(
        const std::string& name,
        const arma::mat& rows,
        const std::string& options
) {
    DEFINE_WRITER
    writer.StartObject();

    writer.Key("name");
    writer.String(name.c_str());

    writer.Key("type");
    writer.String("points");

    writer.Key("params");
    writer.StartObject();
    writer.Key("options");
    writer.String(options.c_str());
    writer.EndObject();

    writer.Key("value");
    writer.StartArray();
    for (int i=0; i<rows.n_rows;i++) {
        bounds.should_include(arma::conv_to<arma::vec>::from(rows.row(i)));

        writer.StartArray();
        for (int j=0; j<rows.n_cols;j++) {
            writer.Double(rows(i, j));
        }
        writer.EndArray();
    }
    writer.EndArray();

    writer.EndObject();

}

void Plot::addPoint(
        const std::string& name,
        const arma::vec& point,
        const std::string& options
) {
    DEFINE_WRITER
    writer.StartObject();

    writer.Key("name");
    writer.String(name.c_str());

    writer.Key("type");
    writer.String("point");

    writer.Key("params");
    writer.StartObject();
    writer.Key("options");
    writer.String(options.c_str());
    writer.EndObject();

    writer.Key("value");
    writer.StartArray();
    for (int i=0; i<point.n_elem;i++) {
        writer.Double(point(i));
        bounds.should_include(point);
    }
    writer.EndArray();

    writer.EndObject();
}

void Plot::addArrow(
        const std::string& name,
        const arma::vec& from,
        const arma::vec& direction,
        const std::string& color
) {
    DEFINE_WRITER
    writer.StartObject();

    writer.Key("name");
    writer.String(name.c_str());

    writer.Key("type");
    writer.String("arrow");

    writer.Key("params");
    writer.StartObject();
    writer.Key("color");
    writer.String(color.c_str());
    writer.EndObject();

    writer.Key("value");
    writer.StartObject();

    writer.Key("from");
    writer.StartArray();
    for (int i=0; i<from.n_elem;i++) {
        writer.Double(from(i));
        bounds.should_include(from);
    }
    writer.EndArray();

    writer.Key("delta");
    writer.StartArray();
    for (int i=0; i<direction.n_elem;i++) {
        writer.Double(direction(i));
        bounds.should_include(from + direction);
    }
    writer.EndArray();
    writer.EndObject(); // value
    writer.EndObject();
}


Plot PlotFactory::createPlot(
        const std::string& name,
        const double padding,
        int n
) {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H-%M-%S");
    auto time_str = oss.str();


    std::stringstream fname;
    fname << plot_directory << '/';
    fname << std::setw(4) << std::setfill('0');
    fname << image_count;
    fname << "_" << name;
    fname << ".json";

    PlotInternals *writer = new PlotInternals(fname.str());
    writer->writer.StartObject();

    writer->writer.Key("time");
    writer->writer.String(time_str.c_str());

    writer->writer.Key("image-number");
    writer->writer.Int(image_count++);

    writer->writer.Key("elements");
    writer->writer.StartArray();

    return Plot(
      writer,
      padding,
      n
    );
}

Plot::~Plot() {
    writeContours();

    internals->writer.EndArray();

    // Now we know the bounds
    internals->writer.Key("upper-bounds");
    internals->writer.StartArray();
    for (int i=0; i<bounds.upper.n_elem; i++) {
        internals->writer.Double(bounds.upper(i) + padding * (bounds.upper(i) - bounds.lower(i)));
    }
    internals->writer.EndArray();

    internals->writer.Key("lower-bounds");
    internals->writer.StartArray();
    for (int i=0; i<bounds.lower.n_elem; i++) {
        internals->writer.Double(bounds.lower(i) - padding * (bounds.upper(i) - bounds.lower(i)));
    }
    internals->writer.EndArray();

    internals->writer.EndObject();
    delete internals;
}




static PlotFactory plot_factory{"images"};

PlotFactory& get_plot_factory_instance() {
    return plot_factory;
}