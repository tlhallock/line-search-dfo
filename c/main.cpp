#include <iostream>

#include <armadillo>

#include "base/PolynomialBasis.h"
#include "dfo/Lagrange.h"
#include "fcn/Utils.h"

#include "opt/NLOpt.h"

#include "sys/Plotter.h"
#include "base/TrustRegion.h"
#include "sys/Config.h"

//void testThis();

int main() {
    PolynomialBasis basis {2, 2};
    std::cout << basis;

//    arma::mat set = {
//      {0, 1},
//      {0, 1},
//      {0, 1},
//      {0, 1},
//      {0, 1},
//      {0, 1},
//    };

    PlotFactory& factory = get_plot_factory_instance();

    arma::mat set = {
            {0, 1},
            {2, 1},
            {1, 4},
            {0, 0},
            {11.5, .5},
            {3, 2},
    };

//    arma::mat set = {
//            {0, 1},
//            {.2, .1},
//            {.1, -.4},
//            {0, 0},
//            {.115, .7},
//            {-.3, 2},
//    };

    {
        Plot plot = factory.createPlot("original_set");
        plot.addPoints("original set", set);
    }

    arma::vec center = {0, 0};
    TrustRegionObj tr{
        1,
        center
    };

    LagrangeParams params{
        [](
            const PolynomialBasis *basis,
            const arma::vec& coeff,
            arma::vec& out_x,
            double& out_f
        ) {
            circular_tr_maximize_lagrange_poly(
                    basis,
                    coeff,
                    out_x,
                    out_f
            );
        },
        &basis,
        &tr,
        1e-10,
        false,
        1e-8,
        1e-12,
        [tr](const arma::vec& v) {
            return tr.contains(v);
        },
        {},
        true
    };

    LagrangeCertification cert;

    arma::mat newset(set.n_rows, set.n_cols);
    computeLagrangePolynomials(
            params,
            set,
            cert,
            newset
    );


    std::cout << newset << std::endl;
    {
        Plot plot = factory.createPlot("poised_set");
        plot.addPoints("poised set", newset);
    }
//
//    Function *obj = basis.createPolynomial(phi);
//
//    Program p;
//
//    arma::vec x0;
//    x0 << 0 << 0 << arma::endr;
//    p.setX0(x0);
//    p.setDimension(2);
//    p.setTolerance(1e-8);
//
//    TrustRegion tr{2};
//
//    p.setObjective(obj);
//    p.addInequalityConstraint(&tr);
//
//    arma::vec test = {0, 0};
//
//    OptResult res = minimize(p);
//    std::cout << res;

    const Config& config = get_config_instance();
    std::cout << config.get_property("foo") << std::endl;
    return 0;
}