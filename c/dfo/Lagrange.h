//
// Created by thallock on 12/17/17.
//

#ifndef C_LAGRANGE_H
#define C_LAGRANGE_H

#include <functional>
#include <armadillo>

#include "../base/PolynomialBasis.h"
#include "../base/Shifter.h"

class LagrangeParams;

typedef std::function<void(
  const PolynomialBasis *basis,
  const arma::vec& coeff,
  arma::vec& out_x,
  double& out_f
)> lagrange_maximizer;

typedef std::function<bool(
  const arma::vec&point
)> point_filter;

void circular_tr_maximize_lagrange_poly(
        const PolynomialBasis *basis,
        const arma::vec& coeff,
        arma::vec& out_x,
        double& out_f
);

class LagrangeParams {
public:
    lagrange_maximizer maximize;
    PolynomialBasis *basis;

    Shifter *shifter;

    double tol;

    bool xsiIsModifiable;
    double xsi;
    double minXsi;

    point_filter pf;

    arma::ivec indexes;

    bool canEvaluateNewPoints;

    LagrangeParams(
            lagrange_maximizer maximize,
            PolynomialBasis *basis,
            Shifter *shifter,
            double tol,
            bool xsiIsModifiable,
            double xsi,
            double minXsi,
            point_filter pf,
            arma::ivec indexes,
            bool canEvaluateNewPoints
    )
      : maximize{maximize}
      , basis{basis}
      , shifter{shifter}
      , tol{tol}
      , xsi{xsi}
      , minXsi{1e-12}
      , pf{pf}
      , indexes()
      , canEvaluateNewPoints{true}
    {}
};

class LagrangeCertification {
public:
    bool isPoised;
    double outputXsi;

    arma::mat outputSet;
};

void computeLagrangePolynomials(
    const LagrangeParams& params,
    const arma::mat& in_unshifted,
    LagrangeCertification& cert,
    arma::mat& out_unshifted
);

#endif //C_LAGRANGE_H
