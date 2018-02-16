//
// Created by thallock on 12/17/17.
//

#include "Lagrange.h"
#include "../base/PolynomialBasis.h"
#include "../opt/Optimization.h"

#include <armadillo>

#define HAVE_NAMESPACES 1
//#define HAVE_STD 1

#include <iostream>

#include <functional>


#include "NLF.h"
#include "BoundConstraint.h"
#include "NonLinearInequality.h"
#include "CompoundConstraint.h"
#include "OptNIPS.h"

#include "NLP.h"

#include <armadillo>

#include "../opt/Optimization.h"
#include "../opt/NewMat2Armadillo.h"


void circular_tr_maximize_lagrange_poly(
  const PolynomialBasis *basis,
  const arma::vec& coeff,
  arma::vec& out_x,
  double& out_f
) {
    int n = basis->get_n();

    Program p;
    p.setDimension(n);
    p.setTolerance(1e-8);

    TrustRegion tr;
    p.addInequalityConstraint(&tr);

    Function *obj = basis->createPolynomial(coeff);
    p.setObjective(obj);

    arma::vec x0(n, arma::fill::zeros);
    p.setX0(x0);

    OptResult res_min = minimize(p);
    out_f = res_min.fVal;
    out_x = res_min.resultX;
    std::cout << out_x << std::endl;

    for (int i=0; i<10; i++) {
        x0.randu();
        x0 *= 2;
        x0 -= 1;
        while (arma::norm(x0) >= 1) {
            x0 /= 2;
        }
        p.setX0(x0);

        OptResult res_rand = minimize(p);
        if (abs(res_rand.fVal) > abs(out_f)) {
            out_f = res_rand.fVal;
            out_x = res_rand.resultX;
        }
    }

    Function *neg = new Negation(obj);
    p.setObjective(neg);

    x0.fill(arma::fill::zeros);
    p.setX0(x0);

    OptResult res_max = minimize(p);
    if (abs(res_max.fVal) > abs(out_f)) {
        out_f = res_max.fVal;
        out_x = res_max.resultX;
    }
    for (int i=0; i<10;i++) {
        x0.randu();
        x0 *= 2;
        x0 -= 1;
        while (arma::norm(x0) >= 1) {
            x0 /= 2;
        }
        p.setX0(x0);

        OptResult res_rand = minimize(p);
        if (abs(res_rand.fVal) > abs(out_f)) {
            out_f = res_rand.fVal;
            out_x = res_rand.resultX;
        }
    }

    std::cout << "The new maximum vector is " << std::endl;
    std::cout << out_x << std::endl;
    std::cout << "With value" << std::endl;
    std::cout << out_f << std::endl;

    delete neg;
    delete obj;
}

static void ensure_within_region(const LagrangeParams& params, arma::mat& out_unshifted, bool p) {
    arma::Row<double> center = arma::conv_to<arma::Row<double> >::from(
            params.shifter->unshift(arma::vec(out_unshifted.n_cols, arma::fill::zeros))
    );
    for (int i=0; i<out_unshifted.n_rows; i++) {
        if (params.pf(arma::conv_to<arma::vec>::from(out_unshifted.row(i)))) {
            continue;
        }
        out_unshifted(i, arma::span::all) = center;
    }
    if (p) std::cout << "filtered" << std::endl;
    if (p) std::cout << out_unshifted << std::endl;
}

//static void __eval(int dim, const arma::mat& V, const arma::mat& shifted, int poly, int point) {
//    double s = 0;
//
//    s += V(dim + 0, poly);
//    s += V(dim + 1, poly) * shifted(point, 1);
//    s += V(dim + 2, poly) * shifted(point, 0);
//    s += .5 * V(dim + 3, poly) * shifted(point, 0) * shifted(point, 0);
//    s += .5 * V(dim + 4, poly) * shifted(point, 0) * shifted(point, 0);
//    s += .5 * V(dim + 5, poly) * shifted(point, 0) * shifted(point, 0);
//
//    return s;
//}

static void test_v(const LagrangeParams& params, const arma::mat& unshifted, const int dim, const arma::mat& V) {
//    std::cout << "testing" << std::endl;
    arma::mat shifted = params.shifter->shift(unshifted);
//    std::cout << V(arma::span(0, dim-1), arma::span::all) - params.basis->toRow(shifted) * V(arma::span(dim, 2*dim-1), arma::span::all) << std::endl;
    double m = arma::abs(V(arma::span(0, dim-1), arma::span::all) - params.basis->toRow(shifted) * V(arma::span(dim, 2*dim-1), arma::span::all)).max();
    if (m > 0.0003) {
        throw 1;
    }
}

void computeLagrangePolynomials(
        const LagrangeParams& params,
        const arma::mat& in_unshifted,
        LagrangeCertification& cert,
        arma::mat& out_unshifted
) {
    bool p = true;
    int dim = params.basis->get_dim();
    if (in_unshifted.n_rows != dim) {
        throw 1;
    }
    int h = 2 * dim;

    out_unshifted = in_unshifted;
    ensure_within_region(params, out_unshifted, p);

    arma::mat V(h, dim, arma::fill::none);
    V(arma::span(0, dim-1), arma::span::all) = params.basis->toRow(params.shifter->shift(out_unshifted));
    V(arma::span(dim, h-1), arma::span::all) = arma::eye(dim, dim);

    test_v(params, out_unshifted, dim, V);
    std::cout << out_unshifted << std::endl;

    if (p) std::cout << "original columns" << std::endl;
    if (p) std::cout << V << std::endl;
//
//    cert.outputXsi = params.xsi;
//
    for (int i=0;i<dim;i++) {
        int index_max = abs(V(arma::span(i, dim-1), i)).index_max() + i;
        double max_val = V(index_max, i);

        if (max_val < cert.outputXsi && params.canEvaluateNewPoints || true) {
            arma::vec coeff = V(arma::span(dim, h - 1), i);
            std::cout << "the coefficients" << std::endl;
            std::cout << coeff << std::endl;
            arma::vec new_x;
            double out_f;
            params.maximize(
              params.basis,
              coeff,
              new_x,
              out_f
            );

            out_unshifted.row(i) = arma::conv_to<arma::Row<double> >::from(params.shifter->unshift(new_x));
            V.row(i) = params.basis->toRow(new_x).t() * V(arma::span(dim, h-1), arma::span::all);

            std::cout << "The maximized V" << std::endl;
            std::cout << V << std::endl;
            std::cout << out_unshifted << std::endl;

            index_max = abs(V(arma::span(i, dim-1), i)).index_max() + i;
            max_val = V(index_max, i);

            test_v(params, out_unshifted, dim, V);
        }

//        if (maxVal < cert.outputXsi) {
//            if (params.xsiIsModifiable) {
//                // get lambda
//                // update xsi to be the new minimum of the maximum
//                // as long as newXsi >= params.minXsi
//            }
//
//            arma::vec newX = params.maximize(V(arma::span(npoints, h), i));
//            replace(cert, i, newX, npoints, h, V);
//        }

        std::cout << "the max index " << index_max << std::endl;
        std::cout << "the current index " << i << std::endl;
        if (index_max != i) {
            if (p) std::cout << "swapping " << i << " and " << index_max << std::endl;
            V.swap_rows(i, index_max);
            out_unshifted.swap_rows(i, index_max);

            std::cout << "The swapped V" << std::endl;
            std::cout << out_unshifted << std::endl;

            test_v(params, out_unshifted, dim, V);
        }

        std::cout << "the maximum value: " << max_val << std::endl;
        std::cout << "the maximum value: " << V(i, i) << std::endl;

        std::cout << "dividing by " << V(i, i) << std::endl;
        V.col(i) /= V(i, i);

        if (p) std::cout << "before" << std::endl;
        if (p) std::cout << V << std::endl;
        if (p) std::cout << "----------------------------------" << std::endl;

        for (int j = 0; j < dim; j++) {
            if (i == j) {
                continue;
            }
            V.col(j) = V.col(j) - V(i, j) * V.col(i);
        }

        if (p) std::cout << V << std::endl;
        if (p) std::cout << "----------------------------------" << std::endl;
        if (p) std::cout << "----------------------------------" << std::endl;
        test_v(params, out_unshifted, dim, V);
    }

    if (p) std::cout << "the result" << std::endl;
    if (p) std::cout << V << std::endl;
    test_v(params, out_unshifted, dim, V);
}











































//
//
//
//
//void toOptPPFun(
//        int mode,
//        int ndim,
//        const NEWMAT::ColumnVector& x,
//        double &fx,
//        NEWMAT::ColumnVector& gx,
//        NEWMAT::SymmetricMatrix& Hx,
//        int& result,
//        void *vptr
//) {
//    Function *f = (Function *) vptr;
//    std::cout << "We  are evaluating a function" << std::endl;
//
//    arma::vec ax = new2arma(x, ndim);
//    if (mode & OPTPP::NLPFunction) {
//        fx  = f->evaluate(ax);
//        result = OPTPP::NLPFunction;
//    }
//    if (mode & OPTPP::NLPGradient) {
//        arma::vec g = f->gradient(ax);
//        arma2new(g, gx);
//        result = OPTPP::NLPGradient;
//    }
//    if (mode & OPTPP::NLPHessian) {
//        arma::mat h = f->hessian(ax);
//        arma2new(h, Hx);
//        result = OPTPP::NLPHessian;
//    }
//}
//
//
////void optPPCons(
////        int mode,
////        int ndim,
////        const NEWMAT::ColumnVector& x,
////        NEWMAT::ColumnVector& cx,
////        NEWMAT::Matrix& cgx,
////        OPTPP::OptppArray<NEWMAT::SymmetricMatrix>& cHx,
////        int& result,
////        void *vptr
////) {
////    Function *f = (Function *) vptr;
////    arma::vec ax = new2arma(x, ndim);
////
////    if (mode & OPTPP::NLPFunction) {
////        std::cout << "Evaluating func";
////        cx(1) = f->evaluate(ax);
////        result = OPTPP::NLPFunction;
////    }
////    if (mode & OPTPP::NLPGradient) {
////        std::cout << "Evaluating grad";
////        arma::vec g = f->gradient(ax);
////        for (int i=0;i<g.n_elem;i++) {
////            cgx(i+1, 1) = g[i];
////        }
////        result = OPTPP::NLPGradient;
////    }
////    if (mode & OPTPP::NLPHessian) {
////        std::cout << "Evaluating hess";
////        const arma::mat h = f->hessian(ax);
////
////        NEWMAT::SymmetricMatrix Htmp(ndim);
////        NEWMAT::SymmetricMatrix& Hother = Htmp;
////        arma2new(h, Hother);
////
////        cHx[0] = Htmp;
////        result = OPTPP::NLPHessian;
////    }
////}
//
//void optPPInit(
//    int ndim,
//    NEWMAT::ColumnVector& x
////        ,
////    void *vptr
//) {
////    Program *p = (Program *)vptr;
////    arma2new(p->x0, x);
//    x(1) = 13;
//    x(2) = 233;
//
//    std::cout << "I am called!" << std::endl;
//}
//
//
//void optPPInit2(
//        int ndim,
//        NEWMAT::ColumnVector& x
////        ,
////    void *vptr
//) {
////    Program *p = (Program *)vptr;
////    arma2new(p->x0, x);
//    x(1) = 10000000;
//    x(2) = 42;
//
//    std::cout << "I am called!" << std::endl;
//}
//
//
//void minimize2(Program& p) {
////    OPTPP::OptppArray<OPTPP::Constraint> allConstraints;
////    for (std::list<Constraint>::iterator it = p.constraints.begin(); it != p.constraints.end(); ++it) {
////        Function *f = &((*it).function);
////        OPTPP::NLF2 *s1 = new OPTPP::NLF2(p.n, &toOptPPFun, &optPPInit, (void *)f);
////        OPTPP::NLP* s2 = new OPTPP::NLP(s1);
////        OPTPP::Constraint s3 = new OPTPP::NonLinearInequality(s2);
////        allConstraints.append(s3);
////    }
////
////    OPTPP::CompoundConstraint* constraints = new OPTPP::CompoundConstraint(allConstraints);
////    OPTPP::NLF2 nips(p.n, &toOptPPFun, &optPPInit2, constraints, (void *)&p.obj);
////
////    OPTPP::OptNIPS objfcn(&nips);
////
////    // The "0" in the second argument says to create a new file.  A "1"
////    // would signify appending to an existing file.
////
////    objfcn.setOutputFile("example2.out", 0);
////    objfcn.setFcnTol(p.tol);
////    objfcn.setMaxIter(1000);
////    objfcn.setMeritFcn(OPTPP::ArgaezTapia);
////
////    objfcn.optimize();
////
////    objfcn.printStatus((char *)"Solution from nips");
////    objfcn.cleanup();
//}
