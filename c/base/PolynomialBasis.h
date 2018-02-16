//
// Created by thallock on 12/16/17.
//

#ifndef C_POLY_H
#define C_POLY_H

#include <armadillo>
#include "../fcn/Utils.h"

class PolynomialBasis {
    int n;
    int degree;
    int basis_dimension;

    arma::Mat<int> powers;
    arma::vec coeff;
public:

//    PolynomialBasis(int n, int degree, int basis_dimension);
    PolynomialBasis(int n, int degree);

    arma::mat toRow(const arma::mat& x) const;
    arma::vec toRow(const arma::vec& x) const;
    arma::mat gradients(const arma::vec& x) const;

    BasisFunction *createPolynomial(const arma::vec& coefficients) const;

    friend std::ostream& operator<<(std::ostream& os, const PolynomialBasis& basis);

    int get_n() const {
        return n;
    }
    int get_dim() const {
        return basis_dimension;
    }
};


#endif //C_POLY_H
